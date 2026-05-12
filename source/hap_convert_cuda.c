/*
 hap_convert_cuda.c

 CUDA driver API + NVRTC backend for HAP YCbCr texture conversion.

 Build requirements (when -DHAVE_CUDA is set):
   - Link against: -lcuda  -lnvrtc
   - Include path must contain cuda.h and nvrtc.h.

 All CUDA kernels are embedded as a C string and compiled at runtime via NVRTC,
 so no separate .cu compilation step is required.

 The GPU conversion pipeline for RGB→YCbCr is:
   1. Upload compressed BC input to device.
   2. Decode BC1/BC3 → RGBA pixels on GPU.
   3. Convert RGBA → Y, Cb, Cr planes (BT.709 full-range) on GPU.
   4. Optionally downsample Cb/Cr (4:2:2 or 4:2:0) on GPU.
   5. Encode each plane → BC4/RGTC1 on GPU.
   6. Download compressed BC4 planes to host.

 The reverse (YCbCr→RGB) is the inverse pipeline.
 */

#include "hap_convert_cuda.h"
#include "hap_convert.h"
#include "hap.h"

#ifdef HAVE_CUDA

#include <cuda.h>
#include <nvrtc.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ── Convenience macros ─────────────────────────────────────────────────── */

#define HAP_CU_CHECK(expr) do { CUresult _r = (expr); if (_r != CUDA_SUCCESS) { return 0; } } while(0)
#define HAP_NVRTC_CHECK(expr) do { nvrtcResult _r = (expr); if (_r != NVRTC_SUCCESS) { return 0; } } while(0)

/* ceil-divide a by b */
#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

/* ── Embedded CUDA kernel source ────────────────────────────────────────── */

/*
 All kernels are written against the CUDA device programming model.
 One CUDA thread processes one 4×4 BC block (decode/encode kernels)
 or one pixel (colour-convert / chroma-downsample kernels).
 */
static const char kKernelSource[] =
    "typedef unsigned char  u8;\n"
    "typedef unsigned short u16;\n"
    "typedef unsigned int   u32;\n"
    "typedef unsigned long long u64;\n"
    "\n"
    /* ── helpers ── */
    "static __device__ void unpack565(u16 c, u8*r, u8*g, u8*b) {\n"
    "    *r=(u8)(((c>>11)&0x1f)*255/31);\n"
    "    *g=(u8)(((c>> 5)&0x3f)*255/63);\n"
    "    *b=(u8)(( c     &0x1f)*255/31);\n"
    "}\n"
    "static __device__ u16 pack565(u8 r,u8 g,u8 b){\n"
    "    return ((u16)(r>>3)<<11)|((u16)(g>>2)<<5)|(b>>3);\n"
    "}\n"
    "static __device__ u8 clamp255(float v){\n"
    "    return (u8)fminf(fmaxf(v+0.5f,0.0f),255.0f);\n"
    "}\n"
    "\n"
    /* ══ BC4/RGTC1 decode ══ */
    "extern \"C\" __global__ void hap_bc4_decode(\n"
    "    const u8* __restrict__ input, u8* __restrict__ output,\n"
    "    int blocks_x, int blocks_y, int out_w, int out_h)\n"
    "{\n"
    "    int bx=blockIdx.x*blockDim.x+threadIdx.x;\n"
    "    int by=blockIdx.y*blockDim.y+threadIdx.y;\n"
    "    if(bx>=blocks_x||by>=blocks_y) return;\n"
    "    const u8* blk=input+(by*blocks_x+bx)*8;\n"
    "    u8 r0=blk[0],r1=blk[1],pal[8];\n"
    "    pal[0]=r0; pal[1]=r1;\n"
    "    if(r0>r1){\n"
    "        pal[2]=(u8)((6*r0+1*r1+3)/7); pal[3]=(u8)((5*r0+2*r1+3)/7);\n"
    "        pal[4]=(u8)((4*r0+3*r1+3)/7); pal[5]=(u8)((3*r0+4*r1+3)/7);\n"
    "        pal[6]=(u8)((2*r0+5*r1+3)/7); pal[7]=(u8)((1*r0+6*r1+3)/7);\n"
    "    } else {\n"
    "        pal[2]=(u8)((4*r0+1*r1+2)/5); pal[3]=(u8)((3*r0+2*r1+2)/5);\n"
    "        pal[4]=(u8)((2*r0+3*r1+2)/5); pal[5]=(u8)((1*r0+4*r1+2)/5);\n"
    "        pal[6]=0; pal[7]=255;\n"
    "    }\n"
    "    u64 bits=0;\n"
    "    for(int i=0;i<6;i++) bits|=(u64)blk[2+i]<<(i*8);\n"
    "    for(int py=0;py<4;py++) for(int px=0;px<4;px++){\n"
    "        int ix=bx*4+px, iy=by*4+py;\n"
    "        if(ix<out_w&&iy<out_h)\n"
    "            output[iy*out_w+ix]=pal[(bits>>((py*4+px)*3))&7];\n"
    "    }\n"
    "}\n"
    "\n"
    /* ══ BC1/DXT1 decode ══ */
    "extern \"C\" __global__ void hap_bc1_decode(\n"
    "    const u8* __restrict__ input, u8* __restrict__ rgba,\n"
    "    int blocks_x, int blocks_y, int out_w, int out_h)\n"
    "{\n"
    "    int bx=blockIdx.x*blockDim.x+threadIdx.x;\n"
    "    int by=blockIdx.y*blockDim.y+threadIdx.y;\n"
    "    if(bx>=blocks_x||by>=blocks_y) return;\n"
    "    const u8* blk=input+(by*blocks_x+bx)*8;\n"
    "    u16 c0=(u16)(blk[0]|((u16)blk[1]<<8));\n"
    "    u16 c1=(u16)(blk[2]|((u16)blk[3]<<8));\n"
    "    u8 r[4],g[4],b[4],a[4]; a[0]=a[1]=a[2]=255;\n"
    "    unpack565(c0,&r[0],&g[0],&b[0]); unpack565(c1,&r[1],&g[1],&b[1]);\n"
    "    if(c0>c1){\n"
    "        r[2]=(u8)((2*r[0]+r[1]+1)/3); g[2]=(u8)((2*g[0]+g[1]+1)/3); b[2]=(u8)((2*b[0]+b[1]+1)/3);\n"
    "        r[3]=(u8)((r[0]+2*r[1]+1)/3); g[3]=(u8)((g[0]+2*g[1]+1)/3); b[3]=(u8)((b[0]+2*b[1]+1)/3);\n"
    "        a[3]=255;\n"
    "    } else {\n"
    "        r[2]=(u8)((r[0]+r[1])/2); g[2]=(u8)((g[0]+g[1])/2); b[2]=(u8)((b[0]+b[1])/2);\n"
    "        r[3]=g[3]=b[3]=a[3]=0;\n"
    "    }\n"
    "    u32 idx32=(u32)blk[4]|((u32)blk[5]<<8)|((u32)blk[6]<<16)|((u32)blk[7]<<24);\n"
    "    for(int py=0;py<4;py++) for(int px=0;px<4;px++){\n"
    "        int ix=bx*4+px, iy=by*4+py;\n"
    "        if(ix<out_w&&iy<out_h){\n"
    "            u8 ji=(idx32>>((py*4+px)*2))&3;\n"
    "            int off=(iy*out_w+ix)*4;\n"
    "            rgba[off+0]=r[ji]; rgba[off+1]=g[ji]; rgba[off+2]=b[ji]; rgba[off+3]=a[ji];\n"
    "        }\n"
    "    }\n"
    "}\n"
    "\n"
    /* ══ BC3/DXT5 decode ══ */
    "extern \"C\" __global__ void hap_bc3_decode(\n"
    "    const u8* __restrict__ input, u8* __restrict__ rgba,\n"
    "    int blocks_x, int blocks_y, int out_w, int out_h)\n"
    "{\n"
    "    int bx=blockIdx.x*blockDim.x+threadIdx.x;\n"
    "    int by=blockIdx.y*blockDim.y+threadIdx.y;\n"
    "    if(bx>=blocks_x||by>=blocks_y) return;\n"
    "    /* BC3 = 16 bytes: 8 alpha (BC4-like) + 8 color (BC1 4-color) */\n"
    "    const u8* ablk=input+(by*blocks_x+bx)*16;\n"
    "    const u8* cblk=ablk+8;\n"
    "    u8 a0=ablk[0],a1=ablk[1],apal[8];\n"
    "    apal[0]=a0; apal[1]=a1;\n"
    "    if(a0>a1){\n"
    "        apal[2]=(u8)((6*a0+1*a1+3)/7); apal[3]=(u8)((5*a0+2*a1+3)/7);\n"
    "        apal[4]=(u8)((4*a0+3*a1+3)/7); apal[5]=(u8)((3*a0+4*a1+3)/7);\n"
    "        apal[6]=(u8)((2*a0+5*a1+3)/7); apal[7]=(u8)((1*a0+6*a1+3)/7);\n"
    "    } else {\n"
    "        apal[2]=(u8)((4*a0+1*a1+2)/5); apal[3]=(u8)((3*a0+2*a1+2)/5);\n"
    "        apal[4]=(u8)((2*a0+3*a1+2)/5); apal[5]=(u8)((1*a0+4*a1+2)/5);\n"
    "        apal[6]=0; apal[7]=255;\n"
    "    }\n"
    "    u64 abits=0; for(int i=0;i<6;i++) abits|=(u64)ablk[2+i]<<(i*8);\n"
    "    u16 cc0=(u16)(cblk[0]|((u16)cblk[1]<<8));\n"
    "    u16 cc1=(u16)(cblk[2]|((u16)cblk[3]<<8));\n"
    "    u8 r[4],g[4],b[4];\n"
    "    unpack565(cc0,&r[0],&g[0],&b[0]); unpack565(cc1,&r[1],&g[1],&b[1]);\n"
    "    r[2]=(u8)((2*r[0]+r[1]+1)/3); g[2]=(u8)((2*g[0]+g[1]+1)/3); b[2]=(u8)((2*b[0]+b[1]+1)/3);\n"
    "    r[3]=(u8)((r[0]+2*r[1]+1)/3); g[3]=(u8)((g[0]+2*g[1]+1)/3); b[3]=(u8)((b[0]+2*b[1]+1)/3);\n"
    "    u32 cidx=(u32)cblk[4]|((u32)cblk[5]<<8)|((u32)cblk[6]<<16)|((u32)cblk[7]<<24);\n"
    "    for(int py=0;py<4;py++) for(int px=0;px<4;px++){\n"
    "        int ix=bx*4+px, iy=by*4+py;\n"
    "        if(ix<out_w&&iy<out_h){\n"
    "            int pi=py*4+px;\n"
    "            u8 ci=(cidx>>(pi*2))&3, ai=(abits>>(pi*3))&7;\n"
    "            int off=(iy*out_w+ix)*4;\n"
    "            rgba[off+0]=r[ci]; rgba[off+1]=g[ci]; rgba[off+2]=b[ci]; rgba[off+3]=apal[ai];\n"
    "        }\n"
    "    }\n"
    "}\n"
    "\n"
    /* ══ RGBA → YCbCr (BT.709 full-range) ══ */
    "extern \"C\" __global__ void hap_rgba_to_ycbcr(\n"
    "    const u8* __restrict__ rgba,\n"
    "    u8* __restrict__ Y, u8* __restrict__ Cb, u8* __restrict__ Cr,\n"
    "    int width, int height)\n"
    "{\n"
    "    int x=blockIdx.x*blockDim.x+threadIdx.x;\n"
    "    int y=blockIdx.y*blockDim.y+threadIdx.y;\n"
    "    if(x>=width||y>=height) return;\n"
    "    int idx=y*width+x;\n"
    "    float r=rgba[idx*4+0]/255.0f;\n"
    "    float g=rgba[idx*4+1]/255.0f;\n"
    "    float b=rgba[idx*4+2]/255.0f;\n"
    "    float luma = 0.2126f*r + 0.7152f*g + 0.0722f*b;\n"
    "    float cb   =-0.1146f*r - 0.3854f*g + 0.5000f*b;\n"
    "    float cr   = 0.5000f*r - 0.4542f*g - 0.0458f*b;\n"
    "    Y [idx]=clamp255(luma*255.0f);\n"
    "    Cb[idx]=clamp255(cb  *255.0f+128.0f);\n"
    "    Cr[idx]=clamp255(cr  *255.0f+128.0f);\n"
    "}\n"
    "\n"
    /* ══ YCbCr → RGBA (BT.709, nearest-neighbour chroma upscale) ══ */
    "extern \"C\" __global__ void hap_ycbcr_to_rgba(\n"
    "    const u8* __restrict__ Y, const u8* __restrict__ Cb, const u8* __restrict__ Cr,\n"
    "    u8* __restrict__ rgba,\n"
    "    int y_w, int y_h, int cb_w, int cb_h)\n"
    "{\n"
    "    int x=blockIdx.x*blockDim.x+threadIdx.x;\n"
    "    int y=blockIdx.y*blockDim.y+threadIdx.y;\n"
    "    if(x>=y_w||y>=y_h) return;\n"
    "    int cx=x*cb_w/y_w, cy=y*cb_h/y_h;\n"
    "    float luma= Y[y*y_w+x]/255.0f;\n"
    "    float cbv =(Cb[cy*cb_w+cx]-128.0f)/255.0f;\n"
    "    float crv =(Cr[cy*cb_w+cx]-128.0f)/255.0f;\n"
    "    float rv=luma+1.5748f*crv;\n"
    "    float gv=luma-0.1873f*cbv-0.4681f*crv;\n"
    "    float bv=luma+1.8556f*cbv;\n"
    "    int off=(y*y_w+x)*4;\n"
    "    rgba[off+0]=clamp255(rv*255.0f);\n"
    "    rgba[off+1]=clamp255(gv*255.0f);\n"
    "    rgba[off+2]=clamp255(bv*255.0f);\n"
    "    rgba[off+3]=255;\n"
    "}\n"
    "\n"
    /* ══ Chroma downsample 4:2:2 (halve width) ══ */
    "extern \"C\" __global__ void hap_chroma_down_422(\n"
    "    const u8* __restrict__ src, u8* __restrict__ dst,\n"
    "    int src_w, int height)\n"
    "{\n"
    "    int x=blockIdx.x*blockDim.x+threadIdx.x;\n"
    "    int y=blockIdx.y*blockDim.y+threadIdx.y;\n"
    "    int dst_w=src_w/2;\n"
    "    if(x>=dst_w||y>=height) return;\n"
    "    dst[y*dst_w+x]=(u8)((src[y*src_w+2*x]+src[y*src_w+2*x+1]+1)/2);\n"
    "}\n"
    "\n"
    /* ══ Chroma downsample 4:2:0 (halve width and height) ══ */
    "extern \"C\" __global__ void hap_chroma_down_420(\n"
    "    const u8* __restrict__ src, u8* __restrict__ dst,\n"
    "    int src_w, int src_h)\n"
    "{\n"
    "    int x=blockIdx.x*blockDim.x+threadIdx.x;\n"
    "    int y=blockIdx.y*blockDim.y+threadIdx.y;\n"
    "    int dw=src_w/2, dh=src_h/2;\n"
    "    if(x>=dw||y>=dh) return;\n"
    "    unsigned a=src[(2*y)*src_w+2*x],   b=src[(2*y)*src_w+2*x+1];\n"
    "    unsigned c=src[(2*y+1)*src_w+2*x], d=src[(2*y+1)*src_w+2*x+1];\n"
    "    dst[y*dw+x]=(u8)((a+b+c+d+2)/4);\n"
    "}\n"
    "\n"
    /* ══ BC4 encode ══ */
    "extern \"C\" __global__ void hap_bc4_encode(\n"
    "    const u8* __restrict__ input, u8* __restrict__ output,\n"
    "    int blocks_x, int blocks_y, int in_w, int in_h)\n"
    "{\n"
    "    int bx=blockIdx.x*blockDim.x+threadIdx.x;\n"
    "    int by=blockIdx.y*blockDim.y+threadIdx.y;\n"
    "    if(bx>=blocks_x||by>=blocks_y) return;\n"
    "    u8 px[16]; u8 mn=255,mx=0;\n"
    "    for(int py=0;py<4;py++) for(int ppx=0;ppx<4;ppx++){\n"
    "        int ix=bx*4+ppx, iy=by*4+py;\n"
    "        u8 v=(ix<in_w&&iy<in_h)?input[iy*in_w+ix]:0;\n"
    "        px[py*4+ppx]=v;\n"
    "        if(v<mn)mn=v; if(v>mx)mx=v;\n"
    "    }\n"
    "    u8* blk=output+(by*blocks_x+bx)*8;\n"
    "    blk[0]=mx; blk[1]=mn;\n"
    "    if(mx==mn){for(int i=2;i<8;i++)blk[i]=0; return;}\n"
    "    u8 pal[8];\n"
    "    pal[0]=mx; pal[1]=mn;\n"
    "    pal[2]=(u8)((6*mx+1*mn+3)/7); pal[3]=(u8)((5*mx+2*mn+3)/7);\n"
    "    pal[4]=(u8)((4*mx+3*mn+3)/7); pal[5]=(u8)((3*mx+4*mn+3)/7);\n"
    "    pal[6]=(u8)((2*mx+5*mn+3)/7); pal[7]=(u8)((1*mx+6*mn+3)/7);\n"
    "    u64 bits=0;\n"
    "    for(int i=0;i<16;i++){\n"
    "        u8 best=0; int bd=256*256;\n"
    "        for(int j=0;j<8;j++){int d=(int)px[i]-(int)pal[j]; if(d*d<bd){bd=d*d;best=(u8)j;}}\n"
    "        bits|=(u64)best<<(i*3);\n"
    "    }\n"
    "    for(int i=0;i<6;i++) blk[2+i]=(u8)((bits>>(i*8))&0xff);\n"
    "}\n"
    "\n"
    /* ══ BC1 encode ══ */
    "extern \"C\" __global__ void hap_bc1_encode(\n"
    "    const u8* __restrict__ rgba, u8* __restrict__ output,\n"
    "    int blocks_x, int blocks_y, int in_w, int in_h)\n"
    "{\n"
    "    int bx=blockIdx.x*blockDim.x+threadIdx.x;\n"
    "    int by=blockIdx.y*blockDim.y+threadIdx.y;\n"
    "    if(bx>=blocks_x||by>=blocks_y) return;\n"
    "    u8 R[16],G[16],B[16];\n"
    "    u8 rn=255,rx=0,gn=255,gx=0,bn=255,bx_=0;\n"
    "    for(int py=0;py<4;py++) for(int ppx=0;ppx<4;ppx++){\n"
    "        int ix=bx*4+ppx, iy=by*4+py, pi=py*4+ppx;\n"
    "        if(ix<in_w&&iy<in_h){int off=(iy*in_w+ix)*4; R[pi]=rgba[off];G[pi]=rgba[off+1];B[pi]=rgba[off+2];}\n"
    "        else{R[pi]=G[pi]=B[pi]=0;}\n"
    "        if(R[pi]<rn)rn=R[pi]; if(R[pi]>rx)rx=R[pi];\n"
    "        if(G[pi]<gn)gn=G[pi]; if(G[pi]>gx)gx=G[pi];\n"
    "        if(B[pi]<bn)bn=B[pi]; if(B[pi]>bx_)bx_=B[pi];\n"
    "    }\n"
    "    u16 c0=pack565(rx,gx,bx_), c1=pack565(rn,gn,bn);\n"
    "    if(c0<=c1){u16 t=c0;c0=c1;c1=t;}\n"
    "    u8 pr[4],pg[4],pb[4];\n"
    "    unpack565(c0,&pr[0],&pg[0],&pb[0]); unpack565(c1,&pr[1],&pg[1],&pb[1]);\n"
    "    pr[2]=(u8)((2*pr[0]+pr[1]+1)/3); pg[2]=(u8)((2*pg[0]+pg[1]+1)/3); pb[2]=(u8)((2*pb[0]+pb[1]+1)/3);\n"
    "    pr[3]=(u8)((pr[0]+2*pr[1]+1)/3); pg[3]=(u8)((pg[0]+2*pg[1]+1)/3); pb[3]=(u8)((pb[0]+2*pb[1]+1)/3);\n"
    "    u32 indices=0;\n"
    "    for(int i=0;i<16;i++){\n"
    "        int best=0,bd=256*256*3;\n"
    "        for(int j=0;j<4;j++){int dr=R[i]-pr[j],dg=G[i]-pg[j],db=B[i]-pb[j]; int d=dr*dr+dg*dg+db*db; if(d<bd){bd=d;best=j;}}\n"
    "        indices|=(u32)best<<(i*2);\n"
    "    }\n"
    "    u8* blk=output+(by*blocks_x+bx)*8;\n"
    "    blk[0]=c0&0xff; blk[1]=(c0>>8)&0xff; blk[2]=c1&0xff; blk[3]=(c1>>8)&0xff;\n"
    "    blk[4]=indices&0xff; blk[5]=(indices>>8)&0xff; blk[6]=(indices>>16)&0xff; blk[7]=(indices>>24)&0xff;\n"
    "}\n"
    "\n"
    /* ══ BC3 encode ══ */
    "extern \"C\" __global__ void hap_bc3_encode(\n"
    "    const u8* __restrict__ rgba, u8* __restrict__ output,\n"
    "    int blocks_x, int blocks_y, int in_w, int in_h)\n"
    "{\n"
    "    int bx=blockIdx.x*blockDim.x+threadIdx.x;\n"
    "    int by=blockIdx.y*blockDim.y+threadIdx.y;\n"
    "    if(bx>=blocks_x||by>=blocks_y) return;\n"
    "    u8 R[16],G[16],B[16],A[16];\n"
    "    u8 rn=255,rx=0,gn=255,gx=0,bn=255,bx_=0,an=255,ax=0;\n"
    "    for(int py=0;py<4;py++) for(int ppx=0;ppx<4;ppx++){\n"
    "        int ix=bx*4+ppx,iy=by*4+py,pi=py*4+ppx;\n"
    "        if(ix<in_w&&iy<in_h){int off=(iy*in_w+ix)*4; R[pi]=rgba[off];G[pi]=rgba[off+1];B[pi]=rgba[off+2];A[pi]=rgba[off+3];}\n"
    "        else{R[pi]=G[pi]=B[pi]=0; A[pi]=255;}\n"
    "        if(R[pi]<rn)rn=R[pi]; if(R[pi]>rx)rx=R[pi];\n"
    "        if(G[pi]<gn)gn=G[pi]; if(G[pi]>gx)gx=G[pi];\n"
    "        if(B[pi]<bn)bn=B[pi]; if(B[pi]>bx_)bx_=B[pi];\n"
    "        if(A[pi]<an)an=A[pi]; if(A[pi]>ax)ax=A[pi];\n"
    "    }\n"
    "    u8* blk=output+(by*blocks_x+bx)*16;\n"
    "    /* Alpha block (BC4-like) */\n"
    "    blk[0]=ax; blk[1]=an;\n"
    "    if(ax==an){for(int i=2;i<8;i++)blk[i]=0;}\n"
    "    else{\n"
    "        u8 ap[8]; ap[0]=ax;ap[1]=an;\n"
    "        ap[2]=(u8)((6*ax+1*an+3)/7);ap[3]=(u8)((5*ax+2*an+3)/7);\n"
    "        ap[4]=(u8)((4*ax+3*an+3)/7);ap[5]=(u8)((3*ax+4*an+3)/7);\n"
    "        ap[6]=(u8)((2*ax+5*an+3)/7);ap[7]=(u8)((1*ax+6*an+3)/7);\n"
    "        u64 abits=0;\n"
    "        for(int i=0;i<16;i++){u8 best=0;int bd=256*256;\n"
    "            for(int j=0;j<8;j++){int d=(int)A[i]-(int)ap[j];if(d*d<bd){bd=d*d;best=(u8)j;}}\n"
    "            abits|=(u64)best<<(i*3);}\n"
    "        for(int i=0;i<6;i++) blk[2+i]=(u8)((abits>>(i*8))&0xff);\n"
    "    }\n"
    "    /* Color block (BC1, always 4-color in BC3) */\n"
    "    u16 c0=pack565(rx,gx,bx_),c1=pack565(rn,gn,bn);\n"
    "    if(c0<=c1){u16 t=c0;c0=c1;c1=t;}\n"
    "    u8 pr[4],pg[4],pb[4];\n"
    "    unpack565(c0,&pr[0],&pg[0],&pb[0]); unpack565(c1,&pr[1],&pg[1],&pb[1]);\n"
    "    pr[2]=(u8)((2*pr[0]+pr[1]+1)/3);pg[2]=(u8)((2*pg[0]+pg[1]+1)/3);pb[2]=(u8)((2*pb[0]+pb[1]+1)/3);\n"
    "    pr[3]=(u8)((pr[0]+2*pr[1]+1)/3);pg[3]=(u8)((pg[0]+2*pg[1]+1)/3);pb[3]=(u8)((pb[0]+2*pb[1]+1)/3);\n"
    "    u32 indices=0;\n"
    "    for(int i=0;i<16;i++){int best=0,bd=256*256*3;\n"
    "        for(int j=0;j<4;j++){int dr=R[i]-pr[j],dg=G[i]-pg[j],db=B[i]-pb[j];int d=dr*dr+dg*dg+db*db;if(d<bd){bd=d;best=j;}}\n"
    "        indices|=(u32)best<<(i*2);}\n"
    "    blk[8]=c0&0xff;blk[9]=(c0>>8)&0xff;blk[10]=c1&0xff;blk[11]=(c1>>8)&0xff;\n"
    "    blk[12]=indices&0xff;blk[13]=(indices>>8)&0xff;\n"
    "    blk[14]=(indices>>16)&0xff;blk[15]=(indices>>24)&0xff;\n"
    "}\n";

/* ── Global state (initialised once per process) ─────────────────────────── */

static int        s_init_done   = 0;
static int        s_init_ok     = 0;
static CUcontext  s_ctx         = NULL;
static int        s_ctx_owned   = 0; /* did we create the context? */
static CUmodule   s_module      = NULL;
static CUfunction s_fn_bc4_decode     = NULL;
static CUfunction s_fn_bc1_decode     = NULL;
static CUfunction s_fn_bc3_decode     = NULL;
static CUfunction s_fn_rgba_to_ycbcr  = NULL;
static CUfunction s_fn_ycbcr_to_rgba  = NULL;
static CUfunction s_fn_chroma_down_422= NULL;
static CUfunction s_fn_chroma_down_420= NULL;
static CUfunction s_fn_bc4_encode     = NULL;
static CUfunction s_fn_bc1_encode     = NULL;
static CUfunction s_fn_bc3_encode     = NULL;

/* ── Initialisation ─────────────────────────────────────────────────────── */

/* Attempt to initialise CUDA and compile the kernels.  Returns 1 on success. */
static int hap_cuda_do_init(void)
{
    CUresult cr;
    nvrtcResult nr;

    /* Initialise the CUDA driver. */
    cr = cuInit(0);
    if (cr != CUDA_SUCCESS) return 0;

    /* Get the first available device. */
    CUdevice dev;
    cr = cuDeviceGet(&dev, 0);
    if (cr != CUDA_SUCCESS) return 0;

    /* Reuse an existing context if the caller already set one up; otherwise
       create our own.  This avoids disrupting GPU-accelerated applications
       that manage their own CUDA context. */
    cr = cuCtxGetCurrent(&s_ctx);
    if (cr != CUDA_SUCCESS || s_ctx == NULL) {
        cr = cuCtxCreate(&s_ctx, 0, dev);
        if (cr != CUDA_SUCCESS) return 0;
        s_ctx_owned = 1;
    }

    /* Detect compute capability to target the correct PTX generation. */
    int major = 0, minor = 0;
    cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
    cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
    char arch_flag[32];
    snprintf(arch_flag, sizeof(arch_flag), "--gpu-architecture=compute_%d%d", major, minor);
    const char *compile_options[] = { arch_flag };

    /* Compile the kernel source to PTX via NVRTC. */
    nvrtcProgram prog;
    nr = nvrtcCreateProgram(&prog, kKernelSource, "hap_kernels.cu", 0, NULL, NULL);
    if (nr != NVRTC_SUCCESS) return 0;

    nr = nvrtcCompileProgram(prog, 1, compile_options);
    if (nr != NVRTC_SUCCESS) {
        /* Emit the compilation log to stderr to assist debugging. */
        size_t log_size = 0;
        nvrtcGetProgramLogSize(prog, &log_size);
        if (log_size > 1) {
            char *log = (char *)malloc(log_size);
            if (log) { nvrtcGetProgramLog(prog, log); fprintf(stderr, "hap NVRTC: %s\n", log); free(log); }
        }
        nvrtcDestroyProgram(&prog);
        return 0;
    }

    /* Extract PTX and load as a CUDA module. */
    size_t ptx_size = 0;
    nvrtcGetPTXSize(prog, &ptx_size);
    char *ptx = (char *)malloc(ptx_size);
    if (!ptx) { nvrtcDestroyProgram(&prog); return 0; }
    nvrtcGetPTX(prog, ptx);
    nvrtcDestroyProgram(&prog);

    cr = cuModuleLoadData(&s_module, ptx);
    free(ptx);
    if (cr != CUDA_SUCCESS) return 0;

    /* Resolve all kernel function handles. */
#define GET_FN(var, name) \
    cr = cuModuleGetFunction(&(var), s_module, (name)); \
    if (cr != CUDA_SUCCESS) return 0;

    GET_FN(s_fn_bc4_decode,      "hap_bc4_decode")
    GET_FN(s_fn_bc1_decode,      "hap_bc1_decode")
    GET_FN(s_fn_bc3_decode,      "hap_bc3_decode")
    GET_FN(s_fn_rgba_to_ycbcr,   "hap_rgba_to_ycbcr")
    GET_FN(s_fn_ycbcr_to_rgba,   "hap_ycbcr_to_rgba")
    GET_FN(s_fn_chroma_down_422, "hap_chroma_down_422")
    GET_FN(s_fn_chroma_down_420, "hap_chroma_down_420")
    GET_FN(s_fn_bc4_encode,      "hap_bc4_encode")
    GET_FN(s_fn_bc1_encode,      "hap_bc1_encode")
    GET_FN(s_fn_bc3_encode,      "hap_bc3_encode")
#undef GET_FN

    return 1;
}

int hap_cuda_available(void)
{
    if (!s_init_done) {
        s_init_done = 1;
        s_init_ok   = hap_cuda_do_init();
    }
    return s_init_ok;
}

/* ── Grid / block helpers ───────────────────────────────────────────────── */

/* 16×16 thread block for BC block-level kernels (one thread = one 4×4 BC block). */
#define BLOCK_DIM_BC 16
/* 32×32 thread block for pixel-level kernels. */
#define BLOCK_DIM_PIX 32

/* Launch a 2-D kernel with (grid_x × grid_y) blocks of (bdim × bdim) threads.
   args is the NULL-terminated void* array accepted by cuLaunchKernel. */
static int launch2d(CUfunction fn,
                    unsigned int grid_x, unsigned int grid_y,
                    unsigned int bdim,
                    void **args)
{
    CUresult cr = cuLaunchKernel(fn,
                                 grid_x, grid_y, 1,
                                 bdim,   bdim,   1,
                                 0, NULL, args, NULL);
    return (cr == CUDA_SUCCESS) ? 1 : 0;
}

/* ── RGB → YCbCr ────────────────────────────────────────────────────────── */

int hap_cuda_convert_rgb_to_ycbcr(
    const void  *input_bc,        unsigned long  input_bytes,
    unsigned int input_format,
    unsigned int width,           unsigned int   height,
    unsigned int subsampling,
    void        *output_y,        void          *output_cb,        void *output_cr,
    unsigned long output_y_bytes, unsigned long  output_cb_bytes,  unsigned long output_cr_bytes)
{
    if (!s_init_ok) return 0;

    unsigned int bx  = CEIL_DIV(width,  4);
    unsigned int by  = CEIL_DIV(height, 4);
    unsigned int cb_w, cb_h;
    hap_chroma_plane_dims(width, height, subsampling, &cb_w, &cb_h);
    unsigned int cb_bx = CEIL_DIV(cb_w, 4);
    unsigned int cb_by = CEIL_DIV(cb_h, 4);

    size_t rgba_bytes  = (size_t)width  * height * 4;
    size_t y_pix_bytes = (size_t)width  * height;
    size_t c_pix_bytes = (size_t)width  * height; /* full-res before downsample */
    size_t cb_out_bytes = (size_t)cb_w  * cb_h;

    /* ── Device memory allocation ── */
    CUdeviceptr d_input_bc = 0;
    CUdeviceptr d_rgba     = 0;
    CUdeviceptr d_Y_pix    = 0;
    CUdeviceptr d_Cb_full  = 0;
    CUdeviceptr d_Cr_full  = 0;
    CUdeviceptr d_Cb_pix   = 0; /* may point to d_Cb_full for 4:4:4 */
    CUdeviceptr d_Cr_pix   = 0;
    CUdeviceptr d_Y_bc     = 0;
    CUdeviceptr d_Cb_bc    = 0;
    CUdeviceptr d_Cr_bc    = 0;

    int ok = 1;

#define ALLOC(ptr, sz) \
    if (ok) { ok = (cuMemAlloc(&(ptr), (sz)) == CUDA_SUCCESS); }

    ALLOC(d_input_bc, input_bytes)
    ALLOC(d_rgba,     rgba_bytes)
    ALLOC(d_Y_pix,    y_pix_bytes)
    ALLOC(d_Cb_full,  c_pix_bytes)
    ALLOC(d_Cr_full,  c_pix_bytes)
    if (subsampling != HapYCbCrSubsampling_444) {
        ALLOC(d_Cb_pix, cb_out_bytes)
        ALLOC(d_Cr_pix, cb_out_bytes)
    }
    ALLOC(d_Y_bc,  output_y_bytes)
    ALLOC(d_Cb_bc, output_cb_bytes)
    ALLOC(d_Cr_bc, output_cr_bytes)
#undef ALLOC

    if (!ok) goto cleanup;

    /* Upload compressed input. */
    ok = (cuMemcpyHtoD(d_input_bc, input_bc, input_bytes) == CUDA_SUCCESS);
    if (!ok) goto cleanup;

    /* 1. Decode BC → RGBA */
    {
        int ibx=(int)bx, iby=(int)by, iw=(int)width, ih=(int)height;
        void *args[] = { &d_input_bc, &d_rgba, &ibx, &iby, &iw, &ih, NULL };
        CUfunction decode_fn = (input_format == HapTextureFormat_RGB_DXT1)
                               ? s_fn_bc1_decode : s_fn_bc3_decode;
        unsigned int gx = CEIL_DIV(bx, BLOCK_DIM_BC);
        unsigned int gy = CEIL_DIV(by, BLOCK_DIM_BC);
        ok = launch2d(decode_fn, gx, gy, BLOCK_DIM_BC, args);
    }
    if (!ok) goto cleanup;

    /* 2. RGBA → YCbCr (full resolution) */
    {
        int iw=(int)width, ih=(int)height;
        void *args[] = { &d_rgba, &d_Y_pix, &d_Cb_full, &d_Cr_full, &iw, &ih, NULL };
        unsigned int gx = CEIL_DIV(width,  BLOCK_DIM_PIX);
        unsigned int gy = CEIL_DIV(height, BLOCK_DIM_PIX);
        ok = launch2d(s_fn_rgba_to_ycbcr, gx, gy, BLOCK_DIM_PIX, args);
    }
    if (!ok) goto cleanup;

    /* 3. Downsample chroma if needed */
    if (subsampling != HapYCbCrSubsampling_444) {
        int isw = (int)width, ish = (int)height;
        CUfunction down_fn = (subsampling == HapYCbCrSubsampling_422)
                             ? s_fn_chroma_down_422 : s_fn_chroma_down_420;
        void *args_cb[] = { &d_Cb_full, &d_Cb_pix, &isw, &ish, NULL };
        void *args_cr[] = { &d_Cr_full, &d_Cr_pix, &isw, &ish, NULL };
        unsigned int gx = CEIL_DIV(cb_w, BLOCK_DIM_PIX);
        unsigned int gy = CEIL_DIV(cb_h, BLOCK_DIM_PIX);
        ok = launch2d(down_fn, gx, gy, BLOCK_DIM_PIX, args_cb);
        if (ok) ok = launch2d(down_fn, gx, gy, BLOCK_DIM_PIX, args_cr);
        if (!ok) goto cleanup;
    } else {
        d_Cb_pix = d_Cb_full;
        d_Cr_pix = d_Cr_full;
    }

    /* 4. Encode each plane → BC4 */
    {
        int ibx=(int)bx, iby=(int)by, iw=(int)width, ih=(int)height;
        void *args[] = { &d_Y_pix, &d_Y_bc, &ibx, &iby, &iw, &ih, NULL };
        unsigned int gx = CEIL_DIV(bx, BLOCK_DIM_BC);
        unsigned int gy = CEIL_DIV(by, BLOCK_DIM_BC);
        ok = launch2d(s_fn_bc4_encode, gx, gy, BLOCK_DIM_BC, args);
    }
    if (!ok) goto cleanup;
    {
        int icbx=(int)cb_bx, icby=(int)cb_by, icw=(int)cb_w, ich=(int)cb_h;
        void *args_cb[] = { &d_Cb_pix, &d_Cb_bc, &icbx, &icby, &icw, &ich, NULL };
        void *args_cr[] = { &d_Cr_pix, &d_Cr_bc, &icbx, &icby, &icw, &ich, NULL };
        unsigned int gx = CEIL_DIV(cb_bx, BLOCK_DIM_BC);
        unsigned int gy = CEIL_DIV(cb_by, BLOCK_DIM_BC);
        ok = launch2d(s_fn_bc4_encode, gx, gy, BLOCK_DIM_BC, args_cb);
        if (ok) ok = launch2d(s_fn_bc4_encode, gx, gy, BLOCK_DIM_BC, args_cr);
    }
    if (!ok) goto cleanup;

    /* Synchronise before downloading results. */
    ok = (cuCtxSynchronize() == CUDA_SUCCESS);
    if (!ok) goto cleanup;

    /* Download outputs. */
    ok = (cuMemcpyDtoH(output_y,  d_Y_bc,  output_y_bytes)  == CUDA_SUCCESS) &&
         (cuMemcpyDtoH(output_cb, d_Cb_bc, output_cb_bytes) == CUDA_SUCCESS) &&
         (cuMemcpyDtoH(output_cr, d_Cr_bc, output_cr_bytes) == CUDA_SUCCESS);

cleanup:
    cuMemFree(d_input_bc);
    cuMemFree(d_rgba);
    cuMemFree(d_Y_pix);
    cuMemFree(d_Cb_full);
    cuMemFree(d_Cr_full);
    if (subsampling != HapYCbCrSubsampling_444) {
        cuMemFree(d_Cb_pix);
        cuMemFree(d_Cr_pix);
    }
    cuMemFree(d_Y_bc);
    cuMemFree(d_Cb_bc);
    cuMemFree(d_Cr_bc);
    return ok;
}

/* ── YCbCr → RGB ────────────────────────────────────────────────────────── */

int hap_cuda_convert_ycbcr_to_rgb(
    const void  *input_y,       unsigned long  input_y_bytes,
    const void  *input_cb,      unsigned long  input_cb_bytes,
    const void  *input_cr,      unsigned long  input_cr_bytes,
    unsigned int subsampling,
    unsigned int width,         unsigned int   height,
    unsigned int output_format,
    void        *output_bc,     unsigned long  output_bc_bytes)
{
    if (!s_init_ok) return 0;

    unsigned int bx = CEIL_DIV(width,  4);
    unsigned int by = CEIL_DIV(height, 4);
    unsigned int cb_w, cb_h;
    hap_chroma_plane_dims(width, height, subsampling, &cb_w, &cb_h);

    size_t y_pix_bytes  = (size_t)width  * height;
    size_t cb_pix_bytes = (size_t)cb_w   * cb_h;
    size_t rgba_bytes   = (size_t)width  * height * 4;

    CUdeviceptr d_Y_bc   = 0, d_Cb_bc  = 0, d_Cr_bc  = 0;
    CUdeviceptr d_Y_pix  = 0, d_Cb_pix = 0, d_Cr_pix = 0;
    CUdeviceptr d_rgba   = 0;
    CUdeviceptr d_out_bc = 0;
    int ok = 1;

#define ALLOC(ptr, sz) \
    if (ok) { ok = (cuMemAlloc(&(ptr), (sz)) == CUDA_SUCCESS); }

    ALLOC(d_Y_bc,   input_y_bytes)
    ALLOC(d_Cb_bc,  input_cb_bytes)
    ALLOC(d_Cr_bc,  input_cr_bytes)
    ALLOC(d_Y_pix,  y_pix_bytes)
    ALLOC(d_Cb_pix, cb_pix_bytes)
    ALLOC(d_Cr_pix, cb_pix_bytes)
    ALLOC(d_rgba,   rgba_bytes)
    ALLOC(d_out_bc, output_bc_bytes)
#undef ALLOC

    if (!ok) goto cleanup;

    /* Upload BC4 planes. */
    ok = (cuMemcpyHtoD(d_Y_bc,  input_y,  input_y_bytes)  == CUDA_SUCCESS) &&
         (cuMemcpyHtoD(d_Cb_bc, input_cb, input_cb_bytes) == CUDA_SUCCESS) &&
         (cuMemcpyHtoD(d_Cr_bc, input_cr, input_cr_bytes) == CUDA_SUCCESS);
    if (!ok) goto cleanup;

    /* 1. Decode BC4 planes → single-channel pixels */
    {
        unsigned int ybx = CEIL_DIV(bx, BLOCK_DIM_BC);
        unsigned int yby = CEIL_DIV(by, BLOCK_DIM_BC);
        int ibx=(int)bx, iby=(int)by, iw=(int)width, ih=(int)height;
        void *args[] = { &d_Y_bc, &d_Y_pix, &ibx, &iby, &iw, &ih, NULL };
        ok = launch2d(s_fn_bc4_decode, ybx, yby, BLOCK_DIM_BC, args);
    }
    if (!ok) goto cleanup;
    {
        int icb_bx=(int)CEIL_DIV(cb_w,4), icb_by=(int)CEIL_DIV(cb_h,4);
        int icw=(int)cb_w, ich=(int)cb_h;
        unsigned int gx=CEIL_DIV(icb_bx,BLOCK_DIM_BC), gy=CEIL_DIV(icb_by,BLOCK_DIM_BC);
        void *args_cb[] = { &d_Cb_bc, &d_Cb_pix, &icb_bx, &icb_by, &icw, &ich, NULL };
        void *args_cr[] = { &d_Cr_bc, &d_Cr_pix, &icb_bx, &icb_by, &icw, &ich, NULL };
        ok = launch2d(s_fn_bc4_decode, gx, gy, BLOCK_DIM_BC, args_cb);
        if (ok) ok = launch2d(s_fn_bc4_decode, gx, gy, BLOCK_DIM_BC, args_cr);
    }
    if (!ok) goto cleanup;

    /* 2. YCbCr → RGBA (coordinate mapping handles upsampling) */
    {
        int yw=(int)width, yh=(int)height, cw=(int)cb_w, ch=(int)cb_h;
        void *args[] = { &d_Y_pix, &d_Cb_pix, &d_Cr_pix, &d_rgba, &yw, &yh, &cw, &ch, NULL };
        unsigned int gx=CEIL_DIV(width, BLOCK_DIM_PIX), gy=CEIL_DIV(height, BLOCK_DIM_PIX);
        ok = launch2d(s_fn_ycbcr_to_rgba, gx, gy, BLOCK_DIM_PIX, args);
    }
    if (!ok) goto cleanup;

    /* 3. Encode RGBA → target BC format */
    {
        int ibx=(int)bx, iby=(int)by, iw=(int)width, ih=(int)height;
        void *args[] = { &d_rgba, &d_out_bc, &ibx, &iby, &iw, &ih, NULL };
        CUfunction enc_fn = (output_format == HapTextureFormat_RGB_DXT1)
                            ? s_fn_bc1_encode : s_fn_bc3_encode;
        unsigned int gx=CEIL_DIV(bx,BLOCK_DIM_BC), gy=CEIL_DIV(by,BLOCK_DIM_BC);
        ok = launch2d(enc_fn, gx, gy, BLOCK_DIM_BC, args);
    }
    if (!ok) goto cleanup;

    ok = (cuCtxSynchronize() == CUDA_SUCCESS);
    if (!ok) goto cleanup;

    ok = (cuMemcpyDtoH(output_bc, d_out_bc, output_bc_bytes) == CUDA_SUCCESS);

cleanup:
    cuMemFree(d_Y_bc);   cuMemFree(d_Cb_bc);  cuMemFree(d_Cr_bc);
    cuMemFree(d_Y_pix);  cuMemFree(d_Cb_pix); cuMemFree(d_Cr_pix);
    cuMemFree(d_rgba);   cuMemFree(d_out_bc);
    return ok;
}

#else /* !HAVE_CUDA */

int hap_cuda_available(void) { return 0; }

int hap_cuda_convert_rgb_to_ycbcr(
    const void *input_bc, unsigned long input_bytes,
    unsigned int input_format, unsigned int width, unsigned int height,
    unsigned int subsampling,
    void *output_y, void *output_cb, void *output_cr,
    unsigned long output_y_bytes, unsigned long output_cb_bytes,
    unsigned long output_cr_bytes)
{
    (void)input_bc; (void)input_bytes; (void)input_format;
    (void)width; (void)height; (void)subsampling;
    (void)output_y; (void)output_cb; (void)output_cr;
    (void)output_y_bytes; (void)output_cb_bytes; (void)output_cr_bytes;
    return 0;
}

int hap_cuda_convert_ycbcr_to_rgb(
    const void *input_y, unsigned long input_y_bytes,
    const void *input_cb, unsigned long input_cb_bytes,
    const void *input_cr, unsigned long input_cr_bytes,
    unsigned int subsampling, unsigned int width, unsigned int height,
    unsigned int output_format, void *output_bc, unsigned long output_bc_bytes)
{
    (void)input_y; (void)input_y_bytes;
    (void)input_cb; (void)input_cb_bytes;
    (void)input_cr; (void)input_cr_bytes;
    (void)subsampling; (void)width; (void)height;
    (void)output_format; (void)output_bc; (void)output_bc_bytes;
    return 0;
}

#endif /* HAVE_CUDA */
