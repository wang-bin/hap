/*
 hap_convert.c

 Convert between BC-compressed RGB/RGBA textures and BC4/RGTC1-compressed
 YCbCr planar textures.

 Public entry points:
   HapYCbCrPlaneByteSize()
   HapConvertTextureRGBToYCbCr()
   HapConvertTextureYCbCrToRGB()

 When compiled with -DHAVE_WEBGPU, the WebGPU C API + WGSL compute backend
 (hap_convert_wgpu.c) is tried first.  When compiled with -DHAVE_CUDA, the
 CUDA driver API backend (hap_convert_cuda.c) is then tried.  On failure or
 absence, the CPU path below is used as a fallback.

 Colour space: BT.709 full-range (0-255 for both luma and chroma).
 */

#include "hap_convert.h"
#include "hap.h"

#ifdef HAVE_WEBGPU
#  include "hap_convert_wgpu.h"
#endif

#ifdef HAVE_CUDA
#  include "hap_convert_cuda.h"
#endif

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

/* ═══════════════════════════════════════════════════════════════════════════
   Public helper
   ═══════════════════════════════════════════════════════════════════════════ */

unsigned long HapYCbCrPlaneByteSize(unsigned int width, unsigned int height,
                                    unsigned int subsampling, int isChroma)
{
    if (width == 0 || height == 0 || width % 4 != 0 || height % 4 != 0)
        return 0;
    if (isChroma) {
        if (subsampling == HapYCbCrSubsampling_422 ||
            subsampling == HapYCbCrSubsampling_420)
            width /= 2;
        if (subsampling == HapYCbCrSubsampling_420)
            height /= 2;
    }
    return hap_bc4_plane_bytes(width, height);
}

/* ═══════════════════════════════════════════════════════════════════════════
   BC4/RGTC1 block codec (CPU)
   ═══════════════════════════════════════════════════════════════════════════ */

void hap_decode_bc4_block(const uint8_t *block, uint8_t out[16])
{
    uint8_t r0 = block[0], r1 = block[1];
    uint8_t pal[8];
    pal[0] = r0;
    pal[1] = r1;
    if (r0 > r1) {
        pal[2] = (uint8_t)((6 * r0 + 1 * r1 + 3) / 7);
        pal[3] = (uint8_t)((5 * r0 + 2 * r1 + 3) / 7);
        pal[4] = (uint8_t)((4 * r0 + 3 * r1 + 3) / 7);
        pal[5] = (uint8_t)((3 * r0 + 4 * r1 + 3) / 7);
        pal[6] = (uint8_t)((2 * r0 + 5 * r1 + 3) / 7);
        pal[7] = (uint8_t)((1 * r0 + 6 * r1 + 3) / 7);
    } else {
        pal[2] = (uint8_t)((4 * r0 + 1 * r1 + 2) / 5);
        pal[3] = (uint8_t)((3 * r0 + 2 * r1 + 2) / 5);
        pal[4] = (uint8_t)((2 * r0 + 3 * r1 + 2) / 5);
        pal[5] = (uint8_t)((1 * r0 + 4 * r1 + 2) / 5);
        pal[6] = 0;
        pal[7] = 255;
    }
    /* 48-bit index table stored little-endian in bytes 2-7 */
    uint64_t bits = 0;
    for (int i = 0; i < 6; i++)
        bits |= (uint64_t)block[2 + i] << (i * 8);
    for (int i = 0; i < 16; i++)
        out[i] = pal[(bits >> (i * 3)) & 0x7];
}

void hap_encode_bc4_block(const uint8_t in[16], uint8_t *block)
{
    uint8_t mn = 255, mx = 0;
    for (int i = 0; i < 16; i++) {
        if (in[i] < mn) mn = in[i];
        if (in[i] > mx) mx = in[i];
    }
    /* r0 = max, r1 = min → r0 > r1 → 8-value interpolation mode */
    block[0] = mx;
    block[1] = mn;
    if (mx == mn) {
        memset(block + 2, 0, 6);
        return;
    }
    uint8_t pal[8];
    pal[0] = mx; pal[1] = mn;
    pal[2] = (uint8_t)((6 * mx + 1 * mn + 3) / 7);
    pal[3] = (uint8_t)((5 * mx + 2 * mn + 3) / 7);
    pal[4] = (uint8_t)((4 * mx + 3 * mn + 3) / 7);
    pal[5] = (uint8_t)((3 * mx + 4 * mn + 3) / 7);
    pal[6] = (uint8_t)((2 * mx + 5 * mn + 3) / 7);
    pal[7] = (uint8_t)((1 * mx + 6 * mn + 3) / 7);
    uint64_t bits = 0;
    for (int i = 0; i < 16; i++) {
        uint8_t best = 0;
        int best_d = 256 * 256;
        for (int j = 0; j < 8; j++) {
            int d = (int)in[i] - pal[j];
            if (d * d < best_d) { best_d = d * d; best = (uint8_t)j; }
        }
        bits |= (uint64_t)best << (i * 3);
    }
    for (int i = 0; i < 6; i++)
        block[2 + i] = (uint8_t)((bits >> (i * 8)) & 0xFF);
}

/* ═══════════════════════════════════════════════════════════════════════════
   BC1/DXT1 block codec (CPU)
   ═══════════════════════════════════════════════════════════════════════════ */

static void unpack_rgb565(uint16_t c, uint8_t *r, uint8_t *g, uint8_t *b)
{
    *r = (uint8_t)(((c >> 11) & 0x1F) * 255 / 31);
    *g = (uint8_t)(((c >>  5) & 0x3F) * 255 / 63);
    *b = (uint8_t)( (c        & 0x1F) * 255 / 31);
}

static uint16_t pack_rgb565(uint8_t r, uint8_t g, uint8_t b)
{
    return ((uint16_t)(r >> 3) << 11) |
           ((uint16_t)(g >> 2) <<  5) |
            (uint16_t)(b >> 3);
}

void hap_decode_bc1_block(const uint8_t *block, uint8_t out[64])
{
    uint16_t c0 = (uint16_t)(block[0] | ((uint16_t)block[1] << 8));
    uint16_t c1 = (uint16_t)(block[2] | ((uint16_t)block[3] << 8));
    uint8_t r[4], g[4], b[4], a[4];
    a[0] = a[1] = a[2] = 255;
    unpack_rgb565(c0, &r[0], &g[0], &b[0]);
    unpack_rgb565(c1, &r[1], &g[1], &b[1]);
    if (c0 > c1) {
        r[2] = (uint8_t)((2*r[0]+r[1]+1)/3); g[2] = (uint8_t)((2*g[0]+g[1]+1)/3); b[2] = (uint8_t)((2*b[0]+b[1]+1)/3);
        r[3] = (uint8_t)((r[0]+2*r[1]+1)/3); g[3] = (uint8_t)((g[0]+2*g[1]+1)/3); b[3] = (uint8_t)((b[0]+2*b[1]+1)/3);
        a[3] = 255;
    } else {
        r[2] = (uint8_t)((r[0]+r[1])/2); g[2] = (uint8_t)((g[0]+g[1])/2); b[2] = (uint8_t)((b[0]+b[1])/2);
        r[3] = g[3] = b[3] = a[3] = 0; /* transparent black */
    }
    uint32_t idx32 = (uint32_t)block[4] | ((uint32_t)block[5] << 8) |
                     ((uint32_t)block[6] << 16) | ((uint32_t)block[7] << 24);
    for (int i = 0; i < 16; i++) {
        uint8_t j = (idx32 >> (i * 2)) & 0x3;
        out[i*4+0] = r[j]; out[i*4+1] = g[j];
        out[i*4+2] = b[j]; out[i*4+3] = a[j];
    }
}

void hap_encode_bc1_block(const uint8_t in[64], uint8_t *block)
{
    uint8_t rn=255,rx=0, gn=255,gx=0, bn=255,bx=0;
    for (int i = 0; i < 16; i++) {
        uint8_t rv=in[i*4], gv=in[i*4+1], bv=in[i*4+2];
        if (rv < rn) rn = rv;
        if (rv > rx) rx = rv;
        if (gv < gn) gn = gv;
        if (gv > gx) gx = gv;
        if (bv < bn) bn = bv;
        if (bv > bx) bx = bv;
    }
    uint16_t c0 = pack_rgb565(rx, gx, bx);
    uint16_t c1 = pack_rgb565(rn, gn, bn);
    if (c0 <= c1) { uint16_t t = c0; c0 = c1; c1 = t; } /* force 4-color mode */
    uint8_t pr[4], pg[4], pb[4];
    unpack_rgb565(c0, &pr[0], &pg[0], &pb[0]);
    unpack_rgb565(c1, &pr[1], &pg[1], &pb[1]);
    pr[2]=(uint8_t)((2*pr[0]+pr[1]+1)/3); pg[2]=(uint8_t)((2*pg[0]+pg[1]+1)/3); pb[2]=(uint8_t)((2*pb[0]+pb[1]+1)/3);
    pr[3]=(uint8_t)((pr[0]+2*pr[1]+1)/3); pg[3]=(uint8_t)((pg[0]+2*pg[1]+1)/3); pb[3]=(uint8_t)((pb[0]+2*pb[1]+1)/3);
    uint32_t indices = 0;
    for (int i = 0; i < 16; i++) {
        int best = 0, bd = 256*256*3;
        for (int j = 0; j < 4; j++) {
            int dr=in[i*4]-pr[j], dg=in[i*4+1]-pg[j], db=in[i*4+2]-pb[j];
            int d = dr*dr+dg*dg+db*db;
            if (d < bd) { bd = d; best = j; }
        }
        indices |= (uint32_t)best << (i * 2);
    }
    block[0]=c0&0xFF; block[1]=(c0>>8)&0xFF;
    block[2]=c1&0xFF; block[3]=(c1>>8)&0xFF;
    block[4]= indices     &0xFF; block[5]=(indices>> 8)&0xFF;
    block[6]=(indices>>16)&0xFF; block[7]=(indices>>24)&0xFF;
}

/* ═══════════════════════════════════════════════════════════════════════════
   BC3/DXT5 block codec (CPU)
   ═══════════════════════════════════════════════════════════════════════════ */

void hap_decode_bc3_block(const uint8_t *block, uint8_t out[64])
{
    /* Alpha sub-block: bytes 0-7 (BC4-like) */
    uint8_t a0 = block[0], a1 = block[1];
    uint8_t apal[8];
    apal[0] = a0; apal[1] = a1;
    if (a0 > a1) {
        apal[2]=(uint8_t)((6*a0+1*a1+3)/7); apal[3]=(uint8_t)((5*a0+2*a1+3)/7);
        apal[4]=(uint8_t)((4*a0+3*a1+3)/7); apal[5]=(uint8_t)((3*a0+4*a1+3)/7);
        apal[6]=(uint8_t)((2*a0+5*a1+3)/7); apal[7]=(uint8_t)((1*a0+6*a1+3)/7);
    } else {
        apal[2]=(uint8_t)((4*a0+1*a1+2)/5); apal[3]=(uint8_t)((3*a0+2*a1+2)/5);
        apal[4]=(uint8_t)((2*a0+3*a1+2)/5); apal[5]=(uint8_t)((1*a0+4*a1+2)/5);
        apal[6] = 0; apal[7] = 255;
    }
    uint64_t abits = 0;
    for (int i = 0; i < 6; i++) abits |= (uint64_t)block[2+i] << (i*8);

    /* Colour sub-block: bytes 8-15 (BC1-like, always 4-color in BC3) */
    const uint8_t *cblk = block + 8;
    uint16_t c0 = (uint16_t)(cblk[0] | ((uint16_t)cblk[1] << 8));
    uint16_t c1 = (uint16_t)(cblk[2] | ((uint16_t)cblk[3] << 8));
    uint8_t r[4], g[4], b[4];
    unpack_rgb565(c0, &r[0], &g[0], &b[0]);
    unpack_rgb565(c1, &r[1], &g[1], &b[1]);
    /* BC3 colour block always uses 4-color interpolation */
    r[2]=(uint8_t)((2*r[0]+r[1]+1)/3); g[2]=(uint8_t)((2*g[0]+g[1]+1)/3); b[2]=(uint8_t)((2*b[0]+b[1]+1)/3);
    r[3]=(uint8_t)((r[0]+2*r[1]+1)/3); g[3]=(uint8_t)((g[0]+2*g[1]+1)/3); b[3]=(uint8_t)((b[0]+2*b[1]+1)/3);
    uint32_t cidx = (uint32_t)cblk[4] | ((uint32_t)cblk[5]<<8) |
                    ((uint32_t)cblk[6]<<16) | ((uint32_t)cblk[7]<<24);
    for (int i = 0; i < 16; i++) {
        uint8_t ci = (cidx  >> (i*2)) & 0x3;
        uint8_t ai = (abits >> (i*3)) & 0x7;
        out[i*4+0]=r[ci]; out[i*4+1]=g[ci];
        out[i*4+2]=b[ci]; out[i*4+3]=apal[ai];
    }
}

void hap_encode_bc3_block(const uint8_t in[64], uint8_t *block)
{
    /* Alpha block (BC4-like) */
    uint8_t an=255, ax=0;
    for (int i = 0; i < 16; i++) {
        uint8_t a = in[i*4+3];
        if (a < an) an = a;
        if (a > ax) ax = a;
    }
    block[0] = ax; block[1] = an;
    if (ax == an) {
        memset(block + 2, 0, 6);
    } else {
        uint8_t ap[8];
        ap[0]=ax; ap[1]=an;
        ap[2]=(uint8_t)((6*ax+1*an+3)/7); ap[3]=(uint8_t)((5*ax+2*an+3)/7);
        ap[4]=(uint8_t)((4*ax+3*an+3)/7); ap[5]=(uint8_t)((3*ax+4*an+3)/7);
        ap[6]=(uint8_t)((2*ax+5*an+3)/7); ap[7]=(uint8_t)((1*ax+6*an+3)/7);
        uint64_t abits = 0;
        for (int i = 0; i < 16; i++) {
            uint8_t best = 0; int bd = 256*256;
            for (int j = 0; j < 8; j++) {
                int d = (int)in[i*4+3] - ap[j];
                if (d*d < bd) { bd = d*d; best = (uint8_t)j; }
            }
            abits |= (uint64_t)best << (i*3);
        }
        for (int i = 0; i < 6; i++)
            block[2+i] = (uint8_t)((abits >> (i*8)) & 0xFF);
    }
    /* Colour block (BC1-like, 16 bytes offset by 8) */
    hap_encode_bc1_block(in, block + 8);
    /* Force 4-color mode in BC3: ensure colour endpoint 0 > endpoint 1.
       hap_encode_bc1_block already guarantees this. */
}

/* ═══════════════════════════════════════════════════════════════════════════
   Full-texture CPU codec wrappers
   ═══════════════════════════════════════════════════════════════════════════ */

unsigned int hap_cpu_decode_bc1(const void *input, unsigned long input_bytes,
                                 unsigned int width, unsigned int height,
                                 uint8_t *rgba_out)
{
    unsigned int bx = (width  + 3) / 4;
    unsigned int by = (height + 3) / 4;
    if (input_bytes < hap_bc1_plane_bytes(width, height)) return HapResult_Bad_Frame;
    const uint8_t *src = (const uint8_t *)input;
    for (unsigned int row = 0; row < by; row++) {
        for (unsigned int col = 0; col < bx; col++) {
            uint8_t pix[64];
            hap_decode_bc1_block(src + (row * bx + col) * 8, pix);
            for (int py = 0; py < 4; py++) {
                for (int px = 0; px < 4; px++) {
                    unsigned int ox = col * 4 + px;
                    unsigned int oy = row * 4 + py;
                    if (ox < width && oy < height) {
                        uint8_t *dst = rgba_out + (oy * width + ox) * 4;
                        memcpy(dst, pix + (py * 4 + px) * 4, 4);
                    }
                }
            }
        }
    }
    return HapResult_No_Error;
}

unsigned int hap_cpu_decode_bc3(const void *input, unsigned long input_bytes,
                                 unsigned int width, unsigned int height,
                                 uint8_t *rgba_out)
{
    unsigned int bx = (width  + 3) / 4;
    unsigned int by = (height + 3) / 4;
    if (input_bytes < hap_bc3_plane_bytes(width, height)) return HapResult_Bad_Frame;
    const uint8_t *src = (const uint8_t *)input;
    for (unsigned int row = 0; row < by; row++) {
        for (unsigned int col = 0; col < bx; col++) {
            uint8_t pix[64];
            hap_decode_bc3_block(src + (row * bx + col) * 16, pix);
            for (int py = 0; py < 4; py++) {
                for (int px = 0; px < 4; px++) {
                    unsigned int ox = col * 4 + px;
                    unsigned int oy = row * 4 + py;
                    if (ox < width && oy < height) {
                        uint8_t *dst = rgba_out + (oy * width + ox) * 4;
                        memcpy(dst, pix + (py * 4 + px) * 4, 4);
                    }
                }
            }
        }
    }
    return HapResult_No_Error;
}

unsigned int hap_cpu_decode_bc4(const void *input, unsigned long input_bytes,
                                 unsigned int width, unsigned int height,
                                 uint8_t *luma_out)
{
    unsigned int bx = (width  + 3) / 4;
    unsigned int by = (height + 3) / 4;
    if (input_bytes < hap_bc4_plane_bytes(width, height)) return HapResult_Bad_Frame;
    const uint8_t *src = (const uint8_t *)input;
    for (unsigned int row = 0; row < by; row++) {
        for (unsigned int col = 0; col < bx; col++) {
            uint8_t pix[16];
            hap_decode_bc4_block(src + (row * bx + col) * 8, pix);
            for (int py = 0; py < 4; py++) {
                for (int px = 0; px < 4; px++) {
                    unsigned int ox = col * 4 + px;
                    unsigned int oy = row * 4 + py;
                    if (ox < width && oy < height)
                        luma_out[oy * width + ox] = pix[py * 4 + px];
                }
            }
        }
    }
    return HapResult_No_Error;
}

unsigned int hap_cpu_encode_bc1(const uint8_t *rgba,
                                 unsigned int width, unsigned int height,
                                 void *output, unsigned long output_bytes)
{
    unsigned int bx = (width  + 3) / 4;
    unsigned int by = (height + 3) / 4;
    if (output_bytes < hap_bc1_plane_bytes(width, height)) return HapResult_Buffer_Too_Small;
    uint8_t *dst = (uint8_t *)output;
    for (unsigned int row = 0; row < by; row++) {
        for (unsigned int col = 0; col < bx; col++) {
            uint8_t pix[64] = {0};
            for (int py = 0; py < 4; py++) {
                for (int px = 0; px < 4; px++) {
                    unsigned int ox = col * 4 + px;
                    unsigned int oy = row * 4 + py;
                    if (ox < width && oy < height)
                        memcpy(pix + (py*4+px)*4, rgba + (oy*width+ox)*4, 4);
                }
            }
            hap_encode_bc1_block(pix, dst + (row * bx + col) * 8);
        }
    }
    return HapResult_No_Error;
}

unsigned int hap_cpu_encode_bc3(const uint8_t *rgba,
                                 unsigned int width, unsigned int height,
                                 void *output, unsigned long output_bytes)
{
    unsigned int bx = (width  + 3) / 4;
    unsigned int by = (height + 3) / 4;
    if (output_bytes < hap_bc3_plane_bytes(width, height)) return HapResult_Buffer_Too_Small;
    uint8_t *dst = (uint8_t *)output;
    for (unsigned int row = 0; row < by; row++) {
        for (unsigned int col = 0; col < bx; col++) {
            uint8_t pix[64] = {0};
            for (int py = 0; py < 4; py++) {
                for (int px = 0; px < 4; px++) {
                    unsigned int ox = col * 4 + px;
                    unsigned int oy = row * 4 + py;
                    if (ox < width && oy < height)
                        memcpy(pix + (py*4+px)*4, rgba + (oy*width+ox)*4, 4);
                }
            }
            hap_encode_bc3_block(pix, dst + (row * bx + col) * 16);
        }
    }
    return HapResult_No_Error;
}

unsigned int hap_cpu_encode_bc4(const uint8_t *luma,
                                 unsigned int width, unsigned int height,
                                 void *output, unsigned long output_bytes)
{
    unsigned int bx = (width  + 3) / 4;
    unsigned int by = (height + 3) / 4;
    if (output_bytes < hap_bc4_plane_bytes(width, height)) return HapResult_Buffer_Too_Small;
    uint8_t *dst = (uint8_t *)output;
    for (unsigned int row = 0; row < by; row++) {
        for (unsigned int col = 0; col < bx; col++) {
            uint8_t pix[16] = {0};
            for (int py = 0; py < 4; py++) {
                for (int px = 0; px < 4; px++) {
                    unsigned int ox = col * 4 + px;
                    unsigned int oy = row * 4 + py;
                    if (ox < width && oy < height)
                        pix[py*4+px] = luma[oy * width + ox];
                }
            }
            hap_encode_bc4_block(pix, dst + (row * bx + col) * 8);
        }
    }
    return HapResult_No_Error;
}

/* ═══════════════════════════════════════════════════════════════════════════
   BT.709 colour-space conversion (CPU)
   ═══════════════════════════════════════════════════════════════════════════ */

/* Clamp a float to [0,255] and convert to uint8. */
static uint8_t f2u8(float v)
{
    if (v < 0.0f) return 0;
    if (v > 255.0f) return 255;
    return (uint8_t)(v + 0.5f);
}

void hap_cpu_rgba_to_ycbcr(const uint8_t *rgba,
                             unsigned int width, unsigned int height,
                             uint8_t *Y, uint8_t *Cb, uint8_t *Cr)
{
    for (unsigned int i = 0; i < width * height; i++) {
        float r = rgba[i*4+0] / 255.0f;
        float g = rgba[i*4+1] / 255.0f;
        float b = rgba[i*4+2] / 255.0f;
        /* BT.709 full-range */
        float y  =  0.2126f * r + 0.7152f * g + 0.0722f * b;
        float cb = -0.1146f * r - 0.3854f * g + 0.5000f * b;
        float cr =  0.5000f * r - 0.4542f * g - 0.0458f * b;
        Y [i] = f2u8(y  * 255.0f);
        Cb[i] = f2u8(cb * 255.0f + 128.0f);
        Cr[i] = f2u8(cr * 255.0f + 128.0f);
    }
}

void hap_cpu_ycbcr_to_rgba(const uint8_t *Y,
                             const uint8_t *Cb, const uint8_t *Cr,
                             unsigned int y_width,  unsigned int y_height,
                             unsigned int cb_width, unsigned int cb_height,
                             uint8_t *rgba_out)
{
    for (unsigned int py = 0; py < y_height; py++) {
        for (unsigned int px = 0; px < y_width; px++) {
            unsigned int cx = px * cb_width  / y_width;
            unsigned int cy = py * cb_height / y_height;
            float luma =  Y [py * y_width  + px]          / 255.0f;
            float cb_v = (Cb[cy * cb_width + cx] - 128.0f) / 255.0f;
            float cr_v = (Cr[cy * cb_width + cx] - 128.0f) / 255.0f;
            /* BT.709 inverse */
            float r = luma + 1.5748f * cr_v;
            float g = luma - 0.1873f * cb_v - 0.4681f * cr_v;
            float bv = luma + 1.8556f * cb_v;
            unsigned int off = (py * y_width + px) * 4;
            rgba_out[off+0] = f2u8(r  * 255.0f);
            rgba_out[off+1] = f2u8(g  * 255.0f);
            rgba_out[off+2] = f2u8(bv * 255.0f);
            rgba_out[off+3] = 255;
        }
    }
}

void hap_cpu_chroma_downsample_422(const uint8_t *src,
                                    unsigned int src_width, unsigned int height,
                                    uint8_t *dst)
{
    unsigned int dst_width = src_width / 2;
    for (unsigned int y = 0; y < height; y++)
        for (unsigned int x = 0; x < dst_width; x++)
            dst[y * dst_width + x] = (uint8_t)
                ((src[y*src_width + 2*x] + src[y*src_width + 2*x+1] + 1) / 2);
}

void hap_cpu_chroma_downsample_420(const uint8_t *src,
                                    unsigned int src_width,
                                    unsigned int src_height,
                                    uint8_t *dst)
{
    unsigned int dw = src_width / 2;
    unsigned int dh = src_height / 2;
    for (unsigned int y = 0; y < dh; y++)
        for (unsigned int x = 0; x < dw; x++) {
            unsigned int a = src[(2*y)  *src_width + 2*x];
            unsigned int b = src[(2*y)  *src_width + 2*x + 1];
            unsigned int c = src[(2*y+1)*src_width + 2*x];
            unsigned int d = src[(2*y+1)*src_width + 2*x + 1];
            dst[y * dw + x] = (uint8_t)((a + b + c + d + 2) / 4);
        }
}

/* ═══════════════════════════════════════════════════════════════════════════
   CPU conversion pipeline
   ═══════════════════════════════════════════════════════════════════════════ */

static unsigned int cpu_convert_rgb_to_ycbcr(
    const void *inputBuffer, unsigned long inputBufferBytes,
    unsigned int inputTextureFormat,
    unsigned int width, unsigned int height,
    unsigned int subsampling,
    void *outputY,  unsigned long outputYBytes,  unsigned long *outputYBytesUsed,
    void *outputCb, unsigned long outputCbBytes, unsigned long *outputCbBytesUsed,
    void *outputCr, unsigned long outputCrBytes, unsigned long *outputCrBytesUsed)
{
    unsigned int cb_w, cb_h;
    hap_chroma_plane_dims(width, height, subsampling, &cb_w, &cb_h);

    /* Allocate intermediate RGBA and full-resolution YCbCr buffers. */
    size_t pixel_count = (size_t)width * height;
    uint8_t *rgba = (uint8_t *)malloc(pixel_count * 4);
    uint8_t *Y_full  = (uint8_t *)malloc(pixel_count);
    uint8_t *Cb_full = (uint8_t *)malloc(pixel_count);
    uint8_t *Cr_full = (uint8_t *)malloc(pixel_count);
    if (!rgba || !Y_full || !Cb_full || !Cr_full) {
        free(rgba); free(Y_full); free(Cb_full); free(Cr_full);
        return HapResult_Internal_Error;
    }

    /* 1. Decode BC texture → RGBA */
    unsigned int result;
    if (inputTextureFormat == HapTextureFormat_RGB_DXT1)
        result = hap_cpu_decode_bc1(inputBuffer, inputBufferBytes, width, height, rgba);
    else /* HapTextureFormat_RGBA_DXT5 */
        result = hap_cpu_decode_bc3(inputBuffer, inputBufferBytes, width, height, rgba);

    /* 2. RGBA → full-resolution YCbCr */
    if (result == HapResult_No_Error)
        hap_cpu_rgba_to_ycbcr(rgba, width, height, Y_full, Cb_full, Cr_full);
    free(rgba);

    /* 3. Downsample chroma if needed */
    uint8_t *Cb_plane = Cb_full;
    uint8_t *Cr_plane = Cr_full;
    uint8_t *Cb_down = NULL;
    uint8_t *Cr_down = NULL;
    if (result == HapResult_No_Error && subsampling != HapYCbCrSubsampling_444) {
        size_t cb_pixels = (size_t)cb_w * cb_h;
        Cb_down = (uint8_t *)malloc(cb_pixels);
        Cr_down = (uint8_t *)malloc(cb_pixels);
        if (!Cb_down || !Cr_down) {
            result = HapResult_Internal_Error;
        } else if (subsampling == HapYCbCrSubsampling_422) {
            hap_cpu_chroma_downsample_422(Cb_full, width, height, Cb_down);
            hap_cpu_chroma_downsample_422(Cr_full, width, height, Cr_down);
        } else {
            hap_cpu_chroma_downsample_420(Cb_full, width, height, Cb_down);
            hap_cpu_chroma_downsample_420(Cr_full, width, height, Cr_down);
        }
        Cb_plane = Cb_down;
        Cr_plane = Cr_down;
    }

    /* 4. Encode planes → BC4 */
    if (result == HapResult_No_Error)
        result = hap_cpu_encode_bc4(Y_full, width, height, outputY, outputYBytes);
    if (result == HapResult_No_Error)
        result = hap_cpu_encode_bc4(Cb_plane, cb_w, cb_h, outputCb, outputCbBytes);
    if (result == HapResult_No_Error)
        result = hap_cpu_encode_bc4(Cr_plane, cb_w, cb_h, outputCr, outputCrBytes);

    if (result == HapResult_No_Error) {
        *outputYBytesUsed  = hap_bc4_plane_bytes(width,  height);
        *outputCbBytesUsed = hap_bc4_plane_bytes(cb_w, cb_h);
        *outputCrBytesUsed = hap_bc4_plane_bytes(cb_w, cb_h);
    }

    free(Y_full); free(Cb_full); free(Cr_full);
    free(Cb_down); free(Cr_down);
    return result;
}

static unsigned int cpu_convert_ycbcr_to_rgb(
    const void *inputY,  unsigned long inputYBytes,
    const void *inputCb, unsigned long inputCbBytes,
    const void *inputCr, unsigned long inputCrBytes,
    unsigned int subsampling,
    unsigned int width, unsigned int height,
    unsigned int outputTextureFormat,
    void *outputBuffer, unsigned long outputBufferBytes,
    unsigned long *outputBufferBytesUsed)
{
    unsigned int cb_w, cb_h;
    hap_chroma_plane_dims(width, height, subsampling, &cb_w, &cb_h);

    size_t pixel_count = (size_t)width * height;
    uint8_t *Y_raw  = (uint8_t *)malloc(pixel_count);
    uint8_t *Cb_raw = (uint8_t *)malloc((size_t)cb_w * cb_h);
    uint8_t *Cr_raw = (uint8_t *)malloc((size_t)cb_w * cb_h);
    uint8_t *rgba   = (uint8_t *)malloc(pixel_count * 4);
    if (!Y_raw || !Cb_raw || !Cr_raw || !rgba) {
        free(Y_raw); free(Cb_raw); free(Cr_raw); free(rgba);
        return HapResult_Internal_Error;
    }

    /* 1. Decode BC4 planes → raw pixels */
    unsigned int result;
    result = hap_cpu_decode_bc4(inputY,  inputYBytes,  width, height, Y_raw);
    if (result == HapResult_No_Error)
        result = hap_cpu_decode_bc4(inputCb, inputCbBytes, cb_w, cb_h, Cb_raw);
    if (result == HapResult_No_Error)
        result = hap_cpu_decode_bc4(inputCr, inputCrBytes, cb_w, cb_h, Cr_raw);

    /* 2. YCbCr → RGBA (coordinate mapping handles upsampling) */
    if (result == HapResult_No_Error)
        hap_cpu_ycbcr_to_rgba(Y_raw, Cb_raw, Cr_raw,
                               width, height, cb_w, cb_h, rgba);

    /* 3. Encode RGBA → target BC format */
    if (result == HapResult_No_Error) {
        if (outputTextureFormat == HapTextureFormat_RGB_DXT1)
            result = hap_cpu_encode_bc1(rgba, width, height, outputBuffer, outputBufferBytes);
        else
            result = hap_cpu_encode_bc3(rgba, width, height, outputBuffer, outputBufferBytes);
    }

    if (result == HapResult_No_Error)
        *outputBufferBytesUsed = hap_bc_plane_bytes(width, height, outputTextureFormat);

    free(Y_raw); free(Cb_raw); free(Cr_raw); free(rgba);
    return result;
}

/* ═══════════════════════════════════════════════════════════════════════════
   Public API
   ═══════════════════════════════════════════════════════════════════════════ */

unsigned int HapConvertTextureRGBToYCbCr(
    const void *inputBuffer,     unsigned long  inputBufferBytes,
    unsigned int  inputTextureFormat,
    unsigned int  width,         unsigned int   height,
    unsigned int  subsampling,
    void         *outputY,       unsigned long  outputYBytes,  unsigned long *outputYBytesUsed,
    void         *outputCb,      unsigned long  outputCbBytes, unsigned long *outputCbBytesUsed,
    void         *outputCr,      unsigned long  outputCrBytes, unsigned long *outputCrBytesUsed)
{
    unsigned int cb_w, cb_h;
    int backend_ok = 0;

    /* Validate arguments */
    if (!inputBuffer || !outputY || !outputCb || !outputCr
        || !outputYBytesUsed || !outputCbBytesUsed || !outputCrBytesUsed)
        return HapResult_Bad_Arguments;
    if (width == 0 || height == 0 || width % 4 != 0 || height % 4 != 0)
        return HapResult_Bad_Arguments;
    if (inputTextureFormat != HapTextureFormat_RGB_DXT1 &&
        inputTextureFormat != HapTextureFormat_RGBA_DXT5)
        return HapResult_Bad_Arguments;
    if (subsampling != HapYCbCrSubsampling_444 &&
        subsampling != HapYCbCrSubsampling_422 &&
        subsampling != HapYCbCrSubsampling_420)
        return HapResult_Bad_Arguments;
    /* Width must be divisible by 8 for 4:2:2 / 4:2:0 (so chroma half-width
       is still a multiple of 4 for BC4 block alignment). */
    if (subsampling != HapYCbCrSubsampling_444 && width % 8 != 0)
        return HapResult_Bad_Arguments;
    if (subsampling == HapYCbCrSubsampling_420 && height % 8 != 0)
        return HapResult_Bad_Arguments;
    hap_chroma_plane_dims(width, height, subsampling, &cb_w, &cb_h);

#ifdef HAVE_WEBGPU
    if (hap_wgpu_available()) {
        int ok = hap_wgpu_convert_rgb_to_ycbcr(
            inputBuffer, inputBufferBytes, inputTextureFormat,
            width, height, subsampling,
            outputY, outputCb, outputCr,
            outputYBytes, outputCbBytes, outputCrBytes);
        if (ok) backend_ok = 1;
    }
#endif

#ifdef HAVE_CUDA
    if (!backend_ok && hap_cuda_available()) {
        int ok = hap_cuda_convert_rgb_to_ycbcr(
            inputBuffer, inputBufferBytes, inputTextureFormat,
            width, height, subsampling,
            outputY, outputCb, outputCr,
            outputYBytes, outputCbBytes, outputCrBytes);
        if (ok) backend_ok = 1;
        /* Fall through to CPU path if CUDA conversion fails */
    }
#endif

    if (backend_ok) {
        *outputYBytesUsed  = hap_bc4_plane_bytes(width, height);
        *outputCbBytesUsed = hap_bc4_plane_bytes(cb_w, cb_h);
        *outputCrBytesUsed = hap_bc4_plane_bytes(cb_w, cb_h);
        return HapResult_No_Error;
    }

    return cpu_convert_rgb_to_ycbcr(
        inputBuffer, inputBufferBytes, inputTextureFormat,
        width, height, subsampling,
        outputY,  outputYBytes,  outputYBytesUsed,
        outputCb, outputCbBytes, outputCbBytesUsed,
        outputCr, outputCrBytes, outputCrBytesUsed);
}

unsigned int HapConvertTextureYCbCrToRGB(
    const void   *inputY,        unsigned long  inputYBytes,
    const void   *inputCb,       unsigned long  inputCbBytes,
    const void   *inputCr,       unsigned long  inputCrBytes,
    unsigned int  subsampling,
    unsigned int  width,         unsigned int   height,
    unsigned int  outputTextureFormat,
    void         *outputBuffer,  unsigned long  outputBufferBytes,
    unsigned long *outputBufferBytesUsed)
{
    unsigned long out_bytes = hap_bc_plane_bytes(width, height, outputTextureFormat);
    int backend_ok = 0;

    if (!inputY || !inputCb || !inputCr || !outputBuffer || !outputBufferBytesUsed)
        return HapResult_Bad_Arguments;
    if (width == 0 || height == 0 || width % 4 != 0 || height % 4 != 0)
        return HapResult_Bad_Arguments;
    if (subsampling != HapYCbCrSubsampling_444 &&
        subsampling != HapYCbCrSubsampling_422 &&
        subsampling != HapYCbCrSubsampling_420)
        return HapResult_Bad_Arguments;
    if (outputTextureFormat != HapTextureFormat_RGB_DXT1 &&
        outputTextureFormat != HapTextureFormat_RGBA_DXT5)
        return HapResult_Bad_Arguments;

#ifdef HAVE_WEBGPU
    if (hap_wgpu_available()) {
        int ok = hap_wgpu_convert_ycbcr_to_rgb(
            inputY, inputYBytes, inputCb, inputCbBytes, inputCr, inputCrBytes,
            subsampling, width, height, outputTextureFormat,
            outputBuffer, outputBufferBytes);
        if (ok) backend_ok = 1;
    }
#endif

#ifdef HAVE_CUDA
    if (!backend_ok && hap_cuda_available()) {
        int ok = hap_cuda_convert_ycbcr_to_rgb(
            inputY, inputYBytes, inputCb, inputCbBytes, inputCr, inputCrBytes,
            subsampling, width, height, outputTextureFormat,
            outputBuffer, outputBufferBytes);
        if (ok) backend_ok = 1;
    }
#endif

    if (backend_ok) {
        *outputBufferBytesUsed = out_bytes;
        return HapResult_No_Error;
    }

    return cpu_convert_ycbcr_to_rgb(
        inputY, inputYBytes, inputCb, inputCbBytes, inputCr, inputCrBytes,
        subsampling, width, height, outputTextureFormat,
        outputBuffer, outputBufferBytes, outputBufferBytesUsed);
}
