/*
 hap_convert.h

 Internal helpers for BC texture <-> YCbCr texture conversion.
 Not part of the public API.
 */

#ifndef hap_convert_h
#define hap_convert_h

#include <stddef.h>
#include <stdint.h>
#include "hap.h"

/* ── BC plane byte-size helpers ───────────────────────────────────────────── */

/* BC4/RGTC1 and BC1/DXT1 use 8 bytes per 4×4 block.
   BC3/DXT5 uses 16 bytes per 4×4 block. */
static inline unsigned long hap_bc4_plane_bytes(unsigned int w, unsigned int h)
{
    return (unsigned long)((w + 3) / 4) * ((h + 3) / 4) * 8UL;
}

static inline unsigned long hap_bc1_plane_bytes(unsigned int w, unsigned int h)
{
    return (unsigned long)((w + 3) / 4) * ((h + 3) / 4) * 8UL;
}

static inline unsigned long hap_bc3_plane_bytes(unsigned int w, unsigned int h)
{
    return (unsigned long)((w + 3) / 4) * ((h + 3) / 4) * 16UL;
}

/* Return the compressed byte size for any supported HapTextureFormat, or 0. */
static inline unsigned long hap_bc_plane_bytes(unsigned int w, unsigned int h,
                                                unsigned int fmt)
{
    switch (fmt) {
        case HapTextureFormat_RGB_DXT1:
        case HapTextureFormat_A_RGTC1:
        case HapTextureFormat_Y_BC4:
        case HapTextureFormat_Cb_BC4:
        case HapTextureFormat_Cr_BC4:
            return hap_bc4_plane_bytes(w, h);
        case HapTextureFormat_RGBA_DXT5:
        case HapTextureFormat_YCoCg_DXT5:
        case HapTextureFormat_RGBA_BPTC_UNORM:
        case HapTextureFormat_RGB_BPTC_UNSIGNED_FLOAT:
        case HapTextureFormat_RGB_BPTC_SIGNED_FLOAT:
            return hap_bc3_plane_bytes(w, h);
        default:
            return 0;
    }
}

/* Compute chroma-plane pixel dimensions from Y-plane dimensions. */
static inline void hap_chroma_plane_dims(unsigned int y_w, unsigned int y_h,
                                         unsigned int subsampling,
                                         unsigned int *cb_w, unsigned int *cb_h)
{
    *cb_w = (subsampling == HapYCbCrSubsampling_422 ||
             subsampling == HapYCbCrSubsampling_420) ? y_w / 2 : y_w;
    *cb_h = (subsampling == HapYCbCrSubsampling_420) ? y_h / 2 : y_h;
}

/* ── CPU codec primitives ─────────────────────────────────────────────────── */

/*
 BC4/RGTC1 decode: expand one 8-byte block into 16 single-channel uint8 values
 stored in row-major order (top-left first).
 */
void hap_decode_bc4_block(const uint8_t *block, uint8_t out[16]);

/*
 BC4/RGTC1 encode: compress 16 single-channel uint8 values into one 8-byte block.
 Uses the 8-value interpolation mode (endpoint0 > endpoint1) for best quality.
 */
void hap_encode_bc4_block(const uint8_t in[16], uint8_t *block);

/*
 BC1/DXT1 decode: expand one 8-byte block into 16 RGBA uint8 pixels (64 bytes).
 Alpha is set to 255 for all pixels except the transparent entry in 3-color mode.
 */
void hap_decode_bc1_block(const uint8_t *block, uint8_t out[64]);

/*
 BC1/DXT1 encode: compress 16 RGBA pixels (64 bytes) into one 8-byte BC1 block.
 Uses a bounding-box heuristic; always forces 4-color mode (no transparency).
 */
void hap_encode_bc1_block(const uint8_t in[64], uint8_t *block);

/*
 BC3/DXT5 decode: expand one 16-byte block into 16 RGBA uint8 pixels (64 bytes).
 Alpha is decoded via the BC4-like alpha sub-block.
 */
void hap_decode_bc3_block(const uint8_t *block, uint8_t out[64]);

/*
 BC3/DXT5 encode: compress 16 RGBA pixels (64 bytes) into one 16-byte BC3 block.
 */
void hap_encode_bc3_block(const uint8_t in[64], uint8_t *block);

/*
 Full-texture decode/encode wrappers.
 width and height must be multiples of 4.
 Returns HapResult_No_Error or HapResult_Bad_Arguments / HapResult_Bad_Frame.
 */
unsigned int hap_cpu_decode_bc1(const void *input, unsigned long input_bytes,
                                 unsigned int width, unsigned int height,
                                 uint8_t *rgba_out);

unsigned int hap_cpu_decode_bc3(const void *input, unsigned long input_bytes,
                                 unsigned int width, unsigned int height,
                                 uint8_t *rgba_out);

unsigned int hap_cpu_decode_bc4(const void *input, unsigned long input_bytes,
                                 unsigned int width, unsigned int height,
                                 uint8_t *luma_out);

unsigned int hap_cpu_encode_bc1(const uint8_t *rgba,
                                 unsigned int width, unsigned int height,
                                 void *output, unsigned long output_bytes);

unsigned int hap_cpu_encode_bc3(const uint8_t *rgba,
                                 unsigned int width, unsigned int height,
                                 void *output, unsigned long output_bytes);

unsigned int hap_cpu_encode_bc4(const uint8_t *luma,
                                 unsigned int width, unsigned int height,
                                 void *output, unsigned long output_bytes);

/* ── CPU colour-space conversion ──────────────────────────────────────────── */

/*
 Convert width×height RGBA pixels to three separate full-resolution Y, Cb, Cr
 planes (BT.709 full-range).  All three output buffers must be width×height bytes.
 */
void hap_cpu_rgba_to_ycbcr(const uint8_t *rgba,
                             unsigned int width, unsigned int height,
                             uint8_t *Y, uint8_t *Cb, uint8_t *Cr);

/*
 Convert full-resolution Y plane plus (possibly subsampled) Cb/Cr planes to
 width×height RGBA pixels (BT.709 full-range).  Chroma upsampling is performed
 by nearest-neighbour coordinate mapping.
 */
void hap_cpu_ycbcr_to_rgba(const uint8_t *Y,
                             const uint8_t *Cb, const uint8_t *Cr,
                             unsigned int y_width,  unsigned int y_height,
                             unsigned int cb_width, unsigned int cb_height,
                             uint8_t *rgba_out);

/*
 Downsample a full-width chroma plane to half width (4:2:2).
 src_width must be even.  dst must hold (src_width/2)*height bytes.
 */
void hap_cpu_chroma_downsample_422(const uint8_t *src,
                                    unsigned int src_width, unsigned int height,
                                    uint8_t *dst);

/*
 Downsample a full-size chroma plane to quarter size (4:2:0).
 src_width and src_height must each be even.
 dst must hold (src_width/2)*(src_height/2) bytes.
 */
void hap_cpu_chroma_downsample_420(const uint8_t *src,
                                    unsigned int src_width,
                                    unsigned int src_height,
                                    uint8_t *dst);

#endif /* hap_convert_h */
