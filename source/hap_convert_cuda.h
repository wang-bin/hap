/*
 hap_convert_cuda.h

 CUDA (driver API + NVRTC) backend interface for HAP YCbCr texture conversion.
 Only meaningful when compiled with -DHAVE_CUDA.
 */

#ifndef hap_convert_cuda_h
#define hap_convert_cuda_h

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 Returns 1 if a suitable CUDA device is available and all kernels have been
 compiled and loaded successfully, 0 otherwise.

 The first call initialises the CUDA context and compiles the embedded kernel
 source via NVRTC.  Subsequent calls return the cached result.
 */
int hap_cuda_available(void);

/*
 GPU path for HapConvertTextureRGBToYCbCr.

 All buffer pointers are host (CPU) memory; the function handles all device
 allocation, H→D / D→H transfers internally.

 Returns 1 on success, 0 on failure (caller should fall back to CPU path).
 */
int hap_cuda_convert_rgb_to_ycbcr(
    const void  *input_bc,         unsigned long  input_bytes,
    unsigned int input_format,     /* HapTextureFormat_RGB_DXT1 or _RGBA_DXT5 */
    unsigned int width,            unsigned int   height,
    unsigned int subsampling,      /* HapYCbCrSubsampling_* */
    void        *output_y,         void          *output_cb,         void *output_cr,
    unsigned long output_y_bytes,  unsigned long  output_cb_bytes,   unsigned long output_cr_bytes);

/*
 GPU path for HapConvertTextureYCbCrToRGB.

 Returns 1 on success, 0 on failure.
 */
int hap_cuda_convert_ycbcr_to_rgb(
    const void  *input_y,          unsigned long  input_y_bytes,
    const void  *input_cb,         unsigned long  input_cb_bytes,
    const void  *input_cr,         unsigned long  input_cr_bytes,
    unsigned int subsampling,
    unsigned int width,            unsigned int   height,
    unsigned int output_format,    /* HapTextureFormat_RGB_DXT1 or _RGBA_DXT5 */
    void        *output_bc,        unsigned long  output_bc_bytes);

#ifdef __cplusplus
}
#endif

#endif /* hap_convert_cuda_h */
