/*
 hap_convert_wgpu.h

 WebGPU (C API + WGSL compute shaders) backend interface for HAP YCbCr
 texture conversion.
 */

#ifndef hap_convert_wgpu_h
#define hap_convert_wgpu_h

#ifdef __cplusplus
extern "C" {
#endif

int hap_wgpu_available(void);

int hap_wgpu_convert_rgb_to_ycbcr(
    const void *input_bc, unsigned long input_bytes,
    unsigned int input_format,
    unsigned int width, unsigned int height,
    unsigned int subsampling,
    void *output_y_bc, void *output_cb_bc, void *output_cr_bc,
    unsigned long output_y_bc_bytes,
    unsigned long output_cb_bc_bytes,
    unsigned long output_cr_bc_bytes);

int hap_wgpu_convert_ycbcr_to_rgb(
    const void *input_y_bc, unsigned long input_y_bc_bytes,
    const void *input_cb_bc, unsigned long input_cb_bc_bytes,
    const void *input_cr_bc, unsigned long input_cr_bc_bytes,
    unsigned int subsampling,
    unsigned int width, unsigned int height,
    unsigned int output_format,
    void *output_bc, unsigned long output_bc_bytes);

#ifdef __cplusplus
}
#endif

#endif /* hap_convert_wgpu_h */
