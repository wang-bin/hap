/*
 hap.c

 Copyright (c) 2011-2013, Tom Butterworth and Vidvox LLC. All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright
 notice, this list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright
 notice, this list of conditions and the following disclaimer in the
 documentation and/or other materials provided with the distribution.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "hap.h"
#include <stdlib.h>
#include <stdint.h>
#include <string.h> // For memcpy for uncompressed frames
#include "snappy-c.h"
#if __has_include("lz4.h")
# include "lz4.h"
#endif
/*
 When the CUDA runtime and nvComp headers are present, GPU-accelerated LZ4 and
 Snappy compression/decompression is enabled.  nvcc is not required; any C/C++
 compiler that can find cuda_runtime_api.h and link against libcuda/libcudart and
 the nvComp library will activate this path.  The on-disk format is identical to
 the CPU path so encoded frames are fully interoperable.
*/
#if __has_include("cuda_runtime_api.h") && __has_include("nvcomp/lz4.h") && __has_include("nvcomp/snappy.h")
# define HAP_USE_NVCOMP 1
# include "cuda_runtime_api.h"
# include "nvcomp.h"
# include "nvcomp/lz4.h"
# include "nvcomp/snappy.h"
#endif

#define kHapUInt24Max 0x00FFFFFF

/*
 Hap Constants
 First four bits represent the compressor
 Second four bits represent the format
 */
#define kHapCompressorNone 0xA
#define kHapCompressorSnappy 0xB
#define kHapCompressorComplex 0xC
#define kHapCompressorLZ4 0xD

#define kHapFormatRGBDXT1 0xB
#define kHapFormatRGBADXT5 0xE
#define kHapFormatYCoCgDXT5 0xF
#define kHapFormatARGTC1 0x1
#define kHapFormatRGBABPTC 0xC
#define kHapFormatRGBBPTCUF 0x2
#define kHapFormatRGBBPTCSF 0x3

/*
 Packed byte values for Hap

 Format                     Compressor      Byte Code
 ----------------------------------------------------
 RGB_DXT1                   None            0xAB
 RGB_DXT1                   Snappy          0xBB
 RGB_DXT1                   Complex         0xCB
 RGBA_DXT5                  None            0xAE
 RGBA_DXT5                  Snappy          0xBE
 RGBA_DXT5                  Complex         0xCE
 YCoCg_DXT5                 None            0xAF
 YCoCg_DXT5                 Snappy          0xBF
 YCoCg_DXT5                 Complex         0xCF
 A_RGTC1                    None            0xA1
 A_RGTC1                    Snappy          0xB1
 A_RGTC1                    Complex         0xC1
 RGBA_BPTC_UNORM            None            0xAC
 RGBA_BPTC_UNORM            Snappy          0xBC
 RGBA_BPTC_UNORM            Complex         0xCC
 RGB_BPTC_UNSIGNED_FLOAT    None            0xA2
 RGB_BPTC_UNSIGNED_FLOAT    Snappy          0xB2
 RGB_BPTC_UNSIGNED_FLOAT    Complex         0xC2
 RGB_BPTC_SIGNED_FLOAT      None            0xA3
 RGB_BPTC_SIGNED_FLOAT      Snappy          0xB3
 RGB_BPTC_SIGNED_FLOAT      Complex         0xC3
 */

/*
 Hap Frame Section Types
 */
#define kHapSectionMultipleImages 0x0D
#define kHapSectionDecodeInstructionsContainer 0x01
#define kHapSectionChunkSecondStageCompressorTable 0x02
#define kHapSectionChunkSizeTable 0x03
#define kHapSectionChunkOffsetTable 0x04

/*
 To decode we use a struct to store details of each chunk
 */
typedef struct HapChunkDecodeInfo {
    unsigned int result;
    unsigned int compressor;
    const char *compressed_chunk_data;
    size_t compressed_chunk_size;
    char *uncompressed_chunk_data;
    size_t uncompressed_chunk_size;
} HapChunkDecodeInfo;

// TODO: rename the defines we use for codes used in stored frames
// to better differentiate them from the enums used for the API

// These read and write little-endian values on big or little-endian architectures
static unsigned int hap_read_3_byte_uint(const void *buffer)
{
    return (*(uint8_t *)buffer) + ((*(((uint8_t *)buffer) + 1)) << 8) + ((*(((uint8_t *)buffer) + 2)) << 16);
}

static void hap_write_3_byte_uint(void *buffer, unsigned int value)
{
    *(uint8_t *)buffer = value & 0xFF;
    *(((uint8_t *)buffer) + 1) = (value >> 8) & 0xFF;
    *(((uint8_t *)buffer) + 2) = (value >> 16) & 0xFF;
}

static unsigned int hap_read_4_byte_uint(const void *buffer)
{
    return (*(uint8_t *)buffer) + ((*(((uint8_t *)buffer) + 1)) << 8) + ((*(((uint8_t *)buffer) + 2)) << 16) + ((*(((uint8_t *)buffer) + 3)) << 24);
}

static void hap_write_4_byte_uint(const void *buffer, unsigned int value)
{
    *(uint8_t *)buffer = value & 0xFF;
    *(((uint8_t *)buffer) + 1) = (value >> 8) & 0xFF;
    *(((uint8_t *)buffer) + 2) = (value >> 16) & 0xFF;
    *(((uint8_t *)buffer) + 3) = (value >> 24) & 0xFF;
}

#define hap_top_4_bits(x) (((x) & 0xF0) >> 4)

#define hap_bottom_4_bits(x) ((x) & 0x0F)

#define hap_4_bit_packed_byte(top_bits, bottom_bits) (((top_bits) << 4) | ((bottom_bits) & 0x0F))

static int hap_read_section_header(const void *buffer, uint32_t buffer_length, uint32_t *out_header_length, uint32_t *out_section_length, unsigned int *out_section_type)
{
    /*
     Verify buffer is big enough to contain a four-byte header
     */
    if (buffer_length < 4U)
    {
        return HapResult_Bad_Frame;
    }

    /*
     The first three bytes are the length of the section (not including the header) or zero
     if the length is stored in the last four bytes of an eight-byte header
     */
    *out_section_length = hap_read_3_byte_uint(buffer);

    /*
     If the first three bytes are zero, the size is in the following four bytes
     */
    if (*out_section_length == 0U)
    {
        /*
         Verify buffer is big enough to contain an eight-byte header
         */
        if (buffer_length < 8U)
        {
            return HapResult_Bad_Frame;
        }
        *out_section_length = hap_read_4_byte_uint(((uint8_t *)buffer) + 4U);
        *out_header_length = 8U;
    }
    else
    {
        *out_header_length = 4U;
    }

    /*
     The fourth byte stores the section type
     */
    *out_section_type = *(((uint8_t *)buffer) + 3U);

    /*
     Verify the section does not extend beyond the buffer
     */
    if (*out_header_length + *out_section_length > buffer_length)
    {
        return HapResult_Bad_Frame;
    }

    return HapResult_No_Error;
}

static void hap_write_section_header(void *buffer, size_t header_length, uint32_t section_length, unsigned int section_type)
{
    /*
     The first three bytes are the length of the section (not including the header) or zero
     if using an eight-byte header
     */
    if (header_length == 4U)
    {
        hap_write_3_byte_uint(buffer, (unsigned int)section_length);
    }
    else
    {
        /*
         For an eight-byte header, the length is in the last four bytes
         */
        hap_write_3_byte_uint(buffer, 0U);
        hap_write_4_byte_uint(((uint8_t *)buffer) + 4U, section_length);
    }

    /*
     The fourth byte stores the section type
     */
    *(((uint8_t *)buffer) + 3) = section_type;
}

// Returns an API texture format constant or 0 if not recognised
static unsigned int hap_texture_format_constant_for_format_identifier(unsigned int identifier)
{
    switch (identifier)
    {
        case kHapFormatRGBDXT1:
            return HapTextureFormat_RGB_DXT1;
        case kHapFormatRGBADXT5:
            return HapTextureFormat_RGBA_DXT5;
        case kHapFormatYCoCgDXT5:
            return HapTextureFormat_YCoCg_DXT5;
        case kHapFormatARGTC1:
            return HapTextureFormat_A_RGTC1;
        case kHapFormatRGBABPTC:
            return HapTextureFormat_RGBA_BPTC_UNORM;
        case kHapFormatRGBBPTCUF:
            return HapTextureFormat_RGB_BPTC_UNSIGNED_FLOAT;
        case kHapFormatRGBBPTCSF:
            return HapTextureFormat_RGB_BPTC_SIGNED_FLOAT;
        default:
            return 0;

    }
}

// Returns a frame identifier or 0 if not recognised
static unsigned int hap_texture_format_identifier_for_format_constant(unsigned int constant)
{
    switch (constant)
    {
        case HapTextureFormat_RGB_DXT1:
            return kHapFormatRGBDXT1;
        case HapTextureFormat_RGBA_DXT5:
            return kHapFormatRGBADXT5;
        case HapTextureFormat_YCoCg_DXT5:
            return kHapFormatYCoCgDXT5;
        case HapTextureFormat_A_RGTC1:
            return kHapFormatARGTC1;
        case HapTextureFormat_RGBA_BPTC_UNORM:
            return kHapFormatRGBABPTC;
        case HapTextureFormat_RGB_BPTC_UNSIGNED_FLOAT:
            return kHapFormatRGBBPTCUF;
        case HapTextureFormat_RGB_BPTC_SIGNED_FLOAT:
            return kHapFormatRGBBPTCSF;
        default:
            return 0;
    }
}

// Returns the length of a decode instructions container of chunk_count chunks
// not including the section header
static size_t hap_decode_instructions_length(unsigned int chunk_count)
{
    /*
     Calculate the size of our Decode Instructions Section
     = Second-Stage Compressor Table + Chunk Size Table + headers for both sections
     = chunk_count + (4 * chunk_count) + 4 + 4
     */
    size_t length = (5 * chunk_count) + 8;

    return length;
}

static unsigned int hap_limited_chunk_count_for_frame(size_t input_bytes, unsigned int texture_format, unsigned int chunk_count)
{
    // This is a hard limit due to the 4-byte headers we use for the decode instruction container
    // (0xFFFFFF == count + (4 x count) + 20)
    if (chunk_count > 3355431)
    {
        chunk_count = 3355431;
    }
    // Divide frame equally on DXT block boundries (8 or 16 bytes)
    unsigned long dxt_block_count;
    switch (texture_format) {
        case HapTextureFormat_RGB_DXT1:
        case HapTextureFormat_A_RGTC1:
            dxt_block_count = input_bytes / 8;
            break;
        default:
            dxt_block_count = input_bytes / 16;
    }
    while (dxt_block_count % chunk_count != 0) {
        chunk_count--;
    }

    return chunk_count;
}

#ifdef HAP_USE_NVCOMP
/*
 Returns 1 if a CUDA-capable device is available, 0 otherwise.
 The result is cached after the first call.
*/
static int hap_cuda_device_available(void)
{
    static int available = -1;
    if (available < 0)
    {
        int count = 0;
        available = (cudaGetDeviceCount(&count) == cudaSuccess && count > 0) ? 1 : 0;
    }
    return available;
}

/*
 Batch-compress chunk_count chunks of chunk_size bytes each on the GPU.

 compressor    – HapCompressorLZ4 or HapCompressorSnappy
 input         – contiguous input buffer (chunk_size * chunk_count bytes)
 chunk_size    – size of every input chunk in bytes
 chunk_count   – number of chunks
 output        – destination buffer; must be large enough for all compressed chunks
                 (for LZ4 each chunk is prefixed with a 4-byte original-size field,
                  matching the CPU encoder format)
 output_cap    – available bytes in output
 comp_table    – [chunk_count] output: per-chunk second-stage compressor byte
 size_table    – [chunk_count * 4] output: per-chunk packed size (4-byte LE)
 total_out     – total bytes written to output

 Returns HapResult_No_Error on success, or an error code if the GPU operation
 failed (the caller should fall back to the CPU path).
*/
static unsigned int hap_nvcomp_compress_chunks(
    unsigned int compressor,
    const void *input,
    size_t chunk_size,
    unsigned int chunk_count,
    void *output,
    size_t output_cap,
    uint8_t *comp_table,
    void *size_table,
    size_t *total_out)
{
    unsigned int result = HapResult_Internal_Error;
    cudaError_t cuda_err;
    nvcompStatus_t nvcomp_err;
    cudaStream_t stream = 0;

    void *d_input        = NULL;
    void *d_comp_out     = NULL;
    void *d_temp         = NULL;
    void **d_in_ptrs     = NULL;
    void **d_out_ptrs    = NULL;
    size_t *d_in_sizes   = NULL;
    size_t *d_out_sizes  = NULL;

    void **h_in_ptrs     = NULL;
    void **h_out_ptrs    = NULL;
    size_t *h_in_sizes   = NULL;
    size_t *h_out_sizes  = NULL;

    size_t temp_bytes = 0;
    size_t max_comp_chunk = 0;

    /* Query worst-case sizes from nvComp */
    if (compressor == HapCompressorLZ4)
    {
        nvcompBatchedLZ4Opts_t opts = {NVCOMP_TYPE_CHAR};
        nvcomp_err = nvcompBatchedLZ4CompressGetTempSize(chunk_count, chunk_size, opts, &temp_bytes);
        if (nvcomp_err != nvcompSuccess) goto cleanup;
        nvcomp_err = nvcompBatchedLZ4CompressGetMaxOutputChunkSize(chunk_size, opts, &max_comp_chunk);
        if (nvcomp_err != nvcompSuccess) goto cleanup;
    }
    else
    {
        nvcompBatchedSnappyOpts_t opts = {};
        nvcomp_err = nvcompBatchedSnappyCompressGetTempSize(chunk_count, chunk_size, opts, &temp_bytes);
        if (nvcomp_err != nvcompSuccess) goto cleanup;
        nvcomp_err = nvcompBatchedSnappyCompressGetMaxOutputChunkSize(chunk_size, opts, &max_comp_chunk);
        if (nvcomp_err != nvcompSuccess) goto cleanup;
    }

    /* Allocate host-side pointer / size arrays */
    h_in_ptrs  = (void **)malloc(sizeof(void *) * chunk_count);
    h_out_ptrs = (void **)malloc(sizeof(void *) * chunk_count);
    h_in_sizes = (size_t *)malloc(sizeof(size_t) * chunk_count);
    h_out_sizes = (size_t *)malloc(sizeof(size_t) * chunk_count);
    if (!h_in_ptrs || !h_out_ptrs || !h_in_sizes || !h_out_sizes)
        goto cleanup;

    /* Allocate device memory */
    cuda_err = cudaMalloc(&d_input, chunk_size * chunk_count);
    if (cuda_err != cudaSuccess) goto cleanup;
    cuda_err = cudaMalloc(&d_comp_out, max_comp_chunk * chunk_count);
    if (cuda_err != cudaSuccess) goto cleanup;
    if (temp_bytes > 0)
    {
        cuda_err = cudaMalloc(&d_temp, temp_bytes);
        if (cuda_err != cudaSuccess) goto cleanup;
    }
    cuda_err = cudaMalloc(&d_in_ptrs,  sizeof(void *) * chunk_count);
    if (cuda_err != cudaSuccess) goto cleanup;
    cuda_err = cudaMalloc(&d_out_ptrs, sizeof(void *) * chunk_count);
    if (cuda_err != cudaSuccess) goto cleanup;
    cuda_err = cudaMalloc(&d_in_sizes,  sizeof(size_t) * chunk_count);
    if (cuda_err != cudaSuccess) goto cleanup;
    cuda_err = cudaMalloc(&d_out_sizes, sizeof(size_t) * chunk_count);
    if (cuda_err != cudaSuccess) goto cleanup;

    /* Build pointer arrays */
    for (unsigned int i = 0; i < chunk_count; i++)
    {
        h_in_ptrs[i]  = (char *)d_input    + chunk_size     * i;
        h_out_ptrs[i] = (char *)d_comp_out + max_comp_chunk * i;
        h_in_sizes[i] = chunk_size;
    }

    /* Upload input and metadata to device */
    cuda_err = cudaMemcpy(d_input,    input,       chunk_size * chunk_count,        cudaMemcpyHostToDevice);
    if (cuda_err != cudaSuccess) goto cleanup;
    cuda_err = cudaMemcpy(d_in_ptrs,  h_in_ptrs,  sizeof(void *) * chunk_count,   cudaMemcpyHostToDevice);
    if (cuda_err != cudaSuccess) goto cleanup;
    cuda_err = cudaMemcpy(d_out_ptrs, h_out_ptrs, sizeof(void *) * chunk_count,   cudaMemcpyHostToDevice);
    if (cuda_err != cudaSuccess) goto cleanup;
    cuda_err = cudaMemcpy(d_in_sizes, h_in_sizes, sizeof(size_t) * chunk_count,   cudaMemcpyHostToDevice);
    if (cuda_err != cudaSuccess) goto cleanup;

    /* GPU batch compression */
    if (compressor == HapCompressorLZ4)
    {
        nvcompBatchedLZ4Opts_t opts = {NVCOMP_TYPE_CHAR};
        nvcomp_err = nvcompBatchedLZ4CompressAsync(
            (const void * const *)d_in_ptrs,
            (const size_t *)d_in_sizes,
            chunk_size,
            chunk_count,
            d_temp,
            temp_bytes,
            d_out_ptrs,
            d_out_sizes,
            opts,
            stream);
    }
    else
    {
        nvcompBatchedSnappyOpts_t opts = {};
        nvcomp_err = nvcompBatchedSnappyCompressAsync(
            (const void * const *)d_in_ptrs,
            (const size_t *)d_in_sizes,
            chunk_size,
            chunk_count,
            d_temp,
            temp_bytes,
            d_out_ptrs,
            d_out_sizes,
            opts,
            stream);
    }
    if (nvcomp_err != nvcompSuccess) goto cleanup;

    cuda_err = cudaStreamSynchronize(stream);
    if (cuda_err != cudaSuccess) goto cleanup;

    /* Retrieve compressed sizes */
    cuda_err = cudaMemcpy(h_out_sizes, d_out_sizes, sizeof(size_t) * chunk_count, cudaMemcpyDeviceToHost);
    if (cuda_err != cudaSuccess) goto cleanup;

    /* Copy compressed chunks to output, writing LZ4 4-byte prefix where needed */
    {
        char *out_ptr = (char *)output;
        size_t written = 0;
        for (unsigned int i = 0; i < chunk_count; i++)
        {
            size_t comp_size = h_out_sizes[i];
            size_t prefix = (compressor == HapCompressorLZ4) ? 4 : 0;
            size_t chunk_total = comp_size + prefix;

            if (chunk_total >= chunk_size)
            {
                /* Compressed is not smaller; store this chunk uncompressed */
                if (written + chunk_size > output_cap)
                    goto cleanup;
                cuda_err = cudaMemcpy(out_ptr,
                                      (char *)d_input + chunk_size * i,
                                      chunk_size,
                                      cudaMemcpyDeviceToHost);
                if (cuda_err != cudaSuccess) goto cleanup;
                comp_table[i] = kHapCompressorNone;
                hap_write_4_byte_uint((uint8_t *)size_table + i * 4, (unsigned int)chunk_size);
                out_ptr += chunk_size;
                written += chunk_size;
            }
            else
            {
                if (written + chunk_total > output_cap)
                    goto cleanup;
                if (prefix)
                {
                    hap_write_4_byte_uint(out_ptr, (unsigned int)chunk_size);
                    out_ptr += 4;
                }
                cuda_err = cudaMemcpy(out_ptr,
                                      (char *)d_comp_out + max_comp_chunk * i,
                                      comp_size,
                                      cudaMemcpyDeviceToHost);
                if (cuda_err != cudaSuccess) goto cleanup;
                comp_table[i] = (compressor == HapCompressorLZ4)
                                 ? kHapCompressorLZ4
                                 : kHapCompressorSnappy;
                hap_write_4_byte_uint((uint8_t *)size_table + i * 4, (unsigned int)chunk_total);
                out_ptr += comp_size;
                written += chunk_total;
            }
        }
        *total_out = written;
    }

    result = HapResult_No_Error;

cleanup:
    free(h_in_ptrs);
    free(h_out_ptrs);
    free(h_in_sizes);
    free(h_out_sizes);
    if (d_input)    cudaFree(d_input);
    if (d_comp_out) cudaFree(d_comp_out);
    if (d_temp)     cudaFree(d_temp);
    if (d_in_ptrs)  cudaFree(d_in_ptrs);
    if (d_out_ptrs) cudaFree(d_out_ptrs);
    if (d_in_sizes) cudaFree(d_in_sizes);
    if (d_out_sizes) cudaFree(d_out_sizes);

    return result;
}

/*
 Batch-decompress chunk_count chunks on the GPU, filling in each
 chunks[i].uncompressed_chunk_data and setting chunks[i].result.

 compressor  – kHapCompressorLZ4 or kHapCompressorSnappy
               (must be uniform across all elements of chunks[])

 For LZ4 chunks: compressed_chunk_data points to [4-byte original_size][lz4 data]
                 and compressed_chunk_size includes the 4-byte prefix.
                 uncompressed_chunk_size must already be set (from the prefix).
 For Snappy chunks: compressed_chunk_data/size are the raw snappy stream.
                    uncompressed_chunk_size must already be set.

 Returns HapResult_No_Error if all chunks were decompressed successfully.
 On any GPU failure the function returns an error and leaves chunks[].result
 untouched so the caller can fall back to the CPU path.
*/
static unsigned int hap_nvcomp_decompress_chunks(
    unsigned int compressor,
    HapChunkDecodeInfo *chunks,
    unsigned int chunk_count)
{
    unsigned int result = HapResult_Internal_Error;
    cudaError_t cuda_err;
    nvcompStatus_t nvcomp_err;
    cudaStream_t stream = 0;

    size_t prefix = (compressor == kHapCompressorLZ4) ? 4 : 0;

    void *d_comp    = NULL;
    void *d_uncomp  = NULL;
    void *d_temp    = NULL;
    void **d_comp_ptrs   = NULL;
    void **d_uncomp_ptrs = NULL;
    size_t *d_comp_sizes   = NULL;
    size_t *d_uncomp_sizes = NULL;
    nvcompStatus_t *d_statuses          = NULL;
    size_t         *d_actual_uncomp_sizes = NULL;

    void   **h_comp_ptrs   = NULL;
    void   **h_uncomp_ptrs = NULL;
    size_t  *h_comp_sizes  = NULL;
    size_t  *h_uncomp_sizes = NULL;
    nvcompStatus_t *h_statuses = NULL;

    size_t total_comp   = 0;
    size_t total_uncomp = 0;
    size_t max_uncomp_chunk = 0;
    size_t temp_bytes = 0;

    for (unsigned int i = 0; i < chunk_count; i++)
    {
        total_comp   += chunks[i].compressed_chunk_size - prefix;
        total_uncomp += chunks[i].uncompressed_chunk_size;
        if (chunks[i].uncompressed_chunk_size > max_uncomp_chunk)
            max_uncomp_chunk = chunks[i].uncompressed_chunk_size;
    }

    /* Query temp buffer size */
    if (compressor == kHapCompressorLZ4)
    {
        nvcomp_err = nvcompBatchedLZ4DecompressGetTempSize(chunk_count, max_uncomp_chunk, &temp_bytes);
        if (nvcomp_err != nvcompSuccess) goto cleanup;
    }
    else
    {
        nvcomp_err = nvcompBatchedSnappyDecompressGetTempSize(chunk_count, max_uncomp_chunk, &temp_bytes);
        if (nvcomp_err != nvcompSuccess) goto cleanup;
    }

    /* Allocate host arrays */
    h_comp_ptrs   = (void **)malloc(sizeof(void *) * chunk_count);
    h_uncomp_ptrs = (void **)malloc(sizeof(void *) * chunk_count);
    h_comp_sizes  = (size_t *)malloc(sizeof(size_t) * chunk_count);
    h_uncomp_sizes = (size_t *)malloc(sizeof(size_t) * chunk_count);
    h_statuses    = (nvcompStatus_t *)malloc(sizeof(nvcompStatus_t) * chunk_count);
    if (!h_comp_ptrs || !h_uncomp_ptrs || !h_comp_sizes || !h_uncomp_sizes || !h_statuses)
        goto cleanup;

    /* Allocate device memory */
    cuda_err = cudaMalloc(&d_comp,   total_comp);
    if (cuda_err != cudaSuccess) goto cleanup;
    cuda_err = cudaMalloc(&d_uncomp, total_uncomp);
    if (cuda_err != cudaSuccess) goto cleanup;
    if (temp_bytes > 0)
    {
        cuda_err = cudaMalloc(&d_temp, temp_bytes);
        if (cuda_err != cudaSuccess) goto cleanup;
    }
    cuda_err = cudaMalloc(&d_comp_ptrs,   sizeof(void *) * chunk_count);
    if (cuda_err != cudaSuccess) goto cleanup;
    cuda_err = cudaMalloc(&d_uncomp_ptrs, sizeof(void *) * chunk_count);
    if (cuda_err != cudaSuccess) goto cleanup;
    cuda_err = cudaMalloc(&d_comp_sizes,   sizeof(size_t) * chunk_count);
    if (cuda_err != cudaSuccess) goto cleanup;
    cuda_err = cudaMalloc(&d_uncomp_sizes, sizeof(size_t) * chunk_count);
    if (cuda_err != cudaSuccess) goto cleanup;
    cuda_err = cudaMalloc(&d_statuses,          sizeof(nvcompStatus_t) * chunk_count);
    if (cuda_err != cudaSuccess) goto cleanup;
    cuda_err = cudaMalloc(&d_actual_uncomp_sizes, sizeof(size_t) * chunk_count);
    if (cuda_err != cudaSuccess) goto cleanup;

    /* Upload compressed data and build pointer arrays */
    {
        size_t comp_off   = 0;
        size_t uncomp_off = 0;
        for (unsigned int i = 0; i < chunk_count; i++)
        {
            size_t actual_comp = chunks[i].compressed_chunk_size - prefix;
            cuda_err = cudaMemcpy((char *)d_comp + comp_off,
                                  chunks[i].compressed_chunk_data + prefix,
                                  actual_comp,
                                  cudaMemcpyHostToDevice);
            if (cuda_err != cudaSuccess) goto cleanup;

            h_comp_ptrs[i]   = (char *)d_comp   + comp_off;
            h_uncomp_ptrs[i] = (char *)d_uncomp + uncomp_off;
            h_comp_sizes[i]  = actual_comp;
            h_uncomp_sizes[i] = chunks[i].uncompressed_chunk_size;

            comp_off   += actual_comp;
            uncomp_off += chunks[i].uncompressed_chunk_size;
        }
    }

    cuda_err = cudaMemcpy(d_comp_ptrs,   h_comp_ptrs,   sizeof(void *) * chunk_count, cudaMemcpyHostToDevice);
    if (cuda_err != cudaSuccess) goto cleanup;
    cuda_err = cudaMemcpy(d_uncomp_ptrs, h_uncomp_ptrs, sizeof(void *) * chunk_count, cudaMemcpyHostToDevice);
    if (cuda_err != cudaSuccess) goto cleanup;
    cuda_err = cudaMemcpy(d_comp_sizes,   h_comp_sizes,  sizeof(size_t) * chunk_count, cudaMemcpyHostToDevice);
    if (cuda_err != cudaSuccess) goto cleanup;
    cuda_err = cudaMemcpy(d_uncomp_sizes, h_uncomp_sizes, sizeof(size_t) * chunk_count, cudaMemcpyHostToDevice);
    if (cuda_err != cudaSuccess) goto cleanup;

    /* GPU batch decompression */
    if (compressor == kHapCompressorLZ4)
    {
        nvcomp_err = nvcompBatchedLZ4DecompressAsync(
            (const void * const *)d_comp_ptrs,
            (const size_t *)d_comp_sizes,
            (const size_t *)d_uncomp_sizes,
            d_actual_uncomp_sizes,
            chunk_count,
            d_temp,
            temp_bytes,
            d_uncomp_ptrs,
            d_statuses,
            stream);
    }
    else
    {
        nvcomp_err = nvcompBatchedSnappyDecompressAsync(
            (const void * const *)d_comp_ptrs,
            (const size_t *)d_comp_sizes,
            (const size_t *)d_uncomp_sizes,
            d_actual_uncomp_sizes,
            chunk_count,
            d_temp,
            temp_bytes,
            d_uncomp_ptrs,
            d_statuses,
            stream);
    }
    if (nvcomp_err != nvcompSuccess) goto cleanup;

    cuda_err = cudaStreamSynchronize(stream);
    if (cuda_err != cudaSuccess) goto cleanup;

    /* Check per-chunk statuses */
    cuda_err = cudaMemcpy(h_statuses, d_statuses, sizeof(nvcompStatus_t) * chunk_count, cudaMemcpyDeviceToHost);
    if (cuda_err != cudaSuccess) goto cleanup;

    for (unsigned int i = 0; i < chunk_count; i++)
    {
        if (h_statuses[i] != nvcompSuccess)
            goto cleanup;
    }

    /* Copy decompressed data back to the caller's output buffers */
    {
        size_t uncomp_off = 0;
        for (unsigned int i = 0; i < chunk_count; i++)
        {
            cuda_err = cudaMemcpy(chunks[i].uncompressed_chunk_data,
                                  (char *)d_uncomp + uncomp_off,
                                  chunks[i].uncompressed_chunk_size,
                                  cudaMemcpyDeviceToHost);
            if (cuda_err != cudaSuccess) goto cleanup;
            chunks[i].result = HapResult_No_Error;
            uncomp_off += chunks[i].uncompressed_chunk_size;
        }
    }

    result = HapResult_No_Error;

cleanup:
    free(h_comp_ptrs);
    free(h_uncomp_ptrs);
    free(h_comp_sizes);
    free(h_uncomp_sizes);
    free(h_statuses);
    if (d_comp)              cudaFree(d_comp);
    if (d_uncomp)            cudaFree(d_uncomp);
    if (d_temp)              cudaFree(d_temp);
    if (d_comp_ptrs)         cudaFree(d_comp_ptrs);
    if (d_uncomp_ptrs)       cudaFree(d_uncomp_ptrs);
    if (d_comp_sizes)        cudaFree(d_comp_sizes);
    if (d_uncomp_sizes)      cudaFree(d_uncomp_sizes);
    if (d_statuses)          cudaFree(d_statuses);
    if (d_actual_uncomp_sizes) cudaFree(d_actual_uncomp_sizes);

    return result;
}
#endif /* HAP_USE_NVCOMP */

static size_t hap_max_encoded_length(size_t input_bytes, unsigned int texture_format, unsigned int compressor, unsigned int chunk_count)
{
    size_t decode_instructions_length, max_compressed_length;

    chunk_count = hap_limited_chunk_count_for_frame(input_bytes, texture_format, chunk_count);

    decode_instructions_length = hap_decode_instructions_length(chunk_count);

    if (compressor == HapCompressorSnappy)
    {
        size_t chunk_size = input_bytes / chunk_count;
        max_compressed_length = snappy_max_compressed_length(chunk_size) * chunk_count;
    }
#ifdef LZ4LIB_API
    else if (compressor == HapCompressorLZ4)
    {
        size_t chunk_size = input_bytes / chunk_count;
        max_compressed_length = (size_t)(LZ4_compressBound((int)chunk_size) + 4) * chunk_count;
    }
#endif
    else
    {
        max_compressed_length = input_bytes;
    }

    // top section header + decode instructions section header + decode instructions + compressed data
    return max_compressed_length + 8U + decode_instructions_length + 4U;
}

unsigned long HapMaxEncodedLength(unsigned int count,
                                  unsigned long *inputBytes,
                                  unsigned int *textureFormats,
                                  unsigned int *chunkCounts)
{
    // Start with the length of a multiple-image section header
    unsigned long total_length = 8;

    // Return 0 for bad arguments
    if (count == 0 || count > 2
        || inputBytes == NULL
        || textureFormats == NULL
        || chunkCounts == NULL)
    {
        return 0;
    }

    for (int i = 0; i < count; i++)
    {
        if (chunkCounts[i] == 0)
        {
            return 0;
        }

        // Assume snappy, the worst case
        total_length += hap_max_encoded_length(inputBytes[i], textureFormats[i], HapCompressorSnappy, chunkCounts[i]);
    }

    return total_length;
}

static unsigned int hap_encode_texture(const void *inputBuffer, unsigned long inputBufferBytes, unsigned int textureFormat,
                                       unsigned int compressor, unsigned int chunkCount, void *outputBuffer,
                                       unsigned long outputBufferBytes, unsigned long *outputBufferBytesUsed)
{
    size_t top_section_header_length;
    size_t top_section_length;
    unsigned int storedCompressor;
    unsigned int storedFormat;

    /*
     Check arguments
     */
    if (inputBuffer == NULL
        || inputBufferBytes == 0
        || (textureFormat != HapTextureFormat_RGB_DXT1
            && textureFormat != HapTextureFormat_RGBA_DXT5
            && textureFormat != HapTextureFormat_YCoCg_DXT5
            && textureFormat != HapTextureFormat_A_RGTC1
            && textureFormat != HapTextureFormat_RGBA_BPTC_UNORM
            && textureFormat != HapTextureFormat_RGB_BPTC_UNSIGNED_FLOAT
            && textureFormat != HapTextureFormat_RGB_BPTC_SIGNED_FLOAT
            )
        || (compressor != HapCompressorNone
            && compressor != HapCompressorSnappy
            && compressor != HapCompressorLZ4
            )
        || outputBuffer == NULL
        || outputBufferBytesUsed == NULL
        )
    {
        return HapResult_Bad_Arguments;
    }
    else if (outputBufferBytes < hap_max_encoded_length(inputBufferBytes, textureFormat, compressor, chunkCount))
    {
        return HapResult_Buffer_Too_Small;
    }

    /*
     To store frames of length greater than can be expressed in three bytes, we use an eight byte header (the last four bytes are the
     frame size). We don't know the compressed size until we have performed compression, but we know the worst-case size
     (the uncompressed size), so choose header-length based on that.

     A simpler encoder could always use the eight-byte header variation.
     */
    if (inputBufferBytes > kHapUInt24Max)
    {
        top_section_header_length = 8U;
    }
    else
    {
        top_section_header_length = 4U;
    }

    if (compressor == HapCompressorSnappy
        || compressor == HapCompressorLZ4
        )
    {
        /*
         We attempt to chunk as requested, and if resulting frame is larger than it is uncompressed then
         store frame uncompressed
         */

        size_t decode_instructions_length;
        size_t chunk_size, compress_buffer_remaining;
        uint8_t *second_stage_compressor_table;
        void *chunk_size_table;
        char *compressed_data;
        unsigned int i;

        chunkCount = hap_limited_chunk_count_for_frame(inputBufferBytes, textureFormat, chunkCount);
        decode_instructions_length = hap_decode_instructions_length(chunkCount);

        // Check we have space for the Decode Instructions Container
        if ((inputBufferBytes + decode_instructions_length + 4) > kHapUInt24Max)
        {
            top_section_header_length = 8U;
        }

        second_stage_compressor_table = ((uint8_t *)outputBuffer) + top_section_header_length + 4 + 4;
        chunk_size_table = ((uint8_t *)outputBuffer) + top_section_header_length + 4 + 4 + chunkCount + 4;

        chunk_size = inputBufferBytes / chunkCount;

        // write the Decode Instructions section header
        hap_write_section_header(((uint8_t *)outputBuffer) + top_section_header_length, 4U, decode_instructions_length, kHapSectionDecodeInstructionsContainer);
        // write the Second Stage Compressor Table section header
        hap_write_section_header(((uint8_t *)outputBuffer) + top_section_header_length + 4U, 4U, chunkCount, kHapSectionChunkSecondStageCompressorTable);
        // write the Chunk Size Table section header
        hap_write_section_header(((uint8_t *)outputBuffer) + top_section_header_length + 4U + 4U + chunkCount, 4U, chunkCount * 4U, kHapSectionChunkSizeTable);

        compressed_data = (char *)(((uint8_t *)outputBuffer) + top_section_header_length + 4 + decode_instructions_length);

        compress_buffer_remaining = outputBufferBytes - top_section_header_length - 4 - decode_instructions_length;

        top_section_length = 4 + decode_instructions_length;

#ifdef HAP_USE_NVCOMP
        if (hap_cuda_device_available())
        {
            size_t gpu_written = 0;
            unsigned int gpu_result = hap_nvcomp_compress_chunks(
                compressor,
                inputBuffer,
                chunk_size,
                chunkCount,
                compressed_data,
                compress_buffer_remaining,
                second_stage_compressor_table,
                chunk_size_table,
                &gpu_written);
            if (gpu_result == HapResult_No_Error)
            {
                top_section_length += gpu_written;
                goto done_compressing;
            }
            /* GPU path failed; fall through to CPU path below */
        }
#endif /* HAP_USE_NVCOMP */

        for (i = 0; i < chunkCount; i++) {
            size_t chunk_packed_length = compress_buffer_remaining;
            const char *chunk_input_start = (const char *)(((uint8_t *)inputBuffer) + (chunk_size * i));
            if (compressor == HapCompressorSnappy)
            {
                snappy_status result = snappy_compress(chunk_input_start, chunk_size, (char *)compressed_data, &chunk_packed_length);
                if (result != SNAPPY_OK)
                {
                    return HapResult_Internal_Error;
                }
            }
#ifdef LZ4LIB_API
            else if (compressor == HapCompressorLZ4)
            {
                if (compress_buffer_remaining < 4)
                {
                    return HapResult_Buffer_Too_Small;
                }
                hap_write_4_byte_uint(compressed_data, (unsigned int)chunk_size);
                int max_dest = (int)compress_buffer_remaining - 4;
                int lz4_size = LZ4_compress_default(chunk_input_start,
                                                    compressed_data + 4,
                                                    (int)chunk_size,
                                                    max_dest);
                if (lz4_size <= 0)
                {
                    return HapResult_Internal_Error;
                }
                chunk_packed_length = (size_t)lz4_size + 4;
            }
#endif

            if (compressor == HapCompressorNone || chunk_packed_length >= chunk_size)
            {
                // store the chunk uncompressed
                memcpy(compressed_data, chunk_input_start, chunk_size);
                chunk_packed_length = chunk_size;
                second_stage_compressor_table[i] = kHapCompressorNone;
            }
            else
            {
                // ie we used snappy and saved some space
                second_stage_compressor_table[i] = kHapCompressorSnappy;
                if (compressor == HapCompressorLZ4)
                {
                    second_stage_compressor_table[i] = kHapCompressorLZ4;
                }
            }
            hap_write_4_byte_uint(((uint8_t *)chunk_size_table) + (i * 4), chunk_packed_length);
            compressed_data += chunk_packed_length;
            top_section_length += chunk_packed_length;
            compress_buffer_remaining -= chunk_packed_length;
        }

#ifdef HAP_USE_NVCOMP
done_compressing:;
#endif

        if (top_section_length < inputBufferBytes + top_section_header_length)
        {
            // use the complex storage because snappy compression saved space
            storedCompressor = kHapCompressorComplex;
        }
        else
        {
            // Signal to store the frame uncompressed
            compressor = HapCompressorNone;
        }
    }

    if (compressor == HapCompressorNone)
    {
        memcpy(((uint8_t *)outputBuffer) + top_section_header_length, inputBuffer, inputBufferBytes);
        top_section_length = inputBufferBytes;
        storedCompressor = kHapCompressorNone;
    }

    storedFormat = hap_texture_format_identifier_for_format_constant(textureFormat);

    hap_write_section_header(outputBuffer, top_section_header_length, top_section_length, hap_4_bit_packed_byte(storedCompressor, storedFormat));

    *outputBufferBytesUsed = top_section_length + top_section_header_length;

    return HapResult_No_Error;
}

unsigned int HapEncode(unsigned int count,
                       const void **inputBuffers, unsigned long *inputBuffersBytes,
                       unsigned int *textureFormats,
                       unsigned int *compressors,
                       unsigned int *chunkCounts,
                       void *outputBuffer, unsigned long outputBufferBytes,
                       unsigned long *outputBufferBytesUsed)
{
    size_t top_section_header_length;
    size_t top_section_length;
    unsigned long section_length;

    if (count == 0 || count > 2 // A frame must contain one or two textures
        || inputBuffers == NULL
        || inputBuffersBytes == NULL
        || textureFormats == NULL
        || compressors == NULL
        || chunkCounts == NULL
        || outputBuffer == NULL
        || outputBufferBytes == 0
        || outputBufferBytesUsed == NULL)
    {
        return HapResult_Bad_Arguments;
    }

    for (int i = 0; i < count; i++)
    {
        if (chunkCounts[i] == 0)
        {
            return HapResult_Bad_Arguments;
        }
    }

    if (count == 1)
    {
        // Encode without the multi-image layout
        return hap_encode_texture(inputBuffers[0],
                                  inputBuffersBytes[0],
                                  textureFormats[0],
                                  compressors[0],
                                  chunkCounts[0],
                                  outputBuffer,
                                  outputBufferBytes,
                                  outputBufferBytesUsed);
    }
    else if ((textureFormats[0] != HapTextureFormat_YCoCg_DXT5 && textureFormats[1] != HapTextureFormat_YCoCg_DXT5)
             && (textureFormats[0] != HapTextureFormat_A_RGTC1 && textureFormats[1] != HapTextureFormat_A_RGTC1))
    {
        /*
         Permitted combinations:
         HapTextureFormat_YCoCg_DXT5 + HapTextureFormat_A_RGTC1
         */
        return HapResult_Bad_Arguments;
    }
    else
    {
        // Calculate the worst-case size for the top section and choose a header-length based on that
        top_section_length = 0;
        for (int i = 0; i < count; i++)
        {
            top_section_length += inputBuffersBytes[i] + hap_decode_instructions_length(chunkCounts[i]) + 4;
        }

        if (top_section_length > kHapUInt24Max)
        {
            top_section_header_length = 8U;
        }
        else
        {
            top_section_header_length = 4U;
        }

        // Encode each texture
        top_section_length = 0;
        for (int i = 0; i < count; i++)
        {
            void *section = ((uint8_t *)outputBuffer) + top_section_header_length + top_section_length;
            unsigned int result = hap_encode_texture(inputBuffers[i],
                                                     inputBuffersBytes[i],
                                                     textureFormats[i],
                                                     compressors[i],
                                                     chunkCounts[i],
                                                     section,
                                                     outputBufferBytes - (top_section_header_length + top_section_length),
                                                     &section_length);
            if (result != HapResult_No_Error)
            {
                return result;
            }
            top_section_length += section_length;
        }

        hap_write_section_header(outputBuffer, top_section_header_length, top_section_length, kHapSectionMultipleImages);

        *outputBufferBytesUsed = top_section_length + top_section_header_length;

        return HapResult_No_Error;
    }
}

static void hap_decode_chunk(HapChunkDecodeInfo chunks[], unsigned int index)
{
    if (chunks)
    {
        if (chunks[index].compressor == kHapCompressorSnappy)
        {
            snappy_status snappy_result = snappy_uncompress(chunks[index].compressed_chunk_data,
                                                            chunks[index].compressed_chunk_size,
                                                            chunks[index].uncompressed_chunk_data,
                                                            &chunks[index].uncompressed_chunk_size);

            switch (snappy_result)
            {
                case SNAPPY_INVALID_INPUT:
                    chunks[index].result = HapResult_Bad_Frame;
                    break;
                case SNAPPY_OK:
                    chunks[index].result = HapResult_No_Error;
                    break;
                default:
                    chunks[index].result = HapResult_Internal_Error;
                    break;
            }
        }
#ifdef LZ4LIB_API
        else if (chunks[index].compressor == kHapCompressorLZ4)
        {
            chunks[index].uncompressed_chunk_size = hap_read_4_byte_uint(chunks[index].compressed_chunk_data);
            int lz4_result = LZ4_decompress_safe(chunks[index].compressed_chunk_data + 4, chunks[index].uncompressed_chunk_data,
                chunks[index].compressed_chunk_size, chunks[index].uncompressed_chunk_size);
            if (lz4_result != chunks[index].uncompressed_chunk_size)
            {
                chunks[index].result = HapResult_Bad_Frame;
            }
            else
            {
                chunks[index].result = HapResult_No_Error;
            }
        }
#endif
        else if (chunks[index].compressor == kHapCompressorNone)
        {
            memcpy(chunks[index].uncompressed_chunk_data,
                   chunks[index].compressed_chunk_data,
                   chunks[index].compressed_chunk_size);
            chunks[index].result = HapResult_No_Error;
        }
        else
        {
            chunks[index].result = HapResult_Bad_Frame;
        }
    }
}

static unsigned int hap_decode_header_complex_instructions(const void *texture_section, uint32_t texture_section_length, int * chunk_count,
                                                   const void **compressors, const void **chunk_sizes, const void **chunk_offsets, const char **frame_data){
    int result = HapResult_No_Error;
    const void *section_start;
    uint32_t section_header_length;
    uint32_t section_length;
    unsigned int section_type;
    size_t bytes_remaining = 0;

    *compressors = NULL;
    *chunk_sizes = NULL;
    *chunk_offsets = NULL;

    result = hap_read_section_header(texture_section, texture_section_length, &section_header_length, &section_length, &section_type);

    if (result == HapResult_No_Error && section_type != kHapSectionDecodeInstructionsContainer)
    {
        result = HapResult_Bad_Frame;
    }

    if (result != HapResult_No_Error)
    {
        return result;
    }

    /*
     Frame data follows immediately after the Decode Instructions Container
     */
    *frame_data = ((const char *)texture_section) + section_header_length + section_length;

    /*
     Step through the sections inside the Decode Instructions Container
     */
    section_start = ((uint8_t *)texture_section) + section_header_length;
    bytes_remaining = section_length;

    while (bytes_remaining > 0) {
        unsigned int section_chunk_count = 0;
        result = hap_read_section_header(section_start, bytes_remaining, &section_header_length, &section_length, &section_type);
        if (result != HapResult_No_Error)
        {
            return result;
        }
        section_start = ((uint8_t *)section_start) + section_header_length;
        switch (section_type) {
            case kHapSectionChunkSecondStageCompressorTable:
                *compressors = section_start;
                section_chunk_count = section_length;
                break;
            case kHapSectionChunkSizeTable:
                *chunk_sizes = section_start;
                section_chunk_count = section_length / 4;
                break;
            case kHapSectionChunkOffsetTable:
                *chunk_offsets = section_start;
                section_chunk_count = section_length / 4;
                break;
            default:
                // Ignore unrecognized sections
                break;
        }

        /*
         If we calculated a chunk count and already have one, make sure they match
         */
        if (section_chunk_count != 0)
        {
            if ((*chunk_count) != 0 && section_chunk_count != (*chunk_count))
            {
                return HapResult_Bad_Frame;
            }
            *chunk_count = section_chunk_count;
        }

        section_start = ((uint8_t *)section_start) + section_length;
        bytes_remaining -= section_header_length + section_length;
    }

    /*
     The Chunk Second-Stage Compressor Table and Chunk Size Table are required
     */
    if (*compressors == NULL || *chunk_sizes == NULL)
    {
        return HapResult_Bad_Frame;
    }
    return result;
}

unsigned int hap_decode_single_texture(const void *texture_section, uint32_t texture_section_length,
                                       unsigned int texture_section_type,
                                       HapDecodeCallback callback, void *info,
                                       void *outputBuffer, unsigned long outputBufferBytes,
                                       void** pOutputBuffer,
                                       HapAlloc alloc,
                                       unsigned long *outputBufferBytesUsed,
                                       unsigned int *outputBufferTextureFormat)
{
    int result = HapResult_No_Error;
    unsigned int textureFormat;
    unsigned int compressor;
    size_t bytesUsed = 0;

    /*
     One top-level section type describes texture-format and second-stage compression
     Hap compressor/format constants can be unpacked by reading the top and bottom four bits.
     */
    compressor = hap_top_4_bits(texture_section_type);
    textureFormat = hap_bottom_4_bits(texture_section_type);

    /*
     Pass the texture format out
     */
    *outputBufferTextureFormat = hap_texture_format_constant_for_format_identifier(textureFormat);
    if (*outputBufferTextureFormat == 0)
    {
        return HapResult_Bad_Frame;
    }

    if (compressor == kHapCompressorComplex)
    {
        /*
         The top-level section should contain a Decode Instructions Container followed by frame data
         */
        int chunk_count = 0;
        const void *compressors = NULL;
        const void *chunk_sizes = NULL;
        const void *chunk_offsets = NULL;
        const char *frame_data = NULL;

        result = hap_decode_header_complex_instructions(texture_section, texture_section_length, &chunk_count, &compressors, &chunk_sizes, &chunk_offsets, &frame_data);

        if (result != HapResult_No_Error)
        {
            return result;
        }

        if (chunk_count > 0)
        {
            /*
             Step through the chunks, storing information for their decompression
             */
            HapChunkDecodeInfo *chunk_info = (HapChunkDecodeInfo *)malloc(sizeof(HapChunkDecodeInfo) * chunk_count);

            size_t running_compressed_chunk_size = 0;
            size_t running_uncompressed_chunk_size = 0;
            int i;

            if (chunk_info == NULL)
            {
                return HapResult_Internal_Error;
            }

            for (i = 0; i < chunk_count; i++) {

                chunk_info[i].compressor = *(((uint8_t *)compressors) + i);

                chunk_info[i].compressed_chunk_size = hap_read_4_byte_uint(((uint8_t *)chunk_sizes) + (i * 4));

                if (chunk_offsets)
                {
                    chunk_info[i].compressed_chunk_data = frame_data + hap_read_4_byte_uint(((uint8_t *)chunk_offsets) + (i * 4));
                }
                else
                {
                    chunk_info[i].compressed_chunk_data = frame_data + running_compressed_chunk_size;
                }

                running_compressed_chunk_size += chunk_info[i].compressed_chunk_size;

                if (chunk_info[i].compressor == kHapCompressorSnappy)
                {
                    snappy_status snappy_result = snappy_uncompressed_length(chunk_info[i].compressed_chunk_data,
                        chunk_info[i].compressed_chunk_size,
                        &(chunk_info[i].uncompressed_chunk_size));

                    if (snappy_result != SNAPPY_OK)
                    {
                        switch (snappy_result)
                        {
                        case SNAPPY_INVALID_INPUT:
                            result = HapResult_Bad_Frame;
                            break;
                        default:
                            result = HapResult_Internal_Error;
                            break;
                        }
                        break;
                    }
                }
                else if (chunk_info[i].compressor == kHapCompressorLZ4)
                {
                    chunk_info[i].uncompressed_chunk_size = hap_read_4_byte_uint(chunk_info[i].compressed_chunk_data);
                }
                else
                {
                    chunk_info[i].uncompressed_chunk_size = chunk_info[i].compressed_chunk_size;
                }

                chunk_info[i].uncompressed_chunk_data = (char *)(((uint8_t *)outputBuffer) + running_uncompressed_chunk_size);
                running_uncompressed_chunk_size += chunk_info[i].uncompressed_chunk_size;
            }

            if (!outputBufferBytes) {
                outputBuffer = alloc(running_uncompressed_chunk_size);
                outputBufferBytes = running_uncompressed_chunk_size;
                *pOutputBuffer = outputBuffer;
                size_t offset = 0;
                for (i = 0; i < chunk_count; i++) {
                    chunk_info[i].uncompressed_chunk_data = (char*)outputBuffer + offset;
                    offset += chunk_info[i].uncompressed_chunk_size;
                }
            }

            if (result == HapResult_No_Error && running_uncompressed_chunk_size > outputBufferBytes)
            {
                result = HapResult_Buffer_Too_Small;
            }

            if (result == HapResult_No_Error)
            {
                /*
                 Perform decompression
                 */
                bytesUsed = running_uncompressed_chunk_size;

#ifdef HAP_USE_NVCOMP
                /* If all chunks share the same GPU-acceleratable compressor, batch-decompress
                   on the GPU.  Fall back to the CPU path on any GPU failure. */
                {
                    int all_same = (chunk_count > 0);
                    unsigned int first_comp = all_same ? chunk_info[0].compressor : 0;
                    if (first_comp != kHapCompressorLZ4 && first_comp != kHapCompressorSnappy)
                        all_same = 0;
                    for (int ci = 1; ci < chunk_count && all_same; ci++)
                    {
                        if (chunk_info[ci].compressor != first_comp)
                            all_same = 0;
                    }

                    if (all_same && hap_cuda_device_available() &&
                        hap_nvcomp_decompress_chunks(first_comp, chunk_info, chunk_count) == HapResult_No_Error)
                    {
                        /* GPU decompression succeeded; chunk_info[].result already set */
                        goto check_chunk_results;
                    }
                }
#endif /* HAP_USE_NVCOMP */

                if (chunk_count == 1)
                {
                    /*
                     We don't invoke the callback for one chunk, just decode it directly
                     */
                    hap_decode_chunk(chunk_info, 0);
                }
                else
                {
                    callback((HapDecodeWorkFunction)hap_decode_chunk, chunk_info, chunk_count, info);
                }

#ifdef HAP_USE_NVCOMP
check_chunk_results:;
#endif
                /*
                 Check to see if we encountered any errors and report one of them
                 */
                for (i = 0; i < chunk_count; i++)
                {
                    if (chunk_info[i].result != HapResult_No_Error)
                    {
                        result = chunk_info[i].result;
                        break;
                    }
                }
            }

            free(chunk_info);

            if (result != HapResult_No_Error)
            {
                return result;
            }
        }
    }
    else if (compressor == kHapCompressorSnappy)
    {
        /*
         Only one section is present containing a single block of snappy-compressed texture data
         */
        snappy_status snappy_result = snappy_uncompressed_length((const char *)texture_section, texture_section_length, &bytesUsed);
        if (snappy_result != SNAPPY_OK)
        {
            return HapResult_Internal_Error;
        }
        if (!outputBufferBytes) {
            outputBuffer = alloc(bytesUsed);
            outputBufferBytes = bytesUsed;
            *pOutputBuffer = outputBuffer;
        }
        if (bytesUsed > outputBufferBytes)
        {
            return HapResult_Buffer_Too_Small;
        }
#ifdef HAP_USE_NVCOMP
        if (hap_cuda_device_available())
        {
            HapChunkDecodeInfo chunk = {0};
            chunk.compressor            = kHapCompressorSnappy;
            chunk.compressed_chunk_data = (const char *)texture_section;
            chunk.compressed_chunk_size = texture_section_length;
            chunk.uncompressed_chunk_data = (char *)outputBuffer;
            chunk.uncompressed_chunk_size = bytesUsed;
            if (hap_nvcomp_decompress_chunks(kHapCompressorSnappy, &chunk, 1) == HapResult_No_Error)
                goto end;
            /* GPU failed; fall through to CPU */
        }
#endif /* HAP_USE_NVCOMP */
        snappy_result = snappy_uncompress((const char *)texture_section, texture_section_length, (char *)outputBuffer, &bytesUsed);
        if (snappy_result != SNAPPY_OK)
        {
            return HapResult_Internal_Error;
        }
    }
#ifdef LZ4LIB_API
    else if (compressor == kHapCompressorLZ4)
    {
        bytesUsed = hap_read_4_byte_uint(texture_section);
        if (!outputBufferBytes) {
            outputBuffer = alloc(bytesUsed);
            outputBufferBytes = bytesUsed;
            *pOutputBuffer = outputBuffer;
        }
        if (bytesUsed > outputBufferBytes)
        {
            return HapResult_Buffer_Too_Small;
        }
#ifdef HAP_USE_NVCOMP
        if (hap_cuda_device_available())
        {
            HapChunkDecodeInfo chunk = {0};
            chunk.compressor            = kHapCompressorLZ4;
            chunk.compressed_chunk_data = (const char *)texture_section;
            chunk.compressed_chunk_size = texture_section_length;
            chunk.uncompressed_chunk_data = (char *)outputBuffer;
            chunk.uncompressed_chunk_size = bytesUsed;
            if (hap_nvcomp_decompress_chunks(kHapCompressorLZ4, &chunk, 1) == HapResult_No_Error)
                goto end;
            /* GPU failed; fall through to CPU */
        }
#endif /* HAP_USE_NVCOMP */
        int lz4_result = LZ4_decompress_safe((const char*)texture_section + 4, (char*)outputBuffer,
                                             texture_section_length, outputBufferBytes);
        if (lz4_result <= 0)
        {
            return HapResult_Internal_Error;
        }
    }
#endif
    else if (compressor == kHapCompressorNone)
    {
        /*
         Only one section is present containing a single block of uncompressed texture data
         */
        bytesUsed = texture_section_length;
        if (!outputBuffer || !outputBufferBytes) {
            outputBuffer = (void*)texture_section;
            /*outputBuffer = alloc(bytesUsed);*/
            outputBufferBytes = bytesUsed;
            *pOutputBuffer = outputBuffer;
            goto end;
        }
        if (texture_section_length > outputBufferBytes)
        {
            return HapResult_Buffer_Too_Small;
        }
        memcpy(outputBuffer, texture_section, texture_section_length);
    }
    else
    {
        return HapResult_Bad_Frame;
    }
end:
    /*
     Fill out the remaining return value
     */
    if (outputBufferBytesUsed != NULL)
    {
        *outputBufferBytesUsed = bytesUsed;
    }

    return HapResult_No_Error;
}

int hap_get_section_at_index(const void *input_buffer, uint32_t input_buffer_bytes,
                             unsigned int index,
                             const void **section, uint32_t *section_length, unsigned int *section_type)
{
    int result;
    uint32_t section_header_length;

    result = hap_read_section_header(input_buffer, input_buffer_bytes, &section_header_length, section_length, section_type);

    if (result != HapResult_No_Error)
    {
        return result;
    }

    if (*section_type == kHapSectionMultipleImages)
    {
        /*
         Step through until we find the section at index
         */
        size_t offset = 0;
        size_t top_section_length = *section_length;
        input_buffer = ((uint8_t *)input_buffer) + section_header_length;
        section_header_length = 0;
        *section_length = 0;
        for (int i = 0; i <= index; i++) {
            offset += section_header_length + *section_length;
            if (offset >= top_section_length)
            {
                return HapResult_Bad_Arguments;
            }
            result = hap_read_section_header(((uint8_t *)input_buffer) + offset,
                                             top_section_length - offset,
                                             &section_header_length,
                                             section_length,
                                             section_type);
            if (result != HapResult_No_Error)
            {
                return result;
            }
        }
        offset += section_header_length;
        *section = ((uint8_t *)input_buffer) + offset;
        return HapResult_No_Error;
    }
    else if (index == 0)
    {
        /*
         A single-texture frame with the texture as the top section.
         */
        *section = ((uint8_t *)input_buffer) + section_header_length;
        return HapResult_No_Error;
    }
    else
    {
        *section = NULL;
        *section_length = 0;
        *section_type = 0;
        return HapResult_Bad_Arguments;
    }
}

static void HapDecodeCallbackDefault(HapDecodeWorkFunction function, void *p, unsigned int count, void *info)
{
    int i;
    for (i = 0; i < count; i++) {
        // Invoke your multithreading mechanism to cause this function to be called
        // on a suitable number of threads.
        function(p, i);
    }
}

unsigned int HapDecode(const void *inputBuffer, unsigned long inputBufferBytes,
                       unsigned int index,
                       HapDecodeCallback callback, void *info,
                       void *outputBuffer, unsigned long outputBufferBytes,
                       unsigned long *outputBufferBytesUsed,
                       unsigned int *outputBufferTextureFormat)
{
    int result = HapResult_No_Error;
    const void *section;
    uint32_t section_length;
    unsigned int section_type;

    /*
     Check arguments
     */
    if (inputBuffer == NULL
        || index > 1
        || callback == NULL
        || outputBuffer == NULL
        || outputBufferTextureFormat == NULL
        )
    {
        return HapResult_Bad_Arguments;
    }

    /*
     Locate the section at the given index, which will either be the top-level section in a single texture image, or one of the
     sections inside a multi-image top-level section.
     */
    result = hap_get_section_at_index(inputBuffer, inputBufferBytes, index, &section, &section_length, &section_type);

    if (result == HapResult_No_Error)
    {
        /*
         Decode the located texture
         */
        result = hap_decode_single_texture(section,
                                           section_length,
                                           section_type,
                                           callback, info,
                                           outputBuffer,
                                           outputBufferBytes,
                                           NULL,
                                           NULL,
                                           outputBufferBytesUsed,
                                           outputBufferTextureFormat);
    }

    return result;
}

unsigned int HapDecodeMinCopy(const void *inputBuffer, unsigned long inputBufferBytes,
                       unsigned int index,
                       HapAlloc alloc,
                       void **outputBuffer,
                       unsigned long *outputBufferBytesUsed,
                       unsigned int *outputBufferTextureFormat)
{
    int result = HapResult_No_Error;
    const void *section;
    uint32_t section_length;
    unsigned int section_type;

    /*
     Check arguments
     */
    if (inputBuffer == NULL
        || index > 1
        || alloc == NULL
        || outputBuffer == NULL
        || outputBufferTextureFormat == NULL
        )
    {
        return HapResult_Bad_Arguments;
    }

    /*
     Locate the section at the given index, which will either be the top-level section in a single texture image, or one of the
     sections inside a multi-image top-level section.
     */
    result = hap_get_section_at_index(inputBuffer, inputBufferBytes, index, &section, &section_length, &section_type);

    if (result == HapResult_No_Error)
    {
        /*
         Decode the located texture
         */
        result = hap_decode_single_texture(section,
                                           section_length,
                                           section_type,
                                           HapDecodeCallbackDefault, NULL,
                                           NULL,
                                           0,
                                           outputBuffer,
                                           alloc,
                                           outputBufferBytesUsed,
                                           outputBufferTextureFormat);
    }

    return result;
}

unsigned int HapGetFrameTextureCount(const void *inputBuffer, unsigned long inputBufferBytes, unsigned int *outputTextureCount)
{
    int result;
    uint32_t section_header_length;
    uint32_t section_length;
    unsigned int section_type;

    result = hap_read_section_header(inputBuffer, inputBufferBytes, &section_header_length, &section_length, &section_type);

    if (result != HapResult_No_Error)
    {
        return result;
    }

    if (section_type == kHapSectionMultipleImages)
    {
        /*
         Step through, counting sections
         */
        uint32_t offset = section_header_length;
        uint32_t top_section_length = section_length;
        *outputTextureCount = 0;
        while (offset < top_section_length) {
            result = hap_read_section_header(((uint8_t *)inputBuffer) + offset,
                                             inputBufferBytes - offset,
                                             &section_header_length,
                                             &section_length,
                                             &section_type);
            if (result != HapResult_No_Error)
            {
                return result;
            }
            offset += section_header_length + section_length;
            *outputTextureCount += 1;
        }
        return HapResult_No_Error;
    }
    else
    {
        /*
         A single-texture frame with the texture as the top section.
         */
        *outputTextureCount = 1;
        return HapResult_No_Error;
    }
}

unsigned int HapGetFrameTextureFormat(const void *inputBuffer, unsigned long inputBufferBytes, unsigned int index, unsigned int *outputBufferTextureFormat)
{
    unsigned int result = HapResult_No_Error;
    const void *section;
    uint32_t section_length;
    unsigned int section_type;
    /*
     Check arguments
     */
    if (inputBuffer == NULL
        || index > 1
        || outputBufferTextureFormat == NULL
        )
    {
        return HapResult_Bad_Arguments;
    }
    /*
     Locate the section at the given index, which will either be the top-level section in a single texture image, or one of the
     sections inside a multi-image top-level section.
     */
    result = hap_get_section_at_index(inputBuffer, inputBufferBytes, index, &section, &section_length, &section_type);

    if (result == HapResult_No_Error)
    {
        /*
         Pass the API enum value to match the constant out
         */
        *outputBufferTextureFormat = hap_texture_format_constant_for_format_identifier(hap_bottom_4_bits(section_type));
        /*
         Check a valid format was present
         */
        if (*outputBufferTextureFormat == 0)
        {
            result = HapResult_Bad_Frame;
        }
    }
    return result;
}

unsigned int HapGetFrameTextureChunkCount(const void *inputBuffer, unsigned long inputBufferBytes, unsigned int index, int *chunk_count)
{
    unsigned int result = HapResult_No_Error;
    const void *section;
    uint32_t section_length;
    unsigned int section_type;
    *chunk_count = 0;

    /*
     Check arguments
     */
    if (inputBuffer == NULL
        || index > 1
        )
    {
        return HapResult_Bad_Arguments;
    }
    /*
     Locate the section at the given index, which will either be the top-level section in a single texture image, or one of the
     sections inside a multi-image top-level section.
     */
    result = hap_get_section_at_index(inputBuffer, inputBufferBytes, index, &section, &section_length, &section_type);

    if (result == HapResult_No_Error)
    {
        unsigned int compressor;

        /*
         One top-level section type describes texture-format and second-stage compression
         Hap compressor/format constants can be unpacked by reading the top and bottom four bits.
         */
        compressor = hap_top_4_bits(section_type);

        if (compressor == kHapCompressorComplex)
        {
            /*
             The top-level section should contain a Decode Instructions Container followed by frame data
             */
            const void *compressors = NULL;
            const void *chunk_sizes = NULL;
            const void *chunk_offsets = NULL;
            const char *frame_data = NULL;

            result = hap_decode_header_complex_instructions(section, section_length, chunk_count, &compressors, &chunk_sizes, &chunk_offsets, &frame_data);

            if (result != HapResult_No_Error)
            {
                return result;
            }
        }
        else if (compressor == kHapCompressorSnappy || compressor == kHapCompressorLZ4 || compressor == kHapCompressorNone)
        {
            *chunk_count = 1;
        }
        else
        {
            return HapResult_Bad_Frame;
        }
    }
    return result;
}
