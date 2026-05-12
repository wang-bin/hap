/*
 hap.h

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

#ifndef hap_h
#define hap_h

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 These match the constants defined by GL_EXT_texture_compression_s3tc,
 GL_ARB_texture_compression_rgtc and GL_ARB_texture_compression_bptc.
 Values 0x01-0x0F are custom Hap-defined constants for formats with no
 directly corresponding OpenGL enum.
 */

enum HapTextureFormat {
    HapTextureFormat_RGB_DXT1 = 0x83F0,
    HapTextureFormat_RGBA_DXT5 = 0x83F3,
    HapTextureFormat_YCoCg_DXT5 = 0x01,
    HapTextureFormat_A_RGTC1 = 0x8DBB,
    HapTextureFormat_RGBA_BPTC_UNORM = 0x8E8C,
    HapTextureFormat_RGB_BPTC_UNSIGNED_FLOAT = 0x8E8F,
    HapTextureFormat_RGB_BPTC_SIGNED_FLOAT = 0x8E8E,
    /* YCbCr planar formats: each plane is independently BC4/RGTC1-compressed.
       Three planes (Y + Cb + Cr) form one YCbCr frame.  Subsampling (4:4:4,
       4:2:2, 4:2:0) is inferred from the relative sizes of the planes. */
    HapTextureFormat_Y_BC4 = 0x02,
    HapTextureFormat_Cb_BC4 = 0x03,
    HapTextureFormat_Cr_BC4 = 0x04,
};

enum HapCompressor {
    HapCompressorNone,
    HapCompressorSnappy,
    HapCompressorLZ4,
};

/*
 Chroma subsampling modes for YCbCr three-plane frames.
 The subsampling is encoded in the relative pixel dimensions (and therefore
 in the relative compressed byte sizes) of the Cb/Cr planes vs. the Y plane.
 See HapYCbCrPlaneByteSize() and HapConvertTextureRGBToYCbCr().
 */
enum HapYCbCrSubsampling {
    HapYCbCrSubsampling_444 = 0, /* Cb/Cr same width and height as Y */
    HapYCbCrSubsampling_422 = 1, /* Cb/Cr half width, full height      */
    HapYCbCrSubsampling_420 = 2, /* Cb/Cr half width, half height      */
};

enum HapResult {
    HapResult_No_Error = 0,
    HapResult_Bad_Arguments,
    HapResult_Buffer_Too_Small,
    HapResult_Bad_Frame,
    HapResult_Internal_Error
};

/*
 See HapDecode for descriptions of these function types.
 */
typedef void (*HapDecodeWorkFunction)(void *p, unsigned int index);
typedef void (*HapDecodeCallback)(HapDecodeWorkFunction function, void *p, unsigned int count, void *info);
typedef void* (*HapAlloc)(size_t size);

/*
 Returns the maximum size of an output buffer for a frame composed of one or more textures, or returns 0 on error.
 count is the number of textures (1, 2, or 3) and matches the number of values in the array arguments
 lengths is an array of input texture lengths in bytes
 textureFormats is an array of HapTextureFormats
 chunkCounts is an array of chunk counts (1 or more)
 */
unsigned long HapMaxEncodedLength(unsigned int count,
                                  unsigned long *lengths,
                                  unsigned int *textureFormats,
                                  unsigned int *chunkCounts);

/*
 Encodes one or multiple textures into one Hap frame, or returns an error.

 Permitted multiple-texture combinations are:
  HapTextureFormat_YCoCg_DXT5 + HapTextureFormat_A_RGTC1
  HapTextureFormat_Y_BC4 + HapTextureFormat_Cb_BC4 + HapTextureFormat_Cr_BC4

 Use HapMaxEncodedLength() to discover the minimal value for outputBufferBytes.
 count is the number of textures (1, 2, or 3) and matches the number of values in the array arguments
 inputBuffers is an array of count pointers to texture data
 inputBufferBytes is an array of texture data lengths in bytes
 textureFormats is an array of HapTextureFormats
 compressors is an array of HapCompressors
 chunkCounts is an array of chunk counts to permit multithreaded decoding (1 or more)
 outputBuffer is the destination buffer to receive the encoded frame
 outputBufferBytes is the destination buffer's length in bytes
 outputBufferBytesUsed will be set to the actual encoded length of the frame on return
*/
unsigned int HapEncode(unsigned int count,
                       const void **inputBuffers, unsigned long *inputBuffersBytes,
                       unsigned int *textureFormats,
                       unsigned int *compressors,
                       unsigned int *chunkCounts,
                       void *outputBuffer, unsigned long outputBufferBytes,
                       unsigned long *outputBufferBytesUsed);

/*
 Decodes a texture from inputBuffer which is a Hap frame.

 A frame may contain multiple textures which are to be combined to create the final image. Use HapGetFrameTextureCount()
 to discover the number of textures in a frame, and then access each texture by incrementing the index argument to this
 function.

 If the frame permits multithreaded decoding, callback will be called once for you to invoke a platform-appropriate
 mechanism to assign work to threads, and trigger that work by calling the function passed to your callback the number
 of times indicated by the count argument, usually from a number of different threads. This callback must not return
 until all the work has been completed.

 void MyHapDecodeCallback(HapDecodeWorkFunction function, void *p, unsigned int count, void *info)
 {
     int i;
     for (i = 0; i < count; i++) {
         // Invoke your multithreading mechanism to cause this function to be called
         // on a suitable number of threads.
         function(p, i);
     }
 }
 info is an argument for your own use to pass context to the callback.
 If the frame does not permit multithreaded decoding, callback will not be called.
 If outputBufferBytesUsed is not NULL then it will be set to the decoded length of the output buffer.
 outputBufferTextureFormat must be non-NULL, and will be set to one of the HapTextureFormat constants.
 */
unsigned int HapDecode(const void *inputBuffer, unsigned long inputBufferBytes,
                       unsigned int index,
                       HapDecodeCallback callback, void *info,
                       void *outputBuffer, unsigned long outputBufferBytes,
                       unsigned long *outputBufferBytesUsed,
                       unsigned int *outputBufferTextureFormat);

/*!
 If data is compressed outputBuffer is allocated by user provided alloc.
 Otherwise, result *outputBuffer is the uncompressed data address in input range (inputBuffer, inputBuffer + inputBufferBytes).
 */
unsigned int HapDecodeMinCopy(const void *inputBuffer, unsigned long inputBufferBytes,
                       unsigned int index,
                       HapAlloc alloc,
                       void **outputBuffer,
                       unsigned long *outputBufferBytesUsed,
                       unsigned int *outputBufferTextureFormat);
/*
 If this returns HapResult_No_Error then outputTextureCount is set to the count of textures in the frame.
 */
unsigned int HapGetFrameTextureCount(const void *inputBuffer, unsigned long inputBufferBytes, unsigned int *outputTextureCount);

/*
 On return sets outputBufferTextureFormat to a HapTextureFormat constant describing the format of the texture at index in the frame.
 */
unsigned int HapGetFrameTextureFormat(const void *inputBuffer, unsigned long inputBufferBytes, unsigned int index, unsigned int *outputBufferTextureFormat);

/*
 On return sets chunk_count to the chunk count value of the texture at index in the frame.
*/
unsigned int HapGetFrameTextureChunkCount(const void *inputBuffer, unsigned long inputBufferBytes, unsigned int index, int *chunk_count);

/*
 Returns the number of bytes required for a single BC4/RGTC1-compressed plane
 of a texture with the given dimensions and subsampling.

 For the Y (luma) plane pass isChroma = 0.
 For the Cb or Cr (chroma) planes pass isChroma = 1; the function will
 divide the width (and, for 4:2:0, also the height) as required.

 Width and height must each be a multiple of 4.
 Returns 0 for invalid arguments.
 */
unsigned long HapYCbCrPlaneByteSize(unsigned int width, unsigned int height,
                                    unsigned int subsampling, int isChroma);

/*
 Convert a BC-compressed RGB or RGBA texture to three BC4/RGTC1-compressed
 YCbCr planes (BT.709 full-range).

 The conversion pipeline is:
   1. Decode input BC texture to raw RGBA pixels.
   2. Convert RGBA -> full-resolution Y, Cb, Cr (BT.709).
   3. Optionally downsample Cb/Cr according to subsampling.
   4. Encode each plane to BC4/RGTC1.

 Backend selection priority: WebGPU (when compiled with -DHAVE_WEBGPU),
 then CUDA (when compiled with -DHAVE_CUDA), then a CPU fallback.

 inputBuffer        BC-compressed source texture data.
 inputBufferBytes   Size of inputBuffer in bytes.
 inputTextureFormat One of: HapTextureFormat_RGB_DXT1, HapTextureFormat_RGBA_DXT5.
 width              Texture width in pixels.  Must be a multiple of 4.
 height             Texture height in pixels. Must be a multiple of 4.
 subsampling        One of the HapYCbCrSubsampling values.
 outputY/Cb/Cr      Caller-allocated output buffers for BC4-compressed planes.
 outputYBytes etc.  Sizes of the output buffers; use HapYCbCrPlaneByteSize().
 outputYBytesUsed   On success, set to the number of bytes written to outputY.
 outputCbBytesUsed  On success, set to the number of bytes written to outputCb.
 outputCrBytesUsed  On success, set to the number of bytes written to outputCr.

 Returns HapResult_No_Error on success, or an error code.
 */
unsigned int HapConvertTextureRGBToYCbCr(
    const void *inputBuffer,     unsigned long  inputBufferBytes,
    unsigned int  inputTextureFormat,
    unsigned int  width,         unsigned int   height,
    unsigned int  subsampling,
    void         *outputY,       unsigned long  outputYBytes,  unsigned long *outputYBytesUsed,
    void         *outputCb,      unsigned long  outputCbBytes, unsigned long *outputCbBytesUsed,
    void         *outputCr,      unsigned long  outputCrBytes, unsigned long *outputCrBytesUsed);

/*
 Convert three BC4/RGTC1-compressed YCbCr planes to a BC-compressed
 RGB or RGBA texture (BT.709 full-range).

 The conversion pipeline is the inverse of HapConvertTextureRGBToYCbCr:
   1. Decode Y, Cb, Cr BC4 planes to raw single-channel pixels.
   2. Convert YCbCr -> RGBA (chroma upsampling handled by coordinate mapping).
   3. Encode RGBA to the requested BC format.

 inputY/Cb/Cr       BC4-compressed Y, Cb, Cr plane data.
 inputYBytes etc.   Sizes of the input buffers in bytes.
 subsampling        Subsampling that was used when encoding the planes.
 width              Y-plane texture width in pixels.  Must be a multiple of 4.
 height             Y-plane texture height in pixels. Must be a multiple of 4.
 outputTextureFormat Target format: HapTextureFormat_RGB_DXT1 or
                    HapTextureFormat_RGBA_DXT5.
 outputBuffer       Caller-allocated destination for the BC-compressed texture.
 outputBufferBytes  Size of outputBuffer; use HapYCbCrPlaneByteSize() with
                    isChroma=0 for size (same formula as BC4 planes).
 outputBufferBytesUsed  On success, set to bytes written.

 Returns HapResult_No_Error on success, or an error code.
 */
unsigned int HapConvertTextureYCbCrToRGB(
    const void   *inputY,        unsigned long  inputYBytes,
    const void   *inputCb,       unsigned long  inputCbBytes,
    const void   *inputCr,       unsigned long  inputCrBytes,
    unsigned int  subsampling,
    unsigned int  width,         unsigned int   height,
    unsigned int  outputTextureFormat,
    void         *outputBuffer,  unsigned long  outputBufferBytes,
    unsigned long *outputBufferBytesUsed);

#ifdef __cplusplus
}
#endif

#endif
