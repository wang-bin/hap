/*
 hap_convert_wgpu.c

 WebGPU backend for the BC<->YCbCr conversion path.

 This backend uses WebGPU C API + WGSL compute shaders for the colour-space
 conversion and chroma subsampling/upsampling stages. BC block
 decode/encode is reused from the CPU codec implementation.
 */

#include "hap_convert_wgpu.h"
#include "hap_convert.h"
#include "hap.h"

#include <stdlib.h>
#include <string.h>

#ifdef HAVE_WEBGPU

#include <webgpu/webgpu.h>
#include <time.h>

typedef struct HapWGPUContext {
    WGPUInstance instance;
    WGPUAdapter adapter;
    WGPUDevice device;
    WGPUQueue queue;
    int ready;
} HapWGPUContext;

static HapWGPUContext s_ctx = {0};
static int s_ctx_init_attempted = 0;

typedef struct HapWGPURequestAdapterState {
    int done;
    int ok;
    WGPUAdapter adapter;
} HapWGPURequestAdapterState;

typedef struct HapWGPURequestDeviceState {
    int done;
    int ok;
    WGPUDevice device;
} HapWGPURequestDeviceState;

typedef struct HapWGPUMapState {
    int done;
    int ok;
} HapWGPUMapState;

static void hap_wgpu_request_adapter_cb(WGPURequestAdapterStatus status,
                                        WGPUAdapter adapter,
                                        const char *message,
                                        void *userdata)
{
    (void)message;
    HapWGPURequestAdapterState *state = (HapWGPURequestAdapterState *)userdata;
    state->done = 1;
    state->ok = (status == WGPURequestAdapterStatus_Success && adapter != NULL);
    state->adapter = adapter;
}

static void hap_wgpu_request_device_cb(WGPURequestDeviceStatus status,
                                       WGPUDevice device,
                                       const char *message,
                                       void *userdata)
{
    (void)message;
    HapWGPURequestDeviceState *state = (HapWGPURequestDeviceState *)userdata;
    state->done = 1;
    state->ok = (status == WGPURequestDeviceStatus_Success && device != NULL);
    state->device = device;
}

static void hap_wgpu_map_cb(WGPUBufferMapAsyncStatus status, void *userdata)
{
    HapWGPUMapState *state = (HapWGPUMapState *)userdata;
    state->done = 1;
    state->ok = (status == WGPUBufferMapAsyncStatus_Success);
}

static int hap_wgpu_wait_flag(int *done, int max_ms)
{
    for (int i = 0; i < max_ms; ++i) {
        if (*done) return 1;
        struct timespec ts;
        ts.tv_sec = 0;
        ts.tv_nsec = 1000000L; /* 1ms */
        nanosleep(&ts, NULL);
    }
    return 0;
}

static int hap_wgpu_init_context(void)
{
    if (s_ctx.ready) {
        return 1;
    }
    if (s_ctx_init_attempted) {
        return 0;
    }
    s_ctx_init_attempted = 1;

    WGPUInstanceDescriptor instance_desc;
    memset(&instance_desc, 0, sizeof(instance_desc));
    s_ctx.instance = wgpuCreateInstance(&instance_desc);
    if (!s_ctx.instance) {
        return 0;
    }

    HapWGPURequestAdapterState adapter_state;
    memset(&adapter_state, 0, sizeof(adapter_state));
    WGPURequestAdapterOptions adapter_opts;
    memset(&adapter_opts, 0, sizeof(adapter_opts));
    adapter_opts.powerPreference = WGPUPowerPreference_HighPerformance;
    wgpuInstanceRequestAdapter(s_ctx.instance,
                               &adapter_opts,
                               hap_wgpu_request_adapter_cb,
                               &adapter_state);
    if (!hap_wgpu_wait_flag(&adapter_state.done, 3000) || !adapter_state.ok) {
        return 0;
    }
    s_ctx.adapter = adapter_state.adapter;

    HapWGPURequestDeviceState device_state;
    memset(&device_state, 0, sizeof(device_state));
    WGPUDeviceDescriptor device_desc;
    memset(&device_desc, 0, sizeof(device_desc));
    wgpuAdapterRequestDevice(s_ctx.adapter,
                             &device_desc,
                             hap_wgpu_request_device_cb,
                             &device_state);
    if (!hap_wgpu_wait_flag(&device_state.done, 3000) || !device_state.ok) {
        return 0;
    }
    s_ctx.device = device_state.device;
    s_ctx.queue = wgpuDeviceGetQueue(s_ctx.device);
    s_ctx.ready = (s_ctx.queue != NULL);
    return s_ctx.ready;
}

int hap_wgpu_available(void)
{
    return hap_wgpu_init_context();
}

static const char *kHapWGSL = ""
"struct Params {\n"
"    y_width: u32,\n"
"    y_height: u32,\n"
"    cb_width: u32,\n"
"    cb_height: u32,\n"
"};\n"
"@group(0) @binding(0) var<storage, read> rgba_in: array<u32>;\n"
"@group(0) @binding(1) var<storage, read_write> y_plane: array<u32>;\n"
"@group(0) @binding(2) var<storage, read_write> cb_plane: array<u32>;\n"
"@group(0) @binding(3) var<storage, read_write> cr_plane: array<u32>;\n"
"@group(0) @binding(4) var<uniform> params: Params;\n"
"\n"
"fn f2u8(v: f32) -> u32 {\n"
"    let c = clamp(v, 0.0, 255.0);\n"
"    return u32(c + 0.5);\n"
"}\n"
"\n"
"@compute @workgroup_size(8, 8, 1)\n"
"fn rgba_to_ycbcr_444(@builtin(global_invocation_id) gid: vec3<u32>) {\n"
"    if (gid.x >= params.y_width || gid.y >= params.y_height) { return; }\n"
"    let idx = gid.y * params.y_width + gid.x;\n"
"    let px = rgba_in[idx];\n"
"    let r = f32(px & 0xFFu) / 255.0;\n"
"    let g = f32((px >> 8u) & 0xFFu) / 255.0;\n"
"    let b = f32((px >> 16u) & 0xFFu) / 255.0;\n"
"    let yv  = (0.2126 * r + 0.7152 * g + 0.0722 * b) * 255.0;\n"
"    let cbv = (-0.1146 * r - 0.3854 * g + 0.5 * b) * 255.0 + 128.0;\n"
"    let crv = (0.5 * r - 0.4542 * g - 0.0458 * b) * 255.0 + 128.0;\n"
"    y_plane[idx] = f2u8(yv);\n"
"    cb_plane[idx] = f2u8(cbv);\n"
"    cr_plane[idx] = f2u8(crv);\n"
"}\n"
"\n"
"@compute @workgroup_size(8, 8, 1)\n"
"fn rgba_to_ycbcr_422(@builtin(global_invocation_id) gid: vec3<u32>) {\n"
"    if (gid.x >= params.cb_width || gid.y >= params.cb_height) { return; }\n"
"    let x0 = gid.x * 2u;\n"
"    let x1 = min(x0 + 1u, params.y_width - 1u);\n"
"    let y = gid.y;\n"
"    let i0 = y * params.y_width + x0;\n"
"    let i1 = y * params.y_width + x1;\n"
"    let p0 = rgba_in[i0];\n"
"    let p1 = rgba_in[i1];\n"
"    let r0 = f32(p0 & 0xFFu) / 255.0; let g0 = f32((p0 >> 8u) & 0xFFu) / 255.0; let b0 = f32((p0 >> 16u) & 0xFFu) / 255.0;\n"
"    let r1 = f32(p1 & 0xFFu) / 255.0; let g1 = f32((p1 >> 8u) & 0xFFu) / 255.0; let b1 = f32((p1 >> 16u) & 0xFFu) / 255.0;\n"
"    let y0 = (0.2126 * r0 + 0.7152 * g0 + 0.0722 * b0) * 255.0;\n"
"    let y1 = (0.2126 * r1 + 0.7152 * g1 + 0.0722 * b1) * 255.0;\n"
"    y_plane[i0] = f2u8(y0);\n"
"    y_plane[i1] = f2u8(y1);\n"
"    let cb0 = (-0.1146 * r0 - 0.3854 * g0 + 0.5 * b0) * 255.0 + 128.0;\n"
"    let cb1 = (-0.1146 * r1 - 0.3854 * g1 + 0.5 * b1) * 255.0 + 128.0;\n"
"    let cr0 = (0.5 * r0 - 0.4542 * g0 - 0.0458 * b0) * 255.0 + 128.0;\n"
"    let cr1 = (0.5 * r1 - 0.4542 * g1 - 0.0458 * b1) * 255.0 + 128.0;\n"
"    let cidx = gid.y * params.cb_width + gid.x;\n"
"    cb_plane[cidx] = f2u8((cb0 + cb1) * 0.5);\n"
"    cr_plane[cidx] = f2u8((cr0 + cr1) * 0.5);\n"
"}\n"
"\n"
"@compute @workgroup_size(8, 8, 1)\n"
"fn rgba_to_ycbcr_420(@builtin(global_invocation_id) gid: vec3<u32>) {\n"
"    if (gid.x >= params.cb_width || gid.y >= params.cb_height) { return; }\n"
"    let x0 = gid.x * 2u;\n"
"    let y0 = gid.y * 2u;\n"
"    let x1 = min(x0 + 1u, params.y_width - 1u);\n"
"    let y1 = min(y0 + 1u, params.y_height - 1u);\n"
"    let i00 = y0 * params.y_width + x0;\n"
"    let i01 = y0 * params.y_width + x1;\n"
"    let i10 = y1 * params.y_width + x0;\n"
"    let i11 = y1 * params.y_width + x1;\n"
"    let p00 = rgba_in[i00]; let p01 = rgba_in[i01]; let p10 = rgba_in[i10]; let p11 = rgba_in[i11];\n"
"    let r00 = f32(p00 & 0xFFu) / 255.0; let g00 = f32((p00 >> 8u) & 0xFFu) / 255.0; let b00 = f32((p00 >> 16u) & 0xFFu) / 255.0;\n"
"    let r01 = f32(p01 & 0xFFu) / 255.0; let g01 = f32((p01 >> 8u) & 0xFFu) / 255.0; let b01 = f32((p01 >> 16u) & 0xFFu) / 255.0;\n"
"    let r10 = f32(p10 & 0xFFu) / 255.0; let g10 = f32((p10 >> 8u) & 0xFFu) / 255.0; let b10 = f32((p10 >> 16u) & 0xFFu) / 255.0;\n"
"    let r11 = f32(p11 & 0xFFu) / 255.0; let g11 = f32((p11 >> 8u) & 0xFFu) / 255.0; let b11 = f32((p11 >> 16u) & 0xFFu) / 255.0;\n"
"    let y00 = (0.2126 * r00 + 0.7152 * g00 + 0.0722 * b00) * 255.0;\n"
"    let y01 = (0.2126 * r01 + 0.7152 * g01 + 0.0722 * b01) * 255.0;\n"
"    let y10 = (0.2126 * r10 + 0.7152 * g10 + 0.0722 * b10) * 255.0;\n"
"    let y11 = (0.2126 * r11 + 0.7152 * g11 + 0.0722 * b11) * 255.0;\n"
"    y_plane[i00] = f2u8(y00); y_plane[i01] = f2u8(y01); y_plane[i10] = f2u8(y10); y_plane[i11] = f2u8(y11);\n"
"    let cb00 = (-0.1146 * r00 - 0.3854 * g00 + 0.5 * b00) * 255.0 + 128.0;\n"
"    let cb01 = (-0.1146 * r01 - 0.3854 * g01 + 0.5 * b01) * 255.0 + 128.0;\n"
"    let cb10 = (-0.1146 * r10 - 0.3854 * g10 + 0.5 * b10) * 255.0 + 128.0;\n"
"    let cb11 = (-0.1146 * r11 - 0.3854 * g11 + 0.5 * b11) * 255.0 + 128.0;\n"
"    let cr00 = (0.5 * r00 - 0.4542 * g00 - 0.0458 * b00) * 255.0 + 128.0;\n"
"    let cr01 = (0.5 * r01 - 0.4542 * g01 - 0.0458 * b01) * 255.0 + 128.0;\n"
"    let cr10 = (0.5 * r10 - 0.4542 * g10 - 0.0458 * b10) * 255.0 + 128.0;\n"
"    let cr11 = (0.5 * r11 - 0.4542 * g11 - 0.0458 * b11) * 255.0 + 128.0;\n"
"    let cidx = gid.y * params.cb_width + gid.x;\n"
"    cb_plane[cidx] = f2u8((cb00 + cb01 + cb10 + cb11) * 0.25);\n"
"    cr_plane[cidx] = f2u8((cr00 + cr01 + cr10 + cr11) * 0.25);\n"
"}\n"
"\n"
"@group(0) @binding(0) var<storage, read> y_in: array<u32>;\n"
"@group(0) @binding(1) var<storage, read> cb_in: array<u32>;\n"
"@group(0) @binding(2) var<storage, read> cr_in: array<u32>;\n"
"@group(0) @binding(3) var<storage, read_write> rgba_out: array<u32>;\n"
"@group(0) @binding(4) var<uniform> ycbcr_params: Params;\n"
"\n"
"@compute @workgroup_size(8, 8, 1)\n"
"fn ycbcr_to_rgba(@builtin(global_invocation_id) gid: vec3<u32>) {\n"
"    if (gid.x >= ycbcr_params.y_width || gid.y >= ycbcr_params.y_height) { return; }\n"
"    let x = gid.x;\n"
"    let y = gid.y;\n"
"    let idx = y * ycbcr_params.y_width + x;\n"
"    let cx = x * ycbcr_params.cb_width / ycbcr_params.y_width;\n"
"    let cy = y * ycbcr_params.cb_height / ycbcr_params.y_height;\n"
"    let cidx = cy * ycbcr_params.cb_width + cx;\n"
"    let yf = f32(y_in[idx]) / 255.0;\n"
"    let cb = (f32(cb_in[cidx]) - 128.0) / 255.0;\n"
"    let cr = (f32(cr_in[cidx]) - 128.0) / 255.0;\n"
"    let r = (yf + 1.5748 * cr) * 255.0;\n"
"    let g = (yf - 0.1873 * cb - 0.4681 * cr) * 255.0;\n"
"    let b = (yf + 1.8556 * cb) * 255.0;\n"
"    let ru = f2u8(r);\n"
"    let gu = f2u8(g);\n"
"    let bu = f2u8(b);\n"
"    rgba_out[idx] = ru | (gu << 8u) | (bu << 16u) | (255u << 24u);\n"
"}\n";

typedef struct HapWGPUParams {
    uint32_t y_width;
    uint32_t y_height;
    uint32_t cb_width;
    uint32_t cb_height;
} HapWGPUParams;

static WGPUBuffer hap_wgpu_make_buffer(WGPUDevice device,
                                       size_t size,
                                       WGPUBufferUsage usage)
{
    WGPUBufferDescriptor desc;
    memset(&desc, 0, sizeof(desc));
    desc.size = size;
    desc.usage = usage;
    return wgpuDeviceCreateBuffer(device, &desc);
}

static int hap_wgpu_read_buffer(WGPUDevice device,
                                WGPUQueue queue,
                                WGPUBuffer src,
                                size_t size,
                                void *dst)
{
    WGPUBuffer staging = hap_wgpu_make_buffer(
        device, size, WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead);
    if (!staging) return 0;

    WGPUCommandEncoderDescriptor enc_desc;
    memset(&enc_desc, 0, sizeof(enc_desc));
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, &enc_desc);
    if (!encoder) {
        wgpuBufferRelease(staging);
        return 0;
    }
    wgpuCommandEncoderCopyBufferToBuffer(encoder, src, 0, staging, 0, size);

    WGPUCommandBufferDescriptor cb_desc;
    memset(&cb_desc, 0, sizeof(cb_desc));
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(encoder, &cb_desc);
    wgpuCommandEncoderRelease(encoder);
    if (!cmd) {
        wgpuBufferRelease(staging);
        return 0;
    }
    wgpuQueueSubmit(queue, 1, &cmd);
    wgpuCommandBufferRelease(cmd);

    HapWGPUMapState map_state;
    memset(&map_state, 0, sizeof(map_state));
    wgpuBufferMapAsync(staging,
                       WGPUMapMode_Read,
                       0,
                       size,
                       hap_wgpu_map_cb,
                       &map_state);
    if (!hap_wgpu_wait_flag(&map_state.done, 4000) || !map_state.ok) {
        wgpuBufferRelease(staging);
        return 0;
    }

    const void *mapped = wgpuBufferGetConstMappedRange(staging, 0, size);
    if (!mapped) {
        wgpuBufferUnmap(staging);
        wgpuBufferRelease(staging);
        return 0;
    }
    memcpy(dst, mapped, size);
    wgpuBufferUnmap(staging);
    wgpuBufferRelease(staging);
    return 1;
}

static int hap_wgpu_run_compute_rgba_to_ycbcr(const uint32_t *rgba_in_u32,
                                              unsigned int width,
                                              unsigned int height,
                                              unsigned int cb_w,
                                              unsigned int cb_h,
                                              unsigned int subsampling,
                                              uint8_t *Y,
                                              uint8_t *Cb,
                                              uint8_t *Cr)
{
    WGPUDevice device = s_ctx.device;
    WGPUQueue queue = s_ctx.queue;
    if (!device || !queue) return 0;

    size_t px_count = (size_t)width * height;
    size_t c_count = (size_t)cb_w * cb_h;
    size_t rgba_bytes = px_count * sizeof(uint32_t);
    size_t y_bytes_u32 = px_count * sizeof(uint32_t);
    size_t c_bytes_u32 = c_count * sizeof(uint32_t);
    size_t params_bytes = sizeof(HapWGPUParams);

    WGPUBuffer in_rgba = hap_wgpu_make_buffer(
        device, rgba_bytes, WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);
    WGPUBuffer out_y = hap_wgpu_make_buffer(
        device, y_bytes_u32, WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc);
    WGPUBuffer out_cb = hap_wgpu_make_buffer(
        device, c_bytes_u32, WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc);
    WGPUBuffer out_cr = hap_wgpu_make_buffer(
        device, c_bytes_u32, WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc);
    WGPUBuffer params = hap_wgpu_make_buffer(
        device, params_bytes, WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
    if (!in_rgba || !out_y || !out_cb || !out_cr || !params) {
        if (in_rgba) wgpuBufferRelease(in_rgba);
        if (out_y) wgpuBufferRelease(out_y);
        if (out_cb) wgpuBufferRelease(out_cb);
        if (out_cr) wgpuBufferRelease(out_cr);
        if (params) wgpuBufferRelease(params);
        return 0;
    }

    HapWGPUParams p;
    p.y_width = width;
    p.y_height = height;
    p.cb_width = cb_w;
    p.cb_height = cb_h;
    wgpuQueueWriteBuffer(queue, in_rgba, 0, rgba_in_u32, rgba_bytes);
    wgpuQueueWriteBuffer(queue, params, 0, &p, sizeof(p));

    WGPUShaderModuleWGSLDescriptor wgsl;
    memset(&wgsl, 0, sizeof(wgsl));
    wgsl.chain.sType = WGPUSType_ShaderModuleWGSLDescriptor;
    wgsl.code = kHapWGSL;
    WGPUShaderModuleDescriptor sm_desc;
    memset(&sm_desc, 0, sizeof(sm_desc));
    sm_desc.nextInChain = (const WGPUChainedStruct *)&wgsl;
    WGPUShaderModule module = wgpuDeviceCreateShaderModule(device, &sm_desc);
    if (!module) {
        wgpuBufferRelease(in_rgba);
        wgpuBufferRelease(out_y);
        wgpuBufferRelease(out_cb);
        wgpuBufferRelease(out_cr);
        wgpuBufferRelease(params);
        return 0;
    }

    const char *entry = "rgba_to_ycbcr_444";
    if (subsampling == HapYCbCrSubsampling_422) entry = "rgba_to_ycbcr_422";
    if (subsampling == HapYCbCrSubsampling_420) entry = "rgba_to_ycbcr_420";

    WGPUComputePipelineDescriptor cp_desc;
    memset(&cp_desc, 0, sizeof(cp_desc));
    cp_desc.compute.module = module;
    cp_desc.compute.entryPoint = entry;
    WGPUComputePipeline pipeline = wgpuDeviceCreateComputePipeline(device, &cp_desc);
    if (!pipeline) {
        wgpuShaderModuleRelease(module);
        wgpuBufferRelease(in_rgba);
        wgpuBufferRelease(out_y);
        wgpuBufferRelease(out_cb);
        wgpuBufferRelease(out_cr);
        wgpuBufferRelease(params);
        return 0;
    }

    WGPUBindGroupLayout bgl = wgpuComputePipelineGetBindGroupLayout(pipeline, 0);
    WGPUBindGroupEntry entries[5];
    memset(entries, 0, sizeof(entries));
    entries[0].binding = 0; entries[0].buffer = in_rgba; entries[0].size = rgba_bytes;
    entries[1].binding = 1; entries[1].buffer = out_y; entries[1].size = y_bytes_u32;
    entries[2].binding = 2; entries[2].buffer = out_cb; entries[2].size = c_bytes_u32;
    entries[3].binding = 3; entries[3].buffer = out_cr; entries[3].size = c_bytes_u32;
    entries[4].binding = 4; entries[4].buffer = params; entries[4].size = params_bytes;

    WGPUBindGroupDescriptor bg_desc;
    memset(&bg_desc, 0, sizeof(bg_desc));
    bg_desc.layout = bgl;
    bg_desc.entryCount = 5;
    bg_desc.entries = entries;
    WGPUBindGroup bg = wgpuDeviceCreateBindGroup(device, &bg_desc);
    if (!bg) {
        wgpuBindGroupLayoutRelease(bgl);
        wgpuComputePipelineRelease(pipeline);
        wgpuShaderModuleRelease(module);
        wgpuBufferRelease(in_rgba);
        wgpuBufferRelease(out_y);
        wgpuBufferRelease(out_cb);
        wgpuBufferRelease(out_cr);
        wgpuBufferRelease(params);
        return 0;
    }

    WGPUCommandEncoderDescriptor enc_desc;
    memset(&enc_desc, 0, sizeof(enc_desc));
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, &enc_desc);
    WGPUComputePassDescriptor pass_desc;
    memset(&pass_desc, 0, sizeof(pass_desc));
    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, &pass_desc);
    wgpuComputePassEncoderSetPipeline(pass, pipeline);
    wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, NULL);
    wgpuComputePassEncoderDispatchWorkgroups(
        pass,
        (subsampling == HapYCbCrSubsampling_444 ? (width + 7U) / 8U : (cb_w + 7U) / 8U),
        (subsampling == HapYCbCrSubsampling_444 ? (height + 7U) / 8U : (cb_h + 7U) / 8U),
        1);
    wgpuComputePassEncoderEnd(pass);
    wgpuComputePassEncoderRelease(pass);

    WGPUCommandBufferDescriptor cb_desc;
    memset(&cb_desc, 0, sizeof(cb_desc));
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(encoder, &cb_desc);
    wgpuCommandEncoderRelease(encoder);
    wgpuQueueSubmit(queue, 1, &cmd);
    wgpuCommandBufferRelease(cmd);

    uint32_t *tmp_y_u32 = (uint32_t *)malloc(y_bytes_u32);
    uint32_t *tmp_cb_u32 = (uint32_t *)malloc(c_bytes_u32);
    uint32_t *tmp_cr_u32 = (uint32_t *)malloc(c_bytes_u32);
    int ok = (tmp_y_u32 && tmp_cb_u32 && tmp_cr_u32);
    if (ok) ok = hap_wgpu_read_buffer(device, queue, out_y, y_bytes_u32, tmp_y_u32);
    if (ok) ok = hap_wgpu_read_buffer(device, queue, out_cb, c_bytes_u32, tmp_cb_u32);
    if (ok) ok = hap_wgpu_read_buffer(device, queue, out_cr, c_bytes_u32, tmp_cr_u32);
    if (ok) {
        for (size_t i = 0; i < px_count; ++i) Y[i] = (uint8_t)tmp_y_u32[i];
        for (size_t i = 0; i < c_count; ++i) Cb[i] = (uint8_t)tmp_cb_u32[i];
        for (size_t i = 0; i < c_count; ++i) Cr[i] = (uint8_t)tmp_cr_u32[i];
    }

    free(tmp_y_u32);
    free(tmp_cb_u32);
    free(tmp_cr_u32);
    wgpuBindGroupRelease(bg);
    wgpuBindGroupLayoutRelease(bgl);
    wgpuComputePipelineRelease(pipeline);
    wgpuShaderModuleRelease(module);
    wgpuBufferRelease(in_rgba);
    wgpuBufferRelease(out_y);
    wgpuBufferRelease(out_cb);
    wgpuBufferRelease(out_cr);
    wgpuBufferRelease(params);
    return ok;
}

static int hap_wgpu_run_compute_ycbcr_to_rgba(const uint8_t *Y,
                                              const uint8_t *Cb,
                                              const uint8_t *Cr,
                                              unsigned int width,
                                              unsigned int height,
                                              unsigned int cb_w,
                                              unsigned int cb_h,
                                              uint32_t *rgba_out_u32)
{
    WGPUDevice device = s_ctx.device;
    WGPUQueue queue = s_ctx.queue;
    if (!device || !queue) return 0;

    size_t px_count = (size_t)width * height;
    size_t c_count = (size_t)cb_w * cb_h;
    size_t y_bytes_u32 = px_count * sizeof(uint32_t);
    size_t c_bytes_u32 = c_count * sizeof(uint32_t);
    size_t rgba_bytes = px_count * sizeof(uint32_t);
    size_t params_bytes = sizeof(HapWGPUParams);

    uint32_t *Y_u32 = (uint32_t *)malloc(y_bytes_u32);
    uint32_t *Cb_u32 = (uint32_t *)malloc(c_bytes_u32);
    uint32_t *Cr_u32 = (uint32_t *)malloc(c_bytes_u32);
    if (!Y_u32 || !Cb_u32 || !Cr_u32) {
        free(Y_u32);
        free(Cb_u32);
        free(Cr_u32);
        return 0;
    }
    for (size_t i = 0; i < px_count; ++i) Y_u32[i] = Y[i];
    for (size_t i = 0; i < c_count; ++i) Cb_u32[i] = Cb[i];
    for (size_t i = 0; i < c_count; ++i) Cr_u32[i] = Cr[i];

    WGPUBuffer in_y = hap_wgpu_make_buffer(
        device, y_bytes_u32, WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);
    WGPUBuffer in_cb = hap_wgpu_make_buffer(
        device, c_bytes_u32, WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);
    WGPUBuffer in_cr = hap_wgpu_make_buffer(
        device, c_bytes_u32, WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);
    WGPUBuffer out_rgba = hap_wgpu_make_buffer(
        device, rgba_bytes, WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc);
    WGPUBuffer params = hap_wgpu_make_buffer(
        device, params_bytes, WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
    if (!in_y || !in_cb || !in_cr || !out_rgba || !params) {
        free(Y_u32); free(Cb_u32); free(Cr_u32);
        if (in_y) wgpuBufferRelease(in_y);
        if (in_cb) wgpuBufferRelease(in_cb);
        if (in_cr) wgpuBufferRelease(in_cr);
        if (out_rgba) wgpuBufferRelease(out_rgba);
        if (params) wgpuBufferRelease(params);
        return 0;
    }

    HapWGPUParams p;
    p.y_width = width;
    p.y_height = height;
    p.cb_width = cb_w;
    p.cb_height = cb_h;
    wgpuQueueWriteBuffer(queue, in_y, 0, Y_u32, y_bytes_u32);
    wgpuQueueWriteBuffer(queue, in_cb, 0, Cb_u32, c_bytes_u32);
    wgpuQueueWriteBuffer(queue, in_cr, 0, Cr_u32, c_bytes_u32);
    wgpuQueueWriteBuffer(queue, params, 0, &p, sizeof(p));

    free(Y_u32); free(Cb_u32); free(Cr_u32);

    WGPUShaderModuleWGSLDescriptor wgsl;
    memset(&wgsl, 0, sizeof(wgsl));
    wgsl.chain.sType = WGPUSType_ShaderModuleWGSLDescriptor;
    wgsl.code = kHapWGSL;
    WGPUShaderModuleDescriptor sm_desc;
    memset(&sm_desc, 0, sizeof(sm_desc));
    sm_desc.nextInChain = (const WGPUChainedStruct *)&wgsl;
    WGPUShaderModule module = wgpuDeviceCreateShaderModule(device, &sm_desc);
    if (!module) {
        wgpuBufferRelease(in_y); wgpuBufferRelease(in_cb); wgpuBufferRelease(in_cr);
        wgpuBufferRelease(out_rgba); wgpuBufferRelease(params);
        return 0;
    }

    WGPUComputePipelineDescriptor cp_desc;
    memset(&cp_desc, 0, sizeof(cp_desc));
    cp_desc.compute.module = module;
    cp_desc.compute.entryPoint = "ycbcr_to_rgba";
    WGPUComputePipeline pipeline = wgpuDeviceCreateComputePipeline(device, &cp_desc);
    if (!pipeline) {
        wgpuShaderModuleRelease(module);
        wgpuBufferRelease(in_y); wgpuBufferRelease(in_cb); wgpuBufferRelease(in_cr);
        wgpuBufferRelease(out_rgba); wgpuBufferRelease(params);
        return 0;
    }

    WGPUBindGroupLayout bgl = wgpuComputePipelineGetBindGroupLayout(pipeline, 0);
    WGPUBindGroupEntry entries[5];
    memset(entries, 0, sizeof(entries));
    entries[0].binding = 0; entries[0].buffer = in_y; entries[0].size = y_bytes_u32;
    entries[1].binding = 1; entries[1].buffer = in_cb; entries[1].size = c_bytes_u32;
    entries[2].binding = 2; entries[2].buffer = in_cr; entries[2].size = c_bytes_u32;
    entries[3].binding = 3; entries[3].buffer = out_rgba; entries[3].size = rgba_bytes;
    entries[4].binding = 4; entries[4].buffer = params; entries[4].size = params_bytes;
    WGPUBindGroupDescriptor bg_desc;
    memset(&bg_desc, 0, sizeof(bg_desc));
    bg_desc.layout = bgl;
    bg_desc.entryCount = 5;
    bg_desc.entries = entries;
    WGPUBindGroup bg = wgpuDeviceCreateBindGroup(device, &bg_desc);
    if (!bg) {
        wgpuBindGroupLayoutRelease(bgl);
        wgpuComputePipelineRelease(pipeline);
        wgpuShaderModuleRelease(module);
        wgpuBufferRelease(in_y); wgpuBufferRelease(in_cb); wgpuBufferRelease(in_cr);
        wgpuBufferRelease(out_rgba); wgpuBufferRelease(params);
        return 0;
    }

    WGPUCommandEncoderDescriptor enc_desc;
    memset(&enc_desc, 0, sizeof(enc_desc));
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, &enc_desc);
    WGPUComputePassDescriptor pass_desc;
    memset(&pass_desc, 0, sizeof(pass_desc));
    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, &pass_desc);
    wgpuComputePassEncoderSetPipeline(pass, pipeline);
    wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, NULL);
    wgpuComputePassEncoderDispatchWorkgroups(
        pass, (width + 7U) / 8U, (height + 7U) / 8U, 1);
    wgpuComputePassEncoderEnd(pass);
    wgpuComputePassEncoderRelease(pass);

    WGPUCommandBufferDescriptor cb_desc;
    memset(&cb_desc, 0, sizeof(cb_desc));
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(encoder, &cb_desc);
    wgpuCommandEncoderRelease(encoder);
    wgpuQueueSubmit(queue, 1, &cmd);
    wgpuCommandBufferRelease(cmd);

    int ok = hap_wgpu_read_buffer(device, queue, out_rgba, rgba_bytes, rgba_out_u32);
    wgpuBindGroupRelease(bg);
    wgpuBindGroupLayoutRelease(bgl);
    wgpuComputePipelineRelease(pipeline);
    wgpuShaderModuleRelease(module);
    wgpuBufferRelease(in_y); wgpuBufferRelease(in_cb); wgpuBufferRelease(in_cr);
    wgpuBufferRelease(out_rgba); wgpuBufferRelease(params);
    return ok;
}

int hap_wgpu_convert_rgb_to_ycbcr(
    const void *input_bc, unsigned long input_bytes,
    unsigned int input_format,
    unsigned int width, unsigned int height,
    unsigned int subsampling,
    void *output_y_bc, void *output_cb_bc, void *output_cr_bc,
    unsigned long output_y_bc_bytes,
    unsigned long output_cb_bc_bytes,
    unsigned long output_cr_bc_bytes)
{
    if (!hap_wgpu_available()) return 0;

    unsigned int cb_w, cb_h;
    hap_chroma_plane_dims(width, height, subsampling, &cb_w, &cb_h);
    size_t px_count = (size_t)width * height;
    size_t c_count = (size_t)cb_w * cb_h;

    uint8_t *rgba = (uint8_t *)malloc(px_count * 4);
    uint32_t *rgba_u32 = (uint32_t *)malloc(px_count * sizeof(uint32_t));
    uint8_t *Y = (uint8_t *)malloc(px_count);
    uint8_t *Cb = (uint8_t *)malloc(c_count);
    uint8_t *Cr = (uint8_t *)malloc(c_count);
    if (!rgba || !rgba_u32 || !Y || !Cb || !Cr) {
        free(rgba); free(rgba_u32); free(Y); free(Cb); free(Cr);
        return 0;
    }

    unsigned int r = HapResult_Bad_Arguments;
    if (input_format == HapTextureFormat_RGB_DXT1) {
        r = hap_cpu_decode_bc1(input_bc, input_bytes, width, height, rgba);
    } else if (input_format == HapTextureFormat_RGBA_DXT5) {
        r = hap_cpu_decode_bc3(input_bc, input_bytes, width, height, rgba);
    }
    if (r != HapResult_No_Error) {
        free(rgba); free(rgba_u32); free(Y); free(Cb); free(Cr);
        return 0;
    }

    for (size_t i = 0; i < px_count; ++i) {
        rgba_u32[i] = ((uint32_t)rgba[i * 4 + 0]) |
                      ((uint32_t)rgba[i * 4 + 1] << 8) |
                      ((uint32_t)rgba[i * 4 + 2] << 16) |
                      ((uint32_t)rgba[i * 4 + 3] << 24);
    }

    int ok = hap_wgpu_run_compute_rgba_to_ycbcr(
        rgba_u32, width, height, cb_w, cb_h, subsampling, Y, Cb, Cr);
    if (ok) {
        ok = (hap_cpu_encode_bc4(Y, width, height, output_y_bc, output_y_bc_bytes) == HapResult_No_Error) &&
             (hap_cpu_encode_bc4(Cb, cb_w, cb_h, output_cb_bc, output_cb_bc_bytes) == HapResult_No_Error) &&
             (hap_cpu_encode_bc4(Cr, cb_w, cb_h, output_cr_bc, output_cr_bc_bytes) == HapResult_No_Error);
    }

    free(rgba);
    free(rgba_u32);
    free(Y);
    free(Cb);
    free(Cr);
    return ok;
}

int hap_wgpu_convert_ycbcr_to_rgb(
    const void *input_y_bc, unsigned long input_y_bc_bytes,
    const void *input_cb_bc, unsigned long input_cb_bc_bytes,
    const void *input_cr_bc, unsigned long input_cr_bc_bytes,
    unsigned int subsampling,
    unsigned int width, unsigned int height,
    unsigned int output_format,
    void *output_bc, unsigned long output_bc_bytes)
{
    if (!hap_wgpu_available()) return 0;

    unsigned int cb_w, cb_h;
    hap_chroma_plane_dims(width, height, subsampling, &cb_w, &cb_h);
    size_t px_count = (size_t)width * height;
    size_t c_count = (size_t)cb_w * cb_h;

    uint8_t *Y = (uint8_t *)malloc(px_count);
    uint8_t *Cb = (uint8_t *)malloc(c_count);
    uint8_t *Cr = (uint8_t *)malloc(c_count);
    uint32_t *rgba_u32 = (uint32_t *)malloc(px_count * sizeof(uint32_t));
    uint8_t *rgba = (uint8_t *)malloc(px_count * 4);
    if (!Y || !Cb || !Cr || !rgba_u32 || !rgba) {
        free(Y); free(Cb); free(Cr); free(rgba_u32); free(rgba);
        return 0;
    }

    int ok = 1;
    ok = ok && (hap_cpu_decode_bc4(input_y_bc, input_y_bc_bytes, width, height, Y) == HapResult_No_Error);
    ok = ok && (hap_cpu_decode_bc4(input_cb_bc, input_cb_bc_bytes, cb_w, cb_h, Cb) == HapResult_No_Error);
    ok = ok && (hap_cpu_decode_bc4(input_cr_bc, input_cr_bc_bytes, cb_w, cb_h, Cr) == HapResult_No_Error);

    if (ok) {
        ok = hap_wgpu_run_compute_ycbcr_to_rgba(
            Y, Cb, Cr, width, height, cb_w, cb_h, rgba_u32);
    }

    if (ok) {
        for (size_t i = 0; i < px_count; ++i) {
            uint32_t v = rgba_u32[i];
            rgba[i * 4 + 0] = (uint8_t)(v & 0xFF);
            rgba[i * 4 + 1] = (uint8_t)((v >> 8) & 0xFF);
            rgba[i * 4 + 2] = (uint8_t)((v >> 16) & 0xFF);
            rgba[i * 4 + 3] = (uint8_t)((v >> 24) & 0xFF);
        }
        if (output_format == HapTextureFormat_RGB_DXT1) {
            ok = (hap_cpu_encode_bc1(rgba, width, height, output_bc, output_bc_bytes) == HapResult_No_Error);
        } else if (output_format == HapTextureFormat_RGBA_DXT5) {
            ok = (hap_cpu_encode_bc3(rgba, width, height, output_bc, output_bc_bytes) == HapResult_No_Error);
        } else {
            ok = 0;
        }
    }

    free(Y);
    free(Cb);
    free(Cr);
    free(rgba_u32);
    free(rgba);
    return ok;
}

#else

int hap_wgpu_available(void) { return 0; }

int hap_wgpu_convert_rgb_to_ycbcr(
    const void *input_bc, unsigned long input_bytes,
    unsigned int input_format,
    unsigned int width, unsigned int height,
    unsigned int subsampling,
    void *output_y_bc, void *output_cb_bc, void *output_cr_bc,
    unsigned long output_y_bc_bytes,
    unsigned long output_cb_bc_bytes,
    unsigned long output_cr_bc_bytes)
{
    (void)input_bc; (void)input_bytes; (void)input_format;
    (void)width; (void)height; (void)subsampling;
    (void)output_y_bc; (void)output_cb_bc; (void)output_cr_bc;
    (void)output_y_bc_bytes; (void)output_cb_bc_bytes; (void)output_cr_bc_bytes;
    return 0;
}

int hap_wgpu_convert_ycbcr_to_rgb(
    const void *input_y_bc, unsigned long input_y_bc_bytes,
    const void *input_cb_bc, unsigned long input_cb_bc_bytes,
    const void *input_cr_bc, unsigned long input_cr_bc_bytes,
    unsigned int subsampling,
    unsigned int width, unsigned int height,
    unsigned int output_format,
    void *output_bc, unsigned long output_bc_bytes)
{
    (void)input_y_bc; (void)input_y_bc_bytes;
    (void)input_cb_bc; (void)input_cb_bc_bytes;
    (void)input_cr_bc; (void)input_cr_bc_bytes;
    (void)subsampling; (void)width; (void)height;
    (void)output_format; (void)output_bc; (void)output_bc_bytes;
    return 0;
}

#endif
