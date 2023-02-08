/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

 // Simple 3D volume renderer

#ifndef _VOLUMERENDER_KERNEL_CU_
#define _VOLUMERENDER_KERNEL_CU_


#include <helper_cuda.h>
#include <helper_math.h>
#include "define.h"

typedef unsigned int uint;
typedef unsigned char uchar;

#define EPSILON 1.0e-6f
#define dotFloat4(a, b)	((a.x)*(b.x) + (a.y)*(b.y) + (a.z)*(b.z))

void setTransferFunc(float4* f, int n) {
    float x, y, z, w;

    x = 1.0f;
    y = 0.0f;
    z = 0.0f;
    w = 0.0f;

    for (int i = 0; i < n / 6; i++) {
        f[i].x = x;
        f[i].y = y;
        f[i].z = z;
        f[i].w = w;
        
        w += 1 / (float)n;
        y += (3 / (float)n);
    }

    for (int i = n / 6; i < n / 3; i++) {
        f[i].x = x;
        f[i].y = y;
        f[i].z = z;
        f[i].w = w;

        w += 1 / (float)n;
        y += (3 / (float)n);
    }

    for (int i = n / 3; i < n / 2; i++) {
        f[i].x = x;
        f[i].y = y;
        f[i].z = z;
        f[i].w = w;

        w += 1 / (float)n;
        x -= (6 / (float)n);
    }

    for (int i = n / 2; i < (2 * n) / 3; i++) {
        f[i].x = x;
        f[i].y = y;
        f[i].z = z;
        f[i].w = w;

        w += 1 / (float)n;
        z += (6 / (float)n);
    }

    for (int i = (2 * n) / 3; i < (5 * n) / 6; i++) {
        f[i].x = x;
        f[i].y = y;
        f[i].z = z;
        f[i].w = w;

        w += 1 / (float)n;
        y -= (6 / (float)n);
    }

    for (int i = (5 * n) / 6; i < n; i++) {
        f[i].x = x;
        f[i].y = y;
        f[i].z = z;
        f[i].w = w;

        w += 1 / (float)n;
        x += (6 / (float)n);
    }

}

cudaArray* d_volumeArray = 0;
cudaArray* d_normalArray = 0;
cudaArray* d_transferFuncArray;

typedef unsigned char VolumeType;
// typedef unsigned short VolumeType;

cudaTextureObject_t texObject;    // For 3D texture
cudaTextureObject_t texObjectN;    // For 3D texture
cudaTextureObject_t transferTex;  // For 1D transfer function texture

typedef struct { float4 m[3]; } float3x4;

__constant__ float3x4 c_invViewMatrix;  // inverse view matrix

struct Ray {
    float3 o;  // origin
    float3 d;  // direction
};

// intersect ray with a box
// http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter3.htm

__device__ int intersectBox(Ray r, float3 boxmin, float3 boxmax, float* tnear,
    float* tfar) {
    // compute intersection of ray with all six bbox planes
    float3 invR = make_float3(1.0f) / r.d;
    float3 tbot = invR * (boxmin - r.o);
    float3 ttop = invR * (boxmax - r.o);

    // re-order intersections to find smallest and largest on each axis
    float3 tmin = fminf(ttop, tbot);
    float3 tmax = fmaxf(ttop, tbot);

    // find the largest tmin and the smallest tmax
    float largest_tmin = fmaxf(fmaxf(tmin.x, tmin.y), fmaxf(tmin.x, tmin.z));
    float smallest_tmax = fminf(fminf(tmax.x, tmax.y), fminf(tmax.x, tmax.z));

    *tnear = largest_tmin;
    *tfar = smallest_tmax;

    return smallest_tmax > largest_tmin;
}

__device__ float4 float3to4(float3 f3, int alpha) {
    float4 f4;

    f4.x = f3.x;
    f4.y = f3.y;
    f4.z = f3.z;
    f4.z = alpha;
}
// transform vector by matrix (no translation)
__device__ float3 mul(const float3x4& M, const float3& v) {
    float3 r;
    r.x = dot(v, make_float3(M.m[0]));
    r.y = dot(v, make_float3(M.m[1]));
    r.z = dot(v, make_float3(M.m[2]));
    return r;
}

// transform vector by matrix with translation
__device__ float4 mul(const float3x4& M, const float4& v) {
    float4 r;
    r.x = dot(v, M.m[0]);
    r.y = dot(v, M.m[1]);
    r.z = dot(v, M.m[2]);
    r.w = 1.0f;
    return r;
}

__device__ float4 constMulFloat4(float c, float4 f4) {
    f4.x *= c;
    f4.y *= c;
    f4.z *= c;

    return f4;
}

__device__ float4 addFloat4(float4 dst, float4 src) {
    dst.x += src.x;
    dst.y += src.y;
    dst.z += src.z;

    return dst;
}

__device__ void subFloat4(float4* dst, float4 src) {
    dst->x -= src.x;
    dst->y -= src.y;
    dst->z -= src.z;
}
__device__ void normalizeFloat4(float4* f) {
    float length
        = sqrtf(f->x * f->x + f->y * f->y + f->z * f->z);

    f->x /= length, f->y /= length, f->z /= length;
}

__device__ void crossFloat4(float4* res, float4 a, float4 b) {
    res->x = a.y * b.z - b.y * a.z;
    res->y = -(a.x * b.z - b.x * a.z);
    res->z = a.x * b.y - b.x * a.y;
}
__device__ uint rgbaFloatToInt(float4 rgba) {
    rgba.x = __saturatef(rgba.x);  // clamp to [0.0, 1.0]
    rgba.y = __saturatef(rgba.y);
    rgba.z = __saturatef(rgba.z);
    rgba.w = __saturatef(rgba.w);
    return (uint(rgba.w * 255) << 24) | (uint(rgba.z * 255) << 16) |
        (uint(rgba.y * 255) << 8) | uint(rgba.x * 255);
}
__global__ void d_calculate_normal(float4* d_output, cudaTextureObject_t tex, uint w, uint h, uint d) {
    uint ix = blockIdx.x * blockDim.x + threadIdx.x;
    uint iy = blockIdx.y * blockDim.y + threadIdx.y;
    uint iz = blockIdx.z * blockDim.z + threadIdx.z;
    float u, v, r;
    if ((ix >= w) || (iy >= h) || (iz >= d)) return;

    uint idx = ix + iy * w + iz * w * h;

    {

        u = ix / (float)w;
        v = iy / (float)h;
        r = iz / (float)d;

        float4 tmp;
        tmp.w = 1;

        {   //diffX
            if (!ix) {
                tmp.x = (tex3D<float>(tex, 0, v, r) - tex3D<float>(tex, 1 / (float)w, v, r)) / (2 * (1 / (float)w));
            }
            else {
                tmp.x = (tex3D<float>(tex, (ix - 1) / (float)w, v, r) - tex3D<float>(tex, (ix + 1) / (float)w, v, r)) / (2 * (1 / (float)w));
            }
        }
        {   //diffY
            if (!iy) {
                tmp.y = (tex3D<float>(tex, u, 0, r) - tex3D<float>(tex, u, 1 / (float)h, r)) / (2 * (1 / (float)h));
            }
            else {
                tmp.y = (tex3D<float>(tex, u, (iy - 1) / (float)h, r) - tex3D<float>(tex, u, (iy + 1) / (float)h, r)) / (2 * (1 / (float)h));
            }
        }
        {   //diffZ
            if (!iz) {
                tmp.z = (tex3D<float>(tex, u, v, 0) - tex3D<float>(tex, u, v, 1 / (float)d)) / (2 * (1 / (float)d));
            }
            else {
                tmp.z = (tex3D<float>(tex, u, v, (iz - 1) / (float)d) - tex3D<float>(tex, u, v, (iz + 1) / (float)d)) / (2 * (1 / (float)d));
            }
        }

        normalizeFloat4(&tmp);
        d_output[idx] = tmp;
    }
}

__device__ float4 shadePhong(const float4 cV, const float4 cN, float4 lightPos, const float4 cDiffuseColor) {
    float4* L;
    float4 V = cV;
    normalizeFloat4(&lightPos);
    normalizeFloat4(&V);
    L = &lightPos;   //light direction for DIRECTIONAL LIGHT
    
    float tmp_flt = dotFloat4((*L), cN);
    float4 finalColor = { 0.0f, 0.0f, 0.0f, 1.0f };

    if (tmp_flt > 0.0f) { // front light
        float kd = 0.5;
        finalColor = addFloat4(finalColor, constMulFloat4(tmp_flt * kd, cDiffuseColor));    // Diffuse Reflection
        //printf("finalColor: %f %f %f %f\n", finalColor.x, finalColor.y, finalColor.z, finalColor.w);
        float4 H;

        H = addFloat4(*L, V);
        normalizeFloat4(&H);

        tmp_flt = dotFloat4(cN, H);
        int spec_exp = 1;   
        float ks = 0.5;
        if (tmp_flt > 0.0f) {
            finalColor += constMulFloat4(powf(tmp_flt, spec_exp) * ks, cDiffuseColor);    // Diffuse Reflection
        }
    }

    return finalColor;
}
__global__ void d_render(uint* d_output, uint imageW, uint imageH,
    float density, float brightness, float transferOffset,
    float transferScale, cudaTextureObject_t tex,
    cudaTextureObject_t transferTex, float4* d_normal, cudaTextureObject_t texN) {
    const int maxSteps = 500;
    const float tstep = 0.01f;
    const float opacityThreshold = 0.95f;
    const float3 boxMin = make_float3(-1.0f, -1.0f, -1.0f);
    const float3 boxMax = make_float3(1.0f, 1.0f, 1.0f);

    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;

    if ((x >= imageW) || (y >= imageH)) return;

    float u = (x / (float)imageW) * 2.0f - 1.0f;
    float v = (y / (float)imageH) * 2.0f - 1.0f;

    // calculate eye ray in world space
    Ray eyeRay;
    eyeRay.o =
        make_float3(mul(c_invViewMatrix, make_float4(0.0f, 0.0f, 0.0f, 1.0f)));
    eyeRay.d = normalize(make_float3(u, v, -2.0f));
    eyeRay.d = mul(c_invViewMatrix, eyeRay.d);

    // find intersection with box
    float tnear, tfar;
    int hit = intersectBox(eyeRay, boxMin, boxMax, &tnear, &tfar);

    if (!hit) return;

    if (tnear < 0.0f) tnear = 0.0f;  // clamp to near plane

    // march along ray from front to back, accumulating color
    float4 sum = make_float4(0.0f);
    float t = tnear;
    float3 pos = eyeRay.o + eyeRay.d * tnear;
    float3 step = eyeRay.d * tstep;
   
    for (int i = 0; i < maxSteps; i++) {

        // read from 3D texture
        // remap position to [0, 1] coordinates

        float sample = tex3D<float>(tex, pos.x * 0.5f + 0.5f, pos.y * 0.5f + 0.5f,
            pos.z * 0.5f + 0.5f);

    // lookup in transfer function texture
        float4 col;

#if SHADING
        float4 tmp = tex3D<float4>(texN, pos.x * 0.5f + 0.5f, pos.y * 0.5f + 0.5f,
            pos.z * 0.5f + 0.5f);

   
        float4 lightDir = mul(c_invViewMatrix, make_float4(1000.0f, -1000.0f, -1000.0f, 1.0f));

        if (fabs(sample) <= 0.0001f) {
            col = { 111.0 / 255.0, 79.0 / 255.0, 40.0 / 255.0, 1.0f, };
            col = shadePhong(-float3to4(eyeRay.d, 1.0f), tmp, lightDir, col);
            col = constMulFloat4(brightness, col);
            d_output[y * imageW + x] = rgbaFloatToInt(col);
            return;
        }
        //else {
        //    col =
        //        tex1D<float4>(transferTex, (sample - transferOffset) * transferScale);
        //    pos += step;
        //    continue;
        //    //col.w *= density;
        //}
#endif
        col =
            tex1D<float4>(transferTex, (sample - transferOffset) * transferScale);
        // "under" operator for back-to-front blending
        // sum = lerp(sum, col, col.w);
        if (density > col.w + 0.001f) {

            // pre-multiply alpha
            col.x *= col.w;
            col.y *= col.w;
            col.z *= col.w;
            // "over" operator for front-to-back blending
            sum = sum + col * (1.0f - sum.w);
        }
        // exit early if opaque
        if (sum.w > opacityThreshold) break;

        t += tstep;

        if (t > tfar) break;

        pos += step;
    }

    sum *= brightness;

    // write output color
    d_output[y * imageW + x] = rgbaFloatToInt(sum);
}

extern "C" void setTextureFilterMode(bool bLinearFilter) {
    if (texObject) {
        checkCudaErrors(cudaDestroyTextureObject(texObject));
    }

    cudaResourceDesc texRes;
    memset(&texRes, 0, sizeof(cudaResourceDesc));

    texRes.resType = cudaResourceTypeArray;
    texRes.res.array.array = d_volumeArray;

    cudaTextureDesc texDescr;
    memset(&texDescr, 0, sizeof(cudaTextureDesc));

    texDescr.normalizedCoords = true;
    texDescr.filterMode =
        bLinearFilter ? cudaFilterModeLinear : cudaFilterModePoint;

    texDescr.addressMode[0] = cudaAddressModeWrap;
    texDescr.addressMode[1] = cudaAddressModeWrap;
    texDescr.addressMode[2] = cudaAddressModeWrap;

    texDescr.readMode = cudaReadModeNormalizedFloat;

    checkCudaErrors(
        cudaCreateTextureObject(&texObject, &texRes, &texDescr, NULL));

}

extern "C" void initNormal(void* d_normal, cudaExtent volumeSize) {

    cudaChannelFormatDesc channelDescN = cudaCreateChannelDesc<float4>();
    checkCudaErrors(cudaMalloc3DArray(&d_normalArray, &channelDescN, volumeSize));

    cudaMemcpy3DParms copyParamsN = { 0 };

    copyParamsN.srcPtr =
        make_cudaPitchedPtr(d_normal, volumeSize.width * sizeof(float4),
            volumeSize.width, volumeSize.height);


    copyParamsN.dstArray = d_normalArray;
    copyParamsN.extent = volumeSize;
    copyParamsN.kind = cudaMemcpyDeviceToDevice;
    checkCudaErrors(cudaMemcpy3D(&copyParamsN));

    cudaResourceDesc texRes;
    memset(&texRes, 0, sizeof(cudaResourceDesc));

    texRes.resType = cudaResourceTypeArray;
    texRes.res.array.array = d_normalArray;

    cudaTextureDesc texDescr;
    memset(&texDescr, 0, sizeof(cudaTextureDesc));

    texDescr.normalizedCoords =
        true;  // access with normalized texture coordinates
    texDescr.filterMode = cudaFilterModeLinear;  // linear interpolation

    texDescr.addressMode[0] = cudaAddressModeClamp;  // clamp texture coordinates
    texDescr.addressMode[1] = cudaAddressModeClamp;
    texDescr.addressMode[2] = cudaAddressModeClamp;


    texDescr.readMode = cudaReadModeElementType;


    checkCudaErrors(
        cudaCreateTextureObject(&texObjectN, &texRes, &texDescr, NULL));
}
extern "C" void initCuda(void* h_volume, cudaExtent volumeSize) {
    // create 3D array
#if (FILE_FORMAT == FORMAT_PHI)
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    checkCudaErrors(cudaMalloc3DArray(&d_volumeArray, &channelDesc, volumeSize));

#elif (FILE_FORMAT == FORMAT_RAW)
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<VolumeType>();
    checkCudaErrors(cudaMalloc3DArray(&d_volumeArray, &channelDesc, volumeSize));
#else
    printf("Check Format\n");
    exit(1);
#endif
    // copy data to 3D array
    cudaMemcpy3DParms copyParams = { 0 };
#if (FILE_FORMAT == FORMAT_PHI)
    copyParams.srcPtr =
        make_cudaPitchedPtr(h_volume, volumeSize.width * sizeof(float),
            volumeSize.width, volumeSize.height);
#elif (FILE_FORMAT == FORMAT_RAW)
    copyParams.srcPtr =
        make_cudaPitchedPtr(h_volume, volumeSize.width * sizeof(VolumeType),
            volumeSize.width, volumeSize.height);
#else
    printf("Check Format\n");
    exit(1);
#endif

    copyParams.dstArray = d_volumeArray;
    copyParams.extent = volumeSize;
    copyParams.kind = cudaMemcpyHostToDevice;

    checkCudaErrors(cudaMemcpy3D(&copyParams));

    cudaResourceDesc texRes;
    memset(&texRes, 0, sizeof(cudaResourceDesc));

    texRes.resType = cudaResourceTypeArray;
    texRes.res.array.array = d_volumeArray;

    cudaTextureDesc texDescr;
    memset(&texDescr, 0, sizeof(cudaTextureDesc));

    texDescr.normalizedCoords =
        true;  // access with normalized texture coordinates
    texDescr.filterMode = cudaFilterModeLinear;  // linear interpolation

    texDescr.addressMode[0] = cudaAddressModeClamp;  // clamp texture coordinates
    texDescr.addressMode[1] = cudaAddressModeClamp;
    texDescr.addressMode[2] = cudaAddressModeClamp;

#if (FILE_FORMAT == FORMAT_PHI)
    texDescr.readMode = cudaReadModeElementType;
#elif (FILE_FORMAT == FORMAT_RAW)
    texDescr.readMode = cudaReadModeNormalizedFloat;
#else
    printf("Check Format\n");
    exit(1);
#endif

    checkCudaErrors(
        cudaCreateTextureObject(&texObject, &texRes, &texDescr, NULL));

    float4* transferFunc;
    transferFunc = (float4*)malloc(sizeof(float4) * SAMPLING_SIZE);

    setTransferFunc(transferFunc, SAMPLING_SIZE);
    size_t tSize = SAMPLING_SIZE * sizeof(float4);

    cudaChannelFormatDesc channelDesc2 = cudaCreateChannelDesc<float4>();
    cudaArray* d_transferFuncArray;
    //printf("%d\n\n", cudaDeviceSynchronize());

    checkCudaErrors(cudaMallocArray(&d_transferFuncArray, &channelDesc2,
        tSize / sizeof(float4), 1));

    checkCudaErrors(cudaMemcpy2DToArray(d_transferFuncArray, 0, 0, transferFunc,
        0, tSize, 1,
        cudaMemcpyHostToDevice));

    memset(&texRes, 0, sizeof(cudaResourceDesc));

    texRes.resType = cudaResourceTypeArray;
    texRes.res.array.array = d_transferFuncArray;

    memset(&texDescr, 0, sizeof(cudaTextureDesc));

    texDescr.normalizedCoords =
        true;  // access with normalized texture coordinates
    texDescr.filterMode = cudaFilterModeLinear;

    texDescr.addressMode[0] = cudaAddressModeClamp;  // wrap texture coordinates

    texDescr.readMode = cudaReadModeElementType;

    checkCudaErrors(
        cudaCreateTextureObject(&transferTex, &texRes, &texDescr, NULL));
}

extern "C" void freeCudaBuffers() {
    checkCudaErrors(cudaDestroyTextureObject(texObject));
    checkCudaErrors(cudaDestroyTextureObject(texObjectN));
    checkCudaErrors(cudaDestroyTextureObject(transferTex));
    checkCudaErrors(cudaFreeArray(d_volumeArray));
    checkCudaErrors(cudaFreeArray(d_normalArray));
    checkCudaErrors(cudaFreeArray(d_transferFuncArray));
}

extern "C" void render_kernel(dim3 gridSize, dim3 blockSize, uint * d_output,
    uint imageW, uint imageH, float density,
    float brightness, float transferOffset,
    float transferScale, float4* d_normal) {
    d_render << <gridSize, blockSize >> > (d_output, imageW, imageH, density,
        brightness, transferOffset, transferScale,
        texObject, transferTex, d_normal, texObjectN);
}

extern "C" void calculate_normal(dim3 gridSizeN, dim3 blockSizeN, float4 * d_normal,
    uint imageW, uint imageH, uint imageD) {
    printf("calculate_normal grid (%d %d %d), block (%d %d %d)\n", gridSizeN.x, gridSizeN.y, gridSizeN.z, blockSizeN.x, blockSizeN.y, blockSizeN.z);
    d_calculate_normal << <gridSizeN, blockSizeN >> > (d_normal, texObject, imageW, imageH, imageD);
}

extern "C" void copyInvViewMatrix(float* invViewMatrix, size_t sizeofMatrix) {
    checkCudaErrors(
        cudaMemcpyToSymbol(c_invViewMatrix, invViewMatrix, sizeofMatrix));
}

#endif  // #ifndef _VOLUMERENDER_KERNEL_CU_
