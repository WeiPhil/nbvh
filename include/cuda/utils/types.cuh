#pragma once

#include "cuda.h"
#include "glm_pch.h"

namespace ntwr {

// vec3 buffer in SoA layout
struct vec3_soa {
    vec3_soa() {}
    vec3_soa(float *x, float *y, float *z) : x(x), y(y), z(z) {}
    float *x = nullptr;
    float *y = nullptr;
    float *z = nullptr;

    __device__ void set(uint32_t index, glm::vec3 vector)
    {
        x[index] = vector.x;
        y[index] = vector.y;
        z[index] = vector.z;
    }

    __device__ glm::vec3 get(uint32_t index) const
    {
        return glm::vec3(x[index], y[index], z[index]);
    }
};

// vec4 buffer in SoA layout
struct vec4_soa {
    vec4_soa() {}
    vec4_soa(float *x, float *y, float *z, float *w) : x(x), y(y), z(z), w(w) {}
    float *x = nullptr;
    float *y = nullptr;
    float *z = nullptr;
    float *w = nullptr;

    __device__ void set(uint32_t index, glm::vec4 vector)
    {
        x[index] = vector.x;
        y[index] = vector.y;
        z[index] = vector.z;
        w[index] = vector.w;
    }

    __device__ glm::vec4 get(uint32_t index) const
    {
        return glm::vec4(x[index], y[index], z[index], w[index]);
    }
};

float2 __host__ __device__ __forceinline__ vec2_to_float2(glm::vec2 value)
{
    return make_float2(value.x, value.y);
}

float3 __host__ __device__ __forceinline__ vec3_to_float3(glm::vec3 value)
{
    return make_float3(value.x, value.y, value.z);
}

float4 __host__ __device__ __forceinline__ vec4_to_float4(glm::vec4 value)
{
    return make_float4(value.x, value.y, value.z, value.w);
}

glm::vec2 __host__ __device__ __forceinline__ float2_to_vec2(float2 value)
{
    return glm::vec2(value.x, value.y);
}

glm::vec3 __host__ __device__ __forceinline__ float3_to_vec3(float3 value)
{
    return glm::vec3(value.x, value.y, value.z);
}

glm::vec4 __host__ __device__ __forceinline__ float4_to_vec4(float4 value)
{
    return glm::vec4(value.x, value.y, value.z, value.w);
}

}