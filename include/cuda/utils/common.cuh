#pragma once

#include <cuda_runtime.h>
#include "glm_pch.h"

#include "ntwr.h"

namespace ntwr {

static constexpr float Epsilon = 1e-3f;

static constexpr float E            = 2.71828182845904523536f;
static constexpr float PI           = 3.14159265358979323846f;
static constexpr float TWOPI        = 6.283185307179586231996f;
static constexpr float INV_PI       = 0.31830988618379067154f;
static constexpr float INV_TWOPI    = 0.15915494309189533577f;
static constexpr float INV_FOURPI   = 0.07957747154594766788f;
static constexpr float SQRT_TWO     = 1.41421356237309504880f;
static constexpr float INV_SQRT_TWO = 0.70710678118654752440f;

NTWR_HOST_DEVICE void ortho_basis(glm::vec3 &tangent, glm::vec3 &bitangent, const glm::vec3 &normal)
{
    bitangent = glm::vec3(0.f, 0.f, 0.f);

    if (normal.x < 0.6f && normal.x > -0.6f) {
        bitangent.x = 1.f;
    } else if (normal.y < 0.6f && normal.y > -0.6f) {
        bitangent.y = 1.f;
    } else if (normal.z < 0.6f && normal.z > -0.6f) {
        bitangent.z = 1.f;
    } else {
        bitangent.x = 1.f;
    }
    tangent   = normalize(cross(bitangent, normal));
    bitangent = normalize(cross(normal, tangent));
}

NTWR_HOST_DEVICE float component_max(glm::vec3 value)
{
    return glm::max(glm::max(value.x, value.y), value.z);
}

NTWR_HOST_DEVICE float component_min(glm::vec3 value)
{
    return glm::min(glm::min(value.x, value.y), value.z);
}

NTWR_HOST_DEVICE float sqr(float x)
{
    return x * x;
}

NTWR_HOST_DEVICE float safe_acos(float x)
{
    if (x < -1.0)
        x = -1.0;
    else if (x > 1.0)
        x = 1.0;
    return acos(x);
}

NTWR_HOST_DEVICE glm::vec3 sqr(glm::vec3 x)
{
    return glm::vec3(x.x * x.x, x.y * x.y, x.z * x.z);
}

NTWR_HOST_DEVICE float saturate(const float value)
{
    return glm::clamp(value, 0.f, 1.f);
}

NTWR_HOST_DEVICE float luminance(const glm::vec3 &c)
{
    return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}

NTWR_HOST_DEVICE bool all_zero(const float3 &v)
{
    return v.x == 0.f && v.y == 0.f && v.z == 0.f;
}

NTWR_HOST_DEVICE glm::vec3 reflect(const glm::vec3 &i, const glm::vec3 &n)
{
    return i - 2.f * n * dot(i, n);
}

NTWR_HOST_DEVICE glm::vec3 refract_ray(const glm::vec3 &i, const glm::vec3 &n, float eta)
{
    float n_dot_i = dot(n, i);
    float k       = 1.f - eta * eta * (1.f - n_dot_i * n_dot_i);
    if (k < 0.f) {
        return glm::vec3(0.f);
    }
    return eta * i - (eta * n_dot_i + sqrtf(k)) * n;
}

// Purely device functions
#ifdef __CUDACC__

NTWR_DEVICE static float atomicMax(float *address, float val)
{
    int *address_as_i = (int *)address;
    int old           = *address_as_i, assumed;
    do {
        assumed = old;
        old     = atomicCAS(address_as_i, assumed, __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);

    return __int_as_float(old);
}

template <typename T>
NTWR_DEVICE void swap(T &a, T &b)
{
    T tmp = a;
    a     = b;
    b     = tmp;
}

// Extracts the i-th most significant byte from x
NTWR_DEVICE uint32_t extract_byte(uint32_t x, uint32_t i)
{
    return (x >> (i * 8)) & 0xff;
}

// Most significant bit
NTWR_DEVICE uint32_t msb(uint32_t x)
{
    uint32_t result;
    asm volatile("bfind.u32 %0, %1; " : "=r"(result) : "r"(x));
    return result;
}

// Create byte mask from sign bit
NTWR_DEVICE uint32_t sign_extend_s8x4(uint32_t x)
{
    uint32_t result;
    asm("prmt.b32 %0, %1, 0x0, 0x0000BA98;" : "=r"(result) : "r"(x));
    return result;
}

// VMIN, VMAX functions, see "Understanding the Efficiency of Ray Traversal on GPUs – Kepler and Fermi Addendum" by
// Aila et al.

// Computes min(min(a, b), c)
NTWR_DEVICE float vmin_min(float a, float b, float c)
{
    int result;

    asm("vmin.s32.s32.s32.min %0, %1, %2, %3;"
        : "=r"(result)
        : "r"(__float_as_int(a)), "r"(__float_as_int(b)), "r"(__float_as_int(c)));

    return __int_as_float(result);
}

// Computes max(min(a, b), c)
NTWR_DEVICE float vmin_max(float a, float b, float c)
{
    int result;

    asm("vmin.s32.s32.s32.max %0, %1, %2, %3;"
        : "=r"(result)
        : "r"(__float_as_int(a)), "r"(__float_as_int(b)), "r"(__float_as_int(c)));

    return __int_as_float(result);
}

// Computes min(max(a, b), c)
NTWR_DEVICE float vmax_min(float a, float b, float c)
{
    int result;

    asm("vmax.s32.s32.s32.min %0, %1, %2, %3;"
        : "=r"(result)
        : "r"(__float_as_int(a)), "r"(__float_as_int(b)), "r"(__float_as_int(c)));

    return __int_as_float(result);
}

// Computes max(max(a, b), c)
// BROKEN WITH NEGATIVE VALUES !!!!
// NTWR_DEVICE float vmax_max(float a, float b, float c)
// {
//     int result;

//     asm("vmax.s32.s32.s32.max %0, %1, %2, %3;"
//         : "=r"(result)
//         : "r"(__float_as_int(a)), "r"(__float_as_int(b)), "r"(__float_as_int(c)));

//     return __int_as_float(result);
// }

#endif

}