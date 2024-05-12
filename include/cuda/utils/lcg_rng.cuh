#pragma once

#include <cuda.h>
#include "glm_pch.h"

namespace ntwr {
// https://github.com/ospray/ospray/blob/master/ospray/math/random.ih
struct LCGRand {
    uint32_t state;
};

NTWR_DEVICE uint32_t murmur_hash3_mix(uint32_t hash, uint32_t k)
{
    const uint32_t c1 = 0xcc9e2d51;
    const uint32_t c2 = 0x1b873593;
    const uint32_t r1 = 15;
    const uint32_t r2 = 13;
    const uint32_t m  = 5;
    const uint32_t n  = 0xe6546b64;

    k *= c1;
    k = (k << r1) | (k >> (32 - r1));
    k *= c2;

    hash ^= k;
    hash = ((hash << r2) | (hash >> (32 - r2))) * m + n;

    return hash;
}

NTWR_DEVICE uint32_t murmur_hash3_finalize(uint32_t hash)
{
    hash ^= hash >> 16;
    hash *= 0x85ebca6b;
    hash ^= hash >> 13;
    hash *= 0xc2b2ae35;
    hash ^= hash >> 16;

    return hash;
}

NTWR_DEVICE uint32_t lcg_random(LCGRand &rng)
{
    const uint32_t m = 1664525;
    const uint32_t n = 1013904223;
    rng.state        = rng.state * m + n;
    return rng.state;
}

NTWR_DEVICE float lcg_randomf(LCGRand &rng)
{
    return ldexp((float)lcg_random(rng), -32);
}

NTWR_DEVICE glm::vec2 lcg_random2f(LCGRand &rng)
{
    return glm::vec2(lcg_randomf(rng), lcg_randomf(rng));
}

NTWR_DEVICE glm::vec3 lcg_random3f(LCGRand &rng)
{
    return glm::vec3(lcg_randomf(rng), lcg_randomf(rng), lcg_randomf(rng));
}

NTWR_DEVICE LCGRand get_lcg_rng(uint32_t index, uint32_t frame, uint32_t linear)
{
    LCGRand rng;
    rng.state = murmur_hash3_mix(0, linear);
    rng.state = murmur_hash3_mix(rng.state, 512 * frame + index);
    rng.state = murmur_hash3_finalize(rng.state);
    return rng;
}

NTWR_DEVICE glm::vec3 lcg_deterministic_vec3(uint32_t seed)
{
    LCGRand rng;
    rng.state = seed;
    return glm::vec3(lcg_randomf(rng), lcg_randomf(rng), lcg_randomf(rng));
}

}