#pragma once

#include "cuda/base/ray.cuh"
#include "cuda/utils/lcg_rng.cuh"
#include "cuda/utils/types.cuh"

namespace ntwr {

struct CudaBVHTraversalData {
    vec3_soa ray_o;
    vec3_soa ray_d;
    uint8_t *ray_depth = nullptr;
    // Pixel coordinate (lower 31 bits) and pixel (or path) still active (highest 1 bit)
    uint32_t *pixel_coord_active = nullptr;
    glm::vec3 *illumination      = nullptr;
    glm::vec4 *throughput_rng    = nullptr;

    NTWR_DEVICE Ray3f get_ray(uint32_t idx) const
    {
        return Ray3f(ray_o.get(idx), ray_d.get(idx));
    }

    NTWR_DEVICE void set_ray(uint32_t idx, Ray3f ray)
    {
        ray_o.set(idx, ray.o);
        ray_d.set(idx, ray.d);
    }

    NTWR_DEVICE void set_path_inactive(uint32_t index) const
    {
        // Set highest bit to 0
        pixel_coord_active[index] &= 0x7FFFFFFFu;
    }

    NTWR_DEVICE void set_path_active(uint32_t index) const
    {
        // Set highest bit to 1
        pixel_coord_active[index] |= 0x80000000u;
    }

    NTWR_DEVICE bool path_active(uint32_t index) const
    {
        return (pixel_coord_active[index] & 0x80000000u);
    }

    // Set the pixel coord and sets the path to be active
    NTWR_DEVICE void init_pixel_coord(uint32_t index, uint32_t pixel_coord) const
    {
        assert(pixel_coord <= 0x7FFFFFFFu);
        pixel_coord_active[index] = 0;
        pixel_coord_active[index] |= 0x7FFFFFFFu & pixel_coord;
        pixel_coord_active[index] |= 0x80000000u;
    }

    NTWR_DEVICE uint32_t pixel_coord(uint32_t index) const
    {
        return (0x7FFFFFFFu & pixel_coord_active[index]);
    }

    NTWR_DEVICE glm::vec3 &get_throughput(uint32_t idx)
    {
        return *(glm::vec3 *)&(throughput_rng[idx].x);
    }

    NTWR_DEVICE LCGRand get_rng(uint32_t idx)
    {
        return LCGRand{glm::floatBitsToUint(throughput_rng[idx].w)};
    }

    NTWR_DEVICE void set_rng(uint32_t idx, LCGRand new_rng)
    {
        throughput_rng[idx].w = glm::uintBitsToFloat(new_rng.state);
    }
};

}