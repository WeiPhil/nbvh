#pragma once

#include "cuda/utils/common.cuh"
#include "glm_pch.h"

namespace ntwr {

NTWR_HOST_DEVICE glm::vec3 square_to_uniform_sphere(const glm::vec2 &sample)
{
    float phi       = 2.0f * PI * sample.x;
    float cos_theta = sample.y * 2.0f - 1.0f;
    float sin_theta = sqrt(1.0f - cos_theta * cos_theta);
    return glm::vec3(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
}

NTWR_HOST_DEVICE float square_to_uniform_sphere_pdf()
{
    return INV_FOURPI;
}

NTWR_HOST_DEVICE float square_to_uniform_sphere_inv_pdf()
{
    return (4.f * PI);
}

NTWR_HOST_DEVICE glm::vec3 square_to_hg(const glm::vec2 rng, const glm::vec3 w_o, float g)
{
    float cos_theta;
    if (abs(g) < 1e-3) {
        cos_theta = 1.0 - 2.0 * rng.x;
    } else {
        float sqr_term = (1.0 - g * g) / (1.0 - g + 2.0 * g * rng.x);
        cos_theta      = (1.0 + g * g - sqr_term * sqr_term) / (2.0 * g);
    }

    float sin_theta = sqrt(max(0.f, 1.f - cos_theta * cos_theta));
    float phi       = 2.0 * PI * rng.y;

    float sin_phi = sin(phi);
    float cos_phi = cos(phi);

    glm::vec3 local = glm::vec3(sin_theta * cos_phi, sin_theta * sin_phi, cos_theta);

    glm::vec3 n = -w_o;

    glm::vec3 s;
    if (abs(n.x) > abs(n.y)) {
        float inv_len = 1.f / sqrt(n.x * n.x + n.z * n.z);
        s             = glm::vec3(n.z * inv_len, 0.0, -n.x * inv_len);
    } else {
        float inv_len = 1.f / sqrt(n.y * n.y + n.z * n.z);
        s             = glm::vec3(0.0, n.z * inv_len, -n.y * inv_len);
    }
    glm::vec3 t = cross(n, s);

    return local.x * s + local.y * t + local.z * n;
}

NTWR_HOST_DEVICE float square_to_hg_pdf(const float wo_dot_wi, const float g)
{
    float denom = 1.0 + g * g + 2.0 * g * wo_dot_wi;
    return 0.25 * INV_PI * (1.0 - g * g) / (denom * sqrt(denom));
}

NTWR_HOST_DEVICE float square_to_hg_inv_pdf(const float wo_dot_wi, const float g)
{
    float denom = 1.0 + g * g + 2.0 * g * wo_dot_wi;
    return (denom * sqrt(denom)) * 4 * PI / (1.0 - g * g);
}

// Cube face sampling
NTWR_DEVICE glm::vec3 get_cube_face_normal(uint32_t index)
{
    const glm::vec3 face_normals[6] = {
        glm::vec3(1.f, 0.f, 0.f),
        glm::vec3(-1.f, 0.f, 0.f),
        glm::vec3(0.f, 1.f, 0.f),
        glm::vec3(0.f, -1.f, 0.f),
        glm::vec3(0.f, 0.f, 1.f),
        glm::vec3(0.f, 0.f, -1.f),
    };
    return face_normals[index];
}

NTWR_DEVICE glm::vec3 get_cube_pos_on_face(uint32_t index, float u, float v)
{
    const glm::vec3 cube_positions[6] = {glm::vec3(1.0, u, v),
                                         glm::vec3(0.0, u, v),
                                         glm::vec3(u, 1.0, v),
                                         glm::vec3(u, 0.0, v),
                                         glm::vec3(u, v, 1.0),
                                         glm::vec3(u, v, 0.0)};
    return cube_positions[index];
}

NTWR_DEVICE uint32_t sample_cube_face_idx_from_uniform_dir(const glm::vec3 &dir_sample, const float sample)
{
    const uint32_t num_faces = 6;

    float pmf[num_faces];
    float summed_valid = 0.f;

    for (size_t i = 0; i < num_faces; i++) {
        float dot_product = dot(dir_sample, get_cube_face_normal(i));
        if (dot_product < 0.f)
            dot_product = 0.f;
        summed_valid += dot_product;
        pmf[i] = dot_product;
    }
    float cdf[num_faces + 1];
    float accum_sum = 0.f;

    // Normalize pmf and compute cdf
    for (size_t i = 0; i < num_faces; i++) {
        pmf[i] = pmf[i] / summed_valid;
        cdf[i] = accum_sum;
        accum_sum += pmf[i];
    }
    cdf[num_faces] = 1.0;

    // Finally sample the index proportionally to dot products
    uint32_t index = 0;
    for (size_t i = 0; i < num_faces; i++) {
        if (cdf[i] <= sample)
            index += 1;
    }
    index -= 1;
    return index;
}

NTWR_DEVICE glm::vec3 sample_cube_face_from_index(uint32_t index, const glm::vec2 &rng, glm::vec3 &face_normal)
{
    // Samples a uniform point on a uniformly sampled cube face (in [-1,1])
    assert(index < 6);

    face_normal = get_cube_face_normal(index);

    return get_cube_pos_on_face(index, rng.x, rng.y);
}

}