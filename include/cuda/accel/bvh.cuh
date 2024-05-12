#pragma once

#include "base/accel/bvh.h"
#include "cuda/base/material.cuh"
#include "cuda/base/ray.cuh"
#include "cuda/utils/common.cuh"
#include "cuda/utils/types.cuh"

namespace ntwr {

struct BaseCudaBLAS {
    uint32_t num_nodes;
    size_t *prim_indices;
    uint32_t num_prim_indices;

    glm::vec3 *vertices;
    glm::uvec3 *indices;
    glm::vec3 *normals;
    glm::vec2 *texcoords;
    int *material_ids;

    uint32_t *material_map;
    uint32_t num_materials;

    NTWR_DEVICE uint32_t upper_bound(const uint32_t *data, uint32_t size, uint32_t val) const
    {
        uint32_t left  = 0;
        uint32_t right = size;

        while (left < right) {
            uint32_t mid = left + (right - left) / 2;
            if (data[mid] <= val) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        return left - 1;
    }

    NTWR_DEVICE uint32_t find_material_idx(uint32_t prim_idx) const
    {
        uint32_t material_index;

        material_index = upper_bound(material_map, num_materials, prim_idx);

        assert(material_index < num_materials);

        return material_index;
    }

// A bit annoying but needed
#ifdef __CUDACC__
    NTWR_DEVICE ItsData compute_its_data(const PreliminaryItsData &pi,
                                         PackedMaterial *materials,
                                         cudaTextureObject_t *textures) const
    {
        ItsData its;
        its.t = pi.t;

        glm::uvec3 index = indices[pi.prim_idx];

        const uint32_t material_idx = find_material_idx(pi.prim_idx);
        its.material_id             = material_ids[material_idx];

        const glm::vec3 &v0 = vertices[index.x];
        const glm::vec3 &v1 = vertices[index.y];
        const glm::vec3 &v2 = vertices[index.z];
        its.geo_normal      = normalize(cross(v1 - v0, v2 - v0));
        float w             = (1.f - pi.u - pi.v);

        its.sh_normal = normals ? normalize((w * normals[index.x] + pi.u * normals[index.y] + pi.v * normals[index.z]))
                                : its.geo_normal;

        its.position = (w * v0 + pi.u * v1 + pi.v * v2);
        its.uv       = !texcoords ? glm::vec2(pi.u, pi.v)
                                  : (w * texcoords[index.x] + pi.u * texcoords[index.y] + pi.v * texcoords[index.z]);

        // Lookup textured tangent space normal if any and convert to world space normal
        if (its.material_id != -1) {
            glm::vec3 unpacked_normal = unpack_normal(materials[its.material_id], its.uv, textures);
            if (unpacked_normal != glm::vec3(-1.f)) {
                glm::vec3 tangent_space_normal = normalize((unpacked_normal - 0.5f));

                glm::vec3 tangent_world, bitangent_world;
                ortho_basis(tangent_world, bitangent_world, its.sh_normal);
                glm::mat3 local_to_world(tangent_world, bitangent_world, its.sh_normal);

                its.sh_normal = local_to_world * tangent_space_normal;
            }
        }

        return its;
    }
#endif

    NTWR_DEVICE PreliminaryItsData intersect_triangle(const Ray3f &ray, uint32_t prim_idx) const
    {
        PreliminaryItsData pi;

        pi.prim_idx = prim_idx;

        glm::vec3 v0 = vertices[indices[prim_idx].x];
        glm::vec3 v1 = vertices[indices[prim_idx].y];
        glm::vec3 v2 = vertices[indices[prim_idx].z];

        glm::vec3 e1 = v1 - v0;
        glm::vec3 e2 = v2 - v0;

        glm::vec3 pvec = cross(ray.d, e2);
        const auto det = dot(e1, pvec);
        if (det > -1e-8f && det < 1e-8f)
            return pi;
        const float inv_det = 1.f / det;

        glm::vec3 tvec = ray.o - v0;
        float u        = dot(tvec, pvec) * inv_det;
        if (u < 0.f || u > 1.f)
            return pi;

        glm::vec3 qvec = cross(tvec, e1);
        float v        = dot(ray.d, qvec) * inv_det;
        if (v < 0.f || (u + v) > 1.f)
            return pi;

        float t = dot(e2, qvec) * inv_det;
        if (t < ray.mint || t >= ray.maxt) {
            return pi;
        }

        pi.t = t;
        pi.u = u;
        pi.v = v;
        return pi;
    }
};

}