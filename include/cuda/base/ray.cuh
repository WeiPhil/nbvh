#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include "glm_pch.h"

#include "base/bbox.h"
// #include "cuda/utils/common.cuh"

namespace ntwr {

struct ItsData {
    glm::vec3 position;
    float t;
    glm::vec3 sh_normal;
    glm::vec3 geo_normal;
    glm::vec2 uv;  // mesh uv or texture uv if uv exist
    int material_id;
};

struct PreliminaryItsData {
    float t = FLT_MAX;
    float u;  // mesh uv
    float v;  // mesh uv
    uint32_t prim_idx;
    uint32_t mesh_idx;

    NTWR_HOST_DEVICE bool is_valid() const
    {
        return t != FLT_MAX;
    }
};

struct PreliminaryItsDataSoA {
    uint4 *hits;

#ifdef __CUDACC__
    __device__ void set(int index, const PreliminaryItsData &pi)
    {
        uint32_t uv = int(pi.u * 65535.0f) | (int(pi.v * 65535.0f) << 16);

        hits[index] = make_uint4(pi.mesh_idx, pi.prim_idx, __float_as_uint(pi.t), uv);
    }

    __device__ PreliminaryItsData get(int index) const
    {
        uint4 hit = __ldg(&hits[index]);

        PreliminaryItsData pi;

        pi.mesh_idx = hit.x;
        pi.prim_idx = hit.y;

        pi.t = __uint_as_float(hit.z);

        pi.u = float(hit.w & 0xffff) / 65535.0f;
        pi.v = float(hit.w >> 16) / 65535.0f;

        return pi;
    }
#endif
};

struct Ray3f {
    glm::vec3 o;
    float mint;
    glm::vec3 d;
    float maxt;

    NTWR_HOST_DEVICE Ray3f(const glm::vec3 &origin = glm::vec3(0),
                           const glm::vec3 &dir    = glm::vec3(1, 0, 0),
                           float mint              = 1e-4f,
                           float maxt              = FLT_MAX)
        : o(origin), mint(mint), d(dir), maxt(maxt)
    {
    }

    NTWR_HOST_DEVICE glm::vec3 operator()(const float &t) const
    {
        return o + t * d;
    }

    NTWR_HOST_DEVICE PreliminaryItsData intersect_triangle(const glm::vec3 &p0,
                                                           const glm::vec3 &p1,
                                                           const glm::vec3 &p2) const
    {
        PreliminaryItsData pi;
        glm::vec3 e1 = p1 - p0, e2 = p2 - p0;

        glm::vec3 pvec = cross(this->d, e2);
        float inv_det  = 1.f / dot(e1, pvec);

        glm::vec3 tvec = this->o - p0;
        float u        = dot(tvec, pvec) * inv_det;
        if (u < 0.f || u > 1.f)
            return pi;

        glm::vec3 qvec = cross(tvec, e1);
        float v        = dot(this->d, qvec) * inv_det;
        if (v < 0.f || ((u + v) > 1.f))
            return pi;

        float t = dot(e2, qvec) * inv_det;
        if (t >= 0.f && t <= this->maxt) {
            pi.t = t;
            pi.u = u;
            pi.v = v;
        }
        return pi;
    }

    // IGNORES the ray.maxt value and returns true if mint < maxt (i.e. mint can be negative)
    // Slab test from "A Ray-Box Intersection Algorithm and Efficient Dynamic Voxel
    // Rendering, Majercik et al. 2018", ignore ray maxt (if not indicated otherwise) and mint
    NTWR_HOST_DEVICE bool intersect_bbox_min_max(const glm::vec3 &aabb_min,
                                                 const glm::vec3 &aabb_max,
                                                 float &min_t,
                                                 float &max_t,
                                                 bool check_ray_bounds = false) const
    {
        glm::vec3 t0s = (aabb_min - this->o) / this->d;
        glm::vec3 t1s = (aabb_max - this->o) / this->d;

        glm::vec3 tsmaller = glm::min(t0s, t1s);
        glm::vec3 tbigger  = glm::max(t0s, t1s);

        min_t = check_ray_bounds ? glm::max(this->mint, component_max(tsmaller)) : component_max(tsmaller);
        max_t = check_ray_bounds ? glm::min(this->maxt, component_min(tbigger)) : component_min(tbigger);

        return (min_t <= max_t);
    }

    NTWR_HOST_DEVICE bool intersect_bbox_min_max(const ntwr::Bbox3f &bbox,
                                                 float &min_t,
                                                 float &max_t,
                                                 bool check_ray_bounds = false) const
    {
        return intersect_bbox_min_max(bbox.min, bbox.max, min_t, max_t, check_ray_bounds);
    }
};

}