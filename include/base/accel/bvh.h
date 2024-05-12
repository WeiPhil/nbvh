#pragma once

#include <bit>
#include <bitset>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

#include "base/obj_writer.h"
#include "cuda/base/ray.cuh"
#include "cuda/utils/common.cuh"

#include "app/render_backend.h"

// Only build with C++ compiler (cuda C++ 20 features not supported)
#ifndef __CUDACC__
#include <bvh/v2/bvh.h>
#include <bvh/v2/default_builder.h>
#include <bvh/v2/executor.h>
#include <bvh/v2/node.h>
#include <bvh/v2/thread_pool.h>
#include <bvh/v2/tri.h>
#include <bvh/v2/vec.h>

using LibNode = bvh::v2::Node<float, 3>;
using LibBVH  = bvh::v2::Bvh<LibNode>;
using LibBBox = bvh::v2::BBox<float, 3>;

#endif

#define WRITE_BVH_OBJ 0

#define INVALID_NODE -1

typedef unsigned char byte;

#ifdef _MSC_VER
#include <intrin.h>
#define __builtin_popcount __popcnt
#endif
namespace ntwr {

// Forward declaration
struct BVH8Converter;

struct BaseBVH {
#ifndef __CUDACC__
    virtual void build_from_libbvh(const LibBVH &lib_bvh) = 0;
#endif

    virtual void write_BVH_to_OBJ(
        ObjWriter &obj_writer, uint32_t node_idx, int depth, int current_level, int max_output_level = -1)
    {
    }

    virtual void generate_node_graph(uint32_t node_idx, std::ofstream &dot_file) {}

    virtual void write_BVH_graph() {}

    std::vector<size_t> m_indices;
};

/// BVH node in 32 bytes, note that we can't directly use Vector as it is
/// padded to 16 bytes
struct BVH2Node {
    glm::vec3 aabb_min;
    uint32_t prim_count;
    glm::vec3 aabb_max;
    // Offset of left node (if interior node) or first primitive in list (if leaf node)
    uint32_t left_or_prim_offset;

    void assign_bbox(Bbox3f bbox);

    /// Is this a leaf node?
    NTWR_HOST_DEVICE bool is_leaf() const
    {
        return prim_count > 0;
    }

    /// Assuming this is a leaf node, return the number of primitives
    NTWR_HOST_DEVICE uint32_t primitive_count() const
    {
        assert(is_leaf());
        return prim_count;
    }

    /// Assuming this is a leaf node, return the first primitive index
    NTWR_HOST_DEVICE uint32_t primitive_offset() const
    {
        assert(is_leaf());
        return left_or_prim_offset;
    }

    /// Assuming that this is an inner node, return the relative offset to
    /// the left child
    NTWR_HOST_DEVICE uint32_t left_offset() const
    {
        assert(!is_leaf());
        return left_or_prim_offset;
    }

    /// Return the bounding box of the node (only on host since expensive on gpu)
    NTWR_HOST_DEVICE Bbox3f bbox() const
    {
        return Bbox3f{aabb_min, aabb_max};
    }

    /// Return the minimum of bounding box's node
    NTWR_HOST_DEVICE glm::vec3 bbox_min() const
    {
        return aabb_min;
    }

    /// Return the maximum of bounding box's node
    NTWR_HOST_DEVICE glm::vec3 bbox_max() const
    {
        return aabb_max;
    }

    // Does NOT ignore the ray extents value and returns mint or FLT_MAX if no intersection in
    // (ray.mint,ray.maxt)
    NTWR_HOST_DEVICE float intersect(const Ray3f &ray) const
    {
        glm::vec3 t0s = (aabb_min - ray.o) / ray.d;
        glm::vec3 t1s = (aabb_max - ray.o) / ray.d;

        glm::vec3 tsmaller = min(t0s, t1s);
        glm::vec3 tbigger  = max(t0s, t1s);

        float mint = glm::max(ray.mint, component_max(tsmaller));
        float maxt = glm::min(ray.maxt, component_min(tbigger));

        return (mint <= maxt) ? mint : FLT_MAX;
    }

    // Slab test from "A Ray-Box Intersection Algorithm and Efficient Dynamic Voxel
    // Rendering, Majercik et al. 2018", ignore ray maxt and mint if asked
    NTWR_HOST_DEVICE bool intersect_min_max(const Ray3f &ray,
                                            float &mint,
                                            float &maxt,
                                            bool check_ray_bounds = false) const
    {
        glm::vec3 t0s = (aabb_min - ray.o) / ray.d;
        glm::vec3 t1s = (aabb_max - ray.o) / ray.d;

        glm::vec3 tsmaller = glm::min(t0s, t1s);
        glm::vec3 tbigger  = glm::max(t0s, t1s);

        mint = check_ray_bounds ? glm::max(ray.mint, component_max(tsmaller)) : component_max(tsmaller);
        maxt = check_ray_bounds ? glm::min(ray.maxt, component_min(tbigger)) : component_min(tbigger);

        return (mint <= maxt);
    }

    NTWR_HOST_DEVICE bool solve_quadratic(float a, float b, float c, float &x0, float &x1) const
    {
        /* Linear case */
        if (a == 0) {
            if (b != 0) {
                x0 = x1 = -c / b;
                return true;
            }
            return false;
        }
        float discrim = b * b - 4.0f * a * c;
        /* Leave if there is no solution */
        if (discrim < 0)
            return false;
        float temp, sqrtDiscrim = sqrt(discrim);
        /* Numerically stable version of (-b (+/-) sqrtDiscrim) / (2 * a)
         *
         * Based on the observation that one solution is always
         * accurate while the other is not. Finds the solution of
         * greater magnitude which does not suffer from loss of
         * precision and then uses the identity x1 * x2 = c / a
         */
        if (b < 0)
            temp = -0.5f * (b - sqrtDiscrim);
        else
            temp = -0.5f * (b + sqrtDiscrim);

        x0 = temp / a;
        x1 = c / temp;
        /* Return the results so that x0 < x1 */
        if (x0 > x1) {
            // swap(x0, x1);
            float tmp = x0;
            x0        = x1;
            x1        = tmp;
        }

        return true;
    }

    // Ellipsoid intersection
    NTWR_HOST_DEVICE bool intersect_ellipsoid_min_max(const Ray3f &ray,
                                                      float &mint,
                                                      float &maxt,
                                                      bool check_maxt = false) const
    {
        // ((o + t * d - center()) / radius())^2 = sqrt(2)^2
        // t^2*(d/r)^2 + 2(o-c)/r td/r + (o-c)^2/r^2-sqrt(2)^2 = 0

        const auto diagonal = bbox_extent();
        const auto center   = bbox_center();

        const auto dOverR = 2.f * ray.d / diagonal;
        const auto pOverR = 2.f * (ray.o - center) / diagonal;

        const float a = glm::length2(dOverR);
        const float b = 2.f * dot(pOverR, dOverR);
        const float c = glm::length2(pOverR) - 3.f;

        if (!solve_quadratic(a, b, c, mint, maxt)) {
            mint = +FLT_MAX;
            maxt = -FLT_MAX;
        }
        if (check_maxt) {
            maxt = glm::min(ray.maxt, maxt);
        }
        return (mint <= maxt);
    }

    NTWR_HOST_DEVICE glm::vec3 bbox_center() const
    {
        return aabb_min + (bbox_extent() * 0.5f);
    }

    NTWR_HOST_DEVICE glm::vec3 bbox_extent() const
    {
        return (aabb_max - aabb_min);
    }

    NTWR_HOST_DEVICE glm::vec3 bbox_to_local(glm::vec3 world_pos) const
    {
        glm::vec3 local_pos = (world_pos - aabb_min) / bbox_extent();
        return local_pos;
    }

    NTWR_HOST_DEVICE glm::vec3 local_to_bbox(glm::vec3 local_pos) const
    {
        glm::vec3 world_pos = local_pos * bbox_extent() + aabb_min;
        return world_pos;
    }

    NTWR_HOST_DEVICE Bbox3f ellipsoid_bbox() const
    {
        glm::vec3 half_axes = bbox_extent() * (sqrt(3.f) * 0.5f);
        return Bbox3f{bbox_center() - half_axes, bbox_center() + half_axes};
    }

    NTWR_HOST_DEVICE glm::vec3 ellipsoid_bbox_to_local(glm::vec3 world_pos) const
    {
        glm::vec3 half_axes         = bbox_extent() * (sqrt(3.f) * 0.5f);
        glm::vec3 elipsoid_bbox_min = bbox_center() - half_axes;
        glm::vec3 local_pos         = (world_pos - elipsoid_bbox_min) / (2.f * half_axes);
        return local_pos;
    }

    NTWR_HOST_DEVICE glm::vec3 local_to_ellipsoid_bbox(glm::vec3 local_pos) const
    {
        glm::vec3 half_axes         = bbox_extent() * (sqrt(3.f) * 0.5f);
        glm::vec3 elipsoid_bbox_min = bbox_center() - half_axes;
        glm::vec3 world_pos         = local_pos * (2.f * half_axes) + elipsoid_bbox_min;
        return world_pos;
    }

    /// Return a string representation
    void print_node();
};

struct BVH2 : BaseBVH {
#ifndef __CUDACC__
    virtual void build_from_libbvh(const LibBVH &lib_bvh) override;
#endif

    virtual void write_BVH_to_OBJ(
        ObjWriter &obj_writer, uint32_t node_idx, int depth, int current_level, int max_output_level = -1) override;

    virtual void generate_node_graph(uint32_t node_idx, std::ofstream &dot_file) override;

    virtual void write_BVH_graph() override;

    Bbox3f root_bbox()
    {
        return m_nodes[0].bbox();
    }

    std::vector<BVH2Node> m_nodes;
};

struct BVH8Node {
    glm::vec3 p;
    byte e[3];
    byte imask;

    uint32_t base_index_child;
    uint32_t base_index_triangle;

    byte meta[8] = {};

    byte quantized_min_x[8] = {}, quantized_max_x[8] = {};
    byte quantized_min_y[8] = {}, quantized_max_y[8] = {};
    byte quantized_min_z[8] = {}, quantized_max_z[8] = {};

    inline bool is_leaf(int child_index) const
    {
        bool is_interior_or_leaf = (meta[child_index] & 0b11100000) == 0b00100000;
        return !(is_interior_or_leaf && (meta[child_index] & 0b00011111) >= 24);
    }

    inline bool is_empty(int child_index) const
    {
        return meta[child_index] == 0b00000000;
    }

    inline int get_child_count() const
    {
        int result = 0;

        for (int child_index = 0; child_index < 8; child_index++) {
            if (!is_empty(child_index))
                result++;
        }

        return result;
    }

    // Returns storage index of first triangle if leaf node OR child node index if interior node
    inline uint32_t get_triangle_index(int child_index) const
    {
        assert(is_leaf(child_index));
        uint32_t triangle_index = meta[child_index] & 0b00011111;
        return base_index_triangle + triangle_index;
    }

    // Returns storage index of first triangle if leaf node OR child node index if interior node
    inline uint32_t get_num_triangles(int child_index) const
    {
        assert(is_leaf(child_index));
        return __builtin_popcount(((uint32_t(meta[child_index]) & 0b11100000) >> 5));
    }
};

static_assert(sizeof(BVH8Node) == 80);

struct BVH8 : BaseBVH {
    // Dummy constructors required to allow the unique_ptr of incomplete type
    BVH8();
    ~BVH8();

#ifndef __CUDACC__
    virtual void build_from_libbvh(const LibBVH &lib_bvh) override;
#endif

    Bbox3f root_bbox()
    {
        return m_root_bbox;
    }

    // This function should take LibBVH directly but we can't compile with Cuda Standard 20 because of tinycudann
    // so we bypass this issue by storing the libbvh in the RenderBackend
    void build_fixed_depth_from_cpu_bvh(const BVH2 &bvh,
                                        int max_depth,
                                        const BVH8Converter &precomputed_bvh8_converter,
                                        std::vector<BVH2Node> *leaf_nodes);

    virtual void generate_node_graph(uint32_t node_idx, std::ofstream &dot_file) override;

    virtual void write_BVH_graph() override;

    const BVH8Converter &bvh_converter();

    std::vector<BVH8Node> m_nodes;

    Bbox3f m_root_bbox;

    std::unique_ptr<BVH8Converter> m_bvh_converter;
};

}