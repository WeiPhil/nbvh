#pragma once

#include "base/accel/bvh.h"
#include "cuda/accel/bvh.cuh"
#include "cuda/base/ray.cuh"

namespace ntwr {

struct CudaBVH2 : BaseCudaBLAS {
    BVH2Node *nodes;

    template <bool ShadowRay>
    __device__ PreliminaryItsData intersect(const Ray3f &_ray) const
    {
        Ray3f ray(_ray);
        PreliminaryItsData pi;

        // Starting at root
        uint32_t node_idx = 0;
        BVH2Node *node    = &nodes[node_idx];

        uint32_t stack[32];
        int stack_size = 0;

        while (true) {
            if (node->is_leaf()) {
                uint32_t prim_start = node->primitive_offset();
                uint32_t prim_end   = prim_start + node->primitive_count();

                for (uint32_t prim_id = prim_start; prim_id < prim_end; prim_id++) {
                    uint32_t prim_idx = prim_indices[prim_id];

                    PreliminaryItsData prim_pi = intersect_triangle(ray, prim_idx);

                    if (prim_pi.t < ray.maxt) {
                        pi = prim_pi;
                        if constexpr (ShadowRay) {
                            return pi;
                        }

                        assert(pi.t >= ray.mint && pi.t < ray.maxt);

                        ray.maxt = pi.t;
                    }
                }
                if (stack_size == 0) {
                    return pi;
                } else {
                    node_idx = stack[--stack_size];
                    node     = &nodes[node_idx];
                }
                continue;
            }

            uint32_t child1_index = node->left_offset();
            BVH2Node *child1      = &nodes[child1_index];
            uint32_t child2_index = node->left_offset() + 1;
            BVH2Node *child2      = &nodes[child2_index];

            float child1_mint = child1->intersect(ray);
            float child2_mint = child2->intersect(ray);

            // Swap order for faster traversal
            if (child1_mint > child2_mint) {
                float tmp_dist         = child1_mint;
                child1_mint            = child2_mint;
                child2_mint            = tmp_dist;
                uint32_t tmp_child_idx = child1_index;
                child1_index           = child2_index;
                child2_index           = tmp_child_idx;
            }

            if (child1_mint == FLT_MAX) {
                if (stack_size == 0)
                    return pi;
                else {
                    node_idx = stack[--stack_size];
                    node     = &nodes[node_idx];
                }
            } else {
                node_idx = child1_index;
                node     = &nodes[node_idx];
                if (child2_mint != FLT_MAX)
                    stack[stack_size++] = child2_index;
            }
        }
        return pi;
    }
};

}