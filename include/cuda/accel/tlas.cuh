#pragma once

#include "cuda/cuda_backend_constants.cuh"

#include "cuda/accel/bvh2.cuh"
#include "cuda/accel/bvh8.cuh"

namespace ntwr {

using CudaBLAS = CudaBVH2;

#define TLAS_INTERIOR_NODE 0
#define TLAS_LEAF          1
#define TLAS_NEURAL_LEAF   2

constexpr int TLAS_STACK_SIZE = 10;

// Needed, TODO: understand why
extern __constant__ uint32_t fb_size_flat;

namespace tlas_utils {

    NTWR_DEVICE uint32_t stack_index(uint32_t ray_idx, int stack_size)
    {
        return ray_idx + stack_size * fb_size_flat;
    }

    // Pushes on the stack located in the ray payload
    template <typename T>
    NTWR_DEVICE void stack_push(uint32_t ray_idx, T stack[], int &stack_size, T item)
    {
        assert(stack_size < TLAS_STACK_SIZE);
        stack[stack_index(ray_idx, stack_size)] = item;
        stack_size++;
    }

    // Pops from the stack located in the ray payload
    template <typename T>
    NTWR_DEVICE T stack_pop(uint32_t ray_idx, const T stack[], int &stack_size)
    {
        assert(stack_size > 0);
        stack_size--;
        return stack[stack_index(ray_idx, stack_size)];
    }

    // Pushes to a thread local stack
    template <typename T>
    NTWR_DEVICE void local_stack_push(T stack[], int &stack_size, T item)
    {
        assert(stack_size < TLAS_STACK_SIZE);
        stack[stack_size] = item;
        stack_size++;
    }

    // Pops from a thread local stack
    template <typename T>
    NTWR_DEVICE T local_stack_pop(const T stack[], int &stack_size)
    {
        assert(stack_size > 0);
        stack_size--;
        return stack[stack_size];
    }
}
struct TLASNode {
    glm::vec3 aabb_min;
    uint32_t left_or_blas_offset;
    glm::vec3 aabb_max;
    uint32_t leaf_type;

    NTWR_HOST_DEVICE bool is_leaf() const
    {
        return leaf_type >= TLAS_LEAF;
    }

    NTWR_HOST_DEVICE bool is_neural_leaf() const
    {
        return leaf_type == TLAS_NEURAL_LEAF;
    }

    NTWR_HOST_DEVICE bool is_non_neural_leaf() const
    {
        return leaf_type == TLAS_LEAF;
    }

    /// Assuming that this is an leaf node, return the offset to
    /// the blas
    NTWR_HOST_DEVICE uint32_t blas_offset() const
    {
        // We need to check is_interior_at_lod instead
        assert(is_leaf());
        return left_or_blas_offset;
    }

    /// Assuming that this is an inner node, return the relative offset to
    /// the left child
    NTWR_HOST_DEVICE uint32_t left_offset() const
    {
        // We need to check is_interior_at_lod instead
        assert(!is_leaf());
        return left_or_blas_offset;
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
};

// #define OMMIT_NEURAL
// #define OMMIT_NON_NEURAL

struct CudaTLAS {
    TLASNode *tlas_nodes;
    uint32_t num_tlas_nodes;

    uint32_t *blas_indices;

    CudaBLAS *mesh_bvhs;
    uint32_t num_meshes;

    template <bool ShadowRay>
    NTWR_DEVICE PreliminaryItsData intersect(const Ray3f &_ray) const
    {
        PreliminaryItsData pi;
        Ray3f ray(_ray);

        // Starting at root
        uint32_t tlas_node_idx = 0;
        TLASNode *tlas_node    = &tlas_nodes[tlas_node_idx];

        uint32_t stack[TLAS_STACK_SIZE];
        int stack_size = 0;

        while (true) {
            if (tlas_node->is_leaf()) {
#ifdef OMMIT_NEURAL
                if (!tlas_node->is_neural_leaf()) {
#endif
#ifdef OMMIT_NON_NEURAL
                    if (!tlas_node->is_non_neural_leaf()) {
#endif
                        uint32_t mesh_blas_offset = blas_indices[tlas_node->blas_offset()];
                        assert(mesh_blas_offset < num_meshes);

                        PreliminaryItsData blas_pi = mesh_bvhs[mesh_blas_offset].intersect<ShadowRay>(ray);

                        if (blas_pi.is_valid() && blas_pi.t < ray.maxt) {
                            pi          = blas_pi;
                            pi.mesh_idx = mesh_blas_offset;
                            if constexpr (ShadowRay) {
                                return pi;
                            }

                            assert(pi.t >= ray.mint && pi.t < ray.maxt);

                            ray.maxt = pi.t;
                        }
#ifdef OMMIT_NEURAL
                    }
#endif
#ifdef OMMIT_NON_NEURAL
                }
#endif

                if (stack_size == 0) {
                    return pi;
                } else {
                    tlas_node_idx = stack[--stack_size];
                    tlas_node     = &tlas_nodes[tlas_node_idx];
                }
                continue;
            }
            assert(!tlas_node->is_leaf());

            uint32_t child1_index = tlas_node->left_offset();
            TLASNode *child1      = &tlas_nodes[child1_index];
            uint32_t child2_index = tlas_node->left_offset() + 1;
            TLASNode *child2      = &tlas_nodes[child2_index];

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
                    tlas_node_idx = stack[--stack_size];
                    tlas_node     = &tlas_nodes[tlas_node_idx];
                }
            } else {
                tlas_node_idx = child1_index;
                tlas_node     = &tlas_nodes[tlas_node_idx];
                if (child2_mint != FLT_MAX)
                    stack[stack_size++] = child2_index;
            }
        }
        return pi;
    }

    NTWR_DEVICE bool intersect_tlas_only(const Ray3f &_ray,
                                         uint32_t ray_idx,
                                         uint32_t &blas_idx,
                                         bool &is_neural_blas,
                                         uint32_t *tlas_stack,
                                         int &stack_size,
                                         bool use_local_stack = false) const
    {
        using tlas_utils::local_stack_pop;
        using tlas_utils::local_stack_push;
        using tlas_utils::stack_pop;
        using tlas_utils::stack_push;
        Ray3f ray(_ray);
        assert(ray.maxt == _ray.maxt);

        is_neural_blas = false;
        assert(!use_local_stack || ray_idx == 0);

        assert(stack_size > 0 && stack_size <= TLAS_STACK_SIZE);
        // Starting at root
        uint32_t tlas_node_idx =
            use_local_stack ? local_stack_pop(tlas_stack, stack_size) : stack_pop(ray_idx, tlas_stack, stack_size);
        TLASNode *tlas_node = &tlas_nodes[tlas_node_idx];

        while (true) {
            if (tlas_node->is_leaf()) {
#ifdef OMMIT_NEURAL
                if (tlas_node->is_neural_leaf()) {
                    if (stack_size == 0) {
                        return false;
                    } else {
                        tlas_node_idx = use_local_stack ? local_stack_pop(tlas_stack, stack_size)
                                                        : stack_pop(ray_idx, tlas_stack, stack_size);
                        tlas_node     = &tlas_nodes[tlas_node_idx];
                    }
                    continue;
                }
#endif

#ifdef OMMIT_NON_NEURAL
                if (tlas_node->is_non_neural_leaf()) {
                    if (stack_size == 0) {
                        return false;
                    } else {
                        tlas_node_idx = use_local_stack ? local_stack_pop(tlas_stack, stack_size)
                                                        : stack_pop(ray_idx, tlas_stack, stack_size);
                        tlas_node     = &tlas_nodes[tlas_node_idx];
                    }
                    continue;
                }
#endif
                if (num_meshes == 1 && tlas_node->intersect(ray) == FLT_MAX) {
                    return false;
                }
                assert(tlas_node->intersect(ray) != FLT_MAX);
                is_neural_blas = tlas_node->is_neural_leaf();
                blas_idx       = blas_indices[tlas_node->blas_offset()];
                return true;
            }
            assert(!tlas_node->is_leaf());

            uint32_t child1_index = tlas_node->left_offset();
            TLASNode *child1      = &tlas_nodes[child1_index];
            uint32_t child2_index = tlas_node->left_offset() + 1;
            TLASNode *child2      = &tlas_nodes[child2_index];

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
                    return false;
                else {
                    tlas_node_idx = use_local_stack ? local_stack_pop(tlas_stack, stack_size)
                                                    : stack_pop(ray_idx, tlas_stack, stack_size);
                    tlas_node     = &tlas_nodes[tlas_node_idx];
                }
            } else {
                tlas_node_idx = child1_index;
                tlas_node     = &tlas_nodes[tlas_node_idx];
                if (child2_mint != FLT_MAX) {
                    if (use_local_stack)
                        local_stack_push(tlas_stack, stack_size, child2_index);
                    else
                        stack_push(ray_idx, tlas_stack, stack_size, child2_index);

                    if (stack_size >= TLAS_STACK_SIZE) {
                        printf("Out of bound tlas stack!\n");
                        return false;
                    }
                }
            }
        }
        assert(false);
        return false;
    }
};
}