#pragma once

#include "cuda/neural/accel/neural_bvh.cuh"
#include "cuda/neural/accel/neural_traversal_data.cuh"

#include "cuda/neural/neural_bvh_renderer_constants.cuh"

namespace ntwr {

namespace neural {

    // Both a neural leaf node and corresponds to the original bvh leaf node
    // i.e. this node can't be splitted further, the ULTIMATE leaf node
    static constexpr uint32_t NUM_BVH_LODS = 7;

    static constexpr uint32_t LEAF_FINAL      = 1;
    static constexpr uint32_t HIGHEST_BVH_LOD = LEAF_FINAL + 1;
    static constexpr uint32_t LOWEST_BVH_LOD  = HIGHEST_BVH_LOD + NUM_BVH_LODS - 1;
    static constexpr uint32_t LEAF_LOD(uint32_t bvh_lod)
    {
        assert(bvh_lod < NUM_BVH_LODS);
        return HIGHEST_BVH_LOD + bvh_lod;
    }
    static constexpr uint32_t START_PRIM_COUNT = LOWEST_BVH_LOD + 1;

    // We use a special value of 2^32-1 to indicate a neural node that has no child yet
    // This is only used to build the LoDs incrementally and check if a leaf node (which is NOT at the highest detail)
    // can be overwritten.
    static constexpr uint32_t NO_NEURAL_CHILD = UINT32_MAX;

    // prim_count between 0 and NUM_BVH_LODS all represent different types of

    struct NeuralBVH2Node : private BVH2Node {
        using BVH2Node::aabb_max;
        using BVH2Node::aabb_min;
        using BVH2Node::left_or_prim_offset;
        using BVH2Node::prim_count;
        // We just selectively get the functions we need from BVH2Node to avoid any confusion
        // between the two representation of hybrid nodes, i.e. we either need to get the function or hide it
        // explicitely
        using BVH2Node::assign_bbox;
        using BVH2Node::bbox;
        using BVH2Node::bbox_center;
        using BVH2Node::bbox_extent;
        using BVH2Node::bbox_to_local;
        using BVH2Node::ellipsoid_bbox;
        using BVH2Node::ellipsoid_bbox_to_local;
        using BVH2Node::intersect;
        using BVH2Node::intersect_ellipsoid_min_max;
        using BVH2Node::intersect_min_max;
        using BVH2Node::local_to_bbox;
        using BVH2Node::local_to_ellipsoid_bbox;
        using BVH2Node::solve_quadratic;

        /// Assuming that this is an inner node, return the relative offset to
        /// the left child
        NTWR_HOST_DEVICE uint32_t left_offset() const
        {
            // We need to check is_interior_at_lod instead
            // assert(!is_leaf());
            return left_or_prim_offset;
        }

        NTWR_HOST_DEVICE bool has_neural_child() const
        {
            return left_or_prim_offset != NO_NEURAL_CHILD;
        }

        NTWR_DEVICE bool intersect_primitive_min_max(const Ray3f &ray,
                                                     float &mint,
                                                     float &maxt,
                                                     bool check_ray_bounds  = false,
                                                     float sampled_dilation = 0.f) const
        {
            if (bvh_const.ellipsoidal_bvh) {
                bool valid_its = intersect_ellipsoid_min_max(ray, mint, maxt, check_ray_bounds);
                if (valid_its && bvh_const.rescale_ellipsoid) {
                    auto segment_scaling = (maxt - mint) * (sqrt(3.f) * 0.5f);
                    auto segment_mid     = mint + (maxt - mint) * 0.5f;
                    mint                 = segment_mid - segment_scaling * 0.5f;
                    maxt                 = segment_mid + segment_scaling * 0.5f;
                }
                return valid_its;
            } else {
                if (sampled_dilation != 0.f) {
                    const glm::vec3 node_center      = bbox_center();
                    const glm::vec3 node_extent      = bbox_extent() * (1.f + sampled_dilation);
                    const glm::vec3 dilated_aabb_min = node_center - node_extent * 0.5f;
                    const glm::vec3 dilated_aabb_max = node_center + node_extent * 0.5f;
                    return ray.intersect_bbox_min_max(dilated_aabb_min, dilated_aabb_max, mint, maxt, check_ray_bounds);
                } else {
                    return intersect_min_max(ray, mint, maxt, check_ray_bounds);
                }
            }
        }

        NTWR_DEVICE float intersect_traversal_bbox(const Ray3f &ray, float sampled_dilation, bool inference) const
        {
            float mint, maxt;
            const bool check_ray_bounds = true;
            bool valid_its              = false;
            if (bvh_const.ellipsoidal_bvh)
                valid_its = ray.intersect_bbox_min_max(ellipsoid_bbox(), mint, maxt, check_ray_bounds);
            else {
                if (sampled_dilation != 0.f) {
                    const glm::vec3 node_center      = bbox_center();
                    const glm::vec3 node_extent      = bbox_extent() * (1.f + sampled_dilation);
                    const glm::vec3 dilated_aabb_min = node_center - node_extent * 0.5f;
                    const glm::vec3 dilated_aabb_max = node_center + node_extent * 0.5f;
                    valid_its =
                        ray.intersect_bbox_min_max(dilated_aabb_min, dilated_aabb_max, mint, maxt, check_ray_bounds);
                } else {
                    valid_its = intersect_min_max(ray, mint, maxt, check_ray_bounds);
                }
            }
            return valid_its ? mint : FLT_MAX;
        }

        NTWR_HOST_DEVICE bool is_final_leaf() const
        {
            return prim_count == LEAF_FINAL;
        }

        /// Is this a leaf node at the queried lod?
        /// Higher lods (coareser details) always consider lower lods (finer details) as leaf nodes
        NTWR_HOST_DEVICE bool is_leaf_at_lod(uint32_t bvh_lod) const
        {
            assert(bvh_lod < NUM_BVH_LODS);
            return (prim_count > 0 && prim_count <= LEAF_LOD(bvh_lod));
        }

        /// Is this a leaf node assigned to the queried lod?
        NTWR_HOST_DEVICE bool is_leaf_assigned_to_lod(uint32_t bvh_lod) const
        {
            assert(bvh_lod < NUM_BVH_LODS);
            return prim_count == LEAF_LOD(bvh_lod);
        }

        NTWR_HOST_DEVICE bool is_interior_at_lod(uint32_t bvh_lod) const
        {
            assert(bvh_lod < NUM_BVH_LODS);
            // If the prim count is 0 we can only know what type of lod this is by traversing the tree
            if (prim_count == 0 || (prim_count > LEAF_LOD(bvh_lod) && prim_count <= LOWEST_BVH_LOD)) {
                assert(has_neural_child());
                return true;
            } else {
                return false;
            }
        }

        /// Assuming this is a leaf node, return the number of primitives
        /// For a neural node that is offseted by the special lod flags used
        NTWR_HOST_DEVICE uint32_t primitive_count() const
        {
            return prim_count - START_PRIM_COUNT;
        }
    };

    static_assert(sizeof(NeuralBVH2Node) == sizeof(BVH2Node));

    struct NeuralBVH2LeafNodes {
        NeuralBVH2Node *nodes;
        uint32_t num_nodes;
    };

    struct NeuralBVH2 {
        NeuralBVH2Node *nodes;
        uint32_t num_nodes;

        NTWR_DEVICE bool intersect(uint32_t bvh_lod,
                                   const Ray3f &_ray,
                                   uint32_t ray_idx,
                                   uint32_t *stack,
                                   int &stack_size,
                                   bool use_local_stack,
                                   float *segment_mint,
                                   float *segment_maxt,
                                   bool inference,
                                   float current_maxt        = FLT_MAX,
                                   uint32_t *neural_node_idx = nullptr,
                                   float sampled_dilation    = bvh_const.fixed_node_dilation) const
        {
            assert(!use_local_stack || ray_idx == 0);

            assert(_ray.mint == 1e-4f);

            // This is to make sure we also get the node min/max in case the ray origin is inside a node
            Ray3f ray(_ray);
            ray.mint = -FLT_MAX;
            ray.maxt = current_maxt;

            uint32_t node_idx =
                use_local_stack ? local_stack_pop(stack, stack_size) : stack_pop(ray_idx, stack, stack_size);
            NeuralBVH2Node *node = &nodes[node_idx];

            while (true) {
                if (node->is_leaf_at_lod(bvh_lod) || !node->has_neural_child()) {
                    float mint, maxt;
                    if (node->intersect_primitive_min_max(ray, mint, maxt, false, sampled_dilation)) {
                        float maxt_epsilon =
                            inference ? bvh_const.inference_maxt_epsilon : bvh_const.learning_maxt_epsilon;
                        float mint_epsilon =
                            inference ? bvh_const.inference_mint_epsilon : bvh_const.learning_mint_epsilon;
                        bool valid_its = mint < ray.maxt;
                        if (mint < 0.f) {
                            valid_its &= maxt > maxt_epsilon;
                        } else {
                            valid_its &= mint > mint_epsilon;
                        }
                        // old approach
                        // valid_its &= mint < ray.maxt && maxt > maxt_epsilon && abs(mint) > mint_epsilon;
                        if (valid_its) {
                            segment_mint[ray_idx] = mint;
                            segment_maxt[ray_idx] = maxt;

                            if (neural_node_idx)
                                *neural_node_idx = node_idx;
                            return true;
                        }
                    }

                    if (stack_size == 0) {
                        return false;
                    } else {
                        node_idx = use_local_stack ? local_stack_pop(stack, stack_size)
                                                   : stack_pop(ray_idx, stack, stack_size);
                        node     = &nodes[node_idx];
                    }
                    continue;
                }
                assert(node->is_interior_at_lod(bvh_lod));

                uint32_t child1_index  = node->left_offset();
                NeuralBVH2Node *child1 = &nodes[child1_index];
                uint32_t child2_index  = node->left_offset() + 1;
                NeuralBVH2Node *child2 = &nodes[child2_index];

#if 0
                node_idx = child1_index;
                node     = &nodes[node_idx];
                if (use_local_stack)
                    local_stack_push(stack, stack_size, child2_index);
                else
                    stack_push(ray_idx, stack, stack_size, child2_index);
                if (stack_size >= NEURAL_STACK_SIZE) {
                    printf("Out of bound stack!\n");
                    return false;
                }

#else
                float child1_mint = child1->intersect_traversal_bbox(ray, sampled_dilation, inference);
                float child2_mint = child2->intersect_traversal_bbox(ray, sampled_dilation, inference);

                // Swap order for faster traversal
                if (child1_mint > child2_mint) {
                    float tmp_dist = child1_mint;
                    child1_mint    = child2_mint;
                    child2_mint    = tmp_dist;

                    uint32_t tmp_child_idx = child1_index;
                    child1_index           = child2_index;
                    child2_index           = tmp_child_idx;
                }

                if (child1_mint == FLT_MAX) {
                    if (stack_size == 0) {
                        return false;
                    } else {
                        node_idx = use_local_stack ? local_stack_pop(stack, stack_size)
                                                   : stack_pop(ray_idx, stack, stack_size);
                        node     = &nodes[node_idx];
                    }
                } else {
                    node_idx = child1_index;
                    node     = &nodes[node_idx];
                    if (child2_mint != FLT_MAX) {
                        if (use_local_stack)
                            local_stack_push(stack, stack_size, child2_index);
                        else
                            stack_push(ray_idx, stack, stack_size, child2_index);

                        if (stack_size >= NEURAL_STACK_SIZE) {
                            printf("Out of bound stack!\n");
                            return false;
                        }
                    }
                }
#endif
            }
            assert(false);
            return false;
        }

        NTWR_DEVICE bool intersect_closest_node(uint32_t bvh_lod,
                                                const Ray3f &_ray,
                                                float &segment_mint,
                                                float &segment_maxt,
                                                uint32_t *neural_node_idx = nullptr,
                                                float sampled_dilation    = 0.f) const
        {
            Ray3f ray(_ray);
            // This is to make sure we also get the node min/max in case the ray origin is inside a node
            ray.mint          = -FLT_MAX;
            ray.maxt          = FLT_MAX;
            float closest_its = FLT_MAX;

            uint32_t stack[BVH_STACK_SIZE];
            int stack_size = 0;

            uint32_t node_idx    = 0;
            NeuralBVH2Node *node = &nodes[node_idx];

            while (true) {
                if (node->is_leaf_at_lod(bvh_lod) || !node->has_neural_child()) {
                    float mint, maxt;
                    if (node->intersect_primitive_min_max(ray, mint, maxt, false, sampled_dilation)) {
                        bool hit = false;
                        if (mint >= 0.f && mint <= closest_its) {
                            hit         = true;
                            closest_its = mint;
                        } else if (maxt >= 0.f && maxt <= closest_its) {
                            // mint is negative (or larger than closest_its implying maxt > closest_its)
                            hit         = true;
                            closest_its = maxt;
                        }
                        if (hit) {
                            segment_mint = mint;
                            segment_maxt = maxt;
                            ray.maxt     = maxt;
                            if (neural_node_idx)
                                *neural_node_idx = node_idx;
                        }
                    }

                    if (stack_size == 0) {
                        break;
                    } else {
                        node_idx = local_stack_pop(stack, stack_size);
                        node     = &nodes[node_idx];
                    }
                    continue;
                }
                assert(node->is_interior_at_lod(bvh_lod));

                uint32_t child1_index  = node->left_offset();
                NeuralBVH2Node *child1 = &nodes[child1_index];
                uint32_t child2_index  = node->left_offset() + 1;
                NeuralBVH2Node *child2 = &nodes[child2_index];

                float child1_mint = child1->intersect_traversal_bbox(ray, sampled_dilation, true);
                float child2_mint = child2->intersect_traversal_bbox(ray, sampled_dilation, true);

                // Swap order for faster traversal
                if (child1_mint > child2_mint) {
                    float tmp_dist = child1_mint;
                    child1_mint    = child2_mint;
                    child2_mint    = tmp_dist;

                    uint32_t tmp_child_idx = child1_index;
                    child1_index           = child2_index;
                    child2_index           = tmp_child_idx;
                }

                if (child1_mint == FLT_MAX) {
                    if (stack_size == 0) {
                        break;
                    } else {
                        node_idx = local_stack_pop(stack, stack_size);
                        node     = &nodes[node_idx];
                    }
                } else {
                    node_idx = child1_index;
                    node     = &nodes[node_idx];
                    if (child2_mint != FLT_MAX) {
                        local_stack_push(stack, stack_size, child2_index);
                        if (stack_size >= NEURAL_STACK_SIZE) {
                            printf("Out of bound stack!\n");
                            return false;
                        }
                    }
                }
            }

            return closest_its != FLT_MAX;
        }
    };

}
}