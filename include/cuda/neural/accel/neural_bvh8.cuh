#pragma once

#include "cuda/neural/accel/neural_bvh.cuh"
#include "cuda/neural/accel/neural_traversal_data.cuh"
#include "cuda/neural/neural_bvh_module_constants.cuh"

namespace ntwr {

namespace neural {

    struct NeuralBVH8 {
        CudaBVH8Node *nodes;
        uint32_t num_nodes;

#ifdef __CUDACC__
        // TODO merge with original bvh8 node_intersect ?
        NTWR_DEVICE uint32_t neural_bvh8_node_intersect(const Ray3f &ray,
                                                        uint32_t ray_idx,
                                                        uint32_t oct_inv4,
                                                        float max_distance,
                                                        float4 node_data_0,
                                                        float4 node_data_1,
                                                        float4 node_data_2,
                                                        float4 node_data_3,
                                                        float4 node_data_4,
                                                        float *segment_mints,
                                                        float *segment_maxts,
                                                        uint8_t &leaves_intersected,
                                                        bool local_segment_index = false) const
        {
            glm::vec3 p = glm::vec3(node_data_0.x, node_data_0.y, node_data_0.z);

            uint32_t e_imask = __float_as_uint(node_data_0.w);
            byte e_x         = extract_byte(e_imask, 0);
            byte e_y         = extract_byte(e_imask, 1);
            byte e_z         = extract_byte(e_imask, 2);

            const float ooeps = exp2f(-80.f);

            float idirx = 1.0f / (fabsf(ray.d.x) > ooeps ? ray.d.x : copysignf(ooeps, ray.d.x));
            float idiry = 1.0f / (fabsf(ray.d.y) > ooeps ? ray.d.y : copysignf(ooeps, ray.d.y));
            float idirz = 1.0f / (fabsf(ray.d.z) > ooeps ? ray.d.z : copysignf(ooeps, ray.d.z));

            glm::vec3 adjusted_ray_direction_inv = glm::vec3(__uint_as_float(e_x << 23) * idirx,
                                                             __uint_as_float(e_y << 23) * idiry,
                                                             __uint_as_float(e_z << 23) * idirz);
            glm::vec3 adjusted_ray_origin        = (p - ray.o) / ray.d;

            uint32_t hit_mask = 0;

// #define SORT_MINTS
#ifdef SORT_MINTS
            float mints[8];
            float maxts[8];
#endif

#pragma unroll
            for (int i = 0; i < 2; i++) {
                uint32_t meta4 = __float_as_uint(i == 0 ? node_data_1.z : node_data_1.w);

                uint32_t is_inner4   = (meta4 & (meta4 << 1)) & 0x10101010;
                uint32_t inner_mask4 = sign_extend_s8x4(is_inner4 << 3);
                uint32_t bit_index4  = (meta4 ^ (oct_inv4 & inner_mask4)) & 0x1f1f1f1f;
                uint32_t child_bits4 = (meta4 >> 5) & 0x07070707;

                // Select near and far planes based on ray octant
                uint32_t q_lo_x = __float_as_uint(i == 0 ? node_data_2.x : node_data_2.y);
                uint32_t q_hi_x = __float_as_uint(i == 0 ? node_data_2.z : node_data_2.w);

                uint32_t q_lo_y = __float_as_uint(i == 0 ? node_data_3.x : node_data_3.y);
                uint32_t q_hi_y = __float_as_uint(i == 0 ? node_data_3.z : node_data_3.w);

                uint32_t q_lo_z = __float_as_uint(i == 0 ? node_data_4.x : node_data_4.y);
                uint32_t q_hi_z = __float_as_uint(i == 0 ? node_data_4.z : node_data_4.w);

                uint32_t x_min = idirx < 0.0f ? q_hi_x : q_lo_x;
                uint32_t x_max = idirx < 0.0f ? q_lo_x : q_hi_x;

                uint32_t y_min = idiry < 0.0f ? q_hi_y : q_lo_y;
                uint32_t y_max = idiry < 0.0f ? q_lo_y : q_hi_y;

                uint32_t z_min = idirz < 0.0f ? q_hi_z : q_lo_z;
                uint32_t z_max = idirz < 0.0f ? q_lo_z : q_hi_z;

#pragma unroll
                for (int j = 0; j < 4; j++) {
                    // Extract j-th byte
                    glm::vec3 tmin3 = glm::vec3(
                        float(extract_byte(x_min, j)), float(extract_byte(y_min, j)), float(extract_byte(z_min, j)));
                    glm::vec3 tmax3 = glm::vec3(
                        float(extract_byte(x_max, j)), float(extract_byte(y_max, j)), float(extract_byte(z_max, j)));

                    // Account for grid origin and scale
                    tmin3 = tmin3 * adjusted_ray_direction_inv + adjusted_ray_origin;
                    tmax3 = tmax3 * adjusted_ray_direction_inv + adjusted_ray_origin;

                    // BROKEN vmax_max!
                    float tmin = fmaxf(fmaxf(tmin3.x, tmin3.y), tmin3.z);
                    float tmax = fminf(fminf(fminf(tmax3.x, tmax3.y), tmax3.z), max_distance);
                    // float tmin = fmaxf(vmax_max(tmin3.x, tmin3.y, tmin3.z), min_distance);
                    // float tmax = fminf(vmin_min(tmax3.x, tmax3.y, tmax3.z), max_distance);

                    // bool intersected = tmin <= tmax;
                    bool intersected = tmin <= tmax && (tmin >= 0.f || tmax >= 0.f);
                    if (intersected) {
                        uint32_t child_bits = extract_byte(child_bits4, j);
                        uint32_t bit_index  = extract_byte(bit_index4, j);
                        hit_mask |= child_bits << bit_index;

                        if ((child_bits << bit_index) & 0x00ffffff) {  // Check if a leaf node
#ifdef SORT_MINTS
                            uint32_t segment_index = leaves_intersected++;
                            mints[segment_index]   = tmin;
                            maxts[segment_index]   = tmax;
#else
                            uint32_t segment_index = leaves_intersected++;
                            segment_index =
                                local_segment_index ? segment_index : get_segment_index(ray_idx, segment_index);
                            segment_mints[segment_index] = tmin;
                            segment_maxts[segment_index] = tmax;

                            if (bvh_module_const.merge_operator) {
                                for (int i = 0; i < leaves_intersected - 1; i++) {
                                    uint32_t other_segment_index =
                                        local_segment_index ? i : get_segment_index(ray_idx, i);
                                    if (segment_mints[segment_index] <= segment_maxts[other_segment_index] &&
                                        segment_maxts[segment_index] >= segment_mints[other_segment_index]) {
                                        // The new interval is inside the other. Min/max and dismiss the other
                                        segment_mints[other_segment_index] =
                                            fminf(segment_mints[segment_index], segment_mints[other_segment_index]);
                                        segment_maxts[other_segment_index] =
                                            fmaxf(segment_maxts[segment_index], segment_maxts[other_segment_index]);

                                        leaves_intersected--;
                                        break;
                                    }
                                }
                            }

#endif
                        }
                    }
                }
            }

#ifdef SORT_MINTS
            int indices[8];
            for (int i = 0; i < leaves_intersected; i++) {
                indices[i] = i;
            }
            for (int i = 0; i < leaves_intersected - 1; i++) {
                for (int j = i + 1; j < leaves_intersected; j++) {
                    if (mints[indices[i]] > mints[indices[j]]) {
                        // Swap indices
                        int temp   = indices[i];
                        indices[i] = indices[j];
                        indices[j] = temp;
                    }
                }
            }
            for (int i = 0; i < leaves_intersected; i++) {
                segment_mints[local_segment_index ? i : get_segment_index(ray_idx, i)] = mints[indices[i]];
                segment_maxts[local_segment_index ? i : get_segment_index(ray_idx, i)] = maxts[indices[i]];
            }
#endif

            return hit_mask;
        }

        NTWR_DEVICE uint8_t intersect(NeuralTraversalData &traversal_data,
                                      uint32_t ray_idx,
                                      uint2 *stack,
                                      int &stack_size,
                                      float *segment_mints,
                                      float *segment_maxts,
                                      float maxt             = FLT_MAX,
                                      bool using_local_stack = false)
        {
            Ray3f ray = traversal_data.get_ray(ray_idx);
            ray.maxt  = maxt;
            return intersect(ray, ray_idx, stack, stack_size, segment_mints, segment_maxts, using_local_stack);
        }

        NTWR_DEVICE uint8_t intersect(const Ray3f &ray,
                                      uint32_t ray_idx,
                                      uint2 *stack,
                                      int &stack_size,
                                      float *segment_mints,
                                      float *segment_maxts,
                                      bool using_local_stack = false)
        {
            uint32_t oct_inv4 = ray_get_octant_inv4(ray.d);

            // Current group |child node base index  | hits     | pad     | imask  |
            //               |       32 bits         | 8 bits   | 16 bits | 8 bits |
            uint2 current_node_group =
                using_local_stack ? local_stack_pop(stack, stack_size) : stack_pop(ray_idx, stack, stack_size);

            do {
                // Triangle group |triangle base index  | pad     | triangle hits |
                //                |       32 bits       | 8 bits  | 24 bits       |
                uint32_t leaf_hitmask      = 0;
                uint8_t leaves_intersected = 0;

                assert(current_node_group.y & 0xff000000);

                if (current_node_group.y & 0xff000000) {
                    uint32_t hits_imask = current_node_group.y;

                    // Get first node hit
                    uint32_t child_index_offset = msb(hits_imask);
                    uint32_t child_index_base   = current_node_group.x;

                    // Remove node from current_node_group
                    current_node_group.y &= ~(1 << child_index_offset);

                    // If the node group is not yet empty, push the rest on the stack
                    if (current_node_group.y & 0xff000000) {
                        if (using_local_stack)
                            local_stack_push(stack, stack_size, current_node_group);
                        else
                            stack_push(ray_idx, stack, stack_size, current_node_group);
                    }

                    uint32_t slot_index     = (child_index_offset - 24) ^ (oct_inv4 & 0xff);
                    uint32_t relative_index = __popc(hits_imask & ~(0xffffffff << slot_index));

                    uint32_t child_node_index = child_index_base + relative_index;

                    assert(child_node_index < num_nodes);

                    const CudaBVH8Node &node = nodes[child_node_index];

                    float4 node_data_0 = __ldg((float4 *)&node.data_0);
                    float4 node_data_1 = __ldg((float4 *)&node.data_1);
                    float4 node_data_2 = __ldg((float4 *)&node.data_2);
                    float4 node_data_3 = __ldg((float4 *)&node.data_3);
                    float4 node_data_4 = __ldg((float4 *)&node.data_4);

                    uint32_t hitmask = neural_bvh8_node_intersect(ray,
                                                                  ray_idx,
                                                                  oct_inv4,
                                                                  ray.maxt,
                                                                  node_data_0,
                                                                  node_data_1,
                                                                  node_data_2,
                                                                  node_data_3,
                                                                  node_data_4,
                                                                  segment_mints,
                                                                  segment_maxts,
                                                                  leaves_intersected,
                                                                  using_local_stack);

                    byte imask = extract_byte(__float_as_uint(node_data_0.w), 3);

                    current_node_group.x = __float_as_uint(node_data_1.x);  // child base offset
                    // We don't care about the triangle base offset
                    // leaf_group.x         = __float_as_uint(node_data_1.y);  // triangle base offset

                    current_node_group.y = (hitmask & 0xff000000) | uint32_t(imask);
                    leaf_hitmask         = (hitmask & 0x00ffffff);

                } else {
                    leaf_hitmask       = current_node_group.y;
                    current_node_group = make_uint2(0);
                }

                // If any leaf node was intersected in the current node break
                if (leaf_hitmask) {
                    // If there are nodes to intersect still after those leaf nodes, add them on the stack
                    if (current_node_group.y & 0xff000000) {
                        if (using_local_stack)
                            local_stack_push(stack, stack_size, current_node_group);
                        else
                            stack_push(ray_idx, stack, stack_size, current_node_group);
                    }

                    return leaves_intersected;
                }

                // Get next node from stack if none left
                if ((current_node_group.y & 0xff000000) == 0) {
                    if (stack_size == 0) {
                        return 0;
                    }
                    current_node_group =
                        using_local_stack ? local_stack_pop(stack, stack_size) : stack_pop(ray_idx, stack, stack_size);
                }

            } while (true);

            // Unreachable
            assert(false);
        }

        NTWR_DEVICE bool intersect_closest_node(const Ray3f &ray,
                                                float &segment_mint,
                                                float &segment_maxt,
                                                uint32_t *neural_node_idx = nullptr) const
        {
            float ray_maxt       = ray.maxt;
            uint32_t oct_inv4    = ray_get_octant_inv4(ray.d);
            float best_candidate = FLT_MAX;

            uint2 stack[BVH_STACK_SIZE];
            int stack_size = 0;
            // Current group |child node base index  | hits     | pad     | imask  |
            //               |       32 bits         | 8 bits   | 16 bits | 8 bits |
            uint2 current_node_group = make_uint2(0, 0x80000000);

            do {
                // Triangle group |triangle base index  | pad     | triangle hits |
                //                |       32 bits       | 8 bits  | 24 bits       |
                uint32_t leaf_hitmask      = 0;
                uint8_t leaves_intersected = 0;

                float segment_mints[8];
                float segment_maxts[8];

                assert(current_node_group.y & 0xff000000);

                if (current_node_group.y & 0xff000000) {
                    uint32_t hits_imask = current_node_group.y;

                    // Get first node hit
                    uint32_t child_index_offset = msb(hits_imask);
                    uint32_t child_index_base   = current_node_group.x;

                    // Remove node from current_node_group
                    current_node_group.y &= ~(1 << child_index_offset);

                    // If the node group is not yet empty, push the rest on the stack
                    if (current_node_group.y & 0xff000000) {
                        local_stack_push(stack, stack_size, current_node_group);
                    }

                    uint32_t slot_index     = (child_index_offset - 24) ^ (oct_inv4 & 0xff);
                    uint32_t relative_index = __popc(hits_imask & ~(0xffffffff << slot_index));

                    uint32_t child_node_index = child_index_base + relative_index;

                    assert(child_node_index < num_nodes);

                    const CudaBVH8Node &node = nodes[child_node_index];

                    float4 node_data_0 = __ldg((float4 *)&node.data_0);
                    float4 node_data_1 = __ldg((float4 *)&node.data_1);
                    float4 node_data_2 = __ldg((float4 *)&node.data_2);
                    float4 node_data_3 = __ldg((float4 *)&node.data_3);
                    float4 node_data_4 = __ldg((float4 *)&node.data_4);

                    uint32_t hitmask = neural_bvh8_node_intersect(ray,
                                                                  0,  // not used if local_segment_index == true
                                                                  oct_inv4,
                                                                  ray_maxt,
                                                                  node_data_0,
                                                                  node_data_1,
                                                                  node_data_2,
                                                                  node_data_3,
                                                                  node_data_4,
                                                                  segment_mints,
                                                                  segment_maxts,
                                                                  leaves_intersected,
                                                                  true);

                    byte imask = extract_byte(__float_as_uint(node_data_0.w), 3);

                    current_node_group.x = __float_as_uint(node_data_1.x);  // child base offset
                    // We don't care about the triangle base offset
                    // leaf_group.x         = __float_as_uint(node_data_1.y);  // triangle base offset

                    current_node_group.y = (hitmask & 0xff000000) | uint32_t(imask);
                    leaf_hitmask         = (hitmask & 0x00ffffff);

                } else {
                    leaf_hitmask       = current_node_group.y;
                    current_node_group = make_uint2(0);
                }

                // If any leaf node was intersected in the current node update
                if (leaf_hitmask) {
                    for (uint8_t segment_idx = 0; segment_idx < leaves_intersected; segment_idx++) {
                        if (segment_mints[segment_idx] >= 0.f && segment_mints[segment_idx] <= best_candidate) {
                            segment_mint   = segment_mints[segment_idx];
                            segment_maxt   = segment_maxts[segment_idx];
                            best_candidate = segment_mints[segment_idx];
                            ray_maxt       = segment_maxts[segment_idx];
                        } else if (segment_maxts[segment_idx] >= 0.f && segment_maxts[segment_idx] <= best_candidate) {
                            // mint is negative (or larger than best_candidate implying maxt > best_candidate)
                            segment_mint   = segment_mints[segment_idx];
                            segment_maxt   = segment_maxts[segment_idx];
                            best_candidate = segment_maxts[segment_idx];
                            ray_maxt       = segment_maxts[segment_idx];
                        }
                    }
                }

                // Get next node from stack if none left
                if ((current_node_group.y & 0xff000000) == 0) {
                    if (stack_size == 0) {
                        break;
                    }
                    current_node_group = local_stack_pop(stack, stack_size);
                }

            } while (true);

            return best_candidate != FLT_MAX;
        }
#endif
    };

}
}