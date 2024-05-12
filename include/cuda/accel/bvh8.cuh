#pragma once

#include "base/accel/bvh.h"
#include "cuda/accel/bvh.cuh"
#include "cuda/base/ray.cuh"
#include "cuda/utils/cudart/cuda_math.h"

#define BVH_STACK_SIZE 16
namespace ntwr {

struct CudaBVH8Node {
    glm::float4 data_0;
    glm::float4 data_1;
    glm::float4 data_2;
    glm::float4 data_3;
    glm::float4 data_4;
    /* *** Equivalent to BVH8Node on Host side ***
     *
     *   glm::vec3 p;
     *   byte e[3];
     *   byte imask;
     *
     *   uint32_t base_index_child;
     *   uint32_t base_index_triangle;
     *
     *   byte meta[8] = {};
     *
     *   byte quantized_min_x[8] = {}, quantized_max_x[8] = {};
     *   byte quantized_min_y[8] = {}, quantized_max_y[8] = {};
     *   byte quantized_min_z[8] = {}, quantized_max_z[8] = {};
     */
};

#ifdef __CUDACC__
NTWR_DEVICE uint32_t bvh8_node_intersect(const Ray3f &ray,
                                         uint32_t oct_inv4,
                                         float min_distance,
                                         float max_distance,
                                         float4 node_data_0,
                                         float4 node_data_1,
                                         float4 node_data_2,
                                         float4 node_data_3,
                                         float4 node_data_4)
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

    glm::vec3 adjusted_ray_direction_inv = glm::vec3(
        __uint_as_float(e_x << 23) * idirx, __uint_as_float(e_y << 23) * idiry, __uint_as_float(e_z << 23) * idirz);
    glm::vec3 adjusted_ray_origin = (p - ray.o) / ray.d;

    uint32_t hit_mask = 0;

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
            glm::vec3 tmin3 =
                glm::vec3(float(extract_byte(x_min, j)), float(extract_byte(y_min, j)), float(extract_byte(z_min, j)));
            glm::vec3 tmax3 =
                glm::vec3(float(extract_byte(x_max, j)), float(extract_byte(y_max, j)), float(extract_byte(z_max, j)));

            // Account for grid origin and scale
            tmin3 = tmin3 * adjusted_ray_direction_inv + adjusted_ray_origin;
            tmax3 = tmax3 * adjusted_ray_direction_inv + adjusted_ray_origin;

            // BROKEN vmax_max!
            float tmin = fmaxf(fmaxf(fmaxf(tmin3.x, tmin3.y), tmin3.z), min_distance);
            float tmax = fminf(fminf(fminf(tmax3.x, tmax3.y), tmax3.z), max_distance);
            // float tmin = fmaxf(vmax_max(tmin3.x, tmin3.y, tmin3.z), min_distance);
            // float tmax = fminf(vmin_min(tmax3.x, tmax3.y, tmax3.z), max_distance);

            bool intersected = tmin <= tmax;
            if (intersected) {
                uint32_t child_bits = extract_byte(child_bits4, j);
                uint32_t bit_index  = extract_byte(bit_index4, j);
                hit_mask |= child_bits << bit_index;
            }
        }
    }

    return hit_mask;
}

// Inverse of ray octant, encoded in 3 bits, duplicated for each byte
NTWR_DEVICE uint32_t ray_get_octant_inv4(const glm::vec3 ray_dir)
{
    return (ray_dir.x < 0.0f ? 0 : 0x04040404) | (ray_dir.y < 0.0f ? 0 : 0x02020202) |
           (ray_dir.z < 0.0f ? 0 : 0x01010101);
}
#endif

struct CudaBVH8 : BaseCudaBLAS {
    CudaBVH8Node *nodes;

    // Function that decides whether to push on the shared stack or thread local stack
    template <typename T>
    NTWR_DEVICE void stack_push(T stack[], int &stack_size, T item) const
    {
        assert(stack_size < BVH_STACK_SIZE);
        stack[stack_size] = item;
        stack_size++;
    }

    // Function that decides whether to pop from the shared stack or thread local stack
    template <typename T>
    NTWR_DEVICE T stack_pop(const T stack[], int &stack_size) const
    {
        assert(stack_size > 0);
        stack_size--;
        return stack[stack_size];
    }

#ifdef __CUDACC__
    template <bool ShadowRay>
    NTWR_DEVICE PreliminaryItsData intersect(const Ray3f &_ray) const
    {
        Ray3f ray(_ray);
        PreliminaryItsData pi;

        uint2 stack[BVH_STACK_SIZE];
        int stack_size = 0;

        uint32_t oct_inv4 = ray_get_octant_inv4(ray.d);

        // Root is always traversed, first hits bit set to 1 all the rest 0s
        // Current group |child node base index  | hits     | pad     | imask  |
        //               |       32 bits         | 8 bits   | 16 bits | 8 bits |
        uint2 current_node_group = make_uint2(0, 0x80000000);

        do {
            // Triangle group |triangle base index  | pad     | triangle hits |
            //                |       32 bits       | 8 bits  | 24 bits       |
            uint2 triangle_group = make_uint2(0, 0);

            // At least a node is still active and needs intersection
            assert(current_node_group.y & 0xff000000);

            uint32_t hits_imask = current_node_group.y;

            // Get first node hit
            uint32_t child_index_offset = msb(hits_imask);
            uint32_t child_index_base   = current_node_group.x;

            // Remove node from current_node_group
            current_node_group.y &= ~(1 << child_index_offset);

            // If the node group is not yet empty, push the rest on the stack
            if (current_node_group.y & 0xff000000) {
                stack_push(stack, stack_size, current_node_group);
            }

            // reversing the traversal priority computation
            uint32_t slot_index = (child_index_offset - 24) ^ (oct_inv4 & 0xff);
            // Relative index of the node is obtained by computing the number of neighboring nodes
            // stored in the lower child slots:
            uint32_t relative_index = __popc(hits_imask & ~(0xffffffff << slot_index));

            uint32_t child_node_index = child_index_base + relative_index;

            assert(child_node_index < num_nodes);

            const CudaBVH8Node &node = nodes[child_node_index];

            float4 node_data_0 = __ldg((float4 *)&node.data_0);
            float4 node_data_1 = __ldg((float4 *)&node.data_1);
            float4 node_data_2 = __ldg((float4 *)&node.data_2);
            float4 node_data_3 = __ldg((float4 *)&node.data_3);
            float4 node_data_4 = __ldg((float4 *)&node.data_4);

            uint32_t hitmask = bvh8_node_intersect(
                ray, oct_inv4, ray.mint, ray.maxt, node_data_0, node_data_1, node_data_2, node_data_3, node_data_4);

            byte imask = extract_byte(__float_as_uint(node_data_0.w), 3);

            current_node_group.x = __float_as_uint(node_data_1.x);  // child base offset
            triangle_group.x     = __float_as_uint(node_data_1.y);  // triangle base offset

            current_node_group.y = (hitmask & 0xff000000) | uint32_t(imask);
            triangle_group.y     = (hitmask & 0x00ffffff);

            // Intersect all triangles in the current node
            while (triangle_group.y != 0) {
                // Get next triangle index and remove it from the list
                int triangle_index = msb(triangle_group.y);
                triangle_group.y &= ~(1 << triangle_index);

                uint32_t prim_idx          = prim_indices[triangle_group.x + triangle_index];
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

            // Get next node from stack
            if ((current_node_group.y & 0xff000000) == 0) {
                if (stack_size == 0) {
                    break;
                }

                current_node_group = stack_pop(stack, stack_size);
            }

        } while (true);
        return pi;
    }

#endif
};
}