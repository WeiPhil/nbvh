#pragma once

#include "cuda/accel/bvh_traversal_data.cuh"
#include "cuda/base/ray.cuh"
#include "cuda/cuda_backend_constants.cuh"
#include "cuda/utils/types.cuh"

namespace ntwr {

namespace neural {

    struct NeuralTraversalData : CudaBVHTraversalData {
        // Default constructor to make compiler happy with unique_ptr to types containing NeuralTraversalData
        NeuralTraversalData() {}
        // Initialises a NeuralTraversalData structure from device pointers
        NeuralTraversalData(vec3_soa ray_o,
                            vec3_soa ray_d,
                            uint8_t *ray_depth,
                            uint8_t *inferenced_node_idx_and_intersected_nodes,
                            float *leaf_mints,
                            float *leaf_maxts,
                            int *stack_size,
                            uint32_t *stack,
                            uint32_t *pixel_coord_active,
                            uint32_t *node_idx)
        {
            this->ray_o     = ray_o;
            this->ray_d     = ray_d;
            this->ray_depth = ray_depth;

            this->inferenced_node_idx_and_intersected_nodes = inferenced_node_idx_and_intersected_nodes;

            this->leaf_mint          = leaf_mints;
            this->leaf_maxt          = leaf_maxts;
            this->stack              = stack;
            this->stack_size         = stack_size;
            this->pixel_coord_active = pixel_coord_active;
            this->node_idx           = node_idx;
        }

        NTWR_DEVICE void init_from_first_traversal_data(uint32_t index,
                                                        uint32_t pixel_coord,
                                                        int local_stack_size,
                                                        uint32_t *local_stack,
                                                        uint8_t nodes_intersected,
                                                        float *local_leaf_mint,
                                                        float *local_leaf_maxt,
                                                        const Ray3f &ray,
                                                        uint8_t depth,
                                                        const LCGRand &rng,
                                                        bool init_path_data        = false,
                                                        int local_tlas_stack_size  = 0,
                                                        uint32_t *local_tlas_stack = nullptr,
                                                        uint32_t node_index        = 0)
        {
            // Set Ray data
            ray_o.x[index] = ray.o.x;
            ray_o.y[index] = ray.o.y;
            ray_o.z[index] = ray.o.z;

            ray_d.x[index] = ray.d.x;
            ray_d.y[index] = ray.d.y;
            ray_d.z[index] = ray.d.z;

            ray_depth[index] = depth;

            // Set Traversal data
            set_intersected_nodes(index, nodes_intersected);

            leaf_mint[index] = local_leaf_mint[0];
            leaf_maxt[index] = local_leaf_maxt[0];

            uint32_t dst_index;
            for (int stack_idx = 0; stack_idx < local_stack_size; stack_idx++) {
                dst_index        = index + stack_idx * fb_size_flat;
                stack[dst_index] = local_stack[stack_idx];
            }
            stack_size[index] = local_stack_size;
            init_pixel_coord(index, pixel_coord);

            if (init_path_data) {
                illumination[index]   = glm::vec3(0.f);
                throughput_rng[index] = glm::vec4(1.f, 1.f, 1.f, glm::uintBitsToFloat(rng.state));
            }

            if (local_tlas_stack) {
                for (int stack_idx = 0; stack_idx < local_tlas_stack_size; stack_idx++) {
                    dst_index             = index + stack_idx * fb_size_flat;
                    tlas_stack[dst_index] = local_tlas_stack[stack_idx];
                }
                tlas_stack_size[index] = local_tlas_stack_size;
            }
            if (node_idx != nullptr) {
                node_idx[index] = node_index;
            }
        }

        NTWR_DEVICE void set(uint32_t index, const NeuralTraversalData &other, uint32_t other_index)
        {
            // Set Ray data
            ray_o.x[index] = other.ray_o.x[other_index];
            ray_o.y[index] = other.ray_o.y[other_index];
            ray_o.z[index] = other.ray_o.z[other_index];

            ray_d.x[index] = other.ray_d.x[other_index];
            ray_d.y[index] = other.ray_d.y[other_index];
            ray_d.z[index] = other.ray_d.z[other_index];

            ray_depth[index] = other.ray_depth[other_index];

            // Set Traversal data
            inferenced_node_idx_and_intersected_nodes[index] =
                other.inferenced_node_idx_and_intersected_nodes[other_index];

            leaf_mint[index] = other.leaf_mint[other_index];
            leaf_maxt[index] = other.leaf_maxt[other_index];

            uint32_t src_index, dst_index;
            for (int stack_idx = 0; stack_idx < other.stack_size[other_index]; stack_idx++) {
                src_index        = other_index + stack_idx * fb_size_flat;
                dst_index        = index + stack_idx * fb_size_flat;
                stack[dst_index] = other.stack[src_index];
            }
            stack_size[index]         = other.stack_size[other_index];
            pixel_coord_active[index] = other.pixel_coord_active[other_index];

            if (illumination != nullptr) {
                // Initialise to zero
                illumination[index] = other.illumination[other_index];
            }
            if (throughput_rng != nullptr) {
                throughput_rng[index] = other.throughput_rng[other_index];
            }

            if (tlas_stack != nullptr) {
                for (int stack_idx = 0; stack_idx < other.tlas_stack_size[other_index]; stack_idx++) {
                    src_index             = other_index + stack_idx * fb_size_flat;
                    dst_index             = index + stack_idx * fb_size_flat;
                    tlas_stack[dst_index] = other.tlas_stack[src_index];
                }
                tlas_stack_size[index] = other.tlas_stack_size[other_index];
            }

            if (node_idx != nullptr) {
                node_idx[index] = other.node_idx[other_index];
            }
        }

        NTWR_DEVICE uint8_t depth(uint32_t index) const
        {
            return ray_depth[index];
        }

        NTWR_DEVICE uint8_t intersected_nodes(uint32_t index) const
        {
            return inferenced_node_idx_and_intersected_nodes[index] & 0x0F;
        }

        NTWR_DEVICE void set_intersected_nodes(uint32_t index, uint8_t new_nodes_batch_size)
        {
            // Not only sets the number of intersected nodes but reset the node counter to 0!
            // (upper four bits are set to 0 as long as new_nodes_batch size is smaller than 0x0F)
            inferenced_node_idx_and_intersected_nodes[index] = new_nodes_batch_size;
        }

        NTWR_DEVICE uint8_t inferenced_node_idx(uint32_t index) const
        {
            return inferenced_node_idx_and_intersected_nodes[index] >> 4;
        }

        NTWR_DEVICE void set_inferenced_node_idx(uint32_t index, uint8_t new_inferenced_node)
        {
            inferenced_node_idx_and_intersected_nodes[index] &= 0x0F;
            inferenced_node_idx_and_intersected_nodes[index] |= (0xF0 & (new_inferenced_node << 4));
        }

        NTWR_DEVICE void increment_inferenced_node_idx(uint32_t index)
        {
            set_inferenced_node_idx(index, inferenced_node_idx(index) + 1);
        }

        NTWR_DEVICE bool all_nodes_processed(uint32_t index) const
        {
            // checks whether upper four bits are smaller than lower bits
            return inferenced_node_idx(index) == intersected_nodes(index);
        }

        /* Neural BVH Traversal Data */

        // Lower 4 bits store current node batch size (max 1 for bvh2, up to 8 for BVH8)
        // Upper 4 bits store the index of the inferenced node (always 0 for BVH2, max 8 for BVH8)
        uint8_t *inferenced_node_idx_and_intersected_nodes = nullptr;
        float *leaf_mint                                   = nullptr;
        float *leaf_maxt = nullptr;  // Note : for BVH8 this represents 8 * num_rays mint/maxt

        // Stack data
        int *stack_size = nullptr;
        uint32_t *stack = nullptr;  // Note : this represents NEURAL_STACK_SIZE * num_rays

        // TLAS Stack data
        int *tlas_stack_size = nullptr;
        uint32_t *tlas_stack = nullptr;
        uint32_t *node_idx   = nullptr;
    };

    struct NormalTraversalData : CudaBVHTraversalData {
        // Default constructor to make compiler happy with unique_ptr to types containing NeuralTraversalData
        NormalTraversalData() {}
        // Initialises a NeuralTraversalData structure from device pointers
        NormalTraversalData(vec3_soa ray_o,
                            vec3_soa ray_d,
                            uint8_t *ray_depth,
                            float *last_tlas_t,
                            uint8_t *last_valid_hit,
                            glm::vec3 *normal_or_uv,
                            glm::vec3 *albedo_or_mesh_prim_id,
                            uint32_t *current_blas_idx,
                            uint32_t *pixel_coord_active,
                            int *tlas_stack_size,
                            uint32_t *tlas_stack,
                            glm::vec3 *illumination,
                            glm::vec4 *throughput_rng)
        {
            this->ray_o     = ray_o;
            this->ray_d     = ray_d;
            this->ray_depth = ray_depth;

            this->last_valid_hit         = last_valid_hit;
            this->normal_or_uv           = normal_or_uv;
            this->albedo_or_mesh_prim_id = albedo_or_mesh_prim_id;

            this->current_blas_idx   = current_blas_idx;
            this->pixel_coord_active = pixel_coord_active;

            this->tlas_stack_size = tlas_stack_size;
            this->tlas_stack      = tlas_stack;
            this->last_tlas_t     = last_tlas_t;

            this->illumination   = illumination;
            this->throughput_rng = throughput_rng;
        }

        NTWR_DEVICE void init_from_tlas_traversal(uint32_t index,
                                                  uint32_t pixel_coord,
                                                  uint32_t blas_idx,
                                                  int local_tlas_stack_size,
                                                  uint32_t *local_tlas_stack,
                                                  float current_t,
                                                  uint8_t current_valid_hit,
                                                  glm::vec3 *current_normal_or_uv,
                                                  glm::vec3 *current_albedo_or_mesh_prim_id,
                                                  const Ray3f &ray,
                                                  uint8_t depth,
                                                  const LCGRand &rng,
                                                  glm::vec3 illum,
                                                  glm::vec3 throughput)
        {
            // Set Ray data
            ray_o.x[index] = ray.o.x;
            ray_o.y[index] = ray.o.y;
            ray_o.z[index] = ray.o.z;

            ray_d.x[index] = ray.d.x;
            ray_d.y[index] = ray.d.y;
            ray_d.z[index] = ray.d.z;

            ray_depth[index] = depth;

            current_blas_idx[index] = blas_idx;
            last_tlas_t[index]      = current_t;
            last_valid_hit[index]   = current_valid_hit;

            if (current_normal_or_uv)
                normal_or_uv[index] = *current_normal_or_uv;
            if (current_albedo_or_mesh_prim_id)
                albedo_or_mesh_prim_id[index] = *current_albedo_or_mesh_prim_id;

            uint32_t dst_index;
            for (int stack_idx = 0; stack_idx < local_tlas_stack_size; stack_idx++) {
                dst_index             = index + stack_idx * fb_size_flat;
                tlas_stack[dst_index] = local_tlas_stack[stack_idx];
            }
            tlas_stack_size[index] = local_tlas_stack_size;

            init_pixel_coord(index, pixel_coord);

            illumination[index] = illum;
            throughput_rng[index] =
                glm::vec4(throughput.x, throughput.y, throughput.z, glm::uintBitsToFloat(rng.state));
        }

        /* Hybrid BVH Traversal Data */
        float *last_tlas_t;
        uint8_t *last_valid_hit;  // required to keep track wether the last hit was neural or not
        uint32_t *current_blas_idx;
        glm::vec3 *normal_or_uv;
        glm::vec3 *albedo_or_mesh_prim_id;

        // TLAS Stack data
        int *tlas_stack_size;
        uint32_t *tlas_stack;
    };
}
}