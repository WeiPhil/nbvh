#pragma once

#include "cuda/base/material.cuh"
#include "cuda/base/ray.cuh"
#include "cuda/cuda_backend.h"
#include "cuda/utils/colors.cuh"
#include "cuda/utils/lcg_rng.cuh"
#include "cuda/utils/sampling.cuh"
#include "cuda/utils/types.cuh"

#include "cuda/neural/neural_bvh_module_constants.cuh"
#include "cuda/neural/neural_bvh_renderer_constants.cuh"
#include "cuda/neural/neural_bvh_renderer_kernels.cuh"

namespace ntwr {

namespace neural {

    //////////////////////////////////////////
    //////////// Inference related ////////////
    //////////////////////////////////////////

    NTWR_DEVICE float decode_depth_estimate(uint32_t ray_idx,
                                            const NeuralTraversalData &traversal_data,
                                            const float &segment_mint,
                                            const float &segment_maxt,
                                            float *pt_output_data,
                                            uint32_t output_idx,
                                            bool negative_mint,
                                            const NeuralBVH &neural_bvh)
    {
        float current_depth_value = FLT_MAX;
        switch (pt_target_const.depth_learning_mode) {
        case DepthLearningMode::LocalT: {
            float segment_length = segment_maxt - segment_mint;

            float unit_depth_output =
                glm::clamp(pt_target_const.output_depth_linear_activation ? (pt_output_data[output_idx] - 0.45f) * 10.f
                                                                          : pt_output_data[output_idx],
                           0.f,
                           1.f);
            if (negative_mint) {
                assert(segment_maxt >= 0.f);
                if (pt_target_const.depth_epsilon != 1.f) {
                    const float inward_depth_estimate = unit_depth_output * (segment_length + 1e-3);
                    if (inward_depth_estimate < segment_maxt - segment_length * (pt_target_const.depth_epsilon) -
                                                    pt_target_const.fixed_depth_epsilon) {
                        // Compute the current depth value
                        current_depth_value = (segment_maxt - inward_depth_estimate);
                    }
                }
            } else {
                current_depth_value = unit_depth_output * (segment_length + 1e-3) + segment_mint;
            }

            break;
        }
        case DepthLearningMode::WorldT: {
            float scene_mint;
            float scene_maxt;
            const Ray3f ray = traversal_data.get_ray(ray_idx);
            ray.intersect_bbox_min_max(neural_scene_bbox, scene_mint, scene_maxt);
            float segment_length = scene_maxt - scene_mint;

            float unit_depth_output =
                glm::clamp(pt_target_const.output_depth_linear_activation ? (pt_output_data[output_idx] - 0.45f) * 10.f
                                                                          : pt_output_data[output_idx],
                           0.f,
                           1.f);

            if (negative_mint) {
                if (pt_target_const.depth_epsilon != 1.f) {
                    const float inward_depth_estimate = unit_depth_output * (segment_length + 1e-3);

                    if (inward_depth_estimate < scene_maxt - segment_length * (pt_target_const.depth_epsilon) -
                                                    pt_target_const.fixed_depth_epsilon) {
                        // Compute the current depth value
                        current_depth_value = (scene_maxt - inward_depth_estimate);
                    }
                }
            } else {
                current_depth_value = unit_depth_output * (segment_length + 1e-3) + scene_mint;
            }
            break;
        }
        case DepthLearningMode::Position: {
            glm::vec3 ray_origin = traversal_data.ray_o.get(ray_idx);
            glm::vec3 ray_dir    = traversal_data.ray_d.get(ray_idx);

            glm::vec3 unit_pos_output =
                glm::vec3(pt_output_data[output_idx], pt_output_data[output_idx + 1], pt_output_data[output_idx + 2]);
            if (pt_target_const.output_depth_linear_activation) {
                unit_pos_output = (unit_pos_output - 0.45f) * 10.f;
            }

            glm::vec3 world_position = neural_scene_bbox.local_to_bbox(unit_pos_output);
            // project point onto ray

            glm::vec3 projected_point = glm::dot(world_position, ray_dir) * ray_dir;

            if (negative_mint) {
                if (pt_target_const.depth_epsilon != 1.f) {
                    float t_estimate = length(projected_point - ray_origin);

                    if (dot(ray_dir, projected_point - ray_origin) > 0.f &&
                        t_estimate > pt_target_const.depth_epsilon) {  // same direction? then we are occluded

                        // Compute the current depth value
                        current_depth_value = t_estimate;
                    }
                }
            } else {
                current_depth_value = length(world_position - ray_origin);
            }

            break;
        }
        case DepthLearningMode::LocalPosition: {
            glm::vec3 ray_origin = traversal_data.ray_o.get(ray_idx);
            glm::vec3 ray_dir    = traversal_data.ray_d.get(ray_idx);
            uint32_t node_idx    = traversal_data.node_idx[ray_idx];

            glm::vec3 unit_pos_output =
                glm::vec3(pt_output_data[output_idx], pt_output_data[output_idx + 1], pt_output_data[output_idx + 2]);
            if (pt_target_const.output_depth_linear_activation) {
                unit_pos_output = (unit_pos_output - 0.45f) * 10.f;
            }
            printf("unit_pos_output : (%g,%g,%g)",
                   pt_output_data[output_idx],
                   pt_output_data[output_idx + 1],
                   pt_output_data[output_idx + 2]);
            glm::vec3 world_position = neural_bvh.nodes[node_idx].local_to_bbox(unit_pos_output);
            // project point onto ray

            glm::vec3 projected_point = glm::dot(world_position, ray_dir) * ray_dir;

            if (negative_mint) {
                if (pt_target_const.depth_epsilon != 1.f) {
                    float t_estimate = length(projected_point - ray_origin);

                    if (dot(ray_dir, projected_point - ray_origin) > 0.f &&
                        t_estimate > pt_target_const.depth_epsilon) {  // same direction? then we are occluded

                        // Compute the current depth value
                        current_depth_value = t_estimate;
                    }
                }
            } else {
                current_depth_value = length(world_position - ray_origin);
            }

            break;
        }
        default:
            UNREACHABLE();
        }
        return current_depth_value;
    }

    NTWR_DEVICE bool neural_node_traversal(uint32_t ray_idx,
                                           float *__restrict__ pt_output_data,
                                           const NeuralTraversalData &traversal_data,
                                           float &depth_estimate,
                                           bool visibility_query_only,
                                           const NeuralBVH &neural_bvh)
    {
        assert(traversal_data.path_active(ray_idx));
        uint32_t output_idx = ray_idx * pt_target_const.output_network_dims;

        // Get current node's visibility inference first
        float visibility_threshold = traversal_data.depth(ray_idx) == 1
                                         ? pt_target_const.primary_visibility_threshold
                                         : pt_target_const.secondary_visibility_threshold;
        bool current_node_occluded = pt_output_data[output_idx] > visibility_threshold;
        output_idx += pt_target_const.output_visibility_dim;

        if (!current_node_occluded) {
            // Depth at infinity
            depth_estimate = FLT_MAX;
            return false;
        }

        assert(current_node_occluded);

        float segment_mint = traversal_data.leaf_mint[ray_idx];
        float segment_maxt = traversal_data.leaf_maxt[ray_idx];

        if (pt_target_const.output_depth_dim == 0) {
            assert(visibility_query_only);
            // Fallback if no depth estimate
            depth_estimate = 0.5f * (segment_maxt - segment_mint) + segment_mint;
            return true;
        }

        // Only set the visibility to 1 if occluded, do NOT set to 0 as we might override a previous valid
        // value
        bool inside_node = segment_mint < 0.f;

        // We don't need to check the depth estimate in that case
        if (!inside_node && visibility_query_only) {
            depth_estimate = FLT_MAX;
            return true;
        }

        depth_estimate = decode_depth_estimate(
            ray_idx, traversal_data, segment_mint, segment_maxt, pt_output_data, output_idx, inside_node, neural_bvh);

        return depth_estimate != FLT_MAX;
    }

    /* This function needs the following callback signature
      using InitNeuralTraversalResult = void (*)(uint32_t ray_idx,
                                                 NeuralTraversalResult &,
                                                 uint8_t current_valid_hit = 0,
                                                 float current_t           = FLT_MAX);
      using BackgroundHitResult = void (*)(const Ray3f &ray);
  */
    template <typename NeuralTraversalResult, typename InitNeuralTraversalResult, typename BackgroundHitResult>
    NTWR_DEVICE void camera_raygen_and_intersect_neural_bvh(uint32_t ray_idx,
                                                            int sample_idx,
                                                            int rnd_sample_offset,
                                                            NeuralTraversalData &traversal_data,
                                                            NeuralTraversalResult &results,
                                                            const NeuralBVH &neural_bvh,
                                                            InitNeuralTraversalResult init_neural_query_result,
                                                            BackgroundHitResult background_hit_result,
                                                            uint32_t *active_counter,
                                                            bool init_path_data)
    {
        glm::uvec2 pixel_coord = glm::uvec2(ray_idx % fb_size.x, ray_idx / fb_size.x);

        LCGRand rng = get_lcg_rng(sample_idx, rnd_sample_offset, ray_idx);
        float xf    = (float(pixel_coord.x) + lcg_randomf(rng)) / float(fb_size.x);
        float yf    = (float(pixel_coord.y) + lcg_randomf(rng)) / float(fb_size.y);

        Ray3f ray = camera_raygen(xf, yf);

        // Initially point to root node (always index 0)
        uint32_t local_stack[NEURAL_STACK_SIZE];
        local_stack[0] = 0;
        float leaf_mint[1];
        float leaf_maxt[1];
        int local_stack_size = 1;

        uint32_t neural_node_idx[1];
        uint32_t bvh_lod =
            bvh_const.path_lod_switch && bvh_const.num_lods_learned > 0 && traversal_data.depth(ray_idx) > 1
                ? bvh_const.lod_switch
                : bvh_const.rendered_lod;

        // For the BVH2 the 0 is actually important, it indexes into the correct leaf_mint and leaf_maxt
        uint8_t nodes_intersected = neural_bvh.intersect(bvh_lod,
                                                         ray,
                                                         0u,
                                                         local_stack,
                                                         local_stack_size,
                                                         true,
                                                         leaf_mint,
                                                         leaf_maxt,
                                                         true,
                                                         FLT_MAX,
                                                         neural_node_idx);

        if (nodes_intersected == 0 && local_stack_size == 0) {
            // Ray escaped
            glm::vec3 background_eval = background_hit_result(ray);
            accumulate_pixel_value(sample_idx, pixel_coord, background_eval, 1.0f);
        } else {
            uint32_t idx = atomicAdd(active_counter, 1);

            traversal_data.init_from_first_traversal_data(idx,
                                                          ray_idx,
                                                          local_stack_size,
                                                          local_stack,
                                                          nodes_intersected,
                                                          leaf_mint,
                                                          leaf_maxt,
                                                          ray,
                                                          0,
                                                          rng,
                                                          init_path_data,
                                                          0,
                                                          nullptr,
                                                          neural_node_idx[0]);

            init_neural_query_result(idx, results, uint8_t(0), FLT_MAX, nullptr, nullptr);
        }
    }

    template <typename NeuralTraversalResult, typename InitNeuralTraversalResult>
    NTWR_DEVICE bool tlas_traversal_and_fill_next_hybrid_traversal(int sample_idx,
                                                                   uint32_t pixel_coord_1d,
                                                                   const Ray3f &ray,
                                                                   uint8_t depth,
                                                                   float current_t,
                                                                   uint8_t current_valid_hit,
                                                                   glm::vec3 *current_normal_or_uv,
                                                                   glm::vec3 *current_albedo_or_mesh_prim_id,
                                                                   glm::vec3 illumination,
                                                                   glm::vec3 throughput,
                                                                   LCGRand rng,
                                                                   uint32_t *local_tlas_stack,
                                                                   int &local_tlas_stack_size,
                                                                   NormalTraversalData &normal_traversal_data,
                                                                   const NeuralBVH &neural_bvh,
                                                                   NeuralTraversalData &neural_traversal_data,
                                                                   NeuralTraversalResult &neural_results,
                                                                   const CudaSceneData &scene_data,
                                                                   uint32_t *active_neural_counter,
                                                                   uint32_t *active_normal_counter,
                                                                   InitNeuralTraversalResult init_neural_query_result)
    {
        uint32_t blas_idx;
        bool is_neural_blas;
        // For the BVH2 the 0 is actually important, it indexes into the correct leaf_mint and leaf_maxt
        bool found_blas = scene_data.tlas.intersect_tlas_only(
            ray, 0u, blas_idx, is_neural_blas, local_tlas_stack, local_tlas_stack_size, true);

        if (!found_blas) {
            return true;
        }

        if (is_neural_blas && !bvh_const.ignore_neural_blas) {
            // Initially point to root node (always index 0)
            uint32_t local_stack[NEURAL_STACK_SIZE];
            local_stack[0] = 0;
            float leaf_mint[1];
            float leaf_maxt[1];

            int local_stack_size = 1;

            uint32_t neural_node_idx[1];
            uint32_t bvh_lod = bvh_const.path_lod_switch && bvh_const.num_lods_learned > 0 && depth > 1
                                   ? bvh_const.lod_switch
                                   : bvh_const.rendered_lod;

            // For the BVH2 the 0 is actually important, it indexes into the correct leaf_mint and leaf_maxt
            uint8_t nodes_intersected = neural_bvh.intersect(bvh_lod,
                                                             ray,
                                                             0u,
                                                             local_stack,
                                                             local_stack_size,
                                                             true,
                                                             leaf_mint,
                                                             leaf_maxt,
                                                             true,
                                                             FLT_MAX,
                                                             neural_node_idx);

            if (nodes_intersected > 0) {
                // The only case where the traversal is neural
                uint32_t neural_idx = atomicAdd(active_neural_counter, 1);

                neural_traversal_data.init_from_first_traversal_data(neural_idx,
                                                                     pixel_coord_1d,
                                                                     local_stack_size,
                                                                     local_stack,
                                                                     nodes_intersected,
                                                                     leaf_mint,
                                                                     leaf_maxt,
                                                                     ray,
                                                                     depth,
                                                                     rng,
                                                                     false,
                                                                     local_tlas_stack_size,
                                                                     local_tlas_stack,
                                                                     neural_node_idx[0]);
                neural_traversal_data.illumination[neural_idx] = illumination;
                neural_traversal_data.throughput_rng[neural_idx] =
                    glm::vec4(throughput.x, throughput.y, throughput.z, glm::uintBitsToFloat(rng.state));
                neural_traversal_data.ray_depth[neural_idx] = depth;

                init_neural_query_result(neural_idx,
                                         neural_results,
                                         current_valid_hit,
                                         current_t,
                                         current_normal_or_uv,
                                         current_albedo_or_mesh_prim_id);
                return false;
            }

            assert(local_stack_size == 0);

            if (local_tlas_stack_size == 0) {
                return true;
            }

            assert(local_tlas_stack_size != 0);

            // Otherwise we get next tlas intersection instead
            found_blas = scene_data.tlas.intersect_tlas_only(
                ray, 0u, blas_idx, is_neural_blas, local_tlas_stack, local_tlas_stack_size, true);

            if (!found_blas) {
                return true;
            }

            // If we support more than 1 neural asset could be true
            assert(!is_neural_blas);
        }

        uint32_t normal_idx = atomicAdd(active_normal_counter, 1);

        normal_traversal_data.init_from_tlas_traversal(normal_idx,
                                                       pixel_coord_1d,
                                                       blas_idx,
                                                       local_tlas_stack_size,
                                                       local_tlas_stack,
                                                       current_t,
                                                       current_valid_hit,
                                                       current_normal_or_uv,
                                                       current_albedo_or_mesh_prim_id,
                                                       ray,
                                                       depth,
                                                       rng,
                                                       illumination,
                                                       throughput);

        return false;
    }

    /* This function needs the following callback signature
    using InitNeuralTraversalResult = void (*)(uint32_t ray_idx, NeuralTraversalResult &results);
    using BackgroundHitResult = void (*)(const Ray3f &ray);
    */
    template <typename NeuralTraversalResult, typename InitNeuralTraversalResult, typename BackgroundHitResult>
    NTWR_DEVICE void hybrid_camera_raygen_and_intersect_neural_bvh(uint32_t ray_idx,
                                                                   int sample_idx,
                                                                   int rnd_sample_offset,
                                                                   NormalTraversalData &normal_traversal_data,
                                                                   const NeuralBVH &neural_bvh,
                                                                   NeuralTraversalData &neural_traversal_data,
                                                                   NeuralTraversalResult &neural_results,
                                                                   const CudaSceneData &scene_data,
                                                                   uint32_t *active_neural_counter,
                                                                   uint32_t *active_normal_counter,
                                                                   InitNeuralTraversalResult init_neural_query_result,
                                                                   BackgroundHitResult background_hit_result)
    {
        glm::uvec2 pixel_coord = glm::uvec2(ray_idx % fb_size.x, ray_idx / fb_size.x);

        LCGRand rng = get_lcg_rng(sample_idx, rnd_sample_offset, ray_idx);
        float xf    = (float(pixel_coord.x) + lcg_randomf(rng)) / float(fb_size.x);
        float yf    = (float(pixel_coord.y) + lcg_randomf(rng)) / float(fb_size.y);

        Ray3f ray         = camera_raygen(xf, yf);
        uint8_t ray_depth = 1;

        glm::vec3 illumination = glm::vec3(0.f);
        glm::vec3 throughput   = glm::vec3(1.f);

        uint32_t local_tlas_stack[TLAS_STACK_SIZE];
        local_tlas_stack[0]       = 0;
        int local_tlas_stack_size = 1;

        bool tlas_traversal_done =
            tlas_traversal_and_fill_next_hybrid_traversal(sample_idx,
                                                          ray_idx,  // for the first traversal this is just the ray_idx
                                                          ray,
                                                          ray_depth,
                                                          FLT_MAX,
                                                          0u,
                                                          nullptr,
                                                          nullptr,
                                                          illumination,
                                                          throughput,
                                                          rng,
                                                          local_tlas_stack,
                                                          local_tlas_stack_size,
                                                          normal_traversal_data,
                                                          neural_bvh,
                                                          neural_traversal_data,
                                                          neural_results,
                                                          scene_data,
                                                          active_neural_counter,
                                                          active_normal_counter,
                                                          init_neural_query_result);

        if (tlas_traversal_done) {
            glm::vec3 background_eval = background_hit_result(ray);
            illumination += throughput * background_eval;
            accumulate_pixel_value(sample_idx, pixel_coord, illumination, 1.0f);
            return;
        }
    }

    // Should return whether to continue or stop the traversal, optionally a depth estimate can be computed
    /* This function needs the following callback signature
        using ProcessNeuralBvhTraversalStep = bool (*)(uint32_t ray_idx,
                                                        uint32_t traversal_step,
                                                        float *__restrict__ visibility_output_data,
                                                        NeuralTraversalResult &results,
                                                        NeuralTraversalData &traversal_data,
                                                        float &t_neural);
    */
    template <typename NeuralTraversalResult, typename ProcessNeuralBvhTraversalStep>
    NTWR_DEVICE void process_bvh_network_output_and_intersect_neural_bvh(
        uint32_t ray_idx,
        uint32_t traversal_step,
        float *__restrict__ visibility_output_data,
        NeuralTraversalData &traversal_data,
        NeuralTraversalResult &results,
        const NeuralBVH &neural_bvh,
        ProcessNeuralBvhTraversalStep process_neural_bvh_traversal_step)
    {
        assert(!traversal_data.all_nodes_processed(ray_idx));

        float depth_estimate;
        bool any_hit_only = process_neural_bvh_traversal_step(
            ray_idx, traversal_step, visibility_output_data, results, traversal_data, depth_estimate, neural_bvh);

        // Indicate the current node as proceseed
        traversal_data.increment_inferenced_node_idx(ray_idx);

        // Any hit only?
        if (any_hit_only) {
            traversal_data.set_path_inactive(ray_idx);
            return;
        }

        assert(traversal_data.path_active(ray_idx));

        uint32_t bvh_lod =
            bvh_const.path_lod_switch && bvh_const.num_lods_learned > 0 && traversal_data.depth(ray_idx) > 1
                ? bvh_const.lod_switch
                : bvh_const.rendered_lod;

        // If the stack is not empty and there are no more nodes to process get the next batch of nodes
        if (traversal_data.all_nodes_processed(ray_idx) && traversal_data.stack_size[ray_idx] != 0) {
            uint32_t neural_node_idx[1];
            uint8_t nodes_intersected = neural_bvh.intersect(bvh_lod,
                                                             traversal_data.get_ray(ray_idx),
                                                             ray_idx,
                                                             traversal_data.stack,
                                                             traversal_data.stack_size[ray_idx],
                                                             false,
                                                             traversal_data.leaf_mint,
                                                             traversal_data.leaf_maxt,
                                                             true,
                                                             depth_estimate,
                                                             neural_node_idx);
            traversal_data.set_intersected_nodes(ray_idx, nodes_intersected);

            traversal_data.node_idx[ray_idx] = neural_node_idx[0];
        }

        // If we processed all nodes AND the stack is empty we can safely make the path inactive
        if (traversal_data.stack_size[ray_idx] == 0 && traversal_data.all_nodes_processed(ray_idx))
            traversal_data.set_path_inactive(ray_idx);
    }

    // Should return the current pixel value to accumulate

    /* This function needs the following callback signature
        using AccumulateNeuralResult = glm::vec3 (*)(uint32_t ray_idx,
                                                 const NeuralTraversalResult &results,
                                                 const NeuralTraversalData &traversal_data);
    */
    template <typename NeuralTraversalResult, typename AccumulateNeuralResult>
    NTWR_DEVICE void compact_neural_data_and_accumulate(uint32_t ray_idx,
                                                        int sample_idx,
                                                        const NeuralTraversalData &src_traversal_data,
                                                        const NeuralTraversalResult &src_results,
                                                        NeuralTraversalData &dst_traversal_data,
                                                        NeuralTraversalResult &dst_results,
                                                        uint32_t *active_counter,
                                                        bool last_inference,
                                                        AccumulateNeuralResult accumulate_neural_result)
    {
        if (src_traversal_data.path_active(ray_idx) && !last_inference) {
            uint32_t idx = atomicAdd(active_counter, 1);
            dst_traversal_data.set(idx, src_traversal_data, ray_idx);
            dst_results.set(idx, src_results, ray_idx);
        } else {
            glm::uvec2 pixel_coord;
            pixel_coord.x = src_traversal_data.pixel_coord(ray_idx) % fb_size.x;
            pixel_coord.y = src_traversal_data.pixel_coord(ray_idx) / fb_size.x;

            glm::vec3 pixel_value = accumulate_neural_result(ray_idx, src_results, src_traversal_data);

            // Splat contribution if an inactive path
            accumulate_pixel_value(sample_idx, pixel_coord, pixel_value, 1.0f);
        }
    }

    NTWR_DEVICE glm::vec3 compute_depth_or_position_output(
        const Ray3f &ray, float its_t, float leaf_mint, float leaf_maxt, uint32_t node_idx, const CudaBLAS &neural_blas)
    {
        glm::vec3 depth_pos;
        switch (pt_target_const.depth_learning_mode) {
        case DepthLearningMode::LocalT: {
            float segment_length = (leaf_maxt - leaf_mint);
            float depth          = (its_t - leaf_mint) / (segment_length + 1e-3);

            assert(depth >= 0.f && depth <= 1.f);
            depth_pos = glm::vec3(depth);
            break;
        }
        case DepthLearningMode::WorldT: {
            float mint, maxt;
            if (!ray.intersect_bbox_min_max(neural_scene_bbox, mint, maxt)) {
                assert(false);
            }
            float segment_length = (maxt - mint);
            float depth          = (its_t - mint) / (segment_length + 1e-3);
            assert(depth >= 0.f && depth <= 1.f);
            depth_pos = glm::vec3(depth);
            break;
        }
        case DepthLearningMode::Position: {
            depth_pos = neural_scene_bbox.bbox_to_local(ray(its_t));
            break;
        }
        case DepthLearningMode::LocalPosition: {
            depth_pos = neural_blas.nodes[node_idx].bbox_to_local(ray(its_t));
            break;
        }
        default:
            UNREACHABLE();
        }

        if (pt_target_const.output_depth_linear_activation) {
            // forcing fake linear activation by first order approximation of sigmoid activation around 0
            // mapping depth_pos in [0.45,0.55]
            depth_pos = depth_pos * 0.1f + 0.45f;
        }

        return depth_pos;
    }

    //////////////////////////////////////////
    //////////// Learning related ////////////
    //////////////////////////////////////////

    NTWR_DEVICE void sample_bbox_position(const NeuralBVH2Node &node,
                                          LCGRand &rng,
                                          float sampled_dilation,
                                          glm::vec3 &bbox_entry_pos,
                                          glm::vec3 &bbox_exit_pos,
                                          bool force_different_face = false)
    {
        int face1_index = glm::min(int(lcg_randomf(rng) * 6), 5);
        int face2_index;
        do {
            face2_index = glm::min(int(lcg_randomf(rng) * 6), 5);
        } while (force_different_face || face1_index == face2_index);

        assert(face1_index < 6 && face1_index >= 0);
        assert(face2_index < 6 && face2_index >= 0);

        // In [0,1]
        glm::vec3 local_pos_1 = get_cube_pos_on_face(face1_index, lcg_randomf(rng), lcg_randomf(rng));
        glm::vec3 local_pos_2 = get_cube_pos_on_face(face2_index, lcg_randomf(rng), lcg_randomf(rng));

        const glm::vec3 node_extent = node.bbox_extent() * (1.f + sampled_dilation);
        const glm::vec3 node_min    = node.bbox_center() - node_extent * 0.5f;

        bbox_entry_pos = node_min + (local_pos_1 * node_extent);
        bbox_exit_pos  = node_min + (local_pos_2 * node_extent);
    }

    NTWR_DEVICE void sample_uniform_bvh_density(const NeuralBVH &neural_bvh,
                                                LCGRand &rng,
                                                Ray3f &ray,
                                                float &leaf_mint,
                                                float &leaf_maxt,
                                                uint32_t *sampled_node_index)
    {
        glm::vec3 point_sample =
            bvh_const.sampling_bbox_scaling * lcg_random3f(rng) - (bvh_const.sampling_bbox_scaling - 1.f) * 0.5f;
        bool valid_its = false;

        uint32_t current_num_lods_learned = (NUM_BVH_LODS - bvh_const.current_learned_lod);
        uint32_t sampled_lod              = bvh_const.current_learned_lod +
                               min(uint32_t(lcg_randomf(rng) * current_num_lods_learned), current_num_lods_learned - 1);

        // Using (ratio <= sample) so we can essentially switch completely to leaf boundary mode if desired
        bool sample_leaf_boundary =
            bvh_const.leaf_boundary_sampling_ratio != 0.f && lcg_randomf(rng) <= bvh_const.leaf_boundary_sampling_ratio;

        float sampled_dilation = bvh_const.fixed_node_dilation + lcg_randomf(rng) * bvh_const.node_extent_dilation;
        while (!valid_its) {
            glm::vec3 dir_sample = square_to_uniform_sphere(lcg_random2f(rng));

            ray = Ray3f(neural_scene_bbox.local_to_bbox(point_sample), dir_sample);

            valid_its = neural_bvh.intersect_closest_node(
                sampled_lod, ray, leaf_mint, leaf_maxt, sampled_node_index, sampled_dilation);
        }
        assert(*sampled_node_index < neural_bvh.num_nodes);

        if (sample_leaf_boundary) {
            assert(!bvh_const.ellipsoidal_bvh);
            glm::vec3 entry, exit;
            sample_bbox_position(neural_bvh.nodes[*sampled_node_index], rng, sampled_dilation, entry, exit);
            ray.o     = entry;
            ray.d     = normalize(exit - entry);
            leaf_mint = 0.f;
            leaf_maxt = length(exit - entry);
        }
    }

    NTWR_DEVICE bool stack_based_sample_uniform_bvh_density(const NeuralBVH2 &neural_bvh,
                                                            uint32_t *local_stack,
                                                            int &local_stack_size,
                                                            LCGRand &rng,
                                                            Ray3f &ray,
                                                            float *leaf_mint,
                                                            float *leaf_maxt,
                                                            uint32_t *sampled_node_index,
                                                            bool continue_traversal)
    {
        glm::vec3 dir_sample = square_to_uniform_sphere(lcg_random2f(rng));

        // only case where valid_its should be true is if we should continue the traversal
        bool valid_its          = continue_traversal && local_stack_size != 0;
        bool restarted_sampling = false;

        uint32_t current_num_lods_learned = (NUM_BVH_LODS - bvh_const.current_learned_lod);
        uint32_t sampled_lod              = bvh_const.current_learned_lod +
                               min(uint32_t(lcg_randomf(rng) * current_num_lods_learned), current_num_lods_learned - 1);

        // We always add a little base dilation
        float sampled_dilation = bvh_const.fixed_node_dilation + lcg_randomf(rng) * bvh_const.node_extent_dilation;

        int failed_sampling_attempts = 0;

        do {
            // if no intersection was found the ray escaped and we restart the sampling of a new ray
            if (!valid_its) {
                // reinitialise stack and sample new ray origin (keep same sampled direction)
                local_stack[0]   = 0;
                local_stack_size = 1;
                if (bvh_const.view_dependent_sampling && failed_sampling_attempts < 5 &&
                    lcg_randomf(rng) < bvh_const.view_dependent_ratio) {
                    ray = camera_raygen(lcg_randomf(rng), lcg_randomf(rng));
                } else {
                    glm::vec3 point_sample = bvh_const.sampling_bbox_scaling * lcg_random3f(rng) -
                                             (bvh_const.sampling_bbox_scaling - 1.f) * 0.5f;
                    ray = Ray3f(neural_scene_bbox.local_to_bbox(point_sample), dir_sample);
                }
                restarted_sampling = true;
            }

            // For the BVH2 the 0 is actually important, it indexes into the correct leaf_mint and leaf_maxt
            valid_its = neural_bvh.intersect(sampled_lod,
                                             ray,
                                             0u,
                                             local_stack,
                                             local_stack_size,
                                             true,
                                             leaf_mint,
                                             leaf_maxt,
                                             false,
                                             FLT_MAX,
                                             sampled_node_index,
                                             sampled_dilation);

            if (!valid_its) {
                failed_sampling_attempts++;
            }

        } while (!valid_its);

        bool sample_leaf_boundary =
            bvh_const.leaf_boundary_sampling_ratio != 0.f && lcg_randomf(rng) <= bvh_const.leaf_boundary_sampling_ratio;

        if (sample_leaf_boundary) {
            assert(!bvh_const.ellipsoidal_bvh);
            glm::vec3 entry, exit;
            sample_bbox_position(neural_bvh.nodes[*sampled_node_index], rng, sampled_dilation, entry, exit);
            ray.o              = entry;
            ray.d              = normalize(exit - entry);
            *leaf_mint         = 0.f;
            *leaf_maxt         = length(exit - entry);
            restarted_sampling = true;
        }

        assert(*sampled_node_index < neural_bvh.num_nodes);
        return restarted_sampling;
    }

    NTWR_DEVICE PreliminaryItsData generate_single_bvh_network_output_reference(const Ray3f &_ray,
                                                                                float leaf_mint,
                                                                                float leaf_maxt,
                                                                                const CudaSceneData &scene_data,
                                                                                bool swap_trained_direction,
                                                                                float *__restrict__ single_target,
                                                                                uint32_t node_idx)
    {
        uint32_t output_idx = 0;

        Ray3f ray;
        if (!swap_trained_direction) {
            ray = Ray3f(_ray.o, _ray.d, leaf_mint, leaf_maxt);
        } else {
            ray.o     = _ray.o;
            ray.d     = -_ray.d;
            ray.mint  = -leaf_maxt;
            ray.maxt  = -leaf_mint;
            leaf_mint = ray.mint;
            leaf_maxt = ray.maxt;
        }

        PreliminaryItsData pi;
        const CudaBLAS &neural_blas = scene_data.tlas.mesh_bvhs[bvh_const.neural_bvh_idx];
        // Shadow rays when only learning visibility
        if (pt_target_const.output_visibility_dim == pt_target_const.output_network_dims)
            pi = neural_blas.intersect<true>(ray);
        else
            pi = neural_blas.intersect<false>(ray);
        pi.mesh_idx = bvh_const.neural_bvh_idx;

        single_target[output_idx] = float(pi.is_valid() ? 1 : 0);
        output_idx += pt_target_const.output_visibility_dim;

        if (pt_target_const.output_depth_dim > 0) {
            const glm::vec3 pos_depth =
                pi.is_valid() ? compute_depth_or_position_output(ray, pi.t, leaf_mint, leaf_maxt, node_idx, neural_blas)
                              : glm::vec3(0.f);

            if (pt_target_const.depth_learning_mode != DepthLearningMode::Position &&
                pt_target_const.depth_learning_mode != DepthLearningMode::LocalPosition) {
                single_target[output_idx] = pos_depth.x;
            } else {
                single_target[output_idx]     = pos_depth.x;
                single_target[output_idx + 1] = pos_depth.y;
                single_target[output_idx + 2] = pos_depth.z;
            }
            output_idx += pt_target_const.output_depth_dim;
        }
        if (pt_target_const.output_normal_dim > 0) {
            glm::vec3 normal = glm::vec3(0.f);
            if (pi.is_valid()) {
                ItsData its = neural_blas.compute_its_data(pi, scene_data.materials, scene_data.textures);

                // Learn normals remapped to [0,1]
                normal = (its.sh_normal + 1.f) * 0.5f;
            }
            single_target[output_idx]     = normal.x;
            single_target[output_idx + 1] = normal.y;
            single_target[output_idx + 2] = normal.z;
            output_idx += pt_target_const.output_normal_dim;
        }
        if (pt_target_const.output_albedo_dim > 0) {
            glm::vec3 albedo = glm::vec3(0.f);
            if (pi.is_valid()) {
                ItsData its = neural_blas.compute_its_data(pi, scene_data.materials, scene_data.textures);

                albedo = CudaMaterial{}.base_color;
                if (its.material_id != -1)
                    albedo = unpack_albedo(scene_data.materials[its.material_id], its.uv, scene_data.textures);
            }
            single_target[output_idx]     = albedo.x;
            single_target[output_idx + 1] = albedo.y;
            single_target[output_idx + 2] = albedo.z;
            output_idx += pt_target_const.output_albedo_dim;
        }

        return pi;
    }

}
}