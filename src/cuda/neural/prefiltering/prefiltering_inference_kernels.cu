

#include "cuda/cuda_backend_constants.cuh"
#include "cuda/utils/rendering.cuh"

#include "cuda/neural/neural_bvh_module_constants.cuh"
#include "cuda/neural/neural_bvh_module_kernels.cuh"
#include "cuda/neural/prefiltering/prefiltering_constants.cuh"
#include "cuda/neural/prefiltering/prefiltering_inference_kernels.cuh"

#include "cuda/base/material.cuh"
#include "cuda/utils/sampling.cuh"

namespace ntwr {
namespace neural::prefiltering {

    NTWR_DEVICE void init_neural_prefiltering_result(uint32_t idx,
                                                     NeuralTraversalResult &results,
                                                     uint8_t current_valid_hit,
                                                     float current_t,
                                                     glm::vec3 *,  // extra data
                                                     glm::vec3 *)  // extra data
    {
        // Initialize to unoccluded
        results.valid_hit[idx] = current_valid_hit;
        // The depth result also stores the ray maxt
        results.t_neural[idx] = current_t;
    }

    NTWR_DEVICE glm::vec3 background_hit_eval(const Ray3f &ray)
    {
// #define CHEAP_SKY_LIKE_BACKGROUND
#ifdef CHEAP_SKY_LIKE_BACKGROUND
        const float a = 0.5f * (ray.d.y + 1.f);
        return (1.f - a) * glm::vec3(1.0f) + a * glm::vec3(0.5f, 0.7f, 1.f);
#else
        return glm::vec3(1.0f);
#endif
    }

    NTWR_KERNEL void camera_raygen_and_intersect_neural_bvh(uint32_t n_elements,
                                                            int sample_idx,
                                                            int rnd_sample_offset,
                                                            NeuralTraversalData traversal_data,
                                                            NeuralTraversalResult results,
                                                            NeuralBVH neural_bvh,
                                                            uint32_t *active_counter)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n_elements)
            return;
        camera_raygen_and_intersect_neural_bvh(i,
                                               sample_idx,
                                               rnd_sample_offset,
                                               traversal_data,
                                               results,
                                               neural_bvh,
                                               init_neural_prefiltering_result,
                                               background_hit_eval,
                                               active_counter,
                                               true);
    }

    NTWR_DEVICE bool traversal_rr_terminate(float inference_value,
                                            uint32_t traversal_step,
                                            glm::vec3 &path_throughput,
                                            LCGRand &rng)
    {
        // Russian roulette termination
        if (traversal_step >= 5) {
            // uint32_t constexpr voxel_grid_max_res = 512;
            // float max_traversal_rr_prob           = std::pow(1e-6f, 1.f / (3 * (voxel_grid_max_res)));

            float survival_prob =
                prefiltering_module_const.traversal_rr;  // max(1.f - inference_value, max_traversal_rr_prob);
            float rr_sample = lcg_randomf(rng);
            if (rr_sample < survival_prob)
                path_throughput /= survival_prob;
            else
                return true;
        }

        return false;
    }

    NTWR_DEVICE bool process_neural_bvh_traversal_step(uint32_t ray_idx,
                                                       uint32_t traversal_step,
                                                       float *__restrict__ pt_output_data,
                                                       NeuralTraversalResult &results,
                                                       NeuralTraversalData &traversal_data,
                                                       float &new_depth_estimate,
                                                       const NeuralBVH &neural_bvh)
    {
        // First set again best distance so far
        new_depth_estimate = results.t_neural[ray_idx];

        const bool visibility_query_only = false;
        float depth_estimate;
        bool ray_hit = neural_node_traversal(
            ray_idx, pt_output_data, traversal_data, depth_estimate, visibility_query_only, neural_bvh);

        if (!ray_hit) {
            assert(depth_estimate == FLT_MAX);

            if (prefiltering_module_const.apply_traversal_rr) {
                glm::vec3 &throughput      = traversal_data.get_throughput(ray_idx);
                LCGRand rng                = traversal_data.get_rng(ray_idx);
                float visibility_inference = pt_output_data[ray_idx * pt_target_const.output_network_dims];
                bool terminate = traversal_rr_terminate(visibility_inference, traversal_step, throughput, rng);
                traversal_data.set_rng(ray_idx, rng);
                if (terminate) {
                    // Special value to indicate no hit and just splatting current contribution
                    results.valid_hit[ray_idx] = 2;
                    return true;
                }
            }

            return false;
        }

        assert(ray_hit);

        results.valid_hit[ray_idx] = 1;

        if (depth_estimate < results.t_neural[ray_idx]) {
            // Update with the new depth estimate
            new_depth_estimate        = depth_estimate;
            results.t_neural[ray_idx] = depth_estimate;
        }
        return false;
    }

    NTWR_KERNEL void process_bvh_network_output_and_intersect_neural_bvh(uint32_t n_elements,
                                                                         uint32_t traversal_step,
                                                                         float *__restrict__ visibility_output_data,
                                                                         NeuralTraversalData traversal_data,
                                                                         NeuralTraversalResult results,
                                                                         NeuralBVH neural_bvh)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n_elements)
            return;
        process_bvh_network_output_and_intersect_neural_bvh(i,
                                                            traversal_step,
                                                            visibility_output_data,
                                                            traversal_data,
                                                            results,
                                                            neural_bvh,
                                                            process_neural_bvh_traversal_step);
    }

    NTWR_KERNEL void compact_neural_data_and_accumulate(const uint32_t n_elements,
                                                        int sample_idx,
                                                        NeuralWavefrontData<NeuralTraversalResult> src_wavefront_data,
                                                        NeuralWavefrontData<NeuralTraversalResult> dst_wavefront_data,
                                                        HybridIntersectionResult intersection_result_data,
                                                        uint32_t *active_counter,
                                                        uint32_t *its_counter,
                                                        bool last_inference)
    {
        const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= n_elements)
            return;

        uint32_t ray_idx = i;

        const NeuralTraversalResult &src_results      = src_wavefront_data.results;
        const NeuralTraversalData &src_traversal_data = src_wavefront_data.traversal_data;

        NeuralTraversalResult &dst_results      = dst_wavefront_data.results;
        NeuralTraversalData &dst_traversal_data = dst_wavefront_data.traversal_data;

        if (src_traversal_data.path_active(ray_idx) && !last_inference) {
            assert(src_traversal_data.path_active(ray_idx));

            uint32_t idx = atomicAdd(active_counter, 1);

            dst_traversal_data.set(idx, src_traversal_data, ray_idx);
            dst_results.set(idx, src_results, ray_idx);

        } else {
            assert(!src_traversal_data.path_active(ray_idx) || last_inference);

            glm::uvec2 pixel_coord;
            pixel_coord.x = src_traversal_data.pixel_coord(ray_idx) % fb_size.x;
            pixel_coord.y = src_traversal_data.pixel_coord(ray_idx) / fb_size.x;

            if (src_results.valid_hit[ray_idx] == 1) {
                uint32_t its_idx = atomicAdd(its_counter, 1);

                switch (prefiltering_module_const.output_mode) {
                case OutputMode::NeuralPrefiltering: {
                    // Store
                    intersection_result_data.set(its_idx, src_results, src_traversal_data, ray_idx);
                    break;
                }
                default:
                    UNREACHABLE();
                }

            } else if (src_results.valid_hit[ray_idx] == 0) {
                // No intersection splat the result!

                // glm::vec3 background_eval = evaluate_background(ray_idx, src_results, src_traversal_data);
                glm::vec3 background_eval = background_hit_eval(src_traversal_data.get_ray(ray_idx));

                glm::vec3 final_illumination = src_traversal_data.illumination[ray_idx];
                final_illumination += glm::vec3(src_traversal_data.throughput_rng[ray_idx]) * background_eval;

                // Splat contribution if an inactive path
                accumulate_pixel_value(sample_idx, pixel_coord, final_illumination, 1.0f);
            } else if (src_results.valid_hit[ray_idx] == 2) {
                // Killed by russian roulette splat current estimate
                accumulate_pixel_value(sample_idx,
                                       pixel_coord,
#ifdef DEBUG_TRAVERSAL_RR
                                       glm::vec3(1, 0, 0)
#else
                                       src_traversal_data.illumination[ray_idx]
#endif
                                           ,
                                       1.0f);
            }
        }
    }

    template <typename InitNeuralTraversalResult>
    NTWR_DEVICE void neural_prefiltering_scatter(const uint32_t ray_idx,
                                                 int sample_idx,
                                                 uint32_t depth,
                                                 const HybridIntersectionResult &its_result_data,
                                                 NeuralTraversalData &traversal_data,
                                                 NeuralTraversalResult &results,
                                                 NeuralBVH &neural_bvh,
                                                 float *__restrict__ appearance_output_data,
                                                 uint32_t *active_counter,
                                                 InitNeuralTraversalResult init_neural_query_result)
    {
        assert(its_result_data.valid_hit[ray_idx] == 1);

        glm::vec3 illumination = its_result_data.illumination[ray_idx];
        glm::vec3 throughput   = its_result_data.get_throughput(ray_idx);
        LCGRand rng            = its_result_data.get_rng(ray_idx);
        // Get inferred data
        uint32_t output_idx            = ray_idx * prefiltering_module_const.prefiltering_output_dims;
        glm::vec3 throughput_inference = glm::vec3(appearance_output_data[output_idx + 0],
                                                   appearance_output_data[output_idx + 1],
                                                   appearance_output_data[output_idx + 2]);
        if (prefiltering_module_const.prefiltering_log_learnt) {
            throughput_inference.x = expf(throughput_inference.x) - 1.f;
            throughput_inference.y = expf(throughput_inference.y) - 1.f;
            throughput_inference.z = expf(throughput_inference.z) - 1.f;
        }
        throughput_inference = max(glm::vec3(0.f), throughput_inference);

        if (!prefiltering_module_const.hg_sampling) {
            throughput *= throughput_inference * square_to_uniform_sphere_inv_pdf();
        } else {
            assert(false);
            // // Incorrect direction right now since ray_d is changed
            // glm::vec3 wi = its_result_data.sampled_wi[ray_idx];
            // glm::vec3 wo = -its_result_data.ray_d.get(ray_idx);
            // throughput *= throughput_inference / square_to_hg_pdf(dot(wo, wi), prefiltering_module_const.hg_g);
        }

        bool terminate_path = false;
        if (prefiltering_module_const.apply_russian_roulette) {
            terminate_path = rr_terminate(depth, throughput, prefiltering_module_const.rr_start_depth, rng);
        }

        // If we terminate the path due to russian roulette
        if (terminate_path) {
            glm::uvec2 pixel_coord;
            pixel_coord.x = its_result_data.pixel_coord[ray_idx] % fb_size.x;
            pixel_coord.y = its_result_data.pixel_coord[ray_idx] / fb_size.x;

            accumulate_pixel_value(sample_idx, pixel_coord, illumination, 1.0f);
            return;
        }

        Ray3f sampled_ray = its_result_data.get_ray(ray_idx);
        uint8_t ray_depth = its_result_data.ray_depth[ray_idx];

        // Initially point to root node (always index 0)
        uint32_t local_stack[NEURAL_STACK_SIZE];
        local_stack[0] = 0;
        float leaf_mint[1];
        float leaf_maxt[1];
        int local_stack_size = 1;

        // For the BVH2 the 0 is actually important, it indexes into the correct leaf_mint and leaf_maxt
        uint8_t nodes_intersected = neural_bvh.intersect(
            bvh_const.rendered_lod, sampled_ray, 0u, local_stack, local_stack_size, true, leaf_mint, leaf_maxt, true);

        if (nodes_intersected == 0 && local_stack_size == 0) {
            // Ray escaped
            glm::uvec2 pixel_coord;
            pixel_coord.x = its_result_data.pixel_coord[ray_idx] % fb_size.x;
            pixel_coord.y = its_result_data.pixel_coord[ray_idx] / fb_size.x;

            glm::vec3 background_eval = background_hit_eval(sampled_ray);
            illumination += throughput * background_eval;
            accumulate_pixel_value(sample_idx, pixel_coord, illumination, 1.0f);
        } else {
            uint32_t idx = atomicAdd(active_counter, 1);

            traversal_data.init_from_first_traversal_data(idx,
                                                          its_result_data.pixel_coord[ray_idx],
                                                          local_stack_size,
                                                          local_stack,
                                                          nodes_intersected,
                                                          leaf_mint,
                                                          leaf_maxt,
                                                          sampled_ray,
                                                          ray_depth + 1,
                                                          rng);
            traversal_data.illumination[idx] = illumination;
            traversal_data.throughput_rng[idx] =
                glm::vec4(throughput.x, throughput.y, throughput.z, glm::uintBitsToFloat(rng.state));
            traversal_data.ray_depth[idx] = ray_depth + 1;

            init_neural_query_result(idx, results, uint8_t(0), FLT_MAX, nullptr, nullptr);
        }
    }

    NTWR_KERNEL void neural_prefiltering_scatter(const uint32_t n_elements,
                                                 int sample_idx,
                                                 uint32_t depth,
                                                 HybridIntersectionResult its_result_data,
                                                 NeuralTraversalData traversal_data,
                                                 NeuralTraversalResult results,
                                                 NeuralBVH neural_bvh,
                                                 float *__restrict__ appearance_output_data,
                                                 uint32_t *active_counter)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n_elements)
            return;
        neural_prefiltering_scatter(i,
                                    sample_idx,
                                    depth,
                                    its_result_data,
                                    traversal_data,
                                    results,
                                    neural_bvh,
                                    appearance_output_data,
                                    active_counter,
                                    init_neural_prefiltering_result);
    }

    NTWR_KERNEL void accumulate_neural_results(const uint32_t n_elements,
                                               int sample_idx,
                                               NeuralTraversalData traversal_data)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n_elements)
            return;

        glm::uvec2 pixel_coord;
        pixel_coord.x = traversal_data.pixel_coord(i) % fb_size.x;
        pixel_coord.y = traversal_data.pixel_coord(i) / fb_size.x;

        accumulate_pixel_value(sample_idx, pixel_coord, traversal_data.illumination[i], 1.0f);
    }

    NTWR_KERNEL void camera_raygen_and_first_hit(uint32_t n_elements,
                                                 int sample_idx,
                                                 int rnd_sample_offset,
                                                 HybridIntersectionResult intersection_result,
                                                 CudaSceneData scene_data,
                                                 uint32_t *active_counter)
    {
        int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (ray_idx >= n_elements)
            return;

        glm::uvec2 pixel_coord = glm::uvec2(ray_idx % fb_size.x, ray_idx / fb_size.x);

        LCGRand rng = get_lcg_rng(sample_idx, rnd_sample_offset, ray_idx);
        float xf    = (float(pixel_coord.x) + lcg_randomf(rng)) / float(fb_size.x);
        float yf    = (float(pixel_coord.y) + lcg_randomf(rng)) / float(fb_size.y);

        Ray3f ray = camera_raygen(xf, yf);

        PreliminaryItsData pi = scene_data.tlas.intersect<false>(ray);

        if (!pi.is_valid()) {
            glm::vec3 background_eval = background_hit_eval(ray);
            // Ray escaped
            accumulate_pixel_value(sample_idx, pixel_coord, background_eval, 1.0f);
            return;
        }

        glm::vec3 throughput = glm::vec3(1.f);
        glm::vec3 wo         = -ray.d;
        glm::vec3 hit_pos    = ray(pi.t);

        uint32_t idx = atomicAdd(active_counter, 1);

        intersection_result.ray_o.set(idx, ray.o);
        intersection_result.ray_d.set(idx, ray.d);
        intersection_result.t_neural[idx]     = pi.t;
        intersection_result.illumination[idx] = glm::vec3(0.f);
        intersection_result.throughput_rng[idx] =
            glm::vec4(throughput.x, throughput.y, throughput.z, glm::uintBitsToFloat(rng.state));
        intersection_result.pixel_coord[idx] = ray_idx;
    }

    NTWR_DEVICE glm::vec3 compute_lod_voxel_pos(glm::vec3 world_pos, Bbox3f &voxel_bbox)
    {
        glm::vec3 normalised_lod_voxel_pos = (world_pos - prefiltering_module_const.voxel_grid_bbox.min) /
                                             prefiltering_module_const.voxel_grid_bbox.extent();

        uint32_t voxel_grid_res = prefiltering_module_const.voxel_grid_max_res >> prefiltering_module_const.lod_level;
        glm::uvec3 lod_voxel_idx =
            glm::clamp(glm::uvec3(normalised_lod_voxel_pos * float(voxel_grid_res)), 0u, voxel_grid_res - 1);

        float step_size =
            scalbn((double)prefiltering_module_const.voxel_size_lod_0, prefiltering_module_const.lod_level);

        glm::vec3 voxel_min =
            prefiltering_module_const.voxel_grid_bbox.min + glm::vec3(lod_voxel_idx) * glm::vec3(step_size);

        glm::vec3 voxel_center_pos = voxel_min + glm::vec3(step_size * 0.5f);

        voxel_bbox = Bbox3f{voxel_min, voxel_min + glm::vec3(step_size)};

        return voxel_center_pos;
    }

    NTWR_KERNEL void generate_throughput_inference_data_and_update_ray(uint32_t n_elements,
                                                                       HybridIntersectionResult intersection_result,
                                                                       float *__restrict__ appearance_input_data)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n_elements)
            return;

        uint32_t ray_idx = i;

        LCGRand rng = intersection_result.get_rng(ray_idx);

        Ray3f ray = intersection_result.get_ray(ray_idx);

        glm::vec3 wo = -ray.d;

        glm::vec3 wi;
        if (!prefiltering_module_const.hg_sampling) {
            wi = square_to_uniform_sphere(lcg_random2f(rng));
        } else {
            wi = square_to_hg(lcg_random2f(rng), wo, prefiltering_module_const.hg_g);
            // Need pdf computation
        }

        glm::vec3 hit_pos = ray(intersection_result.t_neural[ray_idx]);

        Bbox3f voxel_bbox;
        glm::vec3 lod_voxel_pos = compute_lod_voxel_pos(hit_pos, voxel_bbox);

        float t_voxel_near, t_voxel_far;
        if (!ray.intersect_bbox_min_max(voxel_bbox, t_voxel_near, t_voxel_far, false)) {
            const glm::vec3 node_center      = voxel_bbox.center();
            const glm::vec3 node_extent      = voxel_bbox.extent() * (1.f + 1e-4f);
            const glm::vec3 dilated_aabb_min = node_center - node_extent * 0.5f;
            const glm::vec3 dilated_aabb_max = node_center + node_extent * 0.5f;

            if (!ray.intersect_bbox_min_max(dilated_aabb_min, dilated_aabb_max, t_voxel_near, t_voxel_far, false)) {
                printf("Voxel at pos : %f %f %f double missed, setting inference input data to 0\n",
                       lod_voxel_pos.x,
                       lod_voxel_pos.y,
                       lod_voxel_pos.z);
                for (uint32_t idx = 0; idx < 9; idx++) {
                    appearance_input_data[ray_idx * prefiltering_module_const.prefiltering_input_dims + idx] = 0.f;
                }
                intersection_result.throughput_rng[ray_idx].x = 0.f;
                intersection_result.throughput_rng[ray_idx].y = 0.f;
                intersection_result.throughput_rng[ray_idx].z = 0.f;
                return;
            }
        }

        const float t_eps = t_voxel_near + 1e-4f * (t_voxel_far - t_voxel_near);

        intersection_result.ray_o.set(ray_idx, ray(t_eps));
        intersection_result.ray_d.set(ray_idx, wi);

        intersection_result.throughput_rng[ray_idx].w = rng.state;

        // Normalized center coordinates
        glm::vec3 input_pos = (lod_voxel_pos - prefiltering_module_const.voxel_grid_bbox.min) /
                              prefiltering_module_const.voxel_grid_bbox.extent();

        uint32_t appearance_input_idx = ray_idx * prefiltering_module_const.prefiltering_input_dims;

        appearance_input_data[appearance_input_idx + 0] = input_pos.x;
        appearance_input_data[appearance_input_idx + 1] = input_pos.y;
        appearance_input_data[appearance_input_idx + 2] = input_pos.z;

        glm::vec3 input_wo                              = (wo + 1.f) * 0.5f;
        appearance_input_data[appearance_input_idx + 3] = input_wo.x;
        appearance_input_data[appearance_input_idx + 4] = input_wo.y;
        appearance_input_data[appearance_input_idx + 5] = input_wo.z;

        glm::vec3 input_wi                              = (wi + 1.f) * 0.5f;
        appearance_input_data[appearance_input_idx + 6] = input_wi.x;
        appearance_input_data[appearance_input_idx + 7] = input_wi.y;
        appearance_input_data[appearance_input_idx + 8] = input_wi.z;
    }

    NTWR_KERNEL void process_prefiltering_network_output(uint32_t n_elements,
                                                         HybridIntersectionResult intersection_result,
                                                         NeuralTraversalData traversal_data,
                                                         float *__restrict__ appearance_output_data)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n_elements)
            return;

        uint32_t output_idx = i * prefiltering_module_const.prefiltering_output_dims;
        glm::vec3 throughput_inference;
        throughput_inference.x = appearance_output_data[output_idx + 0];
        throughput_inference.y = appearance_output_data[output_idx + 1];
        throughput_inference.z = appearance_output_data[output_idx + 2];

        if (prefiltering_module_const.prefiltering_log_learnt) {
            throughput_inference.x = expf(throughput_inference.x) - 1.f;
            throughput_inference.y = expf(throughput_inference.y) - 1.f;
            throughput_inference.z = expf(throughput_inference.z) - 1.f;
        }
        throughput_inference = max(glm::vec3(0.f), throughput_inference);

        float inv_pdf                  = (4.f * PI);
        traversal_data.illumination[i] = throughput_inference * inv_pdf;
        traversal_data.init_pixel_coord(i, intersection_result.pixel_coord[i]);
    }

}
}