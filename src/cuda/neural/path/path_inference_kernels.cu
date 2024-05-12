

#include "cuda/cuda_backend_constants.cuh"

#include "cuda/neural/neural_bvh_module_constants.cuh"
#include "cuda/neural/neural_bvh_module_kernels.cuh"
#include "cuda/neural/neural_bvh_renderer_kernels.cuh"
#include "cuda/neural/path/path_constants.cuh"
#include "cuda/neural/path/path_inference_kernels.cuh"

#include "cuda/base/material.cuh"
#include "cuda/utils/rendering.cuh"

namespace ntwr {
namespace neural::path {

    NTWR_DEVICE void init_neural_traversal_result(uint32_t idx,
                                                  NeuralTraversalResult &results,
                                                  uint8_t current_valid_hit,
                                                  float current_t,
                                                  glm::vec3 *normal_or_uv,
                                                  glm::vec3 *albedo_or_mesh_prim_id)
    {
        // Initialize to unoccluded
        results.valid_hit[idx] = current_valid_hit;
        // The depth result also stores the ray maxt
        results.t_neural[idx] = current_t;

        if (normal_or_uv) {
            results.normal[idx] = *normal_or_uv;
        }
        if (albedo_or_mesh_prim_id) {
            results.albedo[idx] = *albedo_or_mesh_prim_id;
        }
    }

    NTWR_DEVICE glm::vec3 background_hit_eval(const Ray3f &ray)
    {
        if (!scene_constants.envmap_eval)
            return glm::vec3(0.f);
        if (scene_constants.skylike_background) {
            const float a = 0.5f * (ray.d.y + 1.f);
            return (1.f - a) * glm::vec3(1.0f) + a * glm::vec3(0.5f, 0.7f, 1.f);
        } else {
            glm::vec3 dir = ray.d;
            glm::mat4 rotation =
                glm::rotate(glm::mat4(1.f), glm::radians(scene_constants.envmap_rotation), glm::vec3(0.0, 1.0, 0.0));
            dir = glm::vec3(rotation * glm::vec4(dir, 1.0));
            // [-0.5,0.5] -> [0,1]
            float envmap_u = atan2(dir.x, -dir.z) * INV_PI * 0.5f + 0.5f;
            float envmap_v = safe_acos(dir.y) * INV_PI;
            float4 val     = tex2D<float4>(envmap_texture, envmap_u, envmap_v);
            // HACK : clamping to 100.f to avoid fireflies due to strong sun
            glm::vec3 envmap_val =
                glm::min(glm::vec3(100.f), glm::vec3(val.x, val.y, val.z) * scene_constants.envmap_strength);

            return envmap_val;
        }
    }

    NTWR_DEVICE bool process_neural_bvh_traversal_step(uint32_t ray_idx,
                                                       uint32_t _traversal_step,
                                                       float *__restrict__ pt_output_data,
                                                       NeuralTraversalResult &results,
                                                       NeuralTraversalData &traversal_data,
                                                       float &new_depth_estimate,
                                                       const NeuralBVH &neural_bvh)
    {
        new_depth_estimate = results.t_neural[ray_idx];

        bool visibility_query_only = false;
        float depth_estimate;
        bool ray_hit = neural_node_traversal(
            ray_idx, pt_output_data, traversal_data, depth_estimate, visibility_query_only, neural_bvh);

        if (!ray_hit) {
            assert(depth_estimate == FLT_MAX);
            return false;
        }

        assert(ray_hit);

        // terminate traversal early
        if (visibility_query_only) {
            results.valid_hit[ray_idx] = NEURAL_TRAVERSAL_HIT;
            return true;
        }

        if (depth_estimate < results.t_neural[ray_idx] ||
            (pt_target_const.interpolate_depth &&
             abs(results.t_neural[ray_idx] - depth_estimate) <= pt_target_const.depth_interpolation_eps)) {
            float prev_depth_estimate = results.t_neural[ray_idx];

            results.valid_hit[ray_idx] = NEURAL_TRAVERSAL_HIT;
            // Update with the new depth estimate
            results.t_neural[ray_idx] = depth_estimate;
            new_depth_estimate        = depth_estimate;

            float interpolate = false;

            if (pt_target_const.interpolate_depth &&
                abs(prev_depth_estimate - depth_estimate) <= pt_target_const.depth_interpolation_eps) {
                results.t_neural[ray_idx] = 0.5f * (prev_depth_estimate + depth_estimate);
                new_depth_estimate        = results.t_neural[ray_idx];
                interpolate               = true;
            }

            uint32_t output_idx = ray_idx * pt_target_const.output_network_dims +
                                  pt_target_const.output_visibility_dim + pt_target_const.output_depth_dim;
            // Update corresponding intersection's normal and albedo
            if (pt_target_const.output_normal_dim > 0) {
                glm::vec3 prev_normal = results.normal[ray_idx];
                // Normal learned in [0,1] remap to [-1,1] and normalize for safety
                glm::vec3 output_normal = glm::vec3(
                    pt_output_data[output_idx], pt_output_data[output_idx + 1], pt_output_data[output_idx + 2]);
                results.normal[ray_idx] = normalize(output_normal * 2.f - 1.f);

                if (interpolate) {
                    results.normal[ray_idx] = normalize(results.normal[ray_idx] + prev_normal);
                }

                output_idx += pt_target_const.output_normal_dim;
            }
            if (pt_target_const.output_albedo_dim > 0) {
                results.albedo[ray_idx] = glm::vec3(
                    pt_output_data[output_idx], pt_output_data[output_idx + 1], pt_output_data[output_idx + 2]);
                output_idx += pt_target_const.output_albedo_dim;
            }
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

            const glm::uvec2 pixel_coord = glm::uvec2(src_traversal_data.pixel_coord(ray_idx) % fb_size.x,
                                                      src_traversal_data.pixel_coord(ray_idx) / fb_size.x);

            if (path_module_const.output_mode == OutputMode::NeuralPathTracing) {
                uint32_t its_idx = atomicAdd(its_counter, 1);
                intersection_result_data.set_or_update_from_neural_traversal(
                    its_idx, src_results, src_traversal_data, ray_idx);
            }
        }
    }

    NTWR_KERNEL void non_neural_bvh_intersection(uint32_t n_elements,
                                                 int sample_idx,
                                                 NormalTraversalData normal_traversal_data,
                                                 HybridIntersectionResult intersection_result_data,
                                                 CudaSceneData scene_data,
                                                 uint32_t *its_counter)
    {
        int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (ray_idx >= n_elements)
            return;

        assert(normal_traversal_data.path_active(ray_idx));

        Ray3f ray = normal_traversal_data.get_ray(ray_idx);
        ray.maxt  = normal_traversal_data.last_tlas_t[ray_idx];

        const uint32_t blas_idx = normal_traversal_data.current_blas_idx[ray_idx];
        const CudaBLAS blas     = scene_data.tlas.mesh_bvhs[blas_idx];

        PreliminaryItsData pi = blas.intersect<false>(ray);
        pi.mesh_idx           = blas_idx;

        // Required because another cuda stream updates the intersection result as well
        uint32_t its_idx = atomicAdd(its_counter, 1);

        // Update the traversal data related info (irrespective of a hit or no hit)
        intersection_result_data.set_or_update_with_preliminary_its_data(its_idx, normal_traversal_data, ray_idx, pi);
    }

    NTWR_KERNEL void hybrid_scatter_and_get_next_tlas_traversal_step(const uint32_t n_elements,
                                                                     int sample_idx,
                                                                     HybridIntersectionResult hybrid_its_result,
                                                                     NeuralTraversalData next_neural_traversal_data,
                                                                     NeuralTraversalResult next_neural_results,
                                                                     NormalTraversalData next_normal_traversal_data,
                                                                     NeuralBVH neural_bvh,
                                                                     CudaSceneData scene_data,
                                                                     uint32_t *active_neural_counter,
                                                                     uint32_t *active_normal_counter,
                                                                     bool last_tlas_traversal,
                                                                     uint32_t max_depth)
    {
        int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (ray_idx >= n_elements)
            return;

        // Get pixel coordinate
        uint32_t pixel_coord_1d = hybrid_its_result.pixel_coord[ray_idx];
        glm::uvec2 pixel_coord;
        pixel_coord.x = pixel_coord_1d % fb_size.x;
        pixel_coord.y = pixel_coord_1d / fb_size.x;

        const Ray3f ray        = hybrid_its_result.get_ray(ray_idx);
        uint8_t ray_depth      = hybrid_its_result.ray_depth[ray_idx];
        glm::vec3 throughput   = hybrid_its_result.get_throughput(ray_idx);
        LCGRand rng            = hybrid_its_result.get_rng(ray_idx);
        float best_t           = hybrid_its_result.t[ray_idx];
        glm::vec3 illumination = hybrid_its_result.illumination[ray_idx];
        uint8_t valid_hit      = hybrid_its_result.valid_hit[ray_idx];

        if (last_tlas_traversal) {
            // occluded, early exit and splat current illumination
            accumulate_pixel_value(sample_idx, pixel_coord, illumination, 1.0f);
            return;
        }

        bool neural_hit = valid_hit == NEURAL_TRAVERSAL_HIT;

        assert(valid_hit != 0 || hybrid_its_result.t[ray_idx] == FLT_MAX);
        assert(!neural_hit || !bvh_const.ignore_neural_blas);

        // Check the next tlas traversal, if the traversal is done we will correctly update the intersection and start a
        // new traversal
        uint32_t local_tlas_stack[TLAS_STACK_SIZE];
        int local_tlas_stack_size = hybrid_its_result.tlas_stack_size[ray_idx];

        if (local_tlas_stack_size != 0) {
            uint32_t src_index;
            for (uint32_t stack_idx = 0; stack_idx < hybrid_its_result.tlas_stack_size[ray_idx]; stack_idx++) {
                src_index                   = ray_idx + stack_idx * fb_size_flat;
                local_tlas_stack[stack_idx] = hybrid_its_result.tlas_stack[src_index];
            }

            bool tlas_traversal_done =
                tlas_traversal_and_fill_next_hybrid_traversal(sample_idx,
                                                              pixel_coord_1d,
                                                              ray,
                                                              ray_depth,
                                                              best_t,
                                                              valid_hit,
                                                              &hybrid_its_result.normal_or_uv[ray_idx],
                                                              &hybrid_its_result.albedo_or_mesh_prim_id[ray_idx],
                                                              illumination,
                                                              throughput,
                                                              rng,
                                                              local_tlas_stack,
                                                              local_tlas_stack_size,
                                                              next_normal_traversal_data,
                                                              neural_bvh,
                                                              next_neural_traversal_data,
                                                              next_neural_results,
                                                              scene_data,
                                                              active_neural_counter,
                                                              active_normal_counter,
                                                              init_neural_traversal_result);
            // In case we still have other blas to process return
            if (!tlas_traversal_done) {
                return;
            }
        }

        assert(local_tlas_stack_size == 0);

        assert(path_module_const.output_mode == OutputMode::NeuralPathTracing);

        bool last_bounce = ray_depth == max_depth;
        // Ray is terminated because its the last bounce or ray escaped
        const bool its_found = (valid_hit != 0);
        if (!its_found || last_bounce) {
            // If the ray is terminated due to not finding any intersection, we add the background contribution
            if (!its_found) {
                glm::vec3 background_eval = background_hit_eval(ray);
                illumination += throughput * background_eval;
            }
            accumulate_pixel_value(sample_idx, pixel_coord, illumination, 1.0f);
            return;
        }
        // Prepare for another bounce
        assert(!last_bounce && its_found);
        // Treat differently based on neural hit or non neural hit here
        CudaMaterial bsdf;
        const glm::vec3 wo = -ray.d;
        glm::vec3 its_normal;
        glm::vec3 hit_pos;

        if (neural_hit) {
            bsdf.base_color = hybrid_its_result.get_albedo(ray_idx);
            bsdf.metallic   = path_module_const.neural_metallic;
            bsdf.roughness  = path_module_const.neural_roughness;
            bsdf.specular   = path_module_const.neural_specular;
            its_normal      = hybrid_its_result.get_normal(ray_idx);
            hit_pos         = ray(best_t);
        } else {
            const PreliminaryItsData pi = hybrid_its_result.get_preliminary_its_data(ray_idx);
            const CudaBLAS &geom        = scene_data.tlas.mesh_bvhs[pi.mesh_idx];
            ItsData its                 = geom.compute_its_data(pi, scene_data.materials, scene_data.textures);
            hit_pos                     = its.position;
            its_normal                  = its.sh_normal;

            bool use_neural_material = bvh_const.ignore_neural_blas && pi.mesh_idx == bvh_const.neural_bvh_idx;
            if (use_neural_material) {
                bsdf.base_color = unpack_albedo(scene_data.materials[its.material_id], its.uv, scene_data.textures);
                bsdf.metallic   = path_module_const.neural_metallic;
                bsdf.roughness  = path_module_const.neural_roughness;
                bsdf.specular   = path_module_const.neural_specular;
            } else if (scene_constants.force_constant_material) {
                bsdf.base_color = unpack_albedo(scene_data.materials[its.material_id], its.uv, scene_data.textures);
                bsdf.metallic   = scene_constants.constant_metallic;
                bsdf.roughness  = scene_constants.constant_roughness;
                bsdf.specular   = scene_constants.constant_specular;
            } else if (its.material_id != -1) {
                bsdf = unpack_material(scene_data.materials[its.material_id], its.uv, scene_data.textures);
            }
        }
        // Update throughput
        float pdf;
        glm::vec3 wi;

        glm::vec3 sh_normal, tangent, bitangent;
        shading_frame(wo, its_normal, sh_normal, tangent, bitangent);

        // Add emission of bsdf for direct hits only
        if ((ray_depth == 1 || scene_constants.emission_eval) && bsdf.base_emission != glm::vec3(0.f)) {
            illumination += throughput * bsdf.emission();
        }

        // Sample bsdf
        const glm::vec3 bsdf_eval = disney_sample_eval(bsdf, sh_normal, wo, tangent, bitangent, rng, wi, pdf);

        const float cos_theta_i = dot(wi, sh_normal);
        throughput *= bsdf_eval * fabs(cos_theta_i) / pdf;
        if (pdf == 0.f || bsdf_eval == glm::vec3(0.f)) {
            accumulate_pixel_value(sample_idx, pixel_coord, illumination, 1.0f);
            return;
        }

        if (path_module_const.apply_russian_roulette) {
            bool terminate_path = rr_terminate(ray_depth, throughput, path_module_const.rr_start_depth, rng);
            if (terminate_path) {
                // If we terminate the path due to russian roulette
                if (terminate_path) {
                    accumulate_pixel_value(sample_idx, pixel_coord, illumination, 1.0f);
                    return;
                }
            }
        }

        assert(bsdf.specular_transmission != 0.f || same_hemisphere(wo, wi, sh_normal));
        Ray3f sampled_ray = Ray3f(hit_pos + sh_normal * path_module_const.normal_epsilon_offset, wi);

        // Begin a new tlas traversal
        local_tlas_stack[0]   = 0;
        local_tlas_stack_size = 1;

        bool tlas_traversal_done = tlas_traversal_and_fill_next_hybrid_traversal(sample_idx,
                                                                                 pixel_coord_1d,
                                                                                 sampled_ray,
                                                                                 ray_depth + 1,
                                                                                 FLT_MAX,
                                                                                 uint8_t(0),
                                                                                 nullptr,
                                                                                 nullptr,
                                                                                 illumination,
                                                                                 throughput,
                                                                                 rng,
                                                                                 local_tlas_stack,
                                                                                 local_tlas_stack_size,
                                                                                 next_normal_traversal_data,
                                                                                 neural_bvh,
                                                                                 next_neural_traversal_data,
                                                                                 next_neural_results,
                                                                                 scene_data,
                                                                                 active_neural_counter,
                                                                                 active_normal_counter,
                                                                                 init_neural_traversal_result);

        if (tlas_traversal_done) {
            if ((ray_depth + 1) <= max_depth) {
                glm::vec3 background_eval = background_hit_eval(sampled_ray);
                illumination += throughput * background_eval;
            }
            accumulate_pixel_value(sample_idx, pixel_coord, illumination, 1.0f);
            return;
        }
    }

    template <typename InitNeuralTraversalResult>
    NTWR_DEVICE void camera_raygen_and_hybrid_first_hit_and_traversal(
        int32_t ray_idx,
        int sample_idx,
        int rnd_sample_offset,
        NormalTraversalData &normal_traversal_data,
        NeuralBVH &neural_bvh,
        NeuralTraversalData &neural_traversal_data,
        NeuralTraversalResult &neural_results,
        CudaSceneData &scene_data,
        uint32_t *active_neural_counter,
        uint32_t *active_normal_counter,
        InitNeuralTraversalResult init_neural_query_result)
    {
        glm::uvec2 pixel_coord = glm::uvec2(ray_idx % fb_size.x, ray_idx / fb_size.x);

        LCGRand rng = get_lcg_rng(sample_idx, rnd_sample_offset, ray_idx);
        float xf    = (float(pixel_coord.x) + lcg_randomf(rng)) / float(fb_size.x);
        float yf    = (float(pixel_coord.y) + lcg_randomf(rng)) / float(fb_size.y);

        Ray3f ray = camera_raygen(xf, yf);

        PreliminaryItsData pi = scene_data.tlas.intersect<false>(ray);

        if (!pi.is_valid()) {
            // Ray escaped
            glm::vec3 background_eval = background_hit_eval(ray);
            accumulate_pixel_value(sample_idx, pixel_coord, background_eval, 1.0f);
            return;
        }

        glm::vec3 illumination = glm::vec3(0.f);
        glm::vec3 throughput   = glm::vec3(1.f);
        ////////////// Compute first non-neural hit, throughput and illumination
        const CudaBLAS &geom = scene_data.tlas.mesh_bvhs[pi.mesh_idx];
        ItsData its          = geom.compute_its_data(pi, scene_data.materials, scene_data.textures);

        CudaMaterial bsdf;
        if (its.material_id != -1 && !scene_constants.force_constant_material)
            bsdf = unpack_material(scene_data.materials[its.material_id], its.uv, scene_data.textures);
        else if (scene_constants.force_constant_material) {
            bsdf.base_color = unpack_albedo(scene_data.materials[its.material_id], its.uv, scene_data.textures);
            bsdf.metallic   = scene_constants.constant_metallic;
            bsdf.roughness  = scene_constants.constant_roughness;
            bsdf.specular   = scene_constants.constant_specular;
        }

        glm::vec3 wo = normalize(-ray.d);
        glm::vec3 sh_normal, tangent, bitangent;
        shading_frame(wo, its.sh_normal, sh_normal, tangent, bitangent);

        // Add emission of bsdf (just for 1st bounce)
        if (bsdf.base_emission != glm::vec3(0.f))
            illumination += throughput * bsdf.emission();

        // Update throughput
        float pdf;
        glm::vec3 wi;

        glm::vec3 bsdf_eval = disney_sample_eval(bsdf, sh_normal, wo, tangent, bitangent, rng, wi, pdf);

        const float cos_theta_i = dot(wi, sh_normal);
        throughput *= bsdf_eval * fabs(cos_theta_i) / pdf;

        if (pdf == 0.f || bsdf_eval == glm::vec3(0.f)) {
            throughput = glm::vec3(0.f);
            accumulate_pixel_value(sample_idx, pixel_coord, illumination, 1.0f);
            return;
        }

        Ray3f sampled_ray = Ray3f(its.position + sh_normal * path_module_const.normal_epsilon_offset, wi);
        uint8_t ray_depth = 2;
        // Initially point to root node (always index 0)
        uint32_t local_tlas_stack[TLAS_STACK_SIZE];
        local_tlas_stack[0]       = 0;
        int local_tlas_stack_size = 1;

        bool tlas_traversal_done =
            tlas_traversal_and_fill_next_hybrid_traversal(sample_idx,
                                                          ray_idx,  // for the first traversal this is just the ray_idx
                                                          sampled_ray,
                                                          ray_depth,
                                                          FLT_MAX,
                                                          uint8_t(0),
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
            // No hit splat result
            glm::vec3 background_eval = background_hit_eval(ray);
            illumination += throughput * background_eval;
            accumulate_pixel_value(sample_idx, pixel_coord, illumination, 1.0f);
            return;
        }
    }

    NTWR_KERNEL void hybrid_camera_raygen_and_hybrid_first_hit_and_traversal(uint32_t n_elements,
                                                                             int sample_idx,
                                                                             int rnd_sample_offset,
                                                                             NormalTraversalData normal_traversal_data,
                                                                             NeuralBVH neural_bvh,
                                                                             NeuralTraversalData neural_traversal_data,
                                                                             NeuralTraversalResult neural_results,
                                                                             CudaSceneData scene_data,
                                                                             uint32_t *active_neural_counter,
                                                                             uint32_t *active_normal_counter)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n_elements)
            return;

        hybrid_camera_raygen_and_intersect_neural_bvh(i,
                                                      sample_idx,
                                                      rnd_sample_offset,
                                                      normal_traversal_data,
                                                      neural_bvh,
                                                      neural_traversal_data,
                                                      neural_results,
                                                      scene_data,
                                                      active_neural_counter,
                                                      active_normal_counter,
                                                      init_neural_traversal_result,
                                                      background_hit_eval);
    }

    NTWR_KERNEL void split_screen_hybrid_camera_raygen_and_hybrid_first_hit_and_traversal(
        uint32_t n_elements,
        int sample_idx,
        int rnd_sample_offset,
        NormalTraversalData normal_traversal_data,
        NeuralBVH neural_bvh,
        NeuralTraversalData neural_traversal_data,
        NeuralTraversalResult neural_results,
        CudaSceneData scene_data,
        uint32_t *active_neural_counter,
        uint32_t *active_normal_counter,
        NeuralTreeCutData *neural_tree_cut_data,
        float max_loss,
        bool show_tree_cut_depth,
        uint32_t max_tree_cut_depth)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n_elements)
            return;

        glm::uvec2 pixel = glm::uvec2(i % fb_size.x, i / fb_size.x);

        // float x1 = ;
        // float y1 = fb_size.y / 2.f;
        // float m  = -6.0f;
        // float expected_y = m * (pixel.x - x1) + y1;
        // if (pixel.y <= expected_y) {
        bool valid_its = false;
        if (pixel.x <= fb_size.x / 2.f) {
            valid_its = show_neural_tree_cut_data_impl(i,
                                                       sample_idx,
                                                       rnd_sample_offset,
                                                       neural_bvh,
                                                       neural_tree_cut_data,
                                                       max_loss,
                                                       show_tree_cut_depth,
                                                       max_tree_cut_depth);
        }
        if (valid_its) {
            return;
        }

        hybrid_camera_raygen_and_intersect_neural_bvh(i,
                                                      sample_idx,
                                                      rnd_sample_offset,
                                                      normal_traversal_data,
                                                      neural_bvh,
                                                      neural_traversal_data,
                                                      neural_results,
                                                      scene_data,
                                                      active_neural_counter,
                                                      active_normal_counter,
                                                      init_neural_traversal_result,
                                                      background_hit_eval);
    }

    NTWR_KERNEL void camera_raygen_and_hybrid_first_hit_and_traversal(uint32_t n_elements,
                                                                      int sample_idx,
                                                                      int rnd_sample_offset,
                                                                      NormalTraversalData normal_traversal_data,
                                                                      NeuralBVH neural_bvh,
                                                                      NeuralTraversalData traversal_data,
                                                                      NeuralTraversalResult neural_results,
                                                                      CudaSceneData scene_data,
                                                                      uint32_t *active_neural_counter,
                                                                      uint32_t *active_normal_counter)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n_elements)
            return;
        camera_raygen_and_hybrid_first_hit_and_traversal(i,
                                                         sample_idx,
                                                         rnd_sample_offset,
                                                         normal_traversal_data,
                                                         neural_bvh,
                                                         traversal_data,
                                                         neural_results,
                                                         scene_data,
                                                         active_neural_counter,
                                                         active_normal_counter,
                                                         init_neural_traversal_result);
    }
}
}