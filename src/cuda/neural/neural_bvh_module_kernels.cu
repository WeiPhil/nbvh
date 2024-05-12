#include "cuda/base/material.cuh"
#include "cuda/neural/neural_bvh_module_common.cuh"
#include "cuda/neural/neural_bvh_module_constants.cuh"
#include "cuda/neural/neural_bvh_renderer.h"
#include "cuda/utils/rendering.cuh"

namespace ntwr {
namespace neural {

    NTWR_DEVICE void encode_segment_from_world_po_pi(float *__restrict__ bvh_input_data,
                                                     uint32_t start_input_idx,
                                                     const glm::vec3 &world_po,
                                                     const glm::vec3 &world_pi,
                                                     LCGRand &rng,
                                                     uint32_t node_idx)
    {
        glm::vec3 encoded_po, encoded_pi;
        if (bvh_module_const.apply_encoding_trafo) {
            // currently doesn't support dilation
            assert(false);
            Bbox3f transformed_world_bbox = {glm::vec3(FLT_MAX), glm::vec3(-FLT_MAX)};
            // const float angle             = 0.471239;  // 27 degrees
            const float angle         = PI * 0.25f;  // 45 degrees
            glm::mat4 magic_transform = glm::rotate(angle, glm::vec3(1, 0, 0)) * glm::rotate(angle, glm::vec3(0, 0, 1));
            // magic_transform           = glm::shearY3D(magic_transform, 0.4f, 0.2f);
            for (size_t c = 0; c < 8; c++) {
                const glm::vec3 transformed_corner = magic_transform * glm::vec4(neural_scene_bbox.corner(c), 1);
                transformed_world_bbox.expand(transformed_corner);
            }
            glm::mat4 transformed_world_to_encoding =
                glm::scale(1.f / (transformed_world_bbox.extent())) * glm::translate(-transformed_world_bbox.min);
            // Encoded positions are in the hashgrid local [0,1]^3 space
            // corresponding to the neural scene bbox.
            encoded_po = transformed_world_to_encoding * magic_transform *
                         glm::vec4(world_po, 1.f);  // neural_scene_bbox.bbox_to_local(world_po);
            encoded_pi = transformed_world_to_encoding * magic_transform *
                         glm::vec4(world_pi, 1.f);  // neural_scene_bbox.bbox_to_local(world_pi);
        } else {
            const float dilation        = (1.f + bvh_const.node_extent_dilation + bvh_const.fixed_node_dilation);
            const glm::vec3 node_extent = neural_scene_bbox.extent() * dilation;

            encoded_po = (world_po - neural_scene_bbox.center()) / node_extent + 0.5f;
            encoded_pi = (world_pi - neural_scene_bbox.center()) / node_extent + 0.5f;
            // encoded_po = neural_scene_bbox.bbox_to_local(world_po); // Old appraoch without dilation
            // encoded_pi = neural_scene_bbox.bbox_to_local(world_pi); // Old appraoch without dilation
        }

        for (int i = 0; i < 3; i++) {
            assert(encoded_po[i] <= 1.001f && encoded_po[i] >= -1e-3f);
            assert(encoded_pi[i] <= 1.001f && encoded_pi[i] >= -1e-3f);
        }

        if (bvh_module_const.segment_distribution == SegmentDistribution::Uniform) {
            // Segment points input
            for (int k = 0; k < bvh_module_const.input_segment_points; k++) {
                float t = lcg_randomf(rng);

                glm::vec3 encoded_p = (1.0f - t) * encoded_po + t * encoded_pi;

                uint32_t segment_points_input_idx = start_input_idx + bvh_module_const.input_segment_point_dim * k;

                bvh_input_data[segment_points_input_idx]     = encoded_p.x;
                bvh_input_data[segment_points_input_idx + 1] = encoded_p.y;
                bvh_input_data[segment_points_input_idx + 2] = encoded_p.z;
            }
            return;
        }

        float delta, offset;
        if (bvh_module_const.segment_distribution == SegmentDistribution::EquidistantExterior) {
            delta  = 1.f / (bvh_module_const.input_segment_points - 1);
            offset = 0;

            if (bvh_module_const.input_segment_points == 1) {
                delta = 1.f;
            }

        } else {
            assert(bvh_module_const.segment_distribution == SegmentDistribution::EquidistantInterior);

            delta  = 1.f / bvh_module_const.input_segment_points;
            offset = 0.5f;
        }

        // Segment points input
        for (int k = 0; k < bvh_module_const.input_segment_points; k++) {
            float t = delta * (k + offset);

            glm::vec3 encoded_p = (1.0f - t) * encoded_po + t * encoded_pi;

            uint32_t segment_points_input_idx = start_input_idx + bvh_module_const.input_segment_point_dim * k;

            bvh_input_data[segment_points_input_idx]     = encoded_p.x;
            bvh_input_data[segment_points_input_idx + 1] = encoded_p.y;
            bvh_input_data[segment_points_input_idx + 2] = encoded_p.z;
        }
    }

    NTWR_DEVICE void encode_direction_from_ray_dir(float *__restrict__ bvh_input_data,
                                                   uint32_t start_input_idx,
                                                   const glm::vec3 &ray_dir)
    {
        assert(bvh_module_const.input_segment_direction_dim > 0);

        glm::vec3 encoded_dir = ray_dir;
        if (bvh_module_const.antialias_direction)
            encoded_dir = ray_dir.z < 0 ? -encoded_dir : encoded_dir;

        encoded_dir.x = (encoded_dir.x + 1.f) * 0.5f;
        encoded_dir.y = (encoded_dir.y + 1.f) * 0.5f;
        if (!bvh_module_const.antialias_direction)
            encoded_dir.z = (encoded_dir.z + 1.f) * 0.5f;

        bvh_input_data[start_input_idx]     = encoded_dir.x;
        bvh_input_data[start_input_idx + 1] = encoded_dir.y;
        bvh_input_data[start_input_idx + 2] = encoded_dir.z;
    }

    NTWR_KERNEL void granular_sample_bvh_network_input_data(int sub_batch_size,
                                                            int n_sub_batch,
                                                            int training_step,
                                                            NeuralBVH neural_bvh,
                                                            CudaSceneData scene_data,
                                                            uint32_t *__restrict__ sampled_node_indices,
                                                            float *__restrict__ visibility_input_data,
                                                            float *__restrict__ targets,
                                                            float current_max_loss,
                                                            NeuralTreeCutData *neural_tree_cut_data)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;

        if (i >= sub_batch_size)
            return;

        int batch_size = sub_batch_size * n_sub_batch;

        // Initialise traversal stack
        float leaf_mint, leaf_maxt;
        Ray3f ray;

        uint32_t local_stack[NEURAL_STACK_SIZE];
        int local_stack_size = 0;

        bool continue_traversal     = false;
        bool swap_trained_direction = false;

        uint32_t training_bounce = 0;

        assert(bvh_const.leaf_boundary_sampling_ratio == 0.f || n_sub_batch == 1);

        // From thread_idx to thread_idx + n_sub_batch
        uint32_t thread_idx = (i * n_sub_batch);

        for (size_t inner_batch_index = 0; inner_batch_index < n_sub_batch; inner_batch_index++) {
            uint32_t start_input_idx  = thread_idx * bvh_module_const.input_network_dims;
            uint32_t start_target_idx = thread_idx * pt_target_const.output_network_dims;

            auto rng = get_lcg_rng(training_step * batch_size, 0, thread_idx);

            assert(ray.mint == 1e-4f);
            assert(ray.maxt == FLT_MAX);

            bool restarted_sampling = stack_based_sample_uniform_bvh_density(neural_bvh,
                                                                             local_stack,
                                                                             local_stack_size,
                                                                             rng,
                                                                             ray,
                                                                             &leaf_mint,
                                                                             &leaf_maxt,
                                                                             &sampled_node_indices[thread_idx],
                                                                             continue_traversal);

            if (restarted_sampling) {
                training_bounce        = 0;
                swap_trained_direction = false;
            }

            // We might hit another node first rather than the inner node in which case we do NOT want to swap
            if (swap_trained_direction && leaf_mint >= 0.f) {
                swap_trained_direction = false;
            }

            // Check if we should keep the sampled node or not
            if (bvh_const.training_node_survival) {
                auto node_data      = neural_tree_cut_data[sampled_node_indices[thread_idx]];
                float estimate_loss = node_data.compute_loss_heuristic();

                float node_kept_prob = max(bvh_const.min_rr_survival_prob, (estimate_loss / current_max_loss));

                if (!(estimate_loss <= current_max_loss)) {
                    // assert(neural_bvh.nodes[sampled_node_indices[thread_idx]].is_interior_at_lod(bvh_const.current_learned_lod));
                    node_kept_prob = 1.f;
                } else {
                    assert(estimate_loss <= current_max_loss);
                }
                if (bvh_const.survival_probability_scale * node_kept_prob < lcg_randomf(rng)) {
                    continue;
                }
            }

            glm::vec3 world_po = ray(swap_trained_direction ? leaf_maxt : leaf_mint);
            glm::vec3 world_pi = ray(swap_trained_direction ? leaf_mint : leaf_maxt);

            encode_segment_from_world_po_pi(
                visibility_input_data, start_input_idx, world_po, world_pi, rng, sampled_node_indices[thread_idx]);

            assert(bvh_module_const.input_network_dims ==
                   bvh_module_const.input_segment_points_dim + bvh_module_const.input_segment_direction_dim);

            // Directional input encoding if requested
            if (bvh_module_const.input_segment_direction_dim > 0)
                encode_direction_from_ray_dir(
                    visibility_input_data, start_input_idx + bvh_module_const.input_segment_points_dim, ray.d);

            PreliminaryItsData pi  = generate_single_bvh_network_output_reference(ray,
                                                                                 leaf_mint,
                                                                                 leaf_maxt,
                                                                                 scene_data,
                                                                                 swap_trained_direction,
                                                                                 &targets[start_target_idx],
                                                                                 sampled_node_indices[thread_idx]);
            swap_trained_direction = false;  // reset decision

            // If no hit we continue the traversal
            continue_traversal = !pi.is_valid();

            if (pi.is_valid() && bvh_const.max_training_bounces > 0) {
                training_bounce++;
                const CudaBLAS &geom = scene_data.tlas.mesh_bvhs[pi.mesh_idx];
                ItsData its          = geom.compute_its_data(pi, scene_data.materials, scene_data.textures);

                // Sample a cosine weighted hemisphere
                const glm::vec3 wo = -ray.d;
                glm::vec3 sh_normal, tangent, bitangent;
                bool backface_normal = shading_frame(wo, its.sh_normal, sh_normal, tangent, bitangent);

                if ((backface_normal && !bvh_const.backface_bounce_training) ||
                    training_bounce >= bvh_const.max_training_bounces) {
                    training_bounce    = 0;
                    continue_traversal = false;
                } else {
                    glm::vec3 dir_sample;
                    if (bvh_const.sample_cosine_weighted) {
                        dir_sample = sample_lambertian_dir(sh_normal, tangent, bitangent, lcg_random2f(rng));
                    } else {
                        dir_sample = square_to_uniform_sphere(lcg_random2f(rng));
                        if (!same_hemisphere(wo, dir_sample, sh_normal)) {
                            dir_sample *= -1;
                        }
                    }
                    ray = Ray3f(its.position, dir_sample);

                    // We manually set the newly sampled position, hence continue traversal
                    continue_traversal     = true;
                    swap_trained_direction = true;
                    // Reset stack
                    local_stack[0]   = 0;
                    local_stack_size = 1;
                }
            }

            thread_idx++;
        }
    }

    NTWR_KERNEL void encode_bvh_network_input_data(uint32_t n_elements,
                                                   NeuralTraversalData traversal_data,
                                                   float *__restrict__ visibility_input_data,
                                                   bool swap_direction_for_neg_mint)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n_elements)
            return;

        assert(!traversal_data.all_nodes_processed(i));

        Ray3f ray = traversal_data.get_ray(i);

        float segment_mint = traversal_data.leaf_mint[i];
        float segment_maxt = traversal_data.leaf_maxt[i];

        // If queried, swap the mint and maxt to get the swapped direction
        // Note that the traversal_data mints/maxts are not swapped! The user
        // has to handle the negative mint case itself
        if (segment_mint < 0.f && swap_direction_for_neg_mint) {
            swap(segment_mint, segment_maxt);
        }

        glm::vec3 world_po = ray(segment_mint);
        glm::vec3 world_pi = ray(segment_maxt);

        uint32_t input_idx = i * bvh_module_const.input_network_dims;

        LCGRand rng = traversal_data.get_rng(i);
        encode_segment_from_world_po_pi(
            visibility_input_data, input_idx, world_po, world_pi, rng, traversal_data.node_idx[i]);
        traversal_data.set_rng(i, rng);

        // Directional input encoding if requested
        if (bvh_module_const.input_segment_direction_dim > 0)
            encode_direction_from_ray_dir(
                visibility_input_data, input_idx + bvh_module_const.input_segment_points_dim, ray.d);
    }

}
}