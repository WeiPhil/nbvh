

#include "cuda/neural/neural_bvh_module_common.cuh"
#include "cuda/neural/neural_bvh_renderer_kernels.cuh"

#include "cuda/neural/neural_bvh_module_constants.cuh"
#include "cuda/neural/neural_bvh_renderer_constants.cuh"

#include "FLIP.h"

namespace ntwr {
namespace neural {

    NTWR_KERNEL void camera_raygen(int sample_idx, int rnd_sample_offset, CudaBVHTraversalData traversal_data)
    {
        glm::uvec2 pixel = WavefrontPixelID;

        int i = int(pixel.y) * fb_size.x + int(pixel.x);
        if (pixel.x >= unsigned(fb_size.x) || pixel.y >= unsigned(fb_size.y))
            return;

        LCGRand rng = get_lcg_rng(sample_idx, rnd_sample_offset, i);
        float xf    = (float(pixel.x) + lcg_randomf(rng)) / float(fb_size.x);
        float yf    = (float(pixel.y) + lcg_randomf(rng)) / float(fb_size.y);

        Ray3f ray = camera_raygen(xf, yf);
        traversal_data.ray_o.set(i, ray.o);
        traversal_data.ray_d.set(i, ray.d);
        traversal_data.set_rng(i, rng);
    }

    NTWR_KERNEL void accumulate_results(int sample_idx, glm::vec3 *illumination)
    {
        glm::uvec2 pixel = WavefrontPixelID;
        int i            = int(pixel.y) * fb_size.x + int(pixel.x);
        if (pixel.x >= unsigned(fb_size.x) || pixel.y >= unsigned(fb_size.y))
            return;

        accumulate_pixel_value(sample_idx, pixel, illumination[i], 1.0f);
    }

    NTWR_KERNEL void copy_framebuffers_to_flip_image(int sample_idx,
                                                     FLIP::color3 *reference,
                                                     FLIP::color3 *test,
                                                     const int3 dim)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z * blockDim.z + threadIdx.z;
        int i = (z * dim.y + y) * dim.x + x;

        if (x >= dim.x || y >= dim.y || z >= dim.z)
            return;

        float4 ref_data;
        surf2Dread(&ref_data, reference_buffer, x * 4 * sizeof(float), y, cudaBoundaryModeZero);
        reference[i] = FLIP::color3(ref_data.x, ref_data.y, ref_data.z);

        float4 accum_data;
        surf2Dread(&accum_data, accum_buffer, x * 4 * sizeof(float), y, cudaBoundaryModeZero);
        test[i] = FLIP::color3(accum_data.x, accum_data.y, accum_data.z) / float(sample_idx);
    }

    NTWR_KERNEL void flip_image_to_framebuffer(FLIP::color3 *image, const int3 dim)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z * blockDim.z + threadIdx.z;
        int i = (z * dim.y + y) * dim.x + x;

        if (x >= dim.x || y >= dim.y || z >= dim.z)
            return;

        {
            uchar4 data;
            data.x = glm::clamp(int(image[i].x * 256.0f), 0, 255);
            data.y = glm::clamp(int(image[i].y * 256.0f), 0, 255);
            data.z = glm::clamp(int(image[i].z * 256.0f), 0, 255);
            data.w = 255;
            surf2Dwrite(data, framebuffer, x * 4, y, cudaBoundaryModeZero);
        }
    }

    NTWR_DEVICE glm::vec3 filmic_tone_mapping(glm::vec3 color)
    {
        color = max(glm::vec3(0.0), color - glm::vec3(0.004f));
        color = (color * (6.2f * color + 0.5f)) / (color * (6.2f * color + 1.7f) + 0.06f);
        return color;
    }

    NTWR_KERNEL void compute_and_display_render_error(const uint32_t n_elements,
                                                      int sample_idx,
                                                      PostprocessMode postprocess_mode,
                                                      float *mean_error,
                                                      bool false_color)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n_elements)
            return;

        glm::uvec2 pixel = glm::uvec2(i % fb_size.x, i / fb_size.x);

        glm::vec3 accum_color;
        {
            float4 data;
            surf2Dread(&data, accum_buffer, pixel.x * 4 * sizeof(float), pixel.y, cudaBoundaryModeZero);
            accum_color = glm::vec3(data.x, data.y, data.z);
        }
        accum_color /= float(sample_idx);

        glm::vec3 ref_color;
        {
            float4 data;
            surf2Dread(&data, reference_buffer, pixel.x * 4 * sizeof(float), pixel.y, cudaBoundaryModeZero);
            ref_color = glm::vec3(data.x, data.y, data.z);
        }

        glm::vec3 error;
        switch (postprocess_mode) {
        case PostprocessMode::Filmic: {
            error = filmic_tone_mapping(accum_color);
            break;
        }
        case PostprocessMode::Reference: {
            error = ref_color;
            break;
        }
        case PostprocessMode::AbsError: {
            error = abs(ref_color - accum_color);
            break;
        }
        case PostprocessMode::RelAbsError: {
            error = abs(ref_color - accum_color) / (ref_color + 0.1f);
            break;
        }
        case PostprocessMode::MSEError: {
            error = sqr(ref_color - accum_color);
            break;
        }
        case PostprocessMode::RelMSEError: {
            error = sqr(ref_color - accum_color) / (ref_color + 0.1f);
            break;
        }
        default:
            UNREACHABLE();
        }
        atomicAdd(mean_error, (error.x + error.y + error.z) / 3);

        if (false_color) {
            error = viridis_colormap(log(1 + (error.x + error.y + error.z) / 3));
        } else {
            error = linear_to_srgb(error);
        }
        {
            uchar4 data;
            data.x = glm::clamp(int(error.x * 256.0f), 0, 255);
            data.y = glm::clamp(int(error.y * 256.0f), 0, 255);
            data.z = glm::clamp(int(error.z * 256.0f), 0, 255);
            data.w = 255;
            surf2Dwrite(data, framebuffer, pixel.x * 4, pixel.y, cudaBoundaryModeZero);
        }
    }

    NTWR_KERNEL void show_bhv_closest_node(uint32_t n_elements,
                                           int sample_idx,
                                           int rnd_sample_offset,
                                           NeuralTraversalData traversal_data,
                                           NeuralBVH neural_bvh)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n_elements)
            return;

        glm::uvec2 pixel = glm::uvec2(i % fb_size.x, i / fb_size.x);

        LCGRand lcg = get_lcg_rng(sample_idx, rnd_sample_offset, i);
        float xf    = (float(pixel.x) + lcg_randomf(lcg)) / float(fb_size.x);
        float yf    = (float(pixel.y) + lcg_randomf(lcg)) / float(fb_size.y);

        Ray3f ray = camera_raygen(xf, yf);

        float leaf_mint = 0.f;
        float leaf_maxt = 0.f;

        uint32_t neural_node_idx[1];
        bool valid_its = neural_bvh.intersect_closest_node(
            bvh_const.rendered_lod, ray, leaf_mint, leaf_maxt, neural_node_idx, bvh_const.node_extent_dilation);
        assert(!valid_its || *neural_node_idx < neural_bvh.num_nodes);

#define DEBUG_NODE_IDX
        accumulate_pixel_value(
            sample_idx,
            pixel,
#ifndef DEBUG_NODE_IDX
            valid_its ? (ray(leaf_mint) - neural_scene_bbox.min) / neural_scene_bbox.extent() : glm::vec3(0.f),
#else
            valid_its ? lcg_deterministic_vec3(*neural_node_idx) : glm::vec3(0.f),
#endif
            valid_its ? 1.0f : 0.f);
    }

    NTWR_KERNEL void update_neural_tree_cut_loss(const uint32_t n_elements,
                                                 const uint32_t stride,
                                                 const uint32_t dims,
                                                 float *__restrict__ raw_losses,
                                                 uint32_t *__restrict__ sampled_node_indices,
                                                 NeuralTreeCutData *neural_tree_cut_data,
                                                 uint32_t batch_size)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n_elements)
            return;

        const uint32_t intra_elem_idx = i % stride;
        const uint32_t inter_elem_idx = i / stride;
        if (intra_elem_idx >= dims) {
            return;
        }

        assert(inter_elem_idx < batch_size);

        const float channel_loss = raw_losses[i];

        const uint32_t node_idx = sampled_node_indices[inter_elem_idx];

        NeuralTreeCutData &node_data = neural_tree_cut_data[node_idx];

        atomicAdd(&node_data.loss, channel_loss);
        atomicAdd(&node_data.samples, 1);
    }

    NTWR_KERNEL void update_neural_tree_cut_decisions(const uint32_t n_elements,
                                                      NeuralBVH neural_bvh,
                                                      NeuralTreeCutData *neural_tree_cut_data,
                                                      float *max_node_losses,
                                                      uint32_t *split_neural_node_indices,
                                                      uint32_t max_node_loss_idx,
                                                      bool only_set_split_node_index)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n_elements)
            return;

        // Interior nodes can't be split
        if (neural_bvh.nodes[i].has_neural_child()) {
            return;
        }
        // Since we assign lods in order those are the only possibilities
        assert(!neural_bvh.nodes[i].has_neural_child() ||
               neural_bvh.nodes[i].is_leaf_at_lod(bvh_const.current_learned_lod) ||
               neural_bvh.nodes[i].is_leaf_at_lod(bvh_const.current_learned_lod + 1));

        NeuralTreeCutData &node_data = neural_tree_cut_data[i];

        if (max_node_loss_idx == 0) {
            // Update just once
            node_data.iter_alive += 1;
        }

        // leaf nodes that are already at the maximal depth can't be split
        if (neural_bvh.nodes[i].is_final_leaf()) {
            // Don't set the loss to 0 as this would make the training never sample that node with russian roulette
            // node_data.loss = 0.f;
            return;
        }

        // float max_candidate = node_data.loss / node_data.iter_alive;
        float max_candidate = node_data.compute_loss_heuristic();

        if (max_node_loss_idx > 0) {
            float prev_iter_max = max_node_losses[max_node_loss_idx - 1];
            if (max_candidate >= prev_iter_max) {
                // This max value was already found in the previous iteration
                // Set the correct split node index for that one
                if (max_candidate == prev_iter_max) {
                    split_neural_node_indices[max_node_loss_idx - 1] = i;
                }
                return;
            }
        }
        if (only_set_split_node_index)
            return;

        atomicMax(&max_node_losses[max_node_loss_idx], max_candidate);

        return;
    }

    NTWR_KERNEL void show_neural_tree_cut_data(uint32_t n_elements,
                                               int sample_idx,
                                               int rnd_sample_offset,
                                               NeuralBVH neural_bvh,
                                               NeuralTreeCutData *neural_tree_cut_data,
                                               float max_loss,
                                               bool show_tree_cut_depth,
                                               uint32_t neural_tree_cut_max_depth)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n_elements)
            return;

        show_neural_tree_cut_data_impl(i,
                                       sample_idx,
                                       rnd_sample_offset,
                                       neural_bvh,
                                       neural_tree_cut_data,
                                       max_loss,
                                       show_tree_cut_depth,
                                       neural_tree_cut_max_depth);
    }

}
}