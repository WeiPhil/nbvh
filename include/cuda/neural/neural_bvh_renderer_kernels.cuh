#pragma once

#include "cuda/neural/neural_bvh_renderer.h"
#include "cuda/utils/colors.cuh"

namespace FLIP {
struct color3;
}

#ifndef WavefrontPixelIndex
#define WavefrontPixelIndex \
    (threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * (blockIdx.x + gridDim.x * blockIdx.y)))
#endif

#ifndef WavefrontPixelID
#define WavefrontPixelID                                                                                       \
    ((glm::uvec2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y) &               \
      ~glm::uvec2(0x3 << 3, 0x3)) +                                                                            \
     (glm::uvec2((blockIdx.y * blockDim.y + threadIdx.y) << 3, (blockIdx.x * blockDim.x + threadIdx.x) >> 3) & \
      glm::uvec2(0x3 << 3, 0x3)))
#endif

namespace ntwr {
namespace neural {

    NTWR_KERNEL void camera_raygen(int sample_idx, int rnd_sample_offset, CudaBVHTraversalData traversal_data);

    NTWR_KERNEL void accumulate_results(int sample_idx, glm::vec3 *illumination);

    NTWR_KERNEL void copy_framebuffers_to_flip_image(int sample_idx,
                                                     FLIP::color3 *reference,
                                                     FLIP::color3 *test,
                                                     const int3 dim);

    NTWR_KERNEL void flip_image_to_framebuffer(FLIP::color3 *image, const int3 dim);

    NTWR_KERNEL void compute_and_display_render_error(const uint32_t n_elements,
                                                      int sample_idx,
                                                      PostprocessMode postprocess_mode,
                                                      float *mean_error,
                                                      bool false_color = false);

    NTWR_KERNEL void show_bhv_closest_node(uint32_t n_elements,
                                           int sample_idx,
                                           int rnd_sample_offset,
                                           NeuralTraversalData traversal_data,
                                           NeuralBVH neural_bvh);

    NTWR_KERNEL void update_neural_tree_cut_loss(const uint32_t n_elements,
                                                 const uint32_t stride,
                                                 const uint32_t dims,
                                                 float *__restrict__ raw_losses,
                                                 uint32_t *__restrict__ sampled_node_indices,
                                                 NeuralTreeCutData *neural_tree_cut_data,
                                                 uint32_t batch_size);

    NTWR_KERNEL void update_neural_tree_cut_decisions(const uint32_t n_elements,
                                                      NeuralBVH neural_bvh,
                                                      NeuralTreeCutData *neural_tree_cut_data,
                                                      float *max_node_losses,
                                                      uint32_t *split_neural_node_indices,
                                                      uint32_t max_node_loss_idx,
                                                      bool only_set_split_node_index);

    NTWR_KERNEL void show_neural_tree_cut_data(uint32_t n_elements,
                                               int sample_idx,
                                               int rnd_sample_offset,
                                               NeuralBVH neural_bvh,
                                               NeuralTreeCutData *neural_tree_cut_data,
                                               float max_loss,
                                               bool m_show_tree_cut_depth,
                                               uint32_t max_tree_cut_depth);

    NTWR_DEVICE Ray3f camera_raygen(float xf, float yf)
    {
        glm::vec3 dir = normalize(xf * camera_params.du + yf * camera_params.dv + camera_params.dir_top_left);

        return Ray3f(camera_params.origin, dir);
    }

    NTWR_DEVICE void accumulate_pixel_value(int sample_idx,
                                            const glm::uvec2 &pixel_coord,
                                            const glm::vec3 &pixel_value,
                                            float pixel_alpha)
    {
        glm::vec4 alpha_pixel_value = glm::vec4(pixel_value, pixel_alpha);
        glm::vec4 accum_color       = glm::vec4(0.0f, 0.f, 0.f, 0.f);
        if (sample_idx > 0) {
            float4 data;
            surf2Dread(&data, accum_buffer, pixel_coord.x * 4 * sizeof(float), pixel_coord.y, cudaBoundaryModeZero);
            accum_color = glm::vec4(data.x, data.y, data.z, data.w);
        }

        if (scene_constants.ema_accumulate) {
            if (pixel_alpha == 0.f && sample_idx > 0) {
                return;
            }
            if (sample_idx > 0) {
                accum_color =
                    alpha_pixel_value * scene_constants.ema_weight + accum_color * (1 - scene_constants.ema_weight);
            } else {
                accum_color = alpha_pixel_value;
            }

            float4 data = {accum_color.x, accum_color.y, accum_color.z, accum_color.w};
            surf2Dwrite(data, accum_buffer, pixel_coord.x * 4 * sizeof(float), pixel_coord.y, cudaBoundaryModeZero);

        } else {
            accum_color += alpha_pixel_value;
            {
                float4 data = {accum_color.x, accum_color.y, accum_color.z, accum_color.w};
                surf2Dwrite(data, accum_buffer, pixel_coord.x * 4 * sizeof(float), pixel_coord.y, cudaBoundaryModeZero);
            }
            accum_color /= float(sample_idx + 1);
        }

        glm::vec3 ldr_accum_color = linear_to_srgb(accum_color);
        {
            // int alpha_ldr = scene_constants.black_as_transparent && accum_color.w == 0.f ? 0 : 255;
            uchar4 data;
            data.x = glm::clamp(int(ldr_accum_color.x * 256.0f), 0, 255);
            data.y = glm::clamp(int(ldr_accum_color.y * 256.0f), 0, 255);
            data.z = glm::clamp(int(ldr_accum_color.z * 256.0f), 0, 255);
            data.w = 255;
            surf2Dwrite(data, framebuffer, pixel_coord.x * 4, pixel_coord.y, cudaBoundaryModeZero);
        }
    }

    NTWR_DEVICE bool show_neural_tree_cut_data_impl(uint32_t i,
                                                    int sample_idx,
                                                    int rnd_sample_offset,
                                                    const NeuralBVH &neural_bvh,
                                                    NeuralTreeCutData *neural_tree_cut_data,
                                                    float max_loss,
                                                    bool m_show_tree_cut_depth,
                                                    uint32_t neural_tree_cut_max_depth)
    {
        glm::uvec2 pixel = glm::uvec2(i % fb_size.x, i / fb_size.x);

        LCGRand lcg = get_lcg_rng(sample_idx, rnd_sample_offset, i);
        float xf    = (float(pixel.x) + lcg_randomf(lcg)) / float(fb_size.x);
        float yf    = (float(pixel.y) + lcg_randomf(lcg)) / float(fb_size.y);

        Ray3f ray = camera_raygen(xf, yf);

        float leaf_mint = 0.f;
        float leaf_maxt = 0.f;

        uint32_t neural_node_idx[1] = {0};
        bool valid_its =
            neural_bvh.intersect_closest_node(bvh_const.rendered_lod, ray, leaf_mint, leaf_maxt, neural_node_idx);
        assert(!valid_its || *neural_node_idx < neural_bvh.num_nodes);

        NeuralTreeCutData &node_data = neural_tree_cut_data[*neural_node_idx];
        const float node_area        = neural_bvh.nodes[*neural_node_idx].bbox().area();

        if (node_data.samples > 2e31) {
            printf("too large number of samples soon");
        }

        if (!m_show_tree_cut_depth) {
            // float node_loss = node_data.loss / node_data.iter_alive;
            float node_loss = node_data.samples != 0 ? node_data.loss / node_data.samples : 0.f;
            if (bvh_const.show_loss_heuristic) {
                node_loss = node_data.samples != 0 ? node_data.compute_loss_heuristic() : 0.f;
            } else {
                node_loss = node_data.samples != 0 ? node_data.loss / node_data.samples : 0.f;
            }
            accumulate_pixel_value(sample_idx,
                                   pixel,
                                   valid_its ? turbo_colormap(node_loss / bvh_const.loss_exposure) : glm::vec3(0.f),
                                   valid_its ? 1.0f : 0.f);

        } else {
            accumulate_pixel_value(
                sample_idx,
                pixel,
                valid_its ? plasma_colormap(float(node_data.depth) / neural_tree_cut_max_depth) : glm::vec3(0.f),
                valid_its ? 1.0f : 0.f);
        }
        return valid_its;
    }

}
}