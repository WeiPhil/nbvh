

#include "cuda/neural/neural_bvh_renderer.h"
#include "cuda/neural/neural_bvh_renderer_kernels.cuh"

#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/config.h>
#include <tiny-cuda-nn/optimizers/adam.h>

#include <cudaProfiler.h>

#include "FLIP.h"

using precision_t = tcnn::network_precision_t;

namespace ntwr {
namespace neural {

    void NeuralBVHRenderer::update_loss_graph(float loss)
    {
        m_loss_graph[m_loss_graph_samples++ % m_loss_graph.size()] = std::log(loss);
    }

    CudaSceneData NeuralBVHRenderer::learning_scene_data()
    {
        CudaSceneData data;
        data.tlas      = m_backend->m_cuda_tlas;
        data.materials = m_backend->m_cuda_material_buffer.d_pointer();
        data.textures  = m_backend->m_cuda_textures_buffer.d_pointer();

        return data;
    }

    //  Pixels per degree (PPD).
    inline float calculatePPD(const float dist, const float resolutionX, const float monitorWidth)
    {
        return dist * (resolutionX / monitorWidth) * (float(FLIP::PI) / 180.0f);
    }

    float NeuralBVHRenderer::compute_and_display_flip_error()
    {
        struct {
            float PPD                = 0;        // If PPD==0.0, then it will be computed from the parameters below.
            float monitorDistance    = 0.7f;     // Unit: meters.
            float monitorWidth       = 0.7f;     // Unit: meters.
            float monitorResolutionX = 3840.0f;  // Unit: pixels.
        } gFLIPOptions;

        FLIP::image<FLIP::color3> magmaMap(FLIP::MapMagma, 256);
        FLIP::image<float> errorMapFLIP(m_backend->framebuffer_size().x, m_backend->framebuffer_size().y, 0.0f);

        FLIP::image<FLIP::color3> reference(m_backend->framebuffer_size().x, m_backend->framebuffer_size().y);
        FLIP::image<FLIP::color3> test(m_backend->framebuffer_size().x, m_backend->framebuffer_size().y);

        // accum buffer to flip compatible image
        {
            reference.synchronizeDevice();
            test.synchronizeDevice();
            copy_framebuffers_to_flip_image<<<reference.getGridDim(), reference.getBlockDim()>>>(
                m_accumulated_spp + 1, reference.getDeviceData(), test.getDeviceData(), reference.getDimensions());
            CUDA_SYNC_CHECK();
            reference.setState(FLIP::CudaTensorState::DEVICE_ONLY);
            test.setState(FLIP::CudaTensorState::DEVICE_ONLY);
        }
        reference.LinearRGB2sRGB();
        test.LinearRGB2sRGB();

        gFLIPOptions.PPD =
            calculatePPD(gFLIPOptions.monitorDistance, gFLIPOptions.monitorResolutionX, gFLIPOptions.monitorWidth);

        // Compute flip
        errorMapFLIP.FLIP(reference, test, gFLIPOptions.PPD);

        pooling<float> pooledValues;
        for (int y = 0; y < errorMapFLIP.getHeight(); y++) {
            for (int x = 0; x < errorMapFLIP.getWidth(); x++) {
                pooledValues.update(x, y, errorMapFLIP.get(x, y));
            }
        }
        FLIP::image<FLIP::color3> ldr_flip(reference.getWidth(), reference.getHeight());
        ldr_flip.copyFloat2Color3(errorMapFLIP);
        ldr_flip.colorMap(errorMapFLIP, magmaMap);
        // colormapped flip error to framebuffer
        {
            test.synchronizeDevice();
            flip_image_to_framebuffer<<<ldr_flip.getGridDim(), ldr_flip.getBlockDim()>>>(ldr_flip.getDeviceData(),
                                                                                         ldr_flip.getDimensions());
            CUDA_SYNC_CHECK();
        }

        return pooledValues.getMean();
    }

    void NeuralBVHRenderer::reset_learning_data()
    {
        m_all_bvh_lods_assigned = false;

        // Reset training info
        m_tmp_loss           = 0;
        m_training_step      = 0;
        m_tmp_loss_counter   = 0;
        m_loss_graph_samples = 0;
        m_loss_graph         = std::vector<float>(10000, 0.0f);

        reinitialise_neural_bvh_learning_states();
        reinitialise_tree_cut_learning_data(m_bvh_split_scheduler.m_start_split_num);

        m_bvh_split_scheduler.update_stats_and_reset(m_neural_bvh.num_nodes, m_splits_graph, m_max_training_steps);
        m_neural_module->reinitialise_neural_model(m_batch_size);
        m_reset_learning_data = false;
    }

    void NeuralBVHRenderer::render(bool reset_accumulation)
    {
        if (reset_accumulation || m_continuous_reset) {
            CUDA_SYNC_CHECK();

            m_sample_offset += m_accumulated_spp;
            m_accumulated_spp = 0;
        }

        if (m_reset_inference_data) {
            CUDA_SYNC_CHECK();

            m_neural_module->reinitialise_inference_data(m_backend->m_fb_size);
            m_reset_inference_data = false;
        }

        if (m_reset_learning_data) {
            CUDA_SYNC_CHECK();

            if (m_rebuild_neural_bvh || m_tree_cut_learning) {
                build_and_upload_neural_bvh();
                m_rebuild_neural_bvh = false;
            }
            reset_learning_data();
        }

        setup_constants();

        if (m_run_optimisation) {
            if (m_training_step < m_max_training_steps) {
                m_learning_timer.start_frame(m_training_step);

                CudaSceneData scene_data = NeuralBVHRenderer::learning_scene_data();

                float current_max_loss = m_training_step == 0 ? 0.f : m_max_node_loss_host;
                m_neural_module->optimisation_data_generation(m_training_step,
                                                              m_batch_size,
                                                              m_bvh_const.log2_bvh_max_traversals,
                                                              m_neural_bvh,
                                                              scene_data,
                                                              m_sampled_node_indices,
                                                              current_max_loss,
                                                              m_neural_tree_cut_data);

                float current_network_loss   = 0.f;
                bool should_split_neural_bvh = false;

                // Perform training step and potential bvh split and lod assignment
                {
                    if (m_tree_cut_learning) {
                        int num_bvh_splits = m_bvh_split_scheduler.get_bvh_splits(m_training_step);
                        if (num_bvh_splits != m_current_num_bvh_split) {
                            // Min 1 to avoid 0 sized-allocations
                            reinitialise_tree_cut_learning_data(max(1, num_bvh_splits));
                        }
                        should_split_neural_bvh = num_bvh_splits > 0;
                    }

                    cudaStream_t default_stream = 0;
                    auto ctx                    = m_neural_module->m_model->trainer->training_step(
                        default_stream,
                        m_neural_module->m_model->training_input_batch,
                        m_neural_module->m_model->training_target_batch);
                    CUDA_SYNC_CHECK();

// #define CHECK_ZERO_GRADIENTS
#ifdef CHECK_ZERO_GRADIENTS
                    auto adam_optimiser = std::dynamic_pointer_cast<tcnn::AdamOptimizer<tcnn::network_precision_t>>(
                        m_neural_module->m_model->optimizer->nested());

                    auto json_data                       = adam_optimiser->serialize();
                    tcnn::GPUMemory<float> first_moments = json_data["first_moments_binary"];
                    std::vector<float> first_moments_host(first_moments.size());
                    first_moments.copy_to_host(first_moments_host);
                    uint32_t zero_gradients = 0;
                    for (size_t i = 0; i < m_neural_module->m_model->trainer->n_params(); i++) {
                        if (first_moments_host[i] == 0.f) {
                            zero_gradients += 1;
                        }
                    }
                    logger(LogLevel::Info,
                           "%d gradients are 0, that is %.3f%% are non-zero",
                           zero_gradients,
                           100.f - (100.f * zero_gradients / (float)adam_optimiser->n_weights()));
#endif

                    // After computing the loss for every input/ref distribute the losses into every BVH Node loss
                    tcnn::linear_kernel(update_neural_tree_cut_loss,
                                        0,
                                        default_stream,
                                        ctx->L.n_elements(),
                                        ctx->L.m(),
                                        m_neural_module->network_output_dims(),
                                        ctx->L.data(),
                                        m_sampled_node_indices.d_pointer(),
                                        m_neural_tree_cut_data.d_pointer(),
                                        m_batch_size);
                    CUDA_SYNC_CHECK();

                    CUDA_CHECK(
                        cudaMemsetAsync(m_max_node_losses, 0, m_current_num_bvh_split * sizeof(float), default_stream));
                    CUDA_CHECK(cudaMemsetAsync(
                        m_split_neural_node_indices, 0, m_current_num_bvh_split * sizeof(uint32_t), default_stream));
                    for (uint32_t i = 0; i <= m_current_num_bvh_split; i++) {
                        tcnn::linear_kernel(update_neural_tree_cut_decisions,
                                            0,
                                            default_stream,
                                            m_neural_bvh.num_nodes,
                                            m_neural_bvh,
                                            m_neural_tree_cut_data.d_pointer(),
                                            m_max_node_losses,
                                            m_split_neural_node_indices,
                                            i,
                                            i == m_current_num_bvh_split);
                    }
                    CUDA_SYNC_CHECK();
                    CUDA_CHECK(cudaMemcpyAsync(&m_max_node_loss_host,
                                               m_max_node_losses,
                                               sizeof(float),
                                               cudaMemcpyDeviceToHost,
                                               default_stream));
                    CUDA_CHECK(cudaMemcpyAsync(m_split_neural_node_indices_host.data(),
                                               m_split_neural_node_indices,
                                               m_current_num_bvh_split * sizeof(uint32_t),
                                               cudaMemcpyDeviceToHost,
                                               default_stream));
                    CUDA_CHECK(cudaStreamSynchronize(default_stream));

                    current_network_loss = m_neural_module->m_model->trainer->loss(default_stream, *ctx);
                }

                update_loss_graph(current_network_loss);
                m_tmp_loss += current_network_loss;
                ++m_tmp_loss_counter;

                if (!m_using_validation_mode && m_training_step % 100 == 0) {
                    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
                    // logger(LogLevel::Info,
                    //        "Training Step #%d: average loss=%f time=%f[s]",
                    //        m_training_step,
                    //        m_tmp_loss / static_cast<float>(m_tmp_loss_counter),
                    //        static_cast<float>(
                    //            std::chrono::duration_cast<std::chrono::microseconds>(end - m_training_begin_time)
                    //                .count()) *
                    //            1e-6);

                    m_tmp_loss         = 0;
                    m_tmp_loss_counter = 0;

                    m_training_begin_time = std::chrono::steady_clock::now();
                }

                if (m_tree_cut_learning) {
                    if (should_split_neural_bvh) {
                        split_neural_bvh_at(m_split_neural_node_indices_host);
                    }
                    if (m_bvh_const.num_lods_learned > 1 && !m_all_bvh_lods_assigned) {
                        if (m_bvh_split_scheduler.should_assign_new_lod(m_training_step,
                                                                        m_bvh_const.current_learned_lod,
                                                                        m_bvh_const.num_lods_learned,
                                                                        NUM_BVH_LODS)) {
                            assign_lod_to_current_neural_leaf_nodes();
                        }
                        if (m_all_bvh_lods_assigned) {
                            // Make sure we set the number of splits to process to 1 (we still need the max loss error)
                            // when all LoDs have been assigned
                            reinitialise_tree_cut_learning_data(1);
                            should_split_neural_bvh = false;
                        }
                    }
                }

                m_learning_timer.end_frame();

                m_training_step++;
            }
            // Automatically turn of optimisation when done
            if (m_training_step >= m_max_training_steps) {
                m_run_optimisation = false;

                if (m_using_validation_mode) {
                    process_validation_config_end();
                }
            }
        }

        if (m_run_inference && (m_accumulated_spp < m_max_accumulated_spp)) {
            m_inference_timer.start_frame(m_accumulated_spp);

            if (m_show_bvh) {
                cudaStream_t default_stream = 0;
                tcnn::linear_kernel(show_bhv_closest_node,
                                    0,
                                    default_stream,
                                    m_backend->framebuffer_size_flat(),
                                    m_accumulated_spp,
                                    m_sample_offset,
                                    m_neural_module->m_inference_input_data->neural_traversal_data[0],
                                    m_neural_bvh);
                CUDA_SYNC_CHECK();

            } else if (m_show_neural_node_losses) {
                cudaStream_t default_stream = 0;
                CUDA_CHECK(cudaMemsetAsync(m_mean_error, 0, sizeof(float), default_stream));
                tcnn::linear_kernel(show_neural_tree_cut_data,
                                    0,
                                    default_stream,
                                    m_backend->framebuffer_size_flat(),
                                    m_accumulated_spp,
                                    m_sample_offset,
                                    m_neural_bvh,
                                    m_neural_tree_cut_data.d_pointer(),
                                    m_max_node_loss_host,
                                    m_show_tree_cut_depth,
                                    m_neural_tree_cut_max_depth);
                CUDA_SYNC_CHECK();

            } else if (m_split_screen_neural_node_losses) {
                m_neural_module->split_screen_neural_render_node_loss(m_accumulated_spp,
                                                                      m_sample_offset,
                                                                      m_neural_bvh,
                                                                      m_neural_tree_cut_data,
                                                                      m_max_node_loss_host,
                                                                      m_show_tree_cut_depth,
                                                                      m_neural_tree_cut_max_depth);
            } else {
                m_neural_module->neural_render(m_accumulated_spp, m_sample_offset, m_neural_bvh);

                if (m_postprocess_mode != PostprocessMode::None) {
                    if (m_postprocess_mode == PostprocessMode::FLIP) {
                        m_mean_error_host = compute_and_display_flip_error();
                    } else {
                        cudaStream_t default_stream = 0;
                        CUDA_CHECK(cudaMemsetAsync(m_mean_error, 0, sizeof(float), default_stream));
                        tcnn::linear_kernel(compute_and_display_render_error,
                                            0,
                                            default_stream,
                                            m_backend->framebuffer_size_flat(),
                                            m_accumulated_spp + 1,
                                            m_postprocess_mode,
                                            m_mean_error,
                                            false);
                        CUDA_CHECK(cudaMemcpyAsync(
                            &m_mean_error_host, m_mean_error, sizeof(float), cudaMemcpyDeviceToHost, default_stream));
                        CUDA_SYNC_CHECK();
                        m_mean_error_host /= (float)m_backend->framebuffer_size_flat();
                    }
                }
            }

            m_inference_timer.end_frame();

            m_accumulated_spp++;
            // Automatically stop inference
            if (m_accumulated_spp == m_max_accumulated_spp) {
                if (m_backend->m_cuda_scene_constants.ema_accumulate) {
                    // Restart EMA weighting instead of stopping inference
                    m_accumulated_spp = 0;
                } else {
                    m_run_inference = false;
                }

                if (m_using_validation_mode) {
                    process_validation_config_end();
                }
            }
        }
    }
}
}