

#include "cuda/neural/neural_bvh_module_kernels.cuh"
#include "cuda/neural/neural_bvh_renderer_kernels.cuh"

#include "cuda/neural/prefiltering/prefiltering_inference_kernels.cuh"
#include "cuda/neural/prefiltering/prefiltering_module.h"

namespace ntwr {

namespace neural::prefiltering {

    float *NeuralPrefilteringModule::optim_leaf_mints()
    {
        return m_neural_learning_results.leaf_mint;
    };
    float *NeuralPrefilteringModule::optim_leaf_maxts()
    {
        return m_neural_learning_results.leaf_maxt;
    };

    void NeuralPrefilteringModule::generate_and_fill_bvh_output_reference(
        uint32_t training_step,
        uint32_t batch_size,
        uint32_t log2_bvh_max_traversals,
        const NeuralBVH &neural_bvh,
        const CudaSceneData &scene_data,
        const CUDABuffer<uint32_t> &sampled_node_indices,
        float current_max_loss,
        const CUDABuffer<NeuralTreeCutData> &neural_tree_cut_data)
    {
        cudaStream_t default_stream = 0;
        uint32_t log2_batch_size    = (int)std::log2(batch_size);
        assert(1 << log2_batch_size == batch_size);
        uint32_t sub_batch_size = 1 << (log2_batch_size - log2_bvh_max_traversals);
        uint32_t n_sub_batches  = batch_size / sub_batch_size;

        tcnn::linear_kernel(granular_sample_bvh_network_input_data,
                            0,
                            default_stream,
                            sub_batch_size,
                            n_sub_batches,
                            training_step,
                            neural_bvh,
                            scene_data,
                            sampled_node_indices.d_pointer(),
                            m_model->training_input_batch.data(),
                            m_model->training_target_batch.data(),
                            current_max_loss,
                            neural_tree_cut_data.d_pointer());
        CUDA_SYNC_CHECK();
    }

    void NeuralPrefilteringModule::neural_bvh_traversal(uint32_t accumulated_spp,
                                                        const NeuralBVH &neural_bvh,
                                                        uint32_t n_active_rays,
                                                        uint32_t *its_counter)
    {
        cudaStream_t default_stream = 0;

        NeuralWavefrontData<NeuralTraversalResult> *bounce_data      = &m_wavefront_neural_data[0];
        NeuralWavefrontData<NeuralTraversalResult> *bounce_data_prev = &m_wavefront_neural_data[1];

        if (its_counter) {
            CUDA_CHECK(cudaMemsetAsync(its_counter, 0, sizeof(uint32_t), default_stream));
        }

        for (uint32_t traversal_step = 0; traversal_step <= (uint32_t)m_max_traversal_steps; traversal_step++) {
            if (n_active_rays == 0) {
                return;
            }

            // Prepare right sized inference buffers
            uint32_t n_padded_width = tcnn::next_multiple(n_active_rays, tcnn::batch_size_granularity);

            tcnn::GPUMatrix<float> visibility_inputs(
                m_inference_input_data->network_inputs, m_bvh_module_const.input_network_dims, n_padded_width);
            tcnn::GPUMatrix<float> visibility_outputs(
                m_inference_output_data->network_outputs, m_pt_target_const.output_network_dims, n_padded_width);

            // Fill network input
            tcnn::linear_kernel(encode_bvh_network_input_data,
                                0,
                                default_stream,
                                n_active_rays,
                                bounce_data->traversal_data,
                                visibility_inputs.data(),
                                true);

            CUDA_SYNC_CHECK();

            // Compute Inference
            m_model->network->inference(default_stream, visibility_inputs, visibility_outputs);
            CUDA_SYNC_CHECK();

            // check visibility : if closest hit found -> splat, otherwise trace next leaf bounce

            tcnn::linear_kernel(process_bvh_network_output_and_intersect_neural_bvh,
                                0,
                                default_stream,
                                n_active_rays,
                                traversal_step,
                                visibility_outputs.data(),
                                bounce_data->traversal_data,
                                bounce_data->results,
                                neural_bvh);
            CUDA_SYNC_CHECK();

            // Prepare for compaction
            bounce_data      = &m_wavefront_neural_data[(traversal_step + 1) % 2];
            bounce_data_prev = &m_wavefront_neural_data[traversal_step % 2];
            // Compact queries
            {
                CUDA_CHECK(cudaMemsetAsync(m_active_rays_counter, 0, sizeof(uint32_t), default_stream));
                tcnn::linear_kernel(compact_neural_data_and_accumulate,
                                    0,
                                    default_stream,
                                    n_active_rays,
                                    accumulated_spp,
                                    *bounce_data_prev,
                                    *bounce_data,
                                    m_hybrid_intersection_data,
                                    m_active_rays_counter,
                                    its_counter,
                                    traversal_step == m_max_traversal_steps);
                CUDA_CHECK(cudaMemcpyAsync(
                    &n_active_rays, m_active_rays_counter, sizeof(uint32_t), cudaMemcpyDeviceToHost, default_stream));
                CUDA_CHECK(cudaStreamSynchronize(default_stream));
            }
        }
    }

    void NeuralPrefilteringModule::neural_prefiltering(uint32_t accumulated_spp,
                                                       uint32_t sample_offset,
                                                       const NeuralBVH &neural_bvh)
    {
        cudaStream_t default_stream = 0;

        uint32_t n_active_rays = m_cuda_backend->framebuffer_size_flat();

        CUDA_CHECK(cudaMemsetAsync(m_active_rays_counter, 0, sizeof(uint32_t), default_stream));

        // Populate rays and results with first traversal step
        tcnn::linear_kernel(camera_raygen_and_intersect_neural_bvh,
                            0,
                            default_stream,
                            n_active_rays,
                            accumulated_spp,
                            sample_offset,
                            m_wavefront_neural_data[0].traversal_data,
                            m_wavefront_neural_data[0].results,
                            neural_bvh,
                            m_active_rays_counter);
        CUDA_CHECK(cudaMemcpyAsync(
            &n_active_rays, m_active_rays_counter, sizeof(uint32_t), cudaMemcpyDeviceToHost, default_stream));
        CUDA_CHECK(cudaStreamSynchronize(default_stream));

        for (uint32_t depth = 0; depth < m_max_depth; depth++) {
            // One neural bounce
            neural_bvh_traversal(accumulated_spp, neural_bvh, n_active_rays, m_its_counter);
            CUDA_CHECK(cudaMemcpy(&n_active_rays, m_its_counter, sizeof(uint32_t), cudaMemcpyDeviceToHost));

            if (n_active_rays == 0) {
                return;
            }

            // Fill in throughput inference data
            uint32_t n_padded_width = tcnn::next_multiple(n_active_rays, tcnn::batch_size_granularity);

            tcnn::GPUMatrix<float> appearance_inputs(
                m_appearance_input_data, m_prefiltering_module_const.prefiltering_input_dims, n_padded_width);
            tcnn::GPUMatrix<float> appearance_outputs(
                m_appearance_output_data, m_prefiltering_module_const.prefiltering_output_dims, n_padded_width);

            // Fill network input and update hit position
            tcnn::linear_kernel(generate_throughput_inference_data_and_update_ray,
                                0,
                                default_stream,
                                n_active_rays,
                                m_hybrid_intersection_data,
                                appearance_inputs.data());
            CUDA_SYNC_CHECK();

            // Inference for appearance

            m_appearance_model->network->inference(default_stream, appearance_inputs, appearance_outputs);
            CUDA_SYNC_CHECK();

            // Scatter ray
            CUDA_CHECK(cudaMemsetAsync(m_active_rays_counter, 0, sizeof(uint32_t), default_stream));
            tcnn::linear_kernel(neural_prefiltering_scatter,
                                0,
                                default_stream,
                                n_active_rays,
                                accumulated_spp,
                                depth,
                                m_hybrid_intersection_data,
                                m_wavefront_neural_data[0].traversal_data,
                                m_wavefront_neural_data[0].results,
                                neural_bvh,
                                appearance_outputs.data(),
                                m_active_rays_counter);
            CUDA_CHECK(cudaMemcpyAsync(
                &n_active_rays, m_active_rays_counter, sizeof(uint32_t), cudaMemcpyDeviceToHost, default_stream));
            CUDA_CHECK(cudaStreamSynchronize(default_stream));
        }

        tcnn::linear_kernel(accumulate_neural_results,
                            0,
                            default_stream,
                            n_active_rays,
                            accumulated_spp,
                            m_wavefront_neural_data[0].traversal_data);
    }

    void NeuralPrefilteringModule::appearance_network_debug(uint32_t accumulated_spp,
                                                            uint32_t sample_offset,
                                                            const NeuralBVH &neural_bvh)
    {
        CudaSceneData scene_data    = NeuralPrefilteringModule::scene_data();
        cudaStream_t default_stream = 0;

        uint32_t n_active_rays = m_cuda_backend->framebuffer_size_flat();

        CUDA_CHECK(cudaMemsetAsync(m_active_rays_counter, 0, sizeof(uint32_t), default_stream));

        tcnn::linear_kernel(camera_raygen_and_first_hit,
                            0,
                            default_stream,
                            n_active_rays,
                            accumulated_spp,
                            sample_offset,
                            m_hybrid_intersection_data,
                            scene_data,
                            m_active_rays_counter);

        CUDA_CHECK(cudaMemcpyAsync(
            &n_active_rays, m_active_rays_counter, sizeof(uint32_t), cudaMemcpyDeviceToHost, default_stream));
        CUDA_CHECK(cudaStreamSynchronize(default_stream));

        if (n_active_rays == 0) {
            return;
        }

        // Fill in throughput inference data
        uint32_t n_padded_width = tcnn::next_multiple(n_active_rays, tcnn::batch_size_granularity);

        tcnn::GPUMatrix<float> appearance_inputs(
            m_appearance_input_data, m_prefiltering_module_const.prefiltering_input_dims, n_padded_width);
        tcnn::GPUMatrix<float> appearance_outputs(
            m_appearance_output_data, m_prefiltering_module_const.prefiltering_output_dims, n_padded_width);

        // Fill network input
        tcnn::linear_kernel(generate_throughput_inference_data_and_update_ray,
                            0,
                            default_stream,
                            n_active_rays,
                            m_hybrid_intersection_data,
                            appearance_inputs.data());

        CUDA_SYNC_CHECK();

        // Compute Inference
        m_appearance_model->network->inference(default_stream, appearance_inputs, appearance_outputs);
        CUDA_SYNC_CHECK();

        // Get back inference results

        tcnn::linear_kernel(process_prefiltering_network_output,
                            0,
                            default_stream,
                            n_active_rays,
                            m_hybrid_intersection_data,
                            m_wavefront_neural_data[0].traversal_data,
                            appearance_outputs.data());

        // Splat throughput

        tcnn::linear_kernel(accumulate_neural_results,
                            0,
                            default_stream,
                            n_active_rays,
                            accumulated_spp,
                            m_wavefront_neural_data[0].traversal_data);
    }

    void NeuralPrefilteringModule::neural_render_module(uint32_t accumulated_spp,
                                                        uint32_t sample_offset,
                                                        const NeuralBVH &neural_bvh)
    {
        switch (m_prefiltering_module_const.output_mode) {
        case OutputMode::NeuralPrefiltering:
            neural_prefiltering(accumulated_spp, sample_offset, neural_bvh);
            break;
        case OutputMode::AppearanceNetworkDebug:
            appearance_network_debug(accumulated_spp, sample_offset, neural_bvh);
            break;
        default:
            UNREACHABLE();
        }
    }
}

}