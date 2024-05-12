

#include "cuda/neural/neural_bvh_module_kernels.cuh"
#include "cuda/neural/neural_bvh_renderer_kernels.cuh"

#include "cuda/neural/path/path_inference_kernels.cuh"
#include "cuda/neural/path/path_module.h"

namespace ntwr {

namespace neural::path {

    float *NeuralPathModule::optim_leaf_mints()
    {
        return m_neural_learning_results.leaf_mint;
    };
    float *NeuralPathModule::optim_leaf_maxts()
    {
        return m_neural_learning_results.leaf_maxt;
    };

    void NeuralPathModule::generate_and_fill_bvh_output_reference(
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

    void NeuralPathModule::neural_bvh_traversal(uint32_t accumulated_spp,
                                                const NeuralBVH &neural_bvh,
                                                uint32_t n_neural_active_rays,
                                                uint32_t *its_counter)
    {
        cudaEvent_t start_inference, stop_inference;
        cudaEventCreate(&start_inference);
        cudaEventCreate(&stop_inference);

        cudaStream_t default_stream = 0;

        NeuralWavefrontData<NeuralTraversalResult> *bounce_data      = &m_wavefront_neural_data[0];
        NeuralWavefrontData<NeuralTraversalResult> *bounce_data_prev = &m_wavefront_neural_data[1];

        for (uint32_t traversal_step = 0; traversal_step <= (uint32_t)m_max_traversal_steps; traversal_step++) {
            if (n_neural_active_rays == 0) {
                return;
            }

            // Prepare right sized inference buffers
            uint32_t n_padded_width = tcnn::next_multiple(n_neural_active_rays, tcnn::batch_size_granularity);

            tcnn::GPUMatrix<float> visibility_inputs(
                m_inference_input_data->network_inputs, m_bvh_module_const.input_network_dims, n_padded_width);
            tcnn::GPUMatrix<float> visibility_outputs(
                m_inference_output_data->network_outputs, m_pt_target_const.output_network_dims, n_padded_width);

            // Fill network input
            tcnn::linear_kernel(encode_bvh_network_input_data,
                                0,
                                default_stream,
                                n_neural_active_rays,
                                bounce_data->traversal_data,
                                visibility_inputs.data(),
                                true);

            cudaEventRecord(start_inference);
            // Compute Inference
            m_model->network->inference(default_stream, visibility_inputs, visibility_outputs);
            cudaEventRecord(stop_inference);

            // check visibility : if closest hit found -> splat, otherwise trace next leaf bounce

            tcnn::linear_kernel(process_bvh_network_output_and_intersect_neural_bvh,
                                0,
                                default_stream,
                                n_neural_active_rays,
                                traversal_step,
                                visibility_outputs.data(),
                                bounce_data->traversal_data,
                                bounce_data->results,
                                neural_bvh);

            // Prepare for compaction
            bounce_data      = &m_wavefront_neural_data[(traversal_step + 1) % 2];
            bounce_data_prev = &m_wavefront_neural_data[traversal_step % 2];
            // Compact queries
            {
                CUDA_CHECK(cudaMemsetAsync(m_active_neural_rays_counter, 0, sizeof(uint32_t), default_stream));
                tcnn::linear_kernel(compact_neural_data_and_accumulate,
                                    0,
                                    default_stream,
                                    n_neural_active_rays,
                                    accumulated_spp,
                                    *bounce_data_prev,
                                    *bounce_data,
                                    m_hybrid_intersection_data,
                                    m_active_neural_rays_counter,
                                    its_counter,
                                    traversal_step == m_max_traversal_steps);
                CUDA_CHECK(cudaMemcpyAsync(&n_neural_active_rays,
                                           m_active_neural_rays_counter,
                                           sizeof(uint32_t),
                                           cudaMemcpyDeviceToHost,
                                           default_stream));
                CUDA_CHECK(cudaStreamSynchronize(default_stream));
            }

            cudaEventSynchronize(stop_inference);
            float inference_ms = 0.f;
            cudaEventElapsedTime(&inference_ms, start_inference, stop_inference);
            m_total_inference_ms += inference_ms;
        }
    }

    void NeuralPathModule::neural_path_tracing(uint32_t accumulated_spp,
                                               uint32_t sample_offset,
                                               const NeuralBVH &neural_bvh)
    {
        CudaSceneData scene_data    = NeuralPathModule::scene_data();
        cudaStream_t default_stream = 0;

        cudaStream_t normal_rays_stream;
        CUDA_CHECK(cudaStreamCreate(&normal_rays_stream));

        uint32_t n_all_active_rays = m_cuda_backend->framebuffer_size_flat();

        uint32_t n_neural_active_rays = 0;
        uint32_t n_normal_active_rays = 0;

        CUDA_CHECK(cudaMemsetAsync(m_active_neural_rays_counter, 0, sizeof(uint32_t), default_stream));
        CUDA_CHECK(cudaMemsetAsync(m_active_normal_rays_counter, 0, sizeof(uint32_t), default_stream));

        // Populate rays and results with first traversal step
        tcnn::linear_kernel(hybrid_camera_raygen_and_hybrid_first_hit_and_traversal,
                            0,
                            default_stream,
                            n_all_active_rays,
                            accumulated_spp,
                            sample_offset,
                            m_normal_traversal_data,
                            neural_bvh,
                            m_wavefront_neural_data[0].traversal_data,
                            m_wavefront_neural_data[0].results,
                            scene_data,
                            m_active_neural_rays_counter,
                            m_active_normal_rays_counter);

        CUDA_CHECK(cudaMemcpyAsync(&n_neural_active_rays,
                                   m_active_neural_rays_counter,
                                   sizeof(uint32_t),
                                   cudaMemcpyDeviceToHost,
                                   default_stream));
        CUDA_CHECK(cudaMemcpyAsync(&n_normal_active_rays,
                                   m_active_normal_rays_counter,
                                   sizeof(uint32_t),
                                   cudaMemcpyDeviceToHost,
                                   normal_rays_stream));
        CUDA_CHECK(cudaStreamSynchronize(default_stream));

        if (n_normal_active_rays == 0 && n_neural_active_rays == 0) {
            return;
        }

        int tlas_traversal_step = 0;
        while (true) {
            CUDA_CHECK(cudaMemsetAsync(m_hybrid_its_counter, 0, sizeof(uint32_t), default_stream));

            tcnn::linear_kernel(non_neural_bvh_intersection,
                                0,
                                normal_rays_stream,
                                n_normal_active_rays,
                                accumulated_spp,
                                m_normal_traversal_data,
                                m_hybrid_intersection_data,
                                scene_data,
                                m_hybrid_its_counter);

            // One neural bounce
            neural_bvh_traversal(accumulated_spp, neural_bvh, n_neural_active_rays, m_hybrid_its_counter);

            // Synchronize all streams
            CUDA_SYNC_CHECK();

            uint32_t n_active_hybrid_intersection = 0;

            CUDA_CHECK(cudaMemcpyAsync(
                &n_active_hybrid_intersection, m_hybrid_its_counter, sizeof(uint32_t), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemsetAsync(m_active_neural_rays_counter, 0, sizeof(uint32_t), default_stream));
            CUDA_CHECK(cudaMemsetAsync(m_active_normal_rays_counter, 0, sizeof(uint32_t), default_stream));

            bool last_tlas_traversal = (tlas_traversal_step == (m_max_tlas_traversal - 1));
            tcnn::linear_kernel(hybrid_scatter_and_get_next_tlas_traversal_step,
                                0,
                                default_stream,
                                n_active_hybrid_intersection,
                                accumulated_spp,
                                m_hybrid_intersection_data,
                                m_wavefront_neural_data[0].traversal_data,
                                m_wavefront_neural_data[0].results,
                                m_normal_traversal_data,
                                neural_bvh,
                                scene_data,
                                m_active_neural_rays_counter,
                                m_active_normal_rays_counter,
                                last_tlas_traversal,
                                m_max_depth);

            CUDA_CHECK(cudaMemcpyAsync(
                &n_normal_active_rays, m_active_normal_rays_counter, sizeof(uint32_t), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpyAsync(
                &n_neural_active_rays, m_active_neural_rays_counter, sizeof(uint32_t), cudaMemcpyDeviceToHost));

            if ((n_normal_active_rays == 0 && n_neural_active_rays == 0) || last_tlas_traversal) {
                break;
            }
            tlas_traversal_step++;
        };

        CUDA_CHECK(cudaStreamDestroy(normal_rays_stream));
        CUDA_SYNC_CHECK();
    }

    void NeuralPathModule::neural_render_module(uint32_t accumulated_spp,
                                                uint32_t sample_offset,
                                                const NeuralBVH &neural_bvh)
    {
        switch (m_path_module_const.output_mode) {
        case OutputMode::NeuralPathTracing:
            neural_path_tracing(accumulated_spp, sample_offset, neural_bvh);
            break;
        default:
            UNREACHABLE();
        }
    }

    void NeuralPathModule::split_screen_neural_path_tracing(uint32_t accumulated_spp,
                                                            uint32_t sample_offset,
                                                            const NeuralBVH &neural_bvh,
                                                            const CUDABuffer<NeuralTreeCutData> &neural_tree_cut_data,
                                                            float max_node_loss_host,
                                                            bool show_tree_cut_depth,
                                                            uint32_t neural_tree_cut_max_depth)
    {
        CudaSceneData scene_data    = NeuralPathModule::scene_data();
        cudaStream_t default_stream = 0;

        cudaStream_t normal_rays_stream;
        CUDA_CHECK(cudaStreamCreate(&normal_rays_stream));

        uint32_t n_all_active_rays = m_cuda_backend->framebuffer_size_flat();

        uint32_t n_neural_active_rays = 0;
        uint32_t n_normal_active_rays = 0;

        CUDA_CHECK(cudaMemsetAsync(m_active_neural_rays_counter, 0, sizeof(uint32_t), default_stream));
        CUDA_CHECK(cudaMemsetAsync(m_active_normal_rays_counter, 0, sizeof(uint32_t), default_stream));

        // Populate rays and results with first traversal step
        tcnn::linear_kernel(split_screen_hybrid_camera_raygen_and_hybrid_first_hit_and_traversal,
                            0,
                            default_stream,
                            n_all_active_rays,
                            accumulated_spp,
                            sample_offset,
                            m_normal_traversal_data,
                            neural_bvh,
                            m_wavefront_neural_data[0].traversal_data,
                            m_wavefront_neural_data[0].results,
                            scene_data,
                            m_active_neural_rays_counter,
                            m_active_normal_rays_counter,
                            neural_tree_cut_data.d_pointer(),
                            max_node_loss_host,
                            show_tree_cut_depth,
                            neural_tree_cut_max_depth);

        CUDA_CHECK(cudaMemcpyAsync(&n_neural_active_rays,
                                   m_active_neural_rays_counter,
                                   sizeof(uint32_t),
                                   cudaMemcpyDeviceToHost,
                                   default_stream));
        CUDA_CHECK(cudaMemcpyAsync(&n_normal_active_rays,
                                   m_active_normal_rays_counter,
                                   sizeof(uint32_t),
                                   cudaMemcpyDeviceToHost,
                                   normal_rays_stream));
        CUDA_CHECK(cudaStreamSynchronize(default_stream));

        if (n_normal_active_rays == 0 && n_neural_active_rays == 0) {
            return;
        }

        int tlas_traversal_step = 0;
        while (true) {
            CUDA_CHECK(cudaMemsetAsync(m_hybrid_its_counter, 0, sizeof(uint32_t), default_stream));

            tcnn::linear_kernel(non_neural_bvh_intersection,
                                0,
                                normal_rays_stream,
                                n_normal_active_rays,
                                accumulated_spp,
                                m_normal_traversal_data,
                                m_hybrid_intersection_data,
                                scene_data,
                                m_hybrid_its_counter);

            // One neural bounce
            neural_bvh_traversal(accumulated_spp, neural_bvh, n_neural_active_rays, m_hybrid_its_counter);

            // Synchronize all streams
            CUDA_SYNC_CHECK();

            uint32_t n_active_hybrid_intersection = 0;

            CUDA_CHECK(cudaMemcpyAsync(
                &n_active_hybrid_intersection, m_hybrid_its_counter, sizeof(uint32_t), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemsetAsync(m_active_neural_rays_counter, 0, sizeof(uint32_t), default_stream));
            CUDA_CHECK(cudaMemsetAsync(m_active_normal_rays_counter, 0, sizeof(uint32_t), default_stream));

            bool last_tlas_traversal = (tlas_traversal_step == (m_max_tlas_traversal - 1));
            tcnn::linear_kernel(hybrid_scatter_and_get_next_tlas_traversal_step,
                                0,
                                default_stream,
                                n_active_hybrid_intersection,
                                accumulated_spp,
                                m_hybrid_intersection_data,
                                m_wavefront_neural_data[0].traversal_data,
                                m_wavefront_neural_data[0].results,
                                m_normal_traversal_data,
                                neural_bvh,
                                scene_data,
                                m_active_neural_rays_counter,
                                m_active_normal_rays_counter,
                                last_tlas_traversal,
                                m_max_depth);

            CUDA_CHECK(cudaMemcpyAsync(
                &n_normal_active_rays, m_active_normal_rays_counter, sizeof(uint32_t), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpyAsync(
                &n_neural_active_rays, m_active_neural_rays_counter, sizeof(uint32_t), cudaMemcpyDeviceToHost));

            if ((n_normal_active_rays == 0 && n_neural_active_rays == 0) || last_tlas_traversal) {
                break;
            }
            tlas_traversal_step++;
        };

        CUDA_CHECK(cudaStreamDestroy(normal_rays_stream));
        CUDA_SYNC_CHECK();
    }

    void NeuralPathModule::split_screen_neural_render_node_loss_module(
        uint32_t accumulated_spp,
        uint32_t sample_offset,
        const NeuralBVH &neural_bvh,
        const CUDABuffer<NeuralTreeCutData> &neural_tree_cut_data,
        float max_node_loss_host,
        bool show_tree_cut_depth,
        uint32_t neural_tree_cut_max_depth)
    {
        switch (m_path_module_const.output_mode) {
        case OutputMode::NeuralPathTracing:
            split_screen_neural_path_tracing(accumulated_spp,
                                             sample_offset,
                                             neural_bvh,
                                             neural_tree_cut_data,
                                             max_node_loss_host,
                                             show_tree_cut_depth,
                                             neural_tree_cut_max_depth);
            break;
        default:
            UNREACHABLE();
        }
    }
}
}