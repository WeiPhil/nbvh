
#pragma once

#include "cuda/neural/path/path_module.h"

namespace ntwr {
namespace neural::path {

    using WavefrontData = NeuralWavefrontData<NeuralTraversalResult>;

    NTWR_KERNEL void camera_raygen_and_intersect_neural_bvh(uint32_t n_elements,
                                                            int sample_idx,
                                                            int rnd_sample_offset,
                                                            NeuralTraversalData traversal_data,
                                                            NeuralTraversalResult results,
                                                            NeuralBVH neural_bvh,
                                                            uint32_t *active_counter);

    NTWR_KERNEL void hybrid_camera_raygen_and_hybrid_first_hit_and_traversal(uint32_t n_elements,
                                                                             int sample_idx,
                                                                             int rnd_sample_offset,
                                                                             NormalTraversalData normal_traversal_data,
                                                                             NeuralBVH neural_bvh,
                                                                             NeuralTraversalData neural_traversal_data,
                                                                             NeuralTraversalResult neural_results,
                                                                             CudaSceneData scene_data,
                                                                             uint32_t *active_neural_counter,
                                                                             uint32_t *active_normal_counter);

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
        bool m_show_tree_cut_depth,
        uint32_t max_tree_cut_depth);

    NTWR_KERNEL void compact_neural_data_and_accumulate(const uint32_t n_elements,
                                                        int sample_idx,
                                                        WavefrontData src_wavefront_data,
                                                        WavefrontData dst_wavefront_data,
                                                        HybridIntersectionResult intersection_result_data,
                                                        uint32_t *active_counter,
                                                        uint32_t *its_active_counter,
                                                        bool last_inference);

    NTWR_KERNEL void process_bvh_network_output_and_intersect_neural_bvh(uint32_t n_elements,
                                                                         uint32_t traversal_step,
                                                                         float *__restrict__ visibility_output_data,
                                                                         NeuralTraversalData traversal_data,
                                                                         NeuralTraversalResult results,
                                                                         NeuralBVH neural_bvh);

    NTWR_KERNEL void non_neural_bvh_intersection(uint32_t n_elements,
                                                 int sample_idx,
                                                 NormalTraversalData normal_traversal_data,
                                                 HybridIntersectionResult intersection_result_data,
                                                 CudaSceneData scene_data,
                                                 uint32_t *its_counter);

    NTWR_KERNEL void hybrid_scatter_and_get_next_tlas_traversal_step(const uint32_t n_elements,
                                                                     int sample_idx,
                                                                     HybridIntersectionResult its_result_data,
                                                                     NeuralTraversalData next_neural_traversal_data,
                                                                     NeuralTraversalResult next_neural_results,
                                                                     NormalTraversalData next_normal_traversal_data,
                                                                     NeuralBVH neural_bvh,
                                                                     CudaSceneData scene_data,
                                                                     uint32_t *active_neural_counter,
                                                                     uint32_t *active_normal_counter,
                                                                     bool last_tlas_traversal,
                                                                     uint32_t max_depth);

    NTWR_KERNEL void camera_raygen_and_hybrid_first_hit_and_traversal(uint32_t n_elements,
                                                                      int sample_idx,
                                                                      int rnd_sample_offset,
                                                                      NormalTraversalData hybrid_traversal_data,
                                                                      NeuralBVH neural_bvh,
                                                                      NeuralTraversalData traversal_data,
                                                                      NeuralTraversalResult neural_results,
                                                                      CudaSceneData scene_data,
                                                                      uint32_t *active_neural_counter,
                                                                      uint32_t *active_normal_counter);
}

}