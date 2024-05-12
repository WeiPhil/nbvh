
#pragma once

#include "cuda/neural/prefiltering/prefiltering_module.h"

namespace ntwr {
namespace neural::prefiltering {

    using WavefrontData = NeuralWavefrontData<NeuralTraversalResult>;

    NTWR_KERNEL void camera_raygen_and_intersect_neural_bvh(uint32_t n_elements,
                                                            int sample_idx,
                                                            int rnd_sample_offset,
                                                            NeuralTraversalData traversal_data,
                                                            NeuralTraversalResult results,
                                                            NeuralBVH neural_bvh,
                                                            uint32_t *active_counter);

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

    NTWR_KERNEL void neural_prefiltering_scatter(const uint32_t n_elements,
                                                 int sample_idx,
                                                 uint32_t depth,
                                                 HybridIntersectionResult its_result_data,
                                                 NeuralTraversalData traversal_data,
                                                 NeuralTraversalResult results,
                                                 NeuralBVH neural_bvh,
                                                 float *__restrict__ appearance_output_data,
                                                 uint32_t *active_counter);

    NTWR_KERNEL void accumulate_neural_results(const uint32_t n_elements,
                                               int sample_idx,
                                               NeuralTraversalData traversal_data);

    NTWR_KERNEL void camera_raygen_and_first_hit(uint32_t n_elements,
                                                 int sample_idx,
                                                 int rnd_sample_offset,
                                                 HybridIntersectionResult intersection_result,
                                                 CudaSceneData scene_data,
                                                 uint32_t *active_counter);

    NTWR_KERNEL void generate_throughput_inference_data_and_update_ray(uint32_t n_elements,
                                                                       HybridIntersectionResult intersection_result,
                                                                       float *__restrict__ appearance_input_data);

    NTWR_KERNEL void process_prefiltering_network_output(uint32_t n_elements,
                                                         HybridIntersectionResult intersection_result,
                                                         NeuralTraversalData traversal_data,
                                                         float *__restrict__ appearance_output_data);

}

}