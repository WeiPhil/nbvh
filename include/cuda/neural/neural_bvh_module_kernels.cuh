#pragma once

#include "cuda/base/material.cuh"

#include "cuda/neural/neural_bvh_module_common.cuh"
#include "cuda/neural/neural_bvh_module_kernels.cuh"

#include "cuda/neural/neural_bvh_module_constants.cuh"
#include "cuda/neural/neural_bvh_renderer.h"

namespace ntwr {
namespace neural {

    NTWR_KERNEL void granular_sample_bvh_network_input_data(int sub_batch_size,
                                                            int n_sub_batch,
                                                            int training_step,
                                                            NeuralBVH neural_bvh,
                                                            CudaSceneData scene_data,
                                                            uint32_t *__restrict__ sampled_node_indices,
                                                            float *__restrict__ visibility_input_data,
                                                            float *__restrict__ targets,
                                                            float current_max_loss,
                                                            NeuralTreeCutData *neural_tree_cut_data);

    NTWR_KERNEL void encode_bvh_network_input_data(uint32_t n_elements,
                                                   NeuralTraversalData traversal_data,
                                                   float *__restrict__ visibility_input_data,
                                                   bool assert_no_negative_mint);

}
}