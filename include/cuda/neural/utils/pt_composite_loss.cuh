#pragma once

#include "ntwr.h"

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/loss.h>

#include "cuda/neural/utils/loss_gradient.cuh"
#include "cuda/neural/utils/tcnn_config.h"

namespace ntwr {
namespace neural {

    using namespace tcnn;

    template <typename T>
    NTWR_KERNEL void composite_pt_loss(const uint32_t n_elements,
                                       const uint32_t stride,
                                       const uint32_t dims,
                                       const float loss_scale,
                                       const T *__restrict__ predictions,
                                       const float *__restrict__ targets,
                                       float *__restrict__ values,
                                       T *__restrict__ gradients,
                                       const float *__restrict__ data_pdf = nullptr)
    {
        const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= n_elements)
            return;

        const uint32_t intra_elem_idx = i % stride;
        const uint32_t inter_elem_idx = i / stride;
        if (intra_elem_idx >= dims) {
            values[i]    = 0;
            gradients[i] = 0;
            return;
        }

        const uint32_t target_idx = inter_elem_idx * dims + intra_elem_idx;

        const uint32_t n_total = n_elements / stride * dims;

        const float prediction = (float)predictions[i];
        const float target     = (float)targets[target_idx];

        const float pdf = data_pdf ? data_pdf[target_idx] : 1;

        const uint32_t visibility_index_start = 0;
        const uint32_t depth_index_start      = pt_target_const.output_visibility_dim;
        const uint32_t normal_index_start     = depth_index_start + pt_target_const.output_depth_dim;
        const uint32_t albedo_index_start     = normal_index_start + pt_target_const.output_normal_dim;
        const uint32_t albedo_index_end       = albedo_index_start + pt_target_const.output_albedo_dim;

        const float target_visibility = (float)targets[inter_elem_idx * dims + visibility_index_start];
        const bool ignore_non_visibility_loss =
            pt_target_const.visibility_masking_loss ? target_visibility < 0.5f : false;

        LossAndGradient lg = {0.f, 0.f};
        if (pt_target_const.output_visibility_dim > 0 && (intra_elem_idx == visibility_index_start)) {
            lg = loss_and_gradient(target, prediction, pt_target_const.visibility_loss);
            lg.loss *= pt_target_const.visibility_loss_scale;
            lg.gradient *= pt_target_const.visibility_loss_scale;

        } else if (!ignore_non_visibility_loss && pt_target_const.output_depth_dim > 0 &&
                   (intra_elem_idx >= depth_index_start && intra_elem_idx < normal_index_start)) {
            lg = loss_and_gradient(target, prediction, pt_target_const.depth_loss);

            if (pt_target_const.output_depth_linear_activation) {
                if (pt_target_const.depth_loss == LossType::L2 || pt_target_const.depth_loss == LossType::RelativeL2) {
                    lg.gradient *= 100.f;
                } else {
                    // Typically L1 then
                    lg.gradient *= 10.f;
                }
            }

            lg.loss *= pt_target_const.depth_loss_scale;
            lg.gradient *= pt_target_const.depth_loss_scale;

        } else if (!ignore_non_visibility_loss && pt_target_const.output_normal_dim > 0 &&
                   (intra_elem_idx >= normal_index_start && intra_elem_idx < albedo_index_start)) {
            if (pt_target_const.output_normal_linear_activation) {
                // Approximate linear activation behaviour instead of sigmoid activation function
                float n_length              = 0.1f;
                float n_axis                = (prediction * 2.f - 1.f) / n_length;
                float linearized_prediction = n_axis * 0.5f + 0.5f;

                lg = loss_and_gradient(target, linearized_prediction, pt_target_const.normal_loss);
                lg.gradient *= 1.f / n_length;
            } else {
                lg = loss_and_gradient(target, prediction, pt_target_const.normal_loss);
            }

            lg.loss *= pt_target_const.normal_loss_scale;
            lg.gradient *= pt_target_const.normal_loss_scale;

        } else if (!ignore_non_visibility_loss && pt_target_const.output_albedo_dim > 0 &&
                   (intra_elem_idx >= albedo_index_start && intra_elem_idx < albedo_index_end)) {
            if (pt_target_const.output_albedo_linear_activation) {
                // Approximate linear activation behaviour instead of sigmoid activation function
                float n_length              = 0.1f;
                float linearized_prediction = prediction / n_length;

                lg = loss_and_gradient(target, linearized_prediction, pt_target_const.albedo_loss);
                lg.gradient *= 1.f / n_length;
            } else {
                lg = loss_and_gradient(target, prediction, pt_target_const.albedo_loss);
            }

            lg.loss *= pt_target_const.albedo_loss_scale;
            lg.gradient *= pt_target_const.albedo_loss_scale;

        } else {
            assert(ignore_non_visibility_loss);
        }

        values[i] = lg.loss / pdf / n_total;

        float gradient = lg.gradient / pdf;
        gradients[i]   = (T)(loss_scale * gradient / n_total);
    }

    template <typename T>
    class PathTracingCompositeLoss : public Loss<T> {
    public:
        PathTracingCompositeLoss(uint32_t output_visibility_dim,
                                 std::string visibility_loss,
                                 uint32_t output_depth_dim,
                                 std::string depth_loss,
                                 uint32_t output_normal_dim,
                                 std::string normal_loss,
                                 uint32_t output_albedo_dim,
                                 std::string albedo_loss)
            : m_output_visibility_dim{output_visibility_dim},
              m_output_depth_dim{output_depth_dim},
              m_output_normal_dim{output_normal_dim},
              m_output_albedo_dim{output_albedo_dim}
        {
            m_visibility_loss =
                (LossType)find_string_index(loss_types, visibility_loss.c_str(), ARRAY_SIZE(loss_types));
            m_depth_loss  = (LossType)find_string_index(loss_types, depth_loss.c_str(), ARRAY_SIZE(loss_types));
            m_normal_loss = (LossType)find_string_index(loss_types, normal_loss.c_str(), ARRAY_SIZE(loss_types));
            m_albedo_loss = (LossType)find_string_index(loss_types, albedo_loss.c_str(), ARRAY_SIZE(loss_types));
        }

        void evaluate(cudaStream_t stream,
                      const float loss_scale,
                      const GPUMatrix<T> &prediction,
                      const GPUMatrix<float> &target,
                      GPUMatrix<float> &values,
                      GPUMatrix<T> &gradients,
                      const GPUMatrix<float> *data_pdf = nullptr) const override
        {
            const uint32_t dims   = target.m();
            const uint32_t stride = prediction.m();

            CHECK_THROW(m_output_visibility_dim + m_output_normal_dim + m_output_depth_dim + m_output_albedo_dim ==
                        dims);
            CHECK_THROW(prediction.n() == target.n());
            CHECK_THROW(values.m() == stride);
            CHECK_THROW(gradients.m() == stride);
            CHECK_THROW(!data_pdf || data_pdf->m() == dims);

            // printf("dims : %d\n", dims);

            linear_kernel(composite_pt_loss<T>,
                          0,
                          stream,
                          prediction.n_elements(),
                          stride,
                          dims,
                          loss_scale,
                          prediction.data(),
                          target.data(),
                          values.data(),
                          gradients.data(),
                          data_pdf ? data_pdf->data() : nullptr);
        }

        void update_hyperparams(const json &params) override {}

        json hyperparams() const override
        {
            return {
                {"otype", "PathTracingComposite"},
            };
        }

    private:
        uint32_t m_output_visibility_dim;
        uint32_t m_output_depth_dim;
        uint32_t m_output_normal_dim;
        uint32_t m_output_albedo_dim;
        LossType m_visibility_loss;
        LossType m_depth_loss;
        LossType m_normal_loss;
        LossType m_albedo_loss;
    };

}
}
