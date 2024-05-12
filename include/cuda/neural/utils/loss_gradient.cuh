
#pragma once

#include "ntwr.h"

namespace ntwr {

namespace neural {

    struct LossAndGradient {
        float loss;
        float gradient;

        NTWR_HOST_DEVICE LossAndGradient operator*(float scalar)
        {
            return {loss * scalar, gradient * scalar};
        }

        NTWR_HOST_DEVICE LossAndGradient operator/(float scalar)
        {
            return {loss / scalar, gradient / scalar};
        }
    };

    NTWR_DEVICE LossAndGradient l2_loss(const float &target, const float &prediction)
    {
        float difference = prediction - target;
        return {difference * difference, 2.0f * difference};
    }

    NTWR_DEVICE LossAndGradient relative_l2_loss(const float &target, const float &prediction)
    {
        float difference = prediction - target;
        float denom      = prediction * prediction + 1e-2f;
        return {difference * difference / denom, 2.0f * difference / denom};
    }

    NTWR_DEVICE LossAndGradient l1_loss(const float &target, const float &prediction)
    {
        float difference = prediction - target;
        return {
            fabsf(difference),
            copysign(1.0f, difference),
        };
    }

    NTWR_DEVICE LossAndGradient relative_l1_loss(const float &target, const float &prediction)
    {
        const float scale = 1.0f / (fabsf(prediction) + 1e-2f);

        float difference = prediction - target;
        return {
            fabsf(difference) * scale,
            copysign(scale, difference),
        };
    }

    NTWR_DEVICE LossAndGradient binary_cross_entropy_loss(const float &target, const float &prediction)
    {
        const float epsilon = 1e-5;
        const float pred    = min(max(prediction, epsilon), 1.f - epsilon);

        return {
            -(target * log(pred) + (1.f - target) * log(1.f - pred)),
            -(target - pred) / (pred - (pred * pred)),
        };
    }

    NTWR_DEVICE LossAndGradient log_l2_loss(const float &target, const float &prediction)
    {
        float difference = log(1 + prediction) - log(1 + target);
        return {difference * difference, 2.0f * difference / (prediction + 1)};
    }

    NTWR_DEVICE LossAndGradient l_half_loss(const float &target, const float &prediction)
    {
        float sqrt_difference = sqrt(abs(prediction - target));
        return {sqrt_difference, 1.f / (2.f * sqrt_difference + 1e-5f)};
    }

    NTWR_DEVICE LossAndGradient loss_and_gradient(const float &target, const float &prediction, LossType loss_type)
    {
        // clang-format off
        switch (loss_type) {
            case LossType::RelativeL2: 		   return relative_l2_loss(target, prediction); break;
            case LossType::L1:         		   return l1_loss(target, prediction); break; 
            case LossType::RelativeL1: 		   return relative_l1_loss(target, prediction); break; 
            case LossType::BinaryCrossEntropy: return binary_cross_entropy_loss(target,prediction); break;
            case LossType::LogL2:              return log_l2_loss(target,prediction); break;
            case LossType::LHalf:              return l_half_loss(target,prediction); break;
            default: case LossType::L2: 	   return l2_loss(target, prediction); break;
        }
        // clang-format on
    }

}
}