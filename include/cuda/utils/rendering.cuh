#pragma once

#include "glm_pch.h"
#include "ntwr.h"

#include "cuda/utils/common.cuh"
#include "cuda/utils/lcg_rng.cuh"

namespace ntwr {

NTWR_DEVICE bool shading_frame(
    const glm::vec3 &wo, const glm::vec3 &its_normal, glm::vec3 &sh_normal, glm::vec3 &tangent, glm::vec3 &bitangent)
{
    bool backface_normal = false;
    sh_normal            = its_normal;
    if (dot(wo, sh_normal) < 0.f) {
        sh_normal       = -sh_normal;
        backface_normal = true;
    }
    ortho_basis(tangent, bitangent, sh_normal);
    return backface_normal;
}

NTWR_DEVICE bool rr_terminate(uint32_t depth, glm::vec3 &path_throughput, uint32_t rr_start_depth, LCGRand &rng)
{
    // Russian roulette termination
    if (depth >= rr_start_depth) {
        // Since our path throughput is an estimate itself it's important we clamp it to 1
        // aswell
        float prefix_weight = min(1.f, component_max(path_throughput));

        float survival_prob = prefix_weight;
        float rr_sample     = lcg_randomf(rng);
        if (rr_sample < survival_prob)
            path_throughput /= survival_prob;
        else
            return true;
    }

    return false;
}

}