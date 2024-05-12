#pragma once

#include "cuda/neural/neural_bvh_module.h"

#ifndef NEURAL_BVH_MODULE_CONSTANT
#define NEURAL_BVH_MODULE_CONSTANT extern __constant__
#endif

namespace ntwr {
namespace neural {
    NEURAL_BVH_MODULE_CONSTANT NeuralBVHModuleConstants bvh_module_const;
    NEURAL_BVH_MODULE_CONSTANT PathTracingOutputConstants pt_target_const;
}
}

#undef NEURAL_BVH_MODULE_CONSTANT
