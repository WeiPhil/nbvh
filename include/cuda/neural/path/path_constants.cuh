#pragma once

#include "cuda/neural/path/path_module.h"

#ifndef NEURAL_PATH_CONSTANT
#define NEURAL_PATH_CONSTANT extern __constant__
#endif

namespace ntwr {
namespace neural::path {

    NEURAL_PATH_CONSTANT NeuralPathModuleConstants path_module_const;
}

}

#undef NEURAL_PATH_CONSTANT
