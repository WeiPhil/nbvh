#pragma once

#include "cuda/neural/prefiltering/prefiltering_module.h"

#ifndef NEURAL_PREFILTERING_CONSTANT
#define NEURAL_PREFILTERING_CONSTANT extern __constant__
#endif

namespace ntwr {
namespace neural::prefiltering {

    NEURAL_PREFILTERING_CONSTANT NeuralPrefilteringModuleConstants prefiltering_module_const;
}

}

#undef NEURAL_PREFILTERING_CONSTANT
