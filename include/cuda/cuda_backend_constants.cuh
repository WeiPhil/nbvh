#pragma once

#include "cuda/cuda_backend.h"

#ifndef CUDA_BACKEND_CONSTANT
#define CUDA_BACKEND_CONSTANT extern __constant__
#endif

namespace ntwr {

struct CudaSceneConstants;
struct CameraParams;

CUDA_BACKEND_CONSTANT cudaSurfaceObject_t framebuffer;
CUDA_BACKEND_CONSTANT cudaSurfaceObject_t accum_buffer;
CUDA_BACKEND_CONSTANT glm::uvec2 fb_size;
CUDA_BACKEND_CONSTANT uint32_t fb_size_flat;
CUDA_BACKEND_CONSTANT CameraParams camera_params;
CUDA_BACKEND_CONSTANT CudaSceneConstants scene_constants;
CUDA_BACKEND_CONSTANT cudaSurfaceObject_t reference_buffer;
CUDA_BACKEND_CONSTANT cudaSurfaceObject_t envmap_texture;

CUDA_BACKEND_CONSTANT float depth_max_value;
CUDA_BACKEND_CONSTANT float depth_min_value;

}

#undef CUDA_BACKEND_CONSTANT