#pragma once

#include "base/accel/bvh.h"
#include "cuda/base/ray.cuh"
#include "cuda/cuda_backend_constants.cuh"

namespace ntwr {

namespace neural {

    // Forward declarations for easier import of generic neural bvh
    // Unfortunately necessary because stack has different structure for BVH8
    struct NeuralBVH2;
    struct NeuralBVH8;

    struct NeuralBVH2Node;

    using NeuralBVH     = NeuralBVH2;
    using NeuralBVHNode = NeuralBVH2Node;

    // constexpr int NEURAL_STACK_SIZE = 16;  // ~8.55 ms, fixed depth 15 xyzrgb_dragon
    constexpr int NEURAL_STACK_SIZE = 24;  // ~8.6 ms, fixed depth 15 xyzrgb_dragon
    // constexpr int NEURAL_STACK_SIZE = 32;  // ~8.7 ms, fixed depth 15 xyzrgb_dragon

    NTWR_DEVICE uint32_t stack_index(uint32_t ray_idx, int stack_size)
    {
        return ray_idx + stack_size * fb_size_flat;
    }

    // Pushes on the stack located in the ray payload
    template <typename T>
    NTWR_DEVICE void stack_push(uint32_t ray_idx, T stack[], int &stack_size, T item)
    {
        assert(stack_size < NEURAL_STACK_SIZE);
        stack[stack_index(ray_idx, stack_size)] = item;
        stack_size++;
    }

    // Pops from the stack located in the ray payload
    template <typename T>
    NTWR_DEVICE T stack_pop(uint32_t ray_idx, const T stack[], int &stack_size)
    {
        assert(stack_size > 0);
        stack_size--;
        return stack[stack_index(ray_idx, stack_size)];
    }

    // Pushes to a thread local stack
    template <typename T>
    NTWR_DEVICE void local_stack_push(T stack[], int &stack_size, T item)
    {
        assert(stack_size < NEURAL_STACK_SIZE);
        stack[stack_size] = item;
        stack_size++;
    }

    // Pops from a thread local stack
    template <typename T>
    NTWR_DEVICE T local_stack_pop(const T stack[], int &stack_size)
    {
        assert(stack_size > 0);
        stack_size--;
        return stack[stack_size];
    }

}
}