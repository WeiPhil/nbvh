

#include "cuda/cuda_backend.h"
#include "cuda/cuda_backend_constants.cuh"

#include "glm_pch.h"

namespace ntwr {

void CudaBackend::setup_constants()
{
    CUDA_CHECK(cudaMemcpyToSymbol(fb_size, &m_fb_size, sizeof(fb_size)));
    uint32_t fb_size_flat_local = m_fb_size.x * m_fb_size.y;
    CUDA_CHECK(cudaMemcpyToSymbol(fb_size_flat, &fb_size_flat_local, sizeof(fb_size_flat)));
    CUDA_CHECK(cudaMemcpyToSymbol(framebuffer, &m_framebuffer_surface, sizeof(framebuffer)));
    CUDA_CHECK(cudaMemcpyToSymbol(accum_buffer, &m_accum_buffer_surface, sizeof(accum_buffer)));
    CUDA_CHECK(cudaMemcpyToSymbol(camera_params, &m_camera_params, sizeof(camera_params)));
    CUDA_CHECK(cudaMemcpyToSymbol(scene_constants, &m_cuda_scene_constants, sizeof(scene_constants)));
    CUDA_CHECK(cudaMemcpyToSymbol(reference_buffer, &m_reference_image_surface, sizeof(reference_buffer)));
    CUDA_CHECK(cudaMemcpyToSymbol(envmap_texture, &m_envmap_texture_object, sizeof(envmap_texture)));

    Bbox3f neural_scene_bbox = m_cpu_mesh_bvhs[m_neural_bvh_indices[0]].root_bbox();
    // Compute min max depth values for visualising depth
    float t_depth_min_value = FLT_MAX;
    float t_depth_max_value = -FLT_MAX;
    for (size_t corner_idx = 0; corner_idx < 8; corner_idx++) {
        float dist        = glm::length(m_camera_params.origin - neural_scene_bbox.corner(corner_idx));
        t_depth_min_value = min(dist, t_depth_min_value);
        t_depth_max_value = max(dist, t_depth_max_value);
    }
    CUDA_CHECK(cudaMemcpyToSymbol(depth_min_value, &t_depth_min_value, sizeof(depth_min_value)));
    CUDA_CHECK(cudaMemcpyToSymbol(depth_max_value, &t_depth_max_value, sizeof(depth_max_value)));
}

}