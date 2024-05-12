#pragma once

#include "app/render_app.h"
#include "app/render_backend.h"

#include "base/accel/bvh.h"
#include "base/scene.h"

#include <json/json.hpp>

// If commented out uses a hasgrid per encoded point
#define USE_REPEATED_COMPOSITE

#define BBOX_REGULARISATION 1e-4f

#include "cuda/accel/tlas.cuh"

#include "cuda/accel/bvh2.cuh"
#include "cuda/accel/bvh8.cuh"

#include "utils/cuda_buffer.cuh"

// Only build with C++ compiler (cuda C++ 20 span features not supported)
#ifndef __CUDACC__

#include <bvh/v2/bvh.h>
#include <bvh/v2/default_builder.h>
#include <bvh/v2/executor.h>
#include <bvh/v2/node.h>
#include <bvh/v2/thread_pool.h>
#include <bvh/v2/tri.h>
#include <bvh/v2/vec.h>

#else
// Dummy struct for Cuda compiler
struct LibBVH;
#endif

namespace ntwr {

using BVH      = BVH2;
using CudaBLAS = CudaBVH2;
// BVH2 Node have the same struct for CPU and GPU
using CudaBLASNode = BVH2Node;
using BVHNode      = BVH2Node;

#ifndef __CUDACC__
using LibBVH = bvh::v2::Bvh<bvh::v2::Node<float, 3>>;
#endif

struct CameraParams {
    glm::vec3 origin;
    glm::vec3 du;
    glm::vec3 dv;
    glm::vec3 dir_top_left;
};

struct alignas(8) CudaSceneConstants {
    float constant_roughness = 1.f;
    float constant_specular  = 0.0f;
    float constant_metallic  = 0.0f;
    float envmap_strength    = 1.f;
    float envmap_rotation    = 0.f;

    bool envmap_eval        = true;
    bool emission_eval      = true;
    bool skylike_background = false;
    bool sample_light       = false;

    // TODO : find better place
    float ema_weight             = 0.05f;
    bool ema_accumulate          = true;
    bool force_constant_material = false;

    inline NTWR_HOST nlohmann::json to_json()
    {
        return nlohmann::ordered_json{
            {"force_constant_material", force_constant_material},
            {"constant_roughness", constant_roughness},
            {"constant_specular", constant_specular},
            {"constant_metallic", constant_metallic},
            {"envmap_rotation", envmap_rotation},
            {"envmap_strength", envmap_strength},
            {"ema_accumulate", ema_accumulate},
            {"ema_weight", ema_weight},
            {"envmap_eval", envmap_eval},
            {"emission_eval", emission_eval},
            {"skylike_background", skylike_background},
        };
    }

    inline NTWR_HOST void from_json(nlohmann::json json_config)
    {
        CudaSceneConstants default_config = CudaSceneConstants{};
        force_constant_material = json_config.value("force_constant_material", default_config.force_constant_material);
        constant_roughness      = json_config.value("constant_roughness", default_config.constant_roughness);
        constant_specular       = json_config.value("constant_specular", default_config.constant_specular);
        constant_metallic       = json_config.value("constant_metallic", default_config.constant_metallic);
        envmap_rotation         = json_config.value("envmap_rotation", default_config.envmap_rotation);
        envmap_strength         = json_config.value("envmap_strength", default_config.envmap_strength);
        ema_accumulate          = json_config.value("ema_accumulate", default_config.ema_accumulate);
        ema_weight              = json_config.value("ema_weight", default_config.ema_weight);
        envmap_eval             = json_config.value("envmap_eval", default_config.envmap_eval);
        emission_eval           = json_config.value("emission_eval", default_config.emission_eval);
        skylike_background      = json_config.value("skylike_background", default_config.skylike_background);
    }
};

struct CudaSceneData {
    CudaTLAS tlas;
    PackedMaterial *materials;
    cudaTextureObject_t *textures;
};

class CudaBackend : public RenderBackend {
public:
    CudaBackend(const Scene &scene, RenderApp *display);
    ~CudaBackend();

    virtual std::string name() override
    {
        return "Cuda Backend";
    }

    virtual std::filesystem::path scene_filename() override
    {
        return m_scene_filename;
    }
    virtual bool ui(bool dev_mode) override;
    virtual void set_reference_image(Bitmap<float> reference_image) override;
    virtual void readback_framebuffer(size_t buffer_size, unsigned char *buffer) override;
    virtual void readback_framebuffer_float(size_t buffer_size, float *buffer) override;
    virtual void resize(const glm::uvec2 &new_size) override;
    virtual void update_camera(const Camera &camera) override;
    virtual void update_scene_data(const Scene &scene) override;

    virtual void release_display_mapped_memory() override;
    virtual void map_display_memory() override;
    virtual glm::uvec2 framebuffer_size() override
    {
        return m_fb_size;
    }

    void free_scene_data();

    uint32_t framebuffer_size_flat()
    {
        return m_fb_size.x * m_fb_size.y;
    }

    void load_envmap_texture(std::string filename);
    std::string envmap_filename()
    {
        return std::string(m_envmap_filename);
    }

    void setup_constants();

    CudaSceneData scene_data();

    bool bvh_initialized()
    {
        return m_bvh_initialized;
    }
    size_t n_meshes();
    size_t n_vertices();
    size_t n_textures();

    Camera &get_camera() const
    {
        return *m_render_app->get_scene().camera;
    }

    const RenderDisplay &get_display() const
    {
        return *m_display;
    }

    const RenderApp &get_render_app() const
    {
        return *m_render_app;
    }

    RenderAppConfigs &get_render_app_configs()
    {
        return m_render_app->render_app_configs();
    }

    std::filesystem::path m_scene_filename;

    glm::uvec2 m_fb_size;
    cudaSurfaceObject_t m_framebuffer_surface;
    cudaGraphicsResource_t framebuffer_resource;

    cudaSurfaceObject_t m_accum_buffer_surface;
    cudaGraphicsResource_t accum_buffer_resource;

    CameraParams m_camera_params;

    CudaSceneConstants m_cuda_scene_constants = {};

    CUDABuffer<CudaBLAS> m_cuda_mesh_bvhs_buffer;
    CUDABuffer<TLASNode> m_cuda_tlas_nodes_buffer;
    CUDABuffer<uint32_t> m_cuda_tlas_indices_buffer;

    // Structure holding pointers to material data
    CUDABuffer<PackedMaterial> m_cuda_material_buffer;

    // TODO: add buffer of geometries for cleaner access
    std::vector<CUDABuffer<glm::vec3>> vertex_buffers;
    std::vector<CUDABuffer<glm::uvec3>> index_buffers;
    std::vector<CUDABuffer<glm::vec3>> normal_buffers;
    std::vector<CUDABuffer<glm::vec2>> texcoord_buffers;
    std::vector<CUDABuffer<int>> material_buffers;
    std::vector<CUDABuffer<uint32_t>> material_map_buffers;

    /*! @{ one texture object and pixel array per used texture */
    std::vector<cudaArray_t> m_texture_arrays;
    std::vector<cudaTextureObject_t> m_texture_objects;
    // Buffer for device pointers
    CUDABuffer<cudaTextureObject_t> m_cuda_textures_buffer;
    /*! @} */
    cudaArray_t m_envmap_texture_array = nullptr;
    cudaTextureObject_t m_envmap_texture_object;
    char m_envmap_filename[OUTPUT_FILENAME_SIZE] = "";

    // Reference texture buffer
    cudaTextureObject_t m_reference_image_surface;
    CUDABuffer<float> m_reference_image_buffer;
    bool m_reference_image_set = false;

    uint32_t m_libbvh_max_leaf_size = 2;

    CudaTLAS m_cuda_tlas;
    std::vector<BVH> m_cpu_mesh_bvhs;
    std::vector<uint32_t> m_neural_bvh_indices;

private:
    RenderDisplay *m_display;
    RenderApp *m_render_app;

    LibBVH build_libbvh_mesh(const Mesh &mesh);
    LibBVH build_libbvh_tlas();

    void load_scene(const Scene &scene);
    void build_blas(const Scene &scene);
    void build_tlas(const Scene &scene);

    void create_cuda_textures_buffer(const Scene &scene);

    void register_custom_losses();

    size_t m_n_meshes   = 0;
    size_t m_n_vertices = 0;
    size_t m_n_textures = 0;

    std::vector<CUDABuffer<CudaBLASNode>> m_mesh_bvhs_nodes;

    // The flattened primitive indices refered to by nodes
    std::vector<CUDABuffer<size_t>> m_mesh_bvhs_prim_indices;

    bool m_bvh_initialized = false;
};

}