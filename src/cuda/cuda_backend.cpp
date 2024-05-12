

#include "cuda/cuda_backend.h"
#include "cuda/cuda_backend_constants.cuh"
#include "utils/logger.h"

#include "glm_pch.h"

#include <bvh/v2/bvh.h>
#include <bvh/v2/default_builder.h>
#include <bvh/v2/executor.h>
#include <bvh/v2/node.h>
#include <bvh/v2/thread_pool.h>
#include <bvh/v2/tri.h>
#include <bvh/v2/vec.h>

namespace ntwr {

CudaBackend::CudaBackend(const Scene &scene, RenderApp *render_app)
    : m_render_app(render_app), m_display(render_app->display)
{
    assert(render_app);
    assert(render_app->display);
    logger(LogLevel::Info, "Loading Cuda Backend scene");
    load_scene(scene);
    build_blas(scene);
    build_tlas(scene);
    map_display_memory();
}

CudaBackend::~CudaBackend()
{
    release_display_mapped_memory();

    free_scene_data();

    cudaDeviceReset();
}

void CudaBackend::free_scene_data()
{
    m_cuda_mesh_bvhs_buffer.free();
    m_cuda_tlas_nodes_buffer.free();
    m_cuda_tlas_indices_buffer.free();
    m_cuda_material_buffer.free();

    for (auto &vertex_buffer : vertex_buffers)
        vertex_buffer.free();
    for (auto &index_buffer : index_buffers)
        index_buffer.free();
    for (auto &normal_buffer : normal_buffers)
        normal_buffer.free();
    for (auto &texcoord_buffer : texcoord_buffers)
        texcoord_buffer.free();
    for (auto &material_buffer : material_buffers)
        material_buffer.free();
    for (auto &material_map_buffer : material_map_buffers)
        material_map_buffer.free();

    for (auto &mesh_bvh_nodes : m_mesh_bvhs_nodes) {
        mesh_bvh_nodes.free();
    }
    for (auto &mesh_bvhs_prim_indices : m_mesh_bvhs_prim_indices) {
        mesh_bvhs_prim_indices.free();
    }

    for (auto &texture_object : m_texture_objects) {
        CUDA_CHECK(cudaDestroyTextureObject(texture_object));
    }
    for (auto &texture_array : m_texture_arrays) {
        CUDA_CHECK(cudaFreeArray(texture_array));
    }

    m_cuda_textures_buffer.free();
}

CudaSceneData CudaBackend::scene_data()
{
    CudaSceneData data;
    data.tlas      = m_cuda_tlas;
    data.materials = m_cuda_material_buffer.d_pointer();
    data.textures  = m_cuda_textures_buffer.d_pointer();

    return data;
}

void CudaBackend::map_display_memory()
{
    assert(m_display);
    // Register the texture with CUDA
    // Main framebuffer (RGBA 8byte int)
    {
        CUDA_CHECK(cudaGraphicsGLRegisterImage(&framebuffer_resource,
                                               m_display->gl_framebuffer_texture(),
                                               GL_TEXTURE_2D,
                                               cudaGraphicsRegisterFlagsSurfaceLoadStore));

        cudaArray_t cuda_array_ptr;
        CUDA_CHECK(cudaGraphicsMapResources(1, &framebuffer_resource, 0));
        CUDA_CHECK(cudaGraphicsSubResourceGetMappedArray(&cuda_array_ptr, framebuffer_resource, 0, 0));

        cudaResourceDesc res_desc = {};
        res_desc.resType          = cudaResourceTypeArray;
        res_desc.res.array.array  = cuda_array_ptr;
        CUDA_CHECK(cudaCreateSurfaceObject(&m_framebuffer_surface, &res_desc));
    }

    // Accumulation framebuffer (RGBA 32 byte float)
    {
        CUDA_CHECK(cudaGraphicsGLRegisterImage(&accum_buffer_resource,
                                               m_display->gl_accum_buffer_texture(),
                                               GL_TEXTURE_2D,
                                               cudaGraphicsRegisterFlagsSurfaceLoadStore));

        cudaArray_t cuda_array_ptr;
        CUDA_CHECK(cudaGraphicsMapResources(1, &accum_buffer_resource, 0));
        CUDA_CHECK(cudaGraphicsSubResourceGetMappedArray(&cuda_array_ptr, accum_buffer_resource, 0, 0));
        cudaResourceDesc res_desc = {};
        res_desc.resType          = cudaResourceTypeArray;
        res_desc.res.array.array  = cuda_array_ptr;
        CUDA_CHECK(cudaCreateSurfaceObject(&m_accum_buffer_surface, &res_desc));
    }
}

void CudaBackend::release_display_mapped_memory()
{
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &framebuffer_resource, 0));
    CUDA_CHECK(cudaGraphicsUnregisterResource(framebuffer_resource));
    CUDA_CHECK(cudaDestroySurfaceObject(m_framebuffer_surface));

    CUDA_CHECK(cudaGraphicsUnmapResources(1, &accum_buffer_resource, 0));
    CUDA_CHECK(cudaGraphicsUnregisterResource(accum_buffer_resource));
    CUDA_CHECK(cudaDestroySurfaceObject(m_accum_buffer_surface));

    CUDA_SYNC_CHECK();
}

void CudaBackend::load_scene(const Scene &scene)
{
    m_scene_filename = scene.m_scene_filename;

    // split in two functions -> uploading the scene data and building the acceleration?
    m_n_meshes = scene.meshes.size();

    vertex_buffers.resize(m_n_meshes);
    index_buffers.resize(m_n_meshes);
    normal_buffers.resize(m_n_meshes);
    texcoord_buffers.resize(m_n_meshes);
    material_buffers.resize(m_n_meshes);
    material_map_buffers.resize(m_n_meshes);

    // We store the indices to the neural bvh's
    m_neural_bvh_indices = scene.neural_mesh_indices;

    assert(m_neural_bvh_indices[0] <= m_n_meshes);

    // Each mesh corresponds to a BLAS bvh
    m_n_vertices = 0;
    for (size_t mesh_idx = 0; mesh_idx < m_n_meshes; mesh_idx++) {
        const Mesh &mesh = scene.meshes[mesh_idx];

        vertex_buffers[mesh_idx].alloc_and_upload(mesh.vertices);
        index_buffers[mesh_idx].alloc_and_upload(mesh.indices);
        if (!mesh.normals.empty())
            normal_buffers[mesh_idx].alloc_and_upload(mesh.normals);
        if (!mesh.texcoords.empty())
            texcoord_buffers[mesh_idx].alloc_and_upload(mesh.texcoords);
        material_buffers[mesh_idx].alloc_and_upload(mesh.material_ids);
        material_map_buffers[mesh_idx].alloc_and_upload(mesh.material_map);

        m_n_vertices += mesh.vertices.size();
    }

    logger(LogLevel::Info, "Uploading %d materials ...", scene.materials.size());
    if (scene.materials.size() > 0)
        m_cuda_material_buffer.alloc_and_upload(scene.materials);
    else
        logger(LogLevel::Warn, "The scene contains no materials ...");

    logger(LogLevel::Info, "Create and upload cuda textures ...");
    create_cuda_textures_buffer(scene);
    logger(LogLevel::Info, "Create and upload envmap texture ...");
    load_envmap_texture(std::string(m_envmap_filename));
}

void CudaBackend::build_blas(const Scene &scene)
{
    std::vector<CudaBLAS> cuda_mesh_bvhs;
    cuda_mesh_bvhs.resize(m_n_meshes);
    m_cpu_mesh_bvhs.resize(m_n_meshes);
    m_mesh_bvhs_nodes.resize(m_n_meshes);
    m_mesh_bvhs_prim_indices.resize(m_n_meshes);

    uint32_t total_neuralized_vertex_buffer_size    = 0;
    uint32_t total_neuralized_index_buffer_size     = 0;
    uint32_t total_neuralized_normal_buffer_size    = 0;
    uint32_t total_neuralized_texcoords_buffer_size = 0;
    uint32_t total_neuralized_bvh_size              = 0;

    uint32_t total_non_neuralized_vertex_buffer_size    = 0;
    uint32_t total_non_neuralized_index_buffer_size     = 0;
    uint32_t total_non_neuralized_normal_buffer_size    = 0;
    uint32_t total_non_neuralized_texcoords_buffer_size = 0;
    uint32_t total_non_neuralized_bvh_size              = 0;

    // Build CPU side BLAS's
    logger(LogLevel::Info, "Building BLAS for %d meshes", m_n_meshes);
    for (size_t mesh_idx = 0; mesh_idx < m_n_meshes; mesh_idx++) {
        const Mesh &mesh  = scene.meshes[mesh_idx];
        auto lib_bvh_mesh = std::make_unique<LibBVH>(build_libbvh_mesh(mesh));

        for (auto &n : lib_bvh_mesh->nodes) {
            auto diagonal  = n.get_bbox().get_diagonal();
            auto bound_eps = BBOX_REGULARISATION;
            auto extent_eps =
                bvh::v2::Vec<float, 3>(bound_eps, bound_eps, bound_eps);  // n.get_bbox().get_diagonal() * bound_eps;
            for (size_t i = 0; i < 3; i++) {
                if (diagonal[i] <= BBOX_REGULARISATION) {
                    n.bounds[2 * i] -= extent_eps[i] * 0.5f;      // min
                    n.bounds[2 * i + 1] += extent_eps[i] * 0.5f;  // max
                }
            }
        }

        auto lib_root_bbox = lib_bvh_mesh->get_root().get_bbox();
        auto tmp_bbox      = Bbox3f{glm::vec3(lib_root_bbox.min[0], lib_root_bbox.min[1], lib_root_bbox.min[2]),
                               glm::vec3(lib_root_bbox.max[0], lib_root_bbox.max[1], lib_root_bbox.max[2])};

        m_cpu_mesh_bvhs[mesh_idx].build_from_libbvh(*lib_bvh_mesh);

        assert(m_cpu_mesh_bvhs[mesh_idx].root_bbox().min == tmp_bbox.min);
        assert(m_cpu_mesh_bvhs[mesh_idx].root_bbox().max == tmp_bbox.max);

        m_mesh_bvhs_nodes[mesh_idx].alloc_and_upload(m_cpu_mesh_bvhs[mesh_idx].m_nodes);
        m_mesh_bvhs_prim_indices[mesh_idx].alloc_and_upload(m_cpu_mesh_bvhs[mesh_idx].m_indices);

        cuda_mesh_bvhs[mesh_idx].vertices      = vertex_buffers[mesh_idx].d_pointer();
        cuda_mesh_bvhs[mesh_idx].indices       = index_buffers[mesh_idx].d_pointer();
        cuda_mesh_bvhs[mesh_idx].normals       = normal_buffers[mesh_idx].d_pointer();
        cuda_mesh_bvhs[mesh_idx].texcoords     = texcoord_buffers[mesh_idx].d_pointer();
        cuda_mesh_bvhs[mesh_idx].material_ids  = material_buffers[mesh_idx].d_pointer();
        cuda_mesh_bvhs[mesh_idx].num_materials = material_buffers[mesh_idx].num_elements();
        cuda_mesh_bvhs[mesh_idx].material_map  = material_map_buffers[mesh_idx].d_pointer();

        cuda_mesh_bvhs[mesh_idx].nodes            = m_mesh_bvhs_nodes[mesh_idx].d_pointer();
        cuda_mesh_bvhs[mesh_idx].num_nodes        = m_mesh_bvhs_nodes[mesh_idx].num_elements();
        cuda_mesh_bvhs[mesh_idx].prim_indices     = m_mesh_bvhs_prim_indices[mesh_idx].d_pointer();
        cuda_mesh_bvhs[mesh_idx].num_prim_indices = m_mesh_bvhs_prim_indices[mesh_idx].num_elements();

        if (m_neural_bvh_indices[0] == mesh_idx) {
            // neuralized mesh
            total_neuralized_vertex_buffer_size += mesh.vertices.size() * 4 * 3;
            total_neuralized_index_buffer_size += mesh.indices.size() * 4 * 3;
            total_neuralized_normal_buffer_size += mesh.normals.size() * 4 * 3;
            total_neuralized_texcoords_buffer_size += mesh.texcoords.size() * 4 * 2;
            total_neuralized_bvh_size += lib_bvh_mesh->nodes.size() * 32;
        } else {
            total_non_neuralized_vertex_buffer_size += mesh.vertices.size() * 4 * 3;
            total_non_neuralized_index_buffer_size += mesh.indices.size() * 4 * 3;
            total_non_neuralized_normal_buffer_size += mesh.normals.size() * 4 * 3;
            total_non_neuralized_texcoords_buffer_size += mesh.texcoords.size() * 4 * 2;
            total_non_neuralized_bvh_size += lib_bvh_mesh->nodes.size() * 32;
        }
    }
    logger(LogLevel::Info, "Done!");

    m_cuda_mesh_bvhs_buffer.alloc_and_upload(cuda_mesh_bvhs);
}

LibBVH CudaBackend::build_libbvh_mesh(const Mesh &mesh)
{
    using Vec3 = bvh::v2::Vec<float, 3>;
    using BBox = bvh::v2::BBox<float, 3>;
    using Tri  = bvh::v2::Tri<float, 3>;

    // using PrecomputedTri = bvh::v2::PrecomputedTri<float>;

    // This is the original data, which may come in some other data type/structure.
    std::vector<Tri> tris;

    for (size_t triangle_idx = 0; triangle_idx < mesh.indices.size(); triangle_idx++) {
        glm::uvec3 index = mesh.indices[triangle_idx];

        auto v0 = mesh.vertices[index.x];
        auto v1 = mesh.vertices[index.y];
        auto v2 = mesh.vertices[index.z];
        tris.emplace_back(*((Vec3 const *)&v0), *((Vec3 const *)&v1), *((Vec3 const *)&v2));
    }

    bvh::v2::ThreadPool thread_pool;
    bvh::v2::ParallelExecutor executor(thread_pool);

    // Get triangle centers and bounding boxes (required for BVH builder)
    std::vector<BBox> bboxes(tris.size());
    std::vector<Vec3> centers(tris.size());
    executor.for_each(0, tris.size(), [&](size_t begin, size_t end) {
        for (size_t i = begin; i < end; ++i) {
            bboxes[i]  = tris[i].get_bbox();
            centers[i] = tris[i].get_center();
        }
    });

    typename bvh::v2::DefaultBuilder<LibNode>::Config config;
    config.quality       = bvh::v2::DefaultBuilder<LibNode>::Quality::High;
    config.min_leaf_size = 1;
    config.max_leaf_size = m_libbvh_max_leaf_size;
    LibBVH mesh_bvh      = bvh::v2::DefaultBuilder<LibNode>::build(thread_pool, bboxes, centers, config);

    return mesh_bvh;
}

LibBVH CudaBackend::build_libbvh_tlas()
{
    uint32_t num_blas = m_cpu_mesh_bvhs.size();
    assert(num_blas > 0);

    using Vec3 = bvh::v2::Vec<float, 3>;
    using BBox = bvh::v2::BBox<float, 3>;

    bvh::v2::ThreadPool thread_pool;
    bvh::v2::ParallelExecutor executor(thread_pool);

    // Get triangle centers and bounding boxes (required for BVH builder)
    std::vector<BBox> bboxes(num_blas);
    std::vector<Vec3> centers(num_blas);
    executor.for_each(0, num_blas, [&](size_t begin, size_t end) {
        for (size_t i = begin; i < end; ++i) {
            auto bbox  = m_cpu_mesh_bvhs[i].root_bbox();
            bboxes[i]  = BBox(Vec3(bbox.min.x, bbox.min.y, bbox.min.z), Vec3(bbox.max.x, bbox.max.y, bbox.max.z));
            centers[i] = bboxes[i].get_center();
        }
    });

    typename bvh::v2::DefaultBuilder<LibNode>::Config config;
    config.quality       = bvh::v2::DefaultBuilder<LibNode>::Quality::Medium;
    config.min_leaf_size = 1;
    config.max_leaf_size = 1;
    LibBVH tlas_bvh      = bvh::v2::DefaultBuilder<LibNode>::build(thread_pool, bboxes, centers, config);

    return tlas_bvh;
}

void CudaBackend::build_tlas(const Scene &scene)
{
    auto lib_bvh_tlas = std::make_unique<LibBVH>(build_libbvh_tlas());

    std::vector<TLASNode> cpu_tlas_nodes;
    cpu_tlas_nodes.resize(lib_bvh_tlas->nodes.size());

    for (size_t node_id = 0; node_id < lib_bvh_tlas->nodes.size(); node_id++) {
        const auto &node                 = lib_bvh_tlas->nodes[node_id];
        const auto &bbox                 = node.get_bbox();
        cpu_tlas_nodes[node_id].aabb_min = glm::vec3(bbox.min[0], bbox.min[1], bbox.min[2]);
        cpu_tlas_nodes[node_id].aabb_max = glm::vec3(bbox.max[0], bbox.max[1], bbox.max[2]);

        assert((node.index.prim_count == 1 && node.is_leaf()) || (!node.is_leaf() && node.index.prim_count == 0));

        if (!node.is_leaf()) {
            cpu_tlas_nodes[node_id].leaf_type = TLAS_INTERIOR_NODE;
        } else {
            bool is_neural_leaf = std::find(m_neural_bvh_indices.begin(),
                                            m_neural_bvh_indices.end(),
                                            lib_bvh_tlas->prim_ids[node.index.first_id]) != m_neural_bvh_indices.end();

            cpu_tlas_nodes[node_id].leaf_type = is_neural_leaf ? TLAS_NEURAL_LEAF : TLAS_LEAF;
        }
        cpu_tlas_nodes[node_id].left_or_blas_offset = node.index.first_id;
    }
    std::vector<uint32_t> cpu_tlas_indices(lib_bvh_tlas->prim_ids.begin(), lib_bvh_tlas->prim_ids.end());

    m_cuda_tlas_nodes_buffer.alloc_and_upload(cpu_tlas_nodes);
    m_cuda_tlas_indices_buffer.alloc_and_upload(cpu_tlas_indices);

    m_cuda_tlas.tlas_nodes     = m_cuda_tlas_nodes_buffer.d_pointer();
    m_cuda_tlas.num_tlas_nodes = m_cuda_tlas_nodes_buffer.num_elements();
    m_cuda_tlas.blas_indices   = m_cuda_tlas_indices_buffer.d_pointer();
    m_cuda_tlas.mesh_bvhs      = m_cuda_mesh_bvhs_buffer.d_pointer();
    m_cuda_tlas.num_meshes     = m_cuda_mesh_bvhs_buffer.num_elements();

    assert(m_cuda_tlas.num_meshes == m_cuda_tlas_indices_buffer.num_elements());

    logger(LogLevel::Info, "TLAS contains %d nodes", m_cuda_tlas.num_tlas_nodes);

    m_bvh_initialized = true;
}

void CudaBackend::create_cuda_textures_buffer(const Scene &scene)
{
    m_n_textures = scene.textures.size();

    m_texture_arrays.resize(m_n_textures);
    m_texture_objects.resize(m_n_textures);

    for (size_t texture_id = 0; texture_id < m_n_textures; texture_id++) {
        const Image &texture = scene.textures[texture_id];

        cudaResourceDesc res_desc = {};

        cudaChannelFormatDesc channel_desc;
        int32_t width          = texture.width;
        int32_t height         = texture.height;
        int32_t num_components = texture.channels;
        int32_t pitch          = width * num_components * sizeof(uint8_t);
        channel_desc           = cudaCreateChannelDesc<uchar4>();

        cudaArray_t &pixel_array = m_texture_arrays[texture_id];
        CUDA_CHECK(cudaMallocArray(&pixel_array, &channel_desc, width, height));

        CUDA_CHECK(cudaMemcpy2DToArray(pixel_array,
                                       /* offset */ 0,
                                       0,
                                       texture.pixels.data(),
                                       pitch,
                                       pitch,
                                       height,
                                       cudaMemcpyHostToDevice));

        res_desc.resType         = cudaResourceTypeArray;
        res_desc.res.array.array = pixel_array;

        cudaTextureDesc tex_desc     = {};
        tex_desc.addressMode[0]      = cudaAddressModeWrap;
        tex_desc.addressMode[1]      = cudaAddressModeWrap;
        tex_desc.filterMode          = cudaFilterModeLinear;
        tex_desc.readMode            = cudaReadModeNormalizedFloat;
        tex_desc.normalizedCoords    = 1;
        tex_desc.maxAnisotropy       = 1;
        tex_desc.maxMipmapLevelClamp = 99;
        tex_desc.minMipmapLevelClamp = 0;
        tex_desc.mipmapFilterMode    = cudaFilterModePoint;
        tex_desc.borderColor[0]      = 1.0f;
        tex_desc.sRGB                = 0;

        // Create texture object
        cudaTextureObject_t cuda_tex = 0;
        CUDA_CHECK(cudaCreateTextureObject(&cuda_tex, &res_desc, &tex_desc, nullptr));
        m_texture_objects[texture_id] = cuda_tex;
    }

    if (m_texture_objects.size() > 0)
        m_cuda_textures_buffer.alloc_and_upload(m_texture_objects);
}

void CudaBackend::load_envmap_texture(std::string envmap_filename)
{
    if (m_envmap_texture_array) {
        CUDA_CHECK(cudaFreeArray(m_envmap_texture_array));
        CUDA_CHECK(cudaDestroyTextureObject(m_envmap_texture_object));
    }
    // Yes, that's a lot of parenting but that's where it makes sense to store them for now.
    std::filesystem::path envmap_path =
        m_scene_filename.parent_path().parent_path().parent_path() / "hdris" / envmap_filename;

    int width, height;
    int channels = 4;
    float *data;  // width * height * RGBA
    const char *err = nullptr;

    int ret = LoadEXR(&data, &width, &height, envmap_path.string().c_str(), &err);
    if (ret != TINYEXR_SUCCESS) {
        width  = 4;
        height = 4;
        logger(LogLevel::Warn, "Envmap not found at %s, setting white image", envmap_path.string().c_str());
        data = (float *)malloc(width * height * channels * sizeof(float));
        for (size_t i = 0; i < width * height * channels; i++) {
            data[i] = 1.0f;
        }
    } else {
        logger(LogLevel::Info, "Loaded Envmap from %s", envmap_path.string().c_str());
    }

    cudaResourceDesc res_desc = {};

    cudaChannelFormatDesc channel_desc;
    int32_t num_components = channels;
    int32_t pitch          = width * num_components * sizeof(float);
    channel_desc           = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);

    CUDA_CHECK(cudaMallocArray(&m_envmap_texture_array, &channel_desc, width, height));

    CUDA_CHECK(cudaMemcpy2DToArray(m_envmap_texture_array,
                                   /* offset */ 0,
                                   0,
                                   data,
                                   pitch,
                                   pitch,
                                   height,
                                   cudaMemcpyHostToDevice));

    res_desc.resType         = cudaResourceTypeArray;
    res_desc.res.array.array = m_envmap_texture_array;

    cudaTextureDesc tex_desc  = {};
    tex_desc.filterMode       = cudaFilterModeLinear;
    tex_desc.normalizedCoords = true;
    tex_desc.addressMode[0]   = cudaAddressModeClamp;
    tex_desc.addressMode[1]   = cudaAddressModeClamp;

    // Create texture object
    CUDA_CHECK(cudaCreateTextureObject(&m_envmap_texture_object, &res_desc, &tex_desc, nullptr));

    free(data);
}

size_t CudaBackend::n_meshes()
{
    return m_n_meshes;
}

size_t CudaBackend::n_vertices()
{
    return m_n_vertices;
}

size_t CudaBackend::n_textures()
{
    return m_n_textures;
}

void CudaBackend::set_reference_image(Bitmap<float> reference_image)
{
    if (reference_image.m_channels != 4) {
        logger(LogLevel::Error, "Only 4 channel reference images are supported currently, reference not set.");
        return;
    }
    if (m_reference_image_buffer.d_ptr) {
        m_reference_image_buffer.free();
    }
    CUDA_CHECK(cudaDestroyTextureObject(m_reference_image_surface));

    m_reference_image_buffer.alloc_and_upload(reference_image.m_image_data);

    // Create a cuda texture out of this image
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType                  = cudaResourceTypePitch2D;
    resDesc.res.pitch2D.devPtr       = m_reference_image_buffer.d_pointer();
    resDesc.res.pitch2D.desc         = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
    resDesc.res.pitch2D.width        = reference_image.m_width;
    resDesc.res.pitch2D.height       = reference_image.m_height;
    resDesc.res.pitch2D.pitchInBytes = reference_image.m_width * reference_image.m_channels * sizeof(float);

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.filterMode       = cudaFilterModeLinear;
    texDesc.normalizedCoords = true;
    texDesc.addressMode[0]   = cudaAddressModeClamp;
    texDesc.addressMode[1]   = cudaAddressModeClamp;
    texDesc.addressMode[2]   = cudaAddressModeClamp;

    CUDA_CHECK(cudaCreateTextureObject(&m_reference_image_surface, &resDesc, &texDesc, nullptr));

    m_reference_image_set = true;
}

void CudaBackend::readback_framebuffer(size_t buffer_size, unsigned char *buffer)
{
    assert((m_fb_size.x * static_cast<size_t>(m_fb_size.y) * 4 * sizeof(unsigned char)) == buffer_size);

    // Unmap  opengl->cuda mapped memory
    release_display_mapped_memory();

    glBindTexture(GL_TEXTURE_2D, m_display->gl_framebuffer_texture());
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, buffer);
    glBindTexture(GL_TEXTURE_2D, 0);

    // Remap opengl->cuda memory
    map_display_memory();
}

void CudaBackend::readback_framebuffer_float(size_t buffer_size, float *buffer)
{
    assert(m_fb_size.x * static_cast<size_t>(m_fb_size.y) * 4 * sizeof(float) == buffer_size);

    // Unmap  opengl->cuda mapped memory
    release_display_mapped_memory();

    glBindTexture(GL_TEXTURE_2D, m_display->gl_accum_buffer_texture());
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, reinterpret_cast<unsigned char *>(buffer));
    glBindTexture(GL_TEXTURE_2D, 0);

    // Remap opengl->cuda memory
    map_display_memory();
}

void CudaBackend::resize(const glm::uvec2 &new_size)
{
    // if window minimized
    if ((new_size.x == 0) || (new_size.y == 0))
        return;

    m_fb_size = new_size;
}

void CudaBackend::update_camera(const Camera &camera)
{
    glm::vec2 img_plane_size;
    img_plane_size.y  = 2.f * std::tan(glm::radians(0.5f * camera.fovy()));
    float aspect      = static_cast<float>(m_fb_size.x) / m_fb_size.y;
    img_plane_size.x  = img_plane_size.y * aspect;
    glm::vec3 cam_dir = camera.dir();

    const glm::vec3 dir_du       = glm::normalize(glm::cross(cam_dir, camera.up())) * img_plane_size.x;
    const glm::vec3 dir_dv       = -glm::normalize(glm::cross(dir_du, cam_dir)) * img_plane_size.y;
    const glm::vec3 dir_top_left = cam_dir - 0.5f * dir_du - 0.5f * dir_dv;

    m_camera_params.origin       = camera.origin();
    m_camera_params.du           = glm::vec4(dir_du, 0.f);
    m_camera_params.dv           = glm::vec4(dir_dv, 0.f);
    m_camera_params.dir_top_left = glm::vec4(dir_top_left, 0.f);
}

void CudaBackend::update_scene_data(const Scene &scene)
{
    // Free previously allocated resources
    free_scene_data();
    // Load the new scene

    load_scene(scene);
    build_blas(scene);
    build_tlas(scene);
}
}
