

#include "base/scene.h"
#include "base/camera/flycam.h"
#include "utils/logger.h"

#include "base/gltf/gltf_buffer_view.h"
#include "base/gltf/gltf_flatten.h"
#include "base/gltf/gltf_types.h"

// define all those  in only *one* cpp source file
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define TINYGLTF_IMPLEMENTATION
#define TINYGLTF_NOEXCEPTION
#include "tiny_gltf.h"

#include <iostream>
#include "glm_pch.h"

namespace ntwr {

Scene::Scene(std::filesystem::path scene_filename) : m_scene_filename(scene_filename)
{
    // Default camera
    camera = std::make_unique<FlyCam>();

    if (scene_filename.extension() == ".gltf" || scene_filename.extension() == ".glb") {
        load_gltf_scene(scene_filename);
    } else {
        logger(LogLevel::Error,
               "Scene file with extension %s not supported, only gltf and glb format is supported",
               scene_filename.extension().string().c_str());
        exit(-1);
    }
}

void add_primitives_to_mesh(Mesh &mesh,
                            uint32_t &index_offset,
                            const tinygltf::Model &model,
                            const tinygltf::Mesh &gltf_mesh,
                            const glm::mat4 &node_transform)
{
    glm::mat3 node_transform_normal = glm::inverseTranspose(glm::mat3(node_transform));

    for (auto &p : gltf_mesh.primitives) {
        mesh.material_map.push_back(mesh.indices.size());

        if (p.mode != TINYGLTF_MODE_TRIANGLES) {
            logger(LogLevel::Error, "Unsupported primitive mode! File must contain only triangles");
            throw std::runtime_error("Unsupported primitive mode! Only triangles are supported");
        }

        // Note: assumes there is a POSITION (is this required by the gltf spec?)
        AccessorWrapper<glm::vec3> pos_accessor(model.accessors[p.attributes.at("POSITION")], model);
        for (size_t i = 0; i < pos_accessor.size(); ++i) {
            glm::vec3 vertex_pos = glm::vec3(pos_accessor[i]);
            if (node_transform != glm::mat4(1.f)) {
                vertex_pos = node_transform * glm::vec4(vertex_pos, 1.f);
            }
            mesh.vertices.push_back(vertex_pos);
        }

        uint32_t current_indices_count = pos_accessor.size();

        // Note: GLTF can have multiple texture coordinates used by different textures
        // we don't support this yet
        auto fnd = p.attributes.find("TEXCOORD_0");
        if (fnd != p.attributes.end()) {
            // TODO : uv transformation ?
            AccessorWrapper<glm::vec2> uv_accessor(model.accessors[fnd->second], model);
            for (size_t i = 0; i < uv_accessor.size(); ++i) {
                mesh.texcoords.push_back(uv_accessor[i]);
            }
        } else if (index_offset != 0) {
            // Making sure we always offset the data appropriately with dummy values to get the right data for
            // multi-material meshes
            logger(LogLevel::Warn, "Suboptimal texcoord data storage (added dummy values)");
            mesh.texcoords.resize(index_offset + current_indices_count);
        }

        fnd = p.attributes.find("NORMAL");
        if (fnd != p.attributes.end()) {
            AccessorWrapper<glm::vec3> normal_accessor(model.accessors[fnd->second], model);
            for (size_t i = 0; i < normal_accessor.size(); ++i) {
                glm::vec3 normal_vec = glm::vec3(normal_accessor[i]);
                if (node_transform != glm::mat4(1.f)) {
                    normal_vec = node_transform_normal * normal_vec;
                }
                mesh.normals.push_back(normal_vec);
            }
        } else if (index_offset != 0) {
            // Making sure we always offset the data appropriately with dummy values to get the right data for
            // multi-material meshes
            logger(LogLevel::Warn, "Suboptimal normal data storage (added dummy values)");
            mesh.normals.resize(index_offset + current_indices_count);
        }

        //  Finally add vertices
        if (model.accessors[p.indices].componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
            AccessorWrapper<uint16_t> index_accessor(model.accessors[p.indices], model);
            for (size_t i = 0; i < index_accessor.size() / 3; ++i) {
                mesh.indices.push_back(glm::uvec3(index_offset + index_accessor[i * 3],
                                                  index_offset + index_accessor[i * 3 + 1],
                                                  index_offset + index_accessor[i * 3 + 2]));
            }

        } else if (model.accessors[p.indices].componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT) {
            AccessorWrapper<uint32_t> index_accessor(model.accessors[p.indices], model);
            for (size_t i = 0; i < index_accessor.size() / 3; ++i) {
                mesh.indices.push_back(glm::uvec3(index_offset + index_accessor[i * 3],
                                                  index_offset + index_accessor[i * 3 + 1],
                                                  index_offset + index_accessor[i * 3 + 2]));
            }

        } else {
            logger(LogLevel::Error, "Unsupported index type");
            throw std::runtime_error("Unsupported index component type");
        }

        index_offset += current_indices_count;
        mesh.material_ids.push_back(p.material);

        if (p.material != -1 && model.materials[p.material].emissiveFactor.size() > 0) {
            // TODO : Add area lights for sampling
        }
    }
}

void load_node_as_meshes(const tinygltf::Node &node,
                         Mesh *neural_mesh,
                         std::vector<Mesh> &normal_meshes,
                         uint32_t &index_offset,
                         uint32_t &loaded_gltf_meshes,
                         std::set<std::string> &neural_gltf_mesh_names,
                         const tinygltf::Model &model,
                         const glm::mat4 &parent_transform = glm::mat4(1.f))
{
    glm::mat4 node_transform = parent_transform * read_node_transform(node);

    if (node.children.size() > 0) {
        // logger(LogLevel::Warn, "Node contains %d children, experimental handling.", node.children.size());

        for (int child_node_idx : node.children) {
            load_node_as_meshes(model.nodes[child_node_idx],
                                neural_mesh,
                                normal_meshes,
                                index_offset,
                                loaded_gltf_meshes,
                                neural_gltf_mesh_names,
                                model,
                                node_transform);
        }
    }

    if (node.mesh == -1)
        return;

    const tinygltf::Mesh &gltf_mesh = model.meshes[node.mesh];
    if (neural_mesh) {
        neural_gltf_mesh_names.insert(gltf_mesh.name);
    } else if (neural_gltf_mesh_names.contains(gltf_mesh.name)) {
        return;
    }

    if (neural_mesh) {
        add_primitives_to_mesh(*neural_mesh, index_offset, model, gltf_mesh, node_transform);
    } else {
        Mesh normal_mesh;
        uint32_t normal_index_offset = 0;
        add_primitives_to_mesh(normal_mesh, normal_index_offset, model, gltf_mesh, node_transform);
        normal_meshes.push_back(normal_mesh);
    }
    loaded_gltf_meshes++;
}

void Scene::load_gltf_scene(std::filesystem::path scene_filename)
{
    logger(LogLevel::Info, "Loading GLTF scene");

    assert(scene_filename.extension() == ".gltf" || scene_filename.extension() == ".glb");
    gltf_model = std::make_unique<tinygltf::Model>();
    tinygltf::TinyGLTF loader;
    std::string err;
    std::string warn;

    bool ret = false;
    if (scene_filename.extension() == ".gltf")
        ret = loader.LoadASCIIFromFile(gltf_model.get(), &err, &warn, scene_filename.string().c_str());

    if (scene_filename.extension() == ".glb")  // for binary glTF
        ret = loader.LoadBinaryFromFile(gltf_model.get(), &err, &warn, scene_filename.string().c_str());

    if (!warn.empty()) {
        logger(LogLevel::Warn, "%s", warn.c_str());
    }

    if (!err.empty()) {
        logger(LogLevel::Error, "%s", err.c_str());
        exit(-1);
    }

    if (!ret) {
        logger(LogLevel::Error, "Failed to parse glTF");
        exit(-1);
    }

    tinygltf::Model &model = *gltf_model.get();

    logger(LogLevel::Info, "The scene is composed of %d sub-scenes.", model.scenes.size());

    std::set<uint32_t> neural_scene_indices;

    std::string neural_scene_str = "neural";
    for (uint32_t scene_idx = 0; scene_idx < model.scenes.size(); scene_idx++) {
        if (model.scenes[scene_idx].name.find(neural_scene_str) != std::string::npos) {
            logger(LogLevel::Info, "Found neural sub-scene \"%s\"", model.scenes[scene_idx].name.c_str());
            neural_scene_indices.insert(scene_idx);
        }
    }
    if (neural_scene_indices.size() == 0) {
        logger(LogLevel::Warn, "No neural sub-scene found, assuming fully neural scene.");
        neural_scene_indices.insert(0);
    }
    if (neural_scene_indices.size() != 1) {
        logger(LogLevel::Info, "We only support exactly 1 neural sub-scene for now.");
    }

    uint32_t loaded_gltf_meshes = 0;
    // Load first any neural mesh in a a single mesh
    assert(neural_scene_indices.size() == 1);
    std::set<std::string> neural_gltf_mesh_names;
    for (auto neural_scene_idx : neural_scene_indices) {
        auto &neural_scene = model.scenes[neural_scene_idx];

        Mesh mesh;
        uint32_t index_offset = 0;

        for (int node_idx : neural_scene.nodes) {
            auto &node = model.nodes[node_idx];

            load_node_as_meshes(node, &mesh, meshes, index_offset, loaded_gltf_meshes, neural_gltf_mesh_names, model);
        }

        neural_mesh_indices.push_back(meshes.size());
        meshes.push_back(mesh);
    }

    // Only one neural mesh for now
    assert(neural_mesh_indices.size() == 1);
    // Load the remaining scene/meshes while avoiding the loading of the neural ones again
    for (uint32_t scene_idx = 0; scene_idx < model.scenes.size(); scene_idx++) {
        if (neural_scene_indices.contains(scene_idx))
            continue;

        auto &gltf_scene = model.scenes[scene_idx];
        for (int node_idx : gltf_scene.nodes) {
            auto &node = model.nodes[node_idx];

            uint32_t index_offset = 0;
            load_node_as_meshes(node, nullptr, meshes, index_offset, loaded_gltf_meshes, neural_gltf_mesh_names, model);
        }
    }

    std::set<int> neural_material_ids;
    std::set<int> non_neural_material_ids;
    for (size_t mesh_idx = 0; mesh_idx < meshes.size(); mesh_idx++) {
        bool is_neural_mesh =
            std::find(neural_mesh_indices.begin(), neural_mesh_indices.end(), mesh_idx) != neural_mesh_indices.end();

        for (auto &material_id : meshes[mesh_idx].material_ids) {
            if (is_neural_mesh) {
                neural_material_ids.insert(material_id);
            } else {
                non_neural_material_ids.insert(material_id);
            }
        }
    }

    // Load images
    for (const auto &img : model.images) {
        if (img.component != 4) {
            logger(LogLevel::Warn, "Check non-4 component image support");
        }
        if (img.pixel_type != TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE) {
            logger(LogLevel::Error,
                   "Non-uchar images are not supported, found type %d for image %s",
                   img.pixel_type,
                   img.name.c_str());
            // throw std::runtime_error("Unsupported image pixel type");
        }

        Image texture;
        texture.name     = img.name;
        texture.width    = img.width;
        texture.height   = img.height;
        texture.channels = img.component;
        texture.pixels   = img.image;
        // Assume linear unless we find it used as a color texture
        texture.color_space = LINEAR;
        textures.push_back(texture);
    }

    uint32_t total_albedo_normal_texture_size_byte            = 0;
    uint32_t total_neural_albedo_normal_texture_size_byte     = 0;
    uint32_t total_non_neural_albedo_normal_texture_size_byte = 0;
    uint32_t total_non_neural_only_texture_size_byte          = 0;
    // Load materials
    for (uint32_t material_id = 0; material_id < model.materials.size(); material_id++) {
        const auto &m = model.materials[material_id];
        PackedMaterial mat;
        mat.base_color.x = (float)m.pbrMetallicRoughness.baseColorFactor[0];
        mat.base_color.y = (float)m.pbrMetallicRoughness.baseColorFactor[1];
        mat.base_color.z = (float)m.pbrMetallicRoughness.baseColorFactor[2];

        mat.metallic = (float)m.pbrMetallicRoughness.metallicFactor;

        mat.roughness = (float)m.pbrMetallicRoughness.roughnessFactor;

        if (m.pbrMetallicRoughness.baseColorTexture.index != -1) {
            const int32_t id         = model.textures[m.pbrMetallicRoughness.baseColorTexture.index].source;
            textures[id].color_space = SRGB;

            uint32_t tex_mask = TEXTURED_PARAM_MASK;
            SET_TEXTURE_ID(tex_mask, id);
            mat.base_color.r = glm::uintBitsToFloat(tex_mask);

            // Assuming 8bit channels that's juste one byte per channel
            total_albedo_normal_texture_size_byte += textures[id].width * textures[id].height * textures[id].channels;
            if (neural_material_ids.contains(material_id))
                total_neural_albedo_normal_texture_size_byte +=
                    textures[id].width * textures[id].height * textures[id].channels;
            if (non_neural_material_ids.contains(material_id)) {
                total_non_neural_albedo_normal_texture_size_byte +=
                    textures[id].width * textures[id].height * textures[id].channels;
                // logger(LogLevel::Error,
                //         "Base color texture of material id : %d taking %g MB space",
                //         material_id,
                //         textures[id].width * textures[id].height * textures[id].channels * 1e-6f);
            }
        }

        if (m.normalTexture.index != -1) {
            const int32_t id         = model.textures[m.normalTexture.index].source;
            textures[id].color_space = LINEAR;
            uint32_t tex_mask        = TEXTURED_PARAM_MASK;
            SET_TEXTURE_ID(tex_mask, id);
            mat.normal.r = glm::uintBitsToFloat(tex_mask);

            total_albedo_normal_texture_size_byte += textures[id].width * textures[id].height * textures[id].channels;
            if (neural_material_ids.contains(material_id))
                total_neural_albedo_normal_texture_size_byte +=
                    textures[id].width * textures[id].height * textures[id].channels;
            if (non_neural_material_ids.contains(material_id)) {
                total_non_neural_albedo_normal_texture_size_byte +=
                    textures[id].width * textures[id].height * textures[id].channels;
                // logger(LogLevel::Error,
                //         "Normal texture of material id : %d taking %g MB space",
                //         material_id,
                //         textures[id].width * textures[id].height * textures[id].channels * 1e-6f);
            }
        }

        // glTF: metallic is blue channel, roughness is green channel
        if (m.pbrMetallicRoughness.metallicRoughnessTexture.index != -1) {
            const int32_t id         = model.textures[m.pbrMetallicRoughness.metallicRoughnessTexture.index].source;
            textures[id].color_space = LINEAR;

            uint32_t tex_mask = TEXTURED_PARAM_MASK;
            SET_TEXTURE_ID(tex_mask, id);
            SET_TEXTURE_CHANNEL(tex_mask, 2);
            mat.metallic = glm::uintBitsToFloat(tex_mask);

            tex_mask = TEXTURED_PARAM_MASK;
            SET_TEXTURE_ID(tex_mask, id);
            SET_TEXTURE_CHANNEL(tex_mask, 1);
            mat.roughness = glm::uintBitsToFloat(tex_mask);

            if (non_neural_material_ids.contains(material_id))
                total_non_neural_only_texture_size_byte += textures[id].width * textures[id].height * 2;
        }

        // Transmission
        auto const_it = m.extensions.find("KHR_materials_transmission");
        if (const_it != m.extensions.end()) {
            mat.specular_transmission = const_it->second.Get("transmissionFactor").GetNumberAsDouble();
        }

        // IoR
        const_it = m.extensions.find("KHR_materials_ior");
        if (const_it != m.extensions.end()) {
            mat.ior = const_it->second.Get("ior").GetNumberAsDouble();
        }

        // Emission
        if (m.emissiveFactor.size() > 0) {
            mat.base_emission.x = (float)m.emissiveFactor[0];
            mat.base_emission.y = (float)m.emissiveFactor[1];
            mat.base_emission.z = (float)m.emissiveFactor[2];
            if (m.emissiveTexture.index != -1) {
                const int32_t id         = model.textures[m.emissiveTexture.index].source;
                textures[id].color_space = SRGB;

                uint32_t tex_mask = TEXTURED_PARAM_MASK;
                SET_TEXTURE_ID(tex_mask, id);
                mat.base_emission.r = glm::uintBitsToFloat(tex_mask);
            }
            const_it = m.extensions.find("KHR_materials_emissive_strength");
            if (const_it != m.extensions.end()) {
                mat.emission_scale = const_it->second.Get("emissiveStrength").GetNumberAsDouble();
            } else {
                mat.emission_scale = 1.f;
            }
        }

        materials.push_back(mat);
    }

    // Load cameras
    load_gltf_camera(model);
}

void Scene::load_gltf_camera(tinygltf::Model &model)
{
    int camera_idx = -1;
    // Access and parse cameras from the GLTF model
    for (size_t i = 0; i < model.cameras.size(); ++i) {
        // Access camera properties based on camera type (perspective or orthographic)
        if (model.cameras[i].type == "perspective") {
            camera_idx = (int)i;
        } else if (model.cameras[i].type == "orthographic") {
            logger(LogLevel::Warn, "Orthographic camera not supported, ignored");
        }
    }

    if (camera_idx != -1) {
        for (size_t i = 0; i < model.nodes.size(); i++) {
            const tinygltf::Node &node = model.nodes[i];

            if (node.camera == camera_idx) {
                logger(LogLevel::Info, "Found perspective camera with transform");

                const tinygltf::PerspectiveCamera &perspective_camera = model.cameras[camera_idx].perspective;

                glm::mat4 camera_transform = read_node_transform(node);
                camera->set_transform(camera_transform);
                camera->set_fovy_from_radian(static_cast<float>(perspective_camera.yfov));
            }
        }
    }
}
}