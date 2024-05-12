#pragma once

#include "base/bbox.h"
#include "base/camera/camera.h"
#include "base/material.h"
#include "base/mesh.h"

#include "tiny_gltf.h"

#include <filesystem>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace ntwr {

class Scene {
public:
    Scene(std::filesystem::path scene_filename);

    std::vector<Mesh> meshes;
    std::vector<uint32_t> neural_mesh_indices;

    std::vector<PackedMaterial> materials;
    std::vector<Image> textures;
    std::unique_ptr<Camera> camera;

    std::filesystem::path m_scene_filename;

private:
    void load_gltf_scene(std::filesystem::path scene_filename);
    void load_gltf_camera(tinygltf::Model &model);
    std::unique_ptr<tinygltf::Model> gltf_model;
};

}