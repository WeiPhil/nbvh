#pragma once

#include <vector>
#include "glm_pch.h"
#include "material.h"

namespace ntwr {

struct Mesh {
    std::vector<glm::vec3> vertices;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec2> texcoords;
    std::vector<glm::uvec3> indices;
    // material data:
    // glm::vec3 diffuse;

    std::vector<int> material_ids;
    std::vector<uint32_t> material_map;

    Mesh(const std::vector<int> material_ids);

    Mesh() = default;

    inline size_t num_tris() const
    {
        return indices.size();
    }
};

}