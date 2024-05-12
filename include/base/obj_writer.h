#pragma once

#include "base/bbox.h"

#include <fstream>
#include <string>

namespace ntwr {

struct ObjWriter {
    ObjWriter(std::string filename) : m_vertex_index(1), m_obj_file(filename) {}

    void add_bbox(const Bbox3f &bbox)
    {
        int v1 = m_vertex_index;
        m_vertex_index += 8;
        glm::vec3 vertices[] = {bbox.min,
                                glm::vec3(bbox.min.x, bbox.min.y, bbox.max.z),
                                glm::vec3(bbox.min.x, bbox.max.y, bbox.min.z),
                                glm::vec3(bbox.min.x, bbox.max.y, bbox.max.z),
                                glm::vec3(bbox.max.x, bbox.min.y, bbox.min.z),
                                glm::vec3(bbox.max.x, bbox.min.y, bbox.max.z),
                                glm::vec3(bbox.max.x, bbox.max.y, bbox.min.z),
                                bbox.max};

        for (int i = 0; i < 8; i++) {
            write_vertex(vertices[i]);
        }
        // Write faces
        write_face(v1, v1 + 1, v1 + 3);
        write_face(v1, v1 + 3, v1 + 2);
        write_face(v1, v1 + 2, v1 + 6);
        write_face(v1, v1 + 6, v1 + 4);
        write_face(v1, v1 + 4, v1 + 5);
        write_face(v1, v1 + 5, v1 + 1);
        write_face(v1 + 1, v1 + 5, v1 + 7);
        write_face(v1 + 1, v1 + 7, v1 + 3);
        write_face(v1 + 2, v1 + 3, v1 + 7);
        write_face(v1 + 2, v1 + 7, v1 + 6);
        write_face(v1 + 4, v1 + 6, v1 + 7);
        write_face(v1 + 4, v1 + 7, v1 + 5);
    }

    void add_group(std::string group_name)
    {
        m_obj_file << "o " << group_name << "\n";
    }

    void close()
    {
        m_obj_file.close();
    }

private:
    // Helper function to write a vertex to the OBJ file.
    void write_vertex(const glm::vec3 &vertex)
    {
        m_obj_file << "v " << vertex.x << " " << vertex.y << " " << vertex.z << "\n";
    }

    // Helper function to write a face to the OBJ file.
    void write_face(int v1, int v2, int v3)
    {
        m_obj_file << "f " << v1 << " " << v2 << " " << v3 << "\n";
    }

    int m_vertex_index;
    std::ofstream m_obj_file;
};

}
