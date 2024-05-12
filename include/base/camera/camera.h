#pragma once

#include <vector>
#include "glm_pch.h"
#include "imgui.h"

namespace ntwr {

static const char *const camera_movement_types[] = {"Fly", "Arcball"};

enum CameraMovementType { Fly, Arcball };

class Camera {
public:
    virtual ~Camera(){};

    void set_transform(glm::mat4 to_world);
    void set_fovy_from_radian(float fovy_radian);
    void set_fovy_from_degree(float fovy_degree);
    const glm::mat4 &transform() const;
    const glm::mat4 &inv_transform() const;
    glm::vec3 origin() const;
    glm::vec3 dir() const;
    glm::vec3 up() const;
    float fovy() const;

    virtual bool ui() = 0;

    virtual bool movement_io(ImGuiIO &io, glm::uvec2 window_size) = 0;

    virtual CameraMovementType camera_movement_type()
    {
        return m_camera_movement_type;
    }

protected:
    glm::mat4 m_to_world;
    glm::mat4 m_inv_to_world;
    float m_fovy        = 45.0;
    float m_sensitivity = 1;
    float m_speed       = 1;

    CameraMovementType m_camera_movement_type;
};

}