#pragma once

#include <vector>
#include "base/camera/camera.h"
#include "glm_pch.h"

namespace ntwr {

struct ArcballCam : public Camera {
    ArcballCam(glm::vec3 origin = glm::vec3(0.f, 0.f, -4.f),
               glm::vec3 target = glm::vec3(0.f, 0.f, 0.f),
               glm::vec3 up     = glm::vec3(0.f, 1.f, 0.f));

    virtual bool ui() override;
    virtual bool movement_io(ImGuiIO &io, glm::uvec2 window_size) override;

    void move_along_dir(float distance, float delta_time);
    void rotate(glm::vec2 prev_mouse, glm::vec2 cur_mouse);
    void zoom(const float zoom_amount);
    void pan(glm::vec2 mouse_delta);
    void set_origin(glm::vec3 origin);
    void set_target(glm::vec3 target);

private:
    void update_camera();

    glm::mat4 m_target_translation, m_translation;
    glm::quat m_rotation;

    glm::vec3 m_origin;
    glm::vec3 m_target;
    glm::vec3 m_up;
};
}