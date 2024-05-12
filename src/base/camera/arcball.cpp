

#include "base/camera/arcball.h"
#include "utils/logger.h"

#include "glm_pch.h"

namespace ntwr {

ArcballCam::ArcballCam(glm::vec3 origin, glm::vec3 target, glm::vec3 up) : m_origin(origin), m_target(target), m_up(up)
{
    update_camera();
    m_camera_movement_type = CameraMovementType::Arcball;
    m_sensitivity          = 3.0f;
}

bool ArcballCam::ui()
{
    bool camera_changed = Camera::ui();

    return camera_changed;
}

bool ArcballCam::movement_io(ImGuiIO &io, glm::uvec2 window_size)
{
    auto transform_mouse = [&window_size](glm::vec2 in) {
        return glm::vec2(in.x * 2.f / float(window_size.x) - 1.f, 1.f - 2.f * in.y / float(window_size.y));
    };

    bool camera_changed = false;
    if (!io.WantCaptureMouse) {
        if (io.MouseDelta.x != 0.0f || io.MouseDelta.y != 0.0f) {
            const glm::vec2 cur_mouse = transform_mouse(glm::vec2(io.MousePos.x, io.MousePos.y));
            const glm::vec2 prev_mouse =
                transform_mouse(glm::vec2(io.MousePos.x, io.MousePos.y) - glm::vec2(io.MouseDelta.x, io.MouseDelta.y));
            if (io.MouseDown[0]) {
                this->rotate(prev_mouse, cur_mouse);
                camera_changed = true;
            } else if (io.MouseDown[1]) {
                this->pan(cur_mouse - prev_mouse);
                camera_changed = true;
            }
        }
        if (io.MouseWheel != 0.0f) {
            this->zoom(io.MouseWheel * 0.1f);
            camera_changed = true;
        }
    }
    if (!io.WantCaptureKeyboard) {
        if (ImGui::IsKeyDown(ImGuiKey_S)) {
            this->move_along_dir(1.f, io.DeltaTime);
            camera_changed = true;
        }
        if (ImGui::IsKeyDown(ImGuiKey_W)) {
            this->move_along_dir(-1.0, io.DeltaTime);
            camera_changed = true;
        }

        if (ImGui::IsKeyPressed(ImGuiKey_P)) {
            auto origin = m_origin;
            auto center = m_target;
            auto up     = this->up();
            logger(LogLevel::Info,
                   "origin %f %f %f\ncenter %f %f %f\nup %f %f %f",
                   origin.x,
                   origin.y,
                   origin.z,
                   center.x,
                   center.y,
                   center.z,
                   up.x,
                   up.y,
                   up.z);
        }
    }
    return camera_changed;
}

void ArcballCam::move_along_dir(float distance, float delta_time)
{
    glm::vec3 dir = normalize(m_target - m_origin);
    m_origin += dir * (-distance * this->m_speed * delta_time);
    update_camera();
}

void ArcballCam::rotate(glm::vec2 prev_mouse, glm::vec2 cur_mouse)
{
    glm::vec3 dir = normalize(m_target - m_origin);
    float x_angle = (prev_mouse.x - cur_mouse.x) * m_sensitivity;
    float y_angle = -(prev_mouse.y - cur_mouse.y) * m_sensitivity;

    float cosAngle = dot(dir, m_up);
    if (cosAngle * glm::sign(y_angle) > 0.99f)
        y_angle = 0;

    glm::mat4x4 rot_x = glm::rotate(x_angle, m_up);
    m_origin          = glm::vec3(rot_x * glm::vec4((m_origin - m_target), 1.f)) + m_target;

    glm::vec3 right = normalize(cross(dir, m_up));

    glm::mat4x4 rot_y = glm::rotate(y_angle, right);

    m_origin = glm::vec3(rot_y * glm::vec4((m_origin - m_target), 1.f)) + m_target;

    update_camera();
}

void ArcballCam::pan(glm::vec2 mouse_delta)
{
    glm::vec3 dir   = normalize(m_target - m_origin);
    glm::vec3 right = normalize(cross(dir, m_up));

    m_target -= right * (mouse_delta.x * length(m_origin));
    m_target -= m_up * (mouse_delta.y * length(m_origin));

    update_camera();
}

void ArcballCam::set_origin(glm::vec3 origin)
{
    m_origin = origin;
    update_camera();
}

void ArcballCam::set_target(glm::vec3 target)
{
    m_target = target;
    update_camera();
}

void ArcballCam::zoom(const float zoom_amount)
{
    m_speed *= std::exp(zoom_amount);
}

void ArcballCam::update_camera()
{
    m_to_world     = glm::lookAt(m_origin, m_target, m_up);
    m_inv_to_world = glm::inverse(m_to_world);
}

}
