

#include "base/camera/flycam.h"
#include "utils/logger.h"

#include "glm_pch.h"

namespace ntwr {

FlyCam::FlyCam(glm::vec3 up, glm::vec3 origin, glm::vec3 dir) : global_up(up)
{
    set_origin(origin);
    set_direction(dir);
    m_camera_movement_type = CameraMovementType::Fly;
}

bool FlyCam::ui()
{
    bool camera_changed = Camera::ui();

    glm::vec3 origin = this->origin();
    if (ImGui::InputFloat3("origin", &origin.x)) {
        this->set_origin(origin);
        camera_changed = true;
    }
    glm::vec3 dir    = this->dir();
    glm::vec3 up     = this->up();
    bool dir_changed = ImGui::InputFloat3("direction", &dir.x);
    bool up_changed  = ImGui::InputFloat3("up", &up.x);
    if (up_changed) {  // may include dir_changed!
        this->set_direction(dir, up);
        camera_changed = true;
    } else if (dir_changed) {  // only direction changed
        this->set_direction(dir);
        camera_changed = true;
    }

    return camera_changed;
}

bool FlyCam::movement_io(ImGuiIO &io, glm::uvec2 window_size)
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
            this->move_local(glm::vec3(0, 0, 1), io.DeltaTime);
            camera_changed = true;
        }
        if (ImGui::IsKeyDown(ImGuiKey_W)) {
            this->move_local(glm::vec3(0, 0, -1), io.DeltaTime);
            camera_changed = true;
        }
        if (ImGui::IsKeyDown(ImGuiKey_D)) {
            this->move_local(glm::vec3(1, 0, 0), io.DeltaTime);
            camera_changed = true;
        }
        if (ImGui::IsKeyDown(ImGuiKey_A)) {
            this->move_local(glm::vec3(-1, 0, 0), io.DeltaTime);
            camera_changed = true;
        }
        if (ImGui::IsKeyDown(ImGuiKey_Space)) {
            this->move_local(glm::vec3(0, 1, 0), io.DeltaTime);
            camera_changed = true;
        }
        if (ImGui::IsKeyDown(ImGuiKey_Q)) {
            this->move_local(glm::vec3(0, -1, 0), io.DeltaTime);
            camera_changed = true;
        }
        if (ImGui::IsKeyPressed(ImGuiKey_P)) {
            auto origin = this->origin();
            auto center = this->center();
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

void FlyCam::rotate(glm::vec2 prev_mouse, glm::vec2 cur_mouse)
{
    glm::vec2 mouse_delta = cur_mouse - prev_mouse;
    m_to_world            = glm::rotate(mouse_delta.y * m_sensitivity, glm::vec3(-1, 0, 0)) * m_to_world;
    glm::vec3 local_up    = normalize(glm::vec3{m_to_world * glm::vec4{global_up.x, global_up.y, global_up.z, 0}});
    m_to_world            = glm::rotate(mouse_delta.x * m_sensitivity, local_up) * m_to_world;
    update_camera();
}

void FlyCam::pan(glm::vec2 mouse_delta)
{
    move_local(glm::vec3(mouse_delta.x, mouse_delta.y, 0), length(origin()));
    update_camera();
}

void FlyCam::zoom(float zoom_amount)
{
    m_speed *= std::exp(zoom_amount);
}

void FlyCam::move_local(glm::vec3 local_dir, float delta_time)
{
    m_to_world = glm::translate(local_dir * -(this->m_speed * delta_time)) * m_to_world;
    update_camera();
}

void FlyCam::update_camera()
{
    m_inv_to_world = inverse(m_to_world);
}

glm::vec3 FlyCam::center() const
{
    return origin() + dir();
}

void FlyCam::set_origin(glm::vec3 origin)
{
    m_inv_to_world[3] = glm::vec4(origin, 1.0f);
    m_to_world        = inverse(m_inv_to_world);
}

void FlyCam::set_direction(glm::vec3 dir)
{
    dir                  = normalize(dir);
    glm::vec3 prev_right = cross(this->up(), -this->dir());
    glm::vec3 right      = cross(global_up, -dir);
    float sin_theta      = glm::length(right);
    if (sin_theta < 0.2f) {      // ~12 deg off global up
        if (sin_theta < 0.001f)  // ~0.05 deg off global up
            right = prev_right;
        else if (dot(right, prev_right) < 0.0f)
            right = -right;
    }
    glm::vec3 up      = cross(-dir, right);
    right             = normalize(right);
    up                = normalize(up);
    m_inv_to_world[0] = glm::vec4(right, 0.0f);
    m_inv_to_world[1] = glm::vec4(up, 0.0f);
    m_inv_to_world[2] = glm::vec4(-dir, 0.0f);
    m_to_world        = inverse(m_inv_to_world);
}

void FlyCam::set_direction(glm::vec3 dir, glm::vec3 up)
{
    dir               = normalize(dir);
    glm::vec3 right   = normalize(cross(up, -dir));
    up                = normalize(cross(-dir, right));
    m_inv_to_world[0] = glm::vec4(right, 0.0f);
    m_inv_to_world[1] = glm::vec4(up, 0.0f);
    m_inv_to_world[2] = glm::vec4(-dir, 0.0f);
    m_to_world        = inverse(m_inv_to_world);
}
}
