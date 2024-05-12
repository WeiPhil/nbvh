#pragma once

#include <vector>
#include "base/camera/camera.h"
#include "glm_pch.h"

namespace ntwr {

struct FlyCam : public Camera {
    glm::vec3 global_up;

    FlyCam(glm::vec3 up     = glm::vec3(0.f, 1.f, 0.f),
           glm::vec3 origin = glm::vec3(0.f, 0.f, -4.f),
           glm::vec3 dir    = glm::vec3(0.f, 0.f, 1.f));

    virtual bool ui() override;
    virtual bool movement_io(ImGuiIO &io, glm::uvec2 window_size) override;

    /* Rotate the camera from the previous mouse position to the current
     * one. Mouse positions should be in normalized device coordinates
     */
    void rotate(glm::vec2 prev_mouse, glm::vec2 cur_mouse);

    /* Pan the camera given the translation vector. Mouse
     * delta amount should be in normalized device coordinates
     */
    void pan(glm::vec2 mouse_delta);
    /* Zoom the camera given the zoom amount to (i.e., the scroll amount).
     * Positive values zoom in, negative will zoom out.
     */
    void zoom(float zoom_amount);

    /* Move the camera along a local vector.
     */
    void move_local(glm::vec3 local_dir, float speed);

    // Get the center of rotation of the camera in world space
    glm::vec3 center() const;

    void set_origin(glm::vec3 origin);
    void set_direction(glm::vec3 dir);
    void set_direction(glm::vec3 dir, glm::vec3 up);

private:
    void update_camera();
};
}