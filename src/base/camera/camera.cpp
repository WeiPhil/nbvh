

#include "base/camera/camera.h"

#include "glm_pch.h"

namespace ntwr {

bool Camera::ui()
{
    bool camera_changed = false;

    ImGui::SeparatorText("Camera");

    if (ImGui::Button("Print camera transform")) {
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                // Print the element
                printf("%f", this->transform()[i][j]);  // Access elements in column-major order

                // Add a comma and space unless it's the last element
                if (i != 3 || j != 3) {
                    printf(", ");
                }
            }
        }
        printf("\n");
        printf("%s\n", glm::to_string(this->transform()).c_str());
    }
    ImGui::Spacing();
    camera_changed |= ImGui::SliderFloat("fov", &this->m_fovy, 1.0, 100.0f);
    ImGui::Spacing();
    camera_changed |= ImGui::SliderFloat("speed", &m_speed, 0.0001f, 1000.0f);
    camera_changed |= ImGui::SliderFloat("sensitivity", &m_sensitivity, 0.01f, 10.0f);

    return camera_changed;
}

void Camera::set_transform(glm::mat4 to_world)
{
    m_to_world     = to_world;
    m_inv_to_world = glm::inverse(m_to_world);
}

void Camera::set_fovy_from_radian(float fovy_radian)
{
    m_fovy = glm::degrees(fovy_radian);
}

void Camera::set_fovy_from_degree(float fovy_degree)
{
    m_fovy = fovy_degree;
}

const glm::mat4 &Camera::transform() const
{
    return m_to_world;
}

const glm::mat4 &Camera::inv_transform() const
{
    return m_inv_to_world;
}

glm::vec3 Camera::origin() const
{
    return glm::vec3{m_inv_to_world * glm::vec4{0, 0, 0, 1}};
}

glm::vec3 Camera::dir() const
{
    return glm::normalize(glm::vec3{m_inv_to_world * glm::vec4{0, 0, -1, 0}});
}

glm::vec3 Camera::up() const
{
    return glm::normalize(glm::vec3{m_inv_to_world * glm::vec4{0, 1, 0, 0}});
}

float Camera::fovy() const
{
    return m_fovy;
}
}
