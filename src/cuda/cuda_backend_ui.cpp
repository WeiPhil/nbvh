#include "cuda/cuda_backend.h"

namespace ntwr {

bool CudaBackend::ui(bool dev_mode)
{
    bool reset = false;

    if (dev_mode) {
        reset |= ImGui::Checkbox("Enable Light Sampling", &m_cuda_scene_constants.sample_light);
        reset |= ImGui::Checkbox("Enable Envmap Eval", &m_cuda_scene_constants.envmap_eval);
    }

    if (m_cuda_scene_constants.envmap_eval) {
        ImGui::SeparatorText("Envmap");

        reset |= ImGui::Checkbox("Use skylike envmap", &m_cuda_scene_constants.skylike_background);

        if (!m_cuda_scene_constants.skylike_background) {
            ImGui::Separator();
            ImGui::InputText("Envmap filename##envmap", m_envmap_filename, OUTPUT_FILENAME_SIZE);
            if (ImGui::Button("Reload Envmap")) {
                load_envmap_texture(std::string(m_envmap_filename));
            }
            ImGui::Indent();
            reset |= ImGui::DragFloat("Strength",
                                      &m_cuda_scene_constants.envmap_strength,
                                      m_cuda_scene_constants.envmap_strength / 128.f,
                                      10.f,
                                      100.f,
                                      "%.2f");
            reset |= ImGui::DragFloat("Rotation",
                                      &m_cuda_scene_constants.envmap_rotation,
                                      m_cuda_scene_constants.envmap_rotation / 128.f,
                                      0.f,
                                      360.f,
                                      "%.2f");
            ImGui::Unindent();
        }
    }

    if (dev_mode) {
        reset |= ImGui::Checkbox("Enable Emission Eval", &m_cuda_scene_constants.emission_eval);

        reset |= ImGui::Checkbox("Force non-neural constant material", &m_cuda_scene_constants.force_constant_material);
        if (m_cuda_scene_constants.force_constant_material) {
            ImGui::Indent();
            reset |= ImGui::SliderFloat("Specular", &m_cuda_scene_constants.constant_specular, 0.f, 1.f);
            reset |= ImGui::SliderFloat("Metallic", &m_cuda_scene_constants.constant_metallic, 0.f, 1.f);
            reset |= ImGui::SliderFloat("Roughness", &m_cuda_scene_constants.constant_roughness, 0.f, 1.f);
            ImGui::Unindent();
        }
    }
    return reset;
}

}
