#pragma once

#include <filesystem>

#include "app/render_app_configs.h"
#include "app/render_backend.h"
#include "app/render_display.h"
#include "app/renderer.h"
#include "base/camera/camera.h"
#include "utils/logger.h"

#include "base/bitmap.h"

namespace ntwr {

static constexpr size_t OUTPUT_FILENAME_SIZE = 512;

class RenderApp {
public:
    RenderApp(RenderAppConfigs render_app_configs, std::filesystem::path scene_filename, RenderDisplay *display);
    ~RenderApp();

    void register_renderer(Renderer *renderer);
    void register_backend(RenderBackend *backend);
    void assign_scene(std::filesystem::path scene_filename);

    void save_framebuffer(std::string filename) const;

    RenderAppConfigs &render_app_configs()
    {
        return m_render_app_configs;
    }

    const Scene &get_scene()
    {
        return *scene;
    }

    // Returns true if we need to reset the accumulation
    bool app_ui(bool &changed_camera, bool &changed_renderer);

    /*! opens the actual window, and runs the window's events to
      completion. This function will only return once the window
      gets closed */
    void run();

    RenderDisplay *display = nullptr;

    RenderAppConfigs m_render_app_configs;

    std::vector<Renderer *> renderers          = {};
    std::vector<std::string> renderer_variants = {};
    std::vector<RenderBackend *> backends      = {};

    int active_renderer_variant          = -1;
    int previous_active_renderer_variant = -1;

    bool reset_accumulation = false;
    bool hide_ui            = false;
    // Enables further option in the UI
    bool m_dev_mode = false;

    char output_image_filename[OUTPUT_FILENAME_SIZE] = "ntwr";

    ImGui::FileBrowser m_exr_file_dialog;
    ImGui::FileBrowser m_scene_file_dialog;

    std::unique_ptr<Scene> scene = nullptr;
};

}