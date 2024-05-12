

#include <filesystem>
#include <iostream>
#include <stdexcept>

#include "app/render_app.h"
#include "app/render_display.h"
#include "base/scene.h"
#include "cuda/cuda_backend.h"
#include "utils/logger.h"

using namespace ntwr;

int main(int argc, char *argv[])
{
    if (argc < 1) {
        logger(LogLevel::Info,
               "Usage: %s [obj_scene_filename] [validation_mode (true/false)] [config1] [config2] ",
               argv[0]);
        return 1;
    }

    glm::uvec2 window_size(1920, 1080);

    // Create display
    std::unique_ptr<RenderDisplay> display = std::make_unique<RenderDisplay>(window_size);

    std::filesystem::path scene_filename = "";
    bool validation_mode                 = false;

    if (argc >= 2) {
        scene_filename = argv[1];
        // Convert to posix-style path
        scene_filename = std::filesystem::path(scene_filename.generic_string());
    }

    if (argc >= 3) {
        validation_mode = strcmp(argv[2], "true") == 0;
    }

    RenderAppConfigs render_app_configs = RenderAppConfigs(validation_mode, scene_filename);

    if (argc >= 4) {
        for (size_t arg_idx = 3; arg_idx < argc; arg_idx++) {
            render_app_configs.add_config_file(std::string(argv[arg_idx]));
            if (!validation_mode && argc > 4) {
                logger(LogLevel::Warn,
                       "Only one config at can be loaded when not in validation mode! Skipping remaining.");
                break;
            }
        }
    }

    if (validation_mode) {
        logger(LogLevel::Warn, "Running  %d configs in validation mode!", render_app_configs.n_configs());
    }

    // Create main app and attach display to it
    std::unique_ptr<RenderApp> render_app =
        std::make_unique<RenderApp>(render_app_configs, scene_filename, display.get());

    try {
        render_app->run();
    } catch (std::runtime_error &e) {
        logger(LogLevel::Error, "%s", e.what());
        exit(1);
    }
    return 0;
}