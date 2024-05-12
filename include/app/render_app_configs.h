#pragma once

#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <queue>
#include <tuple>
#include <json/json.hpp>

#include "utils/logger.h"

namespace ntwr {

class RenderAppConfigs {
public:
    RenderAppConfigs(bool validation_mode, std::filesystem::path scene_filename)
        : m_validation_mode(validation_mode), m_scene_filename(scene_filename)
    {
    }
    ~RenderAppConfigs() {}

    void set_scene_filename(std::string scene_filename)
    {
        m_scene_filename = scene_filename;
    }

    bool validation_mode()
    {
        return m_validation_mode;
    }

    size_t n_configs()
    {
        return configs_and_base_names.size();
    }

    void add_config_file(std::string config_filename)
    {
        std::filesystem::path config_path(config_filename);

        if (!std::filesystem::exists(config_path)) {
            // if the file does not exist check if its located in the scene configs
            std::string config_name = config_path.filename().string();

            auto config_relative_to_scene = m_scene_filename.parent_path() / "configs" / config_name;
            if (std::filesystem::exists(config_relative_to_scene)) {
                config_path = config_relative_to_scene;
            } else {
                logger(LogLevel::Error,
                       "Configuration with fileaname %s was not found!",
                       config_relative_to_scene.string().c_str());
            }
        }
        logger(LogLevel::Info, "Loaded scene configuration at %s", config_path.string().c_str());

        std::ifstream input_file(config_path.string());
        nlohmann::json full_config = nlohmann::json::parse(input_file);

        // TODO make generic by storing config name in render app
        nlohmann::json neural_bvh_renderer_config = full_config.value("neural_bvh_renderer", nlohmann::json::object());
        nlohmann::json neural_bvh_module_config =
            neural_bvh_renderer_config.value("neural_module", nlohmann::json::object());

        if (!neural_bvh_renderer_config.contains("base_config_name")) {
            logger(LogLevel::Error, "No base_config_name found in the config file!");
            return;
        }
        std::string base_config_name = neural_bvh_renderer_config["base_config_name"];

        configs_and_base_names.push(std::make_pair(full_config, base_config_name));
    }

    std::optional<std::tuple<nlohmann::json, std::string>> get_next_config(bool &same_base_config_as_previous,
                                                                           bool &only_more_training_steps)
    {
        same_base_config_as_previous = false;
        only_more_training_steps     = false;
        if (configs_and_base_names.empty()) {
            return {};
        }
        std::tuple<nlohmann::json, std::string> next_pair = configs_and_base_names.front();
        configs_and_base_names.pop();
        nlohmann::json next_config        = std::get<0>(next_pair);
        std::string next_base_config_name = std::get<1>(next_pair);

        if (next_config.contains("neural_bvh_renderer") && previous_config.contains("neural_bvh_renderer")) {
            nlohmann::json next_config_tmp     = next_config;
            nlohmann::json previous_config_tmp = previous_config;

            auto &next_renderer_config     = next_config_tmp["neural_bvh_renderer"];
            auto &previous_renderer_config = previous_config_tmp["neural_bvh_renderer"];

            next_renderer_config.erase("base_config_name");
            next_renderer_config.erase("variant_config_name");
            previous_renderer_config.erase("base_config_name");
            previous_renderer_config.erase("variant_config_name");

            // Erase the max training steps and check if the two configs are now the same
            if ((next_renderer_config.erase("max_training_steps") == 1) &&
                (previous_renderer_config.erase("max_training_steps") == 1)) {
                if (next_config_tmp == previous_config_tmp) {
                    only_more_training_steps = true;
                    logger(LogLevel::Info, "Only more training steps in next config!");
                }
            }
        }

        same_base_config_as_previous = (next_base_config_name == previous_base_config_name);
        previous_base_config_name    = next_base_config_name;
        previous_config              = next_config;

        return std::make_optional(next_pair);
    }

private:
    bool m_validation_mode;
    nlohmann::json previous_config;
    std::string previous_base_config_name;
    std::queue<std::tuple<nlohmann::json, std::string>> configs_and_base_names;
    std::filesystem::path m_scene_filename;
};

}