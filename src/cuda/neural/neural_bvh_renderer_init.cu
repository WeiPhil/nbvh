

#include "app/render_app.h"

#include "cuda/cuda_backend_constants.cuh"
#include "cuda/neural/neural_bvh_module_constants.cuh"
#include "cuda/neural/neural_bvh_renderer.h"

#include "cuda/neural/utils/pt_composite_loss.cuh"

#include "cuda/neural/path/path_module.h"
#include "cuda/neural/prefiltering/prefiltering_module.h"

#include <tiny-cuda-nn/loss.h>

#include <format>

namespace ntwr {
namespace neural {

    NeuralBVHRenderer::NeuralBVHRenderer(std::shared_ptr<CudaBackend> cuda_backend) : m_backend(std::move(cuda_backend))
    {
        logger(LogLevel::Info, "Initializing Neural BVH Renderer");

        // For now we just support one neural bvh
        assert(m_backend->m_neural_bvh_indices.size() > 0);
        m_bvh_const.neural_bvh_idx = m_backend->m_neural_bvh_indices[0];

        if (!loss_registered) {
            // Register our custom loss
            tcnn::register_loss<tcnn::network_precision_t>("PathTracingComposite", [](const tcnn::json &loss) {
                return new PathTracingCompositeLoss<tcnn::network_precision_t>{
                    loss.value("output_visibility_dim", 1u),
                    loss.value("visibility_loss", "BinaryCrossEntropy"),
                    loss.value("output_depth_dim", 0u),
                    loss.value("depth_loss", "L2"),
                    loss.value("output_normal_dim", 0u),
                    loss.value("normal_loss", "L1"),
                    loss.value("output_albedo_dim", 0u),
                    loss.value("albedo_loss", "L2"),
                };
            });
            loss_registered = true;
        }

        if (!m_neural_module) {
            switch_neural_module();
            m_neural_module->reinitialise_neural_model(m_batch_size);
            m_neural_module->reinitialise_inference_data(m_backend->m_fb_size);
        }

        reinitialise_bvh_learning_data(m_batch_size);
        build_and_upload_neural_bvh();
        reinitialise_neural_bvh_learning_states();
        reinitialise_tree_cut_learning_data(m_bvh_split_scheduler.m_start_split_num);

        load_next_config();
    }

    NeuralBVHRenderer::~NeuralBVHRenderer() {}

    void NeuralBVHRenderer::reinitialise_tree_cut_learning_data(uint32_t num_bvh_splits)
    {
        m_current_num_bvh_split = num_bvh_splits;
        if (m_max_node_losses)
            CUDA_CHECK(cudaFree(m_max_node_losses));
        if (m_split_neural_node_indices)
            CUDA_CHECK(cudaFree(m_split_neural_node_indices));

        m_max_node_loss_host = 0.f;
        m_split_neural_node_indices_host.resize(m_current_num_bvh_split);

        CUDA_CHECK(cudaMalloc(&m_max_node_losses, m_current_num_bvh_split * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&m_split_neural_node_indices, m_current_num_bvh_split * sizeof(float)));
    }

    void NeuralBVHRenderer::reinitialise_bvh_learning_data(const uint32_t batch_size)
    {
        if (m_sampled_node_indices.d_pointer()) {
            m_sampled_node_indices.free();
        }

        if (m_mean_error) {
            CUDA_CHECK(cudaFree(m_mean_error));
        }

        assert(batch_size == m_batch_size);

        m_sampled_node_indices.alloc(m_batch_size);

        CUDA_CHECK(cudaMalloc(&m_mean_error, sizeof(float)));
    }

    void NeuralBVHRenderer::setup_constants()
    {
        m_backend->setup_constants();

        m_neural_module->setup_neural_module_constants();

        // BVH Rendering related constants
        CUDA_CHECK(cudaMemcpyToSymbol(bvh_const, &m_bvh_const, sizeof(bvh_const)));

        CUDA_CHECK(cudaMemcpyToSymbol(neural_scene_bbox, &m_neural_scene_bbox, sizeof(neural_scene_bbox)));
    }

    void NeuralBVHRenderer::switch_neural_module()
    {
        switch (m_bvh_module_type) {
        case BVHModuleType::Path:
            m_neural_module = std::make_unique<path::NeuralPathModule>(m_batch_size, m_backend);
            break;
        case BVHModuleType::Prefiltering:
            m_neural_module = std::make_unique<prefiltering::NeuralPrefilteringModule>(m_batch_size, m_backend);
            break;
        default:
            UNREACHABLE();
        };
    }

    void NeuralBVHRenderer::on_resize(const glm::uvec2 &new_size)
    {
        // if window minimized
        if ((new_size.x == 0) || (new_size.y == 0))
            return;

        m_neural_module->reinitialise_inference_data(new_size);
    }

    uint32_t NeuralBVHRenderer::accumulated_samples()
    {
        return m_accumulated_spp;
    }

    void NeuralBVHRenderer::scene_updated()
    {
        // reset_accumulation
        m_sample_offset   = 0;
        m_accumulated_spp = 0;

        m_reset_inference_data = true;
        m_reset_learning_data  = true;
        m_rebuild_neural_bvh   = true;
        m_run_optimisation     = false;
        m_run_inference        = false;
    }

    void NeuralBVHRenderer::load_next_config()
    {
        logger(LogLevel::Info, "Loading next configuration ...");

        bool same_base_config;
        bool only_more_training_steps;
        m_using_validation_mode = m_backend->get_render_app_configs().validation_mode();
        if (auto config_tuple =
                m_backend->get_render_app_configs().get_next_config(same_base_config, only_more_training_steps)) {
            nlohmann::json full_config;
            std::string base_config_name;
            std::tie(full_config, base_config_name) = *config_tuple;

            if (!full_config.contains("neural_bvh_renderer")) {
                logger(LogLevel::Error, "Configuration not loaded: missing neural_bvh_renderer field!");
                return;
            }

            m_validation_mode = ValidationMode::Ignore;

            if (m_using_validation_mode) {  // if we are in validation mode check in which mode
                if (!full_config["neural_bvh_renderer"].contains("validation_mode")) {
                    logger(LogLevel::Error, "Validation mode not found, skipping configuration!");
                    return;
                }
                std::string validation_str = full_config["neural_bvh_renderer"].value(
                    "validation_mode", std::string(validation_modes[m_validation_mode]));
                m_validation_mode = (ValidationMode)imgui_utils::find_index(
                    validation_str.c_str(), validation_modes, ARRAY_SIZE(validation_modes));

                if (m_validation_mode == ValidationMode::Ignore) {
                    logger(LogLevel::Warn, "Requested validation mode but config has none specified!");
                }
            }

            // Now load config with bvh and weight if required
            bool load_bvh_and_weights = !same_base_config && (m_validation_mode == ValidationMode::Inference ||
                                                              m_validation_mode == ValidationMode::Ignore);

            bool same_base_config_inference = same_base_config && m_validation_mode == ValidationMode::Inference;
            only_more_training_steps &=
                (m_validation_mode == ValidationMode::Training || m_validation_mode == ValidationMode::Full);
            // Don't reload the bvh/weights if only more training steps are performed
            load_bvh_and_weights &= !only_more_training_steps;

            bool skip_learning_data_reset = same_base_config_inference || only_more_training_steps;

            if (!only_more_training_steps) {
                load_config_from_json(full_config, base_config_name, load_bvh_and_weights, skip_learning_data_reset);
            } else {
                // The only configurations that should change!
                // (avoid loading config as it would override the bvh scheduler)
                m_max_training_steps      = full_config["neural_bvh_renderer"]["max_training_steps"].get<uint32_t>();
                auto new_base_config_name = full_config["neural_bvh_renderer"]["base_config_name"].get<std::string>();
                auto new_variant_config_name =
                    full_config["neural_bvh_renderer"]["variant_config_name"].get<std::string>();
                strcpy(m_base_config_name, new_base_config_name.c_str());
                strcpy(m_variant_config_name, new_variant_config_name.c_str());
            }
            // Correctly set m_run_optimisation and m_run_inference based on validation mode
            if (m_validation_mode == ValidationMode::Ignore) {
                if (m_using_validation_mode) {
                    logger(LogLevel::Error,
                           "Skipped a config during validation mode! (Check validation_mode in your configuration)");
                    load_next_config();
                }
            } else if (m_validation_mode == ValidationMode::Full) {
                // Start by running the optimisation then render after a process_validation_config_end
                m_run_optimisation = true;
                m_run_inference    = false;

                m_reset_learning_data  = true && !only_more_training_steps;
                m_rebuild_neural_bvh   = true && !only_more_training_steps;
                m_reset_inference_data = true;

                // Simulate reset
                m_sample_offset   = 0;
                m_accumulated_spp = 0;
            } else if (m_validation_mode == ValidationMode::Training) {
                // Only run the optimisation
                m_run_optimisation   = true && !only_more_training_steps;
                m_rebuild_neural_bvh = true && !only_more_training_steps;

                m_run_inference = false;

                m_reset_learning_data = true;
            } else if (m_validation_mode == ValidationMode::Inference) {
                // Only run the inference
                m_run_optimisation = false;
                m_run_inference    = true;

                m_reset_inference_data = true;
                // Simulate reset
                m_sample_offset   = 0;
                m_accumulated_spp = 0;
            } else {
                UNREACHABLE();
            }

            if (m_using_validation_mode) {
                // Always isable ema_accumulation in validation mode
                if (m_backend->m_cuda_scene_constants.ema_accumulate) {
                    logger(LogLevel::Warn, "Ema accumulation is not supported in validation mode!");
                }
                m_backend->m_cuda_scene_constants.ema_accumulate = false;
            }
        } else if (m_using_validation_mode) {
            logger(LogLevel::Info, "All configurations processed!");
            exit(0);
        }

        logger(LogLevel::Info, "Done!");
    }

    void NeuralBVHRenderer::process_validation_config_end()
    {
        m_run_optimisation = false;
        m_run_inference    = false;

        bool save_image_and_inference_performance = false;
        bool save_learning_performance            = false;
        bool get_next_config                      = false;
        bool save_neural_bvh_and_weights          = false;

        if (m_validation_mode == ValidationMode::Ignore) {
            logger(LogLevel::Error, "This function should only be called in validation mode!");
        } else if (m_validation_mode == ValidationMode::Full) {
            if ((m_max_training_steps == m_training_step) && (m_accumulated_spp != m_max_accumulated_spp)) {
                // Training is over we can perform the inference

                save_learning_performance   = true;
                save_neural_bvh_and_weights = true;
                get_next_config             = false;

                m_run_inference = true;
            } else {
                get_next_config                      = true;
                save_image_and_inference_performance = true;
            }
        } else if (m_validation_mode == ValidationMode::Training) {
            get_next_config             = true;
            save_learning_performance   = true;
            save_neural_bvh_and_weights = true;

        } else if (m_validation_mode == ValidationMode::Inference) {
            if (m_accumulated_spp != m_max_accumulated_spp) {
                logger(LogLevel::Error, "The validation inference was not done yet!");
            }
            get_next_config                      = true;
            save_image_and_inference_performance = true;
        } else {
            UNREACHABLE();
        }

        if (save_image_and_inference_performance) {
            save_image();
            save_performance_metrics(false, true);
        }
        if (save_learning_performance) {
            save_performance_metrics(true, false);
        }

        if (save_neural_bvh_and_weights) {
            std::filesystem::create_directory(m_backend->m_scene_filename.parent_path() / "nbvhs");

            const std::filesystem::path neural_bvh_filename =
                m_backend->m_scene_filename.parent_path() / "nbvhs" / (std::string(m_base_config_name) + ".nbvh");

            save_neural_bvh(neural_bvh_filename.string());

            std::filesystem::create_directory(m_backend->m_scene_filename.parent_path() / "network_models");

            const std::filesystem::path network_model_filename = m_backend->m_scene_filename.parent_path() /
                                                                 "network_models" /
                                                                 (std::string(m_base_config_name) + "_network.json");

            save_network_model(network_model_filename.string());
        }

        if (get_next_config) {
            load_next_config();
        }
    }

    void NeuralBVHRenderer::save_performance_metrics(bool save_learning_perfs, bool save_inference_perfs)
    {
        const std::string base_config_name    = std::string(m_base_config_name);
        const std::string variant_config_name = std::string(m_variant_config_name);

        // Creates directory if non existent
        std::filesystem::create_directory(m_backend->m_scene_filename.parent_path() / "results");

        if (save_learning_perfs) {
            const std::filesystem::path learning_perf_csv_file =
                m_backend->m_scene_filename.parent_path() / "results" /
                (base_config_name + "_" + variant_config_name + "_learning" + "_spp_" +
                 std::to_string(m_accumulated_spp) + ".csv");
            logger(LogLevel::Info,
                   "Saving learning performance metrics to %s ...",
                   learning_perf_csv_file.string().c_str());
            m_learning_timer.write_to_file(learning_perf_csv_file.string(), &m_loss_graph);
        }
        if (save_inference_perfs) {
            const std::filesystem::path inference_perf_csv_file =
                m_backend->m_scene_filename.parent_path() / "results" /
                (base_config_name + "_" + variant_config_name + "_inference" + "_spp_" +
                 std::to_string(m_accumulated_spp) + ".csv");
            logger(LogLevel::Info,
                   "Saving inference performance metrics to %s ...",
                   inference_perf_csv_file.string().c_str());
            m_inference_timer.write_to_file(inference_perf_csv_file.string());
        }
        logger(LogLevel::Info, "Done!");
    }

    void NeuralBVHRenderer::save_image()
    {
        const std::string base_config_name    = std::string(m_base_config_name);
        const std::string variant_config_name = std::string(m_variant_config_name);

        // Creates directory if non existent
        std::filesystem::create_directory(m_backend->m_scene_filename.parent_path() / "results");

        const std::filesystem::path result_image_filenmae =
            m_backend->m_scene_filename.parent_path() / "results" /
            (base_config_name + "_" + variant_config_name + "_spp_" + std::to_string(m_accumulated_spp) + ".exr");

        logger(LogLevel::Info, "Saving image to %s ...", result_image_filenmae.string().c_str());

        // Would just save incorrect images, avoid doing so
        if (m_backend->m_cuda_scene_constants.ema_accumulate) {
            logger(LogLevel::Error, "Saving images when ema weighting is not supported! Aborting.");
            return;
        }

        m_backend->get_render_app().save_framebuffer(result_image_filenmae.string());
        logger(LogLevel::Info, "Done!");
    }

    void NeuralBVHRenderer::save_network_model(const std::string &filename)
    {
        tcnn::json serialized_model = m_neural_module->m_model->trainer->serialize(false);
        std::ofstream file(filename);
        file << serialized_model;
        file.close();
        logger(LogLevel::Info, "Serialized network model to : %s", filename.c_str());
    }

    void NeuralBVHRenderer::load_network_model(const std::string &filename)
    {
        logger(LogLevel::Info, "Deserializing network model from : %s", filename.c_str());
        try {
            std::ifstream file(filename);
            m_neural_module->m_model->trainer->deserialize(tcnn::json::parse(file));
            logger(LogLevel::Info, "Done!");
        } catch (const std::runtime_error &err) {
            logger(LogLevel::Error, "%s", err.what());
        }
    }

    void NeuralBVHRenderer::save_config(bool save_bvh_and_network)
    {
        const std::string base_config_name    = std::string(m_base_config_name);
        const std::string variant_config_name = std::string(m_variant_config_name);
        logger(LogLevel::Info,
               "Saving base configuration %s as variant %s...",
               base_config_name.c_str(),
               variant_config_name.c_str());

        // Creates directory if non existent
        std::filesystem::create_directory(m_backend->m_scene_filename.parent_path() / "configs");

        const std::filesystem::path config_filename = m_backend->m_scene_filename.parent_path() / "configs" /
                                                      (base_config_name + "_" + variant_config_name + ".json");

        // Cuda Backend related
        const Camera &camera_ref = m_backend->get_camera();

        tcnn::json camera_transform = tcnn::json::array();
        for (int row = 0; row < 4; ++row) {
            for (int col = 0; col < 4; ++col) {
                camera_transform.push_back(camera_ref.transform()[row][col]);
            }
        }

        // Using ordered json for nicer config files
        nlohmann::ordered_json segment_encoding_config = {
            {"type", encoding_types[m_neural_module->m_segment_encoding]},
            {"hashgrid_options", m_neural_module->m_segment_points_hashgrid_options.to_json()},
            {"reduction", reduction_types[m_neural_module->m_segment_reduction]},
        };

        nlohmann::ordered_json direction_encoding_config = {
            {"type", encoding_types[m_neural_module->m_segment_direction_encoding]},
            {"hashgrid_options", m_neural_module->m_segment_direction_hashgrid_options.to_json()},
        };

        nlohmann::ordered_json neural_module_config;
        switch (m_bvh_module_type) {
        case BVHModuleType::Path: {
            path::NeuralPathModule *neural_module = dynamic_cast<path::NeuralPathModule *>(m_neural_module.get());
            assert(neural_module);

            neural_module_config = {
                {"max_depth", neural_module->m_max_depth},
                {"pt_output_constants", neural_module->m_pt_target_const.to_json()},
                {"path_module_constants", neural_module->m_path_module_const.to_json()},
            };
            break;
        }
        case BVHModuleType::Prefiltering: {
            prefiltering::NeuralPrefilteringModule *neural_module =
                dynamic_cast<prefiltering::NeuralPrefilteringModule *>(m_neural_module.get());
            assert(neural_module);

            neural_module_config = {
                {"max_depth", neural_module->m_max_depth},
                {"pt_output_constants", neural_module->m_pt_target_const.to_json()},
                {"prefiltering_module_const", neural_module->m_prefiltering_module_const.to_json()},
            };
            break;
        }
        default:
            UNREACHABLE();
        }

        neural_module_config.push_back({"bvh_module_constants", m_neural_module->m_bvh_module_const.to_json()});
        neural_module_config.push_back({"network", m_neural_module->m_network_options.to_json()});
        neural_module_config.push_back({"optimizer", m_neural_module->m_optimizer_options.to_json()});
        neural_module_config.push_back({"segment_encoding", segment_encoding_config});
        neural_module_config.push_back({"direction_encoding", direction_encoding_config});

        nlohmann::ordered_json neural_bvh_renderer_config = {
            {"bvh_module_type", bvh_module_types[m_bvh_module_type]},
            {"validation_mode", validation_modes[m_saved_validation_mode]},
            {"neural_module", neural_module_config},
            {"batch_size", m_batch_size},
            {"max_training_steps", m_max_training_steps},
            {"max_accumulated_spp", m_max_accumulated_spp},  // from renderer base class
            {"training_step", m_training_step},
            {"base_config_name", std::string(m_base_config_name)},
            {"variant_config_name", std::string(m_variant_config_name)},
            {"run_optimisation", m_run_optimisation},
            {"run_inference", m_run_inference},
            {"continuous_reset", m_continuous_reset},
            {"neural_bvh_min_depth", m_neural_bvh_min_depth},
            {"bvh_constants", m_bvh_const.to_json()},
            {"show_bvh", m_show_bvh},
            {"show_neural_node_losses", m_show_neural_node_losses},
            {"show_tree_cut_depth", m_show_tree_cut_depth},
            {"tree_cut_learning", m_tree_cut_learning},
            {"bvh_split_scheduler", m_bvh_split_scheduler.to_json()},
        };

        nlohmann::ordered_json camera_config = {
            {"type", "perspective"}, {"yfov", glm::radians(camera_ref.fovy())}, {"matrix", camera_transform}};

        nlohmann::ordered_json cuda_scene_config = {
            {"fb_size", {m_backend->framebuffer_size().x, m_backend->framebuffer_size().y}},
            {"camera", camera_config},
            {"scene_constants", m_backend->m_cuda_scene_constants.to_json()},
            {"envmap_filename", m_backend->envmap_filename()},
        };

        nlohmann::ordered_json full_config = {{"cuda_scene", cuda_scene_config},
                                              {"neural_bvh_renderer", neural_bvh_renderer_config}};

        std::ofstream file(config_filename.string());
        file << std::setw(4) << full_config << std::endl;
        file.close();

        // Save current imgui ini state
        const std::filesystem::path imgui_ini_filename =
            m_backend->m_scene_filename.parent_path() / "configs" / (base_config_name + "_imgui.ini");
        ImGui::SaveIniSettingsToDisk(imgui_ini_filename.string().c_str());

        ///// Neural BVH

        if (save_bvh_and_network) {
            std::filesystem::create_directory(m_backend->m_scene_filename.parent_path() / "nbvhs");

            const std::filesystem::path neural_bvh_filename =
                m_backend->m_scene_filename.parent_path() / "nbvhs" / (base_config_name + ".nbvh");

            save_neural_bvh(neural_bvh_filename.string());

            std::filesystem::create_directory(m_backend->m_scene_filename.parent_path() / "network_models");

            const std::filesystem::path network_model_filename =
                m_backend->m_scene_filename.parent_path() / "network_models" / (base_config_name + "_network.json");

            save_network_model(network_model_filename.string());
        }

        logger(LogLevel::Info, "Done!");
    }

    void NeuralBVHRenderer::load_config(bool load_bvh_and_network)
    {
        const std::string base_config_name    = std::string(m_base_config_name);
        const std::string variant_config_name = std::string(m_variant_config_name);

        logger(LogLevel::Info,
               "Loading base configuration %s as variant %s...",
               base_config_name.c_str(),
               variant_config_name.c_str());

        const std::filesystem::path config_filename = m_backend->m_scene_filename.parent_path() / "configs" /
                                                      (base_config_name + "_" + variant_config_name + ".json");
        if (!std::filesystem::exists(config_filename)) {
            logger(LogLevel::Error, "Could not find configuration file at %s", config_filename.string().c_str());
            return;
        }

        std::ifstream input_file(config_filename.string());
        tcnn::json full_config = tcnn::json::parse(input_file);

        load_config_from_json(full_config, base_config_name, load_bvh_and_network);
    }

    void NeuralBVHRenderer::load_config_from_json(tcnn::json full_config,
                                                  const std::string base_config_name,
                                                  bool load_bvh_and_network,
                                                  bool skip_learning_data_reset)
    {
        // Loading all configurations into the respective structures

        // Load current imgui ini state
        const std::filesystem::path imgui_ini_filename =
            m_backend->m_scene_filename.parent_path() / "configs" / (base_config_name + "_imgui.ini");

        if (std::filesystem::exists(imgui_ini_filename))
            ImGui::LoadIniSettingsFromDisk(imgui_ini_filename.string().c_str());

        m_run_optimisation  = false;
        m_tree_cut_learning = false;

        nlohmann::json cuda_scene_config = full_config.value("cuda_scene", json::object());

        if (cuda_scene_config.contains("fb_size")) {
            glfwSetWindowSize(m_backend->get_display().glfw_window(),
                              cuda_scene_config["fb_size"].at(0).get<int>(),
                              cuda_scene_config["fb_size"].at(1).get<int>());
        } else {
            logger(LogLevel::Warn, "No framebuffer size specified in config, unchanged");
        }
        nlohmann::json camera_config = cuda_scene_config.value("camera", json::object());

        auto &scene_camera = m_backend->get_camera();

        glm::mat4 camera_mat4 = scene_camera.transform();
        if (camera_config.contains("matrix")) {
            float camera_trafo[16];
            for (size_t i = 0; i < 16; i++) {
                camera_trafo[i] = camera_config["matrix"].at(i).get<float>();
            }
            camera_mat4 = glm::make_mat4(camera_trafo);
        }
        scene_camera.set_transform(camera_mat4);
        scene_camera.set_fovy_from_radian(camera_config.value("yfov", glm::radians(scene_camera.fovy())));
        m_backend->update_camera(scene_camera);

        m_backend->m_cuda_scene_constants.from_json(cuda_scene_config.value("scene_constants", json::object()));

        strcpy(m_backend->m_envmap_filename,
               cuda_scene_config.value("envmap_filename", std::string(m_backend->m_envmap_filename)).c_str());
        m_backend->load_envmap_texture(std::string(m_backend->m_envmap_filename));

        nlohmann::json neural_bvh_renderer_config = full_config.value("neural_bvh_renderer", json::object());

        BVHModuleType bvh_module_type = (BVHModuleType)imgui_utils::find_index(
            neural_bvh_renderer_config.value("bvh_module_type", bvh_module_types[m_bvh_module_type]).c_str(),
            bvh_module_types,
            ARRAY_SIZE(bvh_module_types));

        if (m_bvh_module_type != bvh_module_type) {
            m_bvh_module_type = bvh_module_type;
            switch_neural_module();
        }
        if (skip_learning_data_reset && (m_bvh_module_type != bvh_module_type)) {
            logger(LogLevel::Error, "If skipping the learning data reset, the neural module should also be the same!");
        }

        m_saved_validation_mode = (ValidationMode)imgui_utils::find_index(
            neural_bvh_renderer_config.value("validation_mode", validation_modes[m_saved_validation_mode]).c_str(),
            validation_modes,
            ARRAY_SIZE(validation_modes));

        nlohmann::json neural_bvh_module_config = neural_bvh_renderer_config.value("neural_module", json::object());

        if (neural_bvh_renderer_config.value("base_config_name", std::string(m_base_config_name)) != base_config_name) {
            logger(LogLevel::Error, "Inconsistent base_config_name found in loaded config file!");
        }

        m_neural_module->m_bvh_module_const.from_json(
            neural_bvh_module_config.value("bvh_module_constants", json::object()));
        m_neural_module->m_network_options.from_json(neural_bvh_module_config.value("network", json::object()));
        m_neural_module->m_optimizer_options.from_json(neural_bvh_module_config.value("optimizer", json::object()));

        nlohmann::json segment_encoding_config   = neural_bvh_module_config.value("segment_encoding", json::object());
        nlohmann::json direction_encoding_config = neural_bvh_module_config.value("direction_encoding", json::object());

        m_neural_module->m_segment_encoding = (EncodingType)imgui_utils::find_index(
            segment_encoding_config.value("type", encoding_types[m_neural_module->m_segment_encoding]).c_str(),
            encoding_types,
            ARRAY_SIZE(encoding_types));
        m_neural_module->m_segment_points_hashgrid_options.from_json(
            segment_encoding_config.value("hashgrid_options", json::object()));
        m_neural_module->m_segment_reduction = (CompositeReductionType)imgui_utils::find_index(
            segment_encoding_config.value("reduction", reduction_types[m_neural_module->m_segment_reduction]).c_str(),
            reduction_types,
            ARRAY_SIZE(reduction_types));

        m_neural_module->m_segment_direction_encoding = (EncodingType)imgui_utils::find_index(
            direction_encoding_config.value("type", encoding_types[m_neural_module->m_segment_direction_encoding])
                .c_str(),
            encoding_types,
            ARRAY_SIZE(encoding_types));
        m_neural_module->m_segment_direction_hashgrid_options.from_json(
            direction_encoding_config.value("hashgrid_options", json::object()));

        switch (m_bvh_module_type) {
        case BVHModuleType::Path: {
            path::NeuralPathModule *neural_module = dynamic_cast<path::NeuralPathModule *>(m_neural_module.get());
            assert(neural_module);

            neural_module->m_max_depth = neural_bvh_module_config.value("max_depth", neural_module->m_max_depth);
            neural_module->m_pt_target_const.from_json(
                neural_bvh_module_config.value("pt_output_constants", json::object()));
            neural_module->m_path_module_const.from_json(
                neural_bvh_module_config.value("path_module_constants", json::object()));
            break;
        }
        case BVHModuleType::Prefiltering: {
            prefiltering::NeuralPrefilteringModule *neural_module =
                dynamic_cast<prefiltering::NeuralPrefilteringModule *>(m_neural_module.get());
            assert(neural_module);
            neural_module->m_max_depth = neural_bvh_module_config.value("max_depth", neural_module->m_max_depth);
            neural_module->m_pt_target_const.from_json(
                neural_bvh_module_config.value("pt_output_constants", json::object()));
            neural_module->m_prefiltering_module_const.from_json(
                neural_bvh_module_config.value("prefiltering_module_const", json::object()));
            break;
        }
        default:
            UNREACHABLE();
        }

        m_batch_size          = neural_bvh_renderer_config.value("batch_size", 512 * 512);
        m_max_training_steps  = neural_bvh_renderer_config.value("max_training_steps", m_max_training_steps);
        m_max_accumulated_spp = neural_bvh_renderer_config.value("max_accumulated_spp", m_max_accumulated_spp);
        m_training_step       = neural_bvh_renderer_config.value("training_step", 0);
        strcpy(m_base_config_name,
               neural_bvh_renderer_config.value("base_config_name", std::string(m_base_config_name)).c_str());
        strcpy(m_variant_config_name,
               neural_bvh_renderer_config.value("variant_config_name", std::string(m_variant_config_name)).c_str());
        m_run_optimisation     = neural_bvh_renderer_config.value("run_optimisation", m_run_optimisation);
        m_run_inference        = neural_bvh_renderer_config.value("run_inference", m_run_inference);
        m_continuous_reset     = neural_bvh_renderer_config.value("continuous_reset", m_continuous_reset);
        m_neural_bvh_min_depth = neural_bvh_renderer_config.value("neural_bvh_min_depth", m_neural_bvh_min_depth);
        m_bvh_const.from_json(neural_bvh_renderer_config.value("bvh_constants", json::object()));
        m_show_bvh = neural_bvh_renderer_config.value("show_bvh", m_show_bvh);
        m_show_neural_node_losses =
            neural_bvh_renderer_config.value("show_neural_node_losses", m_show_neural_node_losses);
        m_show_tree_cut_depth = neural_bvh_renderer_config.value("show_tree_cut_depth", m_show_tree_cut_depth);
        m_tree_cut_learning   = neural_bvh_renderer_config.value("tree_cut_learning", m_tree_cut_learning);
        m_bvh_split_scheduler.from_json(neural_bvh_renderer_config.value("bvh_split_scheduler", json::object()));

        // Make sure to reset training and learning data before loading the bvh and weights!
        // Inference data can be reset later but we need to init learning data directly to not override the network
        // being loaded
        m_reset_inference_data = true;
        // We store the training steps and restore it in order to avoid the optimisation to restart if the
        // flag for running the optimisation was still on
        auto training_step_tmp = m_training_step;
        if (!skip_learning_data_reset)
            reset_learning_data();
        m_training_step = training_step_tmp;

        if (load_bvh_and_network) {
            const std::filesystem::path neural_bvh_filename =
                m_backend->m_scene_filename.parent_path() / "nbvhs" / (base_config_name + ".nbvh");
            if (!std::filesystem::exists(neural_bvh_filename)) {
                logger(LogLevel::Error, "Could not find nbvh file at %s", neural_bvh_filename.string().c_str());
                return;
            }

            load_neural_bvh(neural_bvh_filename.string());

            const std::filesystem::path network_model_filename =
                m_backend->m_scene_filename.parent_path() / "network_models" / (base_config_name + "_network.json");

            if (!std::filesystem::exists(network_model_filename)) {
                logger(LogLevel::Error,
                       "Could not find network model file at %s",
                       network_model_filename.string().c_str());
                return;
            }

            load_network_model(network_model_filename.string());
        }

        logger(LogLevel::Info, "Done!");
    }

}
}