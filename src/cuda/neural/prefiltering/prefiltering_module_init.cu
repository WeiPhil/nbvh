#include "cuda/neural/accel/neural_bvh.cuh"
#include "cuda/neural/neural_bvh_module_constants.cuh"
#include "cuda/neural/prefiltering/prefiltering_constants.cuh"
#include "cuda/neural/prefiltering/prefiltering_module.h"
#include "utils/logger.h"

namespace ntwr {

namespace neural::prefiltering {

    NeuralPrefilteringModule::NeuralPrefilteringModule(uint32_t batch_size, std::shared_ptr<CudaBackend> cuda_backend)
        : m_cuda_backend(cuda_backend)
    {
        m_model                 = std::make_unique<NeuralNetworkModel>();
        m_inference_input_data  = std::make_unique<NeuralInferenceInputData>();
        m_inference_output_data = std::make_unique<NeuralInferenceOutputData<NeuralTraversalResult>>();

        if (!m_appearance_model) {
            m_appearance_model = std::make_unique<NeuralNetworkModel>();
        }
        initialise_appearance_network_with_pretrained_weights(m_cuda_backend->m_scene_filename);

        initialize_learning_data(batch_size);
    }

    void NeuralPrefilteringModule::initialise_appearance_network_with_pretrained_weights(
        const std::filesystem::path &scene_filename)
    {
        auto appearance_weights_filename = scene_filename.parent_path() / (scene_filename.stem().string() +
                                                                           "_world_final_512_throughput_weights.json");
        auto appearance_config_filename =
            scene_filename.parent_path() / (scene_filename.stem().string() + "_world_final_512_throughput_config.json");

        logger(
            LogLevel::Info, "Loading appearance network config from : %s", appearance_config_filename.string().c_str());
        try {
            std::ifstream file(appearance_config_filename);
            if (!file.good()) {
                logger(LogLevel::Error,
                       "File %s not found, appearance network config not loaded",
                       appearance_config_filename.string().c_str());
                m_appearance_model->config = nlohmann::json::object();
            } else {
                m_appearance_model->config = tcnn::json::parse(file);
                logger(LogLevel::Info, "Done!");
            }
        } catch (const std::runtime_error &err) {
            logger(LogLevel::Error, "%s", err.what());
        }

        nlohmann::json encoding_opts  = m_appearance_model->config.value("encoding", nlohmann::json::object());
        nlohmann::json loss_opts      = m_appearance_model->config.value("loss", nlohmann::json::object());
        nlohmann::json optimizer_opts = m_appearance_model->config.value("optimizer", nlohmann::json::object());
        nlohmann::json network_opts   = m_appearance_model->config.value("network", nlohmann::json::object());

        m_appearance_model->loss = std::shared_ptr<tcnn::Loss<tcnn::network_precision_t>>{
            tcnn::create_loss<tcnn::network_precision_t>(loss_opts)};
        m_appearance_model->optimizer = std::shared_ptr<tcnn::Optimizer<tcnn::network_precision_t>>{
            tcnn::create_optimizer<tcnn::network_precision_t>(optimizer_opts)};
        m_appearance_model->network = std::make_shared<tcnn::NetworkWithInputEncoding<tcnn::network_precision_t>>(
            m_prefiltering_module_const.prefiltering_input_dims,
            m_prefiltering_module_const.prefiltering_output_dims,
            encoding_opts,
            network_opts);

        m_appearance_model->trainer =
            std::make_shared<tcnn::Trainer<float, tcnn::network_precision_t, tcnn::network_precision_t>>(
                m_appearance_model->network, m_appearance_model->optimizer, m_appearance_model->loss);

        logger(LogLevel::Info,
               "Loading appearance network weights from : %s",
               appearance_weights_filename.string().c_str());
        try {
            std::ifstream file(appearance_weights_filename);
            if (!file.good()) {
                logger(LogLevel::Error,
                       "File %s not found, weights not loaded",
                       appearance_weights_filename.string().c_str());
            } else {
                m_appearance_model->trainer->deserialize(tcnn::json::parse(file));
                logger(LogLevel::Info, "Done!");
            }
        } catch (const std::runtime_error &err) {
            logger(LogLevel::Error, "%s", err.what());
            return;
        }

        // Now compute the voxel grid bbox and related info
        // This is coherent with the neural prefiltering approach
        assert(m_cuda_backend->m_neural_bvh_indices.size() == 1);
        Bbox3f scene_bbox      = m_cuda_backend->m_cpu_mesh_bvhs[m_cuda_backend->m_neural_bvh_indices[0]].root_bbox();
        Bbox3f voxel_grid_bbox = Bbox3f{scene_bbox.min, scene_bbox.max};
        glm::vec3 lengths      = scene_bbox.max - scene_bbox.min;  // check length of given bbox in every direction
        float max_length       = glm::max(lengths.x, glm::max(lengths.y, lengths.z));  // find max length
        for (uint32_t i = 0; i < 3; i++) {                                             // for every direction (X,Y,Z)
            if (max_length == lengths[i]) {
                continue;
            } else {
                float delta = max_length - lengths[i];  // compute difference between largest
                                                        // length and current (X,Y or Z) length
                voxel_grid_bbox.min[i] =
                    scene_bbox.min[i] - (delta / 2.0f);  // pad with half the difference before current min
                voxel_grid_bbox.max[i] =
                    scene_bbox.max[i] + (delta / 2.0f);  // pad with half the difference behind current max
            }
        }
        // Same regularisation as in prefiltering
        glm::vec3 epsilon = (voxel_grid_bbox.max - voxel_grid_bbox.min) * 1e-4f;
        voxel_grid_bbox.min -= epsilon;
        voxel_grid_bbox.max += epsilon;

        uint32_t voxel_grid_max_res = 512;
        glm::vec3 unit              = glm::vec3((voxel_grid_bbox.max.x - voxel_grid_bbox.min.x) / voxel_grid_max_res,
                                   (voxel_grid_bbox.max.y - voxel_grid_bbox.min.y) / voxel_grid_max_res,
                                   (voxel_grid_bbox.max.z - voxel_grid_bbox.min.z) / voxel_grid_max_res);
        assert(abs(unit.x - unit.y) < 1e5 && abs(unit.y - unit.z) < 1e5);

        m_prefiltering_module_const.voxel_grid_bbox    = voxel_grid_bbox;
        m_prefiltering_module_const.voxel_size_lod_0   = unit.x;
        m_prefiltering_module_const.voxel_grid_max_res = voxel_grid_max_res;

        logger(LogLevel::Info,
               "Voxel Grid Bounding Box: (%g,%g,%g)-(%g,%g,%g)",
               voxel_grid_bbox.min.x,
               voxel_grid_bbox.min.y,
               voxel_grid_bbox.min.z,
               voxel_grid_bbox.max.x,
               voxel_grid_bbox.max.y,
               voxel_grid_bbox.max.z);

        logger(LogLevel::Info, "voxelSize: %g", m_prefiltering_module_const.voxel_size_lod_0);

        logger(LogLevel::Info,
               "Loaded Appearance Network info : \n"
               "\tn_params: %u"
               "\tsize in MB : %f",
               m_appearance_model->network->n_params(),
               2.0 * m_appearance_model->network->n_params() / 1000000.0);
    }

    NeuralPrefilteringModule::~NeuralPrefilteringModule()
    {
        destroy_learning_data();
    }

    CudaSceneData NeuralPrefilteringModule::scene_data()
    {
        CudaSceneData data;
        data.tlas      = m_cuda_backend->m_cuda_tlas;
        data.materials = m_cuda_backend->m_cuda_material_buffer.d_pointer();
        data.textures  = m_cuda_backend->m_cuda_textures_buffer.d_pointer();

        return data;
    }

    void NeuralPrefilteringModule::initialize_learning_data(uint32_t batch_size)
    {
        CUDA_CHECK(cudaMalloc(&m_active_rays_counter, sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&m_its_counter, sizeof(uint32_t)));

        // Alloc with right size
        CUDA_CHECK(cudaMalloc(&m_neural_learning_results.leaf_mint, sizeof(float) * batch_size));
        CUDA_CHECK(cudaMalloc(&m_neural_learning_results.leaf_maxt, sizeof(float) * batch_size));
    }

    void NeuralPrefilteringModule::destroy_learning_data()
    {
        cudaFree(m_active_rays_counter);
        cudaFree(m_its_counter);

        // Free if previously allocated
        if (m_neural_learning_results.leaf_mint)
            CUDA_CHECK(cudaFree(m_neural_learning_results.leaf_mint));
        if (m_neural_learning_results.leaf_maxt)
            CUDA_CHECK(cudaFree(m_neural_learning_results.leaf_maxt));
    }

    void NeuralPrefilteringModule::reinitialise_learning_data(uint32_t batch_size)
    {
        destroy_learning_data();
        initialize_learning_data(batch_size);
    }

    void NeuralPrefilteringModule::reinitialise_inference_data_module(const glm::uvec2 &fb_size)
    {
        cudaStream_t default_stream = 0;

        uint32_t n_padded_ray_payloads = tcnn::next_multiple(fb_size.x * fb_size.y, 4U);
        uint32_t n_padded_elements     = tcnn::next_multiple(fb_size.x * fb_size.y, tcnn::batch_size_granularity);

        auto query_scratch = tcnn::allocate_workspace_and_distribute<uint8_t,  // valid_hit[0]
                                                                     float,    // t_neural[0]
                                                                     //  m_wavefront_neural_data[0].results
                                                                     uint8_t,  // valid_hit[1]
                                                                     float,    // t_neural[1]
                                                                     //  m_wavefront_neural_data[1].results
                                                                     uint8_t,  // valid_hit
                                                                     float,    // t_neural
                                                                     float,
                                                                     float,
                                                                     float,  // ray_o
                                                                     float,
                                                                     float,
                                                                     float,      // ray_d
                                                                     uint8_t,    // ray_depth
                                                                     glm::vec3,  // illumination
                                                                     glm::vec4,  // throughput_rng
                                                                     uint32_t,   // pixel_coord
                                                                     ///  m_hybrid_intersection_data
                                                                     glm::vec3,  // illumination[0]
                                                                     glm::vec4,  // throughput_rng[0]
                                                                     // neural_traversal_data[0].illumination
                                                                     glm::vec3,  // illumination[1]
                                                                     glm::vec4,  // throughput_rng[1]
                                                                     // neural_traversal_data[1].illumination
                                                                     float,  // bvh network outputs
                                                                     float,  // prefiltering inputs
                                                                     float   // prefiltering outputs
                                                                     >(
            default_stream,
            &m_inference_output_data->scratch_alloc,
            ////// Ping buffer //////////
            n_padded_ray_payloads,  // valid_hit[0]
            n_padded_ray_payloads,  // t_neural[0]
            ////// Pong buffer //////////
            n_padded_ray_payloads,  // valid_hit[1]
            n_padded_ray_payloads,  // t_neural[1]
            ////// HybridIntersectionResult buffer //////////
            n_padded_ray_payloads,  // valid_hit
            n_padded_ray_payloads,  // t_neural
            n_padded_ray_payloads,
            n_padded_ray_payloads,
            n_padded_ray_payloads,  // ray_o
            n_padded_ray_payloads,
            n_padded_ray_payloads,
            n_padded_ray_payloads,  // ray_d
            n_padded_ray_payloads,  // ray_depth
            n_padded_ray_payloads,  // illumination
            n_padded_ray_payloads,  // throughput_rng
            n_padded_ray_payloads,  // pixel_coord
            //  m_hybrid_intersection_data
            n_padded_ray_payloads,  // illumination[0]
            n_padded_ray_payloads,  // throughput_rng[0]
            // neural_traversal_data[0].illumination
            n_padded_ray_payloads,  // illumination[1]
            n_padded_ray_payloads,  // throughput_rng[1]
            // neural_traversal_data[1].illumination
            n_padded_elements * m_pt_target_const.output_network_dims,
            n_padded_elements * m_prefiltering_module_const.prefiltering_input_dims,  // prefiltering inputs
            n_padded_elements * m_prefiltering_module_const.prefiltering_output_dims  // prefiltering outputs
        );

        NeuralTraversalResult neural_query_result_0 =
            NeuralTraversalResult(std::get<0>(query_scratch), std::get<1>(query_scratch));

        NeuralTraversalResult neural_query_result_1 =
            NeuralTraversalResult(std::get<2>(query_scratch), std::get<3>(query_scratch));

        m_wavefront_neural_data[0].results = neural_query_result_0;
        m_wavefront_neural_data[1].results = neural_query_result_1;

        m_hybrid_intersection_data = HybridIntersectionResult(
            std::get<4>(query_scratch),
            std::get<5>(query_scratch),
            vec3_soa(std::get<6>(query_scratch), std::get<7>(query_scratch), std::get<8>(query_scratch)),
            vec3_soa(std::get<9>(query_scratch), std::get<10>(query_scratch), std::get<11>(query_scratch)),
            std::get<12>(query_scratch),
            std::get<13>(query_scratch),
            std::get<14>(query_scratch),
            std::get<15>(query_scratch));

        // The parent module does not set these as the child module might not necessarly need them
        // so we set them here and assign the traversal data
        m_inference_input_data->neural_traversal_data[0].illumination   = std::get<16>(query_scratch);
        m_inference_input_data->neural_traversal_data[0].throughput_rng = std::get<17>(query_scratch);
        m_inference_input_data->neural_traversal_data[1].illumination   = std::get<18>(query_scratch);
        m_inference_input_data->neural_traversal_data[1].throughput_rng = std::get<19>(query_scratch);

        m_wavefront_neural_data[0].traversal_data = m_inference_input_data->neural_traversal_data[0];
        m_wavefront_neural_data[1].traversal_data = m_inference_input_data->neural_traversal_data[1];

        m_inference_output_data->network_outputs = std::get<20>(query_scratch);

        m_appearance_input_data  = std::get<21>(query_scratch);
        m_appearance_output_data = std::get<22>(query_scratch);
    }

    void NeuralPrefilteringModule::update_neural_model_module()
    {
        assert(m_pt_target_const.output_visibility_dim == 1);
        m_pt_target_const.output_network_dims =
            m_pt_target_const.output_visibility_dim + m_pt_target_const.output_depth_dim +
            m_pt_target_const.output_normal_dim + m_pt_target_const.output_albedo_dim;

        assert(m_pt_target_const.output_normal_dim == 0);
        assert(m_pt_target_const.output_albedo_dim == 0);

        network_loss_config = {
            {"otype", "PathTracingComposite"},
            {"output_visibility_dim", m_pt_target_const.output_visibility_dim},
            {"visibility_loss", loss_types[m_pt_target_const.visibility_loss]},
            {"output_depth_dim", m_pt_target_const.output_depth_dim},
            {"depth_loss", loss_types[m_pt_target_const.depth_loss]},
            {"output_normal_dim", m_pt_target_const.output_normal_dim},
            {"normal_loss", loss_types[m_pt_target_const.normal_loss]},
            {"output_albedo_dim", m_pt_target_const.output_albedo_dim},
            {"albedo_loss", loss_types[m_pt_target_const.albedo_loss]},
        };
    }

    nlohmann::json NeuralPrefilteringModule::network_loss_module()
    {
        return network_loss_config;
    }

    void NeuralPrefilteringModule::setup_neural_module_constants_module()
    {
        CUDA_CHECK(cudaMemcpyToSymbol(
            prefiltering_module_const, &m_prefiltering_module_const, sizeof(prefiltering_module_const)));
        CUDA_CHECK(cudaMemcpyToSymbol(pt_target_const, &m_pt_target_const, sizeof(pt_target_const)));
    }

    uint32_t NeuralPrefilteringModule::network_output_dims_module()
    {
        return m_pt_target_const.output_network_dims;
    }

    bool NeuralPrefilteringModule::ui_module(bool &reset_renderer, bool dev_mode)
    {
        bool changed_model = false;

        ImGui::Begin("Neural Prefiltering");

        ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.4f);

        if (dev_mode) {
#ifndef NDEBUG
            reset_renderer |= ImGui::InputInt("Max BVH traversal steps", &m_max_traversal_steps);
            reset_renderer |= ImGui::InputInt("Max Depth", &m_max_depth);
#endif
        }
        reset_renderer |= ImGui::Checkbox("Apply traversal RR", &m_prefiltering_module_const.apply_traversal_rr);

        if (m_prefiltering_module_const.apply_traversal_rr) {
            ImGui::Indent();
            reset_renderer |= ImGui::DragFloat("Traversal RR",
                                               &m_prefiltering_module_const.traversal_rr,
                                               m_prefiltering_module_const.traversal_rr / 128.f,
                                               1.f);
            ImGui::Unindent();
        }

        if (imgui_utils::combo_box(
                "Output Mode", output_modes, ARRAY_SIZE(output_modes), m_prefiltering_module_const.output_mode)) {
            changed_model = true;
            switch (m_prefiltering_module_const.output_mode) {
            case OutputMode::AppearanceNetworkDebug:
            case OutputMode::NeuralPrefiltering:
                m_pt_target_const.output_visibility_dim = 1;
                m_pt_target_const.output_depth_dim =
                    m_pt_target_const.depth_learning_mode != DepthLearningMode::Position &&
                            m_pt_target_const.depth_learning_mode != DepthLearningMode::LocalPosition
                        ? 1
                        : 3;
                break;
            default:
                UNREACHABLE();
            }
        }

        if (ImGui::CollapsingHeader("Appearance Network Config", nullptr, ImGuiTreeNodeFlags_DefaultOpen)) {
            reset_renderer |= imgui_utils::combo_box_int(
                "LoD level", lod_values, ARRAY_SIZE(lod_values), m_prefiltering_module_const.lod_level);
            if (dev_mode) {
                reset_renderer |=
                    ImGui::Checkbox("Log learnt prefiltering", &m_prefiltering_module_const.prefiltering_log_learnt);

                reset_renderer |=
                    ImGui::Checkbox("Apply russian roulette", &m_prefiltering_module_const.apply_russian_roulette);

                if (m_prefiltering_module_const.apply_russian_roulette) {
                    ImGui::Indent();
                    reset_renderer |= ImGui::InputInt("RR Start depth", &m_prefiltering_module_const.rr_start_depth);
                    ImGui::Unindent();
                }
            }
        }

        if (dev_mode) {
            changed_model |= m_pt_target_const.pt_target_ui(reset_renderer, true, true);
        }
        ImGui::Unindent();

        ImGui::PopItemWidth();

        ImGui::End();

        return changed_model;
    }

}
}