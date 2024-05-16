

#include "cuda/neural/neural_bvh_module.h"
#include "cuda/neural/neural_bvh_module_constants.cuh"
#include "cuda/neural/neural_bvh_renderer_kernels.cuh"

namespace ntwr {

namespace neural {
    NeuralBVHModule::~NeuralBVHModule() {}

    void NeuralBVHModule::reinitialise_inference_data(const glm::uvec2 &fb_size)
    {
        cudaStream_t default_stream = 0;

        // We pad the number of rays to a multiple of 4 to avoid misaligned address when resizing framebuffer
        // During splatting the framebuffer size is still used so we essentially just compute a few more rays
        // than necessary.
        uint32_t n_padded_ray_payloads = tcnn::next_multiple(fb_size.x * fb_size.y, 4U);
        uint32_t n_padded_elements     = tcnn::next_multiple(fb_size.x * fb_size.y, tcnn::batch_size_granularity);

        auto traversal_scratch =
            tcnn::allocate_workspace_and_distribute<float,
                                                    float,
                                                    float,  // ray_o[0]
                                                    float,
                                                    float,
                                                    float,     // ray_d[0]
                                                    uint8_t,   // ray_depth[0]
                                                    uint8_t,   // inferenced_node_idx_and_intersected_nodes[0]
                                                    float,     // leaf_mint[0]
                                                    float,     // leaf_maxt[0]
                                                    int,       // stack_size[0]
                                                    uint32_t,  // stack[0]
                                                    uint32_t,  // pixel_coord_active[0]
                                                    uint32_t,  // node_idx[0]
                                                    // m_wavefront_neural_data.traversal_data[0]
                                                    float,
                                                    float,
                                                    float,  // ray_o[1]
                                                    float,
                                                    float,
                                                    float,     // ray_d[1]
                                                    uint8_t,   // ray_depth[1]
                                                    uint8_t,   // inferenced_node_idx_and_intersected_nodes[1]
                                                    float,     // leaf_mint[1]
                                                    float,     // leaf_maxt[1]
                                                    int,       // stack_size[1]
                                                    uint32_t,  // stack[1]
                                                    uint32_t,  // pixel_coord_active[1]
                                                    uint32_t,  // node_idx[1]
                                                    // m_wavefront_neural_data.traversal_data[1]
                                                    float  // bvh inputs
                                                    >(
                default_stream,
                &m_inference_input_data->traversal_scratch_alloc,
                n_padded_ray_payloads,
                n_padded_ray_payloads,
                n_padded_ray_payloads,  // ray_o[0]
                n_padded_ray_payloads,
                n_padded_ray_payloads,
                n_padded_ray_payloads,                      // ray_d[0]
                n_padded_ray_payloads,                      // ray_depth[0]
                n_padded_ray_payloads,                      // inferenced_node_idx_and_intersected_nodes[0]
                n_padded_ray_payloads,                      // leaf_mint[0]
                n_padded_ray_payloads,                      // leaf_maxt[0]
                n_padded_ray_payloads,                      // stack_size[0]
                n_padded_ray_payloads * NEURAL_STACK_SIZE,  // stack[0]
                n_padded_ray_payloads,                      // pixel_coord_active[0]
                n_padded_ray_payloads,                      // node_idx[0]
                ////////////////////////
                n_padded_ray_payloads,
                n_padded_ray_payloads,
                n_padded_ray_payloads,  // ray_o[1]
                n_padded_ray_payloads,
                n_padded_ray_payloads,
                n_padded_ray_payloads,                      // ray_d[1]
                n_padded_ray_payloads,                      // ray_depth[1]
                n_padded_ray_payloads,                      // inferenced_node_idx_and_intersected_nodes[1]
                n_padded_ray_payloads,                      // leaf_mint[1]
                n_padded_ray_payloads,                      // leaf_maxt[1]
                n_padded_ray_payloads,                      // stack_size[1]
                n_padded_ray_payloads * NEURAL_STACK_SIZE,  // stack[1]
                n_padded_ray_payloads,                      // pixel_coord_active[1]
                n_padded_ray_payloads,                      // node_idx[1]
                n_padded_elements * m_bvh_module_const.input_network_dims);

        NeuralTraversalData neural_traversal_data_0 = NeuralTraversalData(
            vec3_soa(std::get<0>(traversal_scratch), std::get<1>(traversal_scratch), std::get<2>(traversal_scratch)),
            vec3_soa(std::get<3>(traversal_scratch), std::get<4>(traversal_scratch), std::get<5>(traversal_scratch)),
            std::get<6>(traversal_scratch),
            std::get<7>(traversal_scratch),
            std::get<8>(traversal_scratch),
            std::get<9>(traversal_scratch),
            std::get<10>(traversal_scratch),
            std::get<11>(traversal_scratch),
            std::get<12>(traversal_scratch),
            std::get<13>(traversal_scratch));
        NeuralTraversalData neural_traversal_data_1 = NeuralTraversalData(
            vec3_soa(std::get<14>(traversal_scratch), std::get<15>(traversal_scratch), std::get<16>(traversal_scratch)),
            vec3_soa(std::get<17>(traversal_scratch), std::get<18>(traversal_scratch), std::get<19>(traversal_scratch)),
            std::get<20>(traversal_scratch),
            std::get<21>(traversal_scratch),
            std::get<22>(traversal_scratch),
            std::get<23>(traversal_scratch),
            std::get<24>(traversal_scratch),
            std::get<25>(traversal_scratch),
            std::get<26>(traversal_scratch),
            std::get<27>(traversal_scratch));
        m_inference_input_data->network_inputs = std::get<28>(traversal_scratch);

        m_inference_input_data->neural_traversal_data[0] = neural_traversal_data_0;
        m_inference_input_data->neural_traversal_data[1] = neural_traversal_data_1;

        reinitialise_inference_data_module(fb_size);
    }

    void NeuralBVHModule::reinitialise_neural_model(uint32_t batch_size)
    {
        // Call child module to update anything related to output (loss and output dims ususally)
        update_neural_model_module();

        // Now reinitialise learning data (covers the case where the output dim would have changed for example)
        reinitialise_learning_data(batch_size);

        // Define input dimensionality required
        m_bvh_module_const.input_segment_points_dim =
            m_bvh_module_const.input_segment_points * m_bvh_module_const.input_segment_point_dim;

        m_bvh_module_const.input_network_dims =
            m_bvh_module_const.input_segment_points_dim + m_bvh_module_const.input_segment_direction_dim;

        // logger(LogLevel::Warn,
        //         "NeuralBVHModule input dims (point_dim,repetition,total)"
        //         "(%d,%d,%d)",
        //         m_bvh_module_const.input_segment_point_dim,
        //         m_bvh_module_const.input_segment_points,
        //         m_bvh_module_const.input_segment_points_dim);

        m_model->training_input_batch = tcnn::GPUMatrix<float>(m_bvh_module_const.input_network_dims, batch_size);
        m_model->training_input_batch.initialize_constant(0.f);

        m_model->training_target_batch = tcnn::GPUMatrix<float>(network_output_dims(), batch_size);
        m_model->training_target_batch.initialize_constant(0.f);

        tcnn::json optimizer_config = {{"otype", "ExponentialDecay"},
                                       {"decay_start", m_optimizer_options.decay_start},
                                       {"decay_interval", m_optimizer_options.decay_interval},
                                       {"decay_base", m_optimizer_options.decay_base},
                                       {"nested",
                                        {
                                            {"otype", optimizer_types[m_optimizer_options.type]},
                                            {"learning_rate", m_optimizer_options.learning_rate},
                                            {"beta1", m_optimizer_options.beta1},
                                            {"beta2", m_optimizer_options.beta2},
                                            {"epsilon", m_optimizer_options.epsilon},
                                            {"l2_reg", m_optimizer_options.l2_reg},
                                            {"relative_decay", m_optimizer_options.relative_decay},
                                            {"absolute_decay", m_optimizer_options.absolute_decay},
                                            {"adabound", m_optimizer_options.adabound},
                                            // The following parameters are only used when the optimizer is
                                            // "Shampoo".
                                            {"beta3", 0.9f},
                                            {"beta_shampoo", 0.0f},
                                            {"identity", 0.0001f},
                                            {"cg_on_momentum", false},
                                            {"frobenius_normalization", true},
                                        }}};

        tcnn::json point_encoding = {
            {"n_dims_to_encode", m_bvh_module_const.input_segment_point_dim},
            {"otype", encoding_types[m_segment_encoding]},
            {"hash", hash_types[m_segment_points_hashgrid_options.hash]},
            // hashgrid/grid encodings
            {"n_levels", m_segment_points_hashgrid_options.n_levels},
            {"n_features_per_level", m_segment_points_hashgrid_options.n_features_per_level},
            {"log2_hashmap_size", m_segment_points_hashgrid_options.log2_hashmap_size},
            {"base_resolution", m_segment_points_hashgrid_options.base_resolution},
            {"per_level_scale", m_segment_points_hashgrid_options.per_level_scale},
            {"stochastic_interpolation", m_segment_points_hashgrid_options.stochastic_interpolation},
            {"interpolation", interpolation_types[m_segment_points_hashgrid_options.interpolation]},
        };

#ifdef USE_REPEATED_COMPOSITE
        tcnn::json segment_points_encoding_config = {
            {"otype", "RepeatedComposite"},
            {"reduction", reduction_types[m_segment_reduction]},
            {"n_repetitions", m_bvh_module_const.input_segment_points},
            {"n_dims_to_encode", m_bvh_module_const.input_segment_points_dim},
            {"nested", {point_encoding}},
        };
#else
        tcnn::json points_encoding = tcnn::json::array();
        // Loop to add "point_encoding" to the JSON array multiple times
        for (uint32_t i = 0; i < m_bvh_module_const.input_segment_points; ++i) {
            points_encoding.push_back(point_encoding);
        }

        tcnn::json segment_points_encoding_config = {
            {"otype", "Composite"},
            {"reduction", reduction_types[m_segment_reduction]},
            {"n_dims_to_encode", m_bvh_module_const.input_segment_points_dim},
            {"nested", points_encoding},
        };
#endif

        tcnn::json segment_direction_encoding_config = {
            {"otype", encoding_types[m_segment_direction_encoding]},
            {"n_dims_to_encode", m_bvh_module_const.input_segment_direction_dim},
            // hashgrid/grid encodings
            {"n_levels", m_segment_direction_hashgrid_options.n_levels},
            {"n_features_per_level", m_segment_direction_hashgrid_options.n_features_per_level},
            {"log2_hashmap_size", m_segment_direction_hashgrid_options.log2_hashmap_size},
            {"base_resolution", m_segment_direction_hashgrid_options.base_resolution},
            {"per_level_scale", m_segment_direction_hashgrid_options.per_level_scale},
            {"stochastic_interpolation", m_segment_direction_hashgrid_options.stochastic_interpolation},
            {"interpolation", interpolation_types[m_segment_direction_hashgrid_options.interpolation]},
        };

        tcnn::json input_encoding = tcnn::json::array();
        input_encoding.push_back(segment_points_encoding_config);
        if (m_bvh_module_const.input_segment_direction_dim > 0) {
            input_encoding.push_back(segment_direction_encoding_config);
        }

        tcnn::json encoding_config = {
            {"otype", "Composite"},
            {"n_dims_to_encode", m_bvh_module_const.input_network_dims},
            {"nested", input_encoding},
        };

        tcnn::json network_config = {
            {"otype", network_types[m_network_options.network]},
            {"n_neurons", m_network_options.n_neurons},
            {"n_hidden_layers", m_network_options.n_hidden_layers},
            {"activation", activation_types[m_network_options.activation]},
            {"output_activation", activation_types[m_network_options.output_activation]},
        };

        m_model->config = {
            {"loss", network_loss_module()},
            {"optimizer", optimizer_config},
            {"encoding", encoding_config},
            {"network", network_config},
        };

        tcnn::json loss_opts      = m_model->config.value("loss", tcnn::json::object());
        tcnn::json encoding_opts  = m_model->config.value("encoding", tcnn::json::object());
        tcnn::json optimizer_opts = m_model->config.value("optimizer", tcnn::json::object());
        tcnn::json network_opts   = m_model->config.value("network", tcnn::json::object());

        m_model->loss = std::shared_ptr<tcnn::Loss<tcnn::network_precision_t>>{
            tcnn::create_loss<tcnn::network_precision_t>(loss_opts)};

        m_model->optimizer = std::shared_ptr<tcnn::Optimizer<tcnn::network_precision_t>>{
            tcnn::create_optimizer<tcnn::network_precision_t>(optimizer_opts)};

        m_model->network = std::make_shared<tcnn::NetworkWithInputEncoding<tcnn::network_precision_t>>(
            m_bvh_module_const.input_network_dims, network_output_dims(), encoding_opts, network_opts);

        m_model->trainer = std::make_shared<tcnn::Trainer<float, tcnn::network_precision_t, tcnn::network_precision_t>>(
            m_model->network, m_model->optimizer, m_model->loss);
    }

    void NeuralBVHModule::setup_neural_module_constants()
    {
        CUDA_CHECK(cudaMemcpyToSymbol(bvh_module_const, &m_bvh_module_const, sizeof(bvh_module_const)));

        setup_neural_module_constants_module();
    }

    uint32_t NeuralBVHModule::network_output_dims()
    {
        return network_output_dims_module();
    }

    bool NeuralBVHModule::ui_encoding(bool &reset_renderer, bool dev_mode)
    {
        bool changed_model = false;

        ImGui::SeparatorText("Network Stats");
        ImGui::Indent();
        ImGui::Text(
            "Num params: %lu\n"
            "Size in MB : %.6f",
            m_model->network->n_params(),
            2.0 * m_model->network->n_params() * 0.000001f);
        ImGui::Unindent();

        if (dev_mode) {
            if (ImGui::CollapsingHeader("Segment Encoding Config", nullptr, ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::Indent();

                if (imgui_utils::combo_box("Segment Distribution",
                                           segment_distributions,
                                           ARRAY_SIZE(segment_distributions),
                                           m_bvh_module_const.segment_distribution)) {
                    changed_model = true;
                }

                changed_model |= ImGui::InputInt("Segment points",
                                                 reinterpret_cast<int *>(&m_bvh_module_const.input_segment_points));
                changed_model |= imgui_utils::combo_box(
                    "Segment Encoding", encoding_types, ARRAY_SIZE(encoding_types), m_segment_encoding);
                changed_model |= imgui_utils::combo_box(
                    "Segment Encoding Reduction", reduction_types, ARRAY_SIZE(reduction_types), m_segment_reduction);

                if ((m_segment_encoding == EncodingType::HashGrid || m_segment_encoding == EncodingType::DenseGrid) &&
                    ImGui::CollapsingHeader(
                        "Hashgrid Config##SegmentPoints", nullptr, ImGuiTreeNodeFlags_DefaultOpen)) {
                    changed_model |=
                        hashgrid_encoding_ui(m_segment_points_hashgrid_options, std::string("SegmentPoints"));
                }
                ImGui::Unindent();
            }

            bool directional_input_encoding = m_bvh_module_const.input_segment_direction_dim == 3;
            if (ImGui::Checkbox("Include segment direction encoding", &directional_input_encoding)) {
                changed_model |= true;
                m_bvh_module_const.input_segment_direction_dim = directional_input_encoding ? 3 : 0;
            }

            if (directional_input_encoding &&
                ImGui::CollapsingHeader("Segment Direction Encoding Config", nullptr, ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::Indent();

                changed_model |=
                    ImGui::Checkbox("Anti-alias segment direction", &m_bvh_module_const.antialias_direction);

                changed_model |= imgui_utils::combo_box("Segment Direction Encoding",
                                                        encoding_types,
                                                        ARRAY_SIZE(encoding_types),
                                                        m_segment_direction_encoding);

                if ((m_segment_direction_encoding == EncodingType::HashGrid ||
                     m_segment_direction_encoding == EncodingType::DenseGrid) &&
                    ImGui::CollapsingHeader(
                        "Hashgrid Config##SegmentDirection", nullptr, ImGuiTreeNodeFlags_DefaultOpen)) {
                    changed_model |=
                        hashgrid_encoding_ui(m_segment_direction_hashgrid_options, std::string("SegmentDirection"));
                }

                ImGui::Unindent();
            }

            if (ImGui::CollapsingHeader("MLP Config", nullptr, ImGuiTreeNodeFlags_DefaultOpen)) {
                changed_model |= imgui_utils::combo_box(
                    "network type", network_types, ARRAY_SIZE(network_types), m_network_options.network);

                changed_model |= imgui_utils::combo_box_int(
                    "n_neurons", n_neurons_values, ARRAY_SIZE(n_neurons_values), m_network_options.n_neurons);
                if (m_network_options.n_neurons > 128) {
                    m_network_options.network = NetworkType::CutlassMLP;
                }
                changed_model |= imgui_utils::combo_box_int("n_hidden_layers",
                                                            n_hidden_layers_values,
                                                            ARRAY_SIZE(n_hidden_layers_values),
                                                            m_network_options.n_hidden_layers);

                changed_model |= imgui_utils::combo_box(
                    "Activations", activation_types, ARRAY_SIZE(activation_types), m_network_options.activation);
                changed_model |= imgui_utils::combo_box("Output activation",
                                                        activation_types,
                                                        ARRAY_SIZE(activation_types),
                                                        m_network_options.output_activation);
            }

            if (ImGui::CollapsingHeader("Optimisation", nullptr)) {
                ImGui::Indent();

                changed_model |= imgui_utils::combo_box(
                    "Optimizer", optimizer_types, ARRAY_SIZE(optimizer_types), m_optimizer_options.type);
                changed_model |= ImGui::InputFloat("learning_rate", &m_optimizer_options.learning_rate);
                changed_model |= ImGui::InputFloat("beta1", &m_optimizer_options.beta1);
                changed_model |= ImGui::InputFloat("beta2", &m_optimizer_options.beta2);
                changed_model |= ImGui::InputFloat("epsilon", &m_optimizer_options.epsilon);
                changed_model |= ImGui::InputFloat("l2_reg", &m_optimizer_options.l2_reg);
                changed_model |= ImGui::InputFloat("relative_decay", &m_optimizer_options.relative_decay);
                changed_model |= ImGui::InputFloat("absolute_decay", &m_optimizer_options.absolute_decay);
                changed_model |= ImGui::Checkbox("adabound", &m_optimizer_options.adabound);

                ImGui::Unindent();
            }
        } else {
            ImGui::Spacing();
            ImGui::Indent();
            if ((m_segment_encoding == EncodingType::HashGrid || m_segment_encoding == EncodingType::DenseGrid) &&
                ImGui::CollapsingHeader("Hashgrid Config##SegmentPoints", nullptr, ImGuiTreeNodeFlags_DefaultOpen)) {
                changed_model |= hashgrid_encoding_ui(m_segment_points_hashgrid_options, std::string("SegmentPoints"));
            }
            ImGui::Unindent();
        }
        return changed_model;
    }

    void NeuralBVHModule::optimisation_data_generation(uint32_t training_step,
                                                       uint32_t batch_size,
                                                       uint32_t log2_bvh_max_traversals,
                                                       const NeuralBVH &neural_bvh,
                                                       const CudaSceneData &scene_data,
                                                       const CUDABuffer<uint32_t> &sampled_node_indices,
                                                       float current_max_loss,
                                                       const CUDABuffer<NeuralTreeCutData> &neural_tree_cut_data)
    {
        this->generate_and_fill_bvh_output_reference(training_step,
                                                     batch_size,
                                                     log2_bvh_max_traversals,
                                                     neural_bvh,
                                                     scene_data,
                                                     sampled_node_indices,
                                                     current_max_loss,
                                                     neural_tree_cut_data);
    }

    void NeuralBVHModule::neural_render(uint32_t accumulated_spp, uint32_t sample_offset, const NeuralBVH &neural_bvh)
    {
        neural_render_module(accumulated_spp, sample_offset, neural_bvh);
    }

    void NeuralBVHModule::split_screen_neural_render_node_loss(
        uint32_t accumulated_spp,
        uint32_t sample_offset,
        const NeuralBVH &neural_bvh,
        const CUDABuffer<NeuralTreeCutData> &neural_tree_cut_data,
        float max_node_loss_host,
        bool show_tree_cut_depth,
        uint32_t neural_tree_cut_max_depth)
    {
        split_screen_neural_render_node_loss_module(accumulated_spp,
                                                    sample_offset,
                                                    neural_bvh,
                                                    neural_tree_cut_data,
                                                    max_node_loss_host,
                                                    show_tree_cut_depth,
                                                    neural_tree_cut_max_depth);
    }

}
}