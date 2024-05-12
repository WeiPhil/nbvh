#pragma once

#include "cuda/neural/utils/tcnn_config.h"

#ifdef __CUDACC__
#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/config.h>
#endif

#include <json/json.hpp>

#include "cuda/neural/accel/neural_bvh.cuh"
// For neural bvh2 leaf nodes
#include "cuda/neural/accel/neural_bvh2.cuh"

namespace ntwr {

namespace neural {

    class NeuralBVHRenderer;
    struct NeuralTreeCutData;

    static const char *const segment_distributions[] = {
        "EquidistantInterior", "EquidistantExterior", "Uniform"};

    enum SegmentDistribution { EquidistantInterior = 0, EquidistantExterior, Uniform };

#ifdef __CUDACC__
    struct NeuralNetworkModel {
    public:
        tcnn::json config;

        // Auxiliary matrices for training throughput
        tcnn::GPUMatrix<float> training_target_batch;
        tcnn::GPUMatrix<float> training_input_batch;

        //// Optimisation Model

        std::shared_ptr<tcnn::Loss<tcnn::network_precision_t>> loss;
        std::shared_ptr<tcnn::Optimizer<tcnn::network_precision_t>> optimizer;
        std::shared_ptr<tcnn::NetworkWithInputEncoding<tcnn::network_precision_t>> network;
        std::shared_ptr<tcnn::Trainer<float, tcnn::network_precision_t, tcnn::network_precision_t>> trainer;
    };
#else
    class NeuralNetworkModel;
#endif

#ifdef __CUDACC__
    struct NeuralInferenceInputData {
        tcnn::GPUMemoryArena::Allocation traversal_scratch_alloc;

        // Two for allowing compression later
        NeuralTraversalData neural_traversal_data[2];

        float *network_inputs;
    };
#else
    struct NeuralInferenceInputData;
#endif

#ifdef __CUDACC__
    template <typename NeuralTraversalResult>
    struct NeuralInferenceOutputData {
        tcnn::GPUMemoryArena::Allocation scratch_alloc;
        float *network_outputs;
    };
#else
    template <typename NeuralTraversalResult>
    struct NeuralInferenceOutputData;
#endif

    template <typename NeuralTraversalResult>
    struct NeuralWavefrontData {
        NeuralTraversalData traversal_data;
        NeuralTraversalResult results;
    };

    // Input of the network is stored in the BVHModule, output in the child module
    struct alignas(8) NeuralBVHModuleConstants {
        uint32_t input_segment_point_dim = 3;  // repetition dimensionality (3d points)
        uint32_t input_segment_points    = 3;  // number of points along the segment
        uint32_t input_segment_points_dim =
            input_segment_point_dim * input_segment_points;  // input segment encoding dimensionality

        uint32_t input_segment_direction_dim = 0;  // 3 if using directional_input_encoding true 0 otherwise

        uint32_t input_network_dims =
            input_segment_direction_dim + input_segment_points_dim;  // total input dimensionality

        SegmentDistribution segment_distribution = SegmentDistribution::EquidistantInterior;

        bool antialias_direction  = false;
        bool merge_operator       = false;
        bool apply_encoding_trafo = false;

        inline NTWR_HOST nlohmann::json to_json()
        {
            return nlohmann::ordered_json{
                {"input_segment_point_dim", input_segment_point_dim},
                {"input_segment_points", input_segment_points},
                {"input_segment_points_dim", input_segment_points_dim},
                {"input_segment_direction_dim", input_segment_direction_dim},
                {"input_network_dims", input_network_dims},
                {"segment_distribution", segment_distributions[segment_distribution]},
                {"antialias_direction", antialias_direction},
                {"merge_operator", merge_operator},
                {"apply_encoding_trafo", apply_encoding_trafo},
            };
        }

        inline NTWR_HOST void from_json(nlohmann::json json_config)
        {
            NeuralBVHModuleConstants default_config;
            input_segment_point_dim =
                json_config.value("input_segment_point_dim", default_config.input_segment_point_dim);
            input_segment_points = json_config.value("input_segment_points", default_config.input_segment_points);
            input_segment_points_dim =
                json_config.value("input_segment_points_dim", default_config.input_segment_points_dim);
            input_segment_direction_dim =
                json_config.value("input_segment_direction_dim", default_config.input_segment_direction_dim);
            input_network_dims   = json_config.value("input_network_dims", default_config.input_network_dims);
            segment_distribution = (SegmentDistribution)imgui_utils::find_index(
                json_config.value("segment_distribution", segment_distributions[default_config.segment_distribution])
                    .c_str(),
                segment_distributions,
                ARRAY_SIZE(segment_distributions));
            antialias_direction  = json_config.value("antialias_direction", default_config.antialias_direction);
            merge_operator       = json_config.value("merge_operator", default_config.merge_operator);
            apply_encoding_trafo = json_config.value("apply_encoding_trafo", default_config.apply_encoding_trafo);
        }
    };

    static const char *const depth_learning_modes[] = {"LocalT", "WorldT", "Position", "LocalPosition"};

    enum DepthLearningMode : uint8_t {
        LocalT,
        WorldT,
        Position,
        LocalPosition,
    };

    struct PathTracingOutputConstants;

    class NeuralBVHModule {
    public:
        virtual ~NeuralBVHModule() = 0;

        virtual std::string name() = 0;

        void reinitialise_inference_data(const glm::uvec2 &fb_size);
        void reinitialise_neural_model(uint32_t batch_size);

        virtual void reinitialise_learning_data(uint32_t batch_size) = 0;

        void setup_neural_module_constants();
        uint32_t network_output_dims();

        bool ui_encoding(bool &reset_renderer, bool dev_mode);

        void optimisation_data_generation(uint32_t training_step,
                                          uint32_t batch_size,
                                          uint32_t log2_bvh_max_traversals,
                                          const NeuralBVH &neural_bvh,
                                          const CudaSceneData &scene_data,
                                          const CUDABuffer<uint32_t> &sampled_node_indices,
                                          float current_max_loss,
                                          const CUDABuffer<NeuralTreeCutData> &neural_tree_cut_data);

        void neural_render(uint32_t accumulated_spp, uint32_t sample_offset, const NeuralBVH &neural_bvh);

        void split_screen_neural_render_node_loss(uint32_t accumulated_spp,
                                                  uint32_t sample_offset,
                                                  const NeuralBVH &neural_bvh,
                                                  const CUDABuffer<NeuralTreeCutData> &neural_tree_cut_data,
                                                  float m_max_node_loss_host,
                                                  bool m_show_tree_cut_depth,
                                                  uint32_t m_neural_tree_cut_max_depth);

    protected:
        // Optimizer
        OptimizerOptions m_optimizer_options;

        // Network
        NetworkOptions m_network_options;

        // BVH Segment encoding
        EncodingType m_segment_encoding = EncodingType::HashGrid;
        HashGridOptions m_segment_points_hashgrid_options;
        CompositeReductionType m_segment_reduction = CompositeReductionType::Concatenation;

        // BVH Segment direction encoding
        EncodingType m_segment_direction_encoding = EncodingType::SphericalHarmonics;
        HashGridOptions m_segment_direction_hashgrid_options;

        std::unique_ptr<NeuralInferenceInputData> m_inference_input_data;
        std::unique_ptr<NeuralNetworkModel> m_model;

        NeuralBVHModuleConstants m_bvh_module_const;

        int m_max_traversal_steps = 1000;

    private:
        virtual float *optim_leaf_mints() = 0;
        virtual float *optim_leaf_maxts() = 0;

        virtual void generate_and_fill_bvh_output_reference(
            uint32_t training_step,
            uint32_t batch_size,
            uint32_t log2_bvh_max_traversals,
            const NeuralBVH &neural_bvh,
            const CudaSceneData &scene_data,
            const CUDABuffer<uint32_t> &sampled_node_indices,
            float current_max_loss,
            const CUDABuffer<NeuralTreeCutData> &neural_tree_cut_data) = 0;
        // Template method pattern
        virtual void neural_render_module(uint32_t accumulated_spp,
                                          uint32_t sample_offset,
                                          const NeuralBVH &neural_bvh) = 0;

        virtual void split_screen_neural_render_node_loss_module(
            uint32_t accumulated_spp,
            uint32_t sample_offset,
            const NeuralBVH &neural_bvh,
            const CUDABuffer<NeuralTreeCutData> &neural_tree_cut_data,
            float m_max_node_loss_host,
            bool m_show_tree_cut_depth,
            uint32_t m_neural_tree_cut_max_depth)
        {
            // does nothing if not implemented
        }

        virtual void reinitialise_inference_data_module(const glm::uvec2 &fb_size) = 0;
        virtual void update_neural_model_module()                                  = 0;
        virtual void setup_neural_module_constants_module()                        = 0;
        virtual uint32_t network_output_dims_module()                              = 0;
        virtual nlohmann::json network_loss_module()                               = 0;
        virtual bool ui_module(bool &reset_renderer, bool dev_mode)                = 0;

        // The BVH renderer is a good friend of the BVH module
        friend class NeuralBVHRenderer;
    };

    struct alignas(8) PathTracingOutputConstants {
        uint32_t output_visibility_dim = 1;
        uint32_t output_depth_dim      = 1;
        uint32_t output_normal_dim     = 0;
        uint32_t output_albedo_dim     = 0;
        uint32_t output_network_dims = output_visibility_dim + output_depth_dim + output_normal_dim + output_albedo_dim;

        LossType visibility_loss    = LossType::BinaryCrossEntropy;
        float visibility_loss_scale = 1.f;
        LossType depth_loss         = LossType::L1;
        float depth_loss_scale      = 1.f;
        LossType normal_loss        = LossType::L1;
        float normal_loss_scale     = 1.f;
        LossType albedo_loss        = LossType::RelativeL2;
        float albedo_loss_scale     = 1.f;

        DepthLearningMode depth_learning_mode = DepthLearningMode::LocalT;

        // Force (approximate) linear activation function?
        bool output_normal_linear_activation = true;
        bool output_depth_linear_activation  = false;
        bool output_albedo_linear_activation = false;
        // Do we mask loss based on target visibility?
        bool visibility_masking_loss = true;

        float primary_visibility_threshold   = 0.5f;
        float secondary_visibility_threshold = 0.5f;
        float depth_epsilon                  = 0.2f;
        float fixed_depth_epsilon            = 1e-5f;

        bool interpolate_depth        = false;
        float depth_interpolation_eps = 1e-3;

        inline NTWR_HOST nlohmann::json to_json()
        {
            return nlohmann::ordered_json{
                {"output_visibility_dim", output_visibility_dim},
                {"output_depth_dim", output_depth_dim},
                {"output_normal_dim", output_normal_dim},
                {"output_albedo_dim", output_albedo_dim},
                {"output_network_dims", output_network_dims},
                {"visibility_loss", loss_types[visibility_loss]},
                {"visibility_loss_scale", visibility_loss_scale},
                {"depth_loss", loss_types[depth_loss]},
                {"depth_loss_scale", depth_loss_scale},
                {"normal_loss", loss_types[normal_loss]},
                {"normal_loss_scale", normal_loss_scale},
                {"albedo_loss", loss_types[albedo_loss]},
                {"albedo_loss_scale", albedo_loss_scale},
                {"depth_learning_mode", depth_learning_modes[depth_learning_mode]},
                {"output_normal_linear_activation", output_normal_linear_activation},
                {"output_depth_linear_activation", output_depth_linear_activation},
                {"output_albedo_linear_activation", output_albedo_linear_activation},
                {"visibility_masking_loss", visibility_masking_loss},
                {"primary_visibility_threshold", primary_visibility_threshold},
                {"secondary_visibility_threshold", secondary_visibility_threshold},
                {"depth_epsilon", depth_epsilon},
                {"fixed_depth_epsilon", fixed_depth_epsilon},
            };
        }

        inline NTWR_HOST void from_json(nlohmann::json json_config)
        {
            PathTracingOutputConstants default_config;
            output_visibility_dim = json_config.value("output_visibility_dim", default_config.output_visibility_dim);
            output_depth_dim      = json_config.value("output_depth_dim", default_config.output_depth_dim);
            output_normal_dim     = json_config.value("output_normal_dim", default_config.output_normal_dim);
            output_albedo_dim     = json_config.value("output_albedo_dim", default_config.output_albedo_dim);
            output_network_dims   = json_config.value("output_network_dims", default_config.output_network_dims);
            visibility_loss       = (LossType)imgui_utils::find_index(
                json_config.value("visibility_loss", loss_types[default_config.visibility_loss]).c_str(),
                loss_types,
                ARRAY_SIZE(loss_types));
            visibility_loss_scale = json_config.value("visibility_loss_scale", default_config.visibility_loss_scale);
            depth_loss            = (LossType)imgui_utils::find_index(
                json_config.value("depth_loss", loss_types[default_config.depth_loss]).c_str(),
                loss_types,
                ARRAY_SIZE(loss_types));
            depth_loss_scale = json_config.value("depth_loss_scale", default_config.depth_loss_scale);
            normal_loss      = (LossType)imgui_utils::find_index(
                json_config.value("normal_loss", loss_types[default_config.normal_loss]).c_str(),
                loss_types,
                ARRAY_SIZE(loss_types));
            normal_loss_scale = json_config.value("normal_loss_scale", default_config.normal_loss_scale);
            albedo_loss       = (LossType)imgui_utils::find_index(
                json_config.value("albedo_loss", loss_types[default_config.albedo_loss]).c_str(),
                loss_types,
                ARRAY_SIZE(loss_types));
            albedo_loss_scale   = json_config.value("albedo_loss_scale", default_config.albedo_loss_scale);
            depth_learning_mode = (DepthLearningMode)imgui_utils::find_index(
                json_config.value("depth_learning_mode", depth_learning_modes[default_config.depth_learning_mode])
                    .c_str(),
                depth_learning_modes,
                ARRAY_SIZE(depth_learning_modes));
            output_normal_linear_activation =
                json_config.value("output_normal_linear_activation", default_config.output_normal_linear_activation);
            output_depth_linear_activation =
                json_config.value("output_depth_linear_activation", default_config.output_depth_linear_activation);
            output_albedo_linear_activation =
                json_config.value("output_albedo_linear_activation", default_config.output_albedo_linear_activation);
            visibility_masking_loss =
                json_config.value("visibility_masking_loss", default_config.visibility_masking_loss);
            primary_visibility_threshold =
                json_config.value("primary_visibility_threshold", default_config.primary_visibility_threshold);
            secondary_visibility_threshold =
                json_config.value("secondary_visibility_threshold", default_config.secondary_visibility_threshold);
            depth_epsilon       = json_config.value("depth_epsilon", default_config.depth_epsilon);
            fixed_depth_epsilon = json_config.value("fixed_depth_epsilon", default_config.fixed_depth_epsilon);
        }

        bool pt_target_ui(bool &reset_renderer, bool hide_normal = false, bool hide_albedo = false)
        {
            bool changed_model = false;

            reset_renderer |=
                ImGui::SliderFloat("Primary visibility threshold", &primary_visibility_threshold, 0.0f, 1.0f);
            reset_renderer |=
                ImGui::SliderFloat("Secondary visibility threshold", &secondary_visibility_threshold, 0.0f, 1.0f);

            reset_renderer |= ImGui::SliderFloat("Inner node depth epsilon", &depth_epsilon, 0.0f, 1.0f);

            reset_renderer |= ImGui::DragFloat(
                "Inner node fixed depth epsilon", &fixed_depth_epsilon, fixed_depth_epsilon / 128.f, 0.f, 1.f, "%.2e");

            if (ImGui::CollapsingHeader("Network Output Config", nullptr, ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::Indent();

                changed_model |=
                    imgui_utils::combo_box("Visibility Loss", loss_types, ARRAY_SIZE(loss_types), visibility_loss);

                if (ImGui::Checkbox("Masking visibility loss", &visibility_masking_loss)) {
                    changed_model |= true;
                }

                changed_model |= ImGui::SliderFloat("Visibility Loss Scale", &visibility_loss_scale, 0.05f, 20.0f);

                bool learn_depth = output_depth_dim > 0;
                if (ImGui::Checkbox("Learn depth", &learn_depth)) {
                    changed_model |= true;
                    if (learn_depth) {
                        output_depth_dim = depth_learning_mode != DepthLearningMode::Position &&
                                                   depth_learning_mode != DepthLearningMode::LocalPosition
                                               ? 1
                                               : 3;
                    } else {
                        output_depth_dim = 0;
                    }
                }

                if (learn_depth && imgui_utils::combo_box("Depth Learning Mode",
                                                          depth_learning_modes,
                                                          ARRAY_SIZE(depth_learning_modes),
                                                          depth_learning_mode)) {
                    changed_model    = true;
                    output_depth_dim = depth_learning_mode != DepthLearningMode::Position &&
                                               depth_learning_mode != DepthLearningMode::LocalPosition
                                           ? 1
                                           : 3;
                }

                ImGui::Separator();

                if (output_depth_dim > 0) {
                    changed_model |=
                        imgui_utils::combo_box("Depth Loss", loss_types, ARRAY_SIZE(loss_types), depth_loss);
                }

                if (output_depth_dim) {
                    ImGui::Indent();
                    if (ImGui::Checkbox("(depth) force linear activation", &output_depth_linear_activation))
                        changed_model |= true;
                    changed_model |= ImGui::SliderFloat("Depth Loss Scale", &depth_loss_scale, 0.05f, 20.0f);
                    ImGui::Unindent();
                }

                if (!hide_normal) {
                    ImGui::Separator();

                    bool learn_normal = output_normal_dim == 3;
                    if (ImGui::Checkbox("Learn normal", &learn_normal)) {
                        changed_model |= true;
                        output_normal_dim = learn_normal ? 3 : 0;
                    }

                    if (learn_normal) {
                        ImGui::Indent();
                        changed_model |=
                            ImGui::Checkbox("(normals) force linear activation", &output_normal_linear_activation);
                        changed_model |=
                            imgui_utils::combo_box("Normal Loss", loss_types, ARRAY_SIZE(loss_types), normal_loss);
                        changed_model |= ImGui::SliderFloat("Normal Loss Scale", &normal_loss_scale, 0.05f, 20.0f);
                        ImGui::Unindent();
                    }
                }

                if (!hide_albedo) {
                    ImGui::Separator();

                    bool learn_albedo = output_albedo_dim == 3;
                    if (ImGui::Checkbox("Learn albedo", &learn_albedo)) {
                        changed_model |= true;
                        output_albedo_dim = learn_albedo ? 3 : 0;
                    }

                    if (learn_albedo) {
                        ImGui::Indent();
                        changed_model |=
                            ImGui::Checkbox("(albedo) force linear activation", &output_albedo_linear_activation);
                        changed_model |=
                            imgui_utils::combo_box("Albedo Loss", loss_types, ARRAY_SIZE(loss_types), albedo_loss);
                        changed_model |= ImGui::SliderFloat("Albedo Loss Scale", &albedo_loss_scale, 0.05f, 20.0f);
                        ImGui::Unindent();
                    }
                }

                ImGui::Unindent();
            }

            return changed_model;
        }
    };

}

}