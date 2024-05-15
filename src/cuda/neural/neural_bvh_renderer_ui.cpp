

#include <string>
#include "cuda/neural/neural_bvh_renderer.h"

namespace ntwr {
namespace neural {

    bool NeuralBVHRenderer::render_ui(bool dev_mode)
    {
        bool reset = false;

        ImGui::Begin(name().c_str());

        ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.4f);

        if (imgui_utils::combo_box("Neural BVH Module",
                                   bvh_module_types_display,
                                   ARRAY_SIZE(bvh_module_types_display),
                                   m_bvh_module_type)) {
            switch_neural_module();
            m_reset_learning_data  = true;
            m_reset_inference_data = true;
        }

        imgui_utils::large_separator();

        if (dev_mode) {
            ImGui::SeparatorText("Configuration and I/O");

            if (ImGui::Button("Save image"))
                save_image();
            ImGui::SameLine();
            if (ImGui::Button("Save Performance Metrics"))
                save_performance_metrics();

            ImGui::Checkbox("Save/Load Neural BVH and Weights", &m_load_save_neural_bvh_and_weights);

            ImGui::InputText("Base config name", m_base_config_name, OUTPUT_FILENAME_SIZE);
            ImGui::InputText("Variant config name", m_variant_config_name, OUTPUT_FILENAME_SIZE);
            if (ImGui::Button("Save Configuration")) {
                save_config(m_load_save_neural_bvh_and_weights);
            }
            ImGui::SameLine();
            if (ImGui::Button("Load Configuration")) {
                load_config(m_load_save_neural_bvh_and_weights);
                reset = true;
            }

            if (!m_using_validation_mode) {
                imgui_utils::combo_box(
                    "Saved Validation Mode", validation_modes, ARRAY_SIZE(validation_modes), m_saved_validation_mode);
            }

            imgui_utils::large_separator();
        }

        if (dev_mode) {
            ImGui::Checkbox("Continuous update", &m_continuous_reset);
        }
        reset |= ImGui::Checkbox("EMA accumulation", &m_backend->m_cuda_scene_constants.ema_accumulate);
        ImGui::SameLine();
        reset |= ImGui::DragFloat("weight", &m_backend->m_cuda_scene_constants.ema_weight, 0.001f, 0.f, 1.f);

        if (ImGui::Button("Reset##Inference"))
            reset = true;
        ImGui::SameLine();

        ImGui::Text("Accumulated samples : %d\t", m_accumulated_spp);
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Ignored during optimization!");
        }

        ImGui::InputInt("Max accumulated samples", &m_max_accumulated_spp);

        if (ImGui::Button("Reset##Learning"))
            m_reset_learning_data = true;
        ImGui::SameLine();

        ImGui::Text("Training steps : %d\t", m_training_step);

        if (ImGui::InputScalar("Max training steps", ImGuiDataType_U32, &m_max_training_steps)) {
            m_bvh_split_scheduler.update_stats_and_reset(m_neural_bvh.num_nodes, m_splits_graph, m_max_training_steps);
        }

        ImGui::PopItemWidth();

        size_t last_plot_index =
            glm::clamp(m_loss_graph_samples % m_loss_graph.size() - 1, (size_t)0, m_loss_graph.size() - 1);
        ImGui::PlotLines(
            "Training loss",
            m_loss_graph.data(),
            std::min(m_loss_graph_samples, m_loss_graph.size()),
            (m_loss_graph_samples < m_loss_graph.size()) ? 0 : (m_loss_graph_samples % m_loss_graph.size()),
            glm::detail::format("Last loss=%f", std::exp(m_loss_graph[last_plot_index])).c_str(),
            FLT_MAX,
            FLT_MAX,
            ImVec2(0, 120.f));

        ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.4f);

        int log2_batch_size = (int)std::log2(m_batch_size);
        if (ImGui::InputInt("Training batch size (log2)", &log2_batch_size)) {
            m_reset_learning_data = true;
            reset |= true;
            // BATCH_SIZE_GRANULARITY must be at least 2^8 = 256
            m_batch_size = (int)pow(2, max(8, log2_batch_size));
            // Only case where we actually change the batch size and therefore need to reinitialise the cuda backend
            // data used for learning as well
            reinitialise_bvh_learning_data(m_batch_size);
        }

        if (dev_mode) {
            imgui_utils::large_separator();

            if (imgui_utils::combo_box(
                    "Postprocessing Mode", postprocess_modes, ARRAY_SIZE(postprocess_modes), m_postprocess_mode)) {
                if (!m_backend->m_reference_image_set) {
                    m_postprocess_mode = PostprocessMode::None;
                }
            }
            if (m_postprocess_mode != PostprocessMode::None) {
                ImGui::Indent();
                ImGui::Text("Mean %s : %g", postprocess_modes[m_postprocess_mode], m_mean_error_host);
                ImGui::Unindent();
            }
        }

        reset |= ImGui::Checkbox("Run optimisation", &m_run_optimisation);
        ImGui::Checkbox("Run inference", &m_run_inference);
        if (ImGui::Button("Restart optimisation")) {
            m_run_optimisation     = true;
            m_reset_learning_data  = true;
            m_reset_inference_data = true;
        }

        imgui_utils::large_separator();

        reset |= ImGui::Checkbox("Disable Neural BLAS (Reference)", &m_bvh_const.ignore_neural_blas);

        if (dev_mode) {
            reset |= ImGui::Checkbox("With LoD Switch", &m_bvh_const.path_lod_switch);
            if (m_bvh_const.path_lod_switch) {
                ImGui::Indent();
                reset |= ImGui::InputInt("LoD", &m_bvh_const.lod_switch);
                m_bvh_const.lod_switch = clamp(m_bvh_const.lod_switch, 0, (int)NUM_BVH_LODS - 1);
                ImGui::Unindent();
            }
#ifndef NDEBUG
            reset |= ImGui::DragFloat("Inference mint eps",
                                      &m_bvh_const.inference_mint_epsilon,
                                      m_bvh_const.inference_mint_epsilon / 128.f,
                                      0.f,
                                      1.f,
                                      "%.2e");
            reset |= ImGui::DragFloat("Inference maxt eps",
                                      &m_bvh_const.inference_maxt_epsilon,
                                      m_bvh_const.inference_maxt_epsilon / 128.f,
                                      0.f,
                                      1.f,
                                      "%.2e");
#endif

            if (ImGui::InputInt("Rendered BVH LoD", &m_bvh_const.rendered_lod)) {
                reset |= true;
                m_bvh_const.rendered_lod = clamp(m_bvh_const.rendered_lod, 0, (int)NUM_BVH_LODS - 1);
            }

            imgui_utils::large_separator();
        }

        reset |= ImGui::Checkbox("Show Neural BVH", &m_show_bvh);
        if (dev_mode) {
            ImGui::SameLine();
            reset |= ImGui::Checkbox("Show Node Losses", &m_show_neural_node_losses);
            reset |= ImGui::Checkbox("Split Screen Node Losses", &m_split_screen_neural_node_losses);
            if (m_show_neural_node_losses) {
                ImGui::Indent();
                reset |= ImGui::Checkbox("Loss Heuristic / Sample weighted Loss", &m_bvh_const.show_loss_heuristic);
                reset |= ImGui::Checkbox("Show Tree Cut Depth", &m_show_tree_cut_depth);
                reset |= ImGui::DragFloat(
                    "Loss exposure", &m_bvh_const.loss_exposure, m_bvh_const.loss_exposure / 128.f, 1e-8f, 1.f, "%.2e");
                ImGui::Unindent();
            }
        }

        ImGui::SeparatorText("BVH Config");

        if (ImGui::BeginTabBar("Neural BVH Tabs")) {
            if (ImGui::BeginTabItem("Construction")) {
                if (dev_mode) {
                    if (ImGui::Checkbox("Ellipsoidal BVH", &m_bvh_const.ellipsoidal_bvh)) {
                        m_rebuild_neural_bvh = true;
                    }
                    if (m_bvh_const.ellipsoidal_bvh) {
                        ImGui::Indent();
                        m_reset_learning_data |= ImGui::Checkbox("Rescale to Bbox", &m_bvh_const.rescale_ellipsoid);
                        ImGui::Unindent();
                    }
                }

                if (ImGui::InputInt("BVH min depth", &m_neural_bvh_min_depth)) {
                    if (!m_allow_neural_bvh_hot_swap) {
                        m_rebuild_neural_bvh = true;
                    }
                }

                imgui_utils::large_separator();

                ImGui::Text("Current Max Tree Cut depth : %d", m_neural_tree_cut_max_depth);

                if (ImGui::Checkbox("Enable Tree Cut Learning", &m_tree_cut_learning)) {
                    if (m_tree_cut_learning) {
                        m_rebuild_neural_bvh = true;
                    }
                }

                ImGui::BeginDisabled(!m_tree_cut_learning);
                {
                    if (ImGui::CollapsingHeader("BVH Split Scheduler", nullptr)) {
                        if (dev_mode) {
                            ImGui::PlotLines("BVH Splits",
                                             m_splits_graph.data(),
                                             min(m_splits_graph.size(), m_bvh_split_scheduler.m_last_split_iteration),
                                             0,
                                             glm::detail::format("Total Splits").c_str(),
                                             0.f,
                                             FLT_MAX,
                                             ImVec2(0, 120.f));
                        }

                        ImGui::Text("Num cuts : %d\nTotal splits : %d\nNum BVH nodes : %d\nBalanced BVH depth : %d\n ",
                                    m_bvh_split_scheduler.m_stats.num_cuts,
                                    m_bvh_split_scheduler.m_stats.total_splits,
                                    m_bvh_split_scheduler.m_stats.expected_bvh_num_nodes,
                                    m_bvh_split_scheduler.m_stats.expected_bvh_depth);

                        m_reset_learning_data |=
                            ImGui::InputInt("Last split", &m_bvh_split_scheduler.m_last_split_iteration);
                        m_reset_learning_data |= ImGui::DragFloat("Split scaling",
                                                                  &m_bvh_split_scheduler.m_split_scaling,
                                                                  m_bvh_split_scheduler.m_split_scaling / 128.f,
                                                                  1.f);
                        m_reset_learning_data |=
                            ImGui::InputInt("Starting split number", &m_bvh_split_scheduler.m_start_split_num);
                        m_reset_learning_data |=
                            ImGui::InputInt("Splitting interval", &m_bvh_split_scheduler.m_splitting_interval);
                        m_reset_learning_data |=
                            ImGui::DragFloat("Split interval scaling",
                                             &m_bvh_split_scheduler.m_split_interval_scaling,
                                             m_bvh_split_scheduler.m_split_interval_scaling / 128.f,
                                             1.f);
                    }
                }
                ImGui::EndDisabled();

                if (dev_mode) {
                    ImGui::BeginDisabled(m_tree_cut_learning);
                    {
                        ImGui::Checkbox("Neural BVH Hot Swap", &m_allow_neural_bvh_hot_swap);

                        if (ImGui::InputInt("Num manual BVH splits", &m_current_num_bvh_split)) {
                            m_reset_learning_data = true;
                            reinitialise_tree_cut_learning_data(m_current_num_bvh_split);
                        }

                        if (ImGui::Button("Manual BVH Split")) {
                            split_neural_bvh_at(m_split_neural_node_indices_host);
                            if (!m_allow_neural_bvh_hot_swap) {
                                m_reset_learning_data |= true;
                            }
                        }
                    }
                    ImGui::EndDisabled();
                }

                ImGui::EndTabItem();
            }

            if (dev_mode) {
                if (ImGui::BeginTabItem("LoDs")) {
                    ImGui::Text("Current learned LoD : %d\n", m_bvh_const.current_learned_lod);

                    if (ImGui::Button("Manual LoD assignment")) {
                        assign_lod_to_current_neural_leaf_nodes();
                    }

                    if (ImGui::InputInt("Num LoDs learned", &m_bvh_const.num_lods_learned)) {
                        m_rebuild_neural_bvh |= true;
                        m_bvh_const.num_lods_learned = clamp(m_bvh_const.num_lods_learned, 1, (int)NUM_BVH_LODS);
                    }
                    ImGui::BeginDisabled(!m_tree_cut_learning);

                    m_reset_learning_data |= imgui_utils::combo_box("LoD Decision Mode",
                                                                    lod_decision_types,
                                                                    ARRAY_SIZE(lod_decision_types),
                                                                    m_bvh_split_scheduler.m_lod_decision_type);

                    ImGui::EndDisabled();

                    ImGui::EndTabItem();
                }
            }
            if (dev_mode) {
                if (ImGui::BeginTabItem("Sampling")) {
#ifndef NDEBUG
                    m_reset_learning_data |= ImGui::DragFloat("Learning mint eps",
                                                              &m_bvh_const.learning_mint_epsilon,
                                                              m_bvh_const.learning_mint_epsilon / 128.f,
                                                              0.f,
                                                              1.f,
                                                              "%.2e");
                    m_reset_learning_data |= ImGui::DragFloat("Learning maxt eps",
                                                              &m_bvh_const.learning_maxt_epsilon,
                                                              m_bvh_const.learning_maxt_epsilon / 128.f,
                                                              0.f,
                                                              1.f,
                                                              "%.2e");
#endif
                    m_reset_learning_data |=
                        ImGui::Checkbox("View dependent sampling", &m_bvh_const.view_dependent_sampling);
                    if (m_bvh_const.view_dependent_sampling) {
                        ImGui::Indent();
                        m_reset_learning_data |= ImGui::DragFloat("View dependent ratio",
                                                                  &m_bvh_const.view_dependent_ratio,
                                                                  m_bvh_const.view_dependent_ratio / 128.f,
                                                                  0.f,
                                                                  1.f);
                        ImGui::Unindent();
                    }
                    m_reset_learning_data |=
                        ImGui::Checkbox("Training Node Survival", &m_bvh_const.training_node_survival);
                    if (m_bvh_const.training_node_survival) {
                        ImGui::Indent();
                        m_reset_learning_data |= ImGui::DragFloat("Min survival probability",
                                                                  &m_bvh_const.min_rr_survival_prob,
                                                                  m_bvh_const.min_rr_survival_prob / 128.f,
                                                                  0.f,
                                                                  1.f);
                        m_reset_learning_data |= ImGui::DragFloat("Survival probability scaling",
                                                                  &m_bvh_const.survival_probability_scale,
                                                                  m_bvh_const.survival_probability_scale / 128.f,
                                                                  1.f,
                                                                  1000.f);
                        ImGui::Unindent();
                    }

                    if (ImGui::InputInt("Max BVH traversals (log2)", &m_bvh_const.log2_bvh_max_traversals)) {
                        m_reset_learning_data = true;
                        m_bvh_const.log2_bvh_max_traversals =
                            clamp(m_bvh_const.log2_bvh_max_traversals, 0, log2_batch_size);
                        m_bvh_const.leaf_boundary_sampling_ratio = 0.f;
                    }

                    if (m_bvh_const.log2_bvh_max_traversals > 0) {
                        ImGui::Indent();
                        ImGui::SeparatorText("Multi bounce training");
                        m_reset_learning_data |=
                            ImGui::InputInt("Max Training Bounces", &m_bvh_const.max_training_bounces);

                        if (m_bvh_const.max_training_bounces > 0) {
                            ImGui::Indent();
                            m_reset_learning_data |= ImGui::Checkbox("Sample cosine weighted hemisphere",
                                                                     &m_bvh_const.sample_cosine_weighted);
                            m_reset_learning_data |=
                                ImGui::Checkbox("Backface bounce Training", &m_bvh_const.backface_bounce_training);
                            ImGui::Unindent();
                        }
                        ImGui::Unindent();
                    }

                    if (ImGui::DragFloat("Leaf Boundary Sampling Ratio",
                                         &m_bvh_const.leaf_boundary_sampling_ratio,
                                         0.005f,
                                         0.f,
                                         1.f)) {
                        m_reset_learning_data               = true;
                        m_bvh_const.log2_bvh_max_traversals = 0;
                    }

                    m_reset_learning_data |= ImGui::DragFloat("Sampling bbox scaling",
                                                              &m_bvh_const.sampling_bbox_scaling,
                                                              m_bvh_const.sampling_bbox_scaling / 128.f,
                                                              0.1f,
                                                              2.f);
                    m_reset_learning_data |= ImGui::DragFloat("Fixed node extent dilation",
                                                              &m_bvh_const.fixed_node_dilation,
                                                              m_bvh_const.fixed_node_dilation / 128.f,
                                                              0.f,
                                                              1.f,
                                                              "%.2e");
                    m_reset_learning_data |= ImGui::DragFloat("Node extent dilation",
                                                              &m_bvh_const.node_extent_dilation,
                                                              m_bvh_const.node_extent_dilation / 128.f,
                                                              0.f,
                                                              1.f,
                                                              "%.2e");

                    ImGui::EndTabItem();
                }
            }

            if (ImGui::BeginTabItem("Input Encoding")) {
                if (m_neural_module->ui_encoding(reset, dev_mode)) {
                    m_reset_inference_data = true;
                    m_reset_learning_data  = true;
                }
                ImGui::EndTabItem();
            }

            if (ImGui::BeginTabItem("Stats")) {
                ImGui::SeparatorText("Neural BVH Stats");

                ImGui::Indent();

                uint32_t lod_memory = m_all_bvh_lods_assigned ? 0 : NUM_BVH_LODS - 1;

                ImGui::Text("Memory usage (at LoD %d): is using %.6f MB",
                            lod_memory,
                            m_bvh_stats.n_byte_size(lod_memory) * 0.000001f);
                ImGui::Text(
                    "\t Memory saving %.4f %%",
                    (1.f - float(m_bvh_stats.n_bvh_lod_nodes[lod_memory]) / m_bvh_stats.n_original_bvh_nodes) * 100.f);

                for (int i = (int)NUM_BVH_LODS - 1; i >= 0; i--) {
                    ImGui::Text("%d nodes used for BVH LoD %d", m_bvh_stats.n_bvh_lod_nodes[i], i);
                }

                ImGui::Unindent();

                ImGui::SeparatorText("Original BVH Stats");

                ImGui::Indent();

                ImGui::Text("Memory usage : %.6f MB", m_bvh_stats.n_original_byte_size() * 0.000001f);
                ImGui::Text("Num nodes used : %d", m_bvh_stats.n_original_bvh_nodes);

                ImGui::Unindent();

                ImGui::SeparatorText("Scene Stats");

                ImGui::Text("Num meshes : %ld", m_backend->n_meshes());
                ImGui::Text("Num vertices : %ld", m_backend->n_vertices());
                ImGui::Text("Num textures : %ld", m_backend->n_textures());
                ImGui::EndTabItem();
            }
        }
        ImGui::EndTabBar();
        ImGui::PopItemWidth();

        ImGui::End();

        m_reset_learning_data |= m_neural_module->ui_module(reset, dev_mode);
        m_reset_learning_data |= m_rebuild_neural_bvh;
        m_reset_inference_data |= m_reset_learning_data;
        reset |= m_reset_learning_data;

        return reset;
    }
}
}