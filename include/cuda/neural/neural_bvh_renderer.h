#pragma once

#include "app/render_display.h"
#include "app/renderer.h"
#include "base/accel/bvh.h"
#include "base/camera/camera.h"
#include "base/scene.h"
#include "cuda/accel/bvh2.cuh"
#include "cuda/base/ray.cuh"
#include "cuda/cuda_backend.h"
#include "cuda/utils/cuda_buffer.cuh"
#include "utils/frame_timer.h"

#include "cuda/neural/accel/neural_bvh.cuh"
#include "cuda/neural/accel/neural_bvh2.cuh"

#include "cuda/neural/neural_bvh_renderer_constants.cuh"

#include "cuda/neural/neural_bvh_module.h"
#include "cuda/neural/utils/bvh_split_scheduler.h"

#include <chrono>
#include <memory>
#include <vector>

namespace ntwr {

namespace neural {

    static bool loss_registered                         = false;
    static const char *const bvh_module_types[]         = {"Path", "Prefiltering"};
    static const char *const bvh_module_types_display[] = {"Neural Path Tracing", "Neural Prefiltering"};

    enum BVHModuleType { Path, Prefiltering };

    struct NeuralTreeCutData {
        float loss          = 0.f;
        uint32_t iter_alive = 0;
        uint32_t depth      = 0;
        uint32_t samples    = 0;

        NTWR_DEVICE float compute_loss_heuristic()
        {
            // return samples == 0 ? -FLT_MAX : (loss / iter_alive) / sqrt(float(samples) / iter_alive);
            return samples == 0 ? -FLT_MAX : loss / (iter_alive * sqrt(float(samples) / iter_alive));
        }
    };

    struct NeuralBvhStats {
        // single_bvh2_node_byte_size = 32

        uint32_t n_original_bvh_nodes          = 0;
        uint32_t n_bvh_lod_nodes[NUM_BVH_LODS] = {};

        // uint32_t lod_bvh_depth[NUM_BVH_LODS - 1]   = {}; //TODO
        // uint32_t original_bvh_max_depth   = 0; //TODO

        void reset()
        {
            n_original_bvh_nodes = 0;
            for (size_t bvh_lod = 0; bvh_lod < NUM_BVH_LODS; bvh_lod++)
                n_bvh_lod_nodes[bvh_lod] = 0;
        }

        uint32_t n_original_byte_size()
        {
            return n_original_bvh_nodes * 32;
        }

        uint32_t n_byte_size(uint32_t bvh_lod)
        {
            return n_bvh_lod_nodes[bvh_lod] * 32;
        }
    };

    static const char *const validation_modes[] = {"Ignore", "Training", "Inference", "Full"};

    enum ValidationMode { Ignore, Training, Inference, Full };

    static const char *const postprocess_modes[] = {
        "None", "Filmic", "Reference", "AbsError", "RelAbsError", "MSEError", "RelMSEError", "FLIP"};

    enum PostprocessMode { None, Filmic, Reference, AbsError, RelAbsError, MSEError, RelMSEError, FLIP };

    class NeuralBVHRenderer : public Renderer {
    public:
        NeuralBVHRenderer(std::shared_ptr<CudaBackend> cuda_backend);

        ~NeuralBVHRenderer();

        virtual std::string name() override
        {
            return "Neural BVH Renderer";
        }

        virtual RenderBackend *render_backend() override
        {
            return (RenderBackend *)m_backend.get();
        }

        /*! render any imgui component and update needed states, if renderer needs restart
         * returns true  */
        virtual bool render_ui(bool dev_mode) override;

        /*! render one frame */
        virtual void render(bool reset_accumulation) override;

        /*! exectures on window resize. No need to handle the render backend,
         *  it is handled by the render app already*/
        virtual void on_resize(const glm::uvec2 &new_size) override;

        /*! Returns the number of accumulated samples */
        virtual uint32_t accumulated_samples() override;

        /*! Update any required data when the scene is updated/changed */
        virtual void scene_updated() override;

        void toggle_node_loss_display()
        {
            m_show_neural_node_losses = !m_show_neural_node_losses;
        }
        void toggle_split_screen_node_loss_display()
        {
            m_split_screen_neural_node_losses = !m_split_screen_neural_node_losses;
        }
        void restart_optimisation()
        {
            m_run_optimisation     = true;
            m_reset_learning_data  = true;
            m_reset_inference_data = true;
        }

        void load_config_from_json(nlohmann::json full_config,
                                   const std::string base_config_name,
                                   bool load_bvh_and_network,
                                   bool same_base_config_inference = false);

    private:
        CudaSceneData learning_scene_data();

        void reset_learning_data();

        void reinitialise_tree_cut_learning_data(uint32_t num_bvh_splits);

        void reinitialise_bvh_learning_data(const uint32_t batch_size);
        void switch_neural_module();

        void setup_constants();

        void reinitialise_data(const glm::uvec2 &new_size);

        void update_loss_graph(float loss);

        float compute_and_display_flip_error();

        ///// Configuration Related /////

        void load_next_config();

        void process_validation_config_end();

        void save_performance_metrics(bool save_learning_perfs = true, bool save_inference_perfs = true);
        void save_image();

        void save_config(bool save_bvh_and_network);

        void load_config(bool load_bvh_and_network);

        void save_network_model(const std::string &filename);
        void load_network_model(const std::string &filename);

        ///// Acceleration Related /////

        void reinitialise_neural_bvh_learning_states();

        void build_and_upload_neural_bvh();
        void split_neural_bvh_at(const std::vector<uint32_t> &split_neural_node_indices);

        void assign_lod_to_current_neural_leaf_nodes();

        void save_neural_bvh(const std::string &output_filename);

        void load_neural_bvh(const std::string &output_filename);

        bool m_load_save_neural_bvh_and_weights = false;

        /***** Configurable Members **********/

        std::unique_ptr<NeuralBVHModule> m_neural_module;
        BVHModuleType m_bvh_module_type = BVHModuleType::Path;

        // Various constants for the network and optimization
        uint32_t m_batch_size         = 512 * 512;
        uint32_t m_max_training_steps = 10000;
        uint32_t m_training_step      = 0;

        char m_base_config_name[OUTPUT_FILENAME_SIZE]    = "default";
        char m_variant_config_name[OUTPUT_FILENAME_SIZE] = "base";

        ValidationMode m_saved_validation_mode = ValidationMode::Full;

        bool m_run_optimisation = false;
        bool m_run_inference    = false;
        bool m_continuous_reset = false;

        int m_neural_bvh_min_depth = 0;

        NeuralBVHRendererConstants m_bvh_const;

        //////// Neural BVH split related data //////

        bool m_show_bvh                        = false;
        bool m_show_neural_node_losses         = false;
        bool m_show_tree_cut_depth             = false;
        bool m_split_screen_neural_node_losses = false;

        bool m_tree_cut_learning = true;

        BvhSplitScheduler m_bvh_split_scheduler = BvhSplitScheduler{};

        /******************************/

        FrameTimer m_learning_timer;
        FrameTimer m_inference_timer;

        ValidationMode m_validation_mode = ValidationMode::Ignore;
        bool m_using_validation_mode     = false;

        bool m_reset_learning_data  = false;
        bool m_reset_inference_data = false;
        bool m_rebuild_neural_bvh   = false;

        //// Optimization temporaries
        size_t m_loss_graph_samples     = 0;
        std::vector<float> m_loss_graph = std::vector<float>(m_max_training_steps, 0.0f);
        float m_tmp_loss                = 0;
        uint32_t m_tmp_loss_counter     = 0;
        std::chrono::steady_clock::time_point m_training_begin_time;

        CUDABuffer<NeuralTreeCutData> m_neural_tree_cut_data;

        CUDABuffer<NeuralBVHNode> m_neural_bvh_nodes;

        uint32_t m_neural_tree_cut_max_depth;

        std::vector<uint32_t> m_neural_node_map;
        uint32_t m_neural_nodes_alloc_count = 0;
        uint32_t m_neural_nodes_used        = 0;

        NeuralBVH m_neural_bvh;
        NeuralBvhStats m_bvh_stats;
        Bbox3f m_neural_scene_bbox;

        std::shared_ptr<CudaBackend> m_backend;

        PostprocessMode m_postprocess_mode = PostprocessMode::None;
        float *m_mean_error                = nullptr;
        float m_mean_error_host            = 0.f;

        //////// Neural BVH split related data //////

        bool m_all_bvh_lods_assigned     = false;
        bool m_allow_neural_bvh_hot_swap = false;

        std::vector<float> m_splits_graph = std::vector<float>(m_max_training_steps, 0.0f);

        int m_current_num_bvh_split = m_bvh_split_scheduler.m_start_split_num;
        float *m_max_node_losses    = nullptr;
        float m_max_node_loss_host  = 0.f;

        uint32_t *m_split_neural_node_indices                  = nullptr;
        std::vector<uint32_t> m_split_neural_node_indices_host = {};

        // Stores the neural node index sampled during training for every sample
        CUDABuffer<uint32_t> m_sampled_node_indices;

        ////////////////////////////////////////

        // The BVH module is a good friend of the BVH renderer
        friend class NeuralBVHModule;
    };

}
}