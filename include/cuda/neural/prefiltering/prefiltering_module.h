#pragma once

#include "cuda/neural/neural_bvh_module.h" 
#include "glm_pch.h"

namespace ntwr {

namespace neural::prefiltering {

    // Important to set the pointers to nullptr to consistently malloc and free
    struct NeuralLearningResult {
        float *leaf_mint = nullptr;
        float *leaf_maxt = nullptr;
    };

    struct NeuralTraversalResult {
        // Default constructor to make compiler happy with unique_ptr to types containing NeuralTraversalData
        NeuralTraversalResult() {}
        // Initialises a NeuralTraversalData structure from device pointers
        NeuralTraversalResult(uint8_t *valid_hit, float *t_neural)
        {
            this->valid_hit = valid_hit;
            this->t_neural  = t_neural;
        }
        uint8_t *valid_hit;  // Binary visibility result
        float *t_neural;

        NTWR_DEVICE void set(uint32_t index, const NeuralTraversalResult &other, uint32_t other_index)
        {
            valid_hit[index] = other.valid_hit[other_index];
            t_neural[index]  = other.t_neural[other_index];
        }
    };

    // The result of the network evaluation + wathever is needed from the traversal data later (all but the
    // stack basically)
    struct HybridIntersectionResult : public NeuralTraversalResult {
        // Default constructor to make compiler happy with unique_ptr to types containing NeuralTraversalData
        HybridIntersectionResult() {}
        // Initialises a NeuralTraversalData structure from device pointers
        HybridIntersectionResult(uint8_t *valid_hit,
                                 float *t_neural,
                                 vec3_soa ray_o,
                                 vec3_soa ray_d,
                                 uint8_t *ray_depth,
                                 glm::vec3 *illumination,
                                 glm::vec4 *throughput_rng,
                                 uint32_t *pixel_coord)
        {
            this->valid_hit      = valid_hit;
            this->t_neural       = t_neural;
            this->ray_o          = ray_o;
            this->ray_d          = ray_d;
            this->ray_depth      = ray_depth;
            this->pixel_coord    = pixel_coord;
            this->illumination   = illumination;
            this->throughput_rng = throughput_rng;
        }

        vec3_soa ray_o;         //
        vec3_soa ray_d;         //
        uint8_t *ray_depth;     //
        uint32_t *pixel_coord;  // pixel coordinate (without the active flag from the traversal)
        glm::vec3 *illumination   = nullptr;
        glm::vec4 *throughput_rng = nullptr;

        NTWR_DEVICE void set(uint32_t index,
                             const NeuralTraversalResult &result,
                             const NeuralTraversalData &traversal_data,
                             uint32_t other_index)
        {
            valid_hit[index] = result.valid_hit[other_index];
            t_neural[index]  = result.t_neural[other_index];
            ray_o.set(index, traversal_data.ray_o.get(other_index));
            ray_d.set(index, traversal_data.ray_d.get(other_index));
            ray_depth[index]      = traversal_data.ray_depth[other_index];
            pixel_coord[index]    = traversal_data.pixel_coord(other_index);
            illumination[index]   = traversal_data.illumination[other_index];
            throughput_rng[index] = traversal_data.throughput_rng[other_index];
        }

        NTWR_DEVICE Ray3f get_ray(uint32_t idx) const
        {
            return Ray3f(ray_o.get(idx), ray_d.get(idx));
        }

        NTWR_DEVICE glm::vec3 get_throughput(uint32_t idx) const
        {
            return glm::vec3(throughput_rng[idx]);
        }

        NTWR_DEVICE LCGRand get_rng(uint32_t idx) const
        {
            return LCGRand{glm::floatBitsToUint(throughput_rng[idx].w)};
        }
    };

    static const char *const output_modes[] = {"NeuralPrefiltering", "AppearanceNetworkDebug"};

    enum OutputMode : uint8_t { NeuralPrefiltering, AppearanceNetworkDebug };

    static const int lod_values[] = {0, 1, 2, 3, 4, 5, 6};

    // Visibility Network constants that are kept also uploaded to the GPU
    struct alignas(8) NeuralPrefilteringModuleConstants {
        OutputMode output_mode = OutputMode::NeuralPrefiltering;

        // Prefiltering related
        uint32_t prefiltering_input_dims  = 9;  // (voxel_center_position, wo (towards camera), wi)
        uint32_t prefiltering_output_dims = 3;  // RGB throughput
        uint32_t lod_level                = 0;
        Bbox3f voxel_grid_bbox            = Bbox3f();
        uint32_t voxel_grid_max_res       = 512;
        float voxel_size_lod_0            = 0.f;
        bool prefiltering_log_learnt      = false;
        bool apply_russian_roulette       = true;
        int rr_start_depth                = 0;
        bool hg_sampling                  = false;
        float hg_g                        = 0.f;
        bool apply_traversal_rr           = false;
        float traversal_rr                = 0.98f;

        inline NTWR_HOST nlohmann::json to_json()
        {
            return nlohmann::ordered_json{
                {"output_mode", output_modes[output_mode]},
                {"prefiltering_input_dims", prefiltering_input_dims},
                {"prefiltering_output_dims", prefiltering_output_dims},
                {"lod_level", lod_level},
                // {"voxel_grid_bbox", voxel_grid_bbox}, // Set during initialisation always
                // {"voxel_grid_max_res", voxel_grid_max_res}, // Set during initialisation always
                // {"voxel_size_lod_0", voxel_size_lod_0}, // Set during initialisation always
                {"prefiltering_log_learnt", prefiltering_log_learnt},
                {"apply_russian_roulette", apply_russian_roulette},
                {"rr_start_depth", rr_start_depth},
                {"hg_sampling", hg_sampling},
                {"hg_g", hg_g},
                {"apply_traversal_rr", apply_traversal_rr},
                {"traversal_rr", traversal_rr},
            };
        }

        inline NTWR_HOST void from_json(nlohmann::json json_config)
        {
            NeuralPrefilteringModuleConstants default_config;
            output_mode = (OutputMode)imgui_utils::find_index(
                json_config.value("output_mode", output_modes[default_config.output_mode]).c_str(),
                output_modes,
                ARRAY_SIZE(output_modes));
            prefiltering_input_dims =
                json_config.value("prefiltering_input_dims", default_config.prefiltering_input_dims);
            prefiltering_output_dims =
                json_config.value("prefiltering_output_dims", default_config.prefiltering_output_dims);
            lod_level = json_config.value("lod_level", default_config.lod_level);
            // Set during initialisation always
            // voxel_grid_bbox    = json_config.value("voxel_grid_bbox", default_config.voxel_grid_bbox);
            // voxel_grid_max_res = json_config.value("voxel_grid_max_res", default_config.voxel_grid_max_res);
            // voxel_size_lod_0 = json_config.value("voxel_size_lod_0", default_config.voxel_size_lod_0);
            prefiltering_log_learnt =
                json_config.value("prefiltering_log_learnt", default_config.prefiltering_log_learnt);
            apply_russian_roulette = json_config.value("apply_russian_roulette", default_config.apply_russian_roulette);
            rr_start_depth         = json_config.value("rr_start_depth", default_config.rr_start_depth);
            hg_sampling            = json_config.value("hg_sampling", default_config.hg_sampling);
            hg_g                   = json_config.value("hg_g", default_config.hg_g);
            apply_traversal_rr     = json_config.value("apply_traversal_rr", default_config.apply_traversal_rr);
            traversal_rr           = json_config.value("traversal_rr", default_config.traversal_rr);
        }
    };

    struct NeuralPrefilteringModule : NeuralBVHModule {
        NeuralPrefilteringModule(uint32_t batch_size, std::shared_ptr<CudaBackend>);
        ~NeuralPrefilteringModule();

        virtual std::string name()
        {
            return "Neural Prefiltering Module";
        }

        virtual void reinitialise_inference_data_module(const glm::uvec2 &fb_size) override;
        virtual void reinitialise_learning_data(uint32_t batch_size) override;

        virtual void update_neural_model_module() override;
        virtual void setup_neural_module_constants_module() override;
        virtual uint32_t network_output_dims_module() override;
        virtual nlohmann::json network_loss_module() override;

        virtual bool ui_module(bool &reset_renderer, bool dev_mode) override;

        virtual float *optim_leaf_mints() override;
        virtual float *optim_leaf_maxts() override;
        virtual void generate_and_fill_bvh_output_reference(
            uint32_t training_step,
            uint32_t batch_size,
            uint32_t log2_bvh_max_traversals,
            const NeuralBVH &neural_bvh,
            const CudaSceneData &scene_data,
            const CUDABuffer<uint32_t> &sampled_node_indices,
            float current_max_loss,
            const CUDABuffer<NeuralTreeCutData> &neural_tree_cut_data) override;

        virtual void neural_render_module(uint32_t accumulated_spp,
                                          uint32_t sample_offset,
                                          const NeuralBVH &neural_bvh) override;

    private:
        void initialize_learning_data(uint32_t batch_size);
        void destroy_learning_data();

        void initialise_appearance_network_with_pretrained_weights(const std::filesystem::path &scene_filename);

        void neural_prefiltering(uint32_t accumulated_spp, uint32_t sample_offset, const NeuralBVH &neural_bvh);
        void neural_bvh_traversal(uint32_t accumulated_spp,
                                  const NeuralBVH &neural_bvh,
                                  uint32_t n_active_rays,
                                  uint32_t *its_counter);

        void appearance_network_debug(uint32_t accumulated_spp, uint32_t sample_offset, const NeuralBVH &neural_bvh);

        CudaSceneData scene_data();

        nlohmann::json network_loss_config;

        NeuralPrefilteringModuleConstants m_prefiltering_module_const;
        PathTracingOutputConstants m_pt_target_const;

        int m_max_depth = 1000;

        uint32_t *m_active_rays_counter;
        uint32_t *m_its_counter;

        NeuralLearningResult m_neural_learning_results;

        // For compaction we need two

        std::unique_ptr<NeuralInferenceOutputData<NeuralTraversalResult>> m_inference_output_data;
        float *m_appearance_input_data;   // lives as long as m_inference_output_data's scratch space
        float *m_appearance_output_data;  // lives as long as m_inference_output_data's scratch space

        std::unique_ptr<NeuralNetworkModel> m_appearance_model;

        NeuralWavefrontData<NeuralTraversalResult> m_wavefront_neural_data[2];
        HybridIntersectionResult m_hybrid_intersection_data;

        std::shared_ptr<CudaBackend> m_cuda_backend;

        // The Neural BVH renderer is a good friend of the BVH module
        friend class NeuralBVHRenderer;
    };

}
}