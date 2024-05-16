#pragma once

#include "cuda/neural/neural_bvh_module.h"
#include "glm_pch.h"

constexpr uint8_t NEURAL_TRAVERSAL_HIT = uint8_t(1);
constexpr uint8_t NORMAL_TRAVERSAL_HIT = uint8_t(2);

namespace ntwr {

namespace neural::path {

    // Important to set the pointers to nullptr to consistently malloc and free
    struct NeuralLearningResult {
        float *leaf_mint = nullptr;
        float *leaf_maxt = nullptr;
    };

    struct NeuralTraversalResult {
        // Default constructor to make compiler happy with unique_ptr to types containing NeuralTraversalData
        NeuralTraversalResult() {}
        // Initialises a NeuralTraversalData structure from device pointers
        NeuralTraversalResult(uint8_t *valid_hit, float *t_neural, glm::vec3 *normal, glm::vec3 *albedo)
        {
            this->valid_hit = valid_hit;
            this->t_neural  = t_neural;
            this->normal    = normal;
            this->albedo    = albedo;
        }
        uint8_t *valid_hit = nullptr;  // Binary visibility result
        float *t_neural    = nullptr;
        glm::vec3 *normal  = nullptr;
        glm::vec3 *albedo  = nullptr;

        NTWR_DEVICE void set(uint32_t index, const NeuralTraversalResult &other, uint32_t other_index)
        {
            valid_hit[index] = other.valid_hit[other_index];
            t_neural[index]  = other.t_neural[other_index];
            normal[index]    = other.normal[other_index];
            albedo[index]    = other.albedo[other_index];
        }
    };

    // The result of the network evaluation + wathever is needed from the traversal data later (all but the
    // stack basically)
    struct HybridIntersectionResult {
        // Default constructor to make compiler happy with unique_ptr to types containing NeuralTraversalData
        HybridIntersectionResult() {}
        // Initialises a NeuralTraversalData structure from device pointers
        HybridIntersectionResult(uint8_t *valid_hit,
                                 float *t_neural,
                                 glm::vec3 *normal_or_uv,
                                 glm::vec3 *albedo_or_mesh_prim_id,
                                 vec3_soa ray_o,
                                 vec3_soa ray_d,
                                 uint8_t *ray_depth,
                                 glm::vec3 *illumination,
                                 glm::vec4 *throughput_rng,
                                 uint32_t *pixel_coord,
                                 int *tlas_stack_size,
                                 uint32_t *tlas_stack)
        {
            // Traversal data
            this->ray_o           = ray_o;
            this->ray_d           = ray_d;
            this->ray_depth       = ray_depth;
            this->pixel_coord     = pixel_coord;
            this->illumination    = illumination;
            this->throughput_rng  = throughput_rng;
            this->tlas_stack_size = tlas_stack_size;
            this->tlas_stack      = tlas_stack;
            // Hybrid (NeuralTraversalResult / NormalQueryResult)
            this->valid_hit              = valid_hit;
            this->t                      = t_neural;
            this->normal_or_uv           = normal_or_uv;
            this->albedo_or_mesh_prim_id = albedo_or_mesh_prim_id;
        }

        // Traversal data
        vec3_soa ray_o;
        vec3_soa ray_d;
        uint8_t *ray_depth        = nullptr;
        uint32_t *pixel_coord     = nullptr;  // pixel coordinate (without the active flag from the traversal)
        glm::vec3 *illumination   = nullptr;
        glm::vec4 *throughput_rng = nullptr;
        int *tlas_stack_size      = nullptr;
        uint32_t *tlas_stack      = nullptr;

        // Hybrid (NeuralTraversalResult / NormalTraversalResult)
        uint8_t *valid_hit                = nullptr;
        float *t                          = nullptr;
        glm::vec3 *normal_or_uv           = nullptr;
        glm::vec3 *albedo_or_mesh_prim_id = nullptr;

        NTWR_DEVICE Ray3f get_ray(uint32_t idx) const
        {
            return Ray3f(ray_o.get(idx), ray_d.get(idx));
        }

        NTWR_DEVICE void set_or_update_from_neural_traversal(uint32_t index,
                                                             const NeuralTraversalResult &result,
                                                             const NeuralTraversalData &traversal_data,
                                                             uint32_t other_index)
        {
            // Traversal data
            ray_o.set(index, traversal_data.ray_o.get(other_index));
            ray_d.set(index, traversal_data.ray_d.get(other_index));
            ray_depth[index]      = traversal_data.ray_depth[other_index];
            pixel_coord[index]    = traversal_data.pixel_coord(other_index);
            illumination[index]   = traversal_data.illumination[other_index];
            throughput_rng[index] = traversal_data.throughput_rng[other_index];

            if (traversal_data.tlas_stack != nullptr) {
                assert(tlas_stack != nullptr);
                uint32_t src_index, dst_index;
                for (int stack_idx = 0; stack_idx < traversal_data.tlas_stack_size[other_index]; stack_idx++) {
                    src_index             = other_index + stack_idx * fb_size_flat;
                    dst_index             = index + stack_idx * fb_size_flat;
                    tlas_stack[dst_index] = traversal_data.tlas_stack[src_index];
                }
                tlas_stack_size[index] = traversal_data.tlas_stack_size[other_index];
            }

            //  No need for conditional udpate as result.t_neural will only be updated if a new closer hit is found
            valid_hit[index]              = result.valid_hit[other_index];
            t[index]                      = result.t_neural[other_index];
            normal_or_uv[index]           = result.normal[other_index];
            albedo_or_mesh_prim_id[index] = result.albedo[other_index];
        }

        NTWR_DEVICE void set_or_update_with_preliminary_its_data(uint32_t index,
                                                                 const NormalTraversalData &traversal_data,
                                                                 uint32_t other_index,
                                                                 const PreliminaryItsData &pi)
        {
            ray_o.set(index, traversal_data.ray_o.get(other_index));
            ray_d.set(index, traversal_data.ray_d.get(other_index));
            ray_depth[index]      = traversal_data.ray_depth[other_index];
            pixel_coord[index]    = traversal_data.pixel_coord(other_index);
            illumination[index]   = traversal_data.illumination[other_index];
            throughput_rng[index] = traversal_data.throughput_rng[other_index];

            assert(tlas_stack);
            assert(traversal_data.tlas_stack_size);
            uint32_t src_index, dst_index;
            for (int stack_idx = 0; stack_idx < traversal_data.tlas_stack_size[other_index]; stack_idx++) {
                src_index             = other_index + stack_idx * fb_size_flat;
                dst_index             = index + stack_idx * fb_size_flat;
                tlas_stack[dst_index] = traversal_data.tlas_stack[src_index];
            }
            tlas_stack_size[index] = traversal_data.tlas_stack_size[other_index];

            // Conditionally update HybridIntersectionResult from PreliminaryItsData
            if (pi.t < traversal_data.last_tlas_t[other_index]) {
                assert(traversal_data.current_blas_idx[other_index] == pi.mesh_idx);
                assert(pi.is_valid());
                t[index]                        = pi.t;
                normal_or_uv[index].x           = pi.u;
                normal_or_uv[index].y           = pi.v;
                albedo_or_mesh_prim_id[index].x = glm::uintBitsToFloat(pi.mesh_idx);
                albedo_or_mesh_prim_id[index].y = glm::uintBitsToFloat(pi.prim_idx);
                valid_hit[index]                = NORMAL_TRAVERSAL_HIT;
            } else {
                valid_hit[index]              = traversal_data.last_valid_hit[other_index];
                t[index]                      = traversal_data.last_tlas_t[other_index];
                albedo_or_mesh_prim_id[index] = traversal_data.albedo_or_mesh_prim_id[other_index];
                normal_or_uv[index]           = traversal_data.normal_or_uv[other_index];
            }
        }

        NTWR_DEVICE PreliminaryItsData get_preliminary_its_data(uint32_t idx) const
        {
            assert(valid_hit[idx] == NORMAL_TRAVERSAL_HIT);
            PreliminaryItsData pi;
            pi.t        = t[idx];
            pi.u        = normal_or_uv[idx].x;
            pi.v        = normal_or_uv[idx].y;
            pi.mesh_idx = glm::floatBitsToUint(albedo_or_mesh_prim_id[idx].x);
            pi.prim_idx = glm::floatBitsToUint(albedo_or_mesh_prim_id[idx].y);
            return pi;
        }

        NTWR_DEVICE glm::vec3 get_albedo(uint32_t idx) const
        {
            assert(valid_hit[idx] == NEURAL_TRAVERSAL_HIT);
            return albedo_or_mesh_prim_id[idx];
        }

        NTWR_DEVICE glm::vec3 get_normal(uint32_t idx) const
        {
            assert(valid_hit[idx] == NEURAL_TRAVERSAL_HIT);
            return normal_or_uv[idx];
        }

        NTWR_DEVICE glm::vec2 get_uv(uint32_t idx) const
        {
            assert(valid_hit[idx] == NEURAL_TRAVERSAL_HIT);
            return glm::vec2(albedo_or_mesh_prim_id[idx].x, albedo_or_mesh_prim_id[idx].y);
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

    static const char *const output_modes[] = {"NeuralPathTracing"};

    enum OutputMode : uint8_t { NeuralPathTracing };

    // Visibility Network constants that are kept also uploaded to the GPU
    struct alignas(8) NeuralPathModuleConstants {
        OutputMode output_mode = OutputMode::NeuralPathTracing;

        float normal_epsilon_offset = 1e-4f;
        bool apply_russian_roulette = true;
        int rr_start_depth          = 2;
        float neural_roughness      = 1.f;
        float neural_specular       = 0.f;
        float neural_metallic       = 0.f;

        inline NTWR_HOST nlohmann::json to_json()
        {
            return nlohmann::ordered_json{
                {"output_mode", output_modes[output_mode]},
                {"normal_epsilon_offset", normal_epsilon_offset},
                {"apply_russian_roulette", apply_russian_roulette},
                {"rr_start_depth", rr_start_depth},
                {"neural_roughness", neural_roughness},
                {"neural_specular", neural_specular},
                {"neural_metallic", neural_metallic},
            };
        }

        inline NTWR_HOST void from_json(nlohmann::json json_config)
        {
            NeuralPathModuleConstants default_config;
            output_mode = (OutputMode)imgui_utils::find_index(
                json_config.value("output_mode", output_modes[default_config.output_mode]).c_str(),
                output_modes,
                ARRAY_SIZE(output_modes));
            normal_epsilon_offset  = json_config.value("normal_epsilon_offset", default_config.normal_epsilon_offset);
            apply_russian_roulette = json_config.value("apply_russian_roulette", default_config.apply_russian_roulette);
            rr_start_depth         = json_config.value("rr_start_depth", default_config.rr_start_depth);
            neural_roughness       = json_config.value("neural_roughness", default_config.neural_roughness);
            neural_specular        = json_config.value("neural_specular", default_config.neural_specular);
            neural_metallic        = json_config.value("neural_metallic", default_config.neural_metallic);
        }
    };

    struct NeuralPathModule : NeuralBVHModule {
        NeuralPathModule(uint32_t batch_size, std::shared_ptr<CudaBackend>);
        ~NeuralPathModule();

        virtual std::string name()
        {
            return "Path Tracing Module";
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

        virtual void split_screen_neural_render_node_loss_module(
            uint32_t accumulated_spp,
            uint32_t sample_offset,
            const NeuralBVH &neural_bvh,
            const CUDABuffer<NeuralTreeCutData> &neural_tree_cut_data,
            float max_node_loss_host,
            bool show_tree_cut_depth,
            uint32_t neural_tree_cut_max_depth) override;

    private:
        void initialize_learning_data(uint32_t batch_size);
        void destroy_learning_data();

        void neural_path_tracing(uint32_t accumulated_spp, uint32_t sample_offset, const NeuralBVH &neural_bvh);
        void neural_bvh_traversal(uint32_t accumulated_spp,
                                  const NeuralBVH &neural_bvh,
                                  uint32_t n_active_rays,
                                  uint32_t *its_counter = nullptr);

        void split_screen_neural_path_tracing(uint32_t accumulated_spp,
                                              uint32_t sample_offset,
                                              const NeuralBVH &neural_bvh,
                                              const CUDABuffer<NeuralTreeCutData> &neural_tree_cut_data,
                                              float max_node_loss_host,
                                              bool show_tree_cut_depth,
                                              uint32_t neural_tree_cut_max_depth);

        CudaSceneData scene_data();

        nlohmann::json network_loss_config;

        NeuralPathModuleConstants m_path_module_const;
        PathTracingOutputConstants m_pt_target_const;

        int m_max_depth            = 2;
        int m_max_tlas_traversal   = 1000;
        float m_total_inference_ms = 0.f;

        uint32_t *m_active_normal_rays_counter;
        uint32_t *m_active_neural_rays_counter;

        uint32_t *m_hybrid_its_counter;
        uint32_t *m_normal_its_counter;

        NeuralLearningResult m_neural_learning_results;

        // For compaction we need two

        std::unique_ptr<NeuralInferenceOutputData<NeuralTraversalResult>> m_inference_output_data;
        NeuralWavefrontData<NeuralTraversalResult> m_wavefront_neural_data[2];

        NormalTraversalData m_normal_traversal_data;

        HybridIntersectionResult m_hybrid_intersection_data;

        std::shared_ptr<CudaBackend> m_cuda_backend;

        // The Neural BVH renderer is a good friend of the BVH module
        friend class ntwr::neural::NeuralBVHRenderer;
    };
}
}