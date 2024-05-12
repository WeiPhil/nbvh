#include "cuda/neural/accel/neural_bvh.cuh"
#include "cuda/neural/neural_bvh_module_constants.cuh"
#include "cuda/neural/path/path_constants.cuh"
#include "cuda/neural/path/path_module.h"
#include "utils/logger.h"

namespace ntwr {

namespace neural::path {

    NeuralPathModule::NeuralPathModule(uint32_t batch_size, std::shared_ptr<CudaBackend> cuda_backend)
        : m_cuda_backend(cuda_backend)
    {
        m_model                 = std::make_unique<NeuralNetworkModel>();
        m_inference_input_data  = std::make_unique<NeuralInferenceInputData>();
        m_inference_output_data = std::make_unique<NeuralInferenceOutputData<NeuralTraversalResult>>();

        if (m_path_module_const.output_mode == OutputMode::NeuralPathTracing) {
            m_pt_target_const.output_normal_dim = 3;
            m_pt_target_const.output_albedo_dim = 3;
        } else {
            m_pt_target_const.output_normal_dim = 0;
            m_pt_target_const.output_albedo_dim = 0;
        }

        initialize_learning_data(batch_size);
    }

    NeuralPathModule::~NeuralPathModule()
    {
        destroy_learning_data();
    }

    CudaSceneData NeuralPathModule::scene_data()
    {
        CudaSceneData data;
        data.tlas      = m_cuda_backend->m_cuda_tlas;
        data.materials = m_cuda_backend->m_cuda_material_buffer.d_pointer();
        data.textures  = m_cuda_backend->m_cuda_textures_buffer.d_pointer();

        return data;
    }

    void NeuralPathModule::initialize_learning_data(uint32_t batch_size)
    {
        CUDA_CHECK(cudaMalloc(&m_active_normal_rays_counter, sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&m_active_neural_rays_counter, sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&m_hybrid_its_counter, sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&m_normal_its_counter, sizeof(uint32_t)));

        // Alloc with right size
        CUDA_CHECK(cudaMalloc(&m_neural_learning_results.leaf_mint, sizeof(float) * batch_size));
        CUDA_CHECK(cudaMalloc(&m_neural_learning_results.leaf_maxt, sizeof(float) * batch_size));
    }

    void NeuralPathModule::destroy_learning_data()
    {
        CUDA_CHECK(cudaFree(m_active_normal_rays_counter));
        CUDA_CHECK(cudaFree(m_active_neural_rays_counter));
        CUDA_CHECK(cudaFree(m_hybrid_its_counter));
        CUDA_CHECK(cudaFree(m_normal_its_counter));

        // Free if previously allocated
        if (m_neural_learning_results.leaf_mint)
            CUDA_CHECK(cudaFree(m_neural_learning_results.leaf_mint));
        if (m_neural_learning_results.leaf_maxt)
            CUDA_CHECK(cudaFree(m_neural_learning_results.leaf_maxt));
    }

    void NeuralPathModule::reinitialise_learning_data(uint32_t batch_size)
    {
        destroy_learning_data();
        initialize_learning_data(batch_size);
    }

    void NeuralPathModule::reinitialise_inference_data_module(const glm::uvec2 &fb_size)
    {
        cudaStream_t default_stream = 0;

        uint32_t n_padded_ray_payloads = tcnn::next_multiple(fb_size.x * fb_size.y, 4U);
        uint32_t n_padded_elements     = tcnn::next_multiple(fb_size.x * fb_size.y, tcnn::batch_size_granularity);

        auto query_scratch =
            tcnn::allocate_workspace_and_distribute<uint8_t,    // valid_hit[0]
                                                    float,      // t_neural[0]
                                                    glm::vec3,  // normal[0]
                                                    glm::vec3,  // albedo[0]
                                                    //  m_wavefront_neural_data[0].results
                                                    uint8_t,    // valid_hit[1]
                                                    float,      // t_neural[1]
                                                    glm::vec3,  // normal[1]
                                                    glm::vec3,  // albedo[1]
                                                    //  m_wavefront_neural_data[1].results
                                                    uint8_t,    // valid_hit
                                                    float,      // t_neural
                                                    glm::vec3,  // normal_or_uv
                                                    glm::vec3,  // albedo_or_mesh_prim_id
                                                    float,      //
                                                    float,      //
                                                    float,      // ray_o
                                                    float,      //
                                                    float,      //
                                                    float,      // ray_d
                                                    uint8_t,    // ray_depth
                                                    glm::vec3,  // illumination
                                                    glm::vec4,  // throughput_rng
                                                    uint32_t,   // pixel_coord
                                                    int,        // tlas_stack_size
                                                    uint32_t,   // tlas_stack
                                                    ///  m_hybrid_intersection_data
                                                    glm::vec3,  // illumination[0]
                                                    glm::vec4,  // throughput_rng[0]
                                                    int,        // tlas_stack_size[0]
                                                    uint32_t,   // tlas_stack[0]
                                                    // neural_traversal_data[0].illumination / throughput_rng / tlas
                                                    glm::vec3,  // illumination[1]
                                                    glm::vec4,  // throughput_rng[1]
                                                    int,        // tlas_stack_size[1]
                                                    uint32_t,   // tlas_stack[1]
                                                    // neural_traversal_data[1].illumination / throughput_rng / tlas
                                                    //// Normal traversal data
                                                    float,
                                                    float,
                                                    float,  // ray_o
                                                    float,
                                                    float,
                                                    float,      // ray_d
                                                    uint8_t,    // ray_depth
                                                    float,      // last_tlas_t
                                                    uint8_t,    // last_valid_hit
                                                    glm::vec3,  // normal_or_uv
                                                    glm::vec3,  // albedo_or_mesh_prim_id
                                                    uint32_t,   // current_blas_idx
                                                    uint32_t,   // pixel_coord_active
                                                    int,        // tlas_stack_size
                                                    uint32_t,   // tlas_stack
                                                    glm::vec3,  // illumination
                                                    glm::vec4,  // throughput_rng
                                                    float       // network outputs
                                                    >(default_stream,
                                                      &m_inference_output_data->scratch_alloc,
                                                      ////// Ping buffer //////////
                                                      n_padded_ray_payloads,  // valid_hit[0]
                                                      n_padded_ray_payloads,  // t_neural[0]
                                                      n_padded_ray_payloads,  // normal[0]
                                                      n_padded_ray_payloads,  // albedo[0]
                                                      ////// Pong buffer //////////
                                                      n_padded_ray_payloads,  // valid_hit[1]
                                                      n_padded_ray_payloads,  // t_neural[1]
                                                      n_padded_ray_payloads,  // normal[1]
                                                      n_padded_ray_payloads,  // albedo[1]
                                                      ////// HybridIntersectionResult buffer //////////
                                                      n_padded_ray_payloads,  // valid_hit
                                                      n_padded_ray_payloads,  // t_neural
                                                      n_padded_ray_payloads,  // normal_or_uv
                                                      n_padded_ray_payloads,  // albedo_or_mesh_prim_id
                                                      n_padded_ray_payloads,  //
                                                      n_padded_ray_payloads,  //
                                                      n_padded_ray_payloads,  // ray_o
                                                      n_padded_ray_payloads,  //
                                                      n_padded_ray_payloads,  //
                                                      n_padded_ray_payloads,  // ray_do
                                                      n_padded_ray_payloads,  // ray_depth
                                                      n_padded_ray_payloads,  // illumination
                                                      n_padded_ray_payloads,  // throughput_rng
                                                      n_padded_ray_payloads,  // pixel_coord
                                                      n_padded_ray_payloads,  // tlas_stack_size
                                                      n_padded_ray_payloads * TLAS_STACK_SIZE,  // tlas_stack
                                                      //  m_hybrid_intersection_data
                                                      n_padded_ray_payloads,                    // illumination[0]
                                                      n_padded_ray_payloads,                    // throughput_rng[0]
                                                      n_padded_ray_payloads,                    // tlas_stack_size
                                                      n_padded_ray_payloads * TLAS_STACK_SIZE,  // tlas_stack
                                                      // neural_traversal_data[0].illumination / throughput_rng / tlas
                                                      n_padded_ray_payloads,                    // illumination[1]
                                                      n_padded_ray_payloads,                    // throughput_rng[1]
                                                      n_padded_ray_payloads,                    // tlas_stack_size
                                                      n_padded_ray_payloads * TLAS_STACK_SIZE,  // tlas_stack
                                                      // neural_traversal_data[1].illumination / throughput_rng / tlas
                                                      ////// Normal traversal data //////////
                                                      n_padded_ray_payloads,
                                                      n_padded_ray_payloads,
                                                      n_padded_ray_payloads,  // ray_o
                                                      n_padded_ray_payloads,
                                                      n_padded_ray_payloads,
                                                      n_padded_ray_payloads,  // ray_d
                                                      n_padded_ray_payloads,  // ray_depth
                                                      n_padded_ray_payloads,  // last_tlas_t
                                                      n_padded_ray_payloads,  // last_valid_hit
                                                      n_padded_ray_payloads,  // normal_or_uv
                                                      n_padded_ray_payloads,  // albedo_or_mesh_prim_id
                                                      n_padded_ray_payloads,  // current_blas_idx
                                                      n_padded_ray_payloads,  // pixel_coord_active
                                                      n_padded_ray_payloads,  // tlas_stack_size
                                                      n_padded_ray_payloads * TLAS_STACK_SIZE,  // tlas_stack
                                                      n_padded_ray_payloads,                    // illumination
                                                      n_padded_ray_payloads,                    // throughput_rng
                                                      n_padded_elements * m_pt_target_const.output_network_dims);

        NeuralTraversalResult neural_query_result_0 = NeuralTraversalResult(std::get<0>(query_scratch),
                                                                            std::get<1>(query_scratch),
                                                                            std::get<2>(query_scratch),
                                                                            std::get<3>(query_scratch));

        NeuralTraversalResult neural_query_result_1 = NeuralTraversalResult(std::get<4>(query_scratch),
                                                                            std::get<5>(query_scratch),
                                                                            std::get<6>(query_scratch),
                                                                            std::get<7>(query_scratch));

        m_wavefront_neural_data[0].results = neural_query_result_0;
        m_wavefront_neural_data[1].results = neural_query_result_1;

        m_hybrid_intersection_data = HybridIntersectionResult(
            std::get<8>(query_scratch),
            std::get<9>(query_scratch),
            std::get<10>(query_scratch),
            std::get<11>(query_scratch),
            vec3_soa(std::get<12>(query_scratch), std::get<13>(query_scratch), std::get<14>(query_scratch)),
            vec3_soa(std::get<15>(query_scratch), std::get<16>(query_scratch), std::get<17>(query_scratch)),
            std::get<18>(query_scratch),
            std::get<19>(query_scratch),
            std::get<20>(query_scratch),
            std::get<21>(query_scratch),
            std::get<22>(query_scratch),
            std::get<23>(query_scratch));

        // The parent module does not set these as the child module might not necessarly need them
        // so we set them here and assign the traversal data
        m_inference_input_data->neural_traversal_data[0].illumination    = std::get<24>(query_scratch);
        m_inference_input_data->neural_traversal_data[0].throughput_rng  = std::get<25>(query_scratch);
        m_inference_input_data->neural_traversal_data[0].tlas_stack_size = std::get<26>(query_scratch);
        m_inference_input_data->neural_traversal_data[0].tlas_stack      = std::get<27>(query_scratch);
        m_inference_input_data->neural_traversal_data[1].illumination    = std::get<28>(query_scratch);
        m_inference_input_data->neural_traversal_data[1].throughput_rng  = std::get<29>(query_scratch);
        m_inference_input_data->neural_traversal_data[1].tlas_stack_size = std::get<30>(query_scratch);
        m_inference_input_data->neural_traversal_data[1].tlas_stack      = std::get<31>(query_scratch);

        m_normal_traversal_data = NormalTraversalData(
            vec3_soa(std::get<32>(query_scratch), std::get<33>(query_scratch), std::get<34>(query_scratch)),
            vec3_soa(std::get<35>(query_scratch), std::get<36>(query_scratch), std::get<37>(query_scratch)),
            std::get<38>(query_scratch),
            std::get<39>(query_scratch),
            std::get<40>(query_scratch),
            std::get<41>(query_scratch),
            std::get<42>(query_scratch),
            std::get<43>(query_scratch),
            std::get<44>(query_scratch),
            std::get<45>(query_scratch),
            std::get<46>(query_scratch),
            std::get<47>(query_scratch),
            std::get<48>(query_scratch));

        m_wavefront_neural_data[0].traversal_data = m_inference_input_data->neural_traversal_data[0];
        m_wavefront_neural_data[1].traversal_data = m_inference_input_data->neural_traversal_data[1];

        m_inference_output_data->network_outputs = std::get<49>(query_scratch);
    }

    void NeuralPathModule::update_neural_model_module()
    {
        assert(m_pt_target_const.output_visibility_dim == 1);
        m_pt_target_const.output_network_dims =
            m_pt_target_const.output_visibility_dim + m_pt_target_const.output_depth_dim +
            m_pt_target_const.output_normal_dim + m_pt_target_const.output_albedo_dim;

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

    nlohmann::json NeuralPathModule::network_loss_module()
    {
        return network_loss_config;
    }

    void NeuralPathModule::setup_neural_module_constants_module()
    {
        CUDA_CHECK(cudaMemcpyToSymbol(path_module_const, &m_path_module_const, sizeof(path_module_const)));
        CUDA_CHECK(cudaMemcpyToSymbol(pt_target_const, &m_pt_target_const, sizeof(pt_target_const)));
    }

    uint32_t NeuralPathModule::network_output_dims_module()
    {
        return m_pt_target_const.output_network_dims;
    }

    bool NeuralPathModule::ui_module(bool &reset_renderer, bool dev_mode)
    {
        bool changed_model = false;

        ImGui::Begin("Neural Path Tracing");

        ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.4f);

        ImGui::SeparatorText("Path tracer options");

        ImGui::Indent();

        reset_renderer |= ImGui::InputInt("Path Max Depth", &m_max_depth);
        if (dev_mode) {
            reset_renderer |= ImGui::Checkbox("Apply russian roulette", &m_path_module_const.apply_russian_roulette);
            if (m_path_module_const.apply_russian_roulette) {
                ImGui::Indent();
                reset_renderer |= ImGui::InputInt("Start depth", &m_path_module_const.rr_start_depth);
                ImGui::Unindent();
            }
        }

        ImGui::Unindent();
        ImGui::SeparatorText("Neural Material");
        ImGui::Indent();
        reset_renderer |= ImGui::SliderFloat("Specular", &m_path_module_const.neural_specular, 0.f, 1.f);
        reset_renderer |= ImGui::SliderFloat("Metallic", &m_path_module_const.neural_metallic, 0.f, 1.f);
        reset_renderer |= ImGui::SliderFloat("Roughness", &m_path_module_const.neural_roughness, 0.f, 1.f);
        ImGui::Unindent();

        if (dev_mode) {
            changed_model |= m_pt_target_const.pt_target_ui(reset_renderer);
        }
        ImGui::PopItemWidth();

        ImGui::End();

        return changed_model;
    }

}
}