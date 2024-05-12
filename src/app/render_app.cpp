

#include "app/render_app.h"
#include "app/render_backend.h"
#include "app/renderer.h"
#include "base/camera/arcball.h"
#include "base/camera/camera.h"
#include "base/camera/flycam.h"
#include "utils/logger.h"

#include "stb_image_write.h"
#include "tinyexr.h"

#include "cuda/cuda_backend.h"
#include "cuda/neural/neural_bvh_renderer.h"

#include <filesystem>
#include <iostream>

#include "glm_pch.h"
#include "ntwr.h"

namespace ntwr {
namespace {

    // Window callbacks
    static void render_window_resize_cb(GLFWwindow *window, int win_width, int win_height)
    {
        RenderApp *render_app = static_cast<RenderApp *>(glfwGetWindowUserPointer(window));
        assert(render_app);

        if (!render_app->scene) {
            return;
        }

        assert(render_app->active_renderer_variant != -1);
        const glm::uvec2 new_window_size = glm::uvec2(win_width, win_height);

        for (auto backend : render_app->backends)
            backend->release_display_mapped_memory();
        render_app->display->window_resize(new_window_size);
        for (auto backend : render_app->backends) {
            backend->resize(new_window_size);
            backend->update_camera(*render_app->get_scene().camera);
            backend->map_display_memory();
        }

        for (auto renderer : render_app->renderers) {
            renderer->on_resize(render_app->display->window_size());
        }
        render_app->reset_accumulation = true;
    }

    // Drag and drop callback
    static void drop_callback(GLFWwindow *window, int count, const char **paths)
    {
        RenderApp *render_app = static_cast<RenderApp *>(glfwGetWindowUserPointer(window));
        assert(render_app);

        for (int i = 0; i < count; i++) {
            std::filesystem::path candidate_filename(paths[i]);
            if (candidate_filename.extension().string() == ".gltf" ||
                candidate_filename.extension().string() == ".glb") {
                render_app->assign_scene(candidate_filename);
                render_app->m_render_app_configs.set_scene_filename(candidate_filename.string());
            } else {
                logger(LogLevel::Error,
                       "Unsupported filename type %s for file %s",
                       candidate_filename.extension().string().c_str(),
                       candidate_filename.string().c_str());
            }
        }
    }
}

RenderApp::RenderApp(RenderAppConfigs render_app_configs,
                     std::filesystem::path scene_filename,
                     RenderDisplay *render_display)
    : m_render_app_configs(render_app_configs)
{
    if (scene_filename != "") {
        scene = std::make_unique<Scene>(scene_filename);
    }

    if (!display && render_display)
        logger(LogLevel::Info, "Attaching display to render app");
    else if (render_display)
        logger(LogLevel::Warn, "Overriding render app display!");
    else
        logger(LogLevel::Warn, "No render display attached!");
    display = render_display;

    m_exr_file_dialog.SetTitle("Image Browser");
    m_exr_file_dialog.SetTypeFilters({".exr"});

    m_scene_file_dialog.SetTitle("Scene Browser");
    m_scene_file_dialog.SetTypeFilters({".gltf", ".glb"});
}

RenderApp::~RenderApp() {}

void RenderApp::register_renderer(Renderer *renderer)
{
    logger(LogLevel::Info, "Added renderer : %s", renderer->name().c_str());
    active_renderer_variant          = renderers.size();
    previous_active_renderer_variant = renderers.size() != 0 ? renderers.size() : 0;
    renderer_variants.push_back(renderer->name());
    renderers.push_back(renderer);
}

void RenderApp::register_backend(RenderBackend *backend)
{
    backend->resize(display->window_size());
    backend->update_camera(*scene->camera);
    backends.push_back(backend);
}

void RenderApp::assign_scene(std::filesystem::path scene_filename)
{
    // Assign the new scene
    scene = std::make_unique<Scene>(scene_filename);

    for (auto backend : backends) {
        backend->update_scene_data(*scene);
        backend->update_camera(*scene->camera);
    }

    for (auto renderer : renderers)
        renderer->scene_updated();
}

void RenderApp::save_framebuffer(std::string filename) const
{
    // Read back (device-side)rendered image into (host-side)framebuffer
    auto render_backend = renderers[active_renderer_variant]->render_backend();
    assert(render_backend->framebuffer_size() == display->window_size());

    const std::filesystem::path result_dir = render_backend->scene_filename().parent_path() / "results";
    std::filesystem::create_directory(result_dir);

    if (filename.ends_with(".png")) {
        unsigned char *pixels = reinterpret_cast<unsigned char *>(display->ldr_readback_framebuffer().data());
        render_backend->readback_framebuffer(display->ldr_readback_framebuffer().size() * sizeof(uint32_t), pixels);

        logger(LogLevel::Info, "Saving framebuffer to %s...", filename.c_str());
        write_png(filename.c_str(), display->window_size().x, display->window_size().y, 4, pixels);
    } else {
        if (!filename.ends_with(".exr")) {
            logger(LogLevel::Warn, "No extension set, using .exr");
            filename = filename + ".exr";
        }
        render_backend->readback_framebuffer_float(display->hdr_readback_framebuffer().size() * sizeof(float),
                                                   display->hdr_readback_framebuffer().data());

        logger(LogLevel::Info, "Saving framebuffer to %s...", filename.c_str());

        write_exr(filename.c_str(),
                  display->window_size().x,
                  display->window_size().y,
                  4,
                  display->hdr_readback_framebuffer().data(),
                  (float)renderers[active_renderer_variant]->accumulated_samples());
    }
    logger(LogLevel::Info, "Done!");
}

bool RenderApp::app_ui(bool &camera_changed, bool &changed_renderer)
{
    namespace fs = std::filesystem;

    bool reset  = false;
    ImGuiIO &io = ImGui::GetIO();

    if (ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu("File")) {
            if (ImGui::MenuItem("Load Scene")) {
                m_scene_file_dialog.Open();
            }

            if (!renderers.empty()) {
                std::string config_path = "";
                if (ImGui::BeginMenu("Load Config")) {
                    auto config_dir = scene->m_scene_filename.parent_path() / "configs";
                    if (fs::exists(config_dir)) {
                        for (const auto &entry : fs::directory_iterator(config_dir)) {
                            if (entry.path().extension().string() == ".json") {
                                if (ImGui::MenuItem(entry.path().filename().string().c_str())) {
                                    config_path = entry.path().string();
                                }
                            }
                        }
                        if (config_path != "") {
                            std::ifstream input_file(config_path);
                            nlohmann::json full_config = nlohmann::json::parse(input_file);
                            std::string base_config_name =
                                full_config["neural_bvh_renderer"].value("base_config_name", std::string(""));

                            neural::NeuralBVHRenderer *renderer =
                                (neural::NeuralBVHRenderer *)renderers[active_renderer_variant];
                            renderer->load_config_from_json(full_config, base_config_name, true);
                            reset = true;
                        }
                    } else {
                        ImGui::Text("No configs found in scene_dir/configs");
                    }

                    ImGui::EndMenu();
                }

                ImGui::MenuItem("Devoloper mode", "", &m_dev_mode);
            }

            ImGui::EndMenu();
        }
        if (m_scene_file_dialog.HasSelected()) {
            assign_scene(m_scene_file_dialog.GetSelected().string());
            m_scene_file_dialog.ClearSelected();
            return reset;
        }

        ImGui::Separator();

        ImGui::Text("Application time : %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);

        ImGui::EndMainMenuBar();
    }

    if (renderers.empty()) {
        return reset;
    }

    ImGui::Begin("Scene settings");

    ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.5f);

    if (ImGui::Button("Save image")) {
        save_framebuffer(std::string(output_image_filename));
    }
    ImGui::SameLine();
    ImGui::InputText("Filename", output_image_filename, OUTPUT_FILENAME_SIZE);

    auto camera_movement_type = scene->camera->camera_movement_type();

    if (imgui_utils::combo_box(
            "Camera Type", camera_movement_types, ARRAY_SIZE(camera_movement_types), camera_movement_type)) {
        auto prev_trafo = scene->camera->transform();
        switch (camera_movement_type) {
        case CameraMovementType::Fly: {
            scene->camera = std::make_unique<FlyCam>();
            scene->camera->set_transform(prev_trafo);

            break;
        }
        case CameraMovementType::Arcball: {
            auto prev_origin = scene->camera->origin();
            auto arcball     = ArcballCam();
            arcball.set_transform(prev_trafo);
            arcball.set_origin(prev_origin);
            scene->camera = std::make_unique<ArcballCam>(arcball);

            break;
        }
        default:
            UNREACHABLE();
        }
    }

    auto render_backend = renderers[active_renderer_variant]->render_backend();
    reset |= render_backend->ui(m_dev_mode);

    if (m_dev_mode) {
        if (ImGui::Button("Set Reference from file")) {
            m_exr_file_dialog.Open();
        }
        if (ImGui::Button("Set Reference from framebuffer")) {
            render_backend->readback_framebuffer_float(display->hdr_readback_framebuffer().size() * sizeof(float),
                                                       display->hdr_readback_framebuffer().data());
            render_backend->set_reference_image(
                bitmap_from_vector(display->hdr_readback_framebuffer(),
                                   render_backend->framebuffer_size().x,
                                   render_backend->framebuffer_size().y,
                                   4,
                                   (float)renderers[active_renderer_variant]->accumulated_samples()));
        }

        if (m_exr_file_dialog.HasSelected()) {
            render_backend->set_reference_image(load_exr(m_exr_file_dialog.GetSelected().string().c_str()));

            m_exr_file_dialog.ClearSelected();
        }

        imgui_utils::large_separator();

        camera_changed |= scene->camera->ui();
    }

    ImGui::PopItemWidth();

    ImGui::End();

    return reset;
}

/*! opens the actual window, and runs the window's events to
  completion. This function will only return once the window
  gets closed */
void RenderApp::run()
{
    if (!display) {
        // TODO: Headless rendering
        return;
    }
    assert(display);

    glfwSetWindowUserPointer(display->glfw_window(), reinterpret_cast<void *>(this));
    glfwSetFramebufferSizeCallback(display->glfw_window(), render_window_resize_cb);
    glfwSetDropCallback(display->glfw_window(), drop_callback);

    // Running until we have a scene
    while (!glfwWindowShouldClose(display->glfw_window()) && !scene) {
        display->begin_frame();
        bool dummy = false;
        this->app_ui(dummy, dummy);
        m_scene_file_dialog.Display();
        display->end_frame();
    }

    logger(LogLevel::Info, "Initializing backends...");

    // Create all backends
    std::shared_ptr<CudaBackend> cuda_backend = std::make_shared<CudaBackend>(*scene, this);
    register_backend((RenderBackend *)cuda_backend.get());

    logger(LogLevel::Info, "Initializing renderers...");

    // Create the renderer and add it to the application

    std::unique_ptr<neural::NeuralBVHRenderer> neural_bvh_renderer_renderer =
        std::make_unique<neural::NeuralBVHRenderer>(cuda_backend);

    register_renderer(neural_bvh_renderer_renderer.get());

    assert(active_renderer_variant != -1);

    ImGuiIO &io = ImGui::GetIO();

    auto renderer = renderers[active_renderer_variant];

    bool changed_renderer  = false;
    bool changed_camera    = false;
    bool lose_window_focus = false;

    while (!glfwWindowShouldClose(display->glfw_window())) {
        display->begin_frame();

        if (!io.WantCaptureKeyboard) {
            if (ImGui::IsKeyPressed(ImGuiKey_Period)) {
                hide_ui           = !hide_ui;
                lose_window_focus = true;
            } else if (ImGui::IsKeyPressed(ImGuiKey_V)) {
                std::swap(active_renderer_variant, previous_active_renderer_variant);
                changed_renderer  = true;
                lose_window_focus = true;
            } else if (ImGui::IsKeyPressed(ImGuiKey_L)) {
                neural_bvh_renderer_renderer->toggle_node_loss_display();
                changed_renderer = true;
            } else if (ImGui::IsKeyPressed(ImGuiKey_K)) {
                neural_bvh_renderer_renderer->toggle_split_screen_node_loss_display();
                changed_renderer = true;
            } else if (ImGui::IsKeyPressed(ImGuiKey_R)) {
                neural_bvh_renderer_renderer->restart_optimisation();
                changed_renderer = true;
            }
        }

        // Camera movement update
        // Assuming framebuffer is the size of the window
        changed_camera = scene->camera->movement_io(io, display->window_size());
        if (!hide_ui)
            reset_accumulation |= this->app_ui(changed_camera, changed_renderer);
        reset_accumulation |= changed_camera;
        reset_accumulation |= changed_renderer;
        if (changed_renderer) {
            renderer         = renderers[active_renderer_variant];
            changed_renderer = false;
        }

        if (changed_camera) {
            for (auto backend : backends)
                backend->update_camera(*scene->camera);
            changed_camera = false;
        }

        if (!hide_ui)
            reset_accumulation |= renderer->render_ui(m_dev_mode);
        renderer->render(reset_accumulation);
        reset_accumulation = false;

        m_exr_file_dialog.Display();
        m_scene_file_dialog.Display();

        display->end_frame();

        if (lose_window_focus) {
            ImGui::SetWindowFocus(nullptr);
            lose_window_focus = false;
        }
    }
}
}