

#include "app/render_display.h"
#include "utils/logger.h"

#include <iostream>
#include <string>

#include <cuda_gl_interop.h>

namespace ntwr {

const std::string render_display_vs = /*glsl*/ R"(
#version 330 core

const vec4 pos[3] = vec4[3](
	vec4(-1, 5, 0.0, 1),
	vec4(-1, -5, 0.0, 1),
	vec4(5, 5, 0.0, 1)
);

void main(void){
	gl_Position = pos[gl_VertexID];
}
)";

const std::string render_display_fs = /*glsl*/ R"(
#version 330 core

uniform sampler2D img;

out vec4 color;

layout(origin_upper_left) in vec4 gl_FragCoord;

void main(void){ 
	ivec2 uv = ivec2(gl_FragCoord.x, gl_FragCoord.y);
	color = texelFetch(img, uv, 0);
}
)";

const std::string default_ini = R"(
[Window][Debug##Default]
Pos=60,60
Size=400,400
Collapsed=0

[Window][Render App]
Pos=5,35
Size=431,546
Collapsed=0

[Window][Neural BVH Renderer]
Pos=1370,67
Size=538,957
Collapsed=0

[Window][Neural Path Module]
Pos=1468,35
Size=446,1035
Collapsed=0

[Window][Scene Browser##filebrowser_2919764388552]
Pos=945,560
Size=700,450
Collapsed=0

[Window][Scene settings]
Pos=6,41
Size=436,372
Collapsed=1

[Window][Neural Path Tracing]
Pos=7,75
Size=437,318
Collapsed=1


)";

static void glfw_error_callback(int error, const char *description) {}

RenderDisplay::RenderDisplay(glm::uvec2 window_size) : m_window_size(window_size)
{
    initialisation();
}

RenderDisplay::~RenderDisplay()
{
    destroy();
}

void RenderDisplay::destroy()
{
    glDeleteVertexArrays(1, &m_vao);
    if (m_gl_framebuffer_texture != GLuint(-1)) {
        glDeleteTextures(1, &m_gl_framebuffer_texture);
    }
    if (m_gl_accum_buffer_texture != GLuint(-1)) {
        glDeleteTextures(1, &m_gl_accum_buffer_texture);
    }

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(m_window);
    glfwTerminate();
}

void RenderDisplay::initialisation()
{
    logger(LogLevel::Info, "Initializing display...");

    glfwSetErrorCallback(glfw_error_callback);

    if (!glfwInit())
        exit(EXIT_FAILURE);

    std::string glsl_version = "#version 330 core";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    glfwWindowHint(GLFW_DOUBLEBUFFER, GLFW_TRUE);
    glfwWindowHint(GLFW_SRGB_CAPABLE, GLFW_TRUE);

    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

    m_window = glfwCreateWindow(m_window_size.x, m_window_size.y, "N-BVH Renderer", NULL, NULL);

    if (!m_window) {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    glfwMakeContextCurrent(m_window);

    glfwSwapInterval(0);  // Disable vsync

    // Load glad
    if (!gladLoadGL()) {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    (void)io;

    // Check if an imgui file exists and if not load a default config
    if (!std::filesystem::exists("imgui.ini"))
        ImGui::LoadIniSettingsFromMemory(default_ini.c_str());

    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;  // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;   // Enable Gamepad Controls
    // io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;      // Enable Docking
    io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;  // Enable Multi-Viewport

    // Setup Dear ImGui style using Adobe's Spectrum
    ImGui::Spectrum::StyleColorsSpectrum();
    ImGui::Spectrum::LoadFont();

    // Set dpi scale factor to UI
    float dpi_factor_scale;
    glfwGetWindowContentScale(m_window, &dpi_factor_scale, nullptr);
    io.FontGlobalScale = dpi_factor_scale;

    ImGuiStyle &style = ImGui::GetStyle();
    style.ScaleAllSizes(dpi_factor_scale);
    style.WindowRounding = 4.0f;
    style.FrameRounding  = 4.0f;

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(m_window, true);
    ImGui_ImplOpenGL3_Init(glsl_version.c_str());

    m_shader_manager = std::make_unique<ShaderManager>(render_display_vs, render_display_fs);

    glGenVertexArrays(1, &m_vao);
    glBindVertexArray(m_vao);

    glClearColor(0.1f, 0.1f, 0.1f, 1.f);
    glDisable(GL_DEPTH_TEST);

    window_resize(m_window_size);
}

void RenderDisplay::window_resize(const glm::uvec2 &new_size)
{
    logger(LogLevel::Info, "Resizing window to %dx%d", new_size.x, new_size.y);

    // if window minimized
    if ((new_size.x == 0) || (new_size.y == 0))
        return;

    // Resize framebuffer texture
    if (m_gl_framebuffer_texture != GLuint(-1)) {
        glDeleteTextures(1, &m_gl_framebuffer_texture);
    }
    if (m_gl_accum_buffer_texture != GLuint(-1)) {
        glDeleteTextures(1, &m_gl_accum_buffer_texture);
    }
    m_window_size = new_size;
    m_ldr_readback_framebuffer.resize(m_window_size.x * m_window_size.y);
    // RGBA 32 bit floats
    m_hdr_readback_framebuffer.resize(m_window_size.x * m_window_size.y * 4);

    // Framebuffer texture
    glGenTextures(1, &m_gl_framebuffer_texture);
    glBindTexture(GL_TEXTURE_2D, m_gl_framebuffer_texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, m_window_size.x, m_window_size.y, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    // Accumulation texture
    glGenTextures(1, &m_gl_accum_buffer_texture);
    glBindTexture(GL_TEXTURE_2D, m_gl_accum_buffer_texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, m_window_size.x, m_window_size.y, 0, GL_RGBA, GL_FLOAT, nullptr);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
}

glm::uvec2 RenderDisplay::window_size() const
{
    return m_window_size;
}

GLFWwindow *RenderDisplay::glfw_window() const
{
    return m_window;
}

std::vector<uint32_t> &RenderDisplay::ldr_readback_framebuffer()
{
    return m_ldr_readback_framebuffer;
}

std::vector<float> &RenderDisplay::hdr_readback_framebuffer()
{
    return m_hdr_readback_framebuffer;
}

GLuint RenderDisplay::gl_framebuffer_texture()
{
    return m_gl_framebuffer_texture;
}
GLuint RenderDisplay::gl_accum_buffer_texture()
{
    return m_gl_accum_buffer_texture;
}

void RenderDisplay::draw_pixel_buffer_texture()
{
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_gl_framebuffer_texture);
    glViewport(0, 0, m_window_size.x, m_window_size.y);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    m_shader_manager->use();
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 3);
}

void RenderDisplay::begin_frame()
{
    glfwPollEvents();

    // Start the Dear ImGui frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

void RenderDisplay::end_frame()
{
    ImGui::Render();
    draw_pixel_buffer_texture();

    ImGuiIO &io = ImGui::GetIO();

    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    // Update and Render additional Platform Windows
    // (Platform functions may change the current OpenGL context, so we save/restore it to make it easier to paste this
    // code elsewhere.
    if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
        GLFWwindow *backup_current_context = glfwGetCurrentContext();
        ImGui::UpdatePlatformWindows();
        ImGui::RenderPlatformWindowsDefault();
        glfwMakeContextCurrent(backup_current_context);
    }

    glfwSwapBuffers(m_window);
}

}