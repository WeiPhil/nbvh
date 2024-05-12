#pragma once

// Imgui includes
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "imgui_spectrum.h"
// Include imgui file browser after imgui
#include "imfilebrowser.h"

#include <glad/glad.h>
// Order of glad/GLFW important
#include <GLFW/glfw3.h>

#include <memory>
#include <string>
#include "glm_pch.h"

#include "utils/shader_manager.h"

#include <cuda_gl_interop.h>

namespace ntwr {

class RenderDisplay {
public:
    RenderDisplay(glm::uvec2 window_size);
    ~RenderDisplay();

    // Stuff to run at each frame start, e.g. Imgui begin frame, event polling, etc..
    void begin_frame();

    // Stuff to run at every end of frame, e.g Imgui primitives rendering
    void end_frame();

    // Displays the pixel_buffer_texture
    void draw_pixel_buffer_texture();

    // Resizes the render texture
    void window_resize(const glm::uvec2 &new_size);

    glm::uvec2 window_size() const;

    GLFWwindow *glfw_window() const;

    std::vector<uint32_t> &ldr_readback_framebuffer();
    std::vector<float> &hdr_readback_framebuffer();
    GLuint gl_framebuffer_texture();
    GLuint gl_accum_buffer_texture();

private:
    // Initialise all display components, opengl context, glfw, glad, ImGui etc..
    void initialisation();

    // Called on destruction
    void destroy();

    /*! the glfw window handle */
    GLFWwindow *m_window{nullptr};
    // Render Texture
    GLuint m_gl_framebuffer_texture;
    GLuint m_gl_accum_buffer_texture;
    GLuint m_vao;

    std::unique_ptr<ShaderManager> m_shader_manager;
    // Host side framebuffer data
    glm::uvec2 m_window_size;
    std::vector<uint32_t> m_ldr_readback_framebuffer;
    std::vector<float> m_hdr_readback_framebuffer;
};

}