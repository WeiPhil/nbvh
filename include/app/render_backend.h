#pragma once

#include <string>

#include "app/render_display.h"
#include "base/bitmap.h"
#include "base/camera/camera.h"
#include "base/scene.h"

namespace ntwr {

/* Interface for any renderer */
struct RenderBackend {
    virtual std::string name() = 0;

    virtual std::filesystem::path scene_filename() = 0;

    // UI for the backend, returns true if the backend (usually the scene) was changed
    virtual bool ui(bool dev_mode) = 0;

    // Set a potential reference image for the backend
    virtual void set_reference_image(Bitmap<float> reference_image) = 0;

    // Reads back the backend (4 uint4)framebuffer (usually to write to a disk file)
    virtual void readback_framebuffer(size_t buffer_size, unsigned char *buffer) = 0;

    // Reads back the backend (4 float) framebuffer (usually to write to a disk file)
    virtual void readback_framebuffer_float(size_t buffer_size, float *buffer) = 0;

    // Maps a GL display texture to the backend for fast backend->screen transfer
    virtual void map_display_memory() = 0;

    // Releases any mapped memory, important to perform before any window resize
    virtual void release_display_mapped_memory() = 0;

    /*! called when windows gets resized */
    virtual void resize(const glm::uvec2 &new_size) = 0;

    /*! called when scene camera is updated */
    virtual void update_camera(const Camera &camera) = 0;

    /*! called when scene is updated */
    virtual void update_scene_data(const Scene &scene) = 0;

    /*! Returns the framebuffer size */
    virtual glm::uvec2 framebuffer_size() = 0;
};

}
