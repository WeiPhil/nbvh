#pragma once

#include "app/render_backend.h"
#include "app/render_display.h"
#include "base/camera/camera.h"
#include "base/scene.h"

namespace ntwr {

/* Interface for any renderer */
struct Renderer {
    virtual std::string name() = 0;

    virtual RenderBackend *render_backend() = 0;

    /*! render any imgui component and update needed states, if renderer needs restart
     * returns true  */
    virtual bool render_ui(bool dev_mode) = 0;

    /*! render a single frame*/
    virtual void render(bool reset_accumulation) = 0;

    /*! exectures on window resize. No need to handle the render backend,
     *  it is handled by the render app already*/
    virtual void on_resize(const glm::uvec2 &new_size) = 0;

    /*! Returns the number of accumulated samples */
    virtual uint32_t accumulated_samples() = 0;

    /*! Update any required data when the scene is updated/changed */
    virtual void scene_updated() = 0;

protected:
    int m_accumulated_spp     = 0;
    int m_max_accumulated_spp = 2048;
    int m_sample_offset       = 0;
};

}