#pragma once

#include <string>
#include <vector>
#include "cuda/utils/common.cuh"
#include "cuda/utils/lcg_rng.cuh"

/* The material's float parameters (and the r component of the color)
 * can be a positive scalar, representing the input value, or if
 * negative correspond to a texture input. The bits are the interpreted as:
 *
 * [31]: 1, sign bit indicating this is a handle
 * [29:30]: 2 bits indicating the texture channel to use (base_color is
 *          assumed to use all channels and this is ignored)
 * [0:28]: texture ID
 *
 * (Taken from ChameleonRT)
 *
 */

#define TEXTURED_PARAM_MASK  0x80000000
#define IS_TEXTURED_PARAM(x) ((x) & 0x80000000)

#define GET_TEXTURE_CHANNEL(x)    (((x) >> 29) & 0x3)
#define SET_TEXTURE_CHANNEL(x, c) x |= (c & 0x3) << 29

#define GET_TEXTURE_ID(x)    ((x) & 0x1fffffff)
#define SET_TEXTURE_ID(x, i) (x |= i & 0x1fffffff)

namespace ntwr {

enum ColorSpace { LINEAR, SRGB };

struct Image {
    std::string name;
    int width    = -1;
    int height   = -1;
    int channels = -1;
    std::vector<uint8_t> pixels;
    ColorSpace color_space = LINEAR;

    Image(const std::string &file, const std::string &name, ColorSpace color_space = LINEAR);
    Image(const uint8_t *buf,
          int width,
          int height,
          int channels,
          const std::string &name,
          ColorSpace color_space = LINEAR);
    Image() = default;
};

struct Material {
    // Disney Material + Emission
    glm::vec3 base_color = glm::vec3(0.9f);
    float metallic       = 0.f;

    float specular      = 0.f;
    float roughness     = 1.f;
    float specular_tint = 0.f;
    float anisotropy    = 0.f;

    float sheen           = 0.f;
    float sheen_tint      = 0.f;
    float clearcoat       = 0.f;
    float clearcoat_gloss = 0.f;

    glm::vec3 base_emission = glm::vec3(0.0f);
    float emission_scale    = 1.0f;

    glm::vec3 normal = glm::vec3(0.f);
    float ior        = 1.5f;

    float specular_transmission = 0.f;

    virtual bool is_packed()
    {
        return false;
    }
};

// A packed material represents a material that has encoded texture information as well
struct PackedMaterial : public Material {
    bool is_packed() override
    {
        return true;
    }
};

}