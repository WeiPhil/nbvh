#pragma once

#include "base/material.h"

#include "cuda/base/wip_disney.cuh"

#include "cuda/utils/common.cuh"
#include "cuda/utils/lcg_rng.cuh"
#include "cuda/utils/types.cuh"

// Cuda compatible sampling methods for the Disney BRDF
namespace ntwr {

// A CudaPackedMaterial is just an unpacked CudaMaterial
struct CudaMaterial : public Material {
    NTWR_DEVICE glm::vec3 emission()
    {
        return base_emission * emission_scale;
    }
};

#ifdef __CUDACC__
NTWR_DEVICE float textured_scalar_param(const float x, const glm::vec2 &uv, cudaTextureObject_t *textures)
{
    const uint32_t mask = __float_as_int(x);
    if (IS_TEXTURED_PARAM(mask)) {
        const uint32_t tex_id  = GET_TEXTURE_ID(mask);
        const uint32_t channel = GET_TEXTURE_CHANNEL(mask);
        return float4_to_vec4(tex2D<float4>(textures[tex_id], uv.x, uv.y))[channel];
    }
    return x;
}

NTWR_DEVICE CudaMaterial unpack_material(const PackedMaterial &p, glm::vec2 uv, cudaTextureObject_t *textures)
{
    CudaMaterial mat;
    uint32_t mask = __float_as_int(p.base_color.x);
    if (IS_TEXTURED_PARAM(mask)) {
        const uint32_t tex_id = GET_TEXTURE_ID(mask);
        mat.base_color        = float4_to_vec4(tex2D<float4>(textures[tex_id], uv.x, uv.y));
    } else {
        mat.base_color = p.base_color;
    }

    mat.metallic              = textured_scalar_param(p.metallic, uv, textures);
    mat.specular              = textured_scalar_param(p.specular, uv, textures);
    mat.roughness             = textured_scalar_param(p.roughness, uv, textures);
    mat.specular_tint         = textured_scalar_param(p.specular_tint, uv, textures);
    mat.anisotropy            = textured_scalar_param(p.anisotropy, uv, textures);
    mat.sheen                 = textured_scalar_param(p.sheen, uv, textures);
    mat.sheen_tint            = textured_scalar_param(p.sheen_tint, uv, textures);
    mat.clearcoat             = textured_scalar_param(p.clearcoat, uv, textures);
    mat.clearcoat_gloss       = textured_scalar_param(p.clearcoat_gloss, uv, textures);
    mat.ior                   = textured_scalar_param(p.ior, uv, textures);
    mat.specular_transmission = textured_scalar_param(p.specular_transmission, uv, textures);

    // Emission
    mask = __float_as_int(p.base_emission.x);
    if (IS_TEXTURED_PARAM(mask)) {
        const uint32_t tex_id = GET_TEXTURE_ID(mask);
        mat.base_emission     = float4_to_vec4(tex2D<float4>(textures[tex_id], uv.x, uv.y));
    } else {
        mat.base_emission = p.base_emission;
    }
    mat.emission_scale = p.emission_scale;
    return mat;
}

NTWR_DEVICE glm::vec3 unpack_normal(const PackedMaterial &p, glm::vec2 uv, cudaTextureObject_t *textures)
{
    uint32_t mask = __float_as_int(p.normal.x);
    if (IS_TEXTURED_PARAM(mask)) {
        const uint32_t tex_id = GET_TEXTURE_ID(mask);
        return glm::vec3(float4_to_vec4(tex2D<float4>(textures[tex_id], uv.x, uv.y)));
    } else {
        return glm::vec3(-1.f);
    }
}

NTWR_DEVICE glm::vec3 unpack_albedo(const PackedMaterial &p, glm::vec2 uv, cudaTextureObject_t *textures)
{
    uint32_t mask = __float_as_int(p.base_color.x);
    if (IS_TEXTURED_PARAM(mask)) {
        const uint32_t tex_id = GET_TEXTURE_ID(mask);
        return glm::vec3(float4_to_vec4(tex2D<float4>(textures[tex_id], uv.x, uv.y)));
    } else {
        return p.base_color;
    }
}

static_assert(sizeof(PackedMaterial) == sizeof(CudaMaterial));
static_assert(sizeof(CudaMaterial) == sizeof(Material));

NTWR_DEVICE bool same_hemisphere(const glm::vec3 &wo, const glm::vec3 &wi, const glm::vec3 &n)
{
    return dot(wo, n) * dot(wi, n) > 0.f;
}

// Sample the hemisphere using a cosine weighted distribution,
// returns a vector in a hemisphere oriented about (0, 0, 1)
NTWR_DEVICE glm::vec3 cos_sample_hemisphere(glm::vec2 u)
{
    glm::vec2 s = 2.f * u - glm::vec2(1.f);
    glm::vec2 d;
    float radius = 0.f;
    float theta  = 0.f;
    if (s.x == 0.f && s.y == 0.f) {
        d = s;
    } else {
        if (fabs(s.x) > fabs(s.y)) {
            radius = s.x;
            theta  = PI / 4.f * (s.y / s.x);
        } else {
            radius = s.y;
            theta  = PI / 2.f - PI / 4.f * (s.x / s.y);
        }
    }
    d = radius * glm::vec2(cos(theta), sin(theta));
    return glm::vec3(d.x, d.y, sqrt(std::max(0.f, 1.f - d.x * d.x - d.y * d.y)));
}

NTWR_DEVICE glm::vec3 spherical_dir(float sin_theta, float cos_theta, float phi)
{
    return glm::vec3(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
}

NTWR_DEVICE float power_heuristic(float n_f, float pdf_f, float n_g, float pdf_g)
{
    float f = n_f * pdf_f;
    float g = n_g * pdf_g;
    return (f * f) / (f * f + g * g);
}

NTWR_DEVICE float schlick_weight(float cos_theta)
{
    return pow(saturate(1.f - cos_theta), 5.f);
}

// Complete Fresnel Dielectric computation, for transmission at ior near 1
// they mention having issues with the Schlick approximation.
// eta_i: material on incident side's ior
// eta_t: material on transmitted side's ior
NTWR_DEVICE float fresnel_dielectric(float cos_theta_i, float eta_i, float eta_t)
{
    float g = sqr(eta_t) / sqr(eta_i) - 1.f + sqr(cos_theta_i);
    if (g < 0.f) {
        return 1.f;
    }
    return 0.5f * sqr(g - cos_theta_i) / sqr(g + cos_theta_i) *
           (1.f + sqr(cos_theta_i * (g + cos_theta_i) - 1.f) / sqr(cos_theta_i * (g - cos_theta_i) + 1.f));
}

// D_GTR1: Generalized Trowbridge-Reitz with gamma=1
// Burley notes eq. 4
NTWR_DEVICE float gtr_1(float cos_theta_h, float alpha)
{
    if (alpha >= 1.f) {
        return INV_PI;
    }
    float alpha_sqr = alpha * alpha;
    return INV_PI * (alpha_sqr - 1.f) / (log(alpha_sqr) * (1.f + (alpha_sqr - 1.f) * cos_theta_h * cos_theta_h));
}

// D_GTR2: Generalized Trowbridge-Reitz with gamma=2
// Burley notes eq. 8
NTWR_DEVICE float gtr_2(float cos_theta_h, float alpha)
{
    float alpha_sqr = alpha * alpha;
    return INV_PI * alpha_sqr / sqr(1.f + (alpha_sqr - 1.f) * cos_theta_h * cos_theta_h);
}

// D_GTR2 Anisotropic: Anisotropic generalized Trowbridge-Reitz with gamma=2
// Burley notes eq. 13
NTWR_DEVICE float gtr_2_aniso(float h_dot_n, float h_dot_x, float h_dot_y, glm::vec2 alpha)
{
    return INV_PI / (alpha.x * alpha.y * sqr(sqr(h_dot_x / alpha.x) + sqr(h_dot_y / alpha.y) + h_dot_n * h_dot_n));
}

NTWR_DEVICE float smith_shadowing_ggx(float n_dot_o, float alpha_g)
{
    float a = alpha_g * alpha_g;
    float b = n_dot_o * n_dot_o;
    return 1.f / (n_dot_o + sqrt(a + b - a * b));
}

NTWR_DEVICE float smith_shadowing_ggx_aniso(float n_dot_o, float o_dot_x, float o_dot_y, glm::vec2 alpha)
{
    return 1.f / (n_dot_o + sqrt(sqr(o_dot_x * alpha.x) + sqr(o_dot_y * alpha.y) + sqr(n_dot_o)));
}

// Sample a reflection direction the hemisphere oriented along n and spanned by tangent, bitangent using
// the random samples in s
NTWR_DEVICE glm::vec3 sample_lambertian_dir(const glm::vec3 &n,
                                            const glm::vec3 &tangent,
                                            const glm::vec3 &bitangent,
                                            const glm::vec2 &s)
{
    const glm::vec3 hemi_dir = normalize(cos_sample_hemisphere(s));
    return hemi_dir.x * tangent + hemi_dir.y * bitangent + hemi_dir.z * n;
}

// Sample the microfacet normal vectors for the various microfacet distributions
NTWR_DEVICE glm::vec3 sample_gtr_1_h(
    const glm::vec3 &n, const glm::vec3 &tangent, const glm::vec3 &bitangent, float alpha, const glm::vec2 &s)
{
    float phi_h           = 2.f * PI * s.x;
    float alpha_sqr       = alpha * alpha;
    float cos_theta_h_sqr = (1.f - pow(alpha_sqr, 1.f - s.y)) / (1.f - alpha_sqr);
    float cos_theta_h     = sqrt(cos_theta_h_sqr);
    float sin_theta_h     = 1.f - cos_theta_h_sqr;
    glm::vec3 hemi_dir    = normalize(spherical_dir(sin_theta_h, cos_theta_h, phi_h));
    return hemi_dir.x * tangent + hemi_dir.y * bitangent + hemi_dir.z * n;
}

NTWR_DEVICE glm::vec3 sample_gtr_2_h(
    const glm::vec3 &n, const glm::vec3 &tangent, const glm::vec3 &bitangent, float alpha, const glm::vec2 &s)
{
    float phi_h           = 2.f * PI * s.x;
    float cos_theta_h_sqr = (1.f - s.y) / (1.f + (alpha * alpha - 1.f) * s.y);
    float cos_theta_h     = sqrt(cos_theta_h_sqr);
    float sin_theta_h     = 1.f - cos_theta_h_sqr;
    glm::vec3 hemi_dir    = normalize(spherical_dir(sin_theta_h, cos_theta_h, phi_h));
    return hemi_dir.x * tangent + hemi_dir.y * bitangent + hemi_dir.z * n;
}

NTWR_DEVICE glm::vec3 sample_gtr_2_aniso_h(const glm::vec3 &n,
                                           const glm::vec3 &tangent,
                                           const glm::vec3 &bitangent,
                                           const glm::vec2 &alpha,
                                           const glm::vec2 &s)
{
    float x       = 2.f * PI * s.x;
    glm::vec3 w_h = sqrt(s.y / (1.f - s.y)) * (alpha.x * cos(x) * tangent + alpha.y * sin(x) * bitangent) + n;
    return normalize(w_h);
}

NTWR_DEVICE float lambertian_pdf(const glm::vec3 &wi, const glm::vec3 &n)
{
    float d = dot(wi, n);
    if (d > 0.f) {
        return d * INV_PI;
    }
    return 0.f;
}

NTWR_DEVICE float gtr_1_pdf(const glm::vec3 &wo, const glm::vec3 &wi, const glm::vec3 &n, float alpha)
{
    if (!same_hemisphere(wo, wi, n)) {
        return 0.f;
    }
    glm::vec3 w_h     = normalize(wi + wo);
    float cos_theta_h = dot(n, w_h);
    float d           = gtr_1(cos_theta_h, alpha);
    return d * cos_theta_h / (4.f * dot(wo, w_h));
}

NTWR_DEVICE float gtr_2_pdf(const glm::vec3 &wo, const glm::vec3 &wi, const glm::vec3 &n, float alpha)
{
    if (!same_hemisphere(wo, wi, n)) {
        return 0.f;
    }
    glm::vec3 w_h     = normalize(wi + wo);
    float cos_theta_h = dot(n, w_h);
    float d           = gtr_2(cos_theta_h, alpha);
    return d * cos_theta_h / (4.f * dot(wo, w_h));
}

NTWR_DEVICE float gtr_2_transmission_pdf(
    const glm::vec3 &wo, const glm::vec3 &wi, const glm::vec3 &n, float alpha, float ior)
{
    if (same_hemisphere(wo, wi, n)) {
        return 0.f;
    }
    bool entering     = dot(wo, n) > 0.f;
    float eta_o       = entering ? 1.f : ior;
    float eta_i       = entering ? ior : 1.f;
    glm::vec3 w_h     = normalize(wo + wi * eta_i / eta_o);
    float cos_theta_h = fabs(dot(n, w_h));
    float i_dot_h     = dot(wi, w_h);
    float o_dot_h     = dot(wo, w_h);
    float d           = gtr_2(cos_theta_h, alpha);
    float dwh_dwi     = o_dot_h * sqr(eta_o) / sqr(eta_o * o_dot_h + eta_i * i_dot_h);
    return d * cos_theta_h * fabs(dwh_dwi);
}

NTWR_DEVICE float gtr_2_aniso_pdf(const glm::vec3 &wo,
                                  const glm::vec3 &wi,
                                  const glm::vec3 &n,
                                  const glm::vec3 &tangent,
                                  const glm::vec3 &bitangent,
                                  const glm::vec2 alpha)
{
    if (!same_hemisphere(wo, wi, n)) {
        return 0.f;
    }
    glm::vec3 w_h     = normalize(wi + wo);
    float cos_theta_h = dot(n, w_h);
    float d           = gtr_2_aniso(cos_theta_h, fabs(dot(w_h, tangent)), fabs(dot(w_h, bitangent)), alpha);
    return d * cos_theta_h / (4.f * dot(wo, w_h));
}

NTWR_DEVICE glm::vec3 disney_diffuse(const CudaMaterial &mat,
                                     const glm::vec3 &n,
                                     const glm::vec3 &wo,
                                     const glm::vec3 &wi)
{
    glm::vec3 w_h = normalize(wi + wo);
    float n_dot_o = fabs(dot(wo, n));
    float n_dot_i = fabs(dot(wi, n));
    float i_dot_h = dot(wi, w_h);
    float fd90    = 0.5f + 2.f * mat.roughness * i_dot_h * i_dot_h;
    float fi      = schlick_weight(n_dot_i);
    float fo      = schlick_weight(n_dot_o);
    return mat.base_color * INV_PI * glm::lerp(1.f, fd90, fi) * glm::lerp(1.f, fd90, fo);
}

NTWR_DEVICE glm::vec3 disney_microfacet_isotropic(const CudaMaterial &mat,
                                                  const glm::vec3 &n,
                                                  const glm::vec3 &wo,
                                                  const glm::vec3 &wi)
{
    glm::vec3 w_h  = normalize(wi + wo);
    float lum      = luminance(mat.base_color);
    glm::vec3 tint = lum > 0.f ? mat.base_color / lum : glm::vec3(1.f);
    glm::vec3 spec =
        lerp(mat.specular * 0.08f * lerp(glm::vec3(1.f), tint, mat.specular_tint), mat.base_color, mat.metallic);

    float alpha = std::max(0.001f, mat.roughness * mat.roughness);
    float d     = gtr_2(dot(n, w_h), alpha);
    glm::vec3 f = lerp(spec, glm::vec3(1.f), schlick_weight(dot(wi, w_h)));
    float g     = smith_shadowing_ggx(dot(n, wi), alpha) * smith_shadowing_ggx(dot(n, wo), alpha);
    return d * f * g;
}

NTWR_DEVICE glm::vec3 disney_microfacet_transmission_isotropic(const CudaMaterial &mat,
                                                               const glm::vec3 &n,
                                                               const glm::vec3 &wo,
                                                               const glm::vec3 &wi)
{
    float o_dot_n = dot(wo, n);
    float i_dot_n = dot(wi, n);
    if (o_dot_n == 0.f || i_dot_n == 0.f) {
        return glm::vec3(0.f);
    }
    bool entering = o_dot_n > 0.f;
    float eta_o   = entering ? 1.f : mat.ior;
    float eta_i   = entering ? mat.ior : 1.f;
    glm::vec3 w_h = normalize(wo + wi * eta_i / eta_o);

    float alpha = std::max(0.001f, mat.roughness * mat.roughness);
    float d     = gtr_2(fabs(dot(n, w_h)), alpha);

    float f = fresnel_dielectric(fabs(dot(wi, n)), eta_o, eta_i);
    float g = smith_shadowing_ggx(fabs(dot(n, wi)), alpha) * smith_shadowing_ggx(fabs(dot(n, wo)), alpha);

    float i_dot_h = dot(wi, w_h);
    float o_dot_h = dot(wo, w_h);

    float c = fabs(o_dot_h) / fabs(dot(wo, n)) * fabs(i_dot_h) / fabs(dot(wi, n)) * sqr(eta_o) /
              sqr(eta_o * o_dot_h + eta_i * i_dot_h);

    return mat.base_color * c * (1.f - f) * g * d;
}

NTWR_DEVICE glm::vec3 disney_microfacet_anisotropic(const CudaMaterial &mat,
                                                    const glm::vec3 &n,
                                                    const glm::vec3 &wo,
                                                    const glm::vec3 &wi,
                                                    const glm::vec3 &tangent,
                                                    const glm::vec3 &bitangent)
{
    glm::vec3 w_h  = normalize(wi + wo);
    float lum      = luminance(mat.base_color);
    glm::vec3 tint = lum > 0.f ? mat.base_color / lum : glm::vec3(1.f);
    glm::vec3 spec =
        lerp(mat.specular * 0.08f * lerp(glm::vec3(1.f), tint, mat.specular_tint), mat.base_color, mat.metallic);

    float aspect    = sqrt(1.f - mat.anisotropy * 0.9f);
    float a         = mat.roughness * mat.roughness;
    glm::vec2 alpha = glm::vec2(std::max(0.001f, a / aspect), std::max(0.001f, a * aspect));
    float d         = gtr_2_aniso(dot(n, w_h), fabs(dot(w_h, tangent)), fabs(dot(w_h, bitangent)), alpha);
    glm::vec3 f     = lerp(spec, glm::vec3(1.f), schlick_weight(dot(wi, w_h)));
    float g         = smith_shadowing_ggx_aniso(dot(n, wi), fabs(dot(wi, tangent)), fabs(dot(wi, bitangent)), alpha) *
              smith_shadowing_ggx_aniso(dot(n, wo), fabs(dot(wo, tangent)), fabs(dot(wo, bitangent)), alpha);
    return d * f * g;
}

NTWR_DEVICE float disney_clear_coat(const CudaMaterial &mat,
                                    const glm::vec3 &n,
                                    const glm::vec3 &wo,
                                    const glm::vec3 &wi)
{
    glm::vec3 w_h = normalize(wi + wo);
    float alpha   = glm::lerp(0.1f, 0.001f, mat.clearcoat_gloss);
    float d       = gtr_1(dot(n, w_h), alpha);
    float f       = glm::lerp(0.04f, 1.f, schlick_weight(dot(wi, n)));
    float g       = smith_shadowing_ggx(dot(n, wi), 0.25f) * smith_shadowing_ggx(dot(n, wo), 0.25f);
    return 0.25f * mat.clearcoat * d * f * g;
}

NTWR_DEVICE glm::vec3 disney_sheen(const CudaMaterial &mat,
                                   const glm::vec3 &n,  // currently unused, correct?
                                   const glm::vec3 &wo,
                                   const glm::vec3 &wi)
{
    glm::vec3 w_h         = normalize(wi + wo);
    float lum             = luminance(mat.base_color);
    glm::vec3 tint        = lum > 0.f ? mat.base_color / lum : glm::vec3(1.f);
    glm::vec3 sheen_color = lerp(glm::vec3(1.f), tint, mat.sheen_tint);

    // Incorrrect use of wi dot n instead of wi dot w_h ?
    // float f               = schlick_weight(dot(wi, n));
    float f = schlick_weight(dot(wi, w_h));

    return f * mat.sheen * sheen_color;
}

NTWR_DEVICE glm::vec3 disney_eval(const CudaMaterial &mat,
                                  const glm::vec3 &n,
                                  const glm::vec3 &wo,
                                  const glm::vec3 &wi,
                                  const glm::vec3 &tangent,
                                  const glm::vec3 &bitangent)
{
    if (!same_hemisphere(wo, wi, n)) {
        if (mat.specular_transmission > 0.f) {
            glm::vec3 spec_trans = disney_microfacet_transmission_isotropic(mat, n, wo, wi);
            return spec_trans * (1.f - mat.metallic) * mat.specular_transmission;
        }
        return glm::vec3(0.f);
    }

    float coat        = disney_clear_coat(mat, n, wo, wi);
    glm::vec3 sheen   = disney_sheen(mat, n, wo, wi);
    glm::vec3 diffuse = disney_diffuse(mat, n, wo, wi);
    glm::vec3 gloss;
    if (mat.anisotropy == 0.f) {
        gloss = disney_microfacet_isotropic(mat, n, wo, wi);
    } else {
        gloss = disney_microfacet_anisotropic(mat, n, wo, wi, tangent, bitangent);
    }
    return (diffuse + sheen) * (1.f - mat.metallic) * (1.f - mat.specular_transmission) + gloss + coat;
}

NTWR_DEVICE float disney_pdf(const CudaMaterial &mat,
                             const glm::vec3 &n,
                             const glm::vec3 &wo,
                             const glm::vec3 &wi,
                             const glm::vec3 &tangent,
                             const glm::vec3 &bitangent)
{
    float alpha           = std::max(0.001f, mat.roughness * mat.roughness);
    float aspect          = sqrt(1.f - mat.anisotropy * 0.9f);
    glm::vec2 alpha_aniso = glm::vec2(std::max(0.001f, alpha / aspect), std::max(0.001f, alpha * aspect));

    float clearcoat_alpha = glm::lerp(0.1f, 0.001f, mat.clearcoat_gloss);

    float diffuse    = lambertian_pdf(wi, n);
    float clear_coat = gtr_1_pdf(wo, wi, n, clearcoat_alpha);

    float n_comp                  = 3.f;
    float microfacet              = 0.f;
    float microfacet_transmission = 0.f;
    if (mat.anisotropy == 0.f) {
        microfacet = gtr_2_pdf(wo, wi, n, alpha);
    } else {
        microfacet = gtr_2_aniso_pdf(wo, wi, n, tangent, bitangent, alpha_aniso);
    }
    if (mat.specular_transmission > 0.f) {
        n_comp                  = 4.f;
        microfacet_transmission = gtr_2_transmission_pdf(wo, wi, n, alpha, mat.ior);
    }
    return (diffuse + microfacet + microfacet_transmission + clear_coat) / n_comp;
}

NTWR_DEVICE glm::vec3 dielectric_sample_eval(const CudaMaterial &mat,
                                             const glm::vec3 &n,
                                             const glm::vec3 &wo,
                                             const glm::vec3 &tangent,
                                             const glm::vec3 &bitangent,
                                             const glm::vec2 &samples,
                                             glm::vec3 &wi,
                                             float &pdf)
{
    assert(mat.specular_transmission > 0.f);
    // For lighthouse disney transmission we need to invert the mat.ior -> 1.5 -> 1/1.5
    // And always swap the shading normal to be in the same hemisphere as the incoming vector
    float eta = 1.f / mat.ior;

    // This means normal and wo are always on the same hemisphere
    assert(dot(n, wo) >= 0);

    // sample BSDF
    glm::vec3 bsdf = glm::vec3(0.f);
    float F        = Fr(dot(n, wo), eta);
    if (samples.y < F)  // sample reflectance or transmission based on Fresnel term
    {
        wi = reflect(wo * -1.0f, n);

        if (dot(wi, n) > 0.0f) {
            bsdf = glm::vec3(F);
        }
    } else  // sample transmission
    {
        if (Refract(wo, n, eta, wi)) {
            bsdf = glm::vec3(1.0f - F);
        }
    }
    pdf = 1.0;
    // Remove cosine term from rendering equation
    return bsdf / fabs(dot(n, wi));
}

/* Sample a component of the Disney BRDF, returns the sampled BRDF color,
 * ray reflection direction (wi) and sample PDF.
 */
NTWR_DEVICE glm::vec3 disney_sample_eval(const CudaMaterial &mat,
                                         const glm::vec3 &n,
                                         const glm::vec3 &wo,
                                         const glm::vec3 &tangent,
                                         const glm::vec3 &bitangent,
                                         LCGRand &rng,
                                         glm::vec3 &wi,
                                         float &pdf)
{
    int component     = 0;
    glm::vec2 samples = glm::vec2(lcg_randomf(rng), lcg_randomf(rng));
    if (mat.specular_transmission < 1.f) {
        component = lcg_randomf(rng) * 3.f;
        component = glm::clamp(component, 0, 2);
    } else {
        // We entirely switch to transmission for now
        return dielectric_sample_eval(mat, n, wo, tangent, bitangent, samples, wi, pdf);
        // component = lcg_randomf(rng) * 4.f;
        // component = glm::clamp(component, 0, 3);
    }

    if (component == 0) {
        // Sample diffuse component
        wi = sample_lambertian_dir(n, tangent, bitangent, samples);
    } else if (component == 1) {
        glm::vec3 w_h;
        float alpha = std::max(0.001f, mat.roughness * mat.roughness);
        if (mat.anisotropy == 0.f) {
            w_h = sample_gtr_2_h(n, tangent, bitangent, alpha, samples);
        } else {
            float aspect          = sqrt(1.f - mat.anisotropy * 0.9f);
            glm::vec2 alpha_aniso = glm::vec2(std::max(0.001f, alpha / aspect), std::max(0.001f, alpha * aspect));
            w_h                   = sample_gtr_2_aniso_h(n, tangent, bitangent, alpha_aniso, samples);
        }
        wi = reflect(-wo, w_h);

        // Invalid reflection, terminate ray
        if (!same_hemisphere(wo, wi, n)) {
            pdf = 0.f;
            wi  = glm::vec3(0.f);
            return glm::vec3(0.f);
        }
    } else if (component == 2) {
        // Sample clear coat component
        float alpha   = glm::lerp(0.1f, 0.001f, mat.clearcoat_gloss);
        glm::vec3 w_h = sample_gtr_1_h(n, tangent, bitangent, alpha, samples);
        wi            = reflect(-wo, w_h);

        // Invalid reflection, terminate ray
        if (!same_hemisphere(wo, wi, n)) {
            pdf = 0.f;
            wi  = glm::vec3(0.f);
            return glm::vec3(0.f);
        }
    } else {
        assert(false);
        // Seems broken, using lighthouse's transmission only sampling

        // Sample microfacet transmission component
        float alpha   = std::max(0.001f, mat.roughness * mat.roughness);
        glm::vec3 w_h = sample_gtr_2_h(n, tangent, bitangent, alpha, samples);
        if (dot(wo, w_h) < 0.f) {
            w_h = -w_h;
        }
        bool entering = dot(wo, n) > 0.f;
        wi            = refract_ray(-wo, w_h, entering ? 1.f / mat.ior : mat.ior);

        // Invalid refraction, terminate ray
        if (wi == glm::vec3(0.f)) {
            pdf = 0.f;
            return glm::vec3(0.f);
        }
    }
    pdf = disney_pdf(mat, n, wo, wi, tangent, bitangent);
    return disney_eval(mat, n, wo, wi, tangent, bitangent);
}

#endif

}
