/*
# Copyright Disney Enterprises, Inc.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License
# and the following modification to it: Section 6 Trademarks.
# deleted and replaced with:
#
# 6. Trademarks. This License does not grant permission to use the
# trade names, trademarks, service marks, or product names of the
# Licensor and its affiliates, except as required for reproducing
# the content of the NOTICE file.
#
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# Adapted to C++ by Miles Macklin 2016

*/

// #include "base/material.h"
#include "cuda/utils/common.cuh"

namespace ntwr {

#ifdef __CUDACC__

enum BSDFType { eReflected, eTransmitted, eSpecular };

#define CHAR2FLT(a, s) (((float)(((a) >> s) & 255)) * (1.0f / 255.0f))

struct ShadingData {
    // This structure is filled for an intersection point. It will contain the spatially varying material properties.
    glm::vec3 color;
    int flags;
    glm::vec3 transmittance;
    int matID;
    glm::vec4 tint;
    uint4 parameters;
    /* 16 uchars:   x: 0..7 = metallic, 8..15 = subsurface, 16..23 = specular, 24..31 = roughness;
                    y: 0..7 = specTint, 8..15 = anisotropic, 16..23 = sheen, 24..31 = sheenTint;
                    z: 0..7 = clearcoat, 8..15 = clearcoatGloss, 16..23 = transmission, 24..31 = dummy;
                    w: eta (32-bit float). */
    __device__ int IsSpecular(const int layer) const
    {
        return 0; /* for now. */
    }
    __device__ bool IsEmissive() const
    {
        return color.x > 1.0f || color.y > 1.0f || color.z > 1.0f;
    }

    __device__ void InvertETA()
    {
        parameters.w = __float_as_uint(1.0f / __uint_as_float(parameters.w));
    }

#define METALLIC       CHAR2FLT(shadingData.parameters.x, 0)
#define SUBSURFACE     CHAR2FLT(shadingData.parameters.x, 8)
#define SPECULAR       CHAR2FLT(shadingData.parameters.x, 16)
#define ROUGHNESS      (max(0.001f, CHAR2FLT(shadingData.parameters.x, 24)))
#define SPECTINT       CHAR2FLT(shadingData.parameters.y, 0)
#define ANISOTROPIC    CHAR2FLT(shadingData.parameters.y, 8)
#define SHEEN          CHAR2FLT(shadingData.parameters.y, 16)
#define SHEENTINT      CHAR2FLT(shadingData.parameters.y, 24)
#define CLEARCOAT      CHAR2FLT(shadingData.parameters.z, 0)
#define CLEARCOATGLOSS CHAR2FLT(shadingData.parameters.z, 8)
#define TRANSMISSION   CHAR2FLT(shadingData.parameters.z, 16)
#define TINT           make_float3(shadingData.tint)
#define LUMINANCE      shadingData.tint.w
#define DUMMY0         CHAR2FLT(shadingData.parameters.z, 24)
#define ETA            __uint_as_float(shadingData.parameters.w)
};

NTWR_DEVICE glm::vec3 DiffuseReflectionCosWeighted(const float r0, const float r1)
{
    const float term1 = TWOPI * r0, term2 = sqrtf(1 - r1);
    float s, c;
    sincosf(term1, &s, &c);
    return glm::vec3(c * term2, s * term2, sqrtf(r1));
}

NTWR_DEVICE glm::vec3 DiffuseReflectionCosWeighted(
    const float r0, const float r1, const glm::vec3 &N, const glm::vec3 &T, const glm::vec3 &B)
{
    const float term1 = TWOPI * r0, term2 = sqrtf(1 - r1);
    float s, c;
    sincosf(term1, &s, &c);
    return (c * term2 * T) + (s * term2) * B + sqrtf(r1) * N;
}

NTWR_DEVICE glm::vec3 DiffuseReflectionUniform(const float r0, const float r1)
{
    const float term1 = TWOPI * r0, term2 = sqrtf(1 - r1 * r1);
    float s, c;
    sincosf(term1, &s, &c);
    return glm::vec3(c * term2, s * term2, r1);
}

NTWR_DEVICE bool Refract(const glm::vec3 &wi, const glm::vec3 &n, const float eta, glm::vec3 &wt)
{
    float cosThetaI  = dot(n, wi);
    float sin2ThetaI = max(0.0f, 1.0f - cosThetaI * cosThetaI);
    float sin2ThetaT = eta * eta * sin2ThetaI;
    if (sin2ThetaT >= 1)
        return false;  // TIR
    float cosThetaT = sqrtf(1.0f - sin2ThetaT);
    wt              = eta * (wi * -1.0f) + (eta * cosThetaI - cosThetaT) * glm::vec3(n);
    return true;
}

NTWR_DEVICE float SchlickFresnel(const float u)
{
    const float m = glm::clamp(1 - u, 0.0f, 1.0f);
    return float(m * m) * (m * m) * m;
}

NTWR_DEVICE float GTR1(const float NDotH, const float a)
{
    if (a >= 1)
        return INV_PI;
    const float a2 = a * a;
    const float t  = 1 + (a2 - 1) * NDotH * NDotH;
    return (a2 - 1) / (PI * logf(a2) * t);
}

NTWR_DEVICE float GTR2(const float NDotH, const float a)
{
    const float a2 = a * a;
    const float t  = 1.0f + (a2 - 1.0f) * NDotH * NDotH;
    return a2 / (PI * t * t);
}

NTWR_DEVICE float SmithGGX(const float NDotv, const float alphaG)
{
    const float a = alphaG * alphaG;
    const float b = NDotv * NDotv;
    return 1 / (NDotv + sqrtf(a + b - a * b));
}

NTWR_DEVICE float Fr(const float VDotN, const float eio)
{
    const float SinThetaT2 = sqr(eio) * (1.0f - VDotN * VDotN);
    if (SinThetaT2 > 1.0f)
        return 1.0f;  // TIR
    const float LDotN = sqrtf(1.0f - SinThetaT2);
    // todo: reformulate to remove this division
    const float eta = 1.0f / eio;
    const float r1  = (VDotN - eta * LDotN) / (VDotN + eta * LDotN);
    const float r2  = (LDotN - eta * VDotN) / (LDotN + eta * VDotN);
    return 0.5f * (sqr(r1) + sqr(r2));
}

NTWR_DEVICE glm::vec3 SafeNormalize(const glm::vec3 &a)
{
    const float ls = dot(a, a);
    if (ls > 0.0f)
        return a * (1.0f / sqrtf(ls));
    else
        return glm::vec3(0);
}

__device__ static float BSDFPdf(const ShadingData &shadingData,
                                const glm::vec3 &N,
                                const glm::vec3 &wo,
                                const glm::vec3 &wi)
{
    float bsdfPdf = 0.0f, brdfPdf;
    if (dot(wi, N) <= 0.0f)
        brdfPdf = INV_TWOPI * SUBSURFACE * 0.5f;
    else {
        const float F            = Fr(dot(N, wo), ETA);
        const glm::vec3 half     = SafeNormalize(wi + wo);
        const float cosThetaHalf = abs(dot(half, N));
        const float pdfHalf      = GTR2(cosThetaHalf, ROUGHNESS) * cosThetaHalf;
        // calculate pdf for each method given outgoing light vector
        const float pdfSpec = 0.25f * pdfHalf / max(1.e-6f, dot(wi, half));
        const float pdfDiff = abs(dot(wi, N)) * INV_PI * (1.0f - SUBSURFACE);
        bsdfPdf             = pdfSpec * F;
        brdfPdf             = glm::lerp(pdfDiff, pdfSpec, 0.5f);
    }
    return glm::lerp(brdfPdf, bsdfPdf, TRANSMISSION);
}

// evaluate the BSDF for a given pair of directions
__device__ static glm::vec3 BSDFEval(const ShadingData &shadingData,
                                     const glm::vec3 &N,
                                     const glm::vec3 &wo,
                                     const glm::vec3 &wi)
{
    const float NDotL      = dot(N, wi);
    const float NDotV      = dot(N, wo);
    const glm::vec3 H      = normalize(wi + wo);
    const float NDotH      = dot(N, H);
    const float LDotH      = dot(wi, H);
    const glm::vec3 Cdlin  = shadingData.color;
    const float Cdlum      = .3f * Cdlin.x + .6f * Cdlin.y + .1f * Cdlin.z;   // luminance approx.
    const glm::vec3 Ctint  = Cdlum > 0.0f ? Cdlin / Cdlum : glm::vec3(1.0f);  // normalize lum. to isolate hue+sat
    const glm::vec3 Cspec0 = glm::lerp(SPECULAR * .08f * glm::lerp(glm::vec3(1.0f), Ctint, SPECTINT), Cdlin, METALLIC);
    glm::vec3 bsdf         = glm::vec3(0);
    glm::vec3 brdf         = glm::vec3(0);
    if (TRANSMISSION > 0.0f) {
        // evaluate BSDF
        if (NDotL <= 0) {
            // transmission Fresnel
            const float F = Fr(NDotV, ETA);
            bsdf          = glm::vec3((1.0f - F) / abs(NDotL) * (1.0f - METALLIC) * TRANSMISSION);
        } else {
            // specular lobe
            const float a  = ROUGHNESS;
            const float Ds = GTR2(NDotH, a);

            // Fresnel term with the microfacet normal
            const float FH     = Fr(LDotH, ETA);
            const glm::vec3 Fs = glm::lerp(Cspec0, glm::vec3(1.0f), FH);
            const float Gs     = SmithGGX(NDotV, a) * SmithGGX(NDotL, a);
            bsdf               = (Gs * Ds) * Fs;
        }
    }
    if (TRANSMISSION < 1.0f) {
        // evaluate BRDF
        if (NDotL <= 0) {
            if (SUBSURFACE > 0.0f) {
                // take sqrt to account for entry/exit of the ray through the medium
                // this ensures transmitted light corresponds to the diffuse model
                const glm::vec3 s =
                    glm::vec3(sqrtf(shadingData.color.x), sqrtf(shadingData.color.y), sqrtf(shadingData.color.z));
                const float FL = SchlickFresnel(abs(NDotL)), FV = SchlickFresnel(NDotV);
                const float Fd = (1.0f - 0.5f * FL) * (1.0f - 0.5f * FV);
                brdf           = INV_PI * s * SUBSURFACE * Fd * (1.0f - METALLIC);
            }
        } else {
            // specular
            const float a  = ROUGHNESS;
            const float Ds = GTR2(NDotH, a);

            // Fresnel term with the microfacet normal
            const float FH     = SchlickFresnel(LDotH);
            const glm::vec3 Fs = glm::lerp(Cspec0, glm::vec3(1), FH);
            const float Gs     = SmithGGX(NDotV, a) * SmithGGX(NDotL, a);

            // Diffuse fresnel - go from 1 at normal incidence to .5 at grazing
            // and mix in diffuse retro-reflection based on roughness
            const float FL = SchlickFresnel(NDotL), FV = SchlickFresnel(NDotV);
            const float Fd90 = 0.5 + 2.0f * LDotH * LDotH * a;
            const float Fd   = glm::lerp(1.0f, Fd90, FL) * glm::lerp(1.0f, Fd90, FV);

            // clearcoat (ior = 1.5 -> F0 = 0.04)
            const float Dr = GTR1(NDotH, glm::lerp(.1f, .001f, CLEARCOATGLOSS));
            const float Fc = glm::lerp(.04f, 1.0f, FH);
            const float Gr = SmithGGX(NDotL, .25) * SmithGGX(NDotV, .25);

            brdf =
                INV_PI * Fd * Cdlin * (1.0f - METALLIC) * (1.0f - SUBSURFACE) + Gs * Fs * Ds + CLEARCOAT * Gr * Fc * Dr;
        }
    }

    return lerp(brdf, bsdf, TRANSMISSION);
}

// generate an importance sampled BSDF direction
__device__ static void BSDFSample(const ShadingData &shadingData,
                                  const glm::vec3 &T,
                                  const glm::vec3 &B,
                                  const glm::vec3 &N,
                                  const glm::vec3 &wo,
                                  glm::vec3 &wi,
                                  float &pdf,
                                  BSDFType &type,
                                  const float r3,
                                  const float r4)
{
    if (r3 < TRANSMISSION) {
        // sample BSDF
        float F = Fr(dot(N, wo), ETA);
        if (r4 < F)  // sample reflectance or transmission based on Fresnel term
        {
            // sample reflection
            const float r1           = r3 / TRANSMISSION;
            const float r2           = r4 / F;
            const float cosThetaHalf = sqrtf((1.0f - r2) / (1.0f + (sqr(ROUGHNESS) - 1.0f) * r2));
            const float sinThetaHalf = sqrtf(max(0.0f, 1.0f - sqr(cosThetaHalf)));
            float sinPhiHalf, cosPhiHalf;
            __sincosf(r1 * TWOPI, &sinPhiHalf, &cosPhiHalf);
            glm::vec3 half = T * (sinThetaHalf * cosPhiHalf) + B * (sinThetaHalf * sinPhiHalf) + N * cosThetaHalf;
            if (dot(half, wo) <= 0.0f)
                half *= -1.0f;  // ensure half angle in same hemisphere as wo
            type = eReflected;
            wi   = reflect(wo * -1.0f, half);
        } else  // sample transmission
        {
            pdf = 0;
            if (Refract(wo, N, ETA, wi))
                type = eSpecular, pdf = (1.0f - F) * TRANSMISSION;
            return;
        }
    } else  // sample BRDF
    {
        const float r1 = (r3 - TRANSMISSION) / (1 - TRANSMISSION);
        if (r4 < 0.5f) {
            // sample diffuse
            const float r2 = r4 * 2;
            glm::vec3 d;
            if (r2 < SUBSURFACE) {
                const float r5 = r2 / SUBSURFACE;
                d = DiffuseReflectionUniform(r1, r5), type = eTransmitted, d.z *= -1.0f;
            } else {
                const float r5 = (r2 - SUBSURFACE) / (1 - SUBSURFACE);
                d = DiffuseReflectionCosWeighted(r1, r5), type = eReflected;
            }
            wi = T * d.x + B * d.y + N * d.z;
        } else {
            // sample specular
            const float r2           = (r4 - 0.5f) * 2.0f;
            const float cosThetaHalf = sqrtf((1.0f - r2) / (1.0f + (sqr(ROUGHNESS) - 1.0f) * r2));
            const float sinThetaHalf = sqrtf(max(0.0f, 1.0f - sqr(cosThetaHalf)));
            float sinPhiHalf, cosPhiHalf;
            __sincosf(r1 * TWOPI, &sinPhiHalf, &cosPhiHalf);
            glm::vec3 half = T * (sinThetaHalf * cosPhiHalf) + B * (sinThetaHalf * sinPhiHalf) + N * cosThetaHalf;
            if (dot(half, wo) <= 0.0f)
                half *= -1.0f;  // ensure half angle in same hemisphere as wi
            wi   = reflect(wo * -1.0f, half);
            type = eReflected;
        }
    }
    pdf = BSDFPdf(shadingData, N, wo, wi);
}

// ----------------------------------------------------------------

__device__ static glm::vec3 EvaluateBSDF(const ShadingData &shadingData,
                                         const glm::vec3 &iN,
                                         const glm::vec3 &T,
                                         const glm::vec3 wo,
                                         const glm::vec3 wi,
                                         float &pdf)
{
    const glm::vec3 bsdf = BSDFEval(shadingData, iN, wo, wi);
    pdf                  = BSDFPdf(shadingData, iN, wo, wi);
    return bsdf;
}

__device__ static glm::vec3 SampleBSDF(const ShadingData &shadingData,
                                       const glm::vec3 &iN,
                                       const glm::vec3 &N,
                                       const glm::vec3 &T,
                                       const glm::vec3 &wo,
                                       const float r3,
                                       const float r4,
                                       glm::vec3 &wi,
                                       float &pdf)
{
    BSDFType type;
    const glm::vec3 B      = normalize(cross(T, iN));
    const glm::vec3 Tfinal = cross(B, iN);
    BSDFSample(shadingData, Tfinal, B, iN, wo, wi, pdf, type, r3, r4);
    return BSDFEval(shadingData, iN, wo, wi);
}
#endif

}