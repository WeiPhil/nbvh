#pragma once

#include "cuda/utils/common.cuh"

namespace ntwr {

template <glm::length_t D>
struct Bbox {
    glm::vec<D, float> min;
    glm::vec<D, float> max;

    NTWR_HOST_DEVICE glm::vec<D, float> extent() const
    {
        return max - min;
    }

    NTWR_HOST_DEVICE glm::vec<D, float> center() const
    {
        return (max + min) * 0.5f;
    }

    NTWR_HOST_DEVICE float smallest_distance_from(const glm::vec<D, float> &point)
    {
        float value = 0.f;
        for (size_t i = 0; i < D; i++) {
            float dx = glm::max(glm::max(min[i] - point[i], point[i] - max[i]), 0.0f);
            value += dx * dx;
        }
        return glm::sqrt(glm::max(0.f, value));
    }

    /// Return the position of a bounding box corner (index = [0,7] for Bbox3)
    NTWR_HOST_DEVICE glm::vec<D, float> corner(size_t index) const
    {
        glm::vec<D, float> result;
        for (uint32_t i = 0; i < D; ++i)
            result[i] = ((uint32_t)index & (1 << i)) ? max[i] : min[i];
        return result;
    }

    /*! returns new box including both ourselves _and_ the given point */
    NTWR_HOST_DEVICE Bbox<D> &expand(const glm::vec<D, float> &other)
    {
        min = glm::min(min, other);
        max = glm::max(max, other);
        return *this;
    }

    /*! returns new box including both ourselves _and_ the given point */
    NTWR_HOST_DEVICE Bbox<D> &expand(const Bbox<D> &other)
    {
        min = glm::min(min, other.min);
        max = glm::max(max, other.max);
        return *this;
    }

    /*! returns new box including both ourselves _and_ the given point */
    NTWR_HOST_DEVICE float area()
    {
        glm::vec<D, float> d = max - min;

        float result = 0.f;
        for (int i = 0; i < D; ++i) {
            float term = 1.f;
            for (int j = 0; j < D; ++j) {
                if (i == j)
                    continue;
                term *= d[j];
            }
            result += term;
        }
        return result * 2.f;
    }

    NTWR_HOST_DEVICE glm::vec<D, float> bbox_to_local(glm::vec<D, float> world_pos) const
    {
        glm::vec<D, float> local_pos = (world_pos - this->min) / extent();
        return local_pos;
    }

    NTWR_HOST_DEVICE glm::vec<D, float> local_to_bbox(glm::vec<D, float> local_pos) const
    {
        glm::vec<D, float> world_pos = local_pos * extent() + this->min;
        return world_pos;
    }

    NTWR_HOST_DEVICE Bbox<D> ellipsoid_bbox() const
    {
        glm::vec<D, float> half_axes = extent() * (sqrt(3.f) * 0.5f);
        return Bbox<D>{center() - half_axes, center() + half_axes};
    }
};

typedef Bbox<3> Bbox3f;
typedef Bbox<2> Bbox2f;

enum CubeFace {
    XPlus  = 0,
    XMinus = 1,
    YPlus  = 2,
    YMinus = 3,
    ZPlus  = 4,
    ZMinus = 5,
    SIZE   = 6,
};
}