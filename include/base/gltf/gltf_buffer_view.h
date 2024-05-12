#pragma once

#include <stdexcept>
#include "gltf_types.h"
#include "tiny_gltf.h"

namespace ntwr {

// GLTF Buffer view/accessor utilities (source code from ChameleonRT)

struct BufferViewWrapper {
    const uint8_t *buf = nullptr;
    size_t length      = 0;
    size_t stride      = 0;

    BufferViewWrapper() = default;

    BufferViewWrapper(const tinygltf::BufferView &view, const tinygltf::Model &model, size_t base_stride);

    BufferViewWrapper(const uint8_t *buf, size_t byte_length, size_t byte_stride);

    // Return the pointer to some element, based on the stride specified for the view
    const uint8_t *operator[](const size_t i) const;
};

template <typename T>
class AccessorWrapper {
    BufferViewWrapper view;
    size_t count = 0;

public:
    AccessorWrapper() = default;

    AccessorWrapper(const tinygltf::Accessor &accessor, const tinygltf::Model &model);

    AccessorWrapper(const BufferViewWrapper &view);

    const T &operator[](const size_t i) const;

    // Note: begin/end require the buffer to be tightly packed
    const T *begin() const;

    const T *end() const;

    size_t size() const;
};

template <typename T>
AccessorWrapper<T>::AccessorWrapper(const tinygltf::Accessor &accessor, const tinygltf::Model &model)
    : view(model.bufferViews[accessor.bufferView], model, gltf_base_stride(accessor.type, accessor.componentType)),
      count(accessor.count)
{
    // Apply the additional accessor-specific byte offset
    view.buf += accessor.byteOffset;
}

template <typename T>
AccessorWrapper<T>::AccessorWrapper(const BufferViewWrapper &view) : view(view), count(view.length / view.stride)
{
}

template <typename T>
const T *AccessorWrapper<T>::begin() const
{
    if (view.length / view.stride != count) {
        throw std::runtime_error("Accessor<T>::begin cannot be used on non-packed buffer");
    }
    return reinterpret_cast<const T *>(view[0]);
}

template <typename T>
const T *AccessorWrapper<T>::end() const
{
    if (view.length / view.stride != count) {
        throw std::runtime_error("Accessor<T>::end cannot be used on non-packed buffer");
    }
    return reinterpret_cast<const T *>(view[count]);
}

template <typename T>
const T &AccessorWrapper<T>::operator[](const size_t i) const
{
    return *reinterpret_cast<const T *>(view[i]);
}

template <typename T>
size_t AccessorWrapper<T>::size() const
{
    return count;
}

}