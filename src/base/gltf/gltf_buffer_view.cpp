

#include "base/gltf/gltf_buffer_view.h"

namespace ntwr {

BufferViewWrapper::BufferViewWrapper(const tinygltf::BufferView &view, const tinygltf::Model &model, size_t base_stride)
    : buf(model.buffers[view.buffer].data.data() + view.byteOffset),
      length(view.byteLength),
      stride(std::max(view.byteStride, base_stride))
{
}

BufferViewWrapper::BufferViewWrapper(const uint8_t *buf, size_t byte_length, size_t byte_stride)
    : buf(buf), length(byte_length), stride(byte_stride)
{
}

const uint8_t *BufferViewWrapper::operator[](const size_t i) const
{
    return buf + i * stride;
}

}