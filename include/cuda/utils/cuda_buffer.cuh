#pragma once

#include <vector>
#include <assert.h>
#include "utils/logger.h"

namespace ntwr {

/*! simple wrapper for creating, and managing a device-side CUDA
    buffer */
template <typename T>
struct CUDABuffer {
    inline T *d_pointer() const
    {
        return (T *)d_ptr;
    }

    // Assigns an already allocated device pointer
    void assign_device_ptr(T *device_pointer, size_t count)
    {
        size_in_bytes = count * sizeof(T);
        d_ptr         = (void *)device_pointer;
    }

    //! re-size buffer to given number of bytes
    void resize(size_t size)
    {
        if (d_ptr)
            free();
        alloc(size);
    }

    //! allocate to given number of bytes
    void alloc(size_t count)
    {
        assert(d_ptr == nullptr);
        this->size_in_bytes = count * sizeof(T);
        CUDA_CHECK(cudaMalloc((void **)&d_ptr, this->size_in_bytes));
    }

    //! free allocated memory
    void free()
    {
        CUDA_CHECK(cudaFree(d_ptr));
        d_ptr         = nullptr;
        size_in_bytes = 0;
    }

    void alloc_and_upload(const std::vector<T> &vt)
    {
        alloc(vt.size());
        upload((const T *)vt.data(), vt.size());
    }

    void alloc_and_upload(const T &vt)
    {
        alloc(1);
        upload((const T *)vt, 1);
    }

    T *upload(const T *t, size_t count, size_t offset = 0)
    {
        assert(d_ptr != nullptr);
        assert(size_in_bytes >= (count + offset) * sizeof(T));

        // Calculate the destination pointer with the offset
        T *offset_d_ptr = (T *)d_ptr + offset;

        CUDA_CHECK(cudaMemcpy(offset_d_ptr, (void *)t, count * sizeof(T), cudaMemcpyHostToDevice));
        return offset_d_ptr;
    }

    void download(T *t, size_t count, bool count_can_be_less = false)
    {
        assert(d_ptr != nullptr);
        assert((count_can_be_less && size_in_bytes >= count * sizeof(T)) || size_in_bytes == count * sizeof(T));
        CUDA_CHECK(cudaMemcpy((void *)t, d_ptr, count * sizeof(T), cudaMemcpyDeviceToHost));
    }

    size_t num_elements() const
    {
        return this->size_in_bytes / sizeof(T);
    }

    size_t size_in_bytes{0};
    void *d_ptr{nullptr};
};

}
