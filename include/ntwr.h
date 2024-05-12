#pragma once

#include <sstream>
#include <stdexcept>

#define NTWR_KERNEL              __global__
#define NTWR_HOST_DEVICE         inline __host__ __device__
#define NTWR_HOST                __host__
#define NTWR_DEVICE              inline __device__
#define NTWR_DEVICE_FORCE_INLINE __forceinline__ __device__

[[noreturn]] NTWR_HOST_DEVICE void UNREACHABLE()
{
    // Uses compiler specific extensions if possible.
    // Even if no extension is used, undefined behavior is still raised by
    // an empty function body and the noreturn attribute.
#ifdef __GNUC__  // GCC, Clang, ICC
    __builtin_unreachable();
#elif _MSC_VER  // MSVC
    __assume(false);
#endif
}

// disable default lib for linker due to the potential issue
// LINK : warning LNK4098: defaultlib 'LIBCMT' conflicts with use of other libs
#if defined(WIN32)
#pragma comment(linker, "/NODEFAULTLIB:libcmt.lib")
#endif

#define CUDA_CHECK(call)                                                                             \
    {                                                                                                \
        cudaError_t rc = call;                                                                       \
        if (rc != cudaSuccess) {                                                                     \
            std::stringstream txt;                                                                   \
            cudaError_t err = rc; /*cudaGetLastError();*/                                            \
            txt << "CUDA Error " << cudaGetErrorName(err) << " (" << cudaGetErrorString(err) << ")"; \
            throw std::runtime_error(txt.str());                                                     \
        }                                                                                            \
    }

#define CUDA_CHECK_NOEXCEPT(call) \
    {                             \
        call;                     \
    }

#define CUDA_SYNC_CHECK()                                                                                \
    {                                                                                                    \
        cudaDeviceSynchronize();                                                                         \
        cudaError_t error = cudaGetLastError();                                                          \
        if (error != cudaSuccess) {                                                                      \
            fprintf(stderr, "error (%s: line %d): %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(2);                                                                                     \
        }                                                                                                \
    }
