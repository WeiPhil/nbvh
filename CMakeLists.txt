cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
project(ntwr C CXX CUDA)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(CUDA_DEBUG "Build cuda files with -G for nsight debugging" OFF)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
else()
  # Hack to force tiny-cuda-nn to use the provided cuda architecture
  set(ENV{TCNN_CUDA_ARCHITECTURES} ${CMAKE_CUDA_ARCHITECTURES})
endif()

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

set(CMAKE_CUDA_EXTENSIONS OFF)
set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

# ======== Tinyformat ======== #
include_directories(ext/tinyformat)

# ======== Color Diagnostic ========= #

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    add_compile_options (-fdiagnostics-color=always)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    # add_compile_options (-fcolor-diagnostics)
endif ()

# ====== CUDA and import gdt submodule helper with macros for cuda compilation ====== 
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake/")

set(CUDA_COMPILE_DEFNS "-lineinfo;-O3;--use_fast_math;-rdc=true")

# ====== Tiny Cuda NN ======== 
set(TCNN_BUILD_BENCHMARK OFF CACHE BOOL "disable benchmarks")
set(TCNN_BUILD_EXAMPLES OFF CACHE BOOL "disable examples")

add_subdirectory(ext/tiny-cuda-nn)

# Make sure we match the CUDA ARCHITECTURE CHOSEN BY tiny cuda nn
set(CMAKE_CUDA_ARCHITECTURES ${TCNN_CUDA_ARCHITECTURES})

# ====== OpenGL ========
find_package(OpenGL REQUIRED) 

# ====== GLFW ======== 
add_subdirectory(ext/glfw) 

# ====== IMGUI ======== 
# (1.89.5 + Spectrum)
add_subdirectory(ext/imgui) 

# ====== Tiny obj loader ========
include_directories(ext/tinyobjloader)

# ====== Tiny gltf loader / STB ======== 
include_directories(ext/tinygltf)

# ====== TinyEXR ======== 
add_library(tinyexr STATIC ext/tinyexr/tinyexr.cpp)
target_compile_definitions(tinyexr PUBLIC TINYEXR_USE_MINIZ=0 TINYEXR_USE_STB_ZLIB=1)
include_directories(ext/tinyexr)

# ====== GLAD ======== 
add_subdirectory(ext/glad)

# ====== LibBVH ======== 
add_subdirectory(ext/bvh)

# ====== LibBVH ========
include_directories(${LIBBVH_INCLUDE_DIRS})

# ====== GLM ======== 
add_definitions(-DGLM_ENABLE_EXPERIMENTAL)
add_subdirectory(ext/glm) 

# ====== FLIP ======== 
set(FLIP_LIBRARY ON)
add_subdirectory(ext/flip-cuda)
include_directories(ext/flip-cuda)

# ============= Project source files ==========

if(MSVC)
    # Force to always compile with W4
    if(CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
        string(REGEX REPLACE "/W[0-4]" "/W4" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
    endif()
elseif(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
    # Update if necessary
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-long-long -pedantic")
endif()


# Setting cpp files per directory
set(SRC_DIR src) 
file(GLOB_RECURSE CPP_FILES ${SRC_DIR}/*.cpp)
file(GLOB_RECURSE CU_FILES ${SRC_DIR}/*.cu)

set(SRC_FILES ${CPP_FILES} ${CU_FILES})

set(NTWR_INCLUDE_DIRECTORIES include)

# Adding source files to project executable
add_executable(ntwr
  ${SRC_FILES}
)

target_include_directories(ntwr PUBLIC ${NTWR_INCLUDE_DIRECTORIES} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

set_target_properties(ntwr PROPERTIES
                      CUDA_SEPARABLE_COMPILATION ON)

# -g : Generate debug information for host code.
# --generate-line-info : Generate line-number information for device code
if (MSVC)
	list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=/bigobj")
else()
	list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-Wno-float-conversion")
	list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-fno-strict-aliasing")
	list(APPEND CUDA_NVCC_FLAGS "-Xcudafe=--diag_suppress=unrecognized_gcc_pragma")
endif()
list(APPEND CUDA_NVCC_FLAGS "--extended-lambda")
if(CUDA_DEBUG)
  list(APPEND CUDA_NVCC_FLAGS  "-G")
else()
  list(APPEND CUDA_NVCC_FLAGS "-g")
  list(APPEND CUDA_NVCC_FLAGS "--generate-line-info")
endif()
list(APPEND CUDA_NVCC_FLAGS "--expt-relaxed-constexpr")
list(APPEND CUDA_NVCC_FLAGS "-Xcudafe=--diag_suppress=esa_on_defaulted_function_ignored") # Remove annoying glm warnings

target_compile_options(ntwr PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)

# Remove any min and max macros defined by windows 
if(MSVC)
  target_compile_definitions(ntwr PRIVATE NOMINMAX _USE_MATH_DEFINES)
endif()

if(UNIX)
  include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
endif()

# Precompile glm header which are often included
target_precompile_headers(ntwr PRIVATE include/glm_pch.h)
 
target_link_libraries(ntwr PUBLIC
  flip-cuda  
  bvh
  glfw
  glad
  imgui
  tiny-cuda-nn
  glm::glm
  tinyexr
  ${CUDA_LIBRARIES}
  # for cuCtxGetCurrent
  ${CUDA_CUDA_LIBRARY}
  ${CUDA_RT}
)
