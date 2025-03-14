
// genmod.h: header for main.cpp
#ifndef GLLM_H
#define GLLM_H 1

// for parallel computation
/**
 * if LINUX
 *  - use CUDA or OpenCL if available
 *  - else use CPU
 * if WINDOWS
 *  - use CUDA or OpenCL if available
 *  - else use CPU
 * if APPLE
 *  - use CUDA or OpenCL if available
 *  - else use CPU
 */

// self defined headers
#include <maths.hpp>
#include <neural.hpp>
#include <model.hpp>
#include <script.hpp>

#endif


/**
find_package(CUDAToolkit REQUIRED)
if(CUDA_FOUND)
    set(CMAKE_CUDA_ARCHITECTURES 75)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
endif()

set(OpenCL_CL_VERSION "300") 
set(CL_TARGET_OPENCL_VERSION "300")
find_package(OpenCL REQUIRED)
if(OpenCL_FOUND)
    include_directories(${OpenCL_INCLUDE_DIRS})
endif()
 */