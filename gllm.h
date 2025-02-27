
// genmod.h: header for main.cpp
#ifndef GLLM_H
#define GLLM_H 1

/**
// fir parallel computation
#ifdef HAVE_CUDA
    #include <cuda_runtime.h>
#elif defined(__APPLE__)
    #include <OpenCL/opencl.h>
#else
    #include <CL/cl.hpp>
#endif
 */

// self defined headers
#include <maths.h>
#include <neural.h>
#include <model.h>

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