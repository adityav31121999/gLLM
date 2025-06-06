
// genmod.h: header for main.cpp
#ifndef GLLM_H
#define GLLM_H 1

// self defined headers
#include <memory_map.h>
#include <vector>
#include <maths.hpp>            // math functions for ai/ml
#include <neural.hpp>           // LLM related classes
#include <mod.hpp>            // model related operations

#ifdef USE_CUDA
    #include <cuda.h>
    #include <cuda_runtime.h>
#elif USE_OPENCL
    #define CL_HPP_ENABLE_EXCEPTIONS // Define before including cl.hpp
    #define CL_HPP_TARGET_OPENCL_VERSION 300
    #include <CL/cl.hpp>

    extern std::vector<std::string> kernelSourceFiles;
    extern std::vector<std::string> kernelNames;

#endif

void printCrown();
void printBase();

#endif
