
// genmod.h: header for main.cpp
#ifndef GLLM_H
#define GLLM_H 1

// self defined headers
#include <memory_map.h>
#include <vector>
#include <maths.hpp>            // math functions for ai/ml
#include <neural.hpp>           // LLM related classes
#include <mod.hpp>            // model related operations

#ifdef USE_OPENCL
#include <CL/cl.hpp> // Required for OpenCL C++ bindings, e.g., cl::BuildError
    extern std::vector<std::string> kernelSourceFiles;
    extern std::vector<std::string> kernelNames;
#endif

#endif
