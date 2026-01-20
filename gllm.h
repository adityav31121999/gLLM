#ifndef GLLM_H
#define GLLM_H 1

// self defined headers
#include <memory_map.h>
#include <vector>
#include <maths.hpp>            // math functions for ai/ml
#include <neural.hpp>           // LLM related classes
#include <model.hpp>            // model related operations

#ifdef USE_CL
    #if defined(_WIN64)
        #include <CL/cl.hpp>
    #elif defined(__linux__)
        #include <CL/opencl.hpp>
    #endif
    extern std::vector<std::string> kernelSourceFiles;
    extern std::vector<std::string> kernelNames;
#endif

#endif
