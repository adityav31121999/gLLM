
// genmod.h: header for main.cpp
#ifndef GLLM_H
#define GLLM_H 1

// 3rd party headers
#include <stdio.h>
#include <stdlib.h>

#ifdef __APPLE__
    #include <OpenCL/opencl.h>
#else
    #include <CL/cl.hpp>
#endif

// self defined headers
#include <maths.hpp>
#include <neural.hpp>
#include <model.hpp>
#include <script.hpp>

#endif
