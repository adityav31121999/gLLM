
#ifndef OPENCL_SETUP_HPP
#define OPENCL_SETUP_HPP

#ifndef CL_HPP_ENABLE_EXCEPTIONS
    #define CL_HPP_ENABLE_EXCEPTIONS
#endif

#ifndef CL_HPP_TARGET_OPENCL_VERSION
    #define CL_HPP_TARGET_OPENCL_VERSION 300 // Or your target version (e.g., 120, 200)
#endif

#ifndef CL_HPP_MINIMUM_OPENCL_VERSION
    #define CL_HPP_MINIMUM_OPENCL_VERSION 120 // Set to the minimum your code requires
#endif

#include <CL/cl.hpp> // Use the C++ bindings header

#pragma OPENCL EXTENSION cl_khr_fp32 : enable // For float precision
#pragma OPENCL EXTENSION cl_khr_int32_base_atomics : enable // For atomic operations if used
// #pragma OPENCL EXTENSION cl_khr_fp64 : enable // For float precision
// #pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable // For atomic operations if used

// Declare the global OpenCL objects that other files will use via 'extern'
// These will be defined in opencl_setup.cpp
extern cl::Context context;
extern cl::CommandQueue queue;
extern cl::Program program;
extern cl::Device default_device; // The device selected for execution

std::string readFile(const std::string& filePath);

/**
 * @brief Initializes the OpenCL environment, loads and builds kernels.
 *
 * This function performs the following steps:
 * 1. Selects an OpenCL platform and device (preferring GPU).
 * 2. Creates an OpenCL context for the selected device.
 * 3. Creates an OpenCL command queue for the device and context.
 * 4. Reads kernel source code from a predefined list of .cl files.
 * 5. Concatenates the source code.
 * 6. Creates an OpenCL program object from the concatenated source.
 * 7. Builds (compiles) the program for the selected device.
 * 8. Assigns the created objects to the global 'context', 'queue',
 *    'program', and 'default_device' variables.
 *
 * @throws std::runtime_error if any step of the initialization fails (e.g.,
 *         no suitable OpenCL devices found, kernel file not found, kernel
 *         compilation error). Build logs are printed to stderr on failure.
 */
void initializeOpenCL();

#endif // OPENCL_SETUP_HPP
