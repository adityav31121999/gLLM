
// basic.hpp: header source of basic library
#ifndef BASIC_HPP
#define BASIC_HPP 1

#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <functional>
#include <atomic>
#include <random>

// Define MAXFLOAT if not implicitly available
#ifndef MAXFLOAT
    #define MAXFLOAT 3.402823466e+38F
#endif

// vect.cpp

bool operator==(std::vector<float>, std::vector<float>);
bool operator!=(std::vector<float>, std::vector<float>);
std::vector<float> operator+(std::vector<float>, std::vector<float>);
std::vector<float> operator-(std::vector<float>, std::vector<float>);
std::vector<float> operator*(std::vector<float>, float);
std::vector<float> operator*(float, std::vector<float>);
std::vector<float> operator/(std::vector<float>, float);
std::vector<float> operator+=(std::vector<float>, std::vector<float>);
std::vector<float> operator-=(std::vector<float>, std::vector<float>);
std::vector<float> operator*=(std::vector<float>, float);
std::vector<float> operator*=(float, std::vector<float>);
std::vector<float> operator/=(std::vector<float>, float);
std::vector<std::vector<float>> operator+(std::vector<std::vector<float>>, std::vector<std::vector<float>>);
std::vector<std::vector<float>> operator-(std::vector<std::vector<float>>, std::vector<std::vector<float>>);
std::vector<std::vector<float>> operator*(std::vector<std::vector<float>>, float y);
std::vector<std::vector<float>> operator/(std::vector<std::vector<float>>, float y);
std::vector<std::vector<float>> operator+=(std::vector<std::vector<float>>, std::vector<std::vector<float>>);
std::vector<std::vector<float>> operator-=(std::vector<std::vector<float>>, std::vector<std::vector<float>>);
std::vector<std::vector<float>> operator*=(std::vector<std::vector<float>>, float y);
std::vector<std::vector<float>> operator/=(std::vector<std::vector<float>>, float y);

float errorofv(std::vector<float>&, std::vector<float>&);
float gradientdesc1(std::vector<float>, std::vector<float>);
float vdotv2val(std::vector<float>, std::vector<float>);
float vdotv2scal(std::vector<float> , std::vector<float>);
float MAE(std::vector<float>&, std::vector<float>&);
float MSE(std::vector<float>&, std::vector<float>&);
float crossEntropy(std::vector<float>&, std::vector<float>&);
float sum(std::vector<float>);
float sum(std::vector<std::vector<float>>);
float product(std::vector<float>);
float product(std::vector<std::vector<float>>);

std::vector<float> error(std::vector<float>&, std::vector<float>&);
std::vector<float> percenterrorofvec(std::vector<float> , std::vector<float>);
std::vector<float> gradient_descent(std::vector<float>, std::vector<float>, float);
std::vector<float> power(std::vector<float>, float);
std::vector<float> sumofrow(std::vector<std::vector<float>>);
std::vector<float> sumofcol(std::vector<std::vector<float>>);
std::vector<float> vxv2v(std::vector<float>, std::vector<float>);
std::vector<float> vdotv2v(std::vector<float>, std::vector<float>);
std::vector<float> vxmat2vec(std::vector<float>, std::vector<std::vector<float>>);
std::vector<float> mat2vec(std::vector<std::vector<float>>);
std::vector<std::vector<float>> vec2mat(std::vector<float>, unsigned int, unsigned int);
std::vector<std::vector<float>> vdotmat2mat(std::vector<float>, std::vector<std::vector<float>>);
std::vector<std::vector<float>> vxv2mat(std::vector<float>, std::vector<float>);
std::vector<std::vector<float>> iproduct(std::vector<std::vector<float>>);
std::vector<std::vector<float>> power(std::vector<std::vector<float>>, float);

// activations.cpp

float sigmoid(const float& x);
float sigmoidder(const float& x);
std::vector<float> sigmoid(const std::vector<float>& x);
std::vector<float> sigmoidder(const std::vector<float>& x);
std::vector<std::vector<float>> sigmoid(const std::vector<std::vector<float>>& x);
std::vector<std::vector<float>> sigmoidder(const std::vector<std::vector<float>>& x);
std::vector<float> softmax(const std::vector<float>& x, float temp);
std::vector<float> softmaxder(const std::vector<float>& x, float temp);
std::vector<std::vector<float>> softmax(const std::vector<std::vector<float>>& x, float temp);
std::vector<std::vector<float>> softmaxder(const std::vector<std::vector<float>>& x, float temp);
float ReLU(const float& x);
float ReLUder(const float& x);
std::vector<float> ReLU(const std::vector<float>& x);
std::vector<float> ReLUder(const std::vector<float>& x);
std::vector<std::vector<float>> ReLU(const std::vector<std::vector<float>>& x, int& t);
std::vector<std::vector<float>> ReLUder(const std::vector<std::vector<float>>& x, int& t);
std::vector<float> LOTA(const std::vector<float>& y);
std::vector<float> LOTAder(const std::vector<float>& y);
std::vector<std::vector<float>> LOTA(const std::vector<std::vector<float>>& y, int& t, bool& attentionType);
std::vector<std::vector<float>> LOTAder(const std::vector<std::vector<float>>& y, int& t, bool& attentionType);

// weights.cpp

void randomweights(std::vector<std::vector<float>>);
void jumbledwbs(std::vector<std::vector<float>>);
void ijbasedwbs(std::vector<std::vector<float>>);
void Random(std::vector<std::vector<float>>);


#ifdef USE_CUDA

#include <cuda_runtime.h>

__global__ void cuSigmoid(float x, float* result);
__global__ void cuSigmoid(float* x, float* out, int size);
__global__ void cuSigmoid(float* x, float* out, int rows, int cols);
__global__ void cuSoftmax(const float* __restrict__ x, float* __restrict__ out, float temp, int size);
__global__ void cuSoftmax(const float* __restrict__ x, float* __restrict__ out, float temp, int rows, int cols);
__global__ void cuReLU(float x, float* result);
__global__ void cuReLU(float* x, float* out, int size);
__global__ void cuLOTA(float* y, float* out, int size);
__global__ void cuLOTA(float* y, float* out, int rows, int cols);
__global__ void cuLOTA(float* y, float* out, int rows, int cols, int limit, bool attentionType);

__global__ void cuSigmoidder(float x, float* result);
__global__ void cuSigmoidder(float* x, float* out, int rows, int cols);
__global__ void cuSoftmaxder(float* x, float* out, float temp, int size);
__global__ void cuSoftmaxder(float* x, float* out, float temp, int rows, int cols);
__global__ void cuReLUder(float x, float* result);
__global__ void cuReLUder(float* x, float* out, int size);
__global__ void cuLOTAder(float* y, float* out, int size);
__global__ void cuLOTAder(float* y, float* out, int rows, int cols);
__global__ void cuLOTAder(float* y, float* out, int rows, int cols, int limit, bool attentionType);


__global__ void operator_add(const float* a, const float* b, float* result, int size);
__global__ void operator_sub(const float* a, const float* b, float* result, int size);
__global__ void operator_mul(const float* a, float scalar, float* result, int size);
__global__ void operator_mul_reverse(float scalar, const float* a, float* result, int size);
__global__ void operator_div(const float* a, float scalar, float* result, int size);
__global__ void operator_add_2d(const float* a, const float* b, float* result, int rows, int cols);
__global__ void operator_sub_2d(const float* a, const float* b, float* result, int rows, int cols);
__global__ void operator_mul_2d(const float* a, float scalar, float* result, int rows, int cols);
__global__ void operator_div_2d(const float* a, float scalar, float* result, int rows, int cols);

__global__ void errorofv(const float* a, const float* b, float* result, int size);
__global__ void gradientdesc(const float* a, const float* b, float* result, int size);
__global__ void vdotv2val(const float* a, const float* b, float* result, int size);
__global__ void vdotv2scal(const float* a, const float* b, float* result, int size);
__global__ void MSE(const float* a, const float* b, float* result, int size);
__global__ void sum(const float* a, float* result, int size);
__global__ void sum_2d(const float* a, float* result, int rows, int cols);
__global__ void product(const float* a, float* result, int size);
__global__ void product_2d(const float* a, float* result, int rows, int cols);

__device__ float compute_dot_product(const float* vec1, const float* vec2, int dim);
__global__ void matrixMultiplyKernel(const float* A, const float* B, float* C, int rowsA, int colsA, int colsB);
__global__ void vectorAddKernel(const float* A, const float* B, float* C, int len);

#elif USE_OPENCL

// Conditional inclusion of OpenCL C++ header based on OS
#if defined(_WIN64)
    #define CL_HPP_ENABLE_EXCEPTIONS
    #define CL_HPP_TARGET_OPENCL_VERSION 300
    // For Windows, use the older/common cl.hpp
    #include <CL/cl.hpp>
#elif defined(__linux__)
    #define CL_HPP_TARGET_OPENCL_VERSION 220
    #define CL_TARGET_OPENCL_VERSION 220 // Inform C headers to target OpenCL 2.2
    #include <CL/opencl.hpp>
#endif
#include <sstream>
#include <string>
#include <fstream>
#include <stdexcept>
#include <map>
#include <set>
#include <vector> // Ensure vector is included

#define CL_CHECK(call)                                                      \
    do {                                                                        \
        cl_int err_code_ = call;                                                \
        if (err_code_ != CL_SUCCESS) {                                          \
            std::string error_message_ = "OpenCL API Error in ";                \
            error_message_ += __FILE__;                                         \
            error_message_ += " at line " + std::to_string(__LINE__) + ": ";    \
            error_message_ += oclErrorString(err_code_);                        \
            error_message_ += " (" + std::to_string(err_code_) + ")";           \
            throw std::runtime_error(error_message_);                           \
        }                                                                       \
    } while (0)

// --- OpenCL Error String Helper ---
// (Add this function definition before the CL_CHECK macro)
inline const char* oclErrorString(cl_int error) {
    switch (error) {
        // Run-time and JIT compiler errors
        case 0: return "CL_SUCCESS";
        case -1: return "CL_DEVICE_NOT_FOUND";
        case -2: return "CL_DEVICE_NOT_AVAILABLE";
        case -3: return "CL_COMPILER_NOT_AVAILABLE";
        case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case -5: return "CL_OUT_OF_RESOURCES";
        case -6: return "CL_OUT_OF_HOST_MEMORY";
        case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case -8: return "CL_MEM_COPY_OVERLAP";
        case -9: return "CL_IMAGE_FORMAT_MISMATCH";
        case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case -11: return "CL_BUILD_PROGRAM_FAILURE";
        case -12: return "CL_MAP_FAILURE";
        case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
        case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
        case -15: return "CL_COMPILE_PROGRAM_FAILURE";
        case -16: return "CL_LINKER_NOT_AVAILABLE";
        case -17: return "CL_LINK_PROGRAM_FAILURE";
        case -18: return "CL_DEVICE_PARTITION_FAILED";
        case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

        // Compile-time errors
        case -30: return "CL_INVALID_VALUE";
        case -31: return "CL_INVALID_DEVICE_TYPE";
        case -32: return "CL_INVALID_PLATFORM";
        case -33: return "CL_INVALID_DEVICE";
        case -34: return "CL_INVALID_CONTEXT";
        case -35: return "CL_INVALID_QUEUE_PROPERTIES";
        case -36: return "CL_INVALID_COMMAND_QUEUE";
        case -37: return "CL_INVALID_HOST_PTR";
        case -38: return "CL_INVALID_MEM_OBJECT";
        case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case -40: return "CL_INVALID_IMAGE_SIZE";
        case -41: return "CL_INVALID_SAMPLER";
        case -42: return "CL_INVALID_BINARY";
        case -43: return "CL_INVALID_BUILD_OPTIONS";
        case -44: return "CL_INVALID_PROGRAM";
        case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
        case -46: return "CL_INVALID_KERNEL_NAME";
        case -47: return "CL_INVALID_KERNEL_DEFINITION";
        case -48: return "CL_INVALID_KERNEL";
        case -49: return "CL_INVALID_ARG_INDEX";
        case -50: return "CL_INVALID_ARG_VALUE";
        case -51: return "CL_INVALID_ARG_SIZE";
        case -52: return "CL_INVALID_KERNEL_ARGS";
        case -53: return "CL_INVALID_WORK_DIMENSION";
        case -54: return "CL_INVALID_WORK_GROUP_SIZE";
        case -55: return "CL_INVALID_WORK_ITEM_SIZE";
        case -56: return "CL_INVALID_GLOBAL_OFFSET";
        case -57: return "CL_INVALID_EVENT_WAIT_LIST";
        case -58: return "CL_INVALID_EVENT";
        case -59: return "CL_INVALID_OPERATION";
        case -60: return "CL_INVALID_GL_OBJECT";
        case -61: return "CL_INVALID_BUFFER_SIZE";
        case -62: return "CL_INVALID_MIP_LEVEL";
        case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
        case -64: return "CL_INVALID_PROPERTY";
        case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
        case -66: return "CL_INVALID_COMPILER_OPTIONS";
        case -67: return "CL_INVALID_LINKER_OPTIONS";
        case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";
        // case -69: return "CL_INVALID_PIPE_SIZE";                 // OpenCL 2.0
        // case -70: return "CL_INVALID_DEVICE_QUEUE";              // OpenCL 2.0
        // case -71: return "CL_INVALID_SPEC_ID";                   // OpenCL 2.2
        // case -72: return "CL_MAX_SIZE_RESTRICTION_EXCEEDED";     // OpenCL 2.2
        default: return "Unknown OpenCL error";
    }
}

/**
 * @brief Manages a shared OpenCL context, device, queue, and program compilation
 *        from multiple source files.
 */
class OpenCLContext {
public:
    cl::Context context;        // Represents the OpenCL context.
    cl::Device device;          // Represents the selected OpenCL device.
    cl::CommandQueue queue;     // Command queue for the selected device.
    cl::Program program;        // Compiled OpenCL program from all sources.
    std::map<std::string, cl::Kernel> kernels; // Map to store kernel objects by name

    /**
     * @brief Constructs and initializes the OpenCL environment from kernel source files.
     * @param kernelSourceFiles A vector of strings containing paths to OpenCL kernel (.cl) files.
     * @param kernelNames A vector of strings containing the names of the kernel functions to create.
     * @param device_type The preferred device type (e.g., CL_DEVICE_TYPE_GPU).
     * @throws std::runtime_error on OpenCL setup, file reading, or compilation errors.
     */
    OpenCLContext(const std::vector<std::string>& kernelSourceFiles, const std::vector<std::string>& kernelNames, 
        cl_device_type device_type = CL_DEVICE_TYPE_GPU) 
    {
        std::cout << "CL context Preparation." << std::endl;
        cl_int err; // To store error codes from OpenCL calls
        // Input validation remains important
        if (kernelSourceFiles.empty()) { // Check only files, names check below
            throw std::runtime_error("OpenCL Error: No kernel source files provided.");
        }
        if (kernelNames.empty()) {
            throw std::runtime_error("OpenCL Error: No kernel names provided.");
        }

        // The main try-catch for std::runtime_error can be kept if you want a general catch-all
        // for other runtime errors, but cl::Error specific catches must be removed.
        // try {
        // --- Platform and Device Selection (No changes needed) ---
        std::vector<cl::Platform> platforms;
        CL_CHECK(cl::Platform::get(&platforms)); // cl::Platform::get returns cl_int
        if (platforms.empty()) {
            throw std::runtime_error("OpenCL Error: No platforms found.");
        }
        cl::Platform platform = platforms[0];
        std::vector<cl::Device> devices;
        err = platform.getDevices(device_type, &devices); // platform.getDevices returns cl_int
        if (devices.empty() && device_type != CL_DEVICE_TYPE_CPU) {
            std::cerr << "Warning: No OpenCL devices found for preferred type (" << device_type << "). Trying CPU..." << std::endl;
            err = platform.getDevices(CL_DEVICE_TYPE_CPU, &devices);
        }
        CL_CHECK(err); // Check the result of the last getDevices call
        if (devices.empty()) {
            throw std::runtime_error("OpenCL Error: No devices found (GPU or CPU).");
        }
        device = devices[0];
        std::cout << "Using OpenCL device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
        std::cout << "Device Version: " << device.getInfo<CL_DEVICE_VERSION>() << std::endl;
        context = cl::Context({device}, nullptr, nullptr, nullptr, &err);
        CL_CHECK(err);
        queue = cl::CommandQueue(context, device, 0, &err); // 0 for default properties
        CL_CHECK(err);
        std::cout << "CONTEXT and QUEUE created successfully" << std::endl;

        // --- Load and Compile Program from Multiple Files (No changes needed) ---
        cl::Program::Sources sources;
        std::string allKernelCode;
        // Use a set to avoid reading the same file multiple times if paths are duplicated
        std::set<std::string> uniqueFiles(kernelSourceFiles.begin(), kernelSourceFiles.end());
        for (const std::string& filePath : uniqueFiles) { // Iterate over unique files
            std::ifstream file(filePath);
            if (!file.is_open()) {
                throw std::runtime_error("OpenCL Error: Could not open kernel file: " + filePath);
            }
            std::stringstream buffer;
            buffer << file.rdbuf();
            allKernelCode += buffer.str() + "\n";
            file.close();
            std::cout << "File " << filePath << " loaded successfully." << std::endl;
        }
        if (allKernelCode.empty()) {
            throw std::runtime_error("OpenCL Error: No kernel code loaded from files.");
        }
        sources.push_back({allKernelCode.c_str(), allKernelCode.length()});
        program = cl::Program(context, sources, &err);
        CL_CHECK(err);
        std::cout << "Programs from Kernel sources created successfully" << std::endl;

        // --- Build Program ---
        std::stringstream options_ss;
        options_ss << "-cl-std=CL2.0";
        std::string options = options_ss.str();
        // Add any other necessary build options here
        std::cout << "CL standard: " << options << std::endl;
        err = program.build({device}, options.c_str()); // program.build() returns cl_int
        std::cout << "Program built successfully" << std::endl;
        if (err != CL_SUCCESS) { // Manual check for build error
            std::string error_message = "OpenCL Error during program build: ";
            // Ensure string concatenation is safe by converting C-style strings to std::string first
            error_message += std::string(oclErrorString(err)) +
                                std::string(" [") + std::to_string(err) + std::string("]");
            if (err == CL_BUILD_PROGRAM_FAILURE) {
                error_message += "\n--- Build Log ---";
                std::string build_log_details;
                // Use the non-template getBuildInfo that returns cl_int
                cl_int build_log_err = program.getBuildInfo(device, CL_PROGRAM_BUILD_LOG, &build_log_details);
                if (build_log_err == CL_SUCCESS) {
                    // This line should be fine as device.getInfo<CL_DEVICE_NAME>() returns std::string,
                    // breaking any problematic const char* + const char* chain.
                    error_message += "\nDevice " + device.getInfo<CL_DEVICE_NAME>() + " Log:\n" + build_log_details;
                } 
                else {
                    // Ensure string concatenation is safe
                    error_message += std::string("\nFailed to retrieve build log: ") + oclErrorString(build_log_err) +
                                        std::string(" [") + std::to_string(build_log_err) + std::string("]");
                }
                // Also print directly to cerr for immediate visibility
                std::cerr << "OpenCL Program Build Failed. Build Log:" << std::endl;
                std::cerr << "--------------------------------------------------------" << std::endl;
                std::cerr << build_log_details << std::endl; // Print whatever was retrieved, even if getBuildInfo failed (might be empty)
                std::cerr << "--------------------------------------------------------" << std::endl;
            }
            throw std::runtime_error(error_message);
        }


        // --- Create and store kernels (No changes needed) ---
        if (kernelNames.empty()) {
                throw std::runtime_error("OpenCL Error: No kernel names provided.");
        }

        for (const std::string& kernelName : kernelNames) {
            // Check if kernel name is empty string before creating
            if (kernelName.empty()) {
                std::cerr << "Warning: Skipping empty kernel name." << std::endl;
                continue;
            }
            kernels[kernelName] = createKernel(kernelName); // Calls internal createKernel
            std::cout << "Created OpenCL kernel: " << kernelName << std::endl;
        }
        std::cout << "Successfully created " << kernels.size() << " OpenCL kernels." << std::endl;
    }

    // Disable copy constructor and assignment operator (Good practice)
    OpenCLContext(const OpenCLContext&) = default; // Now enabled
    OpenCLContext& operator=(const OpenCLContext&) = default; // Now enabled
    // Allow move constructor and assignment (Good practice)
    OpenCLContext(OpenCLContext&&) = default; // Remains enabled
    OpenCLContext& operator=(OpenCLContext&&) = default;

    // Default destructor is sufficient as cl:: objects manage their resources via RAII.
    ~OpenCLContext() = default;

    /**
     * @brief Creates a cl::Kernel object from the compiled program. (Internal helper)
     * @param kernelName The name of the kernel function in the source code.
     * @return A cl::Kernel object.
     * @throws std::runtime_error if the kernel cannot be created.
     */
    cl::Kernel createKernel(const std::string& kernelName) {
        cl_int err;
        cl::Kernel kernel_obj(program, kernelName.c_str(), &err); // Use constructor with cl_int*
        if (err != CL_SUCCESS) { // Manual error check
            throw std::runtime_error("OpenCL Error creating kernel '" + kernelName + "': " + oclErrorString(err) + " (" + std::to_string(err) + ")");
        }
        return kernel_obj;
    }

    std::string readKernelFile(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open kernel file: " + filename);
        }
        std::stringstream buffer;
        buffer << file.rdbuf();
        return buffer.str();
    }
};

#endif
#endif