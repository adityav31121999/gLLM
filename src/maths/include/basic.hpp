
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
float MSE(std::vector<float>, std::vector<float>);
float sum(std::vector<float>);
float sum(std::vector<std::vector<float>>);
float product(std::vector<float>);
float product(std::vector<std::vector<float>>);

std::vector<float> error(std::vector<float>, std::vector<float>);
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

#define CL_HPP_ENABLE_EXCEPTIONS // Define before including cl.hpp
#define CL_HPP_TARGET_OPENCL_VERSION 300
#include <CL/cl.hpp>
// #pragma OPENCL EXTENSION cl_khr_fp32 : enable // Keep if needed by kernels
// #pragma OPENCL EXTENSION cl_khr_int32_base_atomics : enable // Keep if needed
// #pragma OPENCL EXTENSION cl_khr_int32_extended_atomics : enable // Keep if needed
#include <sstream>
#include <string>
#include <fstream>
#include <stdexcept>
#include <map>
#include <set>
#include <vector> // Ensure vector is included

// Helper macro for indexing flattened matrix (assuming row-major)
#define IDX(row, col, dim) ((row) * (dim) + (col))

/**
 * @brief Manages a shared OpenCL context, device, queue, and program compilation
 *        from multiple source files.
 */
class OpenCLContext {
public:
    // --- REMOVED member references ---
    // std::vector<std::string>& kernelSourceFiles; // Removed - dangerous reference member
    // std::vector<std::string>& kernelNames;       // Removed - dangerous reference member

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
    OpenCLContext(const std::vector<std::string>& kernelSourceFiles, const std::vector<std::string>& kernelNames, cl_device_type device_type = CL_DEVICE_TYPE_GPU) {
        // Input validation remains important
        if (kernelSourceFiles.empty()) { // Check only files, names check below
            throw std::runtime_error("OpenCL Error: No kernel source files provided.");
        }
        if (kernelNames.empty()) {
            throw std::runtime_error("OpenCL Error: No kernel names provided.");
        }

        // This check might be overly restrictive if one file contains multiple kernels.
        // Consider removing it if you list each file only once, even if it has multiple kernels.
        // However, keeping it as per your original code for now.
        if (kernelSourceFiles.size() != kernelNames.size()) {
            throw std::runtime_error("OpenCL Error: Number of kernel source files provided does not match the number of kernel names. Provide one file path per kernel name (can be duplicate paths).");
        }

        try {
            // --- REMOVED dangerous assignments ---
            // this->kernelNames = kernelNames;
            // this->kernelSourceFiles = kernelSourceFiles;

            // --- Platform and Device Selection (No changes needed) ---
            std::vector<cl::Platform> platforms;
            cl::Platform::get(&platforms);
            if (platforms.empty()) {
                throw std::runtime_error("OpenCL Error: No platforms found.");
            }
            cl::Platform platform = platforms[0];
            std::vector<cl::Device> devices;
            platform.getDevices(device_type, &devices);
            if (devices.empty() && device_type != CL_DEVICE_TYPE_CPU) {
                std::cerr << "Warning: No OpenCL devices found for preferred type (" << device_type << "). Trying CPU..." << std::endl;
                platform.getDevices(CL_DEVICE_TYPE_CPU, &devices);
            }
            if (devices.empty()) {
                throw std::runtime_error("OpenCL Error: No devices found (GPU or CPU).");
            }
            device = devices[0];
            std::cout << "Using OpenCL device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
            std::cout << "Device Version: " << device.getInfo<CL_DEVICE_VERSION>() << std::endl;

            // --- Context & Queue (No changes needed) ---
            context = cl::Context(device);
            queue = cl::CommandQueue(context, device);

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
            }
            if (allKernelCode.empty()) {
                throw std::runtime_error("OpenCL Error: No kernel code loaded from files.");
            }
            sources.push_back({ allKernelCode.c_str(), allKernelCode.length() });
            program = cl::Program(context, sources);

            // --- Build Program (No changes needed) ---
            try {
                std::string options = "-cl-std=CL" + std::to_string(CL_HPP_TARGET_OPENCL_VERSION / 100) + "." + std::to_string((CL_HPP_TARGET_OPENCL_VERSION % 100) / 10);
                // Add any other necessary build options here, e.g. -I include_path
                // options += " -I../kernels/includes"; // Example include path
                program.build({ device }, options.c_str());
            }
            catch (cl::BuildError &e) {
                std::string log = "OpenCL Build Error:\n";
                log += "Status: " + std::to_string(e.err()) + "\n";
                for (const auto& pair : e.getBuildLog()) {
                    log += "Device " + pair.first.getInfo<CL_DEVICE_NAME>() + " Log:\n" + pair.second + "\n";
                }
                throw std::runtime_error(log);
            }
            catch (cl::Error &err) {
                throw std::runtime_error("OpenCL Error during program build: " + std::string(err.what()) + " (" + std::to_string(err.err()) + ")");
            }

            // --- Create and store kernels (No changes needed) ---
            // This uses the kernelNames vector passed to the constructor
            // Check that kernelNames is not empty *after* checking files
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
            }
            std::cout << "Successfully created " << kernels.size() << " OpenCL kernels." << std::endl;
        }
        catch (cl::Error &err) {
            throw std::runtime_error("OpenCL Setup Error: " + std::string(err.what()) + " (" + std::to_string(err.err()) + ")");
        }
        catch (const std::runtime_error& e) {
            throw; // Re-throw runtime errors from setup/file logic
        }
    }


    // Disable copy constructor and assignment operator (Good practice)
    OpenCLContext(const OpenCLContext&) = delete;
    OpenCLContext& operator=(const OpenCLContext&) = delete;
    // Allow move constructor and assignment (Good practice)
    OpenCLContext(OpenCLContext&&) = default;
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
        try {
            return cl::Kernel(program, kernelName.c_str());
        }
        catch (cl::Error &err) {
            throw std::runtime_error("OpenCL Error creating kernel '" + kernelName + "': " + std::string(err.what()) + " (" + std::to_string(err.err()) + ")");
        }
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
