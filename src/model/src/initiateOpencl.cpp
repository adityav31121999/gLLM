
#include "include/opencl_setup.hpp" // Include the header we just defined
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>

// --- Define the global OpenCL objects (implementation) ---
// These definitions correspond to the 'extern' declarations in the header.
cl::Context context;
cl::CommandQueue queue;
cl::Program program;
cl::Device default_device;

// --- Configuration: List of Kernel Files ---
// *** IMPORTANT: Adjust these paths relative to your build/execution directory ***
// You might need to experiment or use absolute paths if relative paths are tricky.
const std::vector<std::string> kernelFiles = {
    "src/maths/basic/cl/activations.cl",        // Activation functions (Sigmoid, ReLU, Softmax, LOTA)
    "src/maths/basic/cl/vect.cl",               // vector functions
    "src/neural/src/mlp/cl/kernel.cl",          // MLP kernels (MSE, L1/L2, backprop etc.)
    "src/neural/src/attention/cl/kernels.cl",   // Attention helper kernels (dot products, sums, backprop etc.)
    "src/neural/src/transformer/cl/kernels.cl"  // Transformer specific kernels (KdotQ variants)
    // Add any other .cl files needed for your program object here
};

// --- Helper Function to Read File Content ---
std::string readFile(const std::string& filePath) {
    std::ifstream fileStream(filePath);
    if (!fileStream.is_open()) {
        // Try prepending the absolute base path as a fallback (less portable)
        std::string altFilePath = "d:/gLLM/" + filePath; // Adjust "d:/gLLM/" if needed
        fileStream.open(altFilePath);
        if (!fileStream.is_open()) {
            throw std::runtime_error("Failed to open kernel file: " + filePath + " or " + altFilePath);
        }
        std::cout << "INFO: Loaded kernel source using alternative path: " << altFilePath << std::endl;
    } else {
        std::cout << "INFO: Loaded kernel source from: " << filePath << std::endl;
    }

    std::stringstream buffer;
    buffer << fileStream.rdbuf();
    return buffer.str();
}


// --- Initialization Function Implementation ---
void initializeOpenCL() {
    std::cout << "--- Initializing OpenCL ---" << std::endl;
    try {
        // 1. Platform Selection
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.empty()) {
            throw std::runtime_error("FATAL: No OpenCL platforms found.");
        }

        cl::Platform platform = platforms[0]; // Choose the first platform
        // Optional: Add logic here to select a specific platform by name if needed
        std::cout << "Using Platform: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;

        // 2. Device Selection
        std::vector<cl::Device> devices;
        // Try to get GPU devices first
        platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
        if (devices.empty()) {
            std::cout << "WARNING: No OpenCL GPU devices found on this platform. Trying CPU..." << std::endl;
            // If no GPU, try CPU devices
            platform.getDevices(CL_DEVICE_TYPE_CPU, &devices);
            if (devices.empty()) {
                throw std::runtime_error("FATAL: No OpenCL GPU or CPU devices found on this platform.");
            }
        }

        default_device = devices[0]; // Choose the first available device
        // Optional: Add logic here to select a specific device by name
        std::cout << "Using Device:   " << default_device.getInfo<CL_DEVICE_NAME>() << std::endl;
        std::cout << "Device Version: " << default_device.getInfo<CL_DEVICE_VERSION>() << std::endl;


        // 3. Context Creation
        // Create a context using the selected device
        context = cl::Context(default_device);
        std::cout << "OpenCL Context created." << std::endl;

        // 4. Command Queue Creation
        // Create a command queue for the selected device within the context
        // Add cl::QueueProperties::Profiling if you need profiling later
        queue = cl::CommandQueue(context, default_device);
        std::cout << "OpenCL Command Queue created." << std::endl;

        // 5. Load and Concatenate Kernel Source Code
        std::cout << "Loading kernel source files..." << std::endl;
        std::string combinedKernelSource = "";
        for (const auto& filePath : kernelFiles) {
            combinedKernelSource += readFile(filePath) + "\n\n"; // Add newline separators
        }

        if (combinedKernelSource.empty()) {
             throw std::runtime_error("FATAL: No kernel source code was loaded. Check file paths.");
        }
        std::cout << "Kernel source files loaded and combined." << std::endl;

        // 6. Program Creation
        program = cl::Program(context, combinedKernelSource);
        std::cout << "OpenCL Program object created." << std::endl;

        // 7. Build Program
        std::cout << "Building OpenCL program... (This may take a moment)" << std::endl;
        try {
            // Build the program for the selected device
            // You can add build options here if needed, e.g., "-cl-std=CL2.0"
            program.build({default_device}, ""); // Empty options string
            std::cout << "SUCCESS: OpenCL program built successfully." << std::endl;
        } catch (const cl::Error& build_err) {
            // Specific handling for build errors is crucial
            std::cerr << "FATAL: OpenCL Program Build Failed: " << build_err.what() << " (" << build_err.err() << ")" << std::endl;

            // Get and print the build log for debugging kernel code errors
            if (build_err.err() == CL_BUILD_PROGRAM_FAILURE) {
                std::string log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device);
                std::cerr << "-------------------- Build Log (" << default_device.getInfo<CL_DEVICE_NAME>() << ") --------------------" << std::endl;
                std::cerr << log << std::endl;
                std::cerr << "-------------------- End Build Log --------------------" << std::endl;
            }
            throw; // Re-throw the exception to signal initialization failure
        }

        std::cout << "--- OpenCL Initialization Complete ---" << std::endl;

    }
    catch (const cl::Error& cl_err) {
        // Catch any OpenCL API errors during setup
        std::cerr << "FATAL: OpenCL API Error during initialization: " << cl_err.what() << " (" << cl_err.err() << ")" << std::endl;
        throw std::runtime_error("OpenCL Initialization failed due to API error."); // Throw a standard exception
    }
    catch (const std::exception& e) {
        // Catch standard exceptions (like file not found, memory allocation)
        std::cerr << "FATAL: Standard Exception during initialization: " << e.what() << std::endl;
        throw; // Re-throw
    }
}
