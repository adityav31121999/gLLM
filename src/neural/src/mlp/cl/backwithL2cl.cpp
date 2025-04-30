#ifdef USE_OPENCL

#include "include/mlp.hpp" // Includes basic.hpp where OpenCLContext is defined
#include <maths.hpp>       // Includes basic utilities like flatten, unflatten
#include <vector>
#include <stdexcept>
#include <iostream>
#include <string>
#include <CL/cl.hpp> // Keep OpenCL headers

/**
 * @brief OpenCL implementation of backpropagation with L2 regularization using shared OpenCLContext.
 *        Calculates deltas and updates weights using the L2 update rule directly.
 *        Does NOT calculate or store gradients (gweights).
 * @param in Input/layer size (assuming all layers have the same size 'in')
 * @param layers Number of hidden layers (total weight matrices = layers + 1)
 * @param learning Learning rate
 */
void mlp::clBackwithL2(int in, int layers, float learning) {
    float lambda = 0.01f; // L2 regularization parameter (hardcoded or pass as argument)

    // --- Basic Sanity Checks ---
    if (output.size() != static_cast<size_t>(in) || expected.size() != static_cast<size_t>(in)) {
        throw std::invalid_argument("MLP clBackwithL2: Output/Expected vector size mismatch.");
    }
    if (weights.empty() || weights.size() != static_cast<size_t>(layers + 1)) {
         throw std::runtime_error("MLP clBackwithL2: Weights vector size mismatch.");
    }
     if (layers > 0 && (activations.empty() || activations.size() != static_cast<size_t>(layers))) {
        throw std::runtime_error("MLP clBackwithL2: Activations vector size mismatch.");
    }
    if (input.size() != static_cast<size_t>(in)) {
        throw std::runtime_error("MLP clBackwithL2: Input vector size mismatch.");
    }
    for (size_t l = 0; l <= static_cast<size_t>(layers); ++l) {
        if (weights[l].empty() || weights[l].size() != static_cast<size_t>(in) || weights[l][0].size() != static_cast<size_t>(in)) {
             throw std::runtime_error("MLP clBackwithL2: Weight dimensions error at layer " + std::to_string(l));
        }
    }
     if (layers > 0) {
        for (size_t l = 0; l < static_cast<size_t>(layers); ++l) {
            if (activations[l].empty() || activations[l].size() != static_cast<size_t>(in)) {
                throw std::runtime_error("MLP clBackwithL2: Activation dimensions error at layer " + std::to_string(l));
            }
        }
     }
    // Note: gweights are NOT used or updated by this function.

    try {
        // --- Access Shared OpenCL Context ---
        OpenCLContext& context_obj = this->clContext; // Use the member reference

        // --- OpenCL Kernel Preparation (Retrieve from context) ---
        // Ensure these kernel names were provided during OpenCLContext initialization
        cl::Kernel kernelOutDelta = context_obj.kernels.at("kernelOutputDeltaSigmoid");
        cl::Kernel kernelHiddenDelta = context_obj.kernels.at("kernelHiddenDeltaSigmoid");
        cl::Kernel kernelUpdateWL2 = context_obj.kernels.at("kernelUpdateWeightsL2"); // L2 update kernel

        // --- Device Buffer Allocation (using shared context) ---
        size_t layer_size_bytes = sizeof(float) * in;
        size_t weights_size_bytes = sizeof(float) * in * in; // Assumes square matrices

        // Buffers for inputs/outputs/activations
        cl::Buffer d_input(context_obj.context, CL_MEM_READ_ONLY, layer_size_bytes);
        // Assuming output holds *activated* values for kernelOutputDeltaSigmoid
        cl::Buffer d_output_activations(context_obj.context, CL_MEM_READ_ONLY, layer_size_bytes);
        cl::Buffer d_expected(context_obj.context, CL_MEM_READ_ONLY, layer_size_bytes);
        std::vector<cl::Buffer> d_activations(layers); // Hidden layer activations

        // Buffers for weights and deltas (one per layer, including output)
        std::vector<cl::Buffer> d_weights(layers + 1);
        std::vector<cl::Buffer> d_layer_deltas(layers + 1);

        for (int l = 0; l <= layers; ++l) {
            d_weights[l] = cl::Buffer(context_obj.context, CL_MEM_READ_WRITE, weights_size_bytes); // Weights are updated
            d_layer_deltas[l] = cl::Buffer(context_obj.context, CL_MEM_READ_WRITE, layer_size_bytes); // Deltas are calculated
        }
        for (int l = 0; l < layers; ++l) {
             d_activations[l] = cl::Buffer(context_obj.context, CL_MEM_READ_ONLY, layer_size_bytes);
        }

        // --- Data Transfer: Host -> Device (using shared queue) ---
        context_obj.queue.enqueueWriteBuffer(d_input, CL_TRUE, 0, layer_size_bytes, input.data());
        context_obj.queue.enqueueWriteBuffer(d_output_activations, CL_TRUE, 0, layer_size_bytes, output.data());
        context_obj.queue.enqueueWriteBuffer(d_expected, CL_TRUE, 0, layer_size_bytes, expected.data());
        for (int l = 0; l < layers; ++l) {
            context_obj.queue.enqueueWriteBuffer(d_activations[l], CL_TRUE, 0, layer_size_bytes, activations[l].data());
        }
        for (int l = 0; l <= layers; ++l) {
            std::vector<float> flat_weights_l = flatten(weights[l]);
            context_obj.queue.enqueueWriteBuffer(d_weights[l], CL_TRUE, 0, weights_size_bytes, flat_weights_l.data());
        }

        // --- NDRange Configuration ---
        cl::NDRange global_1d(in);
        cl::NDRange local_1d = cl::NullRange;
        cl::NDRange global_2d(in, in); // For weight update kernel
        cl::NDRange local_2d = cl::NullRange;

        cl_int cl_in = static_cast<cl_int>(in);
        cl_float cl_learning = static_cast<cl_float>(learning);
        cl_float cl_lambda = static_cast<cl_float>(lambda);

        // --- Backpropagation Steps ---

        // 1. Calculate Output Layer Deltas (Layer index 'layers')
        kernelOutDelta.setArg(0, d_output_activations);
        kernelOutDelta.setArg(1, d_expected);
        kernelOutDelta.setArg(2, d_layer_deltas[layers]);
        kernelOutDelta.setArg(3, cl_in);
        context_obj.queue.enqueueNDRangeKernel(kernelOutDelta, cl::NullRange, global_1d, local_1d);

        // 2. Calculate Hidden Layer Deltas (Propagate backwards from layers-1 down to 0)
        for (int l = layers - 1; l >= 0; --l) {
            kernelHiddenDelta.setArg(0, d_layer_deltas[l + 1]);
            kernelHiddenDelta.setArg(1, d_weights[l + 1]);      // Weights W[l+1]
            kernelHiddenDelta.setArg(2, d_activations[l]);      // Activations A[l]
            kernelHiddenDelta.setArg(3, d_layer_deltas[l]);     // Output delta[l]
            kernelHiddenDelta.setArg(4, cl_in);                 // current_layer_size
            kernelHiddenDelta.setArg(5, cl_in);                 // next_layer_size
            context_obj.queue.enqueueNDRangeKernel(kernelHiddenDelta, cl::NullRange, global_1d, local_1d);
        }

        // 3. Update Weights using L2 Kernel (Iterate through layers 0 to layers)
        for (int l = 0; l <= layers; ++l) {
            cl::Buffer& d_prev_activations_buffer = (l == 0) ? d_input : d_activations[l - 1];

            kernelUpdateWL2.setArg(0, d_weights[l]);              // Weights W[l] (read/write)
            kernelUpdateWL2.setArg(1, d_layer_deltas[l]);         // Deltas delta[l]
            kernelUpdateWL2.setArg(2, d_prev_activations_buffer); // Activations A[l-1] or input
            kernelUpdateWL2.setArg(3, cl_learning);               // Learning rate
            kernelUpdateWL2.setArg(4, cl_lambda);                 // L2 lambda parameter
            kernelUpdateWL2.setArg(5, cl_in);                     // current_layer_size
            kernelUpdateWL2.setArg(6, cl_in);                     // prev_layer_size
            context_obj.queue.enqueueNDRangeKernel(kernelUpdateWL2, cl::NullRange, global_2d, local_2d);

            // Read updated weights back to host for this layer
            std::vector<float> updated_flat_weights(in * in);
            context_obj.queue.enqueueReadBuffer(d_weights[l], CL_TRUE, 0, weights_size_bytes, updated_flat_weights.data());
            unflatten(updated_flat_weights, weights[l], in, in);
        }

        // --- Cleanup ---
        // context_obj.queue.finish(); // Optional, reads are blocking (CL_TRUE)
        // Buffers are released automatically by RAII destructors

    }
    catch (const cl::Error& err) {
        std::cerr << "OpenCL Error in mlp::clBackwithL2: " << err.what() << " (" << err.err() << ")" << std::endl;
        // Print build log if it was a build error
        if (err.err() == CL_BUILD_PROGRAM_FAILURE) {
             try {
                // Use this->clContext to get program and device
                std::string log = this->clContext.program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(this->clContext.device);
                std::cerr << "Build Log:\n" << log << std::endl;
             } catch (const cl::Error& log_err) {
                 std::cerr << "Could not retrieve build log: " << log_err.what() << " (" << log_err.err() << ")" << std::endl;
             }
        }
        throw std::runtime_error("OpenCL error during L2 backpropagation."); // Re-throw as runtime_error
    }
     catch (const std::out_of_range& oor) {
        // Specific catch for kernel lookup failure from the map
        std::cerr << "Error: Kernel not found in the shared OpenCLContext kernel map during clBackwithL2. "
                  << "Ensure kernels 'kernelOutputDeltaSigmoid', 'kernelHiddenDeltaSigmoid', and 'kernelUpdateWeightsL2' were provided during OpenCLContext initialization. "
                  << "Details: " << oor.what() << std::endl;
        throw; // Re-throw exception
    }
    catch (const std::exception& e) {
        std::cerr << "Standard Exception in mlp::clBackwithL2: " << e.what() << std::endl;
        throw; // Re-throw standard exceptions
    }
}

#endif // USE_OPENCL
