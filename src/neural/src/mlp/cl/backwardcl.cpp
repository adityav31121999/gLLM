#ifdef USE_OPENCL

#include "include/mlp.hpp" // Includes basic.hpp where OpenCLContext is defined
#include <maths.hpp>       // Includes basic utilities like flatten, unflatten
#include <vector>
#include <stdexcept>
#include <iostream>
#include <string>
#include <CL/cl.hpp> // Keep OpenCL headers


/**
 * @brief OpenCL implementation of the simple backward propagation function using shared OpenCLContext.
 *        Calculates deltas and updates weights directly without storing gradients.
 *        Mirrors the logic of mlp::backward (CPU version).
 * @param in dimension of input, output and size of weight layers
 * @param layers number of hidden layers (total weight matrices = layers + 1)
 * @param learning learning rate for mlp
 */
void mlp::clBackward(int in, int layers, float learning) {
    // --- Basic Sanity Checks & Initialization ---
    if (output.size() != static_cast<size_t>(in) || expected.size() != static_cast<size_t>(in)) {
        throw std::invalid_argument("MLP clBackward: Output/Expected vector size mismatch. Expected: " + std::to_string(in));
    }
    if (weights.empty() || weights.size() != static_cast<size_t>(layers + 1)) {
         throw std::runtime_error("MLP clBackward: Weights vector size mismatch. Expected: " + std::to_string(layers + 1) + ", Got: " + std::to_string(weights.size()));
    }
    // Need activations for hidden delta calculation and weight updates
    if (layers > 0 && (activations.empty() || activations.size() != static_cast<size_t>(layers))) {
        throw std::runtime_error("MLP clBackward: Activations vector size mismatch. Expected: " + std::to_string(layers));
    }
    if (input.size() != static_cast<size_t>(in)) {
        throw std::runtime_error("MLP clBackward: Input vector size mismatch.");
    }
    // Check dimensions
    for (size_t l = 0; l <= static_cast<size_t>(layers); ++l) {
        if (weights[l].empty() || weights[l].size() != static_cast<size_t>(in) || weights[l][0].size() != static_cast<size_t>(in)) {
             throw std::runtime_error("MLP clBackward: Weight dimensions error at layer " + std::to_string(l));
        }
    }
    if (layers > 0) {
        for (size_t l = 0; l < static_cast<size_t>(layers); ++l) {
            if (activations[l].empty() || activations[l].size() != static_cast<size_t>(in)) {
                throw std::runtime_error("MLP clBackward: Activation dimensions error at layer " + std::to_string(l));
            }
        }
    }

    try {
        // --- Access Shared OpenCL Context ---
        OpenCLContext& context_obj = this->clContext; // Use the member reference

        // --- OpenCL Kernel Preparation (Retrieve from context) ---
        // Ensure these kernel names were provided during OpenCLContext initialization
        cl::Kernel kernelOutDelta = context_obj.kernels.at("kernelOutputDeltaSigmoid");
        cl::Kernel kernelHiddenDelta = context_obj.kernels.at("kernelHiddenDeltaSigmoid");
        cl::Kernel kernelUpdateW = context_obj.kernels.at("kernelUpdateWeights");

        // --- Device Buffer Allocation (using shared context) ---
        size_t layer_size_bytes = sizeof(float) * in;
        size_t weights_size_bytes = sizeof(float) * in * in; // Assumes square matrices

        // Buffers for delta calculation
        cl::Buffer d_output_activations(context_obj.context, CL_MEM_READ_ONLY, layer_size_bytes); // Assuming output holds *activated* values for kernel
        cl::Buffer d_expected(context_obj.context, CL_MEM_READ_ONLY, layer_size_bytes);
        std::vector<cl::Buffer> d_activations(layers); // Hidden layer activations
        cl::Buffer d_weights_next(context_obj.context, CL_MEM_READ_ONLY, weights_size_bytes); // W[l+1]

        // Ping-pong buffers for deltas
        cl::Buffer d_delta_buffer1(context_obj.context, CL_MEM_READ_WRITE, layer_size_bytes);
        cl::Buffer d_delta_buffer2(context_obj.context, CL_MEM_READ_WRITE, layer_size_bytes);
        cl::Buffer* d_delta_next = &d_delta_buffer1; // Holds deltas for layer l+1 (starts with output layer)
        cl::Buffer* d_delta_curr = &d_delta_buffer2; // Holds deltas for layer l (calculated in loop)

        // Buffers for weight update
        cl::Buffer d_weights_l(context_obj.context, CL_MEM_READ_WRITE, weights_size_bytes); // W[l]
        cl::Buffer d_activations_prev(context_obj.context, CL_MEM_READ_ONLY, layer_size_bytes); // Input or A[l-1]
        cl::Buffer d_input_buffer(context_obj.context, CL_MEM_READ_ONLY, layer_size_bytes); // Separate buffer for initial input

        // Store all delta buffers for the weight update phase
        std::vector<cl::Buffer> d_all_deltas(layers + 1); // Need space for output delta + hidden deltas

        // Allocate hidden activation buffers
        for (int l = 0; l < layers; ++l) {
             d_activations[l] = cl::Buffer(context_obj.context, CL_MEM_READ_ONLY, layer_size_bytes);
        }

        // --- NDRange Configuration ---
        cl::NDRange global_1d(in);
        cl::NDRange local_1d = cl::NullRange; // Let runtime choose
        cl::NDRange global_2d(in, in); // Assuming square matrices
        cl::NDRange local_2d = cl::NullRange; // Let runtime choose, or tune e.g., cl::NDRange(16, 16)

        cl_int cl_in = static_cast<cl_int>(in);
        cl_float cl_learning = static_cast<cl_float>(learning);

        // --- Data Transfer: Host -> Device (using shared queue) ---
        // NOTE: Assuming 'output' holds the *activated* output for kernelOutputDeltaSigmoid.
        context_obj.queue.enqueueWriteBuffer(d_output_activations, CL_TRUE, 0, layer_size_bytes, output.data());
        context_obj.queue.enqueueWriteBuffer(d_expected, CL_TRUE, 0, layer_size_bytes, expected.data());
        context_obj.queue.enqueueWriteBuffer(d_input_buffer, CL_TRUE, 0, layer_size_bytes, input.data());
        for (int l = 0; l < layers; ++l) {
            context_obj.queue.enqueueWriteBuffer(d_activations[l], CL_TRUE, 0, layer_size_bytes, activations[l].data());
        }

        // === 1. Calculate Output Layer Deltas ===
        kernelOutDelta.setArg(0, d_output_activations); // 'activations' argument
        kernelOutDelta.setArg(1, d_expected);
        kernelOutDelta.setArg(2, *d_delta_next);        // Write output deltas here
        kernelOutDelta.setArg(3, cl_in);                // size
        context_obj.queue.enqueueNDRangeKernel(kernelOutDelta, cl::NullRange, global_1d, local_1d);

        // Store the output delta buffer (index 'layers')
        d_all_deltas[layers] = *d_delta_next;

        // === 2. Backpropagate Deltas through Hidden Layers ===
        for (int l = layers - 1; l >= 0; --l) {
            std::vector<float> flat_weights_next = flatten(weights[l + 1]);
            context_obj.queue.enqueueWriteBuffer(d_weights_next, CL_TRUE, 0, weights_size_bytes, flat_weights_next.data());
            // d_activations[l] was already written above

            kernelHiddenDelta.setArg(0, *d_delta_next);     // Deltas from next layer (l+1)
            kernelHiddenDelta.setArg(1, d_weights_next);    // Weights W[l+1]
            kernelHiddenDelta.setArg(2, d_activations[l]);  // Activations A[l]
            kernelHiddenDelta.setArg(3, *d_delta_curr);     // Output: Deltas for current layer (l)
            kernelHiddenDelta.setArg(4, cl_in);             // current_layer_size
            kernelHiddenDelta.setArg(5, cl_in);             // next_layer_size
            context_obj.queue.enqueueNDRangeKernel(kernelHiddenDelta, cl::NullRange, global_1d, local_1d);

            // Store the calculated delta buffer for layer 'l'
            d_all_deltas[l] = *d_delta_curr;

            // Swap pointers for next iteration
            std::swap(d_delta_curr, d_delta_next);
        }

        // === 3. Update Weights ===
        for (int l = 0; l <= layers; ++l) {
            // Get previous layer's activations buffer
            cl::Buffer& d_prev_activations_buffer = (l == 0) ? d_input_buffer : d_activations[l - 1];

            // Get current weights W[l]
            std::vector<float> flat_weights_l = flatten(weights[l]);
            context_obj.queue.enqueueWriteBuffer(d_weights_l, CL_TRUE, 0, weights_size_bytes, flat_weights_l.data());

            // Get deltas for layer 'l'
            cl::Buffer& current_delta_buffer = d_all_deltas[l];

            // Launch kernel to update weights W[l] in place
            kernelUpdateW.setArg(0, current_delta_buffer);      // deltas[l]
            kernelUpdateW.setArg(1, d_prev_activations_buffer); // activations[l-1] or input
            kernelUpdateW.setArg(2, d_weights_l);               // weights[l] (read/write)
            kernelUpdateW.setArg(3, cl_learning);               // learning_rate
            kernelUpdateW.setArg(4, cl_in);                     // current_layer_size
            kernelUpdateW.setArg(5, cl_in);                     // prev_layer_size
            context_obj.queue.enqueueNDRangeKernel(kernelUpdateW, cl::NullRange, global_2d, local_2d); // Use 2D launch

            // Read updated weights back to host
            context_obj.queue.enqueueReadBuffer(d_weights_l, CL_TRUE, 0, weights_size_bytes, flat_weights_l.data());

            // Unflatten weights back into the host mlp object
            unflatten(flat_weights_l, weights[l], in, in); // Assuming square matrices
        }

        // --- Cleanup ---
        // context_obj.queue.finish(); // Optional, reads are blocking (CL_TRUE)
        // Buffers are released automatically by RAII destructors

    }
    catch (const cl::Error& err) {
        std::cerr << "OpenCL Error in mlp::clBackward: " << err.what() << " (" << err.err() << ")" << std::endl;
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
        throw std::runtime_error("OpenCL error during backward propagation."); // Re-throw as runtime_error
    }
     catch (const std::out_of_range& oor) {
        // Specific catch for kernel lookup failure from the map
        std::cerr << "Error: Kernel not found in the shared OpenCLContext kernel map during clBackward. "
                  << "Ensure kernels 'kernelOutputDeltaSigmoid', 'kernelHiddenDeltaSigmoid', and 'kernelUpdateWeights' were provided during OpenCLContext initialization. "
                  << "Details: " << oor.what() << std::endl;
        throw; // Re-throw exception
    }
    catch (const std::exception& e) {
        std::cerr << "Standard Exception in mlp::clBackward: " << e.what() << std::endl;
        throw; // Re-throw standard exceptions
    }
}

#endif // USE_OPENCL
