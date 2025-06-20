#ifdef USE_OPENCL
#if defined(_WIN64)
    #define CL_HPP_ENABLE_EXCEPTIONS
    #define CL_HPP_TARGET_OPENCL_VERSION 300
    // For Windows, use the older/common cl.hpp
    #include <CL/cl.hpp>
#elif defined(__linux__)
    #define CL_HPP_TARGET_OPENCL_VERSION 220
    #include <CL/opencl.hpp>
#endif
#include "include/mlp.hpp" // Includes basic.hpp where OpenCLContext is defined
#include <maths.hpp>       // Includes basic utilities like flatten, unflatten
#include <vector>
#include <stdexcept>
#include <iostream>
#include <string>

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
        // Check if mat dimensions are consistent with 'in'
        if (this->weights[l].row != in || this->weights[l].col != in) {
             // This indicates a mismatch between 'in' and actual mat dimensions.
             // Depending on strictness, could throw or log. For now, proceed with 'in' for buffer sizes.
             std::cerr << "Warning: MLP clBackwithL2: Weight mat dimensions at layer " << l << " ("
                       << this->weights[l].row << "x" << this->weights[l].col << ") do not match 'in' parameter (" << in << ")." << std::endl;
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
        cl_int err; // For OpenCL error codes
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
        cl::Buffer d_input(context_obj.context, CL_MEM_READ_ONLY, layer_size_bytes, nullptr, &err);
        CL_CHECK(err);
        // Assuming output holds *activated* values for kernelOutputDeltaSigmoid
        cl::Buffer d_output_activations(context_obj.context, CL_MEM_READ_ONLY, layer_size_bytes, nullptr, &err);
        CL_CHECK(err);
        cl::Buffer d_expected(context_obj.context, CL_MEM_READ_ONLY, layer_size_bytes, nullptr, &err);
        CL_CHECK(err);
        std::vector<cl::Buffer> d_activations(layers); // Hidden layer activations

        // Buffers for weights and deltas (one per layer, including output)
        std::vector<cl::Buffer> d_weights(layers + 1);
        std::vector<cl::Buffer> d_layer_deltas(layers + 1);
        for (int l = 0; l <= layers; ++l) {
            d_weights[l] = cl::Buffer(context_obj.context, CL_MEM_READ_WRITE, weights_size_bytes, nullptr, &err); // Weights are updated
            CL_CHECK(err);
            d_layer_deltas[l] = cl::Buffer(context_obj.context, CL_MEM_READ_WRITE, layer_size_bytes, nullptr, &err); // Deltas are calculated
            CL_CHECK(err);
        }
        for (int l = 0; l < layers; ++l) {
             d_activations[l] = cl::Buffer(context_obj.context, CL_MEM_READ_ONLY, layer_size_bytes, nullptr, &err);
             CL_CHECK(err);
        }

        // --- Data Transfer: Host -> Device (using shared queue) ---
        CL_CHECK(context_obj.queue.enqueueWriteBuffer(d_input, CL_TRUE, 0, layer_size_bytes, input.data()));
        CL_CHECK(context_obj.queue.enqueueWriteBuffer(d_output_activations, CL_TRUE, 0, layer_size_bytes, output.data()));
        CL_CHECK(context_obj.queue.enqueueWriteBuffer(d_expected, CL_TRUE, 0, layer_size_bytes, expected.data()));
        for (int l = 0; l < layers; ++l) {
            CL_CHECK(context_obj.queue.enqueueWriteBuffer(d_activations[l], CL_TRUE, 0, layer_size_bytes, activations[l].data()));
        }
        for (int l = 0; l <= layers; ++l) {
            // Use mapped_data directly
            const mat& weights_l_mat = this->weights[l];
            CL_CHECK(context_obj.queue.enqueueWriteBuffer(d_weights[l], CL_TRUE, 0, weights_size_bytes, weights_l_mat.mapped_data));
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
        CL_CHECK(kernelOutDelta.setArg(0, d_output_activations));
        CL_CHECK(kernelOutDelta.setArg(1, d_expected));
        CL_CHECK(kernelOutDelta.setArg(2, d_layer_deltas[layers]));
        CL_CHECK(kernelOutDelta.setArg(3, cl_in));
        CL_CHECK(context_obj.queue.enqueueNDRangeKernel(kernelOutDelta, cl::NullRange, global_1d, local_1d));

        // 2. Calculate Hidden Layer Deltas (Propagate backwards from layers-1 down to 0)
        for (int l = layers - 1; l >= 0; --l) {
            CL_CHECK(kernelHiddenDelta.setArg(0, d_layer_deltas[l + 1]));
            CL_CHECK(kernelHiddenDelta.setArg(1, d_weights[l + 1]));      // Weights W[l+1]
            CL_CHECK(kernelHiddenDelta.setArg(2, d_activations[l]));      // Activations A[l]
            CL_CHECK(kernelHiddenDelta.setArg(3, d_layer_deltas[l]));     // Output delta[l]
            CL_CHECK(kernelHiddenDelta.setArg(4, cl_in));                 // current_layer_size
            CL_CHECK(kernelHiddenDelta.setArg(5, cl_in));                 // next_layer_size
            CL_CHECK(context_obj.queue.enqueueNDRangeKernel(kernelHiddenDelta, cl::NullRange, global_1d, local_1d));
        }

        // 3. Update Weights using L2 Kernel (Iterate through layers 0 to layers)
        for (int l = 0; l <= layers; ++l) {
            cl::Buffer& d_prev_activations_buffer = (l == 0) ? d_input : d_activations[l - 1];

            CL_CHECK(kernelUpdateWL2.setArg(0, d_weights[l]));              // Weights W[l] (read/write)
            CL_CHECK(kernelUpdateWL2.setArg(1, d_layer_deltas[l]));         // Deltas delta[l]
            CL_CHECK(kernelUpdateWL2.setArg(2, d_prev_activations_buffer)); // Activations A[l-1] or input
            CL_CHECK(kernelUpdateWL2.setArg(3, cl_learning));               // Learning rate
            CL_CHECK(kernelUpdateWL2.setArg(4, cl_lambda));                 // L2 lambda parameter
            CL_CHECK(kernelUpdateWL2.setArg(5, cl_in));                     // current_layer_size
            CL_CHECK(kernelUpdateWL2.setArg(6, cl_in));                     // prev_layer_size
            CL_CHECK(context_obj.queue.enqueueNDRangeKernel(kernelUpdateWL2, cl::NullRange, global_2d, local_2d));

            // Read updated weights back to host for this layer
            // Read directly into the mapped_data of the mat object
            CL_CHECK(context_obj.queue.enqueueReadBuffer(d_weights[l], CL_TRUE, 0, weights_size_bytes, this->weights[l].mapped_data));
        }

        // --- Cleanup ---
        // context_obj.queue.finish(); // Optional, reads are blocking (CL_TRUE)
        // Buffers are released automatically by RAII destructors

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
