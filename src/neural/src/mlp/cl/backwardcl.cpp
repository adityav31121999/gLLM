#ifdef USE_OPENCL
#include "include/mlp.hpp" // Includes basic.hpp where OpenCLContext is defined
#include <maths.hpp>       // Includes basic utilities like flatten, unflatten
#include <vector>
#include <stdexcept>
#include <iostream>
#include <string>

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
        // Check if mat dimensions are consistent with 'in'
        if (weights[l].row != in || weights[l].col != in) {
             // This indicates a mismatch between 'in' and actual mat dimensions.
             // Depending on strictness, could throw or log. For now, proceed with 'in' for buffer sizes.
             std::cerr << "Warning: MLP clBackward: Weight mat dimensions at layer " << l << " ("
                       << weights[l].row << "x" << weights[l].col << ") do not match 'in' parameter (" << in << ")." << std::endl;
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
        cl_int err; // For OpenCL error codes
        // --- Access Shared OpenCL Context ---
        OpenCLContext& context_obj = clContext; // Use the member reference

        // --- OpenCL Kernel Preparation (Retrieve from context) ---
        // Ensure these kernel names were provided during OpenCLContext initialization
        cl::Kernel kernelOutDelta = context_obj.kernels.at("kernelOutputDeltaSigmoid");
        cl::Kernel kernelHiddenDelta = context_obj.kernels.at("kernelHiddenDeltaSigmoid");
        cl::Kernel kernelUpdateW = context_obj.kernels.at("kernelUpdateWeights");

        // --- Device Buffer Allocation (using shared context) ---
        size_t layer_size_bytes = sizeof(float) * in;
        size_t weights_size_bytes = sizeof(float) * in * in; // Assumes square matrices

        // Buffers for delta calculation
        cl::Buffer d_output_activations(context_obj.context, CL_MEM_READ_ONLY, layer_size_bytes, nullptr, &err); // Assuming output holds *activated* values for kernel
        CL_CHECK(err);
        cl::Buffer d_expected(context_obj.context, CL_MEM_READ_ONLY, layer_size_bytes, nullptr, &err);
        CL_CHECK(err);
        std::vector<cl::Buffer> d_activations(layers); // Hidden layer activations
        cl::Buffer d_weights_next(context_obj.context, CL_MEM_READ_ONLY, weights_size_bytes, nullptr, &err); // W[l+1]
        CL_CHECK(err);

        // Ping-pong buffers for deltas
        cl::Buffer d_delta_buffer1(context_obj.context, CL_MEM_READ_WRITE, layer_size_bytes, nullptr, &err);
        CL_CHECK(err);
        cl::Buffer d_delta_buffer2(context_obj.context, CL_MEM_READ_WRITE, layer_size_bytes, nullptr, &err);
        CL_CHECK(err);
        cl::Buffer* d_delta_next = &d_delta_buffer1; // Holds deltas for layer l+1 (starts with output layer)
        cl::Buffer* d_delta_curr = &d_delta_buffer2; // Holds deltas for layer l (calculated in loop)

        // Buffers for weight update
        cl::Buffer d_weights_l(context_obj.context, CL_MEM_READ_WRITE, weights_size_bytes, nullptr, &err); // W[l]
        CL_CHECK(err);
        // d_activations_prev is not directly created; it points to d_input_buffer or d_activations[l-1]
        cl::Buffer d_input_buffer(context_obj.context, CL_MEM_READ_ONLY, layer_size_bytes, nullptr, &err); // Separate buffer for initial input
        CL_CHECK(err);

        // Store all delta buffers for the weight update phase
        // Initialize with dummy/null buffers if direct assignment is tricky, or manage carefully.
        // For simplicity, we'll assign directly after calculation.
        // Ensure d_all_deltas can hold cl::Buffer objects.
        // If d_all_deltas[l] = *d_delta_curr; is used, d_all_deltas should be std::vector<cl::Buffer>
        // and the buffers pointed to by d_delta_curr must remain valid.
        std::vector<cl::Buffer> d_all_deltas(layers + 1); // Will store copies of the delta buffers

        // Allocate hidden activation buffers
        for (int l = 0; l < layers; ++l) {
             d_activations[l] = cl::Buffer(context_obj.context, CL_MEM_READ_ONLY, layer_size_bytes, nullptr, &err);
             CL_CHECK(err);
        }

        // --- NDRange Configuration ---
        cl::NDRange global_1d(in);
        cl::NDRange local_1d = cl::NullRange; // Let runtime choose
        cl::NDRange global_2d(in, in); // Assuming square matrices
        cl::NDRange local_2d = cl::NullRange; // Let runtime choose, or tune e.g., cl::NDRange(16, 16)

        cl_int cl_in = static_cast<cl_int>(in);
        float cl_learning = static_cast<float>(learning);

        // --- Data Transfer: Host -> Device (using shared queue) ---
        // NOTE: Assuming 'output' holds the *activated* output for kernelOutputDeltaSigmoid.
        CL_CHECK(context_obj.queue.enqueueWriteBuffer(d_output_activations, CL_TRUE, 0, layer_size_bytes, output.data()));
        CL_CHECK(context_obj.queue.enqueueWriteBuffer(d_expected, CL_TRUE, 0, layer_size_bytes, expected.data()));
        CL_CHECK(context_obj.queue.enqueueWriteBuffer(d_input_buffer, CL_TRUE, 0, layer_size_bytes, input.data()));
        for (int l = 0; l < layers; ++l) {
            CL_CHECK(context_obj.queue.enqueueWriteBuffer(d_activations[l], CL_TRUE, 0, layer_size_bytes, activations[l].data()));
        }

        // === 1. Calculate Output Layer Deltas ===
        CL_CHECK(kernelOutDelta.setArg(0, d_output_activations)); // 'activations' argument
        CL_CHECK(kernelOutDelta.setArg(1, d_expected));
        CL_CHECK(kernelOutDelta.setArg(2, *d_delta_next));        // Write output deltas here
        CL_CHECK(kernelOutDelta.setArg(3, cl_in));                // size
        CL_CHECK(context_obj.queue.enqueueNDRangeKernel(kernelOutDelta, cl::NullRange, global_1d, local_1d));

        // Store the output delta buffer (index 'layers')
        d_all_deltas[layers] = *d_delta_next; // This copies the cl::Buffer object

        // === 2. Backpropagate Deltas through Hidden Layers ===
        for (int l = layers - 1; l >= 0; --l) {
            // Use mapped_data directly for weights[l+1]
            const mat& weights_l_plus_1_mat = weights[l + 1];
            CL_CHECK(context_obj.queue.enqueueWriteBuffer(d_weights_next, CL_TRUE, 0, weights_size_bytes, weights_l_plus_1_mat.mapped_data));
            // d_activations[l] was already written above

            CL_CHECK(kernelHiddenDelta.setArg(0, *d_delta_next));     // Deltas from next layer (l+1)
            CL_CHECK(kernelHiddenDelta.setArg(1, d_weights_next));    // Weights W[l+1]
            CL_CHECK(kernelHiddenDelta.setArg(2, d_activations[l]));  // Activations A[l]
            CL_CHECK(kernelHiddenDelta.setArg(3, *d_delta_curr));     // Output: Deltas for current layer (l)
            CL_CHECK(kernelHiddenDelta.setArg(4, cl_in));             // current_layer_size
            CL_CHECK(kernelHiddenDelta.setArg(5, cl_in));             // next_layer_size
            CL_CHECK(context_obj.queue.enqueueNDRangeKernel(kernelHiddenDelta, cl::NullRange, global_1d, local_1d));

            // Store the calculated delta buffer for layer 'l'
            d_all_deltas[l] = *d_delta_curr; // This copies the cl::Buffer object

            // Swap pointers for next iteration
            std::swap(d_delta_curr, d_delta_next);
        }

        // === 3. Update Weights ===
        for (int l = 0; l <= layers; ++l) {
            // Get previous layer's activations buffer
            cl::Buffer& d_prev_activations_buffer = (l == 0) ? d_input_buffer : d_activations[l - 1];

            // Get current weights W[l]
            // Use mapped_data directly for weights[l]
            const mat& weights_l_mat = weights[l];
            CL_CHECK(context_obj.queue.enqueueWriteBuffer(d_weights_l, CL_TRUE, 0, weights_size_bytes, weights_l_mat.mapped_data));

            // Get deltas for layer 'l'
            cl::Buffer& current_delta_buffer = d_all_deltas[l];

            // Launch kernel to update weights W[l] in place
            CL_CHECK(kernelUpdateW.setArg(0, current_delta_buffer));      // deltas[l]
            CL_CHECK(kernelUpdateW.setArg(1, d_prev_activations_buffer)); // activations[l-1] or input
            CL_CHECK(kernelUpdateW.setArg(2, d_weights_l));               // weights[l] (read/write)
            CL_CHECK(kernelUpdateW.setArg(3, cl_learning));               // learning_rate
            CL_CHECK(kernelUpdateW.setArg(4, cl_in));                     // current_layer_size
            CL_CHECK(kernelUpdateW.setArg(5, cl_in));                     // prev_layer_size
            CL_CHECK(context_obj.queue.enqueueNDRangeKernel(kernelUpdateW, cl::NullRange, global_2d, local_2d)); // Use 2D launch

            // Read updated weights back to host
            // Read directly into the mapped_data of the mat object
            CL_CHECK(context_obj.queue.enqueueReadBuffer(d_weights_l, CL_TRUE, 0, weights_size_bytes, weights[l].mapped_data));
        }

        // --- Cleanup ---
        // context_obj.queue.finish(); // Optional, reads are blocking (CL_TRUE)
        // Buffers are released automatically by RAII destructors
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
