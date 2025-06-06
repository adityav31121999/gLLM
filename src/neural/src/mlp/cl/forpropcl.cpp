#ifdef USE_OPENCL

#include "include/mlp.hpp" // This now includes basic.hpp where OpenCLContext is defined
#include <maths.hpp>       // Includes basic utilities like flatten
#include <vector>
#include <stdexcept>
#include <iostream>
#include <string>

/**
 * @brief Performs forward propagation using OpenCL kernels.
 *        Calculates weighted sums and applies sigmoid activation for each layer.
 * @param in Number of neurons per layer (assumed constant for hidden layers and output).
 * @param layers Number of hidden layers (total weight matrices = layers + 1).
 */
void mlp::clForward(int in, int layers) {
    if (input.empty() || weights.empty() || layers < 0) {
        throw std::runtime_error("MLP not properly initialized for clForward.");
    }
    if (weights.size() != static_cast<size_t>(layers + 1)) {
         throw std::runtime_error("Mismatch between layers count and weights size in clForward.");
    }

    hlayers.resize(layers, std::vector<float>(in, 0.0f));
    activations.resize(layers, std::vector<float>(in, 0.0f));
    output.resize(in, 0.0f); // Resize host output vector

    try {
        cl_int err; // For OpenCL error codes
        OpenCLContext& context_obj = this->clContext; // Use the member reference

        cl::Kernel layerForwardKernel = context_obj.kernels.at("kernelLayerForward"); // From kernel.cl
        cl::Kernel sigmoidKernel = context_obj.kernels.at("clSigmoid1d"); // From activations.cl

        size_t layer_size = static_cast<size_t>(in);
        size_t buffer_size_bytes = layer_size * sizeof(float);

        cl::Buffer d_current_input(context_obj.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, buffer_size_bytes, const_cast<float*>(input.data()), &err);
        CL_CHECK(err);

        // Buffers for intermediate results (read/write between kernels)
        cl::Buffer d_current_hlayer(context_obj.context, CL_MEM_READ_WRITE, buffer_size_bytes, nullptr, &err);
        CL_CHECK(err);
        cl::Buffer d_current_activation(context_obj.context, CL_MEM_READ_WRITE, buffer_size_bytes, nullptr, &err);
        CL_CHECK(err);

        // Buffer to hold the activation from the *previous* layer (starts as input)
        cl::Buffer d_prev_activation = d_current_input; // Initial input for the first layer

        // --- Process Hidden Layers (0 to layers-1) ---
        for (int i = 0; i < layers; ++i) {
            size_t current_input_size = layer_size; // Input size for this layer (output of previous)
            size_t current_output_size = layer_size; // Output size for this layer

            // Access the mat object for the current layer's weights
            const mat& current_weights_mat = this->weights[i];
            // Validate that the mat dimensions are consistent with 'in'
            // This function assumes weights are conceptually 'in x in'
            if (current_weights_mat.row != static_cast<int>(current_output_size) || current_weights_mat.col != static_cast<int>(current_input_size)) {
                 throw std::runtime_error("MLP clForward: Weight mat dimensions (" + std::to_string(current_weights_mat.row) + "x" + std::to_string(current_weights_mat.col) +
                                          ") mismatch for hidden layer " + std::to_string(i) + ", expected " + std::to_string(in) + "x" + std::to_string(in) + " based on 'in' parameter.");
            }

            // Calculate weights_bytes based on 'in', as the function assumes in x in matrices
            size_t weights_bytes = static_cast<size_t>(in) * in * sizeof(float);

            // Create and write weights buffer for this layer (read-only for kernel)
            cl::Buffer d_weights(context_obj.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, weights_bytes, current_weights_mat.mapped_data, &err);
            CL_CHECK(err);

            // --- Kernel Execution for Hidden Layer i ---

            // 1. Compute Weighted Sum (Layer Forward Kernel)
            //    output = input * weights^T (effectively, handled by kernel indexing)
            //    kernelLayerForward(inputs, weights, outputs, input_size, output_size)
            CL_CHECK(layerForwardKernel.setArg(0, d_prev_activation));    // Input activations (from previous layer or initial input)
            CL_CHECK(layerForwardKernel.setArg(1, d_weights));            // Weights for this layer
            CL_CHECK(layerForwardKernel.setArg(2, d_current_hlayer));     // Output buffer for weighted sum (pre-activation)
            CL_CHECK(layerForwardKernel.setArg(3, static_cast<cl_int>(current_input_size)));
            CL_CHECK(layerForwardKernel.setArg(4, static_cast<cl_int>(current_output_size)));

            cl::NDRange global_fwd(current_output_size); // One work-item per output neuron
            cl::NDRange local_fwd = cl::NullRange;       // Let OpenCL decide local size (or specify if needed)
            CL_CHECK(context_obj.queue.enqueueNDRangeKernel(layerForwardKernel, cl::NullRange, global_fwd, local_fwd));

            // 2. Apply Activation Function (Sigmoid Kernel)
            //    activation = sigmoid(weighted_sum)
            //    clSigmoid1d(input_buffer, output_buffer, size)
            CL_CHECK(sigmoidKernel.setArg(0, d_current_hlayer));      // Input buffer (weighted sum)
            CL_CHECK(sigmoidKernel.setArg(1, d_current_activation));  // Output buffer for activations
            CL_CHECK(sigmoidKernel.setArg(2, static_cast<cl_int>(current_output_size)));

            cl::NDRange global_act(current_output_size); // One work-item per neuron
            cl::NDRange local_act = cl::NullRange;
            CL_CHECK(context_obj.queue.enqueueNDRangeKernel(sigmoidKernel, cl::NullRange, global_act, local_act));

            // --- Prepare for next layer ---
            // The activation of the current layer becomes the input for the next layer
            d_prev_activation = d_current_activation;

            // Optional: Read back intermediate results to host (useful for debugging or backprop)
            CL_CHECK(context_obj.queue.enqueueReadBuffer(d_current_hlayer, CL_TRUE, 0, buffer_size_bytes, hlayers[i].data()));
            CL_CHECK(context_obj.queue.enqueueReadBuffer(d_current_activation, CL_TRUE, 0, buffer_size_bytes, activations[i].data()));
        } // End of hidden layer loop

        // --- Process Output Layer (using weights[layers]) ---
        size_t output_layer_input_size = layer_size; // Input is activation from last hidden layer
        size_t output_layer_output_size = layer_size; // Final output size

        // Access the mat object for the output layer's weights
        const mat& output_weights_mat = this->weights[layers];
        // Validate that the mat dimensions are consistent with 'in'
        if (output_weights_mat.row != static_cast<int>(output_layer_output_size) || output_weights_mat.col != static_cast<int>(output_layer_input_size)) {
             throw std::runtime_error("MLP clForward: Output weight mat dimensions (" + std::to_string(output_weights_mat.row) + "x" + std::to_string(output_weights_mat.col) +
                                      ") mismatch, expected " + std::to_string(in) + "x" + std::to_string(in) + " based on 'in' parameter.");
        }

        // Calculate weights_bytes based on 'in'
        size_t output_weights_bytes = static_cast<size_t>(in) * in * sizeof(float);

        // Create and write output weights buffer (read-only for kernel)
        cl::Buffer d_output_weights(context_obj.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, output_weights_bytes, output_weights_mat.mapped_data, &err);
        CL_CHECK(err);

        // Create buffer for the final network output (write-only for kernel)
        cl::Buffer d_output(context_obj.context, CL_MEM_WRITE_ONLY, buffer_size_bytes, nullptr, &err);
        CL_CHECK(err);

        // --- Kernel Execution for Output Layer ---

        // 1. Compute Weighted Sum for Output Layer (using layerForwardKernel again)
        CL_CHECK(layerForwardKernel.setArg(0, d_prev_activation));    // Input activations (from last hidden layer)
        CL_CHECK(layerForwardKernel.setArg(1, d_output_weights));     // Weights for the output layer
        CL_CHECK(layerForwardKernel.setArg(2, d_output));             // Output buffer for final result (pre-activation if activation applied separately)
        CL_CHECK(layerForwardKernel.setArg(3, static_cast<cl_int>(output_layer_input_size)));
        CL_CHECK(layerForwardKernel.setArg(4, static_cast<cl_int>(output_layer_output_size)));

        cl::NDRange global_out(output_layer_output_size);
        cl::NDRange local_out = cl::NullRange;
        CL_CHECK(context_obj.queue.enqueueNDRangeKernel(layerForwardKernel, cl::NullRange, global_out, local_out));

        // --- Read Final Output Back to Host ---
        // Read from the buffer that holds the final desired result (d_output in this case)
        CL_CHECK(context_obj.queue.enqueueReadBuffer(d_output, CL_TRUE, 0, buffer_size_bytes, output.data()));
    }
     catch (const std::out_of_range& oor) {
        // Specific catch for kernel lookup failure from the map
        std::cerr << "Error: Kernel not found in the shared OpenCLContext kernel map. "
                  << "Ensure kernels 'kernelLayerForward' and 'clSigmoid1d' were provided during OpenCLContext initialization. "
                  << "Details: " << oor.what() << std::endl;
        throw; // Re-throw exception
    }
    catch (const std::exception& e) {
        std::cerr << "Standard Exception in mlp::clForward: " << e.what() << std::endl;
        throw; // Re-throw exception
    }
}

#endif // USE_OPENCL
