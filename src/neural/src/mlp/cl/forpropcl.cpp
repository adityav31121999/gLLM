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
        OpenCLContext& context_obj = this->clContext; // Use the member reference

        cl::Kernel layerForwardKernel = context_obj.kernels.at("kernelLayerForward"); // From kernel.cl
        cl::Kernel sigmoidKernel = context_obj.kernels.at("clSigmoid1d"); // From activations.cl

        size_t layer_size = static_cast<size_t>(in);
        size_t buffer_size_bytes = layer_size * sizeof(float);

        cl::Buffer d_current_input(context_obj.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, buffer_size_bytes, const_cast<float*>(input.data()));

        // Buffers for intermediate results (read/write between kernels)
        cl::Buffer d_current_hlayer(context_obj.context, CL_MEM_READ_WRITE, buffer_size_bytes);
        cl::Buffer d_current_activation(context_obj.context, CL_MEM_READ_WRITE, buffer_size_bytes);

        // Buffer to hold the activation from the *previous* layer (starts as input)
        cl::Buffer d_prev_activation = d_current_input; // Initial input for the first layer

        // --- Process Hidden Layers (0 to layers-1) ---
        for (int i = 0; i < layers; ++i) {
            size_t current_input_size = layer_size; // Input size for this layer (output of previous)
            size_t current_output_size = layer_size; // Output size for this layer

            if (weights[i].size() != current_output_size || (!weights[i].empty() && weights[i][0].size() != current_input_size)) {
                 throw std::runtime_error("Weight dimensions mismatch for hidden layer " + std::to_string(i));
            }
            // Flatten weights for the current layer
            std::vector<float> flat_weights = flatten(weights[i]);
            size_t weights_bytes = flat_weights.size() * sizeof(float);

            // Create and write weights buffer for this layer (read-only for kernel)
            cl::Buffer d_weights(context_obj.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, weights_bytes, flat_weights.data());

            // --- Kernel Execution for Hidden Layer i ---

            // 1. Compute Weighted Sum (Layer Forward Kernel)
            //    output = input * weights^T (effectively, handled by kernel indexing)
            //    kernelLayerForward(inputs, weights, outputs, input_size, output_size)
            layerForwardKernel.setArg(0, d_prev_activation);    // Input activations (from previous layer or initial input)
            layerForwardKernel.setArg(1, d_weights);            // Weights for this layer
            layerForwardKernel.setArg(2, d_current_hlayer);     // Output buffer for weighted sum (pre-activation)
            layerForwardKernel.setArg(3, static_cast<cl_int>(current_input_size));
            layerForwardKernel.setArg(4, static_cast<cl_int>(current_output_size));

            cl::NDRange global_fwd(current_output_size); // One work-item per output neuron
            cl::NDRange local_fwd = cl::NullRange;       // Let OpenCL decide local size (or specify if needed)
            context_obj.queue.enqueueNDRangeKernel(layerForwardKernel, cl::NullRange, global_fwd, local_fwd);

            // 2. Apply Activation Function (Sigmoid Kernel)
            //    activation = sigmoid(weighted_sum)
            //    clSigmoid1d(input_buffer, output_buffer, size)
            sigmoidKernel.setArg(0, d_current_hlayer);      // Input buffer (weighted sum)
            sigmoidKernel.setArg(1, d_current_activation);  // Output buffer for activations
            sigmoidKernel.setArg(2, static_cast<cl_int>(current_output_size));

            cl::NDRange global_act(current_output_size); // One work-item per neuron
            cl::NDRange local_act = cl::NullRange;
            context_obj.queue.enqueueNDRangeKernel(sigmoidKernel, cl::NullRange, global_act, local_act);

            // --- Prepare for next layer ---
            // The activation of the current layer becomes the input for the next layer
            d_prev_activation = d_current_activation;

            // Optional: Read back intermediate results to host (useful for debugging or backprop)
            context_obj.queue.enqueueReadBuffer(d_current_hlayer, CL_TRUE, 0, buffer_size_bytes, hlayers[i].data());
            context_obj.queue.enqueueReadBuffer(d_current_activation, CL_TRUE, 0, buffer_size_bytes, activations[i].data());
        } // End of hidden layer loop

        // --- Process Output Layer (using weights[layers]) ---
        size_t output_layer_input_size = layer_size; // Input is activation from last hidden layer
        size_t output_layer_output_size = layer_size; // Final output size

        if (weights[layers].size() != output_layer_output_size || (!weights[layers].empty() && weights[layers][0].size() != output_layer_input_size)) {
             throw std::runtime_error("Weight dimensions mismatch for output layer");
        }
        std::vector<float> flat_output_weights = flatten(weights[layers]);
        size_t output_weights_bytes = flat_output_weights.size() * sizeof(float);

        // Create and write output weights buffer (read-only for kernel)
        cl::Buffer d_output_weights(context_obj.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, output_weights_bytes, flat_output_weights.data());

        // Create buffer for the final network output (write-only for kernel)
        cl::Buffer d_output(context_obj.context, CL_MEM_WRITE_ONLY, buffer_size_bytes);

        // --- Kernel Execution for Output Layer ---

        // 1. Compute Weighted Sum for Output Layer (using layerForwardKernel again)
        layerForwardKernel.setArg(0, d_prev_activation);    // Input activations (from last hidden layer)
        layerForwardKernel.setArg(1, d_output_weights);     // Weights for the output layer
        layerForwardKernel.setArg(2, d_output);             // Output buffer for final result (pre-activation if activation applied separately)
        layerForwardKernel.setArg(3, static_cast<cl_int>(output_layer_input_size));
        layerForwardKernel.setArg(4, static_cast<cl_int>(output_layer_output_size));

        cl::NDRange global_out(output_layer_output_size);
        cl::NDRange local_out = cl::NullRange;
        context_obj.queue.enqueueNDRangeKernel(layerForwardKernel, cl::NullRange, global_out, local_out);

        // NOTE: Applying activation to the output layer depends on the network design.
        // If the output layer needs sigmoid activation, you would enqueue the sigmoidKernel here:
        /*
        cl::Buffer d_final_activation(context_obj.context, CL_MEM_WRITE_ONLY, buffer_size_bytes); // Need a separate buffer if d_output was pre-activation
        sigmoidKernel.setArg(0, d_output); // Input is the weighted sum from layerForwardKernel
        sigmoidKernel.setArg(1, d_final_activation); // Output is the final activated output
        sigmoidKernel.setArg(2, static_cast<cl_int>(output_layer_output_size));
        context_obj.queue.enqueueNDRangeKernel(sigmoidKernel, cl::NullRange, global_out, local_out); // Reuse global range
        // Then read from d_final_activation instead of d_output
        */
        // Assuming for now the output layer does NOT have activation applied by clForward itself.

        // --- Read Final Output Back to Host ---
        // Read from the buffer that holds the final desired result (d_output in this case)
        context_obj.queue.enqueueReadBuffer(d_output, CL_TRUE, 0, buffer_size_bytes, output.data());

        // Explicitly wait for queue to finish (optional, enqueueReadBuffer with CL_TRUE is blocking)
        // context_obj.queue.finish();

    }
    catch (const cl::Error& err) {
        std::cerr << "OpenCL Error in mlp::clForward: " << err.what() << " (" << err.err() << ")" << std::endl;
        // Print build log if it was a build error - Access program and device via the SHARED context object
        if (err.err() == CL_BUILD_PROGRAM_FAILURE) {
             try {
                // Use this->clContext to get program and device
                std::string log = this->clContext.program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(this->clContext.device);
                std::cerr << "Build Log:\n" << log << std::endl;
             } catch (const cl::Error& log_err) {
                 std::cerr << "Could not retrieve build log: " << log_err.what() << " (" << log_err.err() << ")" << std::endl;
             }
        }
        throw; // Re-throw exception
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
