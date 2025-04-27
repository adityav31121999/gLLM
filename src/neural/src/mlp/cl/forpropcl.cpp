
#ifdef USE_OPENCL

#include <vector>
#include <stdexcept>
#include <iostream>
#include <numeric> // For std::inner_product (if needed for verification, though not in CL path)
#include "include/mlp.hpp" // Adjusted path relative to forpropcl.cpp location
#include <maths.hpp>          // For flatten function declaration

#ifndef CL_HPP_ENABLE_EXCEPTIONS
    #define CL_HPP_ENABLE_EXCEPTIONS
#endif
#ifndef CL_HPP_TARGET_OPENCL_VERSION
    #define CL_HPP_TARGET_OPENCL_VERSION 300 // Match setup
#endif
#ifndef CL_HPP_MINIMUM_OPENCL_VERSION
    #define CL_HPP_MINIMUM_OPENCL_VERSION 120 // Match setup
#endif

#include <CL/cl.hpp> // Use C++ bindings


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

    // Ensure host-side vectors for storing results are sized correctly
    // Note: Device buffers will be created with appropriate sizes.
    // These host vectors might only be needed if reading back intermediate results.
    hlayers.resize(layers, std::vector<float>(in, 0.0f));
    activations.resize(layers, std::vector<float>(in, 0.0f));
    output.resize(in, 0.0f); // Resize host output vector

    try {
        // --- Create Kernels ---
        cl::Kernel layerForwardKernel(program, "kernelLayerForward");
        cl::Kernel sigmoidKernel(program, "clSigmoid1d"); // Use the 1D sigmoid kernel

        // --- Determine Sizes ---
        // Assuming 'in' is the size for input, hidden, and output layers based on CPU code.
        size_t layer_size = static_cast<size_t>(in);
        size_t buffer_size_bytes = layer_size * sizeof(float);

        // --- Device Buffers ---
        // Input buffer (read-only for the first layer)
        cl::Buffer d_current_input(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, buffer_size_bytes, const_cast<float*>(input.data()));

        // Buffers for intermediate results (activations) - We only need the previous layer's activation
        cl::Buffer d_prev_activation = d_current_input; // Start with the input buffer
        cl::Buffer d_current_hlayer(context, CL_MEM_READ_WRITE, buffer_size_bytes); // To store pre-activation sums
        cl::Buffer d_current_activation(context, CL_MEM_READ_WRITE, buffer_size_bytes); // To store activations

        // --- Process Hidden Layers (0 to layers-1) ---
        for (int i = 0; i < layers; ++i) {
            // Determine input/output sizes for this layer connection
            // Layer i connects previous layer (size 'in') to current layer (size 'in')
            size_t current_input_size = layer_size; // Size of previous layer's activations (or initial input)
            size_t current_output_size = layer_size; // Size of the current layer

            // Flatten weights for the current layer (weights[i])
            // weights[i] connects previous layer (size current_input_size) to current layer (size current_output_size)
            // It should have dimensions [current_output_size x current_input_size]
            if (weights[i].size() != current_output_size || (!weights[i].empty() && weights[i][0].size() != current_input_size)) {
                 throw std::runtime_error("Weight dimensions mismatch for layer " + std::to_string(i));
            }
            std::vector<float> flat_weights = flatten(weights[i]); // Flatten the current layer's weights
            size_t weights_bytes = flat_weights.size() * sizeof(float);

            // Create and write weights buffer for this layer
            cl::Buffer d_weights(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, weights_bytes, flat_weights.data());

            // 1. Compute Weighted Sum (Forward Kernel)
            // kernelLayerForward(inputs, weights, outputs, input_size, output_size)
            layerForwardKernel.setArg(0, d_prev_activation); // Input is previous layer's activation (or initial input)
            layerForwardKernel.setArg(1, d_weights);
            layerForwardKernel.setArg(2, d_current_hlayer); // Output is the pre-activation sum for this layer
            layerForwardKernel.setArg(3, static_cast<cl_int>(current_input_size));
            layerForwardKernel.setArg(4, static_cast<cl_int>(current_output_size));

            // NDRange for layerForwardKernel (parallel over output neurons)
            cl::NDRange global_fwd(current_output_size);
            // Let OpenCL choose local size, or specify e.g., cl::NDRange(256) if appropriate
            cl::NDRange local_fwd = cl::NullRange;
            queue.enqueueNDRangeKernel(layerForwardKernel, cl::NullRange, global_fwd, local_fwd);

            // 2. Apply Activation Function (Sigmoid Kernel)
            // clSigmoid1d(x, out, size)
            sigmoidKernel.setArg(0, d_current_hlayer); // Input is the pre-activation sum
            sigmoidKernel.setArg(1, d_current_activation); // Output is the activation for this layer
            sigmoidKernel.setArg(2, static_cast<cl_int>(current_output_size));

            // NDRange for sigmoidKernel (parallel over layer neurons)
            cl::NDRange global_act(current_output_size);
            cl::NDRange local_act = cl::NullRange; // Or specify
            queue.enqueueNDRangeKernel(sigmoidKernel, cl::NullRange, global_act, local_act);

            // The output of this layer's activation becomes the input for the next layer
            d_prev_activation = d_current_activation;

            // Optional: Read back intermediate hlayers/activations if needed on host
            // queue.enqueueReadBuffer(d_current_hlayer, CL_TRUE, 0, buffer_size_bytes, hlayers[i].data());
            // queue.enqueueReadBuffer(d_current_activation, CL_TRUE, 0, buffer_size_bytes, activations[i].data());
        }

        // --- Process Output Layer (layers) ---
        // Connects last hidden layer (activations[layers-1]) to output layer
        size_t output_layer_input_size = layer_size; // Size of last hidden layer activation
        size_t output_layer_output_size = layer_size; // Size of the final output layer

        // Flatten weights for the output layer connection (weights[layers])
        if (weights[layers].size() != output_layer_output_size || (!weights[layers].empty() && weights[layers][0].size() != output_layer_input_size)) {
             throw std::runtime_error("Weight dimensions mismatch for output layer");
        }
        std::vector<float> flat_output_weights = flatten(weights[layers]);
        size_t output_weights_bytes = flat_output_weights.size() * sizeof(float);

        // Create and write output weights buffer
        cl::Buffer d_output_weights(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, output_weights_bytes, flat_output_weights.data());

        // Create buffer for the final output (pre-activation, as CPU version doesn't activate output)
        cl::Buffer d_output(context, CL_MEM_WRITE_ONLY, buffer_size_bytes);

        // Compute Weighted Sum for Output Layer
        layerForwardKernel.setArg(0, d_prev_activation); // Input is the last hidden layer's activation
        layerForwardKernel.setArg(1, d_output_weights);
        layerForwardKernel.setArg(2, d_output); // Output is the final network output
        layerForwardKernel.setArg(3, static_cast<cl_int>(output_layer_input_size));
        layerForwardKernel.setArg(4, static_cast<cl_int>(output_layer_output_size));

        cl::NDRange global_out(output_layer_output_size);
        cl::NDRange local_out = cl::NullRange; // Or specify
        queue.enqueueNDRangeKernel(layerForwardKernel, cl::NullRange, global_out, local_out);

        // --- Read Final Output Back to Host ---
        // Blocking read to ensure computation is finished and data is available
        queue.enqueueReadBuffer(d_output, CL_TRUE, 0, buffer_size_bytes, output.data());

        // Optional: Explicitly wait for queue to finish if using non-blocking operations elsewhere
        // queue.finish();

    }
    catch (const cl::Error& err) {
        std::cerr << "OpenCL Error in mlp::clForward: " << err.what() << " (" << err.err() << ")" << std::endl;
        // Print build log if it was a build error (though build happens in initializeOpenCL)
        if (err.err() == CL_BUILD_PROGRAM_FAILURE) {
             std::string log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device);
             std::cerr << "Build Log:\n" << log << std::endl;
        }
        throw; // Re-throw exception
    }
    catch (const std::exception& e) {
        std::cerr << "Standard Exception in mlp::clForward: " << e.what() << std::endl;
        throw; // Re-throw exception
    }
}

#endif // USE_OPENCL
