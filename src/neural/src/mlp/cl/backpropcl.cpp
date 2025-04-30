#ifdef USE_OPENCL

#include "include/mlp.hpp" // Includes basic.hpp where OpenCLContext is defined
#include <maths.hpp>       // Includes basic utilities like flatten, unflatten
#include <vector>
#include <stdexcept>
#include <iostream>
#include <string>
#include <CL/cl.hpp> // Keep OpenCL headers

/**
 * @brief OpenCL implementation of backpropagation for MLP with gradients using shared OpenCLContext.
 *        Calculates deltas (error * derivative), calculates gradients,
 *        and updates weights. Mirrors cuBackprop logic.
 * @param in Input/layer size (assuming all layers have the same size 'in')
 * @param layers Number of hidden layers (total weight matrices = layers + 1)
 * @param learning Learning rate
 */
void mlp::clBackprop(int in, int layers, float learning) {
    // --- Basic Sanity Checks ---
    if (output.size() != static_cast<size_t>(in) || expected.size() != static_cast<size_t>(in)) {
        throw std::invalid_argument("MLP clBackprop: Output/Expected vector size mismatch.");
    }
    if (weights.empty() || weights.size() != static_cast<size_t>(layers + 1)) {
         throw std::runtime_error("MLP clBackprop: Weights vector size mismatch.");
    }
     if (layers > 0 && (activations.empty() || activations.size() != static_cast<size_t>(layers))) {
        throw std::runtime_error("MLP clBackprop: Activations vector size mismatch.");
    }
    if (input.size() != static_cast<size_t>(in)) {
        throw std::runtime_error("MLP clBackprop: Input vector size mismatch.");
    }
    // Check dimensions
    for (size_t l = 0; l <= static_cast<size_t>(layers); ++l) {
        if (weights[l].empty() || weights[l].size() != static_cast<size_t>(in) || weights[l][0].size() != static_cast<size_t>(in)) {
             throw std::runtime_error("MLP clBackprop: Weight dimensions error at layer " + std::to_string(l));
        }
    }
     if (layers > 0) {
        for (size_t l = 0; l < static_cast<size_t>(layers); ++l) {
            if (activations[l].empty() || activations[l].size() != static_cast<size_t>(in)) {
                throw std::runtime_error("MLP clBackprop: Activation dimensions error at layer " + std::to_string(l));
            }
        }
     }


    // --- Initialize Host gweights if necessary ---
    if (gweights.size() != static_cast<size_t>(layers + 1)) {
        gweights.resize(layers + 1);
    }
    for (int l = 0; l <= layers; ++l) {
        if (gweights[l].size() != static_cast<size_t>(in)) {
            gweights[l].resize(in);
        }
        for (int i = 0; i < in; ++i) {
            if (gweights[l][i].size() != static_cast<size_t>(in)) {
                gweights[l][i].resize(in, 0.0f); // Initialize with 0.0f
            } else {
                 // Optionally clear existing gradients if accumulating over batches is not intended here
                 std::fill(gweights[l][i].begin(), gweights[l][i].end(), 0.0f);
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
        cl::Kernel kernelUpdateWAndG = context_obj.kernels.at("kernelUpdateWeightsAndGradients"); // Kernel that calculates gradients

        // --- Device Buffer Allocation (using shared context) ---
        size_t layer_size_bytes = sizeof(float) * in;
        size_t weights_size_bytes = sizeof(float) * in * in; // Assumes square matrices

        // Buffers for inputs/outputs/activations
        cl::Buffer d_input(context_obj.context, CL_MEM_READ_ONLY, layer_size_bytes);
        // Assuming output holds *activated* values for kernelOutputDeltaSigmoid
        cl::Buffer d_output_activations(context_obj.context, CL_MEM_READ_ONLY, layer_size_bytes);
        cl::Buffer d_expected(context_obj.context, CL_MEM_READ_ONLY, layer_size_bytes);
        std::vector<cl::Buffer> d_activations(layers); // Hidden layer activations only

        // Buffers for weights, gradients, and deltas (one per layer, including output)
        std::vector<cl::Buffer> d_weights(layers + 1);
        std::vector<cl::Buffer> d_gweights(layers + 1); // Gradients
        std::vector<cl::Buffer> d_layer_deltas(layers + 1);

        for (int l = 0; l <= layers; ++l) {
            d_weights[l] = cl::Buffer(context_obj.context, CL_MEM_READ_WRITE, weights_size_bytes); // Weights are updated
            d_gweights[l] = cl::Buffer(context_obj.context, CL_MEM_WRITE_ONLY, weights_size_bytes); // Gradients are calculated
            d_layer_deltas[l] = cl::Buffer(context_obj.context, CL_MEM_READ_WRITE, layer_size_bytes); // Deltas are read/written
        }
        // Allocate hidden layer activation buffers
        for (int l = 0; l < layers; ++l) {
             d_activations[l] = cl::Buffer(context_obj.context, CL_MEM_READ_ONLY, layer_size_bytes);
        }


        // --- Data Transfer: Host -> Device (using shared queue) ---
        context_obj.queue.enqueueWriteBuffer(d_input, CL_TRUE, 0, layer_size_bytes, input.data());
        context_obj.queue.enqueueWriteBuffer(d_output_activations, CL_TRUE, 0, layer_size_bytes, output.data()); // Assuming output holds activated values
        context_obj.queue.enqueueWriteBuffer(d_expected, CL_TRUE, 0, layer_size_bytes, expected.data());

        // Copy hidden activations
        for (int l = 0; l < layers; ++l) {
            context_obj.queue.enqueueWriteBuffer(d_activations[l], CL_TRUE, 0, layer_size_bytes, activations[l].data());
        }
        // Copy initial weights
        for (int l = 0; l <= layers; ++l) {
            std::vector<float> flat_weights_l = flatten(weights[l]);
            context_obj.queue.enqueueWriteBuffer(d_weights[l], CL_TRUE, 0, weights_size_bytes, flat_weights_l.data());
        }

        // --- NDRange Configuration ---
        cl::NDRange global_1d(in);
        cl::NDRange local_1d = cl::NullRange;
        cl::NDRange global_2d(in, in); // For weight/gradient updates
        cl::NDRange local_2d = cl::NullRange;

        cl_int cl_in = static_cast<cl_int>(in);
        cl_float cl_learning = static_cast<cl_float>(learning);

        // --- Backpropagation Steps ---

        // 1. Calculate Output Layer Deltas (Layer index 'layers')
        kernelOutDelta.setArg(0, d_output_activations); // Activated output
        kernelOutDelta.setArg(1, d_expected);
        kernelOutDelta.setArg(2, d_layer_deltas[layers]); // Output delta buffer
        kernelOutDelta.setArg(3, cl_in);
        context_obj.queue.enqueueNDRangeKernel(kernelOutDelta, cl::NullRange, global_1d, local_1d);

        // 2. Calculate Hidden Layer Deltas (Propagate backwards from layers-1 down to 0)
        for (int l = layers - 1; l >= 0; --l) {
            kernelHiddenDelta.setArg(0, d_layer_deltas[l + 1]); // Deltas from the next layer (l+1)
            kernelHiddenDelta.setArg(1, d_weights[l + 1]);      // Weights connecting layer l to layer l+1
            kernelHiddenDelta.setArg(2, d_activations[l]);      // Activations of the current layer (l)
            kernelHiddenDelta.setArg(3, d_layer_deltas[l]);     // Deltas to compute for the current layer (l)
            kernelHiddenDelta.setArg(4, cl_in);                 // current_layer_size
            kernelHiddenDelta.setArg(5, cl_in);                 // next_layer_size
            context_obj.queue.enqueueNDRangeKernel(kernelHiddenDelta, cl::NullRange, global_1d, local_1d);
        }

        // 3. Calculate Gradients and Update Weights (Iterate through layers 0 to layers)
        for (int l = 0; l <= layers; ++l) {
            // Determine the buffer containing activations from the previous layer
            cl::Buffer& d_prev_activations_buffer = (l == 0) ? d_input : d_activations[l - 1];

            kernelUpdateWAndG.setArg(0, d_layer_deltas[l]);         // Deltas for the current layer (l)
            kernelUpdateWAndG.setArg(1, d_prev_activations_buffer); // Activations from the previous layer (l-1 or input)
            kernelUpdateWAndG.setArg(2, d_weights[l]);              // Weights W[l] (read/write)
            kernelUpdateWAndG.setArg(3, d_gweights[l]);             // Gradients gW[l] (write-only)
            kernelUpdateWAndG.setArg(4, cl_learning);               // Learning rate
            kernelUpdateWAndG.setArg(5, cl_in);                     // current_layer_size
            kernelUpdateWAndG.setArg(6, cl_in);                     // prev_layer_size
            context_obj.queue.enqueueNDRangeKernel(kernelUpdateWAndG, cl::NullRange, global_2d, local_2d);
        }

        // --- Data Transfer: Device -> Host ---
        // Wait for kernels to finish before reading back (or use blocking reads CL_TRUE)
        context_obj.queue.finish(); // Ensure all kernels complete before reading

        // Copy updated weights back to the host mlp object
        for (int l = 0; l <= layers; ++l) {
            std::vector<float> updated_flat_weights(in * in);
            context_obj.queue.enqueueReadBuffer(d_weights[l], CL_TRUE, 0, weights_size_bytes, updated_flat_weights.data());
            unflatten(updated_flat_weights, weights[l], in, in);
        }

        // Copy calculated gradients back to the host mlp object
        for (int l = 0; l <= layers; ++l) {
            std::vector<float> calculated_flat_gradients(in * in);
            context_obj.queue.enqueueReadBuffer(d_gweights[l], CL_TRUE, 0, weights_size_bytes, calculated_flat_gradients.data());
            unflatten(calculated_flat_gradients, gweights[l], in, in);
        }

        // --- Cleanup ---
        // Buffers are released automatically by RAII destructors

    }
    catch (const cl::Error& err) {
        std::cerr << "OpenCL Error in mlp::clBackprop: " << err.what() << " (" << err.err() << ")" << std::endl;
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
        throw std::runtime_error("OpenCL error during backpropagation."); // Re-throw as runtime_error
    }
     catch (const std::out_of_range& oor) {
        // Specific catch for kernel lookup failure from the map
        std::cerr << "Error: Kernel not found in the shared OpenCLContext kernel map during clBackprop. "
                  << "Ensure kernels 'kernelOutputDeltaSigmoid', 'kernelHiddenDeltaSigmoid', and 'kernelUpdateWeightsAndGradients' were provided during OpenCLContext initialization. "
                  << "Details: " << oor.what() << std::endl;
        throw; // Re-throw exception
    }
    catch (const std::exception& e) {
        std::cerr << "Standard Exception in mlp::clBackprop: " << e.what() << std::endl;
        throw; // Re-throw standard exceptions
    }
}

/**
 * @brief OpenCL implementation of backpropagation that also updates the input vector using shared OpenCLContext.
 *        Calculates deltas, gradients, updates weights, and updates the input vector.
 * @param in Input/layer size (assuming all layers have the same size 'in')
 * @param layers Number of hidden layers (total weight matrices = layers + 1)
 * @param learning Learning rate
 */
void mlp::clBackprop2in(int in, int layers, float learning) {
    // --- Basic Sanity Checks ---
    // (Same checks as clBackprop)
    if (output.size() != static_cast<size_t>(in) || expected.size() != static_cast<size_t>(in)) {
        throw std::invalid_argument("MLP clBackprop2in: Output/Expected vector size mismatch.");
    }
    if (weights.empty() || weights.size() != static_cast<size_t>(layers + 1)) {
         throw std::runtime_error("MLP clBackprop2in: Weights vector size mismatch.");
    }
     if (layers > 0 && (activations.empty() || activations.size() != static_cast<size_t>(layers))) {
        throw std::runtime_error("MLP clBackprop2in: Activations vector size mismatch.");
    }
    if (input.size() != static_cast<size_t>(in)) {
        throw std::runtime_error("MLP clBackprop2in: Input vector size mismatch.");
    }
    for (size_t l = 0; l <= static_cast<size_t>(layers); ++l) {
        if (weights[l].empty() || weights[l].size() != static_cast<size_t>(in) || weights[l][0].size() != static_cast<size_t>(in)) {
             throw std::runtime_error("MLP clBackprop2in: Weight dimensions error at layer " + std::to_string(l));
        }
    }
    if (layers > 0) {
        for (size_t l = 0; l < static_cast<size_t>(layers); ++l) {
            if (activations[l].empty() || activations[l].size() != static_cast<size_t>(in)) {
                throw std::runtime_error("MLP clBackprop2in: Activation dimensions error at layer " + std::to_string(l));
            }
        }
    }

    // --- Initialize Host gweights if necessary ---
    if (gweights.size() != static_cast<size_t>(layers + 1)) {
        gweights.resize(layers + 1);
    }
    for (int l = 0; l <= layers; ++l) {
        if (gweights[l].size() != static_cast<size_t>(in)) {
            gweights[l].resize(in);
        }
        for (int i = 0; i < in; ++i) {
            if (gweights[l][i].size() != static_cast<size_t>(in)) {
                gweights[l][i].resize(in, 0.0f);
            } else {
                 // Optionally clear existing gradients
                 std::fill(gweights[l][i].begin(), gweights[l][i].end(), 0.0f);
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
        cl::Kernel kernelUpdateWAndG = context_obj.kernels.at("kernelUpdateWeightsAndGradients");
        cl::Kernel kernelUpdateIn = context_obj.kernels.at("kernelUpdateInputMLP"); // Kernel to update input

        // --- Device Buffer Allocation (using shared context) ---
        size_t layer_size_bytes = sizeof(float) * in;
        size_t weights_size_bytes = sizeof(float) * in * in;

        // Buffers for inputs/outputs/activations
        cl::Buffer d_input(context_obj.context, CL_MEM_READ_WRITE, layer_size_bytes); // Input is updated
        cl::Buffer d_output_activations(context_obj.context, CL_MEM_READ_ONLY, layer_size_bytes);
        cl::Buffer d_expected(context_obj.context, CL_MEM_READ_ONLY, layer_size_bytes);
        std::vector<cl::Buffer> d_activations(layers); // Hidden layer activations

        // Buffers for weights, gradients, and deltas
        std::vector<cl::Buffer> d_weights(layers + 1);
        std::vector<cl::Buffer> d_gweights(layers + 1); // Gradients
        std::vector<cl::Buffer> d_layer_deltas(layers + 1);

        for (int l = 0; l <= layers; ++l) {
            d_weights[l] = cl::Buffer(context_obj.context, CL_MEM_READ_WRITE, weights_size_bytes);
            d_gweights[l] = cl::Buffer(context_obj.context, CL_MEM_WRITE_ONLY, weights_size_bytes);
            d_layer_deltas[l] = cl::Buffer(context_obj.context, CL_MEM_READ_WRITE, layer_size_bytes);
        }
        for (int l = 0; l < layers; ++l) {
             d_activations[l] = cl::Buffer(context_obj.context, CL_MEM_READ_ONLY, layer_size_bytes);
        }

        // --- Data Transfer: Host -> Device (using shared queue) ---
        context_obj.queue.enqueueWriteBuffer(d_input, CL_TRUE, 0, layer_size_bytes, input.data()); // Initial input
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
        cl::NDRange global_2d(in, in);
        cl::NDRange local_2d = cl::NullRange;

        cl_int cl_in = static_cast<cl_int>(in);
        cl_float cl_learning = static_cast<cl_float>(learning);

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
            kernelHiddenDelta.setArg(1, d_weights[l + 1]);
            kernelHiddenDelta.setArg(2, d_activations[l]); // Activations of current layer l
            kernelHiddenDelta.setArg(3, d_layer_deltas[l]);
            kernelHiddenDelta.setArg(4, cl_in);
            kernelHiddenDelta.setArg(5, cl_in);
            context_obj.queue.enqueueNDRangeKernel(kernelHiddenDelta, cl::NullRange, global_1d, local_1d);
        }

        // 3. Calculate Gradients and Update Weights (Iterate through layers 0 to layers)
        for (int l = 0; l <= layers; ++l) {
            cl::Buffer& d_prev_activations_buffer = (l == 0) ? d_input : d_activations[l - 1];

            kernelUpdateWAndG.setArg(0, d_layer_deltas[l]);
            kernelUpdateWAndG.setArg(1, d_prev_activations_buffer);
            kernelUpdateWAndG.setArg(2, d_weights[l]);
            kernelUpdateWAndG.setArg(3, d_gweights[l]); // Output gradients
            kernelUpdateWAndG.setArg(4, cl_learning);
            kernelUpdateWAndG.setArg(5, cl_in);
            kernelUpdateWAndG.setArg(6, cl_in);
            context_obj.queue.enqueueNDRangeKernel(kernelUpdateWAndG, cl::NullRange, global_2d, local_2d);
        }

        // 4. Update Input Vector using first layer's deltas and weights
        kernelUpdateIn.setArg(0, d_input);              // Input buffer (read/write)
        kernelUpdateIn.setArg(1, d_weights[0]);         // Weights W[0]
        kernelUpdateIn.setArg(2, d_layer_deltas[0]);    // Deltas delta[0]
        kernelUpdateIn.setArg(3, cl_learning);          // Learning rate
        kernelUpdateIn.setArg(4, cl_in);                // first_hidden_layer_size (assuming == in)
        kernelUpdateIn.setArg(5, cl_in);                // input_size (assuming == in)
        context_obj.queue.enqueueNDRangeKernel(kernelUpdateIn, cl::NullRange, global_1d, local_1d); // 1D kernel over input size

        // --- Data Transfer: Device -> Host ---
        context_obj.queue.finish(); // Ensure all kernels are done before reading

        // Copy updated weights back
        for (int l = 0; l <= layers; ++l) {
            std::vector<float> updated_flat_weights(in * in);
            context_obj.queue.enqueueReadBuffer(d_weights[l], CL_TRUE, 0, weights_size_bytes, updated_flat_weights.data());
            unflatten(updated_flat_weights, weights[l], in, in);
        }

        // Copy calculated gradients back
        for (int l = 0; l <= layers; ++l) {
            std::vector<float> calculated_flat_gradients(in * in);
            context_obj.queue.enqueueReadBuffer(d_gweights[l], CL_TRUE, 0, weights_size_bytes, calculated_flat_gradients.data());
            unflatten(calculated_flat_gradients, gweights[l], in, in);
        }

        // Copy updated input back
        context_obj.queue.enqueueReadBuffer(d_input, CL_TRUE, 0, layer_size_bytes, input.data());


        // --- Cleanup ---
        // Buffers are released automatically by RAII destructors

    }
    catch (const cl::Error& err) {
        std::cerr << "OpenCL Error in mlp::clBackprop2in: " << err.what() << " (" << err.err() << ")" << std::endl;
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
        throw std::runtime_error("OpenCL error during backpropagation to input."); // Re-throw as runtime_error
    }
     catch (const std::out_of_range& oor) {
        // Specific catch for kernel lookup failure from the map
        std::cerr << "Error: Kernel not found in the shared OpenCLContext kernel map during clBackprop2in. "
                  << "Ensure kernels 'kernelOutputDeltaSigmoid', 'kernelHiddenDeltaSigmoid', 'kernelUpdateWeightsAndGradients', and 'kernelUpdateInputMLP' were provided during OpenCLContext initialization. "
                  << "Details: " << oor.what() << std::endl;
        throw; // Re-throw exception
    }
    catch (const std::exception& e) {
        std::cerr << "Standard Exception in mlp::clBackprop2in: " << e.what() << std::endl;
        throw; // Re-throw standard exceptions
    }
}

#endif // USE_OPENCL
