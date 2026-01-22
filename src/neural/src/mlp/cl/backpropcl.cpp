#ifdef USE_CL
#include "include/mlp.hpp"
#include <maths.hpp>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <string>


/**
 * @brief OpenCL implementation of backpropagation for MLP with gradients using shared OpenCLContext.
 *        Calculates deltas (error * derivative), calculates gradients,
 *        and updates weights. Mirrors cuBackprop logic.
 * @param in Input/layer size (assuming all layers have the same size 'in')
 * @param layers Number of hidden layers (total weight matrices = layers + 1)
 * @param learning Learning rate
 */
void mlp::clBackprop(float learning) {
    // --- Basic Sanity Checks ---
    // Check weight mat dimensions against 'in' (conceptual size for this function)
    for (size_t l = 0; l <= static_cast<size_t>(num_layers); ++l) {
        if (weights[l].row != in || weights[l].col != in) {
             // This indicates a mismatch between 'in' and actual mat dimensions.
             // Depending on strictness, could throw or log. For now, proceed with 'in'.
             std::cerr << "Warning: MLP clBackprop: Weight mat dimensions at layer " << l << " ("
                       << weights[l].row << "x" << weights[l].col << ") do not match 'in' parameter (" << in << ")." << std::endl;
        }
    }
    if (num_layers > 0) {
        for (size_t l = 0; l < static_cast<size_t>(num_layers); ++l) {
            if (activations[l].empty() || activations[l].size() != static_cast<size_t>(in)) {
                throw std::runtime_error("MLP clBackprop: Activation dimensions error at layer " + std::to_string(l));
            }
        }
    }


    // --- Initialize Host gweights (mat objects) ---
    // The mlp constructor already sizes gweights. This loop zeros out their mapped_data.
    for (int l = 0; l <= num_layers; ++l) {
        // Assuming gweights[l] is conceptually in x in for this function.
        // The actual mat dimensions are set by layer_sizes in constructor.
        if (gweights[l].mapped_data && (gweights[l].row == in && gweights[l].col == in)) {
            std::fill_n(gweights[l].mapped_data, static_cast<size_t>(in) * in, 0.0f);
        } else if (gweights[l].mapped_data) {
             std::cerr << "Warning: MLP clBackprop: gweights mat dimensions at layer " << l << " ("
                       << gweights[l].row << "x" << gweights[l].col << ") do not match 'in' parameter (" << in << "). Zeroing based on 'in'." << std::endl;
            // Still zero out based on 'in * in' if mapped_data is valid, respecting function's logic.
            std::fill_n(gweights[l].mapped_data, static_cast<size_t>(in) * in, 0.0f);
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
        cl::Kernel kernelUpdateWAndG = context_obj.kernels.at("kernelUpdateWeightsAndGradients"); // Kernel that calculates gradients

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
        std::vector<cl::Buffer> d_activations(num_layers); // Hidden layer activations only

        // Buffers for weights, gradients, and deltas (one per layer, including output)
        std::vector<cl::Buffer> d_weights(num_layers + 1);
        std::vector<cl::Buffer> d_gweights(num_layers + 1); // Gradients
        std::vector<cl::Buffer> d_layer_deltas(num_layers + 1);
        for (int l = 0; l <= num_layers; ++l) {
            d_weights[l] = cl::Buffer(context_obj.context, CL_MEM_READ_WRITE, weights_size_bytes, nullptr, &err); // Weights are updated
            CL_CHECK(err);
            d_gweights[l] = cl::Buffer(context_obj.context, CL_MEM_WRITE_ONLY, weights_size_bytes, nullptr, &err); // Gradients are calculated
            CL_CHECK(err);
            d_layer_deltas[l] = cl::Buffer(context_obj.context, CL_MEM_READ_WRITE, layer_size_bytes, nullptr, &err); // Deltas are read/written
            CL_CHECK(err);
        }
        // Allocate hidden layer activation buffers
        for (int l = 0; l < num_layers; ++l) {
             d_activations[l] = cl::Buffer(context_obj.context, CL_MEM_READ_ONLY, layer_size_bytes, nullptr, &err);
             CL_CHECK(err);
        }


        // --- Data Transfer: Host -> Device (using shared queue) ---
        CL_CHECK(context_obj.queue.enqueueWriteBuffer(d_input, CL_TRUE, 0, layer_size_bytes, input.data()));
        CL_CHECK(context_obj.queue.enqueueWriteBuffer(d_output_activations, CL_TRUE, 0, layer_size_bytes, output.data())); // Assuming output holds activated values
        CL_CHECK(context_obj.queue.enqueueWriteBuffer(d_expected, CL_TRUE, 0, layer_size_bytes, expected.data()));

        // Copy hidden activations
        for (int l = 0; l < num_layers; ++l) {
            CL_CHECK(context_obj.queue.enqueueWriteBuffer(d_activations[l], CL_TRUE, 0, layer_size_bytes, activations[l].data()));
        }
        // Copy initial weights
        for (int l = 0; l <= num_layers; ++l) {
            // Use mapped_data directly
            CL_CHECK(context_obj.queue.enqueueWriteBuffer(d_weights[l], CL_TRUE, 0, weights_size_bytes, weights[l].mapped_data));
        }

        // --- NDRange Configuration ---
        cl::NDRange global_1d(in);
        cl::NDRange local_1d = cl::NullRange;
        cl::NDRange global_2d(in, in); // For weight/gradient updates
        cl::NDRange local_2d = cl::NullRange;

        cl_int cl_in = static_cast<cl_int>(in);
        float cl_learning = static_cast<float>(learning);

        // --- Backpropagation Steps ---

        // 1. Calculate Output Layer Deltas (Layer index 'layers')
        CL_CHECK(kernelOutDelta.setArg(0, d_output_activations)); // Activated output
        CL_CHECK(kernelOutDelta.setArg(1, d_expected));
        CL_CHECK(kernelOutDelta.setArg(2, d_layer_deltas[num_layers])); // Output delta buffer
        CL_CHECK(kernelOutDelta.setArg(3, cl_in));
        CL_CHECK(context_obj.queue.enqueueNDRangeKernel(kernelOutDelta, cl::NullRange, global_1d, local_1d));

        // 2. Calculate Hidden Layer Deltas (Propagate backwards from layers-1 down to 0)
        for (int l = num_layers - 1; l >= 0; --l) {
            CL_CHECK(kernelHiddenDelta.setArg(0, d_layer_deltas[l + 1])); // Deltas from the next layer (l+1)
            CL_CHECK(kernelHiddenDelta.setArg(1, d_weights[l + 1]));      // Weights connecting layer l to layer l+1
            CL_CHECK(kernelHiddenDelta.setArg(2, d_activations[l]));      // Activations of the current layer (l)
            CL_CHECK(kernelHiddenDelta.setArg(3, d_layer_deltas[l]));     // Deltas to compute for the current layer (l)
            CL_CHECK(kernelHiddenDelta.setArg(4, cl_in));                 // current_layer_size
            CL_CHECK(kernelHiddenDelta.setArg(5, cl_in));                 // next_layer_size
            CL_CHECK(context_obj.queue.enqueueNDRangeKernel(kernelHiddenDelta, cl::NullRange, global_1d, local_1d));
        }

        // 3. Calculate Gradients and Update Weights (Iterate through layers 0 to layers)
        for (int l = 0; l <= num_layers; ++l) {
            // Determine the buffer containing activations from the previous layer
            cl::Buffer& d_prev_activations_buffer = (l == 0) ? d_input : d_activations[l - 1];

            CL_CHECK(kernelUpdateWAndG.setArg(0, d_layer_deltas[l]));         // Deltas for the current layer (l)
            CL_CHECK(kernelUpdateWAndG.setArg(1, d_prev_activations_buffer)); // Activations from the previous layer (l-1 or input)
            CL_CHECK(kernelUpdateWAndG.setArg(2, d_weights[l]));              // Weights W[l] (read/write)
            CL_CHECK(kernelUpdateWAndG.setArg(3, d_gweights[l]));             // Gradients gW[l] (write-only)
            CL_CHECK(kernelUpdateWAndG.setArg(4, cl_learning));               // Learning rate
            CL_CHECK(kernelUpdateWAndG.setArg(5, cl_in));                     // current_layer_size
            CL_CHECK(kernelUpdateWAndG.setArg(6, cl_in));                     // prev_layer_size
            CL_CHECK(context_obj.queue.enqueueNDRangeKernel(kernelUpdateWAndG, cl::NullRange, global_2d, local_2d));
        }

        // --- Data Transfer: Device -> Host ---
        // Wait for kernels to finish before reading back (or use blocking reads CL_TRUE)
        CL_CHECK(context_obj.queue.finish()); // Ensure all kernels complete before reading

        // Copy updated weights back to the host mlp object
        for (int l = 0; l <= num_layers; ++l) {
            CL_CHECK(context_obj.queue.enqueueReadBuffer(d_weights[l], CL_TRUE, 0, weights_size_bytes, weights[l].mapped_data));
        }

        // Copy calculated gradients back to the host mlp object
        for (int l = 0; l <= num_layers; ++l) {
            CL_CHECK(context_obj.queue.enqueueReadBuffer(d_gweights[l], CL_TRUE, 0, weights_size_bytes, gweights[l].mapped_data));
        }

        // --- Cleanup ---
        // Buffers are released automatically by RAII destructors
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
void mlp::clBackprop2in(float learning) {
    // --- Basic Sanity Checks ---
    // --- Initialize Host gweights (mat objects) ---
    for (int l = 0; l <= num_layers; ++l) {
        if (gweights[l].mapped_data && (gweights[l].row == in && gweights[l].col == in)) {
            std::fill_n(gweights[l].mapped_data, static_cast<size_t>(in) * in, 0.0f);
        } else if (gweights[l].mapped_data) {
             std::cerr << "Warning: MLP clBackprop2in: gweights mat dimensions at layer " << l << " ("
                       << gweights[l].row << "x" << gweights[l].col << ") do not match 'in' parameter (" << in << "). Zeroing based on 'in'." << std::endl;
            std::fill_n(gweights[l].mapped_data, static_cast<size_t>(in) * in, 0.0f);
        } else {
            // Potentially throw if gweights[l] is not usable
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
        cl::Kernel kernelUpdateWAndG = context_obj.kernels.at("kernelUpdateWeightsAndGradients");
        cl::Kernel kernelUpdateIn = context_obj.kernels.at("kernelUpdateInputMLP"); // Kernel to update input

        // --- Device Buffer Allocation (using shared context) ---
        size_t layer_size_bytes = sizeof(float) * in;
        size_t weights_size_bytes = sizeof(float) * in * in;

        // Buffers for inputs/outputs/activations
        cl::Buffer d_input(context_obj.context, CL_MEM_READ_WRITE, layer_size_bytes, nullptr, &err); // Input is updated
        CL_CHECK(err);
        cl::Buffer d_output_activations(context_obj.context, CL_MEM_READ_ONLY, layer_size_bytes, nullptr, &err);
        CL_CHECK(err);
        cl::Buffer d_expected(context_obj.context, CL_MEM_READ_ONLY, layer_size_bytes, nullptr, &err);
        CL_CHECK(err);
        std::vector<cl::Buffer> d_activations(num_layers); // Hidden layer activations

        // Buffers for weights, gradients, and deltas
        std::vector<cl::Buffer> d_weights(num_layers + 1);
        std::vector<cl::Buffer> d_gweights(num_layers + 1); // Gradients
        std::vector<cl::Buffer> d_layer_deltas(num_layers + 1);
        for (int l = 0; l <= num_layers; ++l) {
            d_weights[l] = cl::Buffer(context_obj.context, CL_MEM_READ_WRITE, weights_size_bytes, nullptr, &err);
            CL_CHECK(err);
            d_gweights[l] = cl::Buffer(context_obj.context, CL_MEM_WRITE_ONLY, weights_size_bytes, nullptr, &err);
            CL_CHECK(err);
            d_layer_deltas[l] = cl::Buffer(context_obj.context, CL_MEM_READ_WRITE, layer_size_bytes, nullptr, &err);
            CL_CHECK(err);
        }
        for (int l = 0; l < num_layers; ++l) {
             d_activations[l] = cl::Buffer(context_obj.context, CL_MEM_READ_ONLY, layer_size_bytes, nullptr, &err);
             CL_CHECK(err);
        }

        // --- Data Transfer: Host -> Device (using shared queue) ---
        CL_CHECK(context_obj.queue.enqueueWriteBuffer(d_input, CL_TRUE, 0, layer_size_bytes, input.data())); // Initial input
        CL_CHECK(context_obj.queue.enqueueWriteBuffer(d_output_activations, CL_TRUE, 0, layer_size_bytes, output.data()));
        CL_CHECK(context_obj.queue.enqueueWriteBuffer(d_expected, CL_TRUE, 0, layer_size_bytes, expected.data()));
        for (int l = 0; l < num_layers; ++l) {
            CL_CHECK(context_obj.queue.enqueueWriteBuffer(d_activations[l], CL_TRUE, 0, layer_size_bytes, activations[l].data()));
        }
        for (int l = 0; l <= num_layers; ++l) {
            CL_CHECK(context_obj.queue.enqueueWriteBuffer(d_weights[l], CL_TRUE, 0, weights_size_bytes, weights[l].mapped_data));
        }

        // --- NDRange Configuration ---
        cl::NDRange global_1d(in);
        cl::NDRange local_1d = cl::NullRange;
        cl::NDRange global_2d(in, in);
        cl::NDRange local_2d = cl::NullRange;

        cl_int cl_in = static_cast<cl_int>(in);
        float cl_learning = static_cast<float>(learning);

        // --- Backpropagation Steps ---

        // 1. Calculate Output Layer Deltas (Layer index 'layers')
        CL_CHECK(kernelOutDelta.setArg(0, d_output_activations));
        CL_CHECK(kernelOutDelta.setArg(1, d_expected));
        CL_CHECK(kernelOutDelta.setArg(2, d_layer_deltas[num_layers]));
        CL_CHECK(kernelOutDelta.setArg(3, cl_in));
        CL_CHECK(context_obj.queue.enqueueNDRangeKernel(kernelOutDelta, cl::NullRange, global_1d, local_1d));

        // 2. Calculate Hidden Layer Deltas (Propagate backwards from layers-1 down to 0)
        for (int l = num_layers - 1; l >= 0; --l) {
            CL_CHECK(kernelHiddenDelta.setArg(0, d_layer_deltas[l + 1]));
            CL_CHECK(kernelHiddenDelta.setArg(1, d_weights[l + 1]));
            CL_CHECK(kernelHiddenDelta.setArg(2, d_activations[l])); // Activations of current layer l
            CL_CHECK(kernelHiddenDelta.setArg(3, d_layer_deltas[l]));
            CL_CHECK(kernelHiddenDelta.setArg(4, cl_in));
            CL_CHECK(kernelHiddenDelta.setArg(5, cl_in));
            CL_CHECK(context_obj.queue.enqueueNDRangeKernel(kernelHiddenDelta, cl::NullRange, global_1d, local_1d));
        }

        // 3. Calculate Gradients and Update Weights (Iterate through layers 0 to layers)
        for (int l = 0; l <= num_layers; ++l) {
            cl::Buffer& d_prev_activations_buffer = (l == 0) ? d_input : d_activations[l - 1];

            CL_CHECK(kernelUpdateWAndG.setArg(0, d_layer_deltas[l]));
            CL_CHECK(kernelUpdateWAndG.setArg(1, d_prev_activations_buffer));
            CL_CHECK(kernelUpdateWAndG.setArg(2, d_weights[l]));
            CL_CHECK(kernelUpdateWAndG.setArg(3, d_gweights[l])); // Output gradients
            CL_CHECK(kernelUpdateWAndG.setArg(4, cl_learning));
            CL_CHECK(kernelUpdateWAndG.setArg(5, cl_in));
            CL_CHECK(kernelUpdateWAndG.setArg(6, cl_in));
            CL_CHECK(context_obj.queue.enqueueNDRangeKernel(kernelUpdateWAndG, cl::NullRange, global_2d, local_2d));
        }

        // 4. Update Input Vector using first layer's deltas and weights
        CL_CHECK(kernelUpdateIn.setArg(0, d_input));              // Input buffer (read/write)
        CL_CHECK(kernelUpdateIn.setArg(1, d_weights[0]));         // Weights W[0]
        CL_CHECK(kernelUpdateIn.setArg(2, d_layer_deltas[0]));    // Deltas delta[0]
        CL_CHECK(kernelUpdateIn.setArg(3, cl_learning));          // Learning rate
        CL_CHECK(kernelUpdateIn.setArg(4, cl_in));                // first_hidden_layer_size (assuming == in)
        CL_CHECK(kernelUpdateIn.setArg(5, cl_in));                // input_size (assuming == in)
        CL_CHECK(context_obj.queue.enqueueNDRangeKernel(kernelUpdateIn, cl::NullRange, global_1d, local_1d)); // 1D kernel over input size

        // --- Data Transfer: Device -> Host ---
        CL_CHECK(context_obj.queue.finish()); // Ensure all kernels are done before reading

        // Copy updated weights back
        for (int l = 0; l <= num_layers; ++l) {
            CL_CHECK(context_obj.queue.enqueueReadBuffer(d_weights[l], CL_TRUE, 0, weights_size_bytes, weights[l].mapped_data));
        }

        // Copy calculated gradients back
        for (int l = 0; l <= num_layers; ++l) {
            CL_CHECK(context_obj.queue.enqueueReadBuffer(d_gweights[l], CL_TRUE, 0, weights_size_bytes, gweights[l].mapped_data));
        }

        // Copy updated input back
        CL_CHECK(context_obj.queue.enqueueReadBuffer(d_input, CL_TRUE, 0, layer_size_bytes, input.data()));


        // --- Cleanup ---
        // Buffers are released automatically by RAII destructors
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

#endif // USE_CL
