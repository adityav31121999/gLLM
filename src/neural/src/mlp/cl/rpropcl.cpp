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
#include <vector>
#include <stdexcept>
#include <iostream>
#include <string>
#include <numeric> // For std::accumulate if calculating host MSE
#include <cmath>   // For std::pow if calculating host MSE

/**
 * @brief OpenCL implementation of Rprop algorithm for MLP using shared OpenCLContext.
 * @param dataset Input dataset (vector of input samples). Assumes `this->expected` is set correctly for each sample before processing.
 * @param layers Number of hidden layers (total weight matrices = layers + 1)
 * @param in Input/layer size
 * @param learning Learning rate (Passed to clBackprop, not directly used by Rprop update rule)
 * @param epochs Number of epochs
 */
void mlp::clRprop(std::vector<std::vector<float>>& dataset, int layers, int in, float learning, int epochs) {
    // Rprop parameters
    const float etaPlus = 1.2f;
    const float etaMinus = 0.5f;
    const float deltaMax = 50.0f;
    const float deltaMin = 1e-6f;

    // --- Basic Sanity Checks ---
    if (dataset.empty()) {
        throw std::invalid_argument("MLP clRprop: Dataset cannot be empty.");
    }
    if (dataset[0].size() != static_cast<size_t>(in)) {
         throw std::invalid_argument("MLP clRprop: Dataset sample size mismatch with network input size.");
    }
    // Assuming 'layers' parameter refers to number of hidden layers.
    // Number of weight matrices = layers + 1.
    // This should match this->weights.size() which is this->num_layers - 1.
    if (weights.empty() || weights.size() != static_cast<size_t>(layers + 1) ||
        gweights.size() != static_cast<size_t>(layers + 1) ) {
         throw std::runtime_error("MLP clRprop: Weights or gweights vector size mismatch with 'layers' parameter.");
    }
    // gweights will be calculated by clBackprop

    // --- Host Initialization for Rprop state ---
    std::vector<mat> prev_gradients_mats(layers + 1);
    std::vector<mat> delta_weights_mats(layers + 1);

    for (int l = 0; l <= layers; ++l) {
        // Assuming each mat for Rprop state is in x in
        prev_gradients_mats[l] = mat(in, in); // Creates temp mapped file
        std::fill_n(prev_gradients_mats[l].mapped_data, static_cast<size_t>(in) * in, 0.0f);

        delta_weights_mats[l] = mat(in, in); // Creates temp mapped file
        std::fill_n(delta_weights_mats[l].mapped_data, static_cast<size_t>(in) * in, deltaMin);
    }

    try {
        cl_int err; // For OpenCL error codes
        // --- Access Shared OpenCL Context ---
        OpenCLContext& context_obj = this->clContext; // Use the member reference

        // --- OpenCL Kernel Preparation (Retrieve from context) ---
        // Ensure this kernel name was provided during OpenCLContext initialization
        cl::Kernel kernelRprop = context_obj.kernels.at("kernelRpropUpdate");

        // --- Device Buffer Allocation (Allocate ONCE outside loops using shared context) ---
        size_t weights_size_bytes = sizeof(float) * in * in;
        std::vector<cl::Buffer> d_weights(layers + 1);
        std::vector<cl::Buffer> d_gradients(layers + 1);
        std::vector<cl::Buffer> d_prev_gradients(layers + 1);
        std::vector<cl::Buffer> d_delta_weights(layers + 1);

        for (int l = 0; l <= layers; ++l) {
            d_weights[l] = cl::Buffer(context_obj.context, CL_MEM_READ_WRITE, weights_size_bytes, nullptr, &err);
            CL_CHECK(err);
            d_gradients[l] = cl::Buffer(context_obj.context, CL_MEM_READ_ONLY, weights_size_bytes, nullptr, &err); // Gradients are input to Rprop kernel
            CL_CHECK(err);
            d_prev_gradients[l] = cl::Buffer(context_obj.context, CL_MEM_READ_WRITE, weights_size_bytes, nullptr, &err);
            CL_CHECK(err);
            d_delta_weights[l] = cl::Buffer(context_obj.context, CL_MEM_READ_WRITE, weights_size_bytes, nullptr, &err);
            CL_CHECK(err);

            // --- Initial Data Transfer for Rprop state (using shared queue) ---
            CL_CHECK(context_obj.queue.enqueueWriteBuffer(d_prev_gradients[l], CL_TRUE, 0, weights_size_bytes, prev_gradients_mats[l].mapped_data));
            CL_CHECK(context_obj.queue.enqueueWriteBuffer(d_delta_weights[l], CL_TRUE, 0, weights_size_bytes, delta_weights_mats[l].mapped_data));
        }


        // --- NDRange Configuration (for Rprop kernel) ---
        // Kernel is 1D, iterating over all weights in a layer matrix
        cl::NDRange global_rprop(in * in);
        cl::NDRange local_rprop = cl::NullRange; // Let runtime choose

        cl_int cl_size = static_cast<cl_int>(in * in);
        cl_float cl_etaPlus = static_cast<cl_float>(etaPlus);
        cl_float cl_etaMinus = static_cast<cl_float>(etaMinus);
        cl_float cl_deltaMax = static_cast<cl_float>(deltaMax);
        cl_float cl_deltaMin = static_cast<cl_float>(deltaMin);

        // --- Epoch Loop ---
        for (int epoch = 0; epoch < epochs; ++epoch) {
            float totalEpochError = 0.0f;

            // --- Dataset Loop ---
            // Note: Processing samples sequentially here. Batch processing would require modifications.
            for (size_t sample_idx = 0; sample_idx < dataset.size(); ++sample_idx) {
                // CRITICAL: Assumes 'this->expected' is correctly set for the current sample
                // before calling clRprop, or the dataset includes expected outputs.
                this->input = dataset[sample_idx];
                // Ensure expected output is set if it varies per sample.
                // e.g., if dataset is vector<pair<vector<float>, vector<float>>>
                // this->expected = dataset[sample_idx].second;

                // 1. Forward Pass (Uses shared context internally)
                this->clForward(in, layers); // Updates this->output

                // 2. Backward Pass (Calculate Gradients using clBackprop - uses shared context internally)
                // This updates host gweights and host weights (using standard GD step, which Rprop overrides)
                this->clBackprop(in, layers, learning); // 'learning' is used by clBackprop, not Rprop kernel

                // 3. Calculate Sample Error (on host for simplicity)
                float sample_error = 0.0f;
                if (this->expected.size() != this->output.size()) {
                     throw std::runtime_error("MLP clRprop: Mismatch between expected and output vector sizes during error calculation.");
                }
                for (int i = 0; i < in; ++i) {
                    sample_error += std::pow(this->expected[i] - this->output[i], 2);
                }
                sample_error /= in; // MSE for this sample
                totalEpochError += sample_error;

                // 4. Rprop Weight Update (using calculated gradients)
                for (int l = 0; l <= layers; ++l) {
                    // Copy current weights (potentially modified by clBackprop)
                    // and calculated gradients (this->gweights) H->D for this layer (using shared queue)
                    const mat& current_weights_mat = this->weights[l];
                    const mat& current_gweights_mat = this->gweights[l]; // Use gradients from clBackprop

                    // Validate mat dimensions against 'in' if strict checking is desired
                    if (current_weights_mat.row != in || current_weights_mat.col != in ||
                        current_gweights_mat.row != in || current_gweights_mat.col != in) {
                        // throw std::runtime_error("MLP clRprop: Mat dimensions do not match 'in' parameter.");
                    }

                    CL_CHECK(context_obj.queue.enqueueWriteBuffer(d_weights[l], CL_TRUE, 0, weights_size_bytes, current_weights_mat.mapped_data));
                    CL_CHECK(context_obj.queue.enqueueWriteBuffer(d_gradients[l], CL_TRUE, 0, weights_size_bytes, current_gweights_mat.mapped_data));
                    // d_prev_gradients and d_delta_weights are already on device and hold state

                    // Launch Rprop Kernel for layer l (using shared queue)
                    CL_CHECK(kernelRprop.setArg(0, d_weights[l]));
                    CL_CHECK(kernelRprop.setArg(1, d_gradients[l]));
                    CL_CHECK(kernelRprop.setArg(2, d_prev_gradients[l])); // Read/Write
                    CL_CHECK(kernelRprop.setArg(3, d_delta_weights[l]));  // Read/Write
                    CL_CHECK(kernelRprop.setArg(4, cl_etaPlus));
                    CL_CHECK(kernelRprop.setArg(5, cl_etaMinus));
                    CL_CHECK(kernelRprop.setArg(6, cl_deltaMax));
                    CL_CHECK(kernelRprop.setArg(7, cl_deltaMin));
                    CL_CHECK(kernelRprop.setArg(8, cl_size));
                    CL_CHECK(context_obj.queue.enqueueNDRangeKernel(kernelRprop, cl::NullRange, global_rprop, local_rprop));

                    // Read updated weights, prev_gradients, delta_weights D->H (using shared queue)
                    // Data is read directly into the mapped_data of the mat objects
                    CL_CHECK(context_obj.queue.enqueueReadBuffer(d_weights[l], CL_TRUE, 0, weights_size_bytes, this->weights[l].mapped_data));
                    CL_CHECK(context_obj.queue.enqueueReadBuffer(d_prev_gradients[l], CL_TRUE, 0, weights_size_bytes, prev_gradients_mats[l].mapped_data));
                    CL_CHECK(context_obj.queue.enqueueReadBuffer(d_delta_weights[l], CL_TRUE, 0, weights_size_bytes, delta_weights_mats[l].mapped_data));
                } // End layer loop for Rprop update
            } // End dataset loop

            // --- End of Epoch ---
            totalEpochError /= dataset.size();
            std::cout << "Epoch " << epoch + 1 << "/" << epochs << " - Mean Squared Error: " << totalEpochError << std::endl;

            // Check for convergence
            if (totalEpochError < 0.01) { // Example threshold
                this->status = true;
                std::cout << "Convergence threshold reached." << std::endl;
                break;
            }
        } // End epoch loop

        // --- Cleanup ---
        // Buffers are released automatically by RAII destructors

    } catch (const std::out_of_range& oor) {
        // Specific catch for kernel lookup failure from the map
        std::cerr << "Error: Kernel not found in the shared OpenCLContext kernel map during clRprop. "
                  << "Ensure kernel 'kernelRpropUpdate' was provided during OpenCLContext initialization. "
                  << "Details: " << oor.what() << std::endl;
        throw; // Re-throw exception
    }
    catch (const std::exception& e) {
        std::cerr << "Standard Exception in mlp::clRprop: " << e.what() << std::endl;
        throw; // Re-throw standard exceptions
    }
}

#endif // USE_OPENCL
