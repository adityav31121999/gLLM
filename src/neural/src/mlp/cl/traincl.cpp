#ifdef USE_OPENCL
#include "include/mlp.hpp" // Includes OpenCL headers, OpenCLContext, etc.
#include <maths.hpp>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <string>
#include <numeric> // Required for std::accumulate
#include <cmath>   // For std::pow (though MSE kernel handles squaring)

/**
 * @brief Helper function to calculate MSE using the kernelMseReduction OpenCL kernel
 *        and the provided OpenCLContext.
 * @param context_obj Reference to the shared OpenCLContext object.
 * @param expected_vec Host expected vector.
 * @param output_vec Host output vector.
 * @param in Size of the vectors.
 * @return Calculated Mean Squared Error.
 */
// Make it static as it's only used within this file
static float calculateMseOpenCL(OpenCLContext& context_obj, const std::vector<float>& expected_vec, const std::vector<float>& output_vec, int in) {
    if (expected_vec.size() != static_cast<size_t>(in) || output_vec.size() != static_cast<size_t>(in)) {
        throw std::runtime_error("calculateMseOpenCL: Vector size mismatch. Expected: " + std::to_string(in));
    }
    if (in == 0) return 0.0f;

    float mse = 0.0f;

    try {
        cl_int cl_err; // For OpenCL error codes
        // --- Access Kernel from Shared Context ---
        cl::Kernel kernelMse = context_obj.kernels.at("kernelMseReduction"); // Use the specific kernel name

        // --- NDRange Configuration for Reduction ---
        size_t local_size = 64; // Example: Tune this based on device.
        // Optional: Query device limits if needed
        try {
             size_t max_wg_size = 0;
             // Use the getWorkGroupInfo version that takes an error code pointer
             max_wg_size = kernelMse.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(context_obj.device, &cl_err);
             if (cl_err == CL_SUCCESS) {
                if (local_size > max_wg_size && max_wg_size > 0) {
                    // std::cout << "Warning: Requested local size (" << local_size
                    //           << ") exceeds device limit (" << max_wg_size
                    //           << "). Clamping to max limit." << std::endl;
                    local_size = max_wg_size;
                }
             } else {
                std::cerr << "Warning: Could not query CL_KERNEL_WORK_GROUP_SIZE for kernelMseReduction. Using default local_size=" << local_size << ". Error: " << oclErrorString(cl_err) << " (" << cl_err << ")" << std::endl;
             }
        } catch (const std::out_of_range& oor_kernel_lookup) { // Catch if kernel itself wasn't found for getWorkGroupInfo
            std::cerr << "Warning: Kernel 'kernelMseReduction' not found when trying to get work group info. Using default local_size=" << local_size << ". Details: " << oor_kernel_lookup.what() << std::endl;
        } // Other std::exceptions will propagate

        size_t num_groups = (in + local_size - 1) / local_size;
        size_t global_size = num_groups * local_size;

        // --- Device Buffers (using shared context) ---
        size_t vector_size_bytes = sizeof(float) * in;
        size_t partial_mse_size_bytes = sizeof(float) * num_groups;

        // Create buffers using the shared context
        cl::Buffer d_expected(context_obj.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, vector_size_bytes, const_cast<float*>(expected_vec.data()), &cl_err);
        CL_CHECK(cl_err);
        cl::Buffer d_output(context_obj.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, vector_size_bytes, const_cast<float*>(output_vec.data()), &cl_err);
        CL_CHECK(cl_err);
        cl::Buffer d_partial_mse(context_obj.context, CL_MEM_WRITE_ONLY, partial_mse_size_bytes, nullptr, &cl_err);
        CL_CHECK(cl_err);

        // --- Kernel Execution (using shared queue) ---
        CL_CHECK(kernelMse.setArg(0, d_expected));
        CL_CHECK(kernelMse.setArg(1, d_output));
        CL_CHECK(kernelMse.setArg(2, d_partial_mse));
        CL_CHECK(kernelMse.setArg(3, cl::Local(sizeof(float) * local_size))); // local memory
        CL_CHECK(kernelMse.setArg(4, static_cast<cl_int>(in)));

        // Enqueue the kernel using the shared queue
        CL_CHECK(context_obj.queue.enqueueNDRangeKernel(kernelMse, cl::NullRange, cl::NDRange(global_size), cl::NDRange(local_size)));

        // --- Read Partial Results and Sum on Host (using shared queue) ---
        std::vector<float> h_partial_mse(num_groups);
        // Blocking read using the shared queue
        CL_CHECK(context_obj.queue.enqueueReadBuffer(d_partial_mse, CL_TRUE, 0, partial_mse_size_bytes, h_partial_mse.data()));

        // Sum the partial results from each work-group on the host
        float total_squared_error = std::accumulate(h_partial_mse.begin(), h_partial_mse.end(), 0.0f);

        // Calculate final MSE
        mse = total_squared_error / static_cast<float>(in);
    }
    catch (const std::out_of_range& oor) {
        // Specific catch for kernel lookup failure from the map
        std::cerr << "Error: Kernel 'kernelMseReduction' not found in the shared OpenCLContext kernel map during MSE calculation. "
                  << "Ensure this kernel was provided during OpenCLContext initialization. "
                  << "Details: " << oor.what() << std::endl;
        throw; // Re-throw exception
    }
    // Note: std::runtime_error from CL_CHECK will be caught by the caller's std::exception handler
    catch (const std::exception& e) {
        std::cerr << "Standard Exception during MSE calculation: " << e.what() << std::endl;
        throw; // Re-throw standard exceptions
    }


    return mse;
}


/**
 * @brief Trains the MLP using OpenCL for a single input-output pair.
 *        Mirrors the logic of mlp::cuTrain (single sample version).
 * @param mse Reference to the Mean Squared Error (MSE) value.
 * @param in Number of input/output neurons.
 * @param layers Number of hidden layers.
 * @param learning The learning rate.
 */
void mlp::clTrain(float& mse, int in, int layers, float learning) {
    unsigned int e = 0;

    const unsigned int max_epochs = 10000; // Add a max epoch limit to prevent infinite loops
    const float convergence_threshold = 1e-6f;

    if (input.empty() || expected.empty()) {
        throw std::runtime_error("clTrain (single): Input or Expected vector is empty.");
    }

    std::cout << "Starting OpenCL Training (Single Sample)..." << std::endl;

    while (e < max_epochs) {
        try {
            // 1. Forward Pass (Uses shared context internally)
            clForward(in, layers); // Updates output

            // 2. Calculate MSE using OpenCL helper function (Pass shared context)
            mse = calculateMseOpenCL(clContext, expected, output, in);

            // 3. Check for convergence
            if (mse < convergence_threshold) {
                std::cout << "Convergence reached after " << e + 1 << " epochs. MSE: " << mse << std::endl;
                break; // Exit loop
            }

            // Print progress (optional, consider printing less frequently)
            if (e % 100 == 0 || e < 10) { // Print every 100 epochs or the first 10
                 std::cout << "Epoch: " << e + 1 << " MSE: " << mse << std::endl;
            }


            // 4. Backward Pass (Uses shared context internally)
            // Choose the appropriate backward function based on desired method
            // clBackward(layers, in, learning); // Simple update
            clBackprop(layers, in, learning); // Update with gradient calculation
            // clBackwithL1(layers, in, learning); // L1 regularization
            // clBackwithL2(layers, in, learning); // L2 regularization
            // clBackprop2in(layers, in, learning); // Update input too

            e++;

        }  catch (const std::exception& ex) { // Catches std::runtime_error from CL_CHECK and other std exceptions
            std::cerr << "Standard Exception during training epoch " << e + 1 << ": " << ex.what() << std::endl;
            throw; // Re-throw
        }
    } // End while loop

    if (e == max_epochs) {
        std::cout << "Training stopped after reaching max epochs (" << max_epochs << "). Final MSE: " << mse << std::endl;
    }

    // Perform a final forward pass to ensure 'output' reflects the trained weights
    try {
        clForward(in, layers);
    } catch (const std::exception& final_fwd_ex) { // Catch standard exception too
        std::cerr << "Warning: Error during final forward pass after training: " << final_fwd_ex.what() << std::endl;
        // Decide how to handle this - maybe the last calculated MSE is sufficient
    }
}


/**
 * @brief Trains the MLP using OpenCL for a dataset of input-output pairs.
 *        Mirrors the logic of mlp::cuTrain (dataset version).
 * @param dataset A vector of input vectors. Assumes `expected` is set correctly for each input.
 * @param mse Reference to the Mean Squared Error (MSE) value (average over the last epoch).
 * @param in Number of input/output neurons.
 * @param layers Number of hidden layers.
 * @param learning The learning rate.
 */
void mlp::clTrain(std::vector<std::vector<float>>& dataset, float& mse, int in, int layers, float learning) {
    unsigned int e = 0;
    float average_epoch_mse = 0.0;
    const unsigned int max_epochs = 1000; // Add a max epoch limit
    const float convergence_threshold = 1e-7f; // Convergence threshold for average MSE

    if (dataset.empty()) {
        throw std::runtime_error("clTrain (dataset): Input dataset cannot be empty.");
    }

    std::cout << "Starting OpenCL Training (Dataset)..." << std::endl;

    while (e < max_epochs) {
        float total_epoch_squared_error = 0.0; // Accumulate squared error for the epoch

        try {
            // Loop through each sample in the dataset
            for (size_t i = 0; i < dataset.size(); ++i) {
                // Set the current input and expected output
                // CRITICAL: Assumes 'expected' is correctly set *before* calling clTrain,
                // or that the dataset structure includes expected outputs.
                // If expected output changes per input sample, it needs to be updated here.
                input = dataset[i];
                if (input.size() != static_cast<size_t>(in)) {
                     throw std::runtime_error("clTrain (dataset): Sample size mismatch at index " + std::to_string(i));
                }
                // Example: If dataset was std::vector<std::pair<std::vector<float>, std::vector<float>>>
                // expected = dataset[i].second;


                // 1. Forward Pass (Uses shared context internally)
                clForward(in, layers); // Updates output

                // 2. Calculate MSE for this sample using OpenCL helper (Pass shared context)
                float current_sample_mse = calculateMseOpenCL(clContext, expected, output, in);
                total_epoch_squared_error += current_sample_mse * static_cast<float>(in); // Add total squared error for the sample

                // 3. Backward Pass for this sample (Uses shared context internally)
                // Choose the appropriate backward function
                // clBackward(layers, in, learning);
                clBackprop(layers, in, learning);
                // clBackwithL1(layers, in, learning);
                // clBackwithL2(layers, in, learning);
                // clBackprop2in(layers, in, learning);

            } // End loop over dataset samples

            e++; // Increment epoch count

            // Calculate average MSE for the completed epoch
            if (dataset.empty() || in == 0) {
                average_epoch_mse = 0.0f; // Avoid division by zero
            } else {
                average_epoch_mse = total_epoch_squared_error / (dataset.size() * static_cast<float>(in));
            }


            // Print progress
            std::cout << "Epoch " << e << " Average MSE: " << average_epoch_mse << std::endl;

            // Check for convergence
            if (average_epoch_mse < convergence_threshold) {
                 std::cout << "Convergence reached after " << e << " epochs. Average MSE: " << average_epoch_mse << std::endl;
                break; // Exit loop
            }

        } catch (const std::exception& ex) { // Catches std::runtime_error from CL_CHECK and other std exceptions
            std::cerr << "Standard Exception during training epoch " << e << ": " << ex.what() << std::endl;
            throw; // Re-throw
        }
    } // End while loop (epochs)

     if (e == max_epochs) {
        std::cout << "Training stopped after reaching max epochs (" << max_epochs << "). Final Average MSE: " << average_epoch_mse << std::endl;
    }

    // Set the output mse parameter to the average MSE of the last completed epoch
    mse = average_epoch_mse;
}

#endif // USE_OPENCL
