#ifdef USE_OPENCL

#ifndef CL_HPP_ENABLE_EXCEPTIONS
    #define CL_HPP_ENABLE_EXCEPTIONS
#endif
#ifndef CL_HPP_TARGET_OPENCL_VERSION
    #define CL_HPP_TARGET_OPENCL_VERSION 300 // Or the version you are targeting
#endif
#ifndef CL_HPP_MINIMUM_OPENCL_VERSION
    #define CL_HPP_MINIMUM_OPENCL_VERSION 120 // Or the minimum version you support
#endif

#include "include/mlp.hpp" // Includes OpenCL headers, getOpenCLErrorString, etc.
#include <vector>
#include <stdexcept>
#include <iostream>
#include <string>
#include <numeric> // Required for std::accumulate
#include <cmath>   // For std::pow (though MSE kernel handles squaring)

#include <CL/cl.hpp> // Explicit include for OpenCL C++ bindings


/**
 * @brief Helper function to calculate MSE using the kernelMseReduction OpenCL kernel.
 * @param expected_vec Host expected vector.
 * @param output_vec Host output vector.
 * @param in Size of the vectors.
 * @return Calculated Mean Squared Error.
 */
float calculateMseOpenCL(const std::vector<float>& expected_vec, const std::vector<float>& output_vec, int in) {
    if (expected_vec.size() != static_cast<size_t>(in) || output_vec.size() != static_cast<size_t>(in)) {
        throw std::runtime_error("calculateMseOpenCL: Vector size mismatch. Expected: " + std::to_string(in));
    }
    if (in == 0) return 0.0f;

    float mse = 0.0f;

    try {
        cl::Kernel kernelMse(program, "kernelMseReduction"); // Use the specific kernel name

        // --- NDRange Configuration for Reduction ---
        // Choose a suitable local work-group size (power of 2 often good, check device limits)        
        size_t local_size = 64; // Example: 64 work-items per group. Tune this based on device.
        // Ensure local_size doesn't exceed device capabilities for this kernel
        // size_t max_work_group_size = kernelMse.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(cl::Device::getDefault());
        // local_size = std::min(local_size, max_work_group_size);

        size_t num_groups = (in + local_size - 1) / local_size; // Number of work-groups needed
        size_t global_size = num_groups * local_size; // Global size must be a multiple of local size

        // --- Device Buffers ---
        size_t vector_size_bytes = sizeof(float) * in;
        size_t partial_mse_size_bytes = sizeof(float) * num_groups; // Buffer to hold one result per group

        // Create buffers and copy host data
        // Using CL_MEM_COPY_HOST_PTR can be efficient if supported well by the driver
        cl::Buffer d_expected(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, vector_size_bytes, const_cast<float*>(expected_vec.data()));
        cl::Buffer d_output(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, vector_size_bytes, const_cast<float*>(output_vec.data()));
        cl::Buffer d_partial_mse(context, CL_MEM_WRITE_ONLY, partial_mse_size_bytes);

        // --- Kernel Execution ---
        kernelMse.setArg(0, d_expected);
        kernelMse.setArg(1, d_output);
        kernelMse.setArg(2, d_partial_mse);
        // Argument 3 is the local memory buffer, specified using cl::Local
        kernelMse.setArg(3, cl::Local(sizeof(float) * local_size)); // local memory
        kernelMse.setArg(4, static_cast<cl_int>(in));             // Pass total size

        // Enqueue the kernel
        queue.enqueueNDRangeKernel(kernelMse, cl::NullRange, cl::NDRange(global_size), cl::NDRange(local_size));

        // --- Read Partial Results and Sum on Host ---
        std::vector<float> h_partial_mse(num_groups);
        // Blocking read to ensure kernel is finished and data is available
        queue.enqueueReadBuffer(d_partial_mse, CL_TRUE, 0, partial_mse_size_bytes, h_partial_mse.data());

        // Sum the partial results from each work-group on the host
        float total_squared_error = std::accumulate(h_partial_mse.begin(), h_partial_mse.end(), 0.0f);

        // Calculate final MSE
        mse = total_squared_error / static_cast<float>(in);

        // Buffers are released automatically by RAII destructors

    }
    catch (const cl::Error& err) {        
        std::cerr << "OpenCL Error during training epoch "<< err.what();
        throw std::runtime_error("OpenCL error during MSE calculation."); // Re-throw
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
            // 1. Forward Pass
            clForward(in, layers); // Updates this->output

            // 2. Calculate MSE using OpenCL helper function
            mse = calculateMseOpenCL(this->expected, this->output, in);

            // 3. Check for convergence
            if (mse < convergence_threshold) {
                std::cout << "Convergence reached after " << e + 1 << " epochs. MSE: " << mse << std::endl;
                break; // Exit loop
            }

            // Print progress (optional, consider printing less frequently)
            if (e % 100 == 0 || e < 10) { // Print every 100 epochs or the first 10
                 std::cout << "Epoch: " << e + 1 << " MSE: " << mse << std::endl;
            }


            // 4. Backward Pass
            // Choose the appropriate backward function based on desired method
            // clBackward(layers, in, learning); // Simple update
            clBackprop(layers, in, learning); // Update with gradient calculation
            // clBackwithL1(layers, in, learning); // L1 regularization
            // clBackwithL2(layers, in, learning); // L2 regularization
            // clBackprop2in(layers, in, learning); // Update input too

            e++;

        } catch (const cl::Error& err) {
            std::cerr << "OpenCL Error during training epoch " << e + 1 << ": "<< err.what();
            throw; // Re-throw after logging
        } catch (const std::exception& ex) {
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
    } catch (...) {
        std::cerr << "Warning: Error during final forward pass after training." << std::endl;
        // Decide how to handle this - maybe the last calculated MSE is sufficient
    }
}


/**
 * @brief Trains the MLP using OpenCL for a dataset of input-output pairs.
 *        Mirrors the logic of mlp::cuTrain (dataset version).
 * @param dataset A vector of input vectors. Assumes `this->expected` is set correctly for each input.
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
                // CRITICAL: Assumes 'this->expected' is correctly set *before* calling clTrain,
                // or that the dataset structure includes expected outputs.
                // If expected output changes per input sample, it needs to be updated here.
                this->input = dataset[i];
                if (this->input.size() != static_cast<size_t>(in)) {
                     throw std::runtime_error("clTrain (dataset): Sample size mismatch at index " + std::to_string(i));
                }


                // 1. Forward Pass
                clForward(in, layers); // Updates this->output

                // 2. Calculate MSE for this sample using OpenCL helper
                float current_sample_mse = calculateMseOpenCL(this->expected, this->output, in);
                total_epoch_squared_error += current_sample_mse * static_cast<float>(in); // Add total squared error for the sample

                // 3. Backward Pass for this sample
                // Choose the appropriate backward function
                // clBackward(layers, in, learning);
                clBackprop(layers, in, learning);
                // clBackwithL1(layers, in, learning);
                // clBackwithL2(layers, in, learning);
                // clBackprop2in(layers, in, learning);

            } // End loop over dataset samples

            e++; // Increment epoch count

            // Calculate average MSE for the completed epoch
            average_epoch_mse = total_epoch_squared_error / (dataset.size() * static_cast<float>(in));

            // Print progress
            std::cout << "Epoch " << e << " Average MSE: " << average_epoch_mse << std::endl;

            // Check for convergence
            if (average_epoch_mse < convergence_threshold) {
                 std::cout << "Convergence reached after " << e << " epochs. Average MSE: " << average_epoch_mse << std::endl;
                break; // Exit loop
            }

        } catch (const cl::Error& err) {
            std::cerr << "OpenCL Error during training epoch " << e + 1 << ": "<< err.what();
            throw; // Re-throw after logging
        } catch (const std::exception& ex) {
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

/**
 * @brief Validates the MLP using OpenCL.
 *        Performs forward propagation on validation data and calculates MSE.
 *        Mirrors the logic of mlp::cuValidate.
 * @param in Number of input/output neurons.
 * @param layers Number of hidden layers.
 */
void mlp::clValidate(int in, int layers) {
    // --- Placeholder for Validation Data ---
    // In a real application, load actual validation data here.
    // For now, using placeholders similar to the CUDA version.
    std::vector<float> validation_input(in, 0.5f); // Example placeholder input
    std::vector<float> validation_expected(in, 0.8f); // Example placeholder expected output
    std::cout << "--- Running OpenCL Validation ---" << std::endl;
    std::cout << "(Using placeholder validation data)" << std::endl;
    // -----------------------------------------

    try {
        // Set the input and expected output for validation
        this->input = validation_input;
        this->expected = validation_expected;

        // Perform forward propagation using OpenCL
        clForward(in, layers); // Updates this->output

        // Calculate Mean Squared Error using the OpenCL helper
        float mse = calculateMseOpenCL(this->expected, this->output, in);

        // Output the validation MSE
        std::cout << "Validation MSE: " << mse << std::endl;

    }
    catch (const cl::Error& err) {
        std::cerr << "OpenCL Error during validation: "<< err.what();
        throw; // Re-throw after logging
    }
    catch (const std::exception& ex) {
        std::cerr << "Standard Exception during validation: " << ex.what() << std::endl;
        throw; // Re-throw
    }
    std::cout << "--- OpenCL Validation Complete ---" << std::endl;
}


/**
 * @brief Tests the MLP using OpenCL.
 *        Performs forward propagation on test data and prints the output vs expected.
 *        Mirrors the logic of mlp::cuTest.
 * @param in Number of input/output neurons.
 * @param layers Number of hidden layers.
 */
void mlp::clTest(int in, int layers) {
    // --- Placeholder for Test Data ---
    // In a real application, load actual test data here.
    // For now, using placeholders similar to the CUDA version.
    std::vector<float> test_input(in, 0.2f);    // Example placeholder input
    std::vector<float> test_expected(in, 0.4f); // Example placeholder expected output
    std::cout << "--- Running OpenCL Test ---" << std::endl;
    std::cout << "(Using placeholder test data)" << std::endl;
    // -------------------------------------

    try {
        // Set the input and expected output for testing
        this->input = test_input;
        this->expected = test_expected; // Store expected for comparison printing

        // Perform forward propagation using OpenCL
        clForward(in, layers); // Updates this->output

        // Output the results: Expected vs Actual Output
        std::cout << "Test Results (Expected <-> Output):" << std::endl;
        if (this->output.size() != this->expected.size()) {
             std::cerr << "Warning: Output and Expected sizes differ during test printout." << std::endl;
        }
        size_t print_count = std::min(this->output.size(), this->expected.size());
        for (size_t i = 0; i < print_count; ++i) {
            std::cout << this->expected[i] << " <-> " << this->output[i] << std::endl;
        }
        if (print_count < this->output.size() || print_count < this->expected.size()) {
             std::cout << "... (sizes mismatched, output truncated)" << std::endl;
        }


    }
    catch (const cl::Error& err) {
        std::cerr << "OpenCL Error during testing: "<< err.what();
        throw; // Re-throw after logging
    } 
    catch (const std::exception& ex) {
        std::cerr << "Standard Exception during testing: " << ex.what() << std::endl;
        throw; // Re-throw
    }
    std::cout << "--- OpenCL Test Complete ---" << std::endl;
}

#endif // USE_OPENCL
