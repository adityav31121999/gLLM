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

#include <stdexcept>
#include <iostream>
#include <cmath>
#include <numeric>
#include "include/mlp.hpp"  // Adjusted path relative to losscl.cpp location
#include <maths.hpp>        // Assumed accessible via include paths

#include <CL/cl.hpp> // Use C++ bindings

/**
 * @brief Calculate Mean Squared Error (MSE) using the 'kernelMseReduction' OpenCL kernel.
 *        MSE = (1 / size) * sum((expected[i] - output[i])^2)
 * @param expected_vec Host vector of expected values.
 * @param output_vec Host vector of actual output values.
 * @param in Unused parameter (size is derived from vectors).
 * @return The calculated Mean Squared Error.
 */
float mlp::clMSE(const std::vector<float> &expected_vec, const std::vector<float> &output_vec, int /*in*/) // Marked 'in' as unused
{
    if (expected_vec.size() != output_vec.size()) {
        throw std::invalid_argument("Expected and output vector sizes must match for MSE calculation.");
    }

    size_t size = expected_vec.size();
    if (size == 0) {
        return 0.0f; // MSE of empty vectors is typically 0.
    }

    float final_mse = 0.0f;

    try {
        // 1. Determine work sizes
        size_t local_size = 256; // Example: 256 work-items per group. Should be <= device max work group size.
        size_t num_groups = (size + local_size - 1) / local_size; // Ceiling division
        size_t global_size = num_groups * local_size; // Global size must be multiple of local size

        // 2. Allocate device memory and copy host data
        cl::Buffer d_expected(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size * sizeof(float), const_cast<float*>(expected_vec.data()));
        cl::Buffer d_output(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size * sizeof(float), const_cast<float*>(output_vec.data()));
        // Buffer for partial results (one float per work-group)
        cl::Buffer d_partial_mse(context, CL_MEM_WRITE_ONLY, num_groups * sizeof(float));

        // 3. Create kernel object from the global program object
        cl::Kernel kernel(program, "kernelMseReduction"); // Using kernel from mlp/cl/kernel.cl

        // 4. Set kernel arguments (matching kernelMseReduction signature)
        kernel.setArg(0, d_expected);
        kernel.setArg(1, d_output);
        kernel.setArg(2, d_partial_mse);
        kernel.setArg(3, cl::Local(local_size * sizeof(float))); // Allocate local memory
        kernel.setArg(4, static_cast<cl_int>(size));             // Pass total size

        // 5. Launch kernel
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(global_size), cl::NDRange(local_size));

        // 6. Read partial results back to host
        std::vector<float> h_partial_mse(num_groups);
        queue.enqueueReadBuffer(d_partial_mse, CL_TRUE, 0, num_groups * sizeof(float), h_partial_mse.data()); // Blocking read

        // 7. Perform final reduction (summation) on the host
        float sum_sq_diff = std::accumulate(h_partial_mse.begin(), h_partial_mse.end(), 0.0f);

        // 8. Calculate the final MSE
        final_mse = sum_sq_diff / static_cast<float>(size);

        // queue.finish(); // Not strictly needed after blocking read

    }
    catch (const cl::Error& err) {
        std::cerr << "OpenCL Error in clMSE: " << err.what() << " (" << err.err() << ")" << std::endl;
        throw; // Re-throw
    }
    catch (const std::exception& e) {
        std::cerr << "Standard Exception in clMSE: " << e.what() << std::endl;
        throw; // Re-throw
    }

    return final_mse;
}


/**
 * @brief Calculate L1 penalty (sum of absolute weights) using the 'l1PenaltyKernel' OpenCL kernel.
 * @param weights 3D vector of weights (flattened before processing).
 * @return L1 penalty value.
 */
float clgetL1Penalty(const std::vector<std::vector<std::vector<float>>>& weights) {
    std::vector<float> flat_weights = flattenWeights(weights); // Flatten the weights
    if (flat_weights.empty()) return 0.0f;

    size_t size = flat_weights.size();
    float final_result = 0.0f;

    try {
        // 1. Determine work sizes
        size_t local_size = 256;
        size_t num_groups = (size + local_size - 1) / local_size;
        size_t global_size = num_groups * local_size;

        // 2. Allocate device memory
        cl::Buffer d_weights(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size * sizeof(float), const_cast<float*>(flat_weights.data()));
        cl::Buffer d_partial_results(context, CL_MEM_WRITE_ONLY, num_groups * sizeof(float));

        // 3. Create kernel
        cl::Kernel kernel(program, "l1PenaltyKernel"); // Using kernel from mlp/cl/kernel.cl

        // 4. Set kernel arguments (matching l1PenaltyKernel signature)
        kernel.setArg(0, d_weights);
        kernel.setArg(1, d_partial_results);
        kernel.setArg(2, cl::Local(local_size * sizeof(float))); // Local memory
        kernel.setArg(3, static_cast<cl_int>(size));             // Total size

        // 5. Launch kernel
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(global_size), cl::NDRange(local_size));

        // 6. Read partial results back to host
        std::vector<float> h_partial_results(num_groups);
        queue.enqueueReadBuffer(d_partial_results, CL_TRUE, 0, num_groups * sizeof(float), h_partial_results.data()); // Blocking read

        // 7. Perform final reduction on host
        final_result = std::accumulate(h_partial_results.begin(), h_partial_results.end(), 0.0f);

        // queue.finish(); // Not strictly needed

    } 
    catch (const cl::Error& err) {
        std::cerr << "OpenCL Error in clgetL1Penalty: " << err.what() << " (" << err.err() << ")" << std::endl;
        throw;
    } 
    catch (const std::exception& e) {
        std::cerr << "Standard Exception in clgetL1Penalty: " << e.what() << std::endl;
        throw;
    }

    return final_result;
}

/**
 * @brief Calculate L2 penalty (sum of squared weights) using the 'l2PenaltyKernel' OpenCL kernel.
 * @param weights 3D vector of weights (flattened before processing).
 * @return L2 penalty value.
 */
float clgetL2Penalty(const std::vector<std::vector<std::vector<float>>>& weights) {
    std::vector<float> flat_weights = flattenWeights(weights); // Flatten the weights
    if (flat_weights.empty()) return 0.0f;

    size_t size = flat_weights.size();
    float final_result = 0.0f;

    try {
        // 1. Determine work sizes
        size_t local_size = 256;
        size_t num_groups = (size + local_size - 1) / local_size;
        size_t global_size = num_groups * local_size;

        // 2. Allocate device memory
        cl::Buffer d_weights(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size * sizeof(float), const_cast<float*>(flat_weights.data()));
        cl::Buffer d_partial_results(context, CL_MEM_WRITE_ONLY, num_groups * sizeof(float));

        // 3. Create kernel
        cl::Kernel kernel(program, "l2PenaltyKernel"); // Using kernel from mlp/cl/kernel.cl

        // 4. Set kernel arguments (matching l2PenaltyKernel signature)
        kernel.setArg(0, d_weights);
        kernel.setArg(1, d_partial_results);
        kernel.setArg(2, cl::Local(local_size * sizeof(float))); // Local memory
        kernel.setArg(3, static_cast<cl_int>(size));             // Total size

        // 5. Launch kernel
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(global_size), cl::NDRange(local_size));

        // 6. Read partial results back to host
        std::vector<float> h_partial_results(num_groups);
        queue.enqueueReadBuffer(d_partial_results, CL_TRUE, 0, num_groups * sizeof(float), h_partial_results.data()); // Blocking read

        // 7. Perform final reduction on host
        final_result = std::accumulate(h_partial_results.begin(), h_partial_results.end(), 0.0f);

        // queue.finish(); // Not strictly needed

    } catch (const cl::Error& err) {
        std::cerr << "OpenCL Error in clgetL2Penalty: " << err.what() << " (" << err.err() << ")" << std::endl;
        throw;
    } catch (const std::exception& e) {
        std::cerr << "Standard Exception in clgetL2Penalty: " << e.what() << std::endl;
        throw;
    }

    return final_result;
}

// --- OpenCL Loss Functions Combining Data Loss and Regularization ---

/**
 * @brief Calculate loss with L1 regularization using OpenCL kernels.
 *        Loss = sum(|outputs - targets|) + 0.5 * lambda * sum(|weights|)
 *        Uses 'absDiffKernel' for data loss and 'l1PenaltyKernel' via clgetL1Penalty.
 * @param outputs Output obtained from process.
 * @param targets Expected output from process.
 * @param network Network object to access weights.
 * @param lambda L1 regularization parameter.
 * @return Combined loss value.
 */
float clcomputeLossWithL1(const std::vector<float>& outputs, const std::vector<float>& targets, mlp& network, float lambda) {
    if (outputs.size() != targets.size()) {
        throw std::invalid_argument("Output and target vector sizes must match in clcomputeLossWithL1.");
    }
    if (outputs.empty()) {
        return 0.0f;
    }

    size_t size = outputs.size();
    float data_loss = 0.0f; // Sum of absolute differences

    try {
        // --- Calculate Data Loss Term (Sum of Absolute Differences) ---
        size_t local_size = 256;
        size_t num_groups = (size + local_size - 1) / local_size;
        size_t global_size = num_groups * local_size;

        cl::Buffer d_outputs(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size * sizeof(float), const_cast<float*>(outputs.data()));
        cl::Buffer d_targets(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size * sizeof(float), const_cast<float*>(targets.data()));
        cl::Buffer d_partial_loss(context, CL_MEM_WRITE_ONLY, num_groups * sizeof(float));

        cl::Kernel kernel(program, "absDiffKernel"); // Using kernel from mlp/cl/kernel.cl

        kernel.setArg(0, d_outputs);
        kernel.setArg(1, d_targets);
        kernel.setArg(2, d_partial_loss);
        kernel.setArg(3, cl::Local(local_size * sizeof(float)));
        kernel.setArg(4, static_cast<cl_int>(size));

        queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(global_size), cl::NDRange(local_size));

        std::vector<float> h_partial_loss(num_groups);
        queue.enqueueReadBuffer(d_partial_loss, CL_TRUE, 0, num_groups * sizeof(float), h_partial_loss.data()); // Blocking read

        data_loss = std::accumulate(h_partial_loss.begin(), h_partial_loss.end(), 0.0f);
        // --- End Data Loss Calculation ---

        // queue.finish(); // Ensure data loss kernel is done before penalty calculation (though blocking read helps)

        // --- Calculate L1 Penalty Term ---
        // Assumes network.weights is accessible and contains the current weights
        float l1_penalty = clgetL1Penalty(network.weights);

        // --- Combine Data Loss and Penalty ---
        // Formula matches comment in original code, potentially derived from a specific source/CUDA version.
        // Standard L1 loss often doesn't have the 0.5 factor on the penalty.
        return data_loss + 0.5f * lambda * l1_penalty;

    } catch (const cl::Error& err) {
        std::cerr << "OpenCL Error in clcomputeLossWithL1: " << err.what() << " (" << err.err() << ")" << std::endl;
        throw;
    } catch (const std::exception& e) {
        std::cerr << "Standard Exception in clcomputeLossWithL1: " << e.what() << std::endl;
        throw;
    }
}

/**
 * @brief Calculate loss with L2 regularization using OpenCL kernels.
 *        Loss = 0.5 * sum((outputs - targets)^2) + 0.5 * lambda * sum(weights^2)
 *        Uses 'squaredDiffKernel' for data loss and 'l2PenaltyKernel' via clgetL2Penalty.
 * @param outputs Output obtained from process.
 * @param targets Expected output from process.
 * @param network Network object to access weights.
 * @param lambda L2 regularization parameter.
 * @return Combined loss value.
 */
float clcomputeLossWithL2(const std::vector<float>& outputs, const std::vector<float>& targets, mlp& network, float lambda) {
    if (outputs.size() != targets.size()) {
        throw std::invalid_argument("Output and target vector sizes must match in clcomputeLossWithL2.");
    }
    if (outputs.empty()) {
        return 0.0f;
    }

    size_t size = outputs.size();
    float sum_sq_diff = 0.0f; // Sum of squared differences

    try {
        // --- Calculate Data Loss Term (Sum of Squared Differences) ---
        size_t local_size = 256;
        size_t num_groups = (size + local_size - 1) / local_size;
        size_t global_size = num_groups * local_size;

        cl::Buffer d_outputs(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size * sizeof(float), const_cast<float*>(outputs.data()));
        cl::Buffer d_targets(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size * sizeof(float), const_cast<float*>(targets.data()));
        cl::Buffer d_partial_loss(context, CL_MEM_WRITE_ONLY, num_groups * sizeof(float));

        cl::Kernel kernel(program, "squaredDiffKernel"); // Using kernel from mlp/cl/kernel.cl

        kernel.setArg(0, d_outputs);
        kernel.setArg(1, d_targets);
        kernel.setArg(2, d_partial_loss);
        kernel.setArg(3, cl::Local(local_size * sizeof(float)));
        kernel.setArg(4, static_cast<cl_int>(size));

        queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(global_size), cl::NDRange(local_size));

        std::vector<float> h_partial_loss(num_groups);
        queue.enqueueReadBuffer(d_partial_loss, CL_TRUE, 0, num_groups * sizeof(float), h_partial_loss.data()); // Blocking read

        sum_sq_diff = std::accumulate(h_partial_loss.begin(), h_partial_loss.end(), 0.0f);
        // --- End Data Loss Calculation ---

        // queue.finish(); // Ensure data loss kernel is done

        // --- Calculate L2 Penalty Term ---
        float l2_penalty = clgetL2Penalty(network.weights);

        // --- Combine Data Loss and Penalty ---
        // Formula matches standard L2 regularized loss (e.g., Ridge regression)
        return 0.5f * sum_sq_diff + 0.5f * lambda * l2_penalty;

    } catch (const cl::Error& err) {
        std::cerr << "OpenCL Error in clcomputeLossWithL2: " << err.what() << " (" << err.err() << ")" << std::endl;
        throw;
    } catch (const std::exception& e) {
        std::cerr << "Standard Exception in clcomputeLossWithL2: " << e.what() << std::endl;
        throw;
    }
}

/**
 * @brief Calculate loss simulating dropout generalization using OpenCL kernels.
 *        Loss = 0.5 * sum((outputs - targets)^2) + 0.5 * (1/(1-p)) * sum(weights^2)
 *        Uses 'squaredDiffKernel' for data loss and 'l2PenaltyKernel' via clgetL2Penalty.
 * @param outputs Output obtained from process.
 * @param targets Expected output from process.
 * @param network Network object to access weights.
 * @param p Dropout probability (probability of dropping a unit, 0 <= p < 1).
 * @return Combined loss value.
 */
float cldropoutGeneralisation(const std::vector<float>& outputs, const std::vector<float>& targets, mlp& network, float p) {
    if (outputs.size() != targets.size()) {
        throw std::invalid_argument("Output and target vector sizes must match in cldropoutGeneralisation.");
    }
    if (outputs.empty()) {
        return 0.0f;
    }
    if (p < 0.0f || p >= 1.0f) { // p=1 would cause division by zero
        throw std::invalid_argument("Dropout probability p must be in the range [0, 1).");
    }

    size_t size = outputs.size();
    float sum_sq_diff = 0.0f; // Sum of squared differences

    try {
        // --- Calculate Data Loss Term (Sum of Squared Differences) ---
        // This part is identical to clcomputeLossWithL2
        size_t local_size = 256;
        size_t num_groups = (size + local_size - 1) / local_size;
        size_t global_size = num_groups * local_size;

        cl::Buffer d_outputs(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size * sizeof(float), const_cast<float*>(outputs.data()));
        cl::Buffer d_targets(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size * sizeof(float), const_cast<float*>(targets.data()));
        cl::Buffer d_partial_loss(context, CL_MEM_WRITE_ONLY, num_groups * sizeof(float));

        cl::Kernel kernel(program, "squaredDiffKernel"); // Using kernel from mlp/cl/kernel.cl

        kernel.setArg(0, d_outputs);
        kernel.setArg(1, d_targets);
        kernel.setArg(2, d_partial_loss);
        kernel.setArg(3, cl::Local(local_size * sizeof(float)));
        kernel.setArg(4, static_cast<cl_int>(size));

        queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(global_size), cl::NDRange(local_size));

        std::vector<float> h_partial_loss(num_groups);
        queue.enqueueReadBuffer(d_partial_loss, CL_TRUE, 0, num_groups * sizeof(float), h_partial_loss.data()); // Blocking read

        sum_sq_diff = std::accumulate(h_partial_loss.begin(), h_partial_loss.end(), 0.0f);
        // --- End Data Loss Calculation ---

        // queue.finish(); // Ensure data loss kernel is done

        // --- Calculate L2 Penalty Term ---
        float l2_penalty = clgetL2Penalty(network.weights);

        // --- Calculate Dropout Scaling Factor ---
        // Corresponds to scaling weights by 1/sqrt(1-p) or activations by 1/(1-p) during training.
        // The loss function uses the squared weights, hence the 1/(1-p) factor.
        float dropout_factor = 1.0f / (1.0f - p);

        // --- Combine Data Loss and Scaled Penalty ---
        // Formula matches comment in original code.
        return 0.5f * sum_sq_diff + 0.5f * dropout_factor * l2_penalty;

    } catch (const cl::Error& err) {
        std::cerr << "OpenCL Error in cldropoutGeneralisation: " << err.what() << " (" << err.err() << ")" << std::endl;
        throw;
    } catch (const std::exception& e) {
        std::cerr << "Standard Exception in cldropoutGeneralisation: " << e.what() << std::endl;
        throw;
    }
}

#endif // USE_OPENCL
