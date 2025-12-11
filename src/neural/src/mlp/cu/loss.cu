#ifdef USE_CU
// loss.cu: CUDA implementations for calculating losses and penalties for MLP
#include "include/mlp.hpp"
#include <maths.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <numeric>
#include <cmath>
#include <iostream>
#include <stdexcept>

// Helper macro for CUDA error checking
#define CUDA_CHECK(call) do {           \
    cudaError_t err = call;             \
    if (err != cudaSuccess) {           \
        fprintf(stderr, "CUDA Error in %s at line %d: %s\n",    \
             __FILE__, __LINE__, cudaGetErrorString(err));      \
        throw std::runtime_error(cudaGetErrorString(err));      \
    }                                   \
} while (0)


// --- CUDA Penalty Functions ---

/**
 * @brief calculate L1 penalty with cuda kernel
 * @param weights 3D vector of whose penalty is to be calculated
 * @return L1 penalty
 */
float cugetL1Penalty(const std::vector<mat>& weights) {
    std::vector<float> flat_weights = flattenWeights(weights);
    if (flat_weights.empty()) return 0.0f;
    
    int size = flat_weights.size();
    float *d_weights, *d_result;
    float result = 0.0f;
    
    try {
        // Allocate device memory
        CUDA_CHECK(cudaMalloc(&d_weights, size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_result, sizeof(float)));
        
        // Initialize result to 0
        CUDA_CHECK(cudaMemset(d_result, 0, sizeof(float)));
        
        // Copy data to device
        CUDA_CHECK(cudaMemcpy(d_weights, flat_weights.data(), size * sizeof(float), cudaMemcpyHostToDevice));
        
        // Launch kernel
        int threadsPerBlock = 256;
        int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
        l1PenaltyKernel<<<blocksPerGrid, threadsPerBlock>>>(d_weights, d_result, size);
        
        // Check for kernel launch errors
        CUDA_CHECK(cudaGetLastError());
        
        // Copy result back to host
        CUDA_CHECK(cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost));
        
        // Free device memory
        CUDA_CHECK(cudaFree(d_weights));
        CUDA_CHECK(cudaFree(d_result));
        
        return result;
    } 
    catch (const std::exception& e) {
        std::cerr << "CUDA Exception in getL1Penalty: " << e.what() << std::endl;
        // Cleanup on error
        cudaFree(d_weights);
        cudaFree(d_result);
        throw;
    }
}

/**
 * @brief calculate L2 penalty with cuda kernel
 * @param weights 3D vector of whose penalty is to be calculated
 * @return L2 penalty
 */
float cugetL2Penalty(const std::vector<mat>& weights) {
    std::vector<float> flat_weights = flattenWeights(weights);
    if (flat_weights.empty()) return 0.0f;
    
    int size = flat_weights.size();
    float *d_weights, *d_result;
    float result = 0.0f;
    
    try {
        // Allocate device memory
        CUDA_CHECK(cudaMalloc(&d_weights, size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_result, sizeof(float)));
        
        // Initialize result to 0
        CUDA_CHECK(cudaMemset(d_result, 0, sizeof(float)));
        
        // Copy data to device
        CUDA_CHECK(cudaMemcpy(d_weights, flat_weights.data(), size * sizeof(float), cudaMemcpyHostToDevice));
        
        // Launch kernel
        int threadsPerBlock = 256;
        int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
        l2PenaltyKernel<<<blocksPerGrid, threadsPerBlock>>>(d_weights, d_result, size);
        
        // Check for kernel launch errors
        CUDA_CHECK(cudaGetLastError());
        
        // Copy result back to host
        CUDA_CHECK(cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost));
        
        // Free device memory
        CUDA_CHECK(cudaFree(d_weights));
        CUDA_CHECK(cudaFree(d_result));
        
        return result;
    } 
    catch (const std::exception& e) {
        std::cerr << "CUDA Exception in getL2Penalty: " << e.what() << std::endl;
        // Cleanup on error
        cudaFree(d_weights);
        cudaFree(d_result);
        throw;
    }
}

// --- CUDA Loss Functions ---

/**
 * @brief calculate loss via L1 penalty with cuda kernel
 * @param outputs output obtained from process
 * @param expected expected output from process
 * @param mlp network which performs process
 * @param lambda L1 regularization parameter
 * @return loss
 */
float cucomputeLossWithL1(const std::vector<float>& outputs, const std::vector<float>& targets, mlp& network, float lambda) {
    if (outputs.size() != targets.size()) {
        throw std::invalid_argument("Output and target vector sizes must match in computeLossWithL1.");
    }
    if (outputs.empty()) {
        return 0.0f;
    }
    
    int size = outputs.size();
    float *d_outputs, *d_targets, *d_result;
    float data_loss = 0.0f;
    
    try {
        // Allocate device memory
        CUDA_CHECK(cudaMalloc(&d_outputs, size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_targets, size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_result, sizeof(float)));
        
        // Initialize result to 0
        CUDA_CHECK(cudaMemset(d_result, 0, sizeof(float)));
        
        // Copy data to device
        CUDA_CHECK(cudaMemcpy(d_outputs, outputs.data(), size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_targets, targets.data(), size * sizeof(float), cudaMemcpyHostToDevice));
        
        // Launch kernel
        int threadsPerBlock = 256;
        int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
        absDiffKernel<<<blocksPerGrid, threadsPerBlock>>>(d_outputs, d_targets, d_result, size);
        
        // Check for kernel launch errors
        CUDA_CHECK(cudaGetLastError());
        
        // Copy result back to host
        CUDA_CHECK(cudaMemcpy(&data_loss, d_result, sizeof(float), cudaMemcpyDeviceToHost));
        
        // Free device memory
        CUDA_CHECK(cudaFree(d_outputs));
        CUDA_CHECK(cudaFree(d_targets));
        CUDA_CHECK(cudaFree(d_result));
        
        float l1_penalty = cugetL1Penalty(network.weights);
        
        return data_loss + 0.5f * lambda * l1_penalty;
    } 
    catch (const std::exception& e) {
        std::cerr << "CUDA Exception in computeLossWithL1: " << e.what() << std::endl;
        // Cleanup on error
        cudaFree(d_outputs);
        cudaFree(d_targets);
        cudaFree(d_result);
        throw;
    }
}

/**
 * @brief calculate loss via L2 penalty with cuda kernel
 * @param outputs output obtained from process
 * @param expected expected output from process
 * @param mlp network which performs process
 * @param lambda L2 regularization parameter
 * @return loss
 */
float cucomputeLossWithL2(const std::vector<float>& outputs, const std::vector<float>& targets, mlp& network, float lambda) {
    if (outputs.size() != targets.size()) {
        throw std::invalid_argument("Output and target vector sizes must match in computeLossWithL2.");
    }
    if (outputs.empty()) {
        return 0.0f;
    }
    
    int size = outputs.size();
    float *d_outputs, *d_targets, *d_result;
    float sum_sq_diff = 0.0f;
    
    try {
        // Allocate device memory
        CUDA_CHECK(cudaMalloc(&d_outputs, size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_targets, size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_result, sizeof(float)));
        
        // Initialize result to 0
        CUDA_CHECK(cudaMemset(d_result, 0, sizeof(float)));
        
        // Copy data to device
        CUDA_CHECK(cudaMemcpy(d_outputs, outputs.data(), size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_targets, targets.data(), size * sizeof(float), cudaMemcpyHostToDevice));
        
        // Launch kernel
        int threadsPerBlock = 256;
        int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
        squaredDiffKernel<<<blocksPerGrid, threadsPerBlock>>>(d_outputs, d_targets, d_result, size);
        
        // Check for kernel launch errors
        CUDA_CHECK(cudaGetLastError());
        
        // Copy result back to host
        CUDA_CHECK(cudaMemcpy(&sum_sq_diff, d_result, sizeof(float), cudaMemcpyDeviceToHost));
        
        // Free device memory
        CUDA_CHECK(cudaFree(d_outputs));
        CUDA_CHECK(cudaFree(d_targets));
        CUDA_CHECK(cudaFree(d_result));
        
        float l2_penalty = cugetL2Penalty(network.weights);
        
        return 0.5f * sum_sq_diff + 0.5f * lambda * l2_penalty;
    } 
    catch (const std::exception& e) {
        std::cerr << "CUDA Exception in computeLossWithL2: " << e.what() << std::endl;
        // Cleanup on error
        cudaFree(d_outputs);
        cudaFree(d_targets);
        cudaFree(d_result);
        throw;
    }
}

/**
 * @brief calculate loss via dropout generalisation
 * @param outputs output obtained from process
 * @param expected expected output from process
 * @param mlp network which performs process
 * @param p dropout probability
 * @return loss
 */
float cudropoutGeneralisation(const std::vector<float>& outputs, const std::vector<float>& targets, mlp& network, float p) {
    if (outputs.size() != targets.size()) {
        throw std::invalid_argument("Output and target vector sizes must match in dropoutGeneralisation.");
    }
    if (outputs.empty()) {
        return 0.0f;
    }
    if (p >= 1.0f || p < 0.0f) {
        throw std::invalid_argument("Dropout probability p must be in [0, 1).");
    }
    
    int size = outputs.size();
    float *d_outputs, *d_targets, *d_result;
    float sum_sq_diff = 0.0f;
    
    try {
        // Allocate device memory
        CUDA_CHECK(cudaMalloc(&d_outputs, size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_targets, size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_result, sizeof(float)));
        
        // Initialize result to 0
        CUDA_CHECK(cudaMemset(d_result, 0, sizeof(float)));
        
        // Copy data to device
        CUDA_CHECK(cudaMemcpy(d_outputs, outputs.data(), size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_targets, targets.data(), size * sizeof(float), cudaMemcpyHostToDevice));
        
        // Launch kernel
        int threadsPerBlock = 256;
        int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
        squaredDiffKernel<<<blocksPerGrid, threadsPerBlock>>>(d_outputs, d_targets, d_result, size);
        
        // Check for kernel launch errors
        CUDA_CHECK(cudaGetLastError());
        
        // Copy result back to host
        CUDA_CHECK(cudaMemcpy(&sum_sq_diff, d_result, sizeof(float), cudaMemcpyDeviceToHost));
        
        // Free device memory
        CUDA_CHECK(cudaFree(d_outputs));
        CUDA_CHECK(cudaFree(d_targets));
        CUDA_CHECK(cudaFree(d_result));
        
        // Apply dropout regularization - scale by (1-p) for each weight
        float dropout_factor = 1.0f / (1.0f - p);
        float dropout_penalty = dropout_factor * cugetL2Penalty(network.weights);
        
        return 0.5f * sum_sq_diff + 0.5f * dropout_penalty;
    } 
    catch (const std::exception& e) {
        std::cerr << "CUDA Exception in dropoutGeneralisation: " << e.what() << std::endl;
        // Cleanup on error
        cudaFree(d_outputs);
        cudaFree(d_targets);
        cudaFree(d_result);
        throw;
    }
}
#endif
