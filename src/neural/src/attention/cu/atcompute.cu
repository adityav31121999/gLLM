#ifdef USE_CUDA
#include "include/attention.hpp"
#include <maths.hpp>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <iostream>

void attention::cuAdamUpdate(unsigned long long t_adam, float beta1, float beta2, float epsilon, float learning_rate) {
    cudaError_t cuda_err = cudaSuccess; // Declare here to control its scope

    // Helper lambda to apply Adam to a single matrix pair on CUDA
    auto apply_adam_to_matrix_pair_cuda = [&](mat& weight_mat, mat& grad_mat, mat& m_mat, mat& v_mat) {
        // Device pointers for the current matrix pair
        float *d_weights = nullptr, *d_gradients = nullptr, *d_m = nullptr, *d_v = nullptr;
        int blockSize = 0; // Initialize to 0
        int numBlocks = 0; // Initialize to 0

        // --- VERY DETAILED DIAGNOSTIC LOGGING (OPTIONAL) ---
        // You can uncomment this section if you need to debug unmapped data issues again.
        /*
        std::cerr << "DEBUG (cuAdamUpdate): Checking matrix pair:" << std::endl;
        std::cerr << "  weight_mat: row=" << weight_mat.row << ", col=" << weight_mat.col
                  << ", mapped_data=" << static_cast<void*>(weight_mat.mapped_data)
                  << ", mapped_size=" << weight_mat.mapped_size
                  << ", is_shared=" << weight_mat.is_shared_segment << std::endl;

        std::cerr << "  grad_mat:   row=" << grad_mat.row << ", col=" << grad_mat.col
                  << ", mapped_data=" << static_cast<void*>(grad_mat.mapped_data)
                  << ", mapped_size=" << grad_mat.mapped_size
                  << ", is_shared=" << grad_mat.is_shared_segment << std::endl;

        std::cerr << "  m_mat:      row=" << m_mat.row << ", col=" << m_mat.col
                  << ", mapped_data=" << static_cast<void*>(m_mat.mapped_data)
                  << ", mapped_size=" << m_mat.mapped_size
                  << ", is_shared=" << m_mat.is_shared_segment << std::endl;

        std::cerr << "  v_mat:      row=" << v_mat.row << ", col=" << v_mat.col
                  << ", mapped_data=" << static_cast<void*>(v_mat.mapped_data)
                  << ", mapped_size=" << v_mat.mapped_size
                  << ", is_shared=" << v_mat.is_shared_segment << std::endl;
        */
        // --- END DIAGNOSTIC LOGGING ---

        // Check for unmapped host data BEFORE attempting CUDA operations
        if (weight_mat.mapped_data == nullptr || grad_mat.mapped_data == nullptr ||
            m_mat.mapped_data == nullptr || v_mat.mapped_data == nullptr) {
            std::cerr << "Warning: Attention Adam update detected unmapped data for a matrix. Skipping this pair." << std::endl;
            // Set error status to indicate a skip, handled by the 'cleanup' logic
            cuda_err = cudaErrorInvalidValue; // A non-success code to trigger cleanup
            return; // Exit lambda for this pair
        }


        int total_elements = weight_mat.row * weight_mat.col;
        if (total_elements == 0) { // Handle empty matrices gracefully
            std::cerr << "Info: Skipping Adam update for a 0-element matrix pair." << std::endl;
            return; // Exit lambda for this pair
        }
        size_t matrix_byte_size = total_elements * sizeof(float);

        // --- Allocation and HtoD Transfer ---
        cuda_err = cudaMalloc(&d_weights, matrix_byte_size);
        if (cuda_err != cudaSuccess) { fprintf(stderr, "CUDA malloc for d_weights failed: %s\n", cudaGetErrorString(cuda_err)); goto cleanup_lambda; }
        cuda_err = cudaMemcpy(d_weights, weight_mat.mapped_data, matrix_byte_size, cudaMemcpyHostToDevice);
        if (cuda_err != cudaSuccess) { fprintf(stderr, "CUDA memcpyH2D for d_weights failed: %s\n", cudaGetErrorString(cuda_err)); goto cleanup_lambda; }

        cuda_err = cudaMalloc(&d_gradients, matrix_byte_size);
        if (cuda_err != cudaSuccess) { fprintf(stderr, "CUDA malloc for d_gradients failed: %s\n", cudaGetErrorString(cuda_err)); goto cleanup_lambda; }
        cuda_err = cudaMemcpy(d_gradients, grad_mat.mapped_data, matrix_byte_size, cudaMemcpyHostToDevice);
        if (cuda_err != cudaSuccess) { fprintf(stderr, "CUDA memcpyH2D for d_gradients failed: %s\n", cudaGetErrorString(cuda_err)); goto cleanup_lambda; }

        cuda_err = cudaMalloc(&d_m, matrix_byte_size);
        if (cuda_err != cudaSuccess) { fprintf(stderr, "CUDA malloc for d_m failed: %s\n", cudaGetErrorString(cuda_err)); goto cleanup_lambda; }
        cuda_err = cudaMemcpy(d_m, m_mat.mapped_data, matrix_byte_size, cudaMemcpyHostToDevice);
        if (cuda_err != cudaSuccess) { fprintf(stderr, "CUDA memcpyH2D for d_m failed: %s\n", cudaGetErrorString(cuda_err)); goto cleanup_lambda; }

        cuda_err = cudaMalloc(&d_v, matrix_byte_size);
        if (cuda_err != cudaSuccess) { fprintf(stderr, "CUDA malloc for d_v failed: %s\n", cudaGetErrorString(cuda_err)); goto cleanup_lambda; }
        cuda_err = cudaMemcpy(d_v, v_mat.mapped_data, matrix_byte_size, cudaMemcpyHostToDevice);
        if (cuda_err != cudaSuccess) { fprintf(stderr, "CUDA memcpyH2D for d_v failed: %s\n", cudaGetErrorString(cuda_err)); goto cleanup_lambda; }

        // Determine grid and block dimensions (now declared at the top of the lambda's scope)
        blockSize = 256; // Standard block size
        numBlocks = (total_elements + blockSize - 1) / blockSize;

        // --- Kernel Launch ---
        adam_optimizer_kernel_cuda<<<numBlocks, blockSize>>>(d_weights, d_gradients, d_m, d_v,
                                                             learning_rate, beta1, beta2, epsilon,
                                                             t_adam, // Pass the global time step
                                                             total_elements);

        // --- Error Checking and Synchronization ---
        cuda_err = cudaGetLastError(); // Check for errors from the kernel launch
        if (cuda_err != cudaSuccess) { fprintf(stderr, "CUDA kernel launch failed: %s\n", cudaGetErrorString(cuda_err)); goto cleanup_lambda; }

        // Synchronize for this pair before copying results back
        cuda_err = cudaDeviceSynchronize(); // Or use cudaStreamSynchronize if you're using explicit streams
        if (cuda_err != cudaSuccess) { fprintf(stderr, "CUDA device synchronize failed: %s\n", cudaGetErrorString(cuda_err)); goto cleanup_lambda; }

        // --- DtoH Transfer ---
        cuda_err = cudaMemcpy(weight_mat.mapped_data, d_weights, matrix_byte_size, cudaMemcpyDeviceToHost);
        if (cuda_err != cudaSuccess) { fprintf(stderr, "CUDA memcpyD2H for weights failed: %s\n", cudaGetErrorString(cuda_err)); goto cleanup_lambda; }
        cuda_err = cudaMemcpy(m_mat.mapped_data, d_m, matrix_byte_size, cudaMemcpyDeviceToHost);
        if (cuda_err != cudaSuccess) { fprintf(stderr, "CUDA memcpyD2H for moments failed: %s\n", cudaGetErrorString(cuda_err)); goto cleanup_lambda; }
        cuda_err = cudaMemcpy(v_mat.mapped_data, d_v, matrix_byte_size, cudaMemcpyDeviceToHost);
        if (cuda_err != cudaSuccess) { fprintf(stderr, "CUDA memcpyD2H for velocity failed: %s\n", cudaGetErrorString(cuda_err)); goto cleanup_lambda; }

    cleanup_lambda: // Label for cleaning up resources for the current matrix pair
        if (d_weights) cudaFree(d_weights);
        if (d_gradients) cudaFree(d_gradients);
        if (d_m) cudaFree(d_m);
        if (d_v) cudaFree(d_v);

        // If an error occurred (cuda_err is not cudaSuccess), propagate or handle it.
        // For lambda, just setting cuda_err and returning is fine, the caller will check it.
    };

/*
    // Apply Adam to attention head's core matrices
    // Check cuda_err after each call to propagate error early if needed
    apply_adam_to_matrix_pair_cuda(MQ, gMQ, m_MQ, v_MQ);
    if (cuda_err != cudaSuccess) return; // Propagate error
    apply_adam_to_matrix_pair_cuda(MK, gMK, m_MK, v_MK);
    if (cuda_err != cudaSuccess) return;
    apply_adam_to_matrix_pair_cuda(MV, gMV, m_MV, v_MV);
    if (cuda_err != cudaSuccess) return;
    apply_adam_to_matrix_pair_cuda(MH, gMH, m_MH, v_MH);
    if (cuda_err != cudaSuccess) return;
*/
    // Apply Adam to internal MLPs (recursive call)
    hor.cuAdamUpdate(t_adam, beta1, beta2, epsilon, learning_rate);
    ver.cuAdamUpdate(t_adam, beta1, beta2, epsilon, learning_rate);
}

#endif // USE_CUDA