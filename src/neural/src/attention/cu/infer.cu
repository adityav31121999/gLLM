#include "include/mlp.hpp"       // For mlp class definition
#include "include/attention.hpp" // For attention class definition
#include <maths.hpp>             // Should declare cuLOTA, cuReLU, cuSigmoid from activations.cu

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <numeric>
#include <algorithm> // For std::max, std::abs, std::min
#include <cmath>     // For std::abs used in count calculation

// Helper macro for CUDA error checking
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        throw std::runtime_error(cudaGetErrorString(err)); \
    } \
} while (0)


/**
 * @brief CUDA inference for first block's attention head.
 * Mirrors: void attention::inferHead(int &in, int &layers, int &tokenCount)
 * @param d_embedding embedding dimension (in)
 * @param layers_mlp layers of hidden weights in mlp (layers)
 * @param currentTokenCount token count for this attention head (tokenCount)
 */
void attention::cuInferHead(const mat& tokens, int& d_embedding, int& layers_mlp, int& currentTokenCount)
{
    const int d = EMBEDDING;        // Embedding dimension. For inference, MATHEIGHTS (h) is assumed to be equal to d.
    const int n = currentTokenCount;// Number of tokens for K, Q, KdotQ processing
    const int context_win_size = EV.row; // Number of rows in EV, should be CONTEXT_WIN

    if (d != d_embedding) {
        throw std::runtime_error("Embedding dimension mismatch (EMBEDDING vs d_embedding).");
    }

    if (n <= 0) {
        std::cerr << "Warning: cuInferHead(..., tokenCount=" << n << ") called with tokenCount <= 0. Skipping." << std::endl;
        std::fill(EH.begin(), EH.end(), 0.0f);
        // EV is not modified if n=0, consistent with CPU infer.cpp not entering loop.
        return;
    }

    // Validation: K, Q cols must be d. khCache, qvCache must be d x d.
    // EV must have context_win_size rows and d cols.
    if (K.row != n || K.col != d ||
        Q.row != n || Q.col != d ||
        KdotQ.row != n || KdotQ.col != n ||
        khCache.row != d || khCache.col != d ||
        qvCache.row != d || qvCache.col != d ||
        EH.size() != static_cast<size_t>(d) ||
        (!EV.mapped_data || EV.row != context_win_size || EV.col != d) ||
        hor.hlayers.empty() || ver.hlayers.empty() || hor.weights.empty() || ver.weights.empty())
    {
        throw std::runtime_error("Attention component dimension mismatch or uninitialized member in cuInferHead(..., tokenCount). Ensure MATHEIGHTS=EMBEDDING for inference caches.");
    }

    float *d_tokens = nullptr, *d_KdotQ = nullptr, *d_head = nullptr;
    float *d_row_sums = nullptr, *d_col_sums = nullptr;
    float *d_dh_accum = nullptr, *d_dv_accum = nullptr; // These will be size d
    float *d_khCache = nullptr, *d_qvCache = nullptr;   // d x d
    float *d_dh = nullptr, *d_dv = nullptr;             // Projected result: 1 x d (size d)
    float *d_EH = nullptr;
    float *d_EV_buffer = nullptr;      // Buffer for all CONTEXT_WIN rows of EV
    float *d_ver_accumulated_ev = nullptr; // Buffer for the sum of EV rows (first n rows)
    float *d_hor_inputs = nullptr, *d_ver_inputs = nullptr;
    float *d_hor_output = nullptr, *d_ver_output = nullptr;
    float *d_relu_hor_output = nullptr, *d_relu_ver_output = nullptr;

    float *d_mlp_bufferA_hor = nullptr, *d_mlp_bufferB_hor = nullptr;
    float *d_mlp_bufferA_ver = nullptr, *d_mlp_bufferB_ver = nullptr;
    float *d_mlp_pre_activation = nullptr;
    float *d_mlp_weights = nullptr;

    try {
        size_t tokensBytes = static_cast<size_t>(n) * d * sizeof(float); // K, Q are n x d
        size_t kdotq_bytes = static_cast<size_t>(n) * n * sizeof(float);
        size_t head_bytes = static_cast<size_t>(n) * n * sizeof(float);
        size_t sums_bytes = static_cast<size_t>(n) * sizeof(float);
        size_t accum_bytes = static_cast<size_t>(d) * sizeof(float); // dh_accum, dv_accum are size d
        size_t cache_mat_bytes = static_cast<size_t>(d) * d * sizeof(float); // khCache, qvCache are d x d
        size_t ev_full_bytes = static_cast<size_t>(context_win_size) * d * sizeof(float); // For all EV rows
        size_t embed_bytes = static_cast<size_t>(d) * sizeof(float);

        CUDA_CHECK(cudaMalloc(&d_tokens, tokensBytes));
        CUDA_CHECK(cudaMalloc(&d_KdotQ, kdotq_bytes));
        CUDA_CHECK(cudaMalloc(&d_head, head_bytes));
        CUDA_CHECK(cudaMalloc(&d_row_sums, sums_bytes));
        CUDA_CHECK(cudaMalloc(&d_col_sums, sums_bytes));
        CUDA_CHECK(cudaMalloc(&d_dh_accum, accum_bytes));
        CUDA_CHECK(cudaMalloc(&d_dv_accum, accum_bytes));
        CUDA_CHECK(cudaMalloc(&d_khCache, cache_mat_bytes));
        CUDA_CHECK(cudaMalloc(&d_qvCache, cache_mat_bytes));
        CUDA_CHECK(cudaMalloc(&d_dh, embed_bytes));
        CUDA_CHECK(cudaMalloc(&d_dv, embed_bytes));
        CUDA_CHECK(cudaMalloc(&d_EH, embed_bytes));
        CUDA_CHECK(cudaMalloc(&d_EV_buffer, ev_full_bytes)); // Full EV buffer
        CUDA_CHECK(cudaMalloc(&d_ver_accumulated_ev, embed_bytes));
        CUDA_CHECK(cudaMalloc(&d_hor_inputs, embed_bytes));
        CUDA_CHECK(cudaMalloc(&d_ver_inputs, embed_bytes));
        CUDA_CHECK(cudaMalloc(&d_hor_output, embed_bytes));
        CUDA_CHECK(cudaMalloc(&d_ver_output, embed_bytes));
        CUDA_CHECK(cudaMalloc(&d_relu_hor_output, embed_bytes));
        CUDA_CHECK(cudaMalloc(&d_relu_ver_output, embed_bytes));
        CUDA_CHECK(cudaMalloc(&d_mlp_bufferA_hor, embed_bytes));
        CUDA_CHECK(cudaMalloc(&d_mlp_bufferB_hor, embed_bytes));
        CUDA_CHECK(cudaMalloc(&d_mlp_bufferA_ver, embed_bytes));
        CUDA_CHECK(cudaMalloc(&d_mlp_bufferB_ver, embed_bytes));
        CUDA_CHECK(cudaMalloc(&d_mlp_pre_activation, embed_bytes));
        CUDA_CHECK(cudaMemset(d_dh_accum, 0, accum_bytes));
        CUDA_CHECK(cudaMemset(d_dv_accum, 0, accum_bytes));

        CUDA_CHECK(cudaMemcpy(d_tokens, tokens.mapped_data, tokensBytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_KdotQ, KdotQ.mapped_data, kdotq_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_khCache, khCache.mapped_data, cache_mat_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_qvCache, qvCache.mapped_data, cache_mat_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_EH, EH.data(), embed_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_EV_buffer, EV.mapped_data, ev_full_bytes, cudaMemcpyHostToDevice)); // Copy all EV rows

        const int threadsPerBlock = 256;

        int totalElementsLOTA = n * n;
        int blocksLOTA = (totalElementsLOTA + threadsPerBlock - 1) / threadsPerBlock;
        cuLOTA<<<blocksLOTA, threadsPerBlock>>>(d_KdotQ, d_head, n, n);
        CUDA_CHECK(cudaGetLastError());

        int blocksSums = (n + threadsPerBlock - 1) / threadsPerBlock;
        computeHeadSumsMaskedKernel<<<blocksSums, threadsPerBlock>>>(d_head, d_row_sums, d_col_sums, n, isSelfAttention);
        CUDA_CHECK(cudaGetLastError());

        // K, Q cols are d, dh_accum/dv_accum are size d. The 'h_dim' for kernel is d.
        int blocksAccum = (d + threadsPerBlock - 1) / threadsPerBlock;
        accumulateWeightedVectorsKernel<<<blocksAccum, threadsPerBlock>>>(d_row_sums, d_col_sums, d_tokens, d_tokens, d_dh_accum, d_dv_accum, n, d);
        CUDA_CHECK(cudaGetLastError());

        // Project: dh_accum (1xd) * khCache (dxd) -> dh (1xd)
        int blocksPerGridMatMul = (d + threadsPerBlock - 1) / threadsPerBlock;
        matrixMultiplyKernel<<<blocksPerGridMatMul, threadsPerBlock>>>(d_dh_accum, d_khCache, d_dh, 1, d, d);
        CUDA_CHECK(cudaGetLastError());
        matrixMultiplyKernel<<<blocksPerGridMatMul, threadsPerBlock>>>(d_dv_accum, d_qvCache, d_dv, 1, d, d);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        int blocksAdd = (d + threadsPerBlock - 1) / threadsPerBlock;
        vectorAddKernel<<<blocksAdd, threadsPerBlock>>>(d_EH, d_dh, d_hor_inputs, d);
        CUDA_CHECK(cudaGetLastError());

        // Accumulate first 'n' (currentTokenCount) rows of EV for ver_input from d_EV_buffer
        int blocksAccumEV = (d + threadsPerBlock - 1) / threadsPerBlock;
        accumulateEVRowsKernel<<<blocksAccumEV, threadsPerBlock>>>(d_EV_buffer, d_ver_accumulated_ev, n, d);
        CUDA_CHECK(cudaGetLastError());
        vectorAddKernel<<<blocksAdd, threadsPerBlock>>>(d_ver_accumulated_ev, d_dv, d_ver_inputs, d);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // --- Run MLPs Forward ( 그대로 사용 ) ---
        CUDA_CHECK(cudaMemcpy(d_mlp_bufferA_hor, d_hor_inputs, embed_bytes, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_mlp_bufferA_ver, d_ver_inputs, embed_bytes, cudaMemcpyDeviceToDevice));
        float* d_current_in_hor = d_mlp_bufferA_hor;
        float* d_current_out_hor = d_mlp_bufferB_hor;
        float* d_current_in_ver = d_mlp_bufferA_ver;
        float* d_current_out_ver = d_mlp_bufferB_ver;
        size_t num_weight_matrices = hor.weights.size();

        for (size_t layer_idx = 0; layer_idx < num_weight_matrices; ++layer_idx) {
            bool is_last_layer = (layer_idx == num_weight_matrices - 1);
            int mlp_io_size = d; // Assuming MLP layers are d x d
            const mat& weights_hor = hor.weights[layer_idx];
            const mat& weights_ver = ver.weights[layer_idx];
            size_t mlp_weights_bytes = static_cast<size_t>(mlp_io_size) * mlp_io_size * sizeof(float);
            CUDA_CHECK(cudaMalloc(&d_mlp_weights, mlp_weights_bytes));

            // Hor MLP
            CUDA_CHECK(cudaMemcpy(d_mlp_weights, weights_hor.mapped_data, mlp_weights_bytes, cudaMemcpyHostToDevice));
            if (is_last_layer) {
                layerForwardKernel<<<blocksPerGridMatMul, threadsPerBlock>>>(d_current_in_hor, d_mlp_weights, d_hor_output, mlp_io_size, mlp_io_size);
            } else {
                layerForwardKernel<<<blocksPerGridMatMul, threadsPerBlock>>>(d_current_in_hor, d_mlp_weights, d_mlp_pre_activation, mlp_io_size, mlp_io_size);
                cuSigmoid<<<blocksPerGridMatMul, threadsPerBlock>>>(d_mlp_pre_activation, d_current_out_hor, mlp_io_size);
                std::swap(d_current_in_hor, d_current_out_hor);
            }
            CUDA_CHECK(cudaGetLastError());

            // Ver MLP
            CUDA_CHECK(cudaMemcpy(d_mlp_weights, weights_ver.mapped_data, mlp_weights_bytes, cudaMemcpyHostToDevice));
             if (is_last_layer) {
                layerForwardKernel<<<blocksPerGridMatMul, threadsPerBlock>>>(d_current_in_ver, d_mlp_weights, d_ver_output, mlp_io_size, mlp_io_size);
            } 
            else {
                layerForwardKernel<<<blocksPerGridMatMul, threadsPerBlock>>>(d_current_in_ver, d_mlp_weights, d_mlp_pre_activation, mlp_io_size, mlp_io_size);
                cuSigmoid<<<blocksPerGridMatMul, threadsPerBlock>>>(d_mlp_pre_activation, d_current_out_ver, mlp_io_size);
                std::swap(d_current_in_ver, d_current_out_ver);
            }
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaFree(d_mlp_weights));
            d_mlp_weights = nullptr;
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        // --- End MLPs ---

        cuReLU<<<blocksAdd, threadsPerBlock>>>(d_hor_output, d_relu_hor_output, d);
        CUDA_CHECK(cudaGetLastError());
        cuReLU<<<blocksAdd, threadsPerBlock>>>(d_ver_output, d_relu_ver_output, d);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        vectorAddKernel<<<blocksAdd, threadsPerBlock>>>(d_EH, d_relu_hor_output, d_EH, d);
        CUDA_CHECK(cudaGetLastError());

        // Update all context_win_size rows of EV
        int blocksUpdateEV = (context_win_size + threadsPerBlock - 1) / threadsPerBlock;
        updateEVRowsKernel<<<blocksUpdateEV, threadsPerBlock>>>(d_EV_buffer, d_relu_ver_output, context_win_size, d);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(EH.data(), d_EH, embed_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(EV.mapped_data, d_EV_buffer, ev_full_bytes, cudaMemcpyDeviceToHost)); // Copy all EV rows back

        cudaFree(d_tokens); cudaFree(d_KdotQ); cudaFree(d_head);
        cudaFree(d_row_sums); cudaFree(d_col_sums);
        cudaFree(d_dh_accum); cudaFree(d_dv_accum);
        cudaFree(d_khCache); cudaFree(d_qvCache);
        cudaFree(d_dh); cudaFree(d_dv);
        cudaFree(d_EH); cudaFree(d_EV_buffer);
        cudaFree(d_ver_accumulated_ev);
        cudaFree(d_hor_inputs); cudaFree(d_ver_inputs);
        cudaFree(d_hor_output); cudaFree(d_ver_output);
        cudaFree(d_relu_hor_output); cudaFree(d_relu_ver_output);
        cudaFree(d_mlp_bufferA_hor); cudaFree(d_mlp_bufferB_hor);
        cudaFree(d_mlp_bufferA_ver); cudaFree(d_mlp_bufferB_ver);
        cudaFree(d_mlp_pre_activation);
    }
    catch (const std::exception& e) {
        std::cerr << "CUDA Exception in attention cuInferHead(..., tokenCount): " << e.what() << std::endl;
        cudaFree(d_tokens); cudaFree(d_KdotQ); cudaFree(d_head);
        cudaFree(d_row_sums); cudaFree(d_col_sums);
        cudaFree(d_dh_accum); cudaFree(d_dv_accum);
        cudaFree(d_khCache); cudaFree(d_qvCache);
        cudaFree(d_dh); cudaFree(d_dv);
        cudaFree(d_EH); cudaFree(d_EV_buffer);
        cudaFree(d_ver_accumulated_ev);
        cudaFree(d_hor_inputs); cudaFree(d_ver_inputs);
        cudaFree(d_hor_output); cudaFree(d_ver_output);
        cudaFree(d_relu_hor_output); cudaFree(d_relu_ver_output);
        cudaFree(d_mlp_bufferA_hor); cudaFree(d_mlp_bufferB_hor);
        cudaFree(d_mlp_bufferA_ver); cudaFree(d_mlp_bufferB_ver);
        cudaFree(d_mlp_pre_activation); cudaFree(d_mlp_weights);
        throw;
    }
}


/**
 * @brief CUDA inference for a 2nd to last block's attention head.
 * Mirrors: void attention::inferHead(mat& EVp, int& in, int& layers, int& totalTokenCount, int& blockCount, int& n_win)
 * @param EVp_mat EV matrix from previous block
 * @param d_embedding input embedding dimension (in)
 * @param layers_mlp layers of MLPs (layers)
 * @param totalTokenCount total number of tokens processed so far (tokenCount in C++)
 * @param blockIdx which block is being processed (blockCount in C++)
 * @param contextWindowSize number of tokens for each attention head (n_win in C++)
 */
void attention::cuInferHead(const mat& EVp_mat, const mat& tokForBlock, int& d_embedding, int& layers_mlp, int& totalTokenCount,
            int& blockIdx, int& contextWindowSize)
{
    if (blockIdx == 0) {
        int firstBlockTokenCount = std::min(totalTokenCount, contextWindowSize);
        // Assuming K, Q, KdotQ are already set for this firstBlockTokenCount
        cuInferHead(tokForBlock, d_embedding, layers_mlp, firstBlockTokenCount);
        return;
    }

    const int d = EMBEDDING;
    const int context_win_size = EV.row; // For this->EV update

    int start_idx_in_full_context = (blockIdx - 1) * contextWindowSize;
    int end_idx_in_full_context = std::min(totalTokenCount, blockIdx * contextWindowSize);
    const int currentBlockTokenCount = std::max(0, end_idx_in_full_context - start_idx_in_full_context); // 'n' for K,Q,KdotQ

    if (d != d_embedding) {
        throw std::runtime_error("Embedding dimension mismatch (EMBEDDING vs d_embedding).");
    }

    if (currentBlockTokenCount <= 0) {
        std::cerr << "Warning: cuInferHead(..., blockIdx=" << blockIdx << ") with currentBlockTokenCount <= 0. Skipping." << std::endl;
        std::fill(EH.begin(), EH.end(), 0.0f);
        // this->EV is not modified.
        return;
    }

    // Validation: K, Q for currentBlockTokenCount, cols must be d. khCache, qvCache dxd.
    // EVp_mat must have totalTokenCount rows and d cols. this->EV for context_win_size rows.
    if (K.row != currentBlockTokenCount || K.col != d || Q.row != currentBlockTokenCount || Q.col != d ||
        KdotQ.row != currentBlockTokenCount || KdotQ.col != currentBlockTokenCount ||
        khCache.row != d || khCache.col != d || qvCache.row != d || qvCache.col != d || EH.size() != static_cast<size_t>(d) ||
        (!EVp_mat.mapped_data || EVp_mat.row != totalTokenCount || EVp_mat.col != d) ||
        (!EV.mapped_data || EV.row != context_win_size || EV.col != d) ||
        hor.hlayers.empty() || ver.hlayers.empty() || hor.weights.empty() || ver.weights.empty())
    {
        throw std::runtime_error("Attention component dimension mismatch or uninitialized member in cuInferHead(..., blockIdx).");
    }

    // Similar device pointers as the first overload
    float *d_tokens = nullptr, *d_KdotQ = nullptr, *d_head = nullptr;
    float *d_row_sums = nullptr, *d_col_sums = nullptr;
    float *d_dh_accum = nullptr, *d_dv_accum = nullptr;
    float *d_khCache = nullptr, *d_qvCache = nullptr;
    float *d_dh = nullptr, *d_dv = nullptr;
    float *d_EH = nullptr;
    float *d_EVp_data = nullptr; // For EVp from previous block
    float *d_this_EV_buffer = nullptr; // For this head's EV update
    float *d_ver_accumulated_ev = nullptr;
    float *d_hor_inputs = nullptr, *d_ver_inputs = nullptr;
    float *d_hor_output = nullptr, *d_ver_output = nullptr;
    float *d_relu_hor_output = nullptr, *d_relu_ver_output = nullptr;

    float *d_mlp_bufferA_hor = nullptr, *d_mlp_bufferB_hor = nullptr;
    float *d_mlp_bufferA_ver = nullptr, *d_mlp_bufferB_ver = nullptr;
    float *d_mlp_pre_activation = nullptr;
    float *d_mlp_weights = nullptr;

    try {
        size_t k_q_bytes = static_cast<size_t>(currentBlockTokenCount) * d * sizeof(float);
        size_t kdotq_bytes = static_cast<size_t>(currentBlockTokenCount) * currentBlockTokenCount * sizeof(float);
        size_t head_bytes = static_cast<size_t>(currentBlockTokenCount) * currentBlockTokenCount * sizeof(float);
        size_t sums_bytes = static_cast<size_t>(currentBlockTokenCount) * sizeof(float);
        size_t accum_bytes = static_cast<size_t>(d) * sizeof(float);
        size_t cache_mat_bytes = static_cast<size_t>(d) * d * sizeof(float);
        size_t evp_bytes = static_cast<size_t>(totalTokenCount) * d * sizeof(float); // EVp_mat.row is totalTokenCount
        size_t this_ev_bytes = static_cast<size_t>(context_win_size) * d * sizeof(float);
        size_t embed_bytes = static_cast<size_t>(d) * sizeof(float);

        CUDA_CHECK(cudaMalloc(&d_tokens, k_q_bytes));
        CUDA_CHECK(cudaMalloc(&d_KdotQ, kdotq_bytes));
        CUDA_CHECK(cudaMalloc(&d_head, head_bytes));
        CUDA_CHECK(cudaMalloc(&d_row_sums, sums_bytes));
        CUDA_CHECK(cudaMalloc(&d_col_sums, sums_bytes));
        CUDA_CHECK(cudaMalloc(&d_dh_accum, accum_bytes));
        CUDA_CHECK(cudaMalloc(&d_dv_accum, accum_bytes));
        CUDA_CHECK(cudaMalloc(&d_khCache, cache_mat_bytes));
        CUDA_CHECK(cudaMalloc(&d_qvCache, cache_mat_bytes));
        CUDA_CHECK(cudaMalloc(&d_dh, embed_bytes));
        CUDA_CHECK(cudaMalloc(&d_dv, embed_bytes));
        CUDA_CHECK(cudaMalloc(&d_EH, embed_bytes));
        CUDA_CHECK(cudaMalloc(&d_EVp_data, evp_bytes));
        CUDA_CHECK(cudaMalloc(&d_this_EV_buffer, this_ev_bytes));
        CUDA_CHECK(cudaMalloc(&d_ver_accumulated_ev, embed_bytes));
        CUDA_CHECK(cudaMalloc(&d_hor_inputs, embed_bytes));
        CUDA_CHECK(cudaMalloc(&d_ver_inputs, embed_bytes));
        CUDA_CHECK(cudaMalloc(&d_hor_output, embed_bytes));
        CUDA_CHECK(cudaMalloc(&d_ver_output, embed_bytes));
        CUDA_CHECK(cudaMalloc(&d_relu_hor_output, embed_bytes));
        CUDA_CHECK(cudaMalloc(&d_relu_ver_output, embed_bytes));
        CUDA_CHECK(cudaMalloc(&d_mlp_bufferA_hor, embed_bytes));
        CUDA_CHECK(cudaMalloc(&d_mlp_bufferB_hor, embed_bytes));
        CUDA_CHECK(cudaMalloc(&d_mlp_bufferA_ver, embed_bytes));
        CUDA_CHECK(cudaMalloc(&d_mlp_bufferB_ver, embed_bytes));
        CUDA_CHECK(cudaMalloc(&d_mlp_pre_activation, embed_bytes));

        CUDA_CHECK(cudaMemset(d_dh_accum, 0, accum_bytes));
        CUDA_CHECK(cudaMemset(d_dv_accum, 0, accum_bytes));

        // K, Q, KdotQ are for currentBlockTokenCount
        CUDA_CHECK(cudaMemcpy(d_tokens, tokForBlock.mapped_data, k_q_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_KdotQ, KdotQ.mapped_data, kdotq_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_khCache, khCache.mapped_data, cache_mat_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_qvCache, qvCache.mapped_data, cache_mat_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_EH, EH.data(), embed_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_EVp_data, EVp_mat.mapped_data, evp_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_this_EV_buffer, EV.mapped_data, this_ev_bytes, cudaMemcpyHostToDevice));

        const int threadsPerBlock = 256;

        int totalElementsLOTA = currentBlockTokenCount * currentBlockTokenCount;
        int blocksLOTA = (totalElementsLOTA + threadsPerBlock - 1) / threadsPerBlock;
        cuLOTA<<<blocksLOTA, threadsPerBlock>>>(d_KdotQ, d_head, currentBlockTokenCount, currentBlockTokenCount);
        CUDA_CHECK(cudaGetLastError());

        int blocksSums = (currentBlockTokenCount + threadsPerBlock - 1) / threadsPerBlock;
        computeHeadSumsMaskedKernel<<<blocksSums, threadsPerBlock>>>(d_head, d_row_sums, d_col_sums, currentBlockTokenCount, isSelfAttention);
        CUDA_CHECK(cudaGetLastError());

        int blocksAccum = (d + threadsPerBlock - 1) / threadsPerBlock; // h_dim is d
        accumulateWeightedVectorsKernel<<<blocksAccum, threadsPerBlock>>>(d_row_sums, d_col_sums, d_tokens, d_EVp_data, d_dh_accum, d_dv_accum, currentBlockTokenCount, d);
        CUDA_CHECK(cudaGetLastError());

        int blocksPerGridMatMul = (d + threadsPerBlock - 1) / threadsPerBlock;
        matrixMultiplyKernel<<<blocksPerGridMatMul, threadsPerBlock>>>(d_dh_accum, d_khCache, d_dh, 1, d, d);
        CUDA_CHECK(cudaGetLastError());
        matrixMultiplyKernel<<<blocksPerGridMatMul, threadsPerBlock>>>(d_dv_accum, d_qvCache, d_dv, 1, d, d);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        int blocksAdd = (d + threadsPerBlock - 1) / threadsPerBlock;
        vectorAddKernel<<<blocksAdd, threadsPerBlock>>>(d_EH, d_dh, d_hor_inputs, d);
        CUDA_CHECK(cudaGetLastError());

        // Accumulate EVp_mat.row (totalTokenCount) rows from d_EVp_data for ver_input
        int blocksAccumEV = (d + threadsPerBlock - 1) / threadsPerBlock;
        accumulateEVRowsKernel<<<blocksAccumEV, threadsPerBlock>>>(d_EVp_data, d_ver_accumulated_ev, totalTokenCount, d);
        CUDA_CHECK(cudaGetLastError());
        vectorAddKernel<<<blocksAdd, threadsPerBlock>>>(d_ver_accumulated_ev, d_dv, d_ver_inputs, d);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // --- Run MLPs Forward ( 그대로 사용 ) ---
        // (Same MLP logic as first overload)
        CUDA_CHECK(cudaMemcpy(d_mlp_bufferA_hor, d_hor_inputs, embed_bytes, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_mlp_bufferA_ver, d_ver_inputs, embed_bytes, cudaMemcpyDeviceToDevice));
        float* d_current_in_hor = d_mlp_bufferA_hor;
        float* d_current_out_hor = d_mlp_bufferB_hor;
        float* d_current_in_ver = d_mlp_bufferA_ver;
        float* d_current_out_ver = d_mlp_bufferB_ver;
        size_t num_weight_matrices = hor.weights.size();

        for (size_t layer_idx = 0; layer_idx < num_weight_matrices; ++layer_idx) {
            bool is_last_layer = (layer_idx == num_weight_matrices - 1);
            int mlp_io_size = d;
            const mat& weights_hor = hor.weights[layer_idx];
            const mat& weights_ver = ver.weights[layer_idx];
            size_t mlp_weights_bytes = static_cast<size_t>(mlp_io_size) * mlp_io_size * sizeof(float);
            CUDA_CHECK(cudaMalloc(&d_mlp_weights, mlp_weights_bytes));
            // Hor MLP
            CUDA_CHECK(cudaMemcpy(d_mlp_weights, weights_hor.mapped_data, mlp_weights_bytes, cudaMemcpyHostToDevice));
            if (is_last_layer) { layerForwardKernel<<<blocksPerGridMatMul, threadsPerBlock>>>(d_current_in_hor, d_mlp_weights, d_hor_output, mlp_io_size, mlp_io_size); }
            else { layerForwardKernel<<<blocksPerGridMatMul, threadsPerBlock>>>(d_current_in_hor, d_mlp_weights, d_mlp_pre_activation, mlp_io_size, mlp_io_size);
                   cuSigmoid<<<blocksPerGridMatMul, threadsPerBlock>>>(d_mlp_pre_activation, d_current_out_hor, mlp_io_size); std::swap(d_current_in_hor, d_current_out_hor); }
            CUDA_CHECK(cudaGetLastError());
            // Ver MLP
            CUDA_CHECK(cudaMemcpy(d_mlp_weights, weights_ver.mapped_data, mlp_weights_bytes, cudaMemcpyHostToDevice));
            if (is_last_layer) { layerForwardKernel<<<blocksPerGridMatMul, threadsPerBlock>>>(d_current_in_ver, d_mlp_weights, d_ver_output, mlp_io_size, mlp_io_size); }
            else { layerForwardKernel<<<blocksPerGridMatMul, threadsPerBlock>>>(d_current_in_ver, d_mlp_weights, d_mlp_pre_activation, mlp_io_size, mlp_io_size);
                   cuSigmoid<<<blocksPerGridMatMul, threadsPerBlock>>>(d_mlp_pre_activation, d_current_out_ver, mlp_io_size); std::swap(d_current_in_ver, d_current_out_ver); }
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaFree(d_mlp_weights)); d_mlp_weights = nullptr;
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        // --- End MLPs ---

        cuReLU<<<blocksAdd, threadsPerBlock>>>(d_hor_output, d_relu_hor_output, d);
        CUDA_CHECK(cudaGetLastError());
        cuReLU<<<blocksAdd, threadsPerBlock>>>(d_ver_output, d_relu_ver_output, d);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        vectorAddKernel<<<blocksAdd, threadsPerBlock>>>(d_EH, d_relu_hor_output, d_EH, d);
        CUDA_CHECK(cudaGetLastError());

        // Update all context_win_size rows of this->EV
        int blocksUpdateEV_this = (context_win_size + threadsPerBlock - 1) / threadsPerBlock;
        updateEVRowsKernel<<<blocksUpdateEV_this, threadsPerBlock>>>(d_this_EV_buffer, d_relu_ver_output, context_win_size, d);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(EH.data(), d_EH, embed_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(EV.mapped_data, d_this_EV_buffer, this_ev_bytes, cudaMemcpyDeviceToHost));

        cudaFree(d_tokens); cudaFree(d_KdotQ); cudaFree(d_head);
        cudaFree(d_row_sums); cudaFree(d_col_sums);
        cudaFree(d_dh_accum); cudaFree(d_dv_accum);
        cudaFree(d_khCache); cudaFree(d_qvCache);
        cudaFree(d_dh); cudaFree(d_dv);
        cudaFree(d_EH); cudaFree(d_EVp_data); cudaFree(d_this_EV_buffer);
        cudaFree(d_ver_accumulated_ev);
        cudaFree(d_hor_inputs); cudaFree(d_ver_inputs);
        cudaFree(d_hor_output); cudaFree(d_ver_output);
        cudaFree(d_relu_hor_output); cudaFree(d_relu_ver_output);
        cudaFree(d_mlp_bufferA_hor); cudaFree(d_mlp_bufferB_hor);
        cudaFree(d_mlp_bufferA_ver); cudaFree(d_mlp_bufferB_ver);
        cudaFree(d_mlp_pre_activation);
    }
    catch (const std::exception& e) {
        std::cerr << "CUDA Exception in attention cuInferHead(..., blockIdx): " << e.what() << std::endl;
        cudaFree(d_tokens); cudaFree(d_KdotQ); cudaFree(d_head);
        cudaFree(d_row_sums); cudaFree(d_col_sums);
        cudaFree(d_dh_accum); cudaFree(d_dv_accum);
        cudaFree(d_khCache); cudaFree(d_qvCache);
        cudaFree(d_dh); cudaFree(d_dv);
        cudaFree(d_EH); cudaFree(d_EVp_data); cudaFree(d_this_EV_buffer);
        cudaFree(d_ver_accumulated_ev);
        cudaFree(d_hor_inputs); cudaFree(d_ver_inputs);
        cudaFree(d_hor_output); cudaFree(d_ver_output);
        cudaFree(d_relu_hor_output); cudaFree(d_relu_ver_output);
        cudaFree(d_mlp_bufferA_hor); cudaFree(d_mlp_bufferB_hor);
        cudaFree(d_mlp_bufferA_ver); cudaFree(d_mlp_bufferB_ver);
        cudaFree(d_mlp_pre_activation); cudaFree(d_mlp_weights);
        throw;
    }
}