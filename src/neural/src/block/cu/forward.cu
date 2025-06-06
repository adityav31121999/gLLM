
#include "include/attention.hpp"
#include "include/block.hpp"
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>
#include <string>

// Helper macro for CUDA error checking (copied from attention/cu/forward.cu for self-containment if not globally available)
#ifndef CUDA_CHECK
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error in %s at line %d (%s): %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(err)); \
        throw std::runtime_error(cudaGetErrorString(err)); \
    } \
} while (0)
#endif


// Helper function for transposing a mat object's data into a flat vector (row-major)
// Takes an R x C matrix m and produces output_flat representing a C x R matrix.
// (Copied from attention/cu/forward.cu for self-containment)
static void transposeMatToFlatVector(const mat& m, std::vector<float>& output_flat) {
    if (!m.mapped_data) {
        output_flat.clear();
        if (m.row != 0 || m.col != 0) { // Invalid state: dimensions but no data
            throw std::runtime_error("Mat has non-zero dimensions but null mapped_data in transposeMatToFlatVector.");
        }
        return; // Valid empty mat
    }
    if (m.row == 0 || m.col == 0) { // Valid empty mat
        output_flat.clear();
        return;
    }
    int R = m.row; // Original rows
    int C = m.col; // Original columns
    output_flat.resize(static_cast<size_t>(R) * C); // Will store data for a C x R matrix

    for (int j = 0; j < C; ++j) {        // Iterate original columns (these become rows in the transposed version)
        for (int i_orig = 0; i_orig < R; ++i_orig) {    // Iterate original rows (these become columns in the transposed version)
            output_flat[static_cast<size_t>(j) * R + i_orig] = m(i_orig, j); // Access m(original_row, original_col)
        }
    }
}

// Helper function to flatten a 2D vector<vector<float>> into a flat vector (row-major)
static void flatten2DVector(const std::vector<std::vector<float>>& vec2d, std::vector<float>& output_flat, size_t expected_rows, size_t expected_cols) {
    if (vec2d.empty()) {
        output_flat.clear();
        if (expected_rows != 0) { // Only throw if rows were expected but vec2d is empty
            throw std::runtime_error("Input 2D vector is empty but expected " + std::to_string(expected_rows) + " rows.");
        }
        return; // Valid empty if 0 rows expected
    }
    size_t R = vec2d.size();
    if (R != expected_rows) {
        throw std::runtime_error("Row count mismatch in flatten2DVector. Expected " + std::to_string(expected_rows) + ", got " + std::to_string(R));
    }

    size_t C = 0;
    if (R > 0) {
        C = vec2d[0].size();
        if (C != expected_cols) {
             throw std::runtime_error("Column count mismatch in flatten2DVector for row 0. Expected " + std::to_string(expected_cols) + ", got " + std::to_string(C));
        }
    } else if (expected_cols != 0) { // R is 0, but expected_cols is not.
         throw std::runtime_error("Column count mismatch in flatten2DVector: 0 rows but expected " + std::to_string(expected_cols) + " columns.");
    }

    output_flat.resize(R * C);
    for (size_t r_idx = 0; r_idx < R; ++r_idx) {
        if (vec2d[r_idx].size() != C) {
            throw std::runtime_error("Inconsistent column count in flatten2DVector at row " + std::to_string(r_idx) + ". Expected " + std::to_string(C) + ", got " + std::to_string(vec2d[r_idx].size()));
        }
        for (size_t c_idx = 0; c_idx < C; ++c_idx) {
            output_flat[r_idx * C + c_idx] = vec2d[r_idx][c_idx];
        }
    }
}


/**
 * @brief CUDA forward propagation on single ith column of the FIRST block.
 * @param in dimension of embeddings and mlp input
 * @param tokenCount number of tokens available in full context
 * @param layers number of mlp weight matrices
 */
void block::cu1parallelForprop(int& in, int& tokenCount, int i, int& layers)
{
    if (i < 0 || i >= this->y) {
        throw std::out_of_range("cu1parallelForprop (first block): column index 'i' (" + std::to_string(i) + ") is out of range [0, " + std::to_string(this->y - 1) + "].");
    }

    for (int layer_idx = 0; layer_idx < this->x; ++layer_idx) {
        attention& head = this->b[layer_idx][i];

        const int d = EMBEDDING;        // Embedding dimension
        const int h_dim = MATHEIGHTS;       // Height dimension of K/Q, projection matrices
        const int n_tokens = tokenCount;    // Number of tokens for this operation

        if (n_tokens <= 0) {
            // std::cerr << "Warning: cu1parallelForprop (first block) for head [" << layer_idx << "][" << i << "] called with tokenCount <= 0. Skipping." << std::endl;
            std::fill(head.EH.begin(), head.EH.end(), 0.0f);
            if (n_tokens == 0 && head.EV.mapped_data && head.EV.row > 0 && head.EV.col == d) {
                std::fill_n(head.EV.mapped_data, head.EV.col, 0.0f);
            }
            continue; // Skip to next head in the column
        }
        if(n_tokens > CONTEXT_WIN) {
            throw std::runtime_error("cu1parallelForprop (first block) is not for subsequent blocks, shift to next block.");
        }

        if (head.K.row != CONTEXT_WIN || head.K.col != h_dim ||
            head.Q.row != CONTEXT_WIN || head.Q.col != h_dim ||
            head.KdotQ.row != CONTEXT_WIN || head.KdotQ.col != CONTEXT_WIN ||
            head.MH.row != EMBEDDING || head.MH.col != MATHEIGHTS ||
            head.MV.row != EMBEDDING || head.MV.col != MATHEIGHTS ||
            head.EH.size() != static_cast<size_t>(d) ||
            (!head.EV.mapped_data || head.EV.row != CONTEXT_WIN || head.EV.col != d) ||
            head.hor.hlayers.empty() || head.ver.hlayers.empty() || head.hor.weights.empty() || head.ver.weights.empty())
        {
            throw std::runtime_error("Attention component dimension mismatch or uninitialized member in cu1parallelForprop (first block) for head [" +
                                     std::to_string(layer_idx) + "][" + std::to_string(i) + "]. K.row=" + std::to_string(head.K.row) + ", n_tokens=" + std::to_string(n_tokens));
        }
        if (d != in) {
            throw std::runtime_error("Embedding dimension mismatch (EMBEDDING vs in) for head [" + std::to_string(layer_idx) + "][" + std::to_string(i) + "].");
        }
        if (head.hor.hlayers[0].size() != static_cast<size_t>(d) || head.ver.hlayers[0].size() != static_cast<size_t>(d) ||
            head.hor.weights.back().row != d || head.ver.weights.back().row != d) {
            throw std::runtime_error("MLP input/output layer dimension mismatch with 'd' for head [" + std::to_string(layer_idx) + "][" + std::to_string(i) + "].");
        }

        float *d_K = nullptr, *d_Q = nullptr, *d_KdotQ = nullptr, *d_head_attention = nullptr;
        float *d_row_sums = nullptr, *d_col_sums = nullptr;
        float *d_dh_accum = nullptr, *d_dv_accum = nullptr;
        float *d_MH_hxd = nullptr, *d_MV_hxd = nullptr;
        float *d_dh = nullptr, *d_dv = nullptr;
        float *d_EH = nullptr, *d_EV_processed_data = nullptr;
        float *d_ver_accumulated_ev = nullptr;
        float *d_hor_inputs = nullptr, *d_ver_inputs = nullptr;
        float *d_hor_output = nullptr, *d_ver_output = nullptr;
        float *d_relu_hor_output = nullptr, *d_relu_ver_output = nullptr;
        float *d_mlp_bufferA_hor = nullptr, *d_mlp_bufferB_hor = nullptr;
        float *d_mlp_bufferA_ver = nullptr, *d_mlp_bufferB_ver = nullptr;
        float *d_mlp_pre_activation = nullptr;
        float *d_mlp_weights = nullptr;

        try {
            // std::cout << "Try: sizes" << std::endl;
            size_t k_bytes = static_cast<size_t>(CONTEXT_WIN) * h_dim * sizeof(float);
            size_t q_bytes = static_cast<size_t>(CONTEXT_WIN) * h_dim * sizeof(float);
            size_t kdotq_bytes = static_cast<size_t>(CONTEXT_WIN) * CONTEXT_WIN * sizeof(float);
            size_t head_attention_bytes = static_cast<size_t>(CONTEXT_WIN) * CONTEXT_WIN * sizeof(float);
            size_t sums_bytes = static_cast<size_t>(CONTEXT_WIN) * sizeof(float);
            size_t accum_bytes = static_cast<size_t>(h_dim) * sizeof(float);
            size_t proj_mat_bytes = static_cast<size_t>(h_dim) * d * sizeof(float);
            size_t ev_processed_bytes = static_cast<size_t>(CONTEXT_WIN) * d * sizeof(float);
            size_t embed_bytes = static_cast<size_t>(d) * sizeof(float);

            // std::cout << "Try: malloc" << std::endl;
            CUDA_CHECK(cudaMalloc(&d_K, k_bytes));
            CUDA_CHECK(cudaMalloc(&d_Q, q_bytes));
            CUDA_CHECK(cudaMalloc(&d_KdotQ, kdotq_bytes));
            CUDA_CHECK(cudaMalloc(&d_head_attention, head_attention_bytes));
            CUDA_CHECK(cudaMalloc(&d_row_sums, sums_bytes));
            CUDA_CHECK(cudaMalloc(&d_col_sums, sums_bytes));
            CUDA_CHECK(cudaMalloc(&d_dh_accum, accum_bytes)); CUDA_CHECK(cudaMemset(d_dh_accum, 0, accum_bytes));
            CUDA_CHECK(cudaMalloc(&d_dv_accum, accum_bytes)); CUDA_CHECK(cudaMemset(d_dv_accum, 0, accum_bytes));
            CUDA_CHECK(cudaMalloc(&d_MH_hxd, proj_mat_bytes));
            CUDA_CHECK(cudaMalloc(&d_MV_hxd, proj_mat_bytes));
            CUDA_CHECK(cudaMalloc(&d_dh, embed_bytes));
            CUDA_CHECK(cudaMalloc(&d_dv, embed_bytes));
            CUDA_CHECK(cudaMalloc(&d_EH, embed_bytes));
            CUDA_CHECK(cudaMalloc(&d_EV_processed_data, ev_processed_bytes));
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

            // std::cout << "Try: memcpy" << std::endl;
            if (!head.K.mapped_data || !head.Q.mapped_data || !head.KdotQ.mapped_data) 
                throw std::runtime_error("K, Q, or KdotQ have null mapped_data.");
            CUDA_CHECK(cudaMemcpy(d_K, head.K.mapped_data, k_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_Q, head.Q.mapped_data, q_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_KdotQ, head.KdotQ.mapped_data, kdotq_bytes, cudaMemcpyHostToDevice));
            std::vector<float> flat_MH_hxd, flat_MV_hxd;
            transposeMatToFlatVector(head.MH, flat_MH_hxd);
            transposeMatToFlatVector(head.MV, flat_MV_hxd);
            CUDA_CHECK(cudaMemcpy(d_MH_hxd, flat_MH_hxd.data(), proj_mat_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_MV_hxd, flat_MV_hxd.data(), proj_mat_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_EH, head.EH.data(), embed_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_EV_processed_data, head.EV.mapped_data, ev_processed_bytes, cudaMemcpyHostToDevice));

            // std::cout << "Try: kernels for forward propagation" << std::endl;
            const int threadsPerBlock = 256;
            cuLOTA<<<( (n_tokens * n_tokens) + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(d_KdotQ, d_head_attention, n_tokens, n_tokens); 
            CUDA_CHECK(cudaGetLastError());
            computeHeadSumsMaskedKernel<<<(n_tokens + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(d_head_attention, d_row_sums, d_col_sums, n_tokens, head.isSelfAttention); 
            CUDA_CHECK(cudaGetLastError());
            accumulateWeightedVectorsKernel<<<(h_dim + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(d_row_sums, d_col_sums, d_K, d_Q, d_dh_accum, d_dv_accum, n_tokens, h_dim); 
            CUDA_CHECK(cudaGetLastError());
            matrixMultiplyKernel<<<(d + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(d_dh_accum, d_MH_hxd, d_dh, 1, h_dim, d); 
            CUDA_CHECK(cudaGetLastError());
            matrixMultiplyKernel<<<(d + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(d_dv_accum, d_MV_hxd, d_dv, 1, h_dim, d); 
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            vectorAddKernel<<<(d + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(d_EH, d_dh, d_hor_inputs, d); CUDA_CHECK(cudaGetLastError());
            accumulateEVRowsKernel<<<(d + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(d_EV_processed_data, d_ver_accumulated_ev, n_tokens, d); CUDA_CHECK(cudaGetLastError());
            vectorAddKernel<<<(d + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(d_ver_accumulated_ev, d_dv, d_ver_inputs, d); CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            // std::cout << "Try: MLP" << std::endl;
            CUDA_CHECK(cudaMemcpy(d_mlp_bufferA_hor, d_hor_inputs, embed_bytes, cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_mlp_bufferA_ver, d_ver_inputs, embed_bytes, cudaMemcpyDeviceToDevice));
            float* d_curr_in_h = d_mlp_bufferA_hor, *d_curr_out_h = d_mlp_bufferB_hor;
            float* d_curr_in_v = d_mlp_bufferA_ver, *d_curr_out_v = d_mlp_bufferB_ver;
            size_t num_weights_mats = head.hor.weights.size();
            if (num_weights_mats != head.ver.weights.size() || num_weights_mats == 0 || num_weights_mats != head.hor.hlayers.size()) 
                throw std::runtime_error("MLP config error.");

            for (size_t l_idx = 0; l_idx < num_weights_mats; ++l_idx) {
                // std::cout << "For MLP layer: " << l_idx << std::endl;
                bool is_last = (l_idx == num_weights_mats - 1);
                int in_sz = d, out_sz = d; // Assuming all MLP layers are d x d
                // Hor MLP
                const mat& w_h = head.hor.weights[l_idx]; 
                if(!w_h.mapped_data) throw std::runtime_error("MLP hor weights null.");
                size_t w_bytes = static_cast<size_t>(w_h.row) * w_h.col * sizeof(float); CUDA_CHECK(cudaMalloc(&d_mlp_weights, w_bytes));
                CUDA_CHECK(cudaMemcpy(d_mlp_weights, w_h.mapped_data, w_bytes, cudaMemcpyHostToDevice));
                layerForwardKernel<<<(out_sz + threadsPerBlock - 1)/threadsPerBlock, threadsPerBlock>>>(d_curr_in_h, d_mlp_weights, is_last ? d_hor_output : d_mlp_pre_activation, in_sz, out_sz); CUDA_CHECK(cudaGetLastError());
                if (!is_last) {
                    cuSigmoid<<<(out_sz + threadsPerBlock -1)/threadsPerBlock, threadsPerBlock>>>(d_mlp_pre_activation, d_curr_out_h, out_sz); CUDA_CHECK(cudaGetLastError()); 
                    std::swap(d_curr_in_h, d_curr_out_h); 
                }
                CUDA_CHECK(cudaFree(d_mlp_weights)); d_mlp_weights = nullptr;
                // Ver MLP
                const mat& w_v = head.ver.weights[l_idx]; 
                if(!w_v.mapped_data) throw std::runtime_error("MLP ver weights null.");
                CUDA_CHECK(cudaMalloc(&d_mlp_weights, w_bytes)); // Assuming same size
                CUDA_CHECK(cudaMemcpy(d_mlp_weights, w_v.mapped_data, w_bytes, cudaMemcpyHostToDevice));
                layerForwardKernel<<<(out_sz + threadsPerBlock - 1)/threadsPerBlock, threadsPerBlock>>>(d_curr_in_v, d_mlp_weights, is_last ? d_ver_output : d_mlp_pre_activation, in_sz, out_sz); CUDA_CHECK(cudaGetLastError());
                if (!is_last) {
                    cuSigmoid<<<(out_sz + threadsPerBlock -1)/threadsPerBlock, threadsPerBlock>>>(d_mlp_pre_activation, d_curr_out_v, out_sz); CUDA_CHECK(cudaGetLastError()); 
                    std::swap(d_curr_in_v, d_curr_out_v); 
                }
                CUDA_CHECK(cudaFree(d_mlp_weights)); d_mlp_weights = nullptr;
            }
            CUDA_CHECK(cudaDeviceSynchronize());

            cuReLU<<<(d + threadsPerBlock - 1)/threadsPerBlock, threadsPerBlock>>>(d_hor_output, d_relu_hor_output, d); CUDA_CHECK(cudaGetLastError());
            cuReLU<<<(d + threadsPerBlock - 1)/threadsPerBlock, threadsPerBlock>>>(d_ver_output, d_relu_ver_output, d); CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            vectorAddKernel<<<(d + threadsPerBlock - 1)/threadsPerBlock, threadsPerBlock>>>(d_EH, d_relu_hor_output, d_EH, d); CUDA_CHECK(cudaGetLastError());
            updateEVRowsKernel<<<(n_tokens + threadsPerBlock - 1)/threadsPerBlock, threadsPerBlock>>>(d_EV_processed_data, d_relu_ver_output, n_tokens, d); CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            CUDA_CHECK(cudaMemcpy(head.EH.data(), d_EH, d, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(head.EV.mapped_data, d_EV_processed_data, ev_processed_bytes, cudaMemcpyDeviceToHost));
            // std::cout << "cuForward done for column " << i << std::endl;
        } 
        catch (const std::exception& e) {
            std::cerr << "CUDA Exception in cu1parallelForprop (first block) for head [" << layer_idx << "][" << i << "]: " << e.what() << std::endl;
            // Free all allocated memory
            cudaFree(d_K); cudaFree(d_Q); cudaFree(d_KdotQ); cudaFree(d_head_attention); cudaFree(d_row_sums); cudaFree(d_col_sums);
            cudaFree(d_dh_accum); cudaFree(d_dv_accum); cudaFree(d_MH_hxd); cudaFree(d_MV_hxd); cudaFree(d_dh); cudaFree(d_dv);
            cudaFree(d_EH); cudaFree(d_EV_processed_data); cudaFree(d_ver_accumulated_ev); cudaFree(d_hor_inputs); cudaFree(d_ver_inputs);
            cudaFree(d_hor_output); cudaFree(d_ver_output); cudaFree(d_relu_hor_output); cudaFree(d_relu_ver_output);
            cudaFree(d_mlp_bufferA_hor); cudaFree(d_mlp_bufferB_hor); cudaFree(d_mlp_bufferA_ver); cudaFree(d_mlp_bufferB_ver);
            cudaFree(d_mlp_pre_activation); cudaFree(d_mlp_weights);
            throw;
        }
        // std::cout << "Free all allocated memory on success" << std::endl;
        // Free all allocated memory on success
        cudaFree(d_K); cudaFree(d_Q); cudaFree(d_KdotQ); cudaFree(d_head_attention); cudaFree(d_row_sums); cudaFree(d_col_sums);
        cudaFree(d_dh_accum); cudaFree(d_dv_accum); cudaFree(d_MH_hxd); cudaFree(d_MV_hxd); cudaFree(d_dh); cudaFree(d_dv);
        cudaFree(d_EH); cudaFree(d_EV_processed_data); cudaFree(d_ver_accumulated_ev); cudaFree(d_hor_inputs); cudaFree(d_ver_inputs);
        cudaFree(d_hor_output); cudaFree(d_ver_output); cudaFree(d_relu_hor_output); cudaFree(d_relu_ver_output);
        cudaFree(d_mlp_bufferA_hor); cudaFree(d_mlp_bufferB_hor); cudaFree(d_mlp_bufferA_ver); cudaFree(d_mlp_bufferB_ver);
        cudaFree(d_mlp_pre_activation); // d_mlp_weights is already freed in loop
        // --- End of inlined attention::cuforprop ---
    }
}


/**
 * @brief CUDA forward propagation on single ith column of a SUBSEQUENT block (blockCount > 0).
 * @param EVp vector of EV
 * @param in dimension of embeddings and mlp input
 * @param tokenCount number of tokens available in full context
 * @param layers number of mlp weight matrices
 */
void block::cu1ParallelForprop(std::vector<std::vector<std::vector<float>>>& EVp, int& in, int& tokenCount, int& blockCount,
    int i, int& layers, int& n)
{
    if (i < 0 || i >= this->y) {
        throw std::out_of_range("cu1ParallelForprop (subsequent block): column index 'i' (" + std::to_string(i) + ") is out of range [0, " + std::to_string(this->y - 1) + "].");
    }
    if (EVp.size() != static_cast<size_t>(this->x)) {
        throw std::runtime_error("cu1ParallelForprop (subsequent block): EVp layer dimension mismatch for column " + std::to_string(i) +
                                 ". Expected " + std::to_string(this->x) + " layers, got " + std::to_string(EVp.size()) + ".");
    }

    for (int layer_idx = 0; layer_idx < this->x; ++layer_idx) {
        attention& head = this->b[layer_idx][i];
        const std::vector<std::vector<float>>& EVp_head = EVp[layer_idx];

        if (blockCount == 0) { // Should call the first block's logic
            int firstBlockTokenCount = std::min(tokenCount, n); // tokenCount is totalTokenCount, n is contextWindowSize
            if (blockCount <= 0) throw std::logic_error("cu1ParallelForprop (subsequent) called with blockCount <= 0");
        }

        const int d = EMBEDDING;
        const int h_dim = MATHEIGHTS;
        const int num_ev_rows_to_process = tokenCount; // This is totalTokenCount
        
        int start_idx_in_full_context = (blockCount - 1) * n; // n is contextWindowSize
        int end_idx_in_full_context = std::min(tokenCount, blockCount * n); // tokenCount is totalTokenCount
        const int count_tokens_this_block = std::max(0, end_idx_in_full_context - start_idx_in_full_context);

        if (count_tokens_this_block <= 0) {
            // std::cerr << "Warning: cu1ParallelForprop (subsequent block) for head [" << layer_idx << "][" << i << "] with calculated count <= 0. Skipping." << std::endl;
            std::fill(head.EH.begin(), head.EH.end(), 0.0f);
            // EV might not need zeroing if count is 0, depends on overall logic.
            continue; // Skip to next head
        }

        if (head.K.row != CONTEXT_WIN || head.K.col != MATHEIGHTS ||
            head.Q.row != CONTEXT_WIN || head.Q.col != MATHEIGHTS ||
            head.KdotQ.row != CONTEXT_WIN || head.KdotQ.col != CONTEXT_WIN ||
            head.MH.row != EMBEDDING || head.MH.col != MATHEIGHTS ||
            head.MV.row != EMBEDDING || head.MV.col != MATHEIGHTS ||
            head.EH.size() != static_cast<size_t>(d) ||
            (!head.EV.mapped_data || num_ev_rows_to_process > head.EV.row || head.EV.col != d) || // head.EV stores full context up to totalTokenCount
            head.hor.hlayers.empty() || head.ver.hlayers.empty() || head.hor.weights.empty() || head.ver.weights.empty())
        {
            throw std::runtime_error("Attention component dimension mismatch (subsequent block) for head [" +
                                     std::to_string(layer_idx) + "][" + std::to_string(i) + "]. K.row=" + std::to_string(head.K.row) + ", count_tokens_this_block=" + std::to_string(count_tokens_this_block));
        }
        if (d != in) {
            throw std::runtime_error("Embedding dimension mismatch (EMBEDDING vs in) for head [" + std::to_string(layer_idx) + "][" + std::to_string(i) + "].");
        }
         if (EVp_head.size() != static_cast<size_t>(num_ev_rows_to_process) || (!EVp_head.empty() && EVp_head[0].size() != static_cast<size_t>(d)) ) {
            throw std::runtime_error("EVp_head dimension mismatch for head [" + std::to_string(layer_idx) + "][" + std::to_string(i) +
                                     "]. Expected rows " + std::to_string(num_ev_rows_to_process) + " (totalTokenCount), got " + std::to_string(EVp_head.size()) +
                                     ". Expected cols " + std::to_string(d) + ", got " + (EVp_head.empty() ? "N/A" : std::to_string(EVp_head[0].size())) );
        }

        float *d_K = nullptr, *d_Q = nullptr, *d_KdotQ = nullptr, *d_head_attention = nullptr;
        float *d_row_sums = nullptr, *d_col_sums = nullptr;
        float *d_dh_accum = nullptr, *d_dv_accum = nullptr;
        float *d_MH_hxd = nullptr, *d_MV_hxd = nullptr;
        float *d_dh = nullptr, *d_dv = nullptr;
        float *d_EH = nullptr, *d_EV_processed_data_from_prev_block = nullptr; // Data from EVp_head
        float *d_EV_output_current_block = nullptr; // Corresponds to head.EV, for updating
        float *d_ver_accumulated_ev = nullptr;
        float *d_hor_inputs = nullptr, *d_ver_inputs = nullptr;
        float *d_hor_output = nullptr, *d_ver_output = nullptr;
        float *d_relu_hor_output = nullptr, *d_relu_ver_output = nullptr;
        float *d_mlp_bufferA_hor = nullptr, *d_mlp_bufferB_hor = nullptr;
        float *d_mlp_bufferA_ver = nullptr, *d_mlp_bufferB_ver = nullptr;
        float *d_mlp_pre_activation = nullptr;
        float *d_mlp_weights = nullptr;

        try {
            size_t k_bytes = static_cast<size_t>(count_tokens_this_block) * h_dim * sizeof(float);
            size_t q_bytes = static_cast<size_t>(count_tokens_this_block) * h_dim * sizeof(float);
            size_t kdotq_bytes = static_cast<size_t>(count_tokens_this_block) * count_tokens_this_block * sizeof(float);
            size_t head_attention_bytes = static_cast<size_t>(count_tokens_this_block) * count_tokens_this_block * sizeof(float);
            size_t sums_bytes = static_cast<size_t>(count_tokens_this_block) * sizeof(float);
            size_t accum_bytes = static_cast<size_t>(h_dim) * sizeof(float);
            size_t proj_mat_bytes = static_cast<size_t>(h_dim) * d * sizeof(float);
            size_t ev_prev_block_bytes = static_cast<size_t>(num_ev_rows_to_process) * d * sizeof(float); // For EVp_head
            size_t ev_current_block_output_bytes = static_cast<size_t>(head.EV.row) * d * sizeof(float); // For head.EV output
            size_t embed_bytes = static_cast<size_t>(d) * sizeof(float);

            CUDA_CHECK(cudaMalloc(&d_K, k_bytes)); // ... (rest of allocations as in 1st overload, using count_tokens_this_block or d where appropriate)
            CUDA_CHECK(cudaMalloc(&d_Q, q_bytes));
            CUDA_CHECK(cudaMalloc(&d_KdotQ, kdotq_bytes));
            CUDA_CHECK(cudaMalloc(&d_head_attention, head_attention_bytes));
            CUDA_CHECK(cudaMalloc(&d_row_sums, sums_bytes));
            CUDA_CHECK(cudaMalloc(&d_col_sums, sums_bytes));
            CUDA_CHECK(cudaMalloc(&d_dh_accum, accum_bytes)); CUDA_CHECK(cudaMemset(d_dh_accum, 0, accum_bytes));
            CUDA_CHECK(cudaMalloc(&d_dv_accum, accum_bytes)); CUDA_CHECK(cudaMemset(d_dv_accum, 0, accum_bytes));
            CUDA_CHECK(cudaMalloc(&d_MH_hxd, proj_mat_bytes));
            CUDA_CHECK(cudaMalloc(&d_MV_hxd, proj_mat_bytes));
            CUDA_CHECK(cudaMalloc(&d_dh, embed_bytes));
            CUDA_CHECK(cudaMalloc(&d_dv, embed_bytes));
            CUDA_CHECK(cudaMalloc(&d_EH, embed_bytes));
            CUDA_CHECK(cudaMalloc(&d_EV_processed_data_from_prev_block, ev_prev_block_bytes));
            CUDA_CHECK(cudaMalloc(&d_EV_output_current_block, ev_current_block_output_bytes)); // For writing to head.EV
            CUDA_CHECK(cudaMalloc(&d_ver_accumulated_ev, embed_bytes));
            // ... (MLP buffers as before)
            CUDA_CHECK(cudaMalloc(&d_hor_inputs, embed_bytes)); CUDA_CHECK(cudaMalloc(&d_ver_inputs, embed_bytes));
            CUDA_CHECK(cudaMalloc(&d_hor_output, embed_bytes)); CUDA_CHECK(cudaMalloc(&d_ver_output, embed_bytes));
            CUDA_CHECK(cudaMalloc(&d_relu_hor_output, embed_bytes)); CUDA_CHECK(cudaMalloc(&d_relu_ver_output, embed_bytes));
            CUDA_CHECK(cudaMalloc(&d_mlp_bufferA_hor, embed_bytes)); CUDA_CHECK(cudaMalloc(&d_mlp_bufferB_hor, embed_bytes));
            CUDA_CHECK(cudaMalloc(&d_mlp_bufferA_ver, embed_bytes)); CUDA_CHECK(cudaMalloc(&d_mlp_bufferB_ver, embed_bytes));
            CUDA_CHECK(cudaMalloc(&d_mlp_pre_activation, embed_bytes));

            if (!head.K.mapped_data || !head.Q.mapped_data || !head.KdotQ.mapped_data) 
                throw std::runtime_error("K, Q, or KdotQ have null mapped_data.");
            CUDA_CHECK(cudaMemcpy(d_K, head.K.mapped_data, k_bytes, cudaMemcpyHostToDevice)); // K, Q, KdotQ are for count_tokens_this_block
            CUDA_CHECK(cudaMemcpy(d_Q, head.Q.mapped_data, q_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_KdotQ, head.KdotQ.mapped_data, kdotq_bytes, cudaMemcpyHostToDevice));
            std::vector<float> flat_MH_hxd, flat_MV_hxd;
            transposeMatToFlatVector(head.MH, flat_MH_hxd); transposeMatToFlatVector(head.MV, flat_MV_hxd);
            CUDA_CHECK(cudaMemcpy(d_MH_hxd, flat_MH_hxd.data(), proj_mat_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_MV_hxd, flat_MV_hxd.data(), proj_mat_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_EH, head.EH.data(), embed_bytes, cudaMemcpyHostToDevice));
            
            std::vector<float> flat_EVp_head;
            flatten2DVector(EVp_head, flat_EVp_head, num_ev_rows_to_process, d);
            CUDA_CHECK(cudaMemcpy(d_EV_processed_data_from_prev_block, flat_EVp_head.data(), ev_prev_block_bytes, cudaMemcpyHostToDevice));

            const int threadsPerBlock = 256;
            cuLOTA<<<( (count_tokens_this_block * count_tokens_this_block) + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(d_KdotQ, d_head_attention, count_tokens_this_block, count_tokens_this_block); CUDA_CHECK(cudaGetLastError());
            computeHeadSumsMaskedKernel<<<(count_tokens_this_block + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(d_head_attention, d_row_sums, d_col_sums, count_tokens_this_block, head.isSelfAttention); CUDA_CHECK(cudaGetLastError());
            accumulateWeightedVectorsKernel<<<(h_dim + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(d_row_sums, d_col_sums, d_K, d_Q, d_dh_accum, d_dv_accum, count_tokens_this_block, h_dim); CUDA_CHECK(cudaGetLastError());
            matrixMultiplyKernel<<<(d + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(d_dh_accum, d_MH_hxd, d_dh, 1, h_dim, d); CUDA_CHECK(cudaGetLastError());
            matrixMultiplyKernel<<<(d + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(d_dv_accum, d_MV_hxd, d_dv, 1, h_dim, d); CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            vectorAddKernel<<<(d + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(d_EH, d_dh, d_hor_inputs, d); CUDA_CHECK(cudaGetLastError());
            // Accumulate EV from previous block (d_EV_processed_data_from_prev_block)
            accumulateEVRowsKernel<<<(d + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(d_EV_processed_data_from_prev_block, d_ver_accumulated_ev, num_ev_rows_to_process, d); CUDA_CHECK(cudaGetLastError());
            vectorAddKernel<<<(d + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(d_ver_accumulated_ev, d_dv, d_ver_inputs, d); CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            // --- MLP Forward (Identical to 1st overload's MLP logic) ---
            CUDA_CHECK(cudaMemcpy(d_mlp_bufferA_hor, d_hor_inputs, embed_bytes, cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_mlp_bufferA_ver, d_ver_inputs, embed_bytes, cudaMemcpyDeviceToDevice));
            float* d_curr_in_h = d_mlp_bufferA_hor, *d_curr_out_h = d_mlp_bufferB_hor;
            float* d_curr_in_v = d_mlp_bufferA_ver, *d_curr_out_v = d_mlp_bufferB_ver;
            size_t num_weights_mats = head.hor.weights.size(); // Should be layers + 1
            if (num_weights_mats != head.ver.weights.size() || num_weights_mats == 0 || num_weights_mats != head.hor.hlayers.size() + 1) throw std::runtime_error("MLP config error.");

            for (size_t l_idx = 0; l_idx < num_weights_mats; ++l_idx) {
                bool is_last = (l_idx == num_weights_mats - 1);
                int in_sz = d, out_sz = d; // Assuming all MLP layers are d x d
                const mat& w_h = head.hor.weights[l_idx]; if(!w_h.mapped_data) throw std::runtime_error("MLP hor weights null.");
                size_t w_bytes = static_cast<size_t>(w_h.row) * w_h.col * sizeof(float); CUDA_CHECK(cudaMalloc(&d_mlp_weights, w_bytes));
                CUDA_CHECK(cudaMemcpy(d_mlp_weights, w_h.mapped_data, w_bytes, cudaMemcpyHostToDevice));
                layerForwardKernel<<<(out_sz + threadsPerBlock - 1)/threadsPerBlock, threadsPerBlock>>>(d_curr_in_h, d_mlp_weights, is_last ? d_hor_output : d_mlp_pre_activation, in_sz, out_sz); CUDA_CHECK(cudaGetLastError());
                if (!is_last) { cuSigmoid<<<(out_sz + threadsPerBlock -1)/threadsPerBlock, threadsPerBlock>>>(d_mlp_pre_activation, d_curr_out_h, out_sz); CUDA_CHECK(cudaGetLastError()); std::swap(d_curr_in_h, d_curr_out_h); }
                CUDA_CHECK(cudaFree(d_mlp_weights)); d_mlp_weights = nullptr;
                
                const mat& w_v = head.ver.weights[l_idx]; if(!w_v.mapped_data) throw std::runtime_error("MLP ver weights null.");
                CUDA_CHECK(cudaMalloc(&d_mlp_weights, w_bytes));
                CUDA_CHECK(cudaMemcpy(d_mlp_weights, w_v.mapped_data, w_bytes, cudaMemcpyHostToDevice));
                layerForwardKernel<<<(out_sz + threadsPerBlock - 1)/threadsPerBlock, threadsPerBlock>>>(d_curr_in_v, d_mlp_weights, is_last ? d_ver_output : d_mlp_pre_activation, in_sz, out_sz); CUDA_CHECK(cudaGetLastError());
                if (!is_last) { cuSigmoid<<<(out_sz + threadsPerBlock -1)/threadsPerBlock, threadsPerBlock>>>(d_mlp_pre_activation, d_curr_out_v, out_sz); CUDA_CHECK(cudaGetLastError()); std::swap(d_curr_in_v, d_curr_out_v); }
                CUDA_CHECK(cudaFree(d_mlp_weights)); d_mlp_weights = nullptr;
            }
            CUDA_CHECK(cudaDeviceSynchronize());
            // --- End MLP Forward ---

            cuReLU<<<(d + threadsPerBlock - 1)/threadsPerBlock, threadsPerBlock>>>(d_hor_output, d_relu_hor_output, d); CUDA_CHECK(cudaGetLastError());
            cuReLU<<<(d + threadsPerBlock - 1)/threadsPerBlock, threadsPerBlock>>>(d_ver_output, d_relu_ver_output, d); CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            vectorAddKernel<<<(d + threadsPerBlock - 1)/threadsPerBlock, threadsPerBlock>>>(d_EH, d_relu_hor_output, d_EH, d); CUDA_CHECK(cudaGetLastError());
            // Update d_EV_processed_data_from_prev_block with ReLU(ver_output)
            // This effectively updates the EV that came from the previous block.
            updateEVRowsKernel<<<(num_ev_rows_to_process + threadsPerBlock - 1)/threadsPerBlock, threadsPerBlock>>>(d_EV_processed_data_from_prev_block, d_relu_ver_output, num_ev_rows_to_process, d); CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            CUDA_CHECK(cudaMemcpy(head.EH.data(), d_EH, embed_bytes, cudaMemcpyDeviceToHost));
            // Copy the updated EV (which originated from EVp_head) back to head.EV
            // This assumes head.EV is meant to store the output EV for this block, based on previous block's EV.
            // Ensure head.EV has enough space (num_ev_rows_to_process * d)
            if (static_cast<size_t>(head.EV.row) < static_cast<size_t>(num_ev_rows_to_process) || static_cast<size_t>(head.EV.col) != static_cast<size_t>(d)) {
                 throw std::runtime_error("head.EV dimensions are insufficient to store updated EV from previous block.");
            }
            CUDA_CHECK(cudaMemcpy(head.EV.mapped_data, d_EV_processed_data_from_prev_block, ev_prev_block_bytes, cudaMemcpyDeviceToHost));

        } 
        catch (const std::exception& e) {
            std::cerr << "CUDA Exception in cu1ParallelForprop (subsequent block) for head [" << layer_idx << "][" << i << "]: " << e.what() << std::endl;
            // Free all allocated memory
            cudaFree(d_K); cudaFree(d_Q); cudaFree(d_KdotQ); cudaFree(d_head_attention); cudaFree(d_row_sums); cudaFree(d_col_sums);
            cudaFree(d_dh_accum); cudaFree(d_dv_accum); cudaFree(d_MH_hxd); cudaFree(d_MV_hxd); cudaFree(d_dh); cudaFree(d_dv);
            cudaFree(d_EH); cudaFree(d_EV_processed_data_from_prev_block); cudaFree(d_EV_output_current_block); cudaFree(d_ver_accumulated_ev);
            cudaFree(d_hor_inputs); cudaFree(d_ver_inputs); cudaFree(d_hor_output); cudaFree(d_ver_output);
            cudaFree(d_relu_hor_output); cudaFree(d_relu_ver_output);
            cudaFree(d_mlp_bufferA_hor); cudaFree(d_mlp_bufferB_hor); cudaFree(d_mlp_bufferA_ver); cudaFree(d_mlp_bufferB_ver);
            cudaFree(d_mlp_pre_activation); cudaFree(d_mlp_weights);
            throw;
        }
        // Free all allocated memory on success
        cudaFree(d_K); cudaFree(d_Q); cudaFree(d_KdotQ); cudaFree(d_head_attention); cudaFree(d_row_sums); cudaFree(d_col_sums);
        cudaFree(d_dh_accum); cudaFree(d_dv_accum); cudaFree(d_MH_hxd); cudaFree(d_MV_hxd); cudaFree(d_dh); cudaFree(d_dv);
        cudaFree(d_EH); cudaFree(d_EV_processed_data_from_prev_block); cudaFree(d_EV_output_current_block); cudaFree(d_ver_accumulated_ev);
        cudaFree(d_hor_inputs); cudaFree(d_ver_inputs); cudaFree(d_hor_output); cudaFree(d_ver_output);
        cudaFree(d_relu_hor_output); cudaFree(d_relu_ver_output);
        cudaFree(d_mlp_bufferA_hor); cudaFree(d_mlp_bufferB_hor); cudaFree(d_mlp_bufferA_ver); cudaFree(d_mlp_bufferB_ver);
        cudaFree(d_mlp_pre_activation); // d_mlp_weights is already freed in loop
        // --- End of inlined attention::cuforprop ---
    }
}


/**
 * @brief CUDA forward propagation on the FIRST block.
 *        Iterates through all columns (parallels) and calls cu1parallelForprop for each.
 * @param in dimension size (maps to d_embedding in attention::cuforprop)
 * @param tokenCount number of tokens (maps to currentTokenCount in attention::cuforprop)
 * @param layers number of layers in mlp (maps to layers_mlp in attention::cuforprop)
 */
void block::cuForprop(int& in, int& tokenCount, int& layers)
{
    // serialise(blockFilePath);
    for (int j = 0; j < this->y; ++j) {
        try {
            this->cu1parallelForprop(in, tokenCount, j, layers);
        }
        catch (const std::exception& e) {
            throw std::runtime_error("Exception in cu1parallelForprop (first block) for column ["
                                     + std::to_string(j) + "]: " + e.what());
        }
    }
}


/**
 * @brief CUDA forward propagation of a SUBSEQUENT block (blockCount > 0).
 *        Iterates through all columns (parallels) and calls cu1ParallelForprop for each,
 *        passing the relevant EVp slice for that column.
 * @param EVp vertical retention vectors from previous blocks (shape [x][y][token][embedding]).
 * @param in dimension size (maps to d_embedding in attention::cuforprop)
 * @param tokenCount number of tokens in full context (maps to totalTokenCount in attention::cuforprop)
 * @param blockCount position of block in full context (1-based, maps to blockIdx in attention::cuforprop)
 * @param layers number of layers in mlp (maps to layers_mlp in attention::cuforprop)
 * @param n context window size (maps to contextWindowSize in attention::cuforprop)
 */
void block::cuForprop(std::vector<std::vector<std::vector<std::vector<float>>>>& EVp, int& in, int& tokenCount,
    int& blockCount, int& layers, int& n)
{
    if (EVp.size() != static_cast<size_t>(this->x)) {
        throw std::runtime_error("cuForprop (subsequent block): EVp layer dimension mismatch. Expected "
                                 + std::to_string(this->x) + " layers, got " + std::to_string(EVp.size()) + ".");
    }
    if (!EVp.empty() && EVp[0].size() != static_cast<size_t>(this->y)) {
        throw std::runtime_error("cuForprop (subsequent block): EVp column dimension mismatch. Expected "
                                 + std::to_string(this->y) + " columns, got " + std::to_string(EVp[0].size()) + ".");
    }
    // serialise(blockFilePath);
    for (int j = 0; j < this->y; ++j) {
        std::vector<std::vector<std::vector<float>>> EVp_col_j(this->x);
        for (int layer_idx = 0; layer_idx < this->x; ++layer_idx) {
            EVp_col_j[layer_idx] = EVp[layer_idx][j];
        }

        try {
            this->cu1ParallelForprop(EVp_col_j, in, tokenCount, blockCount, j, layers, n);
        }
        catch (const std::exception& e) {
            throw std::runtime_error("Exception in cu1ParallelForprop (subsequent block) for column ["
                                     + std::to_string(j) + "]: " + e.what());
        }
    }
}
