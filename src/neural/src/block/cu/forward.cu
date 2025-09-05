
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

    const int num_heads_in_col = this->x;
    const int d_embedding = EMBEDDING;
    const int h_attention = MATHEIGHTS;
    const int n_tokens = tokenCount; // Number of tokens for current processing

    // Initial validation (can be done once before the loop if parameters are consistent for all heads)
    // For now, keeping some per-head validation inside the loop for robustness.
    if (d_embedding != in) {
        throw std::runtime_error("Embedding dimension mismatch (EMBEDDING vs in) for column " + std::to_string(i) + ".");
    }

    for(int layer_idx = 0; layer_idx < this->x; layer_idx++)
    {
        if (n_tokens <= 0) {
            // std::cerr << "Warning: cu1parallelForprop (first block) for head [" << layer_idx << "][" << i << "] called with tokenCount <= 0. Skipping." << std::endl;
            std::fill(b[layer_idx][i].EH.begin(), b[layer_idx][i].EH.end(), 0.0f);
            if (n_tokens == 0 && b[layer_idx][i].EV.mapped_data && b[layer_idx][i].EV.row > 0 && b[layer_idx][i].EV.col == in) {
                std::fill_n(b[layer_idx][i].EV.mapped_data, b[layer_idx][i].EV.col, 0.0f);
            } // This loop structure will change, so this continue will be handled differently
            // For now, if n_tokens <=0, the main loop over heads won't do much for those heads.
        }
        if (n_tokens > CONTEXT_WIN) {
            throw std::runtime_error("cu1parallelForprop (first block) is not for subsequent blocks, shift to next block.");
        }

        if (b[layer_idx][i].K.row != CONTEXT_WIN || b[layer_idx][i].K.col != MATHEIGHTS ||
            b[layer_idx][i].Q.row != CONTEXT_WIN || b[layer_idx][i].Q.col != MATHEIGHTS ||
            b[layer_idx][i].KdotQ.row != CONTEXT_WIN || b[layer_idx][i].KdotQ.col != CONTEXT_WIN ||
            b[layer_idx][i].MH.row != EMBEDDING || b[layer_idx][i].MH.col != MATHEIGHTS ||
            b[layer_idx][i].MV.row != EMBEDDING || b[layer_idx][i].MV.col != MATHEIGHTS)
        {
            throw std::runtime_error("Attention component dimension mismatch or uninitialized member in cu1parallelForprop (first block) for head [" +
                                        std::to_string(layer_idx) + "][" + std::to_string(i) + "]. K.row=" + std::to_string(b[layer_idx][i].K.row) + ", n_tokens=" + std::to_string(n_tokens));
        }
        if (b[layer_idx][i].hor.hlayers[0].size() != static_cast<size_t>(in) || b[layer_idx][i].ver.hlayers[0].size() != static_cast<size_t>(in) ||
            b[layer_idx][i].hor.weights.back().row != in || b[layer_idx][i].ver.weights.back().row != in) {
            throw std::runtime_error("MLP input/output layer dimension mismatch with 'd' for head [" + std::to_string(layer_idx) + "][" + std::to_string(i) + "].");
        }
    }

    // Per-head byte sizes
    // K, Q are CONTEXT_WIN x h_attention, but kernels use n_tokens rows. Data transfer is for CONTEXT_WIN rows.
    size_t k_bytes_ph = static_cast<size_t>(CONTEXT_WIN) * h_attention * sizeof(float);
    size_t q_bytes_ph = static_cast<size_t>(CONTEXT_WIN) * h_attention * sizeof(float);
    // KdotQ is CONTEXT_WIN x CONTEXT_WIN, kernels use n_tokens x n_tokens.
    size_t kdotq_bytes_ph = static_cast<size_t>(CONTEXT_WIN) * CONTEXT_WIN * sizeof(float);
    // head_attention is n_tokens x n_tokens, but allocated for max (CONTEXT_WIN x CONTEXT_WIN).
    size_t head_attention_bytes_ph = static_cast<size_t>(CONTEXT_WIN) * CONTEXT_WIN * sizeof(float);
    // row_sums, col_sums are for n_tokens, but allocated for max (CONTEXT_WIN).
    size_t sums_bytes_ph = static_cast<size_t>(CONTEXT_WIN) * sizeof(float);
    size_t accum_bytes_ph = static_cast<size_t>(h_attention) * sizeof(float);
    // MH, MV are d_embedding x h_attention on host, transposed to h_attention x d_embedding on device.
    size_t proj_mat_bytes_ph = static_cast<size_t>(h_attention) * d_embedding * sizeof(float);
    // EV is CONTEXT_WIN x d_embedding, kernels use n_tokens rows.
    size_t ev_processed_bytes_ph = static_cast<size_t>(CONTEXT_WIN) * d_embedding * sizeof(float);
    size_t embed_bytes_ph = static_cast<size_t>(d_embedding) * sizeof(float);

    // --- Aggregate Buffer Allocation ---
    float *agg_d_K = nullptr, *agg_d_Q = nullptr, *agg_d_KdotQ = nullptr, *agg_d_head_attention = nullptr;
    float *agg_d_row_sums = nullptr, *agg_d_col_sums = nullptr;
    float *agg_d_dh_accum = nullptr, *agg_d_dv_accum = nullptr;
    float *agg_d_MH_hxd = nullptr, *agg_d_MV_hxd = nullptr; // hxd means h_attention x d_embedding (transposed)
    float *agg_d_dh = nullptr, *agg_d_dv = nullptr;
    float *agg_d_EH = nullptr, *agg_d_EV_processed_data = nullptr;
    float *agg_d_ver_accumulated_ev = nullptr;
    float *agg_d_hor_inputs = nullptr, *agg_d_ver_inputs = nullptr;
    float *agg_d_hor_output = nullptr, *agg_d_ver_output = nullptr;
    float *agg_d_relu_hor_output = nullptr, *agg_d_relu_ver_output = nullptr;
    float *agg_d_mlp_bufferA_hor = nullptr, *agg_d_mlp_bufferB_hor = nullptr;
    float *agg_d_mlp_bufferA_ver = nullptr, *agg_d_mlp_bufferB_ver = nullptr;
    float *agg_d_mlp_pre_activation = nullptr;

    std::vector<cudaStream_t> streams(num_heads_in_col);
    std::vector<HeadDevicePointers> head_gpu_data(num_heads_in_col); // Using struct from block.hpp

    try {
        // Allocate aggregate buffers
        CUDA_CHECK(cudaMalloc(&agg_d_K, num_heads_in_col * k_bytes_ph));
        CUDA_CHECK(cudaMalloc(&agg_d_Q, num_heads_in_col * q_bytes_ph));
        CUDA_CHECK(cudaMalloc(&agg_d_KdotQ, num_heads_in_col * kdotq_bytes_ph));
        CUDA_CHECK(cudaMalloc(&agg_d_head_attention, num_heads_in_col * head_attention_bytes_ph));
        CUDA_CHECK(cudaMalloc(&agg_d_row_sums, num_heads_in_col * sums_bytes_ph));
        CUDA_CHECK(cudaMalloc(&agg_d_col_sums, num_heads_in_col * sums_bytes_ph));
        CUDA_CHECK(cudaMalloc(&agg_d_dh_accum, num_heads_in_col * accum_bytes_ph));
        CUDA_CHECK(cudaMalloc(&agg_d_dv_accum, num_heads_in_col * accum_bytes_ph));
        CUDA_CHECK(cudaMalloc(&agg_d_MH_hxd, num_heads_in_col * proj_mat_bytes_ph));
        CUDA_CHECK(cudaMalloc(&agg_d_MV_hxd, num_heads_in_col * proj_mat_bytes_ph));
        CUDA_CHECK(cudaMalloc(&agg_d_dh, num_heads_in_col * embed_bytes_ph));
        CUDA_CHECK(cudaMalloc(&agg_d_dv, num_heads_in_col * embed_bytes_ph));
        CUDA_CHECK(cudaMalloc(&agg_d_EH, num_heads_in_col * embed_bytes_ph));
        CUDA_CHECK(cudaMalloc(&agg_d_EV_processed_data, num_heads_in_col * ev_processed_bytes_ph));
        CUDA_CHECK(cudaMalloc(&agg_d_ver_accumulated_ev, num_heads_in_col * embed_bytes_ph));
        CUDA_CHECK(cudaMalloc(&agg_d_hor_inputs, num_heads_in_col * embed_bytes_ph));
        CUDA_CHECK(cudaMalloc(&agg_d_ver_inputs, num_heads_in_col * embed_bytes_ph));
        CUDA_CHECK(cudaMalloc(&agg_d_hor_output, num_heads_in_col * embed_bytes_ph));
        CUDA_CHECK(cudaMalloc(&agg_d_ver_output, num_heads_in_col * embed_bytes_ph));
        CUDA_CHECK(cudaMalloc(&agg_d_relu_hor_output, num_heads_in_col * embed_bytes_ph));
        CUDA_CHECK(cudaMalloc(&agg_d_relu_ver_output, num_heads_in_col * embed_bytes_ph));
        CUDA_CHECK(cudaMalloc(&agg_d_mlp_bufferA_hor, num_heads_in_col * embed_bytes_ph));
        CUDA_CHECK(cudaMalloc(&agg_d_mlp_bufferB_hor, num_heads_in_col * embed_bytes_ph));
        CUDA_CHECK(cudaMalloc(&agg_d_mlp_bufferA_ver, num_heads_in_col * embed_bytes_ph));
        CUDA_CHECK(cudaMalloc(&agg_d_mlp_bufferB_ver, num_heads_in_col * embed_bytes_ph));
        CUDA_CHECK(cudaMalloc(&agg_d_mlp_pre_activation, num_heads_in_col * embed_bytes_ph));

        for (int layer_idx = 0; layer_idx < num_heads_in_col; ++layer_idx) 
        {
            attention& head_cpu = this->b[layer_idx][i];
            HeadDevicePointers& current_head_pointers = head_gpu_data[layer_idx];
            CUDA_CHECK(cudaStreamCreateWithFlags(&streams[layer_idx], cudaStreamNonBlocking));
            cudaStream_t current_stream = streams[layer_idx];

            // Initialize accumulators on their respective streams
            CUDA_CHECK(cudaMemsetAsync(agg_d_dh_accum + layer_idx * (accum_bytes_ph / sizeof(float)), 0, accum_bytes_ph, current_stream));
            CUDA_CHECK(cudaMemsetAsync(agg_d_dv_accum + layer_idx * (accum_bytes_ph / sizeof(float)), 0, accum_bytes_ph, current_stream));

            // Per-head validation
            if (head_cpu.EH.size() != static_cast<size_t>(d_embedding) ||
                (!head_cpu.EV.mapped_data || head_cpu.EV.row != CONTEXT_WIN || head_cpu.EV.col != d_embedding) ||
                head_cpu.hor.hlayers.empty() || head_cpu.ver.hlayers.empty() || head_cpu.hor.weights.empty() || head_cpu.ver.weights.empty()) {
                throw std::runtime_error("Per-head data structure (EH, EV, MLP) invalid for head [" + std::to_string(layer_idx) + "][" + std::to_string(i) + "].");
            }
            if (head_cpu.hor.hlayers[0].size() != static_cast<size_t>(d_embedding) || head_cpu.ver.hlayers[0].size() != static_cast<size_t>(d_embedding) ||
                head_cpu.hor.weights.back().row != d_embedding || head_cpu.ver.weights.back().row != d_embedding) {
                throw std::runtime_error("MLP input/output layer dimension mismatch with 'd_embedding' for head [" + std::to_string(layer_idx) + "][" + std::to_string(i) + "].");
            }

            if (n_tokens <= 0) { // Handle skip for this specific head
                std::fill(head_cpu.EH.begin(), head_cpu.EH.end(), 0.0f);
                if (n_tokens == 0 && head_cpu.EV.mapped_data && head_cpu.EV.row > 0 && head_cpu.EV.col == d_embedding) {
                    std::fill_n(head_cpu.EV.mapped_data, head_cpu.EV.col, 0.0f);
                }
                continue;
            }

            // Setup device pointers for the current head by offsetting into aggregate buffers
            current_head_pointers.d_K = agg_d_K + layer_idx * (k_bytes_ph / sizeof(float));
            current_head_pointers.d_Q = agg_d_Q + layer_idx * (q_bytes_ph / sizeof(float));
            current_head_pointers.d_KdotQ = agg_d_KdotQ + layer_idx * (kdotq_bytes_ph / sizeof(float));
            current_head_pointers.d_head = agg_d_head_attention + layer_idx * (head_attention_bytes_ph / sizeof(float));
            current_head_pointers.d_pre_MH = agg_d_dh_accum + layer_idx * (accum_bytes_ph / sizeof(float));
            current_head_pointers.d_pre_MV = agg_d_dv_accum + layer_idx * (accum_bytes_ph / sizeof(float));
            current_head_pointers.d_MH_a = agg_d_MH_hxd + layer_idx * (proj_mat_bytes_ph / sizeof(float));
            current_head_pointers.d_MV_a = agg_d_MV_hxd + layer_idx * (proj_mat_bytes_ph / sizeof(float));
            current_head_pointers.d_EH = agg_d_EH + layer_idx * (embed_bytes_ph / sizeof(float));
            current_head_pointers.d_EV = agg_d_EV_processed_data + layer_idx * (ev_processed_bytes_ph / sizeof(float));

            // Local pointers for buffers not directly in HeadDevicePointers or for clarity
            float* local_d_row_sums = agg_d_row_sums + layer_idx * (sums_bytes_ph / sizeof(float));
            float* local_d_col_sums = agg_d_col_sums + layer_idx * (sums_bytes_ph / sizeof(float));
            float* local_d_dh = agg_d_dh + layer_idx * (embed_bytes_ph / sizeof(float));
            float* local_d_dv = agg_d_dv + layer_idx * (embed_bytes_ph / sizeof(float));
            float* local_d_ver_accumulated_ev = agg_d_ver_accumulated_ev + layer_idx * (embed_bytes_ph / sizeof(float));
            float* local_d_hor_inputs = agg_d_hor_inputs + layer_idx * (embed_bytes_ph / sizeof(float));
            float* local_d_ver_inputs = agg_d_ver_inputs + layer_idx * (embed_bytes_ph / sizeof(float));
            float* local_d_hor_output = agg_d_hor_output + layer_idx * (embed_bytes_ph / sizeof(float));
            float* local_d_ver_output = agg_d_ver_output + layer_idx * (embed_bytes_ph / sizeof(float));
            float* local_d_relu_hor_output = agg_d_relu_hor_output + layer_idx * (embed_bytes_ph / sizeof(float));
            float* local_d_relu_ver_output = agg_d_relu_ver_output + layer_idx * (embed_bytes_ph / sizeof(float));
            float* local_d_mlp_bufferA_hor = agg_d_mlp_bufferA_hor + layer_idx * (embed_bytes_ph / sizeof(float));
            float* local_d_mlp_bufferB_hor = agg_d_mlp_bufferB_hor + layer_idx * (embed_bytes_ph / sizeof(float));
            float* local_d_mlp_bufferA_ver = agg_d_mlp_bufferA_ver + layer_idx * (embed_bytes_ph / sizeof(float));
            float* local_d_mlp_bufferB_ver = agg_d_mlp_bufferB_ver + layer_idx * (embed_bytes_ph / sizeof(float));
            float* local_d_mlp_pre_activation = agg_d_mlp_pre_activation + layer_idx * (embed_bytes_ph / sizeof(float));

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
            // Assign from current_head_pointers or local_... to kernel argument pointers for clarity
            d_K = current_head_pointers.d_K;
            d_Q = current_head_pointers.d_Q;
            d_KdotQ = current_head_pointers.d_KdotQ;
            d_head_attention = current_head_pointers.d_head;
            d_row_sums = local_d_row_sums;
            d_col_sums = local_d_col_sums;
            d_dh_accum = current_head_pointers.d_pre_MH;
            d_dv_accum = current_head_pointers.d_pre_MV;
            d_MH_hxd = current_head_pointers.d_MH_a;
            d_MV_hxd = current_head_pointers.d_MV_a;
            d_dh = local_d_dh;
            d_dv = local_d_dv;
            d_EH = current_head_pointers.d_EH;
            d_EV_processed_data = current_head_pointers.d_EV;
            d_ver_accumulated_ev = local_d_ver_accumulated_ev;
            d_hor_inputs = local_d_hor_inputs;
            d_ver_inputs = local_d_ver_inputs;
            d_hor_output = local_d_hor_output;
            d_ver_output = local_d_ver_output;
            d_relu_hor_output = local_d_relu_hor_output;
            d_relu_ver_output = local_d_relu_ver_output;
            d_mlp_bufferA_hor = local_d_mlp_bufferA_hor;
            d_mlp_bufferB_hor = local_d_mlp_bufferB_hor;
            d_mlp_bufferA_ver = local_d_mlp_bufferA_ver;
            d_mlp_bufferB_ver = local_d_mlp_bufferB_ver;
            d_mlp_pre_activation = local_d_mlp_pre_activation;

            // Data Transfer H->D (Asynchronous)
            if (!head_cpu.K.mapped_data || !head_cpu.Q.mapped_data || !head_cpu.KdotQ.mapped_data)
                throw std::runtime_error("K, Q, or KdotQ have null mapped_data.");
            CUDA_CHECK(cudaMemcpyAsync(d_K, head_cpu.K.mapped_data, k_bytes_ph, cudaMemcpyHostToDevice, current_stream));
            CUDA_CHECK(cudaMemcpyAsync(d_Q, head_cpu.Q.mapped_data, q_bytes_ph, cudaMemcpyHostToDevice, current_stream));
            CUDA_CHECK(cudaMemcpyAsync(d_KdotQ, head_cpu.KdotQ.mapped_data, kdotq_bytes_ph, cudaMemcpyHostToDevice, current_stream));
            
            std::vector<float> flat_MH_hxd, flat_MV_hxd;
            transposeMatToFlatVector(head_cpu.MH, flat_MH_hxd); // MH is d_embedding x h_attention on host
            transposeMatToFlatVector(head_cpu.MV, flat_MV_hxd); // MV is d_embedding x h_attention on host
            CUDA_CHECK(cudaMemcpyAsync(d_MH_hxd, flat_MH_hxd.data(), proj_mat_bytes_ph, cudaMemcpyHostToDevice, current_stream)); // d_MH_hxd is h_attention x d_embedding
            CUDA_CHECK(cudaMemcpyAsync(d_MV_hxd, flat_MV_hxd.data(), proj_mat_bytes_ph, cudaMemcpyHostToDevice, current_stream)); // d_MV_hxd is h_attention x d_embedding
            CUDA_CHECK(cudaMemcpyAsync(d_EH, head_cpu.EH.data(), embed_bytes_ph, cudaMemcpyHostToDevice, current_stream));
            CUDA_CHECK(cudaMemcpyAsync(d_EV_processed_data, head_cpu.EV.mapped_data, ev_processed_bytes_ph, cudaMemcpyHostToDevice, current_stream));

            const int threadsPerBlock = 256;
            // Kernel: Perform KdotQ score normalisation (and masking if needed)
            cuLOTA<<<( (n_tokens * n_tokens) + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock, 0, current_stream>>>(d_KdotQ, d_head_attention, CONTEXT_WIN, CONTEXT_WIN, n_tokens, head_cpu.isSelfAttention); CUDA_CHECK(cudaGetLastError());
            // Kernel: Compute row and column sums of the attention matrix
            computeHeadSumsMaskedKernel<<<(n_tokens + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock, 0, current_stream>>>(d_head_attention, d_row_sums, d_col_sums, n_tokens, head_cpu.isSelfAttention); CUDA_CHECK(cudaGetLastError());
            // Kernel: Accumulate weighted K and Q vectors based on attention sums
            accumulateWeightedVectorsKernel<<<(h_attention + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock, 0, current_stream>>>(d_row_sums, d_col_sums, d_K, d_Q, d_dh_accum, d_dv_accum, n_tokens, h_attention); CUDA_CHECK(cudaGetLastError());
            // Kernel: Project accumulated dh_accum through MH to get dh
            matrixMultiplyKernel<<<(d_embedding + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock, 0, current_stream>>>(d_dh_accum, d_MH_hxd, d_dh, 1, h_attention, d_embedding); CUDA_CHECK(cudaGetLastError());
            // Kernel: Project accumulated dv_accum through MV to get dv
            matrixMultiplyKernel<<<(d_embedding + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock, 0, current_stream>>>(d_dv_accum, d_MV_hxd, d_dv, 1, h_attention, d_embedding); CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaStreamSynchronize(current_stream)); // Sync for this head before next stage

            // Kernel: Add dh to EH to get horizontal MLP input
            vectorAddKernel<<<(d_embedding + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock, 0, current_stream>>>(d_EH, d_dh, d_hor_inputs, d_embedding); CUDA_CHECK(cudaGetLastError());
            // Kernel: Accumulate rows of EV_processed_data (summing token embeddings)
            accumulateEVRowsKernel<<<(d_embedding + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock, 0, current_stream>>>(d_EV_processed_data, d_ver_accumulated_ev, n_tokens, d_embedding); CUDA_CHECK(cudaGetLastError());
            // Kernel: Add dv to accumulated EV to get vertical MLP input
            vectorAddKernel<<<(d_embedding + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock, 0, current_stream>>>(d_ver_accumulated_ev, d_dv, d_ver_inputs, d_embedding); CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaStreamSynchronize(current_stream)); // Sync for this head

            CUDA_CHECK(cudaMemcpyAsync(d_mlp_bufferA_hor, d_hor_inputs, embed_bytes_ph, cudaMemcpyDeviceToDevice, current_stream));
            CUDA_CHECK(cudaMemcpyAsync(d_mlp_bufferA_ver, d_ver_inputs, embed_bytes_ph, cudaMemcpyDeviceToDevice, current_stream));
            float* d_curr_in_h = d_mlp_bufferA_hor, *d_curr_out_h = d_mlp_bufferB_hor;
            float* d_curr_in_v = d_mlp_bufferA_ver, *d_curr_out_v = d_mlp_bufferB_ver;
            size_t num_weights_mats = head_cpu.hor.weights.size();
            if (num_weights_mats != head_cpu.ver.weights.size() || num_weights_mats == 0 || num_weights_mats != head_cpu.hor.hlayers.size())
                throw std::runtime_error("MLP config error.");

            for (size_t l_idx = 0; l_idx < num_weights_mats; ++l_idx) {
                bool is_last = (l_idx == num_weights_mats - 1);
                int in_sz = d_embedding, out_sz = d_embedding;
                const mat& w_h = head_cpu.hor.weights[l_idx];
                if(!w_h.mapped_data) throw std::runtime_error("MLP hor weights null.");
                size_t w_bytes = static_cast<size_t>(w_h.row) * w_h.col * sizeof(float); CUDA_CHECK(cudaMalloc(&d_mlp_weights, w_bytes));
                CUDA_CHECK(cudaMemcpyAsync(d_mlp_weights, w_h.mapped_data, w_bytes, cudaMemcpyHostToDevice, current_stream));
                // Kernel: MLP layer forward pass (matrix multiply)
                layerForwardKernel<<<(out_sz + threadsPerBlock - 1)/threadsPerBlock, threadsPerBlock, 0, current_stream>>>(d_curr_in_h, d_mlp_weights, is_last ? d_hor_output : d_mlp_pre_activation, in_sz, out_sz); CUDA_CHECK(cudaGetLastError());
                if (!is_last) {
                    // Kernel: Sigmoid activation for hidden MLP layer
                    cuSigmoid<<<(out_sz + threadsPerBlock -1)/threadsPerBlock, threadsPerBlock, 0, current_stream>>>(d_mlp_pre_activation, d_curr_out_h, out_sz); CUDA_CHECK(cudaGetLastError());
                    std::swap(d_curr_in_h, d_curr_out_h); 
                }
                CUDA_CHECK(cudaFree(d_mlp_weights)); d_mlp_weights = nullptr;
                const mat& w_v = head_cpu.ver.weights[l_idx];
                if(!w_v.mapped_data) throw std::runtime_error("MLP ver weights null.");
                CUDA_CHECK(cudaMalloc(&d_mlp_weights, w_bytes));
                CUDA_CHECK(cudaMemcpyAsync(d_mlp_weights, w_v.mapped_data, w_bytes, cudaMemcpyHostToDevice, current_stream));
                // Kernel: MLP layer forward pass (matrix multiply)
                layerForwardKernel<<<(out_sz + threadsPerBlock - 1)/threadsPerBlock, threadsPerBlock, 0, current_stream>>>(d_curr_in_v, d_mlp_weights, is_last ? d_ver_output : d_mlp_pre_activation, in_sz, out_sz); CUDA_CHECK(cudaGetLastError());
                if (!is_last) {
                    // Kernel: Sigmoid activation for hidden MLP layer
                    cuSigmoid<<<(out_sz + threadsPerBlock -1)/threadsPerBlock, threadsPerBlock, 0, current_stream>>>(d_mlp_pre_activation, d_curr_out_v, out_sz); CUDA_CHECK(cudaGetLastError());
                    std::swap(d_curr_in_v, d_curr_out_v); 
                }
                CUDA_CHECK(cudaFree(d_mlp_weights)); d_mlp_weights = nullptr;
            }
            CUDA_CHECK(cudaStreamSynchronize(current_stream)); // Sync for this head

            // Kernel: Apply ReLU to horizontal MLP output
            cuReLU<<<(d_embedding + threadsPerBlock - 1)/threadsPerBlock, threadsPerBlock, 0, current_stream>>>(d_hor_output, d_relu_hor_output, d_embedding); CUDA_CHECK(cudaGetLastError());
            // Kernel: Apply ReLU to vertical MLP output
            cuReLU<<<(d_embedding + threadsPerBlock - 1)/threadsPerBlock, threadsPerBlock, 0, current_stream>>>(d_ver_output, d_relu_ver_output, d_embedding); CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaStreamSynchronize(current_stream)); // Sync for this head

            // Kernel: Final residual update for EH
            vectorAddKernel<<<(d_embedding + threadsPerBlock - 1)/threadsPerBlock, threadsPerBlock, 0, current_stream>>>(d_EH, d_relu_hor_output, d_EH, d_embedding); CUDA_CHECK(cudaGetLastError());
            // Kernel: Final update for EV rows
            updateEVRowsKernel<<<(n_tokens + threadsPerBlock - 1)/threadsPerBlock, threadsPerBlock, 0, current_stream>>>(d_EV_processed_data, d_relu_ver_output, n_tokens, d_embedding); CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaStreamSynchronize(current_stream)); // Sync for this head

            CUDA_CHECK(cudaMemcpyAsync(head_cpu.EH.data(), d_EH, embed_bytes_ph, cudaMemcpyDeviceToHost, current_stream));
            CUDA_CHECK(cudaMemcpyAsync(head_cpu.EV.mapped_data, d_EV_processed_data, ev_processed_bytes_ph, cudaMemcpyDeviceToHost, current_stream));
        } // End of loop over heads (layer_idx)

        // Final synchronization for all streams
        for (int k = 0; k < num_heads_in_col; ++k) {
            if (streams[k]) { // Check if stream was created (e.g. if n_tokens > 0 for that head)
                CUDA_CHECK(cudaStreamSynchronize(streams[k]));
                CUDA_CHECK(cudaStreamDestroy(streams[k]));
            }
        }
    } // End try block for aggregate allocations
    catch (const std::exception& e) {
        std::cerr << "CUDA Exception in cu1parallelForprop (first block) for column " << i << ": " << e.what() << std::endl;
        // Cleanup aggregate memory and streams on error
        for (int k=0; k<num_heads_in_col; ++k) { if(streams[k]) cudaStreamDestroy(streams[k]); }
        cudaFree(agg_d_K); cudaFree(agg_d_Q); cudaFree(agg_d_KdotQ); cudaFree(agg_d_head_attention); cudaFree(agg_d_row_sums); cudaFree(agg_d_col_sums);
        cudaFree(agg_d_dh_accum); cudaFree(agg_d_dv_accum); cudaFree(agg_d_MH_hxd); cudaFree(agg_d_MV_hxd); cudaFree(agg_d_dh); cudaFree(agg_d_dv);
        cudaFree(agg_d_EH); cudaFree(agg_d_EV_processed_data); cudaFree(agg_d_ver_accumulated_ev); cudaFree(agg_d_hor_inputs); cudaFree(agg_d_ver_inputs);
        cudaFree(agg_d_hor_output); cudaFree(agg_d_ver_output); cudaFree(agg_d_relu_hor_output); cudaFree(agg_d_relu_ver_output);
        cudaFree(agg_d_mlp_bufferA_hor); cudaFree(agg_d_mlp_bufferB_hor); cudaFree(agg_d_mlp_bufferA_ver); cudaFree(agg_d_mlp_bufferB_ver);
        cudaFree(agg_d_mlp_pre_activation);
        throw;
    }
    // Free aggregate memory on success
    cudaFree(agg_d_K); cudaFree(agg_d_Q); cudaFree(agg_d_KdotQ); cudaFree(agg_d_head_attention); cudaFree(agg_d_row_sums); cudaFree(agg_d_col_sums);
    cudaFree(agg_d_dh_accum); cudaFree(agg_d_dv_accum); cudaFree(agg_d_MH_hxd); cudaFree(agg_d_MV_hxd); cudaFree(agg_d_dh); cudaFree(agg_d_dv);
    cudaFree(agg_d_EH); cudaFree(agg_d_EV_processed_data); cudaFree(agg_d_ver_accumulated_ev); cudaFree(agg_d_hor_inputs); cudaFree(agg_d_ver_inputs);
    cudaFree(agg_d_hor_output); cudaFree(agg_d_ver_output); cudaFree(agg_d_relu_hor_output); cudaFree(agg_d_relu_ver_output);
    cudaFree(agg_d_mlp_bufferA_hor); cudaFree(agg_d_mlp_bufferB_hor); cudaFree(agg_d_mlp_bufferA_ver); cudaFree(agg_d_mlp_bufferB_ver);
    cudaFree(agg_d_mlp_pre_activation);
}


/**
 * @brief CUDA forward propagation on single ith column of a SUBSEQUENT block (blockCount > 0).
 * @param EVp vector of EV
 * @param in dimension of embeddings and mlp input/output
 * @param tokenCount number of tokens available in full context
 * @param blockCount current block in full context
 * @param layers number of mlp weight matrices
 * @param n local context for block
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

    const int num_heads_in_col = this->x;
    const int d_embedding = EMBEDDING;
    const int h_attention = MATHEIGHTS;
    const int total_token_count_param = tokenCount; // Renaming for clarity
    const int block_idx_param = blockCount;
    // const int context_window_size_param = n;

    if (block_idx_param == 0) {
        throw std::logic_error("cu1ParallelForprop (subsequent) called with blockCount == 0. Use the first block overload.");
    }
    const int num_ev_rows_to_process_for_evp = total_token_count_param; // EVp comes with totalTokenCount rows
        
    int start_idx_in_full_context = (blockCount - 1) * n; // n is contextWindowSize
    int end_idx_in_full_context = std::min(tokenCount, blockCount * n); // tokenCount is totalTokenCount
    const int count_tokens_this_block = std::max(0, end_idx_in_full_context - start_idx_in_full_context);

    if (count_tokens_this_block <= 0) {
        std::cerr << "Warning: cu1ParallelForprop (subsequent block) for head [" << layers << "][" << i << "] with calculated count <= 0. Skipping." << std::endl;
    }

    for(int layer_idx = 0; layer_idx < layers; ++layer_idx) 
    {
        if (b[layer_idx][i].K.row != CONTEXT_WIN || b[layer_idx][i].K.col != MATHEIGHTS ||
            b[layer_idx][i].Q.row != CONTEXT_WIN || b[layer_idx][i].Q.col != MATHEIGHTS ||
            b[layer_idx][i].KdotQ.row != CONTEXT_WIN || b[layer_idx][i].KdotQ.col != CONTEXT_WIN ||
            b[layer_idx][i].MH.row != EMBEDDING || b[layer_idx][i].MH.col != MATHEIGHTS ||
            b[layer_idx][i].MV.row != EMBEDDING || b[layer_idx][i].MV.col != MATHEIGHTS ||
            b[layer_idx][i].EH.size() != static_cast<size_t>(in) ||
            (!b[layer_idx][i].EV.mapped_data || CONTEXT_WIN > b[layer_idx][i].EV.row || 
            b[layer_idx][i].EV.col != in) || b[layer_idx][i].hor.hlayers.empty() || b[layer_idx][i].ver.hlayers.empty() 
            || b[layer_idx][i].hor.weights.empty() || b[layer_idx][i].ver.weights.empty()) 
            {
                throw std::runtime_error("Attention component dimension mismatch (subsequent block) for head [" +
                                            std::to_string(layer_idx) + "][" + std::to_string(i) + "]. K.row=" + std::to_string(b[layer_idx][i].K.row) + ", count_tokens_this_block=" + std::to_string(count_tokens_this_block));
            }
            if (EVp[layer_idx].size() != static_cast<size_t>(CONTEXT_WIN) || (!EVp[layer_idx].empty() && EVp[layer_idx][0].size() != static_cast<size_t>(in))) {
                throw std::runtime_error("EVp_head dimension mismatch for head [" + std::to_string(layer_idx) + "][" + std::to_string(i) +
                                            "]. Expected rows " + std::to_string(CONTEXT_WIN) + " (totalTokenCount), got " + std::to_string(EVp.size()) +
                                            ". Expected cols " + std::to_string(in) + ", got " + (EVp[layer_idx].empty() ? "N/A" : std::to_string(EVp[layer_idx][0].size())) );
        }
    }

    // Per-head byte sizes
    // K, Q, KdotQ, head_attention, sums are based on count_tokens_this_block.
    size_t k_bytes_ph = static_cast<size_t>(count_tokens_this_block) * h_attention * sizeof(float);
    size_t q_bytes_ph = static_cast<size_t>(count_tokens_this_block) * h_attention * sizeof(float);
    size_t kdotq_bytes_ph = static_cast<size_t>(count_tokens_this_block) * count_tokens_this_block * sizeof(float);
    size_t head_attention_bytes_ph = static_cast<size_t>(count_tokens_this_block) * count_tokens_this_block * sizeof(float);
    size_t sums_bytes_ph = static_cast<size_t>(count_tokens_this_block) * sizeof(float);

    size_t accum_bytes_ph = static_cast<size_t>(h_attention) * sizeof(float);
    size_t proj_mat_bytes_ph = static_cast<size_t>(h_attention) * d_embedding * sizeof(float); // Transposed
    // EVp_head data (input from previous block)
    size_t ev_from_prev_block_bytes_ph = static_cast<size_t>(num_ev_rows_to_process_for_evp) * d_embedding * sizeof(float);
    // Output EV for the current head (head_cpu.EV) is typically CONTEXT_WIN x d_embedding
    // size_t ev_output_current_block_bytes_ph = static_cast<size_t>(CONTEXT_WIN) * d_embedding * sizeof(float);
    size_t embed_bytes_ph = static_cast<size_t>(d_embedding) * sizeof(float);

    // --- Aggregate Buffer Allocation ---
    float *agg_d_K = nullptr, *agg_d_Q = nullptr, *agg_d_KdotQ = nullptr, *agg_d_head_attention = nullptr;
    float *agg_d_row_sums = nullptr, *agg_d_col_sums = nullptr;
    float *agg_d_dh_accum = nullptr, *agg_d_dv_accum = nullptr;
    float *agg_d_MH_hxd = nullptr, *agg_d_MV_hxd = nullptr;
    float *agg_d_dh = nullptr, *agg_d_dv = nullptr;
    float *agg_d_EH = nullptr, *agg_d_EV_processed_data_from_prev_block = nullptr;
    // Note: The output EV for the current block (head_cpu.EV) will be updated based on EVp.
    // We'll use agg_d_EV_processed_data_from_prev_block to hold EVp on device, update it, then copy to head_cpu.EV.
    float *agg_d_ver_accumulated_ev = nullptr;
    float *agg_d_mlp_pre_activation = nullptr;

    std::vector<cudaStream_t> streams(num_heads_in_col);
    std::vector<HeadDevicePointers> head_gpu_data(num_heads_in_col);

    try {
        // Allocate aggregate buffers
        if (count_tokens_this_block > 0) { // Buffers dependent on count_tokens_this_block
            CUDA_CHECK(cudaMalloc(&agg_d_K, num_heads_in_col * k_bytes_ph));
            CUDA_CHECK(cudaMalloc(&agg_d_Q, num_heads_in_col * q_bytes_ph));
            CUDA_CHECK(cudaMalloc(&agg_d_KdotQ, num_heads_in_col * kdotq_bytes_ph));
            CUDA_CHECK(cudaMalloc(&agg_d_head_attention, num_heads_in_col * head_attention_bytes_ph));
            CUDA_CHECK(cudaMalloc(&agg_d_row_sums, num_heads_in_col * sums_bytes_ph));
            CUDA_CHECK(cudaMalloc(&agg_d_col_sums, num_heads_in_col * sums_bytes_ph));
        }
        // Buffers always needed
        CUDA_CHECK(cudaMalloc(&agg_d_dh_accum, num_heads_in_col * accum_bytes_ph));
        CUDA_CHECK(cudaMalloc(&agg_d_dv_accum, num_heads_in_col * accum_bytes_ph));
        CUDA_CHECK(cudaMalloc(&agg_d_MH_hxd, num_heads_in_col * proj_mat_bytes_ph));
        CUDA_CHECK(cudaMalloc(&agg_d_MV_hxd, num_heads_in_col * proj_mat_bytes_ph));
        CUDA_CHECK(cudaMalloc(&agg_d_dh, num_heads_in_col * embed_bytes_ph));
        CUDA_CHECK(cudaMalloc(&agg_d_dv, num_heads_in_col * embed_bytes_ph));
        CUDA_CHECK(cudaMalloc(&agg_d_EH, num_heads_in_col * embed_bytes_ph));
        CUDA_CHECK(cudaMalloc(&agg_d_EV_processed_data_from_prev_block, num_heads_in_col * ev_from_prev_block_bytes_ph));
        CUDA_CHECK(cudaMalloc(&agg_d_ver_accumulated_ev, num_heads_in_col * embed_bytes_ph));
        CUDA_CHECK(cudaMalloc(&agg_d_mlp_pre_activation, num_heads_in_col * embed_bytes_ph));

        for (int layer_idx = 0; layer_idx < num_heads_in_col; ++layer_idx) 
        {
            attention& head_cpu = this->b[layer_idx][i];
            const std::vector<std::vector<float>>& EVp_head = EVp[layer_idx]; // EVp for the current head
            HeadDevicePointers& current_head_pointers = head_gpu_data[layer_idx];
            CUDA_CHECK(cudaStreamCreateWithFlags(&streams[layer_idx], cudaStreamNonBlocking));
            cudaStream_t current_stream = streams[layer_idx];

            // Initialize accumulators on their respective streams
            CUDA_CHECK(cudaMemsetAsync(agg_d_dh_accum + layer_idx * (accum_bytes_ph / sizeof(float)), 0, accum_bytes_ph, current_stream));
            CUDA_CHECK(cudaMemsetAsync(agg_d_dv_accum + layer_idx * (accum_bytes_ph / sizeof(float)), 0, accum_bytes_ph, current_stream));

            if (count_tokens_this_block <= 0) {
                std::fill(head_cpu.EH.begin(), head_cpu.EH.end(), 0.0f);
                continue;
            }
            // Per-head validation (already done partially outside, can add more specifics if needed)
            if (head_cpu.EV.row < num_ev_rows_to_process_for_evp || head_cpu.EV.col != d_embedding) {
                throw std::runtime_error("head_cpu.EV (output) not properly sized for EVp in subsequent block for head [" + std::to_string(layer_idx) + "][" + std::to_string(i) + "].");
            }

            // Setup device pointers for the current head (offsets into aggregate buffers)
            if (count_tokens_this_block > 0) {
                current_head_pointers.d_K = agg_d_K + layer_idx * (k_bytes_ph / sizeof(float));
                current_head_pointers.d_Q = agg_d_Q + layer_idx * (q_bytes_ph / sizeof(float));
                current_head_pointers.d_KdotQ = agg_d_KdotQ + layer_idx * (kdotq_bytes_ph / sizeof(float));
                current_head_pointers.d_head = agg_d_head_attention + layer_idx * (head_attention_bytes_ph / sizeof(float));
            }
            current_head_pointers.d_pre_MH = agg_d_dh_accum + layer_idx * (accum_bytes_ph / sizeof(float));
            current_head_pointers.d_pre_MV = agg_d_dv_accum + layer_idx * (accum_bytes_ph / sizeof(float));
            current_head_pointers.d_MH_a = agg_d_MH_hxd + layer_idx * (proj_mat_bytes_ph / sizeof(float));
            current_head_pointers.d_MV_a = agg_d_MV_hxd + layer_idx * (proj_mat_bytes_ph / sizeof(float));
            current_head_pointers.d_EH = agg_d_EH + layer_idx * (embed_bytes_ph / sizeof(float));
            // d_EV in HeadDevicePointers will point to the buffer holding EVp data for this head
            current_head_pointers.d_EV = agg_d_EV_processed_data_from_prev_block + layer_idx * (ev_from_prev_block_bytes_ph / sizeof(float));

            // Local pointers for buffers not directly in HeadDevicePointers or for clarity
            float* local_d_row_sums = (count_tokens_this_block > 0) ? (agg_d_row_sums + layer_idx * (sums_bytes_ph / sizeof(float))) : nullptr;
            float* local_d_col_sums = (count_tokens_this_block > 0) ? (agg_d_col_sums + layer_idx * (sums_bytes_ph / sizeof(float))) : nullptr;
            // ... (Assign all other local_... pointers similarly to the first overload)
            float* local_d_dh = agg_d_dh + layer_idx * (embed_bytes_ph / sizeof(float));
            // ...

            float *d_K = nullptr, *d_Q = nullptr, *d_KdotQ = nullptr, *d_head_attention = nullptr;
            float *d_row_sums = nullptr, *d_col_sums = nullptr;
            float *d_dh_accum = nullptr, *d_dv_accum = nullptr;
            float *d_MH_hxd = nullptr, *d_MV_hxd = nullptr;
            float *d_dh = nullptr, *d_dv = nullptr;
            float *d_EH = nullptr, *d_EV_processed_data_from_prev_block = nullptr; // Data from EVp_head
            // float *d_EV_output_current_block = nullptr; // Corresponds to head.EV, for updating
            float *d_ver_accumulated_ev = nullptr;
            float *d_hor_inputs = nullptr, *d_ver_inputs = nullptr;
            float *d_hor_output = nullptr, *d_ver_output = nullptr;
            float *d_relu_hor_output = nullptr, *d_relu_ver_output = nullptr;
            float *d_mlp_bufferA_hor = nullptr, *d_mlp_bufferB_hor = nullptr;
            float *d_mlp_bufferA_ver = nullptr, *d_mlp_bufferB_ver = nullptr;
            float *d_mlp_pre_activation = nullptr;
            float *d_mlp_weights = nullptr;
            // Assign from current_head_pointers or local_... to kernel argument pointers
            d_K = current_head_pointers.d_K;
            d_Q = current_head_pointers.d_Q;
            d_KdotQ = current_head_pointers.d_KdotQ;
            d_head_attention = current_head_pointers.d_head;
            d_row_sums = local_d_row_sums;
            d_col_sums = local_d_col_sums;
            d_dh_accum = current_head_pointers.d_pre_MH;
            d_dv_accum = current_head_pointers.d_pre_MV;
            d_MH_hxd = current_head_pointers.d_MH_a;
            d_MV_hxd = current_head_pointers.d_MV_a;
            d_dh = local_d_dh;
            d_dv = agg_d_dv + layer_idx * (embed_bytes_ph / sizeof(float)); // Assign local_d_dv
            // d_dv = local_d_dv;
            d_EH = current_head_pointers.d_EH;
            d_EV_processed_data_from_prev_block = current_head_pointers.d_EV; // EVp data

            // Data Transfer H->D (Asynchronous)
            if (count_tokens_this_block > 0) {
                if (!head_cpu.K.mapped_data || !head_cpu.Q.mapped_data || !head_cpu.KdotQ.mapped_data)
                throw std::runtime_error("K, Q, or KdotQ have null mapped_data.");
                CUDA_CHECK(cudaMemcpyAsync(d_K, head_cpu.K.mapped_data, k_bytes_ph, cudaMemcpyHostToDevice, current_stream));
                CUDA_CHECK(cudaMemcpyAsync(d_Q, head_cpu.Q.mapped_data, q_bytes_ph, cudaMemcpyHostToDevice, current_stream));
                CUDA_CHECK(cudaMemcpyAsync(d_KdotQ, head_cpu.KdotQ.mapped_data, kdotq_bytes_ph, cudaMemcpyHostToDevice, current_stream));
            }
            std::vector<float> flat_MH_hxd, flat_MV_hxd;
            transposeMatToFlatVector(head_cpu.MH, flat_MH_hxd); transposeMatToFlatVector(head_cpu.MV, flat_MV_hxd);
            CUDA_CHECK(cudaMemcpyAsync(d_MH_hxd, flat_MH_hxd.data(), proj_mat_bytes_ph, cudaMemcpyHostToDevice, current_stream));
            CUDA_CHECK(cudaMemcpyAsync(d_MV_hxd, flat_MV_hxd.data(), proj_mat_bytes_ph, cudaMemcpyHostToDevice, current_stream));
            CUDA_CHECK(cudaMemcpyAsync(d_EH, head_cpu.EH.data(), embed_bytes_ph, cudaMemcpyHostToDevice, current_stream));
            
            std::vector<float> flat_EVp_head;
            flatten2DVector(EVp_head, flat_EVp_head, num_ev_rows_to_process_for_evp, d_embedding);
            CUDA_CHECK(cudaMemcpyAsync(d_EV_processed_data_from_prev_block, flat_EVp_head.data(), ev_from_prev_block_bytes_ph, cudaMemcpyHostToDevice, current_stream));

            const int threadsPerBlock = 256;
            if (count_tokens_this_block > 0) {
                // Kernel: Perform KdotQ score normalisation (and masking if needed)
                cuLOTA<<<( (count_tokens_this_block * count_tokens_this_block) + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock, 0, current_stream>>>(d_KdotQ, d_head_attention, count_tokens_this_block, count_tokens_this_block); CUDA_CHECK(cudaGetLastError());
                // Kernel: Compute row and column sums of the attention matrix
                computeHeadSumsMaskedKernel<<<(count_tokens_this_block + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock, 0, current_stream>>>(d_head_attention, d_row_sums, d_col_sums, count_tokens_this_block, head_cpu.isSelfAttention); CUDA_CHECK(cudaGetLastError());
                // Kernel: Accumulate weighted K and Q vectors
                accumulateWeightedVectorsKernel<<<(h_attention + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock, 0, current_stream>>>(d_row_sums, d_col_sums, d_K, d_Q, d_dh_accum, d_dv_accum, count_tokens_this_block, h_attention); CUDA_CHECK(cudaGetLastError());
            }
            // Kernel: Project accumulated dh_accum through MH
            matrixMultiplyKernel<<<(d_embedding + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock, 0, current_stream>>>(d_dh_accum, d_MH_hxd, d_dh, 1, h_attention, d_embedding); CUDA_CHECK(cudaGetLastError());
            // Kernel: Project accumulated dv_accum through MV
            matrixMultiplyKernel<<<(d_embedding + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock, 0, current_stream>>>(d_dv_accum, d_MV_hxd, d_dv, 1, h_attention, d_embedding); CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaStreamSynchronize(current_stream));

            // Kernel: Add dh to EH for horizontal MLP input
            vectorAddKernel<<<(d_embedding + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock, 0, current_stream>>>(d_EH, d_dh, d_hor_inputs, d_embedding); CUDA_CHECK(cudaGetLastError());
            // Kernel: Sum rows of EVp data (from previous block)
            accumulateEVRowsKernel<<<(d_embedding + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock, 0, current_stream>>>(d_EV_processed_data_from_prev_block, d_ver_accumulated_ev, num_ev_rows_to_process_for_evp, d_embedding); CUDA_CHECK(cudaGetLastError());
            // Kernel: Add dv to accumulated EVp for vertical MLP input
            vectorAddKernel<<<(d_embedding + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock, 0, current_stream>>>(d_ver_accumulated_ev, d_dv, d_ver_inputs, d_embedding); CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaStreamSynchronize(current_stream));

            // --- MLP Forward (Identical to 1st overload's MLP logic) ---
            CUDA_CHECK(cudaMemcpyAsync(d_mlp_bufferA_hor, d_hor_inputs, embed_bytes_ph, cudaMemcpyDeviceToDevice, current_stream));
            CUDA_CHECK(cudaMemcpyAsync(d_mlp_bufferA_ver, d_ver_inputs, embed_bytes_ph, cudaMemcpyDeviceToDevice, current_stream));
            float* d_curr_in_h = d_mlp_bufferA_hor, *d_curr_out_h = d_mlp_bufferB_hor;
            float* d_curr_in_v = d_mlp_bufferA_ver, *d_curr_out_v = d_mlp_bufferB_ver;
            size_t num_weights_mats = head_cpu.hor.weights.size();
            if (num_weights_mats != head_cpu.ver.weights.size() || num_weights_mats == 0 || num_weights_mats != head_cpu.hor.hlayers.size()) throw std::runtime_error("MLP config error.");

            for (size_t l_idx = 0; l_idx < num_weights_mats; ++l_idx) {
                bool is_last = (l_idx == num_weights_mats - 1);
                int in_sz = d_embedding, out_sz = d_embedding;
                const mat& w_h = head_cpu.hor.weights[l_idx]; if(!w_h.mapped_data) throw std::runtime_error("MLP hor weights null.");
                size_t w_bytes = static_cast<size_t>(w_h.row) * w_h.col * sizeof(float); CUDA_CHECK(cudaMalloc(&d_mlp_weights, w_bytes));
                CUDA_CHECK(cudaMemcpyAsync(d_mlp_weights, w_h.mapped_data, w_bytes, cudaMemcpyHostToDevice, current_stream));
                // Kernel: MLP layer forward (hor)
                layerForwardKernel<<<(out_sz + threadsPerBlock - 1)/threadsPerBlock, threadsPerBlock, 0, current_stream>>>(d_curr_in_h, d_mlp_weights, is_last ? d_hor_output : d_mlp_pre_activation, in_sz, out_sz); CUDA_CHECK(cudaGetLastError());
                if (!is_last) { 
                    // Kernel: Sigmoid activation (hor)
                    cuSigmoid<<<(out_sz + threadsPerBlock -1)/threadsPerBlock, threadsPerBlock, 0, current_stream>>>(d_mlp_pre_activation, d_curr_out_h, out_sz); CUDA_CHECK(cudaGetLastError()); std::swap(d_curr_in_h, d_curr_out_h); 
                }
                CUDA_CHECK(cudaFree(d_mlp_weights)); d_mlp_weights = nullptr;
                
                const mat& w_v = head_cpu.ver.weights[l_idx]; if(!w_v.mapped_data) throw std::runtime_error("MLP ver weights null.");
                CUDA_CHECK(cudaMalloc(&d_mlp_weights, w_bytes));
                CUDA_CHECK(cudaMemcpyAsync(d_mlp_weights, w_v.mapped_data, w_bytes, cudaMemcpyHostToDevice, current_stream));
                // Kernel: MLP layer forward (ver)
                layerForwardKernel<<<(out_sz + threadsPerBlock - 1)/threadsPerBlock, threadsPerBlock, 0, current_stream>>>(d_curr_in_v, d_mlp_weights, is_last ? d_ver_output : d_mlp_pre_activation, in_sz, out_sz); CUDA_CHECK(cudaGetLastError());
                if (!is_last) { 
                    // Kernel: Sigmoid activation (ver)
                    cuSigmoid<<<(out_sz + threadsPerBlock -1)/threadsPerBlock, threadsPerBlock, 0, current_stream>>>(d_mlp_pre_activation, d_curr_out_v, out_sz); CUDA_CHECK(cudaGetLastError()); std::swap(d_curr_in_v, d_curr_out_v); 
                }
                CUDA_CHECK(cudaFree(d_mlp_weights)); d_mlp_weights = nullptr;
            }
            CUDA_CHECK(cudaStreamSynchronize(current_stream));
            // --- End MLP Forward ---

            // Kernel: Apply ReLU to horizontal MLP output
            cuReLU<<<(d_embedding + threadsPerBlock - 1)/threadsPerBlock, threadsPerBlock, 0, current_stream>>>(d_hor_output, d_relu_hor_output, d_embedding); CUDA_CHECK(cudaGetLastError());
            // Kernel: Apply ReLU to vertical MLP output
            cuReLU<<<(d_embedding + threadsPerBlock - 1)/threadsPerBlock, threadsPerBlock, 0, current_stream>>>(d_ver_output, d_relu_ver_output, d_embedding); CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaStreamSynchronize(current_stream));

            // Kernel: Final residual update for EH
            vectorAddKernel<<<(d_embedding + threadsPerBlock - 1)/threadsPerBlock, threadsPerBlock, 0, current_stream>>>(d_EH, d_relu_hor_output, d_EH, d_embedding); CUDA_CHECK(cudaGetLastError());
            // Kernel: Update EV rows (from EVp) with the vertical MLP output
            updateEVRowsKernel<<<(num_ev_rows_to_process_for_evp + threadsPerBlock - 1)/threadsPerBlock, threadsPerBlock, 0, current_stream>>>(d_EV_processed_data_from_prev_block, d_relu_ver_output, num_ev_rows_to_process_for_evp, d_embedding); CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaStreamSynchronize(current_stream));

            CUDA_CHECK(cudaMemcpyAsync(head_cpu.EH.data(), d_EH, embed_bytes_ph, cudaMemcpyDeviceToHost, current_stream));
            // Copy the updated EV (which originated from EVp_head) back to head.EV
            // This assumes head.EV is meant to store the output EV for this block, based on previous block's EV.
            if (static_cast<size_t>(head_cpu.EV.row) < static_cast<size_t>(num_ev_rows_to_process_for_evp) || static_cast<size_t>(head_cpu.EV.col) != static_cast<size_t>(d_embedding)) {
                 throw std::runtime_error("head.EV dimensions are insufficient to store updated EV from previous block.");
            }
            CUDA_CHECK(cudaMemcpyAsync(head_cpu.EV.mapped_data, d_EV_processed_data_from_prev_block, ev_from_prev_block_bytes_ph, cudaMemcpyDeviceToHost, current_stream));
        } // End loop over heads

        // Final synchronization for all streams
        for (int k = 0; k < num_heads_in_col; ++k) {
            if (streams[k]) {
                CUDA_CHECK(cudaStreamSynchronize(streams[k]));
                CUDA_CHECK(cudaStreamDestroy(streams[k]));
            }
        }
    } // End try block for aggregate allocations
    catch (const std::exception& e) {
        std::cerr << "CUDA Exception in cu1ParallelForprop (subsequent block) for column " << i << ": " << e.what() << std::endl;
        // Cleanup aggregate memory and streams
        for (int k=0; k<num_heads_in_col; ++k) { if(streams[k]) cudaStreamDestroy(streams[k]); }
        cudaFree(agg_d_K); cudaFree(agg_d_Q); /* ... free ALL aggregate buffers ... */ cudaFree(agg_d_mlp_pre_activation);
        throw;
    }
    // Free aggregate memory on success
    cudaFree(agg_d_K); cudaFree(agg_d_Q); /* ... free ALL aggregate buffers ... */ cudaFree(agg_d_mlp_pre_activation);
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
