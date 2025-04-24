
#include "include/mlp.hpp"
#include "include/attention.hpp" // Includes constants like EMBEDDING, MATHEIGHTS, etc.
#include "include/block.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept> // For runtime_error, out_of_range
#include <iostream> // For error logging
#include <string>   // For std::to_string in error messages
#include <maths.hpp> // For SCALING, LOTA, LOTAder (if needed host-side, unlikely here)

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        throw std::runtime_error(cudaGetErrorString(err)); \
    } \
} while (0)
 

/**
 * @brief CUDA backward propagation for a single column in the FIRST block,
 *        driven by horizontal error (EH). Processes heads b[row][layno].
 *        IMPLEMENTATION USES DIRECT KERNEL CALLS PER HEAD (INEFFICIENT).
 * @param expectedH Expected horizontal embedding for the output of this column's heads.
 * @param in Embedding dimension.
 * @param layers Number of MLP layers.
 * @param layno The column index within the block (0 to y-1).
 */
void block::cu1ParallelBackward1stBlock(std::vector<float>& expectedH, int& in, int& layers, int layno)
{
    // Validate column number
    if (layno < 0 || layno >= y) {
        throw std::out_of_range("cu1ParallelBackward1stBlock(H): Column index 'layno' (" + std::to_string(layno) + 
            ") is out of range [0, " + std::to_string(y - 1) + "].");
    }
    if (expectedH.size() != EMBEDDING) {
        throw std::runtime_error("ExpectedH vector size mismatch in cu1ParallelBackward1stBlock(H). Expected " + 
            std::to_string(EMBEDDING) + ", got " + std::to_string(expectedH.size()));
    }
    if (EMBEDDING != in) {
        throw std::runtime_error("Embedding dimension mismatch: EMBEDDING vs in");
    }

    // Constants used within the loop
    const int embedding_dim = EMBEDDING;
    const int mat_heights = MATHEIGHTS;
    const int context_win = CONTEXT_WIN;
    const float learning_rate = LEARNING;
    const float scaling_factor = SCALING;
    const size_t embed_bytes = embedding_dim * sizeof(float);
    const size_t weights_bytes = embedding_dim * embedding_dim * sizeof(float); // MLP weights
    const int mh_mv_mq_mk_size = mat_heights * embedding_dim;
    const size_t mh_mv_mq_mk_bytes = mh_mv_mq_mk_size * sizeof(float);
    const int ev_size = context_win * embedding_dim;
    const size_t ev_bytes = ev_size * sizeof(float);

    // Kernel Launch Config (Define once)
    int threadsPerBlock1D = 256;
    dim3 blockDim1D(threadsPerBlock1D);
    int blocksPerGridEmbed = (embedding_dim + threadsPerBlock1D - 1) / threadsPerBlock1D;
    dim3 gridDimEmbed(blocksPerGridEmbed);
    int blocksPerGridMatHeights = (mat_heights + threadsPerBlock1D - 1) / threadsPerBlock1D;
    dim3 gridDimMatHeights(blocksPerGridMatHeights);
    int blocksPerGridMatrix = (mh_mv_mq_mk_size + threadsPerBlock1D - 1) / threadsPerBlock1D;
    dim3 gridDimMatrix(blocksPerGridMatrix);

    dim3 blockDim2D(16, 16);
    dim3 gridDimEmbed2D((embedding_dim + blockDim2D.x - 1) / blockDim2D.x, (embedding_dim + blockDim2D.y - 1) / blockDim2D.y);
    dim3 gridDimMatrix2D((embedding_dim + blockDim2D.x - 1) / blockDim2D.x, (mat_heights + blockDim2D.y - 1) / blockDim2D.y);


    // Iterate backwards through the rows (parallels/layers) for the specified column
    for (int i = x - 1; i >= 0; --i) { // 'i' is the row index
        attention& head_obj = b[i][layno]; // Reference to the current head object
        const int token_count = head_obj.tokenCount;
        const int head_size = token_count * token_count;
        const size_t head_bytes = head_size * sizeof(float);
        const int k_q_size = token_count * mat_heights;
        const size_t k_q_bytes = k_q_size * sizeof(float);
        // Adjust 2D grids based on token_count
        dim3 gridDimHead2D((token_count + blockDim2D.x - 1) / blockDim2D.x, (token_count + blockDim2D.y - 1) / blockDim2D.y);
        dim3 gridDimKQGrad2D((mat_heights + blockDim2D.x - 1) / blockDim2D.x, (token_count + blockDim2D.y - 1) / blockDim2D.y);
        int blocksPerGridHead = (head_size + threadsPerBlock1D - 1) / threadsPerBlock1D;
        dim3 gridDimHead(blocksPerGridHead);

        bool is_first_head = (i == 0 && layno == 0);

        // --- Device Pointers (Allocate per head - Inefficient) ---
        float *d_expected_h = nullptr, *d_EH = nullptr, *d_EV = nullptr;
        float *d_grad_EH = nullptr, *d_grad_EV_scaled = nullptr;
        float *d_grad_dh = nullptr, *d_grad_dv = nullptr;
        float *d_KdotQ = nullptr, *d_head = nullptr;
        float *d_K = nullptr, *d_Q = nullptr;
        float *d_pre_MH = nullptr, *d_pre_MV = nullptr;
        float *d_MH_a = nullptr, *d_MV_a = nullptr, *d_MQ_a = nullptr, *d_MK_a = nullptr;
        float *d_grad_MH = nullptr, *d_grad_MV = nullptr;
        float *d_grad_head = nullptr;
        float *d_lota_deriv = nullptr;
        float *d_grad_KdotQ = nullptr;
        float *d_grad_K = nullptr, *d_grad_Q = nullptr;
        float *d_grad_MQ = nullptr, *d_grad_MK = nullptr;
        // MLP Internals
        std::vector<float*> d_hor_activations(layers, nullptr);
        std::vector<float*> d_hor_weights(layers, nullptr);
        std::vector<float*> d_hor_gweights(layers, nullptr);
        std::vector<float*> d_hor_deltas(layers, nullptr);
        std::vector<float*> d_ver_activations(layers, nullptr);
        std::vector<float*> d_ver_weights(layers, nullptr);
        std::vector<float*> d_ver_gweights(layers, nullptr);
        std::vector<float*> d_ver_deltas(layers, nullptr);
        // Temp flat vectors
        std::vector<std::vector<float>> flat_hor_weights(layers);
        std::vector<std::vector<float>> flat_ver_weights(layers);

        try {
            // --- Allocate Memory (Attention) ---
            CUDA_CHECK(cudaMalloc(&d_expected_h, embed_bytes));
            CUDA_CHECK(cudaMalloc(&d_EH, embed_bytes));
            CUDA_CHECK(cudaMalloc(&d_EV, ev_bytes)); // Allocate full EV size
            CUDA_CHECK(cudaMalloc(&d_grad_EH, embed_bytes));
            CUDA_CHECK(cudaMalloc(&d_grad_EV_scaled, embed_bytes));
            CUDA_CHECK(cudaMalloc(&d_grad_dh, embed_bytes));
            CUDA_CHECK(cudaMalloc(&d_grad_dv, embed_bytes));
            CUDA_CHECK(cudaMalloc(&d_KdotQ, head_bytes));
            CUDA_CHECK(cudaMalloc(&d_head, head_bytes));
            CUDA_CHECK(cudaMalloc(&d_K, k_q_bytes));
            CUDA_CHECK(cudaMalloc(&d_Q, k_q_bytes));
            CUDA_CHECK(cudaMalloc(&d_pre_MH, mat_heights * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_pre_MV, mat_heights * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_MH_a, mh_mv_mq_mk_bytes));
            CUDA_CHECK(cudaMalloc(&d_MV_a, mh_mv_mq_mk_bytes));
            CUDA_CHECK(cudaMalloc(&d_MQ_a, mh_mv_mq_mk_bytes));
            CUDA_CHECK(cudaMalloc(&d_MK_a, mh_mv_mq_mk_bytes));
            CUDA_CHECK(cudaMalloc(&d_grad_MH, mh_mv_mq_mk_bytes));
            CUDA_CHECK(cudaMalloc(&d_grad_MV, mh_mv_mq_mk_bytes));
            CUDA_CHECK(cudaMalloc(&d_grad_head, head_bytes));
            CUDA_CHECK(cudaMalloc(&d_lota_deriv, head_bytes));
            CUDA_CHECK(cudaMalloc(&d_grad_KdotQ, head_bytes));
            CUDA_CHECK(cudaMalloc(&d_grad_K, k_q_bytes));
            CUDA_CHECK(cudaMalloc(&d_grad_Q, k_q_bytes));
            CUDA_CHECK(cudaMalloc(&d_grad_MQ, mh_mv_mq_mk_bytes));
            CUDA_CHECK(cudaMalloc(&d_grad_MK, mh_mv_mq_mk_bytes));

            // --- Allocate Memory (MLP Internals) ---
            for (int l = 0; l < layers; ++l) {
                CUDA_CHECK(cudaMalloc(&d_hor_activations[l], embed_bytes));
                CUDA_CHECK(cudaMalloc(&d_hor_weights[l], weights_bytes));
                CUDA_CHECK(cudaMalloc(&d_hor_gweights[l], weights_bytes));
                CUDA_CHECK(cudaMalloc(&d_hor_deltas[l], embed_bytes));
                CUDA_CHECK(cudaMalloc(&d_ver_activations[l], embed_bytes));
                CUDA_CHECK(cudaMalloc(&d_ver_weights[l], weights_bytes));
                CUDA_CHECK(cudaMalloc(&d_ver_gweights[l], weights_bytes));
                CUDA_CHECK(cudaMalloc(&d_ver_deltas[l], embed_bytes));
            }

            // --- Data Transfer H->D (Attention) ---
            std::vector<float> flat_EV = flatten(head_obj.EV);
            std::vector<float> flat_K = flatten(head_obj.K);
            std::vector<float> flat_Q = flatten(head_obj.Q);
            std::vector<float> flat_KdotQ = flatten(head_obj.KdotQ);
            std::vector<float> flat_MH_a = flatten(head_obj.MH.a);
            std::vector<float> flat_MV_a = flatten(head_obj.MV.a);
            std::vector<float> flat_MQ_a = flatten(head_obj.MQ.a);
            std::vector<float> flat_MK_a = flatten(head_obj.MK.a);

            CUDA_CHECK(cudaMemcpy(d_expected_h, expectedH.data(), embed_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_EH, head_obj.EH.data(), embed_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_EV, flat_EV.data(), ev_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_KdotQ, flat_KdotQ.data(), head_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_K, flat_K.data(), k_q_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_Q, flat_Q.data(), k_q_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_MH_a, flat_MH_a.data(), mh_mv_mq_mk_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_MV_a, flat_MV_a.data(), mh_mv_mq_mk_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_MQ_a, flat_MQ_a.data(), mh_mv_mq_mk_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_MK_a, flat_MK_a.data(), mh_mv_mq_mk_bytes, cudaMemcpyHostToDevice));

            // --- Data Transfer H->D (MLP Internals) ---
            for (int l = 0; l < layers; ++l) {
                flat_hor_weights[l].reserve(embedding_dim * embedding_dim);
                flat_ver_weights[l].reserve(embedding_dim * embedding_dim);
                for (int r = 0; r < embedding_dim; ++r) {
                    flat_hor_weights[l].insert(flat_hor_weights[l].end(), head_obj.hor.weights[l][r].begin(), head_obj.hor.weights[l][r].end());
                    flat_ver_weights[l].insert(flat_ver_weights[l].end(), head_obj.ver.weights[l][r].begin(), head_obj.ver.weights[l][r].end());
                }
                CUDA_CHECK(cudaMemcpy(d_hor_activations[l], head_obj.hor.activations[l].data(), embed_bytes, cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_hor_weights[l], flat_hor_weights[l].data(), weights_bytes, cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_ver_activations[l], head_obj.ver.activations[l].data(), embed_bytes, cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_ver_weights[l], flat_ver_weights[l].data(), weights_bytes, cudaMemcpyHostToDevice));
            }

            // --- Backpropagation Steps ---

            // Step 1: Compute grad_EH and grad_EV_scaled
            kernelComputeGradientsEH_EV<<<gridDimEmbed, blockDim1D>>>(d_EH, d_expected_h, d_grad_EH, d_grad_EV_scaled, embedding_dim);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            // --- Step 2: Backprop through MLPs ---
            // --- 2a: Backprop through hor MLP ---
            kernelLastLayerDelta<<<gridDimEmbed, blockDim1D>>>(d_grad_EH, d_hor_activations[layers - 1], d_hor_deltas[layers - 1], embedding_dim);
            CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
            for (int l = layers - 2; l >= 0; --l) {
                hiddenDeltaKernel<<<gridDimEmbed, blockDim1D>>>(d_hor_deltas[l + 1], d_hor_weights[l + 1], d_hor_activations[l], d_hor_deltas[l], embedding_dim, embedding_dim);
                CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
            }
            for (int l = 0; l < layers; ++l) {
                float* d_prev_activations = (l == 0) ? d_EH : d_hor_activations[l - 1];
                updateWeightsKernel<<<gridDimEmbed2D, blockDim2D>>>(d_hor_deltas[l], d_prev_activations, d_hor_weights[l], d_hor_gweights[l], learning_rate, embedding_dim, embedding_dim);
                CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
            }
            // --- 2b: Backprop through ver MLP ---
            kernelLastLayerDelta<<<gridDimEmbed, blockDim1D>>>(d_grad_EV_scaled, d_ver_activations[layers - 1], d_ver_deltas[layers - 1], embedding_dim);
            CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
            for (int l = layers - 2; l >= 0; --l) {
                hiddenDeltaKernel<<<gridDimEmbed, blockDim1D>>>(d_ver_deltas[l + 1], d_ver_weights[l + 1], d_ver_activations[l], d_ver_deltas[l], embedding_dim, embedding_dim);
                CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
            }
            for (int l = 0; l < layers; ++l) {
                float* d_prev_activations = (l == 0) ? d_EV : d_ver_activations[l - 1]; // Assuming d_EV is correct input type/size
                updateWeightsKernel<<<gridDimEmbed2D, blockDim2D>>>(d_ver_deltas[l], d_prev_activations, d_ver_weights[l], d_ver_gweights[l], learning_rate, embedding_dim, embedding_dim);
                CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
            }

            // --- Step 3: Compute grad_dh and grad_dv ---
            kernelComputeGradMLPInput<<<gridDimEmbed, blockDim1D>>>(d_hor_deltas[0], d_hor_weights[0], d_grad_dh, embedding_dim, embedding_dim);
            CUDA_CHECK(cudaGetLastError());
            kernelComputeGradMLPInput<<<gridDimEmbed, blockDim1D>>>(d_ver_deltas[0], d_ver_weights[0], d_grad_dv, embedding_dim, embedding_dim);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            // --- Step 4: Compute grad_MH and grad_MV ---
            cuLOTA<<<gridDimHead2D, blockDim2D>>>(d_KdotQ, d_head, context_win, context_win, token_count, head_obj.isSelfAttention);
            CUDA_CHECK(cudaGetLastError());
            kernelComputePreMH_MV<<<gridDimMatHeights, blockDim1D>>>(d_head, d_K, d_Q, d_pre_MH, d_pre_MV, token_count, mat_heights);
            CUDA_CHECK(cudaGetLastError());
            kernelComputeGradMH_MV<<<gridDimMatrix2D, blockDim2D>>>(d_pre_MH, d_pre_MV, d_grad_dh, d_grad_dv, d_grad_MH, d_grad_MV, mat_heights, embedding_dim);
            CUDA_CHECK(cudaGetLastError());

            // --- Step 5: Compute grad_head ---
            kernelComputeGradHead<<<gridDimHead2D, blockDim2D>>>(d_K, d_Q, d_MH_a, d_MV_a, d_grad_dh, d_grad_dv, d_grad_head, token_count, mat_heights, embedding_dim);
            CUDA_CHECK(cudaGetLastError());

            // --- Step 6: Backprop through LOTA ---
            cuLOTAder<<<gridDimHead2D, blockDim2D>>>(d_KdotQ, d_lota_deriv, context_win, context_win, token_count, head_obj.isSelfAttention);
            CUDA_CHECK(cudaGetLastError());
            kernelComputeGradKdotQ_LOTA<<<gridDimHead, blockDim1D>>>(d_grad_head, d_lota_deriv, d_grad_KdotQ, scaling_factor, head_size);
            CUDA_CHECK(cudaGetLastError());

            // --- Step 7: Compute grad_K and grad_Q ---
            kernelComputeGradK_Q<<<gridDimKQGrad2D, blockDim2D>>>(d_grad_KdotQ, d_K, d_Q, d_grad_K, d_grad_Q, token_count, mat_heights);
            CUDA_CHECK(cudaGetLastError());

            // --- Step 8: Compute grad_MK and grad_MQ (Simplified) ---
            // Requires original token embeddings - assuming they are not directly available here.
            // Using nullptr for k_embed/q_embed, kernel handles null checks.
            kernelComputeGradMK_MQ_Simplified<<<gridDimMatrix2D, blockDim2D>>>(d_grad_K, d_grad_Q, nullptr, nullptr, d_grad_MK, d_grad_MQ, token_count, mat_heights, embedding_dim);
            CUDA_CHECK(cudaGetLastError());

            // --- Step 9 & 10: Update Weights ---
            if (is_first_head) {
                // Use kernelUpdateWeights_1stHead_H (updates MH, MV, MQ, MK, EH)
                kernelUpdateWeights_1stHead_H<<<gridDimMatrix, blockDim1D>>>(
                    d_MH_a, d_MV_a, d_MQ_a, d_MK_a, d_EH,
                    d_grad_MH, d_grad_MV, d_grad_MQ, d_grad_MK, d_grad_EH, // Use d_grad_EH for EH update
                    learning_rate, true, // update_eh = true
                    mat_heights, embedding_dim
                );
                CUDA_CHECK(cudaGetLastError());
            }
            else {
                // Use simpler update (updates MH, MV, MQ, MK, EH, EV based on d_grad_dh/dv)
                // Update MH, MV, MQ, MK
                kernelUpdateSimple<<<gridDimMatrix, blockDim1D>>>(d_MH_a, d_grad_MH, learning_rate, mh_mv_mq_mk_size);
                kernelUpdateSimple<<<gridDimMatrix, blockDim1D>>>(d_MV_a, d_grad_MV, learning_rate, mh_mv_mq_mk_size);
                kernelUpdateSimple<<<gridDimMatrix, blockDim1D>>>(d_MQ_a, d_grad_MQ, learning_rate, mh_mv_mq_mk_size);
                kernelUpdateSimple<<<gridDimMatrix, blockDim1D>>>(d_MK_a, d_grad_MK, learning_rate, mh_mv_mq_mk_size);
                CUDA_CHECK(cudaGetLastError());
                // Update EH (using d_grad_dh)
                kernelUpdateSimple<<<gridDimEmbed, blockDim1D>>>(d_EH, d_grad_dh, learning_rate, embedding_dim);
                CUDA_CHECK(cudaGetLastError());
                // Update EV (using d_grad_dv - CAUTION: Size/Logic check needed)
                // Assuming d_grad_dv applies to the relevant part or element-wise if sizes match.
                // This might need a more specific kernel depending on EV structure/update logic.
                // kernelUpdateSimple<<<gridDimEV, blockDim1D>>>(d_EV, d_grad_dv, learning_rate, ev_size); // Example if d_grad_dv applies to full EV
                kernelUpdateSimple<<<gridDimEmbed, blockDim1D>>>(d_EV, d_grad_dv, learning_rate, embedding_dim); // Example if d_grad_dv applies to first embed_dim elements
                CUDA_CHECK(cudaGetLastError());
            }
            CUDA_CHECK(cudaDeviceSynchronize());

            // --- Data Transfer D->H ---
            // Copy updated MLP weights and gradients back
            for (int l = 0; l < layers; ++l) {
                std::vector<float> updated_flat_weights(embedding_dim * embedding_dim);
                std::vector<float> calculated_flat_gradients(embedding_dim * embedding_dim);
                // Hor MLP
                CUDA_CHECK(cudaMemcpy(updated_flat_weights.data(), d_hor_weights[l], weights_bytes, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(calculated_flat_gradients.data(), d_hor_gweights[l], weights_bytes, cudaMemcpyDeviceToHost));
                for (int r = 0; r < embedding_dim; ++r) {
                    for (int c = 0; c < embedding_dim; ++c) {
                        head_obj.hor.weights[l][r][c] = updated_flat_weights[r * embedding_dim + c];
                        if(head_obj.hor.gweights.size() <= l) head_obj.hor.gweights.resize(l+1);
                        if(head_obj.hor.gweights[l].size() <= r) head_obj.hor.gweights[l].resize(r+1);
                        if(head_obj.hor.gweights[l][r].size() != embedding_dim) head_obj.hor.gweights[l][r].resize(embedding_dim);
                        head_obj.hor.gweights[l][r][c] = calculated_flat_gradients[r * embedding_dim + c];
                    }
                }
                // Ver MLP
                CUDA_CHECK(cudaMemcpy(updated_flat_weights.data(), d_ver_weights[l], weights_bytes, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(calculated_flat_gradients.data(), d_ver_gweights[l], weights_bytes, cudaMemcpyDeviceToHost));
                for (int r = 0; r < embedding_dim; ++r) {
                    for (int c = 0; c < embedding_dim; ++c) {
                        head_obj.ver.weights[l][r][c] = updated_flat_weights[r * embedding_dim + c];
                        if(head_obj.ver.gweights.size() <= l) head_obj.ver.gweights.resize(l+1);
                        if(head_obj.ver.gweights[l].size() <= r) head_obj.ver.gweights[l].resize(r+1);
                        if(head_obj.ver.gweights[l][r].size() != embedding_dim) head_obj.ver.gweights[l][r].resize(embedding_dim);
                        head_obj.ver.gweights[l][r][c] = calculated_flat_gradients[r * embedding_dim + c];
                    }
                }
            }
            // Copy updated Attention parameters back
            std::vector<float> updated_MH_a(mh_mv_mq_mk_size);
            std::vector<float> updated_MV_a(mh_mv_mq_mk_size);
            std::vector<float> updated_MQ_a(mh_mv_mq_mk_size);
            std::vector<float> updated_MK_a(mh_mv_mq_mk_size);
            std::vector<float> updated_EV(ev_size);

            CUDA_CHECK(cudaMemcpy(head_obj.EH.data(), d_EH, embed_bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(updated_EV.data(), d_EV, ev_bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(updated_MH_a.data(), d_MH_a, mh_mv_mq_mk_bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(updated_MV_a.data(), d_MV_a, mh_mv_mq_mk_bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(updated_MQ_a.data(), d_MQ_a, mh_mv_mq_mk_bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(updated_MK_a.data(), d_MK_a, mh_mv_mq_mk_bytes, cudaMemcpyDeviceToHost));

            // Unflatten
            unflatten(updated_EV, head_obj.EV, context_win, embedding_dim);
            unflatten(updated_MH_a, head_obj.MH.a, mat_heights, embedding_dim);
            unflatten(updated_MV_a, head_obj.MV.a, mat_heights, embedding_dim);
            unflatten(updated_MQ_a, head_obj.MQ.a, mat_heights, embedding_dim);
            unflatten(updated_MK_a, head_obj.MK.a, mat_heights, embedding_dim);

        }
        catch (const std::exception& e) {
            std::cerr << "Error during cu1ParallelBackward1stBlock (H) for head [" << i << "][" << layno << "]: " << e.what() << std::endl;
            // Cleanup allocated memory on error
            cudaFree(d_expected_h); cudaFree(d_EH); cudaFree(d_EV);
            cudaFree(d_grad_EH); cudaFree(d_grad_EV_scaled);
            cudaFree(d_grad_dh); cudaFree(d_grad_dv);
            cudaFree(d_KdotQ); cudaFree(d_head); cudaFree(d_K); cudaFree(d_Q);
            cudaFree(d_pre_MH); cudaFree(d_pre_MV);
            cudaFree(d_MH_a); cudaFree(d_MV_a); cudaFree(d_MQ_a); cudaFree(d_MK_a);
            cudaFree(d_grad_MH); cudaFree(d_grad_MV); cudaFree(d_grad_head);
            cudaFree(d_lota_deriv); cudaFree(d_grad_KdotQ);
            cudaFree(d_grad_K); cudaFree(d_grad_Q);
            cudaFree(d_grad_MQ); cudaFree(d_grad_MK);
            for (int l = 0; l < layers; ++l) {
                cudaFree(d_hor_activations[l]); cudaFree(d_hor_weights[l]); cudaFree(d_hor_gweights[l]); cudaFree(d_hor_deltas[l]);
                cudaFree(d_ver_activations[l]); cudaFree(d_ver_weights[l]); cudaFree(d_ver_gweights[l]); cudaFree(d_ver_deltas[l]);
            }
            throw std::runtime_error("Exception processing head [" + std::to_string(i) + "][" + std::to_string(layno) + "]: " + e.what());
        }

        // --- Cleanup Device Memory (Success Case - Per Head) ---
        cudaFree(d_expected_h); cudaFree(d_EH); cudaFree(d_EV);
        cudaFree(d_grad_EH); cudaFree(d_grad_EV_scaled);
        cudaFree(d_grad_dh); cudaFree(d_grad_dv);
        cudaFree(d_KdotQ); cudaFree(d_head); cudaFree(d_K); cudaFree(d_Q);
        cudaFree(d_pre_MH); cudaFree(d_pre_MV);
        cudaFree(d_MH_a); cudaFree(d_MV_a); cudaFree(d_MQ_a); cudaFree(d_MK_a);
        cudaFree(d_grad_MH); cudaFree(d_grad_MV); cudaFree(d_grad_head);
        cudaFree(d_lota_deriv); cudaFree(d_grad_KdotQ);
        cudaFree(d_grad_K); cudaFree(d_grad_Q);
        cudaFree(d_grad_MQ); cudaFree(d_grad_MK);
        for (int l = 0; l < layers; ++l) {
            cudaFree(d_hor_activations[l]); cudaFree(d_hor_weights[l]); cudaFree(d_hor_gweights[l]); cudaFree(d_hor_deltas[l]);
            cudaFree(d_ver_activations[l]); cudaFree(d_ver_weights[l]); cudaFree(d_ver_gweights[l]); cudaFree(d_ver_deltas[l]);
        }
    } // End loop over rows (i)
}


/**
 * @brief CUDA backward propagation for a single column in the FIRST block,
 *        driven by vertical error (EV). Processes heads b[row][layno].
 *        IMPLEMENTATION USES DIRECT KERNEL CALLS PER HEAD (INEFFICIENT).
 * @param expectedV Expected vertical embeddings for all heads in this column. Shape: [x][CONTEXT_WIN][EMBEDDING].
 * @param in Embedding dimension.
 * @param layers Number of MLP layers.
 * @param layno The column index within the block (0 to y-1).
 */
void block::cu1ParallelBackward1stBlock(std::vector<std::vector<std::vector<float>>>& expectedV, int& in, int& layers, int layno)
{
    // Validate column number and input shape
    if (layno < 0 || layno >= y) {
        throw std::out_of_range("cu1ParallelBackward1stBlock(V): Column index 'layno' (" + std::to_string(layno) + ") is out of range [0, " + std::to_string(y - 1) + "].");
    }
    if (expectedV.size() != static_cast<size_t>(x)) {
        throw std::runtime_error("ExpectedV size mismatch (number of rows) in cu1ParallelBackward1stBlock(V). Expected " + std::to_string(x) + ", got " + std::to_string(expectedV.size()));
    }
    if (!expectedV.empty() && (expectedV[0].size() != CONTEXT_WIN || (!expectedV[0].empty() && expectedV[0][0].size() != EMBEDDING))) {
        throw std::runtime_error("ExpectedV dimensions mismatch (context/embedding) in cu1ParallelBackward1stBlock(V).");
    }
    if (EMBEDDING != in) {
        throw std::runtime_error("Embedding dimension mismatch: EMBEDDING vs in");
    }

    // Constants
    const int embedding_dim = EMBEDDING;
    const int mat_heights = MATHEIGHTS;
    const int context_win = CONTEXT_WIN;
    const float learning_rate = LEARNING;
    const float scaling_factor = SCALING;
    const size_t embed_bytes = embedding_dim * sizeof(float);
    const size_t weights_bytes = embedding_dim * embedding_dim * sizeof(float); // MLP weights
    const int mh_mv_mq_mk_size = mat_heights * embedding_dim;
    const size_t mh_mv_mq_mk_bytes = mh_mv_mq_mk_size * sizeof(float);
    const int ev_size = context_win * embedding_dim;
    const size_t ev_bytes = ev_size * sizeof(float);

    // Kernel Launch Config
    int threadsPerBlock1D = 256;
    dim3 blockDim1D(threadsPerBlock1D);
    int blocksPerGridEmbed = (embedding_dim + threadsPerBlock1D - 1) / threadsPerBlock1D;
    dim3 gridDimEmbed(blocksPerGridEmbed);
    int blocksPerGridMatHeights = (mat_heights + threadsPerBlock1D - 1) / threadsPerBlock1D;
    dim3 gridDimMatHeights(blocksPerGridMatHeights);
    int blocksPerGridMatrix = (mh_mv_mq_mk_size + threadsPerBlock1D - 1) / threadsPerBlock1D;
    dim3 gridDimMatrix(blocksPerGridMatrix);
    int blocksPerGridEV = (ev_size + threadsPerBlock1D - 1) / threadsPerBlock1D;
    dim3 gridDimEV(blocksPerGridEV);

    dim3 blockDim2D(16, 16);
    dim3 gridDimEmbed2D((embedding_dim + blockDim2D.x - 1) / blockDim2D.x, (embedding_dim + blockDim2D.y - 1) / blockDim2D.y);
    dim3 gridDimMatrix2D((embedding_dim + blockDim2D.x - 1) / blockDim2D.x, (mat_heights + blockDim2D.y - 1) / blockDim2D.y);


    // Iterate through the rows (parallels/layers) in the specified column
    for (int i = 0; i < x; ++i) { // 'i' is the row index
        attention& head_obj = b[i][layno]; // Reference to the current head object
        std::vector<std::vector<float>>& expectedV_head = expectedV[i]; // Expected output for this head
        const int token_count = head_obj.tokenCount;
        const int head_size = token_count * token_count;
        const size_t head_bytes = head_size * sizeof(float);
        const int k_q_size = token_count * mat_heights;
        const size_t k_q_bytes = k_q_size * sizeof(float);
        // Adjust 2D grids based on token_count
        dim3 gridDimHead2D((token_count + blockDim2D.x - 1) / blockDim2D.x, (token_count + blockDim2D.y - 1) / blockDim2D.y);
        dim3 gridDimKQGrad2D((mat_heights + blockDim2D.x - 1) / blockDim2D.x, (token_count + blockDim2D.y - 1) / blockDim2D.y);
        int blocksPerGridHead = (head_size + threadsPerBlock1D - 1) / threadsPerBlock1D;
        dim3 gridDimHead(blocksPerGridHead);

        bool is_first_head = (i == 0 && layno == 0);

        // --- Device Pointers (Allocate per head - Inefficient) ---
        float *d_expected_v = nullptr, *d_EV = nullptr;
        float *d_grad_EV_full = nullptr, *d_grad_EV_summed = nullptr, *d_grad_EV_scaled = nullptr;
        float *d_grad_dv = nullptr;
        float *d_KdotQ = nullptr, *d_head = nullptr;
        float *d_K = nullptr, *d_Q = nullptr;
        float *d_pre_MV = nullptr;
        float *d_MV_a = nullptr, *d_MQ_a = nullptr, *d_MK_a = nullptr;
        float *d_grad_MV = nullptr;
        float *d_grad_head = nullptr;
        float *d_lota_deriv = nullptr;
        float *d_grad_KdotQ = nullptr;
        float *d_grad_Q = nullptr;
        float *d_grad_MQ = nullptr, *d_grad_MK_correction = nullptr;
        // MLP Internals (ver only)
        std::vector<float*> d_ver_activations(layers, nullptr);
        std::vector<float*> d_ver_weights(layers, nullptr);
        std::vector<float*> d_ver_gweights(layers, nullptr);
        std::vector<float*> d_ver_deltas(layers, nullptr);
        // Temp flat vectors
        std::vector<std::vector<float>> flat_ver_weights(layers);

        try {
            // --- Allocate Memory (Attention) ---
            CUDA_CHECK(cudaMalloc(&d_expected_v, ev_bytes));
            CUDA_CHECK(cudaMalloc(&d_EV, ev_bytes));
            CUDA_CHECK(cudaMalloc(&d_grad_EV_full, ev_bytes));
            CUDA_CHECK(cudaMalloc(&d_grad_EV_summed, embed_bytes));
            CUDA_CHECK(cudaMalloc(&d_grad_EV_scaled, embed_bytes));
            CUDA_CHECK(cudaMalloc(&d_grad_dv, embed_bytes));
            CUDA_CHECK(cudaMalloc(&d_KdotQ, head_bytes));
            CUDA_CHECK(cudaMalloc(&d_head, head_bytes));
            CUDA_CHECK(cudaMalloc(&d_K, k_q_bytes));
            CUDA_CHECK(cudaMalloc(&d_Q, k_q_bytes));
            CUDA_CHECK(cudaMalloc(&d_pre_MV, mat_heights * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_MV_a, mh_mv_mq_mk_bytes));
            CUDA_CHECK(cudaMalloc(&d_MQ_a, mh_mv_mq_mk_bytes));
            CUDA_CHECK(cudaMalloc(&d_MK_a, mh_mv_mq_mk_bytes));
            CUDA_CHECK(cudaMalloc(&d_grad_MV, mh_mv_mq_mk_bytes));
            CUDA_CHECK(cudaMalloc(&d_grad_head, head_bytes));
            CUDA_CHECK(cudaMalloc(&d_lota_deriv, head_bytes));
            CUDA_CHECK(cudaMalloc(&d_grad_KdotQ, head_bytes));
            CUDA_CHECK(cudaMalloc(&d_grad_Q, k_q_bytes));
            CUDA_CHECK(cudaMalloc(&d_grad_MQ, mh_mv_mq_mk_bytes));
            CUDA_CHECK(cudaMalloc(&d_grad_MK_correction, mh_mv_mq_mk_bytes));

            // --- Allocate Memory (ver MLP Internals) ---
            for (int l = 0; l < layers; ++l) {
                CUDA_CHECK(cudaMalloc(&d_ver_activations[l], embed_bytes));
                CUDA_CHECK(cudaMalloc(&d_ver_weights[l], weights_bytes));
                CUDA_CHECK(cudaMalloc(&d_ver_gweights[l], weights_bytes));
                CUDA_CHECK(cudaMalloc(&d_ver_deltas[l], embed_bytes));
            }

            // --- Data Transfer H->D (Attention) ---
            std::vector<float> flat_expectedV_head = flatten(expectedV_head);
            std::vector<float> flat_EV = flatten(head_obj.EV);
            std::vector<float> flat_K = flatten(head_obj.K);
            std::vector<float> flat_Q = flatten(head_obj.Q);
            std::vector<float> flat_KdotQ = flatten(head_obj.KdotQ);
            std::vector<float> flat_MV_a = flatten(head_obj.MV.a);
            std::vector<float> flat_MQ_a = flatten(head_obj.MQ.a);
            std::vector<float> flat_MK_a = flatten(head_obj.MK.a);

            CUDA_CHECK(cudaMemcpy(d_expected_v, flat_expectedV_head.data(), ev_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_EV, flat_EV.data(), ev_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_KdotQ, flat_KdotQ.data(), head_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_K, flat_K.data(), k_q_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_Q, flat_Q.data(), k_q_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_MV_a, flat_MV_a.data(), mh_mv_mq_mk_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_MQ_a, flat_MQ_a.data(), mh_mv_mq_mk_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_MK_a, flat_MK_a.data(), mh_mv_mq_mk_bytes, cudaMemcpyHostToDevice));

            // --- Data Transfer H->D (ver MLP Internals) ---
            for (int l = 0; l < layers; ++l) {
                flat_ver_weights[l].reserve(embedding_dim * embedding_dim);
                for (int r = 0; r < embedding_dim; ++r) {
                    flat_ver_weights[l].insert(flat_ver_weights[l].end(), head_obj.ver.weights[l][r].begin(), head_obj.ver.weights[l][r].end());
                }
                CUDA_CHECK(cudaMemcpy(d_ver_activations[l], head_obj.ver.activations[l].data(), embed_bytes, cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_ver_weights[l], flat_ver_weights[l].data(), weights_bytes, cudaMemcpyHostToDevice));
            }

            // --- Backpropagation Steps ---

            // Step 1: Compute grad_EV (full, summed, scaled)
            kernelComputeGradientsEV_V<<<gridDimEmbed, blockDim1D>>>(d_EV, d_expected_v, d_grad_EV_full, d_grad_EV_summed, d_grad_EV_scaled, learning_rate, context_win, embedding_dim);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            // --- Step 2: Backprop through ver MLP ---
            kernelLastLayerDelta<<<gridDimEmbed, blockDim1D>>>(d_grad_EV_scaled, d_ver_activations[layers - 1], d_ver_deltas[layers - 1], embedding_dim);
            CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
            for (int l = layers - 2; l >= 0; --l) {
                hiddenDeltaKernel<<<gridDimEmbed, blockDim1D>>>(d_ver_deltas[l + 1], d_ver_weights[l + 1], d_ver_activations[l], d_ver_deltas[l], embedding_dim, embedding_dim);
                CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
            }
            for (int l = 0; l < layers; ++l) {
                float* d_prev_activations = (l == 0) ? d_EV : d_ver_activations[l - 1]; // Assuming d_EV is correct input type/size
                updateWeightsKernel<<<gridDimEmbed2D, blockDim2D>>>(d_ver_deltas[l], d_prev_activations, d_ver_weights[l], d_ver_gweights[l], learning_rate, embedding_dim, embedding_dim);
                CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
            }

            // --- Step 3: Compute grad_dv ---
            kernelComputeGradMLPInput<<<gridDimEmbed, blockDim1D>>>(d_ver_deltas[0], d_ver_weights[0], d_grad_dv, embedding_dim, embedding_dim);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            // --- Step 4: Compute grad_MV ---
            cuLOTA<<<gridDimHead2D, blockDim2D>>>(d_KdotQ, d_head, context_win, context_win, token_count, head_obj.isSelfAttention);
            CUDA_CHECK(cudaGetLastError());
            kernelComputePreMV_V<<<gridDimMatHeights, blockDim1D>>>(d_head, d_Q, d_pre_MV, token_count, mat_heights);
            CUDA_CHECK(cudaGetLastError());
            kernelComputeGradMV_V<<<gridDimMatrix2D, blockDim2D>>>(d_pre_MV, d_grad_dv, d_grad_MV, mat_heights, embedding_dim);
            CUDA_CHECK(cudaGetLastError());

            // --- Step 5: Compute grad_head (V path only) ---
            kernelComputeGradHead_V<<<gridDimHead2D, blockDim2D>>>(d_Q, d_MV_a, d_grad_dv, d_grad_head, token_count, mat_heights, embedding_dim);
            CUDA_CHECK(cudaGetLastError());

            // --- Step 6: Backprop through LOTA ---
            cuLOTAder<<<gridDimHead2D, blockDim2D>>>(d_KdotQ, d_lota_deriv, context_win, context_win, token_count, head_obj.isSelfAttention);
            CUDA_CHECK(cudaGetLastError());
            kernelComputeGradKdotQ_LOTA<<<gridDimHead, blockDim1D>>>(d_grad_head, d_lota_deriv, d_grad_KdotQ, scaling_factor, head_size);
            CUDA_CHECK(cudaGetLastError());

            // --- Step 7: Compute grad_Q ---
            kernelComputeGradQ_V<<<gridDimKQGrad2D, blockDim2D>>>(d_grad_KdotQ, d_K, d_grad_Q, token_count, mat_heights);
            CUDA_CHECK(cudaGetLastError());

            // --- Step 8: Compute grad_MQ and grad_MK_correction ---
            // Requires original token embeddings - assuming they are not directly available here.
            // Using nullptr for q_embed.
            kernelComputeGradMQ_V<<<gridDimMatrix2D, blockDim2D>>>(d_grad_Q, nullptr, d_grad_MQ, token_count, mat_heights, embedding_dim);
            CUDA_CHECK(cudaGetLastError());
            kernelComputeGradMKCorrection<<<gridDimMatrix2D, blockDim2D>>>(d_grad_MQ, d_Q, d_K, d_grad_MK_correction, token_count, mat_heights, embedding_dim);
            CUDA_CHECK(cudaGetLastError());

            // --- Step 9 & 10: Update Weights ---
            if (is_first_head) {
                // Use kernelUpdateWeights_1stHead_V (updates MV, MQ, MK_correction)
                kernelUpdateWeights_1stHead_V<<<gridDimMatrix, blockDim1D>>>(
                    d_MV_a, d_MQ_a, d_MK_a,
                    d_grad_MV, d_grad_MQ, d_grad_MK_correction,
                    learning_rate, mat_heights, embedding_dim
                );
                CUDA_CHECK(cudaGetLastError());
                // No EV update in kernelUpdateWeights_1stHead_V
            }
            else {
                // Use simpler update (updates MV, MQ, MK_correction, EV)
                kernelUpdateSimple<<<gridDimMatrix, blockDim1D>>>(d_MV_a, d_grad_MV, learning_rate, mh_mv_mq_mk_size);
                kernelUpdateSimple<<<gridDimMatrix, blockDim1D>>>(d_MQ_a, d_grad_MQ, learning_rate, mh_mv_mq_mk_size);
                kernelUpdateSimple<<<gridDimMatrix, blockDim1D>>>(d_MK_a, d_grad_MK_correction, learning_rate, mh_mv_mq_mk_size); // Update MK with correction
                CUDA_CHECK(cudaGetLastError());
                // Update EV using full gradient
                kernelUpdateSimple<<<gridDimEV, blockDim1D>>>(d_EV, d_grad_EV_full, learning_rate, ev_size);
                CUDA_CHECK(cudaGetLastError());
            }
            CUDA_CHECK(cudaDeviceSynchronize());

            // --- Data Transfer D->H ---
            // Copy updated ver MLP weights and gradients back
            for (int l = 0; l < layers; ++l) {
                std::vector<float> updated_flat_weights(embedding_dim * embedding_dim);
                std::vector<float> calculated_flat_gradients(embedding_dim * embedding_dim);
                CUDA_CHECK(cudaMemcpy(updated_flat_weights.data(), d_ver_weights[l], weights_bytes, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(calculated_flat_gradients.data(), d_ver_gweights[l], weights_bytes, cudaMemcpyDeviceToHost));
                for (int r = 0; r < embedding_dim; ++r) {
                    for (int c = 0; c < embedding_dim; ++c) {
                        head_obj.ver.weights[l][r][c] = updated_flat_weights[r * embedding_dim + c];
                        if(head_obj.ver.gweights.size() <= l) head_obj.ver.gweights.resize(l+1);
                        if(head_obj.ver.gweights[l].size() <= r) head_obj.ver.gweights[l].resize(r+1);
                        if(head_obj.ver.gweights[l][r].size() != embedding_dim) head_obj.ver.gweights[l][r].resize(embedding_dim);
                        head_obj.ver.gweights[l][r][c] = calculated_flat_gradients[r * embedding_dim + c];
                    }
                }
            }
            // Copy updated Attention parameters back
            std::vector<float> updated_MV_a(mh_mv_mq_mk_size);
            std::vector<float> updated_MQ_a(mh_mv_mq_mk_size);
            std::vector<float> updated_MK_a(mh_mv_mq_mk_size);
            std::vector<float> updated_EV(ev_size);

            CUDA_CHECK(cudaMemcpy(updated_EV.data(), d_EV, ev_bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(updated_MV_a.data(), d_MV_a, mh_mv_mq_mk_bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(updated_MQ_a.data(), d_MQ_a, mh_mv_mq_mk_bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(updated_MK_a.data(), d_MK_a, mh_mv_mq_mk_bytes, cudaMemcpyDeviceToHost));

            // Unflatten
            unflatten(updated_EV, head_obj.EV, context_win, embedding_dim);
            unflatten(updated_MV_a, head_obj.MV.a, mat_heights, embedding_dim);
            unflatten(updated_MQ_a, head_obj.MQ.a, mat_heights, embedding_dim);
            unflatten(updated_MK_a, head_obj.MK.a, mat_heights, embedding_dim);

        }
        catch (const std::exception& e) {
            std::cerr << "Error during cu1ParallelBackward1stBlock (V) for head [" << i << "][" << layno << "]: " << e.what() << std::endl;
            // Cleanup allocated memory on error
            cudaFree(d_expected_v); cudaFree(d_EV);
            cudaFree(d_grad_EV_full); cudaFree(d_grad_EV_summed); cudaFree(d_grad_EV_scaled);
            cudaFree(d_grad_dv);
            cudaFree(d_KdotQ); cudaFree(d_head); cudaFree(d_K); cudaFree(d_Q);
            cudaFree(d_pre_MV);
            cudaFree(d_MV_a); cudaFree(d_MQ_a); cudaFree(d_MK_a);
            cudaFree(d_grad_MV); cudaFree(d_grad_head);
            cudaFree(d_lota_deriv); cudaFree(d_grad_KdotQ);
            cudaFree(d_grad_Q); cudaFree(d_grad_MQ); cudaFree(d_grad_MK_correction);
            for (int l = 0; l < layers; ++l) {
                cudaFree(d_ver_activations[l]); cudaFree(d_ver_weights[l]); cudaFree(d_ver_gweights[l]); cudaFree(d_ver_deltas[l]);
            }
            throw std::runtime_error("Exception processing head [" + std::to_string(i) + "][" + std::to_string(layno) + "]: " + e.what());
        }

        // --- Cleanup Device Memory (Success Case - Per Head) ---
        cudaFree(d_expected_v); cudaFree(d_EV);
        cudaFree(d_grad_EV_full); cudaFree(d_grad_EV_summed); cudaFree(d_grad_EV_scaled);
        cudaFree(d_grad_dv);
        cudaFree(d_KdotQ); cudaFree(d_head); cudaFree(d_K); cudaFree(d_Q);
        cudaFree(d_pre_MV);
        cudaFree(d_MV_a); cudaFree(d_MQ_a); cudaFree(d_MK_a);
        cudaFree(d_grad_MV); cudaFree(d_grad_head);
        cudaFree(d_lota_deriv); cudaFree(d_grad_KdotQ);
        cudaFree(d_grad_Q); cudaFree(d_grad_MQ); cudaFree(d_grad_MK_correction);
        for (int l = 0; l < layers; ++l) {
            cudaFree(d_ver_activations[l]); cudaFree(d_ver_weights[l]); cudaFree(d_ver_gweights[l]); cudaFree(d_ver_deltas[l]);
        }
    } // End loop over rows (i)
}


/**
 * @brief CUDA backward propagation for a single column in a NON-FIRST block,
 *        driven by horizontal error (EH). Processes heads b[row][layno].
 *        IMPLEMENTATION USES DIRECT KERNEL CALLS PER HEAD (INEFFICIENT).
 * @param expectedH Expected horizontal embedding for the output of this column's heads.
 * @param in Embedding dimension.
 * @param layers Number of MLP layers.
 * @param layno The column index within the block (0 to y-1).
 */
void block::cu1ParallelBackward(std::vector<float>& expectedH, int& in, int& layers, int layno)
{
    // This function's logic is identical to cu1ParallelBackward1stBlock(expectedH, ...)
    // EXCEPT that it never calls the 1stHead update kernel. It always uses the general update path.
    // We can reuse the code structure.

    // Validate column number
    if (layno < 0 || layno >= y) {
        throw std::out_of_range("cu1ParallelBackward(H): Column index 'layno' (" + std::to_string(layno) + ") is out of range [0, " + std::to_string(y - 1) + "].");
    }
    if (expectedH.size() != EMBEDDING) {
        throw std::runtime_error("ExpectedH vector size mismatch in cu1ParallelBackward(H). Expected " + std::to_string(EMBEDDING) + ", got " + std::to_string(expectedH.size()));
    }
    if (EMBEDDING != in) {
        throw std::runtime_error("Embedding dimension mismatch: EMBEDDING vs in");
    }

    // Constants
    const int embedding_dim = EMBEDDING;
    const int mat_heights = MATHEIGHTS;
    const int context_win = CONTEXT_WIN;
    const float learning_rate = LEARNING;
    const float scaling_factor = SCALING;
    const size_t embed_bytes = embedding_dim * sizeof(float);
    const size_t weights_bytes = embedding_dim * embedding_dim * sizeof(float);
    const int mh_mv_mq_mk_size = mat_heights * embedding_dim;
    const size_t mh_mv_mq_mk_bytes = mh_mv_mq_mk_size * sizeof(float);
    const int ev_size = context_win * embedding_dim;
    const size_t ev_bytes = ev_size * sizeof(float);

    // Kernel Launch Config
    int threadsPerBlock1D = 256;
    dim3 blockDim1D(threadsPerBlock1D);
    int blocksPerGridEmbed = (embedding_dim + threadsPerBlock1D - 1) / threadsPerBlock1D;
    dim3 gridDimEmbed(blocksPerGridEmbed);
    int blocksPerGridMatHeights = (mat_heights + threadsPerBlock1D - 1) / threadsPerBlock1D;
    dim3 gridDimMatHeights(blocksPerGridMatHeights);
    int blocksPerGridMatrix = (mh_mv_mq_mk_size + threadsPerBlock1D - 1) / threadsPerBlock1D;
    dim3 gridDimMatrix(blocksPerGridMatrix);
    int blocksPerGridEV = (ev_size + threadsPerBlock1D - 1) / threadsPerBlock1D; // For potential EV update
    dim3 gridDimEV(blocksPerGridEV);

    dim3 blockDim2D(16, 16);
    dim3 gridDimEmbed2D((embedding_dim + blockDim2D.x - 1) / blockDim2D.x, (embedding_dim + blockDim2D.y - 1) / blockDim2D.y);
    dim3 gridDimMatrix2D((embedding_dim + blockDim2D.x - 1) / blockDim2D.x, (mat_heights + blockDim2D.y - 1) / blockDim2D.y);

    // Iterate backwards through the rows
    for (int i = x - 1; i >= 0; --i) {
        attention& head_obj = b[i][layno];
        const int token_count = head_obj.tokenCount;
        const int head_size = token_count * token_count;
        const size_t head_bytes = head_size * sizeof(float);
        const int k_q_size = token_count * mat_heights;
        const size_t k_q_bytes = k_q_size * sizeof(float);
        dim3 gridDimHead2D((token_count + blockDim2D.x - 1) / blockDim2D.x, (token_count + blockDim2D.y - 1) / blockDim2D.y);
        dim3 gridDimKQGrad2D((mat_heights + blockDim2D.x - 1) / blockDim2D.x, (token_count + blockDim2D.y - 1) / blockDim2D.y);
        int blocksPerGridHead = (head_size + threadsPerBlock1D - 1) / threadsPerBlock1D;
        dim3 gridDimHead(blocksPerGridHead);

        // --- Device Pointers (Allocate per head) ---
        // (Same list as in cu1ParallelBackward1stBlock(H))
        float *d_expected_h = nullptr, *d_EH = nullptr, *d_EV = nullptr;
        float *d_grad_EH = nullptr, *d_grad_EV_scaled = nullptr;
        float *d_grad_dh = nullptr, *d_grad_dv = nullptr;
        float *d_KdotQ = nullptr, *d_head = nullptr;
        float *d_K = nullptr, *d_Q = nullptr;
        float *d_pre_MH = nullptr, *d_pre_MV = nullptr;
        float *d_MH_a = nullptr, *d_MV_a = nullptr, *d_MQ_a = nullptr, *d_MK_a = nullptr;
        float *d_grad_MH = nullptr, *d_grad_MV = nullptr;
        float *d_grad_head = nullptr;
        float *d_lota_deriv = nullptr;
        float *d_grad_KdotQ = nullptr;
        float *d_grad_K = nullptr, *d_grad_Q = nullptr;
        float *d_grad_MQ = nullptr, *d_grad_MK = nullptr;
        std::vector<float*> d_hor_activations(layers, nullptr);
        std::vector<float*> d_hor_weights(layers, nullptr);
        std::vector<float*> d_hor_gweights(layers, nullptr);
        std::vector<float*> d_hor_deltas(layers, nullptr);
        std::vector<float*> d_ver_activations(layers, nullptr);
        std::vector<float*> d_ver_weights(layers, nullptr);
        std::vector<float*> d_ver_gweights(layers, nullptr);
        std::vector<float*> d_ver_deltas(layers, nullptr);
        std::vector<std::vector<float>> flat_hor_weights(layers);
        std::vector<std::vector<float>> flat_ver_weights(layers);

        try {
            // --- Allocate Memory ---
            // (Identical allocations as in cu1ParallelBackward1stBlock(H))
            CUDA_CHECK(cudaMalloc(&d_expected_h, embed_bytes));
            CUDA_CHECK(cudaMalloc(&d_EH, embed_bytes));
            CUDA_CHECK(cudaMalloc(&d_EV, ev_bytes));
            CUDA_CHECK(cudaMalloc(&d_grad_EH, embed_bytes));
            CUDA_CHECK(cudaMalloc(&d_grad_EV_scaled, embed_bytes));
            CUDA_CHECK(cudaMalloc(&d_grad_dh, embed_bytes));
            CUDA_CHECK(cudaMalloc(&d_grad_dv, embed_bytes));
            CUDA_CHECK(cudaMalloc(&d_KdotQ, head_bytes));
            CUDA_CHECK(cudaMalloc(&d_head, head_bytes));
            CUDA_CHECK(cudaMalloc(&d_K, k_q_bytes));
            CUDA_CHECK(cudaMalloc(&d_Q, k_q_bytes));
            CUDA_CHECK(cudaMalloc(&d_pre_MH, mat_heights * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_pre_MV, mat_heights * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_MH_a, mh_mv_mq_mk_bytes));
            CUDA_CHECK(cudaMalloc(&d_MV_a, mh_mv_mq_mk_bytes));
            CUDA_CHECK(cudaMalloc(&d_MQ_a, mh_mv_mq_mk_bytes));
            CUDA_CHECK(cudaMalloc(&d_MK_a, mh_mv_mq_mk_bytes));
            CUDA_CHECK(cudaMalloc(&d_grad_MH, mh_mv_mq_mk_bytes));
            CUDA_CHECK(cudaMalloc(&d_grad_MV, mh_mv_mq_mk_bytes));
            CUDA_CHECK(cudaMalloc(&d_grad_head, head_bytes));
            CUDA_CHECK(cudaMalloc(&d_lota_deriv, head_bytes));
            CUDA_CHECK(cudaMalloc(&d_grad_KdotQ, head_bytes));
            CUDA_CHECK(cudaMalloc(&d_grad_K, k_q_bytes));
            CUDA_CHECK(cudaMalloc(&d_grad_Q, k_q_bytes));
            CUDA_CHECK(cudaMalloc(&d_grad_MQ, mh_mv_mq_mk_bytes));
            CUDA_CHECK(cudaMalloc(&d_grad_MK, mh_mv_mq_mk_bytes));
            for (int l = 0; l < layers; ++l) {
                CUDA_CHECK(cudaMalloc(&d_hor_activations[l], embed_bytes));
                CUDA_CHECK(cudaMalloc(&d_hor_weights[l], weights_bytes));
                CUDA_CHECK(cudaMalloc(&d_hor_gweights[l], weights_bytes));
                CUDA_CHECK(cudaMalloc(&d_hor_deltas[l], embed_bytes));
                CUDA_CHECK(cudaMalloc(&d_ver_activations[l], embed_bytes));
                CUDA_CHECK(cudaMalloc(&d_ver_weights[l], weights_bytes));
                CUDA_CHECK(cudaMalloc(&d_ver_gweights[l], weights_bytes));
                CUDA_CHECK(cudaMalloc(&d_ver_deltas[l], embed_bytes));
            }

            // --- Data Transfer H->D ---
            // (Identical transfers as in cu1ParallelBackward1stBlock(H))
            std::vector<float> flat_EV = flatten(head_obj.EV);
            std::vector<float> flat_K = flatten(head_obj.K);
            std::vector<float> flat_Q = flatten(head_obj.Q);
            std::vector<float> flat_KdotQ = flatten(head_obj.KdotQ);
            std::vector<float> flat_MH_a = flatten(head_obj.MH.a);
            std::vector<float> flat_MV_a = flatten(head_obj.MV.a);
            std::vector<float> flat_MQ_a = flatten(head_obj.MQ.a);
            std::vector<float> flat_MK_a = flatten(head_obj.MK.a);
            CUDA_CHECK(cudaMemcpy(d_expected_h, expectedH.data(), embed_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_EH, head_obj.EH.data(), embed_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_EV, flat_EV.data(), ev_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_KdotQ, flat_KdotQ.data(), head_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_K, flat_K.data(), k_q_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_Q, flat_Q.data(), k_q_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_MH_a, flat_MH_a.data(), mh_mv_mq_mk_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_MV_a, flat_MV_a.data(), mh_mv_mq_mk_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_MQ_a, flat_MQ_a.data(), mh_mv_mq_mk_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_MK_a, flat_MK_a.data(), mh_mv_mq_mk_bytes, cudaMemcpyHostToDevice));
            for (int l = 0; l < layers; ++l) {
                flat_hor_weights[l].reserve(embedding_dim * embedding_dim);
                flat_ver_weights[l].reserve(embedding_dim * embedding_dim);
                for (int r = 0; r < embedding_dim; ++r) {
                    flat_hor_weights[l].insert(flat_hor_weights[l].end(), head_obj.hor.weights[l][r].begin(), head_obj.hor.weights[l][r].end());
                    flat_ver_weights[l].insert(flat_ver_weights[l].end(), head_obj.ver.weights[l][r].begin(), head_obj.ver.weights[l][r].end());
                }
                CUDA_CHECK(cudaMemcpy(d_hor_activations[l], head_obj.hor.activations[l].data(), embed_bytes, cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_hor_weights[l], flat_hor_weights[l].data(), weights_bytes, cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_ver_activations[l], head_obj.ver.activations[l].data(), embed_bytes, cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_ver_weights[l], flat_ver_weights[l].data(), weights_bytes, cudaMemcpyHostToDevice));
            }

            // --- Backpropagation Steps ---
            // (Identical steps 1-8 as in cu1ParallelBackward1stBlock(H))
            // Step 1: Compute grad_EH and grad_EV_scaled
            kernelComputeGradientsEH_EV<<<gridDimEmbed, blockDim1D>>>(d_EH, d_expected_h, d_grad_EH, d_grad_EV_scaled, embedding_dim);
            CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
            // Step 2: Backprop through MLPs
            // 2a: Hor MLP
            kernelLastLayerDelta<<<gridDimEmbed, blockDim1D>>>(d_grad_EH, d_hor_activations[layers - 1], d_hor_deltas[layers - 1], embedding_dim);
            CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
            for (int l = layers - 2; l >= 0; --l) { hiddenDeltaKernel<<<gridDimEmbed, blockDim1D>>>(d_hor_deltas[l + 1], d_hor_weights[l + 1], d_hor_activations[l], d_hor_deltas[l], embedding_dim, embedding_dim); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize()); }
            for (int l = 0; l < layers; ++l) { float* d_prev = (l == 0) ? d_EH : d_hor_activations[l - 1]; updateWeightsKernel<<<gridDimEmbed2D, blockDim2D>>>(d_hor_deltas[l], d_prev, d_hor_weights[l], d_hor_gweights[l], learning_rate, embedding_dim, embedding_dim); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize()); }
            // 2b: Ver MLP
            kernelLastLayerDelta<<<gridDimEmbed, blockDim1D>>>(d_grad_EV_scaled, d_ver_activations[layers - 1], d_ver_deltas[layers - 1], embedding_dim);
            CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
            for (int l = layers - 2; l >= 0; --l) { hiddenDeltaKernel<<<gridDimEmbed, blockDim1D>>>(d_ver_deltas[l + 1], d_ver_weights[l + 1], d_ver_activations[l], d_ver_deltas[l], embedding_dim, embedding_dim); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize()); }
            for (int l = 0; l < layers; ++l) { float* d_prev = (l == 0) ? d_EV : d_ver_activations[l - 1]; updateWeightsKernel<<<gridDimEmbed2D, blockDim2D>>>(d_ver_deltas[l], d_prev, d_ver_weights[l], d_ver_gweights[l], learning_rate, embedding_dim, embedding_dim); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize()); }
            // Step 3: Compute grad_dh and grad_dv
            kernelComputeGradMLPInput<<<gridDimEmbed, blockDim1D>>>(d_hor_deltas[0], d_hor_weights[0], d_grad_dh, embedding_dim, embedding_dim); CUDA_CHECK(cudaGetLastError());
            kernelComputeGradMLPInput<<<gridDimEmbed, blockDim1D>>>(d_ver_deltas[0], d_ver_weights[0], d_grad_dv, embedding_dim, embedding_dim); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
            // Step 4: Compute grad_MH and grad_MV
            cuLOTA<<<gridDimHead2D, blockDim2D>>>(d_KdotQ, d_head, context_win, context_win, token_count, head_obj.isSelfAttention); CUDA_CHECK(cudaGetLastError());
            kernelComputePreMH_MV<<<gridDimMatHeights, blockDim1D>>>(d_head, d_K, d_Q, d_pre_MH, d_pre_MV, token_count, mat_heights); CUDA_CHECK(cudaGetLastError());
            kernelComputeGradMH_MV<<<gridDimMatrix2D, blockDim2D>>>(d_pre_MH, d_pre_MV, d_grad_dh, d_grad_dv, d_grad_MH, d_grad_MV, mat_heights, embedding_dim); CUDA_CHECK(cudaGetLastError());
            // Step 5: Compute grad_head
            kernelComputeGradHead<<<gridDimHead2D, blockDim2D>>>(d_K, d_Q, d_MH_a, d_MV_a, d_grad_dh, d_grad_dv, d_grad_head, token_count, mat_heights, embedding_dim); CUDA_CHECK(cudaGetLastError());
            // Step 6: Backprop through LOTA
            cuLOTAder<<<gridDimHead2D, blockDim2D>>>(d_KdotQ, d_lota_deriv, context_win, context_win, token_count, head_obj.isSelfAttention); CUDA_CHECK(cudaGetLastError());
            kernelComputeGradKdotQ_LOTA<<<gridDimHead, blockDim1D>>>(d_grad_head, d_lota_deriv, d_grad_KdotQ, scaling_factor, head_size); CUDA_CHECK(cudaGetLastError());
            // Step 7: Compute grad_K and grad_Q
            kernelComputeGradK_Q<<<gridDimKQGrad2D, blockDim2D>>>(d_grad_KdotQ, d_K, d_Q, d_grad_K, d_grad_Q, token_count, mat_heights); CUDA_CHECK(cudaGetLastError());
            // Step 8: Compute grad_MK and grad_MQ (Simplified)
            kernelComputeGradMK_MQ_Simplified<<<gridDimMatrix2D, blockDim2D>>>(d_grad_K, d_grad_Q, nullptr, nullptr, d_grad_MK, d_grad_MQ, token_count, mat_heights, embedding_dim); CUDA_CHECK(cudaGetLastError());

            // --- Step 9 & 10: Update Weights (General Case) ---
            // Always use the general update path, never the 1stHead specific kernel.
            kernelUpdateSimple<<<gridDimMatrix, blockDim1D>>>(d_MH_a, d_grad_MH, learning_rate, mh_mv_mq_mk_size);
            kernelUpdateSimple<<<gridDimMatrix, blockDim1D>>>(d_MV_a, d_grad_MV, learning_rate, mh_mv_mq_mk_size);
            kernelUpdateSimple<<<gridDimMatrix, blockDim1D>>>(d_MQ_a, d_grad_MQ, learning_rate, mh_mv_mq_mk_size);
            kernelUpdateSimple<<<gridDimMatrix, blockDim1D>>>(d_MK_a, d_grad_MK, learning_rate, mh_mv_mq_mk_size);
            CUDA_CHECK(cudaGetLastError());
            kernelUpdateSimple<<<gridDimEmbed, blockDim1D>>>(d_EH, d_grad_dh, learning_rate, embedding_dim);
            CUDA_CHECK(cudaGetLastError());
            // kernelUpdateSimple<<<gridDimEV, blockDim1D>>>(d_EV, d_grad_dv, learning_rate, ev_size); // Example EV update
            kernelUpdateSimple<<<gridDimEmbed, blockDim1D>>>(d_EV, d_grad_dv, learning_rate, embedding_dim); // Example EV update (partial)
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            // --- Data Transfer D->H ---
            // (Identical transfers as in cu1ParallelBackward1stBlock(H))
            for (int l = 0; l < layers; ++l) {
                std::vector<float> updated_flat_weights(embedding_dim * embedding_dim);
                std::vector<float> calculated_flat_gradients(embedding_dim * embedding_dim);
                CUDA_CHECK(cudaMemcpy(updated_flat_weights.data(), d_hor_weights[l], weights_bytes, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(calculated_flat_gradients.data(), d_hor_gweights[l], weights_bytes, cudaMemcpyDeviceToHost));
                for (int r = 0; r < embedding_dim; ++r) { for (int c = 0; c < embedding_dim; ++c) { head_obj.hor.weights[l][r][c] = updated_flat_weights[r*embedding_dim+c]; if(head_obj.hor.gweights.size()<=l)head_obj.hor.gweights.resize(l+1); if(head_obj.hor.gweights[l].size()<=r)head_obj.hor.gweights[l].resize(r+1); if(head_obj.hor.gweights[l][r].size()!=embedding_dim)head_obj.hor.gweights[l][r].resize(embedding_dim); head_obj.hor.gweights[l][r][c] = calculated_flat_gradients[r*embedding_dim+c]; } }
                CUDA_CHECK(cudaMemcpy(updated_flat_weights.data(), d_ver_weights[l], weights_bytes, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(calculated_flat_gradients.data(), d_ver_gweights[l], weights_bytes, cudaMemcpyDeviceToHost));
                for (int r = 0; r < embedding_dim; ++r) { for (int c = 0; c < embedding_dim; ++c) { head_obj.ver.weights[l][r][c] = updated_flat_weights[r*embedding_dim+c]; if(head_obj.ver.gweights.size()<=l)head_obj.ver.gweights.resize(l+1); if(head_obj.ver.gweights[l].size()<=r)head_obj.ver.gweights[l].resize(r+1); if(head_obj.ver.gweights[l][r].size()!=embedding_dim)head_obj.ver.gweights[l][r].resize(embedding_dim); head_obj.ver.gweights[l][r][c] = calculated_flat_gradients[r*embedding_dim+c]; } }
            }
            std::vector<float> updated_MH_a(mh_mv_mq_mk_size), updated_MV_a(mh_mv_mq_mk_size), updated_MQ_a(mh_mv_mq_mk_size), updated_MK_a(mh_mv_mq_mk_size), updated_EV(ev_size);
            CUDA_CHECK(cudaMemcpy(head_obj.EH.data(), d_EH, embed_bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(updated_EV.data(), d_EV, ev_bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(updated_MH_a.data(), d_MH_a, mh_mv_mq_mk_bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(updated_MV_a.data(), d_MV_a, mh_mv_mq_mk_bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(updated_MQ_a.data(), d_MQ_a, mh_mv_mq_mk_bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(updated_MK_a.data(), d_MK_a, mh_mv_mq_mk_bytes, cudaMemcpyDeviceToHost));
            unflatten(updated_EV, head_obj.EV, context_win, embedding_dim);
            unflatten(updated_MH_a, head_obj.MH.a, mat_heights, embedding_dim);
            unflatten(updated_MV_a, head_obj.MV.a, mat_heights, embedding_dim);
            unflatten(updated_MQ_a, head_obj.MQ.a, mat_heights, embedding_dim);
            unflatten(updated_MK_a, head_obj.MK.a, mat_heights, embedding_dim);

        }
        catch (const std::exception& e) {
            std::cerr << "Error during cu1ParallelBackward (H) for head [" << i << "][" << layno << "]: " << e.what() << std::endl;
            // Cleanup allocated memory on error
            cudaFree(d_expected_h); cudaFree(d_EH); cudaFree(d_EV); cudaFree(d_grad_EH); cudaFree(d_grad_EV_scaled); cudaFree(d_grad_dh); cudaFree(d_grad_dv); cudaFree(d_KdotQ); cudaFree(d_head); cudaFree(d_K); cudaFree(d_Q); cudaFree(d_pre_MH); cudaFree(d_pre_MV); cudaFree(d_MH_a); cudaFree(d_MV_a); cudaFree(d_MQ_a); cudaFree(d_MK_a); cudaFree(d_grad_MH); cudaFree(d_grad_MV); cudaFree(d_grad_head); cudaFree(d_lota_deriv); cudaFree(d_grad_KdotQ); cudaFree(d_grad_K); cudaFree(d_grad_Q); cudaFree(d_grad_MQ); cudaFree(d_grad_MK);
            for (int l = 0; l < layers; ++l) { cudaFree(d_hor_activations[l]); cudaFree(d_hor_weights[l]); cudaFree(d_hor_gweights[l]); cudaFree(d_hor_deltas[l]); cudaFree(d_ver_activations[l]); cudaFree(d_ver_weights[l]); cudaFree(d_ver_gweights[l]); cudaFree(d_ver_deltas[l]); }
            throw std::runtime_error("Exception processing head [" + std::to_string(i) + "][" + std::to_string(layno) + "]: " + e.what());
        }

        // --- Cleanup Device Memory (Success Case - Per Head) ---
        cudaFree(d_expected_h); cudaFree(d_EH); cudaFree(d_EV); cudaFree(d_grad_EH); cudaFree(d_grad_EV_scaled); cudaFree(d_grad_dh); cudaFree(d_grad_dv); cudaFree(d_KdotQ); cudaFree(d_head); cudaFree(d_K); cudaFree(d_Q); cudaFree(d_pre_MH); cudaFree(d_pre_MV); cudaFree(d_MH_a); cudaFree(d_MV_a); cudaFree(d_MQ_a); cudaFree(d_MK_a); cudaFree(d_grad_MH); cudaFree(d_grad_MV); cudaFree(d_grad_head); cudaFree(d_lota_deriv); cudaFree(d_grad_KdotQ); cudaFree(d_grad_K); cudaFree(d_grad_Q); cudaFree(d_grad_MQ); cudaFree(d_grad_MK);
        for (int l = 0; l < layers; ++l) { cudaFree(d_hor_activations[l]); cudaFree(d_hor_weights[l]); cudaFree(d_hor_gweights[l]); cudaFree(d_hor_deltas[l]); cudaFree(d_ver_activations[l]); cudaFree(d_ver_weights[l]); cudaFree(d_ver_gweights[l]); cudaFree(d_ver_deltas[l]); }

    } // End loop over rows (i)
}


/**
 * @brief CUDA backward propagation for a single column in a NON-FIRST block,
 *        driven by vertical error (EV). Processes heads b[row][layno].
 *        IMPLEMENTATION USES DIRECT KERNEL CALLS PER HEAD (INEFFICIENT).
 * @param expectedV Expected vertical embeddings for all heads in this column. Shape: [x][CONTEXT_WIN][EMBEDDING].
 * @param in Embedding dimension.
 * @param layers Number of MLP layers.
 * @param layno The column index within the block (0 to y-1).
 */
void block::cu1ParallelBackward(std::vector<std::vector<std::vector<float>>>& expectedV, int& in, int& layers, int layno)
{
    // This function's logic is identical to cu1ParallelBackward1stBlock(expectedV, ...)
    // EXCEPT that it never calls the 1stHead update kernel. It always uses the general update path.
    // We can reuse the code structure.

    // Validate column number and input shape
    if (layno < 0 || layno >= y) {
        throw std::out_of_range("cu1ParallelBackward(V): Column index 'layno' (" + std::to_string(layno) + ") is out of range [0, " + std::to_string(y - 1) + "].");
    }
    if (expectedV.size() != static_cast<size_t>(x)) {
        throw std::runtime_error("ExpectedV size mismatch (number of rows) in cu1ParallelBackward(V). Expected " + std::to_string(x) + ", got " + std::to_string(expectedV.size()));
    }
    if (!expectedV.empty() && (expectedV[0].size() != CONTEXT_WIN || (!expectedV[0].empty() && expectedV[0][0].size() != EMBEDDING))) {
        throw std::runtime_error("ExpectedV dimensions mismatch (context/embedding) in cu1ParallelBackward(V).");
    }
    if (EMBEDDING != in) {
        throw std::runtime_error("Embedding dimension mismatch: EMBEDDING vs in");
    }

    // Constants
    const int embedding_dim = EMBEDDING;
    const int mat_heights = MATHEIGHTS;
    const int context_win = CONTEXT_WIN;
    const float learning_rate = LEARNING;
    const float scaling_factor = SCALING;
    const size_t embed_bytes = embedding_dim * sizeof(float);
    const size_t weights_bytes = embedding_dim * embedding_dim * sizeof(float);
    const int mh_mv_mq_mk_size = mat_heights * embedding_dim;
    const size_t mh_mv_mq_mk_bytes = mh_mv_mq_mk_size * sizeof(float);
    const int ev_size = context_win * embedding_dim;
    const size_t ev_bytes = ev_size * sizeof(float);

    // Kernel Launch Config
    int threadsPerBlock1D = 256;
    dim3 blockDim1D(threadsPerBlock1D);
    int blocksPerGridEmbed = (embedding_dim + threadsPerBlock1D - 1) / threadsPerBlock1D;
    dim3 gridDimEmbed(blocksPerGridEmbed);
    int blocksPerGridMatHeights = (mat_heights + threadsPerBlock1D - 1) / threadsPerBlock1D;
    dim3 gridDimMatHeights(blocksPerGridMatHeights);
    int blocksPerGridMatrix = (mh_mv_mq_mk_size + threadsPerBlock1D - 1) / threadsPerBlock1D;
    dim3 gridDimMatrix(blocksPerGridMatrix);
    int blocksPerGridEV = (ev_size + threadsPerBlock1D - 1) / threadsPerBlock1D;
    dim3 gridDimEV(blocksPerGridEV);

    dim3 blockDim2D(16, 16);
    dim3 gridDimEmbed2D((embedding_dim + blockDim2D.x - 1) / blockDim2D.x, (embedding_dim + blockDim2D.y - 1) / blockDim2D.y);
    dim3 gridDimMatrix2D((embedding_dim + blockDim2D.x - 1) / blockDim2D.x, (mat_heights + blockDim2D.y - 1) / blockDim2D.y);

    // Iterate through the rows
    for (int i = 0; i < x; ++i) {
        attention& head_obj = b[i][layno];
        std::vector<std::vector<float>>& expectedV_head = expectedV[i];
        const int token_count = head_obj.tokenCount;
        const int head_size = token_count * token_count;
        const size_t head_bytes = head_size * sizeof(float);
        const int k_q_size = token_count * mat_heights;
        const size_t k_q_bytes = k_q_size * sizeof(float);
        dim3 gridDimHead2D((token_count + blockDim2D.x - 1) / blockDim2D.x, (token_count + blockDim2D.y - 1) / blockDim2D.y);
        dim3 gridDimKQGrad2D((mat_heights + blockDim2D.x - 1) / blockDim2D.x, (token_count + blockDim2D.y - 1) / blockDim2D.y);
        int blocksPerGridHead = (head_size + threadsPerBlock1D - 1) / threadsPerBlock1D;
        dim3 gridDimHead(blocksPerGridHead);

        // --- Device Pointers (Allocate per head) ---
        // (Same list as in cu1ParallelBackward1stBlock(V))
        float *d_expected_v = nullptr, *d_EV = nullptr;
        float *d_grad_EV_full = nullptr, *d_grad_EV_summed = nullptr, *d_grad_EV_scaled = nullptr;
        float *d_grad_dv = nullptr;
        float *d_KdotQ = nullptr, *d_head = nullptr;
        float *d_K = nullptr, *d_Q = nullptr;
        float *d_pre_MV = nullptr;
        float *d_MV_a = nullptr, *d_MQ_a = nullptr, *d_MK_a = nullptr;
        float *d_grad_MV = nullptr;
        float *d_grad_head = nullptr;
        float *d_lota_deriv = nullptr;
        float *d_grad_KdotQ = nullptr;
        float *d_grad_Q = nullptr;
        float *d_grad_MQ = nullptr, *d_grad_MK_correction = nullptr;
        std::vector<float*> d_ver_activations(layers, nullptr);
        std::vector<float*> d_ver_weights(layers, nullptr);
        std::vector<float*> d_ver_gweights(layers, nullptr);
        std::vector<float*> d_ver_deltas(layers, nullptr);
        std::vector<std::vector<float>> flat_ver_weights(layers);

        try {
            // --- Allocate Memory ---
            // (Identical allocations as in cu1ParallelBackward1stBlock(V))
            CUDA_CHECK(cudaMalloc(&d_expected_v, ev_bytes));
            CUDA_CHECK(cudaMalloc(&d_EV, ev_bytes));
            CUDA_CHECK(cudaMalloc(&d_grad_EV_full, ev_bytes));
            CUDA_CHECK(cudaMalloc(&d_grad_EV_summed, embed_bytes));
            CUDA_CHECK(cudaMalloc(&d_grad_EV_scaled, embed_bytes));
            CUDA_CHECK(cudaMalloc(&d_grad_dv, embed_bytes));
            CUDA_CHECK(cudaMalloc(&d_KdotQ, head_bytes));
            CUDA_CHECK(cudaMalloc(&d_head, head_bytes));
            CUDA_CHECK(cudaMalloc(&d_K, k_q_bytes));
            CUDA_CHECK(cudaMalloc(&d_Q, k_q_bytes));
            CUDA_CHECK(cudaMalloc(&d_pre_MV, mat_heights * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_MV_a, mh_mv_mq_mk_bytes));
            CUDA_CHECK(cudaMalloc(&d_MQ_a, mh_mv_mq_mk_bytes));
            CUDA_CHECK(cudaMalloc(&d_MK_a, mh_mv_mq_mk_bytes));
            CUDA_CHECK(cudaMalloc(&d_grad_MV, mh_mv_mq_mk_bytes));
            CUDA_CHECK(cudaMalloc(&d_grad_head, head_bytes));
            CUDA_CHECK(cudaMalloc(&d_lota_deriv, head_bytes));
            CUDA_CHECK(cudaMalloc(&d_grad_KdotQ, head_bytes));
            CUDA_CHECK(cudaMalloc(&d_grad_Q, k_q_bytes));
            CUDA_CHECK(cudaMalloc(&d_grad_MQ, mh_mv_mq_mk_bytes));
            CUDA_CHECK(cudaMalloc(&d_grad_MK_correction, mh_mv_mq_mk_bytes));
            for (int l = 0; l < layers; ++l) {
                CUDA_CHECK(cudaMalloc(&d_ver_activations[l], embed_bytes));
                CUDA_CHECK(cudaMalloc(&d_ver_weights[l], weights_bytes));
                CUDA_CHECK(cudaMalloc(&d_ver_gweights[l], weights_bytes));
                CUDA_CHECK(cudaMalloc(&d_ver_deltas[l], embed_bytes));
            }

            // --- Data Transfer H->D ---
            // (Identical transfers as in cu1ParallelBackward1stBlock(V))
            std::vector<float> flat_expectedV_head = flatten(expectedV_head);
            std::vector<float> flat_EV = flatten(head_obj.EV);
            std::vector<float> flat_K = flatten(head_obj.K);
            std::vector<float> flat_Q = flatten(head_obj.Q);
            std::vector<float> flat_KdotQ = flatten(head_obj.KdotQ);
            std::vector<float> flat_MV_a = flatten(head_obj.MV.a);
            std::vector<float> flat_MQ_a = flatten(head_obj.MQ.a);
            std::vector<float> flat_MK_a = flatten(head_obj.MK.a);
            CUDA_CHECK(cudaMemcpy(d_expected_v, flat_expectedV_head.data(), ev_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_EV, flat_EV.data(), ev_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_KdotQ, flat_KdotQ.data(), head_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_K, flat_K.data(), k_q_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_Q, flat_Q.data(), k_q_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_MV_a, flat_MV_a.data(), mh_mv_mq_mk_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_MQ_a, flat_MQ_a.data(), mh_mv_mq_mk_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_MK_a, flat_MK_a.data(), mh_mv_mq_mk_bytes, cudaMemcpyHostToDevice));
            for (int l = 0; l < layers; ++l) {
                flat_ver_weights[l].reserve(embedding_dim * embedding_dim);
                for (int r = 0; r < embedding_dim; ++r) { flat_ver_weights[l].insert(flat_ver_weights[l].end(), head_obj.ver.weights[l][r].begin(), head_obj.ver.weights[l][r].end()); }
                CUDA_CHECK(cudaMemcpy(d_ver_activations[l], head_obj.ver.activations[l].data(), embed_bytes, cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_ver_weights[l], flat_ver_weights[l].data(), weights_bytes, cudaMemcpyHostToDevice));
            }

            // --- Backpropagation Steps ---
            // (Identical steps 1-8 as in cu1ParallelBackward1stBlock(V))
            // Step 1: Compute grad_EV
            kernelComputeGradientsEV_V<<<gridDimEmbed, blockDim1D>>>(d_EV, d_expected_v, d_grad_EV_full, d_grad_EV_summed, d_grad_EV_scaled, learning_rate, context_win, embedding_dim); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
            // Step 2: Backprop through ver MLP
            kernelLastLayerDelta<<<gridDimEmbed, blockDim1D>>>(d_grad_EV_scaled, d_ver_activations[layers - 1], d_ver_deltas[layers - 1], embedding_dim); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
            for (int l = layers - 2; l >= 0; --l) { hiddenDeltaKernel<<<gridDimEmbed, blockDim1D>>>(d_ver_deltas[l + 1], d_ver_weights[l + 1], d_ver_activations[l], d_ver_deltas[l], embedding_dim, embedding_dim); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize()); }
            for (int l = 0; l < layers; ++l) { float* d_prev = (l == 0) ? d_EV : d_ver_activations[l - 1]; updateWeightsKernel<<<gridDimEmbed2D, blockDim2D>>>(d_ver_deltas[l], d_prev, d_ver_weights[l], d_ver_gweights[l], learning_rate, embedding_dim, embedding_dim); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize()); }
            // Step 3: Compute grad_dv
            kernelComputeGradMLPInput<<<gridDimEmbed, blockDim1D>>>(d_ver_deltas[0], d_ver_weights[0], d_grad_dv, embedding_dim, embedding_dim); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
            // Step 4: Compute grad_MV
            cuLOTA<<<gridDimHead2D, blockDim2D>>>(d_KdotQ, d_head, context_win, context_win, token_count, head_obj.isSelfAttention); CUDA_CHECK(cudaGetLastError());
            kernelComputePreMV_V<<<gridDimMatHeights, blockDim1D>>>(d_head, d_Q, d_pre_MV, token_count, mat_heights); CUDA_CHECK(cudaGetLastError());
            kernelComputeGradMV_V<<<gridDimMatrix2D, blockDim2D>>>(d_pre_MV, d_grad_dv, d_grad_MV, mat_heights, embedding_dim); CUDA_CHECK(cudaGetLastError());
            // Step 5: Compute grad_head (V path only)
            kernelComputeGradHead_V<<<gridDimHead2D, blockDim2D>>>(d_Q, d_MV_a, d_grad_dv, d_grad_head, token_count, mat_heights, embedding_dim); CUDA_CHECK(cudaGetLastError());
            // Step 6: Backprop through LOTA
            cuLOTAder<<<gridDimHead2D, blockDim2D>>>(d_KdotQ, d_lota_deriv, context_win, context_win, token_count, head_obj.isSelfAttention); CUDA_CHECK(cudaGetLastError());
            kernelComputeGradKdotQ_LOTA<<<gridDimHead, blockDim1D>>>(d_grad_head, d_lota_deriv, d_grad_KdotQ, scaling_factor, head_size); CUDA_CHECK(cudaGetLastError());
            // Step 7: Compute grad_Q
            kernelComputeGradQ_V<<<gridDimKQGrad2D, blockDim2D>>>(d_grad_KdotQ, d_K, d_grad_Q, token_count, mat_heights); CUDA_CHECK(cudaGetLastError());
            // Step 8: Compute grad_MQ and grad_MK_correction
            kernelComputeGradMQ_V<<<gridDimMatrix2D, blockDim2D>>>(d_grad_Q, nullptr, d_grad_MQ, token_count, mat_heights, embedding_dim); CUDA_CHECK(cudaGetLastError());
            kernelComputeGradMKCorrection<<<gridDimMatrix2D, blockDim2D>>>(d_grad_MQ, d_Q, d_K, d_grad_MK_correction, token_count, mat_heights, embedding_dim); CUDA_CHECK(cudaGetLastError());

            // --- Step 9 & 10: Update Weights (General Case) ---
            // Always use the general update path.
            kernelUpdateSimple<<<gridDimMatrix, blockDim1D>>>(d_MV_a, d_grad_MV, learning_rate, mh_mv_mq_mk_size);
            kernelUpdateSimple<<<gridDimMatrix, blockDim1D>>>(d_MQ_a, d_grad_MQ, learning_rate, mh_mv_mq_mk_size);
            kernelUpdateSimple<<<gridDimMatrix, blockDim1D>>>(d_MK_a, d_grad_MK_correction, learning_rate, mh_mv_mq_mk_size); // Update MK with correction
            CUDA_CHECK(cudaGetLastError());
            kernelUpdateSimple<<<gridDimEV, blockDim1D>>>(d_EV, d_grad_EV_full, learning_rate, ev_size); // Update full EV using full gradient
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            // --- Data Transfer D->H ---
            // (Identical transfers as in cu1ParallelBackward1stBlock(V))
            for (int l = 0; l < layers; ++l) {
                std::vector<float> updated_flat_weights(embedding_dim * embedding_dim);
                std::vector<float> calculated_flat_gradients(embedding_dim * embedding_dim);
                CUDA_CHECK(cudaMemcpy(updated_flat_weights.data(), d_ver_weights[l], weights_bytes, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(calculated_flat_gradients.data(), d_ver_gweights[l], weights_bytes, cudaMemcpyDeviceToHost));
                for (int r = 0; r < embedding_dim; ++r) { for (int c = 0; c < embedding_dim; ++c) { head_obj.ver.weights[l][r][c] = updated_flat_weights[r*embedding_dim+c]; if(head_obj.ver.gweights.size()<=l)head_obj.ver.gweights.resize(l+1); if(head_obj.ver.gweights[l].size()<=r)head_obj.ver.gweights[l].resize(r+1); if(head_obj.ver.gweights[l][r].size()!=embedding_dim)head_obj.ver.gweights[l][r].resize(embedding_dim); head_obj.ver.gweights[l][r][c] = calculated_flat_gradients[r*embedding_dim+c]; } }
            }
            std::vector<float> updated_MV_a(mh_mv_mq_mk_size), updated_MQ_a(mh_mv_mq_mk_size), updated_MK_a(mh_mv_mq_mk_size), updated_EV(ev_size);
            CUDA_CHECK(cudaMemcpy(updated_EV.data(), d_EV, ev_bytes, cudaMemcpyDeviceToHost)); 
            CUDA_CHECK(cudaMemcpy(updated_MV_a.data(), d_MV_a, mh_mv_mq_mk_bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(updated_MQ_a.data(), d_MQ_a, mh_mv_mq_mk_bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(updated_MK_a.data(), d_MK_a, mh_mv_mq_mk_bytes, cudaMemcpyDeviceToHost));
            unflatten(updated_EV, head_obj.EV, context_win, embedding_dim);
            unflatten(updated_MV_a, head_obj.MV.a, mat_heights, embedding_dim);
            unflatten(updated_MQ_a, head_obj.MQ.a, mat_heights, embedding_dim);
            unflatten(updated_MK_a, head_obj.MK.a, mat_heights, embedding_dim);

        }
        catch (const std::exception& e) {
            std::cerr << "Error during cu1ParallelBackward (V) for head [" << i << "][" << layno << "]: " << e.what() << std::endl;
            // Cleanup allocated memory on error
            cudaFree(d_expected_v); cudaFree(d_EV); cudaFree(d_grad_EV_full); cudaFree(d_grad_EV_summed); cudaFree(d_grad_EV_scaled); cudaFree(d_grad_dv); cudaFree(d_KdotQ); cudaFree(d_head); cudaFree(d_K); cudaFree(d_Q); cudaFree(d_pre_MV); cudaFree(d_MV_a); cudaFree(d_MQ_a); cudaFree(d_MK_a); cudaFree(d_grad_MV); cudaFree(d_grad_head); cudaFree(d_lota_deriv); cudaFree(d_grad_KdotQ); cudaFree(d_grad_Q); cudaFree(d_grad_MQ); cudaFree(d_grad_MK_correction);
            for (int l = 0; l < layers; ++l) { cudaFree(d_ver_activations[l]); cudaFree(d_ver_weights[l]); cudaFree(d_ver_gweights[l]); cudaFree(d_ver_deltas[l]); }
            throw std::runtime_error("Exception processing head [" + std::to_string(i) + "][" + std::to_string(layno) + "]: " + e.what());
        }

        // --- Cleanup Device Memory (Success Case - Per Head) ---
        cudaFree(d_expected_v); cudaFree(d_EV); cudaFree(d_grad_EV_full); cudaFree(d_grad_EV_summed); cudaFree(d_grad_EV_scaled); cudaFree(d_grad_dv); cudaFree(d_KdotQ); cudaFree(d_head); cudaFree(d_K); cudaFree(d_Q); cudaFree(d_pre_MV); cudaFree(d_MV_a); cudaFree(d_MQ_a); cudaFree(d_MK_a); cudaFree(d_grad_MV); cudaFree(d_grad_head); cudaFree(d_lota_deriv); cudaFree(d_grad_KdotQ); cudaFree(d_grad_Q); cudaFree(d_grad_MQ); cudaFree(d_grad_MK_correction);
        for (int l = 0; l < layers; ++l) { cudaFree(d_ver_activations[l]); cudaFree(d_ver_weights[l]); cudaFree(d_ver_gweights[l]); cudaFree(d_ver_deltas[l]); }

    } // End loop over rows (i)
}
