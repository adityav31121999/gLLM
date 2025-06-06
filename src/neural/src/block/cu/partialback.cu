
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

    // MLP structure parameters based on 'layers' (L = number of hidden layers)
    const int num_total_layers_mlp = layers; // 'layers' parameter represents the total number of layers (Input + Hidden + Output)
    const int num_neuron_layers_mlp = num_total_layers_mlp; // Number of neuron layers is the total number of layers
    const int num_weight_matrices_mlp = num_total_layers_mlp - 1; // Number of weight matrices is total layers - 1

    const size_t embed_bytes = embedding_dim * sizeof(float);
    const size_t mlp_weights_elements = static_cast<size_t>(embedding_dim) * embedding_dim;
    const size_t mlp_weights_bytes = mlp_weights_elements * sizeof(float);
    const size_t proj_mat_elements = static_cast<size_t>(mat_heights) * embedding_dim; // For MH, MV, MQ, MK
    const size_t proj_mat_bytes = proj_mat_elements * sizeof(float);
    const size_t ev_total_elements = static_cast<size_t>(context_win) * embedding_dim; // For EV
    const size_t ev_total_bytes = ev_total_elements * sizeof(float);
    // Kernel Launch Config (Define once)
    int threadsPerBlock1D = 256;
    dim3 blockDim1D(threadsPerBlock1D);
    int blocksPerGridEmbed = (embedding_dim + threadsPerBlock1D - 1) / threadsPerBlock1D;
    dim3 gridDimEmbed(blocksPerGridEmbed);
    int blocksPerGridMatHeights = (mat_heights + threadsPerBlock1D - 1) / threadsPerBlock1D;
    dim3 gridDimMatHeights(blocksPerGridMatHeights);
    int blocksPerGridProjMat = (static_cast<int>(proj_mat_elements) + threadsPerBlock1D - 1) / threadsPerBlock1D;
    dim3 gridDimProjMat(blocksPerGridProjMat);

    dim3 blockDim2D(16, 16);
    dim3 gridDimEmbed2D((embedding_dim + blockDim2D.x - 1) / blockDim2D.x, (embedding_dim + blockDim2D.y - 1) / blockDim2D.y);
    dim3 gridDimMatrix2D((embedding_dim + blockDim2D.x - 1) / blockDim2D.x, (mat_heights + blockDim2D.y - 1) / blockDim2D.y);

    // Iterate backwards through the rows (parallels/layers) for the specified column
    for (int i = x - 1; i >= 0; --i) { // 'i' is the row index
        attention& head_obj = b[i][layno]; // Reference to the current head object
        const int token_count = head_obj.tokenCount;
        bool att = head_obj.isSelfAttention;
        const size_t active_head_elements = static_cast<size_t>(token_count) * token_count;
        const size_t active_head_bytes = active_head_elements * sizeof(float);
        const size_t active_k_q_elements = static_cast<size_t>(token_count) * mat_heights;
        const size_t active_k_q_bytes = active_k_q_elements * sizeof(float);
        // Adjust 2D grids based on token_count
        dim3 gridDimHead2D((token_count + blockDim2D.x - 1) / blockDim2D.x, (token_count + blockDim2D.y - 1) / blockDim2D.y);
        dim3 gridDimKQGrad2D((mat_heights + blockDim2D.x - 1) / blockDim2D.x, (token_count + blockDim2D.y - 1) / blockDim2D.y);
        int blocksPerGridHead = (static_cast<int>(active_head_elements) + threadsPerBlock1D - 1) / threadsPerBlock1D;
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
        std::vector<float*> d_hor_activations(num_neuron_layers_mlp, nullptr);
        std::vector<float*> d_hor_weights(num_weight_matrices_mlp, nullptr); // W[0] to W[N-2]
        std::vector<float*> d_hor_gweights(num_weight_matrices_mlp, nullptr); // gW[0] to gW[N-2]
        std::vector<float*> d_hor_deltas(num_weight_matrices_mlp, nullptr); // Delta[1] to Delta[N-1] (stored at index 0 to N-2)
        std::vector<float*> d_ver_activations(num_neuron_layers_mlp, nullptr);
        std::vector<float*> d_ver_weights(num_weight_matrices_mlp, nullptr); // W[0] to W[N-2]
        std::vector<float*> d_ver_gweights(num_weight_matrices_mlp, nullptr); // gW[0] to gW[N-2]
        std::vector<float*> d_ver_deltas(num_weight_matrices_mlp, nullptr); // For deltas of neuron layers 1 to L+1

        try {
            // --- Allocate Memory (Attention) ---
            CUDA_CHECK(cudaMalloc(&d_expected_h, embed_bytes));
            CUDA_CHECK(cudaMalloc(&d_EH, embed_bytes));
            CUDA_CHECK(cudaMalloc(&d_EV, ev_total_bytes)); // Allocate full EV size
            CUDA_CHECK(cudaMalloc(&d_grad_EH, embed_bytes));
            CUDA_CHECK(cudaMalloc(&d_grad_EV_scaled, embed_bytes));
            CUDA_CHECK(cudaMalloc(&d_grad_dh, embed_bytes));
            CUDA_CHECK(cudaMalloc(&d_grad_dv, embed_bytes));
            CUDA_CHECK(cudaMalloc(&d_KdotQ, active_head_bytes));
            CUDA_CHECK(cudaMalloc(&d_head, active_head_bytes));
            CUDA_CHECK(cudaMalloc(&d_K, active_k_q_bytes));
            CUDA_CHECK(cudaMalloc(&d_Q, active_k_q_bytes));
            CUDA_CHECK(cudaMalloc(&d_pre_MH, mat_heights * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_pre_MV, mat_heights * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_MH_a, proj_mat_bytes));
            CUDA_CHECK(cudaMalloc(&d_MV_a, proj_mat_bytes));
            CUDA_CHECK(cudaMalloc(&d_MQ_a, proj_mat_bytes));
            CUDA_CHECK(cudaMalloc(&d_MK_a, proj_mat_bytes));
            CUDA_CHECK(cudaMalloc(&d_grad_MH, proj_mat_bytes));
            CUDA_CHECK(cudaMalloc(&d_grad_MV, proj_mat_bytes));
            CUDA_CHECK(cudaMalloc(&d_grad_head, active_head_bytes));
            CUDA_CHECK(cudaMalloc(&d_lota_deriv, active_head_bytes));
            CUDA_CHECK(cudaMalloc(&d_grad_KdotQ, active_head_bytes));
            CUDA_CHECK(cudaMalloc(&d_grad_K, active_k_q_bytes));
            CUDA_CHECK(cudaMalloc(&d_grad_Q, active_k_q_bytes));
            CUDA_CHECK(cudaMalloc(&d_grad_MQ, proj_mat_bytes));
            CUDA_CHECK(cudaMalloc(&d_grad_MK, proj_mat_bytes));

            // --- Allocate Memory (MLP Internals) ---
            for (int l = 0; l < num_neuron_layers_mlp; ++l) { // All neuron layers (0 to N-1)
                CUDA_CHECK(cudaMalloc(&d_hor_activations[l], embed_bytes));
                CUDA_CHECK(cudaMalloc(&d_ver_activations[l], embed_bytes));
            }
            for (int l = 0; l < num_weight_matrices_mlp; ++l) { // Deltas for neuron layers 1 to N-1 (stored at index 0 to N-2)
                CUDA_CHECK(cudaMalloc(&d_hor_deltas[l], embed_bytes));
                CUDA_CHECK(cudaMalloc(&d_ver_deltas[l], embed_bytes));
            }

            for (int l = 0; l < num_weight_matrices_mlp; ++l) { // All weight matrices
                CUDA_CHECK(cudaMalloc(&d_hor_weights[l], mlp_weights_bytes));
                CUDA_CHECK(cudaMalloc(&d_hor_gweights[l], mlp_weights_bytes));
                CUDA_CHECK(cudaMalloc(&d_ver_weights[l], mlp_weights_bytes));
                CUDA_CHECK(cudaMalloc(&d_ver_gweights[l], mlp_weights_bytes));
            }

            // --- Data Transfer H->D (Attention) ---
            // Validate mat objects before use
            if (!head_obj.EV.mapped_data || !head_obj.K.mapped_data || !head_obj.Q.mapped_data || !head_obj.KdotQ.mapped_data ||
                !head_obj.MH.mapped_data || !head_obj.MV.mapped_data || !head_obj.MQ.mapped_data || !head_obj.MK.mapped_data) {
                throw std::runtime_error("One or more attention mat objects have null mapped_data for head [" + std::to_string(i) + "][" + std::to_string(layno) + "]");
            }

            CUDA_CHECK(cudaMemcpy(d_expected_h, expectedH.data(), embed_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_EH, head_obj.EH.data(), embed_bytes, cudaMemcpyHostToDevice));
            // Copy full EV, MH, MV, MQ, MK
            CUDA_CHECK(cudaMemcpy(d_EV, head_obj.EV.mapped_data, ev_total_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_MH_a, head_obj.MH.mapped_data, proj_mat_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_MV_a, head_obj.MV.mapped_data, proj_mat_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_MQ_a, head_obj.MQ.mapped_data, proj_mat_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_MK_a, head_obj.MK.mapped_data, proj_mat_bytes, cudaMemcpyHostToDevice));
            // Copy active parts of K, Q, KdotQ
            CUDA_CHECK(cudaMemcpy(d_KdotQ, head_obj.KdotQ.mapped_data, active_head_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_K, head_obj.K.mapped_data, active_k_q_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_Q, head_obj.Q.mapped_data, active_k_q_bytes, cudaMemcpyHostToDevice));


            // --- Data Transfer H->D (MLP Internals) ---
            if(head_obj.hor.activations.size() != static_cast<size_t>(num_neuron_layers_mlp) ||
               head_obj.ver.activations.size() != static_cast<size_t>(num_neuron_layers_mlp)) {
                throw std::runtime_error("MLP host activations vector size mismatch.");
            }
            for (int l = 0; l < num_neuron_layers_mlp; ++l) {
                if (head_obj.hor.activations[l].empty() || head_obj.ver.activations[l].empty()) {
                    throw std::runtime_error("MLP activation vector is empty for head [" + std::to_string(i) + "][" + std::to_string(layno) + "], layer " + std::to_string(l));
                }
                CUDA_CHECK(cudaMemcpy(d_hor_activations[l], head_obj.hor.activations[l].data(), embed_bytes, cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_ver_activations[l], head_obj.ver.activations[l].data(), embed_bytes, cudaMemcpyHostToDevice));
            }
            if(head_obj.hor.weights.size() != static_cast<size_t>(num_weight_matrices_mlp) ||
               head_obj.ver.weights.size() != static_cast<size_t>(num_weight_matrices_mlp)) {
                throw std::runtime_error("MLP host weights vector size mismatch.");
            }
            for (int l = 0; l < num_weight_matrices_mlp; ++l) {
                if (!head_obj.hor.weights[l].mapped_data || head_obj.hor.weights[l].row != embedding_dim || head_obj.hor.weights[l].col != embedding_dim ||
                    !head_obj.ver.weights[l].mapped_data || head_obj.ver.weights[l].row != embedding_dim || head_obj.ver.weights[l].col != embedding_dim ||
                    !head_obj.hor.gweights[l].mapped_data || head_obj.hor.gweights[l].row != embedding_dim || head_obj.hor.gweights[l].col != embedding_dim ||
                    !head_obj.ver.gweights[l].mapped_data || head_obj.ver.gweights[l].row != embedding_dim || head_obj.ver.gweights[l].col != embedding_dim) {
                    throw std::runtime_error("Invalid MLP weight/gweight mat for head [" + std::to_string(i) + "][" + std::to_string(layno) + "], layer " + std::to_string(l));
                }
                CUDA_CHECK(cudaMemcpy(d_hor_weights[l], head_obj.hor.weights[l].mapped_data, mlp_weights_bytes, cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_ver_weights[l], head_obj.ver.weights[l].mapped_data, mlp_weights_bytes, cudaMemcpyHostToDevice));
            }

            // --- Backpropagation Steps ---
            // Step 1: Compute grad_EH and grad_EV_scaled
            kernelComputeGradientsEH_EV<<<gridDimEmbed, blockDim1D>>>(d_EH, d_expected_h, d_grad_EH, d_grad_EV_scaled, embedding_dim);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            // --- Step 2: Backprop through MLPs ---
            // --- 2a: Backprop through hor MLP (L_mlp hidden layers) ---
            // Calculate Output Layer Deltas for hor MLP (delta for neuron layer N-1, stored in d_hor_deltas[N-2])
            kernelLastLayerDelta<<<gridDimEmbed, blockDim1D>>>(d_grad_EH, d_hor_activations[num_total_layers_mlp - 1], d_hor_deltas[num_total_layers_mlp - 2], embedding_dim);
            CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
            
            // Calculate Hidden Layer Deltas for hor MLP (for neuron layers N-2 down to 1)
            // Loop iterates over the index 'k' of the delta being calculated (0 to N-3, for layers 1 to N-2)
            for (int k = num_total_layers_mlp - 2; k >= 1; --k) { // k is the neuron layer index (1 to N-2)
                hiddenDeltaKernel<<<gridDimEmbed, blockDim1D>>>(d_hor_deltas[k], d_hor_weights[k], d_hor_activations[k], d_hor_deltas[k-1], embedding_dim, embedding_dim);
                CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
            }
            // Calculate Gradients and Update Weights for hor MLP (N-1 weight matrices, W[0] to W[N-2])
            for (int l_weight_idx = 0; l_weight_idx < num_weight_matrices_mlp; ++l_weight_idx) {
                updateWeightsKernel<<<gridDimEmbed2D, blockDim2D>>>(d_hor_deltas[l_weight_idx], d_hor_activations[l_weight_idx], d_hor_weights[l_weight_idx], d_hor_gweights[l_weight_idx], learning_rate, embedding_dim, embedding_dim);
                CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
            }

            // --- 2b: Backprop through ver MLP ---
            kernelLastLayerDelta<<<gridDimEmbed, blockDim1D>>>(d_grad_EV_scaled, d_ver_activations[num_total_layers_mlp - 1], d_ver_deltas[num_total_layers_mlp - 2], embedding_dim);
            CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
            for (int l_neuron_idx = num_total_layers_mlp - 2; l_neuron_idx >= 1; --l_neuron_idx) {
                hiddenDeltaKernel<<<gridDimEmbed, blockDim1D>>>(d_ver_deltas[l_neuron_idx], d_ver_weights[l_neuron_idx], d_ver_activations[l_neuron_idx], d_ver_deltas[l_neuron_idx-1], embedding_dim, embedding_dim);
                CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
            }
            for (int l_weight_idx = 0; l_weight_idx < num_weight_matrices_mlp; ++l_weight_idx) {
                updateWeightsKernel<<<gridDimEmbed2D, blockDim2D>>>(d_ver_deltas[l_weight_idx], d_ver_activations[l_weight_idx], d_ver_weights[l_weight_idx], d_ver_gweights[l_weight_idx], learning_rate, embedding_dim, embedding_dim);
                CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
            }

            // --- Step 3: Compute grad_dh and grad_dv ---
            // grad_dh = delta_for_first_hidden_layer * W_input_to_first_hidden_layer
            // d_hor_deltas[0] is delta for neuron layer 1 (stored at index 0)
            // d_hor_weights[0] is W[0] (Input -> Layer 1, stored at index 0)
            kernelComputeGradMLPInput<<<gridDimEmbed, blockDim1D>>>(d_hor_deltas[0], d_hor_weights[0], d_grad_dh, embedding_dim, embedding_dim);
            CUDA_CHECK(cudaGetLastError());
            kernelComputeGradMLPInput<<<gridDimEmbed, blockDim1D>>>(d_ver_deltas[0], d_ver_weights[0], d_grad_dv, embedding_dim, embedding_dim);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            if(token_count > 0) {
                // --- Step 4: Compute grad_MH and grad_MV ---
                cuLOTA<<<gridDimHead2D, blockDim2D>>>(d_KdotQ, d_head, context_win, context_win, token_count, att);
                CUDA_CHECK(cudaGetLastError());
                kernelComputePreMH_MV<<<gridDimMatHeights, blockDim1D>>>(d_head, d_K, d_Q, d_pre_MH, d_pre_MV, token_count, mat_heights);
                CUDA_CHECK(cudaGetLastError());
                // gridDimMatrix2D might need adjustment if mat_heights != embedding_dim
                kernelComputeGradMH_MV<<<gridDimMatrix2D, blockDim2D>>>(d_pre_MH, d_pre_MV, d_grad_dh, d_grad_dv, d_grad_MH, d_grad_MV, mat_heights, embedding_dim);
                CUDA_CHECK(cudaGetLastError());

                // --- Step 5: Compute grad_head ---
                kernelComputeGradHead<<<gridDimHead2D, blockDim2D>>>(d_K, d_Q, d_MH_a, d_MV_a, d_grad_dh, d_grad_dv, d_grad_head, token_count, mat_heights, embedding_dim);
                CUDA_CHECK(cudaGetLastError());

                // --- Step 6: Backprop through LOTA ---
                cuLOTAder<<<gridDimHead2D, blockDim2D>>>(d_KdotQ, d_lota_deriv, context_win, context_win, token_count, att);
                CUDA_CHECK(cudaGetLastError());
                kernelComputeGradKdotQ_LOTA<<<gridDimHead, blockDim1D>>>(d_grad_head, d_lota_deriv, d_grad_KdotQ, scaling_factor, static_cast<int>(active_head_elements));
                CUDA_CHECK(cudaGetLastError());

                // --- Step 7: Compute grad_K and grad_Q ---
                kernelComputeGradK_Q<<<gridDimKQGrad2D, blockDim2D>>>(d_grad_KdotQ, d_K, d_Q, d_grad_K, d_grad_Q, token_count, mat_heights);
                CUDA_CHECK(cudaGetLastError());

                // --- Step 8: Compute grad_MK and grad_MQ (Simplified) ---
                // Requires original token embeddings - assuming they are not directly available here.
                // Using nullptr for k_embed/q_embed, kernel handles null checks.
                kernelComputeGradMK_MQ_Simplified<<<gridDimMatrix2D, blockDim2D>>>(d_grad_K, d_grad_Q, nullptr, nullptr, d_grad_MK, d_grad_MQ, token_count, mat_heights, embedding_dim);
                CUDA_CHECK(cudaGetLastError());
            }
            else {
                // If token_count is 0, attention-related gradients (MH, MV, MQ, MK) should be zero
                // as they won't be computed by the skipped steps.
                // Other gradients like d_grad_K, d_grad_Q, d_grad_head etc. are for buffers
                // sized by token_count. If token_count is 0, these buffers are 0-sized,
                // and cudaMalloc/cudaMemset on them are effectively no-ops or handle 0 size.
                CUDA_CHECK(cudaMemset(d_grad_MH, 0, proj_mat_bytes));
                CUDA_CHECK(cudaMemset(d_grad_MV, 0, proj_mat_bytes));
                CUDA_CHECK(cudaMemset(d_grad_MQ, 0, proj_mat_bytes));
                CUDA_CHECK(cudaMemset(d_grad_MK, 0, proj_mat_bytes));
            }
            CUDA_CHECK(cudaDeviceSynchronize());

            // --- Step 9 & 10: Update Weights ---
            if (is_first_head) {
                // Use kernelUpdateWeights_1stHead_H (updates MH, MV, MQ, MK, EH)
                kernelUpdateWeights_1stHead_H<<<gridDimProjMat, blockDim1D>>>(
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
                kernelUpdateSimple<<<gridDimProjMat, blockDim1D>>>(d_MH_a, d_grad_MH, learning_rate, proj_mat_elements);
                kernelUpdateSimple<<<gridDimProjMat, blockDim1D>>>(d_MV_a, d_grad_MV, learning_rate, proj_mat_elements);
                kernelUpdateSimple<<<gridDimProjMat, blockDim1D>>>(d_MQ_a, d_grad_MQ, learning_rate, proj_mat_elements);
                kernelUpdateSimple<<<gridDimProjMat, blockDim1D>>>(d_MK_a, d_grad_MK, learning_rate, proj_mat_elements);
                CUDA_CHECK(cudaGetLastError());
                // Update EH (using d_grad_EH, which is the gradient w.r.t. EH after ReLU)
                kernelUpdateSimple<<<gridDimEmbed, blockDim1D>>>(d_EH, d_grad_EH, learning_rate, embedding_dim);
                CUDA_CHECK(cudaGetLastError());
                // Update EV using d_grad_EV_scaled (broadcasted)
                kernelUpdateEVBroadcasted<<< (context_win + threadsPerBlock1D -1) / threadsPerBlock1D, blockDim1D >>>(d_EV, d_grad_EV_scaled, learning_rate, context_win, embedding_dim);
                CUDA_CHECK(cudaGetLastError());
            }
            CUDA_CHECK(cudaDeviceSynchronize());

            // --- Data Transfer D->H ---
            // Copy updated MLP weights and gradients back
            for (int l = 0; l < num_weight_matrices_mlp; ++l) {
                // Hor MLP
                CUDA_CHECK(cudaMemcpy(head_obj.hor.weights[l].mapped_data, d_hor_weights[l], mlp_weights_bytes, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(head_obj.hor.gweights[l].mapped_data, d_hor_gweights[l], mlp_weights_bytes, cudaMemcpyDeviceToHost));
                // Ver MLP
                CUDA_CHECK(cudaMemcpy(head_obj.ver.weights[l].mapped_data, d_ver_weights[l], mlp_weights_bytes, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(head_obj.ver.gweights[l].mapped_data, d_ver_gweights[l], mlp_weights_bytes, cudaMemcpyDeviceToHost));
            }
            for (int l = 0; l < num_neuron_layers_mlp; ++l) { // Copy back activations if needed by host
                // CUDA_CHECK(cudaMemcpy(head_obj.hor.activations[l].data(), d_hor_activations[l], embed_bytes, cudaMemcpyDeviceToHost));
                // CUDA_CHECK(cudaMemcpy(head_obj.ver.activations[l].data(), d_ver_activations[l], embed_bytes, cudaMemcpyDeviceToHost));
            }

            // Copy updated Attention parameters back
            CUDA_CHECK(cudaMemcpy(head_obj.EH.data(), d_EH, embed_bytes, cudaMemcpyDeviceToHost));
            // Copy directly to mat mapped_data
            CUDA_CHECK(cudaMemcpy(head_obj.EV.mapped_data, d_EV, ev_total_bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(head_obj.MH.mapped_data, d_MH_a, proj_mat_bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(head_obj.MV.mapped_data, d_MV_a, proj_mat_bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(head_obj.MQ.mapped_data, d_MQ_a, proj_mat_bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(head_obj.MK.mapped_data, d_MK_a, proj_mat_bytes, cudaMemcpyDeviceToHost));
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
            for (int l = 0; l < num_neuron_layers_mlp; ++l) { cudaFree(d_hor_activations[l]); cudaFree(d_ver_activations[l]); }
            for (int l = 0; l < num_weight_matrices_mlp; ++l) { // Deltas, Weights, GWeights (0 to N-2)
                cudaFree(d_hor_deltas[l]);
                cudaFree(d_hor_weights[l]); cudaFree(d_hor_gweights[l]);
                cudaFree(d_ver_deltas[l]); // Corrected: Should free d_ver_deltas[l]
                cudaFree(d_ver_weights[l]); cudaFree(d_ver_gweights[l]);
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
        for (int l = 0; l < num_neuron_layers_mlp; ++l) {
            cudaFree(d_hor_activations[l]); 
            cudaFree(d_ver_activations[l]); 
        }
        for (int l = 0; l < num_weight_matrices_mlp; ++l) {
            // cudaFree(d_hor_activations[l]); 
            cudaFree(d_hor_deltas[l]); // Corrected: Should free d_hor_deltas[l]
            cudaFree(d_hor_weights[l]); cudaFree(d_hor_gweights[l]);
            cudaFree(d_ver_deltas[l]);
            cudaFree(d_ver_weights[l]); cudaFree(d_ver_gweights[l]);
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
    const size_t mlp_weights_elements = static_cast<size_t>(embedding_dim) * embedding_dim; // Assuming square matrices for simplicity here
    const size_t mlp_weights_bytes = mlp_weights_elements * sizeof(float);
    const size_t proj_mat_elements = static_cast<size_t>(mat_heights) * embedding_dim; // For MV, MQ, MK
    const size_t proj_mat_bytes = proj_mat_elements * sizeof(float);
    const size_t ev_total_elements = static_cast<size_t>(context_win) * embedding_dim; // For full EV
    const size_t ev_total_bytes = ev_total_elements * sizeof(float);

    // Kernel Launch Config
    int threadsPerBlock1D = 256;
    dim3 blockDim1D(threadsPerBlock1D);
    int blocksPerGridEmbed = (embedding_dim + threadsPerBlock1D - 1) / threadsPerBlock1D;
    dim3 gridDimEmbed(blocksPerGridEmbed);
    int blocksPerGridMatHeights = (mat_heights + threadsPerBlock1D - 1) / threadsPerBlock1D;
    dim3 gridDimMatHeights(blocksPerGridMatHeights);
    int blocksPerGridProjMat = (static_cast<int>(proj_mat_elements) + threadsPerBlock1D - 1) / threadsPerBlock1D;
    dim3 gridDimProjMat(blocksPerGridProjMat);
    int blocksPerGridEV = (static_cast<int>(ev_total_elements) + threadsPerBlock1D - 1) / threadsPerBlock1D;
    dim3 gridDimEV(blocksPerGridEV); // For full EV updates

    dim3 blockDim2D(16, 16);
    dim3 gridDimEmbed2D((embedding_dim + blockDim2D.x - 1) / blockDim2D.x, (embedding_dim + blockDim2D.y - 1) / blockDim2D.y);
    dim3 gridDimMatrix2D((embedding_dim + blockDim2D.x - 1) / blockDim2D.x, (mat_heights + blockDim2D.y - 1) / blockDim2D.y);


    // Iterate through the rows (parallels/layers) in the specified column
    for (int i = 0; i < x; ++i) { // 'i' is the row index
        attention& head_obj = b[i][layno]; // Reference to the current head object
        std::vector<std::vector<float>>& expectedV_head = expectedV[i]; // Expected output for this head
        const int token_count = head_obj.tokenCount;
        const size_t active_head_elements = static_cast<size_t>(token_count) * token_count;
        const size_t active_head_bytes = active_head_elements * sizeof(float);
        const size_t active_k_q_elements = static_cast<size_t>(token_count) * mat_heights;
        const size_t active_k_q_bytes = active_k_q_elements * sizeof(float);
        // Adjust 2D grids based on token_count
        dim3 gridDimHead2D((token_count + blockDim2D.x - 1) / blockDim2D.x, (token_count + blockDim2D.y - 1) / blockDim2D.y);
        dim3 gridDimKQGrad2D((mat_heights + blockDim2D.x - 1) / blockDim2D.x, (token_count + blockDim2D.y - 1) / blockDim2D.y);
        int blocksPerGridHead = (static_cast<int>(active_head_elements) + threadsPerBlock1D - 1) / threadsPerBlock1D;
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
        std::vector<float*> d_ver_activations(layers, nullptr); // Size 'layers' (N) for layers 0 to N-1
        std::vector<float*> d_ver_weights(layers - 1, nullptr); // Size 'layers - 1' (N-1) for W[0] to W[N-2]
        std::vector<float*> d_ver_gweights(layers - 1, nullptr); // Size 'layers - 1' (N-1) for gW[0] to gW[N-2]
        std::vector<float*> d_ver_deltas(layers - 1, nullptr); // Size 'layers - 1' (N-1) for Delta[1] to Delta[N-1] (indices 0 to N-2)

        try {
            // --- Allocate Memory (Attention) ---
            CUDA_CHECK(cudaMalloc(&d_expected_v, ev_total_bytes)); // expectedV_head is flattened to this size
            CUDA_CHECK(cudaMalloc(&d_EV, ev_total_bytes));
            CUDA_CHECK(cudaMalloc(&d_grad_EV_full, ev_total_bytes));
            CUDA_CHECK(cudaMalloc(&d_grad_EV_summed, embed_bytes));
            CUDA_CHECK(cudaMalloc(&d_grad_EV_scaled, embed_bytes));
            CUDA_CHECK(cudaMalloc(&d_grad_dv, embed_bytes));
            CUDA_CHECK(cudaMalloc(&d_KdotQ, active_head_bytes));
            CUDA_CHECK(cudaMalloc(&d_head, active_head_bytes));
            CUDA_CHECK(cudaMalloc(&d_K, active_k_q_bytes));
            CUDA_CHECK(cudaMalloc(&d_Q, active_k_q_bytes));
            CUDA_CHECK(cudaMalloc(&d_pre_MV, mat_heights * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_MV_a, proj_mat_bytes));
            CUDA_CHECK(cudaMalloc(&d_MQ_a, proj_mat_bytes));
            CUDA_CHECK(cudaMalloc(&d_MK_a, proj_mat_bytes));
            CUDA_CHECK(cudaMalloc(&d_grad_MV, proj_mat_bytes));
            CUDA_CHECK(cudaMalloc(&d_grad_head, active_head_bytes));
            CUDA_CHECK(cudaMalloc(&d_lota_deriv, active_head_bytes));
            CUDA_CHECK(cudaMalloc(&d_grad_KdotQ, active_head_bytes));
            CUDA_CHECK(cudaMalloc(&d_grad_Q, active_k_q_bytes));
            CUDA_CHECK(cudaMalloc(&d_grad_MQ, proj_mat_bytes));
            CUDA_CHECK(cudaMalloc(&d_grad_MK_correction, proj_mat_bytes));

            // --- Allocate Memory (ver MLP Internals) ---
            // Activations are for each neuron layer (N of them)
            for (int l = 0; l < layers; ++l) { // Loop 0 to N-1
                CUDA_CHECK(cudaMalloc(&d_ver_activations[l], embed_bytes));
            }
            // Deltas, Weights, Gweights are between neuron layers (N-1 of them)
            for (int l = 0; l < layers - 1; ++l) {
                CUDA_CHECK(cudaMalloc(&d_ver_weights[l], mlp_weights_bytes));
                CUDA_CHECK(cudaMalloc(&d_ver_deltas[l], embed_bytes));
                CUDA_CHECK(cudaMalloc(&d_ver_gweights[l], mlp_weights_bytes));
            }

            // --- Data Transfer H->D (Attention) ---
            std::vector<float> flat_expectedV_head = flatten(expectedV_head);
            // Validate mat objects
            if (!head_obj.EV.mapped_data || !head_obj.K.mapped_data || !head_obj.Q.mapped_data || !head_obj.KdotQ.mapped_data ||
                !head_obj.MV.mapped_data || !head_obj.MQ.mapped_data || !head_obj.MK.mapped_data) {
                throw std::runtime_error("One or more attention mat objects have null mapped_data for head [" + std::to_string(i) + "][" + std::to_string(layno) + "]");
            }

            CUDA_CHECK(cudaMemcpy(d_expected_v, flat_expectedV_head.data(), ev_total_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_EV, head_obj.EV.mapped_data, ev_total_bytes, cudaMemcpyHostToDevice));
            // Copy active parts of K, Q, KdotQ
            CUDA_CHECK(cudaMemcpy(d_KdotQ, head_obj.KdotQ.mapped_data, active_head_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_K, head_obj.K.mapped_data, active_k_q_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_Q, head_obj.Q.mapped_data, active_k_q_bytes, cudaMemcpyHostToDevice));
            // Copy full MV, MQ, MK
            CUDA_CHECK(cudaMemcpy(d_MV_a, head_obj.MV.mapped_data, proj_mat_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_MQ_a, head_obj.MQ.mapped_data, proj_mat_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_MK_a, head_obj.MK.mapped_data, proj_mat_bytes, cudaMemcpyHostToDevice));

            // --- Data Transfer H->D (ver MLP Internals) ---
            for (int l = 0; l < layers; ++l) { // Activations (0 to N-1)
                if (head_obj.ver.activations[l].empty()) {
                    throw std::runtime_error("MLP ver.activations vector is empty for head [" + std::to_string(i) + "][" + std::to_string(layno) + "], layer " + std::to_string(l));
                }
                CUDA_CHECK(cudaMemcpy(d_ver_activations[l], head_obj.ver.activations[l].data(), embed_bytes, cudaMemcpyHostToDevice));
            }
            // Copy weights for all layers-1 weight matrices
            for (int l = 0; l < layers - 1; ++l) {
                // Validate ver.weights[l] and ver.gweights[l]
                if (!head_obj.ver.weights[l].mapped_data || head_obj.ver.weights[l].row != embedding_dim || head_obj.ver.weights[l].col != embedding_dim ||
                    !head_obj.ver.gweights[l].mapped_data || head_obj.ver.gweights[l].row != embedding_dim || head_obj.ver.gweights[l].col != embedding_dim) {
                    throw std::runtime_error("Invalid ver.weights/gweights mat for head [" + std::to_string(i) + "][" + std::to_string(layno) + "], layer " + std::to_string(l));
                }
                // CUDA_CHECK(cudaMemcpy(d_ver_activations[l], head_obj.ver.activations[l].data(), embed_bytes, cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_ver_weights[l], head_obj.ver.weights[l].mapped_data, mlp_weights_bytes, cudaMemcpyHostToDevice));
            }

            // --- Backpropagation Steps ---

            // Step 1: Compute grad_EV (full, summed, scaled)
            kernelComputeGradientsEV_V<<<gridDimEV, blockDim1D>>>(d_EV, d_expected_v, d_grad_EV_full, d_grad_EV_summed, d_grad_EV_scaled, learning_rate, context_win, embedding_dim);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            // --- Step 2: Backprop through ver MLP ---
            // Calculate Output Layer Deltas (Layer N-1, stored in d_ver_deltas[N-2])
            // The input gradient to the MLP is d_grad_EV_scaled (or d_grad_EV_summed if scaling/LR is handled differently)
            kernelLastLayerDelta<<<gridDimEmbed, blockDim1D>>>(d_grad_EV_scaled, d_ver_activations[layers - 1], d_ver_deltas[layers - 2], embedding_dim);
            CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
            for (int k = layers - 2; k >= 1; --k) { // k is the neuron layer index (1 to N-2)
                hiddenDeltaKernel<<<gridDimEmbed, blockDim1D>>>(d_ver_deltas[k], d_ver_weights[k], d_ver_activations[k], d_ver_deltas[k-1], embedding_dim, embedding_dim);
                CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
            }
            // Calculate Gradients and Update Weights (Iterate through layers 0 to layers-1)
            // d_ver_activations[0] is the input to the ver MLP.
            // W[l] connects activations[l] to activations[l+1]. gW[l] = delta[l+1] * activations[l]^T
            for (int l = 0; l < layers - 1; ++l) { // layers-1 weight matrices
                // float* d_input_activation_to_Wl = d_ver_activations[l];
                // float* d_output_delta_of_Wl = d_ver_deltas[l]; // Delta for layer l+1 is at index l
                updateWeightsKernel<<<gridDimEmbed2D, blockDim2D>>>(d_ver_deltas[l], d_ver_activations[l], d_ver_weights[l], d_ver_gweights[l], learning_rate, embedding_dim, embedding_dim);
                CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
            }

            // --- Step 3: Compute grad_dv ---
            kernelComputeGradMLPInput<<<gridDimEmbed, blockDim1D>>>(d_ver_deltas[0], d_ver_weights[0], d_grad_dv, embedding_dim, embedding_dim); // Delta for layer 1 is at index 0, W[0] is at index 0
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
            kernelComputeGradKdotQ_LOTA<<<gridDimHead, blockDim1D>>>(d_grad_head, d_lota_deriv, d_grad_KdotQ, scaling_factor, static_cast<int>(active_head_elements));
            CUDA_CHECK(cudaGetLastError());

            // --- Step 7: Compute grad_Q ---
            kernelComputeGradQ_V<<<gridDimKQGrad2D, blockDim2D>>>(d_grad_KdotQ, d_K, d_grad_Q, token_count, mat_heights);
            CUDA_CHECK(cudaGetLastError());

            // --- Step 8: Compute grad_MQ and grad_MK_correction ---
            kernelComputeGradMQ_V<<<gridDimMatrix2D, blockDim2D>>>(d_grad_Q, nullptr, d_grad_MQ, token_count, mat_heights, embedding_dim);
            CUDA_CHECK(cudaGetLastError());
            kernelComputeGradMKCorrection<<<gridDimMatrix2D, blockDim2D>>>(d_grad_MQ, d_Q, d_K, d_grad_MK_correction, token_count, mat_heights, embedding_dim);
            CUDA_CHECK(cudaGetLastError());

            // --- Step 9 & 10: Update Weights ---
            if (is_first_head) {
                // Use kernelUpdateWeights_1stHead_V (updates MV, MQ, MK_correction)
                kernelUpdateWeights_1stHead_V<<<gridDimProjMat, blockDim1D>>>(
                    d_MV_a, d_MQ_a, d_MK_a,
                    d_grad_MV, d_grad_MQ, d_grad_MK_correction,
                    learning_rate, mat_heights, embedding_dim
                );
                CUDA_CHECK(cudaGetLastError());
                // No EV update in kernelUpdateWeights_1stHead_V
            }
            else {
                // Use simpler update (updates MV, MQ, MK_correction, EV)
                kernelUpdateSimple<<<gridDimProjMat, blockDim1D>>>(d_MV_a, d_grad_MV, learning_rate, proj_mat_elements);
                kernelUpdateSimple<<<gridDimProjMat, blockDim1D>>>(d_MQ_a, d_grad_MQ, learning_rate, proj_mat_elements);
                kernelUpdateSimple<<<gridDimProjMat, blockDim1D>>>(d_MK_a, d_grad_MK_correction, learning_rate, proj_mat_elements); // Update MK with correction
                CUDA_CHECK(cudaGetLastError());
                // Update EV using full gradient
                kernelUpdateSimple<<<gridDimEV, blockDim1D>>>(d_EV, d_grad_EV_full, learning_rate, ev_total_elements);
                CUDA_CHECK(cudaGetLastError());
            }
            CUDA_CHECK(cudaDeviceSynchronize());

            // --- Data Transfer D->H ---
            // Copy updated ver MLP weights and gradients back
            for (int l = 0; l < layers - 1; ++l) { // layers-1 weight matrices
                CUDA_CHECK(cudaMemcpy(head_obj.ver.weights[l].mapped_data, d_ver_weights[l], mlp_weights_bytes, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(head_obj.ver.gweights[l].mapped_data, d_ver_gweights[l], mlp_weights_bytes, cudaMemcpyDeviceToHost));
            }
            // Note: MLP activations and deltas are intermediate and usually not copied back.
            // Copy updated Attention parameters back
            CUDA_CHECK(cudaMemcpy(head_obj.EV.mapped_data, d_EV, ev_total_bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(head_obj.MV.mapped_data, d_MV_a, proj_mat_bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(head_obj.MQ.mapped_data, d_MQ_a, proj_mat_bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(head_obj.MK.mapped_data, d_MK_a, proj_mat_bytes, cudaMemcpyDeviceToHost));
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
            for (int l = 0; l < layers; ++l) { // Activations and Deltas
                cudaFree(d_ver_activations[l]);
            }
            for (int l = 0; l < layers - 1; ++l) { // Deltas, Weights, GWeights (0 to N-2)
                cudaFree(d_ver_weights[l]); cudaFree(d_ver_gweights[l]);
                cudaFree(d_ver_deltas[l]);

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
        for (int l = 0; l < layers; ++l) { // Activations and Deltas
            cudaFree(d_ver_activations[l]); 
        }
        for (int l = 0; l < layers - 1; ++l) { // Deltas, Weights, GWeights (0 to N-2)
            cudaFree(d_ver_weights[l]); cudaFree(d_ver_gweights[l]);
            cudaFree(d_ver_deltas[l]);
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
    const size_t mlp_weights_elements = static_cast<size_t>(embedding_dim) * embedding_dim; // Assuming square matrices
    const size_t mlp_weights_bytes = mlp_weights_elements * sizeof(float);
    const size_t proj_mat_elements = static_cast<size_t>(mat_heights) * embedding_dim; // For MH, MV, MQ, MK
    const size_t proj_mat_bytes = proj_mat_elements * sizeof(float);
    const size_t ev_total_elements = static_cast<size_t>(context_win) * embedding_dim; // For EV
    const size_t ev_total_bytes = ev_total_elements * sizeof(float);

    // Kernel Launch Config
    int threadsPerBlock1D = 256;
    dim3 blockDim1D(threadsPerBlock1D);
    int blocksPerGridEmbed = (embedding_dim + threadsPerBlock1D - 1) / threadsPerBlock1D;
    dim3 gridDimEmbed(blocksPerGridEmbed);
    int blocksPerGridMatHeights = (mat_heights + threadsPerBlock1D - 1) / threadsPerBlock1D;
    dim3 gridDimMatHeights(blocksPerGridMatHeights);
    int blocksPerGridProjMat = (static_cast<int>(proj_mat_elements) + threadsPerBlock1D - 1) / threadsPerBlock1D;
    dim3 gridDimProjMat(blocksPerGridProjMat);

    dim3 blockDim2D(16, 16);
    dim3 gridDimEmbed2D((embedding_dim + blockDim2D.x - 1) / blockDim2D.x, (embedding_dim + blockDim2D.y - 1) / blockDim2D.y);
    dim3 gridDimMatrix2D((embedding_dim + blockDim2D.x - 1) / blockDim2D.x, (mat_heights + blockDim2D.y - 1) / blockDim2D.y);

    // Iterate backwards through the rows
    for (int i = x - 1; i >= 0; --i) {
        attention& head_obj = b[i][layno];
        const int token_count = head_obj.tokenCount;
        const size_t active_head_elements = static_cast<size_t>(token_count) * token_count;
        const size_t active_head_bytes = active_head_elements * sizeof(float);
        const size_t active_k_q_elements = static_cast<size_t>(token_count) * mat_heights;
        const size_t active_k_q_bytes = active_k_q_elements * sizeof(float);
        dim3 gridDimHead2D((token_count + blockDim2D.x - 1) / blockDim2D.x, (token_count + blockDim2D.y - 1) / blockDim2D.y);
        dim3 gridDimKQGrad2D((mat_heights + blockDim2D.x - 1) / blockDim2D.x, (token_count + blockDim2D.y - 1) / blockDim2D.y);
        int blocksPerGridHead = (static_cast<int>(active_head_elements) + threadsPerBlock1D - 1) / threadsPerBlock1D;
        dim3 gridDimHead(blocksPerGridHead);

        // --- Device Pointers (Allocate per head) ---
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
        std::vector<float*> d_hor_weights(layers-1, nullptr); // Size N-1
        std::vector<float*> d_hor_gweights(layers-1, nullptr); // Size N-1
        std::vector<float*> d_hor_deltas(layers-1, nullptr); // Size N-1
        std::vector<float*> d_ver_activations(layers, nullptr);
        std::vector<float*> d_ver_weights(layers-1, nullptr); // Size N-1
        std::vector<float*> d_ver_gweights(layers-1, nullptr); // Size N-1
        std::vector<float*> d_ver_deltas(layers-1, nullptr); // Size N-1

        try {
            // --- Allocate Memory ---
            CUDA_CHECK(cudaMalloc(&d_expected_h, embed_bytes));
            CUDA_CHECK(cudaMalloc(&d_EH, embed_bytes));
            CUDA_CHECK(cudaMalloc(&d_EV, ev_total_bytes));
            CUDA_CHECK(cudaMalloc(&d_grad_EH, embed_bytes));
            CUDA_CHECK(cudaMalloc(&d_grad_EV_scaled, embed_bytes));
            CUDA_CHECK(cudaMalloc(&d_grad_dh, embed_bytes));
            CUDA_CHECK(cudaMalloc(&d_grad_dv, embed_bytes));
            CUDA_CHECK(cudaMalloc(&d_KdotQ, active_head_bytes));
            CUDA_CHECK(cudaMalloc(&d_head, active_head_bytes));
            CUDA_CHECK(cudaMalloc(&d_K, active_k_q_bytes));
            CUDA_CHECK(cudaMalloc(&d_Q, active_k_q_bytes));
            CUDA_CHECK(cudaMalloc(&d_pre_MH, mat_heights * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_pre_MV, mat_heights * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_MH_a, proj_mat_bytes));
            CUDA_CHECK(cudaMalloc(&d_MV_a, proj_mat_bytes));
            CUDA_CHECK(cudaMalloc(&d_MQ_a, proj_mat_bytes));
            CUDA_CHECK(cudaMalloc(&d_MK_a, proj_mat_bytes));
            CUDA_CHECK(cudaMalloc(&d_grad_MH, proj_mat_bytes));
            CUDA_CHECK(cudaMalloc(&d_grad_MV, proj_mat_bytes));
            CUDA_CHECK(cudaMalloc(&d_grad_head, active_head_bytes));
            CUDA_CHECK(cudaMalloc(&d_lota_deriv, active_head_bytes));
            CUDA_CHECK(cudaMalloc(&d_grad_KdotQ, active_head_bytes));
            CUDA_CHECK(cudaMalloc(&d_grad_K, active_k_q_bytes));
            CUDA_CHECK(cudaMalloc(&d_grad_Q, active_k_q_bytes));
            CUDA_CHECK(cudaMalloc(&d_grad_MQ, proj_mat_bytes));
            CUDA_CHECK(cudaMalloc(&d_grad_MK, proj_mat_bytes));
            // Activations and Deltas are for each neuron layer (layers of them)
            // MLP Internals Allocation (N total layers)
            for (int l = 0; l < layers; ++l) { // Activations (0 to N-1)
                CUDA_CHECK(cudaMalloc(&d_hor_activations[l], embed_bytes));
                CUDA_CHECK(cudaMalloc(&d_ver_activations[l], embed_bytes));
            }
            // Deltas, Weights, Gweights are between neuron layers (N-1 of them)
            for (int l = 0; l < layers - 1; ++l) {
                CUDA_CHECK(cudaMalloc(&d_hor_deltas[l], embed_bytes));
                CUDA_CHECK(cudaMalloc(&d_hor_weights[l], mlp_weights_bytes));
                CUDA_CHECK(cudaMalloc(&d_hor_gweights[l], mlp_weights_bytes));
                CUDA_CHECK(cudaMalloc(&d_ver_weights[l], mlp_weights_bytes));
                CUDA_CHECK(cudaMalloc(&d_ver_gweights[l], mlp_weights_bytes));
            }

            // --- Data Transfer H->D ---
            // (Identical transfers as in cu1ParallelBackward1stBlock(H))
            if (!head_obj.EV.mapped_data || !head_obj.K.mapped_data || !head_obj.Q.mapped_data || !head_obj.KdotQ.mapped_data ||
                !head_obj.MH.mapped_data || !head_obj.MV.mapped_data || !head_obj.MQ.mapped_data || !head_obj.MK.mapped_data) {
                throw std::runtime_error("One or more attention mat objects have null mapped_data for head [" + std::to_string(i) + "][" + std::to_string(layno) + "]");
            }
            CUDA_CHECK(cudaMemcpy(d_expected_h, expectedH.data(), embed_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_EH, head_obj.EH.data(), embed_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_EV, head_obj.EV.mapped_data, ev_total_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_MH_a, head_obj.MH.mapped_data, proj_mat_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_MV_a, head_obj.MV.mapped_data, proj_mat_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_MQ_a, head_obj.MQ.mapped_data, proj_mat_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_MK_a, head_obj.MK.mapped_data, proj_mat_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_KdotQ, head_obj.KdotQ.mapped_data, active_head_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_K, head_obj.K.mapped_data, active_k_q_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_Q, head_obj.Q.mapped_data, active_k_q_bytes, cudaMemcpyHostToDevice));
            // MLP Internals Data Transfer H->D
            for (int l = 0; l < layers; ++l) {
                if (head_obj.hor.activations[l].empty() || head_obj.ver.activations[l].empty()) { // Activations (0 to N-1)
                    throw std::runtime_error("MLP activation vector is empty for head [" + std::to_string(i) + "][" + std::to_string(layno) + "], layer " + std::to_string(l));
                }
                CUDA_CHECK(cudaMemcpy(d_hor_activations[l], head_obj.hor.activations[l].data(), embed_bytes, cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_ver_activations[l], head_obj.ver.activations[l].data(), embed_bytes, cudaMemcpyHostToDevice));
            }
            for (int l = 0; l < layers - 1; ++l) { // Weights (0 to N-2)
                if (!head_obj.hor.weights[l].mapped_data || head_obj.hor.weights[l].row != embedding_dim || head_obj.hor.weights[l].col != embedding_dim ||
                    !head_obj.ver.weights[l].mapped_data || head_obj.ver.weights[l].row != embedding_dim || head_obj.ver.weights[l].col != embedding_dim ||
                    !head_obj.hor.gweights[l].mapped_data || head_obj.hor.gweights[l].row != embedding_dim || head_obj.hor.gweights[l].col != embedding_dim ||
                    !head_obj.ver.gweights[l].mapped_data || head_obj.ver.gweights[l].row != embedding_dim || head_obj.ver.gweights[l].col != embedding_dim) {
                    throw std::runtime_error("Invalid MLP weight/gweight mat for head [" + std::to_string(i) + "][" + std::to_string(layno) + "], layer " + std::to_string(l));
                }
                // CUDA_CHECK(cudaMemcpy(d_hor_activations[l], head_obj.hor.activations[l].data(), embed_bytes, cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_hor_weights[l], head_obj.hor.weights[l].mapped_data, mlp_weights_bytes, cudaMemcpyHostToDevice));
                // CUDA_CHECK(cudaMemcpy(d_ver_activations[l], head_obj.ver.activations[l].data(), embed_bytes, cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_ver_weights[l], head_obj.ver.weights[l].mapped_data, mlp_weights_bytes, cudaMemcpyHostToDevice));
            }
            // Deltas are not copied from host, they are calculated on device.
            // Gweights are not copied from host, they are updated on device.

            // --- Backpropagation Steps ---
            // (Identical steps 1-8 as in cu1ParallelBackward1stBlock(H))
            // Step 1: Compute grad_EH and grad_EV_scaled
            kernelComputeGradientsEH_EV<<<gridDimEmbed, blockDim1D>>>(d_EH, d_expected_h, d_grad_EH, d_grad_EV_scaled, embedding_dim);
            CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
            // Step 2: Backprop through MLPs
            // 2a: Hor MLP
            kernelLastLayerDelta<<<gridDimEmbed, blockDim1D>>>(d_grad_EH, d_hor_activations[layers - 1], d_hor_deltas[layers - 2], embedding_dim); // Delta for layer N-1 is at index N-2
            CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
            for (int k = layers - 2; k >= 1; --k) { // k is the neuron layer index (1 to N-2)
                hiddenDeltaKernel<<<gridDimEmbed, blockDim1D>>>(d_hor_deltas[k], d_hor_weights[k], d_hor_activations[k], d_hor_deltas[k-1], embedding_dim, embedding_dim);
                CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
            }
            for (int l = 0; l < layers - 1; ++l) {
                // Assuming d_hor_activations[0] is the input to the MLP (e.g., d_EH if head_obj.hor.activations[0] was EH)
                updateWeightsKernel<<<gridDimEmbed2D, blockDim2D>>>(d_hor_deltas[l], d_hor_activations[l], d_hor_weights[l], d_hor_gweights[l], learning_rate, embedding_dim, embedding_dim); // Delta for layer l+1 is at index l
                CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
            }
            // 2b: Ver MLP
            kernelLastLayerDelta<<<gridDimEmbed, blockDim1D>>>(d_grad_EV_scaled, d_ver_activations[layers - 1], d_ver_deltas[layers - 2], embedding_dim); // Delta for layer N-1 is at index N-2
            CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
            for (int k = layers - 2; k >= 1; --k) { // k is the neuron layer index (1 to N-2)
                hiddenDeltaKernel<<<gridDimEmbed, blockDim1D>>>(d_ver_deltas[k], d_ver_weights[k], d_ver_activations[k], d_ver_deltas[k-1], embedding_dim, embedding_dim);
                CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
            }
            for (int l = 0; l < layers - 1; ++l) {
                // Assuming d_ver_activations[0] is the input to the ver MLP
                updateWeightsKernel<<<gridDimEmbed2D, blockDim2D>>>(d_ver_deltas[l], d_ver_activations[l], d_ver_weights[l], d_ver_gweights[l], learning_rate, embedding_dim, embedding_dim); // Delta for layer l+1 is at index l
                CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
            }
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
            kernelComputeGradKdotQ_LOTA<<<gridDimHead, blockDim1D>>>(d_grad_head, d_lota_deriv, d_grad_KdotQ, scaling_factor, static_cast<int>(active_head_elements)); CUDA_CHECK(cudaGetLastError());
            // Step 7: Compute grad_K and grad_Q
            kernelComputeGradK_Q<<<gridDimKQGrad2D, blockDim2D>>>(d_grad_KdotQ, d_K, d_Q, d_grad_K, d_grad_Q, token_count, mat_heights); CUDA_CHECK(cudaGetLastError());
            // Step 8: Compute grad_MK and grad_MQ (Simplified)
            kernelComputeGradMK_MQ_Simplified<<<gridDimMatrix2D, blockDim2D>>>(d_grad_K, d_grad_Q, nullptr, nullptr, d_grad_MK, d_grad_MQ, token_count, mat_heights, embedding_dim); CUDA_CHECK(cudaGetLastError());

            // --- Step 9 & 10: Update Weights (General Case) ---
            kernelUpdateSimple<<<gridDimProjMat, blockDim1D>>>(d_MH_a, d_grad_MH, learning_rate, proj_mat_elements);
            kernelUpdateSimple<<<gridDimProjMat, blockDim1D>>>(d_MV_a, d_grad_MV, learning_rate, proj_mat_elements);
            kernelUpdateSimple<<<gridDimProjMat, blockDim1D>>>(d_MQ_a, d_grad_MQ, learning_rate, proj_mat_elements);
            kernelUpdateSimple<<<gridDimProjMat, blockDim1D>>>(d_MK_a, d_grad_MK, learning_rate, proj_mat_elements);
            CUDA_CHECK(cudaGetLastError());
            kernelUpdateSimple<<<gridDimEmbed, blockDim1D>>>(d_EH, d_grad_dh, learning_rate, embedding_dim);
            CUDA_CHECK(cudaGetLastError());
            // Update the relevant part of EV (e.g., first embedding_dim elements if d_EV points to a specific row)
            kernelUpdateSimple<<<gridDimEmbed, blockDim1D>>>(d_EV, d_grad_dv, learning_rate, embedding_dim); // Example EV update (partial)
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            // --- Data Transfer D->H ---
            for (int l = 0; l < layers-1; ++l) {
                CUDA_CHECK(cudaMemcpy(head_obj.hor.weights[l].mapped_data, d_hor_weights[l], mlp_weights_bytes, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(head_obj.hor.gweights[l].mapped_data, d_hor_gweights[l], mlp_weights_bytes, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(head_obj.ver.weights[l].mapped_data, d_ver_weights[l], mlp_weights_bytes, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(head_obj.ver.gweights[l].mapped_data, d_ver_gweights[l], mlp_weights_bytes, cudaMemcpyDeviceToHost));
            }
            CUDA_CHECK(cudaMemcpy(head_obj.EH.data(), d_EH, embed_bytes, cudaMemcpyDeviceToHost));
            // CUDA_CHECK(cudaMemcpy(head_obj.EV.mapped_data, d_EV, ev_total_bytes, cudaMemcpyDeviceToHost));
            // For non-first block, EV update is tricky. The current kernelUpdateSimple updates only a slice.
            // If full EV update is needed based on d_grad_dv (which is embed_dim), a broadcasted update is required.
            kernelUpdateEVBroadcasted<<< (context_win + threadsPerBlock1D -1) / threadsPerBlock1D, blockDim1D >>>(d_EV, d_grad_dv, learning_rate, context_win, embedding_dim);
            CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(head_obj.EV.mapped_data, d_EV, ev_total_bytes, cudaMemcpyDeviceToHost)); // Copy EV after potential update
            CUDA_CHECK(cudaMemcpy(head_obj.MH.mapped_data, d_MH_a, proj_mat_bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(head_obj.MV.mapped_data, d_MV_a, proj_mat_bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(head_obj.MQ.mapped_data, d_MQ_a, proj_mat_bytes, cudaMemcpyDeviceToHost));

        }
        catch (const std::exception& e) {
            std::cerr << "Error during cu1ParallelBackward (H) for head [" << i << "][" << layno << "]: " << e.what() << std::endl;
            // Cleanup allocated memory on error
            cudaFree(d_expected_h); cudaFree(d_EH); cudaFree(d_EV); cudaFree(d_grad_EH); cudaFree(d_grad_EV_scaled); cudaFree(d_grad_dh); cudaFree(d_grad_dv); cudaFree(d_KdotQ); cudaFree(d_head); cudaFree(d_K); cudaFree(d_Q); cudaFree(d_pre_MH); cudaFree(d_pre_MV); cudaFree(d_MH_a); cudaFree(d_MV_a); cudaFree(d_MQ_a); cudaFree(d_MK_a); cudaFree(d_grad_MH); cudaFree(d_grad_MV); cudaFree(d_grad_head); cudaFree(d_lota_deriv); cudaFree(d_grad_KdotQ); cudaFree(d_grad_K); cudaFree(d_grad_Q); cudaFree(d_grad_MQ); cudaFree(d_grad_MK); // Free Attention/Gradient buffers
            for (int l = 0; l < layers; ++l) { // Free Activations (0 to N-1)
                cudaFree(d_hor_activations[l]); cudaFree(d_ver_activations[l]);
            }
            for (int l = 0; l < layers - 1; ++l) {
                cudaFree(d_hor_weights[l]); cudaFree(d_hor_gweights[l]);
                cudaFree(d_ver_weights[l]); cudaFree(d_ver_gweights[l]);
            }
            throw std::runtime_error("Exception processing head [" + std::to_string(i) + "][" + std::to_string(layno) + "]: " + e.what());
        }

        // --- Cleanup Device Memory (Success Case - Per Head) ---
        cudaFree(d_expected_h); cudaFree(d_EH); cudaFree(d_EV); cudaFree(d_grad_EH); cudaFree(d_grad_EV_scaled); cudaFree(d_grad_dh); cudaFree(d_grad_dv); cudaFree(d_KdotQ); cudaFree(d_head); cudaFree(d_K); cudaFree(d_Q); cudaFree(d_pre_MH); cudaFree(d_pre_MV); cudaFree(d_MH_a); cudaFree(d_MV_a); cudaFree(d_MQ_a); cudaFree(d_MK_a); cudaFree(d_grad_MH); cudaFree(d_grad_MV); cudaFree(d_grad_head); cudaFree(d_lota_deriv); cudaFree(d_grad_KdotQ); cudaFree(d_grad_K); cudaFree(d_grad_Q); cudaFree(d_grad_MQ); cudaFree(d_grad_MK);
        for (int l = 0; l < layers; ++l) {
            cudaFree(d_hor_activations[l]); cudaFree(d_hor_deltas[l]);
            cudaFree(d_ver_activations[l]);
        }
        for (int l = 0; l < layers - 1; ++l) {
            cudaFree(d_hor_deltas[l]); cudaFree(d_hor_weights[l]); cudaFree(d_hor_gweights[l]);
            cudaFree(d_ver_weights[l]); cudaFree(d_ver_gweights[l]);
        }
    }
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
    const size_t mlp_weights_elements = static_cast<size_t>(embedding_dim) * embedding_dim; // Assuming square matrices
    const size_t mlp_weights_bytes = mlp_weights_elements * sizeof(float);
    const size_t proj_mat_elements = static_cast<size_t>(mat_heights) * embedding_dim; // For MV, MQ, MK (assuming mat_heights x embedding_dim)
    const size_t proj_mat_bytes = proj_mat_elements * sizeof(float);
    const size_t ev_total_elements = static_cast<size_t>(context_win) * embedding_dim; // For EV
    const size_t ev_total_bytes = ev_total_elements * sizeof(float);

    // Kernel Launch Config
    int threadsPerBlock1D = 256;
    dim3 blockDim1D(threadsPerBlock1D);
    int blocksPerGridEmbed = (embedding_dim + threadsPerBlock1D - 1) / threadsPerBlock1D;
    dim3 gridDimEmbed(blocksPerGridEmbed);
    int blocksPerGridMatHeights = (mat_heights + threadsPerBlock1D - 1) / threadsPerBlock1D;
    dim3 gridDimMatHeights(blocksPerGridMatHeights);
    int blocksPerGridProjMat = (static_cast<int>(proj_mat_elements) + threadsPerBlock1D - 1) / threadsPerBlock1D;
    dim3 gridDimProjMat(blocksPerGridProjMat);
    int blocksPerGridEV = (static_cast<int>(ev_total_elements) + threadsPerBlock1D - 1) / threadsPerBlock1D;
    dim3 gridDimEV(blocksPerGridEV); // For full EV updates

    dim3 blockDim2D(16, 16);
    dim3 gridDimEmbed2D((embedding_dim + blockDim2D.x - 1) / blockDim2D.x, (embedding_dim + blockDim2D.y - 1) / blockDim2D.y);
    dim3 gridDimMatrix2D((embedding_dim + blockDim2D.x - 1) / blockDim2D.x, (mat_heights + blockDim2D.y - 1) / blockDim2D.y);

    // Iterate through the rows
    for (int i = 0; i < x; ++i) {
        attention& head_obj = b[i][layno];
        std::vector<std::vector<float>>& expectedV_head = expectedV[i];
        const int token_count = head_obj.tokenCount;
        const size_t active_head_elements = static_cast<size_t>(token_count) * token_count;
        const size_t active_head_bytes = active_head_elements * sizeof(float);
        const size_t active_k_q_elements = static_cast<size_t>(token_count) * mat_heights;
        const size_t active_k_q_bytes = active_k_q_elements * sizeof(float);
        dim3 gridDimHead2D((token_count + blockDim2D.x - 1) / blockDim2D.x, (token_count + blockDim2D.y - 1) / blockDim2D.y);
        dim3 gridDimKQGrad2D((mat_heights + blockDim2D.x - 1) / blockDim2D.x, (token_count + blockDim2D.y - 1) / blockDim2D.y);
        int blocksPerGridHead = (static_cast<int>(active_head_elements) + threadsPerBlock1D - 1) / threadsPerBlock1D;
        dim3 gridDimHead(blocksPerGridHead);

        // --- Device Pointers (Allocate per head) ---
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
        std::vector<float*> d_ver_weights(layers-1, nullptr); // Size N-1
        std::vector<float*> d_ver_gweights(layers-1, nullptr); // Size N-1
        std::vector<float*> d_ver_deltas(layers-1, nullptr); // Size N-1

        try {
            // --- Allocate Memory ---
            CUDA_CHECK(cudaMalloc(&d_expected_v, ev_total_bytes));
            CUDA_CHECK(cudaMalloc(&d_EV, ev_total_bytes));
            CUDA_CHECK(cudaMalloc(&d_grad_EV_full, ev_total_bytes));
            CUDA_CHECK(cudaMalloc(&d_grad_EV_summed, embed_bytes));
            CUDA_CHECK(cudaMalloc(&d_grad_EV_scaled, embed_bytes));
            CUDA_CHECK(cudaMalloc(&d_grad_dv, embed_bytes));
            CUDA_CHECK(cudaMalloc(&d_KdotQ, active_head_bytes));
            CUDA_CHECK(cudaMalloc(&d_head, active_head_bytes));
            CUDA_CHECK(cudaMalloc(&d_K, active_k_q_bytes));
            CUDA_CHECK(cudaMalloc(&d_Q, active_k_q_bytes));
            CUDA_CHECK(cudaMalloc(&d_pre_MV, mat_heights * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_MV_a, proj_mat_bytes));
            CUDA_CHECK(cudaMalloc(&d_MQ_a, proj_mat_bytes));
            CUDA_CHECK(cudaMalloc(&d_MK_a, proj_mat_bytes));
            CUDA_CHECK(cudaMalloc(&d_grad_MV, proj_mat_bytes));
            CUDA_CHECK(cudaMalloc(&d_grad_head, active_head_bytes));
            CUDA_CHECK(cudaMalloc(&d_lota_deriv, active_head_bytes));
            CUDA_CHECK(cudaMalloc(&d_grad_KdotQ, active_head_bytes));
            CUDA_CHECK(cudaMalloc(&d_grad_Q, active_k_q_bytes));
            CUDA_CHECK(cudaMalloc(&d_grad_MQ, proj_mat_bytes));
            CUDA_CHECK(cudaMalloc(&d_grad_MK_correction, proj_mat_bytes));
            // MLP Internals Allocation
            for (int l = 0; l < layers; ++l) { // Activations (0 to N-1)
                CUDA_CHECK(cudaMalloc(&d_ver_activations[l], embed_bytes));
            }
            // Deltas, Weights, Gweights are between neuron layers (N-1 of them)
            for (int l = 0; l < layers - 1; ++l) {
                CUDA_CHECK(cudaMalloc(&d_ver_weights[l], mlp_weights_bytes));
                CUDA_CHECK(cudaMalloc(&d_ver_gweights[l], mlp_weights_bytes));
            }

            // --- Data Transfer H->D ---
            std::vector<float> flat_expectedV_head = flatten(expectedV_head);
            if (!head_obj.EV.mapped_data || !head_obj.K.mapped_data || !head_obj.Q.mapped_data || !head_obj.KdotQ.mapped_data ||
                !head_obj.MV.mapped_data || !head_obj.MQ.mapped_data || !head_obj.MK.mapped_data) {
                throw std::runtime_error("One or more attention mat objects have null mapped_data for head [" + std::to_string(i) + "][" + std::to_string(layno) + "]");
            }
            CUDA_CHECK(cudaMemcpy(d_expected_v, flat_expectedV_head.data(), ev_total_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_EV, head_obj.EV.mapped_data, ev_total_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_KdotQ, head_obj.KdotQ.mapped_data, active_head_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_K, head_obj.K.mapped_data, active_k_q_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_Q, head_obj.Q.mapped_data, active_k_q_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_MV_a, head_obj.MV.mapped_data, proj_mat_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_MQ_a, head_obj.MQ.mapped_data, proj_mat_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_MK_a, head_obj.MK.mapped_data, proj_mat_bytes, cudaMemcpyHostToDevice));
            // MLP Internals Data Transfer H->D
            for (int l = 0; l < layers; ++l) { // Activations (0 to N-1)
                if (head_obj.ver.activations[l].empty()) {
                    throw std::runtime_error("MLP ver.activations vector is empty for head [" + std::to_string(i) + "][" + std::to_string(layno) + "], layer " + std::to_string(l));
                }
                CUDA_CHECK(cudaMemcpy(d_ver_activations[l], head_obj.ver.activations[l].data(), embed_bytes, cudaMemcpyHostToDevice));
            }
            for (int l = 0; l < layers - 1; ++l) {
                if (!head_obj.ver.weights[l].mapped_data || head_obj.ver.weights[l].row != embedding_dim || head_obj.ver.weights[l].col != embedding_dim ||
                    !head_obj.ver.gweights[l].mapped_data || head_obj.ver.gweights[l].row != embedding_dim || head_obj.ver.gweights[l].col != embedding_dim) {
                    throw std::runtime_error("Invalid ver.weights/gweights mat for head [" + std::to_string(i) + "][" + std::to_string(layno) + "], layer " + std::to_string(l));
                } 
                // Activations are copied in the previous loop
                CUDA_CHECK(cudaMemcpy(d_ver_weights[l], head_obj.ver.weights[l].mapped_data, mlp_weights_bytes, cudaMemcpyHostToDevice));
            }

            // --- Backpropagation Steps ---
            // Step 1: Compute grad_EV
            kernelComputeGradientsEV_V<<<gridDimEV, blockDim1D>>>(d_EV, d_expected_v, d_grad_EV_full, d_grad_EV_summed, d_grad_EV_scaled, learning_rate, context_win, embedding_dim); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
            // Step 2: Backprop through ver MLP
            kernelLastLayerDelta<<<gridDimEmbed, blockDim1D>>>(d_grad_EV_scaled, d_ver_activations[layers - 1], d_ver_deltas[layers - 2], embedding_dim); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize()); // Delta for layer N-1 is at index N-2
            for (int k = layers - 2; k >= 1; --k) { // k is the neuron layer index (1 to N-2)
                hiddenDeltaKernel<<<gridDimEmbed, blockDim1D>>>(d_ver_deltas[k], d_ver_weights[k], d_ver_activations[k], d_ver_deltas[k-1], embedding_dim, embedding_dim);
                CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
            }
            for (int l = 0; l < layers - 1; ++l) {
                updateWeightsKernel<<<gridDimEmbed2D, blockDim2D>>>(d_ver_deltas[l], d_ver_activations[l], d_ver_weights[l], d_ver_gweights[l], learning_rate, embedding_dim, embedding_dim); // Delta for layer l+1 is at index l
                CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
            }
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
            kernelComputeGradKdotQ_LOTA<<<gridDimHead, blockDim1D>>>(d_grad_head, d_lota_deriv, d_grad_KdotQ, scaling_factor, static_cast<int>(active_head_elements)); CUDA_CHECK(cudaGetLastError());
            // Step 7: Compute grad_Q
            kernelComputeGradQ_V<<<gridDimKQGrad2D, blockDim2D>>>(d_grad_KdotQ, d_K, d_grad_Q, token_count, mat_heights); CUDA_CHECK(cudaGetLastError());
            // Step 8: Compute grad_MQ and grad_MK_correction
            kernelComputeGradMQ_V<<<gridDimMatrix2D, blockDim2D>>>(d_grad_Q, nullptr, d_grad_MQ, token_count, mat_heights, embedding_dim); CUDA_CHECK(cudaGetLastError());
            kernelComputeGradMKCorrection<<<gridDimMatrix2D, blockDim2D>>>(d_grad_MQ, d_Q, d_K, d_grad_MK_correction, token_count, mat_heights, embedding_dim); CUDA_CHECK(cudaGetLastError());

            // --- Step 9 & 10: Update Weights (General Case) ---
            kernelUpdateSimple<<<gridDimProjMat, blockDim1D>>>(d_MV_a, d_grad_MV, learning_rate, proj_mat_elements);
            kernelUpdateSimple<<<gridDimProjMat, blockDim1D>>>(d_MQ_a, d_grad_MQ, learning_rate, proj_mat_elements);
            kernelUpdateSimple<<<gridDimProjMat, blockDim1D>>>(d_MK_a, d_grad_MK_correction, learning_rate, proj_mat_elements); // Update MK with correction
            CUDA_CHECK(cudaGetLastError());
            kernelUpdateSimple<<<gridDimEV, blockDim1D>>>(d_EV, d_grad_EV_full, learning_rate, ev_total_elements); // Update full EV using full gradient
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            // --- Data Transfer D->H ---
            for (int l = 0; l < layers - 1; ++l) { // Weights and Gweights (0 to N-2)
                CUDA_CHECK(cudaMemcpy(head_obj.ver.weights[l].mapped_data, d_ver_weights[l], mlp_weights_bytes, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(head_obj.ver.gweights[l].mapped_data, d_ver_gweights[l], mlp_weights_bytes, cudaMemcpyDeviceToHost));
            }
            CUDA_CHECK(cudaMemcpy(head_obj.EV.mapped_data, d_EV, ev_total_bytes, cudaMemcpyDeviceToHost)); 
            CUDA_CHECK(cudaMemcpy(head_obj.MV.mapped_data, d_MV_a, proj_mat_bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(head_obj.MQ.mapped_data, d_MQ_a, proj_mat_bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(head_obj.MK.mapped_data, d_MK_a, proj_mat_bytes, cudaMemcpyDeviceToHost));

        }
        catch (const std::exception& e) {
            std::cerr << "Error during cu1ParallelBackward (V) for head [" << i << "][" << layno << "]: " << e.what() << std::endl;
            // Cleanup allocated memory on error
            cudaFree(d_expected_v); cudaFree(d_EV); cudaFree(d_grad_EV_full); cudaFree(d_grad_EV_summed); cudaFree(d_grad_EV_scaled); cudaFree(d_grad_dv); cudaFree(d_KdotQ); cudaFree(d_head); cudaFree(d_K); cudaFree(d_Q); cudaFree(d_pre_MV); cudaFree(d_MV_a); cudaFree(d_MQ_a); cudaFree(d_MK_a); cudaFree(d_grad_MV); cudaFree(d_grad_head); cudaFree(d_lota_deriv); cudaFree(d_grad_KdotQ); cudaFree(d_grad_Q); cudaFree(d_grad_MQ); cudaFree(d_grad_MK_correction);
            for (int l = 0; l < layers; ++l) {
                cudaFree(d_ver_activations[l]); cudaFree(d_ver_deltas[l]);
            } // Activations (0 to N-1)
            for (int l = 0; l < layers - 1; ++l) { // Deltas, Weights, Gweights (0 to N-2)
                cudaFree(d_ver_weights[l]); cudaFree(d_ver_gweights[l]);
                cudaFree(d_ver_deltas[l]);
            }
            throw std::runtime_error("Exception processing head [" + std::to_string(i) + "][" + std::to_string(layno) + "]: " + e.what());
        }

        // --- Cleanup Device Memory (Success Case - Per Head) ---
        cudaFree(d_expected_v); cudaFree(d_EV); cudaFree(d_grad_EV_full); cudaFree(d_grad_EV_summed); cudaFree(d_grad_EV_scaled); cudaFree(d_grad_dv); cudaFree(d_KdotQ); cudaFree(d_head); cudaFree(d_K); cudaFree(d_Q); cudaFree(d_pre_MV); cudaFree(d_MV_a); cudaFree(d_MQ_a); cudaFree(d_MK_a); cudaFree(d_grad_MV); cudaFree(d_grad_head); cudaFree(d_lota_deriv); cudaFree(d_grad_KdotQ); cudaFree(d_grad_Q); cudaFree(d_grad_MQ); cudaFree(d_grad_MK_correction);
        for (int l = 0; l < layers; ++l) {
            cudaFree(d_ver_activations[l]); cudaFree(d_ver_deltas[l]);
        } // Activations (0 to N-1)
        for (int l = 0; l < layers - 1; ++l) { // Deltas, Weights, Gweights (0 to N-2)
            cudaFree(d_ver_weights[l]); cudaFree(d_ver_gweights[l]);
            cudaFree(d_ver_deltas[l]);
        }
    }
}
