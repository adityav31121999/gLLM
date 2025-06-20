
#include <maths.hpp>
#include "include/attention.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <cfloat>
#include <vector>
#include <stdexcept>
#include <iostream>

// --- Error Checking Macro ---
#define CUDA_CHECK(call) do {                                   \
    cudaError_t err = call;                                     \
    if (err != cudaSuccess) {                                   \
        fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));  \
        throw std::runtime_error(cudaGetErrorString(err));      \
    }                                                           \
} while (0)

/**
 * @brief Backward Propagation for the attention class using gradients from expected Horizontal output.
 *      Use for first (when sentence ends in first block itself) and last block only.
 * @param expected Expected output vector (target embedding for next token prediction)
 * @param in Input size (embedding dimension) - Corresponds to EMBEDDING
 * @param layers Number of layers in the MLPs
 */
void attention::cuBackward(std::vector<float>& expected, int& in, int& layers, int headnumber)
{
    // get values for all kernels and functions
    const int embedding_dim = EMBEDDING; // Should match 'in'
    const int mat_heights = MATHEIGHTS;
    const int context_win = CONTEXT_WIN;
    // const int ev_size = CONTEXT_WIN;
    const float learning_rate = learning;
    const float scaling_factor = SCALING;
    const int token_count = this->tokenCount; // Use member variable

    // Validate essential mat objects
    if (!this->K.mapped_data || !this->Q.mapped_data || !this->KdotQ.mapped_data ||
        !this->MH.mapped_data || !this->MV.mapped_data ||
        !this->MQ.mapped_data || !this->MK.mapped_data || !this->EV.mapped_data) {
        throw std::runtime_error("One or more attention mat members have null mapped_data in cuBackward(expected).");
    }
    if (this->hor.weights.empty() || !this->hor.weights[0].mapped_data ||
        this->ver.weights.empty() || !this->ver.weights[0].mapped_data ||
        this->hor.gweights.empty() || !this->hor.gweights[0].mapped_data ||
        this->ver.gweights.empty() || !this->ver.gweights[0].mapped_data) {
        throw std::runtime_error("One or more MLP weight/gradient mat members have null mapped_data or are empty in cuBackward(expected).");
    }

    const size_t head_elements = static_cast<size_t>(CONTEXT_WIN) * CONTEXT_WIN;            // Should be token_count * token_count
    const size_t proj_mat_elements = static_cast<size_t>(this->MH.row) * this->MH.col;      // Should be mat_heights * embedding_dim
    const size_t k_q_elements = static_cast<size_t>(this->K.row) * this->K.col;             // Should be token_count * mat_heights
    const size_t ev_elements = static_cast<size_t>(this->EV.row) * this->EV.col;            // Should be context_win * embedding_dim

    const size_t embed_bytes = embedding_dim * sizeof(float);
    const size_t mlp_weights_elements = static_cast<size_t>(this->hor.weights[0].row) * this->hor.weights[0].col;
    const size_t mlp_weights_bytes = mlp_weights_elements * sizeof(float);

    if (embedding_dim != in) {
         throw std::runtime_error("Embedding dimension mismatch: EMBEDDING vs in");
    }
    if (expected.size() != embedding_dim) {
         throw std::runtime_error("Expected vector size mismatch");
    }
    // Dimension validation for mat objects
    if (this->KdotQ.row != context_win || this->KdotQ.col != context_win) throw std::runtime_error("KdotQ dimensions mismatch");
    if (this->MH.row != embedding_dim || this->MH.col != mat_heights) throw std::runtime_error("MH dimensions mismatch");
    if (this->MV.row != embedding_dim || this->MV.col != mat_heights) throw std::runtime_error("MV dimensions mismatch");
    if (this->MQ.row != mat_heights || this->MQ.col != embedding_dim) throw std::runtime_error("MQ dimensions mismatch");
    if (this->MK.row != mat_heights || this->MK.col != embedding_dim) throw std::runtime_error("MK dimensions mismatch");
    if (this->K.row != context_win || this->K.col != mat_heights) throw std::runtime_error("K dimensions mismatch");
    if (this->Q.row != context_win || this->Q.col != mat_heights) throw std::runtime_error("Q dimensions mismatch");
    if (this->EV.row != context_win || this->EV.col != embedding_dim) throw std::runtime_error("EV dimensions mismatch");
    if (this->hor.weights[0].row != embedding_dim || this->hor.weights[0].col != embedding_dim ||
        this->ver.weights[0].row != embedding_dim || this->ver.weights[0].col != embedding_dim) throw std::runtime_error("MLP weights[0] dimensions mismatch");

    // --- Device Memory Pointers ---
    float *d_expected_h = nullptr, *d_EH = nullptr, *d_EV = nullptr;        // size = embedding
    float *d_grad_EH = nullptr, *d_grad_EV_scaled = nullptr;                // size = embedding
    float *d_hor_output = nullptr, *d_ver_output = nullptr;                 // size = embedding (MLP final layer output)
    float *d_grad_hor_input_mlp = nullptr, *d_grad_ver_input_mlp = nullptr; // size = embedding (Gradient input to MLP backprop)
    // float *d_hor_gweights0 = nullptr, *d_ver_gweights0 = nullptr;        // No longer needed H->D
    float *d_grad_dh = nullptr, *d_grad_dv = nullptr;                       // size = embedding (Gradient w.r.t MLP input)
    float *d_KdotQ = nullptr;                                               // size = context_win^2
    float *d_K = nullptr, *d_Q = nullptr;                                   // size = context_wn * embedding
    float *d_pre_MH = nullptr, *d_pre_MV = nullptr;                         // size = matheigh * embedding
    float *d_MH_a = nullptr, *d_MV_a = nullptr;                             // size = matheight * embedding
    float *d_MQ_a = nullptr, *d_MK_a = nullptr;                             // size = embedding * matheight
    float *d_grad_MH = nullptr, *d_grad_MV = nullptr;                       // size = embedding * matheight
    float *d_grad_head = nullptr, *d_head = nullptr;                        // size = tokenCount^2
    float *d_lota_deriv = nullptr;                                          // size = tokenCount^2
    float *d_grad_KdotQ = nullptr;                                          // size = context_win^2
    float *d_grad_K = nullptr, *d_grad_Q = nullptr;                         // size = context_win * embedding
    float *d_grad_MQ = nullptr, *d_grad_MK = nullptr;                       // size = matheight*embedding

    // --- MLP Internal Device Memory ---
    std::vector<float*> d_hor_activations(layers, nullptr);
    std::vector<float*> d_hor_weights(layers - 1, nullptr); // layers-1 weight matrices
    std::vector<float*> d_hor_gweights(layers - 1, nullptr); // layers-1 gweight matrices
    std::vector<float*> d_hor_deltas(layers, nullptr);
    std::vector<float*> d_ver_activations(layers, nullptr);
    std::vector<float*> d_ver_weights(layers - 1, nullptr); // layers-1 weight matrices
    std::vector<float*> d_ver_gweights(layers - 1, nullptr); // layers-1 gweight matrices
    std::vector<float*> d_ver_deltas(layers, nullptr);

    try {
        // --- Allocate Device Memory (Attention Part) ---
        CUDA_CHECK(cudaMalloc(&d_expected_h, embed_bytes));
        CUDA_CHECK(cudaMalloc(&d_EH, embed_bytes)); 
        CUDA_CHECK(cudaMalloc(&d_EV, ev_elements * sizeof(float))); // EV might have context_win dim
        CUDA_CHECK(cudaMalloc(&d_grad_EH, embed_bytes));
        CUDA_CHECK(cudaMalloc(&d_grad_EV_scaled, embed_bytes)); // Scaled EV grad is embed_dim
        CUDA_CHECK(cudaMalloc(&d_hor_output, embed_bytes)); // Output of hor MLP
        CUDA_CHECK(cudaMalloc(&d_ver_output, embed_bytes)); // Output of ver MLP
        CUDA_CHECK(cudaMalloc(&d_grad_hor_input_mlp, embed_bytes)); // Grad input to hor MLP backprop
        CUDA_CHECK(cudaMalloc(&d_grad_ver_input_mlp, embed_bytes)); // Grad input to ver MLP backprop
        CUDA_CHECK(cudaMalloc(&d_grad_dh, embed_bytes)); // Grad w.r.t. hor MLP input (EH)
        CUDA_CHECK(cudaMalloc(&d_grad_dv, embed_bytes)); // Grad w.r.t. ver MLP input (EV)
        CUDA_CHECK(cudaMalloc(&d_KdotQ, head_elements * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_head, head_elements * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_K, k_q_elements * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_Q, k_q_elements * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_pre_MH, mat_heights * sizeof(float))); // Should be mat_heights? Check usage
        CUDA_CHECK(cudaMalloc(&d_pre_MV, mat_heights * sizeof(float))); // Should be mat_heights? Check usage
        CUDA_CHECK(cudaMalloc(&d_MH_a, proj_mat_elements * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_MV_a, proj_mat_elements * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_MQ_a, proj_mat_elements * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_MK_a, proj_mat_elements * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad_MH, proj_mat_elements * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad_MV, proj_mat_elements * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad_head, head_elements * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_lota_deriv, head_elements * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad_KdotQ, head_elements * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad_K, k_q_elements * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad_Q, k_q_elements * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad_MQ, proj_mat_elements * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad_MK, proj_mat_elements * sizeof(float)));

        // --- Allocate Device Memory (MLP Internals) ---
        // Activations and Deltas are for each neuron layer (layers of them)
        for (int l = 0; l < layers; ++l) {
            CUDA_CHECK(cudaMalloc(&d_hor_activations[l], embed_bytes));
            CUDA_CHECK(cudaMalloc(&d_hor_deltas[l], embed_bytes));
            CUDA_CHECK(cudaMalloc(&d_ver_activations[l], embed_bytes));
            CUDA_CHECK(cudaMalloc(&d_ver_deltas[l], embed_bytes));
        }
        // Weights and Gweights are between neuron layers (layers-1 of them)
        for (int l = 0; l < layers - 1; ++l) {
            CUDA_CHECK(cudaMalloc(&d_hor_weights[l], mlp_weights_bytes));
            CUDA_CHECK(cudaMalloc(&d_hor_gweights[l], mlp_weights_bytes));
            CUDA_CHECK(cudaMalloc(&d_ver_weights[l], mlp_weights_bytes));
            CUDA_CHECK(cudaMalloc(&d_ver_gweights[l], mlp_weights_bytes));
        }

        // --- Data Transfer: Host -> Device (Attention Part) ---
        CUDA_CHECK(cudaMemcpy(d_expected_h, expected.data(), embed_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_EH, this->EH.data(), embed_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_EV, this->EV.mapped_data, ev_elements * sizeof(float), cudaMemcpyHostToDevice)); // Copy full EV
        CUDA_CHECK(cudaMemcpy(d_hor_output, this->hor.output.data(), embed_bytes, cudaMemcpyHostToDevice)); // MLP final activation
        CUDA_CHECK(cudaMemcpy(d_ver_output, this->ver.output.data(), embed_bytes, cudaMemcpyHostToDevice)); // MLP final activation
        CUDA_CHECK(cudaMemcpy(d_KdotQ, this->KdotQ.mapped_data, head_elements * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_K, this->K.mapped_data, k_q_elements * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_Q, this->Q.mapped_data, k_q_elements * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_MH_a, this->MH.mapped_data, proj_mat_elements * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_MV_a, this->MV.mapped_data, proj_mat_elements * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_MQ_a, this->MQ.mapped_data, proj_mat_elements * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_MK_a, this->MK.mapped_data, proj_mat_elements * sizeof(float), cudaMemcpyHostToDevice));

        // --- Data Transfer: Host -> Device (MLP Internals) ---
        // Copy activations for all neuron layers
        for (int l = 0; l < layers; ++l) {
             if (this->hor.activations[l].size() != embedding_dim || this->ver.activations[l].size() != embedding_dim) {
                 throw std::runtime_error("MLP Activation vector dimension error during copy.");
             }
            CUDA_CHECK(cudaMemcpy(d_hor_activations[l], this->hor.activations[l].data(), embed_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_ver_activations[l], this->ver.activations[l].data(), embed_bytes, cudaMemcpyHostToDevice));
        }
        // Copy weights for all layers-1 weight matrices
        for (int l = 0; l < layers - 1; ++l) {
            CUDA_CHECK(cudaMemcpy(d_hor_weights[l], this->hor.weights[l].mapped_data, mlp_weights_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_ver_weights[l], this->ver.weights[l].mapped_data, mlp_weights_bytes, cudaMemcpyHostToDevice));
            // Gradients (gweights) and deltas are computed on device, no H->D needed.
        }

        // --- Kernel Launch Configuration ---
        int threadsPerBlock1D = 256;
        dim3 blockDim1D(threadsPerBlock1D);
        int blocksPerGridEmbed = (embedding_dim + threadsPerBlock1D - 1) / threadsPerBlock1D;
        dim3 gridDimEmbed(blocksPerGridEmbed);
        // Other grid dims... (head, matheights, matrix, ev) - define as needed
        int blocksPerGridHead = (static_cast<int>(head_elements) + threadsPerBlock1D - 1) / threadsPerBlock1D;
        int blocksPerGridMatHeights = (mat_heights + threadsPerBlock1D - 1) / threadsPerBlock1D;
        int blocksPerGridMatrix = (static_cast<int>(proj_mat_elements) + threadsPerBlock1D - 1) / threadsPerBlock1D;
        int blocksPerGridEV = (static_cast<int>(ev_elements) + threadsPerBlock1D - 1) / threadsPerBlock1D;
        dim3 gridDimHead(blocksPerGridHead);
        dim3 gridDimMatHeights(blocksPerGridMatHeights);
        dim3 gridDimMatrix(blocksPerGridMatrix);
        dim3 gridDimEV(blocksPerGridEV);

        // 2D Kernels
        dim3 blockDim2D(16, 16); // Example: 256 threads
        dim3 gridDimEmbed2D((embedding_dim + blockDim2D.x - 1) / blockDim2D.x, (embedding_dim + blockDim2D.y - 1) / blockDim2D.y); // For weight updates
        dim3 gridDimHead2D((token_count + blockDim2D.x - 1) / blockDim2D.x, (token_count + blockDim2D.y - 1) / blockDim2D.y);
        dim3 gridDimMatrix2D((embedding_dim + blockDim2D.x - 1) / blockDim2D.x, (mat_heights + blockDim2D.y - 1) / blockDim2D.y);
        dim3 gridDimKQGrad2D((mat_heights + blockDim2D.x - 1) / blockDim2D.x, (token_count + blockDim2D.y - 1) / blockDim2D.y);


        // --- Step 1: Compute loss gradient w.r.t. EH and EV (scaled) ---
        // This kernel computes the initial gradient *before* the MLPs.
        kernelComputeGradientsEH_EV<<<gridDimEmbed, blockDim1D>>>(d_EH, d_expected_h, d_grad_EH, d_grad_EV_scaled, embedding_dim);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize()); // Sync for safety

        // --- Step 2: Backprop through MLPs ---
        // --- 2a: Backprop through hor MLP ---
        // Calculate gradient input for hor MLP (apply activation derivative)
        // Assuming final MLP layer used Sigmoid (matches kernelOutputDelta)
        // If ReLU: use cuReLUder<<<...>>>(d_hor_output, d_temp_deriv, embedding_dim);
        //          then elementWiseMultiply<<<...>>>(d_grad_hor_input_mlp, d_grad_EH, d_temp_deriv, embedding_dim);
        // For Sigmoid, the derivative is part of kernelOutputDelta calculation.
        // We treat d_grad_EH as the "error" signal (target - output) for the MLP's last layer.
        // However, kernelOutputDelta calculates delta = (act - expected) * deriv.
        // Here, the 'error' is already computed (d_grad_EH). We need delta = d_grad_EH * deriv.
        // Let's modify the thinking: d_grad_EH is the gradient dL/d(hor_output).


        // Calculate Output Layer Deltas for hor MLP (Layer layers-1)
        kernelLastLayerDelta<<<gridDimEmbed, blockDim1D>>>(d_grad_EH, d_hor_activations[layers - 1], d_hor_deltas[layers - 1], embedding_dim);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Calculate Hidden Layer Deltas for hor MLP (Propagate backwards from layers-2 down to 0)
        for (int l = layers - 2; l >= 0; --l) {
            hiddenDeltaKernel<<<gridDimEmbed, blockDim1D>>>(
                d_hor_deltas[l + 1],        // Deltas from the layer l+1 (output of W[l])
                d_hor_weights[l],           // Weights W[l] connecting layer l to layer l+1
                d_hor_activations[l],       // Activations of the current layer (l)
                d_hor_deltas[l],            // Deltas to compute for the current layer (l)
                embedding_dim,              // current_layer_size
                embedding_dim               // next_layer_size
            );
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // Calculate Gradients and Update Weights for hor MLP (Iterate through layers-1 weight matrices)
        // W[l] (d_hor_weights[l]) uses d_hor_activations[l] as input. Its output delta is d_hor_deltas[l+1].
        // d_hor_activations[0] is the input to the MLP (e.g., d_EH if hor.activations[0] was EH).
        // For this generic backward, hor.activations[0] is the actual input to the first layer of weights.
        for (int l = 0; l < layers - 1; ++l) {
            updateWeightsKernel<<<gridDimEmbed2D, blockDim2D>>>( // Use 2D grid/block for weights
                d_hor_deltas[l+1],          // Deltas for the output layer of W[l]
                d_hor_activations[l],       // Activations for the input layer of W[l]
                d_hor_weights[l],           // Weights W[l] - TO BE UPDATED
                d_hor_gweights[l],          // Gradients gW[l] - TO BE CALCULATED
                learning_rate,              // Learning rate
                embedding_dim,              // current_layer_size
                embedding_dim               // prev_layer_size (input size of W[l])
            );
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // --- 2b: Backprop through ver MLP ---
        // Calculate Output Layer Deltas for ver MLP (Layer layers-1)
        // Input gradient is d_grad_EV_scaled
        kernelLastLayerDelta<<<gridDimEmbed, blockDim1D>>>(d_grad_EV_scaled, d_ver_activations[layers - 1], d_ver_deltas[layers - 1], embedding_dim);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Calculate Hidden Layer Deltas for ver MLP (Propagate backwards from layers-2 down to 0)
        for (int l = layers - 2; l >= 0; --l) {
            hiddenDeltaKernel<<<gridDimEmbed, blockDim1D>>>(
                d_ver_deltas[l + 1],        // Deltas from the layer l+1
                d_ver_weights[l],           // Weights W[l]
                d_ver_activations[l],       // Activations of the current layer (l)
                d_ver_deltas[l],            // Deltas to compute for the current layer (l)
                embedding_dim,              // current_layer_size
                embedding_dim               // next_layer_size
            );
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // Calculate Gradients and Update Weights for ver MLP (Iterate through layers-1 weight matrices)
        // W[l] (d_ver_weights[l]) uses d_ver_activations[l] as input. Its output delta is d_ver_deltas[l+1].
        // d_ver_activations[0] is the input to the MLP.
        for (int l = 0; l < layers - 1; ++l) {
            updateWeightsKernel<<<gridDimEmbed2D, blockDim2D>>>( // Use 2D grid/block for weights
                d_ver_deltas[l+1],          // Deltas for the output layer of W[l]
                d_ver_activations[l],       // Activations for the input layer of W[l]
                d_ver_weights[l],           // Weights W[l] - TO BE UPDATED
                d_ver_gweights[l],          // Gradients gW[l] - TO BE CALCULATED
                learning_rate,              // Learning rate
                embedding_dim,              // current_layer_size
                embedding_dim               // prev_layer_size
            );
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // --- Step 3: Compute gradients w.r.t. MLP inputs (dh and dv) ---
        // This computes dL/d(EH) and dL/d(EV) based on the first layer's deltas and weights.
        // Note: d_grad_dh will be dL/d(input_to_hor_MLP)
        // Note: d_grad_dv will be dL/d(input_to_ver_MLP)
        // These are then used to update EH and EV respectively.
        // The original CPU code's Step 3 was different. This CUDA approach is more standard for MLP grad_input.
        kernelComputeGradMLPInput<<<gridDimEmbed, blockDim1D>>>(
            d_hor_deltas[0], d_hor_weights[0], d_grad_dh, embedding_dim, embedding_dim);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // --- Step 4: Compute gradients w.r.t. MH and MV ---
        // (Original logic using d_grad_dh, d_grad_dv - now correctly computed)
        // 4a: Compute attention head
        cuLOTA<<<gridDimHead2D, blockDim2D>>>(d_KdotQ, d_head, context_win, context_win, token_count, isSelfAttention);
        CUDA_CHECK(cudaGetLastError());

        // 4b: Compute pre_MH and pre_MV
        kernelComputePreMH_MV<<<gridDimMatHeights, blockDim1D>>>(d_head, d_K, d_Q, d_pre_MH, d_pre_MV, token_count, mat_heights);
        CUDA_CHECK(cudaGetLastError());

        // 4c: Compute grad_MH and grad_MV
        kernelComputeGradMH_MV<<<gridDimMatrix2D, blockDim2D>>>(d_pre_MH, d_pre_MV, d_grad_dh, d_grad_dv, d_grad_MH, d_grad_MV, mat_heights, embedding_dim);
        CUDA_CHECK(cudaGetLastError());

        // --- Step 5: Compute gradients w.r.t. head ---
        kernelComputeGradHead<<<gridDimHead2D, blockDim2D>>>(d_K, d_Q, d_MH_a, d_MV_a, d_grad_dh, d_grad_dv, d_grad_head, token_count, mat_heights, embedding_dim);
        CUDA_CHECK(cudaGetLastError());

        // --- Step 6: Backprop through LOTA ---
        // 6a: Compute LOTA derivative
        cuLOTAder<<<gridDimHead2D, blockDim2D>>>(d_KdotQ, d_lota_deriv, token_count, token_count, token_count, isSelfAttention);
        CUDA_CHECK(cudaGetLastError());

        // 6b: Compute grad_KdotQ
        kernelComputeGradKdotQ_LOTA<<<gridDimHead, blockDim1D>>>(d_grad_head, d_lota_deriv, d_grad_KdotQ, scaling_factor, static_cast<int>(head_elements));
        CUDA_CHECK(cudaGetLastError());

        // --- Step 7: Compute gradients w.r.t. K and Q ---
        kernelComputeGradK_Q<<<gridDimKQGrad2D, blockDim2D>>>(d_grad_KdotQ, d_K, d_Q, d_grad_K, d_grad_Q, token_count, mat_heights);
        CUDA_CHECK(cudaGetLastError());

        // --- Step 8: Compute gradients w.r.t. MQ and MK ---
        // Use the simplified kernel that expects K_embed and Q_embed (token_count x embedding_dim)
        // Pass d_K and d_Q (token_count x mat_heights). The kernel will use embedding_dim for column indexing.
        kernelComputeGradMK_MQ_Simplified<<<gridDimMatrix2D, blockDim2D>>>(
            d_grad_K, d_grad_Q, d_K, d_Q, d_grad_MK, d_grad_MQ, 
            token_count, mat_heights, embedding_dim);
        CUDA_CHECK(cudaGetLastError());

        // --- Step 9: Update weights MH, MV, MQ, MK ---
        // Update MH, MV, MQ, MK
        kernelUpdateSimple<<<gridDimMatrix, blockDim1D>>>(d_MH_a, d_grad_MH, learning_rate, proj_mat_elements);
        CUDA_CHECK(cudaGetLastError());
        kernelUpdateSimple<<<gridDimMatrix, blockDim1D>>>(d_MV_a, d_grad_MV, learning_rate, proj_mat_elements);
        CUDA_CHECK(cudaGetLastError());
        kernelUpdateSimple<<<gridDimMatrix, blockDim1D>>>(d_MQ_a, d_grad_MQ, learning_rate, proj_mat_elements);
        CUDA_CHECK(cudaGetLastError());
        kernelUpdateSimple<<<gridDimMatrix, blockDim1D>>>(d_MK_a, d_grad_MK, learning_rate, proj_mat_elements);
        CUDA_CHECK(cudaGetLastError());

        // --- Step 10: Update EH and EV ---
        // Update EH conditionally using d_grad_EH (gradient before hor MLP)
        if (headnumber > 1) { // Align with C++ logic
            kernelUpdateSimple<<<gridDimEmbed, blockDim1D>>>(d_EH, d_grad_EH, learning_rate, embedding_dim);
            CUDA_CHECK(cudaGetLastError());
        }
        // Update EV by broadcasting d_grad_EV_scaled
        // Update EV: d_EV[r*embed_dim + c] -= lr * d_grad_EV_scaled[c] for r in [0, context_win-1]
        // This requires a kernel that broadcasts d_grad_EV_scaled to all rows of d_EV.
        // Example: kernelUpdateEVBroadcasted<<<gridDimEV, blockDim1D>>>(d_EV, d_grad_EV_scaled, learning_rate, context_win, embedding_dim);
        kernelUpdateEVBroadcasted<<<gridDimEV, blockDim1D>>>(d_EV, d_grad_EV_scaled, learning_rate, context_win, embedding_dim);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // --- Data Transfer: Device -> Host ---
        // Copy updated MLP weights and gradients back
        for (int l = 0; l < layers - 1; ++l) { // layers-1 weight matrices
            // Copy hor weights/gradients
            CUDA_CHECK(cudaMemcpy(this->hor.weights[l].mapped_data, d_hor_weights[l], mlp_weights_bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(this->hor.gweights[l].mapped_data, d_hor_gweights[l], mlp_weights_bytes, cudaMemcpyDeviceToHost));
            // Copy ver weights/gradients
            CUDA_CHECK(cudaMemcpy(this->ver.weights[l].mapped_data, d_ver_weights[l], mlp_weights_bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(this->ver.gweights[l].mapped_data, d_ver_gweights[l], mlp_weights_bytes, cudaMemcpyDeviceToHost));
        }
        // Note: MLP activations (this->hor.activations, this->ver.activations) and deltas
        // are intermediate values computed on the device. They are typically not copied back
        // unless specifically needed for analysis or if the host-side MLP objects
        // (this->hor, this->ver) are expected to reflect these intermediate states.
        // The current code only copies back weights and gweights.

        // Copy updated Attention parameters back
        CUDA_CHECK(cudaMemcpy(this->EH.data(), d_EH, embed_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(this->EV.mapped_data, d_EV, ev_elements * sizeof(float), cudaMemcpyDeviceToHost)); // Copy full EV
        CUDA_CHECK(cudaMemcpy(this->MH.mapped_data, d_MH_a, proj_mat_elements * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(this->MV.mapped_data, d_MV_a, proj_mat_elements * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(this->MQ.mapped_data, d_MQ_a, proj_mat_elements * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(this->MK.mapped_data, d_MK_a, proj_mat_elements * sizeof(float), cudaMemcpyDeviceToHost));
    }
    catch (const std::exception& e) {
        std::cerr << "CUDA Exception in cuBackward(expected): " << e.what() << std::endl;
        // Cleanup allocated memory (Attention part)
        cudaFree(d_expected_h); cudaFree(d_EH); cudaFree(d_EV);
        cudaFree(d_grad_EH); cudaFree(d_grad_EV_scaled);
        cudaFree(d_hor_output); cudaFree(d_ver_output);
        cudaFree(d_grad_hor_input_mlp); cudaFree(d_grad_ver_input_mlp);
        cudaFree(d_grad_dh); cudaFree(d_grad_dv);
        cudaFree(d_KdotQ); cudaFree(d_head);
        cudaFree(d_K); cudaFree(d_Q);
        cudaFree(d_pre_MH); cudaFree(d_pre_MV);
        cudaFree(d_MH_a); cudaFree(d_MV_a); cudaFree(d_MQ_a); cudaFree(d_MK_a);
        cudaFree(d_grad_MH); cudaFree(d_grad_MV);
        cudaFree(d_grad_head); cudaFree(d_lota_deriv);
        cudaFree(d_grad_KdotQ); cudaFree(d_grad_K); cudaFree(d_grad_Q);
        cudaFree(d_grad_MQ); cudaFree(d_grad_MK);
        // Cleanup allocated memory (MLP Internals)
        for (int l = 0; l < layers; ++l) { // Activations and Deltas
            cudaFree(d_hor_activations[l]); cudaFree(d_hor_deltas[l]);
            cudaFree(d_ver_activations[l]); cudaFree(d_ver_deltas[l]);
        }
        for (int l = 0; l < layers - 1; ++l) { // Weights and GWeights
            cudaFree(d_hor_weights[l]); cudaFree(d_hor_gweights[l]);
            cudaFree(d_ver_weights[l]); cudaFree(d_ver_gweights[l]);
        }
        throw; // Re-throw exception
    }

    // --- Cleanup Device Memory (Success Case) ---
    // (Attention part)
    cudaFree(d_expected_h); cudaFree(d_EH); cudaFree(d_EV);
    cudaFree(d_grad_EH); cudaFree(d_grad_EV_scaled);
    cudaFree(d_hor_output); cudaFree(d_ver_output);
    cudaFree(d_grad_hor_input_mlp); cudaFree(d_grad_ver_input_mlp);
    cudaFree(d_grad_dh); cudaFree(d_grad_dv);
    cudaFree(d_KdotQ); cudaFree(d_head);
    cudaFree(d_K); cudaFree(d_Q);
    cudaFree(d_pre_MH); cudaFree(d_pre_MV);
    cudaFree(d_MH_a); cudaFree(d_MV_a); cudaFree(d_MQ_a); cudaFree(d_MK_a);
    cudaFree(d_grad_MH); cudaFree(d_grad_MV);
    cudaFree(d_grad_head); cudaFree(d_lota_deriv);
    cudaFree(d_grad_KdotQ); cudaFree(d_grad_K); cudaFree(d_grad_Q);
    cudaFree(d_grad_MQ); cudaFree(d_grad_MK);
    // (MLP Internals)
    for (int l = 0; l < layers; ++l) { // Activations and Deltas
        cudaFree(d_hor_activations[l]); cudaFree(d_hor_deltas[l]);
        cudaFree(d_ver_activations[l]); cudaFree(d_ver_deltas[l]);
    }
    for (int l = 0; l < layers - 1; ++l) { // Weights and GWeights
        cudaFree(d_hor_weights[l]); cudaFree(d_hor_gweights[l]);
        cudaFree(d_ver_weights[l]); cudaFree(d_ver_gweights[l]);
    }
}


/**
 * @brief Backward Propagation for the attention class using gradients from expected Vertical output only.
 *      Used for intermediate blocks. Adjusts MQ, MV, and MK (correction).
 *      Corresponds to C++: attention::backward(std::vector<std::vector<float>>& expectedV, ...)
 * @param expectedV vertical retention vector (host)
 * @param in Input size (number of tokens) - Corresponds to tokenCount (used indirectly via member)
 * @param layers Number of layers in the MLPs
 */
void attention::cuBackward(std::vector<std::vector<float>>& expectedV, int& layers, int blocknumber)
{
    // get values for all kernels and functions
    const int embedding_dim = EMBEDDING;
    const int mat_heights = MATHEIGHTS;
    const int context_win = CONTEXT_WIN;
    const float learning_rate = learning;
    const float scaling_factor = SCALING;
    const int token_count = this->tokenCount; // Use member variable

    // Validate essential mat objects
    if (!this->K.mapped_data || !this->Q.mapped_data || !this->KdotQ.mapped_data ||
        !this->MV.mapped_data || !this->MQ.mapped_data || !this->MK.mapped_data || !this->EV.mapped_data) {
        throw std::runtime_error("One or more attention mat members have null mapped_data in cuBackward(expectedV).");
    }
    if (this->ver.weights.empty() || !this->ver.weights[0].mapped_data ||
        this->ver.gweights.empty() || !this->ver.gweights[0].mapped_data) {
        throw std::runtime_error("MLP ver weight/gradient mat members have null mapped_data or are empty in cuBackward(expectedV).");
    }

    const size_t head_elements = static_cast<size_t>(this->KdotQ.row) * this->KdotQ.col;
    const size_t proj_mat_elements = static_cast<size_t>(this->MV.row) * this->MV.col; // Using MV as MH is not focus here
    const size_t k_q_elements = static_cast<size_t>(this->K.row) * this->K.col;
    const size_t ev_elements = static_cast<size_t>(this->EV.row) * this->EV.col;

    const size_t embed_bytes = embedding_dim * sizeof(float);
    const size_t mlp_weights_elements = static_cast<size_t>(this->ver.weights[0].row) * this->ver.weights[0].col;
    const size_t mlp_weights_bytes = mlp_weights_elements * sizeof(float);

    // Basic validation
    if (expectedV.size() != context_win || (!expectedV.empty() && expectedV[0].size() != embedding_dim)) {
        throw std::runtime_error("ExpectedV dimensions mismatch");
    }
    // Dimension validation for mat objects
    if (this->KdotQ.row != token_count || this->KdotQ.col != token_count) throw std::runtime_error("KdotQ dimensions mismatch");
    if (this->MV.row != embedding_dim || this->MV.col != mat_heights) throw std::runtime_error("MV dimensions mismatch");
    if (this->MQ.row != mat_heights || this->MQ.col != embedding_dim) throw std::runtime_error("MQ dimensions mismatch");
    if (this->MK.row != mat_heights || this->MK.col != embedding_dim) throw std::runtime_error("MK dimensions mismatch");
    if (this->K.row != context_win || this->K.col != mat_heights) throw std::runtime_error("K dimensions mismatch");
    if (this->Q.row != context_win || this->Q.col != mat_heights) throw std::runtime_error("Q dimensions mismatch");
    if (this->EV.row != context_win || this->EV.col != embedding_dim) throw std::runtime_error("EV dimensions mismatch");
    if (this->ver.weights[0].row != embedding_dim || this->ver.weights[0].col != embedding_dim) throw std::runtime_error("MLP ver.weights[0] dimensions mismatch");

    // --- Device Memory Pointers ---
    float *d_expected_v = nullptr, *d_EV = nullptr;
    float *d_grad_EV_full = nullptr, *d_grad_EV_summed = nullptr, *d_grad_EV_scaled = nullptr;
    float *d_ver_output = nullptr; // MLP final layer output
    float *d_grad_dv = nullptr; // Gradient w.r.t ver MLP input (EV)
    float *d_KdotQ = nullptr, *d_head = nullptr;
    float *d_K = nullptr, *d_Q = nullptr; // token x mat_heights version
    float *d_pre_MV = nullptr;
    float *d_MV_a = nullptr, *d_MQ_a = nullptr, *d_MK_a = nullptr;
    float *d_grad_MV = nullptr;
    float *d_grad_head = nullptr;
    float *d_lota_deriv = nullptr;
    float *d_grad_KdotQ = nullptr;
    float *d_grad_Q = nullptr; // Gradients for Q (token x mat_heights)
    float *d_grad_MQ = nullptr, *d_grad_MK_correction = nullptr;

    // --- MLP Internal Device Memory (Only ver) ---
    std::vector<float*> d_ver_activations(layers, nullptr);
    std::vector<float*> d_ver_weights(layers - 1, nullptr); // layers-1 weight matrices
    std::vector<float*> d_ver_gweights(layers - 1, nullptr); // layers-1 gweight matrices
    std::vector<float*> d_ver_deltas(layers, nullptr);

    try {
        // --- Allocate Device Memory (Attention Part) ---
        CUDA_CHECK(cudaMalloc(&d_expected_v, ev_elements * sizeof(float))); // expectedV is flattened from std::vector<std::vector<float>>
        CUDA_CHECK(cudaMalloc(&d_EV, ev_elements * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad_EV_full, ev_elements * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad_EV_summed, embed_bytes));
        CUDA_CHECK(cudaMalloc(&d_grad_EV_scaled, embed_bytes)); // Grad input to ver MLP backprop
        CUDA_CHECK(cudaMalloc(&d_ver_output, embed_bytes)); // Output of ver MLP
        CUDA_CHECK(cudaMalloc(&d_grad_dv, embed_bytes)); // Grad w.r.t. ver MLP input (EV)
        CUDA_CHECK(cudaMalloc(&d_KdotQ, head_elements * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_head, head_elements * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_K, k_q_elements * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_Q, k_q_elements * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_pre_MV, mat_heights * sizeof(float))); // Check size
        CUDA_CHECK(cudaMalloc(&d_MV_a, proj_mat_elements * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_MQ_a, proj_mat_elements * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_MK_a, proj_mat_elements * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad_MV, proj_mat_elements * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad_head, head_elements * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_lota_deriv, head_elements * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad_KdotQ, head_elements * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad_Q, k_q_elements * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad_MQ, proj_mat_elements * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad_MK_correction, proj_mat_elements * sizeof(float)));

        // --- Allocate Device Memory (ver MLP Internals) ---
        // Activations and Deltas are for each neuron layer (layers of them)
        for (int l = 0; l < layers; ++l) {
            CUDA_CHECK(cudaMalloc(&d_ver_activations[l], embed_bytes));
            CUDA_CHECK(cudaMalloc(&d_ver_deltas[l], embed_bytes));
        }
        // Weights and Gweights are between neuron layers (layers-1 of them)
        for (int l = 0; l < layers - 1; ++l) {
            CUDA_CHECK(cudaMalloc(&d_ver_weights[l], mlp_weights_bytes));
            CUDA_CHECK(cudaMalloc(&d_ver_gweights[l], mlp_weights_bytes));
        }

        // --- Data Transfer: Host -> Device (Attention Part) ---
        std::vector<float> flat_expectedV = flatten(expectedV);

        CUDA_CHECK(cudaMemcpy(d_expected_v, flat_expectedV.data(), ev_elements * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_EV, this->EV.mapped_data, ev_elements * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ver_output, this->ver.output.data(), embed_bytes, cudaMemcpyHostToDevice)); // MLP final activation
        CUDA_CHECK(cudaMemcpy(d_KdotQ, this->KdotQ.mapped_data, head_elements * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_K, this->K.mapped_data, k_q_elements * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_Q, this->Q.mapped_data, k_q_elements * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_MV_a, this->MV.mapped_data, proj_mat_elements * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_MQ_a, this->MQ.mapped_data, proj_mat_elements * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_MK_a, this->MK.mapped_data, proj_mat_elements * sizeof(float), cudaMemcpyHostToDevice));

        // --- Data Transfer: Host -> Device (ver MLP Internals) ---
        // Copy activations for all neuron layers
        for (int l = 0; l < layers; ++l) {
             if (this->ver.activations[l].size() != embedding_dim) {
                 throw std::runtime_error("MLP Activation vector dimension error during copy.");
             }
            CUDA_CHECK(cudaMemcpy(d_ver_activations[l], this->ver.activations[l].data(), embed_bytes, cudaMemcpyHostToDevice));
        }
        // Copy weights for all layers-1 weight matrices
        for (int l = 0; l < layers - 1; ++l) {
            CUDA_CHECK(cudaMemcpy(d_ver_weights[l], this->ver.weights[l].mapped_data, mlp_weights_bytes, cudaMemcpyHostToDevice));
        }

        // --- Kernel Launch Configuration ---
        int threadsPerBlock1D = 256;
        dim3 blockDim1D(threadsPerBlock1D);
        int blocksPerGridEmbed = (embedding_dim + threadsPerBlock1D - 1) / threadsPerBlock1D;
        dim3 gridDimEmbed(blocksPerGridEmbed);
        // Other grid dims...
        int blocksPerGridHead = (static_cast<int>(head_elements) + threadsPerBlock1D - 1) / threadsPerBlock1D;
        int blocksPerGridMatHeights = (mat_heights + threadsPerBlock1D - 1) / threadsPerBlock1D;
        int blocksPerGridMatrix = (static_cast<int>(proj_mat_elements) + threadsPerBlock1D - 1) / threadsPerBlock1D;
        int blocksPerGridEV = (static_cast<int>(ev_elements) + threadsPerBlock1D - 1) / threadsPerBlock1D;
        dim3 gridDimHead(blocksPerGridHead);
        dim3 gridDimMatHeights(blocksPerGridMatHeights);
        dim3 gridDimMatrix(blocksPerGridMatrix);
        dim3 gridDimEV(blocksPerGridEV);

        // 2D Kernels
        dim3 blockDim2D(16, 16);
        dim3 gridDimEmbed2D((embedding_dim + blockDim2D.x - 1) / blockDim2D.x, (embedding_dim + blockDim2D.y - 1) / blockDim2D.y);
        dim3 gridDimHead2D((token_count + blockDim2D.x - 1) / blockDim2D.x, (token_count + blockDim2D.y - 1) / blockDim2D.y);
        dim3 gridDimMatrix2D((embedding_dim + blockDim2D.x - 1) / blockDim2D.x, (mat_heights + blockDim2D.y - 1) / blockDim2D.y);
        dim3 gridDimKQGrad2D((mat_heights + blockDim2D.x - 1) / blockDim2D.x, (token_count + blockDim2D.y - 1) / blockDim2D.y);


        // --- Step 1: Compute loss gradient w.r.t. EV ---
        kernelComputeGradientsEV_V<<<gridDimEmbed, blockDim1D>>> // Grid based on embedding dim for reduction
            (d_EV, d_expected_v, d_grad_EV_full, d_grad_EV_summed, d_grad_EV_scaled,
             learning_rate, context_win, embedding_dim); // d_grad_EV_scaled is output here
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // --- Step 2: Backprop through ver MLP ---
        // Define local lambda for last layer delta calc (same as in other overload)
        // Calculate Output Layer Deltas for ver MLP (Layer layers-1)
        // Input gradient is d_grad_EV_scaled
        kernelLastLayerDelta<<<gridDimEmbed, blockDim1D>>>(d_grad_EV_scaled, d_ver_activations[layers - 1], d_ver_deltas[layers - 1], embedding_dim);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Calculate Hidden Layer Deltas for ver MLP (Propagate backwards from layers-2 down to 0)
        for (int l = layers - 2; l >= 0; --l) {
            hiddenDeltaKernel<<<gridDimEmbed, blockDim1D>>>(
                d_ver_deltas[l + 1],        // Deltas from layer l+1
                d_ver_weights[l],           // Weights W[l]
                d_ver_activations[l],       // Activations of the current layer (l)
                d_ver_deltas[l],            // Deltas to compute for the current layer (l)
                embedding_dim,              // current_layer_size
                embedding_dim               // next_layer_size
            );
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // Calculate Gradients and Update Weights for ver MLP (Iterate through layers-1 weight matrices)
        // W[l] (d_ver_weights[l]) uses d_ver_activations[l] as input. Its output delta is d_ver_deltas[l+1].
        // d_ver_activations[0] is the input to the MLP.
        for (int l = 0; l < layers - 1; ++l) {
            updateWeightsKernel<<<gridDimEmbed2D, blockDim2D>>>(
                d_ver_deltas[l+1],          // Deltas for the output layer of W[l]
                d_ver_activations[l],       // Activations for the input layer of W[l]
                d_ver_weights[l],           // Weights W[l] - TO BE UPDATED
                d_ver_gweights[l],          // Gradients gW[l] - TO BE CALCULATED
                learning_rate,              // Learning rate
                embedding_dim,              // current_layer_size
                embedding_dim               // prev_layer_size
            );
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // --- Step 3: Compute gradients w.r.t. ver MLP input (dv) ---
        kernelComputeGradMLPInput<<<gridDimEmbed, blockDim1D>>>(
            d_ver_deltas[0], d_ver_weights[0], d_grad_dv, embedding_dim, embedding_dim); // Assuming ver input size is embedding_dim
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // --- Step 4: Compute gradients w.r.t. MV ---
        // (Original logic using d_grad_dv - now correctly computed)
        // 4a: Compute attention head
        cuLOTA<<<gridDimHead2D, blockDim2D>>>(d_KdotQ, d_head, context_win, context_win, token_count, isSelfAttention);
        CUDA_CHECK(cudaGetLastError());

        // 4b: Compute pre_MV
        kernelComputePreMV_V<<<gridDimMatHeights, blockDim1D>>>(d_head, d_Q, d_pre_MV, token_count, mat_heights);
        CUDA_CHECK(cudaGetLastError());

        // 4c: Compute grad_MV
        kernelComputeGradMV_V<<<gridDimMatrix2D, blockDim2D>>>(d_pre_MV, d_grad_dv, d_grad_MV, mat_heights, embedding_dim);
        CUDA_CHECK(cudaGetLastError());

        // --- Step 5: Compute gradients w.r.t. head ---
        kernelComputeGradHead_V<<<gridDimHead2D, blockDim2D>>>(d_Q, d_MV_a, d_grad_dv, d_grad_head, token_count, mat_heights, embedding_dim);
        CUDA_CHECK(cudaGetLastError());

        // --- Step 6: Backprop through LOTA ---
        // 6a: Compute LOTA derivative
        cuLOTAder<<<gridDimHead2D, blockDim2D>>>(d_KdotQ, d_lota_deriv, token_count, token_count, token_count, isSelfAttention);
        CUDA_CHECK(cudaGetLastError());

        // 6b: Compute grad_KdotQ
        kernelComputeGradKdotQ_LOTA<<<gridDimHead, blockDim1D>>>(d_grad_head, d_lota_deriv, d_grad_KdotQ, scaling_factor, static_cast<int>(head_elements));
        CUDA_CHECK(cudaGetLastError());

        // --- Step 7: Compute gradients w.r.t. Q ---
        kernelComputeGradQ_V<<<gridDimKQGrad2D, blockDim2D>>>(d_grad_KdotQ, d_K, d_grad_Q, token_count, mat_heights);
        CUDA_CHECK(cudaGetLastError());

        // --- Step 8: Compute gradients w.r.t. MQ and MK_correction ---
        kernelComputeGradMQ_V<<<gridDimMatrix2D, blockDim2D>>>(d_grad_Q, d_Q, d_grad_MQ, token_count, mat_heights, embedding_dim);
        CUDA_CHECK(cudaGetLastError());
        // Compute MK correction
        kernelComputeGradMKCorrection<<<gridDimMatrix2D, blockDim2D>>>(d_grad_MQ, d_Q, d_K, d_grad_MK_correction, token_count, mat_heights, embedding_dim);
        CUDA_CHECK(cudaGetLastError());


        // --- Step 9: Update weights MV, MQ, MK ---
        // Note: C++ updates MK first, then MV, MQ. Order for independent updates doesn't strictly matter
        // as long as correct gradients are used.
        // kernelUpdateWeights_EV_V from backward1sthead.cu is more specific, but kernelUpdateSimple is used here.
        kernelUpdateSimple<<<gridDimMatrix, blockDim1D>>>(d_MV_a, d_grad_MV, learning_rate, proj_mat_elements);
        CUDA_CHECK(cudaGetLastError());
        kernelUpdateSimple<<<gridDimMatrix, blockDim1D>>>(d_MQ_a, d_grad_MQ, learning_rate, proj_mat_elements);
        CUDA_CHECK(cudaGetLastError());
        kernelUpdateSimple<<<gridDimMatrix, blockDim1D>>>(d_MK_a, d_grad_MK_correction, learning_rate, proj_mat_elements); // Update MK with correction
        CUDA_CHECK(cudaGetLastError());

        // --- Step 10: Update EV (conditionally) ---
        if (blocknumber > 1) { // Align with C++ logic
            kernelUpdateSimple<<<gridDimEV, blockDim1D>>>(d_EV, d_grad_EV_full, learning_rate, ev_elements); // Update full EV using full gradient
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());


        // --- Data Transfer: Device -> Host ---
        // Copy updated ver MLP weights and gradients back
        for (int l = 0; l < layers - 1; ++l) { // layers-1 weight matrices
            CUDA_CHECK(cudaMemcpy(this->ver.weights[l].mapped_data, d_ver_weights[l], mlp_weights_bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(this->ver.gweights[l].mapped_data, d_ver_gweights[l], mlp_weights_bytes, cudaMemcpyDeviceToHost));
        }
        // Note: MLP activations (this->ver.activations) and deltas are intermediate.
        // Only weights and gweights are typically copied back.

        // Copy updated Attention parameters back
        CUDA_CHECK(cudaMemcpy(this->EV.mapped_data, d_EV, ev_elements * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(this->MV.mapped_data, d_MV_a, proj_mat_elements * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(this->MQ.mapped_data, d_MQ_a, proj_mat_elements * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(this->MK.mapped_data, d_MK_a, proj_mat_elements * sizeof(float), cudaMemcpyDeviceToHost));

    }
    catch (const std::exception& e) {
        std::cerr << "CUDA Exception in cuBackward(expectedV): " << e.what() << std::endl;
        // Cleanup allocated memory (Attention part)
        cudaFree(d_expected_v); cudaFree(d_EV);
        cudaFree(d_grad_EV_full); cudaFree(d_grad_EV_summed); cudaFree(d_grad_EV_scaled);
        cudaFree(d_ver_output); cudaFree(d_grad_dv);
        cudaFree(d_KdotQ); cudaFree(d_head);
        cudaFree(d_K); cudaFree(d_Q);
        cudaFree(d_pre_MV);
        cudaFree(d_MV_a); cudaFree(d_MQ_a); cudaFree(d_MK_a);
        cudaFree(d_grad_MV); cudaFree(d_grad_head);
        cudaFree(d_lota_deriv); cudaFree(d_grad_KdotQ);
        cudaFree(d_grad_Q); cudaFree(d_grad_MQ); cudaFree(d_grad_MK_correction);
        // Cleanup allocated memory (ver MLP Internals)
        for (int l = 0; l < layers; ++l) { // Activations and Deltas
            cudaFree(d_ver_activations[l]); cudaFree(d_ver_deltas[l]);
        }
        for (int l = 0; l < layers - 1; ++l) { // Weights and GWeights
            cudaFree(d_ver_weights[l]); cudaFree(d_ver_gweights[l]);
        }
        throw; // Re-throw exception
    }

    // --- Cleanup Device Memory (Success Case) ---
    // (Attention part)
    cudaFree(d_expected_v); cudaFree(d_EV);
    cudaFree(d_grad_EV_full); cudaFree(d_grad_EV_summed); cudaFree(d_grad_EV_scaled);
    cudaFree(d_ver_output); cudaFree(d_grad_dv);
    cudaFree(d_KdotQ); cudaFree(d_head);
    cudaFree(d_K); cudaFree(d_Q);
    cudaFree(d_pre_MV);
    cudaFree(d_MV_a); cudaFree(d_MQ_a); cudaFree(d_MK_a);
    cudaFree(d_grad_MV); cudaFree(d_grad_head);
    cudaFree(d_lota_deriv); cudaFree(d_grad_KdotQ);
    cudaFree(d_grad_Q); cudaFree(d_grad_MQ); cudaFree(d_grad_MK_correction);
    // (ver MLP Internals)
    for (int l = 0; l < layers; ++l) { // Activations and Deltas
        cudaFree(d_ver_activations[l]); cudaFree(d_ver_deltas[l]);
    }
    for (int l = 0; l < layers - 1; ++l) { // Weights and GWeights
        cudaFree(d_ver_weights[l]); cudaFree(d_ver_gweights[l]);
    }
}
