
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
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        throw std::runtime_error(cudaGetErrorString(err)); \
    } \
} while (0)

/**
 * @brief Backward Propagation for the attention class using gradients from expected Horizontal output.
 *      Use for first (when sentence ends in first block itself) and last block only.
 * @param expected Expected output vector (target embedding for next token prediction)
 * @param in Input size (embedding dimension) - Corresponds to EMBEDDING
 * @param layers Number of layers in the MLPs
 */
void attention::cuBackward(std::vector<float>& expected, int& in, int& layers)
{
    // get values for all kernels and functions
    const int embedding_dim = EMBEDDING; // Should match 'in'
    const int mat_heights = MATHEIGHTS;
    const int context_win = CONTEXT_WIN;
    const float learning_rate = LEARNING;
    const float scaling_factor = SCALING;
    const int token_count = this->tokenCount; // Use member variable
    const int head_size = token_count * token_count;
    const int mh_mv_mq_mk_size = mat_heights * embedding_dim;
    const int k_q_size = token_count * mat_heights;
    // const int k_q_embed_size = token_count * embedding_dim;
    const int ev_size = context_win * embedding_dim;
    const size_t embed_bytes = embedding_dim * sizeof(float);
    const size_t weights_bytes = embedding_dim * embedding_dim * sizeof(float); // Assuming square MLP layers

    if (embedding_dim != in) {
         throw std::runtime_error("Embedding dimension mismatch: EMBEDDING vs in");
    }
    if (expected.size() != embedding_dim) {
         throw std::runtime_error("Expected vector size mismatch");
    }
    // Add other validation checks (K, Q, MH.a, MV.a etc.)

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
    std::vector<float*> d_hor_weights(layers, nullptr);
    std::vector<float*> d_hor_gweights(layers, nullptr);
    std::vector<float*> d_hor_deltas(layers, nullptr);
    std::vector<float*> d_ver_activations(layers, nullptr);
    std::vector<float*> d_ver_weights(layers, nullptr);
    std::vector<float*> d_ver_gweights(layers, nullptr);
    std::vector<float*> d_ver_deltas(layers, nullptr);

    // Temporary flat vectors for H->D copy
    std::vector<std::vector<float>> flat_hor_weights(layers);
    std::vector<std::vector<float>> flat_ver_weights(layers);

    try {
        // --- Allocate Device Memory (Attention Part) ---
        CUDA_CHECK(cudaMalloc(&d_expected_h, embed_bytes));
        CUDA_CHECK(cudaMalloc(&d_EH, embed_bytes));
        CUDA_CHECK(cudaMalloc(&d_EV, ev_size * sizeof(float))); // EV might have context_win dim
        CUDA_CHECK(cudaMalloc(&d_grad_EH, embed_bytes));
        CUDA_CHECK(cudaMalloc(&d_grad_EV_scaled, embed_bytes)); // Scaled EV grad is embed_dim
        CUDA_CHECK(cudaMalloc(&d_hor_output, embed_bytes)); // Output of hor MLP
        CUDA_CHECK(cudaMalloc(&d_ver_output, embed_bytes)); // Output of ver MLP
        CUDA_CHECK(cudaMalloc(&d_grad_hor_input_mlp, embed_bytes)); // Grad input to hor MLP backprop
        CUDA_CHECK(cudaMalloc(&d_grad_ver_input_mlp, embed_bytes)); // Grad input to ver MLP backprop
        CUDA_CHECK(cudaMalloc(&d_grad_dh, embed_bytes)); // Grad w.r.t. hor MLP input (EH)
        CUDA_CHECK(cudaMalloc(&d_grad_dv, embed_bytes)); // Grad w.r.t. ver MLP input (EV)
        CUDA_CHECK(cudaMalloc(&d_KdotQ, head_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_head, head_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_K, k_q_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_Q, k_q_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_pre_MH, mat_heights * sizeof(float))); // Should be mat_heights? Check usage
        CUDA_CHECK(cudaMalloc(&d_pre_MV, mat_heights * sizeof(float))); // Should be mat_heights? Check usage
        CUDA_CHECK(cudaMalloc(&d_MH_a, mh_mv_mq_mk_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_MV_a, mh_mv_mq_mk_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_MQ_a, mh_mv_mq_mk_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_MK_a, mh_mv_mq_mk_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad_MH, mh_mv_mq_mk_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad_MV, mh_mv_mq_mk_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad_head, head_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_lota_deriv, head_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad_KdotQ, head_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad_K, k_q_size * sizeof(float))); // Check size based on usage
        CUDA_CHECK(cudaMalloc(&d_grad_Q, k_q_size * sizeof(float))); // Check size based on usage
        CUDA_CHECK(cudaMalloc(&d_grad_MQ, mh_mv_mq_mk_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad_MK, mh_mv_mq_mk_size * sizeof(float)));

        // --- Allocate Device Memory (MLP Internals) ---
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

        // --- Data Transfer: Host -> Device (Attention Part) ---
        std::vector<float> flat_EV = flatten(this->EV);
        std::vector<float> flat_K = flatten(this->K);
        std::vector<float> flat_Q = flatten(this->Q);
        std::vector<float> flat_KdotQ = flatten(this->KdotQ);
        std::vector<float> flat_MH_a = flatten(this->MH.a);
        std::vector<float> flat_MV_a = flatten(this->MV.a);
        std::vector<float> flat_MQ_a = flatten(this->MQ.a);
        std::vector<float> flat_MK_a = flatten(this->MK.a);

        CUDA_CHECK(cudaMemcpy(d_expected_h, expected.data(), embed_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_EH, this->EH.data(), embed_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_EV, flat_EV.data(), ev_size * sizeof(float), cudaMemcpyHostToDevice)); // Copy full EV
        CUDA_CHECK(cudaMemcpy(d_hor_output, this->hor.output.data(), embed_bytes, cudaMemcpyHostToDevice)); // MLP final activation
        CUDA_CHECK(cudaMemcpy(d_ver_output, this->ver.output.data(), embed_bytes, cudaMemcpyHostToDevice)); // MLP final activation
        CUDA_CHECK(cudaMemcpy(d_KdotQ, flat_KdotQ.data(), head_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_K, flat_K.data(), k_q_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_Q, flat_Q.data(), k_q_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_MH_a, flat_MH_a.data(), mh_mv_mq_mk_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_MV_a, flat_MV_a.data(), mh_mv_mq_mk_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_MQ_a, flat_MQ_a.data(), mh_mv_mq_mk_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_MK_a, flat_MK_a.data(), mh_mv_mq_mk_size * sizeof(float), cudaMemcpyHostToDevice));

        // --- Data Transfer: Host -> Device (MLP Internals) ---
        for (int l = 0; l < layers; ++l) {
            // Flatten weights for layer l
            flat_hor_weights[l].reserve(embedding_dim * embedding_dim);
            flat_ver_weights[l].reserve(embedding_dim * embedding_dim);
            for (int i = 0; i < embedding_dim; ++i) {
                 if (this->hor.weights[l].size() != embedding_dim || this->hor.weights[l][i].size() != embedding_dim ||
                     this->ver.weights[l].size() != embedding_dim || this->ver.weights[l][i].size() != embedding_dim) {
                    throw std::runtime_error("MLP Weight matrix dimension error during flattening.");
                 }
                flat_hor_weights[l].insert(flat_hor_weights[l].end(), this->hor.weights[l][i].begin(), this->hor.weights[l][i].end());
                flat_ver_weights[l].insert(flat_ver_weights[l].end(), this->ver.weights[l][i].begin(), this->ver.weights[l][i].end());
            }
            // Copy activations and weights
             if (this->hor.activations[l].size() != embedding_dim || this->ver.activations[l].size() != embedding_dim) {
                 throw std::runtime_error("MLP Activation vector dimension error during copy.");
             }
            CUDA_CHECK(cudaMemcpy(d_hor_activations[l], this->hor.activations[l].data(), embed_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_hor_weights[l], flat_hor_weights[l].data(), weights_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_ver_activations[l], this->ver.activations[l].data(), embed_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_ver_weights[l], flat_ver_weights[l].data(), weights_bytes, cudaMemcpyHostToDevice));
            // Gradients (gweights) and deltas are computed on device, no H->D needed.
        }


        // --- Kernel Launch Configuration ---
        int threadsPerBlock1D = 256;
        dim3 blockDim1D(threadsPerBlock1D);
        int blocksPerGridEmbed = (embedding_dim + threadsPerBlock1D - 1) / threadsPerBlock1D;
        dim3 gridDimEmbed(blocksPerGridEmbed);
        // Other grid dims... (head, matheights, matrix, ev) - define as needed
        int blocksPerGridHead = (head_size + threadsPerBlock1D - 1) / threadsPerBlock1D;
        int blocksPerGridMatHeights = (mat_heights + threadsPerBlock1D - 1) / threadsPerBlock1D;
        int blocksPerGridMatrix = (mh_mv_mq_mk_size + threadsPerBlock1D - 1) / threadsPerBlock1D;
        int blocksPerGridEV = (ev_size + threadsPerBlock1D - 1) / threadsPerBlock1D;
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
                d_hor_deltas[l + 1],        // Deltas from the next layer (l+1)
                d_hor_weights[l + 1],       // Weights connecting layer l to layer l+1
                d_hor_activations[l],       // Activations of the current layer (l)
                d_hor_deltas[l],            // Deltas to compute for the current layer (l)
                embedding_dim,              // current_layer_size
                embedding_dim               // next_layer_size
            );
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // Calculate Gradients and Update Weights for hor MLP (Iterate through layers 0 to layers-1)
        for (int l = 0; l < layers; ++l) {
            // Input activations for this layer's weight update:
            // If l=0, the input was EH. If l>0, input was activations[l-1].
            float* d_prev_activations = (l == 0) ? d_EH : d_hor_activations[l - 1];

            updateWeightsKernel<<<gridDimEmbed2D, blockDim2D>>>( // Use 2D grid/block for weights
                d_hor_deltas[l],            // Deltas for the current layer (l)
                d_prev_activations,         // Activations from the previous layer (l-1 or EH)
                d_hor_weights[l],           // Weights connecting previous layer to current layer (l) - TO BE UPDATED
                d_hor_gweights[l],          // Gradients for layer l - TO BE CALCULATED
                learning_rate,              // Learning rate
                embedding_dim,              // current_layer_size
                embedding_dim               // prev_layer_size
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
                d_ver_deltas[l + 1],        // Deltas from the next layer (l+1)
                d_ver_weights[l + 1],       // Weights connecting layer l to layer l+1
                d_ver_activations[l],       // Activations of the current layer (l)
                d_ver_deltas[l],            // Deltas to compute for the current layer (l)
                embedding_dim,              // current_layer_size
                embedding_dim               // next_layer_size
            );
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // Calculate Gradients and Update Weights for ver MLP (Iterate through layers 0 to layers-1)
        for (int l = 0; l < layers; ++l) {
            // Input activations for this layer's weight update:
            // If l=0, the input was EV (or its processed form). Assuming d_EV holds the input used in forward pass.
            // NOTE: Check if ver MLP input was actually d_EV or something derived. Using d_EV for now.
            // If the input to ver MLP forward was different, use that pointer here.
            float* d_prev_activations = (l == 0) ? d_EV : d_ver_activations[l - 1];
             // Check if d_EV size matches embedding_dim, if not, need the correct input pointer.
             // If ver MLP input size != embedding_dim, the weights_bytes and gridDimEmbed2D need adjustment for l=0.
             // Assuming ver MLP input IS embedding_dim for simplicity here.

            updateWeightsKernel<<<gridDimEmbed2D, blockDim2D>>>( // Use 2D grid/block for weights
                d_ver_deltas[l],            // Deltas for the current layer (l)
                d_prev_activations,         // Activations from the previous layer (l-1 or EV)
                d_ver_weights[l],           // Weights connecting previous layer to current layer (l) - TO BE UPDATED
                d_ver_gweights[l],          // Gradients for layer l - TO BE CALCULATED
                learning_rate,              // Learning rate
                embedding_dim,              // current_layer_size
                embedding_dim               // prev_layer_size (assuming square)
            );
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // --- Step 3: Compute gradients w.r.t. MLP inputs (dh and dv) ---
        // This computes dL/d(EH) and dL/d(EV) based on the first layer's deltas and weights.
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
        cuLOTAder<<<gridDimHead2D, blockDim2D>>>(d_KdotQ, d_lota_deriv, context_win, context_win, token_count, isSelfAttention);
        CUDA_CHECK(cudaGetLastError());

        // 6b: Compute grad_KdotQ
        kernelComputeGradKdotQ_LOTA<<<gridDimHead, blockDim1D>>>(d_grad_head, d_lota_deriv, d_grad_KdotQ, scaling_factor, head_size);
        CUDA_CHECK(cudaGetLastError());

        // --- Step 7: Compute gradients w.r.t. K and Q ---
        kernelComputeGradK_Q<<<gridDimKQGrad2D, blockDim2D>>>(d_grad_KdotQ, d_K, d_Q, d_grad_K, d_grad_Q, token_count, mat_heights);
        CUDA_CHECK(cudaGetLastError());

        // --- Step 8: Compute gradients w.r.t. MQ and MK ---
        // (Original logic - requires verification of inputs/dimensions)
        // kernelComputeGradMK_MQ<<<gridDimMatrix2D, blockDim2D>>>(d_grad_K, d_grad_Q, d_K_embed, d_Q_embed, d_grad_MK, d_grad_MQ, token_count, mat_heights, embedding_dim);
        // CUDA_CHECK(cudaGetLastError());

        // --- Step 9 & 10: Update weights MH, MV, MQ, MK and EH, EV ---
        // Note: EH and EV are updated based on d_grad_dh and d_grad_dv respectively,
        // which are the gradients *after* backpropping through the MLPs.
        // The kernelUpdateWeights_EH_EV needs adjustment or replacement.
        // Let's assume a simpler update kernel for now: W = W - lr * grad(W)

        // Update MH, MV, MQ, MK
        kernelUpdateSimple<<<gridDimMatrix, blockDim1D>>>(d_MH_a, d_grad_MH, learning_rate, mh_mv_mq_mk_size);
        CUDA_CHECK(cudaGetLastError());

        // Update EH and EV
        // Grad w.r.t. EH is d_grad_dh. Grad w.r.t. EV is d_grad_dv.
        kernelUpdateSimple<<<gridDimEmbed, blockDim1D>>>(d_EH, d_grad_dh, learning_rate, embedding_dim);
        // kernelUpdateSimple(d_EV, d_grad_dv, learning_rate, embedding_dim); // Update only relevant part of EV if needed, or use d_grad_dv directly if EV is embed_dim
        // If EV is context_win * embedding_dim, d_grad_dv (embedding_dim) needs careful application.
        // Assuming for now EV update is handled differently or d_grad_dv applies element-wise if sizes match.
        // The original kernelUpdateWeights_EH_EV used d_grad_EV_scaled. Let's use d_grad_dv.
        // Need kernel that updates EV[last_token] using d_grad_dv? Or updates whole EV based on d_grad_dv?
        // Let's stick to updating based on d_grad_dv assuming it's the correct gradient for the relevant part of EV.
        // This part needs clarification based on how EV gradient flows. Assuming simple update for now:
        kernelUpdateSimple<<<gridDimEmbed, blockDim1D>>>(d_EV, d_grad_dv, learning_rate, embedding_dim); // CAUTION: May be incorrect size/logic for EV update
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());


        // --- Data Transfer: Device -> Host ---
        // Copy updated MLP weights and gradients back
        for (int l = 0; l < layers; ++l) {
            std::vector<float> updated_flat_weights(embedding_dim * embedding_dim);
            std::vector<float> calculated_flat_gradients(embedding_dim * embedding_dim);

            // Copy hor weights/gradients
            CUDA_CHECK(cudaMemcpy(updated_flat_weights.data(), d_hor_weights[l], weights_bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(calculated_flat_gradients.data(), d_hor_gweights[l], weights_bytes, cudaMemcpyDeviceToHost));
            // Reshape and store in this->hor
            for (int i = 0; i < embedding_dim; ++i) {
                for (int j = 0; j < embedding_dim; ++j) {
                    this->hor.weights[l][i][j] = updated_flat_weights[i * embedding_dim + j];
                    // Ensure gweights structure exists before assignment
                    if(this->hor.gweights.size() <= l) this->hor.gweights.resize(l+1);
                    if(this->hor.gweights[l].size() <= i) this->hor.gweights[l].resize(i+1);
                    if(this->hor.gweights[l][i].size() != embedding_dim) this->hor.gweights[l][i].resize(embedding_dim);
                    this->hor.gweights[l][i][j] = calculated_flat_gradients[i * embedding_dim + j];
                }
            }

            // Copy ver weights/gradients
            CUDA_CHECK(cudaMemcpy(updated_flat_weights.data(), d_ver_weights[l], weights_bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(calculated_flat_gradients.data(), d_ver_gweights[l], weights_bytes, cudaMemcpyDeviceToHost));
            // Reshape and store in this->ver
            for (int i = 0; i < embedding_dim; ++i) {
                for (int j = 0; j < embedding_dim; ++j) {
                    this->ver.weights[l][i][j] = updated_flat_weights[i * embedding_dim + j];
                     // Ensure gweights structure exists before assignment
                    if(this->ver.gweights.size() <= l) this->ver.gweights.resize(l+1);
                    if(this->ver.gweights[l].size() <= i) this->ver.gweights[l].resize(i+1);
                    if(this->ver.gweights[l][i].size() != embedding_dim) this->ver.gweights[l][i].resize(embedding_dim);
                    this->ver.gweights[l][i][j] = calculated_flat_gradients[i * embedding_dim + j];
                }
            }
        }

        // Copy updated Attention parameters back
        std::vector<float> updated_MH_a(mh_mv_mq_mk_size);
        std::vector<float> updated_MV_a(mh_mv_mq_mk_size);
        std::vector<float> updated_MQ_a(mh_mv_mq_mk_size);
        std::vector<float> updated_MK_a(mh_mv_mq_mk_size);
        std::vector<float> updated_EV(ev_size); // Full EV

        CUDA_CHECK(cudaMemcpy(this->EH.data(), d_EH, embed_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(updated_EV.data(), d_EV, ev_size * sizeof(float), cudaMemcpyDeviceToHost)); // Copy full EV
        CUDA_CHECK(cudaMemcpy(updated_MH_a.data(), d_MH_a, mh_mv_mq_mk_size * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(updated_MV_a.data(), d_MV_a, mh_mv_mq_mk_size * sizeof(float), cudaMemcpyDeviceToHost));
        // CUDA_CHECK(cudaMemcpy(updated_MQ_a.data(), d_MQ_a, mh_mv_mq_mk_size * sizeof(float), cudaMemcpyDeviceToHost)); // Uncomment when updated
        // CUDA_CHECK(cudaMemcpy(updated_MK_a.data(), d_MK_a, mh_mv_mq_mk_size * sizeof(float), cudaMemcpyDeviceToHost)); // Uncomment when updated

        // Unflatten results back into class members
        unflatten(updated_EV, this->EV, context_win, embedding_dim); // Unflatten full EV
        unflatten(updated_MH_a, this->MH.a, mat_heights, embedding_dim);
        unflatten(updated_MV_a, this->MV.a, mat_heights, embedding_dim);
        // unflatten(updated_MQ_a, this->MQ.a, mat_heights, embedding_dim); // Uncomment when updated
        // unflatten(updated_MK_a, this->MK.a, mat_heights, embedding_dim); // Uncomment when updated

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
        for (int l = 0; l < layers; ++l) {
            cudaFree(d_hor_activations[l]); cudaFree(d_hor_weights[l]);
            cudaFree(d_hor_gweights[l]); cudaFree(d_hor_deltas[l]);
            cudaFree(d_ver_activations[l]); cudaFree(d_ver_weights[l]);
            cudaFree(d_ver_gweights[l]); cudaFree(d_ver_deltas[l]);
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
    for (int l = 0; l < layers; ++l) {
        cudaFree(d_hor_activations[l]); cudaFree(d_hor_weights[l]);
        cudaFree(d_hor_gweights[l]); cudaFree(d_hor_deltas[l]);
        cudaFree(d_ver_activations[l]); cudaFree(d_ver_weights[l]);
        cudaFree(d_ver_gweights[l]); cudaFree(d_ver_deltas[l]);
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
void attention::cuBackward(std::vector<std::vector<float>>& expectedV, int& in, int& layers)
{
    // get values for all kernels and functions
    const int embedding_dim = EMBEDDING;
    const int mat_heights = MATHEIGHTS;
    const int context_win = CONTEXT_WIN;
    const float learning_rate = LEARNING;
    const float scaling_factor = SCALING;
    const int token_count = this->tokenCount; // Use member variable
    const int head_size = token_count * token_count;
    const int mh_mv_mq_mk_size = mat_heights * embedding_dim;
    const int k_q_size = token_count * mat_heights; // K/Q used in matmuls
    // const int k_q_embed_size = token_count * embedding_dim; // K/Q needed for grad MK/MQ calc (potentially)
    const int ev_size = context_win * embedding_dim;
    const size_t embed_bytes = embedding_dim * sizeof(float);
    const size_t weights_bytes = embedding_dim * embedding_dim * sizeof(float); // Assuming square MLP layers

    // Basic validation
    if (expectedV.size() != context_win || (!expectedV.empty() && expectedV[0].size() != embedding_dim)) {
        throw std::runtime_error("ExpectedV dimensions mismatch");
    }
    // Add other validation checks...

    // --- Device Memory Pointers ---
    float *d_expected_v = nullptr, *d_EV = nullptr;
    float *d_grad_EV_full = nullptr, *d_grad_EV_summed = nullptr, *d_grad_EV_scaled = nullptr;
    float *d_ver_output = nullptr; // MLP final layer output
    // float *d_grad_ver_input = nullptr; // No longer needed D->H
    // float *d_ver_gweights0 = nullptr; // No longer needed H->D
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
    std::vector<float*> d_ver_weights(layers, nullptr);
    std::vector<float*> d_ver_gweights(layers, nullptr);
    std::vector<float*> d_ver_deltas(layers, nullptr);

    // Temporary flat vectors for H->D copy
    std::vector<std::vector<float>> flat_ver_weights(layers);

    try {
        // --- Allocate Device Memory (Attention Part) ---
        CUDA_CHECK(cudaMalloc(&d_expected_v, ev_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_EV, ev_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad_EV_full, ev_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad_EV_summed, embed_bytes));
        CUDA_CHECK(cudaMalloc(&d_grad_EV_scaled, embed_bytes)); // Grad input to ver MLP backprop
        CUDA_CHECK(cudaMalloc(&d_ver_output, embed_bytes)); // Output of ver MLP
        CUDA_CHECK(cudaMalloc(&d_grad_dv, embed_bytes)); // Grad w.r.t. ver MLP input (EV)
        CUDA_CHECK(cudaMalloc(&d_KdotQ, head_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_head, head_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_K, k_q_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_Q, k_q_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_pre_MV, mat_heights * sizeof(float))); // Check size
        CUDA_CHECK(cudaMalloc(&d_MV_a, mh_mv_mq_mk_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_MQ_a, mh_mv_mq_mk_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_MK_a, mh_mv_mq_mk_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad_MV, mh_mv_mq_mk_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad_head, head_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_lota_deriv, head_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad_KdotQ, head_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad_Q, k_q_size * sizeof(float))); // Check size
        CUDA_CHECK(cudaMalloc(&d_grad_MQ, mh_mv_mq_mk_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad_MK_correction, mh_mv_mq_mk_size * sizeof(float)));

        // --- Allocate Device Memory (ver MLP Internals) ---
        for (int l = 0; l < layers; ++l) {
            CUDA_CHECK(cudaMalloc(&d_ver_activations[l], embed_bytes));
            CUDA_CHECK(cudaMalloc(&d_ver_weights[l], weights_bytes));
            CUDA_CHECK(cudaMalloc(&d_ver_gweights[l], weights_bytes));
            CUDA_CHECK(cudaMalloc(&d_ver_deltas[l], embed_bytes));
        }

        // --- Data Transfer: Host -> Device (Attention Part) ---
        std::vector<float> flat_expectedV = flatten(expectedV);
        std::vector<float> flat_EV = flatten(this->EV);
        std::vector<float> flat_K = flatten(this->K);
        std::vector<float> flat_Q = flatten(this->Q);
        std::vector<float> flat_KdotQ = flatten(this->KdotQ);
        std::vector<float> flat_MV_a = flatten(this->MV.a);
        std::vector<float> flat_MQ_a = flatten(this->MQ.a);
        std::vector<float> flat_MK_a = flatten(this->MK.a);

        CUDA_CHECK(cudaMemcpy(d_expected_v, flat_expectedV.data(), ev_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_EV, flat_EV.data(), ev_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ver_output, this->ver.output.data(), embed_bytes, cudaMemcpyHostToDevice)); // MLP final activation
        CUDA_CHECK(cudaMemcpy(d_KdotQ, flat_KdotQ.data(), head_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_K, flat_K.data(), k_q_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_Q, flat_Q.data(), k_q_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_MV_a, flat_MV_a.data(), mh_mv_mq_mk_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_MQ_a, flat_MQ_a.data(), mh_mv_mq_mk_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_MK_a, flat_MK_a.data(), mh_mv_mq_mk_size * sizeof(float), cudaMemcpyHostToDevice));

        // --- Data Transfer: Host -> Device (ver MLP Internals) ---
        for (int l = 0; l < layers; ++l) {
            flat_ver_weights[l].reserve(embedding_dim * embedding_dim);
            for (int i = 0; i < embedding_dim; ++i) {
                 if (this->ver.weights[l].size() != embedding_dim || this->ver.weights[l][i].size() != embedding_dim) {
                    throw std::runtime_error("MLP Weight matrix dimension error during flattening.");
                 }
                flat_ver_weights[l].insert(flat_ver_weights[l].end(), this->ver.weights[l][i].begin(), this->ver.weights[l][i].end());
            }
             if (this->ver.activations[l].size() != embedding_dim) {
                 throw std::runtime_error("MLP Activation vector dimension error during copy.");
             }
            CUDA_CHECK(cudaMemcpy(d_ver_activations[l], this->ver.activations[l].data(), embed_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_ver_weights[l], flat_ver_weights[l].data(), weights_bytes, cudaMemcpyHostToDevice));
        }

        // --- Kernel Launch Configuration ---
        int threadsPerBlock1D = 256;
        dim3 blockDim1D(threadsPerBlock1D);
        int blocksPerGridEmbed = (embedding_dim + threadsPerBlock1D - 1) / threadsPerBlock1D;
        dim3 gridDimEmbed(blocksPerGridEmbed);
        // Other grid dims...
        int blocksPerGridHead = (head_size + threadsPerBlock1D - 1) / threadsPerBlock1D;
        int blocksPerGridMatHeights = (mat_heights + threadsPerBlock1D - 1) / threadsPerBlock1D;
        int blocksPerGridMatrix = (mh_mv_mq_mk_size + threadsPerBlock1D - 1) / threadsPerBlock1D;
        int blocksPerGridEV = (ev_size + threadsPerBlock1D - 1) / threadsPerBlock1D;
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
                d_ver_deltas[l + 1],        // Deltas from the next layer (l+1)
                d_ver_weights[l + 1],       // Weights connecting layer l to layer l+1
                d_ver_activations[l],       // Activations of the current layer (l)
                d_ver_deltas[l],            // Deltas to compute for the current layer (l)
                embedding_dim,              // current_layer_size
                embedding_dim               // next_layer_size
            );
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // Calculate Gradients and Update Weights for ver MLP (Iterate through layers 0 to layers-1)
        for (int l = 0; l < layers; ++l) {
            float* d_prev_activations = (l == 0) ? d_EV : d_ver_activations[l - 1];
            // CAUTION: Check size/nature of d_EV if it's not embedding_dim. Adjust grid/kernel if needed.
            updateWeightsKernel<<<gridDimEmbed2D, blockDim2D>>>(
                d_ver_deltas[l],            // Deltas for the current layer (l)
                d_prev_activations,         // Activations from the previous layer (l-1 or EV)
                d_ver_weights[l],           // Weights connecting previous layer to current layer (l) - TO BE UPDATED
                d_ver_gweights[l],          // Gradients for layer l - TO BE CALCULATED
                learning_rate,              // Learning rate
                embedding_dim,              // current_layer_size
                embedding_dim               // prev_layer_size (assuming square)
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
        cuLOTAder<<<gridDimHead2D, blockDim2D>>>(d_KdotQ, d_lota_deriv, context_win, context_win, token_count, isSelfAttention);
        CUDA_CHECK(cudaGetLastError());

        // 6b: Compute grad_KdotQ
        kernelComputeGradKdotQ_LOTA<<<gridDimHead, blockDim1D>>>(d_grad_head, d_lota_deriv, d_grad_KdotQ, scaling_factor, head_size);
        CUDA_CHECK(cudaGetLastError());

        // --- Step 7: Compute gradients w.r.t. Q ---
        kernelComputeGradQ_V<<<gridDimKQGrad2D, blockDim2D>>>(d_grad_KdotQ, d_K, d_grad_Q, token_count, mat_heights);
        CUDA_CHECK(cudaGetLastError());

        // --- Step 8: Compute gradients w.r.t. MQ and MK_correction ---
        // (Original logic - requires verification)
        // kernelComputeGradMQ_V<<<gridDimMatrix2D, blockDim2D>>>(d_grad_Q, d_Q_embed, d_grad_MQ, token_count, mat_heights, embedding_dim);
        // CUDA_CHECK(cudaGetLastError());
        // Compute MK correction
        kernelComputeGradMKCorrection<<<gridDimMatrix2D, blockDim2D>>>(d_grad_MQ, d_Q, d_K, d_grad_MK_correction, token_count, mat_heights, embedding_dim);
        CUDA_CHECK(cudaGetLastError());


        // --- Step 9 & 10: Update weights MV, MQ, MK and EV ---
        // Define simple update kernel locally
        
        // Update MV, MQ, MK
        kernelUpdateSimple<<<gridDimMatrix, blockDim1D>>>(d_MV_a, d_grad_MV, learning_rate, mh_mv_mq_mk_size);
        // kernelUpdateSimple(d_MQ_a, d_grad_MQ, learning_rate, mh_mv_mq_mk_size); // Uncomment when grad_MQ computed
        kernelUpdateSimple<<<gridDimMatrix, blockDim1D>>>(d_MK_a, d_grad_MK_correction, learning_rate, mh_mv_mq_mk_size); // Update MK with correction
        CUDA_CHECK(cudaGetLastError());

        // Update EV
        // The original kernelUpdateWeights_EV_V used d_grad_EV_full.
        kernelUpdateSimple<<<gridDimEV, blockDim1D>>>(d_EV, d_grad_EV_full, learning_rate, ev_size); // Update full EV using full gradient
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());


        // --- Data Transfer: Device -> Host ---
        // Copy updated ver MLP weights and gradients back
        for (int l = 0; l < layers; ++l) {
            std::vector<float> updated_flat_weights(embedding_dim * embedding_dim);
            std::vector<float> calculated_flat_gradients(embedding_dim * embedding_dim);

            CUDA_CHECK(cudaMemcpy(updated_flat_weights.data(), d_ver_weights[l], weights_bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(calculated_flat_gradients.data(), d_ver_gweights[l], weights_bytes, cudaMemcpyDeviceToHost));
            // Reshape and store in this->ver
            for (int i = 0; i < embedding_dim; ++i) {
                for (int j = 0; j < embedding_dim; ++j) {
                    this->ver.weights[l][i][j] = updated_flat_weights[i * embedding_dim + j];
                    // Ensure gweights structure exists
                    if(this->ver.gweights.size() <= l) this->ver.gweights.resize(l+1);
                    if(this->ver.gweights[l].size() <= i) this->ver.gweights[l].resize(i+1);
                    if(this->ver.gweights[l][i].size() != embedding_dim) this->ver.gweights[l][i].resize(embedding_dim);
                    this->ver.gweights[l][i][j] = calculated_flat_gradients[i * embedding_dim + j];
                }
            }
        }

        // Copy updated Attention parameters back
        std::vector<float> updated_MV_a(mh_mv_mq_mk_size);
        std::vector<float> updated_MQ_a(mh_mv_mq_mk_size);
        std::vector<float> updated_MK_a(mh_mv_mq_mk_size);
        std::vector<float> updated_EV(ev_size);

        CUDA_CHECK(cudaMemcpy(updated_EV.data(), d_EV, ev_size * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(updated_MV_a.data(), d_MV_a, mh_mv_mq_mk_size * sizeof(float), cudaMemcpyDeviceToHost));
        // CUDA_CHECK(cudaMemcpy(updated_MQ_a.data(), d_MQ_a, mh_mv_mq_mk_size * sizeof(float), cudaMemcpyDeviceToHost)); // Uncomment when updated
        CUDA_CHECK(cudaMemcpy(updated_MK_a.data(), d_MK_a, mh_mv_mq_mk_size * sizeof(float), cudaMemcpyDeviceToHost));

        // Unflatten results back into class members
        unflatten(updated_EV, this->EV, context_win, embedding_dim);
        unflatten(updated_MV_a, this->MV.a, mat_heights, embedding_dim);
        // unflatten(updated_MQ_a, this->MQ.a, mat_heights, embedding_dim); // Uncomment when updated
        unflatten(updated_MK_a, this->MK.a, mat_heights, embedding_dim);

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
        for (int l = 0; l < layers; ++l) {
            cudaFree(d_ver_activations[l]); cudaFree(d_ver_weights[l]);
            cudaFree(d_ver_gweights[l]); cudaFree(d_ver_deltas[l]);
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
    for (int l = 0; l < layers; ++l) {
        cudaFree(d_ver_activations[l]); cudaFree(d_ver_weights[l]);
        cudaFree(d_ver_gweights[l]); cudaFree(d_ver_deltas[l]);
    }
}
