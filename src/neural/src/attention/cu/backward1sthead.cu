
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
* @brief Backward Propagation (for first head) using gradients from expected Horizontal output.
*      Updates MH, MV, MQ, MK, and conditionally EH.
* @param expected Expected output vector (target embedding for next token prediction)
* @param in Input size (embedding dimension)
* @param layers Number of layers in the MLPs
* @param first Boolean flag - Determines if EH is updated. (Now implicitly true in this overload)
*/
void attention::cuBackward1stHead(std::vector<float>& expected, int& in, int& layers)
{
    // --- Setup ---
    const int embedding_dim = EMBEDDING;
    const int mat_heights = MATHEIGHTS;
    const int context_win = CONTEXT_WIN;
    const float learning_rate = LEARNING;
    const float scaling_factor = SCALING;
    const int token_count = this->tokenCount;
    const int head_size = token_count * token_count;
    const int mh_mv_mq_mk_size = mat_heights * embedding_dim;
    const int k_q_size = token_count * mat_heights;
    // const int k_q_embed_size = token_count * embedding_dim; // For simplified grad MK/MQ
    const size_t embed_bytes = embedding_dim * sizeof(float);
    const size_t weights_bytes = embedding_dim * embedding_dim * sizeof(float); // Assuming square MLP layers
    const size_t ev_size = context_win * embedding_dim; // Size of EV

    bool first = true; // This overload implies first=true for EH update

    // Validation
    if (embedding_dim != in) throw std::runtime_error("Embedding dimension mismatch");
    if (expected.size() != embedding_dim) throw std::runtime_error("Expected vector size mismatch");
    // Add other necessary validation checks...

    // --- Device Pointers (Attention) ---
    float *d_expected_h = nullptr, *d_EH = nullptr, *d_EV = nullptr; // Added d_EV
    float *d_grad_EH = nullptr, *d_grad_EV_scaled = nullptr;
    // float *d_hor_output = nullptr, *d_ver_output = nullptr; // Use MLP activations directly
    // float *d_grad_hor_input = nullptr, *d_grad_ver_input = nullptr; // Not needed with kernelLastLayerDelta
    // float *d_hor_gweights0 = nullptr, *d_ver_gweights0 = nullptr; // Gradients computed and stored in d_hor/ver_gweights
    float *d_grad_dh = nullptr, *d_grad_dv = nullptr; // Gradients w.r.t MLP inputs
    float *d_KdotQ = nullptr, *d_head = nullptr;
    float *d_K = nullptr, *d_Q = nullptr; // token x mat_heights
    float *d_K_embed = nullptr, *d_Q_embed = nullptr; // token x embedding_dim (Needed for Step 8?)
    float *d_pre_MH = nullptr, *d_pre_MV = nullptr;
    float *d_MH_a = nullptr, *d_MV_a = nullptr, *d_MQ_a = nullptr, *d_MK_a = nullptr;
    float *d_grad_MH = nullptr, *d_grad_MV = nullptr;
    float *d_grad_head = nullptr;
    float *d_lota_deriv = nullptr;
    float *d_grad_KdotQ = nullptr;
    float *d_grad_K = nullptr, *d_grad_Q = nullptr;
    float *d_grad_MQ = nullptr, *d_grad_MK = nullptr;

    // --- Device Pointers (MLP Internals) ---
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
        // --- Allocate Memory (Attention) ---
        CUDA_CHECK(cudaMalloc(&d_expected_h, embed_bytes));
        CUDA_CHECK(cudaMalloc(&d_EH, embed_bytes));
        CUDA_CHECK(cudaMalloc(&d_EV, ev_size * sizeof(float))); // Allocate EV
        CUDA_CHECK(cudaMalloc(&d_grad_EH, embed_bytes));
        CUDA_CHECK(cudaMalloc(&d_grad_EV_scaled, embed_bytes));
        // CUDA_CHECK(cudaMalloc(&d_hor_output, embed_bytes)); // Not needed
        // CUDA_CHECK(cudaMalloc(&d_ver_output, embed_bytes)); // Not needed
        // CUDA_CHECK(cudaMalloc(&d_grad_hor_input, embed_bytes)); // Not needed
        // CUDA_CHECK(cudaMalloc(&d_grad_ver_input, embed_bytes)); // Not needed
        // CUDA_CHECK(cudaMalloc(&d_hor_gweights0, weights_bytes)); // Not needed
        // CUDA_CHECK(cudaMalloc(&d_ver_gweights0, weights_bytes)); // Not needed
        CUDA_CHECK(cudaMalloc(&d_grad_dh, embed_bytes));
        CUDA_CHECK(cudaMalloc(&d_grad_dv, embed_bytes));
        CUDA_CHECK(cudaMalloc(&d_KdotQ, head_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_head, head_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_K, k_q_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_Q, k_q_size * sizeof(float)));
        // CUDA_CHECK(cudaMalloc(&d_K_embed, k_q_embed_size * sizeof(float))); // If needed
        // CUDA_CHECK(cudaMalloc(&d_Q_embed, k_q_embed_size * sizeof(float))); // If needed
        CUDA_CHECK(cudaMalloc(&d_pre_MH, mat_heights * sizeof(float))); // Check size
        CUDA_CHECK(cudaMalloc(&d_pre_MV, mat_heights * sizeof(float))); // Check size
        CUDA_CHECK(cudaMalloc(&d_MH_a, mh_mv_mq_mk_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_MV_a, mh_mv_mq_mk_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_MQ_a, mh_mv_mq_mk_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_MK_a, mh_mv_mq_mk_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad_MH, mh_mv_mq_mk_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad_MV, mh_mv_mq_mk_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad_head, head_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_lota_deriv, head_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad_KdotQ, head_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad_K, k_q_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad_Q, k_q_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad_MQ, mh_mv_mq_mk_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad_MK, mh_mv_mq_mk_size * sizeof(float)));

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
        std::vector<float> flat_K = flatten(this->K);
        std::vector<float> flat_Q = flatten(this->Q);
        std::vector<float> flat_EV = flatten(this->EV); // Flatten EV
        std::vector<float> flat_KdotQ = flatten(this->KdotQ);
        std::vector<float> flat_MH_a = flatten(this->MH.a);
        std::vector<float> flat_MV_a = flatten(this->MV.a);
        std::vector<float> flat_MQ_a = flatten(this->MQ.a);
        std::vector<float> flat_MK_a = flatten(this->MK.a);
        // Flatten and copy K_embed/Q_embed if they exist and are needed
        // std::vector<float> flat_K_embed = flatten(this->K_embed);
        // std::vector<float> flat_Q_embed = flatten(this->Q_embed);

        CUDA_CHECK(cudaMemcpy(d_expected_h, expected.data(), embed_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_EH, this->EH.data(), embed_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_EV, flat_EV.data(), ev_size * sizeof(float), cudaMemcpyHostToDevice)); // Copy EV
        // CUDA_CHECK(cudaMemcpy(d_hor_output, this->hor.output.data(), embed_bytes, cudaMemcpyHostToDevice)); // Not needed
        // CUDA_CHECK(cudaMemcpy(d_ver_output, this->ver.output.data(), embed_bytes, cudaMemcpyHostToDevice)); // Not needed
        CUDA_CHECK(cudaMemcpy(d_KdotQ, flat_KdotQ.data(), head_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_K, flat_K.data(), k_q_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_Q, flat_Q.data(), k_q_size * sizeof(float), cudaMemcpyHostToDevice));
        // CUDA_CHECK(cudaMemcpy(d_K_embed, flat_K_embed.data(), k_q_embed_size * sizeof(float), cudaMemcpyHostToDevice)); // If needed
        // CUDA_CHECK(cudaMemcpy(d_Q_embed, flat_Q_embed.data(), k_q_embed_size * sizeof(float), cudaMemcpyHostToDevice)); // If needed
        CUDA_CHECK(cudaMemcpy(d_MH_a, flat_MH_a.data(), mh_mv_mq_mk_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_MV_a, flat_MV_a.data(), mh_mv_mq_mk_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_MQ_a, flat_MQ_a.data(), mh_mv_mq_mk_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_MK_a, flat_MK_a.data(), mh_mv_mq_mk_size * sizeof(float), cudaMemcpyHostToDevice));

        // --- Data Transfer H->D (MLP Internals) ---
        for (int l = 0; l < layers; ++l) {
            // Flatten weights
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
        }


        // --- Kernel Launch Config ---
        int threadsPerBlock1D = 256;
        int blocksPerGridEmbed = (embedding_dim + threadsPerBlock1D - 1) / threadsPerBlock1D;
        int blocksPerGridHead = (head_size + threadsPerBlock1D - 1) / threadsPerBlock1D;
        int blocksPerGridMatHeights = (mat_heights + threadsPerBlock1D - 1) / threadsPerBlock1D;
        int blocksPerGridMatrix = (mh_mv_mq_mk_size + threadsPerBlock1D - 1) / threadsPerBlock1D;
        // int blocksPerGridToken = (token_count + threadsPerBlock1D - 1) / threadsPerBlock1D; // If needed

        dim3 blockDim1D(threadsPerBlock1D);
        dim3 gridDimEmbed(blocksPerGridEmbed);
        dim3 gridDimHead(blocksPerGridHead);
        dim3 gridDimMatHeights(blocksPerGridMatHeights);
        dim3 gridDimMatrix(blocksPerGridMatrix);
        // dim3 gridDimToken(blocksPerGridToken); // If needed

        dim3 blockDim2D(16, 16);
        dim3 gridDimEmbed2D((embedding_dim + blockDim2D.x - 1) / blockDim2D.x, (embedding_dim + blockDim2D.y - 1) / blockDim2D.y); // For MLP weights
        dim3 gridDimHead2D((token_count + blockDim2D.x - 1) / blockDim2D.x, (token_count + blockDim2D.y - 1) / blockDim2D.y);
        dim3 gridDimMatrix2D((embedding_dim + blockDim2D.x - 1) / blockDim2D.x, (mat_heights + blockDim2D.y - 1) / blockDim2D.y);
        dim3 gridDimKQGrad2D((mat_heights + blockDim2D.x - 1) / blockDim2D.x, (token_count + blockDim2D.y - 1) / blockDim2D.y);

        // --- Backpropagation Steps ---

        // Step 1: Compute grad_EH and grad_EV_scaled
        kernelComputeGradientsEH_EV<<<gridDimEmbed, blockDim1D>>>(d_EH, d_expected_h, d_grad_EH, d_grad_EV_scaled, embedding_dim);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize()); // Sync before MLP backprop

        // --- Step 2: Backprop through MLPs ---
        // --- 2a: Backprop through hor MLP ---
        // Calculate Output Layer Deltas (Layer layers-1)
        kernelLastLayerDelta<<<gridDimEmbed, blockDim1D>>>(
            d_grad_EH, d_hor_activations[layers - 1], d_hor_deltas[layers - 1], embedding_dim);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Calculate Hidden Layer Deltas (Propagate backwards from layers-2 down to 0)
        for (int l = layers - 2; l >= 0; --l) {
            hiddenDeltaKernel<<<gridDimEmbed, blockDim1D>>>(
                d_hor_deltas[l + 1], d_hor_weights[l + 1], d_hor_activations[l], d_hor_deltas[l],
                embedding_dim, embedding_dim);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // Calculate Gradients and Update Weights (Iterate through layers 0 to layers-1)
        for (int l = 0; l < layers; ++l) {
            float* d_prev_activations = (l == 0) ? d_EH : d_hor_activations[l - 1];
            updateWeightsKernel<<<gridDimEmbed2D, blockDim2D>>>(
                d_hor_deltas[l], d_prev_activations, d_hor_weights[l], d_hor_gweights[l],
                learning_rate, embedding_dim, embedding_dim);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // --- 2b: Backprop through ver MLP ---
        // Calculate Output Layer Deltas (Layer layers-1)
        kernelLastLayerDelta<<<gridDimEmbed, blockDim1D>>>(
            d_grad_EV_scaled, d_ver_activations[layers - 1], d_ver_deltas[layers - 1], embedding_dim);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Calculate Hidden Layer Deltas (Propagate backwards from layers-2 down to 0)
        for (int l = layers - 2; l >= 0; --l) {
            hiddenDeltaKernel<<<gridDimEmbed, blockDim1D>>>(
                d_ver_deltas[l + 1], d_ver_weights[l + 1], d_ver_activations[l], d_ver_deltas[l],
                embedding_dim, embedding_dim);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // Calculate Gradients and Update Weights (Iterate through layers 0 to layers-1)
        for (int l = 0; l < layers; ++l) {
            // Input to ver MLP layer 0 is EV. Need to handle its size (ev_size vs embed_dim).
            // Assuming updateWeightsKernel expects prev_layer_size = embed_dim.
            // This requires clarification: Was the input to ver MLP forward pass the full EV or just a part?
            // If full EV (context_win * embed_dim), the first layer weights are [embed_dim x ev_size].
            // If only last token's EV part (embed_dim), weights are [embed_dim x embed_dim].
            // Assuming the latter for consistency with square weights assumption.
            // If EV is the input, need to adjust kernel/grid/weights size for l=0.
            // *** Sticking to square assumption: input is embed_dim ***
            // *** This implies the forward pass likely used only a slice of EV or processed it first ***
            float* d_prev_activations = (l == 0) ? d_EV : d_ver_activations[l - 1]; // CAUTION: Using d_EV directly might be wrong size/pointer
            int prev_layer_size = (l == 0) ? embedding_dim : embedding_dim; // Adjust if EV is not embed_dim input
            dim3 currentGridDim2D = gridDimEmbed2D; // Adjust if prev_layer_size changes for l=0

            updateWeightsKernel<<<currentGridDim2D, blockDim2D>>>(
                d_ver_deltas[l], d_prev_activations, d_ver_weights[l], d_ver_gweights[l],
                learning_rate, embedding_dim, prev_layer_size);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        // --- End of MLP Backprop ---


        // --- Step 3: Compute grad_dh and grad_dv ---
        // Use the computed deltas and weights from the first MLP layer
        kernelComputeGradMLPInput<<<gridDimEmbed, blockDim1D>>>(
            d_hor_deltas[0], d_hor_weights[0], d_grad_dh, embedding_dim, embedding_dim);
        CUDA_CHECK(cudaGetLastError());

        kernelComputeGradMLPInput<<<gridDimEmbed, blockDim1D>>>(
            d_ver_deltas[0], d_ver_weights[0], d_grad_dv, embedding_dim, embedding_dim); // Assuming ver input size is embed_dim
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());


        // --- Step 4: Compute grad_MH and grad_MV ---
        cuLOTA<<<gridDimHead2D, blockDim2D>>>(d_KdotQ, d_head, context_win, context_win, token_count, isSelfAttention);
        CUDA_CHECK(cudaGetLastError());
        kernelComputePreMH_MV<<<gridDimMatHeights, blockDim1D>>>(d_head, d_K, d_Q, d_pre_MH, d_pre_MV, token_count, mat_heights);
        CUDA_CHECK(cudaGetLastError());
        kernelComputeGradMH_MV<<<gridDimMatrix2D, blockDim2D>>>(d_pre_MH, d_pre_MV, d_grad_dh, d_grad_dv, d_grad_MH, d_grad_MV, mat_heights, embedding_dim);
        CUDA_CHECK(cudaGetLastError());

        // --- Step 5: Compute grad_head ---
        kernelComputeGradHead<<<gridDimHead2D, blockDim2D>>>(d_K, d_Q, d_MH_a, d_MV_a, d_grad_dh, d_grad_dv, d_grad_head, token_count, mat_heights, embedding_dim);
        CUDA_CHECK(cudaGetLastError());

        // --- Step 6: Backprop through LOTA (Simple Derivative) ---
        cuLOTAder<<<gridDimHead2D, blockDim2D>>>(d_KdotQ, d_lota_deriv, context_win, context_win, token_count, isSelfAttention); // Use d_head as input? Check kernel def. Original used KdotQ.
        // Assuming cuLOTAder takes KdotQ as input based on mlp::cuBackward modification:
        // cuLOTAder<<<gridDimHead2D, blockDim2D>>>(d_KdotQ, d_lota_deriv, context_win, context_win, token_count, isSelfAttention);
        CUDA_CHECK(cudaGetLastError());
        kernelComputeGradKdotQ_LOTA<<<gridDimHead, blockDim1D>>>(d_grad_head, d_lota_deriv, d_grad_KdotQ, scaling_factor, head_size);
        CUDA_CHECK(cudaGetLastError());

        // --- Step 7: Compute grad_K and grad_Q ---
        kernelComputeGradK_Q<<<gridDimKQGrad2D, blockDim2D>>>(d_grad_KdotQ, d_K, d_Q, d_grad_K, d_grad_Q, token_count, mat_heights);
        CUDA_CHECK(cudaGetLastError());

        // --- Step 8: Compute grad_MK and grad_MQ (Simplified) ---
        float* actual_d_K_embed = nullptr; // Replace with d_K_embed if allocated and copied
        float* actual_d_Q_embed = nullptr; // Replace with d_Q_embed if allocated and copied
        // if (!actual_d_K_embed || !actual_d_Q_embed) {
        //     std::cerr << "Warning: Simplified grad_MK/MQ calculation in cuBackward1stHead(expected,...) might be incorrect without K_embed/Q_embed." << std::endl;
        // }
        kernelComputeGradMK_MQ_Simplified<<<gridDimMatrix2D, blockDim2D>>>(
            d_grad_K, d_grad_Q, actual_d_K_embed, actual_d_Q_embed, d_grad_MK, d_grad_MQ,
            token_count, mat_heights, embedding_dim);
        CUDA_CHECK(cudaGetLastError());

        // --- Step 9 & 10: Update Weights (Conditionally EH) ---
        kernelUpdateWeights_1stHead_H<<<gridDimMatrix, blockDim1D>>>(
            d_MH_a, d_MV_a, d_MQ_a, d_MK_a, d_EH,
            d_grad_MH, d_grad_MV, d_grad_MQ, d_grad_MK, d_grad_EH,
            learning_rate, first, // Pass the 'first' flag (always true here)
            mat_heights, embedding_dim
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize()); // Sync before D->H copy


        // --- Data Transfer D->H ---
        // Copy updated MLP weights and gradients back
        for (int l = 0; l < layers; ++l) {
            std::vector<float> updated_flat_weights(embedding_dim * embedding_dim);
            std::vector<float> calculated_flat_gradients(embedding_dim * embedding_dim);
            // Hor MLP
            CUDA_CHECK(cudaMemcpy(updated_flat_weights.data(), d_hor_weights[l], weights_bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(calculated_flat_gradients.data(), d_hor_gweights[l], weights_bytes, cudaMemcpyDeviceToHost));
            for (int i = 0; i < embedding_dim; ++i) {
                for (int j = 0; j < embedding_dim; ++j) {
                    this->hor.weights[l][i][j] = updated_flat_weights[i * embedding_dim + j];
                    if(this->hor.gweights.size() <= l) this->hor.gweights.resize(l+1);
                    if(this->hor.gweights[l].size() <= i) this->hor.gweights[l].resize(i+1);
                    if(this->hor.gweights[l][i].size() != embedding_dim) this->hor.gweights[l][i].resize(embedding_dim);
                    this->hor.gweights[l][i][j] = calculated_flat_gradients[i * embedding_dim + j];
                }
            }
            // Ver MLP
            CUDA_CHECK(cudaMemcpy(updated_flat_weights.data(), d_ver_weights[l], weights_bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(calculated_flat_gradients.data(), d_ver_gweights[l], weights_bytes, cudaMemcpyDeviceToHost));
            for (int i = 0; i < embedding_dim; ++i) {
                for (int j = 0; j < embedding_dim; ++j) {
                    this->ver.weights[l][i][j] = updated_flat_weights[i * embedding_dim + j];
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

        if (first) { // Always true in this overload
            CUDA_CHECK(cudaMemcpy(this->EH.data(), d_EH, embed_bytes, cudaMemcpyDeviceToHost));
        }
        // Note: EV is not updated in this kernel (kernelUpdateWeights_1stHead_H)
        CUDA_CHECK(cudaMemcpy(updated_MH_a.data(), d_MH_a, mh_mv_mq_mk_size * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(updated_MV_a.data(), d_MV_a, mh_mv_mq_mk_size * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(updated_MQ_a.data(), d_MQ_a, mh_mv_mq_mk_size * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(updated_MK_a.data(), d_MK_a, mh_mv_mq_mk_size * sizeof(float), cudaMemcpyDeviceToHost));

        // Unflatten
        unflatten(updated_MH_a, this->MH.a, mat_heights, embedding_dim);
        unflatten(updated_MV_a, this->MV.a, mat_heights, embedding_dim);
        unflatten(updated_MQ_a, this->MQ.a, mat_heights, embedding_dim);
        unflatten(updated_MK_a, this->MK.a, mat_heights, embedding_dim);

    }
    catch (const std::exception& e) {
        std::cerr << "CUDA Exception in cuBackward1stHead(expected): " << e.what() << std::endl;
        // Cleanup (Attention)
        cudaFree(d_expected_h); cudaFree(d_EH); cudaFree(d_EV);
        cudaFree(d_grad_EH); cudaFree(d_grad_EV_scaled);
        cudaFree(d_grad_dh); cudaFree(d_grad_dv);
        cudaFree(d_KdotQ); cudaFree(d_head); cudaFree(d_K); cudaFree(d_Q);
        cudaFree(d_K_embed); cudaFree(d_Q_embed); // Free if allocated
        cudaFree(d_pre_MH); cudaFree(d_pre_MV);
        cudaFree(d_MH_a); cudaFree(d_MV_a); cudaFree(d_MQ_a); cudaFree(d_MK_a);
        cudaFree(d_grad_MH); cudaFree(d_grad_MV); cudaFree(d_grad_head);
        cudaFree(d_lota_deriv); cudaFree(d_grad_KdotQ);
        cudaFree(d_grad_K); cudaFree(d_grad_Q);
        cudaFree(d_grad_MQ); cudaFree(d_grad_MK);
        // Cleanup (MLP Internals)
        for (int l = 0; l < layers; ++l) {
            cudaFree(d_hor_activations[l]); cudaFree(d_hor_weights[l]); cudaFree(d_hor_gweights[l]); cudaFree(d_hor_deltas[l]);
            cudaFree(d_ver_activations[l]); cudaFree(d_ver_weights[l]); cudaFree(d_ver_gweights[l]); cudaFree(d_ver_deltas[l]);
        }
        throw;
    }

    // --- Cleanup (Success Case) ---
    // (Attention)
    cudaFree(d_expected_h); cudaFree(d_EH); cudaFree(d_EV);
    cudaFree(d_grad_EH); cudaFree(d_grad_EV_scaled);
    cudaFree(d_grad_dh); cudaFree(d_grad_dv);
    cudaFree(d_KdotQ); cudaFree(d_head); cudaFree(d_K); cudaFree(d_Q);
    cudaFree(d_K_embed); cudaFree(d_Q_embed); // Free if allocated
    cudaFree(d_pre_MH); cudaFree(d_pre_MV); cudaFree(d_MH_a); cudaFree(d_MV_a); cudaFree(d_MQ_a); cudaFree(d_MK_a);
    cudaFree(d_grad_MH); cudaFree(d_grad_MV); cudaFree(d_grad_head);
    cudaFree(d_lota_deriv); cudaFree(d_grad_KdotQ); cudaFree(d_grad_K); cudaFree(d_grad_Q);
    cudaFree(d_grad_MQ); cudaFree(d_grad_MK);
    // (MLP Internals)
    for (int l = 0; l < layers; ++l) {
        cudaFree(d_hor_activations[l]); cudaFree(d_hor_weights[l]); cudaFree(d_hor_gweights[l]); cudaFree(d_hor_deltas[l]);
        cudaFree(d_ver_activations[l]); cudaFree(d_ver_weights[l]); cudaFree(d_ver_gweights[l]); cudaFree(d_ver_deltas[l]);
    }
}


/**
* @brief Backward Propagation (for first head) using gradients from expected Vertical output only.
*      Adjusts MQ, MV, and MK (correction). No update to EH/EV.
* @param expectedV vertical retention vector (host)
* @param in Input size (embedding dimension - used for MLP)
* @param layers Number of layers in the MLPs
*/
void attention::cuBackward1stHead(std::vector<std::vector<float>>& expectedV, int& in, int& layers)
{
    // --- Setup ---
    const int embedding_dim = EMBEDDING;
    const int mat_heights = MATHEIGHTS;
    const int context_win = CONTEXT_WIN;
    const float learning_rate = LEARNING;
    const float scaling_factor = SCALING;
    const int token_count = this->tokenCount;
    const int head_size = token_count * token_count;
    const int mh_mv_mq_mk_size = mat_heights * embedding_dim;
    const int k_q_size = token_count * mat_heights;
    // const int k_q_embed_size = token_count * embedding_dim; // For complex grad MQ/MK
    const int ev_size = context_win * embedding_dim;
    const size_t embed_bytes = embedding_dim * sizeof(float);
    const size_t weights_bytes = embedding_dim * embedding_dim * sizeof(float); // Assuming square MLP layers

    // Validation
    if (expectedV.size() != context_win || (!expectedV.empty() && expectedV[0].size() != embedding_dim)) {
        throw std::runtime_error("ExpectedV dimensions mismatch");
    }
    // Add other necessary validation checks...

    // --- Device Pointers (Attention) ---
    float *d_expected_v = nullptr, *d_EV = nullptr;
    float *d_grad_EV_full = nullptr, *d_grad_EV_summed = nullptr, *d_grad_EV_scaled = nullptr;
    // float *d_ver_output = nullptr; // Use MLP activations directly
    // float *d_grad_ver_input = nullptr; // Not needed
    // float *d_ver_gweights0 = nullptr; // Not needed
    float *d_grad_dv = nullptr; // Gradient w.r.t ver MLP input
    float *d_KdotQ = nullptr, *d_head = nullptr;
    float *d_K = nullptr, *d_Q = nullptr; // token x mat_heights
    float *d_K_embed = nullptr, *d_Q_embed = nullptr; // token x embedding_dim (Needed for Step 8?)
    float *d_pre_MV = nullptr;
    float *d_MV_a = nullptr, *d_MQ_a = nullptr, *d_MK_a = nullptr;
    float *d_grad_MV = nullptr;
    float *d_grad_head = nullptr;
    float *d_lota_deriv = nullptr;
    float *d_grad_KdotQ = nullptr;
    float *d_grad_Q = nullptr;
    float *d_grad_MQ = nullptr, *d_grad_MK_correction = nullptr; // Complex gradients

    // --- Device Pointers (ver MLP Internals) ---
    std::vector<float*> d_ver_activations(layers, nullptr);
    std::vector<float*> d_ver_weights(layers, nullptr);
    std::vector<float*> d_ver_gweights(layers, nullptr);
    std::vector<float*> d_ver_deltas(layers, nullptr);

    // Temporary flat vectors for H->D copy
    std::vector<std::vector<float>> flat_ver_weights(layers);

    try {
        // --- Allocate Memory (Attention) ---
        CUDA_CHECK(cudaMalloc(&d_expected_v, ev_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_EV, ev_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad_EV_full, ev_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad_EV_summed, embed_bytes));
        CUDA_CHECK(cudaMalloc(&d_grad_EV_scaled, embed_bytes));
        // CUDA_CHECK(cudaMalloc(&d_ver_output, embed_bytes)); // Not needed
        // CUDA_CHECK(cudaMalloc(&d_grad_ver_input, embed_bytes)); // Not needed
        // CUDA_CHECK(cudaMalloc(&d_ver_gweights0, weights_bytes)); // Not needed
        CUDA_CHECK(cudaMalloc(&d_grad_dv, embed_bytes));
        CUDA_CHECK(cudaMalloc(&d_KdotQ, head_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_head, head_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_K, k_q_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_Q, k_q_size * sizeof(float)));
        // CUDA_CHECK(cudaMalloc(&d_K_embed, k_q_embed_size * sizeof(float))); // If needed
        // CUDA_CHECK(cudaMalloc(&d_Q_embed, k_q_embed_size * sizeof(float))); // If needed
        CUDA_CHECK(cudaMalloc(&d_pre_MV, mat_heights * sizeof(float))); // Check size
        CUDA_CHECK(cudaMalloc(&d_MV_a, mh_mv_mq_mk_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_MQ_a, mh_mv_mq_mk_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_MK_a, mh_mv_mq_mk_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad_MV, mh_mv_mq_mk_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad_head, head_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_lota_deriv, head_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad_KdotQ, head_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad_Q, k_q_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad_MQ, mh_mv_mq_mk_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad_MK_correction, mh_mv_mq_mk_size * sizeof(float)));

        // --- Allocate Memory (ver MLP Internals) ---
        for (int l = 0; l < layers; ++l) {
            CUDA_CHECK(cudaMalloc(&d_ver_activations[l], embed_bytes));
            CUDA_CHECK(cudaMalloc(&d_ver_weights[l], weights_bytes));
            CUDA_CHECK(cudaMalloc(&d_ver_gweights[l], weights_bytes));
            CUDA_CHECK(cudaMalloc(&d_ver_deltas[l], embed_bytes));
        }

        // --- Data Transfer H->D (Attention) ---
        std::vector<float> flat_expectedV = flatten(expectedV);
        std::vector<float> flat_EV = flatten(this->EV);
        std::vector<float> flat_K = flatten(this->K);
        std::vector<float> flat_Q = flatten(this->Q);
        std::vector<float> flat_KdotQ = flatten(this->KdotQ);
        std::vector<float> flat_MV_a = flatten(this->MV.a);
        std::vector<float> flat_MQ_a = flatten(this->MQ.a);
        std::vector<float> flat_MK_a = flatten(this->MK.a);
        // Flatten and copy K_embed/Q_embed if they exist and are needed
        // std::vector<float> flat_K_embed = flatten(this->K_embed);
        // std::vector<float> flat_Q_embed = flatten(this->Q_embed);

        CUDA_CHECK(cudaMemcpy(d_expected_v, flat_expectedV.data(), ev_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_EV, flat_EV.data(), ev_size * sizeof(float), cudaMemcpyHostToDevice));
        // CUDA_CHECK(cudaMemcpy(d_ver_output, this->ver.output.data(), embed_bytes, cudaMemcpyHostToDevice)); // Not needed
        CUDA_CHECK(cudaMemcpy(d_KdotQ, flat_KdotQ.data(), head_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_K, flat_K.data(), k_q_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_Q, flat_Q.data(), k_q_size * sizeof(float), cudaMemcpyHostToDevice));
        // CUDA_CHECK(cudaMemcpy(d_K_embed, flat_K_embed.data(), k_q_embed_size * sizeof(float), cudaMemcpyHostToDevice)); // If needed
        // CUDA_CHECK(cudaMemcpy(d_Q_embed, flat_Q_embed.data(), k_q_embed_size * sizeof(float), cudaMemcpyHostToDevice)); // If needed
        CUDA_CHECK(cudaMemcpy(d_MV_a, flat_MV_a.data(), mh_mv_mq_mk_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_MQ_a, flat_MQ_a.data(), mh_mv_mq_mk_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_MK_a, flat_MK_a.data(), mh_mv_mq_mk_size * sizeof(float), cudaMemcpyHostToDevice));

        // --- Data Transfer H->D (ver MLP Internals) ---
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

        // --- Kernel Launch Config ---
        int threadsPerBlock1D = 256;
        int blocksPerGridEmbed = (embedding_dim + threadsPerBlock1D - 1) / threadsPerBlock1D;
        int blocksPerGridHead = (head_size + threadsPerBlock1D - 1) / threadsPerBlock1D;
        int blocksPerGridMatHeights = (mat_heights + threadsPerBlock1D - 1) / threadsPerBlock1D;
        int blocksPerGridMatrix = (mh_mv_mq_mk_size + threadsPerBlock1D - 1) / threadsPerBlock1D;
        // int blocksPerGridToken = (token_count + threadsPerBlock1D - 1) / threadsPerBlock1D; // If needed

        dim3 blockDim1D(threadsPerBlock1D);
        dim3 gridDimEmbed(blocksPerGridEmbed);
        dim3 gridDimHead(blocksPerGridHead);
        dim3 gridDimMatHeights(blocksPerGridMatHeights);
        dim3 gridDimMatrix(blocksPerGridMatrix);
        // dim3 gridDimToken(blocksPerGridToken); // If needed

        dim3 blockDim2D(16, 16);
        dim3 gridDimEmbed2D((embedding_dim + blockDim2D.x - 1) / blockDim2D.x, (embedding_dim + blockDim2D.y - 1) / blockDim2D.y); // For MLP weights
        dim3 gridDimHead2D((token_count + blockDim2D.x - 1) / blockDim2D.x, (token_count + blockDim2D.y - 1) / blockDim2D.y);
        dim3 gridDimMatrix2D((embedding_dim + blockDim2D.x - 1) / blockDim2D.x, (mat_heights + blockDim2D.y - 1) / blockDim2D.y);
        dim3 gridDimKQGrad2D((mat_heights + blockDim2D.x - 1) / blockDim2D.x, (token_count + blockDim2D.y - 1) / blockDim2D.y);

        // --- Backpropagation Steps ---

        // Step 1: Compute grad_EV (full, summed, scaled)
        kernelComputeGradientsEV_V<<<gridDimEmbed, blockDim1D>>>(
            d_EV, d_expected_v, d_grad_EV_full, d_grad_EV_summed, d_grad_EV_scaled,
            learning_rate, context_win, embedding_dim);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize()); // Sync before MLP backprop

        // --- Step 2: Backprop through ver MLP ---
        // Calculate Output Layer Deltas (Layer layers-1)
        kernelLastLayerDelta<<<gridDimEmbed, blockDim1D>>>(
            d_grad_EV_scaled, d_ver_activations[layers - 1], d_ver_deltas[layers - 1], embedding_dim);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Calculate Hidden Layer Deltas (Propagate backwards from layers-2 down to 0)
        for (int l = layers - 2; l >= 0; --l) {
            hiddenDeltaKernel<<<gridDimEmbed, blockDim1D>>>(
                d_ver_deltas[l + 1], d_ver_weights[l + 1], d_ver_activations[l], d_ver_deltas[l],
                embedding_dim, embedding_dim);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // Calculate Gradients and Update Weights (Iterate through layers 0 to layers-1)
        for (int l = 0; l < layers; ++l) {
            // See comment in previous overload regarding d_EV input size assumption
            float* d_prev_activations = (l == 0) ? d_EV : d_ver_activations[l - 1]; // CAUTION: Size/pointer check needed
            int prev_layer_size = (l == 0) ? embedding_dim : embedding_dim; // Adjust if needed
            dim3 currentGridDim2D = gridDimEmbed2D; // Adjust if needed

            updateWeightsKernel<<<currentGridDim2D, blockDim2D>>>(
                d_ver_deltas[l], d_prev_activations, d_ver_weights[l], d_ver_gweights[l],
                learning_rate, embedding_dim, prev_layer_size);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        // --- End of MLP Backprop ---

        // --- Step 3: Compute grad_dv ---
        kernelComputeGradMLPInput<<<gridDimEmbed, blockDim1D>>>(
            d_ver_deltas[0], d_ver_weights[0], d_grad_dv, embedding_dim, embedding_dim); // Assuming ver input size is embed_dim
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // --- Step 4: Compute grad_MV ---
        cuLOTA<<<gridDimHead2D, blockDim2D>>>(d_KdotQ, d_head, context_win, context_win, token_count, isSelfAttention);
        CUDA_CHECK(cudaGetLastError());
        kernelComputePreMV_V<<<gridDimMatHeights, blockDim1D>>>(d_head, d_Q, d_pre_MV, token_count, mat_heights);
        CUDA_CHECK(cudaGetLastError());
        kernelComputeGradMV_V<<<gridDimMatrix2D, blockDim2D>>>(d_pre_MV, d_grad_dv, d_grad_MV, mat_heights, embedding_dim);
        CUDA_CHECK(cudaGetLastError());

        // --- Step 5: Compute grad_head (dv part only) ---
        kernelComputeGradHead_V<<<gridDimHead2D, blockDim2D>>>(d_Q, d_MV_a, d_grad_dv, d_grad_head, token_count, mat_heights, embedding_dim);
        CUDA_CHECK(cudaGetLastError());

        // --- Step 6: Backprop through LOTA (Simple Derivative) ---
        // Assuming cuLOTAder takes KdotQ as input:
        cuLOTAder<<<gridDimHead2D, blockDim2D>>>(d_KdotQ, d_lota_deriv, context_win, context_win, token_count, isSelfAttention);
        CUDA_CHECK(cudaGetLastError());
        kernelComputeGradKdotQ_LOTA<<<gridDimHead, blockDim1D>>>(d_grad_head, d_lota_deriv, d_grad_KdotQ, scaling_factor, head_size);
        CUDA_CHECK(cudaGetLastError());

        // --- Step 7: Compute grad_Q ---
        kernelComputeGradQ_V<<<gridDimKQGrad2D, blockDim2D>>>(d_grad_KdotQ, d_K, d_grad_Q, token_count, mat_heights);
        CUDA_CHECK(cudaGetLastError());

        // --- Step 8: Compute grad_MQ and grad_MK_correction (Complex) ---
        // float* actual_d_K_embed = nullptr; // Replace if available
        float* actual_d_Q_embed = nullptr; // Replace if available
        // if (!actual_d_K_embed || !actual_d_Q_embed) {
        //     std::cerr << "Warning: Complex grad_MQ/MK calculation in cuBackward1stHead(expectedV,...) might be incorrect without K_embed/Q_embed." << std::endl;
        // }
        kernelComputeGradMQ_V<<<gridDimMatrix2D, blockDim2D>>>(d_grad_Q, actual_d_Q_embed, d_grad_MQ, token_count, mat_heights, embedding_dim);
        CUDA_CHECK(cudaGetLastError());
        kernelComputeGradMKCorrection<<<gridDimMatrix2D, blockDim2D>>>(d_grad_MQ, d_Q, d_K, d_grad_MK_correction, token_count, mat_heights, embedding_dim);
        CUDA_CHECK(cudaGetLastError());

        // --- Step 9: Update Weights MV, MQ, MK (correction) ---
        kernelUpdateWeights_1stHead_V<<<gridDimMatrix, blockDim1D>>>(
            d_MV_a, d_MQ_a, d_MK_a,
            d_grad_MV, d_grad_MQ, d_grad_MK_correction,
            learning_rate, mat_heights, embedding_dim
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize()); // Sync before D->H copy

        // Step 10: No EH/EV update in this path

        // --- Data Transfer D->H ---
        // Copy updated ver MLP weights and gradients back
        for (int l = 0; l < layers; ++l) {
            std::vector<float> updated_flat_weights(embedding_dim * embedding_dim);
            std::vector<float> calculated_flat_gradients(embedding_dim * embedding_dim);
            CUDA_CHECK(cudaMemcpy(updated_flat_weights.data(), d_ver_weights[l], weights_bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(calculated_flat_gradients.data(), d_ver_gweights[l], weights_bytes, cudaMemcpyDeviceToHost));
            for (int i = 0; i < embedding_dim; ++i) {
                for (int j = 0; j < embedding_dim; ++j) {
                    this->ver.weights[l][i][j] = updated_flat_weights[i * embedding_dim + j];
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
        // EV is not updated in kernelUpdateWeights_1stHead_V

        CUDA_CHECK(cudaMemcpy(updated_MV_a.data(), d_MV_a, mh_mv_mq_mk_size * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(updated_MQ_a.data(), d_MQ_a, mh_mv_mq_mk_size * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(updated_MK_a.data(), d_MK_a, mh_mv_mq_mk_size * sizeof(float), cudaMemcpyDeviceToHost));

        // Unflatten
        unflatten(updated_MV_a, this->MV.a, mat_heights, embedding_dim);
        unflatten(updated_MQ_a, this->MQ.a, mat_heights, embedding_dim);
        unflatten(updated_MK_a, this->MK.a, mat_heights, embedding_dim);
    }
    catch (const std::exception& e) {
        std::cerr << "CUDA Exception in cuBackward1stHead(expectedV): " << e.what() << std::endl;
        // Cleanup (Attention)
        cudaFree(d_expected_v); cudaFree(d_EV); cudaFree(d_grad_EV_full); cudaFree(d_grad_EV_summed); cudaFree(d_grad_EV_scaled);
        cudaFree(d_grad_dv);
        cudaFree(d_KdotQ); cudaFree(d_head); cudaFree(d_K); cudaFree(d_Q);
        cudaFree(d_K_embed); cudaFree(d_Q_embed); // Free if allocated
        cudaFree(d_pre_MV); cudaFree(d_MV_a); cudaFree(d_MQ_a); cudaFree(d_MK_a);
        cudaFree(d_grad_MV); cudaFree(d_grad_head);
        cudaFree(d_lota_deriv); cudaFree(d_grad_KdotQ); cudaFree(d_grad_Q);
        cudaFree(d_grad_MQ); cudaFree(d_grad_MK_correction);
        // Cleanup (ver MLP Internals)
        for (int l = 0; l < layers; ++l) {
            cudaFree(d_ver_activations[l]); cudaFree(d_ver_weights[l]); cudaFree(d_ver_gweights[l]); cudaFree(d_ver_deltas[l]);
        }
        throw;
    }

    // --- Cleanup (Success Case) ---
    // (Attention)
    cudaFree(d_expected_v); cudaFree(d_EV); cudaFree(d_grad_EV_full); cudaFree(d_grad_EV_summed); cudaFree(d_grad_EV_scaled);
    cudaFree(d_grad_dv);
    cudaFree(d_KdotQ); cudaFree(d_head); cudaFree(d_K); cudaFree(d_Q);
    cudaFree(d_K_embed); cudaFree(d_Q_embed); // Free if allocated
    cudaFree(d_pre_MV); cudaFree(d_MV_a); cudaFree(d_MQ_a); cudaFree(d_MK_a);
    cudaFree(d_grad_MV); cudaFree(d_grad_head);
    cudaFree(d_lota_deriv); cudaFree(d_grad_KdotQ); cudaFree(d_grad_Q);
    cudaFree(d_grad_MQ); cudaFree(d_grad_MK_correction);
    // (ver MLP Internals)
    for (int l = 0; l < layers; ++l) {
        cudaFree(d_ver_activations[l]); cudaFree(d_ver_weights[l]); cudaFree(d_ver_gweights[l]); cudaFree(d_ver_deltas[l]);
    }
}


/**
* @brief Backward Propagation (for first head) using gradients from both Horizontal and Vertical outputs.
*      Updates MH, MV, MQ, MK. No update to EH/EV.
* @param expectedH Horizontal embedding vector (next token prediction) (host)
* @param expectedV Vertical retention vector (host)
* @param in Input size (embedding dimension - used for MLP)
* @param layers Number of layers in the MLPs
*/
void attention::cuBackward1stHead(std::vector<float>& expectedH, std::vector<std::vector<float>>& expectedV, int& in, int& layers)
{
    // --- Setup ---
    const int embedding_dim = EMBEDDING;
    const int mat_heights = MATHEIGHTS;
    const int context_win = CONTEXT_WIN;
    const float learning_rate = LEARNING;
    const float scaling_factor = SCALING;
    const int token_count = this->tokenCount;
    const int head_size = token_count * token_count;
    const int mh_mv_mq_mk_size = mat_heights * embedding_dim;
    const int k_q_size = token_count * mat_heights;
    // const int k_q_embed_size = token_count * embedding_dim; // For simplified grad MK/MQ
    const int ev_size = context_win * embedding_dim;
    const size_t embed_bytes = embedding_dim * sizeof(float);
    const size_t weights_bytes = embedding_dim * embedding_dim * sizeof(float); // Assuming square MLP layers

    // Validation
    if (embedding_dim != in) throw std::runtime_error("Embedding dimension mismatch");
    if (expectedH.size() != embedding_dim) throw std::runtime_error("ExpectedH vector size mismatch");
    if (expectedV.size() != context_win || (!expectedV.empty() && expectedV[0].size() != embedding_dim)) {
        throw std::runtime_error("ExpectedV dimensions mismatch");
    }
    // Add other necessary validation checks...

    // --- Device Pointers (Attention) ---
    float *d_expected_h = nullptr, *d_expected_v = nullptr;
    float *d_EH = nullptr, *d_EV = nullptr;
    float *d_grad_EH = nullptr, *d_grad_EV_full = nullptr, *d_grad_EV_summed = nullptr, *d_grad_EV_scaled = nullptr;
    // float *d_hor_output = nullptr, *d_ver_output = nullptr; // Not needed
    // float *d_grad_hor_input = nullptr, *d_grad_ver_input = nullptr; // Not needed
    // float *d_hor_gweights0 = nullptr, *d_ver_gweights0 = nullptr; // Not needed
    float *d_grad_dh = nullptr, *d_grad_dv = nullptr; // Gradients w.r.t MLP inputs
    float *d_KdotQ = nullptr, *d_head = nullptr;
    float *d_K = nullptr, *d_Q = nullptr; // token x mat_heights
    float *d_K_embed = nullptr, *d_Q_embed = nullptr; // token x embedding_dim (Needed for Step 8?)
    float *d_pre_MH = nullptr, *d_pre_MV = nullptr;
    float *d_MH_a = nullptr, *d_MV_a = nullptr, *d_MQ_a = nullptr, *d_MK_a = nullptr;
    float *d_grad_MH = nullptr, *d_grad_MV = nullptr;
    float *d_grad_head = nullptr;
    float *d_lota_deriv = nullptr;
    float *d_grad_KdotQ = nullptr;
    float *d_grad_K = nullptr, *d_grad_Q = nullptr;
    float *d_grad_MQ = nullptr, *d_grad_MK = nullptr; // Simplified gradients

    // --- Device Pointers (MLP Internals) ---
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
        // --- Allocate Memory (Attention) ---
        CUDA_CHECK(cudaMalloc(&d_expected_h, embed_bytes));
        CUDA_CHECK(cudaMalloc(&d_expected_v, ev_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_EH, embed_bytes));
        CUDA_CHECK(cudaMalloc(&d_EV, ev_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad_EH, embed_bytes));
        CUDA_CHECK(cudaMalloc(&d_grad_EV_full, ev_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad_EV_summed, embed_bytes));
        CUDA_CHECK(cudaMalloc(&d_grad_EV_scaled, embed_bytes));
        // CUDA_CHECK(cudaMalloc(&d_hor_output, embed_bytes)); // Not needed
        // CUDA_CHECK(cudaMalloc(&d_ver_output, embed_bytes)); // Not needed
        // CUDA_CHECK(cudaMalloc(&d_grad_hor_input, embed_bytes)); // Not needed
        // CUDA_CHECK(cudaMalloc(&d_grad_ver_input, embed_bytes)); // Not needed
        // CUDA_CHECK(cudaMalloc(&d_hor_gweights0, weights_bytes)); // Not needed
        // CUDA_CHECK(cudaMalloc(&d_ver_gweights0, weights_bytes)); // Not needed
        CUDA_CHECK(cudaMalloc(&d_grad_dh, embed_bytes));
        CUDA_CHECK(cudaMalloc(&d_grad_dv, embed_bytes));
        CUDA_CHECK(cudaMalloc(&d_KdotQ, head_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_head, head_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_K, k_q_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_Q, k_q_size * sizeof(float)));
        // CUDA_CHECK(cudaMalloc(&d_K_embed, k_q_embed_size * sizeof(float))); // If needed
        // CUDA_CHECK(cudaMalloc(&d_Q_embed, k_q_embed_size * sizeof(float))); // If needed
        CUDA_CHECK(cudaMalloc(&d_pre_MH, mat_heights * sizeof(float))); // Check size
        CUDA_CHECK(cudaMalloc(&d_pre_MV, mat_heights * sizeof(float))); // Check size
        CUDA_CHECK(cudaMalloc(&d_MH_a, mh_mv_mq_mk_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_MV_a, mh_mv_mq_mk_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_MQ_a, mh_mv_mq_mk_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_MK_a, mh_mv_mq_mk_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad_MH, mh_mv_mq_mk_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad_MV, mh_mv_mq_mk_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad_head, head_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_lota_deriv, head_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad_KdotQ, head_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad_K, k_q_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad_Q, k_q_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad_MQ, mh_mv_mq_mk_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad_MK, mh_mv_mq_mk_size * sizeof(float)));

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
        std::vector<float> flat_expectedV = flatten(expectedV);
        std::vector<float> flat_EV = flatten(this->EV);
        std::vector<float> flat_K = flatten(this->K);
        std::vector<float> flat_Q = flatten(this->Q);
        std::vector<float> flat_KdotQ = flatten(this->KdotQ);
        std::vector<float> flat_MH_a = flatten(this->MH.a);
        std::vector<float> flat_MV_a = flatten(this->MV.a);
        std::vector<float> flat_MQ_a = flatten(this->MQ.a);
        std::vector<float> flat_MK_a = flatten(this->MK.a);
        // Flatten and copy K_embed/Q_embed if they exist and are needed
        // std::vector<float> flat_K_embed = flatten(this->K_embed);
        // std::vector<float> flat_Q_embed = flatten(this->Q_embed);

        CUDA_CHECK(cudaMemcpy(d_expected_h, expectedH.data(), embed_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_expected_v, flat_expectedV.data(), ev_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_EH, this->EH.data(), embed_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_EV, flat_EV.data(), ev_size * sizeof(float), cudaMemcpyHostToDevice));
        // CUDA_CHECK(cudaMemcpy(d_hor_output, this->hor.output.data(), embed_bytes, cudaMemcpyHostToDevice)); // Not needed
        // CUDA_CHECK(cudaMemcpy(d_ver_output, this->ver.output.data(), embed_bytes, cudaMemcpyHostToDevice)); // Not needed
        CUDA_CHECK(cudaMemcpy(d_KdotQ, flat_KdotQ.data(), head_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_K, flat_K.data(), k_q_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_Q, flat_Q.data(), k_q_size * sizeof(float), cudaMemcpyHostToDevice));
        // CUDA_CHECK(cudaMemcpy(d_K_embed, flat_K_embed.data(), k_q_embed_size * sizeof(float), cudaMemcpyHostToDevice)); // If needed
        // CUDA_CHECK(cudaMemcpy(d_Q_embed, flat_Q_embed.data(), k_q_embed_size * sizeof(float), cudaMemcpyHostToDevice)); // If needed
        CUDA_CHECK(cudaMemcpy(d_MH_a, flat_MH_a.data(), mh_mv_mq_mk_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_MV_a, flat_MV_a.data(), mh_mv_mq_mk_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_MQ_a, flat_MQ_a.data(), mh_mv_mq_mk_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_MK_a, flat_MK_a.data(), mh_mv_mq_mk_size * sizeof(float), cudaMemcpyHostToDevice));

        // --- Data Transfer H->D (MLP Internals) ---
        for (int l = 0; l < layers; ++l) {
            // Flatten weights
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
        }

        // --- Kernel Launch Config ---
        int threadsPerBlock1D = 256;
        int blocksPerGridEmbed = (embedding_dim + threadsPerBlock1D - 1) / threadsPerBlock1D;
        int blocksPerGridHead = (head_size + threadsPerBlock1D - 1) / threadsPerBlock1D;
        int blocksPerGridMatHeights = (mat_heights + threadsPerBlock1D - 1) / threadsPerBlock1D;
        int blocksPerGridMatrix = (mh_mv_mq_mk_size + threadsPerBlock1D - 1) / threadsPerBlock1D;
        // int blocksPerGridToken = (token_count + threadsPerBlock1D - 1) / threadsPerBlock1D; // If needed

        dim3 blockDim1D(threadsPerBlock1D);
        dim3 gridDimEmbed(blocksPerGridEmbed);
        dim3 gridDimHead(blocksPerGridHead);
        dim3 gridDimMatHeights(blocksPerGridMatHeights);
        dim3 gridDimMatrix(blocksPerGridMatrix);
        // dim3 gridDimToken(blocksPerGridToken); // If needed

        dim3 blockDim2D(16, 16);
        dim3 gridDimEmbed2D((embedding_dim + blockDim2D.x - 1) / blockDim2D.x, (embedding_dim + blockDim2D.y - 1) / blockDim2D.y); // For MLP weights
        dim3 gridDimHead2D((token_count + blockDim2D.x - 1) / blockDim2D.x, (token_count + blockDim2D.y - 1) / blockDim2D.y);
        dim3 gridDimMatrix2D((embedding_dim + blockDim2D.x - 1) / blockDim2D.x, (mat_heights + blockDim2D.y - 1) / blockDim2D.y);
        dim3 gridDimKQGrad2D((mat_heights + blockDim2D.x - 1) / blockDim2D.x, (token_count + blockDim2D.y - 1) / blockDim2D.y);

        // --- Backpropagation Steps ---

        // Step 1: Compute grad_EH and grad_EV (full, summed, scaled)
        // 1a: Compute grad_EH (using kernelComputeGradientsEH_EV, ignoring EV output)
        float* d_dummy_ev_grad = nullptr;
        CUDA_CHECK(cudaMalloc(&d_dummy_ev_grad, embed_bytes)); // Allocate dummy
        kernelComputeGradientsEH_EV<<<gridDimEmbed, blockDim1D>>>(d_EH, d_expected_h, d_grad_EH, d_dummy_ev_grad, embedding_dim);
        CUDA_CHECK(cudaGetLastError());
        cudaFree(d_dummy_ev_grad); // Free dummy

        // 1b: Compute grad_EV (full, summed, scaled)
        kernelComputeGradientsEV_V<<<gridDimEmbed, blockDim1D>>>(
            d_EV, d_expected_v, d_grad_EV_full, d_grad_EV_summed, d_grad_EV_scaled,
            learning_rate, context_win, embedding_dim);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize()); // Sync before MLP backprop

        // --- Step 2: Backprop through MLPs ---
        // --- 2a: Backprop through hor MLP ---
        kernelLastLayerDelta<<<gridDimEmbed, blockDim1D>>>(
            d_grad_EH, d_hor_activations[layers - 1], d_hor_deltas[layers - 1], embedding_dim);
        CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
        for (int l = layers - 2; l >= 0; --l) {
            hiddenDeltaKernel<<<gridDimEmbed, blockDim1D>>>(
                d_hor_deltas[l + 1], d_hor_weights[l + 1], d_hor_activations[l], d_hor_deltas[l],
                embedding_dim, embedding_dim);
            CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
        }
        for (int l = 0; l < layers; ++l) {
            float* d_prev_activations = (l == 0) ? d_EH : d_hor_activations[l - 1];
            updateWeightsKernel<<<gridDimEmbed2D, blockDim2D>>>(
                d_hor_deltas[l], d_prev_activations, d_hor_weights[l], d_hor_gweights[l],
                learning_rate, embedding_dim, embedding_dim);
            CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
        }

        // --- 2b: Backprop through ver MLP ---
        kernelLastLayerDelta<<<gridDimEmbed, blockDim1D>>>(
            d_grad_EV_scaled, d_ver_activations[layers - 1], d_ver_deltas[layers - 1], embedding_dim);
        CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
        for (int l = layers - 2; l >= 0; --l) {
            hiddenDeltaKernel<<<gridDimEmbed, blockDim1D>>>(
                d_ver_deltas[l + 1], d_ver_weights[l + 1], d_ver_activations[l], d_ver_deltas[l],
                embedding_dim, embedding_dim);
            CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
        }
        for (int l = 0; l < layers; ++l) {
            // See comment in first overload regarding d_EV input size assumption
            float* d_prev_activations = (l == 0) ? d_EV : d_ver_activations[l - 1]; // CAUTION: Size/pointer check needed
            int prev_layer_size = (l == 0) ? embedding_dim : embedding_dim; // Adjust if needed
            dim3 currentGridDim2D = gridDimEmbed2D; // Adjust if needed
            updateWeightsKernel<<<currentGridDim2D, blockDim2D>>>(
                d_ver_deltas[l], d_prev_activations, d_ver_weights[l], d_ver_gweights[l],
                learning_rate, embedding_dim, prev_layer_size);
            CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
        }
        // --- End of MLP Backprop ---

        // --- Step 3: Compute grad_dh and grad_dv ---
        kernelComputeGradMLPInput<<<gridDimEmbed, blockDim1D>>>(
            d_hor_deltas[0], d_hor_weights[0], d_grad_dh, embedding_dim, embedding_dim);
        CUDA_CHECK(cudaGetLastError());
        kernelComputeGradMLPInput<<<gridDimEmbed, blockDim1D>>>(
            d_ver_deltas[0], d_ver_weights[0], d_grad_dv, embedding_dim, embedding_dim); // Assuming ver input size is embed_dim
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // --- Step 4: Compute grad_MH and grad_MV ---
        cuLOTA<<<gridDimHead2D, blockDim2D>>>(d_KdotQ, d_head, context_win, context_win, token_count, isSelfAttention);
        CUDA_CHECK(cudaGetLastError());
        kernelComputePreMH_MV<<<gridDimMatHeights, blockDim1D>>>(d_head, d_K, d_Q, d_pre_MH, d_pre_MV, token_count, mat_heights);
        CUDA_CHECK(cudaGetLastError());
        kernelComputeGradMH_MV<<<gridDimMatrix2D, blockDim2D>>>(d_pre_MH, d_pre_MV, d_grad_dh, d_grad_dv, d_grad_MH, d_grad_MV, mat_heights, embedding_dim);
        CUDA_CHECK(cudaGetLastError());

        // --- Step 5: Compute grad_head ---
        kernelComputeGradHead<<<gridDimHead2D, blockDim2D>>>(d_K, d_Q, d_MH_a, d_MV_a, d_grad_dh, d_grad_dv, d_grad_head, token_count, mat_heights, embedding_dim);
        CUDA_CHECK(cudaGetLastError());

        // --- Step 6: Backprop through LOTA (Simple Derivative) ---
        // Assuming cuLOTAder takes KdotQ as input:
        cuLOTAder<<<gridDimHead2D, blockDim2D>>>(d_KdotQ, d_lota_deriv, context_win, context_win, token_count, isSelfAttention);
        CUDA_CHECK(cudaGetLastError());
        kernelComputeGradKdotQ_LOTA<<<gridDimHead, blockDim1D>>>(d_grad_head, d_lota_deriv, d_grad_KdotQ, scaling_factor, head_size);
        CUDA_CHECK(cudaGetLastError());

        // --- Step 7: Compute grad_K and grad_Q ---
        kernelComputeGradK_Q<<<gridDimKQGrad2D, blockDim2D>>>(d_grad_KdotQ, d_K, d_Q, d_grad_K, d_grad_Q, token_count, mat_heights);
        CUDA_CHECK(cudaGetLastError());

        // --- Step 8: Compute grad_MK and grad_MQ (Simplified) ---
        float* actual_d_K_embed = nullptr; // Replace if available
        float* actual_d_Q_embed = nullptr; // Replace if available
        // if (!actual_d_K_embed || !actual_d_Q_embed) {
        //     std::cerr << "Warning: Simplified grad_MK/MQ calculation in cuBackward1stHead(expectedH,expectedV,...) might be incorrect without K_embed/Q_embed." << std::endl;
        // }
        kernelComputeGradMK_MQ_Simplified<<<gridDimMatrix2D, blockDim2D>>>(
            d_grad_K, d_grad_Q, actual_d_K_embed, actual_d_Q_embed, d_grad_MK, d_grad_MQ,
            token_count, mat_heights, embedding_dim);
        CUDA_CHECK(cudaGetLastError());

        // --- Step 9: Update Weights MH, MV, MQ, MK ---
        kernelUpdateWeights_1stHead_HV<<<gridDimMatrix, blockDim1D>>>(
            d_MH_a, d_MV_a, d_MQ_a, d_MK_a,
            d_grad_MH, d_grad_MV, d_grad_MQ, d_grad_MK,
            learning_rate, mat_heights, embedding_dim);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize()); // Sync before D->H copy

        // Step 10: No EH/EV update in this path

        // --- Data Transfer D->H ---
        // Copy updated MLP weights and gradients back
        for (int l = 0; l < layers; ++l) {
            std::vector<float> updated_flat_weights(embedding_dim * embedding_dim);
            std::vector<float> calculated_flat_gradients(embedding_dim * embedding_dim);
            // Hor MLP
            CUDA_CHECK(cudaMemcpy(updated_flat_weights.data(), d_hor_weights[l], weights_bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(calculated_flat_gradients.data(), d_hor_gweights[l], weights_bytes, cudaMemcpyDeviceToHost));
            for (int i = 0; i < embedding_dim; ++i) {
                for (int j = 0; j < embedding_dim; ++j) {
                    this->hor.weights[l][i][j] = updated_flat_weights[i * embedding_dim + j];
                    if(this->hor.gweights.size() <= l) this->hor.gweights.resize(l+1);
                    if(this->hor.gweights[l].size() <= i) this->hor.gweights[l].resize(i+1);
                    if(this->hor.gweights[l][i].size() != embedding_dim) this->hor.gweights[l][i].resize(embedding_dim);
                    this->hor.gweights[l][i][j] = calculated_flat_gradients[i * embedding_dim + j];
                }
            }
            // Ver MLP
            CUDA_CHECK(cudaMemcpy(updated_flat_weights.data(), d_ver_weights[l], weights_bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(calculated_flat_gradients.data(), d_ver_gweights[l], weights_bytes, cudaMemcpyDeviceToHost));
            for (int i = 0; i < embedding_dim; ++i) {
                for (int j = 0; j < embedding_dim; ++j) {
                    this->ver.weights[l][i][j] = updated_flat_weights[i * embedding_dim + j];
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
        // EH/EV not updated in kernelUpdateWeights_1stHead_HV

        CUDA_CHECK(cudaMemcpy(updated_MH_a.data(), d_MH_a, mh_mv_mq_mk_size * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(updated_MV_a.data(), d_MV_a, mh_mv_mq_mk_size * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(updated_MQ_a.data(), d_MQ_a, mh_mv_mq_mk_size * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(updated_MK_a.data(), d_MK_a, mh_mv_mq_mk_size * sizeof(float), cudaMemcpyDeviceToHost));

        // Unflatten
        unflatten(updated_MH_a, this->MH.a, mat_heights, embedding_dim);
        unflatten(updated_MV_a, this->MV.a, mat_heights, embedding_dim);
        unflatten(updated_MQ_a, this->MQ.a, mat_heights, embedding_dim);
        unflatten(updated_MK_a, this->MK.a, mat_heights, embedding_dim);

    }
    catch (const std::exception& e) {
        std::cerr << "CUDA Exception in cuBackward1stHead(expectedH, expectedV): " << e.what() << std::endl;
        // Cleanup (Attention)
        cudaFree(d_expected_h); cudaFree(d_expected_v); cudaFree(d_EH); cudaFree(d_EV);
        cudaFree(d_grad_EH); cudaFree(d_grad_EV_full); cudaFree(d_grad_EV_summed); cudaFree(d_grad_EV_scaled);
        cudaFree(d_grad_dh); cudaFree(d_grad_dv);
        cudaFree(d_KdotQ); cudaFree(d_head); cudaFree(d_K); cudaFree(d_Q);
        cudaFree(d_K_embed); cudaFree(d_Q_embed); // Free if allocated
        cudaFree(d_pre_MH); cudaFree(d_pre_MV); cudaFree(d_MH_a); cudaFree(d_MV_a); cudaFree(d_MQ_a); cudaFree(d_MK_a);
        cudaFree(d_grad_MH); cudaFree(d_grad_MV);
        cudaFree(d_grad_head); cudaFree(d_lota_deriv);
        cudaFree(d_grad_KdotQ); cudaFree(d_grad_K); cudaFree(d_grad_Q);
        cudaFree(d_grad_MQ); cudaFree(d_grad_MK);
        // Cleanup (MLP Internals)
        for (int l = 0; l < layers; ++l) {
            cudaFree(d_hor_activations[l]); cudaFree(d_hor_weights[l]); cudaFree(d_hor_gweights[l]); cudaFree(d_hor_deltas[l]);
            cudaFree(d_ver_activations[l]); cudaFree(d_ver_weights[l]); cudaFree(d_ver_gweights[l]); cudaFree(d_ver_deltas[l]);
        }
        throw;
    }

    // --- Cleanup (Success Case) ---
    // (Attention)
    cudaFree(d_expected_h); cudaFree(d_expected_v); cudaFree(d_EH); cudaFree(d_EV);
    cudaFree(d_grad_EH); cudaFree(d_grad_EV_full); cudaFree(d_grad_EV_summed); cudaFree(d_grad_EV_scaled);
    cudaFree(d_grad_dh); cudaFree(d_grad_dv);
    cudaFree(d_KdotQ); cudaFree(d_head); cudaFree(d_K); cudaFree(d_Q);
    cudaFree(d_K_embed); cudaFree(d_Q_embed); // Free if allocated
    cudaFree(d_pre_MH); cudaFree(d_pre_MV); cudaFree(d_MH_a); cudaFree(d_MV_a); cudaFree(d_MQ_a); cudaFree(d_MK_a);
    cudaFree(d_grad_MH); cudaFree(d_grad_MV);
    cudaFree(d_grad_head); cudaFree(d_lota_deriv);
    cudaFree(d_grad_KdotQ); cudaFree(d_grad_K); cudaFree(d_grad_Q);
    cudaFree(d_grad_MQ); cudaFree(d_grad_MK);
    // (MLP Internals)
    for (int l = 0; l < layers; ++l) {
        cudaFree(d_hor_activations[l]); cudaFree(d_hor_weights[l]); cudaFree(d_hor_gweights[l]); cudaFree(d_hor_deltas[l]);
        cudaFree(d_ver_activations[l]); cudaFree(d_ver_weights[l]); cudaFree(d_ver_gweights[l]); cudaFree(d_ver_deltas[l]);
    }
}
