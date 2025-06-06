
#include <maths.hpp>
#include "include/attention.hpp"
#include <cmath>
#include <cfloat>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

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

    // Validate essential mat objects
    if (!this->K.mapped_data || !this->Q.mapped_data || !this->KdotQ.mapped_data ||
        !this->MH.mapped_data || !this->MV.mapped_data ||
        !this->MQ.mapped_data || !this->MK.mapped_data || !this->EV.mapped_data) {
        throw std::runtime_error("One or more attention mat members have null mapped_data.");
    }
    if (this->hor.weights.empty() || !this->hor.weights[0].mapped_data ||
        this->ver.weights.empty() || !this->ver.weights[0].mapped_data ||
        this->hor.gweights.empty() || !this->hor.gweights[0].mapped_data || // Assuming gweights are pre-initialized
        this->ver.gweights.empty() || !this->ver.gweights[0].mapped_data) { // Assuming gweights are pre-initialized
        throw std::runtime_error("One or more MLP weight/gradient mat members have null mapped_data or are empty.");
    }


    const size_t head_elements = static_cast<size_t>(this->KdotQ.row) * this->KdotQ.col; // Should be token_count * token_count
    const size_t proj_mat_elements = static_cast<size_t>(this->MH.row) * this->MH.col; // Should be mat_heights * embedding_dim
    const size_t k_q_elements = static_cast<size_t>(this->K.row) * this->K.col;       // Should be token_count * mat_heights

    const size_t embed_bytes = embedding_dim * sizeof(float);
    const size_t mlp_weights_elements = static_cast<size_t>(this->hor.weights[0].row) * this->hor.weights[0].col;
    const size_t mlp_weights_bytes = mlp_weights_elements * sizeof(float);
    const size_t ev_elements = static_cast<size_t>(this->EV.row) * this->EV.col; // Should be context_win * embedding_dim
    bool first = true; // This overload implies first=true for EH update

    // Validation
    if (embedding_dim != in) throw std::runtime_error("Embedding dimension mismatch");
    if (expected.size() != embedding_dim) throw std::runtime_error("Expected vector size mismatch");
    // Add other necessary validation checks...

    if (this->KdotQ.row != context_win || this->KdotQ.col != context_win) throw std::runtime_error("KdotQ dimensions mismatch token_count");
    if (this->MH.row != embedding_dim || this->MH.col != mat_heights) throw std::runtime_error("MH dimensions mismatch");
    if (this->MV.row != embedding_dim || this->MV.col != mat_heights) throw std::runtime_error("MV dimensions mismatch");
    if (this->MQ.row != mat_heights || this->MQ.col != embedding_dim) throw std::runtime_error("MQ dimensions mismatch");
    if (this->MK.row != mat_heights || this->MK.col != embedding_dim) throw std::runtime_error("MK dimensions mismatch");
    if (this->K.row != context_win || this->K.col != mat_heights) throw std::runtime_error("K dimensions mismatch");
    if (this->Q.row != context_win || this->Q.col != mat_heights) throw std::runtime_error("Q dimensions mismatch");
    if (this->EV.row != context_win || this->EV.col != embedding_dim) throw std::runtime_error("EV dimensions mismatch");
    if (this->hor.weights[0].row != embedding_dim || this->hor.weights[0].col != embedding_dim) throw std::runtime_error("MLP hor.weights[0] dimensions mismatch");

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
    std::vector<float*> d_hor_weights(layers-1, nullptr);
    std::vector<float*> d_hor_gweights(layers-1, nullptr);
    std::vector<float*> d_hor_deltas(layers, nullptr);
    std::vector<float*> d_ver_activations(layers, nullptr);
    std::vector<float*> d_ver_weights(layers-1, nullptr);
    std::vector<float*> d_ver_gweights(layers-1, nullptr);
    std::vector<float*> d_ver_deltas(layers, nullptr);

    try {
        // --- Allocate Memory (Attention) ---
        CUDA_CHECK(cudaMalloc(&d_expected_h, embed_bytes));
        CUDA_CHECK(cudaMalloc(&d_EH, embed_bytes));
        CUDA_CHECK(cudaMalloc(&d_EV, ev_elements * sizeof(float))); // Allocate EV
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
        CUDA_CHECK(cudaMalloc(&d_KdotQ, head_elements * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_head, head_elements * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_K, k_q_elements * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_Q, k_q_elements * sizeof(float)));
        // CUDA_CHECK(cudaMalloc(&d_K_embed, k_q_embed_size * sizeof(float))); // If needed
        // CUDA_CHECK(cudaMalloc(&d_Q_embed, k_q_embed_size * sizeof(float))); // If needed
        CUDA_CHECK(cudaMalloc(&d_pre_MH, mat_heights * sizeof(float))); // Check size
        CUDA_CHECK(cudaMalloc(&d_pre_MV, mat_heights * sizeof(float))); // Check size
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

        // --- Allocate Memory (MLP Internals) ---
        for (int l = 0; l < layers; ++l) {
            CUDA_CHECK(cudaMalloc(&d_hor_activations[l], embed_bytes));
            CUDA_CHECK(cudaMalloc(&d_hor_deltas[l], embed_bytes));
            CUDA_CHECK(cudaMalloc(&d_ver_activations[l], embed_bytes));
            CUDA_CHECK(cudaMalloc(&d_ver_deltas[l], embed_bytes));
        }
        // Allocate weights and gweights (size: layers-1)
        for (int l = 0; l < layers - 1; ++l) {
            CUDA_CHECK(cudaMalloc(&d_hor_weights[l], mlp_weights_bytes));
            CUDA_CHECK(cudaMalloc(&d_hor_gweights[l], mlp_weights_bytes));
            CUDA_CHECK(cudaMalloc(&d_ver_weights[l], mlp_weights_bytes));
            CUDA_CHECK(cudaMalloc(&d_ver_gweights[l], mlp_weights_bytes));
        }

        // --- Data Transfer H->D (Attention) ---
        CUDA_CHECK(cudaMemcpy(d_expected_h, expected.data(), embed_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_EH, this->EH.data(), embed_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_EV, this->EV.mapped_data, ev_elements * sizeof(float), cudaMemcpyHostToDevice)); // Copy EV
        // CUDA_CHECK(cudaMemcpy(d_hor_output, this->hor.output.data(), embed_bytes, cudaMemcpyHostToDevice)); // Not needed
        // CUDA_CHECK(cudaMemcpy(d_ver_output, this->ver.output.data(), embed_bytes, cudaMemcpyHostToDevice)); // Not needed
        CUDA_CHECK(cudaMemcpy(d_KdotQ, this->KdotQ.mapped_data, head_elements * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_K, this->K.mapped_data, k_q_elements * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_Q, this->Q.mapped_data, k_q_elements * sizeof(float), cudaMemcpyHostToDevice));
        // CUDA_CHECK(cudaMemcpy(d_K_embed, flat_K_embed.data(), k_q_embed_size * sizeof(float), cudaMemcpyHostToDevice)); // If needed
        // CUDA_CHECK(cudaMemcpy(d_Q_embed, flat_Q_embed.data(), k_q_embed_size * sizeof(float), cudaMemcpyHostToDevice)); // If needed
        CUDA_CHECK(cudaMemcpy(d_MH_a, this->MH.mapped_data, proj_mat_elements * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_MV_a, this->MV.mapped_data, proj_mat_elements * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_MQ_a, this->MQ.mapped_data, proj_mat_elements * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_MK_a, this->MK.mapped_data, proj_mat_elements * sizeof(float), cudaMemcpyHostToDevice));

        // --- Data Transfer H->D (MLP Internals) ---
        for (int l = 0; l < layers; ++l) {
            // Copy activations and weights
             if (this->hor.activations[l].size() != embedding_dim || this->ver.activations[l].size() != embedding_dim) {
                 throw std::runtime_error("MLP Activation vector dimension error during copy.");
             }
            CUDA_CHECK(cudaMemcpy(d_hor_activations[l], this->hor.activations[l].data(), embed_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_ver_activations[l], this->ver.activations[l].data(), embed_bytes, cudaMemcpyHostToDevice));
        }
        // Copy weights for all layers-1 weight matrices
        for (int l = 0; l < layers - 1; ++l) {
            CUDA_CHECK(cudaMemcpy(d_hor_weights[l], this->hor.weights[l].mapped_data, mlp_weights_bytes, cudaMemcpyHostToDevice));
            // Assuming gweights are initialized on host if they need to be copied, or they are computed on device
            // CUDA_CHECK(cudaMemcpy(d_hor_gweights[l], this->hor.gweights[l].mapped_data, mlp_weights_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_ver_weights[l], this->ver.weights[l].mapped_data, mlp_weights_bytes, cudaMemcpyHostToDevice));
            // CUDA_CHECK(cudaMemcpy(d_ver_gweights[l], this->ver.gweights[l].mapped_data, mlp_weights_bytes, cudaMemcpyHostToDevice));
        }

        // --- Kernel Launch Config ---
        int threadsPerBlock1D = 256;
        int blocksPerGridEmbed = (embedding_dim + threadsPerBlock1D - 1) / threadsPerBlock1D;
        int blocksPerGridHead = (static_cast<int>(head_elements) + threadsPerBlock1D - 1) / threadsPerBlock1D;
        int blocksPerGridMatHeights = (mat_heights + threadsPerBlock1D - 1) / threadsPerBlock1D;
        int blocksPerGridMatrix = (static_cast<int>(proj_mat_elements) + threadsPerBlock1D - 1) / threadsPerBlock1D;
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
                d_hor_deltas[l + 1], d_hor_weights[l], d_hor_activations[l], d_hor_deltas[l], // Use W[l] for delta[l]
                embedding_dim, embedding_dim);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // Calculate Gradients and Update Weights (Iterate through layers 0 to layers-1)
        // W[l] connects activations[l] to activations[l+1]. gW[l] = delta[l+1] * activations[l]^T
        // d_hor_activations[0] is the input to the MLP (e.g. d_EH)
        // The first weight matrix d_hor_weights[0] uses d_hor_activations[0] as input.
        // The last weight matrix d_hor_weights[layers-2] uses d_hor_activations[layers-2] as input.
        for (int l = 0; l < layers - 1; ++l) { // layers-1 weight matrices
            // float* d_input_activation_to_Wl = d_hor_activations[l]; // Standard: act[l] is input to W[l]
            // float* d_output_delta_of_Wl = d_hor_deltas[l+1];       // Standard: delta[l+1] is output delta for W[l]
            // The current CUDA kernel signature for updateWeightsKernel seems to be:
            // updateWeightsKernel(delta_of_output_layer_of_W, activation_of_input_layer_to_W, W, gW, ...)
            // If d_hor_activations[0] is actual MLP input (EH), and d_hor_activations[k] is output of k-th neuron layer
            // then W[l] (d_hor_weights[l]) takes d_hor_activations[l] as input and its output delta is d_hor_deltas[l+1]
            updateWeightsKernel<<<gridDimEmbed2D, blockDim2D>>>(
                d_hor_deltas[l+1], d_hor_activations[l], d_hor_weights[l], d_hor_gweights[l],
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
                d_ver_deltas[l + 1], d_ver_weights[l], d_ver_activations[l], d_ver_deltas[l], // Use W[l] for delta[l]
                embedding_dim, embedding_dim);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // Calculate Gradients and Update Weights (Iterate through layers 0 to layers-1)
        /*for (int l = 0; l < layers-1; ++l) {
            // Input to ver MLP layer 0 is EV. Need to handle its size (ev_size vs embed_dim).
            // Assuming this->ver.activations[0] (and thus d_ver_activations[0]) was the input to the ver MLP.
            // d_EV is available on device but d_ver_activations[0] should be the direct input processed by ver MLP.
            float* d_prev_activations = (l == 0) ? d_ver_activations[0] : d_ver_activations[l - 1];
            int prev_layer_size = (l == 0) ? embedding_dim : embedding_dim; // Adjust if EV is not embed_dim input
            dim3 currentGridDim2D = gridDimEmbed2D; // Adjust if prev_layer_size changes for l=0

            updateWeightsKernel<<<currentGridDim2D, blockDim2D>>>(
                d_ver_deltas[l], d_prev_activations, d_ver_weights[l], d_ver_gweights[l],
                learning_rate, embedding_dim, prev_layer_size);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        } */
        // d_ver_activations[0] is the input to the ver MLP.
        for (int l = 0; l < layers - 1; ++l) { // layers-1 weight matrices
            updateWeightsKernel<<<gridDimEmbed2D, blockDim2D>>>(
                d_ver_deltas[l+1], d_ver_activations[l], d_ver_weights[l], d_ver_gweights[l],
                learning_rate, embedding_dim, embedding_dim);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        // --- End of MLP Backprop ---

        // --- Step 3: Compute grad_dh and grad_dv ---
        // Use the computed deltas from the first MLP layer (d_hor_deltas[0])
        // and the weights of the first MLP layer (d_hor_weights[0])
        // to compute the gradient of the MLP's input (d_grad_dh).
        // grad_input = W[0]^T * delta[0] (if delta[0] is for output of W[0])
        // Or, grad_input = W[0]^T * delta[1] (if delta[0] is for input layer, delta[1] for first hidden)
        kernelComputeGradMLPInput<<<gridDimEmbed, blockDim1D>>>(
            d_hor_deltas[0], d_hor_weights[0], d_grad_dh, embedding_dim, embedding_dim); // Assumes d_hor_deltas[0] is for the input layer, d_hor_weights[0] is first weight matrix
        CUDA_CHECK(cudaGetLastError());
        // Similar for ver MLP
        kernelComputeGradMLPInput<<<gridDimEmbed, blockDim1D>>>(
            d_ver_deltas[0], d_ver_weights[0], d_grad_dv, embedding_dim, embedding_dim); // Assuming ver input size is embed_dim
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());


        // --- Step 4: Compute grad_MH and grad_MV ---
        // Assuming KdotQ and head are token_count x token_count.
        // The parameters context_win, context_win might be specific to the kernel's internal logic if it expects fixed-size blocks.
        cuLOTA<<<gridDimHead2D, blockDim2D>>>(d_KdotQ, d_head, token_count, token_count, token_count, isSelfAttention);
        CUDA_CHECK(cudaGetLastError());
        kernelComputePreMH_MV<<<gridDimMatHeights, blockDim1D>>>(d_head, d_K, d_Q, d_pre_MH, d_pre_MV, token_count, mat_heights);
        CUDA_CHECK(cudaGetLastError());
        kernelComputeGradMH_MV<<<gridDimMatrix2D, blockDim2D>>>(d_pre_MH, d_pre_MV, d_grad_dh, d_grad_dv, d_grad_MH, d_grad_MV, mat_heights, embedding_dim);
        CUDA_CHECK(cudaGetLastError());

        // --- Step 5: Compute grad_head ---
        kernelComputeGradHead<<<gridDimHead2D, blockDim2D>>>(d_K, d_Q, d_MH_a, d_MV_a, d_grad_dh, d_grad_dv, d_grad_head, token_count, mat_heights, embedding_dim);
        CUDA_CHECK(cudaGetLastError());

        // --- Step 6: Backprop through LOTA (Simple Derivative) ---
        cuLOTAder<<<gridDimHead2D, blockDim2D>>>(d_KdotQ, d_lota_deriv, token_count, token_count, token_count, isSelfAttention);
        // Assuming cuLOTAder takes KdotQ as input based on mlp::cuBackward modification:
        // cuLOTAder<<<gridDimHead2D, blockDim2D>>>(d_KdotQ, d_lota_deriv, context_win, context_win, token_count, isSelfAttention);
        CUDA_CHECK(cudaGetLastError());
        kernelComputeGradKdotQ_LOTA<<<gridDimHead, blockDim1D>>>(d_grad_head, d_lota_deriv, d_grad_KdotQ, scaling_factor, static_cast<int>(head_elements));
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
        for (int l = 0; l < layers-1; ++l) {
            // Hor MLP
            CUDA_CHECK(cudaMemcpy(this->hor.weights[l].mapped_data, d_hor_weights[l], mlp_weights_bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(this->hor.gweights[l].mapped_data, d_hor_gweights[l], mlp_weights_bytes, cudaMemcpyDeviceToHost));
            // Ver MLP
            CUDA_CHECK(cudaMemcpy(this->ver.weights[l].mapped_data, d_ver_weights[l], mlp_weights_bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(this->ver.gweights[l].mapped_data, d_ver_gweights[l], mlp_weights_bytes, cudaMemcpyDeviceToHost));
        }
        // Note: Activations and Deltas are intermediate and usually not copied back unless needed for debugging/specific logic.
        // If this->hor.activations or this->hor.deltas need to be updated on host, they would be copied here.
        // The current code only copies weights and gweights.

        // Copy updated Attention parameters back
        if (first) { // Always true in this overload
            CUDA_CHECK(cudaMemcpy(this->EH.data(), d_EH, embed_bytes, cudaMemcpyDeviceToHost));
        }
        // Note: EV is not updated in this kernel (kernelUpdateWeights_1stHead_H)
        CUDA_CHECK(cudaMemcpy(this->MH.mapped_data, d_MH_a, proj_mat_elements * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(this->MV.mapped_data, d_MV_a, proj_mat_elements * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(this->MQ.mapped_data, d_MQ_a, proj_mat_elements * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(this->MK.mapped_data, d_MK_a, proj_mat_elements * sizeof(float), cudaMemcpyDeviceToHost));

        // Ensure data is flushed to disk if necessary (OS typically handles this for mmap)
        // If explicit flush is needed for mat objects, call it here.
        // e.g., this->MH.flush(); this->MV.flush(); etc.

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
                for (int l = 0; l < layers; ++l) { // Activations and Deltas
            cudaFree(d_hor_activations[l]); cudaFree(d_hor_deltas[l]);
            cudaFree(d_ver_activations[l]); cudaFree(d_ver_deltas[l]);
        }
        for (int l = 0; l < layers - 1; ++l) { // Weights and GWeights
            cudaFree(d_hor_weights[l]); cudaFree(d_hor_gweights[l]);
            cudaFree(d_ver_weights[l]); cudaFree(d_ver_gweights[l]);
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

    // Validate essential mat objects
    if (!this->K.mapped_data || !this->Q.mapped_data || !this->KdotQ.mapped_data ||
        !this->MV.mapped_data || !this->MQ.mapped_data || !this->MK.mapped_data || !this->EV.mapped_data) {
        throw std::runtime_error("One or more attention mat members have null mapped_data.");
    }
     if (this->ver.weights.empty() || !this->ver.weights[0].mapped_data ||
        this->ver.gweights.empty() || !this->ver.gweights[0].mapped_data) { // Assuming gweights are pre-initialized
        throw std::runtime_error("MLP ver weight/gradient mat members have null mapped_data or are empty.");
    }

    const size_t head_elements = static_cast<size_t>(this->KdotQ.row) * this->KdotQ.col;
    const size_t proj_mat_elements = static_cast<size_t>(this->MV.row) * this->MV.col;
    const size_t k_q_elements = static_cast<size_t>(this->K.row) * this->K.col;
    const size_t ev_elements = static_cast<size_t>(this->EV.row) * this->EV.col;

    const size_t embed_bytes = embedding_dim * sizeof(float);
    const size_t mlp_weights_elements = static_cast<size_t>(this->ver.weights[0].row) * this->ver.weights[0].col;
    const size_t mlp_weights_bytes = mlp_weights_elements * sizeof(float);

    // Validation
    if (expectedV.size() != context_win || (!expectedV.empty() && expectedV[0].size() != embedding_dim)) {
        throw std::runtime_error("ExpectedV dimensions mismatch");
    }
    if (this->KdotQ.row != token_count || this->KdotQ.col != token_count) throw std::runtime_error("KdotQ dimensions mismatch token_count");
    if (this->MV.row != mat_heights || this->MV.col != embedding_dim) throw std::runtime_error("MV dimensions mismatch");
    if (this->MQ.row != mat_heights || this->MQ.col != embedding_dim) throw std::runtime_error("MQ dimensions mismatch");
    if (this->MK.row != mat_heights || this->MK.col != embedding_dim) throw std::runtime_error("MK dimensions mismatch");
    if (this->K.row != token_count || this->K.col != mat_heights) throw std::runtime_error("K dimensions mismatch");
    if (this->Q.row != token_count || this->Q.col != mat_heights) throw std::runtime_error("Q dimensions mismatch");
    if (this->EV.row != context_win || this->EV.col != embedding_dim) throw std::runtime_error("EV dimensions mismatch");
    if (this->ver.weights[0].row != embedding_dim || this->ver.weights[0].col != embedding_dim) throw std::runtime_error("MLP ver.weights[0] dimensions mismatch");
    
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
    std::vector<float*> d_ver_weights(layers - 1, nullptr); // MLP has layers-1 weight matrices
    std::vector<float*> d_ver_gweights(layers - 1, nullptr); // MLP has layers-1 gweight matrices
    std::vector<float*> d_ver_deltas(layers, nullptr);

    try {
        // --- Allocate Memory (Attention) ---
        CUDA_CHECK(cudaMalloc(&d_expected_v, ev_elements * sizeof(float))); // expectedV is flattened later
        CUDA_CHECK(cudaMalloc(&d_EV, ev_elements * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad_EV_full, ev_elements * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad_EV_summed, embed_bytes));
        CUDA_CHECK(cudaMalloc(&d_grad_EV_scaled, embed_bytes));
        // CUDA_CHECK(cudaMalloc(&d_ver_output, embed_bytes)); // Not needed
        // CUDA_CHECK(cudaMalloc(&d_grad_ver_input, embed_bytes)); // Not needed
        // CUDA_CHECK(cudaMalloc(&d_ver_gweights0, weights_bytes)); // Not needed
        CUDA_CHECK(cudaMalloc(&d_grad_dv, embed_bytes));
        CUDA_CHECK(cudaMalloc(&d_KdotQ, head_elements * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_head, head_elements * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_K, k_q_elements * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_Q, k_q_elements * sizeof(float)));
        // CUDA_CHECK(cudaMalloc(&d_K_embed, k_q_embed_size * sizeof(float))); // If needed
        // CUDA_CHECK(cudaMalloc(&d_Q_embed, k_q_embed_size * sizeof(float))); // If needed
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

        // --- Allocate Memory (ver MLP Internals) ---
        // Allocate activations and deltas (size: layers)
        for (int l = 0; l < layers; ++l) {
            CUDA_CHECK(cudaMalloc(&d_ver_activations[l], embed_bytes));
            CUDA_CHECK(cudaMalloc(&d_ver_deltas[l], embed_bytes));
        }
        // Allocate weights and gweights (size: layers-1)
        for (int l = 0; l < layers - 1; ++l) {
            CUDA_CHECK(cudaMalloc(&d_ver_weights[l], mlp_weights_bytes));
            CUDA_CHECK(cudaMalloc(&d_ver_gweights[l], mlp_weights_bytes));
        }

        // --- Data Transfer H->D (Attention) ---
        std::vector<float> flat_expectedV = flatten(expectedV);

        CUDA_CHECK(cudaMemcpy(d_expected_v, flat_expectedV.data(), ev_elements * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_EV, this->EV.mapped_data, ev_elements * sizeof(float), cudaMemcpyHostToDevice));
        // CUDA_CHECK(cudaMemcpy(d_ver_output, this->ver.output.data(), embed_bytes, cudaMemcpyHostToDevice)); // Not needed
        CUDA_CHECK(cudaMemcpy(d_KdotQ, this->KdotQ.mapped_data, head_elements * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_K, this->K.mapped_data, k_q_elements * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_Q, this->Q.mapped_data, k_q_elements * sizeof(float), cudaMemcpyHostToDevice));
        // CUDA_CHECK(cudaMemcpy(d_K_embed, flat_K_embed.data(), k_q_embed_size * sizeof(float), cudaMemcpyHostToDevice)); // If needed
        // CUDA_CHECK(cudaMemcpy(d_Q_embed, flat_Q_embed.data(), k_q_embed_size * sizeof(float), cudaMemcpyHostToDevice)); // If needed
        CUDA_CHECK(cudaMemcpy(d_MV_a, this->MV.mapped_data, proj_mat_elements * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_MQ_a, this->MQ.mapped_data, proj_mat_elements * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_MK_a, this->MK.mapped_data, proj_mat_elements * sizeof(float), cudaMemcpyHostToDevice));

        // --- Data Transfer H->D (ver MLP Internals) ---
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
            // CUDA_CHECK(cudaMemcpy(d_ver_gweights[l], this->ver.gweights[l].mapped_data, mlp_weights_bytes, cudaMemcpyHostToDevice)); // If gweights are initialized on host
        }

        // --- Kernel Launch Config ---
        int threadsPerBlock1D = 256;
        int blocksPerGridEmbed = (embedding_dim + threadsPerBlock1D - 1) / threadsPerBlock1D;
        int blocksPerGridHead = (static_cast<int>(head_elements) + threadsPerBlock1D - 1) / threadsPerBlock1D;
        int blocksPerGridMatHeights = (mat_heights + threadsPerBlock1D - 1) / threadsPerBlock1D;
        int blocksPerGridMatrix = (static_cast<int>(proj_mat_elements) + threadsPerBlock1D - 1) / threadsPerBlock1D;
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
                d_ver_deltas[l + 1], d_ver_weights[l], d_ver_activations[l], d_ver_deltas[l], // Use W[l] for delta[l]
                embedding_dim, embedding_dim);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // Calculate Gradients and Update Weights (Iterate through layers 0 to layers-1)
        // d_ver_activations[0] is the input to the ver MLP.
        // W[l] connects activations[l] to activations[l+1]. gW[l] = delta[l+1] * activations[l]^T
        for (int l = 0; l < layers - 1; ++l) { // layers-1 weight matrices
            updateWeightsKernel<<<gridDimEmbed2D, blockDim2D>>>(
                d_ver_deltas[l+1], d_ver_activations[l], d_ver_weights[l], d_ver_gweights[l],
                learning_rate, embedding_dim, embedding_dim);
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
        cuLOTA<<<gridDimHead2D, blockDim2D>>>(d_KdotQ, d_head, token_count, token_count, token_count, isSelfAttention);
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
        cuLOTAder<<<gridDimHead2D, blockDim2D>>>(d_KdotQ, d_lota_deriv, token_count, token_count, token_count, isSelfAttention);
        CUDA_CHECK(cudaGetLastError());
        kernelComputeGradKdotQ_LOTA<<<gridDimHead, blockDim1D>>>(d_grad_head, d_lota_deriv, d_grad_KdotQ, scaling_factor, static_cast<int>(head_elements));
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
        for (int l = 0; l < layers - 1; ++l) { // layers-1 weight matrices
            CUDA_CHECK(cudaMemcpy(this->ver.weights[l].mapped_data, d_ver_weights[l], mlp_weights_bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(this->ver.gweights[l].mapped_data, d_ver_gweights[l], mlp_weights_bytes, cudaMemcpyDeviceToHost));
        }
        // Note: Activations and Deltas are intermediate and usually not copied back.
        // If this->ver.activations or this->ver.deltas need to be updated on host, they would be copied here.

        // Copy updated Attention parameters back
        // EV is not updated in kernelUpdateWeights_1stHead_V
        CUDA_CHECK(cudaMemcpy(this->MV.mapped_data, d_MV_a, proj_mat_elements * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(this->MQ.mapped_data, d_MQ_a, proj_mat_elements * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(this->MK.mapped_data, d_MK_a, proj_mat_elements * sizeof(float), cudaMemcpyDeviceToHost));

        // Optional: Flush mat objects if explicit flush is part of their API
        // this->MV.flush(); this->MQ.flush(); this->MK.flush();
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
        for (int l = 0; l < layers; ++l) { // Activations and Deltas
            cudaFree(d_ver_activations[l]); cudaFree(d_ver_deltas[l]);
        }
        for (int l = 0; l < layers - 1; ++l) { // Weights and GWeights
            cudaFree(d_ver_weights[l]); cudaFree(d_ver_gweights[l]);
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
    for (int l = 0; l < layers; ++l) { // Activations and Deltas
        cudaFree(d_ver_activations[l]); cudaFree(d_ver_deltas[l]);
    }
    for (int l = 0; l < layers - 1; ++l) { // Weights and GWeights
        cudaFree(d_ver_weights[l]); cudaFree(d_ver_gweights[l]);
    }
}
