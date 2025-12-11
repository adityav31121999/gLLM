#ifdef USE_CU
#include "include/transformer.hpp"
#include "include/block.hpp"
#include "include/attention.hpp" // Include necessary headers
#include "include/mlp.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <string> // For std::to_string

// Helper macro for CUDA error checking
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        throw std::runtime_error(cudaGetErrorString(err)); \
    } \
} while (0)


///////////////////////// static training /////////////////////////

/**
 * @brief OpenCL backward propagation from a specific block 'k' down to the first block (0).
 *        Uses a single common expected horizontal error vector for block 'k-1'.
 * @param expectedH Expected horizontal embedding for block 'k-1's output.
 * @param k The block number (1-based index) to start backpropagation from.
 * @param clip_norm Maximum L2 norm for gradient clipping (new parameter).
 */
void transformer::cuBackward(std::vector<float>& expectedH, int& k) {
    if (k <= 0 || k > m) {
        throw std::out_of_range("cuBackward(vector<float>, k): Block index k=" + std::to_string(k) + " is out of range [1, " + std::to_string(m) + "].");
    }

    int start_block_index = k - 1; // 0-based index
    // std::cout << "-> cuBackward (H, start_block = " << k << ")" << std::endl;

    try {
        // --- Starting Block (k-1) ---
        // Receives the external horizontal error signal 'expectedH'.
        // If k=1, this is the first block.
        if (start_block_index == 0) { // Starting from the first block (k=1)
            blocks[0].tokenCount = currentTokenCount;
            blocks[0].cubackward1stBlock(expectedH, d, l, learning, lambda_L1, lambda_L2);
        }
        else { // Handles all k > 1
            blocks[start_block_index].tokenCount = currentTokenCount % CONTEXT_WIN;
            blocks[start_block_index].cubackward(expectedH, start_block_index, d, l, learning, lambda_L1, lambda_L2);
        }
    } 
    catch (const std::exception& e) {
         throw std::runtime_error("Exception during transformer::cuBackward(vector<float>, k=" + std::to_string(k) + "): " + std::string(e.what()));
    }
}


/**
 * @brief OpenCL backward propagation from a specific block 'k' down to the first block (0).
 *        Uses distinct expected horizontal error vectors (one per column/parallel) for block 'k-1'.
 * @param expectedH Vector of expected horizontal embeddings for block 'k-1's output (shape [y][EMBEDDING]).
 * @param k The block number (1-based index) to start backpropagation from.
 * @param clip_norm Maximum L2 norm for gradient clipping (new parameter).
 */
void transformer::cuBackward(std::vector<std::vector<float>>& expectedH, int& k) {
    if (k <= 0 || k > m) {
        throw std::out_of_range("cuBackward(vector<vector<float>>, k): Block index k=" + std::to_string(k) + " is out of range [1, " + std::to_string(m) + "].");
    }
    if (expectedH.size() != static_cast<size_t>(x)) { // Changed `y` to `x`
        throw std::runtime_error("cuBackward(vector<vector<float>>, k): Outer dimension of expectedH (" + std::to_string(expectedH.size()) + ") does not match number of heads x (" + std::to_string(x) + ").");
    }
    if (!expectedH.empty() && expectedH[0].size() != EMBEDDING) {
        throw std::runtime_error("cuBackward(vector<vector<float>>, k): Inner dimension of expectedH (" + std::to_string(expectedH[0].size()) + ") does not match EMBEDDING (" + std::to_string(EMBEDDING) + ").");
    }

    int start_block_index = k - 1; // 0-based index
    // std::cout << "-> cuBackward (H_2D, start_block = " << k << ")" << std::endl;

    try {
        // --- Starting Block (k-1) ---
        // Receives the external horizontal error signal 'expectedH'.
        // If k=1, this is the first block.
        if (start_block_index == 0) { // Starting from the first block (k=1)
            blocks[0].tokenCount = currentTokenCount;
            blocks[0].cubackward1stBlock(expectedH, d, l, learning, lambda_L1, lambda_L2);
        }
        else {
            blocks[start_block_index].tokenCount = currentTokenCount % CONTEXT_WIN;
            blocks[start_block_index].cubackward(expectedH, start_block_index, d, l, learning, lambda_L1, lambda_L2);
        }
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Exception during transformer::cuBackward(vector<vector<float>>, k=" + std::to_string(k) + "): " + std::string(e.what()));
    }
}

///////////////////////// contextualised training /////////////////////////

/**
 * @brief OpenCL backward propagation from a specific block 'k' down to the first block (0).
 *        Uses distinct expected horizontal error vectors (one per column/parallel) for block 'k-1'.
 * @param expectedH Vector of expected horizontal embeddings for block 'k-1's output (shape [y][EMBEDDING]).
 * @param k The block number (1-based index) to start backpropagation from.
 * @param clip_norm Maximum L2 norm for gradient clipping (new parameter).
 */
void transformer::cuBackwardContext(std::vector<std::vector<float>>& expectedH, int& k) {
    if (k <= 0 || k > m) {
        throw std::out_of_range("cuBackwardContext(vector<vector<float>>, k): Block index k=" + std::to_string(k) + " is out of range [1, " + std::to_string(m) + "].");
    }
    if (expectedH.size() != static_cast<size_t>(x)) { // Changed `y` to `x`
        throw std::runtime_error("cuBackwardContext(vector<vector<float>>, k): Outer dimension of expectedH (" + std::to_string(expectedH.size()) + ") does not match number of heads x (" + std::to_string(x) + ").");
    }
    if (!expectedH.empty() && expectedH[0].size() != EMBEDDING) {
        throw std::runtime_error("cuBackwardContext(vector<vector<float>>, k): Inner dimension of expectedH (" + std::to_string(expectedH[0].size()) + ") does not match EMBEDDING (" + std::to_string(EMBEDDING) + ").");
    }

    int start_block_index = k - 1; // 0-based index
    // std::cout << "-> cuBackwardContext (H_2D, start_block = " << k << ")" << std::endl;

    try {
        // --- Starting Block (k-1) ---
        // Receives the external horizontal error signal 'expectedH'.
        // If k=1, this is the first block.
        if (start_block_index == 0) { // Starting from the first block (k=1)
            blocks[0].tokenCount = currentTokenCount;
            blocks[0].curbackward1stBlock(expectedH, d, l, learning, lambda_L1, lambda_L2);
        }
        else {
            blocks[start_block_index].tokenCount = currentTokenCount % CONTEXT_WIN;
            blocks[start_block_index].cubackward(expectedH, start_block_index, d, l, learning, lambda_L1, lambda_L2);
        }
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Exception during transformer::cuBackwardContext(vector<vector<float>>, k=" + std::to_string(k) + "): " + std::string(e.what()));
    }
}

//////////////////// contextualised de-embedding from backprop gradient ////////////////////

/**
 * @brief CUDA implementation to update the deEmbeddings matrix and calculate gradients for the EH vector.
 *        This function performs:
 *        1. LOTA activation on raw logits to get probabilities.
 *        2. Categorical Cross-Entropy backward pass for dL/du.
 *        3. Computes gradients for deEmbeddings (dL/dW_deEmbeddings).
 *        4. Updates deEmbeddings using Elastic Net regularization.
 *        5. Propagates error back to the input of deEmbeddings (dL/dEH).
 * @param deEmbeddings deEmbeddings matrix ((d*x) * vocabsize).
 * @param otok Final hidden state (EH), concatenated, from the last layer of the block (size = d*x).
 * @param prediction Raw logits for each token in the vocabulary (size: vocabsize).
 * @param oneHotEncode Host-side one-hot vector of true label (size: vocabsize).
 * @param indexForToken Index of the token for training.
 * @param learning Learning rate.
 * @param lambda_L1 L1 regularization parameter.
 * @param lambda_L2 L2 regularization parameter.
 * @param gradForEh Host-side vector to store gradients for the EH vector (dL/dEH).
 */
void transformer::cuUpdateDeEmbeddings(mat& deEmbeddings, std::vector<float> prediction, std::vector<float> oneHotEncode,
                                     float learning, float lambda_L1, float lambda_L2, std::vector<float>& gradForEh)
{
    int vocab_size = vocabsize;
    int p_dim = d * x;

    // --- 0. Validate sizes ---
    if (prediction.size() != vocab_size || oneHotEncode.size() != vocab_size) {
        throw std::runtime_error("cuUpdateDeEmbeddings: Input vector size mismatch.");
    }
    if (otok.size() != static_cast<size_t>(p_dim)) {
        throw std::runtime_error("cuUpdateDeEmbeddings: 'otok' size mismatch.");
    }
    if (gradForEh.size() != static_cast<size_t>(p_dim)) {
        gradForEh.resize(p_dim);
    }

    // --- CUDA device pointers ---
    float *d_deEmbed, *d_gdeEmbed, *d_prediction_raw_logits, *d_final_hidden_state_input;
    float *d_oneHotEncode, *d_predNorm, *d_delta, *d_gradForEh_device;

    size_t deEmbed_size_bytes = vocab_size * p_dim * sizeof(float);
    size_t vocab_size_bytes = vocab_size * sizeof(float);
    size_t p_dim_bytes = p_dim * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_deEmbed, deEmbed_size_bytes));
    CUDA_CHECK(cudaMalloc(&d_gdeEmbed, deEmbed_size_bytes));
    CUDA_CHECK(cudaMalloc(&d_prediction_raw_logits, vocab_size_bytes));
    CUDA_CHECK(cudaMalloc(&d_final_hidden_state_input, p_dim_bytes));
    CUDA_CHECK(cudaMalloc(&d_oneHotEncode, vocab_size_bytes));
    CUDA_CHECK(cudaMalloc(&d_predNorm, vocab_size_bytes));
    CUDA_CHECK(cudaMalloc(&d_delta, vocab_size_bytes));
    CUDA_CHECK(cudaMalloc(&d_gradForEh_device, p_dim_bytes));

    // --- Copy data from host to device ---
    CUDA_CHECK(cudaMemcpy(d_deEmbed, deEmbeddings.mapped_data, deEmbed_size_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_prediction_raw_logits, prediction.data(), vocab_size_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_final_hidden_state_input, otok.data(), p_dim_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_oneHotEncode, oneHotEncode.data(), vocab_size_bytes, cudaMemcpyHostToDevice));

    // --- Zero out gradient buffer ---
    CUDA_CHECK(cudaMemset(d_gdeEmbed, 0, deEmbed_size_bytes));

    // --- Kernel launch parameters ---
    int threads_per_block_1d = 256;
    int blocks_per_grid_1d = (vocab_size + threads_per_block_1d - 1) / threads_per_block_1d;

    // --- 1. LOTA activation ---
    cuLOTA<<<blocks_per_grid_1d, threads_per_block_1d>>>((float*)d_prediction_raw_logits, d_predNorm, vocab_size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- 2. Categorical Cross-Entropy backward pass (dL/dz) ---
    // This kernel computes: prediction - label
    kernelComputeGradpred<<<blocks_per_grid_1d, threads_per_block_1d>>>(d_predNorm, d_oneHotEncode, d_delta, vocab_size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- 3. Compute gradients for deEmbeddings (outer product) ---
    // This is an outer product: grad[i,j] = delta[i] * otok[j].
    dim3 threads_2d(8, 8);
    dim3 blocks_2d((p_dim / 4 + threads_2d.x - 1) / threads_2d.x, (vocab_size + threads_2d.y - 1) / threads_2d.y);
    KernelComputeGradDeEmbeddings<<<blocks_2d, threads_2d>>>
        (d_delta, (const float4*)d_final_hidden_state_input, (float4*)d_gdeEmbed, vocab_size, p_dim / 4);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- 4. Update deEmbeddings weights ---
    // Update weights with gradients and elastic net regularization
    int totalElements = vocab_size * p_dim;
    int blocks_update = (totalElements / 4 + threads_per_block_1d - 1) / threads_per_block_1d;
    kernelUpdateWeightsGeneral_f4<<<blocks_update, threads_per_block_1d>>>(d_deEmbed, d_gdeEmbed, learning, lambda_L1, lambda_L2, totalElements / 4);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- 5. Propagate error back to hidden state (dL/dEH) ---
    // gradForEh = delta * deEmbeddings
    int blocks_propagate = (p_dim / 4 + threads_per_block_1d - 1) / threads_per_block_1d;
    kernelGradForAttentionOutput<<<blocks_propagate, threads_per_block_1d>>>(d_deEmbed, d_delta, d_gradForEh_device, vocab_size, p_dim);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- Transfer results back to host ---
    CUDA_CHECK(cudaMemcpy(deEmbeddings.mapped_data, d_deEmbed, deEmbed_size_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gradForEh.data(), d_gradForEh_device, p_dim_bytes, cudaMemcpyDeviceToHost));

    // --- Free device memory ---
    cudaFree(d_deEmbed);
    cudaFree(d_gdeEmbed);
    cudaFree(d_prediction_raw_logits);
    cudaFree(d_final_hidden_state_input);
    cudaFree(d_oneHotEncode);
    cudaFree(d_predNorm);
    cudaFree(d_delta);
    cudaFree(d_gradForEh_device);
}

/**
 * @brief CUDA implementation to update embedding using propagated gradients from training.
 */
void transformer::cuUpdateEmbeddings(mat tokenEmbedding, std::vector<float>& gradForEh, float learning,
    float lambda_L1, float lambda_L2, int rows)
{
    int dim = d;
    size_t rowBytes = static_cast<size_t>(rows) * dim * sizeof(float);
    size_t gradBytes = gradForEh.size() * sizeof(float);

    // --- Device pointers ---
    float *d_embed, *d_gEmbed;

    // --- Allocate and copy ---
    CUDA_CHECK(cudaMalloc(&d_embed, rowBytes));
    CUDA_CHECK(cudaMalloc(&d_gEmbed, gradBytes));
    CUDA_CHECK(cudaMemcpy(d_embed, tokenEmbedding.mapped_data, rowBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gEmbed, gradForEh.data(), gradBytes, cudaMemcpyHostToDevice));

    // --- Kernel launch ---
    int threads_per_block = 256;
    int blocks_per_grid = (dim + threads_per_block - 1) / threads_per_block;
    updateEmbeddings<<<blocks_per_grid, threads_per_block>>>(d_embed, d_gEmbed, learning, lambda_L1, lambda_L2, rows, dim);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // --- Free memory ---
    cudaFree(d_embed);
    cudaFree(d_gEmbed);
}

#endif