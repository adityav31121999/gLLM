
#include "include/block.hpp"
#include "include/attention.hpp" // For attention class definition
#include "include/mlp.hpp"       // For mlp class definition (assuming block has an mlp member)
// #include "include/layernorm.hpp" // For LayerNorm kernels/class (assuming block has norm members)

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <stdexcept>
#include <numeric>   // For std::iota
#include <algorithm> // For std::min, std::max
#include <iostream>  // For cerr, cout
#include <string>    // For std::to_string in error messages

// Helper macro for CUDA error checking
// Ensure this is defined, possibly in a common header.
#ifndef CUDA_CHECK
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error in %s at line %d (%s): %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(err)); \
        throw std::runtime_error(std::string("CUDA Error: ") + cudaGetErrorString(err)); \
    } \
} while (0)
#endif

// --- Placeholder Kernels (These would need actual implementations) ---

// Kernel to aggregate EH vectors from multiple heads (e.g., by summing them)
// d_all_head_eh_outputs: flat array of [head0_eh, head1_eh, ...]
// d_aggregated_output: single vector of size d_embedding
__global__ void aggregate_eh_vectors_sum(const float* d_all_head_eh_outputs, float* d_aggregated_output, int num_total_heads, int d_embedding) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < d_embedding) {
        float sum = 0.0f;
        for (int head_idx = 0; head_idx < num_total_heads; ++head_idx) {
            sum += d_all_head_eh_outputs[head_idx * d_embedding + idx];
        }
        d_aggregated_output[idx] = sum;
    }
}

// Kernel for broadcasting a vector and adding it to each row of a matrix
// matrix: tokenCount x d_embedding
// vector_to_add: 1 x d_embedding
// output: tokenCount x d_embedding
__global__ void add_broadcast_vector_to_matrix_rows(const float* matrix, const float* vector_to_add, float* output, int tokenCount, int d_embedding) {
    int token_idx = blockIdx.y * blockDim.y + threadIdx.y; // Row
    int embed_idx = blockIdx.x * blockDim.x + threadIdx.x; // Column

    if (token_idx < tokenCount && embed_idx < d_embedding) {
        output[token_idx * d_embedding + embed_idx] = matrix[token_idx * d_embedding + embed_idx] + vector_to_add[embed_idx];
    }
}

// Kernel for element-wise addition of two matrices (or sequences of vectors)
__global__ void elementwise_add_sequences(const float* A, const float* B, float* C, int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        C[idx] = A[idx] + B[idx];
    }
}

// Placeholder for Layer Normalization forward pass kernel
// Assumes d_input and d_output can be the same for in-place
__global__ void layer_norm_forward_kernel(const float* d_input, float* d_output, /* LayerNorm weights/biases, */ int tokenCount, int d_embedding, float eps) {
    // Simplified: actual LayerNorm is more complex (mean, variance per token)
    // This is just a pass-through for structure.
     int token_idx = blockIdx.y * blockDim.y + threadIdx.y;
     int embed_idx = blockIdx.x * blockDim.x + threadIdx.x;
     if (token_idx < tokenCount && embed_idx < d_embedding) {
        if (d_input != d_output) {
            d_output[token_idx * d_embedding + embed_idx] = d_input[token_idx * d_embedding + embed_idx];
        }
     }
}


void block::cuInferParallel(const mat& tokens, int& in, int& tokenCount, int& layers, int& parallelNumber)
{
    const int d_embedding = in; // Assuming 'embedding_dim' is a member of block
    // const int ffn_mlp_layers = layers; // 'layers' parameter for the block's main FFN

    if (in != d_embedding) {
        throw std::runtime_error("Input embedding dimension 'in' (" + std::to_string(in) +
                                 ") does not match block's embedding_dim (" + std::to_string(d_embedding) + ").");
    }
    if (tokenCount <= 0) {
        std::cout << "block::cuInferParallel (first overload) called with tokenCount <= 0. Skipping." << std::endl;
        // Ensure block_output_buffer is appropriately sized and zeroed if necessary by caller
        return;
    }
    if (this->x <= 0 || this->y <= 0) {
        throw std::runtime_error("Block has invalid attention head configuration (x or y is zero).");
    }
    if (tokens.col != d_embedding || tokens.row != tokenCount) {
        throw std::runtime_error("Input token dimensions mismatch. Expected " + std::to_string(tokenCount) +
                                 "x" + std::to_string(d_embedding) + ", got " + std::to_string(tokens.row) +
                                 "x" + std::to_string(tokens.col) + ".");
    }
    // Assume this->mlp, this->norm1, this->norm2, this->block_output_buffer are initialized

    // --- Device Memory ---
    float* d_input_tokens = nullptr;
    float* d_all_head_eh_outputs = nullptr; // Flat buffer for [head0_eh, head1_eh, ...]
    float* d_aggregated_attention_output = nullptr;
    float* d_after_addnorm1 = nullptr;
    float* d_ffn_input = nullptr; // Could be d_after_addnorm1 if norm is in-place
    float* d_ffn_output = nullptr;
    float* d_final_block_output_device = nullptr;

    std::vector<cudaStream_t> streams;

    try {
        size_t tokens_bytes = static_cast<size_t>(tokenCount) * d_embedding * sizeof(float);
        size_t single_eh_bytes = static_cast<size_t>(d_embedding) * sizeof(float);
        int num_total_heads = this->x * this->y;
        size_t all_eh_bytes = static_cast<size_t>(num_total_heads) * d_embedding * sizeof(float);

        CUDA_CHECK(cudaMalloc(&d_input_tokens, tokens_bytes));
        CUDA_CHECK(cudaMemcpy(d_input_tokens, tokens.mapped_data, tokens_bytes, cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc(&d_all_head_eh_outputs, all_eh_bytes));
        CUDA_CHECK(cudaMalloc(&d_aggregated_attention_output, single_eh_bytes));
        CUDA_CHECK(cudaMalloc(&d_after_addnorm1, tokens_bytes));
        d_ffn_input = d_after_addnorm1; // Assuming norm1 is in-place or its output is d_after_addnorm1
        CUDA_CHECK(cudaMalloc(&d_ffn_output, tokens_bytes));
        CUDA_CHECK(cudaMalloc(&d_final_block_output_device, tokens_bytes));

        // --- Parallel Attention Head Inference ---
        int num_streams_to_use = std::min(std::max(1, parallelNumber), num_total_heads);
        streams.resize(num_streams_to_use);
        for (int i = 0; i < num_streams_to_use; ++i) {
            CUDA_CHECK(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
        }

        for (int head_global_idx = 0; head_global_idx < num_total_heads; ++head_global_idx) {
            int layer_h_idx = head_global_idx / this->y; // Head's layer index
            int col_p_idx = head_global_idx % this->y;   // Head's column index

            attention& current_attn_head = this->b[layer_h_idx][col_p_idx];
            cudaStream_t current_stream = streams[head_global_idx % num_streams_to_use];
            float* d_current_head_eh_output_ptr = d_all_head_eh_outputs + head_global_idx * d_embedding;

            // `cuInferHead` is synchronous and updates current_attn_head.EH on host.
            // For true stream parallelism, cuInferHead would need to be async and take device ptrs.
            // Here, we call it, then async copy its host result to the designated device spot.
            int attn_mlp_layers = current_attn_head.hor.weights.empty() ? 0 : current_attn_head.hor.weights.size(); // Example
            current_attn_head.cuInferHead(tokens, in, layers, tokenCount);

            CUDA_CHECK(cudaMemcpyAsync(d_current_head_eh_output_ptr,
                                       current_attn_head.EH.data(), // EH is std::vector<float>
                                       single_eh_bytes,
                                       cudaMemcpyHostToDevice,
                                       current_stream));
        }
        for (int i = 0; i < num_streams_to_use; ++i) 
            CUDA_CHECK(cudaStreamSynchronize(streams[i]));

        // --- Aggregate Attention Outputs ---
        // Example: Summing all EH vectors. A projection might be needed in a full Transformer.
        int threadsPerBlockAgg = 256;
        int blocksAgg = (d_embedding + threadsPerBlockAgg - 1) / threadsPerBlockAgg;
        aggregate_eh_vectors_sum<<<blocksAgg, threadsPerBlockAgg, 0, streams[0]>>>(
            d_all_head_eh_outputs, d_aggregated_attention_output, num_total_heads, d_embedding);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(streams[0]));

        // --- First Add & Norm ---
        // Add d_aggregated_attention_output (broadcast) to d_input_tokens, then LayerNorm.
        dim3 blockDimAddNorm(16, 16); // For 2D data
        dim3 gridDimAddNorm((d_embedding + blockDimAddNorm.x -1) / blockDimAddNorm.x, (tokenCount + blockDimAddNorm.y - 1) / blockDimAddNorm.y);
        add_broadcast_vector_to_matrix_rows<<<gridDimAddNorm, blockDimAddNorm, 0, streams[0]>>>(
            d_input_tokens, d_aggregated_attention_output, d_after_addnorm1, tokenCount, d_embedding);
        CUDA_CHECK(cudaGetLastError());
        // this->norm1.cuForward(d_after_addnorm1, d_after_addnorm1, tokenCount, d_embedding, streams[0]); // Placeholder
        layer_norm_forward_kernel<<<gridDimAddNorm, blockDimAddNorm, 0, streams[0]>>>(d_after_addnorm1, d_after_addnorm1, tokenCount, d_embedding, 1e-5f);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(streams[0]));

        // --- Feed-Forward Network (FFN) ---
        // this->mlp.cuInfer(d_ffn_input, d_ffn_output, tokenCount, d_embedding, ffn_mlp_layers, streams[0]); // Placeholder
        // For now, as a placeholder, let's just copy input to output for FFN part
        CUDA_CHECK(cudaMemcpyAsync(d_ffn_output, d_ffn_input, tokens_bytes, cudaMemcpyDeviceToDevice, streams[0]));
        CUDA_CHECK(cudaStreamSynchronize(streams[0]));

        // --- Second Add & Norm ---
        // Add d_ffn_output to d_ffn_input (which is d_after_addnorm1, the residual), then LayerNorm.
        int total_elements_final = tokenCount * d_embedding;
        int blocksFinal = (total_elements_final + threadsPerBlockAgg - 1) / threadsPerBlockAgg;
        elementwise_add_sequences<<<blocksFinal, threadsPerBlockAgg, 0, streams[0]>>>(
            d_ffn_input, d_ffn_output, d_final_block_output_device, total_elements_final);
        CUDA_CHECK(cudaGetLastError());
        // this->norm2.cuForward(d_final_block_output_device, d_final_block_output_device, tokenCount, d_embedding, streams[0]); // Placeholder
        layer_norm_forward_kernel<<<gridDimAddNorm, blockDimAddNorm, 0, streams[0]>>>(d_final_block_output_device, d_final_block_output_device, tokenCount, d_embedding, 1e-5f);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(streams[0]));
    }
    catch (const std::exception& e) {
        std::cerr << "Exception in block::cuInferParallel (first overload): " << e.what() << std::endl;
        // Fall-through to cleanup
    }

    // Cleanup
    for (cudaStream_t stream : streams) if(stream) CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_input_tokens));
    CUDA_CHECK(cudaFree(d_all_head_eh_outputs));
    CUDA_CHECK(cudaFree(d_aggregated_attention_output));
    CUDA_CHECK(cudaFree(d_after_addnorm1)); // d_ffn_input might be an alias
    CUDA_CHECK(cudaFree(d_ffn_output));
    CUDA_CHECK(cudaFree(d_final_block_output_device));
}


void block::cuInferParallel(const std::vector<mat>& expectedV, const mat& tokForBlock, int& in, int& tokenCount, int& blockCount, int& layers, 
        int& n, int& parallelNumber)
{
    // This overload is for subsequent blocks in a sequence, potentially using cross-attention
    // or state (`expectedV`) passed from the previous block's attention heads.
    // `expectedV` would be a vector of `mat` objects, one for each attention head in the previous block state.
    // `tokForBlock` are the query tokens for the current block.
    // `tokenCount` is likely the total tokens processed so far in the sequence.
    // `blockCount` is the index of the current block.
    // `n` is the context window size for attention processing.

    const int d_embedding = EMBEDDING; // Block's embedding dimension
    // const int ffn_mlp_layers = layers; // 'layers' parameter for the block's main FFN
    const int num_total_heads = this->x * this->y;
    const int current_block_token_count = tokForBlock.row; // Number of tokens specific to this block

    // --- Validations ---
    if (in != d_embedding) {
        throw std::runtime_error("Input embedding dimension 'in' (" + std::to_string(in) +
                                 ") does not match block's embedding_dim (" + std::to_string(d_embedding) + ") in second overload.");
    }
    if (blockCount <= 0) { // blockCount is 1-indexed for subsequent blocks in attention::cuInferHead
        throw std::runtime_error("blockCount must be > 0 for the second overload of block::cuInferParallel.");
    }
    if (current_block_token_count <= 0) {
        std::cout << "block::cuInferParallel (second overload) called with tokForBlock.row <= 0. Skipping." << std::endl;
        return;
    }
    if (this->x <= 0 || this->y <= 0) {
        throw std::runtime_error("Block has invalid attention head configuration (x or y is zero) in second overload.");
    }
    if (tokForBlock.col != d_embedding) {
        throw std::runtime_error("tokForBlock dimensions mismatch. Expected cols: " + std::to_string(d_embedding) +
                                 ", got " + std::to_string(tokForBlock.col) + ".");
    }
    if (static_cast<int>(expectedV.size()) != num_total_heads) {
        throw std::runtime_error("expectedV size (" + std::to_string(expectedV.size()) +
                                 ") does not match number of heads (" + std::to_string(num_total_heads) + ").");
    }
    for (size_t i = 0; i < expectedV.size(); ++i) {
        if (expectedV[i].row != tokenCount || expectedV[i].col != d_embedding) { // tokenCount is totalTokenCount
            throw std::runtime_error("Dimension mismatch for expectedV[" + std::to_string(i) +
                                     "]. Expected " + std::to_string(tokenCount) + "x" + std::to_string(d_embedding) +
                                     ", got " + std::to_string(expectedV[i].row) + "x" + std::to_string(expectedV[i].col));
        }
    }

    // --- Device Memory ---
    float* d_input_tokens_current_block = nullptr; // For tokForBlock
    float* d_all_head_eh_outputs = nullptr;
    float* d_aggregated_attention_output = nullptr;
    float* d_after_addnorm1 = nullptr;
    float* d_ffn_input = nullptr;
    float* d_ffn_output = nullptr;
    float* d_final_block_output_device = nullptr;

    std::vector<cudaStream_t> streams;

    try {
        size_t current_block_tokens_bytes = static_cast<size_t>(current_block_token_count) * d_embedding * sizeof(float);
        size_t single_eh_bytes = static_cast<size_t>(d_embedding) * sizeof(float);
        size_t all_eh_bytes = static_cast<size_t>(num_total_heads) * d_embedding * sizeof(float);

        CUDA_CHECK(cudaMalloc(&d_input_tokens_current_block, current_block_tokens_bytes));
        CUDA_CHECK(cudaMemcpy(d_input_tokens_current_block, tokForBlock.mapped_data, current_block_tokens_bytes, cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc(&d_all_head_eh_outputs, all_eh_bytes));
        CUDA_CHECK(cudaMalloc(&d_aggregated_attention_output, single_eh_bytes));
        CUDA_CHECK(cudaMalloc(&d_after_addnorm1, current_block_tokens_bytes));
        d_ffn_input = d_after_addnorm1;
        CUDA_CHECK(cudaMalloc(&d_ffn_output, current_block_tokens_bytes));
        CUDA_CHECK(cudaMalloc(&d_final_block_output_device, current_block_tokens_bytes));

        // --- Parallel Attention Head Inference ---
        int num_streams_to_use = std::min(std::max(1, parallelNumber), num_total_heads);
        streams.resize(num_streams_to_use);
        for (int i = 0; i < num_streams_to_use; ++i) {
            CUDA_CHECK(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
        }

        for (int head_global_idx = 0; head_global_idx < num_total_heads; ++head_global_idx) {
            int layer_h_idx = head_global_idx / this->y;
            int col_p_idx = head_global_idx % this->y;

            attention& current_attn_head = this->b[layer_h_idx][col_p_idx];
            const mat& evp_for_this_head = expectedV[head_global_idx];
            cudaStream_t current_stream = streams[head_global_idx % num_streams_to_use];
            float* d_current_head_eh_output_ptr = d_all_head_eh_outputs + head_global_idx * d_embedding;

            int attn_mlp_layers = current_attn_head.hor.weights.empty() ? 0 : current_attn_head.hor.weights.size() - 1;
            
            // Call the second overload of attention::cuInferHead
            // Parameters: const mat& EVp_mat, const mat& tokForBlock, int& d_embedding, int& layers_mlp, 
            //             int& totalTokenCount, int& blockIdx, int& contextWindowSize_n
            current_attn_head.cuInferHead(evp_for_this_head, tokForBlock, in, layers, tokenCount, blockCount, n);

            CUDA_CHECK(cudaMemcpyAsync(d_current_head_eh_output_ptr,
                                       current_attn_head.EH.data(),
                                       single_eh_bytes,
                                       cudaMemcpyHostToDevice,
                                       current_stream));
        }

        for (int i = 0; i < num_streams_to_use; ++i) 
            CUDA_CHECK(cudaStreamSynchronize(streams[i]));

        // --- Aggregate Attention Outputs ---
        int threadsPerBlockAgg = 256;
        int blocksAgg = (d_embedding + threadsPerBlockAgg - 1) / threadsPerBlockAgg;
        aggregate_eh_vectors_sum<<<blocksAgg, threadsPerBlockAgg, 0, streams[0]>>>(
                d_all_head_eh_outputs, d_aggregated_attention_output, num_total_heads, d_embedding);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(streams[0]));

        // --- First Add & Norm ---
        dim3 blockDimAddNorm(16, 16);
        dim3 gridDimAddNorm((d_embedding + blockDimAddNorm.x -1) / blockDimAddNorm.x, (current_block_token_count + blockDimAddNorm.y - 1) / blockDimAddNorm.y);
        add_broadcast_vector_to_matrix_rows<<<gridDimAddNorm, blockDimAddNorm, 0, streams[0]>>>(
            d_input_tokens_current_block, d_aggregated_attention_output, d_after_addnorm1, current_block_token_count, d_embedding);
        CUDA_CHECK(cudaGetLastError());
        layer_norm_forward_kernel<<<gridDimAddNorm, blockDimAddNorm, 0, streams[0]>>>(d_after_addnorm1, d_after_addnorm1, current_block_token_count, d_embedding, 1e-5f);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(streams[0]));

        // --- Feed-Forward Network (FFN) ---
        // this->mlp.cuInfer(d_ffn_input, d_ffn_output, current_block_token_count, d_embedding, ffn_mlp_layers, streams[0]); // Placeholder
        CUDA_CHECK(cudaMemcpyAsync(d_ffn_output, d_ffn_input, current_block_tokens_bytes, cudaMemcpyDeviceToDevice, streams[0]));
        CUDA_CHECK(cudaStreamSynchronize(streams[0]));

        // --- Second Add & Norm ---
        int total_elements_final = current_block_token_count * d_embedding;
        int blocksFinal = (total_elements_final + threadsPerBlockAgg - 1) / threadsPerBlockAgg;
        elementwise_add_sequences<<<blocksFinal, threadsPerBlockAgg, 0, streams[0]>>>(
            d_ffn_input, d_ffn_output, d_final_block_output_device, total_elements_final);
        CUDA_CHECK(cudaGetLastError());
        layer_norm_forward_kernel<<<gridDimAddNorm, blockDimAddNorm, 0, streams[0]>>>(d_final_block_output_device, d_final_block_output_device, current_block_token_count, d_embedding, 1e-5f);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(streams[0]));

    } 
    catch (const std::exception& e) {
        std::cerr << "Exception in block::cuInferParallel (second overload): " << e.what() << std::endl;
    }

    // Cleanup
    for (cudaStream_t stream : streams) if(stream) CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_input_tokens_current_block));
    CUDA_CHECK(cudaFree(d_all_head_eh_outputs));
    CUDA_CHECK(cudaFree(d_aggregated_attention_output));
    CUDA_CHECK(cudaFree(d_after_addnorm1));
    CUDA_CHECK(cudaFree(d_ffn_output));
    CUDA_CHECK(cudaFree(d_final_block_output_device));
}



void block::cuInfer(const mat& tokens, int& in, int& tokenCount, int& layers) 
{
    for(int j = 0; j < y-1; j++) {
        // parallel execution
        cuInferParallel(tokens, in, tokenCount, layers, j);
        if(j == y-1) break;
        for(int i = 0; i < x-1; i++) {
            b[i][j+1].EH = b[i][j].EH;
        }
    }
}


void block::cuInfer(const std::vector<std::vector<mat>>& expectedV, const mat& tokForBlock, int& in, 
    int& tokenCount, int& blockCount, int& layers, int& n, int& parallelNumber) 
{
    for(int j = 0; j < y-1; j++) {
        std::vector<mat> eV(x, mat(CONTEXT_WIN, EMBEDDING));
        for(int i = 0; i < x-1; i++) {
            eV[i] = expectedV[i][j];
        }
        // parallel execution
        cuInferParallel(eV, tokForBlock, in, tokenCount, blockCount, layers, n, j);
        if(j == y-1) break;
        for(int i = 0; i < x-1; i++) {
            b[i][j+1].EH = b[i][j].EH;
        }
    }
}
