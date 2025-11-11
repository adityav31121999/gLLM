#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "include/attention.hpp"
#include <vector>
#include <iostream>
#include <stdexcept>
#include <numeric>
#include <cmath>
#include <maths.hpp>
#include <string>

float* attention::getDeviceEVPointer() {
    // Return the device pointer for the EV matrix
    return d_EV;
}

/**------------------------------------MULTIPLICATION------------------------------------**/

/**
 * @brief CUDA kernel for element-wise vector multiplication. Multiplies `target_and_output` by `factor` in place.
 * @param[in,out] target_and_output Device pointer to the vector to be multiplied (and store the result).
 * @param[in] factor Device pointer to the vector acting as the multiplier.
 * @param[in] size The number of elements in the vectors.
 */
__global__ void kernelElementwiseMultiply(float* target_and_output, const float* factor, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        target_and_output[idx] *= factor[idx];
    }
}

/**
 * @brief CUDA device function to compute the predicted token index by finding the highest dot product
 *        between a vector (EH) and rows of an embedding matrix.
 * @param[in] EH Device pointer to the horizontal retention vector (size dim).
 * @param[in] embeddings Device pointer to the token embeddings matrix (row-major: voc x dim).
 * @param[in] dim The embedding dimension (size of EH and columns of embeddings).
 * @param[in] voc The vocabulary size (number of rows in embeddings).
 * @return The index of the token embedding with the highest dot product. Returns -1 if voc <= 0 or embeddings is null.
 */
__device__ int computePrediction(const float* EH, const float* embeddings, int dim, int voc) 
{
    if (voc <= 0 || embeddings == nullptr) {
        return -1; // Handle invalid input
    }
    float max_dot_product = -FLT_MAX;
    int predicted_index = 0;

    // Assuming dim is a multiple of 4 for float4 operations
    int dim_float4 = dim / 4;
    const float4* EH_f4 = (const float4*)EH;

    for (int i = 0; i < voc; ++i) {
        const float4* current_embedding_row_f4 = (const float4*)(embeddings + i * dim);

        float current_dot_product = 0.0f;
        for (int k = 0; k < dim_float4; ++k) {
            current_dot_product += EH_f4[k].x * current_embedding_row_f4[k].x + EH_f4[k].y * current_embedding_row_f4[k].y + EH_f4[k].z * current_embedding_row_f4[k].z + EH_f4[k].w * current_embedding_row_f4[k].w;
        }

        if (current_dot_product > max_dot_product) {
            max_dot_product = current_dot_product;
            predicted_index = i;
        }
    }
    return predicted_index;
}

/**
 * @brief CUDA device function to compute the predicted token index by finding the highest dot product
 *        between a vector (EH) and rows of an embedding matrix.
 * @param[in] EH Device pointer to the horizontal retention vector (size dim).
 * @param[in] embeddings Device pointer to the token embeddings matrix (row-major: voc x dim).
 * @param[out] predictionLogits Device pointer to store the dot products for all tokens (size voc).
 * @param[in] dim The embedding dimension (size of EH and columns of embeddings).
 * @param[in] voc The vocabulary size (number of rows in embeddings).
 * @return The index of the token embedding with the highest dot product. Returns -1 if voc <= 0 or embeddings is null.
 * @note Assumes that the case of "all dot products being exactly equal" is handled implicitly by returning the first max index found.
 * @note Assumes FLT_MAX is available (usually via <cfloat> or CUDA includes).
 */
__device__ int computePredictionWithScores(const float* EH, const float* embeddings, float* predictionLogits,
         int dim, int voc) 
{
    if (voc <= 0 || embeddings == nullptr) {
        return -1; // Handle invalid input
    }
    float max_dot_product = -FLT_MAX;
    int predicted_index = 0;

    // Assuming dim is a multiple of 4 for float4 operations
    int dim_float4 = dim / 4;
    const float4* EH_f4 = (const float4*)EH;

    for (int i = 0; i < voc; ++i) {
        const float4* current_embedding_row_f4 = (const float4*)(embeddings + i * dim);

        float current_dot_product = 0.0f;
        for (int k = 0; k < dim_float4; ++k) {
            current_dot_product += EH_f4[k].x * current_embedding_row_f4[k].x + EH_f4[k].y * current_embedding_row_f4[k].y + EH_f4[k].z * current_embedding_row_f4[k].z + EH_f4[k].w * current_embedding_row_f4[k].w;
        }
        predictionLogits[i] = current_dot_product;

        if (current_dot_product > max_dot_product) {
            max_dot_product = current_dot_product;
            predicted_index = i;
        }
    }
    return predicted_index;
}

/**
 * @brief CUDA kernel wrapper to launch the computePrediction device function.
 *        This kernel is intended to be launched with a single thread.
 * @param[in] EH Device pointer to the horizontal retention vector (size dim).
 * @param[in] embeddings Device pointer to the token embeddings matrix (row-major: voc x dim).
 * @param[out] result_index Device pointer to an integer where the predicted token index will be stored.
 * @param[in] dim The embedding dimension.
 * @param[in] voc The vocabulary size.
 */
__global__ void kernelComputePrediction(const float* EH, const float* embeddings, int* result_index, int dim, int voc) {
    // This kernel is launched with a single thread, which calls the device function.
    *result_index = computePrediction(EH, embeddings, dim, voc);
}

/**
 * @brief CUDA kernel wrapper to launch the computePredictionWithScores device function.
 *        This kernel is intended to be launched with a single thread.
 * @param[in] EH Device pointer to the input vector (size dim).
 * @param[in] embeddings Device pointer to the token embeddings matrix (row-major: voc x dim).
 * @param[out] predictionLogits Device pointer to store the dot products for all tokens (size voc).
 * @param[out] result_index Device pointer to an integer where the predicted token index will be stored.
 * @param[in] dim The embedding dimension.
 * @param[in] voc The vocabulary size.
 */
__global__ void kernelComputePredictionWithScores(const float* EH, const float* embeddings, float* predictionLogits, 
    int* result_index, int dim, int voc)
{
    // This kernel is launched with a single thread, which calls the device function.
    *result_index = computePredictionWithScores(EH, embeddings, predictionLogits, dim, voc);
}

__global__ void kernelVecDotVec(const float* vec1, const float* vec2, float* result, int dim)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *result = cuComputeDot(vec1, vec2, dim);
    }
}

__global__ void kernelDotvecmatvec(const float* vec1, const float* vec2, const float* matrix, float* result, int dim)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *result = cuComputeDotvmv(vec1, vec2, matrix, dim);
    }
}

__global__ void kernelComputeHeadSumsMaskedev(const float* d_head, float* d_col_sums,
                                          int num_tokens, bool isSelfAttention)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Parallelize over token index 'i'

    if (i < num_tokens) {
        float col_sum_l = 0.0f;

        // Calculate column sum (l) for token i: sum head[j][i] for j < num_tokens, applying mask if needed
        for (int j = 0; j < num_tokens; ++j) {
            // Apply self-attention mask: only sum if j <= i
            if (!isSelfAttention || j <= i) {
                col_sum_l += d_head[j * num_tokens + i];
            }
        }

        d_col_sums[i] = col_sum_l;
    }
}

/**
 * @brief CUDA kernel to compute row sums and column sums of the attention head matrix,
 *        applying causal masking for self-attention if specified.
 *        Row sum k[i] = sum_{j=0}^{limit-1}(head[i][j])
 *        Col sum l[i] = sum_{j=0}^{limit-1}(head[j][i])
 *        where limit = isSelfAttention ? i : num_tokens.
 * @param[in] d_head Device pointer to the input head matrix (row-major, num_tokens x num_tokens).
 * @param[out] d_row_sums Device pointer to store the computed row sums (size num_tokens).
 * @param[out] d_col_sums Device pointer to store the computed column sums (size num_tokens).
 * @param[in] num_tokens The dimension of the square head matrix (number of tokens).
 * @param[in] isSelfAttention Boolean flag; if true, applies causal masking (j < limit = i).
 */
__global__ void kernelComputeHeadSumsMasked(const float* d_head, float* d_row_sums, float* d_col_sums, int num_tokens,
    bool isSelfAttention)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Parallelize over token index 'i'

    if (i < num_tokens) {
        float row_sum_k = 0.0f;
        float col_sum_l = 0.0f;

        // Calculate row sum (k) for token i: sum head[i][j] for j < num_tokens, applying mask if needed
        for (int j = 0; j < num_tokens; ++j) {
            // Apply self-attention mask: only sum if j <= i
            if (!isSelfAttention || j <= i) {
                row_sum_k += d_head[i * num_tokens + j];
            }
        }

        // Calculate column sum (l) for token i: sum head[j][i] for j < num_tokens, applying mask if needed
        for (int j = 0; j < num_tokens; ++j) {
            // Apply self-attention mask: only sum if j <= i
            if (!isSelfAttention || j <= i) {
                col_sum_l += d_head[j * num_tokens + i];
            }
        }

        d_row_sums[i] = row_sum_k;
        d_col_sums[i] = col_sum_l;
    }
}

__global__ void kernelAccumulateWeightedVectorsev(const float* d_row_sums, const float* d_col_sums,
                                                const float* d_K, const float* d_Q,
                                                float* d_dv_accum, int num_tokens, int h_dim)
{
    int h_idx = blockIdx.x * blockDim.x + threadIdx.x; // Parallelize over the h_dim dimension

    if (h_idx < h_dim) {
        float total_dv_for_h_idx = 0.0f;

        // Iterate through each token i
        for (int i = 0; i < num_tokens; ++i) {
            // Access Q[i][h_idx] using row-major indexing
            float q_i_h = d_Q[i * h_dim + h_idx];

            // Accumulate dv contribution for this h_idx
            total_dv_for_h_idx += d_col_sums[i] * q_i_h;
        }

        // Atomically add the computed sums for this h_idx to the global accumulators
        atomicAdd(&d_dv_accum[h_idx], total_dv_for_h_idx);
    }
}

/**
 * @brief CUDA kernel to accumulate weighted Key and Query vectors based on head row/column sums.
 *        Computes:
 *        d_dh_accum += sum_i (d_row_sums[i] * d_K[i])
 *        d_dv_accum += sum_i (d_col_sums[i] * d_Q[i])
 *        Uses atomicAdd for safe accumulation across threads working on the same h_idx.
 * @param[in] d_row_sums Device pointer to the precomputed row sums (size num_tokens).
 * @param[in] d_col_sums Device pointer to the precomputed column sums (size num_tokens).
 * @param[in] d_K Device pointer to the Key matrix (row-major, num_tokens x h_dim).
 * @param[in] d_Q Device pointer to the Query matrix (row-major, num_tokens x h_dim).
 * @param[in,out] d_dh_accum Device pointer to the accumulated dh vector (size h_dim). MUST be zero-initialized before kernel launch.
 * @param[in,out] d_dv_accum Device pointer to the accumulated dv vector (size h_dim). MUST be zero-initialized before kernel launch.
 * @param[in] num_tokens The number of tokens (rows in K/Q, size of sum vectors).
 * @param[in] h_dim The dimension of the Key/Query vectors (e.g., CONTEXT_WIN).
 */
__global__ void kernelAccumulateWeightedVectors(const float* d_row_sums, const float* d_col_sums,
                                                const float* d_K, const float* d_Q, float* d_dh_accum,
                                                float* d_dv_accum, int num_tokens, int h_dim)
{
    int h_idx = blockIdx.x * blockDim.x + threadIdx.x; // Parallelize over the h_dim dimension

    if (h_idx < h_dim) {
        float total_dh_for_h_idx = 0.0f;
        float total_dv_for_h_idx = 0.0f;

        // Iterate through each token i
        for (int i = 0; i < num_tokens; ++i) {
            // Access K[i][h_idx] and Q[i][h_idx] using row-major indexing
            float k_i_h = d_K[i * h_dim + h_idx];
            float q_i_h = d_Q[i * h_dim + h_idx];

            // Accumulate dh contribution for this h_idx
            total_dh_for_h_idx += d_row_sums[i] * k_i_h;
            // Accumulate dv contribution for this h_idx
            total_dv_for_h_idx += d_col_sums[i] * q_i_h;
        }

        // Atomically add the computed sums for this h_idx to the global accumulators
        atomicAdd(&d_dh_accum[h_idx], total_dh_for_h_idx);
        atomicAdd(&d_dv_accum[h_idx], total_dv_for_h_idx);
    }
}


/**
 * @brief CUDA kernel to accumulate weighted Key and Query vectors based on head row/column sums.
 *        Computes:
 *        d_dh_accum += sum_i (d_row_sums[i] * d_K[i])
 *        d_dv_accum += sum_i (d_col_sums[i] * d_Q[i])
 *        Uses atomicAdd for safe accumulation across threads working on the same h_idx.
 * @param[in] d_row_sums Device pointer to the precomputed row sums (size num_tokens).
 * @param[in] d_col_sums Device pointer to the precomputed column sums (size num_tokens).
 * @param[in] d_K Device pointer to the Key matrix (row-major, num_tokens x h_dim).
 * @param[in] d_Q Device pointer to the Query matrix (row-major, num_tokens x h_dim).
 * @param[in,out] d_dh_accum Device pointer to the accumulated dh vector (size h_dim). MUST be zero-initialized before kernel launch.
 * @param[in,out] d_dv_accum Device pointer to the accumulated dv vectors (size n_dim*h_dim). MUST be zero-initialized before kernel launch.
 * @param[in] num_tokens The number of tokens (rows in K/Q, size of sum vectors).
 * @param[in] h_dim The dimension of the Key/Query vectors (e.g., CONTEXT_WIN).
 */
__global__ void kernelAccumulateWeightedVectors(const float* d_row_sums, const float* d_col_sums,
                                                const float* d_K, const float* d_Q, float* d_dh_accum,
                                                float* d_dv_accum, int num_tokens, int h_dim, int n_dim)
{
    int h_idx = blockIdx.x * blockDim.x + threadIdx.x; // Parallelize over the h_dim dimension

    if (h_idx < h_dim) {
        float total_dh_for_h_idx = 0.0f;
        float total_dv_for_h_idx = 0.0f;

        // Iterate through each token i
        for (int i = 0; i < num_tokens; ++i) {
            // Access K[i][h_idx] and Q[i][h_idx] using row-major indexing
            float k_i_h = d_K[i * h_dim + h_idx];
            float q_i_h = d_Q[i * h_dim + h_idx];

            // Accumulate dh contribution for this h_idx
            total_dh_for_h_idx += d_row_sums[i] * k_i_h;
            // Accumulate dv contribution for this h_idx
            total_dv_for_h_idx += d_col_sums[i] * q_i_h;
        }

        // Atomically add the computed sums for this h_idx to the global accumulators
        atomicAdd(&d_dh_accum[h_idx], total_dh_for_h_idx);
        atomicAdd(&d_dv_accum[h_idx], total_dv_for_h_idx);
    }
}

// Kernel to accumulate the first num_rows of d_EV into d_output
// d_EV is num_rows x col_size (conceptually, row-major)
// d_output is 1 x col_size
__global__ void accumulateEVRowsKernel(const float* d_EV, float* d_output, int num_rows, int col_size) {
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x; // Each thread computes one element of d_output

    if (col_idx < col_size) {
        float sum = 0.0f;
        for (int row_idx = 0; row_idx < num_rows; ++row_idx) {
            sum += d_EV[static_cast<size_t>(row_idx) * col_size + col_idx];
        }
        d_output[col_idx] = sum;
    }
}

// Kernel to add d_vector_to_add to each of the first num_rows_to_update of d_EV_rows
// d_EV_rows is num_rows_to_update x num_cols (conceptually, row-major)
// d_vector_to_add is 1 x num_cols
__global__ void updateEVRowsKernel(float* d_EV_rows, const float* d_vector_to_add, int num_rows_to_update, int num_cols) {
    int row_idx = blockIdx.x * blockDim.x + threadIdx.x; // Global thread ID, maps to a row
    if (row_idx < num_rows_to_update) { // Each thread handles one row
        for (int col_idx = 0; col_idx < num_cols; ++col_idx) {
            d_EV_rows[static_cast<size_t>(row_idx) * num_cols + col_idx] += d_vector_to_add[col_idx];
        }
    }
}

/**------------------------------------BACKPROP------------------------------------**/

__global__ void kernelComputeGradpred(const float* predNorm, const float* oneHot,
                                    float* grad_pred, int vocab)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < vocab) {
        float pNorm = predNorm[idx];
        float label = oneHot[idx];
        grad_pred[idx] = pNorm - label;
    }
}

__global__ void KernelComputeGradDeEmbeddings(const float* d_delta,
                                           const float4* otok,      // Read otok as float4
                                           float4* grad,          // Write grad as float4
                                           int vocab,
                                           int dEmbedDim_div_4) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // Corresponds to vocab dimension
    int col4 = blockIdx.x * blockDim.x + threadIdx.x; // Corresponds to dEmbedDim dimension (in float4 units)

    if (row < vocab && col4 < dEmbedDim_div_4) {
        const float delta_val = d_delta[row];
        grad[row * dEmbedDim_div_4 + col4] = make_float4(delta_val * otok[col4].x,
                                                        delta_val * otok[col4].y,
                                                        delta_val * otok[col4].z,
                                                        delta_val * otok[col4].w);
    }
}

__global__ void kernelGradForAttentionOutput(const float* d_deEmbed,
                                          const float* d_delta,
                                          float* grad4EH,
                                          int vocabSize, int dEmbedDim)
{
    int idx4 = (blockIdx.x * blockDim.x + threadIdx.x) * 4; // Each work item processes 4 elements
    if (idx4 < dEmbedDim) {
        float4 sum4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        const float4* d_deEmbed_f4 = (const float4*)d_deEmbed;
        int dEmbedDim_f4 = dEmbedDim / 4;

        for (int j = 0; j < vocabSize; ++j) {
            float4 deEmbed = d_deEmbed_f4[j * dEmbedDim_f4 + (idx4/4)];
            sum4.x += d_delta[j] * deEmbed.x;
            sum4.y += d_delta[j] * deEmbed.y;
            sum4.z += d_delta[j] * deEmbed.z;
            sum4.w += d_delta[j] * deEmbed.w;
        }
        
        grad4EH[idx4] = sum4.x;
        grad4EH[idx4 + 1] = sum4.y;
        grad4EH[idx4 + 2] = sum4.z;
        grad4EH[idx4 + 3] = sum4.w;
    }
}

__global__ void kernelComputeGradientsEH(const float* eh, const float* expected_h,
                                       float* grad_eh, int embedding_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < embedding_dim) {
        grad_eh[idx] = eh[idx] - expected_h[idx];
    }
}

__global__ void kernelComputeGradientsEHEVFromMSE(const float* eh, const float* expected_h,
                                          float* grad_eh, float* grad_ev_scaled, int embedding_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < embedding_dim) {
        float grad = 2.0f * (eh[idx] - expected_h[idx]);
        grad_eh[idx] = grad;
        grad_ev_scaled[idx] = grad * 0.01f;
    }
}

/**
 * @brief CUDA kernel to compute gradients w.r.t. MLP inputs (dh, dv) for the *first head* scenario.
 *        Extracts gradients from the first column (input neuron 0) of the first layer's gradient weights (gweights[0]).
 *        grad_dh[i] = hor_gweights[0][i][0]
 *        grad_dv[i] = ver_gweights[0][i][0]
 * @param[in] d_hor_gweights0 Device pointer to the horizontal MLP's first layer gradient weights (row-major, embedding_dim x embedding_dim). Can be null.
 * @param[in] d_ver_gweights0 Device pointer to the vertical MLP's first layer gradient weights (row-major, embedding_dim x embedding_dim). Can be null.
 * @param[out] grad_dh Device pointer to store the computed gradient w.r.t. horizontal input (size embedding_dim).
 * @param[out] grad_dv Device pointer to store the computed gradient w.r.t. vertical input (size embedding_dim).
 * @param[in] embedding_dim The dimension of the embedding and MLP layers.
 */
__global__ void kernelComputeGradDhDv_1stHead(const float* d_hor_gweights0, const float* d_ver_gweights0,
    float* grad_dh, float* grad_dv, int embedding_dim)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Corresponds to the output neuron index 'i' of the first layer

    if (i < embedding_dim) {
        // Calculate flat index for gweights[0][i][0] assuming row-major [output_neuron][input_neuron]
        int gweight_idx = i * embedding_dim + 0; // Index for the weight connecting input 0 to output i

        if (d_hor_gweights0 != nullptr) {
            grad_dh[i] = d_hor_gweights0[gweight_idx];
        }
        else {
            grad_dh[i] = 0.0f; // Set to zero if horizontal MLP path is not active/provided
        }

        if (d_ver_gweights0 != nullptr) {
            grad_dv[i] = d_ver_gweights0[gweight_idx];
        }
        else {
            grad_dv[i] = 0.0f; // Set to zero if vertical MLP path is not active/provided
        }
    }
}


/**
 * @brief CUDA kernel for Step 1 of `cuBackward(expected)`: Compute initial gradients w.r.t. EH and a scaled version for EV.
 *        grad_eh = 2 * (eh - expected_h)
 *        grad_ev_scaled = grad_eh * 0.1
 * @param[in] eh Device pointer to the horizontal embedding vector EH (size embedding_dim).
 * @param[in] expected_h Device pointer to the target horizontal embedding vector (size embedding_dim).
 * @param[out] grad_eh Device pointer to store the gradient w.r.t. EH (size embedding_dim).
 * @param[out] grad_ev_scaled Device pointer to store the scaled gradient for the EV path (size embedding_dim).
 * @param[in] embedding_dim The dimension of the embedding vectors.
 */
__global__ void kernelComputeGradientsEH_EV(const float* eh, const float* expected_h,
                                            float* grad_eh, float* grad_ev_scaled, int embedding_dim) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < embedding_dim) {
        float pred = eh[idx];          // predicted probability (sigmoid output)
        float label = expected_h[idx]; // true label (0 or 1)

        float grad = pred - label;
 
        grad_eh[idx] = grad;
        grad_ev_scaled[idx] = grad * 0.1f;  // scale for vertical path
    }
}


/**
 * @brief CUDA kernel for Step 3 of `cuBackward(expected)`: Compute gradients w.r.t. MLP inputs (dh, dv)
 *        by summing the first layer's gradient weights across the input dimension.
 *        grad_dh[i] = sum_j (hor_gweights[0][j][i])  <- Note: Indexing seems reversed vs kernelComputeGradDhDv_1stHead
 *        grad_dv[i] = sum_j (ver_gweights[0][j][i])  <- Note: Indexing seems reversed vs kernelComputeGradDhDv_1stHead
 *        Assuming gweights are stored [output_neuron][input_neuron] (row-major).
 *        The kernel computes sum over input neurons 'j' for a given output neuron 'i'.
 * @param[in] d_hor_gweights0 Device pointer to the horizontal MLP's first layer gradient weights (row-major, embedding_dim x embedding_dim).
 * @param[in] d_ver_gweights0 Device pointer to the vertical MLP's first layer gradient weights (row-major, embedding_dim x embedding_dim).
 * @param[out] grad_dh Device pointer to store the computed gradient w.r.t. horizontal input (size embedding_dim).
 * @param[out] grad_dv Device pointer to store the computed gradient w.r.t. vertical input (size embedding_dim).
 * @param[in] embedding_dim The dimension of the embedding and MLP layers.
 */
__global__ void kernelComputeGradDhDv(const float* d_hor_gweights0, const float* d_ver_gweights0,
                                      float* grad_dh, float* grad_dv,
                                      int embedding_dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Corresponds to the output neuron index 'i'

    if (i < embedding_dim) {
        float sum_dh = 0.0f;
        float sum_dv = 0.0f;
        // Sum gweights[0][i][j] over j (input dimension)
        // Flat index for gweights[0][i][j] is i * embedding_dim + j
        for (int j = 0; j < embedding_dim; ++j) {
            int gweight_idx = i * embedding_dim + j;
            sum_dh += d_hor_gweights0[gweight_idx];
            sum_dv += d_ver_gweights0[gweight_idx];
        }
        grad_dh[i] = sum_dh;
        grad_dv[i] = sum_dv;
    }
}


/**
 * @brief CUDA kernel for Step 4 of `cuBackward(expected)`: Compute intermediate values pre_MH and pre_MV.
 *        pre_mh[h] = sum_i ( sum_j(head[i][j]) * K[i][h] )
 *        pre_mv[h] = sum_i ( sum_j(head[j][i]) * Q[i][h] )
 * @param[in] head Device pointer to the attention head matrix (row-major, token_count x token_count).
 * @param[in] k Device pointer to the Key matrix K (row-major, token_count x mat_heights).
 * @param[in] q Device pointer to the Query matrix Q (row-major, token_count x mat_heights).
 * @param[out] pre_mh Device pointer to store the pre_MH vector (size mat_heights).
 * @param[out] pre_mv Device pointer to store the pre_MV vector (size mat_heights).
 * @param[in] token_count The number of tokens (dimension of head matrix, rows of K/Q).
 * @param[in] mat_heights The height dimension (columns of K/Q, size of pre_mh/pre_mv).
 * @note This implementation involves redundant calculations (sums over head). Consider optimizing with separate reduction kernels if performance is critical.
 */
__global__ void kernelComputePreMH_MV(const float* head, const float* k, const float* q,
                                      float* pre_mh, float* pre_mv,
                                      int token_count, int mat_heights) {
    int h = blockIdx.x * blockDim.x + threadIdx.x; // Parallelize over the mat_heights dimension

    if (h < mat_heights) {
        float mh_val_h = 0.0f;
        float mv_val_h = 0.0f;

        // Iterate over tokens 'i'
        for (int i = 0; i < token_count; ++i) {
            float sum_head_row_i = 0.0f; // Sum of row i of head
            float sum_head_col_i = 0.0f; // Sum of column i of head

            // Calculate sum of row i and column i of the head matrix
            for (int j = 0; j < token_count; ++j) {
                sum_head_row_i += head[i * token_count + j]; // head[i][j]
                sum_head_col_i += head[j * token_count + i]; // head[j][i]
            }

            // Access K[i][h] and Q[i][h] using row-major indexing
            float k_ih = k[i * mat_heights + h];
            float q_ih = q[i * mat_heights + h];

            // Accumulate results for dimension h
            mh_val_h += sum_head_row_i * k_ih;
            mv_val_h += sum_head_col_i * q_ih;
        }
        pre_mh[h] = mh_val_h;
        pre_mv[h] = mv_val_h;
    }
}


/**
 * @brief CUDA kernel for Step 4 of `cuBackward(expected)`: Compute gradients grad_MH and grad_MV via outer product.
 *        grad_mh[h][d] = pre_mh[h] * grad_dh[d]
 *        grad_mv[h][d] = pre_mv[h] * grad_dv[d]
 * @param[in] pre_mh Device pointer to the pre_MH vector (size mat_heights).
 * @param[in] pre_mv Device pointer to the pre_MV vector (size mat_heights).
 * @param[in] grad_dh Device pointer to the gradient w.r.t. horizontal input (size embedding_dim).
 * @param[in] grad_dv Device pointer to the gradient w.r.t. vertical input (size embedding_dim).
 * @param[out] grad_mh Device pointer to store the gradient w.r.t. MH (row-major, mat_heights x embedding_dim).
 * @param[out] grad_mv Device pointer to store the gradient w.r.t. MV (row-major, mat_heights x embedding_dim).
 * @param[in] mat_heights The height dimension.
 * @param[in] embedding_dim The embedding dimension.
 */
__global__ void kernelComputeGradMH_MV(const float* pre_mh, const float* pre_mv,
                                       const float* grad_dh, const float* grad_dv,
                                       float* grad_mh, float* grad_mv,
                                       int mat_heights, int embedding_dim) {
    int h = blockIdx.y * blockDim.y + threadIdx.y; // Row index (mat_heights)
    int d = blockIdx.x * blockDim.x + threadIdx.x; // Column index (embedding_dim)

    if (h < mat_heights && d < embedding_dim) {
        int idx = h * embedding_dim + d; // Flat index for row-major matrix
        grad_mh[idx] = pre_mh[h] * grad_dh[d];
        grad_mv[idx] = pre_mv[h] * grad_dv[d];
    }
}


/**
 * @brief CUDA kernel for Step 5 of `cuBackward(expected)`: Compute the gradient w.r.t. the attention head matrix.
 *        grad_head[i][j] = (sum_d (sum_h K[i][h]*MH[h][d]) * grad_dh[d]) + (sum_d (sum_h Q[j][h]*MV[h][d]) * grad_dv[d])
 *        This is equivalent to: grad_head[i][j] = dot(K[i]*MH, grad_dh) + dot(Q[j]*MV, grad_dv)
 * @param[in] k Device pointer to the Key matrix K (row-major, token_count x mat_heights).
 * @param[in] q Device pointer to the Query matrix Q (row-major, token_count x mat_heights).
 * @param[in] mh_a Device pointer to the MH matrix (row-major, mat_heights x embedding_dim).
 * @param[in] mv_a Device pointer to the MV matrix (row-major, mat_heights x embedding_dim).
 * @param[in] grad_dh Device pointer to the gradient w.r.t. horizontal input (size embedding_dim).
 * @param[in] grad_dv Device pointer to the gradient w.r.t. vertical input (size embedding_dim).
 * @param[out] grad_head Device pointer to store the gradient w.r.t. the head matrix (row-major, token_count x token_count).
 * @param[in] token_count The number of tokens.
 * @param[in] mat_heights The height dimension.
 * @param[in] embedding_dim The embedding dimension.
 */
__global__ void kernelComputeGradHead(const float* k, const float* q,
                                      const float* mh_a, const float* mv_a,
                                      const float* grad_dh, const float* grad_dv,
                                      float* grad_head,
                                      int token_count, int mat_heights, int embedding_dim) {
    int i = blockIdx.y * blockDim.y + threadIdx.y; // Row index for grad_head (token i)
    int j = blockIdx.x * blockDim.x + threadIdx.x; // Column index for grad_head (token j)

    if (i < token_count && j < token_count) {
        float grad_dh_term_ij = 0.0f; // Contribution from the dh path for grad_head[i][j]
        float grad_dv_term_ij = 0.0f; // Contribution from the dv path for grad_head[i][j]

        // Calculate dot(K[i]*MH, grad_dh) and dot(Q[j]*MV, grad_dv)
        for (int d = 0; d < embedding_dim; ++d) {
            float k_mh_id = 0.0f; // K[i] * MH[:, d]
            float q_mv_jd = 0.0f; // Q[j] * MV[:, d]

            // Compute the dot products involving K[i]/Q[j] and columns of MH/MV
            for (int h = 0; h < mat_heights; ++h) {
                // Access K[i][h], Q[j][h], MH[h][d], MV[h][d] using row-major indexing
                k_mh_id += k[i * mat_heights + h] * mh_a[h * embedding_dim + d];
                q_mv_jd += q[j * mat_heights + h] * mv_a[h * embedding_dim + d];
            }
            // Accumulate the contribution for dimension d
            grad_dh_term_ij += k_mh_id * grad_dh[d];
            grad_dv_term_ij += q_mv_jd * grad_dv[d];
        }

        // Store the total gradient for head[i][j]
        grad_head[i * token_count + j] = grad_dh_term_ij + grad_dv_term_ij;
    }
}


/**
 * @brief CUDA kernel for Step 7 of `cuBackward(expected)`: Compute gradients w.r.t. K and Q matrices.
 *        grad_K = grad_KdotQ * Q^T
 *        grad_Q = K^T * grad_KdotQ
 * @param[in] grad_kdotq Device pointer to the gradient w.r.t. KdotQ (row-major, token_count x token_count).
 * @param[in] k Device pointer to the Key matrix K (row-major, token_count x mat_heights).
 * @param[in] q Device pointer to the Query matrix Q (row-major, token_count x mat_heights).
 * @param[out] grad_k Device pointer to store the gradient w.r.t. K (row-major, token_count x mat_heights).
 * @param[out] grad_q Device pointer to store the gradient w.r.t. Q (row-major, token_count x mat_heights).
 * @param[in] token_count The number of tokens.
 * @param[in] mat_heights The height dimension.
 */
__global__ void kernelComputeGradK_Q(const float* grad_kdotq, const float* k, const float* q,
                                     float* grad_k, float* grad_q, int token_count, int mat_heights) {
    // Each thread computes one element of grad_k and one element of grad_q
    int i = blockIdx.y * blockDim.y + threadIdx.y; // Row index for grad_k (token i)
    int h = blockIdx.x * blockDim.x + threadIdx.x; // Column index for grad_k/grad_q (height h)

    if (i < token_count && h < mat_heights) {
        float sum_for_grad_k_ih = 0.0f; // Accumulator for grad_K[i][h] = sum_j(grad_kdotq[j][i] * Q[j][h])
        float sum_for_grad_q_ih = 0.0f; // Accumulator for grad_Q[i][h] = sum_j(grad_kdotq[i][j] * K[j][h])

        // Correctly calculate grad_Q[i][h] = sum_j (grad_kdotq[i][j] * K[j][h])
        for (int j = 0; j < token_count; ++j) {
            sum_for_grad_q_ih += grad_kdotq[i * token_count + j] * k[j * mat_heights + h];
        }

        // Correctly calculate grad_K[i][h] = sum_j (grad_kdotq[j][i] * Q[j][h])
        for (int j = 0; j < token_count; ++j) {
            // Access grad_kdotq transposed: grad_kdotq[j][i]
            sum_for_grad_k_ih += grad_kdotq[j * token_count + i] * q[j * mat_heights + h];
        }

        // Store results using row-major indexing
        grad_k[i * mat_heights + h] = sum_for_grad_k_ih;
        grad_q[i * mat_heights + h] = sum_for_grad_q_ih;
    }
}


/**
 * @brief CUDA kernel for Step 8 of `cuBackward(expected)`: Compute gradients w.r.t. MK and MQ.
 *        grad_MK[h][d] = sum_i (grad_K[i][h] * X[i][d])
 *        grad_MQ[h][d] = sum_i (grad_Q[i][h] * X[i][d])
 * @param[in] grad_k Device pointer to the gradient w.r.t. K (row-major, token_count x mat_heights).
 * @param[in] grad_q Device pointer to the gradient w.r.t. Q (row-major, token_count x mat_heights).
 * @param[in] k Device pointer to the Key matrix (row-major, token_count x embedding_dim). **Likely should be original token embeddings X.**
 * @param[in] q Device pointer to the Query matrix (row-major, token_count x embedding_dim). **Likely should be original token embeddings X.**
 * @param[out] grad_mk Device pointer to store the gradient w.r.t. MK (row-major, mat_heights x embedding_dim).
 * @param[out] grad_mq Device pointer to store the gradient w.r.t. MQ (row-major, mat_heights x embedding_dim).
 * @param[in] token_count The number of tokens.
 * @param[in] mat_heights The height dimension.
 * @param[in] embedding_dim The embedding dimension.
 * @note There's a potential inconsistency in the source C++ regarding the dimensions/source of K and Q used in this step versus others.
 *       This kernel assumes `k` and `q` inputs are `token_count x embedding_dim` (likely the original token embeddings or equivalent).
 *       If `k` and `q` passed are `token_count x mat_heights`, this kernel's logic is incorrect based on the C++ formula.
 *       Using `kernelComputeGradMK_MQ_Simplified` might be preferred if original embeddings are available.
 */
__global__ void kernelComputeGradMK_MQ(const float* grad_k, const float* grad_q,
                                       const float* k, const float* q, // Assumed token_count x embedding_dim
                                       float* grad_mk, float* grad_mq,
                                       int token_count, int mat_heights, int embedding_dim) {
    int h = blockIdx.y * blockDim.y + threadIdx.y; // Row index for grad_mk/grad_mq (height h)
    int d = blockIdx.x * blockDim.x + threadIdx.x; // Column index for grad_mk/grad_mq (dimension d)

    if (h < mat_heights && d < embedding_dim) {
        float sum_mk_hd = 0.0f;
        float sum_mq_hd = 0.0f;

        // Sum over tokens 'i'
        for (int i = 0; i < token_count; ++i) {
            // Access grad_K[i][h], grad_Q[i][h] (from token_count x mat_heights matrices)
            float grad_k_ih = grad_k[i * mat_heights + h];
            float grad_q_ih = grad_q[i * mat_heights + h];

            // Access K[i][d], Q[i][d] (from assumed token_count x embedding_dim matrices)
            float k_id = k[i * embedding_dim + d];
            float q_id = q[i * embedding_dim + d];

            sum_mk_hd += grad_k_ih * k_id;
            sum_mq_hd += grad_q_ih * q_id;
        }

        // Store results using row-major indexing
        grad_mk[h * embedding_dim + d] = sum_mk_hd;
        grad_mq[h * embedding_dim + d] = sum_mq_hd;
    }
}

// --- Kernels specific to second cuBackward overload (`cuBackward(expectedV)`) ---

/**
 * @brief CUDA kernel for Step 1 of `cuBackward(expectedV)`: Compute gradients w.r.t. EV.
 *        Calculates the full gradient `grad_ev_full`, sums it along the context window dimension
 *        to get `grad_ev_summed`, and scales the summed gradient for MLP input `grad_ev_scaled`.
 *        grad_ev_full[win][embed] = 2 * (ev[win][embed] - expected_v[win][embed])
 *        grad_ev_summed[embed] = sum_win (grad_ev_full[win][embed])
 *        grad_ev_scaled[embed] = grad_ev_summed[embed] * learning_rate (Note: C++ scales by LR here)
 * @param[in] ev Device pointer to the EV matrix (row-major, context_win x embedding_dim).
 * @param[in] expected_v Device pointer to the target EV matrix (row-major, context_win x embedding_dim).
 * @param[out] grad_ev_full Device pointer to store the full gradient w.r.t. EV (size context_win * embedding_dim).
 * @param[out] grad_ev_summed Device pointer to store the gradient summed over the context window (size embedding_dim).
 * @param[out] grad_ev_scaled Device pointer to store the summed gradient scaled by learning rate (size embedding_dim).
 * @param[in] learning_rate The learning rate (used for scaling the summed gradient).
 * @param[in] context_win The context window size (rows in EV).
 * @param[in] embedding_dim The embedding dimension (columns in EV).
 */
__global__ void kernelComputeGradientsEV_V(const float* ev, const float* expected_v,
                                           float* grad_ev_full, float* grad_ev_summed, float* grad_ev_scaled,
                                           float learning_rate,
                                           int context_win, int embedding_dim) {
    int embed_idx = blockIdx.x * blockDim.x + threadIdx.x; // Parallelize over embedding dimension

    if (embed_idx < embedding_dim) {
        float sum_grad_embed = 0.0f; // Accumulator for the sum along the context window

        // Iterate through the context window for the current embedding dimension
        for (int win_idx = 0; win_idx < context_win; ++win_idx) {
            int idx = win_idx * embedding_dim + embed_idx; // Flat index for ev[win_idx][embed_idx]

            float pred = ev[idx];              // prediction
            float label = expected_v[idx];     // true label (0 or 1)

            // Clamp pred to avoid division by zero
            pred = fminf(fmaxf(pred, 1e-7f), 1.0f - 1e-7f);

            // Binary Cross Entropy gradient
            float grad = (pred - label) / (pred * (1.0f - pred));

            grad_ev_full[idx] = grad;       // Store element-wise gradient
            sum_grad_embed += grad;         // Accumulate for this embed dimension
        }

        grad_ev_summed[embed_idx] = sum_grad_embed;
        grad_ev_scaled[embed_idx] = sum_grad_embed * learning_rate;
    }
}


/**
 * @brief CUDA kernel for Step 4 of `cuBackward(expectedV)`: Compute intermediate value pre_MV only.
 *        pre_mv[h] = sum_i ( sum_j(head[j][i]) * Q[i][h] )
 * @param[in] head Device pointer to the attention head matrix (row-major, token_count x token_count).
 * @param[in] q Device pointer to the Query matrix Q (row-major, token_count x mat_heights).
 * @param[out] pre_mv Device pointer to store the pre_MV vector (size mat_heights).
 * @param[in] token_count The number of tokens (dimension of head matrix, rows of Q).
 * @param[in] mat_heights The height dimension (columns of Q, size of pre_mv).
 * @note This implementation involves redundant calculations (sums over head columns). Consider optimizing if needed.
 */
__global__ void kernelComputePreMV_V(const float* head, const float* q,
                                     float* pre_mv,
                                     int token_count, int mat_heights) {
    int h = blockIdx.x * blockDim.x + threadIdx.x; // Parallelize over the mat_heights dimension

    if (h < mat_heights) {
        float mv_val_h = 0.0f;
        // Iterate over tokens 'i' (which corresponds to column index in C++ head sum)
        for (int i = 0; i < token_count; ++i) {
            float sum_head_col_i = 0.0f; // Sum of column i of head
            // Calculate sum of column i of the head matrix
            for (int j = 0; j < token_count; ++j) {
                sum_head_col_i += head[j * token_count + i]; // head[j][i]
            }
            // Access Q[i][h] using row-major indexing
            float q_ih = q[i * mat_heights + h];
            // Accumulate result for dimension h
            mv_val_h += sum_head_col_i * q_ih;
        }
        pre_mv[h] = mv_val_h;
    }
}


/**
 * @brief CUDA kernel for Step 4 of `cuBackward(expectedV)`: Compute gradient grad_MV only via outer product.
 *        grad_mv[h][d] = pre_mv[h] * grad_dv[d]
 * @param[in] pre_mv Device pointer to the pre_MV vector (size mat_heights).
 * @param[in] grad_dv Device pointer to the gradient w.r.t. vertical input (size embedding_dim).
 * @param[out] grad_mv Device pointer to store the gradient w.r.t. MV (row-major, mat_heights x embedding_dim).
 * @param[in] mat_heights The height dimension.
 * @param[in] embedding_dim The embedding dimension.
 */
__global__ void kernelComputeGradMV_V(const float* pre_mv, const float* grad_dv,
                                      float* grad_mv,
                                      int mat_heights, int embedding_dim) {
    int h = blockIdx.y * blockDim.y + threadIdx.y; // Row index (mat_heights)
    int d = blockIdx.x * blockDim.x + threadIdx.x; // Column index (embedding_dim)

    if (h < mat_heights && d < embedding_dim) {
        int idx = h * embedding_dim + d; // Flat index for row-major matrix
        grad_mv[idx] = pre_mv[h] * grad_dv[d];
    }
}

/**
 * @brief CUDA kernel for Step 5 of `cuBackward(expectedV)`: Compute the gradient w.r.t. the attention head matrix (dv path only).
 *        grad_head[i][j] = sum_d (sum_h Q[j][h]*MV[h][d]) * grad_dv[d]
 *        This is equivalent to: grad_head[i][j] = dot(Q[j]*MV, grad_dv)
 * @param[in] q Device pointer to the Query matrix Q (row-major, token_count x mat_heights).
 * @param[in] mv_a Device pointer to the MV matrix (row-major, mat_heights x embedding_dim).
 * @param[in] grad_dv Device pointer to the gradient w.r.t. vertical input (size embedding_dim).
 * @param[out] grad_head Device pointer to store the gradient w.r.t. the head matrix (row-major, token_count x token_count).
 * @param[in] token_count The number of tokens.
 * @param[in] mat_heights The height dimension.
 * @param[in] embedding_dim The embedding dimension.
 * @note The calculation for grad_head[i][j] only depends on 'j' (via Q[j]) and not 'i'. All elements in a column 'j' will receive the same gradient value.
 */
__global__ void kernelComputeGradHead_V(const float* q, const float* mv_a,
                                        const float* grad_dv, float* grad_head,
                                        int token_count, int mat_heights, int embedding_dim) {
    int i = blockIdx.y * blockDim.y + threadIdx.y; // Row index for grad_head (token i)
    int j = blockIdx.x * blockDim.x + threadIdx.x; // Column index for grad_head (token j)

    if (i < token_count && j < token_count) {
        float grad_dv_term_j = 0.0f; // Contribution from the dv path for grad_head[any_i][j]

        // Calculate dot(Q[j]*MV, grad_dv)
        for (int d = 0; d < embedding_dim; ++d) {
            float q_mv_jd = 0.0f; // Q[j] * MV[:, d]
            // Compute the dot product involving Q[j] and column d of MV
            for (int h = 0; h < mat_heights; ++h) {
                // Access Q[j][h], MV[h][d] using row-major indexing
                q_mv_jd += q[j * mat_heights + h] * mv_a[h * embedding_dim + d];
            }
            // Accumulate the contribution for dimension d
            grad_dv_term_j += q_mv_jd * grad_dv[d];
        }

        // Store the gradient for head[i][j] (value depends only on j)
        grad_head[i * token_count + j] = grad_dv_term_j;
    }
}


/**
 * @brief CUDA kernel for Step 7 of `cuBackward(expectedV)`: Compute gradient w.r.t. Q matrix only.
 *        grad_Q = K^T * grad_KdotQ
 * @param[in] grad_kdotq Device pointer to the gradient w.r.t. KdotQ (row-major, token_count x token_count).
 * @param[in] k Device pointer to the Key matrix K (row-major, token_count x mat_heights).
 * @param[out] grad_q Device pointer to store the gradient w.r.t. Q (row-major, token_count x mat_heights).
 * @param[in] token_count The number of tokens.
 * @param[in] mat_heights The height dimension.
 */
__global__ void kernelComputeGradQ_V(const float* grad_kdotq, const float* k,
                                     float* grad_q,
                                     int token_count, int mat_heights) {
    // Each thread computes one element of grad_q
    int j = blockIdx.y * blockDim.y + threadIdx.y; // Row index for grad_q (token j)
    int h = blockIdx.x * blockDim.x + threadIdx.x; // Column index for grad_q (height h)

    if (j < token_count && h < mat_heights) {
        float sum_for_grad_q_jh = 0.0f; // Accumulator for grad_Q[j][h]

        // Calculate grad_Q[j][h] = sum_i (K[i][h] * grad_KdotQ[i][j])
        for (int i = 0; i < token_count; ++i) {
            sum_for_grad_q_jh += k[i * mat_heights + h] * grad_kdotq[i * token_count + j];
        }

        // Store result using row-major indexing
        grad_q[j * mat_heights + h] = sum_for_grad_q_jh;
    }
}

/**
 * @brief CUDA kernel for Step 8 of `cuBackward(expectedV)`: Compute gradient w.r.t. MQ only.
 *        grad_MQ[h][d] = sum_i (grad_Q[i][h] * X[i][d])
 * @param[in] grad_q Device pointer to the gradient w.r.t. Q (row-major, token_count x mat_heights).
 * @param[in] q Device pointer to the Query matrix (row-major, token_count x embedding_dim). **Likely should be original token embeddings X.**
 * @param[out] grad_mq Device pointer to store the gradient w.r.t. MQ (row-major, mat_heights x embedding_dim).
 * @param[in] token_count The number of tokens.
 * @param[in] mat_heights The height dimension.
 * @param[in] embedding_dim The embedding dimension.
 * @note See note in `kernelComputeGradMK_MQ` regarding potential inconsistency of `q` input. Assumes `q` is `token_count x embedding_dim`.
 */
__global__ void kernelComputeGradMQ_V(const float* grad_q, const float* q, // Assumed token_count x embedding_dim
                                      float* grad_mq,
                                      int token_count, int mat_heights, int embedding_dim) {
    int h = blockIdx.y * blockDim.y + threadIdx.y; // Row index for grad_mq (height h)
    int d = blockIdx.x * blockDim.x + threadIdx.x; // Column index for grad_mq (dimension d)

    if (h < mat_heights && d < embedding_dim) {
        float sum_mq_hd = 0.0f;

        // Sum over tokens 'i' (C++ used 'j', using 'i' for clarity)
        for (int i = 0; i < token_count; ++i) {
            // Access grad_Q[i][h] (from token_count x mat_heights matrix)
            float grad_q_ih = grad_q[i * mat_heights + h];
            // Access Q[i][d] (from assumed token_count x embedding_dim matrix)
            float q_id = q[i * embedding_dim + d];
            sum_mq_hd += grad_q_ih * q_id;
        }

        // Store result using row-major indexing
        grad_mq[h * embedding_dim + d] = sum_mq_hd;
    }
}


/**
 * @brief CUDA kernel for Step 8 of `cuBackward(expectedV)`: Compute the correction term for the MK gradient.
 *        grad_MK_correction[h][d] = sum_i sum_j (-grad_MQ[h][d] * Q[j][h] * K[i][h])
 * @param[in] grad_mq Device pointer to the gradient w.r.t. MQ (row-major, mat_heights x embedding_dim).
 * @param[in] q Device pointer to the Query matrix Q (row-major, token_count x mat_heights).
 * @param[in] k Device pointer to the Key matrix K (row-major, token_count x mat_heights).
 * @param[out] grad_mk_correction Device pointer to store the MK correction term (row-major, mat_heights x embedding_dim).
 * @param[in] token_count The number of tokens.
 * @param[in] mat_heights The height dimension.
 * @param[in] embedding_dim The embedding dimension.
 * @note This kernel has O(token_count^2) complexity per thread, which can be very slow for large token counts. Consider optimization if performance is critical.
 */
__global__ void kernelComputeGradMKCorrection(const float* grad_mq, const float* q, const float* k,
                                              float* grad_mk_correction,
                                              int token_count, int mat_heights, int embedding_dim) {
    int h = blockIdx.y * blockDim.y + threadIdx.y; // Row index (mat_heights)
    int d = blockIdx.x * blockDim.x + threadIdx.x; // Column index (embedding_dim)

    if (h < mat_heights && d < embedding_dim) {
        float correction_sum_hd = 0.0f;
        float grad_mq_hd = grad_mq[h * embedding_dim + d]; // grad_MQ[h][d]

        // Double summation over tokens i and j
        for (int i = 0; i < token_count; ++i) {
            for (int j = 0; j < token_count; ++j) {
                // Access Q[j][h] and K[i][h] using row-major indexing
                float q_jh = q[j * mat_heights + h];
                float k_ih = k[i * mat_heights + h];
                correction_sum_hd -= grad_mq_hd * q_jh * k_ih; // Accumulate the negative term
            }
        }
        grad_mk_correction[h * embedding_dim + d] = correction_sum_hd;
    }
}

/**
 * @brief Naive CUDA kernel for row-wise sum reduction of a matrix.
 * @param[in] matrix Device pointer to the input matrix (row-major, rows x cols).
 * @param[out] sums Device pointer to store the row sums (size rows).
 * @param[in] rows The number of rows in the matrix.
 * @param[in] cols The number of columns in the matrix.
 * @note This is a basic implementation. For better performance on large matrices, consider using optimized reduction techniques (e.g., shared memory, CUB library).
 */
__global__ void kernelRowSum(const float* matrix, float* sums, int rows, int cols)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x; // Parallelize over rows

    if (row < rows) {
        float current_sum = 0.0f;
        // Sum elements across the columns for the current row
        for (int j = 0; j < cols; ++j) {
            current_sum += matrix[row * cols + j];
        }
        sums[row] = current_sum;
    }
}

/**
 * @brief CUDA kernel for Step 8 (Simplified): Compute gradients w.r.t. MK and MQ using embedding vectors.
 *        grad_MK[h][d] = sum_i (grad_K[i][h] * K_embed[i][d])
 *        grad_MQ[h][d] = sum_i (grad_Q[i][h] * Q_embed[i][d])
 * @param[in] grad_k Device pointer to the gradient w.r.t. K (row-major, token_count x mat_heights). Can be null.
 * @param[in] grad_q Device pointer to the gradient w.r.t. Q (row-major, token_count x mat_heights). Can be null.
 * @param[in] k_embed Device pointer to the Key embedding vectors (row-major, token_count x embedding_dim). Can be null.
 * @param[in] q_embed Device pointer to the Query embedding vectors (row-major, token_count x embedding_dim). Can be null.
 * @param[out] grad_mk Device pointer to store the gradient w.r.t. MK (row-major, mat_heights x embedding_dim).
 * @param[out] grad_mq Device pointer to store the gradient w.r.t. MQ (row-major, mat_heights x embedding_dim).
 * @param[in] token_count The number of tokens.
 * @param[in] mat_heights The height dimension.
 * @param[in] embedding_dim The embedding dimension.
 * @note This version correctly uses separate embedding vectors (k_embed, q_embed) for the calculation, resolving potential inconsistencies.
 */
__global__ void kernelComputeGradMK_MQ_Simplified(const float* grad_k, const float* grad_q,
        const float* k_embed, const float* q_embed, // Use embedding versions!
        float* grad_mk, float* grad_mq,
        int token_count, int mat_heights, int embedding_dim)
{
    int h = blockIdx.y * blockDim.y + threadIdx.y; // Row index for grad_mk/grad_mq (height h)
    int d = blockIdx.x * blockDim.x + threadIdx.x; // Column index for grad_mk/grad_mq (dimension d)

    if (h < mat_heights && d < embedding_dim) {
        float sum_mk_hd = 0.0f;
        float sum_mq_hd = 0.0f;

        // Sum over tokens 'i'
        for (int i = 0; i < token_count; ++i) {
            // Check if inputs are valid before accessing
            if (grad_k != nullptr && k_embed != nullptr) {
                // Access grad_K[i][h] and K_embed[i][d]
                sum_mk_hd += grad_k[i * mat_heights + h] * k_embed[i * embedding_dim + d];
            }
            if (grad_q != nullptr && q_embed != nullptr) {
                // Access grad_Q[i][h] and Q_embed[i][d]
                sum_mq_hd += grad_q[i * mat_heights + h] * q_embed[i * embedding_dim + d];
            }
        }
        // Store results using row-major indexing
        grad_mk[h * embedding_dim + d] = sum_mk_hd;
        grad_mq[h * embedding_dim + d] = sum_mq_hd;
    }
}

/**
 * @brief CUDA kernel to update the EV matrix by broadcasting a scaled gradient vector.
 *        EV[row][col] -= learning_rate * grad_EV_scaled[col]
 * @param[in,out] d_EV Device pointer to the EV matrix (row-major, context_win x embedding_dim). Updated in place.
 * @param[in] d_grad_EV_scaled Device pointer to the scaled gradient vector for EV (size embedding_dim).
 * @param[in] learning_rate The learning rate for the gradient descent update.
 * @param[in] context_win The number of rows in the EV matrix.
 * @param[in] embedding_dim The number of columns in the EV matrix (and size of d_grad_EV_scaled).
 */
__global__ void kernelUpdateEVBroadcasted(float* d_EV, const float* d_grad_EV_scaled, float learning_rate,
                                          int context_win, int embedding_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Global thread ID, iterates over all elements of EV
    int total_elements = context_win * embedding_dim;

    if (idx < total_elements) {
        int embed_idx = idx % embedding_dim; // Column index
        d_EV[idx] -= learning_rate * d_grad_EV_scaled[embed_idx];
    }
}

#endif