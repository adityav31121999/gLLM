// Forward declarations to prevent implicit declaration warnings/errors
inline float compute_dot_product(__global const float* vec1, __global const float* vec2, int dim);
inline float compute_dot_product_mat(__global const float* vec1, __global const float* vec2, __global const float* matrix, int dim);
int compute_prediction(__global const float* EH, __global const float* embeddings, int dim, int voc);

inline float compute_dot_product(__global const float* vec1, __global const float* vec2, int dim) 
{
    float dot_product = 0.0f;
    for (int k = 0; k < dim; ++k) {
        dot_product += vec1[k] * vec2[k];
    }
    return dot_product;
}

inline float compute_dot_product_mat(__global const float* vec1, 
    __global const float* vec2, __global const float* matrix,
    int dim)
{
    float final_dot_product = 0.0f;
    for (int i = 0; i < dim; ++i) {
        // inner_sum = vec1 . matrix_row_i
        float inner_sum = 0.0f;
        // ith row of matrix
        __global const float* matrix_row_i = matrix + i * dim;

        for (int j = 0; j < dim; ++j) {
            // dot product of vec1 with ith row of matrix
            inner_sum += vec1[j] * matrix_row_i[j];
        }

        // (vec1 . matrix_row_i) * vec2[i]
        final_dot_product += inner_sum * vec2[i];
    }
    return final_dot_product;
}

int compute_prediction(__global const float* EH, __global const float* embeddings, int dim, int voc) 
{
    // for empty embeddings
    if (voc <= 0) {
        return -1;
    }
    // Initialize with the smallest possible float value
    float max_dot_product = -FLT_MAX;
    int predicted_index = 0;
    for (int i = 0; i < voc; ++i) {
        // pointer to ith token embedding row
        __global const float* current_embedding_row = embeddings + i * dim;
        // Use the correctly named inline function
        float current_dot_product = compute_dot_product(EH, current_embedding_row, dim);
        // update index if new maximum dot product is available
        if (current_dot_product > max_dot_product) {
            max_dot_product = current_dot_product;
            predicted_index = i;
        }
    }
    return predicted_index;
}

/**
 * @brief Computes the gradient for a weight matrix in a linear layer (dL/dW).
 *        Assumes the input `h_prev` is a vector and `dL_d_logits` is a vector.
 *        The gradient is calculated as an outer product: dL/dW = dL/du * (h_prev)^T.
 * @param dL_d_logits       Global pointer to the error signal for the output (dL/du). (vocab_size x 1)
 * @param h_prev            Global pointer to the activations from the previous layer (input to this layer). (hidden_dim x 1)
 * @param grad_weights      Global pointer to the gradient matrix (dL/dW). (vocab_size x hidden_dim)
 *                          This buffer should be zeroed before accumulation over a batch/sample if needed.
 * @param vocab_size        Number of rows in grad_weights (e.g., vocab_size for deEmbeddings).
 * @param hidden_dim        Number of columns in grad_weights (e.g., embedding_dim for deEmbeddings).
 */
__kernel void clComputeGradientDeEmbeddings(
    __global const float* dL_d_logits,
    __global const float* h_prev,
    __global float* grad_weights,
    const int vocab_size,
    const int hidden_dim
) {
    int global_idx = get_global_id(0);

    if (global_idx < vocab_size * hidden_dim) {
        int row_idx = global_idx / hidden_dim;
        int col_idx = global_idx % hidden_dim;

        grad_weights[global_idx] = dL_d_logits[row_idx] * h_prev[col_idx];
    }
}


/**
 * @brief Calculates the error signal (dL/du) for logits, assuming Categorical Cross-Entropy loss
 *        and a softmax-like activation (like LOTA). The derivative simplifies to P_hat - y.
 * @param predicted_probs Global pointer to the predicted probabilities (P_hat from LOTA).
 * @param oneHotEncoding Global pointer to the one hot encoded vector for 1 at predicted index and 0 at rest of the places.
 * @param deltas          Global pointer to store the computed error signal (dL/du).
 * @param vocab_size      Total number of vocabulary tokens.
 */
__kernel void clCalculateOutputDeltaLOTA(
    __global const float* predicted_probs,
    __global const float* oneHotEncoding,
    __global float* deltas,
    const int vocab_size
) {
    int idx = get_global_id(0);

    if (idx < vocab_size) {
        deltas[idx] = predicted_probs[idx] - oneHotEncoding[idx];
    }
}

/**
 * @brief Propagates the error signal back to the previous layer's activations (dL/dh_prev).
 *        Calculates: dL/dh_prev = W^T * dL/du.
 * @param weights           Global pointer to the weight matrix (W). (vocab_size x hidden_dim)
 * @param dL_d_logits       Global pointer to the error signal for the current layer's output (dL/du). (vocab_size x 1)
 * @param dL_d_h_prev       Global pointer to store the propagated error for the previous layer (dL/dh_prev). (hidden_dim x 1)
 * @param vocab_size        Number of rows in weights.
 * @param hidden_dim        Number of columns in weights.
 */
__kernel void clPropagateErrorToHidden(
    __global const float* weights,
    __global const float* dL_d_logits,
    __global float* dL_d_h_prev,
    const int vocab_size,
    const int hidden_dim
) {
    int h_idx = get_global_id(0);

    if (h_idx < hidden_dim) {
        float sum_error = 0.0f;
        for (int v_idx = 0; v_idx < vocab_size; ++v_idx) {
            sum_error += weights[v_idx * hidden_dim + h_idx] * dL_d_logits[v_idx];
        }
        dL_d_h_prev[h_idx] = sum_error;
    }
}

/**
 * @brief Computes the gradient for the input TokenEmbed matrix for a single attention head.
 *        Calculates: dL/dX_block = dL/dQ @ W_Q^T + dL/dK @ W_K^T + dL/dV @ W_V^T + dL/dV @ W_V^T
 *        This kernel expects to be launched with (token_count x embedding_dim) work-items.
 */
__kernel void clComputeGradTokenEmbedForHead(
    __global const float* d_grad_K,     // Input: dL/dK (token_count x embedding_dim)
    __global const float* d_grad_Q,     // Input: dL/dQ (token_count x embedding_dim)
    __global const float* d_grad_H,     // Input: dL/dQ (token_count x embedding_dim)
    __global const float* d_grad_V,     // Input: dL/dV (token_count x embedding_dim)
    __global const float* d_MK_weights, // Input: MK (mat_heights x embedding_dim)
    __global const float* d_MQ_weights, // Input: MQ (mat_heights x embedding_dim)
    __global const float* d_MH_weights, // Input: MH (mat_heights x embedding_dim)
    __global const float* d_MV_weights, // Input: MV (mat_heights x embedding_dim)
    __global float* d_grad_token,       // Output: dL/dX_block (token_count x mat_heights)
    int token_count,                    // N_seq
    int mat_heights,                    // D_features
    int embedding_dim                   // D_head
) {
}

// Helper to sum multiple matrices element-wise into one (e.g., summing per-head dL/dTokenEmbed)
// Needs to be launched with (total_elements) work items.
// Assumes output_summed is zeroed initially.
__kernel void clSumMatricesElementwise(
    __global float* output_summed,     // Accumulator
    __global const float* input_matrix, // Matrix to add
    const int num_elements
) {
    int idx = get_global_id(0);
    if (idx < num_elements) {
        // This is safe if work-items for THIS kernel are unique.
        // It's for summing up results from parallel work_groups/heads.
        output_summed[idx] += input_matrix[idx];
    }
}

/**
 * @brief Accumulates gradients from the sequence's embeddings (TokenEmbed)
 *        back into the global vocabulary embedding gradient table.
 *        This kernel expects to be launched with `num_tokens_in_context` work-items
 *        (each work-item handles one token's embedding gradient) and `embedding_dim`
 *        secondary dimension for 2D launch.
 * @param d_gEmbeddings_global   Global pointer to the **global** embedding gradient matrix 
 *        (vocab_size x embedding_dim). This buffer must be zeroed before starting a batch's accumulation.
 * @param d_dL_dTokenEmbed_seq   Global pointer to the gradients for the current sequence's embeddings.
 *                               (num_tokens_in_context x embedding_dim, row-major).
 * @param d_vocab_indices_seq    Global pointer to the vocabulary index for each token in the current sequence.
 *                               (num_tokens_in_context x 1).
 * @param num_tokens_in_context  Number of tokens in the current sequence (rows in d_dL_dTokenEmbed_seq).
 * @param embedding_dim          Dimension of each embedding vector (columns in d_dL_dTokenEmbed_seq and d_gEmbeddings_global).
 * @param vocab_size             Total size of the vocabulary (rows in d_gEmbeddings_global).
 */
__kernel void clAccumulateEmbeddingGradients(
    __global float* d_gEmbeddings_global,
    __global const float* d_dL_dTokenEmbed_seq,
    __global const int* d_vocab_indices_seq, // Indices of tokens in the sequence
    const int num_tokens_in_context,
    const int embedding_dim,
    const int vocab_size
) {
    // Each work-item handles one element of the dL/dTokenEmbed_seq matrix.
    // Map global_id(0) to column (embedding dimension index)
    // Map global_id(1) to row (token position in sequence)
    int embed_dim_idx = get_global_id(0);   // Corresponds to 'j' in (dL/dE)_ij
    int token_pos_idx = get_global_id(1);   // Corresponds to 'i' in (dL/dE)_ij

    if (embed_dim_idx < embedding_dim && token_pos_idx < num_tokens_in_context) {
        // 1. Get the gradient value for this specific embedding dimension and token position.
        // d_dL_dTokenEmbed_seq is (num_tokens_in_context x embedding_dim) row-major.
        float grad_value_at_pos = d_dL_dTokenEmbed_seq[token_pos_idx * embedding_dim + embed_dim_idx];

        // 2. Get the global vocabulary index for the token at this position.
        int vocab_idx = d_vocab_indices_seq[token_pos_idx];

        // 3. Accumulate this gradient into the global embedding gradient matrix.
        // d_gEmbeddings_global is (vocab_size x embedding_dim) row-major.
        if (vocab_idx >= 0 && vocab_idx < vocab_size) { // Ensure valid vocabulary index
            int global_grad_idx = vocab_idx * embedding_dim + embed_dim_idx;
            // Use atomic add to prevent race conditions when multiple tokens
            // in the sequence (or across different samples in a batch) correspond
            // to the same vocabulary word.
            atomic_add_float(&d_gEmbeddings_global[global_grad_idx], grad_value_at_pos);
        }
    }
}

/**------------------------------------TRAINING------------------------------------**/

__kernel void kernelKdotQforSelf_train_transformer(__global float* d_kdotq, __global const float* d_keys, 
            __global const float* d_querys, int num_queries_eff, int num_keys_eff, int kdotq_width, 
            int embedding_dim, float inv_scaling)
{
    // Calculate the global row (query index i) and column (key index j) for this work-item
    int j = get_global_id(0); // Key index (column)
    int i = get_global_id(1); // Query index (row)

    // Boundary check AND self-attention causal mask (j <= i)
    if (i < num_queries_eff && j < num_keys_eff && j <= i) {
        // Pointers to the start of the i-th query vector and j-th key vector
        __global const float* q_vec = d_querys + i * embedding_dim;
        __global const float* k_vec = d_keys + j * embedding_dim;
        // Use the correctly named inline function
        float dot_product = compute_dot_product(q_vec, k_vec, embedding_dim);

        // Calculate the index in the flattened d_kdotq array (row-major)
        int kdotq_index = i * kdotq_width + j;
        d_kdotq[kdotq_index] = dot_product * inv_scaling;
    }
}

__kernel void kernelKdotQforCross_train_transformer(__global float* d_kdotq, __global const float* d_keys, 
            __global const float* d_querys, int num_queries_eff, int num_keys_eff, int kdotq_width, 
            int embedding_dim, float inv_scaling)
{
    // Calculate the global column (key index j) and row (query index i) for this work-item
    int j = get_global_id(0); // Key index (column)
    int i = get_global_id(1); // Query index (row)

    // Boundary check (no causal mask for cross-attention)
    if (i < num_queries_eff && j < num_keys_eff) {
        // Pointers to the start of the i-th query vector and j-th key vector
        __global const float* q_vec = d_querys + i * embedding_dim;
        __global const float* k_vec = d_keys + j * embedding_dim;
        // Use the correctly named inline function
        float dot_product = compute_dot_product(q_vec, k_vec, embedding_dim);
        // Calculate the index in the flattened d_kdotq array (row-major)
        int kdotq_index = i * kdotq_width + j;
        d_kdotq[kdotq_index] = dot_product * inv_scaling;
    }
}

/**------------------------------------INFERENCE------------------------------------**/

__kernel void kernelKdotQBlock1Self_Inference(__global float* d_kdotq, __global const float* d_tokenEmbed, 
            __global const float* d_M, int prompt_start_index, int prompt_len, int context_len, 
            int kdotq_width,int embedding_dim, float inv_scaling)
{
    // Calculate the global key index (j) and the offset for the query index (i_offset)
    int j = get_global_id(0);
    int i_offset = get_global_id(1);
    int i = prompt_start_index + i_offset;
    // Boundary checks for self masking
    if (i_offset < prompt_len && j < context_len && j <= i) {
        // Pointers to the start of the i-th query vector and j-th key vector
        __global const float* q_vec = d_tokenEmbed + i * embedding_dim;
        __global const float* k_vec = d_tokenEmbed + j * embedding_dim;
        // Use the correctly named inline function
        float dot_product = compute_dot_product_mat(q_vec, k_vec, d_M, embedding_dim);
        // index in flatten array
        int kdotq_index = i * kdotq_width + j;
        d_kdotq[kdotq_index] = dot_product * inv_scaling;
    }
}

__kernel void kernelKdotQBlock1Cross_Inference(__global float* d_kdotq, __global const float* d_tokenEmbed, 
            __global const float* d_M, int prompt_start_index, int prompt_len, int context_len, 
            int kdotq_width, int embedding_dim, float inv_scaling)
{
    // Calculate the global key index (j) and the offset for the query index (i_offset)
    int j = get_global_id(0);
    int i_offset = get_global_id(1);
    int i = prompt_start_index + i_offset;
    if (i_offset < prompt_len && j < context_len) {
        // Pointers to the start of the i-th query vector and j-th key vector
        __global const float* q_vec = d_tokenEmbed + i * embedding_dim;
        __global const float* k_vec = d_tokenEmbed + j * embedding_dim;
        // Use the correctly named inline function
        float dot_product = compute_dot_product_mat(q_vec, k_vec, d_M, embedding_dim);
        // Calculate the index in the flattened d_kdotq array (row-major)
        int kdotq_index = i * kdotq_width + j;
        d_kdotq[kdotq_index] = dot_product * inv_scaling;
    }
}

__kernel void kernelKdotQBlockNSelf_Inference(__global float* d_kdotq, __global const float* d_tokForBlock, 
            __global const float* d_EVp, __global const float* d_M, int prompt_start_index_in_block, 
            int prompt_len, int context_len_in_block, int kdotq_width, int embedding_dim, float inv_scaling)
{
    // Calculate the key index (j) and query index offset (i_offset) *within the block's window*
    int j = get_global_id(0);
    int i_offset = get_global_id(1);
    int i = prompt_start_index_in_block + i_offset;
    // self masking and boundary checks
    if (i_offset < prompt_len && j < context_len_in_block && j <= i) {
        // Pointers to the start of the i-th query vector (from tokForBlock) and j-th key vector (from EVp)
        __global const float* q_vec = d_tokForBlock + i * embedding_dim;
        __global const float* k_vec = d_EVp + j * embedding_dim;
        // Use the correctly named inline function
        float dot_product = compute_dot_product_mat(q_vec, k_vec, d_M, embedding_dim);
        // Calculate the index in the flattened d_kdotq array (row-major, relative to block window)
        int kdotq_index = i * kdotq_width + j;
        d_kdotq[kdotq_index] = dot_product * inv_scaling;
    }
}

__kernel void kernelKdotQBlockNCross_Inference(__global float* d_kdotq, __global const float* d_tokForBlock,
            __global const float* d_EVp, __global const float* d_M, int prompt_start_index_in_block, 
            int prompt_len, int context_len_in_block, int kdotq_width, int embedding_dim, float inv_scaling)
{
    // Calculate the key index (j) and query index offset (i_offset) *within the block's window*
    int j = get_global_id(0);
    int i_offset = get_global_id(1);
    int i = prompt_start_index_in_block + i_offset;
    if (i_offset < prompt_len && j < context_len_in_block) {
        // Pointers to the start of the i-th query vector (from tokForBlock) and j-th key vector (from EVp)
        __global const float* q_vec = d_tokForBlock + i * embedding_dim;
        __global const float* k_vec = d_EVp + j * embedding_dim;
        // Use the correctly named inline function
        float dot_product = compute_dot_product_mat(q_vec, k_vec, d_M, embedding_dim);
        // Calculate the index in the flattened d_kdotq array (row-major, relative to block window)
        int kdotq_index = i * kdotq_width + j;
        d_kdotq[kdotq_index] = dot_product * inv_scaling;
    }
}
