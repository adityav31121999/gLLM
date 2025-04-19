
// compute kernels and functions
#include "include/attention.hpp"    // EMBEDDING, SCALING and CONTEXT_WIN, etc.
#include "include/block.hpp"
#include "include/transformer.hpp"
#include <maths.hpp>
#include <cmath>

#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>


// Define thread block dimensions (tune these based on your GPU architecture)
#define THREADS_PER_BLOCK_X 16      // or 32
#define THREADS_PER_BLOCK_Y 16      // or 32

/**
 * @brief compute key or query vector
 * @param[in] tokenEmbed embedding of token for which key or query vector need to be calculated
 * @param[in] matrix key or query matrix
 * @param[out] KorQ Key or Query vector
 * @param[in] dim embedding dimension
 * @param[in] height key and query matrix rows
 */
__device__ void cuComputeKorQ(const float* tokenEmbed, const float* matrix, float* KorQ, int dim, int height) {
    for (int i = 0; i < height; ++i) {
        // ith row of matrix
        const float* matrix_row_i = matrix + i * dim;
        for (int j = 0; j < dim; ++j) {
            // dot product of vec1 with ith row of matrix
            KorQ[i] += tokenEmbed[j] * matrix_row_i[j];
        }
    }
}

/**
 * @brief Computes the dot product of two vectors residing in global memory.
 * @param vec1 Pointer to the first vector.
 * @param vec2 Pointer to the second vector.
 * @param dim The dimension of the vectors.
 * @return The dot product of vec1 and vec2.
 */
__device__ __forceinline__ float compute_dot_product(const float* vec1, const float* vec2, int dim) {
    float dot_product = 0.0f;
    for (int k = 0; k < dim; ++k) {
        dot_product += vec1[k] * vec2[k];
    }
    return dot_product;
}

/**
 * @brief Computes the dot product of vec1 * matrix * vec2.
 * @param vec1 Pointer to the first vector (row vector, size dim).
 * @param vec2 Pointer to the second vector (column vector, size dim).
 * @param matrix Pointer to the matrix (row-major, dim x dim).
 * @param dim The dimension of the vectors and the matrix.
 * @return The scalar result of vec1 * matrix * vec2.
 */
__device__ __forceinline__ float compute_dot_product(const float* vec1, const float* vec2, const float* matrix, int dim)
{
    float final_dot_product = 0.0f;
    for (int i = 0; i < dim; ++i) {
        // inner_sum = vec1.matrix[i]
        float inner_sum = 0.0f;
        // ith row of matrix
        const float* matrix_row_i = matrix + i * dim;

        for (int j = 0; j < dim; ++j) {
            // dot product of vec1 with ith row of matrix
            inner_sum += vec1[j] * matrix_row_i[j];
        }

        // vec1 x matrix[i] x vec2
        final_dot_product += inner_sum * vec2[i];
    }
    return final_dot_product;
}

/**
 * @brief compute prediction by inner products of EH and the rows of embeddings
 * @param EH horizontal retention vector (output from the last block)
 * @param embeddings token embeddings matrix (row-major: voc x dim)
 * @param dim embedding dimension (size of EH and columns of embeddings)
 * @param voc vocabulary size (rows of embeddings)
 * @return The index of the token embedding with the highest dot product (predicted token index).
 * @note it is assumed in this function that the case of "all dot products being zero" will
 *      not occur
 */
__device__ int compute_prediction(const float* EH, const float* embeddings, int dim, int voc) {
    // for empty vocabulary / embeddings
    if (voc <= 0 || embeddings == nullptr) {
        return -1;
    }
    // Initialize with the smallest possible float value
    float max_dot_product = -FLT_MAX;
    int predicted_index = 0;
    for (int i = 0; i < voc; ++i) {
        // pointer to ith token embedding row
        const float* current_embedding_row = embeddings + i * dim;
        float current_dot_product = compute_dot_product(EH, current_embedding_row, dim);
        // update index if new maximum dot product is available
        if (current_dot_product > max_dot_product) {
            max_dot_product = current_dot_product;
            predicted_index = i;
        }
    }
    return predicted_index;
}

/**------------------------------------TRAINING------------------------------------**/

/**
 * @brief kernel for calculating KdotQ using keys and queries for self attention (Q[i] dot K[j] where j <= i)
 * @param[out] d_kdotq         Device pointer to hold dot products (KdotQ matrix, row-major)
 * @param[in]  d_keys          Device pointer to keys (K matrix, row-major)
 * @param[in]  d_querys        Device pointer to queries (Q matrix, row-major)
 * @param[in]  num_queries_eff Number of query rows (i) to compute for in this launch.
 * @param[in]  num_keys_eff    Number of key columns (j) to compute for in this launch.
 * @param[in]  kdotq_width     Total width (number of columns) of the d_kdotq buffer for indexing.
 * @param[in]  embedding_dim   Dimension of each key/query vector.
 * @param[in]  inv_scaling     Value of 1.0f / SCALING.
*/
__global__ void kernelKdotQforSelf_train(float* d_kdotq, const float* d_keys, const float* d_querys, int num_queries_eff, 
    int num_keys_eff, int kdotq_width, int embedding_dim, float inv_scaling)
{
    // Calculate the global row (query index i) and column (key index j) for this thread
    int j = blockIdx.x * blockDim.x + threadIdx.x; // Key index (column)
    int i = blockIdx.y * blockDim.y + threadIdx.y; // Query index (row)

    // Boundary check AND self-attention causal mask (j <= i)
    if (i < num_queries_eff && j < num_keys_eff && j <= i) {
        // Pointers to the start of the i-th query vector and j-th key vector
        const float* q_vec = d_querys + i * embedding_dim;
        const float* k_vec = d_keys + j * embedding_dim;

        // Compute dot product using the device function
        float dot_product = compute_dot_product(q_vec, k_vec, embedding_dim);

        // Calculate the index in the flattened d_kdotq array (row-major)
        int kdotq_index = i * kdotq_width + j;

        // Store the scaled result
        d_kdotq[kdotq_index] = dot_product * inv_scaling;
    }
}


/**
 * @brief kernel for calculating KdotQ using keys and queries for cross attention (Q[i] dot K[j] for all i, j)
 * @param[out] d_kdotq         Device pointer to hold dot products (KdotQ matrix, row-major)
 * @param[in]  d_keys          Device pointer to keys (K matrix, row-major)
 * @param[in]  d_querys        Device pointer to queries (Q matrix, row-major)
 * @param[in]  num_queries_eff Number of query rows (i) to compute for in this launch.
 * @param[in]  num_keys_eff    Number of key columns (j) to compute for in this launch.
 * @param[in]  kdotq_width     Total width (number of columns) of the d_kdotq buffer for indexing.
 * @param[in]  embedding_dim   Dimension of each key/query vector.
 * @param[in]  inv_scaling     Value of 1.0f / SCALING.
 */
__global__ void kernelKdotQforCross_train(float* d_kdotq, const float* d_keys, const float* d_querys, int num_queries_eff,
    int num_keys_eff, int kdotq_width, int embedding_dim, float inv_scaling)
{
    // Calculate the global column (key index j) and row (query index i) for this thread
    int j = blockIdx.x * blockDim.x + threadIdx.x; // Key index (column)
    int i = blockIdx.y * blockDim.y + threadIdx.y; // Query index (row) // <<< CORRECTED LINE

    // Boundary check (no causal mask for cross-attention)
    if (i < num_queries_eff && j < num_keys_eff) {

        // Pointers to the start of the i-th query vector and j-th key vector
        const float* q_vec = d_querys + i * embedding_dim;
        const float* k_vec = d_keys + j * embedding_dim;

        // Compute dot product using the device function
        float dot_product = compute_dot_product(q_vec, k_vec, embedding_dim);

        // Calculate the index in the flattened d_kdotq array (row-major)
        int kdotq_index = i * kdotq_width + j;

        // Store the scaled result
        d_kdotq[kdotq_index] = dot_product * inv_scaling;
    }
}

/**------------------------------------INFERENCE------------------------------------**/

/**
 * @brief INFERENCE kernel for Block 1 SELF-ATTENTION KdotQ.
 * Computes KdotQ[i][j] = (tokenEmbed[i] * M * tokenEmbed[j]) / SCALING for j <= i,
 * where i and j are the index of attention score of KdotQ grid.
 * @param[out] d_kdotq         Device pointer to output KdotQ (potentially sparse, only computes needed values).
 * @param[in]  d_tokenEmbed    Device pointer to token embeddings for the *entire* context (row-major).
 * @param[in]  d_M             Device pointer to the QK' matrix M (row-major, dim x dim).
 * @param[in]  prompt_start_index Index of the first token of the new prompt in d_tokenEmbed.
 * @param[in]  prompt_len      Number of tokens in the new prompt.
 * @param[in]  context_len     Total number of tokens in the context (prompt_start_index + prompt_len).
 * @param[in]  kdotq_width     Total width (number of columns) of the d_kdotq buffer for indexing (usually context_len).
 * @param[in]  embedding_dim   Dimension of each token vector and the matrix M.
 * @param[in]  inv_scaling     Value of 1.0f / SCALING.
 */
__global__ void kernelKdotQ_Block1_Self_Inference(float* d_kdotq, const float* d_tokenEmbed, const float* d_M, 
    int prompt_start_index, int prompt_len, int context_len, int kdotq_width, int embedding_dim, float inv_scaling)
{
    // Calculate the global key index (j) and the offset for the query index (i_offset)
    int j = blockIdx.x * blockDim.x + threadIdx.x; // Key index (column)
    // this is same condition for cpp with promptCount and currentTokenCount
    int i_offset = blockIdx.y * blockDim.y + threadIdx.y; // Query index offset within the prompt
    int i = prompt_start_index + i_offset;

    // forms lower triangle with all elements of diagonal and elements left to diagonal
    if (i_offset < prompt_len && j < context_len && j <= i) {

        // Pointers to the start of the i-th query vector and j-th key vector
        const float* q_vec = d_tokenEmbed + i * embedding_dim;
        const float* k_vec = d_tokenEmbed + j * embedding_dim;

        // Compute dot product: q_vec * M * k_vec
        float dot_product = compute_dot_product(q_vec, k_vec, d_M, embedding_dim);

        // Calculate the index in the flattened d_kdotq array (row-major)
        // We are computing rows corresponding to the new prompt (i)
        // against columns corresponding to the full context (j)
        int kdotq_index = i * kdotq_width + j;

        // Store the scaled result
        d_kdotq[kdotq_index] = dot_product * inv_scaling;
    }
}
  

/**
 * @brief INFERENCE kernel for Block 1 CROSS-ATTENTION KdotQ.
 * Computes KdotQ[i][j] = (tokenEmbed[i] * M * tokenEmbed[j]) / SCALING for all j,
 * where i corresponds to the new prompt tokens and j spans the entire context.
 * @param[out] d_kdotq         Device pointer to output KdotQ (potentially sparse).
 * @param[in]  d_tokenEmbed    Device pointer to token embeddings for the *entire* context (row-major).
 * @param[in]  d_M             Device pointer to the QK' matrix M (row-major, dim x dim).
 * @param[in]  prompt_start_index Index of the first token of the new prompt in d_tokenEmbed.
 * @param[in]  prompt_len      Number of tokens in the new prompt.
 * @param[in]  context_len     Total number of tokens in the context (prompt_start_index + prompt_len).
 * @param[in]  kdotq_width     Total width (number of columns) of the d_kdotq buffer for indexing (usually context_len).
 * @param[in]  embedding_dim   Dimension of each token vector and the matrix M.
 * @param[in]  inv_scaling     Value of 1.0f / SCALING.
 */
__global__ void kernelKdotQ_Block1_Cross_Inference(float* d_kdotq, const float* d_tokenEmbed, const float* d_M,
    int prompt_start_index, int prompt_len, int context_len, int kdotq_width, int embedding_dim, float inv_scaling)
{
    // Calculate the global key index (j) and the offset for the query index (i_offset)
    int j = blockIdx.x * blockDim.x + threadIdx.x; // Key index (column)
    int i_offset = blockIdx.y * blockDim.y + threadIdx.y; // Query index offset within the prompt

    // Calculate the absolute query index (i)
    int i = prompt_start_index + i_offset;

    // Boundary checks:
    // 1. Query index 'i' must be within the new prompt range.
    // 2. Key index 'j' must be within the total context.
    // NO causal mask for cross-attention.
    if (i_offset < prompt_len && j < context_len) {

        // Pointers to the start of the i-th query vector and j-th key vector
        const float* q_vec = d_tokenEmbed + i * embedding_dim;
        const float* k_vec = d_tokenEmbed + j * embedding_dim;

        // Compute dot product: q_vec * M * k_vec
        float dot_product = compute_dot_product(q_vec, k_vec, d_M, embedding_dim);

        // Calculate the index in the flattened d_kdotq array (row-major)
        int kdotq_index = i * kdotq_width + j;

        // Store the scaled result
        d_kdotq[kdotq_index] = dot_product * inv_scaling;
    }
}


/**
 * @brief INFERENCE kernel for Block N (N > 1) SELF-ATTENTION KdotQ.
 * Computes KdotQ[i][j] = (tokForBlock[i] * M * EVp[j]) / SCALING for j <= i,
 * where i corresponds to the new prompt tokens within the block's window,
 * and j spans the relevant context within the block's window.
 * @param[out] d_kdotq         Device pointer to output KdotQ for this block's window.
 * @param[in]  d_tokForBlock   Device pointer to token embeddings for this block's context window (row-major).
 * @param[in]  d_EVp           Device pointer to previous block's output vectors (EV') for this block's context window (row-major).
 * @param[in]  d_M             Device pointer to the QK' matrix M (row-major, dim x dim).
 * @param[in]  prompt_start_index_in_block Index of the first token of the new prompt *within the block's window*.
 * @param[in]  prompt_len      Number of tokens in the new prompt.
 * @param[in]  context_len_in_block Total number of relevant tokens in this block's window.
 * @param[in]  kdotq_width     Total width (number of columns) of the d_kdotq buffer for indexing (usually context_len_in_block).
 * @param[in]  embedding_dim   Dimension of each token vector and the matrix M.
 * @param[in]  inv_scaling     Value of 1.0f / SCALING.
 */
__global__ void kernelKdotQ_BlockN_Self_Inference(float* d_kdotq, const float* d_tokForBlock, const float* d_EVp, 
    const float* d_M, int prompt_start_index_in_block, int prompt_len, int context_len_in_block, int kdotq_width, 
    int embedding_dim, float inv_scaling)
{
    // Calculate the key index (j) and query index offset (i_offset) *within the block's window*
    int j = blockIdx.x * blockDim.x + threadIdx.x; // Key index (column) in block window
    int i_offset = blockIdx.y * blockDim.y + threadIdx.y; // Query index offset within the prompt

    // Calculate the query index (i) *within the block's window*
    int i = prompt_start_index_in_block + i_offset;

    // Boundary checks:
    // 1. Query index 'i' must be within the new prompt range within the block's window.
    // 2. Key index 'j' must be within the relevant context length of the block's window.
    // 3. Self-attention causal mask: j <= i (indices relative to block window).
    if (i_offset < prompt_len && j < context_len_in_block && j <= i) {

        // Pointers to the start of the i-th query vector (from tokForBlock) and j-th key vector (from EVp)
        const float* q_vec = d_tokForBlock + i * embedding_dim;
        const float* k_vec = d_EVp + j * embedding_dim;

        // Compute dot product: q_vec * M * k_vec
        float dot_product = compute_dot_product(q_vec, k_vec, d_M, embedding_dim);

        // Calculate the index in the flattened d_kdotq array (row-major, relative to block window)
        int kdotq_index = i * kdotq_width + j;

        // Store the scaled result
        d_kdotq[kdotq_index] = dot_product * inv_scaling;
    }
}


/**
 * @brief INFERENCE kernel for Block N (N > 1) CROSS-ATTENTION KdotQ.
 * Computes KdotQ[i][j] = (tokForBlock[i] * M * EVp[j]) / SCALING for all j,
 * where i corresponds to the new prompt tokens within the block's window,
 * and j spans the relevant context within the block's window.
 * @param[out] d_kdotq         Device pointer to output KdotQ for this block's window.
 * @param[in]  d_tokForBlock   Device pointer to token embeddings for this block's context window (row-major).
 * @param[in]  d_EVp           Device pointer to previous block's output vectors (EV') for this block's context window (row-major).
 * @param[in]  d_M             Device pointer to the QK' matrix M (row-major, dim x dim).
 * @param[in]  prompt_start_index_in_block Index of the first token of the new prompt *within the block's window*.
 * @param[in]  prompt_len      Number of tokens in the new prompt.
 * @param[in]  context_len_in_block Total number of relevant tokens in this block's window.
 * @param[in]  kdotq_width     Total width (number of columns) of the d_kdotq buffer for indexing (usually context_len_in_block).
 * @param[in]  embedding_dim   Dimension of each token vector and the matrix M.
 * @param[in]  inv_scaling     Value of 1.0f / SCALING.
 */
__global__ void kernelKdotQ_BlockN_Cross_Inference(float* d_kdotq, const float* d_tokForBlock, const float* d_EVp, 
    const float* d_M, int prompt_start_index_in_block, int prompt_len, int context_len_in_block, int kdotq_width, 
    int embedding_dim, float inv_scaling)
{
    // Calculate the key index (j) and query index offset (i_offset) *within the block's window*
    int j = blockIdx.x * blockDim.x + threadIdx.x; // Key index (column) in block window
    int i_offset = blockIdx.y * blockDim.y + threadIdx.y; // Query index offset within the prompt

    // Calculate the query index (i) *within the block's window*
    int i = prompt_start_index_in_block + i_offset;

    // Boundary checks:
    // 1. Query index 'i' must be within the new prompt range within the block's window.
    // 2. Key index 'j' must be within the relevant context length of the block's window.
    // NO causal mask for cross-attention.
    if (i_offset < prompt_len && j < context_len_in_block) {

        // Pointers to the start of the i-th query vector (from tokForBlock) and j-th key vector (from EVp)
        const float* q_vec = d_tokForBlock + i * embedding_dim;
        const float* k_vec = d_EVp + j * embedding_dim;

        // Compute dot product: q_vec * M * k_vec
        float dot_product = compute_dot_product(q_vec, k_vec, d_M, embedding_dim);

        // Calculate the index in the flattened d_kdotq array (row-major, relative to block window)
        int kdotq_index = i * kdotq_width + j;

        // Store the scaled result
        d_kdotq[kdotq_index] = dot_product * inv_scaling;
    }
}

// --- Helper Functions ---

/**
 * @brief Flattens a 2D vector into a 1D vector (row-major).
 * @param vec2d The input 2D vector.
 * @return A 1D vector containing the flattened data. Returns empty if input is empty.
 */
std::vector<float> flatten(const std::vector<std::vector<float>>& vec2d) {
    if (vec2d.empty() || vec2d[0].empty()) {
        return {};
    }
    size_t rows = vec2d.size();
    size_t cols = vec2d[0].size();
    std::vector<float> flat_vec;
    flat_vec.reserve(rows * cols); // Reserve space for efficiency
    for (size_t i = 0; i < rows; ++i) {
        if (vec2d[i].size() != cols) {
             fprintf(stderr, "Warning: Inconsistent column count in flatten (%zu vs %zu). Using first row's column count.\n", vec2d[i].size(), cols);
             // Optionally throw an error or handle differently
        }
        flat_vec.insert(flat_vec.end(), vec2d[i].begin(), vec2d[i].end());
    }
    return flat_vec;
}

/**
 * @brief Flattens a mat object into a 1D vector (row-major).
 * @param matrix The input mat object. Assumes mat has member `a` which is std::vector<std::vector<float>>.
 * @return A 1D vector containing the flattened data.
 */
std::vector<float> flatten(const mat& matrix) {
    // Assuming mat class has a member 'a' of type std::vector<std::vector<float>>
    return flatten(matrix.a);
}

/**
 * @brief Unflattens a 1D vector back into a 2D vector (row-major).
 * @param flat_vec The input 1D vector.
 * @param[out] vec2d The output 2D vector. Will be resized and populated.
 * @param rows The number of rows expected in the output 2D vector.
 * @param cols The number of columns expected in the output 2D vector.
 */
void unflatten(const std::vector<float>& flat_vec, std::vector<std::vector<float>>& vec2d, size_t rows, size_t cols) {
    if (rows == 0 || cols == 0) {
        vec2d.clear();
        return; // Nothing to do for zero dimensions
    }
    if (flat_vec.size() != rows * cols) {
        fprintf(stderr, "Error: Cannot unflatten vector, size mismatch (%zu != %zu * %zu).\n", flat_vec.size(), rows, cols);
        // Set vec2d to a default state or throw?
        vec2d.assign(rows, std::vector<float>(cols, 0.0f)); // Example: fill with zeros
        // Or: throw std::runtime_error("Unflatten size mismatch");
        return;
    }
    vec2d.resize(rows);
    for (size_t i = 0; i < rows; ++i) {
        // Assign elements for the current row
        vec2d[i].assign(flat_vec.begin() + i * cols, flat_vec.begin() + (i + 1) * cols);
    }
}

// --- CUDA Error Checking Macro ---
#define CUDA_CHECK(call)                                                     \
do {                                                                         \
    cudaError_t err = call;                                                  \
    if (err != cudaSuccess) {                                                \
        fprintf(stderr, "CUDA Error in %s at line %d: %s\n",                 \
                __FILE__, __LINE__, cudaGetErrorString(err));                \
        /* Consider throwing an exception or returning an error code */      \
        /* For now, just print and potentially return */                     \
        return; /* Or throw std::runtime_error(cudaGetErrorString(err)); */  \
    }                                                                        \
} while (0)


/**
 * @brief compute KdotQ of specific parallel (or column of block vector b) using CUDA
 * kernels. This optimized version allocates large buffers on the GPU once, copies data 
 * in batches, launches kernels using offsets into these buffers, copies results back, 
 * and then frees. Mirrors the logic of the C++ parallelKdotQs function.
 * @param promptCount number of new tokens in the prompt being processed in this step.
 * @param currentTokenCount total number of tokens processed *before* this step across all blocks.
 * @param blockCount 1-based index of the current block being processed.
 * @param column 0-based index of the parallel (column in the block's attention grid) to compute.
 * @param isSelf true for self-attention, false for cross-attention.
 * @param inTraining true if in training mode, false if in inference mode.
 */
void transformer::cuParallelKdotQs(int& promptCount, int& currentTokenCount, int& blockCount, int& column, bool& isSelf,
    bool& inTraining) 
{
    // --- Basic Sanity Checks ---
    if (blockCount < 1 || blockCount > m) {
        fprintf(stderr, "Error in cuParallelKdotQs: Invalid blockCount %d (max %d)\n", blockCount, m);
        return;
    }
    if (column < 0 || column >= y) {
        fprintf(stderr, "Error in cuParallelKdotQs: Invalid column %d (max %d)\n", column, y-1);
        throw std::out_of_range("Invalid column index");
    }

    // initiate a block that reference the current block operations
    block& current_block = (inTraining == 1) ? t[blockCount - 1] : t[0]; // 0-based index for vector t
    if (x <= 0) {
        fprintf(stderr, "Warning in cuParallelKdotQs: Invalid number of parallels (layers) x = %d. No heads to process.\n", x);
        return; // No heads to process
    }
    if (promptCount < 0) {
         fprintf(stderr, "Warning in cuParallelKdotQs: promptCount is negative (%d). Setting to 0.\n", promptCount);
         return;
    }


    // --- Pre-computation and Setup ---
    const float inv_scaling = 1.0f / sqrtf(EMBEDDING);       // SCALING defined in attention.hpp
    const int embedding_dim = d;                    // 'd' is transformer's embedding dimension
    const int context_win_size = n;                 // 'n' is context window per head from transformer params
    const int kdotq_full_width = context_win_size;  // Max width/height of KdotQ matrix per head
    const int num_heads_in_parallel = x;            // Number of heads in this column (layers)

    // --- Data Sizes Per Head (in elements) ---
    // Ensure context_win_size and embedding_dim are positive
    if (context_win_size <= 0 || embedding_dim <= 0) {
         fprintf(stderr, "Error: Invalid dimensions (context_win_size=%d, embedding_dim=%d)\n", context_win_size, embedding_dim);
         return;
    }
    const size_t k_q_ev_head_elems = static_cast<size_t>(context_win_size) * embedding_dim;    // n * d
    const size_t kdotq_head_elems = static_cast<size_t>(context_win_size) * context_win_size;  // n * n
    const size_t qkcache_head_elems = static_cast<size_t>(embedding_dim) * embedding_dim;      // d * d

    // --- Total Data Sizes for the Parallel (in elements) ---
    const size_t total_k_elems = num_heads_in_parallel * k_q_ev_head_elems;     // x * n * d
    const size_t total_q_elems = num_heads_in_parallel * k_q_ev_head_elems;     // x * n * d
    const size_t total_evp_elems = num_heads_in_parallel * k_q_ev_head_elems;   // x * n * d
    const size_t total_m_elems = num_heads_in_parallel * qkcache_head_elems;    // x * d * d
    const size_t total_kdotq_elems = num_heads_in_parallel * kdotq_head_elems;  // x * n * n

    // --- Total Memory Comsumption (in bytes) ---
    const size_t total_k_bytes = total_k_elems * sizeof(float);
    const size_t total_q_bytes = total_q_elems * sizeof(float);
    const size_t total_evp_bytes = total_evp_elems * sizeof(float);
    const size_t total_m_bytes = total_m_elems * sizeof(float);
    const size_t total_kdotq_bytes = total_kdotq_elems * sizeof(float);

    // --- Device Pointers (Large Buffers) ---
    float* d_all_kdotq = nullptr;
    float* d_all_keys = nullptr;
    float* d_all_querys = nullptr;
    float* d_all_M = nullptr;
    float* d_all_EVp = nullptr;
    float* d_transformer_tokenEmbed_flat = nullptr;     // Shared, single copy
    float* d_block_tokForBlock_flat = nullptr;          // Shared, single copy

    // --- Host-side Aggregated Data Buffers ---
    std::vector<float> h_all_keys_flat;
    std::vector<float> h_all_querys_flat;
    std::vector<float> h_all_M_flat;
    std::vector<float> h_all_EVp_flat;
    std::vector<float> h_all_kdotq_flat;                // Allocate later if needed for copy-back
    std::vector<float> h_transformer_tokenEmbed_flat;   // Flattened global embeddings
    std::vector<float> h_block_tokForBlock_flat;        // Flattened block-local embeddings

    // --- Temporary Host Buffers for Flattening ---
    std::vector<float> temp_flat_buffer;

    try {
        // --- Allocate Large Device Buffers ---
        CUDA_CHECK(cudaMalloc(&d_all_kdotq, total_kdotq_bytes));
        // Initialize KdotQ to 0 or NaN? Let's initialize to 0 for safety.
        CUDA_CHECK(cudaMemset(d_all_kdotq, 0, total_kdotq_bytes));

        if (inTraining) {
            CUDA_CHECK(cudaMalloc(&d_all_keys, total_k_bytes));
            CUDA_CHECK(cudaMalloc(&d_all_querys, total_q_bytes));
            h_all_keys_flat.reserve(total_k_elems);
            h_all_querys_flat.reserve(total_q_elems);
        } 
        else {
            // Inference
            CUDA_CHECK(cudaMalloc(&d_all_M, total_m_bytes));
            h_all_M_flat.reserve(total_m_elems);

            if (blockCount == 1) {
                // Need global tokenEmbed
                h_transformer_tokenEmbed_flat = flatten(this->tokenEmbed);
                if (!h_transformer_tokenEmbed_flat.empty()) {
                    CUDA_CHECK(cudaMalloc(&d_transformer_tokenEmbed_flat, h_transformer_tokenEmbed_flat.size() * sizeof(float)));
                    CUDA_CHECK(cudaMemcpy(d_transformer_tokenEmbed_flat, h_transformer_tokenEmbed_flat.data(), h_transformer_tokenEmbed_flat.size() * sizeof(float), cudaMemcpyHostToDevice));
                } 
                else {
                    fprintf(stderr, "Warning: Global tokenEmbed is empty during Block 1 Inference. Cannot proceed.\n");
                    // If tokenEmbed is essential and empty, we likely cannot compute anything.
                    return;
                }
            }
            else {
                // Block N > 1
                // Need block-local tokForBlock and EVp from previous block
                h_block_tokForBlock_flat = flatten(current_block.tokForBlock);
                if (!h_block_tokForBlock_flat.empty()) {
                    CUDA_CHECK(cudaMalloc(&d_block_tokForBlock_flat, h_block_tokForBlock_flat.size() * sizeof(float)));
                    CUDA_CHECK(cudaMemcpy(d_block_tokForBlock_flat, h_block_tokForBlock_flat.data(), h_block_tokForBlock_flat.size() * sizeof(float), cudaMemcpyHostToDevice));
                }
                else {
                    fprintf(stderr, "Warning: Block-local tokForBlock is empty during Block N > 1 Inference.\n");
                    // If tokForBlock is needed and empty, might be an error or just the start of the block.
                    // Allocate a dummy buffer? Or rely on kernel checks? Let's allocate a small dummy. 
                    CUDA_CHECK(cudaMalloc(&d_block_tokForBlock_flat, 1 * sizeof(float))); // Allocate 1 float
                    CUDA_CHECK(cudaMemset(d_block_tokForBlock_flat, 0, 1 * sizeof(float))); // Set to 0
                }

                CUDA_CHECK(cudaMalloc(&d_all_EVp, total_evp_bytes));
                h_all_EVp_flat.reserve(total_evp_elems);
            }
        }

        // --- Pack Data on Host ---
        block* prev_block_ptr = (blockCount > 1) ? ((inTraining == 1) ? &t[blockCount - 2] : &t[0]): nullptr;

        for (int i = 0; i < num_heads_in_parallel; ++i) {
            // Check if head exists (safety check)
            if (i >= current_block.b.size() || column >= current_block.b[i].size()) {
                fprintf(stderr, "Error: Head index (%d, %d) out of bounds for current block.\n", i, column);
                return;
            }
            attention& head = current_block.b[i][column];

            if (inTraining) {
                // for keys
                temp_flat_buffer = flatten(head.K);
                if (temp_flat_buffer.size() != k_q_ev_head_elems) {
                     fprintf(stderr, "Warning: Head (%d, %d) K size mismatch (Expected %zu, Got %zu). Resizing/Padding host buffer.\n", i, column, k_q_ev_head_elems, temp_flat_buffer.size());
                     temp_flat_buffer.resize(k_q_ev_head_elems, 0.0f); // Pad with 0 if short, truncate if long
                }
                h_all_keys_flat.insert(h_all_keys_flat.end(), temp_flat_buffer.begin(), temp_flat_buffer.end());
                // for queries
                temp_flat_buffer = flatten(head.Q);
                if (temp_flat_buffer.size() != k_q_ev_head_elems) {
                    fprintf(stderr, "Warning: Head (%d, %d) Q size mismatch (Expected %zu, Got %zu). Resizing/Padding host buffer.\n", i, column, k_q_ev_head_elems, temp_flat_buffer.size());
                    temp_flat_buffer.resize(k_q_ev_head_elems, 0.0f);
                }
                h_all_querys_flat.insert(h_all_querys_flat.end(), temp_flat_buffer.begin(), temp_flat_buffer.end());
            } 
            else {
                // Inference
                temp_flat_buffer = flatten(head.qkCache); // head.qkCache is a 'mat'
                if (temp_flat_buffer.size() != qkcache_head_elems) {
                    fprintf(stderr, "Warning: Head (%d, %d) M (qkCache) size mismatch (Expected %zu, Got %zu). Resizing/Padding host buffer.\n", i, column, qkcache_head_elems, temp_flat_buffer.size());
                    temp_flat_buffer.resize(qkcache_head_elems, 0.0f);
                }
                h_all_M_flat.insert(h_all_M_flat.end(), temp_flat_buffer.begin(), temp_flat_buffer.end());

                if (blockCount > 1) {
                    if (!prev_block_ptr) { // Should not happen based on blockCount check, but safety first
                        return;
                    }
                    // Ensure previous block has the required head structure
                    if (i >= prev_block_ptr->b.size() || column >= prev_block_ptr->b[i].size()) {
                        fprintf(stderr, "Error: Head index (%d, %d) out of bounds for previous block.\n", i, column);
                        throw std::out_of_range("Head index out of bounds for previous block");
                    }
                    attention& prev_head = prev_block_ptr->b[i][column];
                    // for EV of head of previous block with same indices
                    temp_flat_buffer = flatten(prev_head.EV);
                    if (temp_flat_buffer.size() != k_q_ev_head_elems) {
                        fprintf(stderr, "Warning: Head (%d, %d) EVp size mismatch (Expected %zu, Got %zu). Resizing/Padding host buffer.\n", i, column, k_q_ev_head_elems, temp_flat_buffer.size());
                        temp_flat_buffer.resize(k_q_ev_head_elems, 0.0f);
                    }
                    h_all_EVp_flat.insert(h_all_EVp_flat.end(), temp_flat_buffer.begin(), temp_flat_buffer.end());
                }
            }
        } // End of host packing loop

        // --- Batch Copy Host -> Device ---
        if (inTraining) {
            if (h_all_keys_flat.size() != total_k_elems || h_all_querys_flat.size() != total_q_elems) {
                fprintf(stderr, "Error: Packed host K/Q size mismatch after loop (K: %zu vs %zu, Q: %zu vs %zu).\n",
                        h_all_keys_flat.size(), total_k_elems, h_all_querys_flat.size(), total_q_elems);
                return;
            }
            CUDA_CHECK(cudaMemcpy(d_all_keys, h_all_keys_flat.data(), total_k_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_all_querys, h_all_querys_flat.data(), total_q_bytes, cudaMemcpyHostToDevice));
        }
        else { 
            // Inference
            if (h_all_M_flat.size() != total_m_elems) {
                fprintf(stderr, "Error: Packed host M size mismatch after loop (%zu vs %zu).\n", h_all_M_flat.size(), total_m_elems);
                return;
            }
            CUDA_CHECK(cudaMemcpy(d_all_M, h_all_M_flat.data(), total_m_bytes, cudaMemcpyHostToDevice));
            if (blockCount > 1) {
                if (h_all_EVp_flat.size() != total_evp_elems) {
                    fprintf(stderr, "Error: Packed host EVp size mismatch after loop (%zu vs %zu).\n", h_all_EVp_flat.size(), total_evp_elems);
                    return;
                } 
                CUDA_CHECK(cudaMemcpy(d_all_EVp, h_all_EVp_flat.data(), total_evp_bytes, cudaMemcpyHostToDevice));
            }
            // d_transformer_tokenEmbed_flat and d_block_tokForBlock_flat already copied if needed
        }

        // --- Loop, Calculate Offsets, and Launch Kernels ---
        dim3 threadsPerBlock(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y); // Defined earlier
        dim3 numBlocks;

        // Calculate context-related variables once before the loop if they are constant for all heads
        int current_tokens_in_window = 0;
        int num_queries_eff = 0;
        int num_keys_eff = 0;
        int prompt_start_index = 0;
        int context_len = 0;
        int effective_prompt_len = 0;
        int tokens_processed_in_prev_blocks = 0;
        int tokens_in_block_before_prompt = 0;
        int prompt_start_index_in_block = 0;
        int context_len_in_block = 0;

        if (inTraining) {
            // Effective number of tokens currently in the window *before* adding the prompt
            current_tokens_in_window = currentTokenCount % context_win_size;
            // Handle edge case: if exactly at the boundary, the window is full from previous step
            if (current_tokens_in_window == 0 && currentTokenCount > 0) {
                // If it's the start of a *new* block (blockCount > 1), the window starts empty
                if (blockCount > 1 && currentTokenCount >= context_win_size * (blockCount - 1)) {
                    current_tokens_in_window = 0;
                }
                else {
                    // Otherwise, it means the window was full
                    current_tokens_in_window = context_win_size;
                }
            }

            // Calculate effective Q/K lengths for this step within the window limit
            num_queries_eff = std::min(current_tokens_in_window + promptCount, context_win_size);
            num_keys_eff = num_queries_eff; // For both self/cross in training, Q and K span the same updated context

            // Calculate grid dimensions based on effective sizes
            numBlocks.x = (num_keys_eff + threadsPerBlock.x - 1) / threadsPerBlock.x;
            numBlocks.y = (num_queries_eff + threadsPerBlock.y - 1) / threadsPerBlock.y;
            numBlocks.z = 1;
        }
        else { // Inference
            if (blockCount == 1) {
                prompt_start_index = currentTokenCount; // Index where the new prompt starts globally
                context_len = std::min(currentTokenCount + promptCount, context_win_size); // Total tokens in window after prompt
                // Effective prompt length considering window limit
                effective_prompt_len = std::min(promptCount, std::max(0, context_win_size - prompt_start_index));

                // Grid dimensions depend on the context length (keys/j) and effective prompt length (queries/i)
                numBlocks.x = (context_len + threadsPerBlock.x - 1) / threadsPerBlock.x;
                numBlocks.y = (effective_prompt_len + threadsPerBlock.y - 1) / threadsPerBlock.y;
                numBlocks.z = 1;
            }
            else { // Block N > 1
                tokens_processed_in_prev_blocks = (blockCount - 1) * context_win_size;
                // Tokens already in this block's window before the current prompt
                tokens_in_block_before_prompt = std::max(0, std::min(currentTokenCount - tokens_processed_in_prev_blocks, context_win_size));
                prompt_start_index_in_block = tokens_in_block_before_prompt;
                // Total relevant tokens in this block's window after adding prompt
                context_len_in_block = std::min(tokens_in_block_before_prompt + promptCount, context_win_size);
                // Effective prompt length considering window limit
                effective_prompt_len = std::min(promptCount, std::max(0, context_win_size - prompt_start_index_in_block));

                // Grid dimensions
                numBlocks.x = (context_len_in_block + threadsPerBlock.x - 1) / threadsPerBlock.x;
                numBlocks.y = (effective_prompt_len + threadsPerBlock.y - 1) / threadsPerBlock.y;
                numBlocks.z = 1;
            }
        }

        // start KdotQ computation for column 'column'
        for (int i = 0; i < num_heads_in_parallel; ++i) {
            // Calculate offsets into the large device buffers for this head
            float* d_kdotq_head = d_all_kdotq + i * kdotq_head_elems;
            float* d_keys_head = d_all_keys + i * k_q_ev_head_elems;     // Only valid if inTraining
            float* d_querys_head = d_all_querys + i * k_q_ev_head_elems; // Only valid if inTraining
            float* d_M_head = d_all_M + i * qkcache_head_elems;         // Only valid if !inTraining
            float* d_EVp_head = d_all_EVp + i * k_q_ev_head_elems;       // Only valid if !inTraining && blockCount > 1

            // --- Select Kernel and Launch ---
            if (inTraining) {
                 if (num_queries_eff > 0 && num_keys_eff > 0) { // Only launch if there's work
                    if (isSelf) {
                        kernelKdotQforSelf_train<<<numBlocks, threadsPerBlock>>>(
                            d_kdotq_head, d_keys_head, d_querys_head, num_queries_eff, num_keys_eff,
                            kdotq_full_width, embedding_dim, inv_scaling);
                    }
                    else {
                        kernelKdotQforCross_train<<<numBlocks, threadsPerBlock>>>(
                            d_kdotq_head, d_keys_head, d_querys_head, num_queries_eff, num_keys_eff,
                            kdotq_full_width, embedding_dim, inv_scaling);
                    }
                    CUDA_CHECK(cudaGetLastError()); // Check kernel launch error immediately
                 }
            }
            else { // Inference Mode
                if (effective_prompt_len > 0) { // Only launch if the prompt has effect in this window
                    if (blockCount == 1) {
                        if (d_transformer_tokenEmbed_flat != nullptr && d_M_head != nullptr) {
                            if (isSelf) {
                                kernelKdotQ_Block1_Self_Inference<<<numBlocks, threadsPerBlock>>>(
                                    d_kdotq_head, d_transformer_tokenEmbed_flat, d_M_head, prompt_start_index, effective_prompt_len,
                                    context_len, kdotq_full_width, embedding_dim, inv_scaling);
                            }
                            else {
                                kernelKdotQ_Block1_Cross_Inference<<<numBlocks, threadsPerBlock>>>(
                                    d_kdotq_head, d_transformer_tokenEmbed_flat, d_M_head, prompt_start_index, effective_prompt_len,
                                    context_len, kdotq_full_width, embedding_dim, inv_scaling);
                            }
                            CUDA_CHECK(cudaGetLastError());
                        } 
                        else {
                            fprintf(stderr, "Warning: Skipping Block 1 inference kernel for head %d due to null input pointers.\n", i);
                        }
                    } 
                    else { // Block N > 1
                        if (d_block_tokForBlock_flat != nullptr && d_EVp_head != nullptr && d_M_head != nullptr) {
                            if (isSelf) {
                                kernelKdotQ_BlockN_Self_Inference<<<numBlocks, threadsPerBlock>>>(
                                    d_kdotq_head, d_block_tokForBlock_flat, d_EVp_head, d_M_head, prompt_start_index_in_block, effective_prompt_len,
                                    context_len_in_block, kdotq_full_width, embedding_dim, inv_scaling);
                            }
                            else {
                                kernelKdotQ_BlockN_Cross_Inference<<<numBlocks, threadsPerBlock>>>(
                                    d_kdotq_head, d_block_tokForBlock_flat, d_EVp_head, d_M_head, prompt_start_index_in_block, effective_prompt_len,
                                    context_len_in_block, kdotq_full_width, embedding_dim, inv_scaling);
                            }
                            CUDA_CHECK(cudaGetLastError());
                        } 
                        else {
                             fprintf(stderr, "Warning: Skipping Block N>1 inference kernel for head %d due to null input pointers (tok=%p, EVp=%p, M=%p).\n",
                                    i, (void*)d_block_tokForBlock_flat, (void*)d_EVp_head, (void*)d_M_head);
                        }
                    }
                } // End if effective_prompt_len > 0
            } // End Inference Mode
        } // End loop over heads (i)

        // --- Batch Copy Result Device -> Host ---
        // Allocate host buffer now that we know the size
        h_all_kdotq_flat.resize(total_kdotq_elems);
        CUDA_CHECK(cudaMemcpy(h_all_kdotq_flat.data(), d_all_kdotq, total_kdotq_bytes, cudaMemcpyDeviceToHost));

        // --- Unpack Results into Attention Heads ---
        auto it_kdotq = h_all_kdotq_flat.begin();
        for (int i = 0; i < num_heads_in_parallel; ++i) {
            attention& head = current_block.b[i][column];

            // Create a temporary vector view or copy the segment
            std::vector<float> head_kdotq_flat(it_kdotq, it_kdotq + kdotq_head_elems);
            it_kdotq += kdotq_head_elems;

            // Unflatten into the head's KdotQ matrix
            // Ensure head.KdotQ is pre-sized correctly or handle resize in unflatten
            // Assuming head.KdotQ is std::vector<std::vector<float>>
            // Resize head.KdotQ if necessary before unflattening
            if (head.KdotQ.size() != context_win_size || (context_win_size > 0 && head.KdotQ[0].size() != context_win_size)) {
                head.KdotQ.assign(context_win_size, std::vector<float>(context_win_size, 0.0f));
            }
            unflatten(head_kdotq_flat, head.KdotQ, context_win_size, context_win_size);
        }

    } catch (const std::exception& e) {
        fprintf(stderr, "Runtime error during cuParallelKdotQs: %s\n", e.what());
        // Cleanup allocated memory before re-throwing or returning
        cudaFree(d_all_kdotq); // Safe to call on nullptr
        cudaFree(d_all_keys);
        cudaFree(d_all_querys);
        cudaFree(d_all_M);
        cudaFree(d_all_EVp);
        cudaFree(d_transformer_tokenEmbed_flat);
        cudaFree(d_block_tokForBlock_flat);
        return; // Re-throw the exception
    }

    // --- Free All Large Device Buffers ---
    cudaFree(d_all_kdotq); // Safe to call on nullptr
    cudaFree(d_all_keys);
    cudaFree(d_all_querys);
    cudaFree(d_all_M);
    cudaFree(d_all_EVp);
    cudaFree(d_transformer_tokenEmbed_flat);
    cudaFree(d_block_tokForBlock_flat);

    // Optional: Synchronize device if needed, but likely handled elsewhere.
    // CUDA_CHECK(cudaDeviceSynchronize());
}
