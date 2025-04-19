
// Add these includes at the top of clcompute.cpp if not already present
#include "include/transformer.hpp" // Should already be there
#include "include/block.hpp"       // For block class definition
#include "include/attention.hpp"   // For attention class definition and constants
#include <vector>
#include <string>
#include <stdexcept> // For std::runtime_error, std::out_of_range
#include <cmath>     // For sqrtf
#include <map>       // For cl_kernels map
#include <cstdio>    // For fprintf, stderr

#ifdef USE_OPENCL

#include <CL/cl.hpp>
#include <cfloat>

/**
 * @brief Computes the dot product of two vectors residing in global memory.
 * @param vec1 Pointer to the first vector in global memory.
 * @param vec2 Pointer to the second vector in global memory.
 * @param dim The dimension of the vectors.
 * @return The dot product of vec1 and vec2.
 */
inline float compute_dot_product(__global const float* vec1, __global const float* vec2, int dim) 
{
    float dot_product = 0.0f;
    for (int k = 0; k < dim; ++k) {
        dot_product += vec1[k] * vec2[k];
    }
    return dot_product;
}

/**
 * @brief Computes the dot product of vec1 * matrix * vec2.
 * @param vec1 Pointer to the first vector (row vector, size dim) in global memory.
 * @param vec2 Pointer to the second vector (column vector, size dim) in global memory.
 * @param matrix Pointer to the matrix (row-major, dim x dim) in global memory.
 * @param dim The dimension of the vectors and the matrix.
 * @return The scalar result of vec1 * matrix * vec2.
 */
inline float compute_dot_product(__global const float* vec1, __global const float* vec2, __global const float* matrix,
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
        float current_dot_product = compute_dot_product_ocl(EH, current_embedding_row, dim);
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
 * @brief OpenCL kernel for calculating KdotQ using keys and queries for self attention (Q[i] dot K[j] where j <= i)
 * @param[out] d_kdotq         Global pointer to hold dot products (KdotQ matrix, row-major)
 * @param[in]  d_keys          Global pointer to keys (K matrix, row-major)
 * @param[in]  d_querys        Global pointer to queries (Q matrix, row-major)
 * @param[in]  num_queries_eff Number of query rows (i) to compute for in this launch.
 * @param[in]  num_keys_eff    Number of key columns (j) to compute for in this launch.
 * @param[in]  kdotq_width     Total width (number of columns) of the d_kdotq buffer for indexing.
 * @param[in]  embedding_dim   Dimension of each key/query vector.
 * @param[in]  inv_scaling     Value of 1.0f / SCALING.
*/
__kernel void kernelKdotQforSelf_train(__global float* d_kdotq, __global const float* d_keys, __global const float* d_querys,
                                int num_queries_eff, int num_keys_eff, int kdotq_width, int embedding_dim, float inv_scaling)
{
    // Calculate the global row (query index i) and column (key index j) for this work-item
    int j = get_global_id(0); // Key index (column)
    int i = get_global_id(1); // Query index (row)

    // Boundary check AND self-attention causal mask (j <= i)
    if (i < num_queries_eff && j < num_keys_eff && j <= i) {
        // Pointers to the start of the i-th query vector and j-th key vector
        __global const float* q_vec = d_querys + i * embedding_dim;
        __global const float* k_vec = d_keys + j * embedding_dim;
        float dot_product = compute_dot_product_ocl(q_vec, k_vec, embedding_dim);

        // Calculate the index in the flattened d_kdotq array (row-major)
        int kdotq_index = i * kdotq_width + j;
        d_kdotq[kdotq_index] = dot_product * inv_scaling;
    }
}

/**
 * @brief OpenCL kernel for calculating KdotQ using keys and queries for cross attention (Q[i] dot K[j] for all i, j)
 * @param[out] d_kdotq         Global pointer to hold dot products (KdotQ matrix, row-major)
 * @param[in]  d_keys          Global pointer to keys (K matrix, row-major)
 * @param[in]  d_querys        Global pointer to queries (Q matrix, row-major)
 * @param[in]  num_queries_eff Number of query rows (i) to compute for in this launch.
 * @param[in]  num_keys_eff    Number of key columns (j) to compute for in this launch.
 * @param[in]  kdotq_width     Total width (number of columns) of the d_kdotq buffer for indexing.
 * @param[in]  embedding_dim   Dimension of each key/query vector.
 * @param[in]  inv_scaling     Value of 1.0f / SCALING.
 */
__kernel void kernelKdotQforCross_train(__global float* d_kdotq, __global const float* d_keys, __global const float* d_querys,
                                int num_queries_eff, int num_keys_eff, int kdotq_width, int embedding_dim, float inv_scaling)
{
    // Calculate the global column (key index j) and row (query index i) for this work-item
    int j = get_global_id(0); // Key index (column)
    int i = get_global_id(1); // Query index (row)

    // Boundary check (no causal mask for cross-attention)
    if (i < num_queries_eff && j < num_keys_eff) {
        // Pointers to the start of the i-th query vector and j-th key vector
        __global const float* q_vec = d_querys + i * embedding_dim;
        __global const float* k_vec = d_keys + j * embedding_dim;
        float dot_product = compute_dot_product_ocl(q_vec, k_vec, embedding_dim);
        // Calculate the index in the flattened d_kdotq array (row-major)
        int kdotq_index = i * kdotq_width + j;
        d_kdotq[kdotq_index] = dot_product * inv_scaling;
    }
}

/**------------------------------------INFERENCE------------------------------------**/

/**
 * @brief INFERENCE OpenCL kernel for Block 1 SELF-ATTENTION KdotQ.
 * Computes KdotQ[i][j] = (tokenEmbed[i] * M * tokenEmbed[j]) / SCALING for j <= i,
 * where i and j are the index of attention score of KdotQ grid.
 * @param[out] d_kdotq         Global pointer to output KdotQ (potentially sparse, only computes needed values).
 * @param[in]  d_tokenEmbed    Global pointer to token embeddings for the *entire* context (row-major).
 * @param[in]  d_M             Global pointer to the QK' matrix M (row-major, dim x dim).
 * @param[in]  prompt_start_index Index of the first token of the new prompt in d_tokenEmbed.
 * @param[in]  prompt_len      Number of tokens in the new prompt.
 * @param[in]  context_len     Total number of tokens in the context (prompt_start_index + prompt_len).
 * @param[in]  kdotq_width     Total width (number of columns) of the d_kdotq buffer for indexing (usually context_len).
 * @param[in]  embedding_dim   Dimension of each token vector and the matrix M.
 * @param[in]  inv_scaling     Value of 1.0f / SCALING.
 */
__kernel void kernelKdotQBlock1Self_Inference(__global float* d_kdotq, __global const float* d_tokenEmbed, __global const float* d_M,
                            int prompt_start_index, int prompt_len, int context_len, int kdotq_width,int embedding_dim, float inv_scaling)
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
        float dot_product = compute_dot_product_mat_ocl(q_vec, k_vec, d_M, embedding_dim);
        // index in flatten array
        int kdotq_index = i * kdotq_width + j;
        d_kdotq[kdotq_index] = dot_product * inv_scaling;
    }
}

/**
 * @brief INFERENCE OpenCL kernel for Block 1 CROSS-ATTENTION KdotQ.
 * Computes KdotQ[i][j] = (tokenEmbed[i] * M * tokenEmbed[j]) / SCALING for all j,
 * where i corresponds to the new prompt tokens and j spans the entire context.
 * @param[out] d_kdotq         Global pointer to output KdotQ (potentially sparse).
 * @param[in]  d_tokenEmbed    Global pointer to token embeddings for the *entire* context (row-major).
 * @param[in]  d_M             Global pointer to the QK' matrix M (row-major, dim x dim).
 * @param[in]  prompt_start_index Index of the first token of the new prompt in d_tokenEmbed.
 * @param[in]  prompt_len      Number of tokens in the new prompt.
 * @param[in]  context_len     Total number of tokens in the context (prompt_start_index + prompt_len).
 * @param[in]  kdotq_width     Total width (number of columns) of the d_kdotq buffer for indexing (usually context_len).
 * @param[in]  embedding_dim   Dimension of each token vector and the matrix M.
 * @param[in]  inv_scaling     Value of 1.0f / SCALING.
 */
__kernel void kernelKdotQBlock1Cross_Inference(__global float* d_kdotq, __global const float* d_tokenEmbed, __global const float* d_M,
                            int prompt_start_index, int prompt_len, int context_len, int kdotq_width, int embedding_dim, float inv_scaling)
{
    // Calculate the global key index (j) and the offset for the query index (i_offset)
    int j = get_global_id(0);
    int i_offset = get_global_id(1);
    int i = prompt_start_index + i_offset;
    if (i_offset < prompt_len && j < context_len) {
        // Pointers to the start of the i-th query vector and j-th key vector
        __global const float* q_vec = d_tokenEmbed + i * embedding_dim;
        __global const float* k_vec = d_tokenEmbed + j * embedding_dim;
        float dot_product = compute_dot_product_mat_ocl(q_vec, k_vec, d_M, embedding_dim);
        // Calculate the index in the flattened d_kdotq array (row-major)
        int kdotq_index = i * kdotq_width + j;
        d_kdotq[kdotq_index] = dot_product * inv_scaling;
    }
}

/**
 * @brief INFERENCE OpenCL kernel for Block N (N > 1) SELF-ATTENTION KdotQ.
 * Computes KdotQ[i][j] = (tokForBlock[i] * M * EVp[j]) / SCALING for j <= i,
 * where i corresponds to the new prompt tokens within the block's window,
 * and j spans the relevant context within the block's window.
 * @param[out] d_kdotq         Global pointer to output KdotQ for this block's window.
 * @param[in]  d_tokForBlock   Global pointer to token embeddings for this block's context window (row-major).
 * @param[in]  d_EVp           Global pointer to previous block's output vectors (EV') for this block's context window (row-major).
 * @param[in]  d_M             Global pointer to the QK' matrix M (row-major, dim x dim).
 * @param[in]  prompt_start_index_in_block Index of the first token of the new prompt *within the block's window*.
 * @param[in]  prompt_len      Number of tokens in the new prompt.
 * @param[in]  context_len_in_block Total number of relevant tokens in this block's window.
 * @param[in]  kdotq_width     Total width (number of columns) of the d_kdotq buffer for indexing (usually context_len_in_block).
 * @param[in]  embedding_dim   Dimension of each token vector and the matrix M.
 * @param[in]  inv_scaling     Value of 1.0f / SCALING.
 */
__kernel void kernelKdotQBlockNSelf_Inference(__global float* d_kdotq, __global const float* d_tokForBlock, __global const float* d_EVp,
                            __global const float* d_M, int prompt_start_index_in_block, int prompt_len, int context_len_in_block,
                            int kdotq_width, int embedding_dim, float inv_scaling)
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
        float dot_product = compute_dot_product_mat_ocl(q_vec, k_vec, d_M, embedding_dim);
        // Calculate the index in the flattened d_kdotq array (row-major, relative to block window)
        int kdotq_index = i * kdotq_width + j;
        d_kdotq[kdotq_index] = dot_product * inv_scaling;
    }
}

/**
 * @brief INFERENCE OpenCL kernel for Block N (N > 1) CROSS-ATTENTION KdotQ.
 * Computes KdotQ[i][j] = (tokForBlock[i] * M * EVp[j]) / SCALING for all j,
 * where i corresponds to the new prompt tokens within the block's window,
 * and j spans the relevant context within the block's window.
 * @param[out] d_kdotq         Global pointer to output KdotQ for this block's window.
 * @param[in]  d_tokForBlock   Global pointer to token embeddings for this block's context window (row-major).
 * @param[in]  d_EVp           Global pointer to previous block's output vectors (EV') for this block's context window (row-major).
 * @param[in]  d_M             Global pointer to the QK' matrix M (row-major, dim x dim).
 * @param[in]  prompt_start_index_in_block Index of the first token of the new prompt *within the block's window*.
 * @param[in]  prompt_len      Number of tokens in the new prompt.
 * @param[in]  context_len_in_block Total number of relevant tokens in this block's window.
 * @param[in]  kdotq__width     Total width (number of columns) of the d_kdotq buffer for indexing (usually context_len_in_block).
 * @param[in]  embedding_dim   Dimension of each token vector and the matrix M.
 * @param[in]  inv_scaling     Value of 1.0f / SCALING.
 */
__kernel void kernelKdotQBlockNCross_Inference(__global float* d_kdotq, __global const float* d_tokForBlock, __global const float* d_EVp,
                                __global const float* d_M, int prompt_start_index_in_block, int prompt_len, int context_len_in_block,
                                int kdotq_width, int embedding_dim, float inv_scaling)
{
    // Calculate the key index (j) and query index offset (i_offset) *within the block's window*
    int j = get_global_id(0);
    int i_offset = get_global_id(1);
    int i = prompt_start_index_in_block + i_offset;
    if (i_offset < prompt_len && j < context_len_in_block) {
        // Pointers to the start of the i-th query vector (from tokForBlock) and j-th key vector (from EVp)
        __global const float* q_vec = d_tokForBlock + i * embedding_dim;
        __global const float* k_vec = d_EVp + j * embedding_dim;
        float dot_product = compute_dot_product_mat_ocl(q_vec, k_vec, d_M, embedding_dim);
        // Calculate the index in the flattened d_kdotq array (row-major, relative to block window)
        int kdotq_index = i * kdotq_width + j;
        d_kdotq[kdotq_index] = dot_product * inv_scaling;
    }
}


// Helper function to convert OpenCL error codes to strings
const char* oclErrorString(cl_int error) {
    static const char* errorString[] = {
        "CL_SUCCESS",                     // 0
        "CL_DEVICE_NOT_FOUND",            // -1
        "CL_DEVICE_NOT_AVAILABLE",        // -2
        "CL_COMPILER_NOT_AVAILABLE",      // -3
        "CL_MEM_OBJECT_ALLOCATION_FAILURE",// -4
        "CL_OUT_OF_RESOURCES",            // -5
        "CL_OUT_OF_HOST_MEMORY",          // -6
        "CL_PROFILING_INFO_NOT_AVAILABLE",// -7
        "CL_MEM_COPY_OVERLAP",            // -8
        "CL_IMAGE_FORMAT_MISMATCH",       // -9
        "CL_IMAGE_FORMAT_NOT_SUPPORTED",  // -10
        "CL_BUILD_PROGRAM_FAILURE",       // -11
        "CL_MAP_FAILURE",                 // -12
        "CL_MISALIGNED_SUB_BUFFER_OFFSET", // -13
        "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST", // -14
        "CL_COMPILE_PROGRAM_FAILURE",     // -15
        "CL_LINKER_NOT_AVAILABLE",        // -16
        "CL_LINK_PROGRAM_FAILURE",        // -17
        "CL_DEVICE_PARTITION_FAILED",     // -18
        "CL_KERNEL_ARG_INFO_NOT_AVAILABLE",// -19
        "Unknown Error",                  // -20
        "Unknown Error",                  // -21
        "Unknown Error",                  // -22
        "Unknown Error",                  // -23
        "Unknown Error",                  // -24
        "Unknown Error",                  // -25
        "Unknown Error",                  // -26
        "Unknown Error",                  // -27
        "Unknown Error",                  // -28
        "Unknown Error",                  // -29
        "CL_INVALID_VALUE",               // -30
        "CL_INVALID_DEVICE_TYPE",         // -31
        "CL_INVALID_PLATFORM",            // -32
        "CL_INVALID_DEVICE",              // -33
        "CL_INVALID_CONTEXT",             // -34
        "CL_INVALID_QUEUE_PROPERTIES",    // -35
        "CL_INVALID_COMMAND_QUEUE",       // -36
        "CL_INVALID_HOST_PTR",            // -37
        "CL_INVALID_MEM_OBJECT",          // -38
        "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR", // -39
        "CL_INVALID_IMAGE_SIZE",          // -40
        "CL_INVALID_SAMPLER",             // -41
        "CL_INVALID_BINARY",              // -42
        "CL_INVALID_BUILD_OPTIONS",       // -43
        "CL_INVALID_PROGRAM",             // -44
        "CL_INVALID_PROGRAM_EXECUTABLE",  // -45
        "CL_INVALID_KERNEL_NAME",         // -46
        "CL_INVALID_KERNEL_DEFINITION",   // -47
        "CL_INVALID_KERNEL",              // -48
        "CL_INVALID_ARG_INDEX",           // -49
        "CL_INVALID_ARG_VALUE",           // -50
        "CL_INVALID_ARG_SIZE",            // -51
        "CL_INVALID_KERNEL_ARGS",         // -52
        "CL_INVALID_WORK_DIMENSION",      // -53
        "CL_INVALID_WORK_GROUP_SIZE",     // -54
        "CL_INVALID_WORK_ITEM_SIZE",      // -55
        "CL_INVALID_GLOBAL_OFFSET",       // -56
        "CL_INVALID_EVENT_WAIT_LIST",     // -57
        "CL_INVALID_EVENT",               // -58
        "CL_INVALID_OPERATION",           // -59
        "CL_INVALID_GL_OBJECT",           // -60
        "CL_INVALID_BUFFER_SIZE",         // -61
        "CL_INVALID_MIP_LEVEL",           // -62
        "CL_INVALID_GLOBAL_WORK_SIZE",    // -63
        "CL_INVALID_PROPERTY",            // -64
        "CL_INVALID_IMAGE_DESCRIPTOR",    // -65
        "CL_INVALID_COMPILER_OPTIONS",    // -66
        "CL_INVALID_LINKER_OPTIONS",      // -67
        "CL_INVALID_DEVICE_PARTITION_COUNT", // -68
    };

    if (error <= 0 && error >= -68) {
        return errorString[-error];
    } else {
        return "Unknown OpenCL Error";
    }
}


#define CL_CHECK(call)                                                      \
do {                                                                        \
    cl_int err = call;                                                      \
    if (err != CL_SUCCESS) {                                                \
        fprintf(stderr, "OpenCL Error in %s at line %d: %s (%d)\n",         \
                __FILE__, __LINE__, oclErrorString(err), err);              \
        /* Consider throwing an exception or returning an error code */     \
        /* For now, just print and potentially return */                    \
        /* Adjust behavior as needed (e.g., throw cl::Error(err)) */        \
        return; /* Or throw cl::Error(err, "OpenCL Error occurred"); */     \
    }                                                                       \
} while (0)

// ========================================================================= //
// Host-side Helper Functions (Identical to CUDA version)                    //
// ========================================================================= //

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
        // Check consistency only if cols > 0 to avoid issues with empty inner vectors
        if (cols > 0 && vec2d[i].size() != cols) {
             fprintf(stderr, "Warning: Inconsistent column count in flatten (Row %zu has %zu, expected %zu). Data might be truncated or padded later.\n", i, vec2d[i].size(), cols);
             // Decide how to handle: Use first row's count? Throw? Pad/Truncate?
             // Current CUDA code inserts anyway, let's mimic that but keep the warning.
        }
        // Ensure we don't read out of bounds if a row is unexpectedly short
        // flat_vec.insert(flat_vec.end(), vec2d[i].begin(), vec2d[i].begin() + std::min(vec2d[i].size(), cols));
        // The original code just inserts whatever the row has. Let's stick to that for direct porting.
         flat_vec.insert(flat_vec.end(), vec2d[i].begin(), vec2d[i].end());
    }
     // Post-check: If sizes were inconsistent, the total size might not be rows*cols.
     if (flat_vec.size() != rows * cols) {
         fprintf(stderr, "Warning: Flattened vector size (%zu) does not match expected (%zu * %zu) due to inconsistent rows.\n", flat_vec.size(), rows, cols);
     }
    return flat_vec;
}

/**
 * @brief Flattens a mat object into a 1D vector (row-major).
 * @param matrix The input mat object. Assumes mat has member `a` which is std::vector<std::vector<float>>.
 * @return A 1D vector containing the flattened data.
 */
std::vector<float> flatten(const mat& matrix) {
    // Assuming mat class has a public member 'a' of type std::vector<std::vector<float>>
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
        // If rows > 0 but cols == 0, maybe create rows empty vectors?
        if (rows > 0) {
            vec2d.resize(rows);
        }
        return; // Nothing to do for zero dimensions in flat_vec expected
    }
    if (flat_vec.size() != rows * cols) {
        fprintf(stderr, "Error: Cannot unflatten vector, size mismatch (%zu != %zu * %zu).\n", flat_vec.size(), rows, cols);
        // Set vec2d to a default state or throw?
        vec2d.assign(rows, std::vector<float>(cols, 0.0f)); // Example: fill with zeros
        // Or: throw std::runtime_error("Unflatten size mismatch");
        return;
    }
    vec2d.resize(rows);
    auto it = flat_vec.begin();
    for (size_t i = 0; i < rows; ++i) {
        // Assign elements for the current row using iterators
        vec2d[i].assign(it, it + cols);
        it += cols; // Move iterator to the start of the next row
    }
}


// ========================================================================= //
// OpenCL Implementation of Parallel KdotQ Computation                       //
// ========================================================================= //

// Define thread block dimensions (tune these based on your GPU architecture)
// These correspond to OpenCL local work group sizes
#define WORKGROUP_SIZE_X 16      // or 32
#define WORKGROUP_SIZE_Y 16      // or 32

/**
 * @brief compute KdotQ of specific parallel (or column of block vector b) using OpenCL
 * kernels. This optimized version allocates large buffers on the device once, copies data
 * in batches, launches kernels using offsets into these buffers, copies results back,
 * and then frees. Mirrors the logic of the C++ parallelKdotQs and CUDA cuParallelKdotQs functions.
 * @param promptCount number of new tokens in the prompt being processed in this step.
 * @param currentTokenCount total number of tokens processed *before* this step across all blocks.
 * @param blockCount 1-based index of the current block being processed.
 * @param column 0-based index of the parallel (column in the block's attention grid) to compute.
 * @param isSelf true for self-attention, false for cross-attention.
 * @param inTraining true if in training mode, false if in inference mode.
 *
 * @note Assumes cl_context, cl_queue, and cl_kernels are accessible (e.g., member variables).
 */
void transformer::clParallelKdotQs(int& promptCount, int& currentTokenCount, int& blockCount, int& column, bool& isSelf,
    bool& inTraining)
{
    // --- Basic Sanity Checks ---
    if (blockCount < 1 || blockCount > m) {
        fprintf(stderr, "Error in clParallelKdotQs: Invalid blockCount %d (max %d)\n", blockCount, m);
        return;
    }
    if (column < 0 || column >= y) {
        fprintf(stderr, "Error in clParallelKdotQs: Invalid column %d (max %d)\n", column, y - 1);
        throw std::out_of_range("Invalid column index");
    }

    // initiate a block that reference the current block operations
    block& current_block = (inTraining == 1) ? t[blockCount - 1] : t[0]; // 0-based index for vector t
    if (x <= 0) {
        fprintf(stderr, "Warning in clParallelKdotQs: Invalid number of parallels (layers) x = %d. No heads to process.\n", x);
        return; // No heads to process
    }
    if (promptCount < 0) {
         fprintf(stderr, "Warning in clParallelKdotQs: promptCount is negative (%d). Setting to 0.\n", promptCount);
         // Decide if we should proceed with promptCount = 0 or return. Let's return for safety.
         return;
    }

    // --- Pre-computation and Setup ---
    const float inv_scaling = 1.0f / sqrtf(static_cast<float>(EMBEDDING)); // Use static_cast for clarity
    const cl_int embedding_dim = d;                    // 'd' is transformer's embedding dimension
    const cl_int context_win_size = n;                 // 'n' is context window per head from transformer params
    const cl_int kdotq_full_width = context_win_size;  // Max width/height of KdotQ matrix per head
    const int num_heads_in_parallel = x;            // Number of heads in this column (layers)

    // --- Data Sizes Per Head (in elements) ---
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

    // --- Total Memory Consumption (in bytes) ---
    const size_t total_k_bytes = total_k_elems * sizeof(cl_float);
    const size_t total_q_bytes = total_q_elems * sizeof(cl_float);
    const size_t total_evp_bytes = total_evp_elems * sizeof(cl_float);
    const size_t total_m_bytes = total_m_elems * sizeof(cl_float);
    const size_t total_kdotq_bytes = total_kdotq_elems * sizeof(cl_float);

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

    // --- OpenCL Device Buffers (using cl::Buffer RAII wrapper) ---
    cl::Buffer d_all_kdotq;
    cl::Buffer d_all_keys;
    cl::Buffer d_all_querys;
    cl::Buffer d_all_M;
    cl::Buffer d_all_EVp;
    cl::Buffer d_transformer_tokenEmbed_flat; // Shared, single copy
    cl::Buffer d_block_tokForBlock_flat;      // Shared, single copy

    try {
        // --- Allocate Large Device Buffers ---
        // Use cl_context (assumed member variable or accessible)
        d_all_kdotq = cl::Buffer(cl_context, CL_MEM_READ_WRITE, total_kdotq_bytes);
        // Initialize KdotQ to 0. Use enqueueFillBuffer for initialization.
        cl_float zero = 0.0f;
        CL_CHECK(cl_queue.enqueueFillBuffer(d_all_kdotq, zero, 0, total_kdotq_bytes));
        // Ensure fill is complete before proceeding (optional, depends on queue properties)
        // CL_CHECK(cl_queue.finish());

        if (inTraining) {
            d_all_keys = cl::Buffer(cl_context, CL_MEM_READ_ONLY, total_k_bytes);
            d_all_querys = cl::Buffer(cl_context, CL_MEM_READ_ONLY, total_q_bytes);
            h_all_keys_flat.reserve(total_k_elems);
            h_all_querys_flat.reserve(total_q_elems);
        }
        else {
            // Inference
            d_all_M = cl::Buffer(cl_context, CL_MEM_READ_ONLY, total_m_bytes);
            h_all_M_flat.reserve(total_m_elems);

            if (blockCount == 1) {
                // Need global tokenEmbed
                h_transformer_tokenEmbed_flat = flatten(this->tokenEmbed);
                if (!h_transformer_tokenEmbed_flat.empty()) {
                    d_transformer_tokenEmbed_flat = cl::Buffer(cl_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                               h_transformer_tokenEmbed_flat.size() * sizeof(cl_float),
                                                               h_transformer_tokenEmbed_flat.data());
                } else {
                    fprintf(stderr, "Warning: Global tokenEmbed is empty during Block 1 Inference. Cannot proceed.\n");
                    return; // Cannot compute without embeddings
                }
            } else {
                // Block N > 1
                // Need block-local tokForBlock and EVp from previous block
                h_block_tokForBlock_flat = flatten(current_block.tokForBlock);
                if (!h_block_tokForBlock_flat.empty()) {
                     d_block_tokForBlock_flat = cl::Buffer(cl_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                          h_block_tokForBlock_flat.size() * sizeof(cl_float),
                                                          h_block_tokForBlock_flat.data());
                } else {
                    fprintf(stderr, "Warning: Block-local tokForBlock is empty during Block N > 1 Inference.\n");
                    // Allocate a small dummy buffer if needed by kernel logic, otherwise might be okay.
                    // Let's assume kernels handle empty inputs gracefully or checks prevent launch.
                    // If a buffer *must* exist, create a minimal one:
                    // cl_float dummy_val = 0.0f;
                    // d_block_tokForBlock_flat = cl::Buffer(cl_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_float), &dummy_val);
                }

                d_all_EVp = cl::Buffer(cl_context, CL_MEM_READ_ONLY, total_evp_bytes);
                h_all_EVp_flat.reserve(total_evp_elems);
            }
        }

        // --- Pack Data on Host ---
        block* prev_block_ptr = (blockCount > 1) ? ((inTraining == 1) ? &t[blockCount - 2] : &t[0]) : nullptr;

        for (int i = 0; i < num_heads_in_parallel; ++i) {
            // Check if head exists (safety check)
            if (i >= static_cast<int>(current_block.b.size()) || column >= static_cast<int>(current_block.b[i].size())) {
                fprintf(stderr, "Error: Head index (%d, %d) out of bounds for current block.\n", i, column);
                return; // Or throw
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
                    if (!prev_block_ptr) {
                         fprintf(stderr, "Error: prev_block_ptr is null for blockCount > 1.\n");
                         return; // Should not happen based on blockCount check
                    }
                    // Ensure previous block has the required head structure
                    if (i >= static_cast<int>(prev_block_ptr->b.size()) || column >= static_cast<int>(prev_block_ptr->b[i].size())) {
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
        // Use enqueueWriteBuffer. CL_TRUE makes it blocking.
        if (inTraining) {
            if (h_all_keys_flat.size() != total_k_elems || h_all_querys_flat.size() != total_q_elems) {
                fprintf(stderr, "Error: Packed host K/Q size mismatch after loop (K: %zu vs %zu, Q: %zu vs %zu).\n",
                        h_all_keys_flat.size(), total_k_elems, h_all_querys_flat.size(), total_q_elems);
                return;
            }
            CL_CHECK(cl_queue.enqueueWriteBuffer(d_all_keys, CL_TRUE, 0, total_k_bytes, h_all_keys_flat.data()));
            CL_CHECK(cl_queue.enqueueWriteBuffer(d_all_querys, CL_TRUE, 0, total_q_bytes, h_all_querys_flat.data()));
        } else {
            // Inference
            if (h_all_M_flat.size() != total_m_elems) {
                fprintf(stderr, "Error: Packed host M size mismatch after loop (%zu vs %zu).\n", h_all_M_flat.size(), total_m_elems);
                return;
            }
             CL_CHECK(cl_queue.enqueueWriteBuffer(d_all_M, CL_TRUE, 0, total_m_bytes, h_all_M_flat.data()));
            if (blockCount > 1) {
                 if (h_all_EVp_flat.size() != total_evp_elems) {
                    fprintf(stderr, "Error: Packed host EVp size mismatch after loop (%zu vs %zu).\n", h_all_EVp_flat.size(), total_evp_elems);
                    return;
                }
                CL_CHECK(cl_queue.enqueueWriteBuffer(d_all_EVp, CL_TRUE, 0, total_evp_bytes, h_all_EVp_flat.data()));
            }
            // d_transformer_tokenEmbed_flat and d_block_tokForBlock_flat already copied via CL_MEM_COPY_HOST_PTR
        }

        // --- Loop, Calculate Offsets, and Launch Kernels ---
        cl::NDRange local_work_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y);
        cl::NDRange global_work_size; // Will be set inside the loop based on effective dimensions

        // Calculate context-related variables once before the loop if they are constant for all heads
        cl_int current_tokens_in_window = 0;
        cl_int num_queries_eff = 0;
        cl_int num_keys_eff = 0;
        cl_int prompt_start_index = 0;
        cl_int context_len = 0;
        cl_int effective_prompt_len = 0;
        cl_int tokens_processed_in_prev_blocks = 0;
        cl_int tokens_in_block_before_prompt = 0;
        cl_int prompt_start_index_in_block = 0;
        cl_int context_len_in_block = 0;

        // Helper lambda to calculate rounded global size
        auto calculate_global_size = & {
            size_t global_x = (total_x + WORKGROUP_SIZE_X - 1) / WORKGROUP_SIZE_X * WORKGROUP_SIZE_X;
            size_t global_y = (total_y + WORKGROUP_SIZE_Y - 1) / WORKGROUP_SIZE_Y * WORKGROUP_SIZE_Y;
            return cl::NDRange(global_x, global_y);
        };

        if (inTraining) {
            // Effective number of tokens currently in the window *before* adding the prompt
            current_tokens_in_window = currentTokenCount % context_win_size;
            if (current_tokens_in_window == 0 && currentTokenCount > 0) {
                 if (blockCount > 1 && currentTokenCount >= context_win_size * (blockCount - 1)) {
                    current_tokens_in_window = 0;
                } else {
                    current_tokens_in_window = context_win_size;
                }
            }
            num_queries_eff = std::min(current_tokens_in_window + promptCount, context_win_size);
            num_keys_eff = num_queries_eff;
            global_work_size = calculate_global_size(num_keys_eff, num_queries_eff);
        } else { // Inference
            if (blockCount == 1) {
                prompt_start_index = currentTokenCount;
                context_len = std::min(currentTokenCount + promptCount, context_win_size);
                effective_prompt_len = std::min(promptCount, std::max(0, context_win_size - prompt_start_index));
                global_work_size = calculate_global_size(context_len, effective_prompt_len);
            } else { // Block N > 1
                tokens_processed_in_prev_blocks = (blockCount - 1) * context_win_size;
                tokens_in_block_before_prompt = std::max(0, std::min(currentTokenCount - tokens_processed_in_prev_blocks, context_win_size));
                prompt_start_index_in_block = tokens_in_block_before_prompt;
                context_len_in_block = std::min(tokens_in_block_before_prompt + promptCount, context_win_size);
                effective_prompt_len = std::min(promptCount, std::max(0, context_win_size - prompt_start_index_in_block));
                global_work_size = calculate_global_size(context_len_in_block, effective_prompt_len);
            }
        }

        // --- Select and Launch Kernels ---
        // The provided OpenCL kernels calculate indices based on get_global_id.
        // We launch one kernel per head, adjusting the arguments.
        // This mirrors the CUDA approach more closely than one giant kernel launch.

        for (int i = 0; i < num_heads_in_parallel; ++i) {

            // Calculate byte offsets and sizes for the current head's data within aggregated buffers
            size_t kdotq_byte_offset = i * kdotq_head_elems * sizeof(cl_float);
            size_t k_byte_offset     = i * k_q_ev_head_elems * sizeof(cl_float);
            size_t q_byte_offset     = i * k_q_ev_head_elems * sizeof(cl_float);
            size_t m_byte_offset     = i * qkcache_head_elems * sizeof(cl_float);
            size_t evp_byte_offset   = i * k_q_ev_head_elems * sizeof(cl_float);

            size_t kdotq_head_bytes = kdotq_head_elems * sizeof(cl_float);
            size_t k_q_ev_head_bytes= k_q_ev_head_elems * sizeof(cl_float);
            size_t qkcache_head_bytes= qkcache_head_elems * sizeof(cl_float);

            // Create sub-buffer for KdotQ output for this head
            cl_buffer_region kdotq_region = {kdotq_byte_offset, kdotq_head_bytes};
            cl::Buffer d_kdotq_head = d_all_kdotq.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &kdotq_region);

            cl::Kernel kernel; // Kernel object for this head's launch

            if (inTraining) {
                 if (num_queries_eff > 0 && num_keys_eff > 0) {
                    // Create sub-buffers for K and Q
                    cl_buffer_region k_region = {k_byte_offset, k_q_ev_head_bytes};
                    cl::Buffer d_keys_head = d_all_keys.createSubBuffer(CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &k_region);

                    cl_buffer_region q_region = {q_byte_offset, k_q_ev_head_bytes};
                    cl::Buffer d_querys_head = d_all_querys.createSubBuffer(CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &q_region);

                    if (isSelf) {
                        kernel = cl_kernels.at("kernelKdotQforSelf_train");
                        CL_CHECK(kernel.setArg(0, d_kdotq_head));  // Pass sub-buffer
                        CL_CHECK(kernel.setArg(1, d_keys_head));   // Pass sub-buffer
                        CL_CHECK(kernel.setArg(2, d_querys_head)); // Pass sub-buffer
                        // ... set remaining args 3-7 ...
                    } else { // Cross Attention
                        kernel = cl_kernels.at("kernelKdotQforCross_train");
                        CL_CHECK(kernel.setArg(0, d_kdotq_head));  // Pass sub-buffer
                        CL_CHECK(kernel.setArg(1, d_keys_head));   // Pass sub-buffer
                        CL_CHECK(kernel.setArg(2, d_querys_head)); // Pass sub-buffer
                        // ... set remaining args 3-7 ...
                    }
                    CL_CHECK(cl_queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_work_size, local_work_size));
                 }
            } else { // Inference Mode
                if (effective_prompt_len > 0) {
                    // Create sub-buffer for M
                    cl_buffer_region m_region = {m_byte_offset, qkcache_head_bytes};
                    cl::Buffer d_M_head = d_all_M.createSubBuffer(CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &m_region);

                    if (blockCount == 1) {
                         if (isSelf) {
                            kernel = cl_kernels.at("kernelKdotQBlock1Self_Inference");
                            CL_CHECK(kernel.setArg(0, d_kdotq_head)); // Pass sub-buffer
                            CL_CHECK(kernel.setArg(1, d_transformer_tokenEmbed_flat)); // Shared buffer (no sub-buffer needed)
                            CL_CHECK(kernel.setArg(2, d_M_head));    // Pass sub-buffer
                            // ... set remaining args 3-8 ...
                        } else {
                            kernel = cl_kernels.at("kernelKdotQBlock1Cross_Inference");
                            CL_CHECK(kernel.setArg(0, d_kdotq_head)); // Pass sub-buffer
                            CL_CHECK(kernel.setArg(1, d_transformer_tokenEmbed_flat));
                            CL_CHECK(kernel.setArg(2, d_M_head));    // Pass sub-buffer
                            // ... set remaining args 3-8 ...
                        }
                    } else { // Block N > 1
                        // Create sub-buffer for EVp
                        cl_buffer_region evp_region = {evp_byte_offset, k_q_ev_head_bytes};
                        cl::Buffer d_EVp_head = d_all_EVp.createSubBuffer(CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &evp_region);

                        if (isSelf) {
                            kernel = cl_kernels.at("kernelKdotQBlockNSelf_Inference");
                            CL_CHECK(kernel.setArg(0, d_kdotq_head)); // Pass sub-buffer
                            CL_CHECK(kernel.setArg(1, d_block_tokForBlock_flat)); // Shared buffer
                            CL_CHECK(kernel.setArg(2, d_EVp_head));   // Pass sub-buffer
                            CL_CHECK(kernel.setArg(3, d_M_head));     // Pass sub-buffer
                            // ... set remaining args 4-9 ...
                        } else {
                            kernel = cl_kernels.at("kernelKdotQBlockNCross_Inference");
                            CL_CHECK(kernel.setArg(0, d_kdotq_head)); // Pass sub-buffer
                            CL_CHECK(kernel.setArg(1, d_block_tokForBlock_flat));
                            CL_CHECK(kernel.setArg(2, d_EVp_head));   // Pass sub-buffer
                            CL_CHECK(kernel.setArg(3, d_M_head));     // Pass sub-buffer
                            // ... set remaining args 4-9 ...
                        }
                    }
                     CL_CHECK(cl_queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_work_size, local_work_size));
                } // End if effective_prompt_len > 0
            } // End Inference Mode
            // Sub-buffers (d_kdotq_head, d_keys_head, etc.) go out of scope and are released here
        } // End loop over heads (i)


        // Ensure all kernels are finished before reading back
        CL_CHECK(cl_queue.finish());

        // --- Batch Copy Result Device -> Host ---
        h_all_kdotq_flat.resize(total_kdotq_elems); // Allocate host buffer
        CL_CHECK(cl_queue.enqueueReadBuffer(d_all_kdotq, CL_TRUE, 0, total_kdotq_bytes, h_all_kdotq_flat.data()));

        // --- Unpack Results into Attention Heads ---
        auto it_kdotq = h_all_kdotq_flat.begin();
        for (int i = 0; i < num_heads_in_parallel; ++i) {
             if (i >= static_cast<int>(current_block.b.size()) || column >= static_cast<int>(current_block.b[i].size())) {
                 // This should have been caught earlier, but double-check
                 fprintf(stderr, "Error: Head index (%d, %d) out of bounds during unpacking.\n", i, column);
                 continue;
             }
            attention& head = current_block.b[i][column];

            // Create a temporary vector view or copy the segment
            std::vector<float> head_kdotq_flat(it_kdotq, it_kdotq + kdotq_head_elems);
            it_kdotq += kdotq_head_elems;

            // Unflatten into the head's KdotQ matrix
            // Resize head.KdotQ if necessary before unflattening
            if (head.KdotQ.size() != static_cast<size_t>(context_win_size) ||
               (context_win_size > 0 && (head.KdotQ.empty() || head.KdotQ[0].size() != static_cast<size_t>(context_win_size))))
            {
                 head.KdotQ.assign(context_win_size, std::vector<float>(context_win_size, 0.0f));
            }
            unflatten(head_kdotq_flat, head.KdotQ, context_win_size, context_win_size);
        }

    } catch (const cl::Error& err) {
        fprintf(stderr, "OpenCL Runtime error during clParallelKdotQs: %s (%d)\n", err.what(), err.err());
        // cl::Buffer RAII will handle cleanup automatically
        return;
    } catch (const std::exception& e) {
        fprintf(stderr, "Standard exception during clParallelKdotQs: %s\n", e.what());
        // cl::Buffer RAII will handle cleanup automatically
        return;
    }

    // --- Buffers are automatically released by cl::Buffer destructors ---

    // Optional: Synchronize device if needed, but blocking calls/finish should suffice.
    // CL_CHECK(cl_queue.finish());
}

#endif // USE_OPENCL
