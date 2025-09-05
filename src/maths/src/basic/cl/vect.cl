
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

inline float compute_dot_product_mat(__global const float* vec1, __global const float* vec2, __global const float* matrix,
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

__kernel void matrixMultiplyKernel(__global const float* A, __global const float* B,
                                   __global float* C, int rowsA, int colsA, int colsB)
{
    // Get the global row and column indices for the output matrix C
    int row = get_global_id(1); // Corresponds to y-dimension in launch
    int col = get_global_id(0); // Corresponds to x-dimension in launch

    // Check if the work-item is within the bounds of the output matrix C
    if (row < rowsA && col < colsB) {
        float sum = 0.0f;
        for (int i = 0; i < colsA; ++i) {
            sum += A[row * colsA + i] * B[i * colsB + col];
        }
        // Store the result in C[row][col]
        C[row * colsB + col] = sum;
    }
}

__kernel void vectorAddKernel(__global const float* A, __global const float* B,
                              __global float* C, int len)
{
    // Get the global index for the 1D vector
    int idx = get_global_id(0); // Corresponds to x-dimension in launch

    // Check if the work-item is within the bounds of the vector length
    if (idx < len) {
        C[idx] = A[idx] + B[idx];
    }
}

__kernel void vectorsAddKernel(__global const float* A, __global const float* B,
                              __global const float* C, __global const float* D,
                              __global float* E, int len)
{
    // Get the global index for the 1D vector
    int idx = get_global_id(0); // Corresponds to x-dimension in launch

    // Check if the work-item is within the bounds of the vector length
    if (idx < len) {
        E[idx] = A[idx] + B[idx] + C[idx] + D[idx];
    }
}
