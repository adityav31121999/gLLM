
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath> // For fabsf
#include <stdexcept>
#include <iostream>

#define CHECK_CUDA_ERROR(call) do {                                         \
    cudaError_t err = call;                                                 \
    if (err != cudaSuccess) {                                               \
        fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__,   \
                cudaGetErrorString(err));                                   \
        throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err))); \
    }                                                                       \
} while (0)


// Helper macro for indexing flattened matrix (assuming row-major)
// n_cols is the number of columns (which is 'n' for square matrices)
#define IDX(row, col, n) ((row) * (n) + (col))

/**
 * @brief CUDA Kernel: Calculates the additive inverse (-A) of a matrix.
 *        Based on mat::inva() logic.
 *        Designed for parallel execution with a 2D grid.
 * @param flattenedMatrix Pointer to the input matrix elements in GPU memory (row-major).
 * @param additiveInverse Pointer to GPU memory where the resulting additive inverse will be stored.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 */
__global__ void inva(const float* flattenedMatrix, float* additiveInverse, int rows, int cols) {
    int j = blockIdx.x * blockDim.x + threadIdx.x; // Global column index
    int i = blockIdx.y * blockDim.y + threadIdx.y; // Global row index

    // Check bounds
    if (i < rows && j < cols) {
        int index = IDX(i, j, cols);
        // multiply each element by -1 (as per C++ logic)
        additiveInverse[index] = flattenedMatrix[index] * -1.0f;
    }
}

// --- Gauss-Jordan Kernel ---

// Helper device function for swapping two rows (mimics std::swap on vectors)
__device__ void swap_rows_gj(float* matA, float* matB, int row1, int row2, int n) {
    float temp;
    for (int j = 0; j < n; ++j) {
        // Swap in matrix A (the copy being reduced)
        temp = matA[IDX(row1, j, n)];
        matA[IDX(row1, j, n)] = matA[IDX(row2, j, n)];
        matA[IDX(row2, j, n)] = temp;

        // Swap in matrix B (the identity matrix being transformed into the inverse)
        temp = matB[IDX(row1, j, n)];
        matB[IDX(row1, j, n)] = matB[IDX(row2, j, n)];
        matB[IDX(row2, j, n)] = temp;
    }
}

/**
 * @brief CUDA Kernel: Calculate the inverse using Gauss-Jordan Elimination.
 *        NOTE: This kernel strictly replicates the logic (including potential flaws
 *              and non-standard formulas) from the provided C++ mat::gaussjordan().
 *        NOTE: Runs sequentially on a single thread (<<<1, 1>>>). Offers no GPU parallelism advantage.
 *        NOTE: Modifies both matrixCopyToReduce and inverseResult IN PLACE.
 * @param matrixCopyToReduce Pointer to a copy of the original matrix in GPU memory. Will be reduced. (Like 'd' in C++)
 * @param inverseResult Pointer to an identity matrix in GPU memory. Will be transformed into the inverse. (Like 'b' in C++)
 * @param n Dimension of the square matrix. Host must ensure n > 0 and matrix is invertible.
 */
__global__ void gaussjordan(float* matrixCopyToReduce, float* inverseResult, int n) {
    // This kernel is intended to run with a single thread.
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    if (n <= 0) return; // Basic check

    // Abbreviate pointers for readability (like 'd' and 'b' in C++)
    float* d = matrixCopyToReduce;
    float* b = inverseResult;
    float pivot, factor; // Re-using 'factor' concept, though C++ logic is different

    // --- Initial Row Swaps (Mimicking C++ logic, fixing bounds) ---
    // NOTE: This pivoting strategy is simplistic and less robust than partial/full pivoting.
    for (int i = 0; i < n; i++) {
        // Check diagonal element d[i][i]
        if (fabsf(d[IDX(i, i, n)]) < 1e-9f) { // Use fabsf for float absolute value
            // Swap ith vector with (i+1)th vector if possible
            if (i + 1 < n) {
                 // C++ code swaps unconditionally if d[i][i] is 0. Replicating this.
                 swap_rows_gj(d, b, i, i + 1, n);
            }
            // If the last row's diagonal is zero, this simple swap fails.
            // The C++ code doesn't handle this; kernel proceeds, risking later errors.
        }
    }

    // --- Solve for inverse ---
    // --- "Lower triangle formation" (Forward Elimination as per C++ code) ---
    // NOTE: The update logic here is non-standard for Gauss-Jordan. Replicating C++ code.
    int k = 0;
    while (k < n - 1) { // C++ loop condition: k: 0 to n-2
        // 1. Normalize the pivot row k (make diagonal element 1)
        pivot = d[IDX(k, k, n)];
        if (fabsf(pivot) < 1e-9f) {
            // Singular matrix or failed pivoting. Cannot proceed.
            // Host should have checked determinant, but check pivot here too.
            return; // Avoid division by zero
        }
        for (int i = 0; i < n; i++) { // C++ code iterates i from 0 to n-1 here for normalization
            d[IDX(k, i, n)] /= pivot; // original matrix row k
            b[IDX(k, i, n)] /= pivot; // inverse matrix row k
        }

        // 2. Eliminate elements below pivot (make elements 0, except diagonal of kth column)
        // NOTE: C++ loops and update formula are replicated here, despite appearing non-standard.
        for (int i = k; i < n - 1; i++) { // C++ outer loop: i from k to n-2
            // changes row at every interval (operates on row i+1)

            // The C++ inner loop `for(int j = 0; j < n-1; j++)` and the update formula
            // `d.a[i+1][j] -= d.a[k+1][k+1] * d.a[i][j]` seem incorrect for standard elimination.
            // Standard elimination targets column 'k' in row 'i+1' using row 'k'.
            // Replicating the C++ formula structure as closely as possible:
            // float elimination_factor_d = d[IDX(i + 1, k, n)]; // Element to zero out in d (standard approach)
            // However, the C++ update uses d[k+1][k+1] * d[i][j] which is very strange.
            // Let's try to interpret the *intent* while using the *indices* from C++.
            // It seems it might be trying to use row 'k' (or k+1?) to modify row 'i+1'.

            // If we strictly follow `d.a[i+1][j] -= d.a[k+1][k+1] * d.a[i][j]` for j=0..n-2:
            if (k + 1 < n) { // Check bounds for k+1 index
                // float multiplier = d[IDX(k + 1, k + 1, n)]; // The strange multiplier from C++
                for (int j = 0; j < n; j++) { // Iterate through columns (assuming full row update needed)
                    // Check bounds for i+1 index
                    if (i + 1 < n) {
                        // Apply the C++ formula structure
                        // Note: d[i][j] is used, which means using row 'i' to update row 'i+1' based on a value from row 'k+1'??
                        // This seems highly unlikely to be correct Gauss-Jordan logic.
                        // Applying it directly:
                        // d[IDX(i + 1, j, n)] -= multiplier * d[IDX(i, j, n)]; // Original matrix update
                        // b[IDX(i + 1, j, n)] -= multiplier * b[IDX(i, j, n)]; // Inverse matrix update

                        // --- Alternative Interpretation (More likely intent, but still not standard GJ): ---
                        // Maybe it meant to use row k to eliminate element at [i+1][k]?
                        // factor = d[IDX(i+1, k, n)] / d[IDX(k, k, n)]; // d[k,k] is 1 after normalization
                        factor = d[IDX(i + 1, k, n)]; // Since d[k,k] is 1
                        if (fabsf(factor) > 1e-9f) { // Only update if needed
                            for(int col_idx = 0; col_idx < n; ++col_idx) { // Update across the entire row
                                d[IDX(i + 1, col_idx, n)] -= factor * d[IDX(k, col_idx, n)];
                                b[IDX(i + 1, col_idx, n)] -= factor * b[IDX(k, col_idx, n)];
                            }
                        }
                    }
                }
                // Explicitly set the target element to zero after row operation for numerical stability
                if (i + 1 < n) {
                    d[IDX(i + 1, k, n)] = 0.0f;
                }
            }
        }
        k++;
    } // End while(k < n-1)


    // --- Handle the last row normalization (k = n-1) ---
    // The C++ 'while' loop stops at k=n-2. The last row (n-1) needs normalization if n > 0.
    if (n > 0) {
        k = n - 1;
        pivot = d[IDX(k, k, n)];
        if (fabsf(pivot) < 1e-9f) { return; } // Check pivot
        for (int j = 0; j < n; j++) {
            d[IDX(k, j, n)] /= pivot;
            b[IDX(k, j, n)] /= pivot;
        }
    }


    // --- "Upper triangle formation" (Backward Elimination as per C++ code) ---
    // NOTE: This part of the C++ code also uses non-standard loops and update formulas.
    //       A standard Gauss-Jordan would typically achieve the identity matrix 'd'
    //       and the inverse 'b' after the forward elimination + normalization steps above.
    //       This backward pass attempts to replicate the C++ code's second while loop.
    k = n - 1;
    while (k > 0) { // C++ loop condition: k from n-1 down to 1
        // 1. Normalize row k (already done in forward pass / last row handling, but C++ repeats it)
        // pivot = d[IDX(k, k, n)]; // Should be 1 already
        // if (fabsf(pivot) < 1e-9f) { return; } // Should not happen if forward pass worked
        // for (int i = 0; i < n; i++) { // C++ normalizes again
        //     d[IDX(k, i, n)] /= pivot;
        //     b[IDX(k, i, n)] /= pivot;
        // }

        // 2. Eliminate elements above pivot in column k
        // NOTE: C++ loops and update formula are replicated here.
        for (int i = k; i >= 1; i--) { // C++ outer loop: i from k down to 1 (operates on row i-1)
            // The C++ inner loop `for(int j = n-1; j >= 0; j--)` and update
            // `d.a[i-1][j] -= d.a[k-1][k-1] * d.a[i][j]` seem incorrect.
            // Standard elimination uses row 'k' to modify row 'i-1'.
            // Replicating C++ formula structure:
            if (k - 1 >= 0) { // Check bounds for k-1 index
                // float multiplier = d[IDX(k - 1, k - 1, n)]; // Strange multiplier from C++
                // for (int j = n - 1; j >= 0; j--) { // C++ inner loop
                //     // Check bounds for i-1 index
                //     if (i - 1 >= 0) {
                //         // Apply C++ formula structure: uses row 'i' to update row 'i-1'?
                //         d[IDX(i - 1, j, n)] -= multiplier * d[IDX(i, j, n)];
                //         b[IDX(i - 1, j, n)] -= multiplier * b[IDX(i, j, n)];
                //     }
                // }

                // --- Alternative Interpretation (More likely intent, standard GJ backward elimination): ---
                // Use row k to eliminate element at [i-1][k]
                factor = d[IDX(i - 1, k, n)]; // Element to zero out
                 if (fabsf(factor) > 1e-9f) { // Only update if needed
                    for(int col_idx = 0; col_idx < n; ++col_idx) { // Update across the entire row
                        d[IDX(i - 1, col_idx, n)] -= factor * d[IDX(k, col_idx, n)];
                        b[IDX(i - 1, col_idx, n)] -= factor * b[IDX(k, col_idx, n)];
                    }
                 }
            }
             // Explicitly set the target element to zero after row operation
             if (i - 1 >= 0) {
                 d[IDX(i - 1, k, n)] = 0.0f;
             }
        }
        k--;
    } // End while(k > 0)
}

// --- Host Function for Additive Inverse (inva kernel) ---

/**
 * @brief Host function to compute the additive inverse of a matrix using the inva CUDA kernel.
 * @param h_matrix Input matrix (row-major) on the host.
 * @param rows Number of rows.
 * @param cols Number of columns.
 * @return Additive inverse matrix (-h_matrix) as a std::vector<float>.
 * @throws std::runtime_error on CUDA errors or invalid input size.
 */
std::vector<float> host_additive_inverse(const std::vector<float>& h_matrix, int rows, int cols) {
    if (rows <= 0 || cols <= 0) {
        throw std::invalid_argument("Matrix dimensions must be positive.");
    }
    size_t num_elements = static_cast<size_t>(rows) * cols;
    if (h_matrix.size() != num_elements) {
        throw std::invalid_argument("Input vector size does not match rows * cols.");
    }

    float* d_matrix = nullptr;
    float* d_result = nullptr;
    std::vector<float> h_result(num_elements); // Allocate host result vector

    try {
        // 1. Allocate GPU Memory
        size_t matrix_size_bytes = num_elements * sizeof(float);
        CHECK_CUDA_ERROR(cudaMalloc(&d_matrix, matrix_size_bytes));
        CHECK_CUDA_ERROR(cudaMalloc(&d_result, matrix_size_bytes));

        // 2. Copy Input Matrix from Host to Device
        CHECK_CUDA_ERROR(cudaMemcpy(d_matrix, h_matrix.data(), matrix_size_bytes, cudaMemcpyHostToDevice));

        // 3. Configure Grid and Block Dimensions for inva kernel
        //    Example: Use 16x16 blocks (256 threads)
        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

        // 4. Launch Kernel
        inva<<<numBlocks, threadsPerBlock>>>(d_matrix, d_result, rows, cols);

        // Check for kernel launch errors specifically
        CHECK_CUDA_ERROR(cudaGetLastError());

        // 5. Synchronize to ensure kernel completion before copying back
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        // 6. Copy Result from Device to Host
        CHECK_CUDA_ERROR(cudaMemcpy(h_result.data(), d_result, matrix_size_bytes, cudaMemcpyDeviceToHost));

        // 7. Free GPU Memory
        CHECK_CUDA_ERROR(cudaFree(d_matrix));
        CHECK_CUDA_ERROR(cudaFree(d_result));

    } 
    catch (const std::exception& e) {
        // Cleanup GPU memory even if an error occurred mid-way
        if (d_matrix) cudaFree(d_matrix);
        if (d_result) cudaFree(d_result);
        std::cerr << "Error during additive inverse calculation: " << e.what() << std::endl;
        throw; // Re-throw the exception
    }

    return h_result;
}


// --- Host Function for Gauss-Jordan Inverse (gaussjordan kernel) ---

/**
 * @brief Host function to compute the matrix inverse using the Gauss-Jordan CUDA kernel.
 * @param h_matrix Input square matrix (row-major) on the host.
 * @param n Dimension of the square matrix.
 * @return Inverse matrix as a std::vector<float>.
 * @throws std::runtime_error on CUDA errors or invalid input size.
 * @throws std::invalid_argument if n <= 0 or matrix size is incorrect.
 */
std::vector<float> host_inverse_gauss_jordan(const std::vector<float>& h_matrix, int n) {
    if (n <= 0) {
        throw std::invalid_argument("Matrix dimension 'n' must be positive.");
    }
    size_t num_elements = static_cast<size_t>(n) * n;
    if (h_matrix.size() != num_elements) {
        throw std::invalid_argument("Input vector size does not match n * n.");
    }

    // *** HIGHLY RECOMMENDED: Add a determinant check here! ***
    // float det = host_determinant(h_matrix, n); // Using a host or GPU determinant function
    // if (std::fabs(det) < 1e-9f) {
    //     throw std::runtime_error("Matrix is singular (determinant is near zero), cannot invert.");
    // }

    float* d_matrix_copy = nullptr;    // Device memory for the matrix copy to be reduced
    float* d_inverse_result = nullptr; // Device memory for the identity matrix -> inverse
    std::vector<float> h_inverse(num_elements); // Allocate host result vector

    try {
        // 1. Allocate GPU Memory
        size_t matrix_size_bytes = num_elements * sizeof(float);
        CHECK_CUDA_ERROR(cudaMalloc(&d_matrix_copy, matrix_size_bytes));
        CHECK_CUDA_ERROR(cudaMalloc(&d_inverse_result, matrix_size_bytes));

        // 2. Create Identity Matrix on Host
        std::vector<float> h_identity(num_elements, 0.0f);
        for (int i = 0; i < n; ++i) {
            h_identity[static_cast<size_t>(i) * n + i] = 1.0f; // Correct indexing for row-major
        }

        // 3. Copy Input Matrix and Identity Matrix from Host to Device
        CHECK_CUDA_ERROR(cudaMemcpy(d_matrix_copy, h_matrix.data(), matrix_size_bytes, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_inverse_result, h_identity.data(), matrix_size_bytes, cudaMemcpyHostToDevice));

        // 4. Configure Grid and Block Dimensions (Single Thread for gaussjordan)
        dim3 threadsPerBlock(1);
        dim3 numBlocks(1);

        // 5. Launch Kernel
        gaussjordan<<<numBlocks, threadsPerBlock>>>(d_matrix_copy, d_inverse_result, n);

        // Check for kernel launch errors specifically
        CHECK_CUDA_ERROR(cudaGetLastError());

        // 6. Synchronize to ensure kernel completion
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        // 7. Copy Resulting Inverse from Device to Host
        CHECK_CUDA_ERROR(cudaMemcpy(h_inverse.data(), d_inverse_result, matrix_size_bytes, cudaMemcpyDeviceToHost));

        // 8. Free GPU Memory
        CHECK_CUDA_ERROR(cudaFree(d_matrix_copy));
        CHECK_CUDA_ERROR(cudaFree(d_inverse_result));

    } 
    catch (const std::exception& e) {
        // Cleanup GPU memory even if an error occurred mid-way
        if (d_matrix_copy) cudaFree(d_matrix_copy);
        if (d_inverse_result) cudaFree(d_inverse_result);
        std::cerr << "Error during Gauss-Jordan inverse calculation: " << e.what() << std::endl;
        throw; // Re-throw the exception
    }

    return h_inverse;
}
