

/**
 * @brief OpenCL Kernel: Calculates the additive inverse (-A) of a matrix.
 *        Based on mat::inva() logic.
 *        Designed for parallel execution with a 2D global work size.
 * @param flattenedMatrix Pointer to the input matrix elements in global GPU memory (row-major).
 * @param additiveInverse Pointer to global GPU memory where the resulting additive inverse will be stored.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 */
__kernel void inva(__global const float* flattenedMatrix,
                   __global float* additiveInverse,
                   int rows,
                   int cols)
{
    int j = get_global_id(0); // Global column index (maps to x-dimension)
    int i = get_global_id(1); // Global row index (maps to y-dimension)

    // Check bounds
    if (i < rows && j < cols) {
        int index = IDX(i, j, cols);
        // multiply each element by -1
        additiveInverse[index] = flattenedMatrix[index] * -1.0f;
    }
}

// --- Gauss-Jordan Kernel ---

// Helper function for swapping two rows within the OpenCL kernel context
// (Equivalent to __device__ swap_rows_gj)
// Note: Pointers passed from the kernel must be marked __global.
static void swap_rows_gj_cl(__global float* matA, __global float* matB, int row1, int row2, int n) {
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
 * @brief OpenCL Kernel: Calculate the inverse using Gauss-Jordan Elimination.
 *        NOTE: This kernel strictly replicates the logic (including potential flaws
 *              and non-standard formulas) from the provided C++ mat::gaussjordan()
 *              as implemented in the CUDA kernel.
 *        NOTE: Runs sequentially on a single work-item (requires global_size[0] >= 1).
 *              Offers no GPU parallelism advantage.
 *        NOTE: Modifies both matrixCopyToReduce and inverseResult IN PLACE.
 * @param matrixCopyToReduce Pointer to a copy of the original matrix in global GPU memory. Will be reduced.
 * @param inverseResult Pointer to an identity matrix in global GPU memory. Will be transformed into the inverse.
 * @param n Dimension of the square matrix. Host must ensure n > 0 and matrix is invertible.
 */
__kernel void gaussjordan(__global float* matrixCopyToReduce,
                          __global float* inverseResult,
                          int n)
{
    // This kernel is intended to run with a single work-item.
    if (get_global_id(0) != 0) return;
    if (n <= 0) return; // Basic check

    // Abbreviate pointers for readability
    __global float* d = matrixCopyToReduce;
    __global float* b = inverseResult;
    float pivot, factor;

    // --- Initial Row Swaps (Mimicking CUDA/C++ logic) ---
    for (int i = 0; i < n; i++) {
        // Check diagonal element d[i][i]
        if (fabs(d[IDX(i, i, n)]) < 1e-9f) { // Use fabs for float absolute value in OpenCL
            // Swap ith vector with (i+1)th vector if possible
            if (i + 1 < n) {
                 swap_rows_gj_cl(d, b, i, i + 1, n);
            }
            // If the last row's diagonal is zero, this simple swap fails.
        }
    }

    // --- Solve for inverse ---
    // --- "Lower triangle formation" (Forward Elimination as per CUDA/C++ code) ---
    // NOTE: Using the "Alternative Interpretation" from the CUDA kernel comments,
    //       which represents standard elimination steps, as the literal C++ formula
    //       replication seemed incorrect for the task.
    int k = 0;
    while (k < n - 1) { // Loop condition: k: 0 to n-2
        // 1. Normalize the pivot row k
        pivot = d[IDX(k, k, n)];
        if (fabs(pivot) < 1e-9f) {
            // Singular matrix or failed pivoting.
            return; // Avoid division by zero
        }
        for (int j = 0; j < n; j++) { // Normalize full row k
            d[IDX(k, j, n)] /= pivot;
            b[IDX(k, j, n)] /= pivot;
        }

        // 2. Eliminate elements below pivot in column k
        for (int i = k + 1; i < n; i++) { // Operate on rows below k
            factor = d[IDX(i, k, n)]; // Element to zero out in column k, row i
            if (fabs(factor) > 1e-9f) { // Only update if needed
                for(int col_idx = 0; col_idx < n; ++col_idx) { // Update across the entire row i
                    d[IDX(i, col_idx, n)] -= factor * d[IDX(k, col_idx, n)];
                    b[IDX(i, col_idx, n)] -= factor * b[IDX(k, col_idx, n)];
                }
            }
             // Explicitly set the target element to zero for numerical stability
            d[IDX(i, k, n)] = 0.0f;
        }
        k++;
    } // End while(k < n-1)


    // --- Handle the last row normalization (k = n-1) ---
    if (n > 0) {
        k = n - 1;
        pivot = d[IDX(k, k, n)];
        if (fabs(pivot) < 1e-9f) { return; } // Check pivot
        for (int j = 0; j < n; j++) {
            d[IDX(k, j, n)] /= pivot;
            b[IDX(k, j, n)] /= pivot;
        }
    }


    // --- "Upper triangle formation" (Backward Elimination) ---
    // Standard Gauss-Jordan backward elimination to zero out elements above the diagonal.
    k = n - 1;
    while (k > 0) { // k from n-1 down to 1
        // Eliminate elements above pivot in column k
        for (int i = k - 1; i >= 0; i--) { // Operate on rows above k (i.e., row i)
            factor = d[IDX(i, k, n)]; // Element to zero out in column k, row i
            if (fabs(factor) > 1e-9f) { // Only update if needed
                for(int col_idx = 0; col_idx < n; ++col_idx) { // Update across the entire row i
                    d[IDX(i, col_idx, n)] -= factor * d[IDX(k, col_idx, n)];
                    b[IDX(i, col_idx, n)] -= factor * b[IDX(k, col_idx, n)];
                }
            }
            // Explicitly set the target element to zero
            d[IDX(i, k, n)] = 0.0f;
        }
        k--;
    } // End while(k > 0)
}

