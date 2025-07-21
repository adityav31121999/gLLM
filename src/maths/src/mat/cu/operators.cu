#ifdef USE_CUDA

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <stdexcept>
#include "mat.hpp" // Include the mat class definition

#define CUDA_CHECK_ERROR(call)                                          \
do {                                                                    \
    cudaError_t err = call;                                             \
    if (err != cudaSuccess) {                                           \
        fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err));                               \
        /* Optionally, perform cleanup before throwing */               \
        throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err))); \
    }                                                                   \
} while (0)

#define IDX(row, col, N_COLS) ((row) * (N_COLS) + (col))

/**
 * @brief CUDA Kernel: Calculates the dot product of each row of a matrix with a vector.
 *        Computes vecOutput = matrix * vec (Matrix-Vector Multiplication).
 *        Based on C++: std::vector<float> dot(std::vector<float> a, mat b)
 *        Uses the signature from operators.cu.
 * @param vec Pointer to the input vector 'a' in GPU memory.
 * @param matrix Pointer to the input matrix 'b' in GPU memory (row-major).
 * @param vecOutput Pointer to the output vector 'c' in GPU memory.
 * @param veclength Size of the input vector 'vec' (N). Must match matrixColumn.
 * @param matrixRow Number of rows in 'matrix' (M). Size of 'vecOutput'.
 * @param matrixColumn Number of columns in 'matrix' (N). Must match veclength.
 */
__global__ void dot(float* vec, float* matrix, float* vecOutput, int veclength, int matrixRow, int matrixColumn) {
    // Calculate the global thread ID, corresponding to the row index 'i' of the matrix
    // and the index 'i' of the output vector 'vecOutput'.
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Check dimensions consistency (optional, better done on host)
    // if (veclength != matrixColumn) return; // Basic check

    // Check if the thread ID is within the bounds of the output vector size (matrixRow)
    if (i < matrixRow) {
        float sum = 0.0f;
        // Calculate the dot product of the i-th row of 'matrix' with 'vec'
        for (int j = 0; j < matrixColumn; ++j) { // matrixColumn is N (veclength)
            // Access element matrix[i][j]
            sum += matrix[IDX(i, j, matrixColumn)] * vec[j];
        }
        // Store the result in the i-th element of the output vector
        vecOutput[i] = sum;
    }
}


/**
 * @brief CUDA Kernel: Computes the scalar dot product: result = vec1 * matrix * vec2_transpose.
 *        Based on C++: void computeDot(std::vector<float>& T1, std::vector<float>& T2, std::vector<std::vector<float>>& M, float& dot)
 *        Uses the signature from operators.cu, interpreting 'scalarValue' as an output pointer.
 *        NOTE: Requires launch with a SINGLE BLOCK and shared memory allocation.
 *              Launch: <<<1, numThreads, sharedMemBytes>>>
 *              numThreads >= N (dimension), ideally power of 2.
 *              sharedMemBytes = numThreads * sizeof(float).
 * @param vec1 Pointer to input vector T1 in GPU memory.
 * @param matrix Pointer to input matrix M in GPU memory (row-major, N x N).
 * @param vec2 Pointer to input vector T2 in GPU memory.
 * @param d_scalarOutput Pointer to a single float in GPU memory for the output scalar result. (Interpreted from 'scalarValue').
 * @param vec1length Dimension N (size of vec1).
 * @param matrixRow Dimension N (rows of matrix). Must match vec1length, matrixColumn, vec2length.
 * @param matrixColumn Dimension N (columns of matrix). Must match vec1length, matrixRow, vec2length.
 * @param vec2length Dimension N (size of vec2). Must match vec1length, matrixRow, matrixColumn.
 */
__global__ void dot(float* vec1, float* matrix, float* vec2, float* d_scalarOutput, int vec1length, int matrixRow, int matrixColumn, int vec2length) {
    // Assume N = vec1length = matrixRow = matrixColumn = vec2length
    // Host should verify these dimensions before launch.
    int N = vec1length;

    // Allocate shared memory dynamically based on launch configuration
    // Size needs to be at least N, but reduction works best with blockDim.x (power of 2)
    extern __shared__ float sdata[];

    // Thread index within the block
    int tid = threadIdx.x;

    // --- Step 1: Calculate intermediate value temp[tid] = dot(vec1, matrix[tid]) ---
    //    Then multiply by vec2[tid] and store in shared memory for reduction.
    if (tid < N) {
        float temp_i = 0.0f;
        // Calculate dot(vec1, matrix[tid])
        for (int k = 0; k < N; ++k) {
            // Access matrix[tid][k]
            temp_i += vec1[k] * matrix[IDX(tid, k, N)];
        }
        // Calculate the contribution to the final dot product: temp[tid] * vec2[tid]
        // Store this contribution in shared memory.
        sdata[tid] = temp_i * vec2[tid];
    } else {
        // Threads outside the range [0, N-1] but within the block need to initialize
        // their shared memory slot if they participate in the reduction.
         if (tid < blockDim.x) { // Check if tid is a valid thread index in the block
             sdata[tid] = 0.0f;
         }
    }

    // Synchronize all threads within the block to ensure all sdata values are written
    __syncthreads();

    // --- Step 2: Parallel Reduction in Shared Memory ---
    // Reduce the values stored in sdata using blockDim.x threads.
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        // Synchronize after each step of the reduction
        __syncthreads();
    }

    // --- Step 3: Write Result ---
    // Thread 0 writes the final sum (located in sdata[0]) to the output pointer in global memory
    if (tid == 0) {
        *d_scalarOutput = sdata[0];
    }
}

// --- Host Function for Matrix-Vector Dot Product (Accepts 2D Vector Matrix) ---

/**
 * @brief Host function to compute matrix-vector product using the 'dot_matrix_vector' CUDA kernel.
 *        Calculates result = matrix * vec.
 * @param h_vec Host input vector.
 * @param input_mat Host input matrix object.
 * @return Resulting vector as a new mat object (Mx1 matrix).
 * @throws std::invalid_argument if dimensions mismatch, matrix is ragged, or dimensions are non-positive.
 * @throws std::runtime_error on CUDA errors.
 */
mat host_dot(const std::vector<float>& h_vec, const mat& input_mat)
{
    // --- Validation ---
    if (input_mat.row <= 0 || input_mat.col <= 0 || !input_mat.mapped_data) {
        throw std::invalid_argument("Input matrix must have positive dimensions and be mapped.");
    }
    int matrixRow = input_mat.row;
    int matrixColumn = input_mat.col;
    if (matrixColumn <= 0) { // Redundant check, but keep for clarity
        throw std::invalid_argument("Matrix must have at least one column.");
    }

    int veclength = matrixColumn; // Vector length must match matrix columns
    if (h_vec.size() != static_cast<size_t>(veclength)) {
        throw std::invalid_argument("Input vector size does not match matrix column count.");
    }

    // No need to flatten, input_mat.mapped_data is already flat.

    // --- GPU Resource Allocation ---
    float *d_vec = nullptr, *d_matrix = nullptr, *d_vecOutput = nullptr;
    // Create result matrix (Mx1)
    mat result_mat(matrixRow, 1);
    if (!result_mat.mapped_data) { throw std::runtime_error("Failed to create result matrix."); }

    try {
        size_t vec_size_bytes = static_cast<size_t>(veclength) * sizeof(float);
        size_t matrix_size_bytes = static_cast<size_t>(input_mat.row) * input_mat.col * sizeof(float); // Use input_mat properties
        size_t output_vec_size_bytes = static_cast<size_t>(matrixRow) * sizeof(float);

        CUDA_CHECK_ERROR(cudaMalloc(&d_vec, vec_size_bytes));
        CUDA_CHECK_ERROR(cudaMalloc(&d_matrix, matrix_size_bytes));
        CUDA_CHECK_ERROR(cudaMalloc(&d_vecOutput, output_vec_size_bytes));

        // --- Data Transfer (Host to Device) ---
        CUDA_CHECK_ERROR(cudaMemcpy(d_vec, h_vec.data(), vec_size_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK_ERROR(cudaMemcpy(d_matrix, input_mat.mapped_data, matrix_size_bytes, cudaMemcpyHostToDevice));

        // --- Kernel Launch Configuration ---
        int threadsPerBlock = 256; // Or another suitable block size
        int blocksPerGrid = (matrixRow + threadsPerBlock - 1) / threadsPerBlock;
        dim3 gridDim(blocksPerGrid);
        dim3 blockDim(threadsPerBlock);

        // --- Kernel Launch ---
        // Use the correct kernel name
        dot<<<gridDim, blockDim>>>(d_vec, d_matrix, d_vecOutput, veclength, matrixRow, matrixColumn);
        CUDA_CHECK_ERROR(cudaGetLastError()); // Check for launch errors

        // --- Synchronization ---
        CUDA_CHECK_ERROR(cudaDeviceSynchronize());

        // --- Data Transfer (Device to Host) into result_mat's mapped memory ---
        CUDA_CHECK_ERROR(cudaMemcpy(result_mat.mapped_data, d_vecOutput, output_vec_size_bytes, cudaMemcpyDeviceToHost));

        // --- GPU Resource Cleanup ---
        CUDA_CHECK_ERROR(cudaFree(d_vec));
        CUDA_CHECK_ERROR(cudaFree(d_matrix));
        CUDA_CHECK_ERROR(cudaFree(d_vecOutput));

    } catch (const std::exception& e) {
        // Ensure cleanup even if error occurs
        if (d_vec) cudaFree(d_vec);
        if (d_matrix) cudaFree(d_matrix);
        if (d_vecOutput) cudaFree(d_vecOutput);
        std::cerr << "Error during matrix-vector dot product (2D input): " << e.what() << std::endl;
        throw; // Re-throw
    }

    return result_mat;
}

// --- Host Function for Vector-Matrix-Vector Dot Product (Accepts 2D Vector Matrix) ---

/**
 * @brief Host function to compute scalar = vec1 * matrix * vec2_transpose using the second 'dot' CUDA kernel.
 *        Accepts matrix as a mat object. Vectors remain std::vector for now.
 * @param h_vec1 Host input vector 1 (T1).
 * @param input_mat Host input matrix M (must be square N x N).
 * @param h_vec2 Host input vector 2 (T2).
 * @return Resulting scalar dot product.
 * @throws std::invalid_argument if dimensions mismatch, matrix not square, matrix ragged, or N is non-positive.
 * @throws std::runtime_error on CUDA errors or if N exceeds kernel limitations.
 */
float host_dot(const std::vector<float>& h_vec1,
    const mat& input_mat, const std::vector<float>& h_vec2)
{
    // --- Validation ---
    if (!input_mat.ifsquare() || !input_mat.mapped_data) {
        throw std::invalid_argument("Input matrix must be square and mapped.");
    }
    int N = input_mat.row; // The common dimension
    if (N <= 0) {
        throw std::invalid_argument("Matrix must have at least one column.");
    }

    size_t expected_vec_size = static_cast<size_t>(N);
    if (h_vec1.size() != expected_vec_size || h_vec2.size() != expected_vec_size) {
        throw std::invalid_argument("Input vector sizes do not match matrix dimension N.");
    }

    // No need to flatten matrix
    /*
    std::vector<float> h_matrix_flat(static_cast<size_t>(N) * N);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            h_matrix_flat[static_cast<size_t>(i) * N + j] = h_matrix_2d[i][j];
        }
    }
    */
    // --- GPU Resource Allocation ---
    float *d_vec1 = nullptr, *d_matrix = nullptr, *d_vec2 = nullptr, *d_scalarOutput = nullptr;
    float h_scalarOutput = 0.0f; // Host variable for the final result

    try {
        size_t vec_size_bytes = expected_vec_size * sizeof(float);
        size_t matrix_size_bytes = static_cast<size_t>(input_mat.row) * input_mat.col * sizeof(float); // Use input_mat properties
        size_t scalar_size_bytes = sizeof(float); // Corrected size calculation

        CUDA_CHECK_ERROR(cudaMalloc(&d_vec1, vec_size_bytes));
        CUDA_CHECK_ERROR(cudaMalloc(&d_matrix, matrix_size_bytes));
        CUDA_CHECK_ERROR(cudaMalloc(&d_vec2, vec_size_bytes));
        CUDA_CHECK_ERROR(cudaMalloc(&d_scalarOutput, scalar_size_bytes));

        // --- Data Transfer (Host to Device) ---
        CUDA_CHECK_ERROR(cudaMemcpy(d_vec1, h_vec1.data(), vec_size_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK_ERROR(cudaMemcpy(d_matrix, input_mat.mapped_data, matrix_size_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK_ERROR(cudaMemcpy(d_vec2, h_vec2.data(), vec_size_bytes, cudaMemcpyHostToDevice));

        // --- Kernel Launch Configuration (Single Block Reduction) ---
        unsigned int numThreads = 1;
        while(numThreads < N && numThreads < 1024) numThreads *= 2;
        numThreads = std::max((unsigned int)N, numThreads);

        cudaDeviceProp prop;
        CUDA_CHECK_ERROR(cudaGetDeviceProperties(&prop, 0));
        if (numThreads > (unsigned int)prop.maxThreadsPerBlock) {
            numThreads = prop.maxThreadsPerBlock;
            if (N > prop.maxThreadsPerBlock) {
                throw std::runtime_error("Dimension N exceeds maximum threads per block supported by this kernel implementation.");
            }
            std::cerr << "Warning: Clamping threads to maxThreadsPerBlock (" << numThreads << ")" << std::endl;
        }

        size_t sharedMemBytes = numThreads * sizeof(float);
        dim3 gridDim(1);
        dim3 blockDim(numThreads);

        // --- Kernel Launch ---
        // Use the correct kernel name
        dot<<<gridDim, blockDim, sharedMemBytes>>>(d_vec1, d_matrix, d_vec2, d_scalarOutput, N, N, N, N);
        CUDA_CHECK_ERROR(cudaGetLastError());
        // --- Synchronization ---
        CUDA_CHECK_ERROR(cudaDeviceSynchronize());
        // --- Data Transfer (Device to Host) ---
        CUDA_CHECK_ERROR(cudaMemcpy(&h_scalarOutput, d_scalarOutput, scalar_size_bytes, cudaMemcpyDeviceToHost));

        // --- GPU Resource Cleanup ---
        CUDA_CHECK_ERROR(cudaFree(d_vec1));
        CUDA_CHECK_ERROR(cudaFree(d_matrix));
        CUDA_CHECK_ERROR(cudaFree(d_vec2));
        CUDA_CHECK_ERROR(cudaFree(d_scalarOutput));

    } catch (const std::exception& e) {
        // Ensure cleanup even if error occurs
        if (d_vec1) cudaFree(d_vec1);
        if (d_matrix) cudaFree(d_matrix);
        if (d_vec2) cudaFree(d_vec2);
        if (d_scalarOutput) cudaFree(d_scalarOutput);
        std::cerr << "Error during vector-matrix-vector dot product (2D input): " << e.what() << std::endl;
        throw; // Re-throw
    }

    return h_scalarOutput;
}

#endif