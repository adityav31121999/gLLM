
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath> // For fabsf, sqrtf, powf
#include "include/stats.hpp"

// --- Reduction Helper (Single Block/Workgroup) ---
// Reduces values in shared memory array 'sdata'.
// Assumes blockDim.x is power of 2, sdata size >= blockDim.x
__device__ inline void reduce_sum_shared(volatile float* sdata) {
    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            // Corrected: Avoid compound assignment on volatile destination
            // Read the value to add separately
            float value_to_add = sdata[threadIdx.x + s];
            // Perform the read-modify-write explicitly
            sdata[threadIdx.x] = sdata[threadIdx.x] + value_to_add;
            // Or simply:
            // sdata[threadIdx.x] = sdata[threadIdx.x] + sdata[threadIdx.x + s];
        }
        __syncthreads(); // Synchronize within the block
    }
}

// --- Mean Kernel (Single Block Reduction) ---
// NOTE: Assumes input size N fits within one block's processing capability.
//       Requires shared memory allocation: blockDim.x * sizeof(float)
__global__ void mean_kernel(const float* vec, int N, float* result) {
    extern __shared__ float sdata[]; // Shared memory for reduction

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; // Global index

    // Load data into shared memory
    // Each thread loads multiple elements if N > blockDim.x (Grid-Stride Loop)
    float thread_sum = 0.0f;
    while (i < N) {
        thread_sum += vec[i];
        i += gridDim.x * blockDim.x; // Move to next element this thread handles
    }
    sdata[tid] = thread_sum;
    __syncthreads();

    // Perform reduction in shared memory
    reduce_sum_shared(sdata); // Use helper

    // Thread 0 writes the final result
    if (tid == 0) {
        *result = sdata[0] / (float)N;
    }
}

// --- Variance/StdDev Kernel (Single Block Reduction) ---
// Calculates variance. StdDev is sqrt(variance).
// NOTE: Assumes input size N fits within one block. Requires pre-calculated mean.
//       Requires shared memory allocation: blockDim.x * sizeof(float)
__global__ void variance_kernel(const float* vec, int N, float mean_val, float* variance_result) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Calculate sum of squared differences
    float thread_sum_sq_diff = 0.0f;
    while (i < N) {
        float diff = vec[i] - mean_val;
        thread_sum_sq_diff += diff * diff;
        i += gridDim.x * blockDim.x;
    }
    sdata[tid] = thread_sum_sq_diff;
    __syncthreads();

    // Reduce the sum of squared differences
    reduce_sum_shared(sdata);

    // Thread 0 calculates final variance
    if (tid == 0) {
        *variance_result = sdata[0] / (float)N;
    }
}

// --- Covariance Kernel (Single Block Reduction) ---
// NOTE: Assumes input size N fits within one block. Requires pre-calculated means.
//       Requires shared memory allocation: blockDim.x * sizeof(float)
__global__ void covariance_kernel(const float* vec_a, const float* vec_b, int N, float mean_a, float mean_b, float* covariance_result) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Calculate sum of product of differences
    float thread_sum_prod_diff = 0.0f;
    while (i < N) {
        float diff_a = vec_a[i] - mean_a;
        float diff_b = vec_b[i] - mean_b;
        thread_sum_prod_diff += diff_a * diff_b;
        i += gridDim.x * blockDim.x;
    }
    sdata[tid] = thread_sum_prod_diff;
    __syncthreads();

    // Reduce the sum
    reduce_sum_shared(sdata);

    // Thread 0 calculates final covariance
    if (tid == 0) {
        *covariance_result = sdata[0] / (float)N;
    }
}

// --- Correlation Kernel (Simple Element-wise) ---
// Assumes covariance and standard deviations are pre-calculated.
__global__ void correlation_kernel(float covariance_ab, float stddev_a, float stddev_b, float* correlation_result) {
    // Only one thread needs to do this simple calculation
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        if (fabsf(stddev_a) < 1e-9f || fabsf(stddev_b) < 1e-9f) {
            *correlation_result = 0.0f; // Avoid division by zero, correlation is undefined/0
        } else {
            *correlation_result = covariance_ab / (stddev_a * stddev_b);
        }
    }
}
// Pearson is the same calculation as correlation.

// --- Rank Kernel (Vector Input/Output for Spearman) ---
// Calculates rank for each element of vec_in, stores in vec_rank_out.
// NOTE: This is O(N^2) work distributed across N threads. Inefficient for large N.
//       Better approach involves sorting.
__global__ void rank_vector_kernel(const float* vec_in, int N, float* vec_rank_out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Global index for output rank

    if (i < N) {
        float val_i = vec_in[i];
        int count = 0;
        // Each thread iterates through the *entire* input vector
        for (int j = 0; j < N; ++j) {
            if (vec_in[j] < val_i) {
                count++;
            }
            // Handling ties (optional, C++ code doesn't specify):
            // Could add 0.5 for each element equal to val_i (excluding self)
        }
        vec_rank_out[i] = (float)count; // Store the rank
    }
}

// --- Expectation Kernel (Single Block Reduction) ---
// NOTE: Assumes input size N fits within one block.
//       Requires shared memory allocation: blockDim.x * sizeof(float)
__global__ void expectation_kernel(const float* vec, const float* prob, int N, float* expectation_result) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Calculate weighted sum contribution
    float thread_weighted_sum = 0.0f;
    while (i < N) {
        thread_weighted_sum += vec[i] * prob[i];
        i += gridDim.x * blockDim.x;
    }
    sdata[tid] = thread_weighted_sum;
    __syncthreads();

    // Reduce the sum
    reduce_sum_shared(sdata);

    // Thread 0 writes final result
    if (tid == 0) {
        *expectation_result = sdata[0];
    }
}

// --- Median/Percentile/Quartile Kernel (Requires Pre-Sorted Data) ---
// Calculates value at index corresponding to percentile p (0-100).
// Median p=50, Q1 p=25, Q3 p=75.
// NOTE: Input 'sorted_vec' MUST be sorted beforehand (e.g., using Thrust).
__global__ void percentile_kernel(const float* sorted_vec, int N, float p, float* result) {
    // Only one thread needs to do this
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        if (N <= 0) {
            *result = 0.0f; // Or NaN
            return;
        }
        // Calculate index (linear interpolation per C++ logic)
        float index_f = p / 100.0f * (float)(N - 1);
        int index_i = (int)floorf(index_f); // Use floorf

        // Clamp index to valid range [0, N-1]
        index_i = max(0, min(N - 1, index_i));

        // C++ code directly uses the possibly fractional index, implying floor.
        // If interpolation between indices is needed:
        // float frac = index_f - index_i;
        // if (index_i + 1 < N && frac > 1e-6f) {
        //     *result = sorted_vec[index_i] * (1.0f - frac) + sorted_vec[index_i + 1] * frac;
        // } else {
        //     *result = sorted_vec[index_i];
        // }

        // Simpler version matching C++ direct access:
         *result = sorted_vec[index_i];
    }
}

// --- Z-Score Kernel (Element-wise) ---
// Calculates z-score for each element. Requires pre-calculated mean and stddev.
__global__ void zscore_vector_kernel(const float* vec_in, int N, float mean_val, float stddev_val, float* vec_zscore_out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        if (fabsf(stddev_val) < 1e-9f) {
            vec_zscore_out[i] = 0.0f; // Avoid division by zero
        } else {
            vec_zscore_out[i] = (vec_in[i] - mean_val) / stddev_val;
        }
    }
}

// --- Outlier Kernel (Element-wise Check) ---
// Checks if each element is an outlier using IQR method. Outputs z-score if outlier, 0 otherwise.
// Requires pre-calculated Q1, Q3, IQR, mean, stddev.
__global__ void outlier_vector_kernel(const float* vec_in, int N, float q1, float q3, float iqr, float mean_val, float stddev_val, float* vec_outlier_zscore_out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        float lowerBound = q1 - 1.5f * iqr;
        float upperBound = q3 + 1.5f * iqr;
        float val = vec_in[i];

        if (val < lowerBound || val > upperBound) {
            // It's an outlier, calculate and store its z-score
            if (fabsf(stddev_val) < 1e-9f) {
                vec_outlier_zscore_out[i] = 0.0f; // Avoid division by zero
            } else {
                vec_outlier_zscore_out[i] = (val - mean_val) / stddev_val;
            }
        } else {
            // Not an outlier
            vec_outlier_zscore_out[i] = 0.0f;
        }
    }
}

