// Helper macro for indexing flattened matrix (assuming row-major)
#define IDX(row, col, dim) ((row) * (dim) + (col))

// Enable extensions for atomics and potentially double precision (which might include float atomics)
// #pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
// #pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable
// #pragma OPENCL EXTENSION cl_khr_fp64 : enable // For double support
// #pragma OPENCL EXTENSION cl_khr_float_atomics : enable // Not supported on target, using manual implementation

// --- Reduction Helper (Single Workgroup) ---
// Reduces values in local memory array 'sdata'.
// Assumes get_local_size(0) is power of 2, sdata size >= get_local_size(0)
inline void reduce_sum_local(__local volatile float* sdata, uint lsz) {
    // Perform reduction in local memory
    for (unsigned int s = lsz / 2; s > 0; s >>= 1) {
        if (get_local_id(0) < s) {
            sdata[get_local_id(0)] += sdata[get_local_id(0) + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE); // Synchronize within the workgroup
    }
}

// --- Mean Kernel (Single Workgroup Reduction) ---
// NOTE: Assumes input size N fits within one workgroup's processing capability.
//       Requires local memory allocation: get_local_size(0) * sizeof(float)
__kernel void mean_kernel(__global const float* vec, int N, __global float* result, __local float* sdata) {
    size_t tid = get_local_id(0);
    size_t gid = get_global_id(0);
    size_t lsz = get_local_size(0);
    size_t gsz = get_global_size(0); // Total global size

    // Load data into local memory (Grid-Stride Loop)
    float thread_sum = 0.0f;
    size_t i = gid;
    while (i < N) {
        thread_sum += vec[i];
        i += gsz; // Move to next element this work-item handles
    }
    sdata[tid] = thread_sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Perform reduction in local memory
    reduce_sum_local(sdata, lsz); // Use helper

    // Work-item 0 writes the final result
    if (tid == 0) {
        *result = sdata[0] / (float)N;
    }
}

// --- Variance/StdDev Kernel (Single Workgroup Reduction) ---
// Calculates variance. StdDev is sqrt(variance).
// NOTE: Assumes input size N fits within one workgroup. Requires pre-calculated mean.
//       Requires local memory allocation: get_local_size(0) * sizeof(float)
__kernel void variance_kernel(__global const float* vec, int N, float mean_val, __global float* variance_result, __local float* sdata) {
    size_t tid = get_local_id(0);
    size_t gid = get_global_id(0);
    size_t lsz = get_local_size(0);
    size_t gsz = get_global_size(0);

    // Calculate sum of squared differences
    float thread_sum_sq_diff = 0.0f;
    size_t i = gid;
    while (i < N) {
        float diff = vec[i] - mean_val;
        thread_sum_sq_diff += diff * diff;
        i += gsz;
    }
    sdata[tid] = thread_sum_sq_diff;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Reduce the sum of squared differences
    reduce_sum_local(sdata, lsz);

    // Work-item 0 calculates final variance
    if (tid == 0) {
        *variance_result = sdata[0] / (float)N;
    }
}

// --- Covariance Kernel (Single Workgroup Reduction) ---
// NOTE: Assumes input size N fits within one workgroup. Requires pre-calculated means.
//       Requires local memory allocation: get_local_size(0) * sizeof(float)
__kernel void covariance_kernel(__global const float* vec_a, __global const float* vec_b, int N, float mean_a, float mean_b, __global float* covariance_result, __local float* sdata) {
    size_t tid = get_local_id(0);
    size_t gid = get_global_id(0);
    size_t lsz = get_local_size(0);
    size_t gsz = get_global_size(0);

    // Calculate sum of product of differences
    float thread_sum_prod_diff = 0.0f;
    size_t i = gid;
    while (i < N) {
        float diff_a = vec_a[i] - mean_a;
        float diff_b = vec_b[i] - mean_b;
        thread_sum_prod_diff += diff_a * diff_b;
        i += gsz;
    }
    sdata[tid] = thread_sum_prod_diff;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Reduce the sum
    reduce_sum_local(sdata, lsz);

    // Work-item 0 calculates final covariance
    if (tid == 0) {
        *covariance_result = sdata[0] / (float)N;
    }
}

// --- Correlation Kernel (Simple Element-wise) ---
// Assumes covariance and standard deviations are pre-calculated.
__kernel void correlation_kernel(float covariance_ab, float stddev_a, float stddev_b, __global float* correlation_result) {
    // Only one work-item needs to do this simple calculation
    if (get_global_id(0) == 0) {
        if (fabs(stddev_a) < 1e-9f || fabs(stddev_b) < 1e-9f) {
            *correlation_result = 0.0f; // Avoid division by zero
        } else {
            *correlation_result = covariance_ab / (stddev_a * stddev_b);
        }
    }
}
// Pearson is the same calculation as correlation.

// --- Rank Kernel (Vector Input/Output for Spearman) ---
// Calculates rank for each element of vec_in, stores in vec_rank_out.
// NOTE: This is O(N^2) work distributed across N work-items. Inefficient for large N.
//       Better approach involves sorting.
__kernel void rank_vector_kernel(__global const float* vec_in, int N, __global float* vec_rank_out) {
    size_t i = get_global_id(0); // Global index for output rank

    if (i < N) {
        float val_i = vec_in[i];
        int count = 0;
        // Each work-item iterates through the *entire* input vector
        for (int j = 0; j < N; ++j) {
            if (vec_in[j] < val_i) {
                count++;
            }
            // Handling ties (optional)
        }
        vec_rank_out[i] = (float)count; // Store the rank
    }
}

// --- Expectation Kernel (Single Workgroup Reduction) ---
// NOTE: Assumes input size N fits within one workgroup.
//       Requires local memory allocation: get_local_size(0) * sizeof(float)
__kernel void expectation_kernel(__global const float* vec, __global const float* prob, int N, __global float* expectation_result, __local float* sdata) {
    size_t tid = get_local_id(0);
    size_t gid = get_global_id(0);
    size_t lsz = get_local_size(0);
    size_t gsz = get_global_size(0);

    // Calculate weighted sum contribution
    float thread_weighted_sum = 0.0f;
    size_t i = gid;
    while (i < N) {
        thread_weighted_sum += vec[i] * prob[i];
        i += gsz;
    }
    sdata[tid] = thread_weighted_sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Reduce the sum
    reduce_sum_local(sdata, lsz);

    // Work-item 0 writes final result
    if (tid == 0) {
        *expectation_result = sdata[0];
    }
}

// --- Median/Percentile/Quartile Kernel (Requires Pre-Sorted Data) ---
// Calculates value at index corresponding to percentile p (0-100).
// Median p=50, Q1 p=25, Q3 p=75.
// NOTE: Input 'sorted_vec' MUST be sorted beforehand.
__kernel void percentile_kernel(__global const float* sorted_vec, int N, float p, __global float* result) {
    // Only one work-item needs to do this
    if (get_global_id(0) == 0) {
         if (N <= 0) {
            *result = 0.0f; // Or NaN
            return;
        }
        // Calculate index (linear interpolation per C++ logic)
        float index_f = p / 100.0f * (float)(N - 1);
        int index_i = (int)floor(index_f); // Use OpenCL floor

        // Clamp index to valid range [0, N-1]
        index_i = max(0, min(N - 1, index_i));

        // Simpler version matching C++ direct access:
        *result = sorted_vec[index_i];
    }
}

// --- Z-Score Kernel (Element-wise) ---
// Calculates z-score for each element. Requires pre-calculated mean and stddev.
__kernel void zscore_vector_kernel(__global const float* vec_in, int N, float mean_val, float stddev_val, __global float* vec_zscore_out) {
    size_t i = get_global_id(0);

    if (i < N) {
        if (fabs(stddev_val) < 1e-9f) {
            vec_zscore_out[i] = 0.0f; // Avoid division by zero
        } else {
            vec_zscore_out[i] = (vec_in[i] - mean_val) / stddev_val;
        }
    }
}

// --- Outlier Kernel (Element-wise Check) ---
// Checks if each element is an outlier using IQR method. Outputs z-score if outlier, 0 otherwise.
// Requires pre-calculated Q1, Q3, IQR, mean, stddev.
__kernel void outlier_vector_kernel(__global const float* vec_in, int N, float q1, float q3, float iqr, float mean_val, float stddev_val, __global float* vec_outlier_zscore_out) {
    size_t i = get_global_id(0);

    if (i < N) {
        float lowerBound = q1 - 1.5f * iqr;
        float upperBound = q3 + 1.5f * iqr;
        float val = vec_in[i];

        if (val < lowerBound || val > upperBound) {
            // It's an outlier, calculate and store its z-score
            if (fabs(stddev_val) < 1e-9f) {
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

