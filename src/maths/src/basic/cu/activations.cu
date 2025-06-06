
// activations and its derivative functions
#include "include/basic.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <cfloat>


/**
 * @brief cuda function for sigmoid
 * @param[in] x input
 * @param[out] result sigmoid(x)
 */
__global__ void cuSigmoid(float x, float* result) {
    *result = 1.0f / (1.0f + expf(-x));
}

/**
 * @brief cuda function for sigmoid
 * @param[in] x input array
 * @param[out] out output array
 * @param[in] size size of array
 */
__global__ void cuSigmoid(float* x, float* out, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        out[i] = 1.0f / (1.0f + expf(-x[i]));
    }
}

/**
 * @brief cuda function for sigmoid
 * @param[in] x input array
 * @param[out] out output array
 * @param[in] rows number of rows
 * @param[in] cols number of cols
 */
__global__ void cuSigmoid(float* x, float* out, int rows, int cols) {
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols) {
        out[row * cols + col] = 1.0f / (1.0f + expf(-x[row * cols + col]));
    }
}


__global__ void cuSoftmax(const float* __restrict__ x, float* __restrict__ out, float temp, int size) 
{
    extern __shared__ float shmem[];
    int tid = threadIdx.x;

    // Load values into shared memory with numerical stability
    float max_val = -FLT_MAX;
    if (tid < size) {
        float val = x[tid];
        shmem[tid] = val;
        max_val = fmaxf(max_val, val);
    }
    __syncthreads();

    // Compute exponentials
    float sum = 0.0f;
    if (tid < size) {
        float expval = expf((shmem[tid] - max_val) / temp);
        shmem[tid] = expval;
        sum += expval;
    }
    __syncthreads();

    // Accumulate sum across block (basic reduce)
    __shared__ float total_sum;
    if (tid == 0) {
        total_sum = 0.0f;
        for (int i = 0; i < size; ++i) total_sum += shmem[i];
    }
    __syncthreads();

    if (tid < size) {
        out[tid] = shmem[tid] / total_sum;
    }
}


/**
 * @brief cuda function for softmax
 * @param[in] x input array
 * @param[out] out output array
 * @param[in] temp temperature
 * @param[in] rows number of rows
 * @param[in] cols number of cols
 */
__global__ void cuSoftmax(const float* __restrict__ x, float* __restrict__ out, float temp, int rows, int cols) 
{
    extern __shared__ float shared[]; // Shared memory: holds one row of `x` and its softmax values
    float* row_vals = shared;         // Temporary buffer for exp values
    float* row_exp  = &shared[cols];  // Separate buffer for final exp values if needed

    int row = blockIdx.y;
    int tid = threadIdx.x;

    if (row < rows && tid < cols) {
        int idx = row * cols + tid;

        // Step 1: Load row into shared memory (x values)
        row_vals[tid] = x[idx];
    }

    __syncthreads();

    // Step 2: Compute max for numerical stability (single thread, or parallel reduce if needed)
    float max_val = -FLT_MAX;
    if (row < rows && tid < cols) {
        max_val = row_vals[tid];
        for (int i = 0; i < cols; ++i) {
            max_val = fmaxf(max_val, row_vals[i]);
        }

        // Step 3: Compute exp((x - max) / temp)
        float exp_val = expf((row_vals[tid] - max_val) / temp);
        row_exp[tid] = exp_val;
    }

    __syncthreads();

    // Step 4: Compute sum of exp
    float sum_exp = 0.0f;
    if (row < rows && tid < cols) {
        for (int i = 0; i < cols; ++i) {
            sum_exp += row_exp[i];
        }

        // Step 5: Normalize
        out[row * cols + tid] = row_exp[tid] / sum_exp;
    }
}


/**
 * @brief cuda function for ReLU
 * @param[in] x input
 * @param[out] result output
 */
__global__ void cuReLU(float x, float* result) {
    *result = (x > 0.0f) ? x : 0.0f;
}

/**
 * @brief cuda function for ReLU
 * @param[in] x input array
 * @param[out] out output array
 * @param[in] size size of array
 */
__global__ void cuReLU(float* x, float* out, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        out[i] = (x[i] > 0.0f) ? x[i] : 0.0f;
    }
}

// Helper for block-wide sum reduction
// Assumes sdata is a shared memory array of size blockDim.x
// my_val is the thread's local value to contribute
// The final sum will be in sdata[0]
__device__ inline void blockReduceSum_shared(float* sdata, float my_val) {
    int tid = threadIdx.x;
    sdata[tid] = my_val;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
}

// Helper for block-wide max reduction
__device__ inline void blockReduceMax_shared(volatile float* sdata, float my_val) {
    int tid = threadIdx.x;
    sdata[tid] = my_val;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
}

// Helper for block-wide min reduction
__device__ inline void blockReduceMin_shared(volatile float* sdata, float my_val) {
    int tid = threadIdx.x;
    sdata[tid] = my_val;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fminf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
}


/**
 * @brief cuda function for LOTA
 * @param[in] y input array
 * @param[out] out output array
 * @param[in] size size of array
 */
__global__ void cuLOTA(float* y, float* out, int size) {
    __shared__ float shared_min;
    __shared__ float shared_sum;

    // Step 1: Compute global min using thread 0
    if (threadIdx.x == 0) {
        float min_val = y[0];
        for (int i = 1; i < size; i++) {
            min_val = fminf(min_val, y[i]);
        }
        shared_min = min_val;

        // Step 2: Compute global sum after min shift
        float sum = 0.0f;
        for (int i = 0; i < size; i++) {
            sum += (y[i] - min_val);
        }
        shared_sum = sum;
    }
    __syncthreads();

    // Step 3: Apply LOTA formula
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float min_val = shared_min;
        float sum = shared_sum;

        if (sum > 0.0f) {
            out[idx] = (y[idx] - min_val) / sum;
        } 
        else {
            out[idx] = 1.0f / size;
        }
    }
}


/**
 * @brief cuda function for LOTA
 * @param[in] y input array
 * @param[out] out output array
 * @param[in] rows number of rows
 * @param[in] cols number of cols
 */
__global__ void cuLOTA(float* y, float* out, int rows, int cols) {
    extern __shared__ float shared[];

    float* shared_min = shared;
    float* shared_sum = shared + 1;

    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int size = rows * cols;

    // One thread computes min and sum
    if (tid == 0) {
        float min_val = y[0];
        for (int i = 1; i < size; i++) {
            min_val = fminf(min_val, y[i]);
        }
        *shared_min = min_val;

        float sum = 0.0f;
        for (int i = 0; i < size; i++) {
            sum += (y[i] - min_val);
        }
        *shared_sum = sum;
    }
    __syncthreads();

    // All threads apply the LOTA transformation
    if (tid < size) {
        float min_val = *shared_min;
        float sum = *shared_sum;

        if (sum > 0.0f) {
            out[tid] = (y[tid] - min_val) / sum;
        } 
        else {
            out[tid] = 1.0f / size;
        }
    }
}


/**
 * @brief cuda function for LOTA
 * @param[in] y input array
 * @param[out] out output array
 * @param[in] rows number of rows
 * @param[in] cols number of cols
 * @param[in] limit limit of LOTA (<= rows, <= cols)
 * @param[in] attentionType boolean flag for attention type
 */
__global__ void cuLOTA(float* y, float* out, int rows, int cols, int limit, bool attentionType) {
   extern __shared__ float shared[];
   float* shared_min = shared;
   float* shared_sum = shared + 1;

   int tid = threadIdx.x + blockIdx.x * blockDim.x;
   int size = rows * cols;

   if (tid == 0) {
       float min_val = FLT_MAX;
       float sum = 0.0f;

       // Iterate only up to the limit
       for (int row = 0; row <= limit; ++row) {
           for (int col = 0; col <= limit; ++col) {
               int idx = row * cols + col; // Correct index calculation
               if (!attentionType || col < row) {
                   float val = y[idx];
                   min_val = fminf(min_val, val);
               }
           }
       }

       if (min_val == FLT_MAX) min_val = 0.0f; // No valid entries found

       // Iterate only up to the limit
       for (int row = 0; row <= limit; ++row) {
           for (int col = 0; col <= limit; ++col) {
               int idx = row * cols + col; // Correct index calculation
               if (!attentionType || col < row) {
                   sum += (y[idx] - min_val);
               }
           }
       }

       *shared_min = min_val;
       *shared_sum = sum;
   }
   __syncthreads();

   if (tid < size) {
        int row = tid / cols;
        int col = tid % cols;

        float min_val = *shared_min;
        float sum = *shared_sum;

        if (row <= limit && col <= limit && (!attentionType || col < row)) {
            out[tid] = (sum > 0.0f) ? (y[tid] - min_val) / sum : 0.0f;
        } 
        else {
            out[tid] = 0.0f;
        }
   }
}

// DERIVATIVES OF ACTIVATION FUNCTIONS

/**
 * @brief cuda function for sigmoid derivative
 * @param[in] x input
 * @param[out] result sigmoid'(x)
 */
__global__ void cuSigmoidder(float x, float* result) {
    float sigmoid_x = 1.0f / (1.0f + expf(-x));
    *result = sigmoid_x * (1.0f - sigmoid_x);
}

/**
 * @brief cuda function for sigmoid derivative
 * @param[in] x input array
 * @param[out] out output array
 * @param[in] rows number of rows
 * @param[in] cols number of cols
 */
__global__ void cuSigmoidder(float* x, float* out, int rows, int cols) {
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols) {
        float sigmoid_x = 1.0f / (1.0f + expf(-x[row * cols + col]));
        out[row * cols + col] = sigmoid_x * (1.0f - sigmoid_x);
    }
}

/**
 * @brief cuda function for softmax derivative
 * @param[in] x input array
 * @param[out] out output array
 * @param[in] temp temperature
 * @param[in] size size of array
 */
__global__ void cuSoftmaxder(const float* x, float* out, float temp, int size) {
    extern __shared__ float shmem[];
    int tid = threadIdx.x;

    float max_val = -FLT_MAX;
    if (tid < size) {
        shmem[tid] = x[tid];
        max_val = fmaxf(max_val, x[tid]);
    }
    __syncthreads();

    float sum = 0.0f;
    if (tid < size) {
        shmem[tid] = expf((shmem[tid] - max_val) / temp);
        sum += shmem[tid];
    }
    __syncthreads();

    __shared__ float total_sum;
    if (tid == 0) {
        total_sum = 0.0f;
        for (int i = 0; i < size; ++i) total_sum += shmem[i];
    }
    __syncthreads();

    if (tid < size) {
        float s = shmem[tid] / total_sum;
        out[tid] = s * (1.0f - s);
    }
}


/**
 * @brief cuda function for softmax derivative
 * @param[in] x input array
 * @param[out] out output array
 * @param[in] temp temperature
 * @param[in] rows number of rows
 * @param[in] cols number of cols
 */
__global__ void cuSoftmaxder(const float* x, float* out, float temp, int rows, int cols) {
    extern __shared__ float shared[];
    float* row_vals = shared;
    float* row_exp  = &shared[cols];

    int row = blockIdx.y;
    int tid = threadIdx.x;

    if (row < rows && tid < cols) {
        int idx = row * cols + tid;
        row_vals[tid] = x[idx];
    }
    __syncthreads();

    float max_val = -FLT_MAX;
    for (int i = 0; i < cols; ++i)
        max_val = fmaxf(max_val, row_vals[i]);
    
    float exp_val = expf((row_vals[tid] - max_val) / temp);
    row_exp[tid] = exp_val;
    __syncthreads();

    float sum = 0.0f;
    for (int i = 0; i < cols; ++i)
        sum += row_exp[i];
    __syncthreads();

    float s = row_exp[tid] / sum;
    out[row * cols + tid] = s * (1.0f - s);
}


/**
 * @brief cuda function for ReLU derivative
 * @param[in] x input
 * @param[out] result output
 */
__global__ void cuReLUder(float x, float* result) {
    *result = (x > 0.0f) ? 1.0f : 0.0f;
}

/**
 * @brief cuda function for ReLU derivative
 * @param[in] x input array
 * @param[out] out output array
 * @param[in] size size of array
 */
__global__ void cuReLUder(float* x, float* out, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        out[i] = (x[i] > 0.0f) ? 1.0f : 0.0f;
    }
}

/**
 * @brief cuda function for LOTA derivative
 * @param[in] y input array
 * @param[out] out output array
 * @param[in] size size of array
 */
__global__ void cuLOTAder(float* y, float* out, int size) {
    __shared__ float shared_min;
    __shared__ float shared_sum;

    if (threadIdx.x == 0) {
        float min_val = y[0];
        for (int i = 1; i < size; i++) {
            min_val = fminf(min_val, y[i]);
        }
        shared_min = min_val;

        float sum = 0.0f;
        for (int i = 0; i < size; i++) {
            sum += (y[i] - min_val);
        }
        shared_sum = sum;
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float min_val = shared_min;
        float sum = shared_sum;

        if (sum > 0.0f) {
            out[idx] = (sum - y[idx] + min_val) / (sum * sum);
        }
        else {
            out[idx] = 0.0f;
        }
    }
}

/**
 * @brief cuda function for LOTA derivative
 * @param[in] y input array
 * @param[out] out output array
 * @param[in] rows number of rows
 * @param[in] cols number of cols
 */
__global__ void cuLOTAder(float* y, float* out, int rows, int cols) {
    extern __shared__ float shared[];

    float* shared_min = shared;
    float* shared_sum = shared + 1;

    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int size = rows * cols;

    // Thread 0 computes global min and sum
    if (tid == 0) {
        float min_val = y[0];
        for (int i = 1; i < size; i++) {
            min_val = fminf(min_val, y[i]);
        }
        *shared_min = min_val;

        float sum = 0.0f;
        for (int i = 0; i < size; i++) {
            sum += (y[i] - min_val);
        }
        *shared_sum = sum;
    }
    __syncthreads();

    if (tid < size) {
        float min_val = *shared_min;
        float sum = *shared_sum;

        if (sum > 0.0f) {
            out[tid] = (sum - y[tid] + min_val) / (sum * sum);
        }
        else {
            out[tid] = 0.0f;
        }
    }
}

/**
 * @brief cuda function for LOTA derivative
 * @param[in] y input array
 * @param[out] out output array
 * @param[in] rows number of rows
 * @param[in] cols number of cols
 * @param[in] limit limit of LOTA (<= rows, <= cols)
 * @param[in] attentionType boolean flag for attention type
 */
__global__ void cuLOTAder(float* y, float* out, int rows, int cols, int limit, bool attentionType) {
    extern __shared__ float shared[];
    float* shared_min = shared;
    float* shared_sum = shared + 1;

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int size = rows * cols;

    if (tid == 0) {
        float min_val = FLT_MAX;
        float sum = 0.0f;

        for (int row = 0; row <= limit; ++row) {
            for (int col = 0; col <= limit; ++col) {
                int idx = row * cols + col;
                if (!attentionType || col < row) {
                    float val = y[idx];
                    min_val = fminf(min_val, val);
                }
            }
       }

        if (min_val == FLT_MAX) min_val = 0.0f; // No valid entries

        for (int row = 0; row <= limit; ++row) {
            for (int col = 0; col <= limit; ++col) {
                int idx = row * cols + col;
                if (!attentionType || col < row) {
                    sum += (y[idx] - min_val);
                }
            }
        }

        *shared_min = min_val;
        *shared_sum = sum;
    }
    __syncthreads();

    if (tid < size) {
        int row = tid / cols;
        int col = tid % cols;

        float min_val = *shared_min;
        float sum = *shared_sum;

        if (row <= limit && col <= limit && (!attentionType || col < row)) {
            out[tid] = (sum > 0.0f) ? (sum - y[tid] + min_val) / (sum * sum) : 0.0f;
        }
        else {
            out[tid] = 0.0f;
        }
    }
}
