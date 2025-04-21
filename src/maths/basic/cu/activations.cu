
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


/**
 * @brief cuda function for softmax
 * @param[in] x input array
 * @param[out] out output array
 * @param[in] temp temperature
 * @param[in] size number of elements
 */
__global__ void cuSoftmax(float* x, float* out, float temp, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        // First thread computes the sum of exponentials
        if (i == 0) {
            float sum = 0.0f;
            for (int j = 0; j < size; j++) {
                out[j] = expf(x[j] / temp); // Store exp values temporarily
                sum += out[j];
            }
            // Store sum in the last element (temporary storage)
            atomicExch((int*)&out[size-1], *((int*)&sum));
        }
        __syncthreads();
        
        // Get the sum computed by the first thread
        float sum = out[size-1];
        
        // Normalize by the sum
        if (sum > 0.0f) {
            out[i] = out[i] / sum; // out[i] already contains exp(x[i]/temp)
        } else {
            out[i] = 1.0f / size; // Equal distribution if all values are the same
        }
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
__global__ void cuSoftmax(float* x, float* out, float temp, int rows, int cols) {
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols) {
        float sum = 0.0f;
        for (int j = 0; j < cols; j++) {
            sum += expf(x[row * cols + j] / temp);
        }
        out[row * cols + col] = expf(x[row * cols + col] / temp) / sum;
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

/**
 * @brief cuda function for SeLU
 * @param[in] x input
 * @param[out] result output
 */
__global__ void cuSeLU(float x, float* result) {
    float alpha = 1.67326f;
    float lambda = 1.0507f;
    *result = (x > 0.0f) ? lambda * x : lambda * alpha * (expf(x) - 1.0f);
}

/**
 * @brief cuda function for SeLU
 * @param[in] x input array
 * @param[out] out output array
 * @param[in] size size of array
 */
__global__ void cuSeLU(float* x, float* out, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float alpha = 1.67326f;
        float lambda = 1.0507f;
        out[i] = (x[i] > 0.0f) ? lambda * x[i] : lambda * alpha * (expf(x[i]) - 1.0f);
    }
}

/**
 * @brief cuda function for LOTA
 * @param[in] y input array
 * @param[out] out output array
 * @param[in] size size of array
 */
__global__ void cuLOTA(float* y, float* out, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        // Find minimum value in the array
        float min_val = y[0];
        for (int j = 1; j < size; j++) {
            min_val = fminf(min_val, y[j]);
        }
        
        // Subtract min value and compute sum
        float sum = 0.0f;
        for (int j = 0; j < size; j++) {
            sum += (y[j] - min_val);
        }
        
        // Normalize if sum is not zero
        if (sum > 0.0f) {
            out[i] = (y[i] - min_val) / sum;
        } else {
            out[i] = 1.0f / size; // Equal distribution if all values are the same
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
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows && col < cols) {
        // Process each row independently
        if (col == 0) { // First thread in each row computes min and sum
            float min_val = y[row * cols];
            // Find minimum value in this row
            for (int j = 1; j < cols; j++) {
                min_val = fminf(min_val, y[row * cols + j]);
            }
            
            // Store min value in the first element of the output row (temporary)
            out[row * cols] = min_val;
            
            // Compute sum of (y - min) for normalization
            float sum = 0.0f;
            for (int j = 0; j < cols; j++) {
                sum += (y[row * cols + j] - min_val);
            }
            
            // Store sum in the second element of the output row (temporary)
            if (cols > 1) {
                out[row * cols + 1] = sum;
            }
        }
        __syncthreads(); // Make sure min and sum are computed before proceeding
        
        // Get the min and sum values computed by the first thread
        float min_val = out[row * cols];
        float sum = (cols > 1) ? out[row * cols + 1] : 0.0f;
        
        // Apply LOTA transformation
        if (sum > 0.0f) {
            out[row * cols + col] = (y[row * cols + col] - min_val) / sum;
        } else {
            out[row * cols + col] = 1.0f / cols; // Equal distribution if all values are the same
        }
    }
}

/**
 * @brief cuda function for LOTA
 * @param[in] y input array
 * @param[out] out output array
 * @param[in] rows number of rows
 * @param[in] cols number of cols
 * @param[in] limit limit of LOTA
 * @param[in] attentionType boolean flag for attention type
 */
__global__ void cuLOTA(float* y, float* out, int rows, int cols, int limit, bool attentionType) {
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Determine processing boundary based on attentionType
    bool process = attentionType ? (col < row) : (col < cols);
    
    if (row < rows && process && col < cols) {
        // Apply limit constraint first
        float val = (y[row * cols + col] > 0.0f && y[row * cols + col] < limit) ? y[row * cols + col] : 0.0f;
        out[row * cols + col] = val;
        
        // First thread in each row computes min and sum for normalization
        if (col == 0) {
            float min_val = FLT_MAX;
            // Find minimum non-zero value in this row
            for (int j = 0; j < cols; j++) {
                if (attentionType && j >= row) continue; // Skip if using triangular attention
                if (out[row * cols + j] > 0.0f) {
                    min_val = fminf(min_val, out[row * cols + j]);
                }
            }
            
            if (min_val == FLT_MAX) min_val = 0.0f; // No positive values found
            
            // Compute sum for normalization
            float sum = 0.0f;
            for (int j = 0; j < cols; j++) {
                if (attentionType && j >= row) continue; // Skip if using triangular attention
                if (out[row * cols + j] > 0.0f) {
                    out[row * cols + j] -= min_val;
                    sum += out[row * cols + j];
                }
            }
            
            // Store sum for other threads
            atomicExch((int*)&out[row * cols + cols - 1], *((int*)&sum));
        }
        __syncthreads();
        
        // Get the sum computed by the first thread
        float sum = out[row * cols + cols - 1];
        
        // Normalize if sum is not zero
        if (sum > 0.0f && out[row * cols + col] > 0.0f) {
            out[row * cols + col] /= sum;
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
__global__ void cuSoftmaxder(float* x, float* out, float temp, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        // First compute softmax values (or reuse if available)
        float* softmax_vals = new float[size];
        float sum = 0.0f;
        for (int j = 0; j < size; j++) {
            softmax_vals[j] = expf(x[j] / temp);
            sum += softmax_vals[j];
        }
        for (int j = 0; j < size; j++) {
            softmax_vals[j] /= sum;
        }
        
        // Calculate the derivative
        float result = softmax_vals[i] * (1.0f - softmax_vals[i]);
        
        // Subtract the softmax of each other element
        for (int j = 0; j < size; j++) {
            if (i != j) {
                result -= softmax_vals[j];
            }
        }
        
        out[i] = result;
        delete[] softmax_vals;
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
__global__ void cuSoftmaxder(float* x, float* out, float temp, int rows, int cols) {
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols) {
        float softmax_x = 0.0f;
        float sum = 0.0f;
        for (int j = 0; j < cols; j++) {
            sum += expf(x[row * cols + j] / temp);
        }
        softmax_x = expf(x[row * cols + col] / temp) / sum;
        out[row * cols + col] = softmax_x * (1.0f - softmax_x);
    }
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
 * @brief cuda function for SeLU derivative
 * @param[in] x input
 * @param[out] result output
 */
__global__ void cuSeLUder(float x, float* result) {
    float alpha = 1.67326f;
    float lambda = 1.0507f;
    *result = (x > 0.0f) ? lambda : lambda * alpha * expf(x);
}

/**
 * @brief cuda function for SeLU derivative
 * @param[in] x input array
 * @param[out] out output array
 * @param[in] size size of array
 */
__global__ void cuSeLUder(float* x, float* out, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float alpha = 1.67326f;
        float lambda = 1.0507f;
        out[i] = (x[i] > 0.0f) ? lambda : lambda * alpha * expf(x[i]);
    }
}

/**
 * @brief cuda function for LOTA derivative
 * @param[in] y input array
 * @param[out] out output array
 * @param[in] size size of array
 */
__global__ void cuLOTAder(float* y, float* out, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        // Find minimum value in the array
        float min_val = y[0];
        for (int j = 1; j < size; j++) {
            min_val = fminf(min_val, y[j]);
        }
        
        // Compute sum for normalization
        float sum = 0.0f;
        for (int j = 0; j < size; j++) {
            sum += (y[j] - min_val);
        }
        
        // Compute derivative
        if (sum > 0.0f) {
            // The derivative is (sum - (y[i] - min_val)) / sum^2 = (sum - y[i] + min_val) / sum^2
            out[i] = (sum - y[i] + min_val) / (sum * sum);
        } else {
            out[i] = 0.0f;
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
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows && col < cols) {
        // First thread in each row computes min and sum
        if (col == 0) {
            float min_val = y[row * cols];
            // Find minimum value in this row
            for (int j = 1; j < cols; j++) {
                min_val = fminf(min_val, y[row * cols + j]);
            }
            
            // Store min value in shared memory or temporary location
            out[row * cols] = min_val;
            
            // Compute sum for normalization
            float sum = 0.0f;
            for (int j = 0; j < cols; j++) {
                sum += (y[row * cols + j] - min_val);
            }
            
            // Store sum in shared memory or temporary location
            if (cols > 1) {
                out[row * cols + 1] = sum;
            }
        }
        __syncthreads(); // Make sure min and sum are computed before proceeding
        
        // Get the min and sum values computed by the first thread
        float min_val = out[row * cols];
        float sum = (cols > 1) ? out[row * cols + 1] : 0.0f;
        
        // Compute derivative
        if (sum > 0.0f) {
            // The derivative is (sum - (y[i] - min_val)) / sum^2 = (sum - y[i] + min_val) / sum^2
            out[row * cols + col] = (sum - y[row * cols + col] + min_val) / (sum * sum);
        } else {
            out[row * cols + col] = 0.0f;
        }
    }
}
/**
 * @brief cuda function for LOTA derivative
 * @param[in] y input array
 * @param[out] out output array
 * @param[in] rows number of rows
 * @param[in] cols number of cols
 * @param[in] limit limit of LOTA
 * @param[in] attentionType boolean flag for attention type
 */
__global__ void cuLOTAder(float* y, float* out, int rows, int cols, int limit, bool attentionType) {
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Determine processing boundary based on attentionType
    bool process = attentionType ? (col < row) : (col < cols);
    
    if (row < rows && process && col < cols) {
        // First apply limit constraint
        bool valid = (y[row * cols + col] > 0.0f && y[row * cols + col] < limit);
        
        if (col == 0) {
            // First thread computes min and sum for the row
            float min_val = FLT_MAX;
            
            // Find minimum non-zero value that meets the limit constraint
            for (int j = 0; j < cols; j++) {
                if (attentionType && j >= row) continue; // Skip if using triangular attention
                if (y[row * cols + j] > 0.0f && y[row * cols + j] < limit) {
                    min_val = fminf(min_val, y[row * cols + j]);
                }
            }
            
            if (min_val == FLT_MAX) min_val = 0.0f; // No valid values found
            
            // Compute sum for normalization
            float sum = 0.0f;
            for (int j = 0; j < cols; j++) {
                if (attentionType && j >= row) continue; // Skip if using triangular attention
                if (y[row * cols + j] > 0.0f && y[row * cols + j] < limit) {
                    sum += (y[row * cols + j] - min_val);
                }
            }
            
            // Store min and sum for other threads
            atomicExch((int*)&out[row * cols], *((int*)&min_val));
            atomicExch((int*)&out[row * cols + 1], *((int*)&sum));
        }
        __syncthreads();
        
        // Get the min and sum values computed by the first thread
        float min_val = out[row * cols];
        float sum = out[row * cols + 1];
        
        // Compute derivative
        if (valid && sum > 0.0f) {
            // The derivative is (sum - (y[i] - min_val)) / sum^2 = (sum - y[i] + min_val) / sum^2
            out[row * cols + col] = (sum - y[row * cols + col] + min_val) / (sum * sum);
        } else {
            out[row * cols + col] = 0.0f;
        }
    }
}
