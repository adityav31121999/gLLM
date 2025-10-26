__kernel void l1PenaltyKernel(__global const float* weights,
                              __global float* result, // Output buffer for partial sums (size = num_groups)
                              __local float* temp_sum, // Local memory buffer (size = local_size)
                              const int size)         // Total number of weights
{
    int global_id = get_global_id(0);
    int local_id = get_local_id(0);
    int group_id = get_group_id(0);
    int local_size = get_local_size(0);

    // Initialize local memory for this work-item
    temp_sum[local_id] = 0.0f;

    // Each work-item accumulates its portion of the sum
    // Loop necessary if total size > global work size
    for (int i = global_id; i < size; i += get_global_size(0)) {
        temp_sum[local_id] += fabs(weights[i]);
    }

    // Synchronize within the work-group to ensure all items finished accumulation
    barrier(CLK_LOCAL_MEM_FENCE);

    // Perform reduction in local memory
    for (int stride = local_size / 2; stride > 0; stride >>= 1) {
        if (local_id < stride) {
            temp_sum[local_id] += temp_sum[local_id + stride];
        }
        // Synchronize after each step of the reduction
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // First work-item in the group writes the partial sum for this group to global memory
    if (local_id == 0) {
        result[group_id] = temp_sum[0];
    }
}

__kernel void l2PenaltyKernel(__global const float* weights,
                              __global float* result,    // Output buffer for partial sums (size = num_groups)
                              __local float* temp_sum,  // Local memory buffer (size = local_size)
                              const int size)          // Total number of weights
{
    int global_id = get_global_id(0);
    int local_id = get_local_id(0);
    int group_id = get_group_id(0);
    int local_size = get_local_size(0);

    // Initialize local memory
    temp_sum[local_id] = 0.0f;

    // Each work-item accumulates its portion of the sum of squares
    for (int i = global_id; i < size; i += get_global_size(0)) {
        float w = weights[i];
        temp_sum[local_id] += w * w;
    }

    // Synchronize within the work-group
    barrier(CLK_LOCAL_MEM_FENCE);

    // Perform reduction in local memory
    for (int stride = local_size / 2; stride > 0; stride >>= 1) {
        if (local_id < stride) {
            temp_sum[local_id] += temp_sum[local_id + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // First work-item in the group writes the partial sum to global memory
    if (local_id == 0) {
        result[group_id] = temp_sum[0];
    }
}

__kernel void absDiffKernel(__global const float* outputs,
                            __global const float* targets,
                            __global float* result,     // Output buffer for partial sums (size = num_groups)
                            __local float* temp_sum,   // Local memory buffer (size = local_size)
                            const int size)           // Size of output/target vectors
{
    int global_id = get_global_id(0);
    int local_id = get_local_id(0);
    int group_id = get_group_id(0);
    int local_size = get_local_size(0);

    // Initialize local memory
    temp_sum[local_id] = 0.0f;

    // Each work-item accumulates its portion of the sum of absolute differences
    for (int i = global_id; i < size; i += get_global_size(0)) {
        temp_sum[local_id] += fabs(outputs[i] - targets[i]);
    }

    // Synchronize within the work-group
    barrier(CLK_LOCAL_MEM_FENCE);

    // Perform reduction in local memory
    for (int stride = local_size / 2; stride > 0; stride >>= 1) {
        if (local_id < stride) {
            temp_sum[local_id] += temp_sum[local_id + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // First work-item in the group writes the partial sum to global memory
    if (local_id == 0) {
        result[group_id] = temp_sum[0];
    }
}

__kernel void squaredDiffKernel(__global const float* outputs,
                                __global const float* targets,
                                __global float* result,     // Output buffer for partial sums (size = num_groups)
                                __local float* temp_sum,   // Local memory buffer (size = local_size)
                                const int size)           // Size of output/target vectors
{
    int global_id = get_global_id(0);
    int local_id = get_local_id(0);
    int group_id = get_group_id(0);
    int local_size = get_local_size(0);

    // Initialize local memory
    temp_sum[local_id] = 0.0f;

    // Each work-item accumulates its portion of the sum of squared differences
    for (int i = global_id; i < size; i += get_global_size(0)) {
        float diff = outputs[i] - targets[i];
        temp_sum[local_id] += diff * diff;
    }

    // Synchronize within the work-group
    barrier(CLK_LOCAL_MEM_FENCE);

    // Perform reduction in local memory
    for (int stride = local_size / 2; stride > 0; stride >>= 1) {
        if (local_id < stride) {
            temp_sum[local_id] += temp_sum[local_id + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // First work-item in the group writes the partial sum to global memory
    if (local_id == 0) {
        result[group_id] = temp_sum[0];
    }
}

__kernel void kernelMseReduction(__global const float* expected,
                                 __global const float* output,
                                 __global float* partial_mse, // Output buffer for partial sums
                                 __local float* temp_sum,    // Local memory buffer
                                 const int size)
{
    int global_id = get_global_id(0);
    int local_id = get_local_id(0);
    int group_id = get_group_id(0);
    int local_size = get_local_size(0);

    // Initialize local memory
    temp_sum[local_id] = 0.0f;

    // Each work-item accumulates its portion of the sum of squared errors
    for (int i = global_id; i < size; i += get_global_size(0)) {
        float diff = expected[i] - output[i];
        temp_sum[local_id] += diff * diff;
    }

    // Synchronize within the work-group
    barrier(CLK_LOCAL_MEM_FENCE);

    // Perform reduction in local memory
    for (int stride = local_size / 2; stride > 0; stride >>= 1) {
        if (local_id < stride) {
            temp_sum[local_id] += temp_sum[local_id + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // First work-item in the group writes the partial sum to global memory
    if (local_id == 0) {
        partial_mse[group_id] = temp_sum[0];
    }
}

// ---------------- forward pass ---------------- //

__kernel void kernelLayerForward(__global const float* inputs,
                                 __global const float* weights,
                                 __global float* outputs,
                                 const int input_size,
                                 const int output_size)
{
    int neuron_idx = get_global_id(0); // Index 'j' for the output neuron

    const int input_size_f4 = input_size / 4;
    if (neuron_idx < output_size) {
        float sum = 0.0f;
        __global const float4* inputs_f4 = (__global const float4*)inputs;
        __global const float4* weights_row_f4 = (__global const float4*)(weights + neuron_idx * input_size);

        for (int i = 0; i < input_size / 4; ++i) {
            sum += dot(inputs_f4[i], weights_row_f4[i]);
        }
        // Store the weighted sum (pre-activation value)
        outputs[neuron_idx] = sum;
        // Activation function (e.g., sigmoid, ReLU) would typically be applied in a separate kernel or here.
        // outputs[neuron_idx] = 1.0f / (1.0f + exp(-sum)); // Example Sigmoid
    }
}

// ---------------- backward pass ---------------- //

__kernel void kernelOutputDelta(__global const float* output, __global const float* expected, 
                    __global float* delta, const int size) 
{
    int idx = get_global_id(0);
    if (idx < size) {
        // This kernel calculates the initial delta for a layer with Sigmoid activation and Binary Cross-Entropy (BCE) loss.
        // The gradient of the loss with respect to the pre-activation inputs (logits) is simply (activation - expected).
        // This is numerically stable.
        delta[idx] = output[idx] - expected[idx];
    }
}

__kernel void kernelComputeGradMLPInput(__global const float* deltas,
                                        __global const float* weights,
                                        __global float* grad_input,
                                        const int current_layer_size,
                                        const int input_size)
{
    // Each work-item computes 4 elements of the grad_input vector.
    int i4 = get_global_id(0) * 4; // Corresponds to input dimension index 'i', processing 4 at a time

    if (i4 < input_size) {
        float4 sum4 = (float4)(0.0f);
        // Sum over the neurons 'j' in the current layer.
        // This new structure provides much better memory coalescing on the 'weights' buffer.
        for (int j = 0; j < current_layer_size; ++j) {
            float delta_j = deltas[j];
            // Load 4 contiguous weights from row 'j' of the weight matrix: W_j,i, W_j,i+1, W_j,i+2, W_j,i+3
            sum4 = fma(delta_j, vload4(0, weights + j * input_size + i4), sum4);
        }
        vstore4(sum4, 0, grad_input + i4);
    }
}

__kernel void kernelOutputDeltaSigmoid(__global const float* activations,
                                       __global const float* expected,
                                       __global float* deltas,
                                       const int size)
{
    int idx = get_global_id(0);
    if (idx < size) {
        float activation = activations[idx];
        float error = activation - expected[idx];
        // Sigmoid derivative: sigmoid(x) * (1 - sigmoid(x))
        float sigmoid_der = activation * (1.0f - activation);
        deltas[idx] = error * sigmoid_der;
    }
}

__kernel void kernelHiddenDeltaSigmoid(__global const float* next_layer_deltas,
                                       __global const float* weights, // Weights W_ji where j=next, i=current
                                       __global const float* activations,
                                       __global float* deltas,
                                       const int current_layer_size,
                                       const int next_layer_size)
{
    int neuron_idx = get_global_id(0); // Index 'i' for the current layer neuron

    if (neuron_idx < current_layer_size) {
        float error_sum = 0.0f;
        // Sum contribution from each neuron 'j' in the next layer
        // weights are row-major: weights[row * num_cols + col]
        // Here: row=j (next layer neuron), num_cols=current_layer_size, col=neuron_idx (current layer neuron)
        // We need weight W_ji which connects current neuron 'i' to next neuron 'j'
        for (int j = 0; j < next_layer_size; j++) {
            error_sum += next_layer_deltas[j] * weights[j * current_layer_size + neuron_idx];
        }

        // Calculate delta using the sigmoid derivative for the current neuron 'i'
        float activation = activations[neuron_idx];
        float sigmoid_der = activation * (1.0f - activation);
        deltas[neuron_idx] = error_sum * sigmoid_der;
    }
}

__kernel void kernelLastLayerDeltaSigmoid(__global const float* grad_output,
                                          __global const float* activations,
                                          __global float* deltas,
                                          const int size)
{
    int idx = get_global_id(0);
    const int size_f4 = size / 4;
    if (idx < size_f4) {
        float4 activation = vload4(idx, activations);
        // ReLU derivative: 1.0f if activation > 0.0f, else 0.0f
        float4 relu_der = select((float4)(0.0f), (float4)(1.0f), activation > 0.0f);
        vstore4(vload4(idx, grad_output) * relu_der, idx, deltas);
    }
}

__kernel void kernelUpdateInputMLP(__global float* input, __global const float* weights,
                                   __global const float* deltas, const float learning_rate,
                                   const int first_hidden_layer_size, const int input_size)
{
    int i4 = get_global_id(0) * 4; // Index 'i' for the input vector dimension, processing 4 at a time

    if (i4 < input_size) {
        float4 grad_input_component4 = (float4)(0.0f);
        // Sum over the neurons 'j' in the first hidden layer
        for (int j = 0; j < first_hidden_layer_size; j++) {
            float delta_j = deltas[j];
            grad_input_component4 = fma(delta_j, vload4(0, weights + j * input_size + i4), grad_input_component4);
        }
        float4 input4 = vload4(0, input + i4);
        // Update the input vector element
        vstore4(input4 - learning_rate * grad_input_component4, 0, input + i4);
    }
}

// ---------------- update weigthts ---------------- //

__kernel void kernelUpdateWeights(__global const float* deltas,
                                  __global const float* prev_activations,
                                  __global float* weights,
                                  const float learning_rate,
                                  const int current_layer_size,
                                  const int prev_layer_size)
{
    // Use 2D indexing
    int col = get_global_id(0); // Index for previous layer neuron (input to weight) 'j'
    int row = get_global_id(1); // Index for current layer neuron (output of weight) 'i'

    if (row < current_layer_size && col < prev_layer_size) {
        int weight_idx = row * prev_layer_size + col; // row-major index for W_ij

        // Calculate gradient implicitly and update weight
        weights[weight_idx] -= learning_rate * deltas[row] * prev_activations[col];
    }
}


/**
 * @brief Updates weights and stores gradients for a layer (no regularization).
 */
__kernel void kernelUpdateWeightsAndGradients(__global const float* deltas, 
                                                __global const float* prev_activations,
                                                __global float* weights, 
                                                __global float* gweights,
                                                float learning_rate,
                                                int current_layer_size,
                                                int prev_layer_size)
{
    int j = get_global_id(0);
    int i = get_global_id(1);

    if (j < current_layer_size && i < prev_layer_size) {
        int weight_idx = j * prev_layer_size + i;
        float gradient = deltas[j] * prev_activations[i];
        
        gweights[weight_idx] = gradient;
        weights[weight_idx] -= learning_rate * gradient;
    }
}


__kernel void kernelUpdateElasticNet(__global const float* deltas, 
                                     __global const float* prev_activations,
                                     __global float* weights,
                                     __global float* gweights, 
                                     int current_layer_size, 
                                     int prev_layer_size,
                                     float learning_rate, 
                                     float lambda_l1,
                                     float lambda_l2)
{
    // Consistent 2D indexing with other update kernels
    int i = get_global_id(0); // neuron index in previous layer ('col')
    int j = get_global_id(1); // neuron index in current layer ('row')

    if (i < prev_layer_size && j < current_layer_size) {
        int weight_idx = j * prev_layer_size + i;
        
        // Gradient of the error term
        float error_gradient = deltas[j] * prev_activations[i];
        
        // Gradient of the L2 regularization term
        float current_weight = weights[weight_idx];
        float l2_gradient = lambda_l2 * current_weight;
        
        // Subgradient of the L1 regularization term
        float sign;
        if (current_weight < 0.0f) {
            sign = -1.0f;
        }
        else if (current_weight > 0.0f) {
            sign = 1.0f;
        }
        else {
            sign = 0.0f; // Subgradient is 0 when weight is 0
        }

        float l1_gradient = sign * lambda_l1;

        // Total gradient
        float total_gradient = error_gradient + l2_gradient + l1_gradient;

        if (gweights != NULL) { // Use NULL instead of nullptr and ensure check exists
            gweights[weight_idx] = total_gradient;
        }
        weights[weight_idx] -= learning_rate * total_gradient;
    }
}

__kernel void kernelUpdateElasticNetWithGrads(__global const float* deltas, 
                                     __global float* weights,
                                     __global float* gweights, 
                                     int current_layer_size, 
                                     int prev_layer_size,
                                     float learning_rate, 
                                     float lambda_l1,
                                     float lambda_l2)
{
    // Consistent 2D indexing with other update kernels
    int i = get_global_id(0); // neuron index in previous layer ('col')
    int j = get_global_id(1); // neuron index in current layer ('row')

    if (i < prev_layer_size && j < current_layer_size) {
        int weight_idx = j * prev_layer_size + i;
        
        // Gradient of the L2 regularization term
        float current_weight = weights[weight_idx];
        float l2_gradient = lambda_l2 * current_weight;
        
        // Subgradient of the L1 regularization term
        float sign;
        if (current_weight < 0.0f) {
            sign = -1.0f;
        }
        else if (current_weight > 0.0f) {
            sign = 1.0f;
        }
        else {
            sign = 0.0f; // Subgradient is 0 when weight is 0
        }

        float l1_gradient = sign * lambda_l1;

        // Total gradient
        float total_gradient = gweights[weight_idx] + l2_gradient + l1_gradient;

        weights[weight_idx] -= learning_rate * total_gradient;
    }
}


__kernel void kernelRpropUpdate(__global float* weights,
                                __global const float* gradients,
                                __global float* prev_gradients, // Read and Write
                                __global float* delta_weights,  // Read and Write
                                const float eta_plus,
                                const float eta_minus,
                                const float delta_max,
                                const float delta_min,
                                const int size)
{
    int idx = get_global_id(0); // Was using CUDA-style indexing (blockIdx.x * blockDim.x + threadIdx.x)

    if (idx < size) {
        float grad = gradients[idx];
        float prev_grad = prev_gradients[idx];
        float delta = delta_weights[idx];
        float weight_update = 0.0f;

        // Rprop update logic
        float sign_change = grad * prev_grad;

        if (sign_change > 0.0f) {
            // No sign change: Increase step size, apply update
            delta = fmin(delta * eta_plus, delta_max);
            weight_update = -copysign(delta, grad); // Update direction is opposite to gradient
            prev_gradients[idx] = grad; // Store current gradient for next iteration
        }
        else if (sign_change < 0.0f) {
            delta = fmax(delta * eta_minus, delta_min);
            weight_update = 0.0f; // No weight update this step
            prev_gradients[idx] = 0.0f; // Reset prev_grad to prevent double penalty
            delta = fmax(delta * eta_minus, delta_min);
            weights[idx] += copysign(delta, prev_grad); // Revert step based on prev_grad sign
            prev_gradients[idx] = 0.0f; // Reset prev_grad
            delta_weights[idx] = delta; // Store updated delta
            return; // Exit early, no further update this step

        }
        else {
            weight_update = -copysign(delta, grad);
            prev_gradients[idx] = grad; // Store current gradient
        }

        // Apply the calculated weight update
        weights[idx] += weight_update;
        // Store the updated step size
        delta_weights[idx] = delta;
    }
}

