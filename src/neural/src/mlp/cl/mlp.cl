// mlp related kernels

// kernel for forward propagation
__kernel void kernelLayerForward(__global const float* inputs,
                                 __global const float* weights,
                                 __global float* outputs,
                                 const int input_size,
                                 const int output_size)
{
    int neuron_idx = get_global_id(0); // Index 'j' for the output neuron

    if (neuron_idx < output_size) {
        float sum = 0.0f;
        // Perform dot product: inputs . weights_row_j
        // weights are row-major: weights[row * num_cols + col]
        // Here: row = neuron_idx, num_cols = input_size, col = i (input dimension)
        for (int i = 0; i < input_size; i++) { // input_size is the number of columns in the weights matrix for the current row
            sum += inputs[i] * weights[neuron_idx * input_size + i];
        }
        // Store the weighted sum (pre-activation value)
        outputs[neuron_idx] = sum;
        // Activation function (e.g., sigmoid, ReLU) would typically be applied in a separate kernel or here.
        // outputs[neuron_idx] = 1.0f / (1.0f + exp(-sum)); // Example Sigmoid
    }
}


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


__kernel void kernelComputeGradMLPInput(__global const float* deltas,
                                        __global const float* weights,
                                        __global float* grad_input,
                                        const int current_layer_size,
                                        const int input_size)
{
    int input_idx = get_global_id(0); // Corresponds to input dimension index 'i'

    if (input_idx < input_size) {
        float sum = 0.0f;
        // Sum over the neurons 'j' in the current layer
        for (int j = 0; j < current_layer_size; ++j) {
            // weights are row-major: weights[row * num_cols + col]
            // Here: row = j (current layer neuron), num_cols = input_size, col = input_idx (input dimension)
            sum += deltas[j] * weights[j * input_size + input_idx];
        }
        grad_input[input_idx] = sum;
    }
}


/**
 * @brief Computes the raw gradient matrix (dL/dW) for a linear layer.
 *        Gradient is calculated as an outer product: dL/dW_ij = delta_i * prev_activation_j.
 * @param deltas           Global pointer to the deltas (dL/du) of the current layer. (current_layer_size x 1 vector)
 * @param prev_activations Global pointer to the activations from the previous layer (input to this weight matrix). (prev_layer_size x 1 vector)
 * @param raw_gradients    Global pointer to the output buffer for the computed raw gradients (dL/dW). (current_layer_size x prev_layer_size matrix)
 *                         This buffer should be zeroed before accumulation over a batch/sample if needed.
 * @param current_layer_size Number of neurons in the current layer (rows in weights matrix, size of deltas vector).
 * @param prev_layer_size  Number of neurons in the previous layer (columns in weights matrix, size of prev_activations vector).
 */
__kernel void clComputeLinearLayerGradient(
    __global const float* deltas,
    __global const float* prev_activations,
    __global float* raw_gradients,
    const int current_layer_size, // Output dimension for this layer's weights
    const int prev_layer_size     // Input dimension for this layer's weights
) {
    // Each work-item computes one element of the raw_gradients matrix
    int row = get_global_id(0); // Index for output neuron in current layer (j)
    int col = get_global_id(1); // Index for input neuron from previous layer (i)

    if (row < current_layer_size && col < prev_layer_size) {
        int grad_idx = row * prev_layer_size + col; // Row-major indexing
        // dL/dW_row,col = delta_row * prev_activation_col
        raw_gradients[grad_idx] = deltas[row] * prev_activations[col];
    }
}


__kernel void kernelOutputDelta(__global const float* output, __global const float* expected, 
                    __global float* delta, const int size) 
{
    int idx = get_global_id(0);
    if (idx < size) {
        // This kernel calculates the initial delta for a layer with Sigmoid activation and Binary Cross-Entropy (BCE) loss.
        // The gradient of the loss with respect to the pre-activation inputs (logits) is simply (activation - expected).
        // This is numerically stable.
        delta[idx] = (output[idx] - expected[idx]);
        delta[idx] /= (1.0f - output[idx]) * output[idx];   // output/logit is not activated
    }
}


__kernel void kernelOutputDeltaSigmoid(
    __global const float* logits,    // Corrected name for clarity: should be raw logits (u)
    __global const float* expected,  // True labels (y), which are 0 or 1
    __global float* deltas,
    const int size)
{
    int idx = get_global_id(0); // Get the global ID of the current work item (neuron index)

    if (idx < size) {
        // Step 1: Get the raw logit (u) for the current output neuron.
        // This is the input value 'u' to the implicit sigmoid function.
        float current_logit = logits[idx];

        // Step 2: Calculate the predicted probability (hat_y) by applying the sigmoid function to the logit.
        // hat_y = sigmoid(u)
        float predicted_prob = 1.0f / (1.0f + exp(-current_logit));

        // Step 3: Calculate the error signal (delta) for this logit.
        // For Binary Cross-Entropy with Logits loss, this is dL/du = hat_y - y.
        deltas[idx] = predicted_prob - expected[idx];
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
    if (idx < size) {
        float activation = activations[idx];
        // Sigmoid derivative: sigmoid(x) * (1 - sigmoid(x))
        float sigmoid_der = activation * (1.0f - activation);
        // If ReLU: float relu_der = (activation > 0.0f) ? 1.0f : 0.0f; deltas[idx] = grad_output[idx] * relu_der;
        deltas[idx] = grad_output[idx] * sigmoid_der;
    }
}


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


__kernel void kernelUpdateInputMLP(__global float* input, __global const float* weights,
                                   __global const float* deltas, const float learning_rate,
                                   const int first_hidden_layer_size, const int input_size)
{
    int input_idx = get_global_id(0); // Index 'i' for the input vector dimension

    if (input_idx < input_size) {
        float grad_input_component = 0.0f;
        // Sum over the neurons 'j' in the first hidden layer
        for (int j = 0; j < first_hidden_layer_size; j++) {
            // weights are row-major: weights[row * num_cols + col]
            // Here: row = j (hidden neuron), num_cols = input_size, col = input_idx (input dimension)
            grad_input_component += deltas[j] * weights[j * input_size + input_idx];
        }
        // Update the input vector element
        input[input_idx] -= learning_rate * grad_input_component;
    }
}


/**
 * @brief Updates weights and stores gradients for a layer (no regularization).
 *        Includes element-wise gradient clipping.
 */
__kernel void kernelUpdateWeightsAndGradients(__global const float* deltas, __global const float* prev_activations,
        __global float* weights, __global float* gweights, float learning_rate,
        const float max_grad_clip_value, // NEW: Clipping threshold
        int current_layer_size, int prev_layer_size)
{
    int j = get_global_id(0); // Row index (neuron in current layer)
    int i = get_global_id(1); // Column index (neuron in previous layer)

    if (j < current_layer_size && i < prev_layer_size) {
        int weight_idx = j * prev_layer_size + i;
        float gradient = deltas[j] * prev_activations[i];

        // Sanitize gradient to prevent inf/nan propagation.
        if (isnan(gradient)) {
            gradient = 0.0001f; // Replace NaN with a small, non-zero value
        } else if (isinf(gradient)) {
            // Replace infinity with a large but finite value, preserving the sign.
            gradient = copysign(1000.0f, gradient);
        }

        // Apply element-wise gradient clipping
        if (fabs(gradient) > max_grad_clip_value) {
            gradient = copysign(max_grad_clip_value, gradient);
        }

        if (gweights != NULL) {
            gweights[weight_idx] = gradient;
        }
        weights[weight_idx] -= learning_rate * gradient;
    }
}


/*
 * @brief OpenCL kernel for L1 regularization weight update.
 *        This kernel is a special case of Elastic Net, using the same robust logic.
 *        Includes element-wise gradient clipping.
 * @param weights Weights to be updated
 * @param deltas Deltas for the current layer
 * @param prev_activations Activations from the previous layer
 * @param learning_rate Learning rate
 * @param lambda L1 regularization parameter
 * @param max_grad_clip_value Maximum absolute value for gradient clipping (NEW)
 * @param gweights Gradients of the weights (output, can be nullptr).
 * @param current_layer_size Size of the current layer
 * @param prev_layer_size Size of the previous layer
 */
__kernel void kernelUpdateWeightsL1(__global float* weights, __global const float* deltas,
                                    __global const float* prev_activations, const float learning_rate,
                                    const float lambda, const float max_grad_clip_value, // NEW: Clipping threshold
                                    __global float* gweights,
                                    const int current_layer_size, const int prev_layer_size)
{
    int i = get_global_id(0); // neuron index in previous layer ('col')
    int j = get_global_id(1); // neuron index in current layer ('row')

    if (i < prev_layer_size && j < current_layer_size) {
        int weight_idx = j * prev_layer_size + i;

        // Gradient of the error term
        float error_gradient = deltas[j] * prev_activations[i];
        float current_weight = weights[weight_idx];

        // For pure L1 regularization
        float lambda_l1 = lambda;

        // Subgradient of the L1 regularization term
        float sign_w;
        if (current_weight < 0.0f) {
            sign_w = -1.0f;
        }
        else if (current_weight > 0.0f) {
            sign_w = 1.0f;
        }
        else {
            sign_w = 0.0f; // Subgradient is 0 when weight is 0
        }
        float l1_gradient = sign_w * lambda_l1;

        // Total gradient (error + regularization)
        float total_gradient = error_gradient + l1_gradient;

        // Sanitize gradient
        if (isnan(total_gradient)) {
            total_gradient = 0.0001f;
        } else if (isinf(total_gradient)) {
            total_gradient = copysign(1000.0f, total_gradient);
        }

        // Apply element-wise gradient clipping
        if (fabs(total_gradient) > max_grad_clip_value) {
            total_gradient = copysign(max_grad_clip_value, total_gradient);
        }

        if (gweights != NULL) {
            gweights[weight_idx] = total_gradient;
        }
        // Apply the weight update
        weights[weight_idx] -= learning_rate * total_gradient;
    }
}


__kernel void kernelUpdateWeightsL2(__global float* weights, __global const float* deltas,
                                    __global const float* prev_activations, const float learning_rate,
                                    const float lambda, const float max_grad_clip_value, // NEW: Clipping threshold
                                    __global float* gweights,
                                    const int current_layer_size, const int prev_layer_size)
{
    // Use 2D indexing
    int col = get_global_id(0); // Index for previous layer neuron 'j'
    int row = get_global_id(1); // Index for current layer neuron 'i'

    if (row < current_layer_size && col < prev_layer_size) {
        int weight_idx = row * prev_layer_size + col;

        // Gradient of the error term
        float error_gradient = deltas[row] * prev_activations[col];

        // Gradient of the L2 regularization term (weight decay)
        float l2_gradient = lambda * weights[weight_idx];

        // Total gradient
        float total_gradient = error_gradient + l2_gradient;

        // Sanitize gradient
        if (isnan(total_gradient)) {
            total_gradient = 0.0001f;
        } else if (isinf(total_gradient)) {
            total_gradient = copysign(1000.0f, total_gradient);
        }

        // Apply element-wise gradient clipping
        if (fabs(total_gradient) > max_grad_clip_value) {
            total_gradient = copysign(max_grad_clip_value, total_gradient);
        }

        if (gweights != NULL) {
            gweights[weight_idx] = total_gradient;
        }
        weights[weight_idx] -= learning_rate * total_gradient;
    }
}


__kernel void kernelUpdateElasticNet(__global const float* deltas, __global const float* prev_activations,
        __global float* weights, __global float* gweights, float learning_rate, float lambda_l1,
        float lambda_l2, const float max_grad_clip_value,
        int current_layer_size, int prev_layer_size)
{
    // Consistent 2D indexing with other update kernels
    int i = get_global_id(0);   // neuron index in previous layer ('col')
    int j = get_global_id(1);   // neuron index in current layer ('row')

    if (i < prev_layer_size && j < current_layer_size) {
        int weight_idx = j * prev_layer_size + i;

        // Gradient of the error term
        float error_gradient = deltas[j] * prev_activations[i];

        // Gradient of the L2 regularization term
        float current_weight = weights[weight_idx];
        float l2_gradient = lambda_l2 * current_weight;

        // Subgradient of the L1 regularization term
        float sign_w;
        if (current_weight < 0.0f) {
            sign_w = -1.0f;
        }
        else if (current_weight > 0.0f) {
            sign_w = 1.0f;
        }
        else {
            sign_w = 0.0f; // Subgradient is 0 when weight is 0
        }

        float l1_gradient = sign_w * lambda_l1;

        // Total gradient (error + regularization)
        float total_gradient = error_gradient + l2_gradient + l1_gradient;

        // Sanitize gradient
        if (isnan(total_gradient)) {
            total_gradient = 0.0001f;
        } else if (isinf(total_gradient)) {
            total_gradient = copysign(1000.0f, total_gradient);
        }

        // Apply element-wise gradient clipping
        if (fabs(total_gradient) > max_grad_clip_value) {
            total_gradient = copysign(max_grad_clip_value, total_gradient);
        }

        if (gweights != NULL) {
            gweights[weight_idx] = total_gradient;
        }
        weights[weight_idx] -= learning_rate * total_gradient;
    }
}


/**
 * @brief OpenCL Kernel for Adam Optimizer weight update.
 *        Includes element-wise gradient clipping for the input gradient.
 * @param weights       Global memory pointer to the weight matrix.
 * @param gradients     Global memory pointer to the raw gradient matrix for the current step.
 * @param moments       Global memory pointer to the first moment (momentum) matrix.
 * @param velocity      Global memory pointer to the second moment (velocity) matrix.
 * @param learning_rate Current learning rate (base LR for Adam).
 * @param beta1         Adam hyperparameter beta1.
 * @param beta2         Adam hyperparameter beta2.
 * @param epsilon       Adam hyperparameter epsilon.
 * @param t_step        Global time step (1-indexed) for bias correction.
 * @param max_grad_clip_value Maximum absolute value for gradient clipping (NEW)
 * @param num_elements  Total number of elements in the matrices.
 */
__kernel void adam_optimizer_kernel(__global float* weights,
                                    __global const float* gradients,
                                    __global float* moments,
                                    __global float* velocity,
                                    const float learning_rate,
                                    const float beta1,
                                    const float beta2,
                                    const float epsilon,
                                    const ulong t_step,
                                    const float max_grad_clip_value, // NEW: Clipping threshold
                                    const int num_elements)
{
    int gid = get_global_id(0); // Global ID for the current element

    if (gid >= num_elements) {
        return;
    }

    float g = gradients[gid]; // Raw gradient for this element

    // Sanitize gradient (before moment updates)
    if (isnan(g)) {
        g = 0.0001f;
    } else if (isinf(g)) {
        g = copysign(1000.0f, g);
    }

    // Apply element-wise gradient clipping to the raw gradient
    if (fabs(g) > max_grad_clip_value) {
        g = copysign(max_grad_clip_value, g);
    }


    float m = moments[gid];
    float v = velocity[gid];

    // Update biased first moment estimate
    m = beta1 * m + (1.0f - beta1) * g;

    // Update biased second raw moment estimate
    v = beta2 * v + (1.0f - beta2) * g * g;

    // Bias correction
    float beta1_pow_t = pow(beta1, (float)t_step);
    float beta2_pow_t = pow(beta2, (float)t_step);

    // Bias correction denominators, adding epsilon for robustness against near-zero values.
    // In strict Adam, epsilon is usually only in the final weight update denominator.
    // Check your reference implementation for exact handling.
    float denom_m = 1.0f - beta1_pow_t;
    if (denom_m < FLT_MIN) denom_m = FLT_MIN; // Use a small float constant instead of epsilon for this specific check
    // Or, more robustly, directly check `fabs(denom_m) < epsilon` if you prefer.

    float denom_v = 1.0f - beta2_pow_t;
    if (denom_v < FLT_MIN) denom_v = FLT_MIN;

    float m_hat = m / denom_m;
    float v_hat = v / denom_v;

    // Update weights
    // sqrt(v_hat) + epsilon in the denominator for numerical stability
    weights[gid] -= (learning_rate / (sqrt(v_hat) + epsilon)) * m_hat;

    // Store updated moments back to global memory
    moments[gid] = m;
    velocity[gid] = v;
}


__kernel void kernelRpropUpdate(__global float* weights,
                                __global const float* gradients, // Input: raw gradients for this step
                                __global float* prev_gradients,
                                __global float* delta_weights,
                                const float eta_plus,
                                const float eta_minus,
                                const float delta_max,
                                const float delta_min,
                                const float max_grad_clip_value, // NEW: Clipping threshold for input gradients
                                const int size)
{
    int idx = get_global_id(0);

    if (idx < size) {
        float grad = gradients[idx];

        // Apply element-wise gradient clipping to the raw input gradient
        if (fabs(grad) > max_grad_clip_value) {
            grad = copysign(max_grad_clip_value, grad);
        }

        // Sanitize gradient (post-clipping is fine, but can be before or after)
        if (isnan(grad)) { // Check the clipped grad for NaN
            grad = 0.0001f;
        } else if (isinf(grad)) { // Check the clipped grad for Inf
            grad = copysign(1000.0f, grad);
        }

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
            // Sign change: Decrease step size, revert previous update, reset prev_grad
            delta = fmax(delta * eta_minus, delta_min);
            // This line `weights[idx] += copysign(delta, prev_grad);` is part of a "revert" logic.
            // If the previous step was already applied, some Rprop variants revert it here.
            // Ensure this aligns with your Rprop implementation's exact rules.
            // For many Rprop variants, it's weights[idx] -= (prev_delta_w); and then update_val is adjusted.
            // Here, it seems to imply reverting the previous step using the *new* delta.
            // Usually it's `weights[idx] += previous_update_step;` to revert
            // and `prev_gradients[idx] = 0.0f;` and `delta_weights[idx] = delta;`
            weights[idx] += copysign(delta, prev_grad); // Revert step based on prev_grad sign (this is a common Rprop rule for sign change)
            prev_gradients[idx] = 0.0f; // Reset prev_grad to prevent double penalty
            delta_weights[idx] = delta; // Store updated delta
            return; // Exit early, no further update this step
        }
        else { // sign_change == 0.0f (current gradient is zero)
            // No change in direction, keep step size, apply update if gradient is not zero
            weight_update = -copysign(delta, grad);
            prev_gradients[idx] = grad; // Store current gradient (which is zero)
        }

        // Apply the calculated weight update
        weights[idx] += weight_update;
        // Store the updated step size
        delta_weights[idx] = delta;
    }
}