// ---------------- forward pass ---------------- //

__kernel void kernelLayerForward(__global const float* inputs,
                                 __global const float* weights,
                                 __global float* outputs,
                                 const int input_size,
                                 const int output_size)
{
    int neuron_idx = get_global_id(0); 

    if (neuron_idx < output_size) {
        float sum = 0.0f;
        int row_offset = neuron_idx * input_size;

        // Process in chunks of 4
        int i = 0;
        for (; i <= input_size - 4; i += 4) {
            float4 in_f4 = vload4(0, inputs + i);
            float4 w_f4 = vload4(0, weights + row_offset + i);
            sum += dot(in_f4, w_f4);
        }
        // Process remaining elements
        for (; i < input_size; ++i) {
            sum += inputs[i] * weights[row_offset + i];
        }
        
        outputs[neuron_idx] = sum;
    }
}

// ---------------- backward pass ---------------- //

__kernel void kernelOutputDelta(__global const float* output, 
                                __global const float* expected, 
                                __global float* delta, 
                                const int size) 
{
    int idx = get_global_id(0);
    if (idx < size) {
        delta[idx] = output[idx] - expected[idx];
    }
}

__kernel void kernelComputeGradMLPInput(__global const float* deltas,
                                        __global const float* weights,
                                        __global float* grad_input,
                                        const int current_layer_size,
                                        const int input_size)
{
    int i = get_global_id(0); 

    if (i < input_size) {
        float sum = 0.0f;
        // Memory coalescing: adjacent threads read adjacent weights in the column
        for (int j = 0; j < current_layer_size; ++j) {
            sum += deltas[j] * weights[j * input_size + i];
        }
        grad_input[i] = sum;
    }
}

__kernel void kernelOutputDeltaSigmoid(__global const float* activations,
                                       __global const float* expected,
                                       __global float* deltas,
                                       const int size)
{
    int idx = get_global_id(0);
    if (idx < size) {
        float act = activations[idx];
        deltas[idx] = (act - expected[idx]) * act * (1.0f - act);
    }
}

__kernel void kernelHiddenDeltaSigmoid(__global const float* next_layer_deltas,
                                       __global const float* weights, 
                                       __global const float* activations,
                                       __global float* deltas,
                                       const int current_layer_size,
                                       const int next_layer_size)
{
    int i = get_global_id(0); 

    if (i < current_layer_size) {
        float error_sum = 0.0f;
        // weights[j * current_layer_size + i] access is coalesced
        for (int j = 0; j < next_layer_size; j++) {
            error_sum += next_layer_deltas[j] * weights[j * current_layer_size + i];
        }

        float act = activations[i];
        deltas[i] = error_sum * act * (1.0f - act);
    }
}

__kernel void kernelLastLayerDeltaSigmoid(__global const float* grad_output,
                                          __global const float* activations,
                                          __global float* deltas,
                                          const int size)
{
    int idx = get_global_id(0);
    if (idx < size) {
        float act = activations[idx];
        // Corrected: Uses sigmoid derivative instead of ReLU logic
        deltas[idx] = grad_output[idx] * act * (1.0f - act);
    }
}

__kernel void kernelUpdateInputMLP(__global float* input, 
                                   __global const float* weights,
                                   __global const float* deltas, 
                                   const float learning_rate,
                                   const int first_hidden_layer_size, 
                                   const int input_size)
{
    int i = get_global_id(0); 

    if (i < input_size) {
        float grad_sum = 0.0f;
        for (int j = 0; j < first_hidden_layer_size; j++) {
            grad_sum += deltas[j] * weights[j * input_size + i];
        }
        input[i] -= learning_rate * grad_sum;
    }
}

// ---------------- update weights ---------------- //

__kernel void kernelUpdateWeights(__global const float* deltas,
                                  __global const float* prev_activations,
                                  __global float* weights,
                                  const float learning_rate,
                                  const int current_layer_size,
                                  const int prev_layer_size)
{
    int j = get_global_id(0); // Col (prev layer)
    int i = get_global_id(1); // Row (curr layer)

    if (i < current_layer_size && j < prev_layer_size) {
        weights[i * prev_layer_size + j] -= learning_rate * deltas[i] * prev_activations[j];
    }
}

__kernel void kernelUpdateWeightsAndGradients(__global const float* deltas, 
                                                __global const float* prev_activations,
                                                __global float* weights, 
                                                __global float* gweights,
                                                float learning_rate,
                                                int current_layer_size,
                                                int prev_layer_size)
{
    int j = get_global_id(0); // Col
    int i = get_global_id(1); // Row

    if (i < current_layer_size && j < prev_layer_size) {
        int idx = i * prev_layer_size + j;
        float grad = deltas[i] * prev_activations[j];
        
        gweights[idx] = grad;
        weights[idx] -= learning_rate * grad;
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
    int j = get_global_id(0); // col
    int i = get_global_id(1); // row

    if (j < prev_layer_size && i < current_layer_size) {
        int idx = i * prev_layer_size + j;
        float err_grad = deltas[i] * prev_activations[j];
        float w = weights[idx];
        
        float sign_w = (w > 0.0f) ? 1.0f : ((w < 0.0f) ? -1.0f : 0.0f);
        float total_grad = err_grad + (lambda_l1 * sign_w) + (lambda_l2 * w);

        if (gweights != NULL) gweights[idx] = total_grad;
        weights[idx] -= learning_rate * total_grad;
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
    int j = get_global_id(0);
    int i = get_global_id(1);

    if (j < prev_layer_size && i < current_layer_size) {
        int idx = i * prev_layer_size + j;
        float w = weights[idx];
        float sign_w = (w > 0.0f) ? 1.0f : ((w < 0.0f) ? -1.0f : 0.0f);
        
        float total_grad = gweights[idx] + (lambda_l1 * sign_w) + (lambda_l2 * w);
        weights[idx] -= learning_rate * total_grad;
    }
}

__kernel void kernelRpropUpdate(__global float* weights,
                                __global const float* gradients,
                                __global float* prev_gradients,
                                __global float* delta_weights,
                                const float eta_plus,
                                const float eta_minus,
                                const float delta_max,
                                const float delta_min,
                                const int size)
{
    int idx = get_global_id(0);

    if (idx < size) {
        float grad = gradients[idx];
        float prev_grad = prev_gradients[idx];
        float delta = delta_weights[idx];
        
        float sign_change = grad * prev_grad;

        if (sign_change > 0.0f) {
            delta = fmin(delta * eta_plus, delta_max);
            weights[idx] -= copysign(delta, grad);
            prev_gradients[idx] = grad;
        }
        else if (sign_change < 0.0f) {
            delta = fmax(delta * eta_minus, delta_min);
            // Backtrack: revert the previous weight update
            weights[idx] += copysign(delta, prev_grad); 
            prev_gradients[idx] = 0.0f; // Reset gradient
        }
        else {
            weights[idx] -= copysign(delta, grad);
            prev_gradients[idx] = grad;
        }

        delta_weights[idx] = delta;
    }
}