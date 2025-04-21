
// cl/loss.cl
// OpenCL kernels for MLP loss and penalty calculations

#include <CL/cl.hpp>
#include <cmath>

/*
 * Kernel to calculate the L1 penalty (sum of absolute values) using parallel reduction.
 * Writes partial sums per work-group to the result buffer.
 */
__kernel void l1PenaltyKernel(__global const float* weights,
                              __global float* result,
                              __local float* temp_sum,
                              const int size)
{
    int global_id = get_global_id(0);
    int local_id = get_local_id(0);
    int group_id = get_group_id(0);
    int local_size = get_local_size(0);

    // Initialize local memory
    temp_sum[local_id] = 0.0f;

    // Each work-item processes multiple elements if necessary
    for (int i = global_id; i < size; i += get_global_size(0)) {
        temp_sum[local_id] += fabs(weights[i]);
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

/*
 * Kernel to calculate the L2 penalty (sum of squares) using parallel reduction.
 * Writes partial sums per work-group to the result buffer.
 */
__kernel void l2PenaltyKernel(__global const float* weights,
                              __global float* result,
                              __local float* temp_sum,
                              const int size)
{
    int global_id = get_global_id(0);
    int local_id = get_local_id(0);
    int group_id = get_group_id(0);
    int local_size = get_local_size(0);

    // Initialize local memory
    temp_sum[local_id] = 0.0f;

    // Each work-item processes multiple elements if necessary
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

/*
 * Kernel to calculate the sum of absolute differences using parallel reduction.
 * Writes partial sums per work-group to the result buffer.
 */
__kernel void absDiffKernel(__global const float* outputs,
                            __global const float* targets,
                            __global float* result,
                            __local float* temp_sum,
                            const int size)
{
    int global_id = get_global_id(0);
    int local_id = get_local_id(0);
    int group_id = get_group_id(0);
    int local_size = get_local_size(0);

    // Initialize local memory
    temp_sum[local_id] = 0.0f;

    // Each work-item processes multiple elements if necessary
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

/*
 * Kernel to calculate the sum of squared differences using parallel reduction.
 * Writes partial sums per work-group to the result buffer.
 */
__kernel void squaredDiffKernel(__global const float* outputs,
                                __global const float* targets,
                                __global float* result,
                                __local float* temp_sum,
                                const int size)
{
    int global_id = get_global_id(0);
    int local_id = get_local_id(0);
    int group_id = get_group_id(0);
    int local_size = get_local_size(0);

    // Initialize local memory
    temp_sum[local_id] = 0.0f;

    // Each work-item processes multiple elements if necessary
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
