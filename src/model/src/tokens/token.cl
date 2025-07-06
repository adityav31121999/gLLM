
/** 
    * --> f(i, j, seed) = (i * j + 1) * C * (seed^[j%d])
    * where: C = 0.01, x = seed, and d is the embedding dimension.
    */
__kernel void embeddingFormulaBatchKernel(__global float* all_embeddings, 
    __global const float* all_seeds, const int N, const int d) 
{
    // Use OpenCL's direct global IDs
    int j = get_global_id(0);
    int i = get_global_id(1);

    if (i < N && j < d) {
        float seed = all_seeds[i];
        const float C = 0.01f;
        float result = (float)(i * j + 1) * C;
        int exponent = j;
        // Use OpenCL's pow() function
        result *= pow(seed, (float)exponent);
        all_embeddings[i * d + j] = result;
    }
}


__kernel void batchedVectorInverseKernel(__global float* output, 
    __global const float* input, const int N, const int d, __local float* s_data)
{
    // Identify which row (vector) this work-group is working on.
    const int row_idx = get_group_id(1);

    // Identify the thread's local and global column indices.
    const int tid_in_block = get_local_id(0);
    const int col_idx = get_global_id(0);

    // --- Step 1: Parallel Reduction to find the squared magnitude ---
    float my_val = 0.0f;
    if (col_idx < d) {
        my_val = input[row_idx * d + col_idx];
    }
    
    s_data[tid_in_block] = my_val * my_val;

    // Synchronize to make sure all work-items have written to local memory.
    barrier(CLK_LOCAL_MEM_FENCE);

    // Perform the reduction in local memory.
    for (unsigned int s = get_local_size(0) / 2; s > 0; s >>= 1) {
        if (tid_in_block < s) {
            s_data[tid_in_block] += s_data[tid_in_block + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // The first work-item holds the final result.
    const float squared_magnitude = s_data[0];

    // --- Step 2: Element-wise division ---
    if (col_idx < d) {
        if (squared_magnitude > 1e-9f) {
            output[row_idx * d + col_idx] = my_val / squared_magnitude;
        } else {
            output[row_idx * d + col_idx] = 0.0f;
        }
    }
}
