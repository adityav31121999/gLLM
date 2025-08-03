// Basic XorShift32 PRNG for OpenCL
// seed must be unique per work-item
unsigned int xorshift32(unsigned int x) {
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return x;
}

float scale_random(unsigned int* seed_ptr, float r1) {
    // Generate a random unsigned int
    *seed_ptr = xorshift32(*seed_ptr);
    // Convert to float in [0, 1] range
    float normalized_val = (float)(*seed_ptr) / (float)0xFFFFFFFFU;
    return r1 + normalized_val * (10.0f - r1);
}

__kernel void generate_embeddings(
    __global float* embeddings_out,
    const int d_dim,
    const float r1,
    const unsigned int initial_seed_offset) {

    int global_id = get_global_id(0); // 0-indexed global work-item ID
    // int total_elements = get_global_size(0); // Not directly used in this snippet

    // --- MISSING LINE ADDED HERE ---
    unsigned int seed = initial_seed_offset + global_id + 1; // Declare and initialize 'seed' for this work-item

    embeddings_out[global_id] = scale_random(&seed, r1);
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
