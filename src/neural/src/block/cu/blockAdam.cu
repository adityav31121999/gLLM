#ifdef USE_CUDA
#include "include/block.hpp" // Assumed to contain block, attention, mlp, mat
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdio> // For fprintf
#include <stdexcept> // For std::out_of_range
#include <vector>


void block::cuParallelAdamUpdate(int layers_mlp, unsigned long long t_adam, int columnNumber, float beta1, float beta2, float epsilon, float learning_rate_param)
{
    const int num_heads_to_process = x; // 'x' is the number of rows/heads in this column

    // Corrected columnNumber range check (0-indexed)
    if (columnNumber < 0 || columnNumber >= y) {
        fprintf(stderr, "cuParallelAdamUpdate: Column index 'columnNumber' (%d) is out of range [0, %d].\n", columnNumber, y - 1);
        return; // Or throw an exception for C++ style error handling
    }

    const int embedding_dim = EMBEDDING;
    const int mat_heights = MATHEIGHTS;
    const float learning_rate = learning_rate_param;

    const int num_total_layers_mlp = layers_mlp;
    const int num_weight_matrices_mlp = num_total_layers_mlp - 1;

    // Sizes for individual matrices (in number of floats)
    const size_t matValuesCount = static_cast<size_t>(mat_heights) * embedding_dim; // For MQ, MK, MV, MH
    const size_t mlpLayerValuesCount = static_cast<size_t>(embedding_dim) * embedding_dim; // For single MLP weight matrix

    const int blockSize = 256; // Typical block size, adjust for performance

    // --- Aggregate Device Pointers ---
    // Attention matrices
    float *d_aggMQ_W=nullptr, *d_aggMK_W=nullptr, *d_aggMH_W=nullptr, *d_aggMV_W=nullptr;
    float *d_aggMQ_G=nullptr, *d_aggMK_G=nullptr, *d_aggMH_G=nullptr, *d_aggMV_G=nullptr;
    float *d_aggMQ_M=nullptr, *d_aggMK_M=nullptr, *d_aggMH_M=nullptr, *d_aggMV_M=nullptr;
    float *d_aggMQ_V=nullptr, *d_aggMK_V=nullptr, *d_aggMH_V=nullptr, *d_aggMV_V=nullptr;
    // MLP matrices
    float *d_aggHor_W=nullptr, *d_aggVer_W=nullptr;
    float *d_aggHor_G=nullptr, *d_aggVer_G=nullptr;
    float *d_aggHor_M=nullptr, *d_aggVer_M=nullptr;
    float *d_aggHor_V=nullptr, *d_aggVer_V=nullptr;

    std::vector<cudaStream_t> streams_cuda(num_heads_to_process);
    std::vector<HeadDevicePointersCUDA> head_gpu_data_cuda(num_heads_to_process);


    // Lambda to calculate global grid size for kernel launch
    auto calculate_grid_size = [&](size_t total_elements) {
        return (static_cast<int>(total_elements) + blockSize - 1) / blockSize;
    };

    // --- Core Adam Update Lambda (operates on raw device pointers and a stream) ---
    auto apply_adam_kernel_on_buffers = [&](float* d_weights, const float* d_gradients,
                                             float* d_moments, float* d_velocity,
                                             size_t num_elements, // Number of FLOAT elements
                                             cudaStream_t stream) {

        if (num_elements == 0) return; // Skip empty matrices
        
        adam_optimizer_kernel_cuda<<<calculate_grid_size(num_elements), blockSize, 0, stream>>>(
            d_weights, d_gradients, d_moments, d_velocity,
            learning_rate, beta1, beta2, epsilon, t_adam, static_cast<int>(num_elements)
        );
        cudaError_t kernel_launch_err = cudaGetLastError(); // Check for errors from async launch
        if (kernel_launch_err != cudaSuccess) {
            fprintf(stderr, "CUDA kernel launch failed in lambda: %s\n", cudaGetErrorString(kernel_launch_err));
            // This is an error within an async lambda. Proper error propagation is tricky here.
            // For now, we print and expect a higher-level cudaStreamSynchronize to catch it.
        }
    };


    try {
        cudaError_t cuda_err;

        // --- Allocate All Aggregate Device Buffers ---
        // Attention Weights
        cuda_err = cudaMalloc(&d_aggMQ_W, num_heads_to_process * matValuesCount * sizeof(float)); if (cuda_err != cudaSuccess) goto cleanup;
        cuda_err = cudaMalloc(&d_aggMK_W, num_heads_to_process * matValuesCount * sizeof(float)); if (cuda_err != cudaSuccess) goto cleanup;
        cuda_err = cudaMalloc(&d_aggMH_W, num_heads_to_process * matValuesCount * sizeof(float)); if (cuda_err != cudaSuccess) goto cleanup;
        cuda_err = cudaMalloc(&d_aggMV_W, num_heads_to_process * matValuesCount * sizeof(float)); if (cuda_err != cudaSuccess) goto cleanup;
        // Attention Gradients
        cuda_err = cudaMalloc(&d_aggMQ_G, num_heads_to_process * matValuesCount * sizeof(float)); if (cuda_err != cudaSuccess) goto cleanup;
        cuda_err = cudaMalloc(&d_aggMK_G, num_heads_to_process * matValuesCount * sizeof(float)); if (cuda_err != cudaSuccess) goto cleanup;
        cuda_err = cudaMalloc(&d_aggMH_G, num_heads_to_process * matValuesCount * sizeof(float)); if (cuda_err != cudaSuccess) goto cleanup;
        cuda_err = cudaMalloc(&d_aggMV_G, num_heads_to_process * matValuesCount * sizeof(float)); if (cuda_err != cudaSuccess) goto cleanup;
        // Attention Momentum
        cuda_err = cudaMalloc(&d_aggMQ_M, num_heads_to_process * matValuesCount * sizeof(float)); if (cuda_err != cudaSuccess) goto cleanup;
        cuda_err = cudaMalloc(&d_aggMK_M, num_heads_to_process * matValuesCount * sizeof(float)); if (cuda_err != cudaSuccess) goto cleanup;
        cuda_err = cudaMalloc(&d_aggMH_M, num_heads_to_process * matValuesCount * sizeof(float)); if (cuda_err != cudaSuccess) goto cleanup;
        cuda_err = cudaMalloc(&d_aggMV_M, num_heads_to_process * matValuesCount * sizeof(float)); if (cuda_err != cudaSuccess) goto cleanup;
        // Attention Velocity
        cuda_err = cudaMalloc(&d_aggMQ_V, num_heads_to_process * matValuesCount * sizeof(float)); if (cuda_err != cudaSuccess) goto cleanup;
        cuda_err = cudaMalloc(&d_aggMK_V, num_heads_to_process * matValuesCount * sizeof(float)); if (cuda_err != cudaSuccess) goto cleanup;
        cuda_err = cudaMalloc(&d_aggMH_V, num_heads_to_process * matValuesCount * sizeof(float)); if (cuda_err != cudaSuccess) goto cleanup;
        cuda_err = cudaMalloc(&d_aggMV_V, num_heads_to_process * matValuesCount * sizeof(float)); if (cuda_err != cudaSuccess) goto cleanup;

        // MLP Weights
        cuda_err = cudaMalloc(&d_aggHor_W, num_heads_to_process * num_weight_matrices_mlp * mlpLayerValuesCount * sizeof(float)); if (cuda_err != cudaSuccess) goto cleanup;
        cuda_err = cudaMalloc(&d_aggVer_W, num_heads_to_process * num_weight_matrices_mlp * mlpLayerValuesCount * sizeof(float)); if (cuda_err != cudaSuccess) goto cleanup;
        // MLP Gradients
        cuda_err = cudaMalloc(&d_aggHor_G, num_heads_to_process * num_weight_matrices_mlp * mlpLayerValuesCount * sizeof(float)); if (cuda_err != cudaSuccess) goto cleanup;
        cuda_err = cudaMalloc(&d_aggVer_G, num_heads_to_process * num_weight_matrices_mlp * mlpLayerValuesCount * sizeof(float)); if (cuda_err != cudaSuccess) goto cleanup;
        // MLP Momentum
        cuda_err = cudaMalloc(&d_aggHor_M, num_heads_to_process * num_weight_matrices_mlp * mlpLayerValuesCount * sizeof(float)); if (cuda_err != cudaSuccess) goto cleanup;
        cuda_err = cudaMalloc(&d_aggVer_M, num_heads_to_process * num_weight_matrices_mlp * mlpLayerValuesCount * sizeof(float)); if (cuda_err != cudaSuccess) goto cleanup;
        // MLP Velocity
        cuda_err = cudaMalloc(&d_aggHor_V, num_heads_to_process * num_weight_matrices_mlp * mlpLayerValuesCount * sizeof(float)); if (cuda_err != cudaSuccess) goto cleanup;
        cuda_err = cudaMalloc(&d_aggVer_V, num_heads_to_process * num_weight_matrices_mlp * mlpLayerValuesCount * sizeof(float)); if (cuda_err != cudaSuccess) goto cleanup;


        // --- Create Streams, Sub-Pointers & Enqueue Initial Host->Device Copies ---
        for (int head_idx = 0; head_idx < num_heads_to_process; ++head_idx) {
            cuda_err = cudaStreamCreate(&streams_cuda[head_idx]);
            if (cuda_err != cudaSuccess) {
                fprintf(stderr, "CUDA stream creation failed for stream %d: %s\n", head_idx, cudaGetErrorString(cuda_err));
                goto cleanup;
            }

            // Access the specific attention head and its matrices on the host
            attention& head_obj = b[head_idx][columnNumber];

            // Resize MLP buffer vectors within HeadDevicePointersCUDA struct
            head_gpu_data_cuda[head_idx].d_hor_weights.resize(num_weight_matrices_mlp);
            head_gpu_data_cuda[head_idx].d_hor_gweights.resize(num_weight_matrices_mlp);
            head_gpu_data_cuda[head_idx].d_hor_moments.resize(num_weight_matrices_mlp);
            head_gpu_data_cuda[head_idx].d_hor_velocity.resize(num_weight_matrices_mlp);
            head_gpu_data_cuda[head_idx].d_ver_weights.resize(num_weight_matrices_mlp);
            head_gpu_data_cuda[head_idx].d_ver_gweights.resize(num_weight_matrices_mlp);
            head_gpu_data_cuda[head_idx].d_ver_moments.resize(num_weight_matrices_mlp);
            head_gpu_data_cuda[head_idx].d_ver_velocity.resize(num_weight_matrices_mlp);

            // Calculate offsets and assign sub-pointers for attention matrices
            size_t offset_mat_elements = head_idx * matValuesCount;
            size_t size_mat_bytes = matValuesCount * sizeof(float);

            head_gpu_data_cuda[head_idx].d_MQ = d_aggMQ_W + offset_mat_elements;
            head_gpu_data_cuda[head_idx].d_MK = d_aggMK_W + offset_mat_elements;
            head_gpu_data_cuda[head_idx].d_MH = d_aggMH_W + offset_mat_elements;
            head_gpu_data_cuda[head_idx].d_MV = d_aggMV_W + offset_mat_elements;

            head_gpu_data_cuda[head_idx].d_gMQ = d_aggMQ_G + offset_mat_elements;
            head_gpu_data_cuda[head_idx].d_gMK = d_aggMK_G + offset_mat_elements;
            head_gpu_data_cuda[head_idx].d_gMH = d_aggMH_G + offset_mat_elements;
            head_gpu_data_cuda[head_idx].d_gMV = d_aggMV_G + offset_mat_elements;

            head_gpu_data_cuda[head_idx].d_m_MQ = d_aggMQ_M + offset_mat_elements;
            head_gpu_data_cuda[head_idx].d_m_MK = d_aggMK_M + offset_mat_elements;
            head_gpu_data_cuda[head_idx].d_m_MH = d_aggMH_M + offset_mat_elements;
            head_gpu_data_cuda[head_idx].d_m_MV = d_aggMV_M + offset_mat_elements;

            head_gpu_data_cuda[head_idx].d_v_MQ = d_aggMQ_V + offset_mat_elements;
            head_gpu_data_cuda[head_idx].d_v_MK = d_aggMK_V + offset_mat_elements;
            head_gpu_data_cuda[head_idx].d_v_MH = d_aggMH_V + offset_mat_elements;
            head_gpu_data_cuda[head_idx].d_v_MV = d_aggMV_V + offset_mat_elements;

            // Enqueue initial host->device copies for attention matrices for this head
            cuda_err = cudaMemcpyAsync(head_gpu_data_cuda[head_idx].d_MQ, head_obj.MQ.mapped_data, size_mat_bytes, cudaMemcpyHostToDevice, streams_cuda[head_idx]); if (cuda_err != cudaSuccess) goto cleanup;
            cuda_err = cudaMemcpyAsync(head_gpu_data_cuda[head_idx].d_MK, head_obj.MK.mapped_data, size_mat_bytes, cudaMemcpyHostToDevice, streams_cuda[head_idx]); if (cuda_err != cudaSuccess) goto cleanup;
            cuda_err = cudaMemcpyAsync(head_gpu_data_cuda[head_idx].d_MH, head_obj.MH.mapped_data, size_mat_bytes, cudaMemcpyHostToDevice, streams_cuda[head_idx]); if (cuda_err != cudaSuccess) goto cleanup;
            cuda_err = cudaMemcpyAsync(head_gpu_data_cuda[head_idx].d_MV, head_obj.MV.mapped_data, size_mat_bytes, cudaMemcpyHostToDevice, streams_cuda[head_idx]); if (cuda_err != cudaSuccess) goto cleanup;

            cuda_err = cudaMemcpyAsync(head_gpu_data_cuda[head_idx].d_gMQ, head_obj.gMQ.mapped_data, size_mat_bytes, cudaMemcpyHostToDevice, streams_cuda[head_idx]); if (cuda_err != cudaSuccess) goto cleanup;
            cuda_err = cudaMemcpyAsync(head_gpu_data_cuda[head_idx].d_gMK, head_obj.gMK.mapped_data, size_mat_bytes, cudaMemcpyHostToDevice, streams_cuda[head_idx]); if (cuda_err != cudaSuccess) goto cleanup;
            cuda_err = cudaMemcpyAsync(head_gpu_data_cuda[head_idx].d_gMH, head_obj.gMH.mapped_data, size_mat_bytes, cudaMemcpyHostToDevice, streams_cuda[head_idx]); if (cuda_err != cudaSuccess) goto cleanup;
            cuda_err = cudaMemcpyAsync(head_gpu_data_cuda[head_idx].d_gMV, head_obj.gMV.mapped_data, size_mat_bytes, cudaMemcpyHostToDevice, streams_cuda[head_idx]); if (cuda_err != cudaSuccess) goto cleanup;

            cuda_err = cudaMemcpyAsync(head_gpu_data_cuda[head_idx].d_m_MQ, head_obj.m_MQ.mapped_data, size_mat_bytes, cudaMemcpyHostToDevice, streams_cuda[head_idx]); if (cuda_err != cudaSuccess) goto cleanup;
            cuda_err = cudaMemcpyAsync(head_gpu_data_cuda[head_idx].d_m_MK, head_obj.m_MK.mapped_data, size_mat_bytes, cudaMemcpyHostToDevice, streams_cuda[head_idx]); if (cuda_err != cudaSuccess) goto cleanup;
            cuda_err = cudaMemcpyAsync(head_gpu_data_cuda[head_idx].d_m_MH, head_obj.m_MH.mapped_data, size_mat_bytes, cudaMemcpyHostToDevice, streams_cuda[head_idx]); if (cuda_err != cudaSuccess) goto cleanup;
            cuda_err = cudaMemcpyAsync(head_gpu_data_cuda[head_idx].d_m_MV, head_obj.m_MV.mapped_data, size_mat_bytes, cudaMemcpyHostToDevice, streams_cuda[head_idx]); if (cuda_err != cudaSuccess) goto cleanup;

            cuda_err = cudaMemcpyAsync(head_gpu_data_cuda[head_idx].d_v_MQ, head_obj.v_MQ.mapped_data, size_mat_bytes, cudaMemcpyHostToDevice, streams_cuda[head_idx]); if (cuda_err != cudaSuccess) goto cleanup;
            cuda_err = cudaMemcpyAsync(head_gpu_data_cuda[head_idx].d_v_MK, head_obj.v_MK.mapped_data, size_mat_bytes, cudaMemcpyHostToDevice, streams_cuda[head_idx]); if (cuda_err != cudaSuccess) goto cleanup;
            cuda_err = cudaMemcpyAsync(head_gpu_data_cuda[head_idx].d_v_MH, head_obj.v_MH.mapped_data, size_mat_bytes, cudaMemcpyHostToDevice, streams_cuda[head_idx]); if (cuda_err != cudaSuccess) goto cleanup;
            cuda_err = cudaMemcpyAsync(head_gpu_data_cuda[head_idx].d_v_MV, head_obj.v_MV.mapped_data, size_mat_bytes, cudaMemcpyHostToDevice, streams_cuda[head_idx]); if (cuda_err != cudaSuccess) goto cleanup;

            // MLP Weight Matrices
            size_t size_mlp_layer_bytes = mlpLayerValuesCount * sizeof(float);
            for (int l = 0; l < num_weight_matrices_mlp; ++l) {
                size_t offset_mlp_elements = (head_idx * num_weight_matrices_mlp + l) * mlpLayerValuesCount;

                head_gpu_data_cuda[head_idx].d_hor_weights[l] = d_aggHor_W + offset_mlp_elements;
                head_gpu_data_cuda[head_idx].d_ver_weights[l] = d_aggVer_W + offset_mlp_elements;
                head_gpu_data_cuda[head_idx].d_hor_gweights[l] = d_aggHor_G + offset_mlp_elements;
                head_gpu_data_cuda[head_idx].d_ver_gweights[l] = d_aggVer_G + offset_mlp_elements;
                head_gpu_data_cuda[head_idx].d_hor_moments[l] = d_aggHor_M + offset_mlp_elements;
                head_gpu_data_cuda[head_idx].d_ver_moments[l] = d_aggVer_M + offset_mlp_elements;
                head_gpu_data_cuda[head_idx].d_hor_velocity[l] = d_aggHor_V + offset_mlp_elements;
                head_gpu_data_cuda[head_idx].d_ver_velocity[l] = d_aggVer_V + offset_mlp_elements;

                // Enqueue initial host->device copies for MLP matrices for this head and layer
                cuda_err = cudaMemcpyAsync(head_gpu_data_cuda[head_idx].d_hor_weights[l], head_obj.hor.weights[l].mapped_data, size_mlp_layer_bytes, cudaMemcpyHostToDevice, streams_cuda[head_idx]); if (cuda_err != cudaSuccess) goto cleanup;
                cuda_err = cudaMemcpyAsync(head_gpu_data_cuda[head_idx].d_hor_gweights[l], head_obj.hor.gweights[l].mapped_data, size_mlp_layer_bytes, cudaMemcpyHostToDevice, streams_cuda[head_idx]); if (cuda_err != cudaSuccess) goto cleanup;
                cuda_err = cudaMemcpyAsync(head_gpu_data_cuda[head_idx].d_hor_moments[l], head_obj.hor.moments[l].mapped_data, size_mlp_layer_bytes, cudaMemcpyHostToDevice, streams_cuda[head_idx]); if (cuda_err != cudaSuccess) goto cleanup;
                cuda_err = cudaMemcpyAsync(head_gpu_data_cuda[head_idx].d_hor_velocity[l], head_obj.hor.velocity[l].mapped_data, size_mlp_layer_bytes, cudaMemcpyHostToDevice, streams_cuda[head_idx]); if (cuda_err != cudaSuccess) goto cleanup;

                cuda_err = cudaMemcpyAsync(head_gpu_data_cuda[head_idx].d_ver_weights[l], head_obj.ver.weights[l].mapped_data, size_mlp_layer_bytes, cudaMemcpyHostToDevice, streams_cuda[head_idx]); if (cuda_err != cudaSuccess) goto cleanup;
                cuda_err = cudaMemcpyAsync(head_gpu_data_cuda[head_idx].d_ver_gweights[l], head_obj.ver.gweights[l].mapped_data, size_mlp_layer_bytes, cudaMemcpyHostToDevice, streams_cuda[head_idx]); if (cuda_err != cudaSuccess) goto cleanup;
                cuda_err = cudaMemcpyAsync(head_gpu_data_cuda[head_idx].d_ver_moments[l], head_obj.ver.moments[l].mapped_data, size_mlp_layer_bytes, cudaMemcpyHostToDevice, streams_cuda[head_idx]); if (cuda_err != cudaSuccess) goto cleanup;
                cuda_err = cudaMemcpyAsync(head_gpu_data_cuda[head_idx].d_ver_velocity[l], head_obj.ver.velocity[l].mapped_data, size_mlp_layer_bytes, cudaMemcpyHostToDevice, streams_cuda[head_idx]); if (cuda_err != cudaSuccess) goto cleanup;
            }
        } // End of per-head stream/sub-pointer creation and initial data transfer


        // --- Enqueue Adam Kernel Launches for Each Head ---
        for (int head_idx = 0; head_idx < num_heads_to_process; ++head_idx) {
            HeadDevicePointersCUDA& device_ptrs = head_gpu_data_cuda[head_idx];
            cudaStream_t current_stream = streams_cuda[head_idx];

            // Attention Matrix Updates
            apply_adam_kernel_on_buffers(device_ptrs.d_MQ, device_ptrs.d_gMQ, device_ptrs.d_m_MQ, device_ptrs.d_v_MQ,
                                          matValuesCount, current_stream);
            apply_adam_kernel_on_buffers(device_ptrs.d_MK, device_ptrs.d_gMK, device_ptrs.d_m_MK, device_ptrs.d_v_MK,
                                          matValuesCount, current_stream);
            apply_adam_kernel_on_buffers(device_ptrs.d_MH, device_ptrs.d_gMH, device_ptrs.d_m_MH, device_ptrs.d_v_MH,
                                          matValuesCount, current_stream);
            apply_adam_kernel_on_buffers(device_ptrs.d_MV, device_ptrs.d_gMV, device_ptrs.d_m_MV, device_ptrs.d_v_MV,
                                          matValuesCount, current_stream);

            // MLP Weight Matrix Updates
            for(int i = 0; i < num_weight_matrices_mlp; i++) {
                apply_adam_kernel_on_buffers(device_ptrs.d_hor_weights[i], device_ptrs.d_hor_gweights[i],
                                              device_ptrs.d_hor_moments[i], device_ptrs.d_hor_velocity[i],
                                              mlpLayerValuesCount, current_stream);
                apply_adam_kernel_on_buffers(device_ptrs.d_ver_weights[i], device_ptrs.d_ver_gweights[i],
                                              device_ptrs.d_ver_moments[i], device_ptrs.d_ver_velocity[i],
                                              mlpLayerValuesCount, current_stream);
            }
        }

        // --- Enqueue Final Device->Host Copies for Each Head ---
        for (int head_idx = 0; head_idx < num_heads_to_process; ++head_idx) {
            attention& head_obj = b[head_idx][columnNumber]; // Get host object reference
            HeadDevicePointersCUDA& device_ptrs = head_gpu_data_cuda[head_idx];
            cudaStream_t current_stream = streams_cuda[head_idx];

            size_t size_mat_bytes = matValuesCount * sizeof(float);
            size_t size_mlp_layer_bytes = mlpLayerValuesCount * sizeof(float);

            // Attention Matrices
            cuda_err = cudaMemcpyAsync(head_obj.MQ.mapped_data, device_ptrs.d_MQ, size_mat_bytes, cudaMemcpyDeviceToHost, current_stream); if (cuda_err != cudaSuccess) goto cleanup;
            cuda_err = cudaMemcpyAsync(head_obj.MK.mapped_data, device_ptrs.d_MK, size_mat_bytes, cudaMemcpyDeviceToHost, current_stream); if (cuda_err != cudaSuccess) goto cleanup;
            cuda_err = cudaMemcpyAsync(head_obj.MH.mapped_data, device_ptrs.d_MH, size_mat_bytes, cudaMemcpyDeviceToHost, current_stream); if (cuda_err != cudaSuccess) goto cleanup;
            cuda_err = cudaMemcpyAsync(head_obj.MV.mapped_data, device_ptrs.d_MV, size_mat_bytes, cudaMemcpyDeviceToHost, current_stream); if (cuda_err != cudaSuccess) goto cleanup;

            cuda_err = cudaMemcpyAsync(head_obj.m_MQ.mapped_data, device_ptrs.d_m_MQ, size_mat_bytes, cudaMemcpyDeviceToHost, current_stream); if (cuda_err != cudaSuccess) goto cleanup;
            cuda_err = cudaMemcpyAsync(head_obj.m_MK.mapped_data, device_ptrs.d_m_MK, size_mat_bytes, cudaMemcpyDeviceToHost, current_stream); if (cuda_err != cudaSuccess) goto cleanup;
            cuda_err = cudaMemcpyAsync(head_obj.m_MH.mapped_data, device_ptrs.d_m_MH, size_mat_bytes, cudaMemcpyDeviceToHost, current_stream); if (cuda_err != cudaSuccess) goto cleanup;
            cuda_err = cudaMemcpyAsync(head_obj.m_MV.mapped_data, device_ptrs.d_m_MV, size_mat_bytes, cudaMemcpyDeviceToHost, current_stream); if (cuda_err != cudaSuccess) goto cleanup;

            cuda_err = cudaMemcpyAsync(head_obj.v_MQ.mapped_data, device_ptrs.d_v_MQ, size_mat_bytes, cudaMemcpyDeviceToHost, current_stream); if (cuda_err != cudaSuccess) goto cleanup;
            cuda_err = cudaMemcpyAsync(head_obj.v_MK.mapped_data, device_ptrs.d_v_MK, size_mat_bytes, cudaMemcpyHostToDevice, current_stream); if (cuda_err != cudaSuccess) goto cleanup;
            cuda_err = cudaMemcpyAsync(head_obj.v_MH.mapped_data, device_ptrs.d_v_MH, size_mat_bytes, cudaMemcpyHostToDevice, current_stream); if (cuda_err != cudaSuccess) goto cleanup;
            cuda_err = cudaMemcpyAsync(head_obj.v_MV.mapped_data, device_ptrs.d_v_MV, size_mat_bytes, cudaMemcpyHostToDevice, current_stream); if (cuda_err != cudaSuccess) goto cleanup;

            // MLP Matrices
            for(int l = 0; l < num_weight_matrices_mlp; ++l) {
                cuda_err = cudaMemcpyAsync(head_obj.hor.weights[l].mapped_data, device_ptrs.d_hor_weights[l], size_mlp_layer_bytes, cudaMemcpyDeviceToHost, current_stream); if (cuda_err != cudaSuccess) goto cleanup;
                cuda_err = cudaMemcpyAsync(head_obj.hor.moments[l].mapped_data, device_ptrs.d_hor_moments[l], size_mlp_layer_bytes, cudaMemcpyDeviceToHost, current_stream); if (cuda_err != cudaSuccess) goto cleanup;
                cuda_err = cudaMemcpyAsync(head_obj.hor.velocity[l].mapped_data, device_ptrs.d_hor_velocity[l], size_mlp_layer_bytes, cudaMemcpyDeviceToHost, current_stream); if (cuda_err != cudaSuccess) goto cleanup;

                cuda_err = cudaMemcpyAsync(head_obj.ver.weights[l].mapped_data, device_ptrs.d_ver_weights[l], size_mlp_layer_bytes, cudaMemcpyDeviceToHost, current_stream); if (cuda_err != cudaSuccess) goto cleanup;
                cuda_err = cudaMemcpyAsync(head_obj.ver.moments[l].mapped_data, device_ptrs.d_ver_moments[l], size_mlp_layer_bytes, cudaMemcpyDeviceToHost, current_stream); if (cuda_err != cudaSuccess) goto cleanup;
                cuda_err = cudaMemcpyAsync(head_obj.ver.velocity[l].mapped_data, device_ptrs.d_ver_velocity[l], size_mlp_layer_bytes, cudaMemcpyHostToDevice, current_stream); if (cuda_err != cudaSuccess) goto cleanup;
            }
        }

        // --- Final Synchronization and Cleanup ---
        // Synchronize all streams to ensure all GPU operations are complete
        for (int head_idx = 0; head_idx < num_heads_to_process; ++head_idx) {
            cuda_err = cudaStreamSynchronize(streams_cuda[head_idx]);
            if (cuda_err != cudaSuccess) {
                fprintf(stderr, "CUDA stream synchronization failed for stream %d: %s\n", head_idx, cudaGetErrorString(cuda_err));
            }
        }

    cleanup:
        // Destroy streams
        for (int head_idx = 0; head_idx < num_heads_to_process; ++head_idx) {
            if (streams_cuda[head_idx]) { // Check if stream was successfully created
                cudaStreamDestroy(streams_cuda[head_idx]);
            }
        }

        // Free aggregate device memory.
        if (d_aggMQ_W) cudaFree(d_aggMQ_W); if (d_aggMK_W) cudaFree(d_aggMK_W); if (d_aggMH_W) cudaFree(d_aggMH_W); if (d_aggMV_W) cudaFree(d_aggMV_W);
        if (d_aggMQ_G) cudaFree(d_aggMQ_G); if (d_aggMK_G) cudaFree(d_aggMK_G); if (d_aggMH_G) cudaFree(d_aggMH_G); if (d_aggMV_G) cudaFree(d_aggMV_G);
        if (d_aggMQ_M) cudaFree(d_aggMQ_M); if (d_aggMK_M) cudaFree(d_aggMK_M); if (d_aggMH_M) cudaFree(d_aggMH_M); if (d_aggMV_M) cudaFree(d_aggMV_M);
        if (d_aggMQ_V) cudaFree(d_aggMQ_V); if (d_aggMK_V) cudaFree(d_aggMK_V); if (d_aggMH_V) cudaFree(d_aggMH_V); if (d_aggMV_V) cudaFree(d_aggMV_V);

        if (d_aggHor_W) cudaFree(d_aggHor_W); if (d_aggVer_W) cudaFree(d_aggVer_W);
        if (d_aggHor_G) cudaFree(d_aggHor_G); if (d_aggVer_G) cudaFree(d_aggVer_G);
        if (d_aggHor_M) cudaFree(d_aggHor_M); if (d_aggVer_M) cudaFree(d_aggVer_M);
        if (d_aggHor_V) cudaFree(d_aggHor_V); if (d_aggVer_V) cudaFree(d_aggVer_V);

        if (cuda_err != cudaSuccess) { // Only re-throw if an error occurred before cleanup
            throw std::runtime_error(std::string("CUDA Error in block::cuParallelAdamUpdate: ") + cudaGetErrorString(cuda_err));
        }

    } // End of try block
    catch (const std::exception& e) {
        std::cerr << "Standard Exception during cuParallelAdamUpdate for column " << columnNumber << ": " << e.what() << std::endl;
        throw; // Re-throw the exception
    }
}

#endif // USE_CUDA