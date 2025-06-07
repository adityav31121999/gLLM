
#include "include/mlp.hpp"
#include "include/attention.hpp" // Includes constants like EMBEDDING, MATHEIGHTS, etc.
#include "include/block.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept> // For runtime_error, out_of_range
#include <iostream> // For error logging
#include <string>   // For std::to_string in error messages
#include <maths.hpp> // For SCALING, LOTA, LOTAder (if needed host-side, unlikely here)

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        throw std::runtime_error(cudaGetErrorString(err)); \
    } \
} while (0)

// Helper function to flatten a 2D vector (used for expectedV_head)
// This should ideally be in a common utility header.
template <typename T>
static std::vector<T> flatten(const std::vector<std::vector<T>>& vec_2d) {
    std::vector<T> res;
    for (const auto& inner_vec : vec_2d) {
        res.insert(res.end(), inner_vec.begin(), inner_vec.end());
    }
    return res;
}

struct HeadDevicePointersV {
    float *d_expected_v;
    float *d_EV;
    float *d_grad_EV_full, *d_grad_EV_summed, *d_grad_EV_scaled;
    float *d_grad_dv;
    float *d_KdotQ, *d_head;
    float *d_K, *d_Q;
    float *d_pre_MV;
    float *d_MV_a, *d_MQ_a, *d_MK_a;
    float *d_grad_MV;
    float *d_grad_head;
    float *d_lota_deriv;
    float *d_grad_KdotQ;
    float *d_grad_Q;
    float *d_grad_MQ, *d_grad_MK_correction;

    std::vector<float*> d_ver_activations;
    std::vector<float*> d_ver_weights, d_ver_gweights, d_ver_deltas;
};

/**
 * @brief CUDA backward propagation for a single column in the FIRST block,
 *        driven by vertical error (EV). Processes heads b[row][layno].
 *        IMPLEMENTATION USES DIRECT KERNEL CALLS PER HEAD (INEFFICIENT).
 * @param expectedV Expected vertical embeddings for all heads in this column. Shape: [x][CONTEXT_WIN][EMBEDDING].
 * @param in Embedding dimension.
 * @param layers Number of MLP layers.
 * @param layno The column index within the block (0 to y-1).
 */
void block::cu1ParallelBackward1stBlock(std::vector<std::vector<std::vector<float>>>& expectedV, int& in, int& layers, int layno)
{
    const int num_heads_to_process = x; // 'x' is the number of rows/heads in this column

    // Validate column number and input shape
    if (layno < 0 || layno >= y) {
        throw std::out_of_range("cu1ParallelBackward1stBlock(V): Column index 'layno' (" + std::to_string(layno) + ") is out of range [0, " + std::to_string(y - 1) + "].");
    }
    if (expectedV.size() != static_cast<size_t>(x)) {
        throw std::runtime_error("ExpectedV size mismatch (number of rows) in cu1ParallelBackward1stBlock(V). Expected " + std::to_string(x) + ", got " + std::to_string(expectedV.size()));
    }
    if (!expectedV.empty() && (expectedV[0].size() != CONTEXT_WIN || (!expectedV[0].empty() && expectedV[0][0].size() != EMBEDDING))) {
        throw std::runtime_error("ExpectedV dimensions mismatch (context/embedding) in cu1ParallelBackward1stBlock(V).");
    }
    if (EMBEDDING != in) {
        throw std::runtime_error("Embedding dimension mismatch: EMBEDDING vs in");
    }

    // Constants
    const int embedding_dim = EMBEDDING;
    const int mat_heights = MATHEIGHTS;
    const int context_win = CONTEXT_WIN;
    const float learning_rate = LEARNING;
    const float scaling_factor = SCALING;

    // MLP structure parameters
    const int num_total_layers_mlp = layers;
    const int num_neuron_layers_mlp = num_total_layers_mlp;
    const int num_weight_matrices_mlp = num_total_layers_mlp - 1;

    const size_t embed_bytes = embedding_dim * sizeof(float);
    const size_t mlp_weights_elements = static_cast<size_t>(embedding_dim) * embedding_dim; // Assuming square matrices for simplicity here
    const size_t mlp_weights_bytes = mlp_weights_elements * sizeof(float);
    const size_t proj_mat_elements = static_cast<size_t>(mat_heights) * embedding_dim; // For MV, MQ, MK
    const size_t proj_mat_bytes = proj_mat_elements * sizeof(float);
    const size_t ev_total_elements = static_cast<size_t>(context_win) * embedding_dim; // For full EV
    const size_t ev_total_bytes = ev_total_elements * sizeof(float);
    const size_t max_head_elements_per_head = static_cast<size_t>(context_win) * context_win;
    const size_t max_head_bytes_per_head = max_head_elements_per_head * sizeof(float);
    const size_t max_k_q_elements_per_head = static_cast<size_t>(context_win) * mat_heights;
    const size_t max_k_q_bytes_per_head = max_k_q_elements_per_head * sizeof(float);
    const size_t pre_mv_elements_per_head = mat_heights;
    const size_t pre_mv_bytes_per_head = pre_mv_elements_per_head * sizeof(float);

    // Kernel Launch Config
    int threadsPerBlock1D = 256;
    dim3 blockDim1D(threadsPerBlock1D);
    int blocksPerGridEmbed = (embedding_dim + threadsPerBlock1D - 1) / threadsPerBlock1D;
    dim3 gridDimEmbed(blocksPerGridEmbed);
    int blocksPerGridMatHeights = (mat_heights + threadsPerBlock1D - 1) / threadsPerBlock1D;
    dim3 gridDimMatHeights(blocksPerGridMatHeights);
    int blocksPerGridProjMat = (static_cast<int>(proj_mat_elements) + threadsPerBlock1D - 1) / threadsPerBlock1D;
    dim3 gridDimProjMat(blocksPerGridProjMat);
    int blocksPerGridEV = (static_cast<int>(ev_total_elements) + threadsPerBlock1D - 1) / threadsPerBlock1D;
    dim3 gridDimEV(blocksPerGridEV); // For full EV updates

    dim3 blockDim2D(16, 16);
    dim3 gridDimEmbed2D((embedding_dim + blockDim2D.x - 1) / blockDim2D.x, (embedding_dim + blockDim2D.y - 1) / blockDim2D.y);
    dim3 gridDimMatrix2D((embedding_dim + blockDim2D.x - 1) / blockDim2D.x, (mat_heights + blockDim2D.y - 1) / blockDim2D.y);

    // --- Aggregate Device Memory Pointers ---
    float *agg_d_expected_v = nullptr, *agg_d_EV = nullptr;
    float *agg_d_grad_EV_full = nullptr, *agg_d_grad_EV_summed = nullptr, *agg_d_grad_EV_scaled = nullptr;
    float *agg_d_grad_dv = nullptr;
    float *agg_d_KdotQ = nullptr, *agg_d_head_storage = nullptr;
    float *agg_d_K = nullptr, *agg_d_Q = nullptr;
    float *agg_d_pre_MV = nullptr;
    float *agg_d_MV_a = nullptr, *agg_d_MQ_a = nullptr, *agg_d_MK_a = nullptr;
    float *agg_d_grad_MV = nullptr;
    float *agg_d_grad_head_storage = nullptr;
    float *agg_d_lota_deriv = nullptr;
    float *agg_d_grad_KdotQ = nullptr;
    float *agg_d_grad_Q_storage = nullptr;
    float *agg_d_grad_MQ = nullptr, *agg_d_grad_MK_correction = nullptr;
    // MLP Aggregate Storage (ver only)
    float *agg_d_ver_activations_storage = nullptr;
    float *agg_d_ver_weights_storage = nullptr;
    float *agg_d_ver_gweights_storage = nullptr;
    float *agg_d_ver_deltas_storage = nullptr;

    std::vector<cudaStream_t> streams(num_heads_to_process, nullptr);
    std::vector<HeadDevicePointersV> head_gpu_data(num_heads_to_process);

    try {
        // --- Allocate Aggregate Memory ---
        CUDA_CHECK(cudaMalloc(&agg_d_expected_v, num_heads_to_process * ev_total_bytes));
        CUDA_CHECK(cudaMalloc(&agg_d_EV, num_heads_to_process * ev_total_bytes));
        CUDA_CHECK(cudaMalloc(&agg_d_grad_EV_full, num_heads_to_process * ev_total_bytes));
        CUDA_CHECK(cudaMalloc(&agg_d_grad_EV_summed, num_heads_to_process * embed_bytes));
        CUDA_CHECK(cudaMalloc(&agg_d_grad_EV_scaled, num_heads_to_process * embed_bytes));
        CUDA_CHECK(cudaMalloc(&agg_d_grad_dv, num_heads_to_process * embed_bytes));
        CUDA_CHECK(cudaMalloc(&agg_d_KdotQ, num_heads_to_process * max_head_bytes_per_head));
        CUDA_CHECK(cudaMalloc(&agg_d_head_storage, num_heads_to_process * max_head_bytes_per_head));
        CUDA_CHECK(cudaMalloc(&agg_d_K, num_heads_to_process * max_k_q_bytes_per_head));
        CUDA_CHECK(cudaMalloc(&agg_d_Q, num_heads_to_process * max_k_q_bytes_per_head));
        CUDA_CHECK(cudaMalloc(&agg_d_pre_MV, num_heads_to_process * pre_mv_bytes_per_head));
        CUDA_CHECK(cudaMalloc(&agg_d_MV_a, num_heads_to_process * proj_mat_bytes));
        CUDA_CHECK(cudaMalloc(&agg_d_MQ_a, num_heads_to_process * proj_mat_bytes));
        CUDA_CHECK(cudaMalloc(&agg_d_MK_a, num_heads_to_process * proj_mat_bytes));
        CUDA_CHECK(cudaMalloc(&agg_d_grad_MV, num_heads_to_process * proj_mat_bytes));
        CUDA_CHECK(cudaMalloc(&agg_d_grad_head_storage, num_heads_to_process * max_head_bytes_per_head));
        CUDA_CHECK(cudaMalloc(&agg_d_lota_deriv, num_heads_to_process * max_head_bytes_per_head));
        CUDA_CHECK(cudaMalloc(&agg_d_grad_KdotQ, num_heads_to_process * max_head_bytes_per_head));
        CUDA_CHECK(cudaMalloc(&agg_d_grad_Q_storage, num_heads_to_process * max_k_q_bytes_per_head));
        CUDA_CHECK(cudaMalloc(&agg_d_grad_MQ, num_heads_to_process * proj_mat_bytes));
        CUDA_CHECK(cudaMalloc(&agg_d_grad_MK_correction, num_heads_to_process * proj_mat_bytes));

        CUDA_CHECK(cudaMalloc(&agg_d_ver_activations_storage, num_heads_to_process * num_neuron_layers_mlp * embed_bytes));
        CUDA_CHECK(cudaMalloc(&agg_d_ver_weights_storage, num_heads_to_process * num_weight_matrices_mlp * mlp_weights_bytes));
        CUDA_CHECK(cudaMalloc(&agg_d_ver_gweights_storage, num_heads_to_process * num_weight_matrices_mlp * mlp_weights_bytes));
        CUDA_CHECK(cudaMalloc(&agg_d_ver_deltas_storage, num_heads_to_process * num_weight_matrices_mlp * embed_bytes));

        // --- Create Streams and Setup Per-Head Pointers ---
        for (int head_idx = 0; head_idx < num_heads_to_process; ++head_idx) {
            CUDA_CHECK(cudaStreamCreateWithFlags(&streams[head_idx], cudaStreamNonBlocking));
            head_gpu_data[head_idx].d_ver_activations.resize(num_neuron_layers_mlp);
            head_gpu_data[head_idx].d_ver_weights.resize(num_weight_matrices_mlp);
            head_gpu_data[head_idx].d_ver_gweights.resize(num_weight_matrices_mlp);
            head_gpu_data[head_idx].d_ver_deltas.resize(num_weight_matrices_mlp);

            HeadDevicePointersV& current_ptrs = head_gpu_data[head_idx];
            current_ptrs.d_expected_v = agg_d_expected_v + head_idx * ev_total_elements;
            current_ptrs.d_EV = agg_d_EV + head_idx * ev_total_elements;
            current_ptrs.d_grad_EV_full = agg_d_grad_EV_full + head_idx * ev_total_elements;
            current_ptrs.d_grad_EV_summed = agg_d_grad_EV_summed + head_idx * embedding_dim;
            current_ptrs.d_grad_EV_scaled = agg_d_grad_EV_scaled + head_idx * embedding_dim;
            current_ptrs.d_grad_dv = agg_d_grad_dv + head_idx * embedding_dim;
            current_ptrs.d_KdotQ = agg_d_KdotQ + head_idx * max_head_elements_per_head;
            current_ptrs.d_head = agg_d_head_storage + head_idx * max_head_elements_per_head;
            current_ptrs.d_K = agg_d_K + head_idx * max_k_q_elements_per_head;
            current_ptrs.d_Q = agg_d_Q + head_idx * max_k_q_elements_per_head;
            current_ptrs.d_pre_MV = agg_d_pre_MV + head_idx * pre_mv_elements_per_head;
            current_ptrs.d_MV_a = agg_d_MV_a + head_idx * proj_mat_elements;
            current_ptrs.d_MQ_a = agg_d_MQ_a + head_idx * proj_mat_elements;
            current_ptrs.d_MK_a = agg_d_MK_a + head_idx * proj_mat_elements;
            current_ptrs.d_grad_MV = agg_d_grad_MV + head_idx * proj_mat_elements;
            current_ptrs.d_grad_head = agg_d_grad_head_storage + head_idx * max_head_elements_per_head;
            current_ptrs.d_lota_deriv = agg_d_lota_deriv + head_idx * max_head_elements_per_head;
            current_ptrs.d_grad_KdotQ = agg_d_grad_KdotQ + head_idx * max_head_elements_per_head;
            current_ptrs.d_grad_Q = agg_d_grad_Q_storage + head_idx * max_k_q_elements_per_head;
            current_ptrs.d_grad_MQ = agg_d_grad_MQ + head_idx * proj_mat_elements;
            current_ptrs.d_grad_MK_correction = agg_d_grad_MK_correction + head_idx * proj_mat_elements;

            for (int l = 0; l < num_neuron_layers_mlp; ++l) {
                current_ptrs.d_ver_activations[l] = agg_d_ver_activations_storage + (head_idx * num_neuron_layers_mlp + l) * embedding_dim;
            }
            for (int l = 0; l < num_weight_matrices_mlp; ++l) {
                current_ptrs.d_ver_weights[l] = agg_d_ver_weights_storage + (head_idx * num_weight_matrices_mlp + l) * mlp_weights_elements;
                current_ptrs.d_ver_gweights[l] = agg_d_ver_gweights_storage + (head_idx * num_weight_matrices_mlp + l) * mlp_weights_elements;
                current_ptrs.d_ver_deltas[l] = agg_d_ver_deltas_storage + (head_idx * num_weight_matrices_mlp + l) * embedding_dim;
            }
        }

        // Iterate through the rows (parallels/layers) in the specified column
        for (int head_idx = 0; head_idx < num_heads_to_process; ++head_idx) {
            attention& head_obj = b[head_idx][layno]; // Reference to the current head object
            std::vector<std::vector<float>>& expectedV_head = expectedV[head_idx]; // Expected output for this head
            HeadDevicePointersV& device_ptrs = head_gpu_data[head_idx];
            cudaStream_t current_stream = streams[head_idx];

            const int token_count = head_obj.tokenCount;
            const size_t active_head_elements = static_cast<size_t>(token_count) * token_count;
            const size_t active_head_bytes = active_head_elements * sizeof(float);
            const size_t active_k_q_elements = static_cast<size_t>(token_count) * mat_heights;
            const size_t active_k_q_bytes = active_k_q_elements * sizeof(float);
            // Adjust 2D grids based on token_count
            dim3 gridDimHead2D((token_count + blockDim2D.x - 1) / blockDim2D.x, (token_count + blockDim2D.y - 1) / blockDim2D.y);
            dim3 gridDimKQGrad2D((mat_heights + blockDim2D.x - 1) / blockDim2D.x, (token_count + blockDim2D.y - 1) / blockDim2D.y);
            int blocksPerGridHead = (static_cast<int>(active_head_elements) + threadsPerBlock1D - 1) / threadsPerBlock1D;
            dim3 gridDimHead(blocksPerGridHead);

            bool is_first_head = (head_idx == 0 && layno == 0);

            // --- Data Transfer H->D (Async Attention) ---
            std::vector<float> flat_expectedV_head = flatten(expectedV_head);
            if (!head_obj.EV.mapped_data || (token_count > 0 && (!head_obj.K.mapped_data || !head_obj.Q.mapped_data || !head_obj.KdotQ.mapped_data)) ||
                !head_obj.MV.mapped_data || !head_obj.MQ.mapped_data || !head_obj.MK.mapped_data) {
                throw std::runtime_error("One or more attention mat objects have null mapped_data for head [" + std::to_string(head_idx) + "][" + std::to_string(layno) + "]");
            }

            CUDA_CHECK(cudaMemcpyAsync(device_ptrs.d_expected_v, flat_expectedV_head.data(), ev_total_bytes, cudaMemcpyHostToDevice, current_stream));
            CUDA_CHECK(cudaMemcpyAsync(device_ptrs.d_EV, head_obj.EV.mapped_data, ev_total_bytes, cudaMemcpyHostToDevice, current_stream));
            if (token_count > 0) {
                CUDA_CHECK(cudaMemcpyAsync(device_ptrs.d_KdotQ, head_obj.KdotQ.mapped_data, active_head_bytes, cudaMemcpyHostToDevice, current_stream));
                CUDA_CHECK(cudaMemcpyAsync(device_ptrs.d_K, head_obj.K.mapped_data, active_k_q_bytes, cudaMemcpyHostToDevice, current_stream));
                CUDA_CHECK(cudaMemcpyAsync(device_ptrs.d_Q, head_obj.Q.mapped_data, active_k_q_bytes, cudaMemcpyHostToDevice, current_stream));
            }
            CUDA_CHECK(cudaMemcpyAsync(device_ptrs.d_MV_a, head_obj.MV.mapped_data, proj_mat_bytes, cudaMemcpyHostToDevice, current_stream));
            CUDA_CHECK(cudaMemcpyAsync(device_ptrs.d_MQ_a, head_obj.MQ.mapped_data, proj_mat_bytes, cudaMemcpyHostToDevice, current_stream));
            CUDA_CHECK(cudaMemcpyAsync(device_ptrs.d_MK_a, head_obj.MK.mapped_data, proj_mat_bytes, cudaMemcpyHostToDevice, current_stream));

            // --- Data Transfer H->D (Async ver MLP Internals) ---
            if (head_obj.ver.activations.size() != static_cast<size_t>(num_neuron_layers_mlp) ||
                head_obj.ver.weights.size() != static_cast<size_t>(num_weight_matrices_mlp) ||
                head_obj.ver.gweights.size() != static_cast<size_t>(num_weight_matrices_mlp)) {
                 throw std::runtime_error("MLP host ver vector size mismatch for head [" + std::to_string(head_idx) + "][" + std::to_string(layno) + "]");
            }
            for (int l = 0; l < num_neuron_layers_mlp; ++l) {
                if (head_obj.ver.activations[l].empty()) {
                    throw std::runtime_error("MLP ver.activations vector is empty for head [" + std::to_string(head_idx) + "][" + std::to_string(layno) + "], layer " + std::to_string(l));
                }
                CUDA_CHECK(cudaMemcpyAsync(device_ptrs.d_ver_activations[l], head_obj.ver.activations[l].data(), embed_bytes, cudaMemcpyHostToDevice, current_stream));
            }
            for (int l = 0; l < num_weight_matrices_mlp; ++l) {
                if (!head_obj.ver.weights[l].mapped_data || head_obj.ver.weights[l].row * head_obj.ver.weights[l].col != mlp_weights_elements ||
                    !head_obj.ver.gweights[l].mapped_data || head_obj.ver.gweights[l].row * head_obj.ver.gweights[l].col != mlp_weights_elements) {
                    throw std::runtime_error("Invalid ver.weights/gweights mat for head [" + std::to_string(head_idx) + "][" + std::to_string(layno) + "], layer " + std::to_string(l));
                }
                CUDA_CHECK(cudaMemcpyAsync(device_ptrs.d_ver_weights[l], head_obj.ver.weights[l].mapped_data, mlp_weights_bytes, cudaMemcpyHostToDevice, current_stream));
                CUDA_CHECK(cudaMemcpyAsync(device_ptrs.d_ver_gweights[l], head_obj.ver.gweights[l].mapped_data, mlp_weights_bytes, cudaMemcpyHostToDevice, current_stream));
            }

            // --- Backpropagation Steps (Async Kernels) ---

            // Step 1: Compute grad_EV (full, summed, scaled)
            kernelComputeGradientsEV_V<<<gridDimEV, blockDim1D, 0, current_stream>>>(device_ptrs.d_EV, device_ptrs.d_expected_v, device_ptrs.d_grad_EV_full, device_ptrs.d_grad_EV_summed, device_ptrs.d_grad_EV_scaled, learning_rate, context_win, embedding_dim);
            CUDA_CHECK(cudaGetLastError());

            // --- Step 2: Backprop through ver MLP ---
            kernelLastLayerDelta<<<gridDimEmbed, blockDim1D, 0, current_stream>>>(device_ptrs.d_grad_EV_scaled, device_ptrs.d_ver_activations[num_neuron_layers_mlp - 1], device_ptrs.d_ver_deltas[num_weight_matrices_mlp - 1], embedding_dim);
            CUDA_CHECK(cudaGetLastError());
            for (int k = num_weight_matrices_mlp - 1; k >= 1; --k) {
                hiddenDeltaKernel<<<gridDimEmbed, blockDim1D, 0, current_stream>>>(device_ptrs.d_ver_deltas[k], device_ptrs.d_ver_weights[k], device_ptrs.d_ver_activations[k], device_ptrs.d_ver_deltas[k-1], embedding_dim, embedding_dim);
                CUDA_CHECK(cudaGetLastError());
            }
            for (int l = 0; l < num_weight_matrices_mlp; ++l) {
                updateWeightsKernel<<<gridDimEmbed2D, blockDim2D, 0, current_stream>>>(device_ptrs.d_ver_deltas[l], device_ptrs.d_ver_activations[l], device_ptrs.d_ver_weights[l], device_ptrs.d_ver_gweights[l], learning_rate, embedding_dim, embedding_dim);
                CUDA_CHECK(cudaGetLastError());
            }

            // --- Step 3: Compute grad_dv ---
            kernelComputeGradMLPInput<<<gridDimEmbed, blockDim1D, 0, current_stream>>>(device_ptrs.d_ver_deltas[0], device_ptrs.d_ver_weights[0], device_ptrs.d_grad_dv, embedding_dim, embedding_dim);
            CUDA_CHECK(cudaGetLastError());

            if (token_count > 0) {
                // --- Step 4: Compute grad_MV ---
                cuLOTA<<<gridDimHead2D, blockDim2D, 0, current_stream>>>(device_ptrs.d_KdotQ, device_ptrs.d_head, context_win, context_win, token_count, head_obj.isSelfAttention);
                CUDA_CHECK(cudaGetLastError());
                kernelComputePreMV_V<<<gridDimMatHeights, blockDim1D, 0, current_stream>>>(device_ptrs.d_head, device_ptrs.d_Q, device_ptrs.d_pre_MV, token_count, mat_heights);
                CUDA_CHECK(cudaGetLastError());
                kernelComputeGradMV_V<<<gridDimMatrix2D, blockDim2D, 0, current_stream>>>(device_ptrs.d_pre_MV, device_ptrs.d_grad_dv, device_ptrs.d_grad_MV, mat_heights, embedding_dim);
                CUDA_CHECK(cudaGetLastError());

                // --- Step 5: Compute grad_head (V path only) ---
                kernelComputeGradHead_V<<<gridDimHead2D, blockDim2D, 0, current_stream>>>(device_ptrs.d_Q, device_ptrs.d_MV_a, device_ptrs.d_grad_dv, device_ptrs.d_grad_head, token_count, mat_heights, embedding_dim);
                CUDA_CHECK(cudaGetLastError());

                // --- Step 6: Backprop through LOTA ---
                cuLOTAder<<<gridDimHead2D, blockDim2D, 0, current_stream>>>(device_ptrs.d_KdotQ, device_ptrs.d_lota_deriv, context_win, context_win, token_count, head_obj.isSelfAttention);
                CUDA_CHECK(cudaGetLastError());
                kernelComputeGradKdotQ_LOTA<<<gridDimHead, blockDim1D, 0, current_stream>>>(device_ptrs.d_grad_head, device_ptrs.d_lota_deriv, device_ptrs.d_grad_KdotQ, scaling_factor, static_cast<int>(active_head_elements));
                CUDA_CHECK(cudaGetLastError());

                // --- Step 7: Compute grad_Q ---
                kernelComputeGradQ_V<<<gridDimKQGrad2D, blockDim2D, 0, current_stream>>>(device_ptrs.d_grad_KdotQ, device_ptrs.d_K, device_ptrs.d_grad_Q, token_count, mat_heights);
                CUDA_CHECK(cudaGetLastError());

                // --- Step 8: Compute grad_MQ and grad_MK_correction ---
                kernelComputeGradMQ_V<<<gridDimMatrix2D, blockDim2D, 0, current_stream>>>(device_ptrs.d_grad_Q, nullptr, device_ptrs.d_grad_MQ, token_count, mat_heights, embedding_dim);
                CUDA_CHECK(cudaGetLastError());
                kernelComputeGradMKCorrection<<<gridDimMatrix2D, blockDim2D, 0, current_stream>>>(device_ptrs.d_grad_MQ, device_ptrs.d_Q, device_ptrs.d_K, device_ptrs.d_grad_MK_correction, token_count, mat_heights, embedding_dim);
                CUDA_CHECK(cudaGetLastError());
            } else {
                CUDA_CHECK(cudaMemsetAsync(device_ptrs.d_grad_MV, 0, proj_mat_bytes, current_stream));
                CUDA_CHECK(cudaMemsetAsync(device_ptrs.d_grad_MQ, 0, proj_mat_bytes, current_stream));
                CUDA_CHECK(cudaMemsetAsync(device_ptrs.d_grad_MK_correction, 0, proj_mat_bytes, current_stream));
            }

            // --- Step 9 & 10: Update Weights ---
            if (is_first_head) {
                kernelUpdateWeights_1stHead_V<<<gridDimProjMat, blockDim1D, 0, current_stream>>>(
                    device_ptrs.d_MV_a, device_ptrs.d_MQ_a, device_ptrs.d_MK_a,
                    device_ptrs.d_grad_MV, device_ptrs.d_grad_MQ, device_ptrs.d_grad_MK_correction,
                    learning_rate, mat_heights, embedding_dim
                );
                CUDA_CHECK(cudaGetLastError());
                // EV is updated outside this kernel for 1st head V-path, using full gradient
                kernelUpdateSimple<<<gridDimEV, blockDim1D, 0, current_stream>>>(device_ptrs.d_EV, device_ptrs.d_grad_EV_full, learning_rate, ev_total_elements);
                CUDA_CHECK(cudaGetLastError());
            } else {
                kernelUpdateSimple<<<gridDimProjMat, blockDim1D, 0, current_stream>>>(device_ptrs.d_MV_a, device_ptrs.d_grad_MV, learning_rate, proj_mat_elements);
                kernelUpdateSimple<<<gridDimProjMat, blockDim1D, 0, current_stream>>>(device_ptrs.d_MQ_a, device_ptrs.d_grad_MQ, learning_rate, proj_mat_elements);
                kernelUpdateSimple<<<gridDimProjMat, blockDim1D, 0, current_stream>>>(device_ptrs.d_MK_a, device_ptrs.d_grad_MK_correction, learning_rate, proj_mat_elements);
                CUDA_CHECK(cudaGetLastError());
                kernelUpdateSimple<<<gridDimEV, blockDim1D, 0, current_stream>>>(device_ptrs.d_EV, device_ptrs.d_grad_EV_full, learning_rate, ev_total_elements);
                CUDA_CHECK(cudaGetLastError());
            }

            // --- Data Transfer D->H (Async) ---
            for (int l = 0; l < num_weight_matrices_mlp; ++l) {
                CUDA_CHECK(cudaMemcpyAsync(head_obj.ver.weights[l].mapped_data, device_ptrs.d_ver_weights[l], mlp_weights_bytes, cudaMemcpyDeviceToHost, current_stream));
                CUDA_CHECK(cudaMemcpyAsync(head_obj.ver.gweights[l].mapped_data, device_ptrs.d_ver_gweights[l], mlp_weights_bytes, cudaMemcpyDeviceToHost, current_stream));
            }
            CUDA_CHECK(cudaMemcpyAsync(head_obj.EV.mapped_data, device_ptrs.d_EV, ev_total_bytes, cudaMemcpyDeviceToHost, current_stream));
            CUDA_CHECK(cudaMemcpyAsync(head_obj.MV.mapped_data, device_ptrs.d_MV_a, proj_mat_bytes, cudaMemcpyDeviceToHost, current_stream));
            CUDA_CHECK(cudaMemcpyAsync(head_obj.MQ.mapped_data, device_ptrs.d_MQ_a, proj_mat_bytes, cudaMemcpyDeviceToHost, current_stream));
            CUDA_CHECK(cudaMemcpyAsync(head_obj.MK.mapped_data, device_ptrs.d_MK_a, proj_mat_bytes, cudaMemcpyDeviceToHost, current_stream));
        } // End loop over heads

        // --- Synchronize all streams ---
        for (int head_idx = 0; head_idx < num_heads_to_process; ++head_idx) {
            if (streams[head_idx]) {
                CUDA_CHECK(cudaStreamSynchronize(streams[head_idx]));
            }
        }
    } // End try block
    catch (const std::exception& e) {
        std::cerr << "Error during cu1ParallelBackward1stBlock (V) for column " << layno << ": " << e.what() << std::endl;
        // Cleanup aggregate memory and streams on error
        for (int head_idx = 0; head_idx < num_heads_to_process; ++head_idx) { if (streams[head_idx]) { cudaStreamDestroy(streams[head_idx]); streams[head_idx] = nullptr; } }
        cudaFree(agg_d_expected_v); cudaFree(agg_d_EV); cudaFree(agg_d_grad_EV_full); cudaFree(agg_d_grad_EV_summed); cudaFree(agg_d_grad_EV_scaled); cudaFree(agg_d_grad_dv);
        cudaFree(agg_d_KdotQ); cudaFree(agg_d_head_storage); cudaFree(agg_d_K); cudaFree(agg_d_Q); cudaFree(agg_d_pre_MV);
        cudaFree(agg_d_MV_a); cudaFree(agg_d_MQ_a); cudaFree(agg_d_MK_a);
        cudaFree(agg_d_grad_MV); cudaFree(agg_d_grad_head_storage); cudaFree(agg_d_lota_deriv); cudaFree(agg_d_grad_KdotQ);
        cudaFree(agg_d_grad_Q_storage); cudaFree(agg_d_grad_MQ); cudaFree(agg_d_grad_MK_correction);
        cudaFree(agg_d_ver_activations_storage); cudaFree(agg_d_ver_weights_storage); cudaFree(agg_d_ver_gweights_storage); cudaFree(agg_d_ver_deltas_storage);
        throw std::runtime_error("Exception during cu1ParallelBackward1stBlock (V) for column " + std::to_string(layno) + ": " + e.what());
    }

    // --- Cleanup Device Memory and Streams (Success Path) ---
    for (int head_idx = 0; head_idx < num_heads_to_process; ++head_idx) {
        if (streams[head_idx]) {
            CUDA_CHECK(cudaStreamDestroy(streams[head_idx]));
        }
    }
    cudaFree(agg_d_expected_v); cudaFree(agg_d_EV); cudaFree(agg_d_grad_EV_full); cudaFree(agg_d_grad_EV_summed); cudaFree(agg_d_grad_EV_scaled); cudaFree(agg_d_grad_dv);
    cudaFree(agg_d_KdotQ); cudaFree(agg_d_head_storage); cudaFree(agg_d_K); cudaFree(agg_d_Q); cudaFree(agg_d_pre_MV);
    cudaFree(agg_d_MV_a); cudaFree(agg_d_MQ_a); cudaFree(agg_d_MK_a);
    cudaFree(agg_d_grad_MV); cudaFree(agg_d_grad_head_storage); cudaFree(agg_d_lota_deriv); cudaFree(agg_d_grad_KdotQ);
    cudaFree(agg_d_grad_Q_storage); cudaFree(agg_d_grad_MQ); cudaFree(agg_d_grad_MK_correction);
    cudaFree(agg_d_ver_activations_storage); cudaFree(agg_d_ver_weights_storage); cudaFree(agg_d_ver_gweights_storage); cudaFree(agg_d_ver_deltas_storage);
}


/**
 * @brief CUDA backward propagation for a single column in a NON-FIRST block,
 *        driven by vertical error (EV). Processes heads b[row][layno].
 *        IMPLEMENTATION USES ASYNCHRONOUS KERNEL CALLS PER HEAD.
 * @param expectedV Expected vertical embeddings for all heads in this column. Shape: [x][CONTEXT_WIN][EMBEDDING].
 * @param in Embedding dimension.
 * @param layers Number of MLP layers.
 * @param layno The column index within the block (0 to y-1).
 */
void block::cu1ParallelBackward(std::vector<std::vector<std::vector<float>>>& expectedV, int& in, int& layers, int layno)
{
    const int num_heads_to_process = x; // 'x' is the number of rows/heads in this column

    if (layno < 0 || layno >= y) {
        throw std::out_of_range("cu1ParallelBackward(V): Column index 'layno' (" + std::to_string(layno) + ") is out of range [0, " + std::to_string(y - 1) + "].");
    }
    if (expectedV.size() != static_cast<size_t>(x)) {
        throw std::runtime_error("ExpectedV size mismatch (number of rows) in cu1ParallelBackward(V). Expected " + std::to_string(x) + ", got " + std::to_string(expectedV.size()));
    }
    if (!expectedV.empty() && (expectedV[0].size() != CONTEXT_WIN || (!expectedV[0].empty() && expectedV[0][0].size() != EMBEDDING))) {
        throw std::runtime_error("ExpectedV dimensions mismatch (context/embedding) in cu1ParallelBackward(V).");
    }
    if (EMBEDDING != in) {
        throw std::runtime_error("Embedding dimension mismatch: EMBEDDING vs in");
    }

    // Constants
    const int embedding_dim = EMBEDDING;
    const int mat_heights = MATHEIGHTS;
    const int context_win = CONTEXT_WIN;
    const float learning_rate = LEARNING;
    const float scaling_factor = SCALING;

    // MLP structure parameters
    const int num_total_layers_mlp = layers;
    const int num_neuron_layers_mlp = num_total_layers_mlp;
    const int num_weight_matrices_mlp = num_total_layers_mlp - 1;

    const size_t embed_bytes = embedding_dim * sizeof(float);
    const size_t mlp_weights_elements = static_cast<size_t>(embedding_dim) * embedding_dim; // Assuming square matrices
    const size_t mlp_weights_bytes = mlp_weights_elements * sizeof(float);
    const size_t proj_mat_elements = static_cast<size_t>(mat_heights) * embedding_dim; // For MV, MQ, MK (assuming mat_heights x embedding_dim)
    const size_t proj_mat_bytes = proj_mat_elements * sizeof(float);
    const size_t ev_total_elements = static_cast<size_t>(context_win) * embedding_dim; // For EV
    const size_t ev_total_bytes = ev_total_elements * sizeof(float);
    const size_t max_head_elements_per_head = static_cast<size_t>(context_win) * context_win;
    const size_t max_head_bytes_per_head = max_head_elements_per_head * sizeof(float);
    const size_t max_k_q_elements_per_head = static_cast<size_t>(context_win) * mat_heights;
    const size_t max_k_q_bytes_per_head = max_k_q_elements_per_head * sizeof(float);
    const size_t pre_mv_elements_per_head = mat_heights;
    const size_t pre_mv_bytes_per_head = pre_mv_elements_per_head * sizeof(float);

    // Kernel Launch Config
    int threadsPerBlock1D = 256;
    dim3 blockDim1D(threadsPerBlock1D);
    int blocksPerGridEmbed = (embedding_dim + threadsPerBlock1D - 1) / threadsPerBlock1D;
    dim3 gridDimEmbed(blocksPerGridEmbed);
    int blocksPerGridMatHeights = (mat_heights + threadsPerBlock1D - 1) / threadsPerBlock1D;
    dim3 gridDimMatHeights(blocksPerGridMatHeights);
    int blocksPerGridProjMat = (static_cast<int>(proj_mat_elements) + threadsPerBlock1D - 1) / threadsPerBlock1D;
    dim3 gridDimProjMat(blocksPerGridProjMat);
    int blocksPerGridEV = (static_cast<int>(ev_total_elements) + threadsPerBlock1D - 1) / threadsPerBlock1D;
    dim3 gridDimEV(blocksPerGridEV); // For full EV updates

    dim3 blockDim2D(16, 16);
    dim3 gridDimEmbed2D((embedding_dim + blockDim2D.x - 1) / blockDim2D.x, (embedding_dim + blockDim2D.y - 1) / blockDim2D.y);
    dim3 gridDimMatrix2D((embedding_dim + blockDim2D.x - 1) / blockDim2D.x, (mat_heights + blockDim2D.y - 1) / blockDim2D.y);

    // --- Aggregate Device Memory Pointers ---
    float *agg_d_expected_v = nullptr, *agg_d_EV = nullptr;
    float *agg_d_grad_EV_full = nullptr, *agg_d_grad_EV_summed = nullptr, *agg_d_grad_EV_scaled = nullptr;
    float *agg_d_grad_dv = nullptr;
    float *agg_d_KdotQ = nullptr, *agg_d_head_storage = nullptr;
    float *agg_d_K = nullptr, *agg_d_Q = nullptr;
    float *agg_d_pre_MV = nullptr;
    float *agg_d_MV_a = nullptr, *agg_d_MQ_a = nullptr, *agg_d_MK_a = nullptr;
    float *agg_d_grad_MV = nullptr;
    float *agg_d_grad_head_storage = nullptr;
    float *agg_d_lota_deriv = nullptr;
    float *agg_d_grad_KdotQ = nullptr;
    float *agg_d_grad_Q_storage = nullptr;
    float *agg_d_grad_MQ = nullptr, *agg_d_grad_MK_correction = nullptr;
    // MLP Aggregate Storage (ver only)
    float *agg_d_ver_activations_storage = nullptr;
    float *agg_d_ver_weights_storage = nullptr;
    float *agg_d_ver_gweights_storage = nullptr;
    float *agg_d_ver_deltas_storage = nullptr;

    std::vector<cudaStream_t> streams(num_heads_to_process, nullptr);
    std::vector<HeadDevicePointersV> head_gpu_data(num_heads_to_process);

    try {
        // --- Allocate Aggregate Memory ---
        CUDA_CHECK(cudaMalloc(&agg_d_expected_v, num_heads_to_process * ev_total_bytes));
        CUDA_CHECK(cudaMalloc(&agg_d_EV, num_heads_to_process * ev_total_bytes));
        CUDA_CHECK(cudaMalloc(&agg_d_grad_EV_full, num_heads_to_process * ev_total_bytes));
        CUDA_CHECK(cudaMalloc(&agg_d_grad_EV_summed, num_heads_to_process * embed_bytes));
        CUDA_CHECK(cudaMalloc(&agg_d_grad_EV_scaled, num_heads_to_process * embed_bytes));
        CUDA_CHECK(cudaMalloc(&agg_d_grad_dv, num_heads_to_process * embed_bytes));
        CUDA_CHECK(cudaMalloc(&agg_d_KdotQ, num_heads_to_process * max_head_bytes_per_head));
        CUDA_CHECK(cudaMalloc(&agg_d_head_storage, num_heads_to_process * max_head_bytes_per_head));
        CUDA_CHECK(cudaMalloc(&agg_d_K, num_heads_to_process * max_k_q_bytes_per_head));
        CUDA_CHECK(cudaMalloc(&agg_d_Q, num_heads_to_process * max_k_q_bytes_per_head));
        CUDA_CHECK(cudaMalloc(&agg_d_pre_MV, num_heads_to_process * pre_mv_bytes_per_head));
        CUDA_CHECK(cudaMalloc(&agg_d_MV_a, num_heads_to_process * proj_mat_bytes));
        CUDA_CHECK(cudaMalloc(&agg_d_MQ_a, num_heads_to_process * proj_mat_bytes));
        CUDA_CHECK(cudaMalloc(&agg_d_MK_a, num_heads_to_process * proj_mat_bytes));
        CUDA_CHECK(cudaMalloc(&agg_d_grad_MV, num_heads_to_process * proj_mat_bytes));
        CUDA_CHECK(cudaMalloc(&agg_d_grad_head_storage, num_heads_to_process * max_head_bytes_per_head));
        CUDA_CHECK(cudaMalloc(&agg_d_lota_deriv, num_heads_to_process * max_head_bytes_per_head));
        CUDA_CHECK(cudaMalloc(&agg_d_grad_KdotQ, num_heads_to_process * max_head_bytes_per_head));
        CUDA_CHECK(cudaMalloc(&agg_d_grad_Q_storage, num_heads_to_process * max_k_q_bytes_per_head));
        CUDA_CHECK(cudaMalloc(&agg_d_grad_MQ, num_heads_to_process * proj_mat_bytes));
        CUDA_CHECK(cudaMalloc(&agg_d_grad_MK_correction, num_heads_to_process * proj_mat_bytes));

        CUDA_CHECK(cudaMalloc(&agg_d_ver_activations_storage, num_heads_to_process * num_neuron_layers_mlp * embed_bytes));
        CUDA_CHECK(cudaMalloc(&agg_d_ver_weights_storage, num_heads_to_process * num_weight_matrices_mlp * mlp_weights_bytes));
        CUDA_CHECK(cudaMalloc(&agg_d_ver_gweights_storage, num_heads_to_process * num_weight_matrices_mlp * mlp_weights_bytes));
        CUDA_CHECK(cudaMalloc(&agg_d_ver_deltas_storage, num_heads_to_process * num_weight_matrices_mlp * embed_bytes));

        // --- Create Streams and Setup Per-Head Pointers ---
        for (int head_idx = 0; head_idx < num_heads_to_process; ++head_idx) {
            CUDA_CHECK(cudaStreamCreateWithFlags(&streams[head_idx], cudaStreamNonBlocking));
            head_gpu_data[head_idx].d_ver_activations.resize(num_neuron_layers_mlp);
            head_gpu_data[head_idx].d_ver_weights.resize(num_weight_matrices_mlp);
            head_gpu_data[head_idx].d_ver_gweights.resize(num_weight_matrices_mlp);
            head_gpu_data[head_idx].d_ver_deltas.resize(num_weight_matrices_mlp);

            HeadDevicePointersV& current_ptrs = head_gpu_data[head_idx];
            current_ptrs.d_expected_v = agg_d_expected_v + head_idx * ev_total_elements;
            current_ptrs.d_EV = agg_d_EV + head_idx * ev_total_elements;
            current_ptrs.d_grad_EV_full = agg_d_grad_EV_full + head_idx * ev_total_elements;
            current_ptrs.d_grad_EV_summed = agg_d_grad_EV_summed + head_idx * embedding_dim;
            current_ptrs.d_grad_EV_scaled = agg_d_grad_EV_scaled + head_idx * embedding_dim;
            current_ptrs.d_grad_dv = agg_d_grad_dv + head_idx * embedding_dim;
            current_ptrs.d_KdotQ = agg_d_KdotQ + head_idx * max_head_elements_per_head;
            current_ptrs.d_head = agg_d_head_storage + head_idx * max_head_elements_per_head;
            current_ptrs.d_K = agg_d_K + head_idx * max_k_q_elements_per_head;
            current_ptrs.d_Q = agg_d_Q + head_idx * max_k_q_elements_per_head;
            current_ptrs.d_pre_MV = agg_d_pre_MV + head_idx * pre_mv_elements_per_head;
            current_ptrs.d_MV_a = agg_d_MV_a + head_idx * proj_mat_elements;
            current_ptrs.d_MQ_a = agg_d_MQ_a + head_idx * proj_mat_elements;
            current_ptrs.d_MK_a = agg_d_MK_a + head_idx * proj_mat_elements;
            current_ptrs.d_grad_MV = agg_d_grad_MV + head_idx * proj_mat_elements;
            current_ptrs.d_grad_head = agg_d_grad_head_storage + head_idx * max_head_elements_per_head;
            current_ptrs.d_lota_deriv = agg_d_lota_deriv + head_idx * max_head_elements_per_head;
            current_ptrs.d_grad_KdotQ = agg_d_grad_KdotQ + head_idx * max_head_elements_per_head;
            current_ptrs.d_grad_Q = agg_d_grad_Q_storage + head_idx * max_k_q_elements_per_head;
            current_ptrs.d_grad_MQ = agg_d_grad_MQ + head_idx * proj_mat_elements;
            current_ptrs.d_grad_MK_correction = agg_d_grad_MK_correction + head_idx * proj_mat_elements;

            for (int l = 0; l < num_neuron_layers_mlp; ++l) {
                current_ptrs.d_ver_activations[l] = agg_d_ver_activations_storage + (head_idx * num_neuron_layers_mlp + l) * embedding_dim;
            }
            for (int l = 0; l < num_weight_matrices_mlp; ++l) {
                current_ptrs.d_ver_weights[l] = agg_d_ver_weights_storage + (head_idx * num_weight_matrices_mlp + l) * mlp_weights_elements;
                current_ptrs.d_ver_gweights[l] = agg_d_ver_gweights_storage + (head_idx * num_weight_matrices_mlp + l) * mlp_weights_elements;
                current_ptrs.d_ver_deltas[l] = agg_d_ver_deltas_storage + (head_idx * num_weight_matrices_mlp + l) * embedding_dim;
            }
        }

        // Iterate through the rows
        for (int head_idx = 0; head_idx < num_heads_to_process; ++head_idx) {
            attention& head_obj = b[head_idx][layno];
            std::vector<std::vector<float>>& expectedV_head = expectedV[head_idx];
            HeadDevicePointersV& device_ptrs = head_gpu_data[head_idx];
            cudaStream_t current_stream = streams[head_idx];

            const int token_count = head_obj.tokenCount;
            const size_t active_head_elements = static_cast<size_t>(token_count) * token_count;
            const size_t active_head_bytes = active_head_elements * sizeof(float);
            const size_t active_k_q_elements = static_cast<size_t>(token_count) * mat_heights;
            const size_t active_k_q_bytes = active_k_q_elements * sizeof(float);
            dim3 gridDimHead2D((token_count + blockDim2D.x - 1) / blockDim2D.x, (token_count + blockDim2D.y - 1) / blockDim2D.y);
            dim3 gridDimKQGrad2D((mat_heights + blockDim2D.x - 1) / blockDim2D.x, (token_count + blockDim2D.y - 1) / blockDim2D.y);
            int blocksPerGridHead = (static_cast<int>(active_head_elements) + threadsPerBlock1D - 1) / threadsPerBlock1D;
            dim3 gridDimHead(blocksPerGridHead);

            // --- Data Transfer H->D (Async) ---
            std::vector<float> flat_expectedV_head = flatten(expectedV_head);
            if (!head_obj.EV.mapped_data || (token_count > 0 && (!head_obj.K.mapped_data || !head_obj.Q.mapped_data || !head_obj.KdotQ.mapped_data)) ||
                !head_obj.MV.mapped_data || !head_obj.MQ.mapped_data || !head_obj.MK.mapped_data) {
                throw std::runtime_error("One or more attention mat objects have null mapped_data for head [" + std::to_string(head_idx) + "][" + std::to_string(layno) + "]");
            }
            CUDA_CHECK(cudaMemcpyAsync(device_ptrs.d_expected_v, flat_expectedV_head.data(), ev_total_bytes, cudaMemcpyHostToDevice, current_stream));
            CUDA_CHECK(cudaMemcpyAsync(device_ptrs.d_EV, head_obj.EV.mapped_data, ev_total_bytes, cudaMemcpyHostToDevice, current_stream));
            if (token_count > 0) {
                CUDA_CHECK(cudaMemcpyAsync(device_ptrs.d_KdotQ, head_obj.KdotQ.mapped_data, active_head_bytes, cudaMemcpyHostToDevice, current_stream));
                CUDA_CHECK(cudaMemcpyAsync(device_ptrs.d_K, head_obj.K.mapped_data, active_k_q_bytes, cudaMemcpyHostToDevice, current_stream));
                CUDA_CHECK(cudaMemcpyAsync(device_ptrs.d_Q, head_obj.Q.mapped_data, active_k_q_bytes, cudaMemcpyHostToDevice, current_stream));
            }
            CUDA_CHECK(cudaMemcpyAsync(device_ptrs.d_MV_a, head_obj.MV.mapped_data, proj_mat_bytes, cudaMemcpyHostToDevice, current_stream));
            CUDA_CHECK(cudaMemcpyAsync(device_ptrs.d_MQ_a, head_obj.MQ.mapped_data, proj_mat_bytes, cudaMemcpyHostToDevice, current_stream));
            CUDA_CHECK(cudaMemcpyAsync(device_ptrs.d_MK_a, head_obj.MK.mapped_data, proj_mat_bytes, cudaMemcpyHostToDevice, current_stream));
            
            // MLP Internals Data Transfer H->D
            if (head_obj.ver.activations.size() != static_cast<size_t>(num_neuron_layers_mlp) ||
                head_obj.ver.weights.size() != static_cast<size_t>(num_weight_matrices_mlp) ||
                head_obj.ver.gweights.size() != static_cast<size_t>(num_weight_matrices_mlp)) {
                 throw std::runtime_error("MLP host ver vector size mismatch for head [" + std::to_string(head_idx) + "][" + std::to_string(layno) + "]");
            }
            for (int l = 0; l < num_neuron_layers_mlp; ++l) {
                if (head_obj.ver.activations[l].empty()) {
                    throw std::runtime_error("MLP ver.activations vector is empty for head [" + std::to_string(head_idx) + "][" + std::to_string(layno) + "], layer " + std::to_string(l));
                }
                CUDA_CHECK(cudaMemcpyAsync(device_ptrs.d_ver_activations[l], head_obj.ver.activations[l].data(), embed_bytes, cudaMemcpyHostToDevice, current_stream));
            }
            for (int l = 0; l < num_weight_matrices_mlp; ++l) {
                if (!head_obj.ver.weights[l].mapped_data || head_obj.ver.weights[l].row * head_obj.ver.weights[l].col != mlp_weights_elements ||
                    !head_obj.ver.gweights[l].mapped_data || head_obj.ver.gweights[l].row * head_obj.ver.gweights[l].col != mlp_weights_elements) {
                    throw std::runtime_error("Invalid ver.weights/gweights mat for head [" + std::to_string(head_idx) + "][" + std::to_string(layno) + "], layer " + std::to_string(l));
                }
                CUDA_CHECK(cudaMemcpyAsync(device_ptrs.d_ver_weights[l], head_obj.ver.weights[l].mapped_data, mlp_weights_bytes, cudaMemcpyHostToDevice, current_stream));
                CUDA_CHECK(cudaMemcpyAsync(device_ptrs.d_ver_gweights[l], head_obj.ver.gweights[l].mapped_data, mlp_weights_bytes, cudaMemcpyHostToDevice, current_stream));
            }

            // --- Backpropagation Steps (Async Kernels) ---
            kernelComputeGradientsEV_V<<<gridDimEV, blockDim1D, 0, current_stream>>>(device_ptrs.d_EV, device_ptrs.d_expected_v, device_ptrs.d_grad_EV_full, device_ptrs.d_grad_EV_summed, device_ptrs.d_grad_EV_scaled, learning_rate, context_win, embedding_dim); CUDA_CHECK(cudaGetLastError());
            kernelLastLayerDelta<<<gridDimEmbed, blockDim1D, 0, current_stream>>>(device_ptrs.d_grad_EV_scaled, device_ptrs.d_ver_activations[num_neuron_layers_mlp - 1], device_ptrs.d_ver_deltas[num_weight_matrices_mlp - 1], embedding_dim); CUDA_CHECK(cudaGetLastError());
            for (int k = num_weight_matrices_mlp - 1; k >= 1; --k) {
                hiddenDeltaKernel<<<gridDimEmbed, blockDim1D, 0, current_stream>>>(device_ptrs.d_ver_deltas[k], device_ptrs.d_ver_weights[k], device_ptrs.d_ver_activations[k], device_ptrs.d_ver_deltas[k-1], embedding_dim, embedding_dim); CUDA_CHECK(cudaGetLastError());
            }
            // ver mlp
            for (int l = 0; l < num_weight_matrices_mlp; ++l) {
                updateWeightsKernel<<<gridDimEmbed2D, blockDim2D, 0, current_stream>>>(device_ptrs.d_ver_deltas[l], device_ptrs.d_ver_activations[l], device_ptrs.d_ver_weights[l], device_ptrs.d_ver_gweights[l], learning_rate, embedding_dim, embedding_dim); CUDA_CHECK(cudaGetLastError());
            }
            kernelComputeGradMLPInput<<<gridDimEmbed, blockDim1D, 0, current_stream>>>(device_ptrs.d_ver_deltas[0], device_ptrs.d_ver_weights[0], device_ptrs.d_grad_dv, embedding_dim, embedding_dim); CUDA_CHECK(cudaGetLastError());

            if (token_count > 0) {
                cuLOTA<<<gridDimHead2D, blockDim2D, 0, current_stream>>>(device_ptrs.d_KdotQ, device_ptrs.d_head, context_win, context_win, token_count, head_obj.isSelfAttention); CUDA_CHECK(cudaGetLastError());
                kernelComputePreMV_V<<<gridDimMatHeights, blockDim1D, 0, current_stream>>>(device_ptrs.d_head, device_ptrs.d_Q, device_ptrs.d_pre_MV, token_count, mat_heights); CUDA_CHECK(cudaGetLastError());
                kernelComputeGradMV_V<<<gridDimMatrix2D, blockDim2D, 0, current_stream>>>(device_ptrs.d_pre_MV, device_ptrs.d_grad_dv, device_ptrs.d_grad_MV, mat_heights, embedding_dim); CUDA_CHECK(cudaGetLastError());
                kernelComputeGradHead_V<<<gridDimHead2D, blockDim2D, 0, current_stream>>>(device_ptrs.d_Q, device_ptrs.d_MV_a, device_ptrs.d_grad_dv, device_ptrs.d_grad_head, token_count, mat_heights, embedding_dim); CUDA_CHECK(cudaGetLastError());
                cuLOTAder<<<gridDimHead2D, blockDim2D, 0, current_stream>>>(device_ptrs.d_KdotQ, device_ptrs.d_lota_deriv, context_win, context_win, token_count, head_obj.isSelfAttention); CUDA_CHECK(cudaGetLastError());
                kernelComputeGradKdotQ_LOTA<<<gridDimHead, blockDim1D, 0, current_stream>>>(device_ptrs.d_grad_head, device_ptrs.d_lota_deriv, device_ptrs.d_grad_KdotQ, scaling_factor, static_cast<int>(active_head_elements)); CUDA_CHECK(cudaGetLastError());
                kernelComputeGradQ_V<<<gridDimKQGrad2D, blockDim2D, 0, current_stream>>>(device_ptrs.d_grad_KdotQ, device_ptrs.d_K, device_ptrs.d_grad_Q, token_count, mat_heights); CUDA_CHECK(cudaGetLastError());
                kernelComputeGradMQ_V<<<gridDimMatrix2D, blockDim2D, 0, current_stream>>>(device_ptrs.d_grad_Q, nullptr, device_ptrs.d_grad_MQ, token_count, mat_heights, embedding_dim); CUDA_CHECK(cudaGetLastError());
                kernelComputeGradMKCorrection<<<gridDimMatrix2D, blockDim2D, 0, current_stream>>>(device_ptrs.d_grad_MQ, device_ptrs.d_Q, device_ptrs.d_K, device_ptrs.d_grad_MK_correction, token_count, mat_heights, embedding_dim); CUDA_CHECK(cudaGetLastError());
            } else {
                CUDA_CHECK(cudaMemsetAsync(device_ptrs.d_grad_MV, 0, proj_mat_bytes, current_stream));
                CUDA_CHECK(cudaMemsetAsync(device_ptrs.d_grad_MQ, 0, proj_mat_bytes, current_stream));
                CUDA_CHECK(cudaMemsetAsync(device_ptrs.d_grad_MK_correction, 0, proj_mat_bytes, current_stream));
            }

            kernelUpdateSimple<<<gridDimProjMat, blockDim1D, 0, current_stream>>>(device_ptrs.d_MV_a, device_ptrs.d_grad_MV, learning_rate, proj_mat_elements);
            kernelUpdateSimple<<<gridDimProjMat, blockDim1D, 0, current_stream>>>(device_ptrs.d_MQ_a, device_ptrs.d_grad_MQ, learning_rate, proj_mat_elements);
            kernelUpdateSimple<<<gridDimProjMat, blockDim1D, 0, current_stream>>>(device_ptrs.d_MK_a, device_ptrs.d_grad_MK_correction, learning_rate, proj_mat_elements);
            CUDA_CHECK(cudaGetLastError());
            kernelUpdateSimple<<<gridDimEV, blockDim1D, 0, current_stream>>>(device_ptrs.d_EV, device_ptrs.d_grad_EV_full, learning_rate, ev_total_elements);
            CUDA_CHECK(cudaGetLastError());

            // --- Data Transfer D->H (Async) ---
            for (int l = 0; l < num_weight_matrices_mlp; ++l) {
                CUDA_CHECK(cudaMemcpyAsync(head_obj.ver.weights[l].mapped_data, device_ptrs.d_ver_weights[l], mlp_weights_bytes, cudaMemcpyDeviceToHost, current_stream));
                CUDA_CHECK(cudaMemcpyAsync(head_obj.ver.gweights[l].mapped_data, device_ptrs.d_ver_gweights[l], mlp_weights_bytes, cudaMemcpyDeviceToHost, current_stream));
            }
            CUDA_CHECK(cudaMemcpyAsync(head_obj.EV.mapped_data, device_ptrs.d_EV, ev_total_bytes, cudaMemcpyDeviceToHost, current_stream));
            CUDA_CHECK(cudaMemcpyAsync(head_obj.MV.mapped_data, device_ptrs.d_MV_a, proj_mat_bytes, cudaMemcpyDeviceToHost, current_stream));
            CUDA_CHECK(cudaMemcpyAsync(head_obj.MQ.mapped_data, device_ptrs.d_MQ_a, proj_mat_bytes, cudaMemcpyDeviceToHost, current_stream));
            CUDA_CHECK(cudaMemcpyAsync(head_obj.MK.mapped_data, device_ptrs.d_MK_a, proj_mat_bytes, cudaMemcpyDeviceToHost, current_stream));
        } // End loop over heads

        // --- Synchronize all streams ---
        for (int head_idx = 0; head_idx < num_heads_to_process; ++head_idx) {
            if (streams[head_idx]) {
                CUDA_CHECK(cudaStreamSynchronize(streams[head_idx]));
            }
        }
    } // End try block
    catch (const std::exception& e) {
        std::cerr << "Error during cu1ParallelBackward (V) for column " << layno << ": " << e.what() << std::endl;
        // Cleanup aggregate memory and streams on error
        for (int head_idx = 0; head_idx < num_heads_to_process; ++head_idx) { if (streams[head_idx]) { cudaStreamDestroy(streams[head_idx]); streams[head_idx] = nullptr; } }
        cudaFree(agg_d_expected_v); cudaFree(agg_d_EV); cudaFree(agg_d_grad_EV_full); cudaFree(agg_d_grad_EV_summed); cudaFree(agg_d_grad_EV_scaled); cudaFree(agg_d_grad_dv);
        cudaFree(agg_d_KdotQ); cudaFree(agg_d_head_storage); cudaFree(agg_d_K); cudaFree(agg_d_Q); cudaFree(agg_d_pre_MV);
        cudaFree(agg_d_MV_a); cudaFree(agg_d_MQ_a); cudaFree(agg_d_MK_a);
        cudaFree(agg_d_grad_MV); cudaFree(agg_d_grad_head_storage); cudaFree(agg_d_lota_deriv); cudaFree(agg_d_grad_KdotQ);
        cudaFree(agg_d_grad_Q_storage); cudaFree(agg_d_grad_MQ); cudaFree(agg_d_grad_MK_correction);
        cudaFree(agg_d_ver_activations_storage); cudaFree(agg_d_ver_weights_storage); cudaFree(agg_d_ver_gweights_storage); cudaFree(agg_d_ver_deltas_storage);
        throw std::runtime_error("Exception during cu1ParallelBackward (V) for column " + std::to_string(layno) + ": " + e.what());
    }

    // --- Cleanup Device Memory and Streams (Success Path) ---
    for (int head_idx = 0; head_idx < num_heads_to_process; ++head_idx) {
        if (streams[head_idx]) {
            CUDA_CHECK(cudaStreamDestroy(streams[head_idx]));
        }
    }
    cudaFree(agg_d_expected_v); cudaFree(agg_d_EV); cudaFree(agg_d_grad_EV_full); cudaFree(agg_d_grad_EV_summed); cudaFree(agg_d_grad_EV_scaled); cudaFree(agg_d_grad_dv);
    cudaFree(agg_d_KdotQ); cudaFree(agg_d_head_storage); cudaFree(agg_d_K); cudaFree(agg_d_Q); cudaFree(agg_d_pre_MV);
    cudaFree(agg_d_MV_a); cudaFree(agg_d_MQ_a); cudaFree(agg_d_MK_a);
    cudaFree(agg_d_grad_MV); cudaFree(agg_d_grad_head_storage); cudaFree(agg_d_lota_deriv); cudaFree(agg_d_grad_KdotQ);
    cudaFree(agg_d_grad_Q_storage); cudaFree(agg_d_grad_MQ); cudaFree(agg_d_grad_MK_correction);
    cudaFree(agg_d_ver_activations_storage); cudaFree(agg_d_ver_weights_storage); cudaFree(agg_d_ver_gweights_storage); cudaFree(agg_d_ver_deltas_storage);
}
