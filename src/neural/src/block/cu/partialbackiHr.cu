#ifdef USE_CUDA
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

// Helper struct to manage device pointers for one head's worth of data
struct HeadDevicePointers {
    // Attention related
    float *d_expected_h = nullptr, *d_EH = nullptr, *d_EV = nullptr;
    float *d_grad_EH = nullptr, *d_grad_EV_scaled = nullptr;
    float *d_grad_dh = nullptr, *d_grad_dv = nullptr;
    float *d_KdotQ = nullptr, *d_head = nullptr;
    float *d_K = nullptr, *d_Q = nullptr;
    float *d_pre_MH = nullptr, *d_pre_MV = nullptr;
    float *d_MH_a = nullptr, *d_MV_a = nullptr, *d_MQ_a = nullptr, *d_MK_a = nullptr;
    float *d_grad_MH = nullptr, *d_grad_MV = nullptr;
    float *d_grad_head = nullptr;
    float *d_lota_deriv = nullptr;
    float *d_grad_KdotQ = nullptr;
    float *d_grad_K = nullptr, *d_grad_Q = nullptr;
    float *d_grad_MQ = nullptr, *d_grad_MK = nullptr;

    // MLP Internals
    std::vector<float*> d_hor_activations;
    std::vector<float*> d_hor_weights;
    std::vector<float*> d_hor_gweights;
    std::vector<float*> d_hor_deltas;
    std::vector<float*> d_ver_activations;
    std::vector<float*> d_ver_weights;
    std::vector<float*> d_ver_gweights;
    std::vector<float*> d_ver_deltas;

    HeadDevicePointers() = default; // Default constructor for vector initialization
};

/**
 * @brief CUDA backward propagation for a single column in the FIRST block,
 *        driven by horizontal error (EH). Processes heads b[row][layno].
 *        IMPLEMENTATION USES DIRECT KERNEL CALLS PER HEAD (INEFFICIENT).
 * @param expectedH Expected horizontal embedding for the output of this column's heads.
 * @param in Embedding dimension.
 * @param layers Number of MLP layers.
 * @param layno The column index within the block (0 to y-1).
 */
void block::curpartialbackward1stBlock(std::vector<std::vector<float>>& expectedH, int& in, int& layers, int& layno, float& learning, float& lambda_l1, float& lambda_l2)
{
    const int num_heads_to_process = x; // 'x' is the number of rows/heads in this column

    // Validate column number
    if (layno < 0 || layno >= y) {
        throw std::out_of_range("cupartialbackward1stBlock(H2d): Column index 'layno' (" + std::to_string(layno) + 
            ") is out of range [0, " + std::to_string(y - 1) + "].");
    }
    if (expectedH.size() != static_cast<size_t>(num_heads_to_process)) {
        throw std::runtime_error("ExpectedH outer vector size mismatch in cupartialbackward1stBlock(std::vector<std::vector<float>>). Expected " +
            std::to_string(num_heads_to_process) + ", got " + std::to_string(expectedH.size()));
    }
    for(int i = 0; i < num_heads_to_process; ++i) {
        if(expectedH[i].size() != EMBEDDING) {
            throw std::runtime_error("cupartialbackward1stBlock(H): ExpectedH inner vector size mismatch: Size is " + std::to_string(expectedH[i].size()) + 
                ", expected " + std::to_string(EMBEDDING) + ".");
        }
    }
    if (EMBEDDING != in) {
        throw std::runtime_error("Embedding dimension mismatch: EMBEDDING vs in");
    }

    // Constants used within the loop
    const int embedding_dim = EMBEDDING;
    const int mat_heights = CONTEXT_WIN;
    const int context_win = CONTEXT_WIN;
    const float learning_rate = learning;
    const float scaling_factor = SCALING;

    // MLP structure parameters based on 'layers' (L = number of hidden layers)
    const int num_total_layers_mlp = layers; // 'layers' parameter represents the total number of layers (Input + Hidden + Output)
    const int num_neuron_layers_mlp = num_total_layers_mlp; // Number of neuron layers is the total number of layers
    const int num_weight_matrices_mlp = num_total_layers_mlp - 1; // Number of weight matrices is total layers - 1

    const size_t embed_bytes = embedding_dim * sizeof(float);
    const size_t mlp_weights_elements = static_cast<size_t>(embedding_dim) * embedding_dim;
    const size_t mlp_weights_bytes = mlp_weights_elements * sizeof(float);
    const size_t proj_mat_elements = static_cast<size_t>(mat_heights) * embedding_dim; // For MH, MV, MQ, MK
    const size_t proj_mat_bytes = proj_mat_elements * sizeof(float);
    const size_t ev_single_head_elements = static_cast<size_t>(context_win) * embedding_dim; // For one head's EV
    const size_t ev_single_head_bytes = ev_single_head_elements * sizeof(float);
    // Max sizes for token-dependent buffers per head
    const size_t max_head_elements_per_head = static_cast<size_t>(context_win) * context_win;
    const size_t max_head_bytes_per_head = max_head_elements_per_head * sizeof(float);
    const size_t max_k_q_elements_per_head = static_cast<size_t>(context_win) * mat_heights;
    const size_t max_k_q_bytes_per_head = max_k_q_elements_per_head * sizeof(float);
    const size_t pre_mh_mv_elements_per_head = mat_heights; // d_pre_MH, d_pre_MV
    const size_t pre_mh_mv_bytes_per_head = pre_mh_mv_elements_per_head * sizeof(float);
    // Kernel Launch Config (Define once)
    int threadsPerBlock1D = 256;
    dim3 blockDim1D(threadsPerBlock1D);
    int blocksPerGridEmbed = (embedding_dim + threadsPerBlock1D - 1) / threadsPerBlock1D;
    dim3 gridDimEmbed(blocksPerGridEmbed);
    int blocksPerGridMatHeights = (mat_heights + threadsPerBlock1D - 1) / threadsPerBlock1D;
    dim3 gridDimMatHeights(blocksPerGridMatHeights);
    int blocksPerGridProjMat = (static_cast<int>(proj_mat_elements) + threadsPerBlock1D - 1) / threadsPerBlock1D;
    dim3 gridDimProjMat(blocksPerGridProjMat);

    dim3 blockDim2D(16, 16);
    dim3 gridDimEmbed2D((embedding_dim + blockDim2D.x - 1) / blockDim2D.x, (embedding_dim + blockDim2D.y - 1) / blockDim2D.y);
    dim3 gridDimMatrix2D((embedding_dim + blockDim2D.x - 1) / blockDim2D.x, (mat_heights + blockDim2D.y - 1) / blockDim2D.y);

    // --- Aggregate Device Memory Pointers ---
    float *agg_d_expected_h = nullptr, *agg_d_EH = nullptr, *agg_d_EV = nullptr;
    float *agg_d_grad_EH = nullptr, *agg_d_grad_EV_scaled = nullptr;
    float *agg_d_grad_dh = nullptr, *agg_d_grad_dv = nullptr;
    float *agg_d_KdotQ = nullptr, *agg_d_head = nullptr;
    float *agg_d_K = nullptr, *agg_d_Q = nullptr;
    float *agg_d_pre_MH = nullptr, *agg_d_pre_MV = nullptr;
    float *agg_d_MH_a = nullptr, *agg_d_MV_a = nullptr, *agg_d_MQ_a = nullptr, *agg_d_MK_a = nullptr;
    float *agg_d_grad_MH = nullptr, *agg_d_grad_MV = nullptr;
    float *agg_d_grad_head_storage = nullptr; // Renamed to avoid conflict
    float *agg_d_lota_deriv = nullptr;
    float *agg_d_grad_KdotQ = nullptr;
    float *agg_d_grad_K = nullptr, *agg_d_grad_Q = nullptr;
    float *agg_d_grad_MQ = nullptr, *agg_d_grad_MK = nullptr;
    // MLP Aggregate Storage
    float *agg_d_hor_activations_storage = nullptr, *agg_d_ver_activations_storage = nullptr;
    float *agg_d_hor_weights_storage = nullptr, *agg_d_ver_weights_storage = nullptr;
    float *agg_d_hor_gweights_storage = nullptr, *agg_d_ver_gweights_storage = nullptr;
    float *agg_d_hor_deltas_storage = nullptr, *agg_d_ver_deltas_storage = nullptr; 

    std::vector<cudaStream_t> streams(num_heads_to_process, nullptr);
    std::vector<HeadDevicePointers> head_gpu_data(num_heads_to_process);

    try {
        // --- Allocate Aggregate Memory ---
        CUDA_CHECK(cudaMalloc(&agg_d_expected_h, num_heads_to_process * embed_bytes)); 
        CUDA_CHECK(cudaMalloc(&agg_d_EH, num_heads_to_process * embed_bytes)); 
        CUDA_CHECK(cudaMalloc(&agg_d_EV, num_heads_to_process * ev_single_head_bytes)); 
        CUDA_CHECK(cudaMalloc(&agg_d_grad_EH, num_heads_to_process * embed_bytes));
        CUDA_CHECK(cudaMalloc(&agg_d_grad_EV_scaled, num_heads_to_process * embed_bytes));
        CUDA_CHECK(cudaMalloc(&agg_d_grad_dh, num_heads_to_process * embed_bytes));
        CUDA_CHECK(cudaMalloc(&agg_d_grad_dv, num_heads_to_process * embed_bytes));
        CUDA_CHECK(cudaMalloc(&agg_d_KdotQ, num_heads_to_process * max_head_bytes_per_head));
        CUDA_CHECK(cudaMalloc(&agg_d_head, num_heads_to_process * max_head_bytes_per_head));
        CUDA_CHECK(cudaMalloc(&agg_d_K, num_heads_to_process * max_k_q_bytes_per_head));
        CUDA_CHECK(cudaMalloc(&agg_d_Q, num_heads_to_process * max_k_q_bytes_per_head));
        CUDA_CHECK(cudaMalloc(&agg_d_pre_MH, num_heads_to_process * pre_mh_mv_bytes_per_head));
        CUDA_CHECK(cudaMalloc(&agg_d_pre_MV, num_heads_to_process * pre_mh_mv_bytes_per_head));
        CUDA_CHECK(cudaMalloc(&agg_d_MH_a, num_heads_to_process * proj_mat_bytes));
        CUDA_CHECK(cudaMalloc(&agg_d_MV_a, num_heads_to_process * proj_mat_bytes));
        CUDA_CHECK(cudaMalloc(&agg_d_MQ_a, num_heads_to_process * proj_mat_bytes));
        CUDA_CHECK(cudaMalloc(&agg_d_MK_a, num_heads_to_process * proj_mat_bytes));
        CUDA_CHECK(cudaMalloc(&agg_d_grad_MH, num_heads_to_process * proj_mat_bytes));
        CUDA_CHECK(cudaMalloc(&agg_d_grad_MV, num_heads_to_process * proj_mat_bytes));
        CUDA_CHECK(cudaMalloc(&agg_d_grad_head_storage, num_heads_to_process * max_head_bytes_per_head));
        CUDA_CHECK(cudaMalloc(&agg_d_lota_deriv, num_heads_to_process * max_head_bytes_per_head));
        CUDA_CHECK(cudaMalloc(&agg_d_grad_KdotQ, num_heads_to_process * max_head_bytes_per_head));
        CUDA_CHECK(cudaMalloc(&agg_d_grad_K, num_heads_to_process * max_k_q_bytes_per_head));
        CUDA_CHECK(cudaMalloc(&agg_d_grad_Q, num_heads_to_process * max_k_q_bytes_per_head));
        CUDA_CHECK(cudaMalloc(&agg_d_grad_MQ, num_heads_to_process * proj_mat_bytes));
        CUDA_CHECK(cudaMalloc(&agg_d_grad_MK, num_heads_to_process * proj_mat_bytes));

        CUDA_CHECK(cudaMalloc(&agg_d_hor_activations_storage, num_heads_to_process * num_neuron_layers_mlp * embed_bytes)); // num_neuron_layers_mlp activations
        CUDA_CHECK(cudaMalloc(&agg_d_ver_activations_storage, num_heads_to_process * num_neuron_layers_mlp * embed_bytes)); // num_neuron_layers_mlp activations
        CUDA_CHECK(cudaMalloc(&agg_d_hor_weights_storage, num_heads_to_process * num_weight_matrices_mlp * mlp_weights_bytes)); // num_weight_matrices_mlp weights
        CUDA_CHECK(cudaMalloc(&agg_d_ver_weights_storage, num_heads_to_process * num_weight_matrices_mlp * mlp_weights_bytes)); // num_weight_matrices_mlp weights
        CUDA_CHECK(cudaMalloc(&agg_d_hor_gweights_storage, num_heads_to_process * num_weight_matrices_mlp * mlp_weights_bytes)); // num_weight_matrices_mlp gweights
        CUDA_CHECK(cudaMalloc(&agg_d_ver_gweights_storage, num_heads_to_process * num_weight_matrices_mlp * mlp_weights_bytes)); // num_weight_matrices_mlp gweights
        CUDA_CHECK(cudaMalloc(&agg_d_hor_deltas_storage, num_heads_to_process * num_neuron_layers_mlp * embed_bytes)); // num_neuron_layers_mlp deltas
        CUDA_CHECK(cudaMalloc(&agg_d_ver_deltas_storage, num_heads_to_process * num_neuron_layers_mlp * embed_bytes)); // num_neuron_layers_mlp deltas

        // --- Create Streams and Setup Per-Head Pointers ---
        for (int head_idx = 0; head_idx < num_heads_to_process; ++head_idx) {
            CUDA_CHECK(cudaStreamCreateWithFlags(&streams[head_idx], cudaStreamNonBlocking));
            // Initialize HeadDevicePointers with correct MLP layer counts
            head_gpu_data[head_idx] = HeadDevicePointers(); // Use default constructor
            head_gpu_data[head_idx].d_hor_activations.resize(num_neuron_layers_mlp);
            head_gpu_data[head_idx].d_hor_weights.resize(num_weight_matrices_mlp); 
            head_gpu_data[head_idx].d_hor_gweights.resize(num_weight_matrices_mlp); 
            head_gpu_data[head_idx].d_hor_deltas.resize(num_neuron_layers_mlp); 
            head_gpu_data[head_idx].d_ver_activations.resize(num_neuron_layers_mlp);
            head_gpu_data[head_idx].d_ver_weights.resize(num_weight_matrices_mlp);
            head_gpu_data[head_idx].d_ver_gweights.resize(num_weight_matrices_mlp);
            head_gpu_data[head_idx].d_ver_deltas.resize(num_neuron_layers_mlp); 
            
            HeadDevicePointers& current_ptrs = head_gpu_data[head_idx];
            current_ptrs.d_expected_h = agg_d_expected_h + head_idx * embedding_dim;
            current_ptrs.d_EH = agg_d_EH + head_idx * embedding_dim;
            current_ptrs.d_EV = agg_d_EV + head_idx * ev_single_head_elements;
            current_ptrs.d_grad_EH = agg_d_grad_EH + head_idx * embedding_dim;
            current_ptrs.d_grad_EV_scaled = agg_d_grad_EV_scaled + head_idx * embedding_dim;
            current_ptrs.d_grad_dh = agg_d_grad_dh + head_idx * embedding_dim;
            current_ptrs.d_grad_dv = agg_d_grad_dv + head_idx * embedding_dim;
            current_ptrs.d_KdotQ = agg_d_KdotQ + head_idx * max_head_elements_per_head;
            current_ptrs.d_head = agg_d_head + head_idx * max_head_elements_per_head;
            current_ptrs.d_K = agg_d_K + head_idx * max_k_q_elements_per_head;
            current_ptrs.d_Q = agg_d_Q + head_idx * max_k_q_elements_per_head;
            current_ptrs.d_pre_MH = agg_d_pre_MH + head_idx * pre_mh_mv_elements_per_head;
            current_ptrs.d_pre_MV = agg_d_pre_MV + head_idx * pre_mh_mv_elements_per_head;
            current_ptrs.d_MH_a = agg_d_MH_a + head_idx * proj_mat_elements;
            current_ptrs.d_MV_a = agg_d_MV_a + head_idx * proj_mat_elements;
            current_ptrs.d_MQ_a = agg_d_MQ_a + head_idx * proj_mat_elements;
            current_ptrs.d_MK_a = agg_d_MK_a + head_idx * proj_mat_elements;
            current_ptrs.d_grad_MH = agg_d_grad_MH + head_idx * proj_mat_elements;
            current_ptrs.d_grad_MV = agg_d_grad_MV + head_idx * proj_mat_elements;
            current_ptrs.d_grad_head = agg_d_grad_head_storage + head_idx * max_head_elements_per_head;
            current_ptrs.d_lota_deriv = agg_d_lota_deriv + head_idx * max_head_elements_per_head;
            current_ptrs.d_grad_KdotQ = agg_d_grad_KdotQ + head_idx * max_head_elements_per_head;
            current_ptrs.d_grad_K = agg_d_grad_K + head_idx * max_k_q_elements_per_head;
            current_ptrs.d_grad_Q = agg_d_grad_Q + head_idx * max_k_q_elements_per_head;
            current_ptrs.d_grad_MQ = agg_d_grad_MQ + head_idx * proj_mat_elements;
            current_ptrs.d_grad_MK = agg_d_grad_MK + head_idx * proj_mat_elements;

            for (int l = 0; l < num_neuron_layers_mlp; ++l) {
                current_ptrs.d_hor_activations[l] = agg_d_hor_activations_storage + (head_idx * num_neuron_layers_mlp + l) * embedding_dim;
                current_ptrs.d_ver_activations[l] = agg_d_ver_activations_storage + (head_idx * num_neuron_layers_mlp + l) * embedding_dim;
            }
            for (int l = 0; l < num_weight_matrices_mlp; ++l) { // Weights and gweights are num_weight_matrices_mlp
                current_ptrs.d_hor_weights[l] = agg_d_hor_weights_storage + (head_idx * num_weight_matrices_mlp + l) * mlp_weights_elements;
                current_ptrs.d_ver_weights[l] = agg_d_ver_weights_storage + (head_idx * num_weight_matrices_mlp + l) * mlp_weights_elements;
                current_ptrs.d_hor_gweights[l] = agg_d_hor_gweights_storage + (head_idx * num_weight_matrices_mlp + l) * mlp_weights_elements;
                current_ptrs.d_ver_gweights[l] = agg_d_ver_gweights_storage + (head_idx * num_weight_matrices_mlp + l) * mlp_weights_elements;
            }
            for (int l = 0; l < num_neuron_layers_mlp; ++l) { // Deltas are num_neuron_layers_mlp
                current_ptrs.d_hor_deltas[l] = agg_d_hor_deltas_storage + (head_idx * num_neuron_layers_mlp + l) * embedding_dim;
                current_ptrs.d_ver_deltas[l] = agg_d_ver_deltas_storage + (head_idx * num_neuron_layers_mlp + l) * embedding_dim;
            }
        }

        // Iterate backwards through the rows (parallels/layers) for the specified column
        for (int i = num_heads_to_process - 1; i >= 0; --i) { // 'i' is the row index, also used as head_idx
            attention& head_obj = b[i][layno]; // Reference to the current head object
            HeadDevicePointers& device_ptrs = head_gpu_data[i]; // Device pointers for this head
            cudaStream_t current_stream = streams[i];

            const int token_count = head_obj.tokenCount;
            bool att = head_obj.isSelfAttention;
            const size_t active_head_elements = static_cast<size_t>(token_count) * token_count;
            const size_t active_head_bytes = active_head_elements * sizeof(float);
            const size_t active_k_q_elements = static_cast<size_t>(token_count) * mat_heights;
            const size_t active_k_q_bytes = active_k_q_elements * sizeof(float);
            // Adjust 2D grids based on token_count
            dim3 gridDimHead2D((token_count + blockDim2D.x - 1) / blockDim2D.x, (token_count + blockDim2D.y - 1) / blockDim2D.y);
            dim3 gridDimKQGrad2D((mat_heights + blockDim2D.x - 1) / blockDim2D.x, (token_count + blockDim2D.y - 1) / blockDim2D.y);
            int blocksPerGridHead = (static_cast<int>(active_head_elements) + threadsPerBlock1D - 1) / threadsPerBlock1D;
            dim3 gridDimHead(blocksPerGridHead);

            bool is_first_head = (i == 0 && layno == 0);

            // --- Data Transfer H->D (Async Attention) ---
            if (!head_obj.EV.mapped_data || !head_obj.K.mapped_data || !head_obj.Q.mapped_data || !head_obj.KdotQ.mapped_data ||
                !head_obj.MH.mapped_data || !head_obj.MV.mapped_data || !head_obj.MQ.mapped_data || !head_obj.MK.mapped_data) {
                throw std::runtime_error("One or more attention mat objects have null mapped_data for head [" + std::to_string(i) + "][" + std::to_string(layno) + "]");
            }
            CUDA_CHECK(cudaMemcpyAsync(device_ptrs.d_expected_h, expectedH[i].data(), embed_bytes, cudaMemcpyHostToDevice, current_stream));
            CUDA_CHECK(cudaMemcpyAsync(device_ptrs.d_EH, head_obj.EH.data(), embed_bytes, cudaMemcpyHostToDevice, current_stream));
            CUDA_CHECK(cudaMemcpyAsync(device_ptrs.d_EV, head_obj.EV.mapped_data, ev_single_head_bytes, cudaMemcpyHostToDevice, current_stream));
            CUDA_CHECK(cudaMemcpyAsync(device_ptrs.d_MH_a, head_obj.MH.mapped_data, proj_mat_bytes, cudaMemcpyHostToDevice, current_stream));
            CUDA_CHECK(cudaMemcpyAsync(device_ptrs.d_MV_a, head_obj.MV.mapped_data, proj_mat_bytes, cudaMemcpyHostToDevice, current_stream));
            CUDA_CHECK(cudaMemcpyAsync(device_ptrs.d_MQ_a, head_obj.MQ.mapped_data, proj_mat_bytes, cudaMemcpyHostToDevice, current_stream));
            CUDA_CHECK(cudaMemcpyAsync(device_ptrs.d_MK_a, head_obj.MK.mapped_data, proj_mat_bytes, cudaMemcpyHostToDevice, current_stream));
            if (token_count > 0) {
                CUDA_CHECK(cudaMemcpyAsync(device_ptrs.d_KdotQ, head_obj.KdotQ.mapped_data, active_head_bytes, cudaMemcpyHostToDevice, current_stream));
                CUDA_CHECK(cudaMemcpyAsync(device_ptrs.d_K, head_obj.K.mapped_data, active_k_q_bytes, cudaMemcpyHostToDevice, current_stream));
                CUDA_CHECK(cudaMemcpyAsync(device_ptrs.d_Q, head_obj.Q.mapped_data, active_k_q_bytes, cudaMemcpyHostToDevice, current_stream));
            }

            // --- Data Transfer H->D (Async MLP Internals) ---
            if(head_obj.hor.activations.size() != static_cast<size_t>(num_neuron_layers_mlp) ||
                head_obj.ver.activations.size() != static_cast<size_t>(num_neuron_layers_mlp)) {
                throw std::runtime_error("MLP host activations vector size mismatch.");
            }
            for (int l = 0; l < num_neuron_layers_mlp; ++l) {
                if (head_obj.hor.activations[l].empty() || head_obj.ver.activations[l].empty()) {
                    throw std::runtime_error("MLP activation vector is empty for head [" + std::to_string(i) + "][" + std::to_string(layno) + "], layer " + std::to_string(l));
                }
                CUDA_CHECK(cudaMemcpyAsync(device_ptrs.d_hor_activations[l], head_obj.hor.activations[l].data(), embed_bytes, cudaMemcpyHostToDevice, current_stream));
                CUDA_CHECK(cudaMemcpyAsync(device_ptrs.d_ver_activations[l], head_obj.ver.activations[l].data(), embed_bytes, cudaMemcpyHostToDevice, current_stream));
            }
            if(head_obj.hor.weights.size() != static_cast<size_t>(num_weight_matrices_mlp) ||
                head_obj.ver.weights.size() != static_cast<size_t>(num_weight_matrices_mlp)) {
                throw std::runtime_error("MLP host weights vector size mismatch.");
            }
            for (int l = 0; l < num_weight_matrices_mlp; ++l) {
                if (!head_obj.hor.weights[l].mapped_data || head_obj.hor.weights[l].row != embedding_dim || head_obj.hor.weights[l].col != embedding_dim ||
                    !head_obj.ver.weights[l].mapped_data || head_obj.ver.weights[l].row != embedding_dim || head_obj.ver.weights[l].col != embedding_dim ||
                    !head_obj.hor.gweights[l].mapped_data || head_obj.hor.gweights[l].row != embedding_dim || head_obj.hor.gweights[l].col != embedding_dim ||
                    !head_obj.ver.gweights[l].mapped_data || head_obj.ver.gweights[l].row != embedding_dim || head_obj.ver.gweights[l].col != embedding_dim) {
                    throw std::runtime_error("Invalid MLP weight/gweight mat for head [" + std::to_string(i) + "][" + std::to_string(layno) + "], layer " + std::to_string(l));
                }
                CUDA_CHECK(cudaMemcpyAsync(device_ptrs.d_hor_weights[l], head_obj.hor.weights[l].mapped_data, mlp_weights_bytes, cudaMemcpyHostToDevice, current_stream));
                CUDA_CHECK(cudaMemcpyAsync(device_ptrs.d_ver_weights[l], head_obj.ver.weights[l].mapped_data, mlp_weights_bytes, cudaMemcpyHostToDevice, current_stream));
                // Copy gweights if they carry state (e.g., momentum)
                CUDA_CHECK(cudaMemcpyAsync(device_ptrs.d_hor_gweights[l], head_obj.hor.gweights[l].mapped_data, mlp_weights_bytes, cudaMemcpyHostToDevice, current_stream));
                CUDA_CHECK(cudaMemcpyAsync(device_ptrs.d_ver_gweights[l], head_obj.ver.gweights[l].mapped_data, mlp_weights_bytes, cudaMemcpyHostToDevice, current_stream));
            }

            // --- Backpropagation Steps (Async Kernels) ---
            // Step 1: Compute grad_EH and grad_EV_scaled
            kernelComputeGradientsEH_EV<<<gridDimEmbed, blockDim1D, 0, current_stream>>>(device_ptrs.d_EH, device_ptrs.d_expected_h, device_ptrs.d_grad_EH, 
                                        device_ptrs.d_grad_EV_scaled, embedding_dim);
            CUDA_CHECK(cudaGetLastError());

            // --- Step 2: Backprop through MLPs ---
            // --- 2a: Backprop through hor MLP (L_mlp hidden layers) ---
            // Calculate Output Layer Deltas for hor MLP (delta for neuron layer N-1, stored in d_hor_deltas[N-2])
            kernelLastLayerDelta<<<gridDimEmbed, blockDim1D, 0, current_stream>>>(device_ptrs.d_grad_EH, device_ptrs.d_hor_activations[num_total_layers_mlp - 1], 
                                        device_ptrs.d_hor_deltas[num_total_layers_mlp - 2], embedding_dim);
            CUDA_CHECK(cudaGetLastError()); 
            
            // Calculate Hidden Layer Deltas for hor MLP (for neuron layers N-2 down to 1)
            // Loop iterates over the index 'k' of the delta being calculated (0 to N-3, for layers 1 to N-2)
            for (int k = num_total_layers_mlp - 2; k >= 1; --k) { // k is the neuron layer index (1 to N-2)
                hiddenDeltaKernel<<<gridDimEmbed, blockDim1D, 0, current_stream>>>(device_ptrs.d_hor_deltas[k], device_ptrs.d_hor_weights[k], device_ptrs.d_hor_activations[k], 
                                        device_ptrs.d_hor_deltas[k-1], embedding_dim, embedding_dim);
                CUDA_CHECK(cudaGetLastError());
            }
            // Calculate Gradients and Update Weights for hor MLP (N-1 weight matrices, W[0] to W[N-2])
            for (int l_weight_idx = 0; l_weight_idx < num_weight_matrices_mlp; ++l_weight_idx) { 
                kernelUpdateElasticNet<<<gridDimEmbed2D, blockDim2D, 0, current_stream>>>(device_ptrs.d_hor_deltas[l_weight_idx], device_ptrs.d_hor_activations[l_weight_idx], 
                                        device_ptrs.d_hor_weights[l_weight_idx], device_ptrs.d_hor_gweights[l_weight_idx], learning_rate, lambda_l1, lambda_l2, embedding_dim, embedding_dim);
                CUDA_CHECK(cudaGetLastError());
            }

            // --- 2b: Backprop through ver MLP ---
            kernelLastLayerDelta<<<gridDimEmbed, blockDim1D, 0, current_stream>>>(device_ptrs.d_grad_EV_scaled, device_ptrs.d_ver_activations[num_total_layers_mlp - 1], 
                                        device_ptrs.d_ver_deltas[num_total_layers_mlp - 2], embedding_dim);
            CUDA_CHECK(cudaGetLastError()); 
            for (int l_neuron_idx = num_total_layers_mlp - 2; l_neuron_idx >= 1; --l_neuron_idx) {
                hiddenDeltaKernel<<<gridDimEmbed, blockDim1D, 0, current_stream>>>(device_ptrs.d_ver_deltas[l_neuron_idx], device_ptrs.d_ver_weights[l_neuron_idx], 
                                        device_ptrs.d_ver_activations[l_neuron_idx], device_ptrs.d_ver_deltas[l_neuron_idx-1], embedding_dim, embedding_dim);
                CUDA_CHECK(cudaGetLastError());
            }
            for (int l_weight_idx = 0; l_weight_idx < num_weight_matrices_mlp; ++l_weight_idx) {
                kernelUpdateElasticNet<<<gridDimEmbed2D, blockDim2D, 0, current_stream>>>(device_ptrs.d_ver_deltas[l_weight_idx], device_ptrs.d_ver_activations[l_weight_idx], 
                                        device_ptrs.d_ver_weights[l_weight_idx], device_ptrs.d_ver_gweights[l_weight_idx], learning_rate, lambda_l1, lambda_l2, embedding_dim, embedding_dim);
                CUDA_CHECK(cudaGetLastError());
            }

            // --- Step 3: Compute grad_dh and grad_dv ---
            // grad_dh = delta_for_first_hidden_layer * W_input_to_first_hidden_layer
            // d_hor_deltas[0] is delta for neuron layer 1 (stored at index 0)
            // d_hor_weights[0] is W[0] (Input -> Layer 1, stored at index 0)
            kernelComputeGradMLPInput<<<gridDimEmbed, blockDim1D, 0, current_stream>>>(device_ptrs.d_hor_deltas[0], device_ptrs.d_hor_weights[0], device_ptrs.d_grad_dh, embedding_dim, embedding_dim);
            CUDA_CHECK(cudaGetLastError());
            kernelComputeGradMLPInput<<<gridDimEmbed, blockDim1D, 0, current_stream>>>(device_ptrs.d_ver_deltas[0], device_ptrs.d_ver_weights[0], device_ptrs.d_grad_dv, embedding_dim, embedding_dim);
            CUDA_CHECK(cudaGetLastError());

            if(token_count > 0) {
                // --- Step 4: Compute grad_MH and grad_MV ---
                cuLOTA<<<gridDimHead2D, blockDim2D, 0, current_stream>>>(device_ptrs.d_KdotQ, device_ptrs.d_head, context_win, context_win, token_count, att);
                CUDA_CHECK(cudaGetLastError());
                kernelComputePreMH_MV<<<gridDimMatHeights, blockDim1D, 0, current_stream>>>(device_ptrs.d_head, device_ptrs.d_K, device_ptrs.d_Q, device_ptrs.d_pre_MH, device_ptrs.d_pre_MV, token_count, mat_heights);
                CUDA_CHECK(cudaGetLastError());
                // gridDimMatrix2D might need adjustment if mat_heights != embedding_dim
                kernelComputeGradMH_MV<<<gridDimMatrix2D, blockDim2D, 0, current_stream>>>(device_ptrs.d_pre_MH, device_ptrs.d_pre_MV, device_ptrs.d_grad_dh, device_ptrs.d_grad_dv, device_ptrs.d_grad_MH, device_ptrs.d_grad_MV, mat_heights, embedding_dim);
                CUDA_CHECK(cudaGetLastError());

                // --- Step 5: Compute grad_head ---
                kernelComputeGradHead<<<gridDimHead2D, blockDim2D, 0, current_stream>>>(device_ptrs.d_K, device_ptrs.d_Q, device_ptrs.d_MH_a, device_ptrs.d_MV_a, device_ptrs.d_grad_dh, device_ptrs.d_grad_dv, device_ptrs.d_grad_head, token_count, mat_heights, embedding_dim);
                CUDA_CHECK(cudaGetLastError());

                // --- Step 6: Backprop through LOTA ---
                cuLOTAder<<<gridDimHead2D, blockDim2D, 0, current_stream>>>(device_ptrs.d_KdotQ, device_ptrs.d_lota_deriv, context_win, context_win, token_count, att);
                CUDA_CHECK(cudaGetLastError());
                kernelComputeGradKdotQ_LOTA<<<gridDimHead, blockDim1D, 0, current_stream>>>(device_ptrs.d_grad_head, device_ptrs.d_lota_deriv, device_ptrs.d_grad_KdotQ, scaling_factor, static_cast<int>(active_head_elements));
                CUDA_CHECK(cudaGetLastError());

                // --- Step 7: Compute grad_K and grad_Q ---
                kernelComputeGradK_Q<<<gridDimKQGrad2D, blockDim2D, 0, current_stream>>>(device_ptrs.d_grad_KdotQ, device_ptrs.d_K, device_ptrs.d_Q, device_ptrs.d_grad_K, device_ptrs.d_grad_Q, token_count, mat_heights);
                CUDA_CHECK(cudaGetLastError());

                // --- Step 8: Compute grad_MK and grad_MQ (Simplified) ---
                // Requires original token embeddings - assuming they are not directly available here.
                // Using nullptr for k_embed/q_embed, kernel handles null checks.
                kernelComputeGradMK_MQ_Simplified<<<gridDimMatrix2D, blockDim2D, 0, current_stream>>>(device_ptrs.d_grad_K, device_ptrs.d_grad_Q, nullptr, nullptr, device_ptrs.d_grad_MK, device_ptrs.d_grad_MQ, token_count, mat_heights, embedding_dim);
                CUDA_CHECK(cudaGetLastError());
            }
            else {
                // If token_count is 0, attention-related gradients (MH, MV, MQ, MK) should be zero
                // as they won't be computed by the skipped steps.
                CUDA_CHECK(cudaMemsetAsync(device_ptrs.d_grad_MH, 0, proj_mat_bytes, current_stream));
                CUDA_CHECK(cudaMemsetAsync(device_ptrs.d_grad_MV, 0, proj_mat_bytes, current_stream));
                CUDA_CHECK(cudaMemsetAsync(device_ptrs.d_grad_MQ, 0, proj_mat_bytes, current_stream));
                CUDA_CHECK(cudaMemsetAsync(device_ptrs.d_grad_MK, 0, proj_mat_bytes, current_stream));
            }

            // --- Step 9 & 10: Update Weights ---
            if (is_first_head) {
                // Use kernelUpdateWeights_1stHead_H (updates MH, MV, MQ, MK, EH)
                bool update_eh_flag = (layno > 0) ? 1 : 0; // EH updated if not the very first column of the block
                kernelUpdateWeights_1stHead_H<<<gridDimProjMat, blockDim1D, 0, current_stream>>>( // gridDimProjMat
                    device_ptrs.d_MH_a, device_ptrs.d_MV_a, device_ptrs.d_MQ_a, device_ptrs.d_MK_a, device_ptrs.d_EH,
                    device_ptrs.d_grad_MH, device_ptrs.d_grad_MV, device_ptrs.d_grad_MQ, device_ptrs.d_grad_MK, device_ptrs.d_grad_EH,
                    learning_rate, update_eh_flag, // update_eh = true
                    mat_heights, embedding_dim
                );
                CUDA_CHECK(cudaGetLastError());
            }
            else {
                // Use simpler update (updates MH, MV, MQ, MK, EH, EV based on d_grad_dh/dv)
                // Update MH, MV, MQ, MK
                kernelUpdateSimple<<<gridDimProjMat, blockDim1D, 0, current_stream>>>(device_ptrs.d_MH_a, device_ptrs.d_grad_MH, learning_rate, proj_mat_elements);
                kernelUpdateSimple<<<gridDimProjMat, blockDim1D, 0, current_stream>>>(device_ptrs.d_MV_a, device_ptrs.d_grad_MV, learning_rate, proj_mat_elements);
                kernelUpdateSimple<<<gridDimProjMat, blockDim1D, 0, current_stream>>>(device_ptrs.d_MQ_a, device_ptrs.d_grad_MQ, learning_rate, proj_mat_elements);
                kernelUpdateSimple<<<gridDimProjMat, blockDim1D, 0, current_stream>>>(device_ptrs.d_MK_a, device_ptrs.d_grad_MK, learning_rate, proj_mat_elements);
                CUDA_CHECK(cudaGetLastError());
                // For non-first heads in the first block, behave like attention::cuBackward
                if (layno > 0) { // Corresponds to headnumber > 1 in attention::cuBackward
                    kernelUpdateSimple<<<gridDimEmbed, blockDim1D, 0, current_stream>>>(device_ptrs.d_EH, device_ptrs.d_grad_EH, learning_rate, embedding_dim);
                    CUDA_CHECK(cudaGetLastError());
                }
                // Update EV using d_grad_EV_scaled (0.1f * d_grad_EH)
                kernelUpdateEVBroadcasted<<< (context_win + threadsPerBlock1D -1) / threadsPerBlock1D, blockDim1D, 0, current_stream >>>(device_ptrs.d_EV, device_ptrs.d_grad_EV_scaled, learning_rate, context_win, embedding_dim);
                CUDA_CHECK(cudaGetLastError());
            }

            // --- Data Transfer D->H (Async) ---
            // Copy updated MLP weights and gradients back
            for (int l = 0; l < num_weight_matrices_mlp; ++l) {
                // Hor MLP
                CUDA_CHECK(cudaMemcpyAsync(head_obj.hor.weights[l].mapped_data, device_ptrs.d_hor_weights[l], mlp_weights_bytes, cudaMemcpyDeviceToHost, current_stream));
                CUDA_CHECK(cudaMemcpyAsync(head_obj.hor.gweights[l].mapped_data, device_ptrs.d_hor_gweights[l], mlp_weights_bytes, cudaMemcpyDeviceToHost, current_stream));
                // Ver MLP
                CUDA_CHECK(cudaMemcpyAsync(head_obj.ver.weights[l].mapped_data, device_ptrs.d_ver_weights[l], mlp_weights_bytes, cudaMemcpyDeviceToHost, current_stream));
                CUDA_CHECK(cudaMemcpyAsync(head_obj.ver.gweights[l].mapped_data, device_ptrs.d_ver_gweights[l], mlp_weights_bytes, cudaMemcpyDeviceToHost, current_stream));
            }

            // Copy updated Attention parameters back
            CUDA_CHECK(cudaMemcpyAsync(head_obj.EH.data(), device_ptrs.d_EH, embed_bytes, cudaMemcpyDeviceToHost, current_stream));
            CUDA_CHECK(cudaMemcpyAsync(head_obj.EV.mapped_data, device_ptrs.d_EV, ev_single_head_bytes, cudaMemcpyDeviceToHost, current_stream));
            CUDA_CHECK(cudaMemcpyAsync(head_obj.MH.mapped_data, device_ptrs.d_MH_a, proj_mat_bytes, cudaMemcpyDeviceToHost, current_stream));
            CUDA_CHECK(cudaMemcpyAsync(head_obj.MV.mapped_data, device_ptrs.d_MV_a, proj_mat_bytes, cudaMemcpyDeviceToHost, current_stream));
            CUDA_CHECK(cudaMemcpyAsync(head_obj.MQ.mapped_data, device_ptrs.d_MQ_a, proj_mat_bytes, cudaMemcpyDeviceToHost, current_stream));
            CUDA_CHECK(cudaMemcpyAsync(head_obj.MK.mapped_data, device_ptrs.d_MK_a, proj_mat_bytes, cudaMemcpyDeviceToHost, current_stream));
        } // End loop over rows (i)

        // --- Synchronize all streams ---
        for (int head_idx = 0; head_idx < num_heads_to_process; ++head_idx) {
            if (streams[head_idx]) {
                CUDA_CHECK(cudaStreamSynchronize(streams[head_idx]));
            }
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error during cu1ParallelBackward1stBlock (H) for column " << layno << ": " << e.what() << std::endl;
        // Cleanup aggregate memory and streams on error
        for (int head_idx = 0; head_idx < num_heads_to_process; ++head_idx) { if (streams[head_idx]) { cudaStreamDestroy(streams[head_idx]); streams[head_idx] = nullptr; } }
        cudaFree(agg_d_expected_h); cudaFree(agg_d_EH); cudaFree(agg_d_EV); cudaFree(agg_d_grad_EH); cudaFree(agg_d_grad_EV_scaled); cudaFree(agg_d_grad_dh); cudaFree(agg_d_grad_dv); cudaFree(agg_d_KdotQ); cudaFree(agg_d_head); cudaFree(agg_d_K); cudaFree(agg_d_Q); cudaFree(agg_d_pre_MH); cudaFree(agg_d_pre_MV); cudaFree(agg_d_MH_a); cudaFree(agg_d_MV_a); cudaFree(agg_d_MQ_a); cudaFree(agg_d_MK_a); cudaFree(agg_d_grad_MH); cudaFree(agg_d_grad_MV); cudaFree(agg_d_grad_head_storage); cudaFree(agg_d_lota_deriv); cudaFree(agg_d_grad_KdotQ); cudaFree(agg_d_grad_K); cudaFree(agg_d_grad_Q); cudaFree(agg_d_grad_MQ); cudaFree(agg_d_grad_MK);
        cudaFree(agg_d_hor_activations_storage); cudaFree(agg_d_ver_activations_storage); cudaFree(agg_d_hor_weights_storage); cudaFree(agg_d_ver_weights_storage); cudaFree(agg_d_hor_gweights_storage); cudaFree(agg_d_ver_gweights_storage); cudaFree(agg_d_hor_deltas_storage); cudaFree(agg_d_ver_deltas_storage);
        throw std::runtime_error("Exception during cu1ParallelBackward1stBlock (H) for column " + std::to_string(layno) + ": " + e.what());
    }

    // --- Cleanup Device Memory and Streams (Success Path) ---
    for (int head_idx = 0; head_idx < num_heads_to_process; ++head_idx) {
        if (streams[head_idx]) {
            CUDA_CHECK(cudaStreamDestroy(streams[head_idx]));
        }
    }
    cudaFree(agg_d_expected_h); cudaFree(agg_d_EH); cudaFree(agg_d_EV);
    cudaFree(agg_d_grad_EH); cudaFree(agg_d_grad_EV_scaled);
    cudaFree(agg_d_grad_dh); cudaFree(agg_d_grad_dv);
    cudaFree(agg_d_KdotQ); cudaFree(agg_d_head); cudaFree(agg_d_K); cudaFree(agg_d_Q);
    cudaFree(agg_d_pre_MH); cudaFree(agg_d_pre_MV);
    cudaFree(agg_d_MH_a); cudaFree(agg_d_MV_a); cudaFree(agg_d_MQ_a); cudaFree(agg_d_MK_a);
    cudaFree(agg_d_grad_MH); cudaFree(agg_d_grad_MV); cudaFree(agg_d_grad_head_storage);
    cudaFree(agg_d_lota_deriv); cudaFree(agg_d_grad_KdotQ);
    cudaFree(agg_d_grad_K); cudaFree(agg_d_grad_Q);
    cudaFree(agg_d_grad_MQ); cudaFree(agg_d_grad_MK);
    cudaFree(agg_d_hor_activations_storage); cudaFree(agg_d_ver_activations_storage);
    cudaFree(agg_d_hor_weights_storage); cudaFree(agg_d_ver_weights_storage);
    cudaFree(agg_d_hor_gweights_storage); cudaFree(agg_d_ver_gweights_storage);
    cudaFree(agg_d_hor_deltas_storage); cudaFree(agg_d_ver_deltas_storage);
}

#endif  // USE_CUDA