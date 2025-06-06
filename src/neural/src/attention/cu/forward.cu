
#include "include/mlp.hpp"       // For mlp class definition
#include "include/attention.hpp" // For attention class definition
#include <maths.hpp>             // Should declare cuLOTA, cuReLU, cuSigmoid from activations.cu

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <numeric>
#include <algorithm> // For std::max, std::abs, std::min
#include <cmath>     // For std::abs used in count calculation

// Helper macro for CUDA error checking
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        throw std::runtime_error(cudaGetErrorString(err)); \
    } \
} while (0)

// Helper function for transposing a mat object's data into a flat vector (row-major)
// Takes an R x C matrix m and produces output_flat representing a C x R matrix.
static void transposeMatToFlatVector(const mat& m, std::vector<float>& output_flat) {
    if (!m.mapped_data) {
        output_flat.clear();
        if (m.row != 0 || m.col != 0) { // Invalid state: dimensions but no data
            throw std::runtime_error("Mat has non-zero dimensions but null mapped_data in transposeMatToFlatVector.");
        }
        return; // Valid empty mat
    }
    if (m.row == 0 || m.col == 0) { // Valid empty mat
        output_flat.clear();
        return;
    }
    int R = m.row; // Original rows
    int C = m.col; // Original columns
    output_flat.resize(static_cast<size_t>(R) * C); // Will store data for a C x R matrix

    for (int j = 0; j < C; ++j) {        // Iterate original columns (these become rows in the transposed version)
        for (int i = 0; i < R; ++i) {    // Iterate original rows (these become columns in the transposed version)
            output_flat[static_cast<size_t>(j) * R + i] = m(i, j); // Access m(original_row, original_col)
        }
    }
}

/**
 * @brief CUDA forward propagation for first block's attention class (incomplete attention)
 * @param d_embedding embedding dimension (in)
 * @param layers_mlp layers of hidden weights in mlp (layers) - NOTE: This seems unused in the logic, mlp.hlayers.size() is used instead.
 * @param currentTokenCount token count for this attention head (tokenCount)
 */
void attention::cuforprop(int& d_embedding, int& layers_mlp, int& currentTokenCount)
{
    // Use constants defined in attention.hpp for clarity
    const int d = EMBEDDING;        // Embedding dimension
    const int h = MATHEIGHTS;       // Height dimension of K/Q, projection matrices
    const int n = currentTokenCount;// Number of tokens
    // const int mlp_hidden_layers = layers_mlp; // Parameter seems redundant if using mlp object sizes

    // --- Basic Validation ---
    if (n <= 0) {
        std::cerr << "Warning: cuforprop(..., tokenCount=" << n << ") called with tokenCount <= 0. Skipping computation." << std::endl;
        // Ensure outputs are zeroed or handled appropriately upstream if needed
        std::fill(EH.begin(), EH.end(), 0.0f);
        // If EV is a mat, it should be pre-allocated. We might zero the relevant row if n=0.
        if (n == 0) { // If tokenCount is exactly 0, implies initializing state for the 0-th token.
            if (EV.mapped_data && EV.row > 0 && EV.col == d) {
                // Zero out the 0-th row of EV on the host side.
                std::fill_n(EV.mapped_data, EV.col, 0.0f);
            }
        }
        return;
    }

    // K (n x h), Q (n x h), KdotQ (n x n)
    // MH (d x h), MV (d x h)
    // EV (CONTEXT_WIN x d), 'n' active rows
    // K, Q, KdotQ are pre-allocated to CONTEXT_WIN based sizes.
    // 'n' is the current active number of tokens.
    if (K.row != CONTEXT_WIN || K.col != h ||
        Q.row != CONTEXT_WIN || Q.col != h ||
        KdotQ.row != CONTEXT_WIN || KdotQ.col != CONTEXT_WIN || // Assuming KdotQ is pre-allocated to CONTEXT_WIN x CONTEXT_WIN
        n > CONTEXT_WIN || // Active token count 'n' must not exceed CONTEXT_WIN
        MH.row != d || MH.col != h || // MH is d x h
        MV.row != d || MV.col != h || // MV is d x h
        EH.size() != static_cast<size_t>(d) ||
        // EV validation: must be mapped, allocated for CONTEXT_WIN rows, col must be d. 'n' active rows.
        (!EV.mapped_data || EV.row != CONTEXT_WIN || EV.col != d) ||
        hor.hlayers.empty() || ver.hlayers.empty() || hor.weights.empty() || ver.weights.empty())
    {
        // Provide more specific error messages if possible
        std::string error_msg = "Attention component dimension mismatch or uninitialized member in cuforprop(..., currentTokenCount=" + std::to_string(n) + ").\n";
        if (K.row != CONTEXT_WIN || K.col != h) error_msg += " K dim: " + std::to_string(K.row) + "x" + std::to_string(K.col) + " (expected " + std::to_string(CONTEXT_WIN) + "x" + std::to_string(h) + ")\n";
        if (Q.row != CONTEXT_WIN || Q.col != h) error_msg += " Q dim: " + std::to_string(Q.row) + "x" + std::to_string(Q.col) + " (expected " + std::to_string(CONTEXT_WIN) + "x" + std::to_string(h) + ")\n";
        if (KdotQ.row != CONTEXT_WIN || KdotQ.col != CONTEXT_WIN) error_msg += " KdotQ dim: " + std::to_string(KdotQ.row) + "x" + std::to_string(KdotQ.col) + " (expected " + std::to_string(CONTEXT_WIN) + "x" + std::to_string(CONTEXT_WIN) + ")\n";
        if (n > CONTEXT_WIN) error_msg += " Active token count n=" + std::to_string(n) + " exceeds CONTEXT_WIN=" + std::to_string(CONTEXT_WIN) + "\n";
        if (!EV.mapped_data || EV.row != CONTEXT_WIN || EV.col != d) error_msg += " EV dim/map error. Mapped: " + std::string(EV.mapped_data ? "yes" : "no") + ", Dim: " + std::to_string(EV.row) + "x" + std::to_string(EV.col) + " (expected " + std::to_string(CONTEXT_WIN) + "x" + std::to_string(d) + ")\n";
        // Add more specific checks for MH, MV, EH, MLPs if needed
        throw std::runtime_error(error_msg);
    }
    if (d != d_embedding) {
        throw std::runtime_error("Embedding dimension mismatch between definition (EMBEDDING) and argument (d_embedding).");
    }
    // Check MLP dimensions against embedding dim 'd'
    // Assuming MLP input/output and hidden layers match 'd' based on previous corrections
    if (hor.hlayers[0].size() != static_cast<size_t>(d) || ver.hlayers[0].size() != static_cast<size_t>(d) ||
        hor.weights.back().row != d || ver.weights.back().row != d) { // Last weight matrix output dim (rows) must be d
         throw std::runtime_error("MLP input/output layer dimension mismatch with embedding dimension 'd'.");
    }


    // --- Device Memory Pointers ---
    float *d_K = nullptr, *d_Q = nullptr, *d_KdotQ = nullptr, *d_head = nullptr;
    float *d_row_sums = nullptr, *d_col_sums = nullptr;
    float *d_dh_accum = nullptr, *d_dv_accum = nullptr;
    float *d_MH_hxd = nullptr, *d_MV_hxd = nullptr; // Transposed: h x d
    float *d_dh = nullptr, *d_dv = nullptr;         // Projected result: 1 x d (size d)
    float *d_EH = nullptr;
    float *d_EV_processed_data = nullptr; // Buffer for the first 'n' rows of EV
    float *d_ver_accumulated_ev = nullptr; // Buffer for the sum of EV rows
    float *d_hor_inputs = nullptr, *d_ver_inputs = nullptr; // Inputs to MLP after residual add
    float *d_hor_output = nullptr, *d_ver_output = nullptr; // Final MLP outputs (pre-ReLU)
    float *d_relu_hor_output = nullptr, *d_relu_ver_output = nullptr; // After ReLU

    // MLP Intermediate Buffers (Merged Logic)
    float *d_mlp_bufferA_hor = nullptr, *d_mlp_bufferB_hor = nullptr; // Ping-pong for hor
    float *d_mlp_bufferA_ver = nullptr, *d_mlp_bufferB_ver = nullptr; // Ping-pong for ver
    float *d_mlp_pre_activation = nullptr; // Reused for pre-activation calculation
    float *d_mlp_weights = nullptr;        // Reused for weights (allocated/freed per layer per MLP)

    try {
        // --- Allocate Device Memory ---
        size_t k_bytes = static_cast<size_t>(n) * h * sizeof(float);
        size_t q_bytes = static_cast<size_t>(n) * h * sizeof(float);
        size_t kdotq_bytes = static_cast<size_t>(n) * n * sizeof(float);
        size_t head_bytes = static_cast<size_t>(n) * n * sizeof(float);
        size_t sums_bytes = static_cast<size_t>(n) * sizeof(float);
        size_t accum_bytes = static_cast<size_t>(h) * sizeof(float);
        size_t proj_mat_bytes = static_cast<size_t>(h) * d * sizeof(float); // h x d
        size_t ev_processed_bytes = static_cast<size_t>(n) * d * sizeof(float);
        size_t embed_bytes = static_cast<size_t>(d) * sizeof(float);        // d

        CUDA_CHECK(cudaMalloc(&d_K, k_bytes));
        CUDA_CHECK(cudaMalloc(&d_Q, q_bytes));
        CUDA_CHECK(cudaMalloc(&d_KdotQ, kdotq_bytes));
        CUDA_CHECK(cudaMalloc(&d_head, head_bytes));
        CUDA_CHECK(cudaMalloc(&d_row_sums, sums_bytes));
        CUDA_CHECK(cudaMalloc(&d_col_sums, sums_bytes));
        CUDA_CHECK(cudaMalloc(&d_dh_accum, accum_bytes));
        CUDA_CHECK(cudaMalloc(&d_dv_accum, accum_bytes));
        CUDA_CHECK(cudaMalloc(&d_MH_hxd, proj_mat_bytes));
        CUDA_CHECK(cudaMalloc(&d_MV_hxd, proj_mat_bytes));
        CUDA_CHECK(cudaMalloc(&d_dh, embed_bytes));
        CUDA_CHECK(cudaMalloc(&d_dv, embed_bytes));
        CUDA_CHECK(cudaMalloc(&d_EH, embed_bytes));
        CUDA_CHECK(cudaMalloc(&d_EV_processed_data, ev_processed_bytes));
        CUDA_CHECK(cudaMalloc(&d_ver_accumulated_ev, embed_bytes));
        CUDA_CHECK(cudaMalloc(&d_hor_inputs, embed_bytes)); // MLP input after add
        CUDA_CHECK(cudaMalloc(&d_ver_inputs, embed_bytes)); // MLP input after add
        CUDA_CHECK(cudaMalloc(&d_hor_output, embed_bytes)); // MLP final output (pre-ReLU)
        CUDA_CHECK(cudaMalloc(&d_ver_output, embed_bytes)); // MLP final output (pre-ReLU)
        CUDA_CHECK(cudaMalloc(&d_relu_hor_output, embed_bytes)); // After ReLU
        CUDA_CHECK(cudaMalloc(&d_relu_ver_output, embed_bytes)); // After ReLU

        // Allocate MLP Intermediate Buffers (Merged Logic)
        CUDA_CHECK(cudaMalloc(&d_mlp_bufferA_hor, embed_bytes));
        CUDA_CHECK(cudaMalloc(&d_mlp_bufferB_hor, embed_bytes));
        CUDA_CHECK(cudaMalloc(&d_mlp_bufferA_ver, embed_bytes));
        CUDA_CHECK(cudaMalloc(&d_mlp_bufferB_ver, embed_bytes));
        CUDA_CHECK(cudaMalloc(&d_mlp_pre_activation, embed_bytes));

        // Initialize accumulators to zero
        CUDA_CHECK(cudaMemset(d_dh_accum, 0, accum_bytes));
        CUDA_CHECK(cudaMemset(d_dv_accum, 0, accum_bytes));

        // --- Flatten Host Data & Copy H->D ---
        // K, Q, KdotQ are mat objects, copy directly from their mapped_data
        if (!K.mapped_data || !Q.mapped_data || !KdotQ.mapped_data)
            throw std::runtime_error("K, Q, or KdotQ have null mapped_data.");

        CUDA_CHECK(cudaMemcpy(d_K, K.mapped_data, k_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_Q, Q.mapped_data, q_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_KdotQ, KdotQ.mapped_data, kdotq_bytes, cudaMemcpyHostToDevice));

        // Transpose MH (d x h) and MV (d x h) from host storage to h x d for device projection kernel
        // MH is d x h (MH.row = d, MH.col = h). flat_MH_hxd needs to represent h x d.
        std::vector<float> flat_MH_hxd, flat_MV_hxd;
        transposeMatToFlatVector(MH, flat_MH_hxd); // MH (d x h) -> flat_MH_hxd (represents h x d)
        transposeMatToFlatVector(MV, flat_MV_hxd); // MV (d x h) -> flat_MV_hxd (represents h x d)

        CUDA_CHECK(cudaMemcpy(d_MH_hxd, flat_MH_hxd.data(), proj_mat_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_MV_hxd, flat_MV_hxd.data(), proj_mat_bytes, cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMemcpy(d_EH, EH.data(), embed_bytes, cudaMemcpyHostToDevice));
        // Copy the first 'n' rows from EV mat to device buffer
        CUDA_CHECK(cudaMemcpy(d_EV_processed_data, EV.mapped_data, ev_processed_bytes, cudaMemcpyHostToDevice));

        // --- Kernel Launches ---
        const int threadsPerBlock = 256; // General purpose block size, adjust as needed

        // 1. Calculate Attention Weights: head = LOTA(KdotQ)
        int totalElementsLOTA = n * n;
        int blocksLOTA = (totalElementsLOTA + threadsPerBlock - 1) / threadsPerBlock;
        cuLOTA<<<blocksLOTA, threadsPerBlock>>>(d_KdotQ, d_head, n, n);
        CUDA_CHECK(cudaGetLastError());

        // 2. Compute Head Row/Column Sums (Masked)
        int blocksSums = (n + threadsPerBlock - 1) / threadsPerBlock;
        computeHeadSumsMaskedKernel<<<blocksSums, threadsPerBlock>>>(d_head, d_row_sums, d_col_sums, n, isSelfAttention);
        CUDA_CHECK(cudaGetLastError());

        // 3. Accumulate Weighted K and Q vectors
        int blocksAccum = (h + threadsPerBlock - 1) / threadsPerBlock;
        accumulateWeightedVectorsKernel<<<blocksAccum, threadsPerBlock>>>(d_row_sums, d_col_sums, d_K, d_Q, d_dh_accum, d_dv_accum, n, h);
        CUDA_CHECK(cudaGetLastError());

        // 4. Project accumulated vectors: dh = dot(dh_accum, MH), dv = dot(dv_accum, MV)
        int threadsPerBlockMatMul = 256;
        int blocksPerGridMatMul = (d + threadsPerBlockMatMul - 1) / threadsPerBlockMatMul;
        matrixMultiplyKernel<<<blocksPerGridMatMul, threadsPerBlockMatMul>>>(d_dh_accum, d_MH_hxd, d_dh, 1, h, d);
        CUDA_CHECK(cudaGetLastError());
        matrixMultiplyKernel<<<blocksPerGridMatMul, threadsPerBlockMatMul>>>(d_dv_accum, d_MV_hxd, d_dv, 1, h, d);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize()); // Ensure d_dh, d_dv are ready

        // 5. Prepare MLP inputs (Residual Add): hor_input = EH + dh, ver_input = EV_current + dv
        int blocksAdd = (d + threadsPerBlock - 1) / threadsPerBlock;
        vectorAddKernel<<<blocksAdd, threadsPerBlock>>>(d_EH, d_dh, d_hor_inputs, d);
        CUDA_CHECK(cudaGetLastError());

        // Accumulate first 'n' rows of EV for ver_input
        int blocksAccumEV = (d + threadsPerBlock - 1) / threadsPerBlock; // d is col_size for accumulateEVRowsKernel
        accumulateEVRowsKernel<<<blocksAccumEV, threadsPerBlock>>>(d_EV_processed_data, d_ver_accumulated_ev, n, d);
        CUDA_CHECK(cudaGetLastError());
        vectorAddKernel<<<blocksAdd, threadsPerBlock>>>(d_ver_accumulated_ev, d_dv, d_ver_inputs, d);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize()); // Sync before MLP uses these inputs

        // --- 6. Run MLPs Forward (Merged Loop) ---
        try {
            // Initial input copies
            CUDA_CHECK(cudaMemcpy(d_mlp_bufferA_hor, d_hor_inputs, embed_bytes, cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_mlp_bufferA_ver, d_ver_inputs, embed_bytes, cudaMemcpyDeviceToDevice));

            // Pointers for ping-pong buffers
            float* d_current_in_hor = d_mlp_bufferA_hor;
            float* d_current_out_hor = d_mlp_bufferB_hor;
            float* d_current_in_ver = d_mlp_bufferA_ver;
            float* d_current_out_ver = d_mlp_bufferB_ver;

            int threadsPerBlockMLP = 256; // Can tune this

            // Loop through all weight matrices (including the final projection)
            size_t num_weight_matrices = hor.weights.size(); // Should be layers + 1
            if (num_weight_matrices != ver.weights.size()) {
                throw std::runtime_error("Horizontal and Vertical MLPs must have the same number of weight matrices.");
            }
             if (num_weight_matrices == 0) { // Handle case of MLP with no layers (only projection)
                 throw std::runtime_error("MLP weights cannot be empty.");
             }
            if (num_weight_matrices != hor.hlayers.size() + 1) {
                 throw std::runtime_error("MLP weights size mismatch with hlayers size.");
            }


            for (size_t layer_idx = 0; layer_idx < num_weight_matrices; ++layer_idx) {
                bool is_last_layer = (layer_idx == num_weight_matrices - 1);

                // --- Process Horizontal MLP Layer ---
                { // Scope for hor layer processing
                    int input_size = (layer_idx == 0) ? d : hor.hlayers[layer_idx-1].size();
                    // Output size is d for the last layer, otherwise size of the corresponding hlayer
                    int output_size = is_last_layer ? d : hor.hlayers[layer_idx].size();

                    if (input_size != d || output_size != d) {
                         throw std::runtime_error("MLP layers currently must have size equal to embedding dimension (d).");
                    }

                    const mat& current_weights_mat = hor.weights[layer_idx];
                    if (current_weights_mat.row != output_size || current_weights_mat.col != input_size) {
                        std::string err_msg = "MLP weight mat dimension mismatch (hor). Expected " +
                                              std::to_string(output_size) + "x" + std::to_string(input_size) +
                                              ", got " + std::to_string(current_weights_mat.row) + "x" +
                                              std::to_string(current_weights_mat.col) + " for layer " + std::to_string(layer_idx);
                        throw std::runtime_error(err_msg);
                    }
                    if (!current_weights_mat.mapped_data) {
                        throw std::runtime_error("MLP weight matrix (hor) has null mapped_data for layer " + std::to_string(layer_idx));
                    }

                    size_t weights_bytes = static_cast<size_t>(current_weights_mat.row) * current_weights_mat.col * sizeof(float);
                    CUDA_CHECK(cudaMalloc(&d_mlp_weights, weights_bytes));
                    CUDA_CHECK(cudaMemcpy(d_mlp_weights, current_weights_mat.mapped_data, weights_bytes, cudaMemcpyHostToDevice));


                    int blocksPerGridMLP = (output_size + threadsPerBlockMLP - 1) / threadsPerBlockMLP;

                    if (is_last_layer) {
                        // Final projection: input -> d_hor_output
                        layerForwardKernel<<<blocksPerGridMLP, threadsPerBlockMLP>>>(
                            d_current_in_hor, d_mlp_weights, d_hor_output, input_size, output_size);
                        CUDA_CHECK(cudaGetLastError());
                    } 
                    else {
                        // Hidden layer: input -> pre-activation -> sigmoid -> output
                        layerForwardKernel<<<blocksPerGridMLP, threadsPerBlockMLP>>>(
                            d_current_in_hor, d_mlp_weights, d_mlp_pre_activation, input_size, output_size);
                        CUDA_CHECK(cudaGetLastError());

                        cuSigmoid<<<blocksPerGridMLP, threadsPerBlockMLP>>>(d_mlp_pre_activation, d_current_out_hor, output_size);
                        CUDA_CHECK(cudaGetLastError());

                        // Swap hor buffers
                        float* temp = d_current_in_hor;
                        d_current_in_hor = d_current_out_hor;
                        d_current_out_hor = temp;
                    }
                    CUDA_CHECK(cudaFree(d_mlp_weights));
                    d_mlp_weights = nullptr; // Reset pointer
                } // End scope for hor layer processing

                // --- Process Vertical MLP Layer ---
                { // Scope for ver layer processing
                    int input_size = (layer_idx == 0) ? d : ver.hlayers[layer_idx-1].size();
                    int output_size = is_last_layer ? d : ver.hlayers[layer_idx].size();

                    if (input_size != d || output_size != d) {
                         throw std::runtime_error("MLP layers currently must have size equal to embedding dimension (d).");
                    }

                    const mat& current_weights_mat = ver.weights[layer_idx];
                    if (current_weights_mat.row != output_size || current_weights_mat.col != input_size) {
                        std::string err_msg = "MLP weight mat dimension mismatch (ver). Expected " +
                                              std::to_string(output_size) + "x" + std::to_string(input_size) +
                                              ", got " + std::to_string(current_weights_mat.row) + "x" +
                                              std::to_string(current_weights_mat.col) + " for layer " + std::to_string(layer_idx);
                        throw std::runtime_error(err_msg);
                    }
                     if (!current_weights_mat.mapped_data) {
                        throw std::runtime_error("MLP weight matrix (ver) has null mapped_data for layer " + std::to_string(layer_idx));
                    }

                    size_t weights_bytes = static_cast<size_t>(current_weights_mat.row) * current_weights_mat.col * sizeof(float);
                    CUDA_CHECK(cudaMalloc(&d_mlp_weights, weights_bytes));
                    CUDA_CHECK(cudaMemcpy(d_mlp_weights, current_weights_mat.mapped_data, weights_bytes, cudaMemcpyHostToDevice));

                    int blocksPerGridMLP = (output_size + threadsPerBlockMLP - 1) / threadsPerBlockMLP;

                    if (is_last_layer) {
                        // Final projection: input -> d_ver_output
                        layerForwardKernel<<<blocksPerGridMLP, threadsPerBlockMLP>>>(
                            d_current_in_ver, d_mlp_weights, d_ver_output, input_size, output_size);
                        CUDA_CHECK(cudaGetLastError());
                    }
                    else {
                        // Hidden layer: input -> pre-activation -> sigmoid -> output
                        layerForwardKernel<<<blocksPerGridMLP, threadsPerBlockMLP>>>(
                            d_current_in_ver, d_mlp_weights, d_mlp_pre_activation, input_size, output_size);
                        CUDA_CHECK(cudaGetLastError());

                        cuSigmoid<<<blocksPerGridMLP, threadsPerBlockMLP>>>(d_mlp_pre_activation, d_current_out_ver, output_size);
                        CUDA_CHECK(cudaGetLastError());

                        // Swap ver buffers
                        float* temp = d_current_in_ver;
                        d_current_in_ver = d_current_out_ver;
                        d_current_out_ver = temp;
                    }
                    CUDA_CHECK(cudaFree(d_mlp_weights));
                    d_mlp_weights = nullptr; // Reset pointer
                } // End scope for ver layer processing

            } // End loop over layers

            CUDA_CHECK(cudaDeviceSynchronize()); // Sync after all MLP layers are processed

        }
        catch (const std::exception& e) {
            std::cerr << "Error during Merged MLP processing: " << e.what() << std::endl;
            cudaFree(d_mlp_weights); // Ensure cleanup if error occurs mid-layer
            // Free other MLP buffers in the main catch block outside this section
            throw;
        }

        // --- Free MLP Intermediate Buffers (No longer needed) ---
        CUDA_CHECK(cudaFree(d_mlp_bufferA_hor)); d_mlp_bufferA_hor = nullptr;
        CUDA_CHECK(cudaFree(d_mlp_bufferB_hor)); d_mlp_bufferB_hor = nullptr;
        CUDA_CHECK(cudaFree(d_mlp_bufferA_ver)); d_mlp_bufferA_ver = nullptr;
        CUDA_CHECK(cudaFree(d_mlp_bufferB_ver)); d_mlp_bufferB_ver = nullptr;
        CUDA_CHECK(cudaFree(d_mlp_pre_activation)); d_mlp_pre_activation = nullptr;


        // 7. Apply ReLU to MLP outputs (d_hor_output, d_ver_output now hold final MLP results)
        //    Using cuReLU(float* x, float* out, int size) from activations.cu
        cuReLU<<<blocksAdd, threadsPerBlock>>>(d_hor_output, d_relu_hor_output, d);
        CUDA_CHECK(cudaGetLastError());
        cuReLU<<<blocksAdd, threadsPerBlock>>>(d_ver_output, d_relu_ver_output, d);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize()); // Sync before final add

        // 8. Final Residual Update: EH = EH + ReLU(hor_output), EV[n] = EV[n] + ReLU(ver_output)
        vectorAddKernel<<<blocksAdd, threadsPerBlock>>>(d_EH, d_relu_hor_output, d_EH, d); // Update d_EH in-place
        CUDA_CHECK(cudaGetLastError());

        // Add ReLU(ver_output) to the first 'n' rows of EV
        int blocksUpdateEV = (n + threadsPerBlock - 1) / threadsPerBlock;
        updateEVRowsKernel<<<blocksUpdateEV, threadsPerBlock>>>(d_EV_processed_data, d_relu_ver_output, n, d);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize()); // Sync before D->H copy

        // --- Copy Results D->H ---
        CUDA_CHECK(cudaMemcpy(EH.data(), d_EH, embed_bytes, cudaMemcpyDeviceToHost));
        // Copy updated EV rows back
        CUDA_CHECK(cudaMemcpy(EV.mapped_data, d_EV_processed_data, ev_processed_bytes, cudaMemcpyDeviceToHost));

        // --- Free Device Memory ---
        cudaFree(d_K); cudaFree(d_Q); cudaFree(d_KdotQ); cudaFree(d_head);
        cudaFree(d_row_sums); cudaFree(d_col_sums);
        cudaFree(d_dh_accum); cudaFree(d_dv_accum);
        cudaFree(d_MH_hxd); cudaFree(d_MV_hxd);
        cudaFree(d_dh); cudaFree(d_dv);
        cudaFree(d_EH);
        cudaFree(d_EV_processed_data);
        cudaFree(d_ver_accumulated_ev);
        cudaFree(d_hor_inputs); cudaFree(d_ver_inputs); // Free the initial MLP input buffers
        cudaFree(d_hor_output); cudaFree(d_ver_output); // Free the final MLP output buffers
        cudaFree(d_relu_hor_output); cudaFree(d_relu_ver_output); // Free the ReLU output buffers
        // MLP intermediate buffers were freed earlier

    }
    catch (const std::exception& e) {
        std::cerr << "CUDA Exception in attention cuforprop(..., tokenCount): " << e.what() << std::endl;
        // --- Cleanup on Error ---
        // Free ALL potentially allocated buffers, including MLP intermediates
        cudaFree(d_K); cudaFree(d_Q); cudaFree(d_KdotQ); cudaFree(d_head);
        cudaFree(d_row_sums); cudaFree(d_col_sums);
        cudaFree(d_dh_accum); cudaFree(d_dv_accum);
        cudaFree(d_MH_hxd); cudaFree(d_MV_hxd);
        cudaFree(d_dh); cudaFree(d_dv);
        cudaFree(d_EH);
        cudaFree(d_EV_processed_data);
        cudaFree(d_ver_accumulated_ev);
        cudaFree(d_hor_inputs); cudaFree(d_ver_inputs);
        cudaFree(d_hor_output); cudaFree(d_ver_output);
        cudaFree(d_relu_hor_output); cudaFree(d_relu_ver_output);
        // Free merged MLP buffers
        cudaFree(d_mlp_bufferA_hor);
        cudaFree(d_mlp_bufferB_hor);
        cudaFree(d_mlp_bufferA_ver);
        cudaFree(d_mlp_bufferB_ver);
        cudaFree(d_mlp_pre_activation);
        cudaFree(d_mlp_weights); // Free weights buffer if allocated during error
        throw; // Re-throw the exception
    }
}


// --- Second overload remains below, unchanged by this request ---

/**
 * @brief CUDA forward propagation for a 2nd to last block's attention class (incomplete attention)
 * Mirrors: void attention::forprop(std::vector<std::vector<float>> EVp, int& in, int& layers,
 * int& tokenCount, int& blockCount, int& n)
 * @param EVp EV vector from previous block (unused in current C++ logic, ignored here)
 * @param d_embedding input embedding dimension (in)
 * @param layers_mlp layers of MLPs (layers) - NOTE: Seems unused, mlp.hlayers.size() is used.
 * @param totalTokenCount total number of tokens processed so far (tokenCount in C++)
 * @param blockIdx which block is being processed (blockCount in C++)
 * @param contextWindowSize number of tokens for each attention head (n in C++)
 */
void attention::cuforprop(std::vector<std::vector<float>> EVp, int& d_embedding, int& layers_mlp, int& totalTokenCount,
    int& blockIdx, int& contextWindowSize)
{
    // Handle first block case by calling the other overload
    if (blockIdx == 0) {
        // Match C++: process min(totalTokenCount, contextWindowSize) for the first block
        int firstBlockTokenCount = std::min(totalTokenCount, contextWindowSize);
        cuforprop(d_embedding, layers_mlp, firstBlockTokenCount);
        return;
    }

    // Use constants defined in attention.hpp for clarity
    const int d = EMBEDDING;        // Embedding dimension
    const int h = MATHEIGHTS;       // Height dimension of K/Q, projection matrices
    // Number of EV rows to process, matching CPU logic for EV.sumRows(totalTokenCount) and EV.addToRows(totalTokenCount, ...)
    const int num_ev_rows_to_process = totalTokenCount;
    // 'count' is for K, Q, KdotQ dimensions for this specific block.
    // blockIdx is 1-based for the second block onwards.
    int start_idx_in_full_context = (blockIdx - 1) * contextWindowSize;
    int end_idx_in_full_context = std::min(totalTokenCount, blockIdx * contextWindowSize);
    const int count = std::max(0, end_idx_in_full_context - start_idx_in_full_context);

    // --- Basic Validation ---
    if (count <= 0) {
        std::cerr << "Warning: cuforprop(..., blockIdx=" << blockIdx << ") called with calculated count <= 0. Skipping computation." << std::endl;
        // Ensure outputs are zeroed or handled appropriately upstream if needed
        std::fill(EH.begin(), EH.end(), 0.0f);
        if (count == 0) {
             if (EV.mapped_data && EV.row > 0 && EV.col == d) { // Assuming index 0 for EV if count is 0
                std::fill_n(EV.mapped_data, EV.col, 0.0f);
            }
        }
        return;
    }

    // K (count x h), Q (count x h), KdotQ (count x count)
    // MH (d x h), MV (d x h)
    // K, Q, KdotQ are pre-allocated to CONTEXT_WIN based sizes.
    // 'count' is the current active number of tokens for this block's segment.
    if (K.row != CONTEXT_WIN || K.col != h ||
        Q.row != CONTEXT_WIN || Q.col != h ||
        KdotQ.row != CONTEXT_WIN || KdotQ.col != CONTEXT_WIN || // Assuming KdotQ is pre-allocated to CONTEXT_WIN x CONTEXT_WIN
        count > CONTEXT_WIN || // Active token count 'count' for this block segment must not exceed CONTEXT_WIN
        MH.row != d || MH.col != h || // MH is d x h
        MV.row != d || MV.col != h || // MV is d x h
        EH.size() != static_cast<size_t>(d) ||
        // EV validation: must be mapped, allocated for CONTEXT_WIN rows, col must be d. 'num_ev_rows_to_process' active rows.
        (!EV.mapped_data || EV.row != CONTEXT_WIN || EV.col != d) ||
        num_ev_rows_to_process > CONTEXT_WIN || // total active EV rows should not exceed CONTEXT_WIN
        hor.hlayers.empty() || ver.hlayers.empty() || hor.weights.empty() || ver.weights.empty())
    {
        // Provide more specific error messages if possible
        std::string error_msg = "Attention component dimension mismatch or uninitialized member in cuforprop(..., blockIdx=" + std::to_string(blockIdx) + ", count=" + std::to_string(count) + ").\n";
        if (K.row != CONTEXT_WIN || K.col != h) error_msg += " K dim: " + std::to_string(K.row) + "x" + std::to_string(K.col) + " (expected " + std::to_string(CONTEXT_WIN) + "x" + std::to_string(h) + ")\n";
        if (Q.row != CONTEXT_WIN || Q.col != h) error_msg += " Q dim: " + std::to_string(Q.row) + "x" + std::to_string(Q.col) + " (expected " + std::to_string(CONTEXT_WIN) + "x" + std::to_string(h) + ")\n";
        if (KdotQ.row != CONTEXT_WIN || KdotQ.col != CONTEXT_WIN) error_msg += " KdotQ dim: " + std::to_string(KdotQ.row) + "x" + std::to_string(KdotQ.col) + " (expected " + std::to_string(CONTEXT_WIN) + "x" + std::to_string(CONTEXT_WIN) + ")\n";
        if (count > CONTEXT_WIN) error_msg += " Active token count for block (count)=" + std::to_string(count) + " exceeds CONTEXT_WIN=" + std::to_string(CONTEXT_WIN) + "\n";
        if (!EV.mapped_data || EV.row != CONTEXT_WIN || EV.col != d) error_msg += " EV dim/map error. Mapped: " + std::string(EV.mapped_data ? "yes" : "no") + ", Dim: " + std::to_string(EV.row) + "x" + std::to_string(EV.col) + " (expected " + std::to_string(CONTEXT_WIN) + "x" + std::to_string(d) + ")\n";
        if (num_ev_rows_to_process > CONTEXT_WIN) error_msg += " num_ev_rows_to_process=" + std::to_string(num_ev_rows_to_process) + " exceeds CONTEXT_WIN=" + std::to_string(CONTEXT_WIN) + "\n";
        throw std::runtime_error(error_msg);
    }
    if (d != d_embedding) {
        throw std::runtime_error("Embedding dimension mismatch between definition (EMBEDDING) and argument (d_embedding).");
    }
    // Check MLP dimensions against embedding dim 'd'
    if (hor.hlayers[0].size() != static_cast<size_t>(d) || ver.hlayers[0].size() != static_cast<size_t>(d) ||
        hor.weights.back().row != d || ver.weights.back().row != d) { // Last weight matrix output dim (rows) must be d
         throw std::runtime_error("MLP input/output layer dimension mismatch with embedding dimension 'd'.");
    }

    // --- Device Memory Pointers ---
    float *d_K = nullptr, *d_Q = nullptr, *d_KdotQ = nullptr, *d_head = nullptr;
    float *d_row_sums = nullptr, *d_col_sums = nullptr;
    float *d_dh_accum = nullptr, *d_dv_accum = nullptr;
    float *d_MH_hxd = nullptr, *d_MV_hxd = nullptr; // Transposed: h x d
    float *d_dh = nullptr, *d_dv = nullptr;         // Projected result: 1 x d (size d)
    float *d_EH = nullptr;
    float *d_EV_processed_data = nullptr; // Buffer for the first 'num_ev_rows_to_process' rows of EV
    float *d_ver_accumulated_ev = nullptr; // Buffer for the sum of EV rows
    float *d_hor_inputs = nullptr, *d_ver_inputs = nullptr; // Inputs to MLP after residual add
    float *d_hor_output = nullptr, *d_ver_output = nullptr; // Final MLP outputs (pre-ReLU)
    float *d_relu_hor_output = nullptr, *d_relu_ver_output = nullptr; // After ReLU

    // MLP Intermediate Buffers (Merged Logic)
    float *d_mlp_bufferA_hor = nullptr, *d_mlp_bufferB_hor = nullptr; // Ping-pong for hor
    float *d_mlp_bufferA_ver = nullptr, *d_mlp_bufferB_ver = nullptr; // Ping-pong for ver
    float *d_mlp_pre_activation = nullptr; // Reused for pre-activation calculation
    float *d_mlp_weights = nullptr;        // Reused for weights (allocated/freed per layer per MLP)


    try {
        // --- Allocate Device Memory ---
        size_t k_bytes = static_cast<size_t>(count) * h * sizeof(float);
        size_t q_bytes = static_cast<size_t>(count) * h * sizeof(float);
        size_t kdotq_bytes = static_cast<size_t>(count) * count * sizeof(float);
        size_t head_bytes = static_cast<size_t>(count) * count * sizeof(float);
        size_t sums_bytes = static_cast<size_t>(count) * sizeof(float);
        size_t accum_bytes = static_cast<size_t>(h) * sizeof(float);
        size_t ev_processed_bytes = static_cast<size_t>(num_ev_rows_to_process) * d * sizeof(float);
        size_t proj_mat_bytes = static_cast<size_t>(h) * d * sizeof(float); // h x d
        size_t embed_bytes = static_cast<size_t>(d) * sizeof(float);        // d

        CUDA_CHECK(cudaMalloc(&d_K, k_bytes));
        CUDA_CHECK(cudaMalloc(&d_Q, q_bytes));
        CUDA_CHECK(cudaMalloc(&d_KdotQ, kdotq_bytes));
        CUDA_CHECK(cudaMalloc(&d_head, head_bytes));
        CUDA_CHECK(cudaMalloc(&d_row_sums, sums_bytes));
        CUDA_CHECK(cudaMalloc(&d_col_sums, sums_bytes));
        CUDA_CHECK(cudaMalloc(&d_dh_accum, accum_bytes));
        CUDA_CHECK(cudaMalloc(&d_dv_accum, accum_bytes));
        CUDA_CHECK(cudaMalloc(&d_MH_hxd, proj_mat_bytes));
        CUDA_CHECK(cudaMalloc(&d_MV_hxd, proj_mat_bytes));
        CUDA_CHECK(cudaMalloc(&d_dh, embed_bytes));
        CUDA_CHECK(cudaMalloc(&d_dv, embed_bytes));
        CUDA_CHECK(cudaMalloc(&d_EH, embed_bytes));
        CUDA_CHECK(cudaMalloc(&d_EV_processed_data, ev_processed_bytes));
        CUDA_CHECK(cudaMalloc(&d_ver_accumulated_ev, embed_bytes));
        CUDA_CHECK(cudaMalloc(&d_hor_inputs, embed_bytes)); // MLP input after add
        CUDA_CHECK(cudaMalloc(&d_ver_inputs, embed_bytes)); // MLP input after add
        CUDA_CHECK(cudaMalloc(&d_hor_output, embed_bytes)); // MLP final output (pre-ReLU)
        CUDA_CHECK(cudaMalloc(&d_ver_output, embed_bytes)); // MLP final output (pre-ReLU)
        CUDA_CHECK(cudaMalloc(&d_relu_hor_output, embed_bytes)); // After ReLU
        CUDA_CHECK(cudaMalloc(&d_relu_ver_output, embed_bytes)); // After ReLU

        // Allocate MLP Intermediate Buffers (Merged Logic)
        CUDA_CHECK(cudaMalloc(&d_mlp_bufferA_hor, embed_bytes));
        CUDA_CHECK(cudaMalloc(&d_mlp_bufferB_hor, embed_bytes));
        CUDA_CHECK(cudaMalloc(&d_mlp_bufferA_ver, embed_bytes));
        CUDA_CHECK(cudaMalloc(&d_mlp_bufferB_ver, embed_bytes));
        CUDA_CHECK(cudaMalloc(&d_mlp_pre_activation, embed_bytes));


        // Initialize accumulators to zero
        CUDA_CHECK(cudaMemset(d_dh_accum, 0, accum_bytes));
        CUDA_CHECK(cudaMemset(d_dv_accum, 0, accum_bytes));

        // --- Flatten Host Data & Copy H->D ---
        // ** Assuming K, Q, KdotQ members contain the 'count'-sized data for this block **
        if (!K.mapped_data || !Q.mapped_data || !KdotQ.mapped_data)
            throw std::runtime_error("K, Q, or KdotQ have null mapped_data for block processing.");

        CUDA_CHECK(cudaMemcpy(d_K, K.mapped_data, k_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_Q, Q.mapped_data, q_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_KdotQ, KdotQ.mapped_data, kdotq_bytes, cudaMemcpyHostToDevice));

        // Transpose MH (d x h) and MV (d x h) from host storage to h x d for device projection kernel
        // MH is d x h (MH.row = d, MH.col = h). flat_MH_hxd needs to represent h x d.
        std::vector<float> flat_MH_hxd, flat_MV_hxd;
        transposeMatToFlatVector(MH, flat_MH_hxd);
        transposeMatToFlatVector(MV, flat_MV_hxd);

        CUDA_CHECK(cudaMemcpy(d_MH_hxd, flat_MH_hxd.data(), proj_mat_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_MV_hxd, flat_MV_hxd.data(), proj_mat_bytes, cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMemcpy(d_EH, EH.data(), embed_bytes, cudaMemcpyHostToDevice));
        // Copy the first 'num_ev_rows_to_process' rows from EV mat to device buffer
        CUDA_CHECK(cudaMemcpy(d_EV_processed_data, EV.mapped_data, ev_processed_bytes, cudaMemcpyHostToDevice));


        // --- Kernel Launches ---
        const int threadsPerBlock = 256; // General purpose block size, adjust as needed

        // 1. Calculate Attention Weights: head = LOTA(KdotQ)
        int totalElementsLOTA = count * count;
        int blocksLOTA = (totalElementsLOTA + threadsPerBlock - 1) / threadsPerBlock;
        cuLOTA<<<blocksLOTA, threadsPerBlock>>>(d_KdotQ, d_head, count, count);
        CUDA_CHECK(cudaGetLastError());

        // 2. Compute Head Row/Column Sums (Masked)
        int blocksSums = (count + threadsPerBlock - 1) / threadsPerBlock;
        computeHeadSumsMaskedKernel<<<blocksSums, threadsPerBlock>>>(d_head, d_row_sums, d_col_sums, count, isSelfAttention);
        CUDA_CHECK(cudaGetLastError());

        // 3. Accumulate Weighted K and Q vectors
        int blocksAccum = (h + threadsPerBlock - 1) / threadsPerBlock;
        accumulateWeightedVectorsKernel<<<blocksAccum, threadsPerBlock>>>(d_row_sums, d_col_sums, d_K, d_Q, d_dh_accum, d_dv_accum, count, h);
        CUDA_CHECK(cudaGetLastError());

        // 4. Project accumulated vectors: dh = dot(dh_accum, MH), dv = dot(dv_accum, MV)
        int threadsPerBlockMatMul = 256;
        int blocksPerGridMatMul = (d + threadsPerBlockMatMul - 1) / threadsPerBlockMatMul;
        matrixMultiplyKernel<<<blocksPerGridMatMul, threadsPerBlockMatMul>>>(d_dh_accum, d_MH_hxd, d_dh, 1, h, d);
        CUDA_CHECK(cudaGetLastError());
        matrixMultiplyKernel<<<blocksPerGridMatMul, threadsPerBlockMatMul>>>(d_dv_accum, d_MV_hxd, d_dv, 1, h, d);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize()); // Ensure d_dh, d_dv are ready

        // 5. Prepare MLP inputs (Residual Add): hor_input = EH + dh, ver_input = EV_current + dv
        int blocksAdd = (d + threadsPerBlock - 1) / threadsPerBlock;
        vectorAddKernel<<<blocksAdd, threadsPerBlock>>>(d_EH, d_dh, d_hor_inputs, d);
        CUDA_CHECK(cudaGetLastError());

        // Accumulate first 'num_ev_rows_to_process' rows of EV for ver_input
        int blocksAccumEV = (d + threadsPerBlock - 1) / threadsPerBlock; // d is col_size
        accumulateEVRowsKernel<<<blocksAccumEV, threadsPerBlock>>>(d_EV_processed_data, d_ver_accumulated_ev, num_ev_rows_to_process, d);
        CUDA_CHECK(cudaGetLastError());
        vectorAddKernel<<<blocksAdd, threadsPerBlock>>>(d_ver_accumulated_ev, d_dv, d_ver_inputs, d);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize()); // Sync before MLP uses these inputs

        // --- 6. Run MLPs Forward (Merged Loop) ---
        try {
            // Initial input copies
            CUDA_CHECK(cudaMemcpy(d_mlp_bufferA_hor, d_hor_inputs, embed_bytes, cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_mlp_bufferA_ver, d_ver_inputs, embed_bytes, cudaMemcpyDeviceToDevice));

            // Pointers for ping-pong buffers
            float* d_current_in_hor = d_mlp_bufferA_hor;
            float* d_current_out_hor = d_mlp_bufferB_hor;
            float* d_current_in_ver = d_mlp_bufferA_ver;
            float* d_current_out_ver = d_mlp_bufferB_ver;

            int threadsPerBlockMLP = 256; // Can tune this

            // Loop through all weight matrices (including the final projection)
            size_t num_weight_matrices = hor.weights.size(); // Should be layers + 1
            if (num_weight_matrices != ver.weights.size()) {
                throw std::runtime_error("Horizontal and Vertical MLPs must have the same number of weight matrices.");
            }
             if (num_weight_matrices == 0) { // Handle case of MLP with no layers (only projection)
                 throw std::runtime_error("MLP weights cannot be empty.");
             }
            if (num_weight_matrices != hor.hlayers.size() + 1) {
                 throw std::runtime_error("MLP weights size mismatch with hlayers size.");
            }


            for (size_t layer_idx = 0; layer_idx < num_weight_matrices; ++layer_idx) {
                bool is_last_layer = (layer_idx == num_weight_matrices - 1);

                // --- Process Horizontal MLP Layer ---
                { // Scope for hor layer processing
                    int input_size = (layer_idx == 0) ? d : hor.hlayers[layer_idx-1].size();
                    // Output size is d for the last layer, otherwise size of the corresponding hlayer
                    int output_size = is_last_layer ? d : hor.hlayers[layer_idx].size();

                    if (input_size != d || output_size != d) {
                         throw std::runtime_error("MLP layers currently must have size equal to embedding dimension (d).");
                    }

                    const mat& current_weights_mat = hor.weights[layer_idx];
                    if (current_weights_mat.row != output_size || current_weights_mat.col != input_size) {
                        std::string err_msg = "MLP weight mat dimension mismatch (hor, block " + std::to_string(blockIdx) + "). Expected " +
                                              std::to_string(output_size) + "x" + std::to_string(input_size) +
                                              ", got " + std::to_string(current_weights_mat.row) + "x" +
                                              std::to_string(current_weights_mat.col) + " for layer " + std::to_string(layer_idx);
                        throw std::runtime_error(err_msg);
                    }
                    if (!current_weights_mat.mapped_data) {
                        throw std::runtime_error("MLP weight matrix (hor, block " + std::to_string(blockIdx) + ") has null mapped_data for layer " + std::to_string(layer_idx));
                    }

                    size_t weights_bytes = static_cast<size_t>(current_weights_mat.row) * current_weights_mat.col * sizeof(float);
                    CUDA_CHECK(cudaMalloc(&d_mlp_weights, weights_bytes));
                    CUDA_CHECK(cudaMemcpy(d_mlp_weights, current_weights_mat.mapped_data, weights_bytes, cudaMemcpyHostToDevice));

                    int blocksPerGridMLP = (output_size + threadsPerBlockMLP - 1) / threadsPerBlockMLP;

                    if (is_last_layer) {
                        // Final projection: input -> d_hor_output
                        layerForwardKernel<<<blocksPerGridMLP, threadsPerBlockMLP>>>(
                            d_current_in_hor, d_mlp_weights, d_hor_output, input_size, output_size);
                        CUDA_CHECK(cudaGetLastError());
                    } else {
                        // Hidden layer: input -> pre-activation -> sigmoid -> output
                        layerForwardKernel<<<blocksPerGridMLP, threadsPerBlockMLP>>>(
                            d_current_in_hor, d_mlp_weights, d_mlp_pre_activation, input_size, output_size);
                        CUDA_CHECK(cudaGetLastError());

                        cuSigmoid<<<blocksPerGridMLP, threadsPerBlockMLP>>>(d_mlp_pre_activation, d_current_out_hor, output_size);
                        CUDA_CHECK(cudaGetLastError());

                        // Swap hor buffers
                        float* temp = d_current_in_hor;
                        d_current_in_hor = d_current_out_hor;
                        d_current_out_hor = temp;
                    }
                    CUDA_CHECK(cudaFree(d_mlp_weights));
                    d_mlp_weights = nullptr; // Reset pointer
                } // End scope for hor layer processing

                // --- Process Vertical MLP Layer ---
                { // Scope for ver layer processing
                    int input_size = (layer_idx == 0) ? d : ver.hlayers[layer_idx-1].size();
                    int output_size = is_last_layer ? d : ver.hlayers[layer_idx].size();

                     if (input_size != d || output_size != d) {
                         throw std::runtime_error("MLP layers currently must have size equal to embedding dimension (d).");
                    }

                    const mat& current_weights_mat = ver.weights[layer_idx];
                    if (current_weights_mat.row != output_size || current_weights_mat.col != input_size) {
                         std::string err_msg = "MLP weight mat dimension mismatch (ver, block " + std::to_string(blockIdx) + "). Expected " +
                                              std::to_string(output_size) + "x" + std::to_string(input_size) +
                                              ", got " + std::to_string(current_weights_mat.row) + "x" +
                                              std::to_string(current_weights_mat.col) + " for layer " + std::to_string(layer_idx);
                        throw std::runtime_error(err_msg);
                    }
                    if (!current_weights_mat.mapped_data) {
                        throw std::runtime_error("MLP weight matrix (ver, block " + std::to_string(blockIdx) + ") has null mapped_data for layer " + std::to_string(layer_idx));
                    }

                    size_t weights_bytes = static_cast<size_t>(current_weights_mat.row) * current_weights_mat.col * sizeof(float);
                    CUDA_CHECK(cudaMalloc(&d_mlp_weights, weights_bytes));
                    CUDA_CHECK(cudaMemcpy(d_mlp_weights, current_weights_mat.mapped_data, weights_bytes, cudaMemcpyHostToDevice));

                    int blocksPerGridMLP = (output_size + threadsPerBlockMLP - 1) / threadsPerBlockMLP;

                    if (is_last_layer) {
                        // Final projection: input -> d_ver_output
                        layerForwardKernel<<<blocksPerGridMLP, threadsPerBlockMLP>>>(
                            d_current_in_ver, d_mlp_weights, d_ver_output, input_size, output_size);
                        CUDA_CHECK(cudaGetLastError());
                    } else {
                        // Hidden layer: input -> pre-activation -> sigmoid -> output
                        layerForwardKernel<<<blocksPerGridMLP, threadsPerBlockMLP>>>(
                            d_current_in_ver, d_mlp_weights, d_mlp_pre_activation, input_size, output_size);
                        CUDA_CHECK(cudaGetLastError());

                        cuSigmoid<<<blocksPerGridMLP, threadsPerBlockMLP>>>(d_mlp_pre_activation, d_current_out_ver, output_size);
                        CUDA_CHECK(cudaGetLastError());

                        // Swap ver buffers
                        float* temp = d_current_in_ver;
                        d_current_in_ver = d_current_out_ver;
                        d_current_out_ver = temp;
                    }
                    CUDA_CHECK(cudaFree(d_mlp_weights));
                    d_mlp_weights = nullptr; // Reset pointer
                } // End scope for ver layer processing

            } // End loop over layers

            CUDA_CHECK(cudaDeviceSynchronize()); // Sync after all MLP layers are processed

        }
        catch (const std::exception& e) {
            std::cerr << "Error during Merged MLP processing: " << e.what() << std::endl;
            cudaFree(d_mlp_weights); // Ensure cleanup if error occurs mid-layer
            // Free other MLP buffers in the main catch block outside this section
            throw;
        }

        // --- Free MLP Intermediate Buffers (No longer needed) ---
        CUDA_CHECK(cudaFree(d_mlp_bufferA_hor)); d_mlp_bufferA_hor = nullptr;
        CUDA_CHECK(cudaFree(d_mlp_bufferB_hor)); d_mlp_bufferB_hor = nullptr;
        CUDA_CHECK(cudaFree(d_mlp_bufferA_ver)); d_mlp_bufferA_ver = nullptr;
        CUDA_CHECK(cudaFree(d_mlp_bufferB_ver)); d_mlp_bufferB_ver = nullptr;
        CUDA_CHECK(cudaFree(d_mlp_pre_activation)); d_mlp_pre_activation = nullptr;
        // d_mlp_weights should be null here if loop completed normally


        // 7. Apply ReLU to MLP outputs (d_hor_output, d_ver_output now hold final MLP results)
        cuReLU<<<blocksAdd, threadsPerBlock>>>(d_hor_output, d_relu_hor_output, d);
        CUDA_CHECK(cudaGetLastError());
        cuReLU<<<blocksAdd, threadsPerBlock>>>(d_ver_output, d_relu_ver_output, d);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize()); // Sync before final add

        // 8. Final Residual Update: EH = EH + ReLU(hor_output), EV[count] = EV[count] + ReLU(ver_output)
        vectorAddKernel<<<blocksAdd, threadsPerBlock>>>(d_EH, d_relu_hor_output, d_EH, d); // Update d_EH in-place
        CUDA_CHECK(cudaGetLastError());

        // Add ReLU(ver_output) to the first 'num_ev_rows_to_process' rows of EV
        int blocksUpdateEV = (num_ev_rows_to_process + threadsPerBlock - 1) / threadsPerBlock;
        updateEVRowsKernel<<<blocksUpdateEV, threadsPerBlock>>>(d_EV_processed_data, d_relu_ver_output, num_ev_rows_to_process, d);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize()); // Sync before D->H copy

        // --- Copy Results D->H ---
        CUDA_CHECK(cudaMemcpy(EH.data(), d_EH, embed_bytes, cudaMemcpyDeviceToHost));
        // Copy updated EV rows back
        CUDA_CHECK(cudaMemcpy(EV.mapped_data, d_EV_processed_data, ev_processed_bytes, cudaMemcpyDeviceToHost));

        // --- Free Device Memory ---
        cudaFree(d_K); cudaFree(d_Q); cudaFree(d_KdotQ); cudaFree(d_head);
        cudaFree(d_row_sums); cudaFree(d_col_sums);
        cudaFree(d_dh_accum); cudaFree(d_dv_accum);
        cudaFree(d_MH_hxd); cudaFree(d_MV_hxd);
        cudaFree(d_dh); cudaFree(d_dv);
        cudaFree(d_EH);
        cudaFree(d_EV_processed_data);
        cudaFree(d_ver_accumulated_ev);
        cudaFree(d_hor_inputs); cudaFree(d_ver_inputs); // Free the initial MLP input buffers
        cudaFree(d_hor_output); cudaFree(d_ver_output); // Free the final MLP output buffers
        cudaFree(d_relu_hor_output); cudaFree(d_relu_ver_output); // Free the ReLU output buffers
        // MLP intermediate buffers were freed earlier

    }
    catch (const std::exception& e) {
        std::cerr << "CUDA Exception in attention cuforprop(..., blockIdx=" << blockIdx << "): " << e.what() << std::endl;
        // --- Cleanup on Error ---
        cudaFree(d_K); cudaFree(d_Q); cudaFree(d_KdotQ); cudaFree(d_head);
        cudaFree(d_row_sums); cudaFree(d_col_sums);
        cudaFree(d_dh_accum); cudaFree(d_dv_accum);
        cudaFree(d_MH_hxd); cudaFree(d_MV_hxd);
        cudaFree(d_dh); cudaFree(d_dv);
        cudaFree(d_EH);
        cudaFree(d_EV_processed_data);
        cudaFree(d_ver_accumulated_ev);
        cudaFree(d_hor_inputs); cudaFree(d_ver_inputs);
        cudaFree(d_hor_output); cudaFree(d_ver_output);
        cudaFree(d_relu_hor_output); cudaFree(d_relu_ver_output);
        // Free merged MLP buffers
        cudaFree(d_mlp_bufferA_hor);
        cudaFree(d_mlp_bufferB_hor);
        cudaFree(d_mlp_bufferA_ver);
        cudaFree(d_mlp_bufferB_ver);
        cudaFree(d_mlp_pre_activation);
        cudaFree(d_mlp_weights); // Free weights buffer if allocated during error
        throw; // Re-throw the exception
    }
}
