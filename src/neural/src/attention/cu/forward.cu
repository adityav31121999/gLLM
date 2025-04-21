
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


/**
 * @brief make transpose of a flatten matrix
 * @param[in] input matrix
 * @param[out] output_flat flattened transpose of input
 * @param[in] rows number of rows
 * @param[in] cols number of columns
 */
void transposeFlattenMatrix(const std::vector<std::vector<float>>& input, std::vector<float>& output_flat, int rows, int cols) {
    if (input.empty()) { // Allow empty input (e.g., if d or h is 0)
        output_flat.clear();
        return;
    }
     if (input[0].empty() && cols != 0) { // Rows exist but are empty, cols expected
        throw std::runtime_error("Transpose input has empty rows but non-zero columns expected.");
    }
     if (input[0].empty() && cols == 0) { // Empty rows and zero cols expected is valid
         output_flat.clear();
         return;
     }
    if (static_cast<int>(input.size()) != rows || static_cast<int>(input[0].size()) != cols) {
        throw std::runtime_error("Transpose dimension mismatch.");
    }
    output_flat.resize(static_cast<size_t>(cols) * rows); // Transposed dimensions
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            // output[j][i] = input[i][j] -> output_flat[j * rows + i]
            output_flat[static_cast<size_t>(j) * rows + i] = input[i][j];
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
        if (!EV.empty()) { // Check if EV has been initialized
             if (EV.size() <= static_cast<size_t>(n)) EV.resize(n + 1); // Ensure EV[n] exists
             if (EV[n].size() != static_cast<size_t>(d)) EV[n].resize(d, 0.0f); // Ensure correct size
             std::fill(EV[n].begin(), EV[n].end(), 0.0f);
        }
        return;
    }
    // Resize EV if necessary before accessing EV[n]
    if (EV.size() <= static_cast<size_t>(n)) EV.resize(n + 1);
    if (EV[n].size() != static_cast<size_t>(d)) EV[n].resize(d, 0.0f);

    if (K.size() != static_cast<size_t>(n) || (!K.empty() && K[0].size() != static_cast<size_t>(h)) ||
        Q.size() != static_cast<size_t>(n) || (!Q.empty() && Q[0].size() != static_cast<size_t>(h)) ||
        KdotQ.size() != static_cast<size_t>(n) || (!KdotQ.empty() && KdotQ[0].size() != static_cast<size_t>(n)) ||
        MH.a.size() != static_cast<size_t>(d) || (!MH.a.empty() && MH.a[0].size() != static_cast<size_t>(h)) ||
        MV.a.size() != static_cast<size_t>(d) || (!MV.a.empty() && MV.a[0].size() != static_cast<size_t>(h)) ||
        EH.size() != static_cast<size_t>(d) ||
        EV[n].size() != static_cast<size_t>(d) || // Checked after resize
        hor.hlayers.empty() || ver.hlayers.empty() || hor.weights.empty() || ver.weights.empty())
    {
        // Provide more specific error messages if possible
        throw std::runtime_error("Attention component dimension mismatch or uninitialized member in cuforprop(..., tokenCount).");
    }
    if (d != d_embedding) {
        throw std::runtime_error("Embedding dimension mismatch between definition (EMBEDDING) and argument (d_embedding).");
    }
    // Check MLP dimensions against embedding dim 'd'
    // Assuming MLP input/output and hidden layers match 'd' based on previous corrections
    if (hor.hlayers[0].size() != static_cast<size_t>(d) || ver.hlayers[0].size() != static_cast<size_t>(d) ||
        hor.weights.back().size() != static_cast<size_t>(d) || ver.weights.back().size() != static_cast<size_t>(d)) {
         throw std::runtime_error("MLP input/output layer dimension mismatch with embedding dimension 'd'.");
    }


    // --- Device Memory Pointers ---
    float *d_K = nullptr, *d_Q = nullptr, *d_KdotQ = nullptr, *d_head = nullptr;
    float *d_row_sums = nullptr, *d_col_sums = nullptr;
    float *d_dh_accum = nullptr, *d_dv_accum = nullptr;
    float *d_MH_hxd = nullptr, *d_MV_hxd = nullptr; // Transposed: h x d
    float *d_dh = nullptr, *d_dv = nullptr;         // Projected result: 1 x d (size d)
    float *d_EH = nullptr, *d_EV_current = nullptr; // Corresponds to EV[n]
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
        CUDA_CHECK(cudaMalloc(&d_EV_current, embed_bytes)); // For EV[n]
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
        std::vector<float> flat_K, flat_Q, flat_KdotQ, flat_MH_hxd, flat_MV_hxd;
        auto flatten = [](const auto& vec2d, auto& flat_vec) {
            if (vec2d.empty()) return;
            size_t rows = vec2d.size();
            size_t cols = vec2d[0].size();
            flat_vec.reserve(rows * cols);
            for (const auto& row : vec2d) {
                if (row.size() != cols) throw std::runtime_error("Inconsistent inner vector size during flattening.");
                flat_vec.insert(flat_vec.end(), row.begin(), row.end());
            }
        };

        flatten(K, flat_K);
        flatten(Q, flat_Q);
        flatten(KdotQ, flat_KdotQ);

        // Transpose MH (d x h) and MV (d x h) from host storage to h x d for device projection kernel
        transposeFlattenMatrix(MH.a, flat_MH_hxd, d, h);
        transposeFlattenMatrix(MV.a, flat_MV_hxd, d, h);

        CUDA_CHECK(cudaMemcpy(d_K, flat_K.data(), k_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_Q, flat_Q.data(), q_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_KdotQ, flat_KdotQ.data(), kdotq_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_MH_hxd, flat_MH_hxd.data(), proj_mat_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_MV_hxd, flat_MV_hxd.data(), proj_mat_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_EH, EH.data(), embed_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_EV_current, EV[n].data(), embed_bytes, cudaMemcpyHostToDevice)); // Copy EV[n]

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

        // 5. Prepare MLP inputs (Residual Add): hor_input = EH + dh, ver_input = EV_current + dv
        int blocksAdd = (d + threadsPerBlock - 1) / threadsPerBlock;
        vectorAddKernel<<<blocksAdd, threadsPerBlock>>>(d_EH, d_dh, d_hor_inputs, d);
        CUDA_CHECK(cudaGetLastError());
        vectorAddKernel<<<blocksAdd, threadsPerBlock>>>(d_EV_current, d_dv, d_ver_inputs, d);
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
                    if (hor.weights[layer_idx].size() != static_cast<size_t>(output_size)) {
                        throw std::runtime_error("MLP weight dimension mismatch (hor outer).");
                    }

                    // Flatten weights
                    std::vector<float> flat_weights_hor;
                    flat_weights_hor.reserve(static_cast<size_t>(input_size) * output_size);
                    for (int i = 0; i < output_size; ++i) {
                         if (hor.weights[layer_idx][i].size() != static_cast<size_t>(input_size)) {
                             throw std::runtime_error("MLP weight dimension mismatch (hor inner).");
                         }
                        flat_weights_hor.insert(flat_weights_hor.end(),
                                           hor.weights[layer_idx][i].begin(),
                                           hor.weights[layer_idx][i].end());
                    }
                    size_t weights_bytes = flat_weights_hor.size() * sizeof(float);

                    // Allocate/copy weights
                    CUDA_CHECK(cudaMalloc(&d_mlp_weights, weights_bytes));
                    CUDA_CHECK(cudaMemcpy(d_mlp_weights, flat_weights_hor.data(), weights_bytes, cudaMemcpyHostToDevice));

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
                     if (ver.weights[layer_idx].size() != static_cast<size_t>(output_size)) {
                        throw std::runtime_error("MLP weight dimension mismatch (ver outer).");
                    }

                    // Flatten weights
                    std::vector<float> flat_weights_ver;
                    flat_weights_ver.reserve(static_cast<size_t>(input_size) * output_size);
                    for (int i = 0; i < output_size; ++i) {
                         if (ver.weights[layer_idx][i].size() != static_cast<size_t>(input_size)) {
                             throw std::runtime_error("MLP weight dimension mismatch (ver inner).");
                         }
                        flat_weights_ver.insert(flat_weights_ver.end(),
                                           ver.weights[layer_idx][i].begin(),
                                           ver.weights[layer_idx][i].end());
                    }
                    size_t weights_bytes = flat_weights_ver.size() * sizeof(float);

                    // Allocate/copy weights
                    CUDA_CHECK(cudaMalloc(&d_mlp_weights, weights_bytes));
                    CUDA_CHECK(cudaMemcpy(d_mlp_weights, flat_weights_ver.data(), weights_bytes, cudaMemcpyHostToDevice));

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
        //    Using cuReLU(float* x, float* out, int size) from activations.cu
        cuReLU<<<blocksAdd, threadsPerBlock>>>(d_hor_output, d_relu_hor_output, d);
        CUDA_CHECK(cudaGetLastError());
        cuReLU<<<blocksAdd, threadsPerBlock>>>(d_ver_output, d_relu_ver_output, d);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize()); // Sync before final add

        // 8. Final Residual Update: EH = EH + ReLU(hor_output), EV[n] = EV[n] + ReLU(ver_output)
        vectorAddKernel<<<blocksAdd, threadsPerBlock>>>(d_EH, d_relu_hor_output, d_EH, d); // Update d_EH in-place
        CUDA_CHECK(cudaGetLastError());
        vectorAddKernel<<<blocksAdd, threadsPerBlock>>>(d_EV_current, d_relu_ver_output, d_EV_current, d); // Update d_EV_current in-place (EV[n])
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize()); // Sync before D->H copy

        // --- Copy Results D->H ---
        CUDA_CHECK(cudaMemcpy(EH.data(), d_EH, embed_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(EV[n].data(), d_EV_current, embed_bytes, cudaMemcpyDeviceToHost)); // Copy back to EV[n]

        // --- Free Device Memory ---
        cudaFree(d_K); cudaFree(d_Q); cudaFree(d_KdotQ); cudaFree(d_head);
        cudaFree(d_row_sums); cudaFree(d_col_sums);
        cudaFree(d_dh_accum); cudaFree(d_dv_accum);
        cudaFree(d_MH_hxd); cudaFree(d_MV_hxd);
        cudaFree(d_dh); cudaFree(d_dv);
        cudaFree(d_EH); cudaFree(d_EV_current);
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
        cudaFree(d_EH); cudaFree(d_EV_current);
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
void attention::cuforprop(std::vector<std::vector<float>> /* EVp */, int& d_embedding, int& layers_mlp, int& totalTokenCount,
    int& blockIdx, int& contextWindowSize)
{
    // Handle first block case by calling the other overload
    if (blockIdx == 0) {
        // Note: The first overload uses totalTokenCount directly as its 'currentTokenCount' parameter
        cuforprop(d_embedding, layers_mlp, totalTokenCount);
        return;
    }

    // Use constants defined in attention.hpp for clarity
    const int d = EMBEDDING;        // Embedding dimension
    const int h = MATHEIGHTS;       // Height dimension of K/Q, projection matrices
    // const int mlp_hidden_layers = layers_mlp; // Parameter seems redundant

    // Calculate the number of tokens relevant to this block, mirroring C++ logic exactly
    int count = std::abs(totalTokenCount - contextWindowSize * (blockIdx - 1));
    // Optional safety clamp if needed, based on how K/Q/KdotQ are populated for the block
    // count = std::min(count, contextWindowSize);

    // --- Basic Validation ---
    if (count <= 0) {
        std::cerr << "Warning: cuforprop(..., blockIdx=" << blockIdx << ") called with calculated count <= 0. Skipping computation." << std::endl;
        // Ensure outputs are zeroed or handled appropriately upstream if needed
        std::fill(EH.begin(), EH.end(), 0.0f);
        if (!EV.empty()) { // Check if EV has been initialized
             if (EV.size() <= static_cast<size_t>(count)) EV.resize(count + 1); // Ensure EV[count] exists
             if (EV[count].size() != static_cast<size_t>(d)) EV[count].resize(d, 0.0f); // Ensure correct size
             std::fill(EV[count].begin(), EV[count].end(), 0.0f);
        }
        return;
    }
    // Resize EV if necessary before accessing EV[count]
    if (EV.size() <= static_cast<size_t>(count)) EV.resize(count + 1);
    if (EV[count].size() != static_cast<size_t>(d)) EV[count].resize(d, 0.0f);

    // ** Assumption: K, Q, KdotQ members are pre-populated with data relevant to this block (size 'count') **
     if (K.size() != static_cast<size_t>(count) || (!K.empty() && K[0].size() != static_cast<size_t>(h)) ||
        Q.size() != static_cast<size_t>(count) || (!Q.empty() && Q[0].size() != static_cast<size_t>(h)) ||
        KdotQ.size() != static_cast<size_t>(count) || (!KdotQ.empty() && KdotQ[0].size() != static_cast<size_t>(count)) ||
        MH.a.size() != static_cast<size_t>(d) || (!MH.a.empty() && MH.a[0].size() != static_cast<size_t>(h)) ||
        MV.a.size() != static_cast<size_t>(d) || (!MV.a.empty() && MV.a[0].size() != static_cast<size_t>(h)) ||
        EH.size() != static_cast<size_t>(d) ||
        EV[count].size() != static_cast<size_t>(d) || // Checked after resize
        hor.hlayers.empty() || ver.hlayers.empty() || hor.weights.empty() || ver.weights.empty())
    {
        // Provide more specific error messages if possible
        throw std::runtime_error("Attention component dimension mismatch or uninitialized member in cuforprop(..., blockIdx). Check K/Q/KdotQ size matches calculated 'count'.");
    }
    if (d != d_embedding) {
        throw std::runtime_error("Embedding dimension mismatch between definition (EMBEDDING) and argument (d_embedding).");
    }
    // Check MLP dimensions against embedding dim 'd'
    if (hor.hlayers[0].size() != static_cast<size_t>(d) || ver.hlayers[0].size() != static_cast<size_t>(d) ||
        hor.weights.back().size() != static_cast<size_t>(d) || ver.weights.back().size() != static_cast<size_t>(d)) {
         throw std::runtime_error("MLP input/output layer dimension mismatch with embedding dimension 'd'.");
    }

    // --- Device Memory Pointers ---
    float *d_K = nullptr, *d_Q = nullptr, *d_KdotQ = nullptr, *d_head = nullptr;
    float *d_row_sums = nullptr, *d_col_sums = nullptr;
    float *d_dh_accum = nullptr, *d_dv_accum = nullptr;
    float *d_MH_hxd = nullptr, *d_MV_hxd = nullptr; // Transposed: h x d
    float *d_dh = nullptr, *d_dv = nullptr;         // Projected result: 1 x d (size d)
    float *d_EH = nullptr, *d_EV_current = nullptr; // Corresponds to EV[count]
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
        CUDA_CHECK(cudaMalloc(&d_EV_current, embed_bytes)); // For EV[count]
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
        std::vector<float> flat_K, flat_Q, flat_KdotQ, flat_MH_hxd, flat_MV_hxd;
        auto flatten = [](const auto& vec2d, auto& flat_vec) {
            if (vec2d.empty()) return;
            size_t rows = vec2d.size();
            size_t cols = vec2d[0].size();
            flat_vec.reserve(rows * cols);
            for (const auto& row : vec2d) {
                if (row.size() != cols) throw std::runtime_error("Inconsistent inner vector size during flattening.");
                flat_vec.insert(flat_vec.end(), row.begin(), row.end());
            }
        };

        flatten(K, flat_K);
        flatten(Q, flat_Q);
        flatten(KdotQ, flat_KdotQ);

        // Transpose MH (d x h) and MV (d x h) from host storage to h x d for device projection kernel
        transposeFlattenMatrix(MH.a, flat_MH_hxd, d, h);
        transposeFlattenMatrix(MV.a, flat_MV_hxd, d, h);

        CUDA_CHECK(cudaMemcpy(d_K, flat_K.data(), k_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_Q, flat_Q.data(), q_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_KdotQ, flat_KdotQ.data(), kdotq_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_MH_hxd, flat_MH_hxd.data(), proj_mat_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_MV_hxd, flat_MV_hxd.data(), proj_mat_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_EH, EH.data(), embed_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_EV_current, EV[count].data(), embed_bytes, cudaMemcpyHostToDevice)); // Copy EV[count]

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

        // 5. Prepare MLP inputs (Residual Add): hor_input = EH + dh, ver_input = EV_current + dv
        int blocksAdd = (d + threadsPerBlock - 1) / threadsPerBlock;
        vectorAddKernel<<<blocksAdd, threadsPerBlock>>>(d_EH, d_dh, d_hor_inputs, d);
        CUDA_CHECK(cudaGetLastError());
        vectorAddKernel<<<blocksAdd, threadsPerBlock>>>(d_EV_current, d_dv, d_ver_inputs, d);
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
                    if (hor.weights[layer_idx].size() != static_cast<size_t>(output_size)) {
                        throw std::runtime_error("MLP weight dimension mismatch (hor outer).");
                    }

                    // Flatten weights
                    std::vector<float> flat_weights_hor;
                    flat_weights_hor.reserve(static_cast<size_t>(input_size) * output_size);
                    for (int i = 0; i < output_size; ++i) {
                         if (hor.weights[layer_idx][i].size() != static_cast<size_t>(input_size)) {
                             throw std::runtime_error("MLP weight dimension mismatch (hor inner).");
                         }
                        flat_weights_hor.insert(flat_weights_hor.end(),
                                           hor.weights[layer_idx][i].begin(),
                                           hor.weights[layer_idx][i].end());
                    }
                    size_t weights_bytes = flat_weights_hor.size() * sizeof(float);

                    // Allocate/copy weights
                    CUDA_CHECK(cudaMalloc(&d_mlp_weights, weights_bytes));
                    CUDA_CHECK(cudaMemcpy(d_mlp_weights, flat_weights_hor.data(), weights_bytes, cudaMemcpyHostToDevice));

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
                     if (ver.weights[layer_idx].size() != static_cast<size_t>(output_size)) {
                        throw std::runtime_error("MLP weight dimension mismatch (ver outer).");
                    }

                    // Flatten weights
                    std::vector<float> flat_weights_ver;
                    flat_weights_ver.reserve(static_cast<size_t>(input_size) * output_size);
                    for (int i = 0; i < output_size; ++i) {
                         if (ver.weights[layer_idx][i].size() != static_cast<size_t>(input_size)) {
                             throw std::runtime_error("MLP weight dimension mismatch (ver inner).");
                         }
                        flat_weights_ver.insert(flat_weights_ver.end(),
                                           ver.weights[layer_idx][i].begin(),
                                           ver.weights[layer_idx][i].end());
                    }
                    size_t weights_bytes = flat_weights_ver.size() * sizeof(float);

                    // Allocate/copy weights
                    CUDA_CHECK(cudaMalloc(&d_mlp_weights, weights_bytes));
                    CUDA_CHECK(cudaMemcpy(d_mlp_weights, flat_weights_ver.data(), weights_bytes, cudaMemcpyHostToDevice));

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
        vectorAddKernel<<<blocksAdd, threadsPerBlock>>>(d_EV_current, d_relu_ver_output, d_EV_current, d); // Update d_EV_current in-place (EV[count])
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize()); // Sync before D->H copy

        // --- Copy Results D->H ---
        CUDA_CHECK(cudaMemcpy(EH.data(), d_EH, embed_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(EV[count].data(), d_EV_current, embed_bytes, cudaMemcpyDeviceToHost)); // Copy back to EV[count]

        // --- Free Device Memory ---
        cudaFree(d_K); cudaFree(d_Q); cudaFree(d_KdotQ); cudaFree(d_head);
        cudaFree(d_row_sums); cudaFree(d_col_sums);
        cudaFree(d_dh_accum); cudaFree(d_dv_accum);
        cudaFree(d_MH_hxd); cudaFree(d_MV_hxd);
        cudaFree(d_dh); cudaFree(d_dv);
        cudaFree(d_EH); cudaFree(d_EV_current);
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
        cudaFree(d_EH); cudaFree(d_EV_current);
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
