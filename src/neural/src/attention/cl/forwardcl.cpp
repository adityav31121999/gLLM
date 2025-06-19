#ifdef USE_OPENCL
#if defined(_WIN64)
    #define CL_HPP_ENABLE_EXCEPTIONS
    #define CL_HPP_TARGET_OPENCL_VERSION 300
    // For Windows, use the older/common cl.hpp
    #include <CL/cl.hpp>
#elif defined(__linux__)
    #define CL_HPP_TARGET_OPENCL_VERSION 220
    #include <CL/opencl.hpp>
#endif
#include "include/attention.hpp" // Includes mlp.hpp and maths.hpp indirectly or directly
#include <maths.hpp>
#include <vector>
#include <string>
#include <stdexcept>
#include <iostream>
#include <numeric>      // For std::accumulate if needed for host-side reduction
#include <algorithm>    // For std::max, std::abs, std::min
#include <cmath>        // For std::abs used in count calculation
#include <map>          // For kernel map

/**
 * @brief OpenCL forward propagation for first block's attention class (incomplete attention)
 * @param d_embedding embedding dimension (in)
 * @param layers_mlp layers of hidden weights in mlp (layers) - NOTE: Unused, mlp object sizes used.
 * @param currentTokenCount token count for this attention head (tokenCount)
 */
void attention::clforprop(int& d_embedding, int& layers_mlp, int& currentTokenCount)
{
    // Use constants defined in attention.hpp for clarity
    const int d = EMBEDDING;        // Embedding dimension
    const int h = MATHEIGHTS;       // Height dimension of K/Q, projection matrices
    const int n = currentTokenCount;// Number of tokens
    // Number of EV rows to process, matching CPU logic for EV.sumRows(totalTokenCount) and EV.addToRows(totalTokenCount, ...)
    const int num_ev_rows_to_process = currentTokenCount;

    // --- Basic Validation ---
    if (n <= 0) {
        std::cerr << "Warning: clforprop(..., tokenCount=" << n << ") called with tokenCount <= 0. Skipping computation." << std::endl;
        std::fill(EH.begin(), EH.end(), 0.0f);
        // If EV is a mat, it should be pre-allocated. We might zero the relevant row if n=0.
        if (n == 0) { // If tokenCount is exactly 0, implies initializing state for the 0-th token.
            if (EV.mapped_data && EV.row > 0 && EV.col == d) {
                std::fill_n(EV.mapped_data, EV.col, 0.0f); // Zero out the 0-th row
            }
        }
        return;
    }
    // EV mat validation: must be large enough for current token n
    if (EV.row < n || EV.col != d) { // If EV has fewer than n rows, it's an error.
        throw std::runtime_error("EV matrix not properly sized for current tokenCount n. Expected rows >= " + std::to_string(n) + " and cols = " + std::to_string(d) +
                                 ", but got rows = " + std::to_string(EV.row) + " and cols = " + std::to_string(EV.col));
    }

    if (K.row != n || K.col != h || // K and Q are n x h (tokenCount x MATHEIGHTS)
        Q.row != n || Q.col != h || // K and Q are n x h (tokenCount x MATHEIGHTS)
        KdotQ.row != n || KdotQ.col != n ||
        MH.row != d || MH.col != h || // MH is mat(d,h)
        MV.row != d || MV.col != h || // MV is mat(d,h)
        EH.size() != static_cast<size_t>(d) ||
        // EV check for 'n' rows is done above
        hor.hlayers.empty() || ver.hlayers.empty() || hor.weights.empty() || ver.weights.empty())
    {
        throw std::runtime_error("Attention component dimension mismatch or uninitialized member in clforprop(..., tokenCount).");
    }

    if (d != d_embedding) {
        throw std::runtime_error("Embedding dimension mismatch between definition (EMBEDDING) and argument (d_embedding).");
    }
    if (hor.hlayers[0].size() != static_cast<size_t>(d) || ver.hlayers[0].size() != static_cast<size_t>(d) ||
        hor.weights.back().row != static_cast<size_t>(d) || ver.weights.back().row != static_cast<size_t>(d)) { // Assuming weights are d x d, so row count is d
         throw std::runtime_error("MLP input/output layer dimension mismatch with embedding dimension 'd'.");
    }

    try {
        cl_int cl_err; // For OpenCL error codes
        OpenCLContext& context_obj = this->clcontext; // Use the member reference
        cl::Context context = context_obj.context;
        cl::CommandQueue queue = context_obj.queue;
        // Use std::map<std::string, cl::Kernel> kernels = context_obj.kernels;
        // Ensure kernel names used below exist in the map keys during OpenCLContext construction

        // --- Allocate Device Memory ---
        // Use cl::Buffer and C++ bindings. Exceptions will be thrown on error.
        size_t k_bytes = static_cast<size_t>(n) * h * sizeof(float);
        size_t q_bytes = static_cast<size_t>(n) * h * sizeof(float);
        size_t kdotq_bytes = static_cast<size_t>(n) * n * sizeof(float);
        size_t head_bytes = static_cast<size_t>(n) * n * sizeof(float);
        size_t sums_bytes = static_cast<size_t>(n) * sizeof(float);
        size_t accum_bytes = static_cast<size_t>(h) * sizeof(float);
        size_t proj_mat_bytes = static_cast<size_t>(d) * h * sizeof(float); // MH/MV are d x h
        // Size for the first 'n' rows of EV
        size_t ev_processed_bytes = static_cast<size_t>(n) * d * sizeof(float);
        size_t embed_bytes = static_cast<size_t>(d) * sizeof(float);        // d

        cl::Buffer d_K(context, CL_MEM_READ_ONLY, k_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        cl::Buffer d_Q(context, CL_MEM_READ_ONLY, q_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        cl::Buffer d_KdotQ(context, CL_MEM_READ_ONLY, kdotq_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        cl::Buffer d_head(context, CL_MEM_READ_WRITE, head_bytes, nullptr, &cl_err); CL_CHECK(cl_err); // LOTA output, read by sums
        cl::Buffer d_row_sums(context, CL_MEM_READ_WRITE, sums_bytes, nullptr, &cl_err); CL_CHECK(cl_err); // Written by sums, read by accum
        cl::Buffer d_col_sums(context, CL_MEM_READ_WRITE, sums_bytes, nullptr, &cl_err); CL_CHECK(cl_err); // Written by sums, read by accum
        cl::Buffer d_dh_accum(context, CL_MEM_READ_WRITE, accum_bytes, nullptr, &cl_err); CL_CHECK(cl_err); // Written by accum (atomic), read by proj
        cl::Buffer d_dv_accum(context, CL_MEM_READ_WRITE, accum_bytes, nullptr, &cl_err); CL_CHECK(cl_err); // Written by accum (atomic), read by proj
        cl::Buffer d_MH_hxd(context, CL_MEM_READ_ONLY, proj_mat_bytes, nullptr, &cl_err); CL_CHECK(cl_err); // Transposed MH
        cl::Buffer d_MV_hxd(context, CL_MEM_READ_ONLY, proj_mat_bytes, nullptr, &cl_err); CL_CHECK(cl_err); // Transposed MV
        cl::Buffer d_dh (context, CL_MEM_READ_WRITE, embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err); // Written by proj, read by add
        cl::Buffer d_dv(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err); // Written by proj, read by add
        cl::Buffer d_EH(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err); // Read by add, written by final add
        cl::Buffer d_EV_processed_data(context, CL_MEM_READ_WRITE, ev_processed_bytes, nullptr, &cl_err); CL_CHECK(cl_err); // For first 'n' rows of EV
        cl::Buffer d_ver_accumulated_ev(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err); // Sum of EV rows
        cl::Buffer d_hor_inputs(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err); // Written by add, read by MLP
        cl::Buffer d_ver_inputs(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err); // Written by add, read by MLP
        cl::Buffer d_hor_output(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err); // Written by MLP, read by ReLU
        cl::Buffer d_ver_output(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err); // Written by MLP, read by ReLU
        cl::Buffer d_relu_hor_output(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err); // Written by ReLU, read by final add
        cl::Buffer d_relu_ver_output(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err); // Written by ReLU, read by final add

        // Allocate MLP Intermediate Buffers
        cl::Buffer d_mlp_bufferA_hor(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        cl::Buffer d_mlp_bufferB_hor(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        cl::Buffer d_mlp_bufferA_ver(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        cl::Buffer d_mlp_bufferB_ver(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        cl::Buffer d_mlp_pre_activation(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err); // Temp buffer for MLP layer output before activation
        // d_mlp_weights buffer will be created inside the loop

        // Initialize accumulators to zero
        float zero = 0.0f;
        CL_CHECK(queue.enqueueFillBuffer(d_dh_accum, zero, 0, accum_bytes));
        CL_CHECK(queue.enqueueFillBuffer(d_dv_accum, zero, 0, accum_bytes));

        // --- Flatten Host Data & Copy H->D ---
        // For mat objects, use mapped_data directly
        CL_CHECK(queue.enqueueWriteBuffer(d_K, CL_TRUE, 0, k_bytes, K.mapped_data));
        CL_CHECK(queue.enqueueWriteBuffer(d_Q, CL_TRUE, 0, q_bytes, Q.mapped_data));
        CL_CHECK(queue.enqueueWriteBuffer(d_KdotQ, CL_TRUE, 0, kdotq_bytes, KdotQ.mapped_data));
        CL_CHECK(queue.enqueueWriteBuffer(d_MH_hxd, CL_TRUE, 0, proj_mat_bytes, MH.mapped_data)); // MH is mat(d,h)
        CL_CHECK(queue.enqueueWriteBuffer(d_MV_hxd, CL_TRUE, 0, proj_mat_bytes, MV.mapped_data)); // MV is mat(d,h)
        CL_CHECK(queue.enqueueWriteBuffer(d_EH, CL_TRUE, 0, embed_bytes, EH.data()));
        // Copy the first 'n' rows of EV to device
        CL_CHECK(queue.enqueueWriteBuffer(d_EV_processed_data, CL_TRUE, 0, ev_processed_bytes, EV.mapped_data));

        // --- Kernel Launches ---
        const size_t local_work_size_1d = 256; // General purpose block size

        // 1. Calculate Attention Weights: head = LOTA(KdotQ)
        // Use clLOTA2d as KdotQ is 2D
        cl::Kernel lota_kernel = context_obj.kernels.at("clLOTA2dmasking");
        size_t totalElementsLOTA = static_cast<size_t>(CONTEXT_WIN) * CONTEXT_WIN;
        size_t global_work_size_lota[1] = { (totalElementsLOTA + local_work_size_1d - 1) / local_work_size_1d * local_work_size_1d };
        size_t local_work_size_lota[1] = { local_work_size_1d };
        if (totalElementsLOTA > 0) {
            size_t global_lota_raw = totalElementsLOTA;
            size_t local_lota_clamped = (std::min)(global_lota_raw, local_work_size_1d);
            if (local_lota_clamped == 0) local_lota_clamped = 1; 
            size_t global_lota_padded = ((global_lota_raw + local_lota_clamped - 1) / local_lota_clamped) * local_lota_clamped;

            cl::NDRange global_lota(global_lota_padded);
            cl::NDRange local_lota(local_lota_clamped);

            CL_CHECK(lota_kernel.setArg(0, d_KdotQ));
            CL_CHECK(lota_kernel.setArg(1, d_head));
            CL_CHECK(lota_kernel.setArg(2, CONTEXT_WIN)); // rows
            CL_CHECK(lota_kernel.setArg(3, CONTEXT_WIN)); // cols
            CL_CHECK(lota_kernel.setArg(4, n)); // cols
            cl_int cl_att_is_self_lota = this->isSelfAttention ? 1 : 0;
            CL_CHECK(lota_kernel.setArg(5, cl_att_is_self_lota));
            CL_CHECK(queue.enqueueNDRangeKernel(lota_kernel, cl::NullRange, global_lota, local_lota));

            // 2. Compute Head Row/Column Sums (Masked)
            cl::Kernel sums_kernel = context_obj.kernels.at("computeHeadSumsMaskedKernel"); // Use the new kernel
            size_t global_sums_raw = static_cast<size_t>(n);
            size_t global_sums_padded = ((global_sums_raw + local_work_size_1d - 1) / local_work_size_1d) * local_work_size_1d;
            cl::NDRange global_sums(global_sums_padded);
            cl::NDRange local_sums(local_work_size_1d);
            cl_int cl_isSelfAttention = isSelfAttention; // Convert bool to cl_int
            CL_CHECK(sums_kernel.setArg(0, d_head));
            CL_CHECK(sums_kernel.setArg(1, d_row_sums));
            CL_CHECK(sums_kernel.setArg(2, d_col_sums));
            CL_CHECK(sums_kernel.setArg(3, n));
            CL_CHECK(sums_kernel.setArg(4, cl_isSelfAttention));
            CL_CHECK(queue.enqueueNDRangeKernel(sums_kernel, cl::NullRange, global_sums, local_sums));

            // 3. Accumulate Weighted K and Q vectors
            // NOTE: Requires float atomics support. Launching one work-item per token.
            cl::Kernel accum_kernel = context_obj.kernels.at("accumulateWeightedVectorsKernel"); // Use the new kernel
            size_t global_accum_raw = static_cast<size_t>(n);
            size_t global_accum_padded = ((global_accum_raw + local_work_size_1d - 1) / local_work_size_1d) * local_work_size_1d;
            cl::NDRange global_accum(global_accum_padded);
            cl::NDRange local_accum(local_work_size_1d);
            CL_CHECK(accum_kernel.setArg(0, d_row_sums));
            CL_CHECK(accum_kernel.setArg(1, d_col_sums));
            CL_CHECK(accum_kernel.setArg(2, d_K));
            CL_CHECK(accum_kernel.setArg(3, d_Q));
            CL_CHECK(accum_kernel.setArg(4, d_dh_accum));
            CL_CHECK(accum_kernel.setArg(5, d_dv_accum));
            CL_CHECK(accum_kernel.setArg(6, n));
            CL_CHECK(accum_kernel.setArg(7, h));
            CL_CHECK(queue.enqueueNDRangeKernel(accum_kernel, cl::NullRange, global_accum, local_accum));

            // 4. Project accumulated vectors: dh = dot(dh_accum, MH), dv = dot(dv_accum, MV)
            // Use kernelLayerForward (matrix-vector): output[j] = sum(input[i] * weights[j*input_size + i])
            // Input: d_dh_accum (size h), Weights: d_MH_hxd (size d x h), Output: d_dh (size d)
            cl::Kernel proj_kernel = context_obj.kernels.at("kernelLayerForward"); // Use the new kernel
            size_t global_proj_raw = static_cast<size_t>(d);
            size_t global_proj_padded = ((global_proj_raw + local_work_size_1d - 1) / local_work_size_1d) * local_work_size_1d;
            cl::NDRange global_proj(global_proj_padded);
            cl::NDRange local_proj(local_work_size_1d); // Or cl::NullRange

            // dh = dh_accum * MH_hxd (conceptually) -> kernelLayerForward(d_dh_accum, d_MH_hxd, d_dh, h, d)
            CL_CHECK(proj_kernel.setArg(0, d_dh_accum)); // input vector (size h)
            CL_CHECK(proj_kernel.setArg(1, d_MH_hxd));   // weight matrix (d rows, h cols)
            CL_CHECK(proj_kernel.setArg(2, d_dh));       // output vector (size d)
            CL_CHECK(proj_kernel.setArg(3, h));          // input_size
            CL_CHECK(proj_kernel.setArg(4, d));          // output_size
            CL_CHECK(queue.enqueueNDRangeKernel(proj_kernel, cl::NullRange, global_proj, local_proj));

            // dv = dv_accum * MV_hxd (conceptually) -> kernelLayerForward(d_dv_accum, d_MV_hxd, d_dv, h, d)
            CL_CHECK(proj_kernel.setArg(0, d_dv_accum)); // input vector (size h)
            CL_CHECK(proj_kernel.setArg(1, d_MV_hxd));   // weight matrix (d rows, h cols)
            CL_CHECK(proj_kernel.setArg(2, d_dv));       // output vector (size d)
            // Args 3 (h) and 4 (d) are already set
            CL_CHECK(queue.enqueueNDRangeKernel(proj_kernel, cl::NullRange, global_proj, local_proj));

            // 5. Prepare MLP inputs (Residual Add): hor_input = EH + dh, ver_input = EV_current + dv
            cl::Kernel add_kernel = context_obj.kernels.at("vectorAddKernel"); // Use the new kernel
            size_t global_add_raw = static_cast<size_t>(d);
            size_t global_add_padded = ((global_add_raw + local_work_size_1d - 1) / local_work_size_1d) * local_work_size_1d;
            cl::NDRange global_add(global_add_padded);
            cl::NDRange local_add(local_work_size_1d); // Or cl::NullRange

            // hor_input = EH + dh
            CL_CHECK(add_kernel.setArg(0, d_EH));
            CL_CHECK(add_kernel.setArg(1, d_dh));
            CL_CHECK(add_kernel.setArg(2, d_hor_inputs));
            CL_CHECK(add_kernel.setArg(3, d));
            CL_CHECK(queue.enqueueNDRangeKernel(add_kernel, cl::NullRange, global_add, local_add));

            // Accumulate first 'n' rows of EV for ver_input
            cl::Kernel accum_ev_kernel = context_obj.kernels.at("accumulateEVRowsKernelCL");
            // global_add is for size 'd' (col_size), local_add is local_work_size_1d
            CL_CHECK(accum_ev_kernel.setArg(0, d_EV_processed_data));
            CL_CHECK(accum_ev_kernel.setArg(1, d_ver_accumulated_ev));
            CL_CHECK(accum_ev_kernel.setArg(2, n)); // num_rows
            CL_CHECK(accum_ev_kernel.setArg(3, d)); // col_size
            CL_CHECK(queue.enqueueNDRangeKernel(accum_ev_kernel, cl::NullRange, global_add, local_add));

            // ver_input = d_ver_accumulated_ev + dv
            CL_CHECK(add_kernel.setArg(0, d_ver_accumulated_ev));
            CL_CHECK(add_kernel.setArg(1, d_dv));
            CL_CHECK(add_kernel.setArg(2, d_ver_inputs));
            // Arg 3 (d) is already set
            CL_CHECK(queue.enqueueNDRangeKernel(add_kernel, cl::NullRange, global_add, local_add));



            CL_CHECK(queue.finish()); // Sync before MLP uses these inputs

            // --- 6. Run MLPs Forward ---
            // try { // Inner try-catch for MLP specific errors can be removed if CL_CHECK handles all OpenCL errors
                // Get kernels
                cl::Kernel mlp_fwd_kernel = context_obj.kernels.at("kernelLayerForward");
                cl::Kernel sigmoid_kernel = context_obj.kernels.at("clSigmoid1d"); // Assuming 1D version

                // Initial input copies using enqueueCopyBuffer
                CL_CHECK(queue.enqueueCopyBuffer(d_hor_inputs, d_mlp_bufferA_hor, 0, 0, embed_bytes));
                CL_CHECK(queue.enqueueCopyBuffer(d_ver_inputs, d_mlp_bufferA_ver, 0, 0, embed_bytes));

                // References for ping-pong buffers
                cl::Buffer& current_in_hor = d_mlp_bufferA_hor;
                cl::Buffer& current_out_hor = d_mlp_bufferB_hor;
                cl::Buffer& current_in_ver = d_mlp_bufferA_ver;
                cl::Buffer& current_out_ver = d_mlp_bufferB_ver;

                size_t num_weight_matrices = hor.weights.size();
                if (num_weight_matrices != ver.weights.size()) {
                    throw std::runtime_error("Horizontal and Vertical MLPs must have the same number of weight matrices.");
                }
                if (num_weight_matrices == 0) {
                    throw std::runtime_error("MLP weights cannot be empty.");
                }
                if (num_weight_matrices != hor.hlayers.size() + 1) {
                    throw std::runtime_error("MLP weights size mismatch with hlayers size.");
                }

                size_t global_mlp_raw = static_cast<size_t>(d);
                size_t global_mlp_padded = ((global_mlp_raw + local_work_size_1d - 1) / local_work_size_1d) * local_work_size_1d;
                cl::NDRange global_mlp(global_mlp_padded);
                cl::NDRange local_mlp(local_work_size_1d); // Or cl::NullRange

                for (size_t layer_idx = 0; layer_idx < num_weight_matrices; ++layer_idx) {
                    bool is_last_layer = (layer_idx == num_weight_matrices - 1);
                    int input_size = d; // Assuming all layers have size d
                    int output_size = d; // Assuming MLP hidden and output layers are size d

                    // --- Process Horizontal MLP Layer ---
                    {
                        mat& current_weights_mat = hor.weights[layer_idx];
                        if (current_weights_mat.row != output_size || current_weights_mat.col != input_size) {
                            throw std::runtime_error("MLP weight dimension mismatch (hor). Expected " +
                                                     std::to_string(output_size) + "x" + std::to_string(input_size) +
                                                     ", got " + std::to_string(current_weights_mat.row) + "x" + std::to_string(current_weights_mat.col));
                        }
                        size_t weights_bytes = static_cast<size_t>(current_weights_mat.row) * current_weights_mat.col * sizeof(float);


                        // Create and write weights buffer for this layer
                        cl::Buffer d_mlp_weights(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, weights_bytes, current_weights_mat.mapped_data, &cl_err); CL_CHECK(cl_err);

                        cl::Buffer& target_output_buffer = is_last_layer ? d_hor_output : d_mlp_pre_activation;

                        CL_CHECK(mlp_fwd_kernel.setArg(0, current_in_hor));
                        CL_CHECK(mlp_fwd_kernel.setArg(1, d_mlp_weights));
                        CL_CHECK(mlp_fwd_kernel.setArg(2, target_output_buffer));
                        CL_CHECK(mlp_fwd_kernel.setArg(3, input_size));
                        CL_CHECK(mlp_fwd_kernel.setArg(4, output_size));
                        CL_CHECK(queue.enqueueNDRangeKernel(mlp_fwd_kernel, cl::NullRange, global_mlp, local_mlp));

                        if (!is_last_layer) {
                            CL_CHECK(sigmoid_kernel.setArg(0, d_mlp_pre_activation));
                            CL_CHECK(sigmoid_kernel.setArg(1, current_out_hor));
                            CL_CHECK(sigmoid_kernel.setArg(2, output_size));
                            CL_CHECK(queue.enqueueNDRangeKernel(sigmoid_kernel, cl::NullRange, global_mlp, local_mlp));

                            // Swap hor buffers (references)
                            std::swap(current_in_hor, current_out_hor);
                        }
                        // d_mlp_weights goes out of scope and is released
                    }

                    // --- Process Vertical MLP Layer ---
                    {
                        mat& current_weights_mat = ver.weights[layer_idx];
                        if (current_weights_mat.row != output_size || current_weights_mat.col != input_size) {
                           throw std::runtime_error("MLP weight dimension mismatch (ver). Expected " +
                                                     std::to_string(output_size) + "x" + std::to_string(input_size) +
                                                     ", got " + std::to_string(current_weights_mat.row) + "x" + std::to_string(current_weights_mat.col));
                        }
                        size_t weights_bytes = static_cast<size_t>(current_weights_mat.row) * current_weights_mat.col * sizeof(float);


                        cl::Buffer d_mlp_weights(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, weights_bytes, current_weights_mat.mapped_data, &cl_err); CL_CHECK(cl_err);

                        cl::Buffer& target_output_buffer = is_last_layer ? d_ver_output : d_mlp_pre_activation;

                        CL_CHECK(mlp_fwd_kernel.setArg(0, current_in_ver));
                        CL_CHECK(mlp_fwd_kernel.setArg(1, d_mlp_weights));
                        CL_CHECK(mlp_fwd_kernel.setArg(2, target_output_buffer));
                        // Args 3 (input_size) and 4 (output_size) are already set
                        CL_CHECK(queue.enqueueNDRangeKernel(mlp_fwd_kernel, cl::NullRange, global_mlp, local_mlp));

                        if (!is_last_layer) {
                            CL_CHECK(sigmoid_kernel.setArg(0, d_mlp_pre_activation));
                            CL_CHECK(sigmoid_kernel.setArg(1, current_out_ver));
                            // Arg 2 (output_size) is already set
                            CL_CHECK(queue.enqueueNDRangeKernel(sigmoid_kernel, cl::NullRange, global_mlp, local_mlp));

                            // Swap ver buffers (references)
                            std::swap(current_in_ver, current_out_ver);
                        }
                        // d_mlp_weights goes out of scope and is released
                    }
                } // End loop over layers
                CL_CHECK(queue.finish()); // Sync after all MLP layers
            // } // End of inner try-catch for MLP

            // 7. Apply ReLU to MLP outputs
            cl::Kernel relu_kernel = context_obj.kernels.at("clReLU1d");
            // ReLU(hor_output) -> d_relu_hor_output
            CL_CHECK(relu_kernel.setArg(0, d_hor_output));
            CL_CHECK(relu_kernel.setArg(1, d_relu_hor_output));
            CL_CHECK(relu_kernel.setArg(2, d));
            CL_CHECK(queue.enqueueNDRangeKernel(relu_kernel, cl::NullRange, global_add, local_add));

            // ReLU(ver_output) -> d_relu_ver_output
            CL_CHECK(relu_kernel.setArg(0, d_ver_output));
            CL_CHECK(relu_kernel.setArg(1, d_relu_ver_output));
            // Arg 2 (d) is already set
            CL_CHECK(queue.enqueueNDRangeKernel(relu_kernel, cl::NullRange, global_add, local_add));

            CL_CHECK(queue.finish()); // Sync before final add

            // 8. Final Residual Update: EH = EH + ReLU(hor_output), EV[n] = EV[n] + ReLU(ver_output)
            // EH = EH + relu_hor_output (update d_EH in-place)
            CL_CHECK(add_kernel.setArg(0, d_EH)); // Input A
            CL_CHECK(add_kernel.setArg(1, d_relu_hor_output)); // Input B
            CL_CHECK(add_kernel.setArg(2, d_EH)); // Output C (in-place)
            // Arg 3 (d) is already set
            CL_CHECK(queue.enqueueNDRangeKernel(add_kernel, cl::NullRange, global_add, local_add));

            // Add ReLU(ver_output) to the first 'n' rows of EV
            cl::Kernel update_ev_kernel = context_obj.kernels.at("updateEVRowsKernelCL");
            size_t global_update_ev_raw = static_cast<size_t>(n); // num_rows_to_update
            size_t global_update_ev_padded = ((global_update_ev_raw + local_work_size_1d - 1) / local_work_size_1d) * local_work_size_1d;
            cl::NDRange global_update_ev(global_update_ev_padded);
            cl::NDRange local_update_ev(local_work_size_1d);
            CL_CHECK(update_ev_kernel.setArg(0, d_EV_processed_data));
            CL_CHECK(update_ev_kernel.setArg(1, d_relu_ver_output));
            CL_CHECK(update_ev_kernel.setArg(2, n)); // num_rows_to_update
            CL_CHECK(update_ev_kernel.setArg(3, d)); // num_cols
            CL_CHECK(queue.enqueueNDRangeKernel(update_ev_kernel, cl::NullRange, global_update_ev, local_update_ev));

            CL_CHECK(queue.finish()); // Sync before D->H copy

            // --- Copy Results D->H ---
            CL_CHECK(queue.enqueueReadBuffer(d_EH, CL_TRUE, 0, embed_bytes, EH.data()));
            CL_CHECK(queue.enqueueReadBuffer(d_EV_processed_data, CL_TRUE, 0, ev_processed_bytes, EV.mapped_data));

            // --- Free Device Memory ---
            // cl::Buffer objects automatically release memory when they go out of scope (RAII)
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Standard Exception in attention clforprop(..., tokenCount): " << e.what() << std::endl;
        // Cleanup is handled by RAII for cl::Buffer
        throw; // Re-throw the exception
    }
}


/**
 * @brief OpenCL forward propagation for a 2nd to last block's attention class (incomplete attention)
 * @param EVp EV vector from previous block (unused in current C++ logic, ignored here)
 * @param d_embedding input embedding dimension (in)
 * @param layers_mlp layers of MLPs (layers) - NOTE: Unused, mlp object sizes used.
 * @param totalTokenCount total number of tokens processed so far (tokenCount in C++)
 * @param blockIdx which block is being processed (blockCount in C++)
 * @param contextWindowSize number of tokens for each attention head (n in C++)
 */
void attention::clforprop(std::vector<std::vector<float>> /* EVp */, int& d_embedding, int& layers_mlp, int& totalTokenCount,
    int& blockIdx, int& contextWindowSize)
{
    // Handle first block case by calling the other overload
    if (blockIdx == 0) {
    // Match C++: process min(totalTokenCount, contextWindowSize) for the first block
    int firstBlockTokenCount = std::min<int>(totalTokenCount, contextWindowSize);
        clforprop(d_embedding, layers_mlp, firstBlockTokenCount);
        return;
    }

    // Use constants defined in attention.hpp for clarity
    const int n = CONTEXT_WIN;      // context window
    const int d = EMBEDDING;        // Embedding dimension
    const int h = MATHEIGHTS;       // Height dimension of K/Q, projection matrices
    // Number of EV rows to process, matching CPU/CUDA logic for EV.sumRows(totalTokenCount) and EV.addToRows(totalTokenCount, ...)
    const int num_ev_rows_to_process = totalTokenCount;

    int start_idx_in_full_context = (blockIdx - 1) * contextWindowSize;
    int end_idx_in_full_context = std::min<int>(totalTokenCount, blockIdx * contextWindowSize);
    const int count = std::max<int>(0, end_idx_in_full_context - start_idx_in_full_context);

    // --- Basic Validation ---
    if (count <= 0) {
        std::cerr << "Warning: clforprop(..., blockIdx=" << blockIdx << ") called with calculated count <= 0. Skipping computation." << std::endl;
        // Ensure EH and EV[count] are valid even if skipping
        if (EH.size() != static_cast<size_t>(d)) EH.resize(d, 0.0f);
        std::fill(EH.begin(), EH.end(), 0.0f); // EH is std::vector<float>
        if (count == 0) { // If calculated block-specific count is 0
            if (EV.mapped_data && EV.row > 0 && EV.col == d) { // Assuming index 0 for EV if count is 0
                std::fill_n(EV.mapped_data, EV.col, 0.0f);
            }
        }
        return;
    }
    // EV mat validation: must be large enough for 'num_ev_rows_to_process'
    if (EV.row < num_ev_rows_to_process || EV.col != d) {
        throw std::runtime_error("EV matrix not properly sized for totalTokenCount. Expected rows >= " + std::to_string(num_ev_rows_to_process) + " and cols = " + std::to_string(d) +
                                 ", but got rows = " + std::to_string(EV.row) + " and cols = " + std::to_string(EV.col));
    }

    // ** Assumption: K, Q, KdotQ members are pre-populated with data relevant to this block (size 'count') **
    if (K.row != n || K.col != h || // K and Q are n x h (tokenCount x MATHEIGHTS)
        Q.row != n || Q.col != h || // K and Q are n x h (tokenCount x MATHEIGHTS)
        KdotQ.row != count || KdotQ.col != count ||
        MH.row != d || MH.col != h || // MH is mat(d,h)
        MV.row != d || MV.col != h || // MV is mat(d,h)
        EH.size() != static_cast<size_t>(d) ||
        hor.hlayers.empty() || ver.hlayers.empty() || hor.weights.empty() || ver.weights.empty())
    {
        throw std::runtime_error("Attention component dimension mismatch or uninitialized member in clforprop(..., blockIdx). Check K/Q/KdotQ size matches calculated 'count'.");
    }
    if (d != d_embedding) {
        throw std::runtime_error("Embedding dimension mismatch between definition (EMBEDDING) and argument (d_embedding).");
    }
    if (hor.hlayers[0].size() != static_cast<size_t>(d) || ver.hlayers[0].size() != static_cast<size_t>(d) ||
        hor.weights.back().row != static_cast<size_t>(d) || ver.weights.back().row != static_cast<size_t>(d)) { // Assuming weights are d x d
        throw std::runtime_error("MLP input/output layer dimension mismatch with embedding dimension 'd'.");
    }

    // --- Device Memory Buffers ---
    // (Same list as the first overload, just using 'count' instead of 'n')
    try {
        cl_int cl_err; // For OpenCL error codes
        OpenCLContext& context_obj = this->clcontext;
        cl::Context context = context_obj.context;
        cl::CommandQueue queue = context_obj.queue;

        // --- Allocate Device Memory ---
        size_t k_bytes = static_cast<size_t>(count) * h * sizeof(float);
        size_t q_bytes = static_cast<size_t>(count) * h * sizeof(float);
        size_t kdotq_bytes = static_cast<size_t>(count) * count * sizeof(float);
        size_t head_bytes = static_cast<size_t>(count) * count * sizeof(float);
        size_t sums_bytes = static_cast<size_t>(count) * sizeof(float);
        size_t accum_bytes = static_cast<size_t>(h) * sizeof(float);
        // Size for the first 'num_ev_rows_to_process' rows of EV
        size_t ev_processed_bytes = static_cast<size_t>(num_ev_rows_to_process) * d * sizeof(float);
        size_t proj_mat_bytes = static_cast<size_t>(d) * h * sizeof(float); // MH/MV are d x h
        size_t embed_bytes = static_cast<size_t>(d) * sizeof(float);

        cl::Buffer d_K(context, CL_MEM_READ_ONLY, k_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        cl::Buffer d_Q(context, CL_MEM_READ_ONLY, q_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        cl::Buffer d_KdotQ(context, CL_MEM_READ_ONLY, kdotq_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        cl::Buffer d_head(context, CL_MEM_READ_WRITE, head_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        cl::Buffer d_row_sums(context, CL_MEM_READ_WRITE, sums_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        cl::Buffer d_col_sums(context, CL_MEM_READ_WRITE, sums_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        cl::Buffer d_dh_accum(context, CL_MEM_READ_WRITE, accum_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        cl::Buffer d_dv_accum(context, CL_MEM_READ_WRITE, accum_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        cl::Buffer d_MH_hxd(context, CL_MEM_READ_ONLY, proj_mat_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        cl::Buffer d_MV_hxd(context, CL_MEM_READ_ONLY, proj_mat_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        cl::Buffer d_dh(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        cl::Buffer d_dv(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        cl::Buffer d_EH(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        cl::Buffer d_EV_processed_data(context, CL_MEM_READ_WRITE, ev_processed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        cl::Buffer d_ver_accumulated_ev(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        cl::Buffer d_hor_inputs(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        cl::Buffer d_ver_inputs(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        cl::Buffer d_hor_output(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        cl::Buffer d_ver_output(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        cl::Buffer d_relu_hor_output(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        cl::Buffer d_relu_ver_output(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        cl::Buffer d_mlp_bufferA_hor(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        cl::Buffer d_mlp_bufferB_hor(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        cl::Buffer d_mlp_bufferA_ver(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        cl::Buffer d_mlp_bufferB_ver(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        cl::Buffer d_mlp_pre_activation(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);

        // Initialize accumulators
        float zero = 0.0f;
        CL_CHECK(queue.enqueueFillBuffer(d_dh_accum, zero, 0, accum_bytes));
        CL_CHECK(queue.enqueueFillBuffer(d_dv_accum, zero, 0, accum_bytes));

        // --- Flatten Host Data & Copy H->D ---
        // ** Assuming K, Q, KdotQ members contain the 'count'-sized data for this block **
        // For mat objects, use mapped_data directly
        CL_CHECK(queue.enqueueWriteBuffer(d_K, CL_TRUE, 0, k_bytes, K.mapped_data));
        CL_CHECK(queue.enqueueWriteBuffer(d_Q, CL_TRUE, 0, q_bytes, Q.mapped_data));
        CL_CHECK(queue.enqueueWriteBuffer(d_KdotQ, CL_TRUE, 0, kdotq_bytes, KdotQ.mapped_data));
        CL_CHECK(queue.enqueueWriteBuffer(d_MH_hxd, CL_TRUE, 0, proj_mat_bytes, MH.mapped_data)); // MH is mat(d,h)
        CL_CHECK(queue.enqueueWriteBuffer(d_MV_hxd, CL_TRUE, 0, proj_mat_bytes, MV.mapped_data)); // MV is mat(d,h)
        CL_CHECK(queue.enqueueWriteBuffer(d_EH, CL_TRUE, 0, embed_bytes, EH.data()));
        // Copy the first 'num_ev_rows_to_process' rows of EV to device
        CL_CHECK(queue.enqueueWriteBuffer(d_EV_processed_data, CL_TRUE, 0, ev_processed_bytes, EV.mapped_data));


        // --- Kernel Launches ---
        const size_t local_work_size_1d = 256;

        // 1. Calculate Attention Weights: head = LOTA(KdotQ)
        cl::Kernel lota_kernel = context_obj.kernels.at("clLOTA2dmasking");
        size_t totalElementsLOTA = static_cast<size_t>(CONTEXT_WIN) * CONTEXT_WIN;
        if (totalElementsLOTA > 0) {
            // Same launch considerations as in the first overload, using 'count'
            size_t global_lota_raw = totalElementsLOTA;
            size_t local_lota_clamped = (std::min)(global_lota_raw, local_work_size_1d);
            if (local_lota_clamped == 0) local_lota_clamped = 1;
            size_t global_lota_padded = ((global_lota_raw + local_lota_clamped - 1) / local_lota_clamped) * local_lota_clamped;

            cl::NDRange global_lota(global_lota_padded);
            cl::NDRange local_lota(local_lota_clamped);

            CL_CHECK(lota_kernel.setArg(0, d_KdotQ));
            CL_CHECK(lota_kernel.setArg(1, d_head));
            CL_CHECK(lota_kernel.setArg(2, CONTEXT_WIN)); // rows
            CL_CHECK(lota_kernel.setArg(3, CONTEXT_WIN)); // cols
            CL_CHECK(lota_kernel.setArg(4, count)); // cols
            cl_int cl_att_is_self_lota = this->isSelfAttention ? 1 : 0;
            CL_CHECK(lota_kernel.setArg(5, cl_att_is_self_lota));
            CL_CHECK(queue.enqueueNDRangeKernel(lota_kernel, cl::NullRange, global_lota, local_lota));
        }
        // Removed the 'else' block that was here, as the if (totalElementsLOTA > 0) covers the skip logic.

        // 2. Compute Head Row/Column Sums (Masked)
        cl::Kernel sums_kernel = context_obj.kernels.at("computeHeadSumsMaskedKernel");
        size_t global_sums_raw = static_cast<size_t>(count);
        size_t global_sums_padded = ((global_sums_raw + local_work_size_1d - 1) / local_work_size_1d) * local_work_size_1d;
        cl::NDRange global_sums(global_sums_padded);
        cl::NDRange local_sums(local_work_size_1d);
        cl_int cl_isSelfAttention = isSelfAttention;
        CL_CHECK(sums_kernel.setArg(0, d_head));
        CL_CHECK(sums_kernel.setArg(1, d_row_sums));
        CL_CHECK(sums_kernel.setArg(2, d_col_sums));
        CL_CHECK(sums_kernel.setArg(3, count)); // Use count
        CL_CHECK(sums_kernel.setArg(4, cl_isSelfAttention));
        CL_CHECK(queue.enqueueNDRangeKernel(sums_kernel, cl::NullRange, global_sums, local_sums));

        // 3. Accumulate Weighted K and Q vectors
        cl::Kernel accum_kernel = context_obj.kernels.at("accumulateWeightedVectorsKernel");
        size_t global_accum_raw = static_cast<size_t>(count);
        size_t global_accum_padded = ((global_accum_raw + local_work_size_1d - 1) / local_work_size_1d) * local_work_size_1d; // Launch per token
        cl::NDRange global_accum(global_accum_padded);
        cl::NDRange local_accum(local_work_size_1d);
        CL_CHECK(accum_kernel.setArg(0, d_row_sums));
        CL_CHECK(accum_kernel.setArg(1, d_col_sums));
        CL_CHECK(accum_kernel.setArg(2, d_K));
        CL_CHECK(accum_kernel.setArg(3, d_Q));
        CL_CHECK(accum_kernel.setArg(4, d_dh_accum));
        CL_CHECK(accum_kernel.setArg(5, d_dv_accum));
        CL_CHECK(accum_kernel.setArg(6, count)); // Use count
        CL_CHECK(accum_kernel.setArg(7, h));
        CL_CHECK(queue.enqueueNDRangeKernel(accum_kernel, cl::NullRange, global_accum, local_accum));

        // 4. Project accumulated vectors
        cl::Kernel proj_kernel = context_obj.kernels.at("kernelLayerForward");
        size_t global_proj_raw = static_cast<size_t>(d);
        size_t global_proj_padded = ((global_proj_raw + local_work_size_1d - 1) / local_work_size_1d) * local_work_size_1d;
        cl::NDRange global_proj(global_proj_padded);
        cl::NDRange local_proj(local_work_size_1d); // Or cl::NullRange
        CL_CHECK(proj_kernel.setArg(3, h)); // input_size
        CL_CHECK(proj_kernel.setArg(4, d)); // output_size

        CL_CHECK(proj_kernel.setArg(0, d_dh_accum));
        CL_CHECK(proj_kernel.setArg(1, d_MH_hxd));
        CL_CHECK(proj_kernel.setArg(2, d_dh));
        CL_CHECK(queue.enqueueNDRangeKernel(proj_kernel, cl::NullRange, global_proj, local_proj));

        CL_CHECK(proj_kernel.setArg(0, d_dv_accum));
        CL_CHECK(proj_kernel.setArg(1, d_MV_hxd));
        CL_CHECK(proj_kernel.setArg(2, d_dv));
        CL_CHECK(queue.enqueueNDRangeKernel(proj_kernel, cl::NullRange, global_proj, local_proj));

        // 5. Prepare MLP inputs
        cl::Kernel add_kernel = context_obj.kernels.at("vectorAddKernel");
        size_t global_add_raw = static_cast<size_t>(d);
        size_t global_add_padded = ((global_add_raw + local_work_size_1d - 1) / local_work_size_1d) * local_work_size_1d;
        cl::NDRange global_add(global_add_padded);
        cl::NDRange local_add(local_work_size_1d); // Or cl::NullRange
        CL_CHECK(add_kernel.setArg(3, d));

        CL_CHECK(add_kernel.setArg(0, d_EH));
        CL_CHECK(add_kernel.setArg(1, d_dh));
        CL_CHECK(add_kernel.setArg(2, d_hor_inputs));
        CL_CHECK(queue.enqueueNDRangeKernel(add_kernel, cl::NullRange, global_add, local_add));

        // Accumulate first 'num_ev_rows_to_process' rows of EV for ver_input
        cl::Kernel accum_ev_kernel = context_obj.kernels.at("accumulateEVRowsKernelCL");
        // global_add is for size 'd' (col_size), local_add is local_work_size_1d
        CL_CHECK(accum_ev_kernel.setArg(0, d_EV_processed_data));
        CL_CHECK(accum_ev_kernel.setArg(1, d_ver_accumulated_ev));
        CL_CHECK(accum_ev_kernel.setArg(2, num_ev_rows_to_process)); // num_rows
        CL_CHECK(accum_ev_kernel.setArg(3, d));                      // col_size
        CL_CHECK(queue.enqueueNDRangeKernel(accum_ev_kernel, cl::NullRange, global_add, local_add));

        // ver_input = d_ver_accumulated_ev + dv
        CL_CHECK(add_kernel.setArg(0, d_ver_accumulated_ev));
        CL_CHECK(add_kernel.setArg(1, d_dv));
        CL_CHECK(add_kernel.setArg(2, d_ver_inputs));
        // Arg 3 (d) is already set
        CL_CHECK(queue.enqueueNDRangeKernel(add_kernel, cl::NullRange, global_add, local_add));

        CL_CHECK(queue.finish());

        // --- 6. Run MLPs Forward ---
        // (Identical launch logic to the first overload, using 'd' for sizes)
        // try { // Inner try-catch for MLP specific errors can be removed
            cl::Kernel mlp_fwd_kernel = context_obj.kernels.at("kernelLayerForward");
            cl::Kernel sigmoid_kernel = context_obj.kernels.at("clSigmoid1d");

            CL_CHECK(queue.enqueueCopyBuffer(d_hor_inputs, d_mlp_bufferA_hor, 0, 0, embed_bytes));
            CL_CHECK(queue.enqueueCopyBuffer(d_ver_inputs, d_mlp_bufferA_ver, 0, 0, embed_bytes));

            cl::Buffer& current_in_hor = d_mlp_bufferA_hor;
            cl::Buffer& current_out_hor = d_mlp_bufferB_hor;
            cl::Buffer& current_in_ver = d_mlp_bufferA_ver;
            cl::Buffer& current_out_ver = d_mlp_bufferB_ver;

            size_t num_weight_matrices = hor.weights.size();
            size_t global_mlp_raw = static_cast<size_t>(d);
            size_t global_mlp_padded = ((global_mlp_raw + local_work_size_1d - 1) / local_work_size_1d) * local_work_size_1d;
            cl::NDRange global_mlp(global_mlp_padded);
            cl::NDRange local_mlp(local_work_size_1d); // Or cl::NullRange

            for (size_t layer_idx = 0; layer_idx < num_weight_matrices; ++layer_idx) {
                bool is_last_layer = (layer_idx == num_weight_matrices - 1);
                int input_size = d;
                int output_size = d;

            // Hor
                {
                    mat& current_weights_mat = hor.weights[layer_idx];
                    if (current_weights_mat.row != output_size || current_weights_mat.col != input_size) {
                        throw std::runtime_error("MLP weight dimension mismatch (hor, block). Expected " +
                                                 std::to_string(output_size) + "x" + std::to_string(input_size) +
                                                 ", got " + std::to_string(current_weights_mat.row) + "x" + std::to_string(current_weights_mat.col));
                    }
                    size_t weights_bytes = static_cast<size_t>(current_weights_mat.row) * current_weights_mat.col * sizeof(float);
                    cl::Buffer d_mlp_weights(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, weights_bytes, current_weights_mat.mapped_data, &cl_err); CL_CHECK(cl_err);

                    cl::Buffer& target_output_buffer = is_last_layer ? d_hor_output : d_mlp_pre_activation;
                    CL_CHECK(mlp_fwd_kernel.setArg(0, current_in_hor));
                    CL_CHECK(mlp_fwd_kernel.setArg(1, d_mlp_weights));
                    CL_CHECK(mlp_fwd_kernel.setArg(2, target_output_buffer));
                    CL_CHECK(mlp_fwd_kernel.setArg(3, input_size));
                    CL_CHECK(mlp_fwd_kernel.setArg(4, output_size));
                    CL_CHECK(queue.enqueueNDRangeKernel(mlp_fwd_kernel, cl::NullRange, global_mlp, local_mlp));
                    if (!is_last_layer) {
                        CL_CHECK(sigmoid_kernel.setArg(0, d_mlp_pre_activation));
                        CL_CHECK(sigmoid_kernel.setArg(1, current_out_hor));
                        CL_CHECK(sigmoid_kernel.setArg(2, output_size));
                        CL_CHECK(queue.enqueueNDRangeKernel(sigmoid_kernel, cl::NullRange, global_mlp, local_mlp));
                        std::swap(current_in_hor, current_out_hor);
                    }
                }
            // Ver
                {
                    mat& current_weights_mat = ver.weights[layer_idx];
                     if (current_weights_mat.row != output_size || current_weights_mat.col != input_size) {
                        throw std::runtime_error("MLP weight dimension mismatch (ver, block). Expected " +
                                                 std::to_string(output_size) + "x" + std::to_string(input_size) +
                                                 ", got " + std::to_string(current_weights_mat.row) + "x" + std::to_string(current_weights_mat.col));
                    }
                    size_t weights_bytes = static_cast<size_t>(current_weights_mat.row) * current_weights_mat.col * sizeof(float);
                    cl::Buffer d_mlp_weights(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, weights_bytes, current_weights_mat.mapped_data, &cl_err); CL_CHECK(cl_err);

                    cl::Buffer& target_output_buffer = is_last_layer ? d_ver_output : d_mlp_pre_activation;
                    CL_CHECK(mlp_fwd_kernel.setArg(0, current_in_ver));
                    CL_CHECK(mlp_fwd_kernel.setArg(1, d_mlp_weights));
                    CL_CHECK(mlp_fwd_kernel.setArg(2, target_output_buffer));
                    CL_CHECK(queue.enqueueNDRangeKernel(mlp_fwd_kernel, cl::NullRange, global_mlp, local_mlp));
                    if (!is_last_layer) {
                        CL_CHECK(sigmoid_kernel.setArg(0, d_mlp_pre_activation));
                        CL_CHECK(sigmoid_kernel.setArg(1, current_out_ver));
                        CL_CHECK(queue.enqueueNDRangeKernel(sigmoid_kernel, cl::NullRange, global_mlp, local_mlp));
                        std::swap(current_in_ver, current_out_ver);
                    }
                }
            }
            CL_CHECK(queue.finish());
        // } // End of inner try-catch for MLP

        // --- Free MLP Intermediate Buffers --- (Handled by RAII)

        // 7. Apply ReLU
        cl::Kernel relu_kernel = context_obj.kernels.at("clReLU1d");
        // Use global_add and local_add from step 5
        CL_CHECK(relu_kernel.setArg(2, d));

        CL_CHECK(relu_kernel.setArg(0, d_hor_output));
        CL_CHECK(relu_kernel.setArg(1, d_relu_hor_output));
        CL_CHECK(queue.enqueueNDRangeKernel(relu_kernel, cl::NullRange, global_add, local_add));

        CL_CHECK(relu_kernel.setArg(0, d_ver_output));
        CL_CHECK(relu_kernel.setArg(1, d_relu_ver_output));
        CL_CHECK(queue.enqueueNDRangeKernel(relu_kernel, cl::NullRange, global_add, local_add));

        CL_CHECK(queue.finish());

        // 8. Final Residual Update
        // Use add_kernel from step 5

        CL_CHECK(add_kernel.setArg(0, d_EH));
        CL_CHECK(add_kernel.setArg(1, d_relu_hor_output));
        CL_CHECK(add_kernel.setArg(2, d_EH)); // In-place update
        CL_CHECK(queue.enqueueNDRangeKernel(add_kernel, cl::NullRange, global_add, local_add));

        // Add ReLU(ver_output) to the first 'num_ev_rows_to_process' rows of EV
        cl::Kernel update_ev_kernel = context_obj.kernels.at("updateEVRowsKernelCL");
        size_t global_update_ev_raw = static_cast<size_t>(num_ev_rows_to_process); // num_rows_to_update
        size_t global_update_ev_padded = ((global_update_ev_raw + local_work_size_1d - 1) / local_work_size_1d) * local_work_size_1d;
        cl::NDRange global_update_ev(global_update_ev_padded);
        cl::NDRange local_update_ev(local_work_size_1d);
        CL_CHECK(update_ev_kernel.setArg(0, d_EV_processed_data));
        CL_CHECK(update_ev_kernel.setArg(1, d_relu_ver_output));
        CL_CHECK(update_ev_kernel.setArg(2, num_ev_rows_to_process)); // num_rows_to_update
        CL_CHECK(update_ev_kernel.setArg(3, d));                      // num_cols
        CL_CHECK(queue.enqueueNDRangeKernel(update_ev_kernel, cl::NullRange, global_update_ev, local_update_ev));

        CL_CHECK(queue.finish());

        // --- Copy Results D->H ---
        CL_CHECK(queue.enqueueReadBuffer(d_EH, CL_TRUE, 0, embed_bytes, EH.data()));
        CL_CHECK(queue.enqueueReadBuffer(d_EV_processed_data, CL_TRUE, 0, ev_processed_bytes, EV.mapped_data));

        // --- Free Device Memory --- (Handled by RAII)

    }
    catch (const std::exception& e) {
        std::cerr << "Standard Exception in attention clforprop(..., blockIdx=" << blockIdx << "): " << e.what() << std::endl;
        throw; // Re-throw the exception
    }
}

#endif // USE_OPENCL
