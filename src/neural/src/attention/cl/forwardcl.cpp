#ifdef USE_OPENCL

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
#include <CL/cl.hpp>

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

    // --- Basic Validation ---
    if (n <= 0) {
        std::cerr << "Warning: clforprop(..., tokenCount=" << n << ") called with tokenCount <= 0. Skipping computation." << std::endl;
        std::fill(EH.begin(), EH.end(), 0.0f);
        if (!EV.empty()) {
             if (EV.size() <= static_cast<size_t>(n)) EV.resize(n + 1);
             if (EV[n].size() != static_cast<size_t>(d)) EV[n].resize(d, 0.0f);
             std::fill(EV[n].begin(), EV[n].end(), 0.0f);
        }
        return;
    }
    if (EV.size() <= static_cast<size_t>(n)) EV.resize(n + 1);
    if (EV[n].size() != static_cast<size_t>(d)) EV[n].resize(d, 0.0f);

    if (K.size() != static_cast<size_t>(n) || (!K.empty() && K[0].size() != static_cast<size_t>(h)) ||
        Q.size() != static_cast<size_t>(n) || (!Q.empty() && Q[0].size() != static_cast<size_t>(h)) ||
        KdotQ.size() != static_cast<size_t>(n) || (!KdotQ.empty() && KdotQ[0].size() != static_cast<size_t>(n)) ||
        MH.a.size() != static_cast<size_t>(d) || (!MH.a.empty() && MH.a[0].size() != static_cast<size_t>(h)) ||
        MV.a.size() != static_cast<size_t>(d) || (!MV.a.empty() && MV.a[0].size() != static_cast<size_t>(h)) ||
        EH.size() != static_cast<size_t>(d) ||
        EV[n].size() != static_cast<size_t>(d) ||
        hor.hlayers.empty() || ver.hlayers.empty() || hor.weights.empty() || ver.weights.empty())
    {
        throw std::runtime_error("Attention component dimension mismatch or uninitialized member in clforprop(..., tokenCount).");
    }
    if (d != d_embedding) {
        throw std::runtime_error("Embedding dimension mismatch between definition (EMBEDDING) and argument (d_embedding).");
    }
    if (hor.hlayers[0].size() != static_cast<size_t>(d) || ver.hlayers[0].size() != static_cast<size_t>(d) ||
        hor.weights.back().size() != static_cast<size_t>(d) || ver.weights.back().size() != static_cast<size_t>(d)) {
         throw std::runtime_error("MLP input/output layer dimension mismatch with embedding dimension 'd'.");
    }

    try {
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
        size_t proj_mat_bytes = static_cast<size_t>(h) * d * sizeof(float); // h x d
        size_t embed_bytes = static_cast<size_t>(d) * sizeof(float);        // d

        cl::Buffer d_K(context, CL_MEM_READ_ONLY, k_bytes);
        cl::Buffer d_Q(context, CL_MEM_READ_ONLY, q_bytes);
        cl::Buffer d_KdotQ(context, CL_MEM_READ_ONLY, kdotq_bytes);
        cl::Buffer d_head(context, CL_MEM_READ_WRITE, head_bytes); // LOTA output, read by sums
        cl::Buffer d_row_sums(context, CL_MEM_READ_WRITE, sums_bytes); // Written by sums, read by accum
        cl::Buffer d_col_sums(context, CL_MEM_READ_WRITE, sums_bytes); // Written by sums, read by accum
        cl::Buffer d_dh_accum(context, CL_MEM_READ_WRITE, accum_bytes); // Written by accum (atomic), read by proj
        cl::Buffer d_dv_accum(context, CL_MEM_READ_WRITE, accum_bytes); // Written by accum (atomic), read by proj
        cl::Buffer d_MH_hxd(context, CL_MEM_READ_ONLY, proj_mat_bytes); // Transposed MH
        cl::Buffer d_MV_hxd(context, CL_MEM_READ_ONLY, proj_mat_bytes); // Transposed MV
        cl::Buffer d_dh (context, CL_MEM_READ_WRITE, embed_bytes); // Written by proj, read by add
        cl::Buffer d_dv(context, CL_MEM_READ_WRITE, embed_bytes); // Written by proj, read by add
        cl::Buffer d_EH(context, CL_MEM_READ_WRITE, embed_bytes); // Read by add, written by final add
        cl::Buffer d_EV_current(context, CL_MEM_READ_WRITE, embed_bytes); // Read by add, written by final add
        cl::Buffer d_hor_inputs(context, CL_MEM_READ_WRITE, embed_bytes); // Written by add, read by MLP
        cl::Buffer d_ver_inputs(context, CL_MEM_READ_WRITE, embed_bytes); // Written by add, read by MLP
        cl::Buffer d_hor_output(context, CL_MEM_READ_WRITE, embed_bytes); // Written by MLP, read by ReLU
        cl::Buffer d_ver_output(context, CL_MEM_READ_WRITE, embed_bytes); // Written by MLP, read by ReLU
        cl::Buffer d_relu_hor_output(context, CL_MEM_READ_WRITE, embed_bytes); // Written by ReLU, read by final add
        cl::Buffer d_relu_ver_output(context, CL_MEM_READ_WRITE, embed_bytes); // Written by ReLU, read by final add

        // Allocate MLP Intermediate Buffers
        cl::Buffer d_mlp_bufferA_hor(context, CL_MEM_READ_WRITE, embed_bytes);
        cl::Buffer d_mlp_bufferB_hor(context, CL_MEM_READ_WRITE, embed_bytes);
        cl::Buffer d_mlp_bufferA_ver(context, CL_MEM_READ_WRITE, embed_bytes);
        cl::Buffer d_mlp_bufferB_ver(context, CL_MEM_READ_WRITE, embed_bytes);
        cl::Buffer d_mlp_pre_activation(context, CL_MEM_READ_WRITE, embed_bytes); // Temp buffer for MLP layer output before activation
        // d_mlp_weights buffer will be created inside the loop

        // Initialize accumulators to zero
        float zero = 0.0f;
        queue.enqueueFillBuffer(d_dh_accum, zero, 0, accum_bytes);
        queue.enqueueFillBuffer(d_dv_accum, zero, 0, accum_bytes);

        // --- Flatten Host Data & Copy H->D ---
        std::vector<float> flat_K = flatten(K);
        std::vector<float> flat_Q = flatten(Q);
        std::vector<float> flat_KdotQ = flatten(KdotQ);
        std::vector<float> flat_MH_hxd, flat_MV_hxd;
        transposeFlattenMatrix(MH.a, flat_MH_hxd, d, h);
        transposeFlattenMatrix(MV.a, flat_MV_hxd, d, h);

        queue.enqueueWriteBuffer(d_K, CL_TRUE, 0, k_bytes, flat_K.data()); // Use CL_TRUE for blocking write
        queue.enqueueWriteBuffer(d_Q, CL_TRUE, 0, q_bytes, flat_Q.data());
        queue.enqueueWriteBuffer(d_KdotQ, CL_TRUE, 0, kdotq_bytes, flat_KdotQ.data());
        queue.enqueueWriteBuffer(d_MH_hxd, CL_TRUE, 0, proj_mat_bytes, flat_MH_hxd.data());
        queue.enqueueWriteBuffer(d_MV_hxd, CL_TRUE, 0, proj_mat_bytes, flat_MV_hxd.data());
        queue.enqueueWriteBuffer(d_EH, CL_TRUE, 0, embed_bytes, EH.data());
        queue.enqueueWriteBuffer(d_EV_current, CL_TRUE, 0, embed_bytes, EV[n].data());

        // --- Kernel Launches ---
        const size_t local_work_size_1d = 256; // General purpose block size

        // 1. Calculate Attention Weights: head = LOTA(KdotQ)
        // Use clLOTA2d as KdotQ is 2D
        cl::Kernel lota_kernel = context_obj.kernels.at("clLOTA2d");
        size_t totalElementsLOTA = static_cast<size_t>(n) * n;
        size_t global_work_size_lota[1] = { (totalElementsLOTA + local_work_size_1d - 1) / local_work_size_1d * local_work_size_1d };
        size_t local_work_size_lota[1] = { local_work_size_1d };
        // NOTE: clLOTA2d expects a single work-group for reduction. Adjust launch if needed.
        // If n*n exceeds max work group size, clLOTA2d needs modification or a different approach.
        // Assuming n*n fits for now.
        if (totalElementsLOTA > 0) {
            if (totalElementsLOTA > local_work_size_1d) {
                // The current clLOTA2d kernel uses local memory reduction over get_local_id(0).
                // It expects a 1D launch where the global size is rows*cols.
                // The reduction happens *within* each workgroup. If global_size > local_size,
                // the reduction is only partial per group.
                // For a *full* reduction as intended by the original LOTA logic (min/sum over all elements),
                // either n*n must be <= max work group size and launched as a single group,
                // or a multi-stage reduction kernel is needed.
                // Let's stick to the current kernel's behavior: reduction within workgroups.
                // If a full reduction is strictly required, the kernel needs changing.
                // Launching with global size = n*n, local size = local_work_size_1d
                size_t global_lota_raw = totalElementsLOTA;
                size_t local_lota_clamped = std::min(global_lota_raw, local_work_size_1d); // Clamp local size if total is smaller
                if (local_lota_clamped == 0) local_lota_clamped = 1; // Avoid zero local size
                size_t global_lota_padded = ((global_lota_raw + local_lota_clamped - 1) / local_lota_clamped) * local_lota_clamped; // Align global size
   
                cl::NDRange global_lota(global_lota_padded);
                cl::NDRange local_lota(local_lota_clamped);
   
                lota_kernel.setArg(0, d_KdotQ);
                lota_kernel.setArg(1, d_head);
                lota_kernel.setArg(2, n); // rows
                lota_kernel.setArg(3, n); // cols
                queue.enqueueNDRangeKernel(lota_kernel, cl::NullRange, global_lota, local_lota);
            }   

            // 2. Compute Head Row/Column Sums (Masked)
            cl::Kernel sums_kernel = context_obj.kernels.at("computeHeadSumsMaskedKernel"); // Use the new kernel
            size_t global_sums_raw = static_cast<size_t>(n);
            size_t global_sums_padded = ((global_sums_raw + local_work_size_1d - 1) / local_work_size_1d) * local_work_size_1d;
            cl::NDRange global_sums(global_sums_padded);
            cl::NDRange local_sums(local_work_size_1d);
            cl_int cl_isSelfAttention = isSelfAttention; // Convert bool to cl_int
            sums_kernel.setArg(0, d_head);
            sums_kernel.setArg(1, d_row_sums);
            sums_kernel.setArg(2, d_col_sums);
            sums_kernel.setArg(3, n);
            sums_kernel.setArg(4, cl_isSelfAttention);
            queue.enqueueNDRangeKernel(sums_kernel, cl::NullRange, global_sums, local_sums);

            // 3. Accumulate Weighted K and Q vectors
            // NOTE: Requires float atomics support. Launching one work-item per token.
            cl::Kernel accum_kernel = context_obj.kernels.at("accumulateWeightedVectorsKernel"); // Use the new kernel
            size_t global_accum_raw = static_cast<size_t>(n);
            size_t global_accum_padded = ((global_accum_raw + local_work_size_1d - 1) / local_work_size_1d) * local_work_size_1d;
            cl::NDRange global_accum(global_accum_padded);
            cl::NDRange local_accum(local_work_size_1d);
            accum_kernel.setArg(0, d_row_sums);
            accum_kernel.setArg(1, d_col_sums);
            accum_kernel.setArg(2, d_K);
            accum_kernel.setArg(3, d_Q);
            accum_kernel.setArg(4, d_dh_accum);
            accum_kernel.setArg(5, d_dv_accum);
            accum_kernel.setArg(6, n);
            accum_kernel.setArg(7, h);
            queue.enqueueNDRangeKernel(accum_kernel, cl::NullRange, global_accum, local_accum);

            // 4. Project accumulated vectors: dh = dot(dh_accum, MH), dv = dot(dv_accum, MV)
            // Use kernelLayerForward (matrix-vector): output[j] = sum(input[i] * weights[j*input_size + i])
            // Input: d_dh_accum (size h), Weights: d_MH_hxd (size d x h), Output: d_dh (size d)
            cl::Kernel proj_kernel = context_obj.kernels.at("kernelLayerForward"); // Use the new kernel
            size_t global_proj_raw = static_cast<size_t>(d);
            size_t global_proj_padded = ((global_proj_raw + local_work_size_1d - 1) / local_work_size_1d) * local_work_size_1d;
            cl::NDRange global_proj(global_proj_padded);
            cl::NDRange local_proj(local_work_size_1d); // Or cl::NullRange

            // dh = dh_accum * MH_hxd (conceptually) -> kernelLayerForward(d_dh_accum, d_MH_hxd, d_dh, h, d)
            proj_kernel.setArg(0, d_dh_accum); // input vector (size h)
            proj_kernel.setArg(1, d_MH_hxd);   // weight matrix (d rows, h cols)
            proj_kernel.setArg(2, d_dh);       // output vector (size d)
            proj_kernel.setArg(3, h);          // input_size
            proj_kernel.setArg(4, d);          // output_size
            queue.enqueueNDRangeKernel(proj_kernel, cl::NullRange, global_proj, local_proj);

            // dv = dv_accum * MV_hxd (conceptually) -> kernelLayerForward(d_dv_accum, d_MV_hxd, d_dv, h, d)
            proj_kernel.setArg(0, d_dv_accum); // input vector (size h)
            proj_kernel.setArg(1, d_MV_hxd);   // weight matrix (d rows, h cols)
            proj_kernel.setArg(2, d_dv);       // output vector (size d)
            // Args 3 (h) and 4 (d) are already set
            queue.enqueueNDRangeKernel(proj_kernel, cl::NullRange, global_proj, local_proj);

            // 5. Prepare MLP inputs (Residual Add): hor_input = EH + dh, ver_input = EV_current + dv
            cl::Kernel add_kernel = context_obj.kernels.at("vectorAddKernel"); // Use the new kernel
            size_t global_add_raw = static_cast<size_t>(d);
            size_t global_add_padded = ((global_add_raw + local_work_size_1d - 1) / local_work_size_1d) * local_work_size_1d;
            cl::NDRange global_add(global_add_padded);
            cl::NDRange local_add(local_work_size_1d); // Or cl::NullRange

            // hor_input = EH + dh
            add_kernel.setArg(0, d_EH);
            add_kernel.setArg(1, d_dh);
            add_kernel.setArg(2, d_hor_inputs);
            add_kernel.setArg(3, d);
            queue.enqueueNDRangeKernel(add_kernel, cl::NullRange, global_add, local_add);

            // ver_input = EV_current + dv
            add_kernel.setArg(0, d_EV_current);
            add_kernel.setArg(1, d_dv);
            add_kernel.setArg(2, d_ver_inputs);
            // Arg 3 (d) is already set
            queue.enqueueNDRangeKernel(add_kernel, cl::NullRange, global_add, local_add);

            queue.finish(); // Sync before MLP uses these inputs

            // --- 6. Run MLPs Forward ---
            try {
                // Get kernels
                cl::Kernel mlp_fwd_kernel = context_obj.kernels.at("kernelLayerForward");
                cl::Kernel sigmoid_kernel = context_obj.kernels.at("clSigmoid1d"); // Assuming 1D version

                // Initial input copies using enqueueCopyBuffer
                queue.enqueueCopyBuffer(d_hor_inputs, d_mlp_bufferA_hor, 0, 0, embed_bytes);
                queue.enqueueCopyBuffer(d_ver_inputs, d_mlp_bufferA_ver, 0, 0, embed_bytes);

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
                    int output_size = d;

                    // --- Process Horizontal MLP Layer ---
                    {
                        std::vector<float> flat_weights_hor = flatten(hor.weights[layer_idx]);
                        size_t weights_bytes = flat_weights_hor.size() * sizeof(float);
                        if (weights_bytes != static_cast<size_t>(input_size) * output_size * sizeof(float)) {
                            throw std::runtime_error("MLP weight dimension mismatch (hor).");
                        }

                        // Create and write weights buffer for this layer
                        cl::Buffer d_mlp_weights(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, weights_bytes, flat_weights_hor.data());

                        cl::Buffer& target_output_buffer = is_last_layer ? d_hor_output : d_mlp_pre_activation;

                        mlp_fwd_kernel.setArg(0, current_in_hor);
                        mlp_fwd_kernel.setArg(1, d_mlp_weights);
                        mlp_fwd_kernel.setArg(2, target_output_buffer);
                        mlp_fwd_kernel.setArg(3, input_size);
                        mlp_fwd_kernel.setArg(4, output_size);
                        queue.enqueueNDRangeKernel(mlp_fwd_kernel, cl::NullRange, global_mlp, local_mlp);

                        if (!is_last_layer) {
                            sigmoid_kernel.setArg(0, d_mlp_pre_activation);
                            sigmoid_kernel.setArg(1, current_out_hor);
                            sigmoid_kernel.setArg(2, output_size);
                            queue.enqueueNDRangeKernel(sigmoid_kernel, cl::NullRange, global_mlp, local_mlp);

                            // Swap hor buffers (references)
                            std::swap(current_in_hor, current_out_hor);
                        }
                        // d_mlp_weights goes out of scope and is released
                    }

                    // --- Process Vertical MLP Layer ---
                    {
                        std::vector<float> flat_weights_ver = flatten(ver.weights[layer_idx]);
                        size_t weights_bytes = flat_weights_ver.size() * sizeof(float);
                        if (weights_bytes != static_cast<size_t>(input_size) * output_size * sizeof(float)) {
                            throw std::runtime_error("MLP weight dimension mismatch (ver).");
                        }

                        cl::Buffer d_mlp_weights(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, weights_bytes, flat_weights_ver.data());

                        cl::Buffer& target_output_buffer = is_last_layer ? d_ver_output : d_mlp_pre_activation;

                        mlp_fwd_kernel.setArg(0, current_in_ver);
                        mlp_fwd_kernel.setArg(1, d_mlp_weights);
                        mlp_fwd_kernel.setArg(2, target_output_buffer);
                        // Args 3 (input_size) and 4 (output_size) are already set
                        queue.enqueueNDRangeKernel(mlp_fwd_kernel, cl::NullRange, global_mlp, local_mlp);

                        if (!is_last_layer) {
                            sigmoid_kernel.setArg(0, d_mlp_pre_activation);
                            sigmoid_kernel.setArg(1, current_out_ver);
                            // Arg 2 (output_size) is already set
                            queue.enqueueNDRangeKernel(sigmoid_kernel, cl::NullRange, global_mlp, local_mlp);

                            // Swap ver buffers (references)
                            std::swap(current_in_ver, current_out_ver);
                        }
                        // d_mlp_weights goes out of scope and is released
                    }
                } // End loop over layers
                queue.finish(); // Sync after all MLP layers
            }
            catch (const cl::Error& err) {
                std::cerr << "OpenCL Error during MLP processing: " << err.what() << " (" << err.err() << ")" << std::endl;
                throw;
            }
            catch (const std::exception& e) {
                std::cerr << "Error during MLP processing: " << e.what() << std::endl;
                throw;
            }

            // Buffers d_mlp_bufferA/B_hor/ver and d_mlp_pre_activation go out of scope here

            // 7. Apply ReLU to MLP outputs
            cl::Kernel relu_kernel = context_obj.kernels.at("clReLU1d");
            // Use global_add and local_add from step 5 as the size 'd' is the same
            // size_t global_relu_raw = static_cast<size_t>(d);
            // size_t global_relu_padded = ((global_relu_raw + local_work_size_1d - 1) / local_work_size_1d) * local_work_size_1d;
            // cl::NDRange global_relu(global_relu_padded);
            // cl::NDRange local_relu(local_work_size_1d); // Or cl::NullRange

            // ReLU(hor_output) -> d_relu_hor_output
            relu_kernel.setArg(0, d_hor_output);
            relu_kernel.setArg(1, d_relu_hor_output);
            relu_kernel.setArg(2, d);
            queue.enqueueNDRangeKernel(relu_kernel, cl::NullRange, global_add, local_add);

            // ReLU(ver_output) -> d_relu_ver_output
            relu_kernel.setArg(0, d_ver_output);
            relu_kernel.setArg(1, d_relu_ver_output);
            // Arg 2 (d) is already set
            queue.enqueueNDRangeKernel(relu_kernel, cl::NullRange, global_add, local_add);

            queue.finish(); // Sync before final add

            // 8. Final Residual Update: EH = EH + ReLU(hor_output), EV[n] = EV[n] + ReLU(ver_output)
            // Use add_kernel from step 5

            // EH = EH + relu_hor_output (update d_EH in-place)
            add_kernel.setArg(0, d_EH); // Input A
            add_kernel.setArg(1, d_relu_hor_output); // Input B
            add_kernel.setArg(2, d_EH); // Output C (in-place)
            // Arg 3 (d) is already set
            queue.enqueueNDRangeKernel(add_kernel, cl::NullRange, global_add, local_add);

            // EV[n] = EV[n] + relu_ver_output (update d_EV_current in-place)
            add_kernel.setArg(0, d_EV_current); // Input A
            add_kernel.setArg(1, d_relu_ver_output); // Input B
            add_kernel.setArg(2, d_EV_current); // Output C (in-place)
            // Arg 3 (d) is already set
            queue.enqueueNDRangeKernel(add_kernel, cl::NullRange, global_add, local_add);

            queue.finish(); // Sync before D->H copy

            // --- Copy Results D->H ---
            queue.enqueueReadBuffer(d_EH, CL_TRUE, 0, embed_bytes, EH.data());
            queue.enqueueReadBuffer(d_EV_current, CL_TRUE, 0, embed_bytes, EV[n].data());

            // --- Free Device Memory ---
            // cl::Buffer objects automatically release memory when they go out of scope (RAII)
        }
    }
    catch (const cl::Error& err) {
        std::cerr << "OpenCL Error in attention clforprop(..., tokenCount): " << err.what() << " (" << err.err() << ")" << std::endl;
        // Add more specific error handling if needed
        throw; // Re-throw the exception
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
    // Need to adjust the call signature if the first overload strictly needs currentTokenCount
    // Assuming totalTokenCount is the relevant count for the first block.
    int currentTokenCount = totalTokenCount; // Or calculate appropriately if different
        clforprop(d_embedding, layers_mlp, currentTokenCount);
        return;
    }

    // Use constants defined in attention.hpp for clarity
    const int d = EMBEDDING;        // Embedding dimension
    const int h = MATHEIGHTS;       // Height dimension of K/Q, projection matrices

    // Calculate the number of tokens relevant to this block
    // This calculation might need adjustment based on how tokens are fed block by block.
    // Assuming 'count' is the number of *new* tokens processed in this block's window.
    int count = totalTokenCount - contextWindowSize * blockIdx; // Tokens processed *within* this block's window start
    count = std::min(count, contextWindowSize); // Clamp to the window size
    count = std::max(count, 0); // Ensure non-negative

    // --- Basic Validation ---
    if (count <= 0) {
        std::cerr << "Warning: clforprop(..., blockIdx=" << blockIdx << ") called with calculated count <= 0. Skipping computation." << std::endl;
        // Ensure EH and EV[count] are valid even if skipping
        if (EH.size() != static_cast<size_t>(d)) EH.resize(d, 0.0f);
        std::fill(EH.begin(), EH.end(), 0.0f);
        if (EV.size() <= static_cast<size_t>(count)) EV.resize(count + 1); // Resize EV vector itself if needed
        if (EV[count].size() != static_cast<size_t>(d)) EV[count].resize(d, 0.0f);
        std::fill(EV[count].begin(), EV[count].end(), 0.0f);
        return;
    }
    // Ensure EV vector is large enough and the specific EV[count] element is sized correctly
    if (EV.size() <= static_cast<size_t>(count)) EV.resize(count + 1);
    if (EV[count].size() != static_cast<size_t>(d)) EV[count].resize(d, 0.0f);

    // ** Assumption: K, Q, KdotQ members are pre-populated with data relevant to this block (size 'count') **
    if (K.size() != static_cast<size_t>(count) || (!K.empty() && K[0].size() != static_cast<size_t>(h)) ||
    Q.size() != static_cast<size_t>(count) || (!Q.empty() && Q[0].size() != static_cast<size_t>(h)) ||
    KdotQ.size() != static_cast<size_t>(count) || (!KdotQ.empty() && KdotQ[0].size() != static_cast<size_t>(count)) ||
    MH.a.size() != static_cast<size_t>(d) || (!MH.a.empty() && MH.a[0].size() != static_cast<size_t>(h)) ||
    MV.a.size() != static_cast<size_t>(d) || (!MV.a.empty() && MV.a[0].size() != static_cast<size_t>(h)) ||
    EH.size() != static_cast<size_t>(d) ||
    EV[count].size() != static_cast<size_t>(d) || // Check EV[count] specifically
    hor.hlayers.empty() || ver.hlayers.empty() || hor.weights.empty() || ver.weights.empty())
    {
        throw std::runtime_error("Attention component dimension mismatch or uninitialized member in clforprop(..., blockIdx). Check K/Q/KdotQ size matches calculated 'count'.");
    }
    if (d != d_embedding) {
        throw std::runtime_error("Embedding dimension mismatch between definition (EMBEDDING) and argument (d_embedding).");
    }
    if (hor.hlayers[0].size() != static_cast<size_t>(d) || ver.hlayers[0].size() != static_cast<size_t>(d) ||
        hor.weights.back().size() != static_cast<size_t>(d) || ver.weights.back().size() != static_cast<size_t>(d)) {
        throw std::runtime_error("MLP input/output layer dimension mismatch with embedding dimension 'd'.");
    }

    // --- Device Memory Buffers ---
    // (Same list as the first overload, just using 'count' instead of 'n')
    try {
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
        size_t proj_mat_bytes = static_cast<size_t>(h) * d * sizeof(float);
        size_t embed_bytes = static_cast<size_t>(d) * sizeof(float);

        cl::Buffer d_K(context, CL_MEM_READ_ONLY, k_bytes);
        cl::Buffer d_Q(context, CL_MEM_READ_ONLY, q_bytes);
        cl::Buffer d_KdotQ(context, CL_MEM_READ_ONLY, kdotq_bytes);
        cl::Buffer d_head(context, CL_MEM_READ_WRITE, head_bytes);
        cl::Buffer d_row_sums(context, CL_MEM_READ_WRITE, sums_bytes);
        cl::Buffer d_col_sums(context, CL_MEM_READ_WRITE, sums_bytes);
        cl::Buffer d_dh_accum(context, CL_MEM_READ_WRITE, accum_bytes);
        cl::Buffer d_dv_accum(context, CL_MEM_READ_WRITE, accum_bytes);
        cl::Buffer d_MH_hxd(context, CL_MEM_READ_ONLY, proj_mat_bytes);
        cl::Buffer d_MV_hxd(context, CL_MEM_READ_ONLY, proj_mat_bytes);
        cl::Buffer d_dh(context, CL_MEM_READ_WRITE, embed_bytes);
        cl::Buffer d_dv(context, CL_MEM_READ_WRITE, embed_bytes);
        cl::Buffer d_EH(context, CL_MEM_READ_WRITE, embed_bytes);
        cl::Buffer d_EV_current(context, CL_MEM_READ_WRITE, embed_bytes); // For EV[count]
        cl::Buffer d_hor_inputs(context, CL_MEM_READ_WRITE, embed_bytes);
        cl::Buffer d_ver_inputs(context, CL_MEM_READ_WRITE, embed_bytes);
        cl::Buffer d_hor_output(context, CL_MEM_READ_WRITE, embed_bytes);
        cl::Buffer d_ver_output(context, CL_MEM_READ_WRITE, embed_bytes);
        cl::Buffer d_relu_hor_output(context, CL_MEM_READ_WRITE, embed_bytes);
        cl::Buffer d_relu_ver_output(context, CL_MEM_READ_WRITE, embed_bytes);
        cl::Buffer d_mlp_bufferA_hor(context, CL_MEM_READ_WRITE, embed_bytes);
        cl::Buffer d_mlp_bufferB_hor(context, CL_MEM_READ_WRITE, embed_bytes);
        cl::Buffer d_mlp_bufferA_ver(context, CL_MEM_READ_WRITE, embed_bytes);
        cl::Buffer d_mlp_bufferB_ver(context, CL_MEM_READ_WRITE, embed_bytes);
        cl::Buffer d_mlp_pre_activation(context, CL_MEM_READ_WRITE, embed_bytes);

        // Initialize accumulators
        float zero = 0.0f;
        queue.enqueueFillBuffer(d_dh_accum, zero, 0, accum_bytes);
        queue.enqueueFillBuffer(d_dv_accum, zero, 0, accum_bytes);

        // --- Flatten Host Data & Copy H->D ---
        // ** Assuming K, Q, KdotQ members contain the 'count'-sized data for this block **
        std::vector<float> flat_K = flatten(K);
        std::vector<float> flat_Q = flatten(Q);
        std::vector<float> flat_KdotQ = flatten(KdotQ);
        std::vector<float> flat_MH_hxd, flat_MV_hxd;
        transposeFlattenMatrix(MH.a, flat_MH_hxd, d, h);
        transposeFlattenMatrix(MV.a, flat_MV_hxd, d, h);

        queue.enqueueWriteBuffer(d_K, CL_TRUE, 0, k_bytes, flat_K.data());
        queue.enqueueWriteBuffer(d_Q, CL_TRUE, 0, q_bytes, flat_Q.data());
        queue.enqueueWriteBuffer(d_KdotQ, CL_TRUE, 0, kdotq_bytes, flat_KdotQ.data());
        queue.enqueueWriteBuffer(d_MH_hxd, CL_TRUE, 0, proj_mat_bytes, flat_MH_hxd.data());
        queue.enqueueWriteBuffer(d_MV_hxd, CL_TRUE, 0, proj_mat_bytes, flat_MV_hxd.data());
        queue.enqueueWriteBuffer(d_EH, CL_TRUE, 0, embed_bytes, EH.data());
        queue.enqueueWriteBuffer(d_EV_current, CL_TRUE, 0, embed_bytes, EV[count].data()); // Copy EV[count]

        // --- Kernel Launches ---
        const size_t local_work_size_1d = 256;

        // 1. Calculate Attention Weights: head = LOTA(KdotQ)
        cl::Kernel lota_kernel = context_obj.kernels.at("clLOTA2d");
        size_t totalElementsLOTA = static_cast<size_t>(count) * count;
        if (totalElementsLOTA > 0) {
            // Same launch considerations as in the first overload, using 'count'
            size_t global_lota_raw = totalElementsLOTA;
            size_t local_lota_clamped = std::min(global_lota_raw, local_work_size_1d);
            if (local_lota_clamped == 0) local_lota_clamped = 1;
            size_t global_lota_padded = ((global_lota_raw + local_lota_clamped - 1) / local_lota_clamped) * local_lota_clamped;

            cl::NDRange global_lota(global_lota_padded);
            cl::NDRange local_lota(local_lota_clamped);

            lota_kernel.setArg(0, d_KdotQ);
            lota_kernel.setArg(1, d_head);
            lota_kernel.setArg(2, count); // rows
            lota_kernel.setArg(3, count); // cols
            queue.enqueueNDRangeKernel(lota_kernel, cl::NullRange, global_lota, local_lota);
        }


        // 2. Compute Head Row/Column Sums (Masked)
        cl::Kernel sums_kernel = context_obj.kernels.at("computeHeadSumsMaskedKernel");
        size_t global_sums_raw = static_cast<size_t>(count);
        size_t global_sums_padded = ((global_sums_raw + local_work_size_1d - 1) / local_work_size_1d) * local_work_size_1d;
        cl::NDRange global_sums(global_sums_padded);
        cl::NDRange local_sums(local_work_size_1d);
        cl_int cl_isSelfAttention = isSelfAttention;
        sums_kernel.setArg(0, d_head);
        sums_kernel.setArg(1, d_row_sums);
        sums_kernel.setArg(2, d_col_sums);
        sums_kernel.setArg(3, count); // Use count
        sums_kernel.setArg(4, cl_isSelfAttention);
        queue.enqueueNDRangeKernel(sums_kernel, cl::NullRange, global_sums, local_sums);

        // 3. Accumulate Weighted K and Q vectors
        cl::Kernel accum_kernel = context_obj.kernels.at("accumulateWeightedVectorsKernel");
        size_t global_accum_raw = static_cast<size_t>(count);
        size_t global_accum_padded = ((global_accum_raw + local_work_size_1d - 1) / local_work_size_1d) * local_work_size_1d; // Launch per token
        cl::NDRange global_accum(global_accum_padded);
        cl::NDRange local_accum(local_work_size_1d);
        accum_kernel.setArg(0, d_row_sums);
        accum_kernel.setArg(1, d_col_sums);
        accum_kernel.setArg(2, d_K);
        accum_kernel.setArg(3, d_Q);
        accum_kernel.setArg(4, d_dh_accum);
        accum_kernel.setArg(5, d_dv_accum);
        accum_kernel.setArg(6, count); // Use count
        accum_kernel.setArg(7, h);
        queue.enqueueNDRangeKernel(accum_kernel, cl::NullRange, global_accum, local_accum);

        // 4. Project accumulated vectors
        cl::Kernel proj_kernel = context_obj.kernels.at("kernelLayerForward");
        size_t global_proj_raw = static_cast<size_t>(d);
        size_t global_proj_padded = ((global_proj_raw + local_work_size_1d - 1) / local_work_size_1d) * local_work_size_1d;
        cl::NDRange global_proj(global_proj_padded);
        cl::NDRange local_proj(local_work_size_1d); // Or cl::NullRange
        proj_kernel.setArg(3, h); // input_size
        proj_kernel.setArg(4, d); // output_size

        proj_kernel.setArg(0, d_dh_accum);
        proj_kernel.setArg(1, d_MH_hxd);
        proj_kernel.setArg(2, d_dh);
        queue.enqueueNDRangeKernel(proj_kernel, cl::NullRange, global_proj, local_proj);

        proj_kernel.setArg(0, d_dv_accum);
        proj_kernel.setArg(1, d_MV_hxd);
        proj_kernel.setArg(2, d_dv);
        queue.enqueueNDRangeKernel(proj_kernel, cl::NullRange, global_proj, local_proj);

        // 5. Prepare MLP inputs
        cl::Kernel add_kernel = context_obj.kernels.at("vectorAddKernel");
        size_t global_add_raw = static_cast<size_t>(d);
        size_t global_add_padded = ((global_add_raw + local_work_size_1d - 1) / local_work_size_1d) * local_work_size_1d;
        cl::NDRange global_add(global_add_padded);
        cl::NDRange local_add(local_work_size_1d); // Or cl::NullRange
        add_kernel.setArg(3, d);

        add_kernel.setArg(0, d_EH);
        add_kernel.setArg(1, d_dh);
        add_kernel.setArg(2, d_hor_inputs);
        queue.enqueueNDRangeKernel(add_kernel, cl::NullRange, global_add, local_add);

        add_kernel.setArg(0, d_EV_current); // Use EV[count] buffer
        add_kernel.setArg(1, d_dv);
        add_kernel.setArg(2, d_ver_inputs);
        queue.enqueueNDRangeKernel(add_kernel, cl::NullRange, global_add, local_add);

        queue.finish();

        // --- 6. Run MLPs Forward ---
        // (Identical launch logic to the first overload, using 'd' for sizes)
        try {
            cl::Kernel mlp_fwd_kernel = context_obj.kernels.at("kernelLayerForward");
            cl::Kernel sigmoid_kernel = context_obj.kernels.at("clSigmoid1d");

            queue.enqueueCopyBuffer(d_hor_inputs, d_mlp_bufferA_hor, 0, 0, embed_bytes);
            queue.enqueueCopyBuffer(d_ver_inputs, d_mlp_bufferA_ver, 0, 0, embed_bytes);

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
                    std::vector<float> flat_weights_hor = flatten(hor.weights[layer_idx]);
                    size_t weights_bytes = flat_weights_hor.size() * sizeof(float);
                    cl::Buffer d_mlp_weights(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, weights_bytes, flat_weights_hor.data());
                    cl::Buffer& target_output_buffer = is_last_layer ? d_hor_output : d_mlp_pre_activation;
                    mlp_fwd_kernel.setArg(0, current_in_hor);
                    mlp_fwd_kernel.setArg(1, d_mlp_weights);
                    mlp_fwd_kernel.setArg(2, target_output_buffer);
                    mlp_fwd_kernel.setArg(3, input_size);
                    mlp_fwd_kernel.setArg(4, output_size);
                    queue.enqueueNDRangeKernel(mlp_fwd_kernel, cl::NullRange, global_mlp, local_mlp);
                    if (!is_last_layer) {
                        sigmoid_kernel.setArg(0, d_mlp_pre_activation);
                        sigmoid_kernel.setArg(1, current_out_hor);
                        sigmoid_kernel.setArg(2, output_size);
                        queue.enqueueNDRangeKernel(sigmoid_kernel, cl::NullRange, global_mlp, local_mlp);
                        std::swap(current_in_hor, current_out_hor);
                    }
                }
            // Ver
                {
                    std::vector<float> flat_weights_ver = flatten(ver.weights[layer_idx]);
                    size_t weights_bytes = flat_weights_ver.size() * sizeof(float);
                    cl::Buffer d_mlp_weights(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, weights_bytes, flat_weights_ver.data());
                    cl::Buffer& target_output_buffer = is_last_layer ? d_ver_output : d_mlp_pre_activation;
                    mlp_fwd_kernel.setArg(0, current_in_ver);
                    mlp_fwd_kernel.setArg(1, d_mlp_weights);
                    mlp_fwd_kernel.setArg(2, target_output_buffer);
                    queue.enqueueNDRangeKernel(mlp_fwd_kernel, cl::NullRange, global_mlp, local_mlp);
                    if (!is_last_layer) {
                        sigmoid_kernel.setArg(0, d_mlp_pre_activation);
                        sigmoid_kernel.setArg(1, current_out_ver);
                        queue.enqueueNDRangeKernel(sigmoid_kernel, cl::NullRange, global_mlp, local_mlp);
                        std::swap(current_in_ver, current_out_ver);
                    }
                }
            }
            queue.finish();
        }
        catch (const cl::Error& err) {
            std::cerr << "OpenCL Error during MLP processing (blockIdx=" << blockIdx << "): " << err.what() << " (" << err.err() << ")" << std::endl;
            throw;
        }
        catch (const std::exception& e) {
        std::cerr << "Error during MLP processing (blockIdx=" << blockIdx << "): " << e.what() << std::endl;
        throw;
        }

        // --- Free MLP Intermediate Buffers --- (Handled by RAII)

        // 7. Apply ReLU
        cl::Kernel relu_kernel = context_obj.kernels.at("clReLU1d");
        // Use global_add and local_add from step 5
        relu_kernel.setArg(2, d);

        relu_kernel.setArg(0, d_hor_output);
        relu_kernel.setArg(1, d_relu_hor_output);
        queue.enqueueNDRangeKernel(relu_kernel, cl::NullRange, global_add, local_add);

        relu_kernel.setArg(0, d_ver_output);
        relu_kernel.setArg(1, d_relu_ver_output);
        queue.enqueueNDRangeKernel(relu_kernel, cl::NullRange, global_add, local_add);

        queue.finish();

        // 8. Final Residual Update
        // Use add_kernel from step 5

        add_kernel.setArg(0, d_EH);
        add_kernel.setArg(1, d_relu_hor_output);
        add_kernel.setArg(2, d_EH); // In-place update
        queue.enqueueNDRangeKernel(add_kernel, cl::NullRange, global_add, local_add);

        add_kernel.setArg(0, d_EV_current); // Use EV[count] buffer
        add_kernel.setArg(1, d_relu_ver_output);
        add_kernel.setArg(2, d_EV_current); // Update EV[count] buffer (in-place)
        queue.enqueueNDRangeKernel(add_kernel, cl::NullRange, global_add, local_add);

        queue.finish();

        // --- Copy Results D->H ---
        queue.enqueueReadBuffer(d_EH, CL_TRUE, 0, embed_bytes, EH.data());
        queue.enqueueReadBuffer(d_EV_current, CL_TRUE, 0, embed_bytes, EV[count].data()); // Copy back to EV[count]

        // --- Free Device Memory --- (Handled by RAII)

    }
    catch (const cl::Error& err) {
        std::cerr << "OpenCL Error in attention clforprop(..., blockIdx=" << blockIdx << "): " << err.what() << " (" << err.err() << ")" << std::endl;
        throw;
    }
    catch (const std::exception& e) {
        std::cerr << "Standard Exception in attention clforprop(..., blockIdx=" << blockIdx << "): " << e.what() << std::endl;
        throw; // Re-throw the exception
    }
}

#endif // USE_OPENCL
