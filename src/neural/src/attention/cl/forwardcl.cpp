#ifdef USE_OPENCL

#ifndef CL_HPP_ENABLE_EXCEPTIONS
    #define CL_HPP_ENABLE_EXCEPTIONS
#endif
#ifndef CL_HPP_TARGET_OPENCL_VERSION
    #define CL_HPP_TARGET_OPENCL_VERSION 300 // Or the version you are targeting
#endif

#include "include/attention.hpp" // Includes mlp.hpp and maths.hpp indirectly or directly
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

    // --- Device Memory Buffers ---
    cl_mem d_K = nullptr, d_Q = nullptr, d_KdotQ = nullptr, d_head = nullptr;
    cl_mem d_row_sums = nullptr, d_col_sums = nullptr;
    cl_mem d_dh_accum = nullptr, d_dv_accum = nullptr;
    cl_mem d_MH_hxd = nullptr, d_MV_hxd = nullptr; // Transposed: h x d
    cl_mem d_dh = nullptr, d_dv = nullptr;         // Projected result: 1 x d (size d)
    cl_mem d_EH = nullptr, d_EV_current = nullptr; // Corresponds to EV[n]
    cl_mem d_hor_inputs = nullptr, d_ver_inputs = nullptr; // Inputs to MLP after residual add
    cl_mem d_hor_output = nullptr, d_ver_output = nullptr; // Final MLP outputs (pre-ReLU)
    cl_mem d_relu_hor_output = nullptr, d_relu_ver_output = nullptr; // After ReLU

    // MLP Intermediate Buffers
    cl_mem d_mlp_bufferA_hor = nullptr, d_mlp_bufferB_hor = nullptr;
    cl_mem d_mlp_bufferA_ver = nullptr, d_mlp_bufferB_ver = nullptr;
    cl_mem d_mlp_pre_activation = nullptr;
    cl_mem d_mlp_weights = nullptr; // Reused

    cl_int err;

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

        d_K = cl_create_buffer(context, CL_MEM_READ_ONLY, k_bytes, nullptr, err); CL_CHECK(err);
        d_Q = cl_create_buffer(context, CL_MEM_READ_ONLY, q_bytes, nullptr, err); CL_CHECK(err);
        d_KdotQ = cl_create_buffer(context, CL_MEM_READ_ONLY, kdotq_bytes, nullptr, err); CL_CHECK(err);
        d_head = cl_create_buffer(context, CL_MEM_READ_WRITE, head_bytes, nullptr, err); CL_CHECK(err); // LOTA output
        d_row_sums = cl_create_buffer(context, CL_MEM_READ_WRITE, sums_bytes, nullptr, err); CL_CHECK(err);
        d_col_sums = cl_create_buffer(context, CL_MEM_READ_WRITE, sums_bytes, nullptr, err); CL_CHECK(err);
        d_dh_accum = cl_create_buffer(context, CL_MEM_READ_WRITE, accum_bytes, nullptr, err); CL_CHECK(err);
        d_dv_accum = cl_create_buffer(context, CL_MEM_READ_WRITE, accum_bytes, nullptr, err); CL_CHECK(err);
        d_MH_hxd = cl_create_buffer(context, CL_MEM_READ_ONLY, proj_mat_bytes, nullptr, err); CL_CHECK(err);
        d_MV_hxd = cl_create_buffer(context, CL_MEM_READ_ONLY, proj_mat_bytes, nullptr, err); CL_CHECK(err);
        d_dh = cl_create_buffer(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, err); CL_CHECK(err); // Projection output
        d_dv = cl_create_buffer(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, err); CL_CHECK(err); // Projection output
        d_EH = cl_create_buffer(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, err); CL_CHECK(err); // Read for add, write final result
        d_EV_current = cl_create_buffer(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, err); CL_CHECK(err); // Read for add, write final result
        d_hor_inputs = cl_create_buffer(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, err); CL_CHECK(err); // MLP input after add
        d_ver_inputs = cl_create_buffer(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, err); CL_CHECK(err); // MLP input after add
        d_hor_output = cl_create_buffer(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, err); CL_CHECK(err); // MLP final output (pre-ReLU)
        d_ver_output = cl_create_buffer(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, err); CL_CHECK(err); // MLP final output (pre-ReLU)
        d_relu_hor_output = cl_create_buffer(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, err); CL_CHECK(err); // After ReLU
        d_relu_ver_output = cl_create_buffer(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, err); CL_CHECK(err); // After ReLU

        // Allocate MLP Intermediate Buffers
        d_mlp_bufferA_hor = cl_create_buffer(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, err); CL_CHECK(err);
        d_mlp_bufferB_hor = cl_create_buffer(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, err); CL_CHECK(err);
        d_mlp_bufferA_ver = cl_create_buffer(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, err); CL_CHECK(err);
        d_mlp_bufferB_ver = cl_create_buffer(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, err); CL_CHECK(err);
        d_mlp_pre_activation = cl_create_buffer(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, err); CL_CHECK(err);

        // Initialize accumulators to zero
        float zero = 0.0f;
        cl_fill_buffer(queue, d_dh_accum, &zero, sizeof(float), 0, accum_bytes);
        cl_fill_buffer(queue, d_dv_accum, &zero, sizeof(float), 0, accum_bytes);

        // --- Flatten Host Data & Copy H->D ---
        std::vector<float> flat_K = flatten(K);
        std::vector<float> flat_Q = flatten(Q);
        std::vector<float> flat_KdotQ = flatten(KdotQ);
        std::vector<float> flat_MH_hxd, flat_MV_hxd;
        transposeFlattenMatrix(MH.a, flat_MH_hxd, d, h);
        transposeFlattenMatrix(MV.a, flat_MV_hxd, d, h);

        cl_write_buffer(queue, d_K, k_bytes, flat_K.data());
        cl_write_buffer(queue, d_Q, q_bytes, flat_Q.data());
        cl_write_buffer(queue, d_KdotQ, kdotq_bytes, flat_KdotQ.data());
        cl_write_buffer(queue, d_MH_hxd, proj_mat_bytes, flat_MH_hxd.data());
        cl_write_buffer(queue, d_MV_hxd, proj_mat_bytes, flat_MV_hxd.data());
        cl_write_buffer(queue, d_EH, embed_bytes, EH.data());
        cl_write_buffer(queue, d_EV_current, embed_bytes, EV[n].data());

        // --- Kernel Launches ---
        const size_t local_work_size_1d = 256; // General purpose block size

        // 1. Calculate Attention Weights: head = LOTA(KdotQ)
        // Use clLOTA2d as KdotQ is 2D
        cl_kernel lota_kernel = kernels.at("clLOTA2d"); // Make sure this kernel name is correct
        size_t totalElementsLOTA = static_cast<size_t>(n) * n;
        size_t global_work_size_lota[1] = { (totalElementsLOTA + local_work_size_1d - 1) / local_work_size_1d * local_work_size_1d };
        size_t local_work_size_lota[1] = { local_work_size_1d };
        // NOTE: clLOTA2d expects a single work-group for reduction. Adjust launch if needed.
        // If n*n exceeds max work group size, clLOTA2d needs modification or a different approach.
        // Assuming n*n fits for now.
        if (totalElementsLOTA > 0) {
             if (totalElementsLOTA > local_work_size_1d) {
                 std::cerr << "Warning: LOTA size (" << totalElementsLOTA << ") > local size (" << local_work_size_1d << "). Reduction might be incorrect." << std::endl;
                 // Adjust global/local size for single group launch if possible/necessary
                 global_work_size_lota[0] = local_work_size_1d; // Or max allowed size
             } else {
                 global_work_size_lota[0] = totalElementsLOTA;
                 local_work_size_lota[0] = totalElementsLOTA;
             }
            cl_set_kernel_arg(lota_kernel, 0, sizeof(cl_mem), &d_KdotQ);
            cl_set_kernel_arg(lota_kernel, 1, sizeof(cl_mem), &d_head);
            cl_set_kernel_arg(lota_kernel, 2, sizeof(cl_int), &n); // rows
            cl_set_kernel_arg(lota_kernel, 3, sizeof(cl_int), &n); // cols
            cl_enqueue_nd_range_kernel(queue, lota_kernel, 1, nullptr, global_work_size_lota, local_work_size_lota);
        }


        // 2. Compute Head Row/Column Sums (Masked)
        cl_kernel sums_kernel = kernels.at("computeHeadSumsMaskedKernel");
        size_t global_work_size_sums[1] = { (static_cast<size_t>(n) + local_work_size_1d - 1) / local_work_size_1d * local_work_size_1d };
        size_t local_work_size_sums[1] = { local_work_size_1d };
        cl_int cl_isSelfAttention = isSelfAttention; // Convert bool to cl_int
        cl_set_kernel_arg(sums_kernel, 0, sizeof(cl_mem), &d_head);
        cl_set_kernel_arg(sums_kernel, 1, sizeof(cl_mem), &d_row_sums);
        cl_set_kernel_arg(sums_kernel, 2, sizeof(cl_mem), &d_col_sums);
        cl_set_kernel_arg(sums_kernel, 3, sizeof(cl_int), &n);
        cl_set_kernel_arg(sums_kernel, 4, sizeof(cl_int), &cl_isSelfAttention);
        cl_enqueue_nd_range_kernel(queue, sums_kernel, 1, nullptr, global_work_size_sums, local_work_size_sums);

        // 3. Accumulate Weighted K and Q vectors
        // NOTE: Requires float atomics support (OpenCL 2.0+ or extensions)
        cl_kernel accum_kernel = kernels.at("accumulateWeightedVectorsKernel");
        size_t global_work_size_accum[1] = { (static_cast<size_t>(h) + local_work_size_1d - 1) / local_work_size_1d * local_work_size_1d };
        size_t local_work_size_accum[1] = { local_work_size_1d };
        cl_set_kernel_arg(accum_kernel, 0, sizeof(cl_mem), &d_row_sums);
        cl_set_kernel_arg(accum_kernel, 1, sizeof(cl_mem), &d_col_sums);
        cl_set_kernel_arg(accum_kernel, 2, sizeof(cl_mem), &d_K);
        cl_set_kernel_arg(accum_kernel, 3, sizeof(cl_mem), &d_Q);
        cl_set_kernel_arg(accum_kernel, 4, sizeof(cl_mem), &d_dh_accum);
        cl_set_kernel_arg(accum_kernel, 5, sizeof(cl_mem), &d_dv_accum);
        cl_set_kernel_arg(accum_kernel, 6, sizeof(cl_int), &n);
        cl_set_kernel_arg(accum_kernel, 7, sizeof(cl_int), &h);
        cl_enqueue_nd_range_kernel(queue, accum_kernel, 1, nullptr, global_work_size_accum, local_work_size_accum);

        // 4. Project accumulated vectors: dh = dot(dh_accum, MH), dv = dot(dv_accum, MV)
        // Use kernelLayerForward (matrix-vector): output[j] = sum(input[i] * weights[j*input_size + i])
        // Here: input=dh_accum (size h), weights=MH_hxd (d rows, h cols), output=dh (size d)
        // Need to transpose MH_hxd conceptually or adjust kernel access.
        // Let's assume kernelLayerForward expects weights[output_idx * input_size + input_idx]
        // Input: d_dh_accum (size h)
        // Weights: d_MH_hxd (size h*d, interpreted as d rows, h cols)
        // Output: d_dh (size d)
        cl_kernel proj_kernel = kernels.at("kernelLayerForward"); // Assuming this is matrix-vector
        size_t global_work_size_proj[1] = { (static_cast<size_t>(d) + local_work_size_1d - 1) / local_work_size_1d * local_work_size_1d };
        size_t local_work_size_proj[1] = { local_work_size_1d };

        // dh = dh_accum * MH_hxd (conceptually) -> kernelLayerForward(d_dh_accum, d_MH_hxd, d_dh, h, d)
        cl_set_kernel_arg(proj_kernel, 0, sizeof(cl_mem), &d_dh_accum); // input vector (size h)
        cl_set_kernel_arg(proj_kernel, 1, sizeof(cl_mem), &d_MH_hxd);   // weight matrix (d rows, h cols)
        cl_set_kernel_arg(proj_kernel, 2, sizeof(cl_mem), &d_dh);       // output vector (size d)
        cl_set_kernel_arg(proj_kernel, 3, sizeof(cl_int), &h);          // input_size
        cl_set_kernel_arg(proj_kernel, 4, sizeof(cl_int), &d);          // output_size
        cl_enqueue_nd_range_kernel(queue, proj_kernel, 1, nullptr, global_work_size_proj, local_work_size_proj);

        // dv = dv_accum * MV_hxd (conceptually) -> kernelLayerForward(d_dv_accum, d_MV_hxd, d_dv, h, d)
        cl_set_kernel_arg(proj_kernel, 0, sizeof(cl_mem), &d_dv_accum); // input vector (size h)
        cl_set_kernel_arg(proj_kernel, 1, sizeof(cl_mem), &d_MV_hxd);   // weight matrix (d rows, h cols)
        cl_set_kernel_arg(proj_kernel, 2, sizeof(cl_mem), &d_dv);       // output vector (size d)
        // Args 3 (h) and 4 (d) are already set
        cl_enqueue_nd_range_kernel(queue, proj_kernel, 1, nullptr, global_work_size_proj, local_work_size_proj);

        // 5. Prepare MLP inputs (Residual Add): hor_input = EH + dh, ver_input = EV_current + dv
        cl_kernel add_kernel = kernels.at("vectorAddKernel");
        size_t global_work_size_add[1] = { (static_cast<size_t>(d) + local_work_size_1d - 1) / local_work_size_1d * local_work_size_1d };
        size_t local_work_size_add[1] = { local_work_size_1d };

        // hor_input = EH + dh
        cl_set_kernel_arg(add_kernel, 0, sizeof(cl_mem), &d_EH);
        cl_set_kernel_arg(add_kernel, 1, sizeof(cl_mem), &d_dh);
        cl_set_kernel_arg(add_kernel, 2, sizeof(cl_mem), &d_hor_inputs);
        cl_set_kernel_arg(add_kernel, 3, sizeof(cl_int), &d);
        cl_enqueue_nd_range_kernel(queue, add_kernel, 1, nullptr, global_work_size_add, local_work_size_add);

        // ver_input = EV_current + dv
        cl_set_kernel_arg(add_kernel, 0, sizeof(cl_mem), &d_EV_current);
        cl_set_kernel_arg(add_kernel, 1, sizeof(cl_mem), &d_dv);
        cl_set_kernel_arg(add_kernel, 2, sizeof(cl_mem), &d_ver_inputs);
        // Arg 3 (d) is already set
        cl_enqueue_nd_range_kernel(queue, add_kernel, 1, nullptr, global_work_size_add, local_work_size_add);

        cl_finish(queue); // Sync before MLP uses these inputs

        // --- 6. Run MLPs Forward ---
        try {
            // Get kernels
            cl_kernel mlp_fwd_kernel = kernels.at("kernelLayerForward");
            cl_kernel sigmoid_kernel = kernels.at("clSigmoid1d"); // Assuming 1D version

            // Initial input copies using clEnqueueCopyBuffer
            err = clEnqueueCopyBuffer(queue, d_hor_inputs, d_mlp_bufferA_hor, 0, 0, embed_bytes, 0, nullptr, nullptr); CL_CHECK(err);
            err = clEnqueueCopyBuffer(queue, d_ver_inputs, d_mlp_bufferA_ver, 0, 0, embed_bytes, 0, nullptr, nullptr); CL_CHECK(err);

            // Pointers for ping-pong buffers
            cl_mem d_current_in_hor = d_mlp_bufferA_hor;
            cl_mem d_current_out_hor = d_mlp_bufferB_hor;
            cl_mem d_current_in_ver = d_mlp_bufferA_ver;
            cl_mem d_current_out_ver = d_mlp_bufferB_ver;

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

            size_t global_work_size_mlp[1] = { (static_cast<size_t>(d) + local_work_size_1d - 1) / local_work_size_1d * local_work_size_1d };
            size_t local_work_size_mlp[1] = { local_work_size_1d };

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

                    d_mlp_weights = cl_create_buffer(context, CL_MEM_READ_ONLY, weights_bytes, nullptr, err); CL_CHECK(err);
                    cl_write_buffer(queue, d_mlp_weights, weights_bytes, flat_weights_hor.data());

                    cl_mem target_output_buffer = is_last_layer ? d_hor_output : d_mlp_pre_activation;

                    cl_set_kernel_arg(mlp_fwd_kernel, 0, sizeof(cl_mem), &d_current_in_hor);
                    cl_set_kernel_arg(mlp_fwd_kernel, 1, sizeof(cl_mem), &d_mlp_weights);
                    cl_set_kernel_arg(mlp_fwd_kernel, 2, sizeof(cl_mem), &target_output_buffer);
                    cl_set_kernel_arg(mlp_fwd_kernel, 3, sizeof(cl_int), &input_size);
                    cl_set_kernel_arg(mlp_fwd_kernel, 4, sizeof(cl_int), &output_size);
                    cl_enqueue_nd_range_kernel(queue, mlp_fwd_kernel, 1, nullptr, global_work_size_mlp, local_work_size_mlp);

                    if (!is_last_layer) {
                        cl_set_kernel_arg(sigmoid_kernel, 0, sizeof(cl_mem), &d_mlp_pre_activation);
                        cl_set_kernel_arg(sigmoid_kernel, 1, sizeof(cl_mem), &d_current_out_hor);
                        cl_set_kernel_arg(sigmoid_kernel, 2, sizeof(cl_int), &output_size);
                        cl_enqueue_nd_range_kernel(queue, sigmoid_kernel, 1, nullptr, global_work_size_mlp, local_work_size_mlp);

                        // Swap hor buffers
                        cl_mem temp = d_current_in_hor;
                        d_current_in_hor = d_current_out_hor;
                        d_current_out_hor = temp;
                    }
                    cl_release_mem_object(d_mlp_weights);
                    d_mlp_weights = nullptr;
                }

                // --- Process Vertical MLP Layer ---
                {
                    std::vector<float> flat_weights_ver = flatten(ver.weights[layer_idx]);
                    size_t weights_bytes = flat_weights_ver.size() * sizeof(float);
                     if (weights_bytes != static_cast<size_t>(input_size) * output_size * sizeof(float)) {
                        throw std::runtime_error("MLP weight dimension mismatch (ver).");
                    }

                    d_mlp_weights = cl_create_buffer(context, CL_MEM_READ_ONLY, weights_bytes, nullptr, err); CL_CHECK(err);
                    cl_write_buffer(queue, d_mlp_weights, weights_bytes, flat_weights_ver.data());

                    cl_mem target_output_buffer = is_last_layer ? d_ver_output : d_mlp_pre_activation;

                    cl_set_kernel_arg(mlp_fwd_kernel, 0, sizeof(cl_mem), &d_current_in_ver);
                    cl_set_kernel_arg(mlp_fwd_kernel, 1, sizeof(cl_mem), &d_mlp_weights);
                    cl_set_kernel_arg(mlp_fwd_kernel, 2, sizeof(cl_mem), &target_output_buffer);
                    // Args 3 (input_size) and 4 (output_size) are already set
                    cl_enqueue_nd_range_kernel(queue, mlp_fwd_kernel, 1, nullptr, global_work_size_mlp, local_work_size_mlp);

                    if (!is_last_layer) {
                        cl_set_kernel_arg(sigmoid_kernel, 0, sizeof(cl_mem), &d_mlp_pre_activation);
                        cl_set_kernel_arg(sigmoid_kernel, 1, sizeof(cl_mem), &d_current_out_ver);
                        // Arg 2 (output_size) is already set
                        cl_enqueue_nd_range_kernel(queue, sigmoid_kernel, 1, nullptr, global_work_size_mlp, local_work_size_mlp);

                        // Swap ver buffers
                        cl_mem temp = d_current_in_ver;
                        d_current_in_ver = d_current_out_ver;
                        d_current_out_ver = temp;
                    }
                    cl_release_mem_object(d_mlp_weights);
                    d_mlp_weights = nullptr;
                }
            } // End loop over layers
            cl_finish(queue); // Sync after all MLP layers
        }
        catch (const std::exception& e) {
            std::cerr << "Error during OpenCL MLP processing: " << e.what() << std::endl;
            cl_release_mem_object(d_mlp_weights); // Ensure cleanup if error occurs mid-layer
            throw;
        }

        // --- Free MLP Intermediate Buffers ---
        cl_release_mem_object(d_mlp_bufferA_hor); d_mlp_bufferA_hor = nullptr;
        cl_release_mem_object(d_mlp_bufferB_hor); d_mlp_bufferB_hor = nullptr;
        cl_release_mem_object(d_mlp_bufferA_ver); d_mlp_bufferA_ver = nullptr;
        cl_release_mem_object(d_mlp_bufferB_ver); d_mlp_bufferB_ver = nullptr;
        cl_release_mem_object(d_mlp_pre_activation); d_mlp_pre_activation = nullptr;

        // 7. Apply ReLU to MLP outputs
        cl_kernel relu_kernel = kernels.at("clReLU1d");
        // Use global_work_size_add, local_work_size_add from step 5

        // ReLU(hor_output) -> d_relu_hor_output
        cl_set_kernel_arg(relu_kernel, 0, sizeof(cl_mem), &d_hor_output);
        cl_set_kernel_arg(relu_kernel, 1, sizeof(cl_mem), &d_relu_hor_output);
        cl_set_kernel_arg(relu_kernel, 2, sizeof(cl_int), &d);
        cl_enqueue_nd_range_kernel(queue, relu_kernel, 1, nullptr, global_work_size_add, local_work_size_add);

        // ReLU(ver_output) -> d_relu_ver_output
        cl_set_kernel_arg(relu_kernel, 0, sizeof(cl_mem), &d_ver_output);
        cl_set_kernel_arg(relu_kernel, 1, sizeof(cl_mem), &d_relu_ver_output);
        // Arg 2 (d) is already set
        cl_enqueue_nd_range_kernel(queue, relu_kernel, 1, nullptr, global_work_size_add, local_work_size_add);

        cl_finish(queue); // Sync before final add

        // 8. Final Residual Update: EH = EH + ReLU(hor_output), EV[n] = EV[n] + ReLU(ver_output)
        // Use add_kernel from step 5

        // EH = EH + relu_hor_output (update d_EH in-place)
        cl_set_kernel_arg(add_kernel, 0, sizeof(cl_mem), &d_EH);
        cl_set_kernel_arg(add_kernel, 1, sizeof(cl_mem), &d_relu_hor_output);
        cl_set_kernel_arg(add_kernel, 2, sizeof(cl_mem), &d_EH); // Output is d_EH
        // Arg 3 (d) is already set
        cl_enqueue_nd_range_kernel(queue, add_kernel, 1, nullptr, global_work_size_add, local_work_size_add);

        // EV[n] = EV[n] + relu_ver_output (update d_EV_current in-place)
        cl_set_kernel_arg(add_kernel, 0, sizeof(cl_mem), &d_EV_current);
        cl_set_kernel_arg(add_kernel, 1, sizeof(cl_mem), &d_relu_ver_output);
        cl_set_kernel_arg(add_kernel, 2, sizeof(cl_mem), &d_EV_current); // Output is d_EV_current
        // Arg 3 (d) is already set
        cl_enqueue_nd_range_kernel(queue, add_kernel, 1, nullptr, global_work_size_add, local_work_size_add);

        cl_finish(queue); // Sync before D->H copy

        // --- Copy Results D->H ---
        cl_read_buffer(queue, d_EH, embed_bytes, EH.data());
        cl_read_buffer(queue, d_EV_current, embed_bytes, EV[n].data());

        // --- Free Device Memory ---
        cl_release_mem_object(d_K); cl_release_mem_object(d_Q); cl_release_mem_object(d_KdotQ); cl_release_mem_object(d_head);
        cl_release_mem_object(d_row_sums); cl_release_mem_object(d_col_sums);
        cl_release_mem_object(d_dh_accum); cl_release_mem_object(d_dv_accum);
        cl_release_mem_object(d_MH_hxd); cl_release_mem_object(d_MV_hxd);
        cl_release_mem_object(d_dh); cl_release_mem_object(d_dv);
        cl_release_mem_object(d_EH); cl_release_mem_object(d_EV_current);
        cl_release_mem_object(d_hor_inputs); cl_release_mem_object(d_ver_inputs);
        cl_release_mem_object(d_hor_output); cl_release_mem_object(d_ver_output);
        cl_release_mem_object(d_relu_hor_output); cl_release_mem_object(d_relu_ver_output);
        // MLP intermediate buffers were freed earlier or are null

    }
    catch (const std::exception& e) {
        std::cerr << "OpenCL Exception in attention clforprop(..., tokenCount): " << e.what() << std::endl;
        // --- Cleanup on Error ---
        cl_release_mem_object(d_K); cl_release_mem_object(d_Q); cl_release_mem_object(d_KdotQ); cl_release_mem_object(d_head);
        cl_release_mem_object(d_row_sums); cl_release_mem_object(d_col_sums);
        cl_release_mem_object(d_dh_accum); cl_release_mem_object(d_dv_accum);
        cl_release_mem_object(d_MH_hxd); cl_release_mem_object(d_MV_hxd);
        cl_release_mem_object(d_dh); cl_release_mem_object(d_dv);
        cl_release_mem_object(d_EH); cl_release_mem_object(d_EV_current);
        cl_release_mem_object(d_hor_inputs); cl_release_mem_object(d_ver_inputs);
        cl_release_mem_object(d_hor_output); cl_release_mem_object(d_ver_output);
        cl_release_mem_object(d_relu_hor_output); cl_release_mem_object(d_relu_ver_output);
        cl_release_mem_object(d_mlp_bufferA_hor); cl_release_mem_object(d_mlp_bufferB_hor);
        cl_release_mem_object(d_mlp_bufferA_ver); cl_release_mem_object(d_mlp_bufferB_ver);
        cl_release_mem_object(d_mlp_pre_activation);
        cl_release_mem_object(d_mlp_weights); // Release if allocated during error
        throw; // Re-throw the exception
    }
}


/**
 * @brief OpenCL forward propagation for a 2nd to last block's attention class (incomplete attention)
 * @param context OpenCL context
 * @param queue OpenCL command queue
 * @param kernels Map of compiled OpenCL kernels
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
        clforprop(d_embedding, layers_mlp, totalTokenCount);
        return;
    }

    // Use constants defined in attention.hpp for clarity
    const int d = EMBEDDING;        // Embedding dimension
    const int h = MATHEIGHTS;       // Height dimension of K/Q, projection matrices

    // Calculate the number of tokens relevant to this block
    int count = std::abs(totalTokenCount - contextWindowSize * (blockIdx - 1));
    // count = std::min(count, contextWindowSize); // Optional clamp

    // --- Basic Validation ---
    if (count <= 0) {
        std::cerr << "Warning: clforprop(..., blockIdx=" << blockIdx << ") called with calculated count <= 0. Skipping computation." << std::endl;
        std::fill(EH.begin(), EH.end(), 0.0f);
        if (!EV.empty()) {
             if (EV.size() <= static_cast<size_t>(count)) EV.resize(count + 1);
             if (EV[count].size() != static_cast<size_t>(d)) EV[count].resize(d, 0.0f);
             std::fill(EV[count].begin(), EV[count].end(), 0.0f);
        }
        return;
    }
    if (EV.size() <= static_cast<size_t>(count)) EV.resize(count + 1);
    if (EV[count].size() != static_cast<size_t>(d)) EV[count].resize(d, 0.0f);

    // ** Assumption: K, Q, KdotQ members are pre-populated with data relevant to this block (size 'count') **
     if (K.size() != static_cast<size_t>(count) || (!K.empty() && K[0].size() != static_cast<size_t>(h)) ||
        Q.size() != static_cast<size_t>(count) || (!Q.empty() && Q[0].size() != static_cast<size_t>(h)) ||
        KdotQ.size() != static_cast<size_t>(count) || (!KdotQ.empty() && KdotQ[0].size() != static_cast<size_t>(count)) ||
        MH.a.size() != static_cast<size_t>(d) || (!MH.a.empty() && MH.a[0].size() != static_cast<size_t>(h)) ||
        MV.a.size() != static_cast<size_t>(d) || (!MV.a.empty() && MV.a[0].size() != static_cast<size_t>(h)) ||
        EH.size() != static_cast<size_t>(d) ||
        EV[count].size() != static_cast<size_t>(d) ||
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
    cl_mem d_K = nullptr, d_Q = nullptr, d_KdotQ = nullptr, d_head = nullptr;
    cl_mem d_row_sums = nullptr, d_col_sums = nullptr;
    cl_mem d_dh_accum = nullptr, d_dv_accum = nullptr;
    cl_mem d_MH_hxd = nullptr, d_MV_hxd = nullptr;
    cl_mem d_dh = nullptr, d_dv = nullptr;
    cl_mem d_EH = nullptr, d_EV_current = nullptr; // Corresponds to EV[count]
    cl_mem d_hor_inputs = nullptr, d_ver_inputs = nullptr;
    cl_mem d_hor_output = nullptr, d_ver_output = nullptr;
    cl_mem d_relu_hor_output = nullptr, d_relu_ver_output = nullptr;
    cl_mem d_mlp_bufferA_hor = nullptr, d_mlp_bufferB_hor = nullptr;
    cl_mem d_mlp_bufferA_ver = nullptr, d_mlp_bufferB_ver = nullptr;
    cl_mem d_mlp_pre_activation = nullptr;
    cl_mem d_mlp_weights = nullptr;

    cl_int err;

    try {
        // --- Allocate Device Memory ---
        size_t k_bytes = static_cast<size_t>(count) * h * sizeof(float);
        size_t q_bytes = static_cast<size_t>(count) * h * sizeof(float);
        size_t kdotq_bytes = static_cast<size_t>(count) * count * sizeof(float);
        size_t head_bytes = static_cast<size_t>(count) * count * sizeof(float);
        size_t sums_bytes = static_cast<size_t>(count) * sizeof(float);
        size_t accum_bytes = static_cast<size_t>(h) * sizeof(float);
        size_t proj_mat_bytes = static_cast<size_t>(h) * d * sizeof(float);
        size_t embed_bytes = static_cast<size_t>(d) * sizeof(float);

        d_K = cl_create_buffer(context, CL_MEM_READ_ONLY, k_bytes, nullptr, err); CL_CHECK(err);
        d_Q = cl_create_buffer(context, CL_MEM_READ_ONLY, q_bytes, nullptr, err); CL_CHECK(err);
        d_KdotQ = cl_create_buffer(context, CL_MEM_READ_ONLY, kdotq_bytes, nullptr, err); CL_CHECK(err);
        d_head = cl_create_buffer(context, CL_MEM_READ_WRITE, head_bytes, nullptr, err); CL_CHECK(err);
        d_row_sums = cl_create_buffer(context, CL_MEM_READ_WRITE, sums_bytes, nullptr, err); CL_CHECK(err);
        d_col_sums = cl_create_buffer(context, CL_MEM_READ_WRITE, sums_bytes, nullptr, err); CL_CHECK(err);
        d_dh_accum = cl_create_buffer(context, CL_MEM_READ_WRITE, accum_bytes, nullptr, err); CL_CHECK(err);
        d_dv_accum = cl_create_buffer(context, CL_MEM_READ_WRITE, accum_bytes, nullptr, err); CL_CHECK(err);
        d_MH_hxd = cl_create_buffer(context, CL_MEM_READ_ONLY, proj_mat_bytes, nullptr, err); CL_CHECK(err);
        d_MV_hxd = cl_create_buffer(context, CL_MEM_READ_ONLY, proj_mat_bytes, nullptr, err); CL_CHECK(err);
        d_dh = cl_create_buffer(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, err); CL_CHECK(err);
        d_dv = cl_create_buffer(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, err); CL_CHECK(err);
        d_EH = cl_create_buffer(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, err); CL_CHECK(err);
        d_EV_current = cl_create_buffer(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, err); CL_CHECK(err); // For EV[count]
        d_hor_inputs = cl_create_buffer(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, err); CL_CHECK(err);
        d_ver_inputs = cl_create_buffer(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, err); CL_CHECK(err);
        d_hor_output = cl_create_buffer(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, err); CL_CHECK(err);
        d_ver_output = cl_create_buffer(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, err); CL_CHECK(err);
        d_relu_hor_output = cl_create_buffer(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, err); CL_CHECK(err);
        d_relu_ver_output = cl_create_buffer(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, err); CL_CHECK(err);
        d_mlp_bufferA_hor = cl_create_buffer(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, err); CL_CHECK(err);
        d_mlp_bufferB_hor = cl_create_buffer(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, err); CL_CHECK(err);
        d_mlp_bufferA_ver = cl_create_buffer(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, err); CL_CHECK(err);
        d_mlp_bufferB_ver = cl_create_buffer(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, err); CL_CHECK(err);
        d_mlp_pre_activation = cl_create_buffer(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, err); CL_CHECK(err);

        // Initialize accumulators
        float zero = 0.0f;
        cl_fill_buffer(queue, d_dh_accum, &zero, sizeof(float), 0, accum_bytes);
        cl_fill_buffer(queue, d_dv_accum, &zero, sizeof(float), 0, accum_bytes);

        // --- Flatten Host Data & Copy H->D ---
        // ** Assuming K, Q, KdotQ members contain the 'count'-sized data for this block **
        std::vector<float> flat_K = flatten(K);
        std::vector<float> flat_Q = flatten(Q);
        std::vector<float> flat_KdotQ = flatten(KdotQ);
        std::vector<float> flat_MH_hxd, flat_MV_hxd;
        transposeFlattenMatrix(MH.a, flat_MH_hxd, d, h);
        transposeFlattenMatrix(MV.a, flat_MV_hxd, d, h);

        cl_write_buffer(queue, d_K, k_bytes, flat_K.data());
        cl_write_buffer(queue, d_Q, q_bytes, flat_Q.data());
        cl_write_buffer(queue, d_KdotQ, kdotq_bytes, flat_KdotQ.data());
        cl_write_buffer(queue, d_MH_hxd, proj_mat_bytes, flat_MH_hxd.data());
        cl_write_buffer(queue, d_MV_hxd, proj_mat_bytes, flat_MV_hxd.data());
        cl_write_buffer(queue, d_EH, embed_bytes, EH.data());
        cl_write_buffer(queue, d_EV_current, embed_bytes, EV[count].data()); // Copy EV[count]

        // --- Kernel Launches ---
        const size_t local_work_size_1d = 256;

        // 1. Calculate Attention Weights: head = LOTA(KdotQ)
        cl_kernel lota_kernel = kernels.at("clLOTA2d");
        size_t totalElementsLOTA = static_cast<size_t>(count) * count;
        size_t global_work_size_lota[1] = { (totalElementsLOTA + local_work_size_1d - 1) / local_work_size_1d * local_work_size_1d };
        size_t local_work_size_lota[1] = { local_work_size_1d };
        // Adjust for single work-group reduction if needed
        if (totalElementsLOTA > 0) {
             if (totalElementsLOTA > local_work_size_1d) {
                 std::cerr << "Warning: LOTA size (" << totalElementsLOTA << ") > local size (" << local_work_size_1d << "). Reduction might be incorrect." << std::endl;
                 global_work_size_lota[0] = local_work_size_1d;
             } else {
                 global_work_size_lota[0] = totalElementsLOTA;
                 local_work_size_lota[0] = totalElementsLOTA;
             }
            cl_set_kernel_arg(lota_kernel, 0, sizeof(cl_mem), &d_KdotQ);
            cl_set_kernel_arg(lota_kernel, 1, sizeof(cl_mem), &d_head);
            cl_set_kernel_arg(lota_kernel, 2, sizeof(cl_int), &count); // rows
            cl_set_kernel_arg(lota_kernel, 3, sizeof(cl_int), &count); // cols
            cl_enqueue_nd_range_kernel(queue, lota_kernel, 1, nullptr, global_work_size_lota, local_work_size_lota);
        }


        // 2. Compute Head Row/Column Sums (Masked)
        cl_kernel sums_kernel = kernels.at("computeHeadSumsMaskedKernel");
        size_t global_work_size_sums[1] = { (static_cast<size_t>(count) + local_work_size_1d - 1) / local_work_size_1d * local_work_size_1d };
        size_t local_work_size_sums[1] = { local_work_size_1d };
        cl_int cl_isSelfAttention = isSelfAttention;
        cl_set_kernel_arg(sums_kernel, 0, sizeof(cl_mem), &d_head);
        cl_set_kernel_arg(sums_kernel, 1, sizeof(cl_mem), &d_row_sums);
        cl_set_kernel_arg(sums_kernel, 2, sizeof(cl_mem), &d_col_sums);
        cl_set_kernel_arg(sums_kernel, 3, sizeof(cl_int), &count); // Use count
        cl_set_kernel_arg(sums_kernel, 4, sizeof(cl_int), &cl_isSelfAttention);
        cl_enqueue_nd_range_kernel(queue, sums_kernel, 1, nullptr, global_work_size_sums, local_work_size_sums);

        // 3. Accumulate Weighted K and Q vectors
        cl_kernel accum_kernel = kernels.at("accumulateWeightedVectorsKernel");
        size_t global_work_size_accum[1] = { (static_cast<size_t>(h) + local_work_size_1d - 1) / local_work_size_1d * local_work_size_1d };
        size_t local_work_size_accum[1] = { local_work_size_1d };
        cl_set_kernel_arg(accum_kernel, 0, sizeof(cl_mem), &d_row_sums);
        cl_set_kernel_arg(accum_kernel, 1, sizeof(cl_mem), &d_col_sums);
        cl_set_kernel_arg(accum_kernel, 2, sizeof(cl_mem), &d_K);
        cl_set_kernel_arg(accum_kernel, 3, sizeof(cl_mem), &d_Q);
        cl_set_kernel_arg(accum_kernel, 4, sizeof(cl_mem), &d_dh_accum);
        cl_set_kernel_arg(accum_kernel, 5, sizeof(cl_mem), &d_dv_accum);
        cl_set_kernel_arg(accum_kernel, 6, sizeof(cl_int), &count); // Use count
        cl_set_kernel_arg(accum_kernel, 7, sizeof(cl_int), &h);
        cl_enqueue_nd_range_kernel(queue, accum_kernel, 1, nullptr, global_work_size_accum, local_work_size_accum);

        // 4. Project accumulated vectors
        cl_kernel proj_kernel = kernels.at("kernelLayerForward");
        size_t global_work_size_proj[1] = { (static_cast<size_t>(d) + local_work_size_1d - 1) / local_work_size_1d * local_work_size_1d };
        size_t local_work_size_proj[1] = { local_work_size_1d };
        cl_set_kernel_arg(proj_kernel, 3, sizeof(cl_int), &h); // input_size
        cl_set_kernel_arg(proj_kernel, 4, sizeof(cl_int), &d); // output_size

        cl_set_kernel_arg(proj_kernel, 0, sizeof(cl_mem), &d_dh_accum);
        cl_set_kernel_arg(proj_kernel, 1, sizeof(cl_mem), &d_MH_hxd);
        cl_set_kernel_arg(proj_kernel, 2, sizeof(cl_mem), &d_dh);
        cl_enqueue_nd_range_kernel(queue, proj_kernel, 1, nullptr, global_work_size_proj, local_work_size_proj);

        cl_set_kernel_arg(proj_kernel, 0, sizeof(cl_mem), &d_dv_accum);
        cl_set_kernel_arg(proj_kernel, 1, sizeof(cl_mem), &d_MV_hxd);
        cl_set_kernel_arg(proj_kernel, 2, sizeof(cl_mem), &d_dv);
        cl_enqueue_nd_range_kernel(queue, proj_kernel, 1, nullptr, global_work_size_proj, local_work_size_proj);

        // 5. Prepare MLP inputs
        cl_kernel add_kernel = kernels.at("vectorAddKernel");
        size_t global_work_size_add[1] = { (static_cast<size_t>(d) + local_work_size_1d - 1) / local_work_size_1d * local_work_size_1d };
        size_t local_work_size_add[1] = { local_work_size_1d };
        cl_set_kernel_arg(add_kernel, 3, sizeof(cl_int), &d);

        cl_set_kernel_arg(add_kernel, 0, sizeof(cl_mem), &d_EH);
        cl_set_kernel_arg(add_kernel, 1, sizeof(cl_mem), &d_dh);
        cl_set_kernel_arg(add_kernel, 2, sizeof(cl_mem), &d_hor_inputs);
        cl_enqueue_nd_range_kernel(queue, add_kernel, 1, nullptr, global_work_size_add, local_work_size_add);

        cl_set_kernel_arg(add_kernel, 0, sizeof(cl_mem), &d_EV_current); // Use EV[count] buffer
        cl_set_kernel_arg(add_kernel, 1, sizeof(cl_mem), &d_dv);
        cl_set_kernel_arg(add_kernel, 2, sizeof(cl_mem), &d_ver_inputs);
        cl_enqueue_nd_range_kernel(queue, add_kernel, 1, nullptr, global_work_size_add, local_work_size_add);

        cl_finish(queue);

        // --- 6. Run MLPs Forward ---
        // (Identical logic to the first overload, using 'd' for sizes)
        try {
            cl_kernel mlp_fwd_kernel = kernels.at("kernelLayerForward");
            cl_kernel sigmoid_kernel = kernels.at("clSigmoid1d");

            err = clEnqueueCopyBuffer(queue, d_hor_inputs, d_mlp_bufferA_hor, 0, 0, embed_bytes, 0, nullptr, nullptr); CL_CHECK(err);
            err = clEnqueueCopyBuffer(queue, d_ver_inputs, d_mlp_bufferA_ver, 0, 0, embed_bytes, 0, nullptr, nullptr); CL_CHECK(err);

            cl_mem d_current_in_hor = d_mlp_bufferA_hor;
            cl_mem d_current_out_hor = d_mlp_bufferB_hor;
            cl_mem d_current_in_ver = d_mlp_bufferA_ver;
            cl_mem d_current_out_ver = d_mlp_bufferB_ver;

            size_t num_weight_matrices = hor.weights.size();
            size_t global_work_size_mlp[1] = { (static_cast<size_t>(d) + local_work_size_1d - 1) / local_work_size_1d * local_work_size_1d };
            size_t local_work_size_mlp[1] = { local_work_size_1d };

            for (size_t layer_idx = 0; layer_idx < num_weight_matrices; ++layer_idx) {
                bool is_last_layer = (layer_idx == num_weight_matrices - 1);
                int input_size = d;
                int output_size = d;

                // Hor
                {
                    std::vector<float> flat_weights_hor = flatten(hor.weights[layer_idx]);
                    size_t weights_bytes = flat_weights_hor.size() * sizeof(float);
                    d_mlp_weights = cl_create_buffer(context, CL_MEM_READ_ONLY, weights_bytes, nullptr, err); CL_CHECK(err);
                    cl_write_buffer(queue, d_mlp_weights, weights_bytes, flat_weights_hor.data());
                    cl_mem target_output_buffer = is_last_layer ? d_hor_output : d_mlp_pre_activation;
                    cl_set_kernel_arg(mlp_fwd_kernel, 0, sizeof(cl_mem), &d_current_in_hor);
                    cl_set_kernel_arg(mlp_fwd_kernel, 1, sizeof(cl_mem), &d_mlp_weights);
                    cl_set_kernel_arg(mlp_fwd_kernel, 2, sizeof(cl_mem), &target_output_buffer);
                    cl_set_kernel_arg(mlp_fwd_kernel, 3, sizeof(cl_int), &input_size);
                    cl_set_kernel_arg(mlp_fwd_kernel, 4, sizeof(cl_int), &output_size);
                    cl_enqueue_nd_range_kernel(queue, mlp_fwd_kernel, 1, nullptr, global_work_size_mlp, local_work_size_mlp);
                    if (!is_last_layer) {
                        cl_set_kernel_arg(sigmoid_kernel, 0, sizeof(cl_mem), &d_mlp_pre_activation);
                        cl_set_kernel_arg(sigmoid_kernel, 1, sizeof(cl_mem), &d_current_out_hor);
                        cl_set_kernel_arg(sigmoid_kernel, 2, sizeof(cl_int), &output_size);
                        cl_enqueue_nd_range_kernel(queue, sigmoid_kernel, 1, nullptr, global_work_size_mlp, local_work_size_mlp);
                        cl_mem temp = d_current_in_hor; d_current_in_hor = d_current_out_hor; d_current_out_hor = temp;
                    }
                    cl_release_mem_object(d_mlp_weights); d_mlp_weights = nullptr;
                }
                // Ver
                {
                    std::vector<float> flat_weights_ver = flatten(ver.weights[layer_idx]);
                    size_t weights_bytes = flat_weights_ver.size() * sizeof(float);
                    d_mlp_weights = cl_create_buffer(context, CL_MEM_READ_ONLY, weights_bytes, nullptr, err); CL_CHECK(err);
                    cl_write_buffer(queue, d_mlp_weights, weights_bytes, flat_weights_ver.data());
                    cl_mem target_output_buffer = is_last_layer ? d_ver_output : d_mlp_pre_activation;
                    cl_set_kernel_arg(mlp_fwd_kernel, 0, sizeof(cl_mem), &d_current_in_ver);
                    cl_set_kernel_arg(mlp_fwd_kernel, 1, sizeof(cl_mem), &d_mlp_weights);
                    cl_set_kernel_arg(mlp_fwd_kernel, 2, sizeof(cl_mem), &target_output_buffer);
                    cl_enqueue_nd_range_kernel(queue, mlp_fwd_kernel, 1, nullptr, global_work_size_mlp, local_work_size_mlp);
                    if (!is_last_layer) {
                        cl_set_kernel_arg(sigmoid_kernel, 0, sizeof(cl_mem), &d_mlp_pre_activation);
                        cl_set_kernel_arg(sigmoid_kernel, 1, sizeof(cl_mem), &d_current_out_ver);
                        cl_enqueue_nd_range_kernel(queue, sigmoid_kernel, 1, nullptr, global_work_size_mlp, local_work_size_mlp);
                        cl_mem temp = d_current_in_ver; d_current_in_ver = d_current_out_ver; d_current_out_ver = temp;
                    }
                    cl_release_mem_object(d_mlp_weights); d_mlp_weights = nullptr;
                }
            }
            cl_finish(queue);
        }
        catch (const std::exception& e) {
            std::cerr << "Error during OpenCL MLP processing (blockIdx=" << blockIdx << "): " << e.what() << std::endl;
            cl_release_mem_object(d_mlp_weights);
            throw;
        }

        // --- Free MLP Intermediate Buffers ---
        cl_release_mem_object(d_mlp_bufferA_hor); d_mlp_bufferA_hor = nullptr;
        cl_release_mem_object(d_mlp_bufferB_hor); d_mlp_bufferB_hor = nullptr;
        cl_release_mem_object(d_mlp_bufferA_ver); d_mlp_bufferA_ver = nullptr;
        cl_release_mem_object(d_mlp_bufferB_ver); d_mlp_bufferB_ver = nullptr;
        cl_release_mem_object(d_mlp_pre_activation); d_mlp_pre_activation = nullptr;

        // 7. Apply ReLU
        cl_kernel relu_kernel = kernels.at("clReLU1d");
        cl_set_kernel_arg(relu_kernel, 2, sizeof(cl_int), &d);

        cl_set_kernel_arg(relu_kernel, 0, sizeof(cl_mem), &d_hor_output);
        cl_set_kernel_arg(relu_kernel, 1, sizeof(cl_mem), &d_relu_hor_output);
        cl_enqueue_nd_range_kernel(queue, relu_kernel, 1, nullptr, global_work_size_add, local_work_size_add);

        cl_set_kernel_arg(relu_kernel, 0, sizeof(cl_mem), &d_ver_output);
        cl_set_kernel_arg(relu_kernel, 1, sizeof(cl_mem), &d_relu_ver_output);
        cl_enqueue_nd_range_kernel(queue, relu_kernel, 1, nullptr, global_work_size_add, local_work_size_add);

        cl_finish(queue);

        // 8. Final Residual Update
        // Use add_kernel from step 5

        cl_set_kernel_arg(add_kernel, 0, sizeof(cl_mem), &d_EH);
        cl_set_kernel_arg(add_kernel, 1, sizeof(cl_mem), &d_relu_hor_output);
        cl_set_kernel_arg(add_kernel, 2, sizeof(cl_mem), &d_EH);
        cl_enqueue_nd_range_kernel(queue, add_kernel, 1, nullptr, global_work_size_add, local_work_size_add);

        cl_set_kernel_arg(add_kernel, 0, sizeof(cl_mem), &d_EV_current); // Use EV[count] buffer
        cl_set_kernel_arg(add_kernel, 1, sizeof(cl_mem), &d_relu_ver_output);
        cl_set_kernel_arg(add_kernel, 2, sizeof(cl_mem), &d_EV_current); // Update EV[count] buffer
        cl_enqueue_nd_range_kernel(queue, add_kernel, 1, nullptr, global_work_size_add, local_work_size_add);

        cl_finish(queue);

        // --- Copy Results D->H ---
        cl_read_buffer(queue, d_EH, embed_bytes, EH.data());
        cl_read_buffer(queue, d_EV_current, embed_bytes, EV[count].data()); // Copy back to EV[count]

        // --- Free Device Memory ---
        cl_release_mem_object(d_K); cl_release_mem_object(d_Q); cl_release_mem_object(d_KdotQ); cl_release_mem_object(d_head);
        cl_release_mem_object(d_row_sums); cl_release_mem_object(d_col_sums);
        cl_release_mem_object(d_dh_accum); cl_release_mem_object(d_dv_accum);
        cl_release_mem_object(d_MH_hxd); cl_release_mem_object(d_MV_hxd);
        cl_release_mem_object(d_dh); cl_release_mem_object(d_dv);
        cl_release_mem_object(d_EH); cl_release_mem_object(d_EV_current);
        cl_release_mem_object(d_hor_inputs); cl_release_mem_object(d_ver_inputs);
        cl_release_mem_object(d_hor_output); cl_release_mem_object(d_ver_output);
        cl_release_mem_object(d_relu_hor_output); cl_release_mem_object(d_relu_ver_output);

    }
    catch (const std::exception& e) {
        std::cerr << "OpenCL Exception in attention clforprop(..., blockIdx=" << blockIdx << "): " << e.what() << std::endl;
        // --- Cleanup on Error ---
        cl_release_mem_object(d_K); cl_release_mem_object(d_Q); cl_release_mem_object(d_KdotQ); cl_release_mem_object(d_head);
        cl_release_mem_object(d_row_sums); cl_release_mem_object(d_col_sums);
        cl_release_mem_object(d_dh_accum); cl_release_mem_object(d_dv_accum);
        cl_release_mem_object(d_MH_hxd); cl_release_mem_object(d_MV_hxd);
        cl_release_mem_object(d_dh); cl_release_mem_object(d_dv);
        cl_release_mem_object(d_EH); cl_release_mem_object(d_EV_current);
        cl_release_mem_object(d_hor_inputs); cl_release_mem_object(d_ver_inputs);
        cl_release_mem_object(d_hor_output); cl_release_mem_object(d_ver_output);
        cl_release_mem_object(d_relu_hor_output); cl_release_mem_object(d_relu_ver_output);
        cl_release_mem_object(d_mlp_bufferA_hor); cl_release_mem_object(d_mlp_bufferB_hor);
        cl_release_mem_object(d_mlp_bufferA_ver); cl_release_mem_object(d_mlp_bufferB_ver);
        cl_release_mem_object(d_mlp_pre_activation);
        cl_release_mem_object(d_mlp_weights);
        throw; // Re-throw the exception
    }
}

#endif // USE_OPENCL
