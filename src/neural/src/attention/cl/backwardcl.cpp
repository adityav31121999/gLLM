
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
#include "include/attention.hpp"
#include <vector>
#include <string>
#include <stdexcept>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <map>

/**
 * @brief OpenCL Backward Propagation using gradients from expected Horizontal output.
 *      Updates MH, MV, MQ, MK, EH, and EV. 
 * @param expected Expected output vector (target embedding for next token prediction)
 * @param in Input size (embedding dimension)
 * @param layers Number of activation layers in the MLPs (for weights = layers - 1)
 */
void attention::clbackward(std::vector<float>& expected, int& in, int& layers, int& headnumber)
{
    // --- Setup ---
    const int embedding_dim = EMBEDDING;
    const int mat_heights = MATHEIGHTS;
    const int context_win = CONTEXT_WIN;
    const float learning_rate = LEARNING;
    const float scaling_factor = SCALING;
    const int token_count = this->tokenCount;

    // Calculate sizes from mat objects
    const size_t k_q_bytes = static_cast<size_t>(this->K.row) * this->K.col * sizeof(float);
    const size_t head_bytes = static_cast<size_t>(this->KdotQ.row) * this->KdotQ.col * sizeof(float);
    const size_t mh_mv_mq_mk_bytes = static_cast<size_t>(this->MH.row) * this->MH.col * sizeof(float);
    const size_t ev_bytes = static_cast<size_t>(this->EV.row) * this->EV.col * sizeof(float);
    const size_t embed_bytes = embedding_dim * sizeof(float);
    const size_t mlp_weights_bytes = (this->hor.weights.empty()) ? 0 :
                                   static_cast<size_t>(this->hor.weights[0].row) * this->hor.weights[0].col * sizeof(float);

    // Validation
    if (embedding_dim != in) throw std::runtime_error("Embedding dimension mismatch");
    if (expected.size() != embedding_dim) throw std::runtime_error("Expected vector size mismatch");
    if (this->K.row != CONTEXT_WIN || this->K.col != MATHEIGHTS) throw std::runtime_error("K dimensions mismatch");
    if (this->Q.row != CONTEXT_WIN || this->Q.col != MATHEIGHTS) throw std::runtime_error("Q dimensions mismatch");
    if (this->KdotQ.row != CONTEXT_WIN || this->KdotQ.col != CONTEXT_WIN) throw std::runtime_error("KdotQ dimensions mismatch");
    if (this->MH.row != MATHEIGHTS || this->MH.col != EMBEDDING) throw std::runtime_error("MH dimensions mismatch");
    if (this->EV.row != CONTEXT_WIN || this->EV.col != EMBEDDING) throw std::runtime_error("EV dimensions mismatch");
    if (!this->hor.weights.empty() && (this->hor.weights[0].row != EMBEDDING || this->hor.weights[0].col != EMBEDDING))
        throw std::runtime_error("MLP hor.weights dimensions mismatch");

    if (token_count <= 0) {
        std::cerr << "Warning: clbackward(expected,...) called with token_count <= 0. Skipping." << std::endl;
        return;
    }

    int update_ev = 1;      // int update_ev = (blocknumber > 1) ? 1 : 0;
    int update_eh = (headnumber == 0) ? 1 : 0;
    cl_int cl_err;          // For OpenCL error codes
    
    OpenCLContext& context_obj = this->clcontext;
    cl::Context context = context_obj.context;
    cl::CommandQueue queue = context_obj.queue;
    // Use context_obj.kernels map below
    // --- Allocate Memory (Attention) ---
    cl::Buffer d_expected_h(context, CL_MEM_READ_ONLY, embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
    cl::Buffer d_EH(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err); // Read/Write for update
    cl::Buffer d_EV(context, CL_MEM_READ_WRITE, ev_bytes, nullptr, &cl_err); CL_CHECK(cl_err); // Read/Write for update
    cl::Buffer d_grad_EH(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
    cl::Buffer d_grad_EV_scaled(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
    cl::Buffer d_grad_dh(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
    cl::Buffer d_grad_dv(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
    cl::Buffer d_KdotQ(context, CL_MEM_READ_ONLY, head_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
    cl::Buffer d_head(context, CL_MEM_READ_WRITE, head_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
    cl::Buffer d_K(context, CL_MEM_READ_ONLY, k_q_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
    cl::Buffer d_Q(context, CL_MEM_READ_ONLY, k_q_bytes, nullptr, &cl_err); CL_CHECK(cl_err);

    cl::Buffer d_pre_MH(context, CL_MEM_READ_WRITE, mat_heights * sizeof(float), nullptr, &cl_err); CL_CHECK(cl_err);
    cl::Buffer d_pre_MV(context, CL_MEM_READ_WRITE, mat_heights * sizeof(float), nullptr, &cl_err); CL_CHECK(cl_err);
    cl::Buffer d_MH_a(context, CL_MEM_READ_WRITE, mh_mv_mq_mk_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
    cl::Buffer d_MV_a(context, CL_MEM_READ_WRITE, mh_mv_mq_mk_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
    cl::Buffer d_MQ_a(context, CL_MEM_READ_WRITE, mh_mv_mq_mk_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
    cl::Buffer d_MK_a(context, CL_MEM_READ_WRITE, mh_mv_mq_mk_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
    cl::Buffer d_grad_MH(context, CL_MEM_READ_WRITE, mh_mv_mq_mk_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
    cl::Buffer d_grad_MV(context, CL_MEM_READ_WRITE, mh_mv_mq_mk_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
    cl::Buffer d_grad_head(context, CL_MEM_READ_WRITE, head_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
    cl::Buffer d_lota_deriv(context, CL_MEM_READ_WRITE, head_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
    cl::Buffer d_grad_KdotQ(context, CL_MEM_READ_WRITE, head_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
    cl::Buffer d_grad_K(context, CL_MEM_READ_WRITE, k_q_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
    cl::Buffer d_grad_Q(context, CL_MEM_READ_WRITE, k_q_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
    cl::Buffer d_grad_MQ(context, CL_MEM_READ_WRITE, mh_mv_mq_mk_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
    cl::Buffer d_grad_MK(context, CL_MEM_READ_WRITE, mh_mv_mq_mk_bytes, nullptr, &cl_err); CL_CHECK(cl_err); // Simplified gradients

    // --- Allocate Memory (MLP Internals) ---
    std::vector<cl::Buffer> d_hor_activations;
    std::vector<cl::Buffer> d_hor_weights;
    std::vector<cl::Buffer> d_hor_gweights;
    std::vector<cl::Buffer> d_hor_deltas;
    std::vector<cl::Buffer> d_ver_activations;
    std::vector<cl::Buffer> d_ver_weights;
    std::vector<cl::Buffer> d_ver_gweights;
    std::vector<cl::Buffer> d_ver_deltas;

    d_hor_activations.reserve(layers);
    d_hor_weights.reserve(layers-1);
    d_hor_gweights.reserve(layers-1);
    d_hor_deltas.reserve(layers);
    d_ver_activations.reserve(layers);
    d_ver_weights.reserve(layers-1);
    d_ver_gweights.reserve(layers-1);
    d_ver_deltas.reserve(layers);

    for (int l = 0; l < layers; ++l) {
        d_hor_activations.emplace_back(context, CL_MEM_READ_ONLY, embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        d_hor_deltas.emplace_back(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        d_ver_activations.emplace_back(context, CL_MEM_READ_ONLY, embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        d_ver_deltas.emplace_back(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
    }

    for (int l = 0; l < layers-1; ++l) {
        d_hor_weights.emplace_back(context, CL_MEM_READ_WRITE, mlp_weights_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        d_hor_gweights.emplace_back(context, CL_MEM_READ_WRITE, mlp_weights_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        d_ver_weights.emplace_back(context, CL_MEM_READ_WRITE, mlp_weights_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        d_ver_gweights.emplace_back(context, CL_MEM_READ_WRITE, mlp_weights_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
    }

    // --- Data Transfer H->D (Attention) ---
    CL_CHECK(queue.enqueueWriteBuffer(d_expected_h, CL_TRUE, 0, embed_bytes, expected.data()));
    CL_CHECK(queue.enqueueWriteBuffer(d_EH, CL_TRUE, 0, embed_bytes, this->EH.data()));
    CL_CHECK(queue.enqueueWriteBuffer(d_EV, CL_TRUE, 0, ev_bytes, this->EV.mapped_data));
    CL_CHECK(queue.enqueueWriteBuffer(d_KdotQ, CL_TRUE, 0, head_bytes, this->KdotQ.mapped_data));
    CL_CHECK(queue.enqueueWriteBuffer(d_K, CL_TRUE, 0, k_q_bytes, this->K.mapped_data));
    CL_CHECK(queue.enqueueWriteBuffer(d_Q, CL_TRUE, 0, k_q_bytes, this->Q.mapped_data));
    CL_CHECK(queue.enqueueWriteBuffer(d_MH_a, CL_TRUE, 0, mh_mv_mq_mk_bytes, this->MH.mapped_data));
    CL_CHECK(queue.enqueueWriteBuffer(d_MV_a, CL_TRUE, 0, mh_mv_mq_mk_bytes, this->MV.mapped_data));
    CL_CHECK(queue.enqueueWriteBuffer(d_MQ_a, CL_TRUE, 0, mh_mv_mq_mk_bytes, this->MQ.mapped_data));
    CL_CHECK(queue.enqueueWriteBuffer(d_MK_a, CL_TRUE, 0, mh_mv_mq_mk_bytes, this->MK.mapped_data));

    // --- Data Transfer H->D (MLP Internals) ---
    for (int l = 0; l < layers; ++l) {
        CL_CHECK(queue.enqueueWriteBuffer(d_hor_activations[l], CL_TRUE, 0, embed_bytes, this->hor.activations[l].data()));
    }
    for (int l = 0; l < layers; ++l) {
        CL_CHECK(queue.enqueueWriteBuffer(d_ver_activations[l], CL_TRUE, 0, embed_bytes, this->ver.activations[l].data()));
    }
    for (int l = 0; l < layers-1; ++l) {
        CL_CHECK(queue.enqueueWriteBuffer(d_hor_weights[l], CL_TRUE, 0, mlp_weights_bytes, this->hor.weights[l].mapped_data));
    }
    for (int l = 0; l < layers-1; ++l) {
        CL_CHECK(queue.enqueueWriteBuffer(d_ver_weights[l], CL_TRUE, 0, mlp_weights_bytes, this->ver.weights[l].mapped_data));
    }

    try {
        // --- Kernel Launch Config ---
        const size_t local_work_size_1d = 256;
        auto calculate_global_1d = [&](size_t total_size) {
            return ((total_size + local_work_size_1d - 1) / local_work_size_1d) * local_work_size_1d;
        };
        cl::NDRange global_embed(calculate_global_1d(embedding_dim));
        cl::NDRange global_head(calculate_global_1d(this->KdotQ.row * this->KdotQ.col));
        cl::NDRange global_mat_heights(calculate_global_1d(mat_heights));
        cl::NDRange global_matrix(calculate_global_1d(this->MH.row * this->MH.col));
        cl::NDRange global_ev(calculate_global_1d(this->EV.row * this->EV.col));
        cl::NDRange local_1d(local_work_size_1d);

        size_t local_work_size_2d[2] = { 16, 16 };
        auto calculate_global_2d = [&](size_t dim0, size_t dim1) {
            size_t global0 = ((dim0 + local_work_size_2d[0] - 1) / local_work_size_2d[0]) * local_work_size_2d[0];
            size_t global1 = ((dim1 + local_work_size_2d[1] - 1) / local_work_size_2d[1]) * local_work_size_2d[1];
            return cl::NDRange(global0, global1);
        };
        cl::NDRange global_embed_2d = calculate_global_2d(embedding_dim, embedding_dim);
        cl::NDRange global_head_2d = calculate_global_2d(token_count, token_count);
        cl::NDRange global_matrix_2d = calculate_global_2d(embedding_dim, mat_heights); // dim0=embed, dim1=height
        // For kernels operating on K, Q and their gradients (token_count x embedding_dim)
        cl::NDRange global_kq_grad_2d = calculate_global_2d(embedding_dim, token_count); // dim0 for embedding_dim, dim1 for token_count
        cl::NDRange local_2d(local_work_size_2d[0], local_work_size_2d[1]);

        // --- Backpropagation Steps ---
        // Step 1: Compute grad_EH
        cl::Kernel grad_eh_ev_kernel = context_obj.kernels.at("kernelComputeGradientsEH_EV");
        CL_CHECK(grad_eh_ev_kernel.setArg(0, d_EH));
        CL_CHECK(grad_eh_ev_kernel.setArg(1, d_expected_h));
        CL_CHECK(grad_eh_ev_kernel.setArg(2, d_grad_EH));
        CL_CHECK(grad_eh_ev_kernel.setArg(3, d_grad_EV_scaled));
        CL_CHECK(grad_eh_ev_kernel.setArg(4, embedding_dim));
        CL_CHECK(queue.enqueueNDRangeKernel(grad_eh_ev_kernel, cl::NullRange, global_embed, local_1d));
        CL_CHECK(queue.finish());

        // --- Step 2: Backprop through MLPs ---
        cl::Kernel last_delta_kernel = context_obj.kernels.at("kernelLastLayerDeltaSigmoid");
        cl::Kernel hidden_delta_kernel = context_obj.kernels.at("kernelHiddenDeltaSigmoid");
        cl::Kernel update_weights_kernel = context_obj.kernels.at("kernelUpdateWeightsAndGradients");
        // --- 2a: Backprop through hor MLP ---
        CL_CHECK(last_delta_kernel.setArg(0, d_grad_EH));
        CL_CHECK(last_delta_kernel.setArg(1, d_hor_activations[layers - 1]));
        CL_CHECK(last_delta_kernel.setArg(2, d_hor_deltas[layers - 1]));
        CL_CHECK(last_delta_kernel.setArg(3, embedding_dim));
        CL_CHECK(queue.enqueueNDRangeKernel(last_delta_kernel, cl::NullRange, global_embed, local_1d));
        CL_CHECK(queue.finish());
        for (int l = layers - 2; l >= 0; --l) {
            CL_CHECK(hidden_delta_kernel.setArg(0, d_hor_deltas[l + 1]));
            CL_CHECK(hidden_delta_kernel.setArg(1, d_hor_weights[l])); // Corrected index
            CL_CHECK(hidden_delta_kernel.setArg(2, d_hor_activations[l]));
            CL_CHECK(hidden_delta_kernel.setArg(3, d_hor_deltas[l]));
            CL_CHECK(hidden_delta_kernel.setArg(4, embedding_dim));
            CL_CHECK(hidden_delta_kernel.setArg(5, embedding_dim));
            CL_CHECK(queue.enqueueNDRangeKernel(hidden_delta_kernel, cl::NullRange, global_embed, local_1d));
            CL_CHECK(queue.finish());
        }
        for (int l = 0; l < layers - 1; ++l) { // Corrected loop bound
            cl::Buffer& d_prev_activations = (l == 0) ? d_EH : d_hor_activations[l - 1];
            CL_CHECK(update_weights_kernel.setArg(0, d_hor_deltas[l]));
            CL_CHECK(update_weights_kernel.setArg(1, d_prev_activations));
            CL_CHECK(update_weights_kernel.setArg(2, d_hor_weights[l]));
            CL_CHECK(update_weights_kernel.setArg(3, d_hor_gweights[l])); // d_hor_gweights also has layers-1 elements
            CL_CHECK(update_weights_kernel.setArg(4, learning_rate));
            CL_CHECK(update_weights_kernel.setArg(5, embedding_dim));
            CL_CHECK(update_weights_kernel.setArg(6, embedding_dim));
            CL_CHECK(queue.enqueueNDRangeKernel(update_weights_kernel, cl::NullRange, global_embed_2d, local_2d));
            CL_CHECK(queue.finish());
        }

        // --- 2b: Backprop through ver MLP ---
        CL_CHECK(last_delta_kernel.setArg(0, d_grad_EV_scaled));
        CL_CHECK(last_delta_kernel.setArg(1, d_ver_activations[layers - 1]));
        CL_CHECK(last_delta_kernel.setArg(2, d_ver_deltas[layers - 1]));
        // Arg 3 (embedding_dim) is already set
        CL_CHECK(queue.enqueueNDRangeKernel(last_delta_kernel, cl::NullRange, global_embed, local_1d));
        CL_CHECK(queue.finish());
        for (int l = layers - 2; l >= 0; --l) {
            CL_CHECK(hidden_delta_kernel.setArg(0, d_ver_deltas[l + 1]));
            CL_CHECK(hidden_delta_kernel.setArg(1, d_ver_weights[l])); // Corrected index
            CL_CHECK(hidden_delta_kernel.setArg(2, d_ver_activations[l]));
            CL_CHECK(hidden_delta_kernel.setArg(3, d_ver_deltas[l]));
            // Args 4, 5 (embedding_dim) are already set
            CL_CHECK(queue.enqueueNDRangeKernel(hidden_delta_kernel, cl::NullRange, global_embed, local_1d));
            CL_CHECK(queue.finish());
        }
        for (int l = 0; l < layers - 1; ++l) {
            // Input to ver.weights[l] is ver.activations[l] (if act[0] is MLP input) or (l==0 ? MLP_INPUT : ver.activations[l-1])
            // Assuming this->ver.activations[0] is the input to the ver MLP.
            cl::Buffer& d_prev_activations_ver = (l == 0) ? d_ver_activations[0] : d_ver_activations[l-1];
            int prev_layer_size = (l == 0) ? embedding_dim : embedding_dim;
            // Use global_embed_2d as sizes are embedding_dim x embedding_dim
            CL_CHECK(update_weights_kernel.setArg(0, d_ver_deltas[l]));
            CL_CHECK(update_weights_kernel.setArg(1, d_prev_activations_ver));
            CL_CHECK(update_weights_kernel.setArg(2, d_ver_weights[l]));
            CL_CHECK(update_weights_kernel.setArg(3, d_ver_gweights[l])); // d_ver_gweights also has layers-1 elements
            // Arg 4 (learning_rate) is already set
            CL_CHECK(update_weights_kernel.setArg(5, embedding_dim));
            CL_CHECK(update_weights_kernel.setArg(6, prev_layer_size)); // Input size might vary if not square
            CL_CHECK(queue.enqueueNDRangeKernel(update_weights_kernel, cl::NullRange, global_embed_2d, local_2d));
            CL_CHECK(queue.finish());
        }
        // --- End of MLP Backprop ---
        // --- Step 3: Compute grad_dh ---
        // Use the kernel that sums gradients, NOT _1stHead version
        cl::Kernel grad_mlp_input_kernel = context_obj.kernels.at("kernelComputeGradMLPInput");
        CL_CHECK(grad_mlp_input_kernel.setArg(0, d_hor_deltas[0]));
        CL_CHECK(grad_mlp_input_kernel.setArg(1, d_hor_weights[0]));
        CL_CHECK(grad_mlp_input_kernel.setArg(2, d_grad_dh));
        CL_CHECK(grad_mlp_input_kernel.setArg(3, embedding_dim)); // current_layer_size
        CL_CHECK(grad_mlp_input_kernel.setArg(4, embedding_dim)); // input_size
        CL_CHECK(queue.enqueueNDRangeKernel(grad_mlp_input_kernel, cl::NullRange, global_embed, local_1d));

        CL_CHECK(grad_mlp_input_kernel.setArg(0, d_ver_deltas[0]));
        CL_CHECK(grad_mlp_input_kernel.setArg(1, d_ver_weights[0]));
        CL_CHECK(grad_mlp_input_kernel.setArg(2, d_grad_dv));
        // Args 3, 4 (embedding_dim) are already set
        CL_CHECK(queue.enqueueNDRangeKernel(grad_mlp_input_kernel, cl::NullRange, global_embed, local_1d));
        CL_CHECK(queue.finish());

        // --- Step 4: Compute grad_MH and grad_MV ---
        cl::Kernel lota_kernel = context_obj.kernels.at("clLOTA2d");
        CL_CHECK(lota_kernel.setArg(0, d_KdotQ));
        CL_CHECK(lota_kernel.setArg(1, d_head));
        CL_CHECK(lota_kernel.setArg(2, token_count));
        CL_CHECK(lota_kernel.setArg(3, token_count));
        size_t lota_global_raw = this->KdotQ.row * this->KdotQ.col;
        size_t lota_local_clamped = (std::min)(lota_global_raw, local_work_size_1d); // Parenthesize std::min
        if (lota_local_clamped == 0) lota_local_clamped = 1;
        size_t lota_global_padded = ((lota_global_raw + lota_local_clamped - 1) / lota_local_clamped) * lota_local_clamped;
        cl::NDRange global_lota(lota_global_padded);
        cl::NDRange local_lota(lota_local_clamped);
        if (this->KdotQ.row * this->KdotQ.col > 0) {
            CL_CHECK(queue.enqueueNDRangeKernel(lota_kernel, cl::NullRange, global_lota, local_lota));
        }
        cl::Kernel pre_mh_mv_kernel = context_obj.kernels.at("kernelComputePreMH_MV");
        CL_CHECK(pre_mh_mv_kernel.setArg(0, d_head));
        CL_CHECK(pre_mh_mv_kernel.setArg(1, d_K));
        CL_CHECK(pre_mh_mv_kernel.setArg(2, d_Q));
        CL_CHECK(pre_mh_mv_kernel.setArg(3, d_pre_MH));
        CL_CHECK(pre_mh_mv_kernel.setArg(4, d_pre_MV));
        CL_CHECK(pre_mh_mv_kernel.setArg(5, token_count));
        CL_CHECK(pre_mh_mv_kernel.setArg(6, mat_heights));
        CL_CHECK(queue.enqueueNDRangeKernel(pre_mh_mv_kernel, cl::NullRange, global_mat_heights, local_1d));
        // CL_CHECK(queue.finish());

        cl::Kernel grad_mh_mv_kernel = context_obj.kernels.at("kernelComputeGradMH_MV");
        CL_CHECK(grad_mh_mv_kernel.setArg(0, d_pre_MH));
        CL_CHECK(grad_mh_mv_kernel.setArg(1, d_pre_MV));
        CL_CHECK(grad_mh_mv_kernel.setArg(2, d_grad_dh));
        CL_CHECK(grad_mh_mv_kernel.setArg(3, d_grad_dv));
        CL_CHECK(grad_mh_mv_kernel.setArg(4, d_grad_MH));
        CL_CHECK(grad_mh_mv_kernel.setArg(5, d_grad_MV));
        CL_CHECK(grad_mh_mv_kernel.setArg(6, mat_heights));
        CL_CHECK(grad_mh_mv_kernel.setArg(7, embedding_dim));
        CL_CHECK(queue.enqueueNDRangeKernel(grad_mh_mv_kernel, cl::NullRange, global_matrix_2d, local_2d));
        // CL_CHECK(queue.finish());

        // --- Step 5: Compute grad_head ---
        cl::Kernel grad_head_kernel = context_obj.kernels.at("kernelComputeGradHead");
        CL_CHECK(grad_head_kernel.setArg(0, d_K));
        CL_CHECK(grad_head_kernel.setArg(1, d_Q));
        CL_CHECK(grad_head_kernel.setArg(2, d_MH_a));
        CL_CHECK(grad_head_kernel.setArg(3, d_MV_a));
        CL_CHECK(grad_head_kernel.setArg(4, d_grad_dh));
        CL_CHECK(grad_head_kernel.setArg(5, d_grad_dv));
        CL_CHECK(grad_head_kernel.setArg(6, d_grad_head));
        CL_CHECK(grad_head_kernel.setArg(7, token_count));
        CL_CHECK(grad_head_kernel.setArg(8, mat_heights));
        CL_CHECK(grad_head_kernel.setArg(9, embedding_dim));
        CL_CHECK(queue.enqueueNDRangeKernel(grad_head_kernel, cl::NullRange, global_head_2d, local_2d));
        // CL_CHECK(queue.finish());

        // --- Step 6: Backprop through LOTA ---
        cl::Kernel lota_deriv_kernel = context_obj.kernels.at("clLOTA2dder"); // Or clComputeSimpleLOTAder
        CL_CHECK(lota_deriv_kernel.setArg(0, d_KdotQ));
        CL_CHECK(lota_deriv_kernel.setArg(1, d_lota_deriv));
        CL_CHECK(lota_deriv_kernel.setArg(2, token_count));
        CL_CHECK(lota_deriv_kernel.setArg(3, token_count));
        if (this->KdotQ.row * this->KdotQ.col > 0) {
            CL_CHECK(queue.enqueueNDRangeKernel(lota_deriv_kernel, cl::NullRange, global_lota, local_lota));
        }
        // CL_CHECK(queue.finish());

        cl::Kernel grad_kdotq_kernel = context_obj.kernels.at("kernelComputeGradKdotQ_LOTA");
        CL_CHECK(grad_kdotq_kernel.setArg(0, d_grad_head));
        CL_CHECK(grad_kdotq_kernel.setArg(1, d_lota_deriv));
        CL_CHECK(grad_kdotq_kernel.setArg(2, d_grad_KdotQ));
        CL_CHECK(grad_kdotq_kernel.setArg(3, scaling_factor));
        CL_CHECK(grad_kdotq_kernel.setArg(4, this->KdotQ.row * this->KdotQ.col));
        CL_CHECK(queue.enqueueNDRangeKernel(grad_kdotq_kernel, cl::NullRange, global_head, local_1d));
        // CL_CHECK(queue.finish());

        // --- Step 7: Compute grad_K and grad_Q ---
        cl::Kernel grad_k_q_kernel = context_obj.kernels.at("kernelComputeGradK_Q");
        CL_CHECK(grad_k_q_kernel.setArg(0, d_grad_KdotQ));
        CL_CHECK(grad_k_q_kernel.setArg(1, d_K));
        CL_CHECK(grad_k_q_kernel.setArg(2, d_Q));
        CL_CHECK(grad_k_q_kernel.setArg(3, d_grad_K));
        CL_CHECK(grad_k_q_kernel.setArg(4, d_grad_Q));
        CL_CHECK(grad_k_q_kernel.setArg(5, token_count));
        CL_CHECK(grad_k_q_kernel.setArg(6, embedding_dim)); // Pass embedding_dim as the K/Q column dimension
        CL_CHECK(queue.enqueueNDRangeKernel(grad_k_q_kernel, cl::NullRange, global_kq_grad_2d, local_2d)); // Launch with embedding_dim for h-like dimension
        // CL_CHECK(queue.finish());

        // --- Step 8: Compute grad_MK and grad_MQ (Simplified) ---
        // Use the correct kernel that takes K and Q vectors as input proxies for embeddings
        cl::Kernel grad_mk_mq_kernel = context_obj.kernels.at("kernelComputeGradMK_MQ");
        CL_CHECK(grad_mk_mq_kernel.setArg(0, d_grad_K));
        CL_CHECK(grad_mk_mq_kernel.setArg(1, d_grad_Q));
        CL_CHECK(grad_mk_mq_kernel.setArg(2, d_K)); // Pass d_K (token_count x embedding_dim)
        CL_CHECK(grad_mk_mq_kernel.setArg(3, d_Q)); // Pass d_Q (token_count x embedding_dim)
        CL_CHECK(grad_mk_mq_kernel.setArg(4, d_grad_MK));
        CL_CHECK(grad_mk_mq_kernel.setArg(5, d_grad_MQ));
        CL_CHECK(grad_mk_mq_kernel.setArg(6, token_count));
        CL_CHECK(grad_mk_mq_kernel.setArg(7, mat_heights));
        CL_CHECK(grad_mk_mq_kernel.setArg(8, embedding_dim));
        CL_CHECK(queue.enqueueNDRangeKernel(grad_mk_mq_kernel, cl::NullRange, global_matrix_2d, local_2d));
        // CL_CHECK(queue.finish());

        // --- Step 9 & 10: Update Weights (MH, MV, MQ, MK) and Embeddings (EH, EV) ---
        cl::Kernel update_weights_eh_ev_kernel = context_obj.kernels.at("kernelUpdateWeights_EH_EV");
        CL_CHECK(update_weights_eh_ev_kernel.setArg(0, d_MH_a));
        CL_CHECK(update_weights_eh_ev_kernel.setArg(1, d_MV_a));
        CL_CHECK(update_weights_eh_ev_kernel.setArg(2, d_MQ_a));
        CL_CHECK(update_weights_eh_ev_kernel.setArg(3, d_MK_a));
        CL_CHECK(update_weights_eh_ev_kernel.setArg(4, d_EH));
        CL_CHECK(update_weights_eh_ev_kernel.setArg(5, d_EV));
        CL_CHECK(update_weights_eh_ev_kernel.setArg(6, d_grad_MH));
        CL_CHECK(update_weights_eh_ev_kernel.setArg(7, d_grad_MV));
        CL_CHECK(update_weights_eh_ev_kernel.setArg(8, d_grad_MQ));
        CL_CHECK(update_weights_eh_ev_kernel.setArg(9, d_grad_MK));
        CL_CHECK(update_weights_eh_ev_kernel.setArg(10, d_grad_EH)); // Use original grad_EH
        CL_CHECK(update_weights_eh_ev_kernel.setArg(11, d_grad_EV_scaled)); // Use scaled grad for EV update
        CL_CHECK(update_weights_eh_ev_kernel.setArg(12, learning_rate));
        CL_CHECK(update_weights_eh_ev_kernel.setArg(13, update_eh));
        CL_CHECK(update_weights_eh_ev_kernel.setArg(14, update_ev));
        CL_CHECK(update_weights_eh_ev_kernel.setArg(15, mat_heights));
        CL_CHECK(update_weights_eh_ev_kernel.setArg(16, embedding_dim));
        CL_CHECK(update_weights_eh_ev_kernel.setArg(17, context_win));
        // Launch with a grid size covering the largest update target (likely EV)
        CL_CHECK(queue.enqueueNDRangeKernel(update_weights_eh_ev_kernel, cl::NullRange, global_ev, local_1d));
        CL_CHECK(queue.finish());

        // --- Data Transfer D->H ---
        // Copy updated MLP weights and gradients back
        for (int l = 0; l < layers-1; ++l) {
            // Hor MLP
            CL_CHECK(queue.enqueueReadBuffer(d_hor_weights[l], CL_TRUE, 0, mlp_weights_bytes, this->hor.weights[l].mapped_data));
            CL_CHECK(queue.enqueueReadBuffer(d_hor_gweights[l], CL_TRUE, 0, mlp_weights_bytes, this->hor.gweights[l].mapped_data));
            // Ver MLP
            CL_CHECK(queue.enqueueReadBuffer(d_ver_weights[l], CL_TRUE, 0, mlp_weights_bytes, this->ver.weights[l].mapped_data));
            CL_CHECK(queue.enqueueReadBuffer(d_ver_gweights[l], CL_TRUE, 0, mlp_weights_bytes, this->ver.gweights[l].mapped_data));
        }

        // Copy updated Attention parameters back
        CL_CHECK(queue.enqueueReadBuffer(d_EH, CL_TRUE, 0, embed_bytes, this->EH.data()));
        CL_CHECK(queue.enqueueReadBuffer(d_EV, CL_TRUE, 0, ev_bytes, this->EV.mapped_data));
        CL_CHECK(queue.enqueueReadBuffer(d_MH_a, CL_TRUE, 0, mh_mv_mq_mk_bytes, this->MH.mapped_data));
        CL_CHECK(queue.enqueueReadBuffer(d_MV_a, CL_TRUE, 0, mh_mv_mq_mk_bytes, this->MV.mapped_data));
        CL_CHECK(queue.enqueueReadBuffer(d_MQ_a, CL_TRUE, 0, mh_mv_mq_mk_bytes, this->MQ.mapped_data));
        CL_CHECK(queue.enqueueReadBuffer(d_MK_a, CL_TRUE, 0, mh_mv_mq_mk_bytes, this->MK.mapped_data));
    }
    catch (const std::exception& e) {
        std::cerr << "Standard Exception in clbackward(expected): " << e.what() << std::endl;
        throw;
    }
    // Buffers are automatically released when they go out of scope (RAII)
}


/**
 * @brief OpenCL Backward Propagation using gradients from expected Vertical output only.
 *      Updates MV, MQ, MK (correction), and EV.
 * @param expectedV vertical retention vector (host)
 * @param in Input size (embedding dimension - used for MLP)
 * @param layers Number of layers in the MLPs
 */
void attention::clbackward(std::vector<std::vector<float>>& expectedV, int& layers, int& blocknumber)
{
    // --- Setup ---
    const int embedding_dim = EMBEDDING;
    const int mat_heights = MATHEIGHTS;
    const int context_win = CONTEXT_WIN;
    const float learning_rate = LEARNING;
    const float scaling_factor = SCALING;
    const int token_count = this->tokenCount;

    // Calculate sizes from mat objects
    const size_t k_q_bytes = static_cast<size_t>(this->K.row) * this->K.col * sizeof(float);
    const size_t head_bytes = static_cast<size_t>(this->KdotQ.row) * this->KdotQ.col * sizeof(float);
    const size_t mh_mv_mq_mk_bytes = static_cast<size_t>(this->MV.row) * this->MV.col * sizeof(float); // Using MV for size, assuming MQ/MK same
    const size_t ev_bytes = static_cast<size_t>(this->EV.row) * this->EV.col * sizeof(float);
    const size_t embed_bytes = embedding_dim * sizeof(float);
    const size_t mlp_weights_bytes = (this->ver.weights.empty()) ? 0 :
                                   static_cast<size_t>(this->ver.weights[0].row) * this->ver.weights[0].col * sizeof(float);

    // Validation
    if (expectedV.size() != context_win || (!expectedV.empty() && expectedV[0].size() != embedding_dim)) {
        throw std::runtime_error("ExpectedV dimensions mismatch");
    }
    if (this->K.row != CONTEXT_WIN || this->K.col != MATHEIGHTS) throw std::runtime_error("K dimensions mismatch");
    if (this->Q.row != CONTEXT_WIN || this->Q.col != MATHEIGHTS) throw std::runtime_error("Q dimensions mismatch");
    if (this->KdotQ.row != CONTEXT_WIN || this->KdotQ.col != CONTEXT_WIN) throw std::runtime_error("KdotQ dimensions mismatch");
    if (this->MV.row != MATHEIGHTS || this->MV.col != EMBEDDING) throw std::runtime_error("MV dimensions mismatch");
    if (this->EV.row != CONTEXT_WIN || this->EV.col != EMBEDDING) throw std::runtime_error("EV dimensions mismatch");
    if (!this->ver.weights.empty() && (this->ver.weights[0].row != EMBEDDING || this->ver.weights[0].col != EMBEDDING))
        throw std::runtime_error("MLP ver.weights dimensions mismatch");

    if (token_count <= 0) {
        std::cerr << "Warning: clbackward(expectedV,...) called with token_count <= 0. Skipping." << std::endl;
        return;
    }

    int update_ev = (blocknumber == 1) ? 0 : 1;

    cl_int cl_err; // For OpenCL error codes

    OpenCLContext& context_obj = this->clcontext;
    cl::Context context = context_obj.context;
    cl::CommandQueue queue = context_obj.queue;
    // Use context_obj.kernels map below

    std::vector<float> flat_expectedV = flatten(expectedV); // expectedV is std::vector<std::vector<float>>

    // --- Allocate Memory (Attention) ---
    cl::Buffer d_expected_v(context, CL_MEM_READ_ONLY, ev_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
    cl::Buffer d_EV(context, CL_MEM_READ_WRITE, ev_bytes, nullptr, &cl_err); CL_CHECK(cl_err); // Read/Write for update
    cl::Buffer d_grad_EV_full(context, CL_MEM_READ_WRITE, ev_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
    cl::Buffer d_grad_EV_summed(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
    cl::Buffer d_grad_EV_scaled(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
    cl::Buffer d_grad_dv(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
    cl::Buffer d_KdotQ(context, CL_MEM_READ_ONLY, head_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
    cl::Buffer d_head(context, CL_MEM_READ_WRITE, head_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
    cl::Buffer d_K(context, CL_MEM_READ_ONLY, k_q_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
    cl::Buffer d_Q(context, CL_MEM_READ_ONLY, k_q_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
    cl::Buffer d_pre_MV(context, CL_MEM_READ_WRITE, mat_heights * sizeof(float), nullptr, &cl_err); CL_CHECK(cl_err);
    cl::Buffer d_MV_a(context, CL_MEM_READ_WRITE, mh_mv_mq_mk_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
    cl::Buffer d_MQ_a(context, CL_MEM_READ_WRITE, mh_mv_mq_mk_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
    cl::Buffer d_MK_a(context, CL_MEM_READ_WRITE, mh_mv_mq_mk_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
    cl::Buffer d_grad_MV(context, CL_MEM_READ_WRITE, mh_mv_mq_mk_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
    cl::Buffer d_grad_head(context, CL_MEM_READ_WRITE, head_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
    cl::Buffer d_lota_deriv(context, CL_MEM_READ_WRITE, head_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
    cl::Buffer d_grad_KdotQ(context, CL_MEM_READ_WRITE, head_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
    cl::Buffer d_grad_Q(context, CL_MEM_READ_WRITE, k_q_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
    cl::Buffer d_grad_MQ(context, CL_MEM_READ_WRITE, mh_mv_mq_mk_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
    cl::Buffer d_grad_MK_correction(context, CL_MEM_READ_WRITE, mh_mv_mq_mk_bytes, nullptr, &cl_err); CL_CHECK(cl_err);

    // --- Allocate Memory (ver MLP Internals) ---
    std::vector<cl::Buffer> d_ver_activations;
    std::vector<cl::Buffer> d_ver_weights;
    std::vector<cl::Buffer> d_ver_gweights;
    std::vector<cl::Buffer> d_ver_deltas;
    d_ver_activations.reserve(layers); // Should be 'layers' if activations[0] is input
    d_ver_weights.reserve(layers-1);   // Should be 'layers-1'
    d_ver_gweights.reserve(layers-1);  // Should be 'layers-1'
    d_ver_deltas.reserve(layers);      // Should be 'layers'

    for (int l = 0; l < layers-1; ++l) {
        d_ver_weights.emplace_back(context, CL_MEM_READ_WRITE, mlp_weights_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        d_ver_gweights.emplace_back(context, CL_MEM_READ_WRITE, mlp_weights_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
    }
    for (int l = 0; l < layers; ++l) {
        d_ver_activations.emplace_back(context, CL_MEM_READ_ONLY, embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        d_ver_deltas.emplace_back(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
    }

    // --- Data Transfer H->D (Attention) ---
    CL_CHECK(queue.enqueueWriteBuffer(d_expected_v, CL_TRUE, 0, ev_bytes, flat_expectedV.data()));
    CL_CHECK(queue.enqueueWriteBuffer(d_EV, CL_TRUE, 0, ev_bytes, this->EV.mapped_data));
    CL_CHECK(queue.enqueueWriteBuffer(d_KdotQ, CL_TRUE, 0, head_bytes, this->KdotQ.mapped_data));
    CL_CHECK(queue.enqueueWriteBuffer(d_K, CL_TRUE, 0, k_q_bytes, this->K.mapped_data));
    CL_CHECK(queue.enqueueWriteBuffer(d_Q, CL_TRUE, 0, k_q_bytes, this->Q.mapped_data));
    CL_CHECK(queue.enqueueWriteBuffer(d_MV_a, CL_TRUE, 0, mh_mv_mq_mk_bytes, this->MV.mapped_data));
    CL_CHECK(queue.enqueueWriteBuffer(d_MQ_a, CL_TRUE, 0, mh_mv_mq_mk_bytes, this->MQ.mapped_data));
    CL_CHECK(queue.enqueueWriteBuffer(d_MK_a, CL_TRUE, 0, mh_mv_mq_mk_bytes, this->MK.mapped_data));

    // --- Data Transfer H->D (ver MLP Internals) ---
    for (int l = 0; l < layers; ++l) { // Copy all 'layers' activations if activations[0] is input
        CL_CHECK(queue.enqueueWriteBuffer(d_ver_activations[l], CL_TRUE, 0, embed_bytes, this->ver.activations[l].data()));
    }
    for (int l = 0; l < layers-1; ++l) { // Copy 'layers-1' weights
        CL_CHECK(queue.enqueueWriteBuffer(d_ver_weights[l], CL_TRUE, 0, mlp_weights_bytes, this->ver.weights[l].mapped_data));
    }

    try {
        // --- Kernel Launch Config ---
        const size_t local_work_size_1d = 256;
        auto calculate_global_1d = [&](size_t total_size) {
            return ((total_size + local_work_size_1d - 1) / local_work_size_1d) * local_work_size_1d;
        };
        cl::NDRange global_embed(calculate_global_1d(embedding_dim));
        cl::NDRange global_head(calculate_global_1d(this->KdotQ.row * this->KdotQ.col));
        cl::NDRange global_mat_heights(calculate_global_1d(mat_heights));
        cl::NDRange global_matrix(calculate_global_1d(this->MV.row * this->MV.col));
        cl::NDRange global_ev(calculate_global_1d(this->EV.row * this->EV.col));
        cl::NDRange local_1d(local_work_size_1d);

        size_t local_work_size_2d[2] = { 16, 16 };
        auto calculate_global_2d = [&](size_t dim0, size_t dim1) {
            size_t global0 = ((dim0 + local_work_size_2d[0] - 1) / local_work_size_2d[0]) * local_work_size_2d[0];
            size_t global1 = ((dim1 + local_work_size_2d[1] - 1) / local_work_size_2d[1]) * local_work_size_2d[1];
            return cl::NDRange(global0, global1);
        };
        cl::NDRange global_embed_2d = calculate_global_2d(embedding_dim, embedding_dim);
        cl::NDRange global_head_2d = calculate_global_2d(token_count, token_count);
        cl::NDRange global_matrix_2d = calculate_global_2d(embedding_dim, mat_heights); // dim0=embed, dim1=height
        // For kernels operating on K, Q and their gradients (token_count x embedding_dim)
        cl::NDRange global_kq_grad_2d = calculate_global_2d(embedding_dim, token_count); // dim0 for embedding_dim, dim1 for token_count
        cl::NDRange local_2d(local_work_size_2d[0], local_work_size_2d[1]);

        // --- Backpropagation Steps ---

        // Step 1: Compute grad_EV (full, summed, scaled)
        cl::Kernel grad_ev_v_kernel = context_obj.kernels.at("kernelComputeGradientsEV_V");
        CL_CHECK(grad_ev_v_kernel.setArg(0, d_EV));
        CL_CHECK(grad_ev_v_kernel.setArg(1, d_expected_v));
        CL_CHECK(grad_ev_v_kernel.setArg(2, d_grad_EV_full));
        CL_CHECK(grad_ev_v_kernel.setArg(3, d_grad_EV_summed));
        CL_CHECK(grad_ev_v_kernel.setArg(4, d_grad_EV_scaled));
        CL_CHECK(grad_ev_v_kernel.setArg(5, learning_rate));
        CL_CHECK(grad_ev_v_kernel.setArg(6, context_win));
        CL_CHECK(grad_ev_v_kernel.setArg(7, embedding_dim));
        CL_CHECK(queue.enqueueNDRangeKernel(grad_ev_v_kernel, cl::NullRange, global_embed, local_1d)); // Use global_embed as it sums to embed_dim
        CL_CHECK(queue.finish());

        // --- Step 2: Backprop through ver MLP ---
        cl::Kernel last_delta_kernel = context_obj.kernels.at("kernelLastLayerDeltaSigmoid");
        cl::Kernel hidden_delta_kernel = context_obj.kernels.at("kernelHiddenDeltaSigmoid");
        cl::Kernel update_weights_kernel = context_obj.kernels.at("kernelUpdateWeightsAndGradients");

        CL_CHECK(last_delta_kernel.setArg(0, d_grad_EV_scaled));
        CL_CHECK(last_delta_kernel.setArg(1, d_ver_activations[layers - 1]));
        CL_CHECK(last_delta_kernel.setArg(2, d_ver_deltas[layers - 1]));
        CL_CHECK(last_delta_kernel.setArg(3, embedding_dim));
        CL_CHECK(queue.enqueueNDRangeKernel(last_delta_kernel, cl::NullRange, global_embed, local_1d));
        CL_CHECK(queue.finish());

        for (int l = layers - 2; l >= 0; --l) {
            CL_CHECK(hidden_delta_kernel.setArg(0, d_ver_deltas[l + 1]));
            CL_CHECK(hidden_delta_kernel.setArg(1, d_ver_weights[l])); // Corrected index
            CL_CHECK(hidden_delta_kernel.setArg(2, d_ver_activations[l]));
            CL_CHECK(hidden_delta_kernel.setArg(3, d_ver_deltas[l]));
            CL_CHECK(hidden_delta_kernel.setArg(4, embedding_dim));
            CL_CHECK(hidden_delta_kernel.setArg(5, embedding_dim));
            CL_CHECK(queue.enqueueNDRangeKernel(hidden_delta_kernel, cl::NullRange, global_embed, local_1d));
            CL_CHECK(queue.finish());
        }

        for (int l = 0; l < layers - 1; ++l) { // Corrected loop bound
            // CAUTION: Assuming input to ver MLP layer 0 is size embedding_dim.
            cl::Buffer& d_prev_activations_ver = (l == 0) ? d_ver_activations[0] : d_ver_activations[l-1];
            int prev_layer_size = (l == 0) ? embedding_dim : embedding_dim;
            // Use global_embed_2d as sizes are embedding_dim x embedding_dim

            CL_CHECK(update_weights_kernel.setArg(0, d_ver_deltas[l]));
            CL_CHECK(update_weights_kernel.setArg(1, d_prev_activations_ver));
            CL_CHECK(update_weights_kernel.setArg(2, d_ver_weights[l]));
            CL_CHECK(update_weights_kernel.setArg(3, d_ver_gweights[l])); // d_ver_gweights also has layers-1 elements
            CL_CHECK(update_weights_kernel.setArg(4, learning_rate));
            CL_CHECK(update_weights_kernel.setArg(5, embedding_dim));
            CL_CHECK(update_weights_kernel.setArg(6, prev_layer_size)); // Input size might vary if not square
            CL_CHECK(queue.enqueueNDRangeKernel(update_weights_kernel, cl::NullRange, global_embed_2d, local_2d));
            CL_CHECK(queue.finish());
        }
        // --- End of MLP Backprop ---

        // --- Step 3: Compute grad_dv ---
        cl::Kernel grad_mlp_input_kernel = context_obj.kernels.at("kernelComputeGradMLPInput");
        CL_CHECK(grad_mlp_input_kernel.setArg(0, d_ver_deltas[0]));
        CL_CHECK(grad_mlp_input_kernel.setArg(1, d_ver_weights[0]));
        CL_CHECK(grad_mlp_input_kernel.setArg(2, d_grad_dv));
        CL_CHECK(grad_mlp_input_kernel.setArg(3, embedding_dim));
        CL_CHECK(grad_mlp_input_kernel.setArg(4, embedding_dim));
        CL_CHECK(queue.enqueueNDRangeKernel(grad_mlp_input_kernel, cl::NullRange, global_embed, local_1d));
        CL_CHECK(queue.finish());

        // --- Step 4: Compute grad_MV ---
        cl::Kernel lota_kernel = context_obj.kernels.at("clLOTA2d");
        CL_CHECK(lota_kernel.setArg(0, d_KdotQ));
        CL_CHECK(lota_kernel.setArg(1, d_head));
        CL_CHECK(lota_kernel.setArg(2, token_count));
        CL_CHECK(lota_kernel.setArg(3, token_count));
        size_t lota_global_raw = this->KdotQ.row * this->KdotQ.col;
        size_t lota_local_clamped = (std::min)(lota_global_raw, local_work_size_1d); // Parenthesize std::min
        if (lota_local_clamped == 0) lota_local_clamped = 1;
        size_t lota_global_padded = ((lota_global_raw + lota_local_clamped - 1) / lota_local_clamped) * lota_local_clamped;
        cl::NDRange global_lota(lota_global_padded);
        cl::NDRange local_lota(lota_local_clamped);
        if (this->KdotQ.row * this->KdotQ.col > 0) {
            CL_CHECK(queue.enqueueNDRangeKernel(lota_kernel, cl::NullRange, global_lota, local_lota));
        }

        cl::Kernel pre_mv_v_kernel = context_obj.kernels.at("kernelComputePreMV_V");
        CL_CHECK(pre_mv_v_kernel.setArg(0, d_head));
        CL_CHECK(pre_mv_v_kernel.setArg(1, d_Q));
        CL_CHECK(pre_mv_v_kernel.setArg(2, d_pre_MV));
        CL_CHECK(pre_mv_v_kernel.setArg(3, token_count));
        CL_CHECK(pre_mv_v_kernel.setArg(4, mat_heights));
        CL_CHECK(queue.enqueueNDRangeKernel(pre_mv_v_kernel, cl::NullRange, global_mat_heights, local_1d));

        cl::Kernel grad_mv_v_kernel = context_obj.kernels.at("kernelComputeGradMV_V");
        CL_CHECK(grad_mv_v_kernel.setArg(0, d_pre_MV));
        CL_CHECK(grad_mv_v_kernel.setArg(1, d_grad_dv));
        CL_CHECK(grad_mv_v_kernel.setArg(2, d_grad_MV));
        CL_CHECK(grad_mv_v_kernel.setArg(3, mat_heights));
        CL_CHECK(grad_mv_v_kernel.setArg(4, embedding_dim));
        CL_CHECK(queue.enqueueNDRangeKernel(grad_mv_v_kernel, cl::NullRange, global_matrix_2d, local_2d));

        // --- Step 5: Compute grad_head (dv part only) ---
        cl::Kernel grad_head_v_kernel = context_obj.kernels.at("kernelComputeGradHead_V");
        CL_CHECK(grad_head_v_kernel.setArg(0, d_Q));
        CL_CHECK(grad_head_v_kernel.setArg(1, d_MV_a));
        CL_CHECK(grad_head_v_kernel.setArg(2, d_grad_dv));
        CL_CHECK(grad_head_v_kernel.setArg(3, d_grad_head));
        CL_CHECK(grad_head_v_kernel.setArg(4, token_count));
        CL_CHECK(grad_head_v_kernel.setArg(5, mat_heights));
        CL_CHECK(grad_head_v_kernel.setArg(6, embedding_dim));
        CL_CHECK(queue.enqueueNDRangeKernel(grad_head_v_kernel, cl::NullRange, global_head_2d, local_2d));

        // --- Step 6: Backprop through LOTA ---
        cl::Kernel lota_deriv_kernel = context_obj.kernels.at("clLOTA2dder");
        CL_CHECK(lota_deriv_kernel.setArg(0, d_KdotQ));
        CL_CHECK(lota_deriv_kernel.setArg(1, d_lota_deriv));
        CL_CHECK(lota_deriv_kernel.setArg(2, token_count));
        CL_CHECK(lota_deriv_kernel.setArg(3, token_count));
        if (this->KdotQ.row * this->KdotQ.col > 0) {
            CL_CHECK(queue.enqueueNDRangeKernel(lota_deriv_kernel, cl::NullRange, global_lota, local_lota));
        }

        cl::Kernel grad_kdotq_kernel = context_obj.kernels.at("kernelComputeGradKdotQ_LOTA");
        CL_CHECK(grad_kdotq_kernel.setArg(0, d_grad_head));
        CL_CHECK(grad_kdotq_kernel.setArg(1, d_lota_deriv));
        CL_CHECK(grad_kdotq_kernel.setArg(2, d_grad_KdotQ));
        CL_CHECK(grad_kdotq_kernel.setArg(3, scaling_factor));
        CL_CHECK(grad_kdotq_kernel.setArg(4, this->KdotQ.row * this->KdotQ.col));
        CL_CHECK(queue.enqueueNDRangeKernel(grad_kdotq_kernel, cl::NullRange, global_head, local_1d));

        // --- Step 7: Compute grad_Q ---
        cl::Kernel grad_q_v_kernel = context_obj.kernels.at("kernelComputeGradQ_V");
        CL_CHECK(grad_q_v_kernel.setArg(0, d_grad_KdotQ));
        CL_CHECK(grad_q_v_kernel.setArg(1, d_K));
        CL_CHECK(grad_q_v_kernel.setArg(2, d_grad_Q));
        CL_CHECK(grad_q_v_kernel.setArg(3, token_count));
        CL_CHECK(grad_q_v_kernel.setArg(4, embedding_dim)); // Pass embedding_dim as the K/Q column dimension
        CL_CHECK(queue.enqueueNDRangeKernel(grad_q_v_kernel, cl::NullRange, global_kq_grad_2d, local_2d)); // Launch with embedding_dim for h-like dimension

        // --- Step 8: Compute grad_MQ and grad_MK_correction (Complex) ---
        cl::Kernel grad_mq_v_kernel = context_obj.kernels.at("kernelComputeGradMQ_V");
        // cl::Buffer d_null(context, 0, 0, nullptr, &cl_err);
        cl::Buffer d_null;
        if (cl_err != CL_SUCCESS && cl_err != CL_INVALID_BUFFER_SIZE) CL_CHECK(cl_err);
        CL_CHECK(grad_mq_v_kernel.setArg(0, d_grad_Q));
        CL_CHECK(grad_mq_v_kernel.setArg(1, d_null)); // d_Q_embed
        CL_CHECK(grad_mq_v_kernel.setArg(2, d_grad_MQ));
        CL_CHECK(grad_mq_v_kernel.setArg(3, token_count));
        CL_CHECK(grad_mq_v_kernel.setArg(4, mat_heights));
        CL_CHECK(grad_mq_v_kernel.setArg(5, embedding_dim));
        CL_CHECK(queue.enqueueNDRangeKernel(grad_mq_v_kernel, cl::NullRange, global_matrix_2d, local_2d));

        cl::Kernel grad_mk_corr_kernel = context_obj.kernels.at("kernelComputeGradMKCorrection");
        CL_CHECK(grad_mk_corr_kernel.setArg(0, d_grad_MQ));
        CL_CHECK(grad_mk_corr_kernel.setArg(1, d_Q));
        CL_CHECK(grad_mk_corr_kernel.setArg(2, d_K));
        CL_CHECK(grad_mk_corr_kernel.setArg(3, d_grad_MK_correction));
        CL_CHECK(grad_mk_corr_kernel.setArg(4, token_count));
        CL_CHECK(grad_mk_corr_kernel.setArg(5, mat_heights));
        CL_CHECK(grad_mk_corr_kernel.setArg(6, embedding_dim));
        CL_CHECK(queue.enqueueNDRangeKernel(grad_mk_corr_kernel, cl::NullRange, global_matrix_2d, local_2d));

        // --- Step 9 & 10: Update Weights (MV, MQ, MK_correction) and EV ---
        cl::Kernel update_weights_ev_v_kernel = context_obj.kernels.at("kernelUpdateWeights_EV_V");
        CL_CHECK(update_weights_ev_v_kernel.setArg(0, d_MV_a));
        CL_CHECK(update_weights_ev_v_kernel.setArg(1, d_MQ_a));
        CL_CHECK(update_weights_ev_v_kernel.setArg(2, d_MK_a));
        CL_CHECK(update_weights_ev_v_kernel.setArg(3, d_EV));
        CL_CHECK(update_weights_ev_v_kernel.setArg(4, d_grad_MV));
        CL_CHECK(update_weights_ev_v_kernel.setArg(5, d_grad_MQ));
        CL_CHECK(update_weights_ev_v_kernel.setArg(6, d_grad_MK_correction));
        CL_CHECK(update_weights_ev_v_kernel.setArg(7, d_grad_EV_full)); // Use full EV gradient
        CL_CHECK(update_weights_ev_v_kernel.setArg(8, learning_rate));
        CL_CHECK(update_weights_ev_v_kernel.setArg(9, update_ev));
        CL_CHECK(update_weights_ev_v_kernel.setArg(10, mat_heights));
        CL_CHECK(update_weights_ev_v_kernel.setArg(11, embedding_dim));
        CL_CHECK(update_weights_ev_v_kernel.setArg(12, context_win));
        // Launch with grid size covering largest update target (EV)
        CL_CHECK(queue.enqueueNDRangeKernel(update_weights_ev_v_kernel, cl::NullRange, global_ev, local_1d));
        CL_CHECK(queue.finish());

        // --- Data Transfer D->H ---
        // Copy updated ver MLP weights and gradients back
        for (int l = 0; l < layers - 1; ++l) { // Corrected loop bound
            CL_CHECK(queue.enqueueReadBuffer(d_ver_weights[l], CL_TRUE, 0, mlp_weights_bytes, this->ver.weights[l].mapped_data));
            CL_CHECK(queue.enqueueReadBuffer(d_ver_gweights[l], CL_TRUE, 0, mlp_weights_bytes, this->ver.gweights[l].mapped_data));
        }

        // Copy updated Attention parameters back
        CL_CHECK(queue.enqueueReadBuffer(d_EV, CL_TRUE, 0, ev_bytes, this->EV.mapped_data));
        CL_CHECK(queue.enqueueReadBuffer(d_MV_a, CL_TRUE, 0, mh_mv_mq_mk_bytes, this->MV.mapped_data));
        CL_CHECK(queue.enqueueReadBuffer(d_MQ_a, CL_TRUE, 0, mh_mv_mq_mk_bytes, this->MQ.mapped_data));
        CL_CHECK(queue.enqueueReadBuffer(d_MK_a, CL_TRUE, 0, mh_mv_mq_mk_bytes, this->MK.mapped_data));
    } 
    catch (const std::exception& e) {
        std::cerr << "Standard Exception in clbackward(expectedV): " << e.what() << std::endl;
        throw;
    }
}

#endif // USE_OPENCL
