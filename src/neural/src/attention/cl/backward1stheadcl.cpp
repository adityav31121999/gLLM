#ifdef USE_OPENCL

#define CL_HPP_ENABLE_EXCEPTIONS // Enable exceptions before including cl.hpp
#include "include/attention.hpp" // Includes mlp.hpp and maths.hpp indirectly or directly
#include <vector>
#include <string>
#include <stdexcept>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <map>
#include <CL/cl.hpp>

/**
* @brief OpenCL Backward Propagation (for first head) using gradients from expected Horizontal output.
*      Updates MH, MV, MQ, MK, and EH.
*      Uses the clContext member for OpenCL resources.
* @param expected Expected output vector (target embedding for next token prediction)
* @param in Input size (embedding dimension)
* @param layers Number of layers in the MLPs
*/
void attention::clbackward1stHead(std::vector<float>& expected, int& in, int& layers)
{
    // --- Setup ---
    const int embedding_dim = EMBEDDING;
    const int mat_heights = MATHEIGHTS;
    const int context_win = CONTEXT_WIN;
    const float learning_rate = LEARNING;
    const float scaling_factor = SCALING;
    const int token_count = this->tokenCount;
    const int head_size = token_count * token_count;
    const int mh_mv_mq_mk_size = mat_heights * embedding_dim;
    const int k_q_size = token_count * mat_heights;
    const size_t embed_bytes = embedding_dim * sizeof(float);
    const size_t weights_bytes = embedding_dim * embedding_dim * sizeof(float);
    const size_t ev_size = context_win * embedding_dim;

    bool first = true; // This overload implies EH update

    // Validation
    if (embedding_dim != in) throw std::runtime_error("Embedding dimension mismatch");
    if (expected.size() != embedding_dim) throw std::runtime_error("Expected vector size mismatch");
    if (token_count <= 0) {
        std::cerr << "Warning: clbackward1stHead(expectedH,...) called with token_count <= 0. Skipping." << std::endl;
        return;
    }
    // Add other necessary validation checks...

    // Temporary flat vectors for H->D copy
    std::vector<std::vector<float>> flat_hor_weights(layers);
    std::vector<std::vector<float>> flat_ver_weights(layers);
    OpenCLContext& context_obj = this->clcontext;
    cl::Context context = context_obj.context;
    cl::CommandQueue queue = context_obj.queue;
    // Use context_obj.kernels map below

    // --- Allocate Memory (Attention) ---
    cl::Buffer d_expected_h(context, CL_MEM_READ_ONLY, embed_bytes);
    cl::Buffer d_EH(context, CL_MEM_READ_WRITE, embed_bytes);
    cl::Buffer d_EV(context, CL_MEM_READ_ONLY, ev_size * sizeof(float)); // Added d_EV for MLP input
    cl::Buffer d_grad_EH(context, CL_MEM_READ_WRITE, embed_bytes);
    cl::Buffer d_grad_EV_scaled(context, CL_MEM_READ_WRITE, embed_bytes); // grad_EV_scaled is dummy here
    cl::Buffer d_grad_dh(context, CL_MEM_READ_WRITE, embed_bytes);
    cl::Buffer d_grad_dv(context, CL_MEM_READ_WRITE, embed_bytes);
    cl::Buffer d_KdotQ(context, CL_MEM_READ_ONLY, head_size * sizeof(float));
    cl::Buffer d_head(context, CL_MEM_READ_WRITE, head_size * sizeof(float));
    cl::Buffer d_K(context, CL_MEM_READ_ONLY, k_q_size * sizeof(float));
    cl::Buffer d_Q(context, CL_MEM_READ_ONLY, k_q_size * sizeof(float));
    cl::Buffer d_pre_MH(context, CL_MEM_READ_WRITE, mat_heights * sizeof(float));
    cl::Buffer d_pre_MV(context, CL_MEM_READ_WRITE, mat_heights * sizeof(float));
    cl::Buffer d_MH_a(context, CL_MEM_READ_WRITE, mh_mv_mq_mk_size * sizeof(float));
    cl::Buffer d_MV_a(context, CL_MEM_READ_WRITE, mh_mv_mq_mk_size * sizeof(float));
    cl::Buffer d_MQ_a(context, CL_MEM_READ_WRITE, mh_mv_mq_mk_size * sizeof(float));
    cl::Buffer d_MK_a(context, CL_MEM_READ_WRITE, mh_mv_mq_mk_size * sizeof(float));
    cl::Buffer d_grad_MH(context, CL_MEM_READ_WRITE, mh_mv_mq_mk_size * sizeof(float));
    cl::Buffer d_grad_MV(context, CL_MEM_READ_WRITE, mh_mv_mq_mk_size * sizeof(float));
    cl::Buffer d_grad_head(context, CL_MEM_READ_WRITE, head_size * sizeof(float));
    cl::Buffer d_lota_deriv(context, CL_MEM_READ_WRITE, head_size * sizeof(float));
    cl::Buffer d_grad_KdotQ(context, CL_MEM_READ_WRITE, head_size * sizeof(float));
    cl::Buffer d_grad_K(context, CL_MEM_READ_WRITE, k_q_size * sizeof(float));
    cl::Buffer d_grad_Q(context, CL_MEM_READ_WRITE, k_q_size * sizeof(float));
    cl::Buffer d_grad_MQ(context, CL_MEM_READ_WRITE, mh_mv_mq_mk_size * sizeof(float));
    cl::Buffer d_grad_MK(context, CL_MEM_READ_WRITE, mh_mv_mq_mk_size * sizeof(float));

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
    d_hor_weights.reserve(layers);
    d_hor_gweights.reserve(layers);
    d_hor_deltas.reserve(layers);
    d_ver_activations.reserve(layers);
    d_ver_weights.reserve(layers);
    d_ver_gweights.reserve(layers);
    d_ver_deltas.reserve(layers);

    for (int l = 0; l < layers; ++l) {
        d_hor_activations.emplace_back(context, CL_MEM_READ_ONLY, embed_bytes);
        d_hor_weights.emplace_back(context, CL_MEM_READ_WRITE, weights_bytes);
        d_hor_gweights.emplace_back(context, CL_MEM_READ_WRITE, weights_bytes);
        d_hor_deltas.emplace_back(context, CL_MEM_READ_WRITE, embed_bytes);
        d_ver_activations.emplace_back(context, CL_MEM_READ_ONLY, embed_bytes);
        d_ver_weights.emplace_back(context, CL_MEM_READ_WRITE, weights_bytes);
        d_ver_gweights.emplace_back(context, CL_MEM_READ_WRITE, weights_bytes);
        d_ver_deltas.emplace_back(context, CL_MEM_READ_WRITE, embed_bytes);
    }

    // --- Data Transfer H->D (Attention) ---
    std::vector<float> flat_K = flatten(this->K);
    std::vector<float> flat_Q = flatten(this->Q);
    std::vector<float> flat_EV = flatten(this->EV);
    std::vector<float> flat_KdotQ = flatten(this->KdotQ);
    std::vector<float> flat_MH_a = flatten(this->MH.a);
    std::vector<float> flat_MV_a = flatten(this->MV.a);
    std::vector<float> flat_MQ_a = flatten(this->MQ.a);
    std::vector<float> flat_MK_a = flatten(this->MK.a);

    queue.enqueueWriteBuffer(d_expected_h, CL_TRUE, 0, embed_bytes, expected.data());
    queue.enqueueWriteBuffer(d_EH, CL_TRUE, 0, embed_bytes, this->EH.data());
    queue.enqueueWriteBuffer(d_EV, CL_TRUE, 0, ev_size * sizeof(float), flat_EV.data());
    queue.enqueueWriteBuffer(d_KdotQ, CL_TRUE, 0, head_size * sizeof(float), flat_KdotQ.data());
    queue.enqueueWriteBuffer(d_K, CL_TRUE, 0, k_q_size * sizeof(float), flat_K.data());
    queue.enqueueWriteBuffer(d_Q, CL_TRUE, 0, k_q_size * sizeof(float), flat_Q.data());
    queue.enqueueWriteBuffer(d_MH_a, CL_TRUE, 0, mh_mv_mq_mk_size * sizeof(float), flat_MH_a.data());
    queue.enqueueWriteBuffer(d_MV_a, CL_TRUE, 0, mh_mv_mq_mk_size * sizeof(float), flat_MV_a.data());
    queue.enqueueWriteBuffer(d_MQ_a, CL_TRUE, 0, mh_mv_mq_mk_size * sizeof(float), flat_MQ_a.data());
    queue.enqueueWriteBuffer(d_MK_a, CL_TRUE, 0, mh_mv_mq_mk_size * sizeof(float), flat_MK_a.data());
    
    // --- Data Transfer H->D (MLP Internals) ---
    for (int l = 0; l < layers; ++l) {
        flat_hor_weights[l] = flatten(this->hor.weights[l]);
        flat_ver_weights[l] = flatten(this->ver.weights[l]);
        queue.enqueueWriteBuffer(d_hor_activations[l], CL_TRUE, 0, embed_bytes, this->hor.activations[l].data());
        queue.enqueueWriteBuffer(d_hor_weights[l], CL_TRUE, 0, weights_bytes, flat_hor_weights[l].data());
        queue.enqueueWriteBuffer(d_ver_activations[l], CL_TRUE, 0, embed_bytes, this->ver.activations[l].data());
        queue.enqueueWriteBuffer(d_ver_weights[l], CL_TRUE, 0, weights_bytes, flat_ver_weights[l].data());
    }

    try {
        // --- Kernel Launch Config ---
        const size_t local_work_size_1d = 256;
        auto calculate_global_1d = [&](size_t total_size) {
            return ((total_size + local_work_size_1d - 1) / local_work_size_1d) * local_work_size_1d;
        };
        cl::NDRange global_embed(calculate_global_1d(embedding_dim));
        cl::NDRange global_head(calculate_global_1d(head_size));
        cl::NDRange global_mat_heights(calculate_global_1d(mat_heights));
        cl::NDRange global_matrix(calculate_global_1d(mh_mv_mq_mk_size));
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
        cl::NDRange global_kq_grad_2d = calculate_global_2d(mat_heights, token_count); // dim0=height, dim1=token
        cl::NDRange local_2d(local_work_size_2d[0], local_work_size_2d[1]);

        // --- Backpropagation Steps ---

        // Step 1: Compute grad_EH and grad_EV_scaled (dummy)
        cl::Kernel grad_eh_ev_kernel = context_obj.kernels.at("clComputeGradientsEH_EV"); // Assumed name
        grad_eh_ev_kernel.setArg(0, d_EH);
        grad_eh_ev_kernel.setArg(1, d_expected_h);
        grad_eh_ev_kernel.setArg(2, d_grad_EH);
        grad_eh_ev_kernel.setArg(3, d_grad_EV_scaled); // Write dummy value
        grad_eh_ev_kernel.setArg(4, embedding_dim);
        queue.enqueueNDRangeKernel(grad_eh_ev_kernel, cl::NullRange, global_embed, local_1d);
        queue.finish(); // Sync before MLP backprop

        // --- Step 2: Backprop through MLPs ---
        cl::Kernel last_delta_kernel = context_obj.kernels.at("clLastLayerDeltaSigmoid"); // Assumed name
        cl::Kernel hidden_delta_kernel = context_obj.kernels.at("clHiddenDeltaSigmoid"); // Assumed name
        cl::Kernel update_weights_kernel = context_obj.kernels.at("clUpdateWeightsAndGradients"); // Assumed name

        // --- 2a: Backprop through hor MLP ---
        last_delta_kernel.setArg(0, d_grad_EH);
        last_delta_kernel.setArg(1, d_hor_activations[layers - 1]);
        last_delta_kernel.setArg(2, d_hor_deltas[layers - 1]);
        last_delta_kernel.setArg(3, embedding_dim);
        queue.enqueueNDRangeKernel(last_delta_kernel, cl::NullRange, global_embed, local_1d);
        queue.finish();

        for (int l = layers - 2; l >= 0; --l) {
            hidden_delta_kernel.setArg(0, d_hor_deltas[l + 1]);
            hidden_delta_kernel.setArg(1, d_hor_weights[l + 1]);
            hidden_delta_kernel.setArg(2, d_hor_activations[l]);
            hidden_delta_kernel.setArg(3, d_hor_deltas[l]);
            hidden_delta_kernel.setArg(4, embedding_dim);
            hidden_delta_kernel.setArg(5, embedding_dim);
            queue.enqueueNDRangeKernel(hidden_delta_kernel, cl::NullRange, global_embed, local_1d);
            queue.finish();
        }

        for (int l = 0; l < layers; ++l) {
            cl::Buffer& d_prev_activations = (l == 0) ? d_EH : d_hor_activations[l - 1];
            update_weights_kernel.setArg(0, d_hor_deltas[l]);
            update_weights_kernel.setArg(1, d_prev_activations);
            update_weights_kernel.setArg(2, d_hor_weights[l]);
            update_weights_kernel.setArg(3, d_hor_gweights[l]);
            update_weights_kernel.setArg(4, learning_rate);
            update_weights_kernel.setArg(5, embedding_dim);
            update_weights_kernel.setArg(6, embedding_dim);
            queue.enqueueNDRangeKernel(update_weights_kernel, cl::NullRange, global_embed_2d, local_2d);
            queue.finish();
        }

        // --- 2b: Backprop through ver MLP ---
        last_delta_kernel.setArg(0, d_grad_EV_scaled); // Use scaled EV gradient
        last_delta_kernel.setArg(1, d_ver_activations[layers - 1]);
        last_delta_kernel.setArg(2, d_ver_deltas[layers - 1]);
        // Arg 3 (embedding_dim) is already set
        queue.enqueueNDRangeKernel(last_delta_kernel, cl::NullRange, global_embed, local_1d);
        queue.finish();

        for (int l = layers - 2; l >= 0; --l) {
            hidden_delta_kernel.setArg(0, d_ver_deltas[l + 1]);
            hidden_delta_kernel.setArg(1, d_ver_weights[l + 1]);
            hidden_delta_kernel.setArg(2, d_ver_activations[l]);
            hidden_delta_kernel.setArg(3, d_ver_deltas[l]);
            // Args 4, 5 (embedding_dim) are already set
            queue.enqueueNDRangeKernel(hidden_delta_kernel, cl::NullRange, global_embed, local_1d);
            queue.finish();
        }

        for (int l = 0; l < layers; ++l) {
            // Assuming input to ver MLP layer 0 is size embedding_dim (needs verification based on forward pass)
            cl::Buffer& d_prev_activations = (l == 0) ? d_EV : d_ver_activations[l - 1]; // CAUTION: d_EV might be wrong size/pointer
            int prev_layer_size = (l == 0) ? embedding_dim : embedding_dim; // Adjust if needed
            // Use global_embed_2d as sizes are embedding_dim x embedding_dim

            update_weights_kernel.setArg(0, d_ver_deltas[l]);
            update_weights_kernel.setArg(1, d_prev_activations);
            update_weights_kernel.setArg(2, d_ver_weights[l]);
            update_weights_kernel.setArg(3, d_ver_gweights[l]);
            // Arg 4 (learning_rate) is already set
            update_weights_kernel.setArg(5, embedding_dim);
            update_weights_kernel.setArg(6, prev_layer_size);
            queue.enqueueNDRangeKernel(update_weights_kernel, cl::NullRange, global_embed_2d, local_2d);
            queue.finish();
        }
        // --- End of MLP Backprop ---

        // --- Step 3: Compute grad_dh and grad_dv ---
        cl::Kernel grad_mlp_input_kernel = context_obj.kernels.at("clComputeGradMLPInput"); // Assumed name
        grad_mlp_input_kernel.setArg(0, d_hor_deltas[0]);
        grad_mlp_input_kernel.setArg(1, d_hor_weights[0]);
        grad_mlp_input_kernel.setArg(2, d_grad_dh);
        grad_mlp_input_kernel.setArg(3, embedding_dim);
        grad_mlp_input_kernel.setArg(4, embedding_dim);
        queue.enqueueNDRangeKernel(grad_mlp_input_kernel, cl::NullRange, global_embed, local_1d);

        grad_mlp_input_kernel.setArg(0, d_ver_deltas[0]);
        grad_mlp_input_kernel.setArg(1, d_ver_weights[0]);
        grad_mlp_input_kernel.setArg(2, d_grad_dv);
        // Args 3, 4 (embedding_dim) are already set
        queue.enqueueNDRangeKernel(grad_mlp_input_kernel, cl::NullRange, global_embed, local_1d);
        queue.finish();

        // --- Step 4: Compute grad_MH and grad_MV ---
        cl::Kernel lota_kernel = context_obj.kernels.at("clLOTA2d"); // Assuming LOTA forward needed for pre_MH/MV calc
        lota_kernel.setArg(0, d_KdotQ);
        lota_kernel.setArg(1, d_head);
        lota_kernel.setArg(2, token_count); // rows
        lota_kernel.setArg(3, token_count); // cols
        // Adjust launch for single work-group if needed
        size_t lota_global_raw = head_size;
        size_t lota_local_clamped = std::min(lota_global_raw, local_work_size_1d);
        if (lota_local_clamped == 0) lota_local_clamped = 1;
        size_t lota_global_padded = ((lota_global_raw + lota_local_clamped - 1) / lota_local_clamped) * lota_local_clamped;
        cl::NDRange global_lota(lota_global_padded);
        cl::NDRange local_lota(lota_local_clamped);
         if (head_size > 0) {
             queue.enqueueNDRangeKernel(lota_kernel, cl::NullRange, global_lota, local_lota);
         }

        cl::Kernel pre_mh_mv_kernel = context_obj.kernels.at("clComputePreMH_MV"); // Assumed name
        pre_mh_mv_kernel.setArg(0, d_head);
        pre_mh_mv_kernel.setArg(1, d_K);
        pre_mh_mv_kernel.setArg(2, d_Q);
        pre_mh_mv_kernel.setArg(3, d_pre_MH);
        pre_mh_mv_kernel.setArg(4, d_pre_MV);
        pre_mh_mv_kernel.setArg(5, token_count);
        pre_mh_mv_kernel.setArg(6, mat_heights);
        queue.enqueueNDRangeKernel(pre_mh_mv_kernel, cl::NullRange, global_mat_heights, local_1d);

        cl::Kernel grad_mh_mv_kernel = context_obj.kernels.at("clComputeGradMH_MV"); // Assumed name
        grad_mh_mv_kernel.setArg(0, d_pre_MH);
        grad_mh_mv_kernel.setArg(1, d_pre_MV);
        grad_mh_mv_kernel.setArg(2, d_grad_dh);
        grad_mh_mv_kernel.setArg(3, d_grad_dv);
        grad_mh_mv_kernel.setArg(4, d_grad_MH);
        grad_mh_mv_kernel.setArg(5, d_grad_MV);
        grad_mh_mv_kernel.setArg(6, mat_heights);
        grad_mh_mv_kernel.setArg(7, embedding_dim);
        queue.enqueueNDRangeKernel(grad_mh_mv_kernel, cl::NullRange, global_matrix_2d, local_2d);

        // --- Step 5: Compute grad_head ---
        cl::Kernel grad_head_kernel = context_obj.kernels.at("clComputeGradHead"); // Assumed name
        grad_head_kernel.setArg(0, d_K);
        grad_head_kernel.setArg(1, d_Q);
        grad_head_kernel.setArg(2, d_MH_a);
        grad_head_kernel.setArg(3, d_MV_a);
        grad_head_kernel.setArg(4, d_grad_dh);
        grad_head_kernel.setArg(5, d_grad_dv);
        grad_head_kernel.setArg(6, d_grad_head);
        grad_head_kernel.setArg(7, token_count);
        grad_head_kernel.setArg(8, mat_heights);
        grad_head_kernel.setArg(9, embedding_dim);
        queue.enqueueNDRangeKernel(grad_head_kernel, cl::NullRange, global_head_2d, local_2d);

        // --- Step 6: Backprop through LOTA ---
        cl::Kernel lota_deriv_kernel = context_obj.kernels.at("clLOTA2dder"); // Or clComputeSimpleLOTAder
        lota_deriv_kernel.setArg(0, d_KdotQ); // Input is KdotQ
        lota_deriv_kernel.setArg(1, d_lota_deriv);
        lota_deriv_kernel.setArg(2, token_count); // rows
        lota_deriv_kernel.setArg(3, token_count); // cols
        // Adjust launch for single work-group if needed
         if (head_size > 0) {
             queue.enqueueNDRangeKernel(lota_deriv_kernel, cl::NullRange, global_lota, local_lota);
         }


        cl::Kernel grad_kdotq_kernel = context_obj.kernels.at("clComputeGradKdotQ_LOTA"); // Assumed name
        grad_kdotq_kernel.setArg(0, d_grad_head);
        grad_kdotq_kernel.setArg(1, d_lota_deriv);
        grad_kdotq_kernel.setArg(2, d_grad_KdotQ);
        grad_kdotq_kernel.setArg(3, scaling_factor);
        grad_kdotq_kernel.setArg(4, head_size);
        queue.enqueueNDRangeKernel(grad_kdotq_kernel, cl::NullRange, global_head, local_1d);

        // --- Step 7: Compute grad_K and grad_Q ---
        cl::Kernel grad_k_q_kernel = context_obj.kernels.at("clComputeGradK_Q"); // Assumed name
        grad_k_q_kernel.setArg(0, d_grad_KdotQ);
        grad_k_q_kernel.setArg(1, d_K);
        grad_k_q_kernel.setArg(2, d_Q);
        grad_k_q_kernel.setArg(3, d_grad_K);
        grad_k_q_kernel.setArg(4, d_grad_Q);
        grad_k_q_kernel.setArg(5, token_count);
        grad_k_q_kernel.setArg(6, mat_heights);
        queue.enqueueNDRangeKernel(grad_k_q_kernel, cl::NullRange, global_kq_grad_2d, local_2d);

        // --- Step 8: Compute grad_MK and grad_MQ (Simplified) ---
        cl::Kernel grad_mk_mq_kernel = context_obj.kernels.at("clComputeGradMK_MQ_Simplified"); // Assumed name
        cl::Buffer d_null; // Null buffer
        grad_mk_mq_kernel.setArg(0, d_grad_K);
        grad_mk_mq_kernel.setArg(1, d_grad_Q);
        grad_mk_mq_kernel.setArg(2, d_null); // d_K_embed (nullptr)
        grad_mk_mq_kernel.setArg(3, d_null); // d_Q_embed (nullptr)
        grad_mk_mq_kernel.setArg(4, d_grad_MK);
        grad_mk_mq_kernel.setArg(5, d_grad_MQ);
        grad_mk_mq_kernel.setArg(6, token_count);
        grad_mk_mq_kernel.setArg(7, mat_heights);
        grad_mk_mq_kernel.setArg(8, embedding_dim);
        queue.enqueueNDRangeKernel(grad_mk_mq_kernel, cl::NullRange, global_matrix_2d, local_2d);

        // --- Step 9 & 10: Update Weights (Conditionally EH) ---
        cl::Kernel update_weights_h_kernel = context_obj.kernels.at("clUpdateWeights_1stHead_H"); // Assumed name
        cl_int cl_first = first; // Convert bool to cl_int
        update_weights_h_kernel.setArg(0, d_MH_a);
        update_weights_h_kernel.setArg(1, d_MV_a);
        update_weights_h_kernel.setArg(2, d_MQ_a);
        update_weights_h_kernel.setArg(3, d_MK_a);
        update_weights_h_kernel.setArg(4, d_EH);
        update_weights_h_kernel.setArg(5, d_grad_MH);
        update_weights_h_kernel.setArg(6, d_grad_MV);
        update_weights_h_kernel.setArg(7, d_grad_MQ);
        update_weights_h_kernel.setArg(8, d_grad_MK);
        update_weights_h_kernel.setArg(9, d_grad_EH);
        update_weights_h_kernel.setArg(10, learning_rate);
        update_weights_h_kernel.setArg(11, cl_first); // Pass cl_int flag
        update_weights_h_kernel.setArg(12, mat_heights);
        update_weights_h_kernel.setArg(13, embedding_dim);
        queue.enqueueNDRangeKernel(update_weights_h_kernel, cl::NullRange, global_matrix, local_1d);
        queue.finish(); // Sync before D->H copy

        // --- Data Transfer D->H ---
        // Copy updated MLP weights and gradients back
        for (int l = 0; l < layers; ++l) {
            std::vector<float> updated_flat_weights(embedding_dim * embedding_dim);
            std::vector<float> calculated_flat_gradients(embedding_dim * embedding_dim);
            // Hor MLP
            queue.enqueueReadBuffer(d_hor_weights[l], CL_TRUE, 0, weights_bytes, updated_flat_weights.data());
            queue.enqueueReadBuffer(d_hor_gweights[l], CL_TRUE, 0, weights_bytes, calculated_flat_gradients.data());
            unflatten(updated_flat_weights, this->hor.weights[l], embedding_dim, embedding_dim);
            unflatten(calculated_flat_gradients, this->hor.gweights[l], embedding_dim, embedding_dim);
            // Ver MLP
            queue.enqueueReadBuffer(d_ver_weights[l], CL_TRUE, 0, weights_bytes, updated_flat_weights.data());
            queue.enqueueReadBuffer(d_ver_gweights[l], CL_TRUE, 0, weights_bytes, calculated_flat_gradients.data());
            unflatten(updated_flat_weights, this->ver.weights[l], embedding_dim, embedding_dim);
            unflatten(calculated_flat_gradients, this->ver.gweights[l], embedding_dim, embedding_dim);
        }

        // Copy updated Attention parameters back
        std::vector<float> updated_MH_a(mh_mv_mq_mk_size);
        std::vector<float> updated_MV_a(mh_mv_mq_mk_size);
        std::vector<float> updated_MQ_a(mh_mv_mq_mk_size);
        std::vector<float> updated_MK_a(mh_mv_mq_mk_size);

        if (first) { // Always true here
            queue.enqueueReadBuffer(d_EH, CL_TRUE, 0, embed_bytes, this->EH.data());
        }
        queue.enqueueReadBuffer(d_MH_a, CL_TRUE, 0, mh_mv_mq_mk_size * sizeof(float), updated_MH_a.data());
        queue.enqueueReadBuffer(d_MV_a, CL_TRUE, 0, mh_mv_mq_mk_size * sizeof(float), updated_MV_a.data());
        queue.enqueueReadBuffer(d_MQ_a, CL_TRUE, 0, mh_mv_mq_mk_size * sizeof(float), updated_MQ_a.data());
        queue.enqueueReadBuffer(d_MK_a, CL_TRUE, 0, mh_mv_mq_mk_size * sizeof(float), updated_MK_a.data());

        unflatten(updated_MH_a, this->MH.a, mat_heights, embedding_dim);
        unflatten(updated_MV_a, this->MV.a, mat_heights, embedding_dim);
        unflatten(updated_MQ_a, this->MQ.a, mat_heights, embedding_dim);
        unflatten(updated_MK_a, this->MK.a, mat_heights, embedding_dim);

    }
    catch (const cl::Error& err) {
        std::cerr << "OpenCL Error in clbackward1stHead(expectedH): " << err.what() << " (" << err.err() << ")" << std::endl;
        // Manual Cleanup (Attention) - RAII is bypassed
        clReleaseMemObject(d_expected_h()); clReleaseMemObject(d_EH()); clReleaseMemObject(d_EV());
        clReleaseMemObject(d_grad_EH()); clReleaseMemObject(d_grad_EV_scaled());
        clReleaseMemObject(d_grad_dh()); clReleaseMemObject(d_grad_dv());
        clReleaseMemObject(d_KdotQ()); clReleaseMemObject(d_head()); clReleaseMemObject(d_K()); clReleaseMemObject(d_Q());
        clReleaseMemObject(d_pre_MH()); clReleaseMemObject(d_pre_MV());
        clReleaseMemObject(d_MH_a()); clReleaseMemObject(d_MV_a()); clReleaseMemObject(d_MQ_a()); clReleaseMemObject(d_MK_a());
        clReleaseMemObject(d_grad_MH()); clReleaseMemObject(d_grad_MV()); clReleaseMemObject(d_grad_head());
        clReleaseMemObject(d_lota_deriv()); clReleaseMemObject(d_grad_KdotQ());
        clReleaseMemObject(d_grad_K()); clReleaseMemObject(d_grad_Q());
        clReleaseMemObject(d_grad_MQ()); clReleaseMemObject(d_grad_MK());
        // Manual Cleanup (MLP Internals)
        for (int l = 0; l < layers; ++l) {
            clReleaseMemObject(d_hor_activations[l]()); clReleaseMemObject(d_hor_weights[l]()); clReleaseMemObject(d_hor_gweights[l]()); clReleaseMemObject(d_hor_deltas[l]());
            clReleaseMemObject(d_ver_activations[l]()); clReleaseMemObject(d_ver_weights[l]()); clReleaseMemObject(d_ver_gweights[l]()); clReleaseMemObject(d_ver_deltas[l]());
        }
        throw;
    }
    catch (const std::exception& e) {
        std::cerr << "Standard Exception in clbackward1stHead(expectedH): " << e.what() << std::endl;
        // Manual Cleanup (Attention) - RAII is bypassed
        clReleaseMemObject(d_expected_h()); clReleaseMemObject(d_EH()); clReleaseMemObject(d_EV());
        clReleaseMemObject(d_grad_EH()); clReleaseMemObject(d_grad_EV_scaled());
        clReleaseMemObject(d_grad_dh()); clReleaseMemObject(d_grad_dv());
        clReleaseMemObject(d_KdotQ()); clReleaseMemObject(d_head()); clReleaseMemObject(d_K()); clReleaseMemObject(d_Q());
        clReleaseMemObject(d_pre_MH()); clReleaseMemObject(d_pre_MV());
        clReleaseMemObject(d_MH_a()); clReleaseMemObject(d_MV_a()); clReleaseMemObject(d_MQ_a()); clReleaseMemObject(d_MK_a());
        clReleaseMemObject(d_grad_MH()); clReleaseMemObject(d_grad_MV()); clReleaseMemObject(d_grad_head());
        clReleaseMemObject(d_lota_deriv()); clReleaseMemObject(d_grad_KdotQ());
        clReleaseMemObject(d_grad_K()); clReleaseMemObject(d_grad_Q());
        clReleaseMemObject(d_grad_MQ()); clReleaseMemObject(d_grad_MK());
        // Manual Cleanup (MLP Internals)
        for (int l = 0; l < layers; ++l) {
            clReleaseMemObject(d_hor_activations[l]()); clReleaseMemObject(d_hor_weights[l]()); clReleaseMemObject(d_hor_gweights[l]()); clReleaseMemObject(d_hor_deltas[l]());
            clReleaseMemObject(d_ver_activations[l]()); clReleaseMemObject(d_ver_weights[l]()); clReleaseMemObject(d_ver_gweights[l]()); clReleaseMemObject(d_ver_deltas[l]());
        }
        throw;
    }
    // Buffers are automatically released when they go out of scope (RAII)
}


/**
* @brief OpenCL Backward Propagation (for first head) using gradients from expected Vertical output only.
*      Adjusts MQ, MV, and MK (correction). No update to EH/EV.
*      Uses the clContext member for OpenCL resources.
* @param expectedV vertical retention vector (host)
* @param in Input size (embedding dimension - used for MLP)
* @param layers Number of layers in the MLPs
*/
void attention::clbackward1stHead(std::vector<std::vector<float>>& expectedV, int& in, int& layers)
{
    // --- Setup ---
    const int embedding_dim = EMBEDDING;
    const int mat_heights = MATHEIGHTS;
    const int context_win = CONTEXT_WIN;
    const float learning_rate = LEARNING;
    const float scaling_factor = SCALING;
    const int token_count = this->tokenCount;
    const int head_size = token_count * token_count;
    const int mh_mv_mq_mk_size = mat_heights * embedding_dim;
    const int k_q_size = token_count * mat_heights;
    const int ev_size = context_win * embedding_dim;
    const size_t embed_bytes = embedding_dim * sizeof(float);
    const size_t weights_bytes = embedding_dim * embedding_dim * sizeof(float);

    // Validation
    if (expectedV.size() != context_win || (!expectedV.empty() && expectedV[0].size() != embedding_dim)) {
        throw std::runtime_error("ExpectedV dimensions mismatch");
    }
     if (token_count <= 0) {
        std::cerr << "Warning: clbackward1stHead(expectedV,...) called with token_count <= 0. Skipping." << std::endl;
        return;
    }
    // Add other necessary validation checks...

    // Temporary flat vectors for H->D copy
    std::vector<std::vector<float>> flat_ver_weights(layers);
    OpenCLContext& context_obj = this->clcontext;
    cl::Context context = context_obj.context;
    cl::CommandQueue queue = context_obj.queue;
    // Use context_obj.kernels map below

    // --- Allocate Memory (Attention) ---
    cl::Buffer d_expected_v(context, CL_MEM_READ_ONLY, ev_size * sizeof(float));
    cl::Buffer d_EV(context, CL_MEM_READ_ONLY, ev_size * sizeof(float)); // Read-only
    cl::Buffer d_grad_EV_full(context, CL_MEM_READ_WRITE, ev_size * sizeof(float));
    cl::Buffer d_grad_EV_summed(context, CL_MEM_READ_WRITE, embed_bytes);
    cl::Buffer d_grad_EV_scaled(context, CL_MEM_READ_WRITE, embed_bytes);
    cl::Buffer d_grad_dv(context, CL_MEM_READ_WRITE, embed_bytes);
    cl::Buffer d_KdotQ(context, CL_MEM_READ_ONLY, head_size * sizeof(float));
    cl::Buffer d_head(context, CL_MEM_READ_WRITE, head_size * sizeof(float));
    cl::Buffer d_K(context, CL_MEM_READ_ONLY, k_q_size * sizeof(float));
    cl::Buffer d_Q(context, CL_MEM_READ_ONLY, k_q_size * sizeof(float));
    cl::Buffer d_pre_MV(context, CL_MEM_READ_WRITE, mat_heights * sizeof(float));
    cl::Buffer d_MV_a(context, CL_MEM_READ_WRITE, mh_mv_mq_mk_size * sizeof(float));
    cl::Buffer d_MQ_a(context, CL_MEM_READ_WRITE, mh_mv_mq_mk_size * sizeof(float));
    cl::Buffer d_MK_a(context, CL_MEM_READ_WRITE, mh_mv_mq_mk_size * sizeof(float));
    cl::Buffer d_grad_MV(context, CL_MEM_READ_WRITE, mh_mv_mq_mk_size * sizeof(float));
    cl::Buffer d_grad_head(context, CL_MEM_READ_WRITE, head_size * sizeof(float));
    cl::Buffer d_lota_deriv(context, CL_MEM_READ_WRITE, head_size * sizeof(float));
    cl::Buffer d_grad_KdotQ(context, CL_MEM_READ_WRITE, head_size * sizeof(float));
    cl::Buffer d_grad_Q(context, CL_MEM_READ_WRITE, k_q_size * sizeof(float));
    cl::Buffer d_grad_MQ(context, CL_MEM_READ_WRITE, mh_mv_mq_mk_size * sizeof(float));
    cl::Buffer d_grad_MK_correction(context, CL_MEM_READ_WRITE, mh_mv_mq_mk_size * sizeof(float));

    // --- Allocate Memory (ver MLP Internals) ---
    std::vector<cl::Buffer> d_ver_activations;
    std::vector<cl::Buffer> d_ver_weights;
    std::vector<cl::Buffer> d_ver_gweights;
    std::vector<cl::Buffer> d_ver_deltas;
    d_ver_activations.reserve(layers);
    d_ver_weights.reserve(layers);
    d_ver_gweights.reserve(layers);
    d_ver_deltas.reserve(layers);

    for (int l = 0; l < layers; ++l) {
        d_ver_activations.emplace_back(context, CL_MEM_READ_ONLY, embed_bytes);
        d_ver_weights.emplace_back(context, CL_MEM_READ_WRITE, weights_bytes);
        d_ver_gweights.emplace_back(context, CL_MEM_READ_WRITE, weights_bytes);
        d_ver_deltas.emplace_back(context, CL_MEM_READ_WRITE, embed_bytes);
    }

    // --- Data Transfer H->D (Attention) ---
    std::vector<float> flat_expectedV = flatten(expectedV);
    std::vector<float> flat_EV = flatten(this->EV);
    std::vector<float> flat_K = flatten(this->K);
    std::vector<float> flat_Q = flatten(this->Q);
    std::vector<float> flat_KdotQ = flatten(this->KdotQ);
    std::vector<float> flat_MV_a = flatten(this->MV.a);
    std::vector<float> flat_MQ_a = flatten(this->MQ.a);
    std::vector<float> flat_MK_a = flatten(this->MK.a);

    queue.enqueueWriteBuffer(d_expected_v, CL_TRUE, 0, ev_size * sizeof(float), flat_expectedV.data());
    queue.enqueueWriteBuffer(d_EV, CL_TRUE, 0, ev_size * sizeof(float), flat_EV.data());
    queue.enqueueWriteBuffer(d_KdotQ, CL_TRUE, 0, head_size * sizeof(float), flat_KdotQ.data());
    queue.enqueueWriteBuffer(d_K, CL_TRUE, 0, k_q_size * sizeof(float), flat_K.data());
    queue.enqueueWriteBuffer(d_Q, CL_TRUE, 0, k_q_size * sizeof(float), flat_Q.data());
    queue.enqueueWriteBuffer(d_MV_a, CL_TRUE, 0, mh_mv_mq_mk_size * sizeof(float), flat_MV_a.data());
    queue.enqueueWriteBuffer(d_MQ_a, CL_TRUE, 0, mh_mv_mq_mk_size * sizeof(float), flat_MQ_a.data());
    queue.enqueueWriteBuffer(d_MK_a, CL_TRUE, 0, mh_mv_mq_mk_size * sizeof(float), flat_MK_a.data());

    // --- Data Transfer H->D (ver MLP Internals) ---
    for (int l = 0; l < layers; ++l) {
        flat_ver_weights[l] = flatten(this->ver.weights[l]);
        queue.enqueueWriteBuffer(d_ver_activations[l], CL_TRUE, 0, embed_bytes, this->ver.activations[l].data());
        queue.enqueueWriteBuffer(d_ver_weights[l], CL_TRUE, 0, weights_bytes, flat_ver_weights[l].data());
    }

    try {
        // --- Kernel Launch Config ---
        const size_t local_work_size_1d = 256;
        size_t global_work_size_embed[1] = { (static_cast<size_t>(embedding_dim) + local_work_size_1d - 1) / local_work_size_1d * local_work_size_1d };
        size_t global_work_size_head[1] = { (static_cast<size_t>(head_size) + local_work_size_1d - 1) / local_work_size_1d * local_work_size_1d };
        size_t global_work_size_mat_heights[1] = { (static_cast<size_t>(mat_heights) + local_work_size_1d - 1) / local_work_size_1d * local_work_size_1d };
        size_t global_work_size_matrix[1] = { (static_cast<size_t>(mh_mv_mq_mk_size) + local_work_size_1d - 1) / local_work_size_1d * local_work_size_1d };
        auto calculate_global_1d = [&](size_t total_size) {
            return ((total_size + local_work_size_1d - 1) / local_work_size_1d) * local_work_size_1d;
        };
        cl::NDRange global_embed(calculate_global_1d(embedding_dim));
        cl::NDRange global_head(calculate_global_1d(head_size));
        cl::NDRange global_mat_heights(calculate_global_1d(mat_heights));
        cl::NDRange global_matrix(calculate_global_1d(mh_mv_mq_mk_size));
        cl::NDRange global_ev(calculate_global_1d(ev_size));
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
        cl::NDRange global_kq_grad_2d = calculate_global_2d(mat_heights, token_count); // dim0=height, dim1=token
        cl::NDRange local_2d(local_work_size_2d[0], local_work_size_2d[1]);

        // --- Backpropagation Steps ---

        // Step 1: Compute grad_EV (full, summed, scaled)
        cl::Kernel grad_ev_v_kernel = context_obj.kernels.at("clComputeGradientsEV_V"); // Assumed name
        grad_ev_v_kernel.setArg(0, d_EV);
        grad_ev_v_kernel.setArg(1, d_expected_v);
        grad_ev_v_kernel.setArg(2, d_grad_EV_full);
        grad_ev_v_kernel.setArg(3, d_grad_EV_summed);
        grad_ev_v_kernel.setArg(4, d_grad_EV_scaled);
        grad_ev_v_kernel.setArg(5, learning_rate);
        grad_ev_v_kernel.setArg(6, context_win);
        grad_ev_v_kernel.setArg(7, embedding_dim);
        queue.enqueueNDRangeKernel(grad_ev_v_kernel, cl::NullRange, global_embed, local_1d); // Use global_embed as it sums to embed_dim
        queue.finish(); // Sync before MLP backprop

        // --- Step 2: Backprop through ver MLP ---
        cl::Kernel last_delta_kernel = context_obj.kernels.at("clLastLayerDeltaSigmoid"); // Assumed name
        cl::Kernel hidden_delta_kernel = context_obj.kernels.at("clHiddenDeltaSigmoid"); // Assumed name
        cl::Kernel update_weights_kernel = context_obj.kernels.at("clUpdateWeightsAndGradients"); // Assumed name

        last_delta_kernel.setArg(0, d_grad_EV_scaled);
        last_delta_kernel.setArg(1, d_ver_activations[layers - 1]);
        last_delta_kernel.setArg(2, d_ver_deltas[layers - 1]);
        last_delta_kernel.setArg(3, embedding_dim);
        queue.enqueueNDRangeKernel(last_delta_kernel, cl::NullRange, global_embed, local_1d);
        queue.finish();

        for (int l = layers - 2; l >= 0; --l) {
            hidden_delta_kernel.setArg(0, d_ver_deltas[l + 1]);
            hidden_delta_kernel.setArg(1, d_ver_weights[l + 1]);
            hidden_delta_kernel.setArg(2, d_ver_activations[l]);
            hidden_delta_kernel.setArg(3, d_ver_deltas[l]);
            hidden_delta_kernel.setArg(4, embedding_dim);
            hidden_delta_kernel.setArg(5, embedding_dim);
            queue.enqueueNDRangeKernel(hidden_delta_kernel, cl::NullRange, global_embed, local_1d);
            queue.finish();
        }

        for (int l = 0; l < layers; ++l) {
            cl::Buffer& d_prev_activations = (l == 0) ? d_EV : d_ver_activations[l - 1]; // CAUTION: Size/pointer check needed
            int prev_layer_size = (l == 0) ? embedding_dim : embedding_dim; // Adjust if needed
            // Use global_embed_2d as sizes are embedding_dim x embedding_dim

            update_weights_kernel.setArg(0, d_ver_deltas[l]);
            update_weights_kernel.setArg(1, d_prev_activations);
            update_weights_kernel.setArg(2, d_ver_weights[l]);
            update_weights_kernel.setArg(3, d_ver_gweights[l]);
            update_weights_kernel.setArg(4, learning_rate);
            update_weights_kernel.setArg(5, embedding_dim);
            update_weights_kernel.setArg(6, prev_layer_size);
            queue.enqueueNDRangeKernel(update_weights_kernel, cl::NullRange, global_embed_2d, local_2d);
            queue.finish();
        }
        // --- End of MLP Backprop ---

        // --- Step 3: Compute grad_dv ---
        cl::Kernel grad_mlp_input_kernel = context_obj.kernels.at("clComputeGradMLPInput"); // Assumed name
        grad_mlp_input_kernel.setArg(0, d_ver_deltas[0]);
        grad_mlp_input_kernel.setArg(1, d_ver_weights[0]);
        grad_mlp_input_kernel.setArg(2, d_grad_dv);
        grad_mlp_input_kernel.setArg(3, embedding_dim);
        grad_mlp_input_kernel.setArg(4, embedding_dim); // Assuming ver input size is embed_dim
        queue.enqueueNDRangeKernel(grad_mlp_input_kernel, cl::NullRange, global_embed, local_1d);
        queue.finish();

        // --- Step 4: Compute grad_MV ---
        cl::Kernel lota_kernel = context_obj.kernels.at("clLOTA2d");
        lota_kernel.setArg(0, d_KdotQ);
        lota_kernel.setArg(1, d_head);
        lota_kernel.setArg(2, token_count);
        lota_kernel.setArg(3, token_count);
        size_t lota_global_raw = head_size;
        size_t lota_local_clamped = std::min(lota_global_raw, local_work_size_1d);
        if (lota_local_clamped == 0) lota_local_clamped = 1;
        size_t lota_global_padded = ((lota_global_raw + lota_local_clamped - 1) / lota_local_clamped) * lota_local_clamped;
        cl::NDRange global_lota(lota_global_padded);
        cl::NDRange local_lota(lota_local_clamped);
         if (head_size > 0) {
             queue.enqueueNDRangeKernel(lota_kernel, cl::NullRange, global_lota, local_lota);
         }

        cl::Kernel pre_mv_v_kernel = context_obj.kernels.at("clComputePreMV_V"); // Assumed name
        pre_mv_v_kernel.setArg(0, d_head);
        pre_mv_v_kernel.setArg(1, d_Q);
        pre_mv_v_kernel.setArg(2, d_pre_MV);
        pre_mv_v_kernel.setArg(3, token_count);
        pre_mv_v_kernel.setArg(4, mat_heights);
        queue.enqueueNDRangeKernel(pre_mv_v_kernel, cl::NullRange, global_mat_heights, local_1d);

        cl::Kernel grad_mv_v_kernel = context_obj.kernels.at("clComputeGradMV_V"); // Assumed name
        grad_mv_v_kernel.setArg(0, d_pre_MV);
        grad_mv_v_kernel.setArg(1, d_grad_dv);
        grad_mv_v_kernel.setArg(2, d_grad_MV);
        grad_mv_v_kernel.setArg(3, mat_heights);
        grad_mv_v_kernel.setArg(4, embedding_dim);
        queue.enqueueNDRangeKernel(grad_mv_v_kernel, cl::NullRange, global_matrix_2d, local_2d);

        // --- Step 5: Compute grad_head (dv part only) ---
        cl::Kernel grad_head_v_kernel = context_obj.kernels.at("clComputeGradHead_V"); // Assumed name
        grad_head_v_kernel.setArg(0, d_Q);
        grad_head_v_kernel.setArg(1, d_MV_a);
        grad_head_v_kernel.setArg(2, d_grad_dv);
        grad_head_v_kernel.setArg(3, d_grad_head);
        grad_head_v_kernel.setArg(4, token_count);
        grad_head_v_kernel.setArg(5, mat_heights);
        grad_head_v_kernel.setArg(6, embedding_dim);
        queue.enqueueNDRangeKernel(grad_head_v_kernel, cl::NullRange, global_head_2d, local_2d);

        // --- Step 6: Backprop through LOTA ---
        cl::Kernel lota_deriv_kernel = context_obj.kernels.at("clLOTA2dder");
        lota_deriv_kernel.setArg(0, d_KdotQ);
        lota_deriv_kernel.setArg(1, d_lota_deriv);
        lota_deriv_kernel.setArg(2, token_count);
        lota_deriv_kernel.setArg(3, token_count);
         if (head_size > 0) {
             queue.enqueueNDRangeKernel(lota_deriv_kernel, cl::NullRange, global_lota, local_lota);
         }

        cl::Kernel grad_kdotq_kernel = context_obj.kernels.at("clComputeGradKdotQ_LOTA"); // Assumed name
        grad_kdotq_kernel.setArg(0, d_grad_head);
        grad_kdotq_kernel.setArg(1, d_lota_deriv);
        grad_kdotq_kernel.setArg(2, d_grad_KdotQ);
        grad_kdotq_kernel.setArg(3, scaling_factor);
        grad_kdotq_kernel.setArg(4, head_size);
        queue.enqueueNDRangeKernel(grad_kdotq_kernel, cl::NullRange, global_head, local_1d);

        // --- Step 7: Compute grad_Q ---
        cl::Kernel grad_q_v_kernel = context_obj.kernels.at("clComputeGradQ_V"); // Assumed name
        grad_q_v_kernel.setArg(0, d_grad_KdotQ);
        grad_q_v_kernel.setArg(1, d_K);
        grad_q_v_kernel.setArg(2, d_grad_Q);
        grad_q_v_kernel.setArg(3, token_count);
        grad_q_v_kernel.setArg(4, mat_heights);
        queue.enqueueNDRangeKernel(grad_q_v_kernel, cl::NullRange, global_kq_grad_2d, local_2d);

        // --- Step 8: Compute grad_MQ and grad_MK_correction (Complex) ---
        cl::Kernel grad_mq_v_kernel = context_obj.kernels.at("clComputeGradMQ_V"); // Assumed name
        cl::Buffer d_null; // Null buffer
        grad_mq_v_kernel.setArg(0, d_grad_Q);
        grad_mq_v_kernel.setArg(1, d_null); // d_Q_embed (nullptr)
        grad_mq_v_kernel.setArg(2, d_grad_MQ);
        grad_mq_v_kernel.setArg(3, token_count);
        grad_mq_v_kernel.setArg(4, mat_heights);
        grad_mq_v_kernel.setArg(5, embedding_dim);
        queue.enqueueNDRangeKernel(grad_mq_v_kernel, cl::NullRange, global_matrix_2d, local_2d);

        cl::Kernel grad_mk_corr_kernel = context_obj.kernels.at("clComputeGradMKCorrection"); // Assumed name
        grad_mk_corr_kernel.setArg(0, d_grad_MQ);
        grad_mk_corr_kernel.setArg(1, d_Q);
        grad_mk_corr_kernel.setArg(2, d_K);
        grad_mk_corr_kernel.setArg(3, d_grad_MK_correction);
        grad_mk_corr_kernel.setArg(4, token_count);
        grad_mk_corr_kernel.setArg(5, mat_heights);
        grad_mk_corr_kernel.setArg(6, embedding_dim);
        queue.enqueueNDRangeKernel(grad_mk_corr_kernel, cl::NullRange, global_matrix_2d, local_2d);

        // --- Step 9: Update Weights MV, MQ, MK (correction) ---
        cl::Kernel update_weights_v_kernel = context_obj.kernels.at("clUpdateWeights_1stHead_V"); // Assumed name
        update_weights_v_kernel.setArg(0, d_MV_a);
        update_weights_v_kernel.setArg(1, d_MQ_a);
        update_weights_v_kernel.setArg(2, d_MK_a);
        update_weights_v_kernel.setArg(3, d_grad_MV);
        update_weights_v_kernel.setArg(4, d_grad_MQ);
        update_weights_v_kernel.setArg(5, d_grad_MK_correction);
        update_weights_v_kernel.setArg(6, learning_rate);
        update_weights_v_kernel.setArg(7, mat_heights);
        update_weights_v_kernel.setArg(8, embedding_dim);
        queue.enqueueNDRangeKernel(update_weights_v_kernel, cl::NullRange, global_matrix, local_1d);
        queue.finish(); // Sync before D->H copy

        // Step 10: No EH/EV update

        // --- Data Transfer D->H ---
        // Copy updated ver MLP weights and gradients back
        for (int l = 0; l < layers; ++l) {
            std::vector<float> updated_flat_weights(embedding_dim * embedding_dim);
            std::vector<float> calculated_flat_gradients(embedding_dim * embedding_dim);
            queue.enqueueReadBuffer(d_ver_weights[l], CL_TRUE, 0, weights_bytes, updated_flat_weights.data());
            queue.enqueueReadBuffer(d_ver_gweights[l], CL_TRUE, 0, weights_bytes, calculated_flat_gradients.data());
            unflatten(updated_flat_weights, this->ver.weights[l], embedding_dim, embedding_dim);
            unflatten(calculated_flat_gradients, this->ver.gweights[l], embedding_dim, embedding_dim);
        }

        // Copy updated Attention parameters back
        std::vector<float> updated_MV_a(mh_mv_mq_mk_size);
        std::vector<float> updated_MQ_a(mh_mv_mq_mk_size);
        std::vector<float> updated_MK_a(mh_mv_mq_mk_size);

        queue.enqueueReadBuffer(d_MV_a, CL_TRUE, 0, mh_mv_mq_mk_size * sizeof(float), updated_MV_a.data());
        queue.enqueueReadBuffer(d_MQ_a, CL_TRUE, 0, mh_mv_mq_mk_size * sizeof(float), updated_MQ_a.data());
        queue.enqueueReadBuffer(d_MK_a, CL_TRUE, 0, mh_mv_mq_mk_size * sizeof(float), updated_MK_a.data());

        unflatten(updated_MV_a, this->MV.a, mat_heights, embedding_dim);
        unflatten(updated_MQ_a, this->MQ.a, mat_heights, embedding_dim);
        unflatten(updated_MK_a, this->MK.a, mat_heights, embedding_dim);

    }
    catch (const cl::Error& err) {
        std::cerr << "OpenCL Error in clbackward1stHead(expectedV): " << err.what() << " (" << err.err() << ")" << std::endl;
        // Manual Cleanup (Attention)
        clReleaseMemObject(d_expected_v()); clReleaseMemObject(d_EV()); clReleaseMemObject(d_grad_EV_full()); clReleaseMemObject(d_grad_EV_summed()); clReleaseMemObject(d_grad_EV_scaled());
        clReleaseMemObject(d_grad_dv());
        clReleaseMemObject(d_KdotQ()); clReleaseMemObject(d_head()); clReleaseMemObject(d_K()); clReleaseMemObject(d_Q());
        clReleaseMemObject(d_pre_MV()); clReleaseMemObject(d_MV_a()); clReleaseMemObject(d_MQ_a()); clReleaseMemObject(d_MK_a());
        clReleaseMemObject(d_grad_MV()); clReleaseMemObject(d_grad_head());
        clReleaseMemObject(d_lota_deriv()); clReleaseMemObject(d_grad_KdotQ()); clReleaseMemObject(d_grad_Q());
        clReleaseMemObject(d_grad_MQ()); clReleaseMemObject(d_grad_MK_correction());
        // Manual Cleanup (ver MLP Internals)
        for (int l = 0; l < layers; ++l) {
            clReleaseMemObject(d_ver_activations[l]()); clReleaseMemObject(d_ver_weights[l]()); clReleaseMemObject(d_ver_gweights[l]()); clReleaseMemObject(d_ver_deltas[l]());
        }
        throw;
    }
    catch (const std::exception& e) {
        std::cerr << "Standard Exception in clbackward1stHead(expectedV): " << e.what() << std::endl;
        // Manual Cleanup (Attention)
        clReleaseMemObject(d_expected_v()); clReleaseMemObject(d_EV()); clReleaseMemObject(d_grad_EV_full()); clReleaseMemObject(d_grad_EV_summed()); clReleaseMemObject(d_grad_EV_scaled());
        clReleaseMemObject(d_grad_dv());
        clReleaseMemObject(d_KdotQ()); clReleaseMemObject(d_head()); clReleaseMemObject(d_K()); clReleaseMemObject(d_Q());
        clReleaseMemObject(d_pre_MV()); clReleaseMemObject(d_MV_a()); clReleaseMemObject(d_MQ_a()); clReleaseMemObject(d_MK_a());
        clReleaseMemObject(d_grad_MV()); clReleaseMemObject(d_grad_head());
        clReleaseMemObject(d_lota_deriv()); clReleaseMemObject(d_grad_KdotQ()); clReleaseMemObject(d_grad_Q());
        clReleaseMemObject(d_grad_MQ()); clReleaseMemObject(d_grad_MK_correction());
        // Manual Cleanup (ver MLP Internals)
        for (int l = 0; l < layers; ++l) {
            clReleaseMemObject(d_ver_activations[l]()); clReleaseMemObject(d_ver_weights[l]()); clReleaseMemObject(d_ver_gweights[l]()); clReleaseMemObject(d_ver_deltas[l]());
        }
        throw;
    } 
    // Buffers are automatically released when they go out of scope (RAII)
}


/**
* @brief OpenCL Backward Propagation (for first head) using gradients from both Horizontal and Vertical outputs.
*      Updates MH, MV, MQ, MK. No update to EH/EV.
*      Uses the clContext member for OpenCL resources.
* @param expectedH Horizontal embedding vector (next token prediction) (host)
* @param expectedV Vertical retention vector (host)
* @param in Input size (embedding dimension - used for MLP)
* @param layers Number of layers in the MLPs
*/
void attention::clbackward1stHead(std::vector<float>& expectedH, std::vector<std::vector<float>>& expectedV, int& in, int& layers)
{
    // --- Setup ---
    const int embedding_dim = EMBEDDING;
    const int mat_heights = MATHEIGHTS;
    const int context_win = CONTEXT_WIN;
    const float learning_rate = LEARNING;
    const float scaling_factor = SCALING;
    const int token_count = this->tokenCount;
    const int head_size = token_count * token_count;
    const int mh_mv_mq_mk_size = mat_heights * embedding_dim;
    const int k_q_size = token_count * mat_heights;
    const int ev_size = context_win * embedding_dim;
    const size_t embed_bytes = embedding_dim * sizeof(float);
    const size_t weights_bytes = embedding_dim * embedding_dim * sizeof(float);

    // Validation
    if (embedding_dim != in) throw std::runtime_error("Embedding dimension mismatch");
    if (expectedH.size() != embedding_dim) throw std::runtime_error("ExpectedH vector size mismatch");
    if (expectedV.size() != context_win || (!expectedV.empty() && expectedV[0].size() != embedding_dim)) {
        throw std::runtime_error("ExpectedV dimensions mismatch");
    }
     if (token_count <= 0) {
        std::cerr << "Warning: clbackward1stHead(expectedH, expectedV,...) called with token_count <= 0. Skipping." << std::endl;
        return;
    }
    // Add other necessary validation checks...

    // Temporary flat vectors for H->D copy
    std::vector<std::vector<float>> flat_hor_weights(layers);
    std::vector<std::vector<float>> flat_ver_weights(layers);

    OpenCLContext& context_obj = this->clcontext;
    cl::Context context = context_obj.context;
    cl::CommandQueue queue = context_obj.queue;
    // Use context_obj.kernels map below

    // --- Allocate Memory (Attention) ---
    cl::Buffer d_expected_h(context, CL_MEM_READ_ONLY, embed_bytes);
    cl::Buffer d_expected_v(context, CL_MEM_READ_ONLY, ev_size * sizeof(float));
    cl::Buffer d_EH(context, CL_MEM_READ_ONLY, embed_bytes); // Read-only
    cl::Buffer d_EV(context, CL_MEM_READ_ONLY, ev_size * sizeof(float)); // Read-only
    cl::Buffer d_grad_EH(context, CL_MEM_READ_WRITE, embed_bytes);
    cl::Buffer d_grad_EV_full(context, CL_MEM_READ_WRITE, ev_size * sizeof(float));
    cl::Buffer d_grad_EV_summed(context, CL_MEM_READ_WRITE, embed_bytes);
    cl::Buffer d_grad_EV_scaled(context, CL_MEM_READ_WRITE, embed_bytes);
    cl::Buffer d_grad_dh(context, CL_MEM_READ_WRITE, embed_bytes);
    cl::Buffer d_grad_dv(context, CL_MEM_READ_WRITE, embed_bytes);
    cl::Buffer d_KdotQ(context, CL_MEM_READ_ONLY, head_size * sizeof(float));
    cl::Buffer d_head(context, CL_MEM_READ_WRITE, head_size * sizeof(float));
    cl::Buffer d_K(context, CL_MEM_READ_ONLY, k_q_size * sizeof(float));
    cl::Buffer d_Q(context, CL_MEM_READ_ONLY, k_q_size * sizeof(float));
    cl::Buffer d_pre_MH(context, CL_MEM_READ_WRITE, mat_heights * sizeof(float));
    cl::Buffer d_pre_MV(context, CL_MEM_READ_WRITE, mat_heights * sizeof(float));
    cl::Buffer d_MH_a(context, CL_MEM_READ_WRITE, mh_mv_mq_mk_size * sizeof(float));
    cl::Buffer d_MV_a(context, CL_MEM_READ_WRITE, mh_mv_mq_mk_size * sizeof(float));
    cl::Buffer d_MQ_a(context, CL_MEM_READ_WRITE, mh_mv_mq_mk_size * sizeof(float));
    cl::Buffer d_MK_a(context, CL_MEM_READ_WRITE, mh_mv_mq_mk_size * sizeof(float));
    cl::Buffer d_grad_MH(context, CL_MEM_READ_WRITE, mh_mv_mq_mk_size * sizeof(float));
    cl::Buffer d_grad_MV(context, CL_MEM_READ_WRITE, mh_mv_mq_mk_size * sizeof(float));
    cl::Buffer d_grad_head(context, CL_MEM_READ_WRITE, head_size * sizeof(float));
    cl::Buffer d_lota_deriv(context, CL_MEM_READ_WRITE, head_size * sizeof(float));
    cl::Buffer d_grad_KdotQ(context, CL_MEM_READ_WRITE, head_size * sizeof(float));
    cl::Buffer d_grad_K(context, CL_MEM_READ_WRITE, k_q_size * sizeof(float));
    cl::Buffer d_grad_Q(context, CL_MEM_READ_WRITE, k_q_size * sizeof(float));
    cl::Buffer d_grad_MQ(context, CL_MEM_READ_WRITE, mh_mv_mq_mk_size * sizeof(float));
    cl::Buffer d_grad_MK(context, CL_MEM_READ_WRITE, mh_mv_mq_mk_size * sizeof(float)); // Simplified gradients

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
    d_hor_weights.reserve(layers);
    d_hor_gweights.reserve(layers);
    d_hor_deltas.reserve(layers);
    d_ver_activations.reserve(layers);
    d_ver_weights.reserve(layers);
    d_ver_gweights.reserve(layers);
    d_ver_deltas.reserve(layers);

    for (int l = 0; l < layers; ++l) {
        d_hor_activations.emplace_back(context, CL_MEM_READ_ONLY, embed_bytes);
        d_hor_weights.emplace_back(context, CL_MEM_READ_WRITE, weights_bytes);
        d_hor_gweights.emplace_back(context, CL_MEM_READ_WRITE, weights_bytes);
        d_hor_deltas.emplace_back(context, CL_MEM_READ_WRITE, embed_bytes);
        d_ver_activations.emplace_back(context, CL_MEM_READ_ONLY, embed_bytes);
        d_ver_weights.emplace_back(context, CL_MEM_READ_WRITE, weights_bytes);
        d_ver_gweights.emplace_back(context, CL_MEM_READ_WRITE, weights_bytes);
        d_ver_deltas.emplace_back(context, CL_MEM_READ_WRITE, embed_bytes);
    }

    // --- Data Transfer H->D (Attention) ---
    std::vector<float> flat_expectedV = flatten(expectedV);
    std::vector<float> flat_EV = flatten(this->EV);
    std::vector<float> flat_K = flatten(this->K);
    std::vector<float> flat_Q = flatten(this->Q);
    std::vector<float> flat_KdotQ = flatten(this->KdotQ);
    std::vector<float> flat_MH_a = flatten(this->MH.a);
    std::vector<float> flat_MV_a = flatten(this->MV.a);
    std::vector<float> flat_MQ_a = flatten(this->MQ.a);
    std::vector<float> flat_MK_a = flatten(this->MK.a);

    queue.enqueueWriteBuffer(d_expected_h, CL_TRUE, 0, embed_bytes, expectedH.data());
    queue.enqueueWriteBuffer(d_expected_v, CL_TRUE, 0, ev_size * sizeof(float), flat_expectedV.data());
    queue.enqueueWriteBuffer(d_EH, CL_TRUE, 0, embed_bytes, this->EH.data());
    queue.enqueueWriteBuffer(d_EV, CL_TRUE, 0, ev_size * sizeof(float), flat_EV.data());
    queue.enqueueWriteBuffer(d_KdotQ, CL_TRUE, 0, head_size * sizeof(float), flat_KdotQ.data());
    queue.enqueueWriteBuffer(d_K, CL_TRUE, 0, k_q_size * sizeof(float), flat_K.data());
    queue.enqueueWriteBuffer(d_Q, CL_TRUE, 0, k_q_size * sizeof(float), flat_Q.data());
    queue.enqueueWriteBuffer(d_MH_a, CL_TRUE, 0, mh_mv_mq_mk_size * sizeof(float), flat_MH_a.data());
    queue.enqueueWriteBuffer(d_MV_a, CL_TRUE, 0, mh_mv_mq_mk_size * sizeof(float), flat_MV_a.data());
    queue.enqueueWriteBuffer(d_MQ_a, CL_TRUE, 0, mh_mv_mq_mk_size * sizeof(float), flat_MQ_a.data());
    queue.enqueueWriteBuffer(d_MK_a, CL_TRUE, 0, mh_mv_mq_mk_size * sizeof(float), flat_MK_a.data());

    // --- Data Transfer H->D (MLP Internals) ---
    for (int l = 0; l < layers; ++l) {
        flat_hor_weights[l] = flatten(this->hor.weights[l]);
        flat_ver_weights[l] = flatten(this->ver.weights[l]);
        queue.enqueueWriteBuffer(d_hor_activations[l], CL_TRUE, 0, embed_bytes, this->hor.activations[l].data());
        queue.enqueueWriteBuffer(d_hor_weights[l], CL_TRUE, 0, weights_bytes, flat_hor_weights[l].data());
        queue.enqueueWriteBuffer(d_ver_activations[l], CL_TRUE, 0, embed_bytes, this->ver.activations[l].data());
        queue.enqueueWriteBuffer(d_ver_weights[l], CL_TRUE, 0, weights_bytes, flat_ver_weights[l].data());
    }

    try {
        // --- Kernel Launch Config ---
        const size_t local_work_size_1d = 256;
        auto calculate_global_1d = [&](size_t total_size) {
            return ((total_size + local_work_size_1d - 1) / local_work_size_1d) * local_work_size_1d;
        };
        cl::NDRange global_embed(calculate_global_1d(embedding_dim));
        cl::NDRange global_head(calculate_global_1d(head_size));
        cl::NDRange global_mat_heights(calculate_global_1d(mat_heights));
        cl::NDRange global_matrix(calculate_global_1d(mh_mv_mq_mk_size));
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
        cl::NDRange global_kq_grad_2d = calculate_global_2d(mat_heights, token_count); // dim0=height, dim1=token
        cl::NDRange local_2d(local_work_size_2d[0], local_work_size_2d[1]);

        // --- Backpropagation Steps ---

        // Step 1: Compute grad_EH and grad_EV (full, summed, scaled)
        cl::Kernel grad_eh_ev_kernel = context_obj.kernels.at("clComputeGradientsEH_EV"); // Assumed name
        cl::Buffer d_dummy_ev_grad(context, CL_MEM_WRITE_ONLY, embed_bytes); // Dummy
        grad_eh_ev_kernel.setArg(0, d_EH);
        grad_eh_ev_kernel.setArg(1, d_expected_h);
        grad_eh_ev_kernel.setArg(2, d_grad_EH);
        grad_eh_ev_kernel.setArg(3, d_dummy_ev_grad); // Write dummy value
        grad_eh_ev_kernel.setArg(4, embedding_dim);
        queue.enqueueNDRangeKernel(grad_eh_ev_kernel, cl::NullRange, global_embed, local_1d);
        // d_dummy_ev_grad released by RAII

        cl::Kernel grad_ev_v_kernel = context_obj.kernels.at("clComputeGradientsEV_V"); // Assumed name
        grad_ev_v_kernel.setArg(0, d_EV);
        grad_ev_v_kernel.setArg(1, d_expected_v);
        grad_ev_v_kernel.setArg(2, d_grad_EV_full);
        grad_ev_v_kernel.setArg(3, d_grad_EV_summed);
        grad_ev_v_kernel.setArg(4, d_grad_EV_scaled);
        grad_ev_v_kernel.setArg(5, learning_rate);
        grad_ev_v_kernel.setArg(6, context_win);
        grad_ev_v_kernel.setArg(7, embedding_dim);
        queue.enqueueNDRangeKernel(grad_ev_v_kernel, cl::NullRange, global_embed, local_1d); // Use global_embed as it sums to embed_dim
        queue.finish(); // Sync before MLP backprop

        // --- Step 2: Backprop through MLPs ---
        cl::Kernel last_delta_kernel = context_obj.kernels.at("clLastLayerDeltaSigmoid"); // Assumed name
        cl::Kernel hidden_delta_kernel = context_obj.kernels.at("clHiddenDeltaSigmoid"); // Assumed name
        cl::Kernel update_weights_kernel = context_obj.kernels.at("clUpdateWeightsAndGradients"); // Assumed name

        // --- 2a: Backprop through hor MLP ---
        last_delta_kernel.setArg(0, d_grad_EH);
        last_delta_kernel.setArg(1, d_hor_activations[layers - 1]);
        last_delta_kernel.setArg(2, d_hor_deltas[layers - 1]);
        last_delta_kernel.setArg(3, embedding_dim);
        queue.enqueueNDRangeKernel(last_delta_kernel, cl::NullRange, global_embed, local_1d);
        queue.finish();

        for (int l = layers - 2; l >= 0; --l) {
            hidden_delta_kernel.setArg(0, d_hor_deltas[l + 1]);
            hidden_delta_kernel.setArg(1, d_hor_weights[l + 1]);
            hidden_delta_kernel.setArg(2, d_hor_activations[l]);
            hidden_delta_kernel.setArg(3, d_hor_deltas[l]);
            hidden_delta_kernel.setArg(4, embedding_dim);
            hidden_delta_kernel.setArg(5, embedding_dim);
            queue.enqueueNDRangeKernel(hidden_delta_kernel, cl::NullRange, global_embed, local_1d);
            queue.finish();
        }

        for (int l = 0; l < layers; ++l) {
            cl::Buffer& d_prev_activations = (l == 0) ? d_EH : d_hor_activations[l - 1];
            update_weights_kernel.setArg(0, d_hor_deltas[l]);
            update_weights_kernel.setArg(1, d_prev_activations);
            update_weights_kernel.setArg(2, d_hor_weights[l]);
            update_weights_kernel.setArg(3, d_hor_gweights[l]);
            update_weights_kernel.setArg(4, learning_rate);
            update_weights_kernel.setArg(5, embedding_dim);
            update_weights_kernel.setArg(6, embedding_dim);
            queue.enqueueNDRangeKernel(update_weights_kernel, cl::NullRange, global_embed_2d, local_2d);
            queue.finish();
        }

        // --- 2b: Backprop through ver MLP ---
        last_delta_kernel.setArg(0, d_grad_EV_scaled);
        last_delta_kernel.setArg(1, d_ver_activations[layers - 1]);
        last_delta_kernel.setArg(2, d_ver_deltas[layers - 1]);
        // Arg 3 (embedding_dim) is already set
        queue.enqueueNDRangeKernel(last_delta_kernel, cl::NullRange, global_embed, local_1d);
        queue.finish();

        for (int l = layers - 2; l >= 0; --l) {
            hidden_delta_kernel.setArg(0, d_ver_deltas[l + 1]);
            hidden_delta_kernel.setArg(1, d_ver_weights[l + 1]);
            hidden_delta_kernel.setArg(2, d_ver_activations[l]);
            hidden_delta_kernel.setArg(3, d_ver_deltas[l]);
            // Args 4, 5 (embedding_dim) are already set
            queue.enqueueNDRangeKernel(hidden_delta_kernel, cl::NullRange, global_embed, local_1d);
            queue.finish();
        }

        for (int l = 0; l < layers; ++l) {
            cl::Buffer& d_prev_activations = (l == 0) ? d_EV : d_ver_activations[l - 1]; // CAUTION: Size/pointer check needed
            int prev_layer_size = (l == 0) ? embedding_dim : embedding_dim; // Adjust if needed
            // Use global_embed_2d as sizes are embedding_dim x embedding_dim

            update_weights_kernel.setArg(0, d_ver_deltas[l]);
            update_weights_kernel.setArg(1, d_prev_activations);
            update_weights_kernel.setArg(2, d_ver_weights[l]);
            update_weights_kernel.setArg(3, d_ver_gweights[l]);
            // Arg 4 (learning_rate) is already set
            update_weights_kernel.setArg(5, embedding_dim);
            update_weights_kernel.setArg(6, prev_layer_size);
            queue.enqueueNDRangeKernel(update_weights_kernel, cl::NullRange, global_embed_2d, local_2d);
            queue.finish();
        }
        // --- End of MLP Backprop ---

        // --- Step 3: Compute grad_dh and grad_dv ---
        cl::Kernel grad_mlp_input_kernel = context_obj.kernels.at("clComputeGradMLPInput"); // Assumed name
        grad_mlp_input_kernel.setArg(0, d_hor_deltas[0]);
        grad_mlp_input_kernel.setArg(1, d_hor_weights[0]);
        grad_mlp_input_kernel.setArg(2, d_grad_dh);
        grad_mlp_input_kernel.setArg(3, embedding_dim);
        grad_mlp_input_kernel.setArg(4, embedding_dim);
        queue.enqueueNDRangeKernel(grad_mlp_input_kernel, cl::NullRange, global_embed, local_1d);

        grad_mlp_input_kernel.setArg(0, d_ver_deltas[0]);
        grad_mlp_input_kernel.setArg(1, d_ver_weights[0]);
        grad_mlp_input_kernel.setArg(2, d_grad_dv);
        // Args 3, 4 (embedding_dim) are already set
        queue.enqueueNDRangeKernel(grad_mlp_input_kernel, cl::NullRange, global_embed, local_1d);
        queue.finish();

        // --- Step 4: Compute grad_MH and grad_MV ---
        cl::Kernel lota_kernel = context_obj.kernels.at("clLOTA2d");
        lota_kernel.setArg(0, d_KdotQ);
        lota_kernel.setArg(1, d_head);
        lota_kernel.setArg(2, token_count);
        lota_kernel.setArg(3, token_count);
        size_t lota_global_raw = head_size;
        size_t lota_local_clamped = std::min(lota_global_raw, local_work_size_1d);
        if (lota_local_clamped == 0) lota_local_clamped = 1;
        size_t lota_global_padded = ((lota_global_raw + lota_local_clamped - 1) / lota_local_clamped) * lota_local_clamped;
        cl::NDRange global_lota(lota_global_padded);
        cl::NDRange local_lota(lota_local_clamped);
         if (head_size > 0) {
             queue.enqueueNDRangeKernel(lota_kernel, cl::NullRange, global_lota, local_lota);
         }

        cl::Kernel pre_mh_mv_kernel = context_obj.kernels.at("clComputePreMH_MV"); // Assumed name
        pre_mh_mv_kernel.setArg(0, d_head);
        pre_mh_mv_kernel.setArg(1, d_K);
        pre_mh_mv_kernel.setArg(2, d_Q);
        pre_mh_mv_kernel.setArg(3, d_pre_MH);
        pre_mh_mv_kernel.setArg(4, d_pre_MV);
        pre_mh_mv_kernel.setArg(5, token_count);
        pre_mh_mv_kernel.setArg(6, mat_heights);
        queue.enqueueNDRangeKernel(pre_mh_mv_kernel, cl::NullRange, global_mat_heights, local_1d);

        cl::Kernel grad_mh_mv_kernel = context_obj.kernels.at("clComputeGradMH_MV"); // Assumed name
        grad_mh_mv_kernel.setArg(0, d_pre_MH);
        grad_mh_mv_kernel.setArg(1, d_pre_MV);
        grad_mh_mv_kernel.setArg(2, d_grad_dh);
        grad_mh_mv_kernel.setArg(3, d_grad_dv);
        grad_mh_mv_kernel.setArg(4, d_grad_MH);
        grad_mh_mv_kernel.setArg(5, d_grad_MV);
        grad_mh_mv_kernel.setArg(6, mat_heights);
        grad_mh_mv_kernel.setArg(7, embedding_dim);
        queue.enqueueNDRangeKernel(grad_mh_mv_kernel, cl::NullRange, global_matrix_2d, local_2d);

        // --- Step 5: Compute grad_head ---
        cl::Kernel grad_head_kernel = context_obj.kernels.at("clComputeGradHead"); // Assumed name
        grad_head_kernel.setArg(0, d_K);
        grad_head_kernel.setArg(1, d_Q);
        grad_head_kernel.setArg(2, d_MH_a);
        grad_head_kernel.setArg(3, d_MV_a);
        grad_head_kernel.setArg(4, d_grad_dh);
        grad_head_kernel.setArg(5, d_grad_dv);
        grad_head_kernel.setArg(6, d_grad_head);
        grad_head_kernel.setArg(7, token_count);
        grad_head_kernel.setArg(8, mat_heights);
        grad_head_kernel.setArg(9, embedding_dim);
        queue.enqueueNDRangeKernel(grad_head_kernel, cl::NullRange, global_head_2d, local_2d);

        // --- Step 6: Backprop through LOTA ---
        cl::Kernel lota_deriv_kernel = context_obj.kernels.at("clLOTA2dder");
        lota_deriv_kernel.setArg(0, d_KdotQ);
        lota_deriv_kernel.setArg(1, d_lota_deriv);
        lota_deriv_kernel.setArg(2, token_count);
        lota_deriv_kernel.setArg(3, token_count);
         if (head_size > 0) {
             queue.enqueueNDRangeKernel(lota_deriv_kernel, cl::NullRange, global_lota, local_lota);
         }

        cl::Kernel grad_kdotq_kernel = context_obj.kernels.at("clComputeGradKdotQ_LOTA"); // Assumed name
        grad_kdotq_kernel.setArg(0, d_grad_head);
        grad_kdotq_kernel.setArg(1, d_lota_deriv);
        grad_kdotq_kernel.setArg(2, d_grad_KdotQ);
        grad_kdotq_kernel.setArg(3, scaling_factor);
        grad_kdotq_kernel.setArg(4, head_size);
        queue.enqueueNDRangeKernel(grad_kdotq_kernel, cl::NullRange, global_head, local_1d);

        // --- Step 7: Compute grad_K and grad_Q ---
        cl::Kernel grad_k_q_kernel = context_obj.kernels.at("clComputeGradK_Q"); // Assumed name
        grad_k_q_kernel.setArg(0, d_grad_KdotQ);
        grad_k_q_kernel.setArg(1, d_K);
        grad_k_q_kernel.setArg(2, d_Q);
        grad_k_q_kernel.setArg(3, d_grad_K);
        grad_k_q_kernel.setArg(4, d_grad_Q);
        grad_k_q_kernel.setArg(5, token_count);
        grad_k_q_kernel.setArg(6, mat_heights);
        queue.enqueueNDRangeKernel(grad_k_q_kernel, cl::NullRange, global_kq_grad_2d, local_2d);

        // --- Step 8: Compute grad_MK and grad_MQ (Simplified) ---
        cl::Kernel grad_mk_mq_kernel = context_obj.kernels.at("clComputeGradMK_MQ_Simplified"); // Assumed name
        cl::Buffer d_null;
        grad_mk_mq_kernel.setArg(0, d_grad_K);
        grad_mk_mq_kernel.setArg(1, d_grad_Q);
        grad_mk_mq_kernel.setArg(2, d_null); // d_K_embed
        grad_mk_mq_kernel.setArg(3, d_null); // d_Q_embed
        grad_mk_mq_kernel.setArg(4, d_grad_MK);
        grad_mk_mq_kernel.setArg(5, d_grad_MQ);
        grad_mk_mq_kernel.setArg(6, token_count);
        grad_mk_mq_kernel.setArg(7, mat_heights);
        grad_mk_mq_kernel.setArg(8, embedding_dim);
        queue.enqueueNDRangeKernel(grad_mk_mq_kernel, cl::NullRange, global_matrix_2d, local_2d);

        // --- Step 9: Update Weights MH, MV, MQ, MK ---
        cl::Kernel update_weights_hv_kernel = context_obj.kernels.at("clUpdateWeights_1stHead_HV"); // Assumed name
        update_weights_hv_kernel.setArg(0, d_MH_a);
        update_weights_hv_kernel.setArg(1, d_MV_a);
        update_weights_hv_kernel.setArg(2, d_MQ_a);
        update_weights_hv_kernel.setArg(3, d_MK_a);
        update_weights_hv_kernel.setArg(4, d_grad_MH);
        update_weights_hv_kernel.setArg(5, d_grad_MV);
        update_weights_hv_kernel.setArg(6, d_grad_MQ);
        update_weights_hv_kernel.setArg(7, d_grad_MK);
        update_weights_hv_kernel.setArg(8, learning_rate);
        update_weights_hv_kernel.setArg(9, mat_heights);
        update_weights_hv_kernel.setArg(10, embedding_dim);
        queue.enqueueNDRangeKernel(update_weights_hv_kernel, cl::NullRange, global_matrix, local_1d);
        queue.finish(); // Sync before D->H copy

        // Step 10: No EH/EV update

        // --- Data Transfer D->H ---
        // Copy updated MLP weights and gradients back
        for (int l = 0; l < layers; ++l) {
            std::vector<float> updated_flat_weights(embedding_dim * embedding_dim);
            std::vector<float> calculated_flat_gradients(embedding_dim * embedding_dim);
            // Hor MLP
            queue.enqueueReadBuffer(d_hor_weights[l], CL_TRUE, 0, weights_bytes, updated_flat_weights.data());
            queue.enqueueReadBuffer(d_hor_gweights[l], CL_TRUE, 0, weights_bytes, calculated_flat_gradients.data());
            unflatten(updated_flat_weights, this->hor.weights[l], embedding_dim, embedding_dim);
            unflatten(calculated_flat_gradients, this->hor.gweights[l], embedding_dim, embedding_dim);
            // Ver MLP
            queue.enqueueReadBuffer(d_ver_weights[l], CL_TRUE, 0, weights_bytes, updated_flat_weights.data());
            queue.enqueueReadBuffer(d_ver_gweights[l], CL_TRUE, 0, weights_bytes, calculated_flat_gradients.data());
            unflatten(updated_flat_weights, this->ver.weights[l], embedding_dim, embedding_dim);
            unflatten(calculated_flat_gradients, this->ver.gweights[l], embedding_dim, embedding_dim);
        }

        // Copy updated Attention parameters back
        std::vector<float> updated_MH_a(mh_mv_mq_mk_size);
        std::vector<float> updated_MV_a(mh_mv_mq_mk_size);
        std::vector<float> updated_MQ_a(mh_mv_mq_mk_size);
        std::vector<float> updated_MK_a(mh_mv_mq_mk_size);

        queue.enqueueReadBuffer(d_MH_a, CL_TRUE, 0, mh_mv_mq_mk_size * sizeof(float), updated_MH_a.data());
        queue.enqueueReadBuffer(d_MV_a, CL_TRUE, 0, mh_mv_mq_mk_size * sizeof(float), updated_MV_a.data());
        queue.enqueueReadBuffer(d_MQ_a, CL_TRUE, 0, mh_mv_mq_mk_size * sizeof(float), updated_MQ_a.data());
        queue.enqueueReadBuffer(d_MK_a, CL_TRUE, 0, mh_mv_mq_mk_size * sizeof(float), updated_MK_a.data());

        unflatten(updated_MH_a, this->MH.a, mat_heights, embedding_dim);
        unflatten(updated_MV_a, this->MV.a, mat_heights, embedding_dim);
        unflatten(updated_MQ_a, this->MQ.a, mat_heights, embedding_dim);
        unflatten(updated_MK_a, this->MK.a, mat_heights, embedding_dim);

    }
    catch (const cl::Error& err) {
        std::cerr << "OpenCL Error in clbackward1stHead(expectedH, expectedV): " << err.what() << " (" << err.err() << ")" << std::endl;
        // Manual Cleanup (Attention)
        clReleaseMemObject(d_expected_h()); clReleaseMemObject(d_expected_v()); clReleaseMemObject(d_EH()); clReleaseMemObject(d_EV());
        clReleaseMemObject(d_grad_EH()); clReleaseMemObject(d_grad_EV_full()); clReleaseMemObject(d_grad_EV_summed()); clReleaseMemObject(d_grad_EV_scaled());
        clReleaseMemObject(d_grad_dh()); clReleaseMemObject(d_grad_dv());
        clReleaseMemObject(d_KdotQ()); clReleaseMemObject(d_head()); clReleaseMemObject(d_K()); clReleaseMemObject(d_Q());
        clReleaseMemObject(d_pre_MH()); clReleaseMemObject(d_pre_MV());
        clReleaseMemObject(d_MH_a()); clReleaseMemObject(d_MV_a()); clReleaseMemObject(d_MQ_a()); clReleaseMemObject(d_MK_a());
        clReleaseMemObject(d_grad_MH()); clReleaseMemObject(d_grad_MV());
        clReleaseMemObject(d_grad_head()); clReleaseMemObject(d_lota_deriv());
        clReleaseMemObject(d_grad_KdotQ()); clReleaseMemObject(d_grad_K()); clReleaseMemObject(d_grad_Q());
        clReleaseMemObject(d_grad_MQ()); clReleaseMemObject(d_grad_MK());
        // Manual Cleanup (MLP Internals)
        for (int l = 0; l < layers; ++l) {
            clReleaseMemObject(d_hor_activations[l]()); clReleaseMemObject(d_hor_weights[l]()); clReleaseMemObject(d_hor_gweights[l]()); clReleaseMemObject(d_hor_deltas[l]());
            clReleaseMemObject(d_ver_activations[l]()); clReleaseMemObject(d_ver_weights[l]()); clReleaseMemObject(d_ver_gweights[l]()); clReleaseMemObject(d_ver_deltas[l]());
        }
        throw;
    }
    catch (const std::exception& e) {
        std::cerr << "Standard Exception in clbackward1stHead(expectedH, expectedV): " << e.what() << std::endl;
        // Manual Cleanup (Attention)
        clReleaseMemObject(d_expected_h()); clReleaseMemObject(d_expected_v()); clReleaseMemObject(d_EH()); clReleaseMemObject(d_EV());
        clReleaseMemObject(d_grad_EH()); clReleaseMemObject(d_grad_EV_full()); clReleaseMemObject(d_grad_EV_summed()); clReleaseMemObject(d_grad_EV_scaled());
        clReleaseMemObject(d_grad_dh()); clReleaseMemObject(d_grad_dv());
        clReleaseMemObject(d_KdotQ()); clReleaseMemObject(d_head()); clReleaseMemObject(d_K()); clReleaseMemObject(d_Q());
        clReleaseMemObject(d_pre_MH()); clReleaseMemObject(d_pre_MV());
        clReleaseMemObject(d_MH_a()); clReleaseMemObject(d_MV_a()); clReleaseMemObject(d_MQ_a()); clReleaseMemObject(d_MK_a());
        clReleaseMemObject(d_grad_MH()); clReleaseMemObject(d_grad_MV());
        clReleaseMemObject(d_grad_head()); clReleaseMemObject(d_lota_deriv());
        clReleaseMemObject(d_grad_KdotQ()); clReleaseMemObject(d_grad_K()); clReleaseMemObject(d_grad_Q());
        clReleaseMemObject(d_grad_MQ()); clReleaseMemObject(d_grad_MK());
        // Manual Cleanup (MLP Internals)
        for (int l = 0; l < layers; ++l) {
            clReleaseMemObject(d_hor_activations[l]()); clReleaseMemObject(d_hor_weights[l]()); clReleaseMemObject(d_hor_gweights[l]()); clReleaseMemObject(d_hor_deltas[l]());
            clReleaseMemObject(d_ver_activations[l]()); clReleaseMemObject(d_ver_weights[l]()); clReleaseMemObject(d_ver_gweights[l]()); clReleaseMemObject(d_ver_deltas[l]());
        }
        throw;
    }
    // Buffers are automatically released when they go out of scope (RAII)
}

#endif // USE_OPENCL
