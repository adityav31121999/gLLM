#ifdef USE_OPENCL
#if defined(_WIN64)
    #include <CL/cl.hpp>
#elif defined(__linux__)
    #include <CL/opencl.hpp>
#endif
#include <iostream>
#include <vector>
#include <cmath>
#include "maths.hpp"
#include "include/mlp.hpp"
#include "include/attention.hpp"
#include "include/block.hpp"

// Helper struct to hold per-head sub-buffers for Horizontal error backprop,
// analogous to HeadDevicePointers in block.hpp but for cl::Buffer.
struct HeadDeviceSubBuffersirH {
    cl::Buffer d_expected_h;
    cl::Buffer d_EH, d_EV;
    cl::Buffer d_grad_EH, d_grad_EV_scaled;
    cl::Buffer d_grad_dh, d_grad_dv;
    cl::Buffer d_KdotQ, d_head;
    cl::Buffer d_K, d_Q;
    cl::Buffer d_pre_MH, d_pre_MV;
    cl::Buffer d_MH_a, d_MV_a, d_MQ_a, d_MK_a;
    cl::Buffer d_grad_MH, d_grad_MV;
    cl::Buffer d_grad_head;
    cl::Buffer d_lota_deriv;
    cl::Buffer d_grad_KdotQ;
    cl::Buffer d_grad_K, d_grad_Q;
    cl::Buffer d_grad_MQ, d_grad_MK;
    cl::Buffer d_grad_token;

    // for gradient accumulation across heads
    // get gradient for token from gradients of MQ, MK, and gradients from h and v
    // dL/dT_Q = d_grad_Q * MQ^T
    // dL/dT_K = d_grad_K * MK^T
    cl::Buffer d_MQt, d_MKt;
    // dL/dT_h = d_grad_dh * (d_head * MK^T)^T * MH
    // dL/dT_v = d_grad_dv * (d_head * MQ^T)^T * MV
    cl::Buffer d_head_MKt, d_head_MQt;
    cl::Buffer d_head_MKt_t, d_head_MQt_t;
    cl::Buffer d_hMKt_t, d_hMQt_t;
    // dL/dT = dL/dT_Q + dL/dT_K + dL/dT_h + dL/dT_v
    cl::Buffer d_grad_T_Q, d_grad_T_K, d_grad_T_h, d_grad_T_v;
    cl::Buffer d_grad_T; // final gradient for token from this head

    std::vector<cl::Buffer> d_hor_activations;
    std::vector<cl::Buffer> d_hor_weights, d_hor_gweights, d_hor_deltas;
    std::vector<cl::Buffer> d_ver_activations;
    std::vector<cl::Buffer> d_ver_weights, d_ver_gweights, d_ver_deltas;
};


/**
 * @brief backward propagation for last column of first block (backprop starts from this) and
 *      get gradient to update embeddings
 * @param expectedH expected token embedding for each head of the column
 * @param in_dim embedding dimension and input-output vector dimension of mlp
 * @param layers number of layers of activations of mlp
 * @param layno_col_idx column number
 * @param learning learning rate
 * @param lambda_l1 L1 regularization parameter
 * @param lambda_l2 L2 regularization parameter
 * @param clip_norm Maximum L2 norm for gradient clipping (new parameter)
 */
void block::clrpartialbackward1stBlock(std::vector<std::vector<float>>& expectedH, int& in_dim, int& layers_mlp, int& layno_col_idx,
    float& learning, float& lambda_l1, float& lambda_l2)
{
    cl_int cl_err;
    const int num_heads_to_process = x; // 'x' is the number of rows/heads in this column

    if (layno_col_idx < 0 || layno_col_idx >= y) {
        throw std::out_of_range("clrpartialbackward1stBlock(H_2D): Column index 'layno_col_idx' (" + std::to_string(layno_col_idx) +
            ") is out of range [0, " + std::to_string(y - 1) + "].");
    }
    if (expectedH.size() != static_cast<size_t>(num_heads_to_process)) {
        throw std::runtime_error("ExpectedH outer vector size mismatch in clrpartialbackward1stBlock(H_2D). Expected " +
            std::to_string(num_heads_to_process) + ", got " + std::to_string(expectedH.size()));
    }
    for(int head_check_idx = 0; head_check_idx < num_heads_to_process; ++head_check_idx) {
        if(expectedH[head_check_idx].size() != EMBEDDING) {
            throw std::runtime_error("clrpartialbackward1stBlock(H_2D): ExpectedH inner vector size mismatch for head " + std::to_string(head_check_idx) +
                ". Expected " + std::to_string(EMBEDDING) + ", got " + std::to_string(expectedH[head_check_idx].size()));
        }
    }
    if (EMBEDDING != in_dim) {
        throw std::runtime_error("Embedding dimension mismatch: EMBEDDING (" + std::to_string(EMBEDDING) 
                                + ") vs in_dim (" + std::to_string(in_dim) +")");
    }

    const float learning_rate = learning;
    const float scaling_factor = SCALING;
    const int embedding_dim = EMBEDDING;
    const int mat_heights = CONTEXT_WIN;
    const int context_win = CONTEXT_WIN;
    const int token_count = tokenCount;
    const int num_total_layers_mlp = layers_mlp;
    const int num_neuron_layers_mlp = num_total_layers_mlp;
    const int num_weight_matrices_mlp = num_total_layers_mlp - 1;
    const size_t embed_bytes = embedding_dim * sizeof(float);
    const size_t token_embed_bytes = context_win * embedding_dim * sizeof(float);
    const size_t mlp_weights_elements = static_cast<size_t>(embedding_dim) * embedding_dim;
    const size_t mlp_weights_bytes = mlp_weights_elements * sizeof(float);
    const size_t proj_mat_elements = static_cast<size_t>(mat_heights) * embedding_dim;
    const size_t proj_mat_bytes = proj_mat_elements * sizeof(float);
    const size_t ev_single_head_elements = static_cast<size_t>(context_win) * embedding_dim;
    const size_t ev_single_head_bytes = ev_single_head_elements * sizeof(float);
    const size_t max_head_elements_per_head = static_cast<size_t>(context_win) * context_win;
    const size_t max_head_bytes_per_head = max_head_elements_per_head * sizeof(float);
    const size_t max_k_q_elements_per_head = static_cast<size_t>(context_win) * mat_heights;
    const size_t max_k_q_bytes_per_head = max_k_q_elements_per_head * sizeof(float);
    const size_t pre_mh_mv_elements_per_head = mat_heights;
    const size_t pre_mh_mv_bytes_per_head = pre_mh_mv_elements_per_head * sizeof(float);

    // --- Local work size definitions (brought into scope) ---
    const size_t local_work_size_1d = 64;
    cl::NDRange local_1d(local_work_size_1d);
    auto calculate_global_1d = [&](size_t total_size) {
        return cl::NDRange(((total_size + local_work_size_1d - 1) / local_work_size_1d) * local_work_size_1d);
    };

    const size_t local_work_size_2d_arr[2] = { 8, 8 };
    cl::NDRange local_2d(local_work_size_2d_arr[0], local_work_size_2d_arr[1]);
    auto calculate_global_2d = [&](size_t dim0, size_t dim1) {
        size_t g0 = ((dim0 + local_work_size_2d_arr[0] - 1) / local_work_size_2d_arr[0]) * local_work_size_2d_arr[0];
        size_t g1 = ((dim1 + local_work_size_2d_arr[1] - 1) / local_work_size_2d_arr[1]) * local_work_size_2d_arr[1];
        return cl::NDRange(g0, g1);
    };

    cl::Buffer d_tokenEmbed;
    cl::Buffer agg_d_expected_h, agg_d_EH, agg_d_EV;
    cl::Buffer agg_d_grad_EH, agg_d_grad_EV_scaled;
    cl::Buffer agg_d_grad_dh, agg_d_grad_dv;
    cl::Buffer agg_d_KdotQ, agg_d_head_storage;
    cl::Buffer agg_d_K, agg_d_Q;
    cl::Buffer agg_d_pre_MH, agg_d_pre_MV;
    cl::Buffer agg_d_MH_a, agg_d_MV_a, agg_d_MQ_a, agg_d_MK_a;
    cl::Buffer agg_d_grad_MH, agg_d_grad_MV;
    cl::Buffer agg_d_grad_head_storage_buf;
    cl::Buffer agg_d_lota_deriv, agg_d_grad_KdotQ_buf;
    cl::Buffer agg_d_grad_K_buf, agg_d_grad_Q_buf;
    cl::Buffer agg_d_grad_MQ_buf, agg_d_grad_MK_buf;
    cl::Buffer agg_d_hor_activations_storage, agg_d_ver_activations_storage;
    cl::Buffer agg_d_hor_weights_storage, agg_d_ver_weights_storage;
    cl::Buffer agg_d_hor_gweights_storage, agg_d_ver_gweights_storage;
    cl::Buffer agg_d_hor_deltas_storage, agg_d_ver_deltas_storage;
    cl::Buffer agg_d_grad_tokens;
    cl::Buffer agg_d_tokenEmbed_grad_for_each_head;

    cl::Buffer agg_d_MQt, agg_d_MKt;
    cl::Buffer agg_d_head_MKt, agg_d_head_MQt;
    cl::Buffer agg_d_head_MKt_t, agg_d_head_MQt_t;
    cl::Buffer agg_d_hMKt_t, agg_d_hMQt_t;
    cl::Buffer agg_d_grad_T_Q, agg_d_grad_T_K, agg_d_grad_T_h, agg_d_grad_T_v;
    cl::Buffer agg_d_grad_T;

    std::vector<cl::CommandQueue> streams_cl(num_heads_to_process);
    std::vector<HeadDeviceSubBuffersirH> head_gpu_data_cl(num_heads_to_process);

    try {
        cl::Buffer d_dL_dTokenEmbed_aggregated_column_output(clcontext.context, CL_MEM_READ_WRITE, static_cast<size_t>(tokenCount) * embedding_dim * sizeof(float), nullptr, &cl_err); 
        CL_CHECK(cl_err);
        CL_CHECK(clcontext.queue.enqueueFillBuffer(d_dL_dTokenEmbed_aggregated_column_output, 0.0f, 0, static_cast<size_t>(tokenCount) * embedding_dim * sizeof(float)));

        d_tokenEmbed = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, token_embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_expected_h = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_EH = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_EV = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * ev_single_head_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_grad_EH = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_grad_EV_scaled = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_grad_dh = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_grad_dv = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_KdotQ = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * max_head_bytes_per_head, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_head_storage = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * max_head_bytes_per_head, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_K = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * max_k_q_bytes_per_head, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_Q = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * max_k_q_bytes_per_head, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_pre_MH = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * pre_mh_mv_bytes_per_head, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_pre_MV = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * pre_mh_mv_bytes_per_head, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_MH_a = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * proj_mat_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_MV_a = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * proj_mat_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_MQ_a = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * proj_mat_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_MK_a = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * proj_mat_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_grad_MH = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * proj_mat_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_grad_MV = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * proj_mat_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_grad_head_storage_buf = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * max_head_bytes_per_head, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_lota_deriv = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * max_head_bytes_per_head, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_grad_KdotQ_buf = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * max_head_bytes_per_head, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_grad_K_buf = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * max_k_q_bytes_per_head, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_grad_Q_buf = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * max_k_q_bytes_per_head, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_grad_MQ_buf = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * proj_mat_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_grad_MK_buf = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * proj_mat_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_hor_activations_storage = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * num_neuron_layers_mlp * embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_ver_activations_storage = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * num_neuron_layers_mlp * embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_hor_weights_storage = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * num_weight_matrices_mlp * mlp_weights_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_ver_weights_storage = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * num_weight_matrices_mlp * mlp_weights_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_hor_gweights_storage = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * num_weight_matrices_mlp * mlp_weights_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_ver_gweights_storage = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * num_weight_matrices_mlp * mlp_weights_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_hor_deltas_storage = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * num_neuron_layers_mlp * embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_ver_deltas_storage = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * num_neuron_layers_mlp * embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_tokenEmbed_grad_for_each_head = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);

        agg_d_MQt = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * proj_mat_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_MKt = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * proj_mat_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_head_MKt = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * proj_mat_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_head_MQt = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * proj_mat_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_head_MKt_t = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * proj_mat_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_head_MQt_t = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * proj_mat_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_hMKt_t = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * proj_mat_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_hMQt_t = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * proj_mat_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_grad_T_Q = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_grad_T_K = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_grad_T_h = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_grad_T_v = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_grad_T = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);

        if(layno_col_idx == 0) {
            gradToken.resize(EMBEDDING, 0.0f);
            agg_d_grad_tokens = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        }

        for (int head_idx = 0; head_idx < num_heads_to_process; ++head_idx) {
            streams_cl[head_idx] = cl::CommandQueue(clcontext.context, clcontext.device, 0, &cl_err); CL_CHECK(cl_err);
            head_gpu_data_cl[head_idx].d_hor_activations.resize(num_neuron_layers_mlp);
            head_gpu_data_cl[head_idx].d_hor_weights.resize(num_weight_matrices_mlp);
            head_gpu_data_cl[head_idx].d_hor_gweights.resize(num_weight_matrices_mlp);
            head_gpu_data_cl[head_idx].d_hor_deltas.resize(num_neuron_layers_mlp);
            head_gpu_data_cl[head_idx].d_ver_activations.resize(num_neuron_layers_mlp);
            head_gpu_data_cl[head_idx].d_ver_weights.resize(num_weight_matrices_mlp);
            head_gpu_data_cl[head_idx].d_ver_gweights.resize(num_weight_matrices_mlp);
            head_gpu_data_cl[head_idx].d_ver_deltas.resize(num_neuron_layers_mlp);

            cl_buffer_region region;
            region = { head_idx * embed_bytes, embed_bytes };
            head_gpu_data_cl[head_idx].d_expected_h = agg_d_expected_h.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
            head_gpu_data_cl[head_idx].d_EH = agg_d_EH.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
            head_gpu_data_cl[head_idx].d_grad_EH = agg_d_grad_EH.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
            head_gpu_data_cl[head_idx].d_grad_EV_scaled = agg_d_grad_EV_scaled.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
            head_gpu_data_cl[head_idx].d_grad_dh = agg_d_grad_dh.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
            head_gpu_data_cl[head_idx].d_grad_dv = agg_d_grad_dv.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);

            region = { head_idx * ev_single_head_bytes, ev_single_head_bytes };
            head_gpu_data_cl[head_idx].d_EV = agg_d_EV.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);

            region = { head_idx * max_head_bytes_per_head, max_head_bytes_per_head };
            head_gpu_data_cl[head_idx].d_KdotQ = agg_d_KdotQ.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
            head_gpu_data_cl[head_idx].d_head = agg_d_head_storage.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
            head_gpu_data_cl[head_idx].d_grad_head = agg_d_grad_head_storage_buf.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
            head_gpu_data_cl[head_idx].d_lota_deriv = agg_d_lota_deriv.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
            head_gpu_data_cl[head_idx].d_grad_KdotQ = agg_d_grad_KdotQ_buf.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);

            region = { head_idx * max_k_q_bytes_per_head, max_k_q_bytes_per_head };
            head_gpu_data_cl[head_idx].d_K = agg_d_K.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
            head_gpu_data_cl[head_idx].d_Q = agg_d_Q.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
            head_gpu_data_cl[head_idx].d_grad_K = agg_d_grad_K_buf.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
            head_gpu_data_cl[head_idx].d_grad_Q = agg_d_grad_Q_buf.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);

            region = { head_idx * pre_mh_mv_bytes_per_head, pre_mh_mv_bytes_per_head };
            head_gpu_data_cl[head_idx].d_pre_MH = agg_d_pre_MH.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
            head_gpu_data_cl[head_idx].d_pre_MV = agg_d_pre_MV.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);

            region = { head_idx * proj_mat_bytes, proj_mat_bytes };
            head_gpu_data_cl[head_idx].d_MH_a = agg_d_MH_a.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
            head_gpu_data_cl[head_idx].d_MV_a = agg_d_MV_a.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
            head_gpu_data_cl[head_idx].d_MQ_a = agg_d_MQ_a.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
            head_gpu_data_cl[head_idx].d_MK_a = agg_d_MK_a.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
            head_gpu_data_cl[head_idx].d_grad_MH = agg_d_grad_MH.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
            head_gpu_data_cl[head_idx].d_grad_MV = agg_d_grad_MV.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
            head_gpu_data_cl[head_idx].d_grad_MQ = agg_d_grad_MQ_buf.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
            head_gpu_data_cl[head_idx].d_grad_MK = agg_d_grad_MK_buf.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);

            // for token embedding gradients
            region = { head_idx * embed_bytes, embed_bytes }; // FIX: Use embed_bytes, not token_embed_bytes
            head_gpu_data_cl[head_idx].d_grad_token = agg_d_grad_tokens.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);

            // for MLPs
            for (int l = 0; l < num_neuron_layers_mlp; ++l) {
                region = { (head_idx * num_neuron_layers_mlp + l) * embed_bytes, embed_bytes };
                head_gpu_data_cl[head_idx].d_hor_activations[l] = agg_d_hor_activations_storage.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
                head_gpu_data_cl[head_idx].d_ver_activations[l] = agg_d_ver_activations_storage.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
                head_gpu_data_cl[head_idx].d_hor_deltas[l] = agg_d_hor_deltas_storage.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
                head_gpu_data_cl[head_idx].d_ver_deltas[l] = agg_d_ver_deltas_storage.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
            }
            for (int l = 0; l < num_weight_matrices_mlp; ++l) {
                region = { (head_idx * num_weight_matrices_mlp + l) * mlp_weights_bytes, mlp_weights_bytes };
                head_gpu_data_cl[head_idx].d_hor_weights[l] = agg_d_hor_weights_storage.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
                head_gpu_data_cl[head_idx].d_ver_weights[l] = agg_d_ver_weights_storage.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
                head_gpu_data_cl[head_idx].d_hor_gweights[l] = agg_d_hor_gweights_storage.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
                head_gpu_data_cl[head_idx].d_ver_gweights[l] = agg_d_ver_gweights_storage.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
            }

            // sub-buffers for gradient accumulation for token embedding from all heads
            region = { head_idx * proj_mat_bytes, proj_mat_bytes };
            head_gpu_data_cl[head_idx].d_MQt = agg_d_MQt.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
            head_gpu_data_cl[head_idx].d_MKt = agg_d_MKt.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
            head_gpu_data_cl[head_idx].d_head_MKt = agg_d_head_MKt.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
            head_gpu_data_cl[head_idx].d_head_MQt = agg_d_head_MQt.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
            head_gpu_data_cl[head_idx].d_head_MKt_t = agg_d_head_MKt_t.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
            head_gpu_data_cl[head_idx].d_head_MQt_t = agg_d_head_MQt_t.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
            head_gpu_data_cl[head_idx].d_hMKt_t = agg_d_hMKt_t.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
            head_gpu_data_cl[head_idx].d_hMQt_t = agg_d_hMQt_t.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
            region = { head_idx * embed_bytes, embed_bytes };
            head_gpu_data_cl[head_idx].d_grad_T = agg_d_grad_T.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
            head_gpu_data_cl[head_idx].d_grad_T_Q = agg_d_grad_T_Q.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
            head_gpu_data_cl[head_idx].d_grad_T_K = agg_d_grad_T_K.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
            head_gpu_data_cl[head_idx].d_grad_T_h = agg_d_grad_T_h.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
            head_gpu_data_cl[head_idx].d_grad_T_v = agg_d_grad_T_v.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
        }

        for (int head_idx = num_heads_to_process - 1; head_idx >= 0; --head_idx) {
            attention& head_obj = b[head_idx][layno_col_idx];
            HeadDeviceSubBuffersirH& device_ptrs = head_gpu_data_cl[head_idx];
            cl::CommandQueue& current_stream = streams_cl[head_idx];

            bool att_is_self = head_obj.isSelfAttention;
            const size_t active_head_elements = static_cast<size_t>(token_count) * token_count;
            const size_t active_head_bytes = active_head_elements * sizeof(float);
            const size_t active_k_q_elements = static_cast<size_t>(token_count) * mat_heights;
            const size_t active_k_q_bytes = active_k_q_elements * sizeof(float);

            auto calculate_global_1d = [&](size_t total_size) { return ((total_size + local_work_size_1d - 1) / local_work_size_1d) * local_work_size_1d; };
            auto calculate_global_2d = [&](size_t dim0, size_t dim1) { 
                size_t g0 = ((dim0 + local_work_size_2d_arr[0] - 1) / local_work_size_2d_arr[0]) * local_work_size_2d_arr[0]; 
                size_t g1 = ((dim1 + local_work_size_2d_arr[1] - 1) / local_work_size_2d_arr[1]) * local_work_size_2d_arr[1]; 
                return cl::NDRange(g0, g1); 
            };

            cl::NDRange global_embed(calculate_global_1d(embedding_dim));
            cl::NDRange global_mat_heights(calculate_global_1d(mat_heights));
            cl::NDRange global_proj_mat(calculate_global_1d(proj_mat_elements));
            cl::NDRange global_ev(calculate_global_1d(ev_single_head_elements));
            cl::NDRange global_embed_2d = calculate_global_2d(embedding_dim, embedding_dim);
            cl::NDRange global_matrix_2d = calculate_global_2d(embedding_dim, mat_heights);
            cl::NDRange global_head_2d = calculate_global_2d(token_count, token_count);
            cl::NDRange global_kq_grad_2d = calculate_global_2d(mat_heights, token_count);
            cl::NDRange global_head_1d(calculate_global_1d(active_head_elements));

            CL_CHECK(current_stream.enqueueWriteBuffer(device_ptrs.d_expected_h, CL_FALSE, 0, embed_bytes, expectedH[head_idx].data()));
            CL_CHECK(current_stream.enqueueWriteBuffer(device_ptrs.d_EH, CL_FALSE, 0, embed_bytes, head_obj.EH.data()));
            CL_CHECK(current_stream.enqueueWriteBuffer(device_ptrs.d_EV, CL_FALSE, 0, ev_single_head_bytes, head_obj.EV.mapped_data));
            CL_CHECK(current_stream.enqueueWriteBuffer(device_ptrs.d_MH_a, CL_FALSE, 0, proj_mat_bytes, head_obj.MH.mapped_data));
            CL_CHECK(current_stream.enqueueWriteBuffer(device_ptrs.d_MV_a, CL_FALSE, 0, proj_mat_bytes, head_obj.MV.mapped_data));
            CL_CHECK(current_stream.enqueueWriteBuffer(device_ptrs.d_MQ_a, CL_FALSE, 0, proj_mat_bytes, head_obj.MQ.mapped_data));
            CL_CHECK(current_stream.enqueueWriteBuffer(device_ptrs.d_MK_a, CL_FALSE, 0, proj_mat_bytes, head_obj.MK.mapped_data));
            if (token_count > 0) {
                CL_CHECK(current_stream.enqueueWriteBuffer(device_ptrs.d_KdotQ, CL_FALSE, 0, active_head_bytes, head_obj.KdotQ.mapped_data));
                CL_CHECK(current_stream.enqueueWriteBuffer(device_ptrs.d_K, CL_FALSE, 0, active_k_q_bytes, head_obj.K.mapped_data));
                CL_CHECK(current_stream.enqueueWriteBuffer(device_ptrs.d_Q, CL_FALSE, 0, active_k_q_bytes, head_obj.Q.mapped_data)); 
            }
            for(int l=0; l<num_neuron_layers_mlp; ++l) { 
                CL_CHECK(current_stream.enqueueWriteBuffer(device_ptrs.d_hor_activations[l], CL_FALSE, 0, embed_bytes, head_obj.hor.activations[l].data()));
                CL_CHECK(current_stream.enqueueWriteBuffer(device_ptrs.d_ver_activations[l], CL_FALSE, 0, embed_bytes, head_obj.ver.activations[l].data()));
            }
            for(int l=0; l<num_weight_matrices_mlp; ++l) { 
                CL_CHECK(current_stream.enqueueWriteBuffer(device_ptrs.d_hor_weights[l], CL_FALSE, 0, mlp_weights_bytes, head_obj.hor.weights[l].mapped_data));
                CL_CHECK(current_stream.enqueueWriteBuffer(device_ptrs.d_ver_weights[l], CL_FALSE, 0, mlp_weights_bytes, head_obj.ver.weights[l].mapped_data));
                CL_CHECK(current_stream.enqueueWriteBuffer(device_ptrs.d_hor_gweights[l], CL_FALSE, 0, mlp_weights_bytes, head_obj.hor.gweights[l].mapped_data));
                CL_CHECK(current_stream.enqueueWriteBuffer(device_ptrs.d_ver_gweights[l], CL_FALSE, 0, mlp_weights_bytes, head_obj.ver.gweights[l].mapped_data));
            }
            current_stream.flush();

            cl::Kernel k_grad_eh_ev = clcontext.kernels.at("kernelComputeGradientsEH_EV");
            CL_CHECK(k_grad_eh_ev.setArg(0, device_ptrs.d_EH));
            CL_CHECK(k_grad_eh_ev.setArg(1, device_ptrs.d_expected_h));
            CL_CHECK(k_grad_eh_ev.setArg(2, device_ptrs.d_grad_EH));
            CL_CHECK(k_grad_eh_ev.setArg(3, device_ptrs.d_grad_EV_scaled));
            CL_CHECK(k_grad_eh_ev.setArg(4, embedding_dim));
            CL_CHECK(current_stream.enqueueNDRangeKernel(k_grad_eh_ev, cl::NullRange, global_embed, local_1d)); 

            cl::Kernel k_last_delta_h = clcontext.kernels.at("kernelLastLayerDeltaSigmoid"); 
            CL_CHECK(k_last_delta_h.setArg(0, device_ptrs.d_grad_EH));
            CL_CHECK(k_last_delta_h.setArg(1, device_ptrs.d_hor_activations[num_neuron_layers_mlp - 1]));
            CL_CHECK(k_last_delta_h.setArg(2, device_ptrs.d_hor_deltas[num_neuron_layers_mlp - 1]));
            CL_CHECK(k_last_delta_h.setArg(3, embedding_dim));
            CL_CHECK(current_stream.enqueueNDRangeKernel(k_last_delta_h, cl::NullRange, global_embed, local_1d));

            cl::Kernel k_hidden_delta_h = clcontext.kernels.at("kernelHiddenDeltaSigmoid"); 
            for (int l = num_neuron_layers_mlp - 2; l >= 0; --l) { 
                CL_CHECK(k_hidden_delta_h.setArg(0, device_ptrs.d_hor_deltas[l+1]));
                CL_CHECK(k_hidden_delta_h.setArg(1, device_ptrs.d_hor_weights[l]));
                CL_CHECK(k_hidden_delta_h.setArg(2, device_ptrs.d_hor_activations[l]));
                CL_CHECK(k_hidden_delta_h.setArg(3, device_ptrs.d_hor_deltas[l]));
                CL_CHECK(k_hidden_delta_h.setArg(4, embedding_dim));
                CL_CHECK(k_hidden_delta_h.setArg(5, embedding_dim));
                CL_CHECK(current_stream.enqueueNDRangeKernel(k_hidden_delta_h, cl::NullRange, global_embed, local_1d));
            }

            cl::Kernel k_update_weights_h = clcontext.kernels.at("kernelUpdateElasticNet"); 
            for (int l = 0; l < num_weight_matrices_mlp; ++l) { 
                CL_CHECK(k_update_weights_h.setArg(0, device_ptrs.d_hor_deltas[l+1]));
                CL_CHECK(k_update_weights_h.setArg(1, device_ptrs.d_hor_activations[l]));
                CL_CHECK(k_update_weights_h.setArg(2, device_ptrs.d_hor_weights[l]));
                CL_CHECK(k_update_weights_h.setArg(3, device_ptrs.d_hor_gweights[l]));
                CL_CHECK(k_update_weights_h.setArg(4, learning_rate));
                CL_CHECK(k_update_weights_h.setArg(5, lambda_l1));
                CL_CHECK(k_update_weights_h.setArg(6, lambda_l2));
                CL_CHECK(k_update_weights_h.setArg(7, embedding_dim));
                CL_CHECK(k_update_weights_h.setArg(8, embedding_dim));
                CL_CHECK(current_stream.enqueueNDRangeKernel(k_update_weights_h, cl::NullRange, global_embed_2d, local_2d));
            }

            cl::Kernel k_last_delta_v = clcontext.kernels.at("kernelLastLayerDeltaSigmoid");
            CL_CHECK(k_last_delta_v.setArg(0, device_ptrs.d_grad_EV_scaled)); // Now uses potentially clipped d_grad_EV_scaled
            CL_CHECK(k_last_delta_v.setArg(1, device_ptrs.d_ver_activations[num_neuron_layers_mlp - 1]));
            CL_CHECK(k_last_delta_v.setArg(2, device_ptrs.d_ver_deltas[num_neuron_layers_mlp - 1]));
            CL_CHECK(k_last_delta_v.setArg(3, embedding_dim));
            CL_CHECK(current_stream.enqueueNDRangeKernel(k_last_delta_v, cl::NullRange, global_embed, local_1d));

            cl::Kernel k_hidden_delta_v = clcontext.kernels.at("kernelHiddenDeltaSigmoid"); 
            for (int l = num_neuron_layers_mlp - 2; l >= 0; --l) { 
                CL_CHECK(k_hidden_delta_v.setArg(0, device_ptrs.d_ver_deltas[l+1]));
                CL_CHECK(k_hidden_delta_v.setArg(1, device_ptrs.d_ver_weights[l]));
                CL_CHECK(k_hidden_delta_v.setArg(2, device_ptrs.d_ver_activations[l]));
                CL_CHECK(k_hidden_delta_v.setArg(3, device_ptrs.d_ver_deltas[l]));
                CL_CHECK(k_hidden_delta_v.setArg(4, embedding_dim));
                CL_CHECK(k_hidden_delta_v.setArg(5, embedding_dim));
                CL_CHECK(current_stream.enqueueNDRangeKernel(k_hidden_delta_v, cl::NullRange, global_embed, local_1d));
            }

            cl::Kernel k_update_weights_v = clcontext.kernels.at("kernelUpdateElasticNet"); 
            for (int l = 0; l < num_weight_matrices_mlp; ++l) { 
                CL_CHECK(k_update_weights_v.setArg(0, device_ptrs.d_ver_deltas[l+1]));
                CL_CHECK(k_update_weights_v.setArg(1, device_ptrs.d_ver_activations[l]));
                CL_CHECK(k_update_weights_v.setArg(2, device_ptrs.d_ver_weights[l]));
                CL_CHECK(k_update_weights_v.setArg(3, device_ptrs.d_ver_gweights[l]));
                CL_CHECK(k_update_weights_v.setArg(4, learning_rate));
                CL_CHECK(k_update_weights_v.setArg(5, lambda_l1));
                CL_CHECK(k_update_weights_v.setArg(6, lambda_l2));
                CL_CHECK(k_update_weights_v.setArg(7, embedding_dim));
                CL_CHECK(k_update_weights_v.setArg(8, embedding_dim));
                CL_CHECK(current_stream.enqueueNDRangeKernel(k_update_weights_v, cl::NullRange, global_embed_2d, local_2d));
            }
            // Ensure MLP weight gradients are clipped before proceeding to MLP input gradient
            current_stream.finish();

            cl::Kernel k_grad_mlp_input = clcontext.kernels.at("kernelComputeGradMLPInput"); 
            CL_CHECK(k_grad_mlp_input.setArg(0, device_ptrs.d_hor_deltas[0]));
            CL_CHECK(k_grad_mlp_input.setArg(1, device_ptrs.d_hor_weights[0]));
            CL_CHECK(k_grad_mlp_input.setArg(2, device_ptrs.d_grad_dh));
            CL_CHECK(k_grad_mlp_input.setArg(3, embedding_dim));
            CL_CHECK(k_grad_mlp_input.setArg(4, embedding_dim));
            CL_CHECK(current_stream.enqueueNDRangeKernel(k_grad_mlp_input, cl::NullRange, global_embed, local_1d));
            
            CL_CHECK(k_grad_mlp_input.setArg(0, device_ptrs.d_ver_deltas[0]));
            CL_CHECK(k_grad_mlp_input.setArg(1, device_ptrs.d_ver_weights[0]));
            CL_CHECK(k_grad_mlp_input.setArg(2, device_ptrs.d_grad_dv));
            CL_CHECK(current_stream.enqueueNDRangeKernel(k_grad_mlp_input, cl::NullRange, global_embed, local_1d));

            if (token_count > 0) {
/*                cl::Kernel k_lota = clcontext.kernels.at("clLOTA2dmasking"); 
                CL_CHECK(k_lota.setArg(0, device_ptrs.d_KdotQ)); 
                CL_CHECK(k_lota.setArg(1, device_ptrs.d_head)); 
                CL_CHECK(k_lota.setArg(2, context_win)); // rows
                CL_CHECK(k_lota.setArg(3, context_win)); // cols
                CL_CHECK(k_lota.setArg(4, token_count)); 
                cl_int cl_att_is_self_lota = att_is_self ? 1 : 0;
                CL_CHECK(k_lota.setArg(5, cl_att_is_self_lota));
                CL_CHECK(current_stream.enqueueNDRangeKernel(k_lota, cl::NullRange, global_head_2d, local_2d));
*/

                cl::Kernel k_pre_mh_mv = clcontext.kernels.at("kernelComputePreMH_MV"); 
                CL_CHECK(k_pre_mh_mv.setArg(0, device_ptrs.d_head));
                CL_CHECK(k_pre_mh_mv.setArg(1, device_ptrs.d_K));
                CL_CHECK(k_pre_mh_mv.setArg(2, device_ptrs.d_Q));
                CL_CHECK(k_pre_mh_mv.setArg(3, device_ptrs.d_pre_MH));
                CL_CHECK(k_pre_mh_mv.setArg(4, device_ptrs.d_pre_MV));
                CL_CHECK(k_pre_mh_mv.setArg(5, token_count));
                CL_CHECK(k_pre_mh_mv.setArg(6, mat_heights));
                CL_CHECK(current_stream.enqueueNDRangeKernel(k_pre_mh_mv, cl::NullRange, global_mat_heights, local_1d));

                cl::Kernel k_grad_mh_mv = clcontext.kernels.at("kernelComputeGradMH_MV"); 
                CL_CHECK(k_grad_mh_mv.setArg(0, device_ptrs.d_pre_MH));
                CL_CHECK(k_grad_mh_mv.setArg(1, device_ptrs.d_pre_MV));
                CL_CHECK(k_grad_mh_mv.setArg(2, device_ptrs.d_grad_dh));
                CL_CHECK(k_grad_mh_mv.setArg(3, device_ptrs.d_grad_dv));
                CL_CHECK(k_grad_mh_mv.setArg(4, device_ptrs.d_grad_MH));
                CL_CHECK(k_grad_mh_mv.setArg(5, device_ptrs.d_grad_MV));
                CL_CHECK(k_grad_mh_mv.setArg(6, mat_heights));
                CL_CHECK(k_grad_mh_mv.setArg(7, embedding_dim));
                CL_CHECK(current_stream.enqueueNDRangeKernel(k_grad_mh_mv, cl::NullRange, global_matrix_2d, local_2d));

                cl::Kernel k_grad_head = clcontext.kernels.at("kernelComputeGradHead"); 
                CL_CHECK(k_grad_head.setArg(0, device_ptrs.d_K));
                CL_CHECK(k_grad_head.setArg(1, device_ptrs.d_Q));
                CL_CHECK(k_grad_head.setArg(2, device_ptrs.d_MH_a));
                CL_CHECK(k_grad_head.setArg(3, device_ptrs.d_MV_a));
                CL_CHECK(k_grad_head.setArg(4, device_ptrs.d_grad_dh));
                CL_CHECK(k_grad_head.setArg(5, device_ptrs.d_grad_dv));
                CL_CHECK(k_grad_head.setArg(6, device_ptrs.d_grad_head));
                CL_CHECK(k_grad_head.setArg(7, token_count));
                CL_CHECK(k_grad_head.setArg(8, mat_heights));
                CL_CHECK(k_grad_head.setArg(9, embedding_dim));
                CL_CHECK(current_stream.enqueueNDRangeKernel(k_grad_head, cl::NullRange, global_head_2d, local_2d));

                cl::Kernel k_lota_der = clcontext.kernels.at("clLOTA2ddermasking"); 
                CL_CHECK(k_lota_der.setArg(0, device_ptrs.d_KdotQ));
                CL_CHECK(k_lota_der.setArg(1, device_ptrs.d_lota_deriv));
                CL_CHECK(k_lota_der.setArg(2, context_win));
                CL_CHECK(k_lota_der.setArg(3, context_win));
                CL_CHECK(k_lota_der.setArg(4, token_count));
                cl_int cl_att_is_self_lota_der = att_is_self ? 1 : 0;
                CL_CHECK(k_lota_der.setArg(5, cl_att_is_self_lota_der));
                CL_CHECK(current_stream.enqueueNDRangeKernel(k_lota_der, cl::NullRange, global_head_2d, local_2d));

                cl::Kernel k_grad_kdotq_lota = clcontext.kernels.at("kernelComputeGradKdotQ_LOTA"); 
                CL_CHECK(k_grad_kdotq_lota.setArg(0, device_ptrs.d_grad_head));
                CL_CHECK(k_grad_kdotq_lota.setArg(1, device_ptrs.d_lota_deriv));
                CL_CHECK(k_grad_kdotq_lota.setArg(2, device_ptrs.d_grad_KdotQ));
                CL_CHECK(k_grad_kdotq_lota.setArg(3, scaling_factor));
                CL_CHECK(k_grad_kdotq_lota.setArg(4, static_cast<int>(active_head_elements)));
                CL_CHECK(current_stream.enqueueNDRangeKernel(k_grad_kdotq_lota, cl::NullRange, global_head_1d, local_1d));

                cl::Kernel k_grad_k_q = clcontext.kernels.at("kernelComputeGradK_Q"); 
                CL_CHECK(k_grad_k_q.setArg(0, device_ptrs.d_grad_KdotQ));
                CL_CHECK(k_grad_k_q.setArg(1, device_ptrs.d_K));
                CL_CHECK(k_grad_k_q.setArg(2, device_ptrs.d_Q));
                CL_CHECK(k_grad_k_q.setArg(3, device_ptrs.d_grad_K));
                CL_CHECK(k_grad_k_q.setArg(4, device_ptrs.d_grad_Q));
                CL_CHECK(k_grad_k_q.setArg(5, token_count));
                CL_CHECK(k_grad_k_q.setArg(6, mat_heights));
                CL_CHECK(current_stream.enqueueNDRangeKernel(k_grad_k_q, cl::NullRange, global_kq_grad_2d, local_2d));

                cl::Kernel k_grad_mk_mq = clcontext.kernels.at("kernelComputeGradMK_MQ"); 
                CL_CHECK(k_grad_mk_mq.setArg(0, device_ptrs.d_grad_K));
                CL_CHECK(k_grad_mk_mq.setArg(1, device_ptrs.d_grad_Q));
                CL_CHECK(k_grad_mk_mq.setArg(2, device_ptrs.d_K));
                CL_CHECK(k_grad_mk_mq.setArg(3, device_ptrs.d_Q));
                CL_CHECK(k_grad_mk_mq.setArg(4, device_ptrs.d_grad_MK));
                CL_CHECK(k_grad_mk_mq.setArg(5, device_ptrs.d_grad_MQ));
                CL_CHECK(k_grad_mk_mq.setArg(6, token_count));
                CL_CHECK(k_grad_mk_mq.setArg(7, mat_heights));
                CL_CHECK(k_grad_mk_mq.setArg(8, embedding_dim));
                CL_CHECK(current_stream.enqueueNDRangeKernel(k_grad_mk_mq, cl::NullRange, global_matrix_2d, local_2d));
                current_stream.finish();
            }
            else {
                CL_CHECK(current_stream.enqueueFillBuffer(device_ptrs.d_grad_MH, 0.0f, 0, proj_mat_bytes)); 
                CL_CHECK(current_stream.enqueueFillBuffer(device_ptrs.d_grad_MV, 0.0f, 0, proj_mat_bytes)); 
                CL_CHECK(current_stream.enqueueFillBuffer(device_ptrs.d_grad_MQ, 0.0f, 0, proj_mat_bytes)); 
                CL_CHECK(current_stream.enqueueFillBuffer(device_ptrs.d_grad_MK, 0.0f, 0, proj_mat_bytes));
            }

            if(layno_col_idx == 0) {
                // get gradient for token from gradients of MQ, MK, and gradients from h and v
                cl::Kernel transpose = clcontext.kernels.at("kernelTransposeMatrix");
                cl::Kernel matmul = clcontext.kernels.at("matrix_multiply");
                cl::Kernel vecmatmul = clcontext.kernels.at("vector_matrix_multiply");
                // dL/dT_Q = d_grad_Q * MQ^T
                CL_CHECK(transpose.setArg(0, device_ptrs.d_MQ_a));
                CL_CHECK(transpose.setArg(1, device_ptrs.d_MQt));
                CL_CHECK(transpose.setArg(2, mat_heights));
                CL_CHECK(transpose.setArg(3, embedding_dim));
                CL_CHECK(current_stream.enqueueNDRangeKernel(transpose, cl::NullRange, global_mat_heights, local_1d));
                CL_CHECK(vecmatmul.setArg(0, device_ptrs.d_grad_Q));
                CL_CHECK(vecmatmul.setArg(1, device_ptrs.d_MQt));
                CL_CHECK(vecmatmul.setArg(2, device_ptrs.d_grad_T_Q));
                CL_CHECK(vecmatmul.setArg(3, mat_heights));
                CL_CHECK(vecmatmul.setArg(4, embedding_dim));
                CL_CHECK(current_stream.enqueueNDRangeKernel(vecmatmul, cl::NullRange, global_embed, local_1d));
                // dL/dT_K = d_grad_K * MK^T
                CL_CHECK(transpose.setArg(0, device_ptrs.d_MK_a));
                CL_CHECK(transpose.setArg(1, device_ptrs.d_MKt));
                CL_CHECK(transpose.setArg(2, mat_heights));
                CL_CHECK(transpose.setArg(3, embedding_dim));
                CL_CHECK(current_stream.enqueueNDRangeKernel(transpose, cl::NullRange, global_mat_heights, local_1d));
                CL_CHECK(vecmatmul.setArg(0, device_ptrs.d_grad_K));
                CL_CHECK(vecmatmul.setArg(1, device_ptrs.d_MKt));
                CL_CHECK(vecmatmul.setArg(2, device_ptrs.d_grad_T_K));
                CL_CHECK(vecmatmul.setArg(3, mat_heights));
                CL_CHECK(vecmatmul.setArg(4, embedding_dim));
                CL_CHECK(current_stream.enqueueNDRangeKernel(vecmatmul, cl::NullRange, global_embed, local_1d));
                // dL/dT_h = d_grad_dh * (d_head * MK^T)^T * MH
                CL_CHECK(matmul.setArg(0, device_ptrs.d_head));
                CL_CHECK(matmul.setArg(1, device_ptrs.d_MKt));
                CL_CHECK(matmul.setArg(2, device_ptrs.d_head_MKt));
                CL_CHECK(matmul.setArg(3, context_win));
                CL_CHECK(matmul.setArg(4, mat_heights));
                CL_CHECK(matmul.setArg(5, embedding_dim));
                CL_CHECK(current_stream.enqueueNDRangeKernel(matmul, cl::NullRange, global_embed, local_1d));
                CL_CHECK(transpose.setArg(0, device_ptrs.d_head_MKt));
                CL_CHECK(transpose.setArg(1, device_ptrs.d_head_MKt_t));
                CL_CHECK(transpose.setArg(2, embedding_dim));
                CL_CHECK(transpose.setArg(3, mat_heights));
                CL_CHECK(current_stream.enqueueNDRangeKernel(transpose, cl::NullRange, global_mat_heights, local_1d));
                CL_CHECK(matmul.setArg(0, device_ptrs.d_head_MKt_t));
                CL_CHECK(matmul.setArg(1, device_ptrs.d_MH_a));
                CL_CHECK(matmul.setArg(2, device_ptrs.d_hMKt_t));
                CL_CHECK(matmul.setArg(3, embedding_dim));
                CL_CHECK(matmul.setArg(4, context_win));
                CL_CHECK(matmul.setArg(5, embedding_dim));
                CL_CHECK(current_stream.enqueueNDRangeKernel(matmul, cl::NullRange, global_embed, local_1d));
                CL_CHECK(vecmatmul.setArg(0, device_ptrs.d_grad_dh));
                CL_CHECK(vecmatmul.setArg(1, device_ptrs.d_hMKt_t));
                CL_CHECK(vecmatmul.setArg(2, device_ptrs.d_grad_T_h));
                CL_CHECK(vecmatmul.setArg(3, embedding_dim));
                CL_CHECK(vecmatmul.setArg(4, embedding_dim));
                CL_CHECK(current_stream.enqueueNDRangeKernel(vecmatmul, cl::NullRange, global_embed, local_1d));
                // dL/dT_v = d_grad_dv * (d_head * MQ^T)^T * MV
                CL_CHECK(matmul.setArg(0, device_ptrs.d_head));
                CL_CHECK(matmul.setArg(1, device_ptrs.d_MQt));
                CL_CHECK(matmul.setArg(2, device_ptrs.d_head_MQt));
                CL_CHECK(matmul.setArg(3, context_win));
                CL_CHECK(matmul.setArg(4, mat_heights));
                CL_CHECK(matmul.setArg(5, embedding_dim));
                CL_CHECK(current_stream.enqueueNDRangeKernel(matmul, cl::NullRange, global_embed, local_1d));
                CL_CHECK(transpose.setArg(0, device_ptrs.d_head_MQt));
                CL_CHECK(transpose.setArg(1, device_ptrs.d_head_MQt_t));
                CL_CHECK(transpose.setArg(2, embedding_dim));
                CL_CHECK(transpose.setArg(3, mat_heights));
                CL_CHECK(current_stream.enqueueNDRangeKernel(transpose, cl::NullRange, global_mat_heights, local_1d));
                CL_CHECK(matmul.setArg(0, device_ptrs.d_head_MQt_t));
                CL_CHECK(matmul.setArg(1, device_ptrs.d_MV_a));
                CL_CHECK(matmul.setArg(2, device_ptrs.d_hMQt_t));
                CL_CHECK(matmul.setArg(3, embedding_dim));
                CL_CHECK(matmul.setArg(4, context_win));
                CL_CHECK(matmul.setArg(5, embedding_dim));
                CL_CHECK(current_stream.enqueueNDRangeKernel(matmul, cl::NullRange, global_embed, local_1d));
                CL_CHECK(vecmatmul.setArg(0, device_ptrs.d_grad_dv));
                CL_CHECK(vecmatmul.setArg(1, device_ptrs.d_hMQt_t));
                CL_CHECK(vecmatmul.setArg(2, device_ptrs.d_grad_T_v));
                CL_CHECK(vecmatmul.setArg(3, embedding_dim));
                CL_CHECK(vecmatmul.setArg(4, embedding_dim));
                CL_CHECK(current_stream.enqueueNDRangeKernel(vecmatmul, cl::NullRange, global_embed, local_1d));
                // dL/dT = dL/dT_Q + dL/dT_K + dL/dT_h + dL/dT_v
                cl::Kernel addAll = clcontext.kernels.at("vectorsAddKernel");
                CL_CHECK(addAll.setArg(0, device_ptrs.d_grad_T_Q));
                CL_CHECK(addAll.setArg(1, device_ptrs.d_grad_T_K));
                CL_CHECK(addAll.setArg(2, device_ptrs.d_grad_T_h));
                CL_CHECK(addAll.setArg(3, device_ptrs.d_grad_T_v));
                CL_CHECK(addAll.setArg(4, device_ptrs.d_grad_token));
                CL_CHECK(addAll.setArg(5, embedding_dim));
                CL_CHECK(current_stream.enqueueNDRangeKernel(addAll, cl::NullRange, global_embed, local_1d));
            }

            // update weights MQ, MK, MH, MV,
            cl_int cl_update_eh_flag = 1;
            cl::Kernel k_update_1st_h = clcontext.kernels.at("kernelUpdateWeightsHeadElastic");
            CL_CHECK(k_update_1st_h.setArg(0, device_ptrs.d_MH_a));
            CL_CHECK(k_update_1st_h.setArg(1, device_ptrs.d_MV_a));
            CL_CHECK(k_update_1st_h.setArg(2, device_ptrs.d_MQ_a));
            CL_CHECK(k_update_1st_h.setArg(3, device_ptrs.d_MK_a));
            CL_CHECK(k_update_1st_h.setArg(4, device_ptrs.d_EH));
            CL_CHECK(k_update_1st_h.setArg(5, device_ptrs.d_grad_MH));
            CL_CHECK(k_update_1st_h.setArg(6, device_ptrs.d_grad_MV));
            CL_CHECK(k_update_1st_h.setArg(7, device_ptrs.d_grad_MQ));
            CL_CHECK(k_update_1st_h.setArg(8, device_ptrs.d_grad_MK));
            CL_CHECK(k_update_1st_h.setArg(9, device_ptrs.d_grad_EH));
            CL_CHECK(k_update_1st_h.setArg(10, cl_update_eh_flag));
            CL_CHECK(k_update_1st_h.setArg(11, mat_heights));
            CL_CHECK(k_update_1st_h.setArg(12, embedding_dim));
            CL_CHECK(k_update_1st_h.setArg(13, learning_rate));
            CL_CHECK(k_update_1st_h.setArg(14, lambda_l1));
            CL_CHECK(k_update_1st_h.setArg(15, lambda_l2));
            CL_CHECK(k_update_1st_h.setArg(16, MAX_GRAD_CLIP));
            CL_CHECK(current_stream.enqueueNDRangeKernel(k_update_1st_h, cl::NullRange, global_proj_mat, local_1d));

            for(int l=0; l<num_weight_matrices_mlp; ++l) {
                CL_CHECK(current_stream.enqueueReadBuffer(device_ptrs.d_hor_weights[l], CL_FALSE, 0, mlp_weights_bytes, head_obj.hor.weights[l].mapped_data)); 
                CL_CHECK(current_stream.enqueueReadBuffer(device_ptrs.d_hor_gweights[l], CL_FALSE, 0, mlp_weights_bytes, head_obj.hor.gweights[l].mapped_data)); 
                CL_CHECK(current_stream.enqueueReadBuffer(device_ptrs.d_ver_weights[l], CL_FALSE, 0, mlp_weights_bytes, head_obj.ver.weights[l].mapped_data)); 
                CL_CHECK(current_stream.enqueueReadBuffer(device_ptrs.d_ver_gweights[l], CL_FALSE, 0, mlp_weights_bytes, head_obj.ver.gweights[l].mapped_data)); 
            }
            CL_CHECK(current_stream.enqueueReadBuffer(device_ptrs.d_EH, CL_FALSE, 0, embed_bytes, head_obj.EH.data()));
            CL_CHECK(current_stream.enqueueReadBuffer(device_ptrs.d_EV, CL_FALSE, 0, ev_single_head_bytes, head_obj.EV.mapped_data));
            CL_CHECK(current_stream.enqueueReadBuffer(device_ptrs.d_MH_a, CL_FALSE, 0, proj_mat_bytes, head_obj.MH.mapped_data));
            CL_CHECK(current_stream.enqueueReadBuffer(device_ptrs.d_MV_a, CL_FALSE, 0, proj_mat_bytes, head_obj.MV.mapped_data));
            CL_CHECK(current_stream.enqueueReadBuffer(device_ptrs.d_MQ_a, CL_FALSE, 0, proj_mat_bytes, head_obj.MQ.mapped_data));
            CL_CHECK(current_stream.enqueueReadBuffer(device_ptrs.d_MK_a, CL_FALSE, 0, proj_mat_bytes, head_obj.MK.mapped_data));
            CL_CHECK(current_stream.finish());
        }

        if(layno_col_idx) {
            // Accumulate token embedding gradients from all heads
            // dL/dT = summation(dL/dT of each head)
            cl::NDRange global_embed(calculate_global_1d(embedding_dim));
            cl::Buffer finalgradtoken = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
            cl::Kernel addvec = clcontext.kernels.at("vectorAddKernel");
            CL_CHECK(addvec.setArg(0, finalgradtoken));
            CL_CHECK(addvec.setArg(2, finalgradtoken));
            CL_CHECK(addvec.setArg(3, embedding_dim));
            for(int i = 0; i < num_heads_to_process; ++i) {
                CL_CHECK(addvec.setArg(1, head_gpu_data_cl[i].d_grad_token));
                CL_CHECK(clcontext.queue.enqueueNDRangeKernel(addvec, cl::NullRange, global_embed, local_1d));
            }
            CL_CHECK(clcontext.queue.enqueueReadBuffer(finalgradtoken, CL_FALSE, 0, mlp_weights_bytes, gradToken.data())); 
            CL_CHECK(clcontext.queue.finish());
        }

        for (int head_idx = 0; head_idx < num_heads_to_process; ++head_idx) {
            CL_CHECK(streams_cl[head_idx].finish());
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Standard Exception during clrpartialbackward1stBlock(H_2D) for column " << layno_col_idx << ": " << e.what() << std::endl;
        throw;
    }

    // Explicitly release all aggregate OpenCL buffers
    agg_d_expected_h = cl::Buffer();
    agg_d_EH = cl::Buffer();
    agg_d_EV = cl::Buffer();
    agg_d_grad_EH = cl::Buffer();
    agg_d_grad_EV_scaled = cl::Buffer();
    agg_d_grad_dh = cl::Buffer();
    agg_d_grad_dv = cl::Buffer();
    agg_d_KdotQ = cl::Buffer();
    agg_d_head_storage = cl::Buffer();
    agg_d_K = cl::Buffer();
    agg_d_Q = cl::Buffer();
    agg_d_pre_MH = cl::Buffer();
    agg_d_pre_MV = cl::Buffer();
    agg_d_MH_a = cl::Buffer();
    agg_d_MV_a = cl::Buffer();
    agg_d_MQ_a = cl::Buffer();
    agg_d_MK_a = cl::Buffer();
    agg_d_grad_MH = cl::Buffer();
    agg_d_grad_MV = cl::Buffer();
    agg_d_grad_head_storage_buf = cl::Buffer();
    agg_d_lota_deriv = cl::Buffer();
    agg_d_grad_KdotQ_buf = cl::Buffer();
    agg_d_grad_K_buf = cl::Buffer();
    agg_d_grad_Q_buf = cl::Buffer();
    agg_d_grad_MQ_buf = cl::Buffer();
    agg_d_grad_MK_buf = cl::Buffer();
    agg_d_hor_activations_storage = cl::Buffer();
    agg_d_ver_activations_storage = cl::Buffer();
    agg_d_hor_weights_storage = cl::Buffer();
    agg_d_ver_weights_storage = cl::Buffer();
    agg_d_hor_gweights_storage = cl::Buffer();
    agg_d_ver_gweights_storage = cl::Buffer();
    agg_d_hor_deltas_storage = cl::Buffer();
    agg_d_ver_deltas_storage = cl::Buffer();

    if(layno_col_idx == 0) {
        agg_d_MQt = cl::Buffer();
        agg_d_MKt = cl::Buffer();
        agg_d_head_MKt = cl::Buffer();
        agg_d_head_MQt = cl::Buffer();
        agg_d_head_MKt_t = cl::Buffer();
        agg_d_head_MQt_t = cl::Buffer();
        agg_d_hMKt_t = cl::Buffer();
        agg_d_hMQt_t = cl::Buffer();
        agg_d_grad_T_Q = cl::Buffer();
        agg_d_grad_T_K = cl::Buffer();
        agg_d_grad_T_h = cl::Buffer();
        agg_d_grad_T_v = cl::Buffer();
        agg_d_grad_T = cl::Buffer();
    }
}

#endif