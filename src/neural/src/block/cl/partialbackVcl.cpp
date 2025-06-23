#ifdef USE_OPENCL
#if defined(_WIN64)
    #include <CL/cl.hpp>
#elif defined(__linux__)
    #define CL_HPP_TARGET_OPENCL_VERSION 220
    #include <CL/opencl.hpp>
#endif
#include <iostream>
#include <maths.hpp>
#include "include/attention.hpp"
#include "include/block.hpp"

// Macro for OpenCL error checking
#ifndef CL_CHECK
#define CL_CHECK(err) \
    if (err != CL_SUCCESS) { \
        fprintf(stderr, "OpenCL Error in %s at line %d: %s (Code: %d)\n", __FILE__, __LINE__, oclErrorString(err), err); \
        throw std::runtime_error("OpenCL Error: " + std::to_string(err)); \
    }
#endif


// Struct to hold per-head sub-buffers. This mirrors HeadDevicePointersV but holds cl::Buffer objects.
struct HeadDeviceSubBuffersV {
    cl::Buffer d_expected_v;
    cl::Buffer d_EV;
    cl::Buffer d_grad_EV_full, d_grad_EV_summed, d_grad_EV_scaled;
    cl::Buffer d_grad_dv;
    cl::Buffer d_KdotQ, d_head;
    cl::Buffer d_K, d_Q;
    cl::Buffer d_pre_MV;
    cl::Buffer d_MV_a, d_MQ_a, d_MK_a;
    cl::Buffer d_grad_MV;
    cl::Buffer d_grad_head;
    cl::Buffer d_lota_deriv;
    cl::Buffer d_grad_KdotQ;
    cl::Buffer d_grad_Q;
    cl::Buffer d_grad_MQ, d_grad_MK_correction;

    std::vector<cl::Buffer> d_ver_activations;
    std::vector<cl::Buffer> d_ver_weights, d_ver_deltas;
    std::vector<cl::Buffer> d_ver_gweights; // Added for ElasticNet
};

/**
 * @brief backpropagation via vertical retention vectors for first block
 * @param expectedV vertical retention vectors for each head of column
 * @param in embedding dimension and mlp input-output vector dimension
 * @param layers number of mlp activations layers
 * @param k column number
 */
void block::clpartialbackward1stBlock(std::vector<std::vector<std::vector<float>>>& expectedV, int& in, int& layers, int& k, float& learning)
{
        cl_int cl_err; // For OpenCL error codes
    const int num_heads_to_process = x; // 'x' is the number of rows/heads in this column

    // Validate column number and input shape
    if (k < 0 || k >= y) {
        throw std::out_of_range("clpartialbackward1stBlock(V): Column index 'k' (" + std::to_string(k) + ") is out of range [0, " + std::to_string(y - 1) + "].");
    }
    if (expectedV.size() != static_cast<size_t>(x)) {
        throw std::runtime_error("ExpectedV size mismatch (number of rows) in clpartialbackward1stBlock(V). Expected " + std::to_string(x) + ", got " + std::to_string(expectedV.size()));
    }
    if (!expectedV.empty() && (expectedV[0].size() != CONTEXT_WIN || (!expectedV[0].empty() && expectedV[0][0].size() != EMBEDDING))) {
        throw std::runtime_error("ExpectedV dimensions mismatch (context/embedding) in clpartialbackward1stBlock(V).");
    }
    if (EMBEDDING != in) {
        throw std::runtime_error("Embedding dimension mismatch: EMBEDDING vs in");
    }

    // Constants
    const int embedding_dim = EMBEDDING;
    const int mat_heights = MATHEIGHTS;
    const int context_win = CONTEXT_WIN;
    const float learning_rate = learning;
    const float scaling_factor = SCALING;
    const float lambda_l1 = 0.001f;     // L1 regularization parameter
    const float lambda_l2 = 0.001f;     // L2 regularization parameter

    // MLP structure parameters
    const int num_total_layers_mlp = layers;
    const int num_neuron_layers_mlp = num_total_layers_mlp; // Number of activation layers (including input)
    const int num_weight_matrices_mlp = num_total_layers_mlp - 1; // Number of weight matrices and delta vectors

    const size_t embed_bytes = embedding_dim * sizeof(float);
    const size_t mlp_weights_elements = static_cast<size_t>(embedding_dim) * embedding_dim; // Assuming square matrices for MLP
    const size_t mlp_weights_bytes = mlp_weights_elements * sizeof(float);
    const size_t proj_mat_elements = static_cast<size_t>(mat_heights) * embedding_dim; // For MV, MQ, MK projection matrices
    const size_t proj_mat_bytes = proj_mat_elements * sizeof(float);
    const size_t ev_total_elements = static_cast<size_t>(context_win) * embedding_dim; // For full EV
    const size_t ev_total_bytes = ev_total_elements * sizeof(float);
    const size_t max_head_elements_per_head = static_cast<size_t>(context_win) * context_win; // For KdotQ, head, grad_head, lota_deriv
    const size_t max_head_bytes_per_head = max_head_elements_per_head * sizeof(float);
    const size_t max_k_q_elements_per_head = static_cast<size_t>(context_win) * mat_heights; // For K, Q, grad_Q
    const size_t max_k_q_bytes_per_head = max_k_q_elements_per_head * sizeof(float);
    const size_t pre_mv_elements_per_head = mat_heights; // For pre_MV
    const size_t pre_mv_bytes_per_head = pre_mv_elements_per_head * sizeof(float);

    // Kernel Launch Config (fixed block/local sizes, global size will be calculated per kernel)
    const size_t local_work_size_1d = 256;
    cl::NDRange local_1d(local_work_size_1d);

    const size_t local_work_size_2d[2] = { 16, 16 };
    cl::NDRange local_2d(local_work_size_2d[0], local_work_size_2d[1]);

    // Aggregate OpenCL Buffers - These are the large buffers holding data for ALL heads
    cl::Buffer agg_d_expected_v_buf, agg_d_EV_buf;
    cl::Buffer agg_d_grad_EV_full_buf, agg_d_grad_EV_summed_buf, agg_d_grad_EV_scaled_buf;
    cl::Buffer agg_d_grad_dv_buf;
    cl::Buffer agg_d_KdotQ_buf, agg_d_head_storage_buf;
    cl::Buffer agg_d_K_buf, agg_d_Q_buf;
    cl::Buffer agg_d_pre_MV_buf;
    cl::Buffer agg_d_MV_a_buf, agg_d_MQ_a_buf, agg_d_MK_a_buf;
    cl::Buffer agg_d_grad_MV_buf;
    cl::Buffer agg_d_grad_head_storage_buf;
    cl::Buffer agg_d_lota_deriv_buf;
    cl::Buffer agg_d_grad_KdotQ_buf;
    cl::Buffer agg_d_grad_Q_storage_buf;
    cl::Buffer agg_d_grad_MQ_buf, agg_d_grad_MK_correction_buf;
    cl::Buffer agg_d_ver_activations_storage_buf;
    cl::Buffer agg_d_ver_weights_storage_buf;
    cl::Buffer agg_d_ver_deltas_storage_buf;

    // OpenCL Command Queues (one per head, like CUDA streams)
    std::vector<cl::CommandQueue> streams_cl(num_heads_to_process);
    // Vector to store per-head sub-buffers
    std::vector<HeadDeviceSubBuffersV> head_gpu_data_cl(num_heads_to_process);
     std::vector<std::string> kernelFiles = {
        "cl_kernels.cl"
    };

    try {
        OpenCLContext& clcontext = this->clcontext;
        // --- Allocate Aggregate Memory on Device ---
        // These buffers are allocated once for all heads combined.
        agg_d_expected_v_buf = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * ev_total_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_EV_buf = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * ev_total_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_grad_EV_full_buf = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * ev_total_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_grad_EV_summed_buf = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_grad_EV_scaled_buf = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_grad_dv_buf = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_KdotQ_buf = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * max_head_bytes_per_head, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_head_storage_buf = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * max_head_bytes_per_head, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_K_buf = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * max_k_q_bytes_per_head, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_Q_buf = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * max_k_q_bytes_per_head, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_pre_MV_buf = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * pre_mv_bytes_per_head, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_MV_a_buf = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * proj_mat_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_MQ_a_buf = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * proj_mat_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_MK_a_buf = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * proj_mat_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_grad_MV_buf = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * proj_mat_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_grad_head_storage_buf = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * max_head_bytes_per_head, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_lota_deriv_buf = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * max_head_bytes_per_head, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_grad_KdotQ_buf = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * max_head_bytes_per_head, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_grad_Q_storage_buf = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * max_k_q_bytes_per_head, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_grad_MQ_buf = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * proj_mat_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_grad_MK_correction_buf = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * proj_mat_bytes, nullptr, &cl_err); CL_CHECK(cl_err);

        // MLP Aggregate Storage (ver only)
        agg_d_ver_activations_storage_buf = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * num_neuron_layers_mlp * embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_ver_weights_storage_buf = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * num_weight_matrices_mlp * mlp_weights_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_ver_gweights_storage_buf = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * num_weight_matrices_mlp * mlp_weights_bytes, nullptr, &cl_err); CL_CHECK(cl_err); // Added
        agg_d_ver_deltas_storage_buf = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * num_weight_matrices_mlp * embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);

        // --- Create Command Queues and Setup Per-Head Sub-Buffers ---
        for (int head_idx = 0; head_idx < num_heads_to_process; ++head_idx) {
            // Create a non-blocking command queue for each head (OpenCL equivalent of a CUDA stream)
            streams_cl[head_idx] = cl::CommandQueue(clcontext.context, clcontext.device, 0, &cl_err);
            CL_CHECK(cl_err);

            // Resize MLP vectors in the sub-buffer struct
            head_gpu_data_cl[head_idx].d_ver_activations.resize(num_neuron_layers_mlp);
            head_gpu_data_cl[head_idx].d_ver_weights.resize(num_weight_matrices_mlp);
            head_gpu_data_cl[head_idx].d_ver_gweights.resize(num_weight_matrices_mlp); // Added
            head_gpu_data_cl[head_idx].d_ver_deltas.resize(num_weight_matrices_mlp); // Following CUDA code's usage

            // Create sub-buffers for each head from the aggregate buffers.
            // These sub-buffers are specific views into the larger aggregate buffers.
            cl_buffer_region region;

            region = { head_idx * ev_total_bytes, ev_total_bytes };
            head_gpu_data_cl[head_idx].d_expected_v = agg_d_expected_v_buf.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
            head_gpu_data_cl[head_idx].d_EV = agg_d_EV_buf.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
            head_gpu_data_cl[head_idx].d_grad_EV_full = agg_d_grad_EV_full_buf.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);

            region = { head_idx * embed_bytes, embed_bytes };
            head_gpu_data_cl[head_idx].d_grad_EV_summed = agg_d_grad_EV_summed_buf.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
            head_gpu_data_cl[head_idx].d_grad_EV_scaled = agg_d_grad_EV_scaled_buf.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
            head_gpu_data_cl[head_idx].d_grad_dv = agg_d_grad_dv_buf.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);

            region = { head_idx * max_head_bytes_per_head, max_head_bytes_per_head };
            head_gpu_data_cl[head_idx].d_KdotQ = agg_d_KdotQ_buf.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
            head_gpu_data_cl[head_idx].d_head = agg_d_head_storage_buf.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
            head_gpu_data_cl[head_idx].d_grad_head = agg_d_grad_head_storage_buf.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
            head_gpu_data_cl[head_idx].d_lota_deriv = agg_d_lota_deriv_buf.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
            head_gpu_data_cl[head_idx].d_grad_KdotQ = agg_d_grad_KdotQ_buf.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);

            region = { head_idx * max_k_q_bytes_per_head, max_k_q_bytes_per_head };
            head_gpu_data_cl[head_idx].d_K = agg_d_K_buf.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
            head_gpu_data_cl[head_idx].d_Q = agg_d_Q_buf.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
            head_gpu_data_cl[head_idx].d_grad_Q = agg_d_grad_Q_storage_buf.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);

            region = { head_idx * pre_mv_bytes_per_head, pre_mv_bytes_per_head };
            head_gpu_data_cl[head_idx].d_pre_MV = agg_d_pre_MV_buf.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);

            region = { head_idx * proj_mat_bytes, proj_mat_bytes };
            head_gpu_data_cl[head_idx].d_MV_a = agg_d_MV_a_buf.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
            head_gpu_data_cl[head_idx].d_MQ_a = agg_d_MQ_a_buf.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
            head_gpu_data_cl[head_idx].d_MK_a = agg_d_MK_a_buf.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
            head_gpu_data_cl[head_idx].d_grad_MV = agg_d_grad_MV_buf.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
            head_gpu_data_cl[head_idx].d_grad_MQ = agg_d_grad_MQ_buf.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
            head_gpu_data_cl[head_idx].d_grad_MK_correction = agg_d_grad_MK_correction_buf.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);

            // MLP sub-buffers (activations, weights, gradients, deltas)
            for (int l = 0; l < num_neuron_layers_mlp; ++l) { // All activation layers
                region = { (head_idx * num_neuron_layers_mlp + l) * embed_bytes, embed_bytes };
                head_gpu_data_cl[head_idx].d_ver_activations[l] = agg_d_ver_activations_storage_buf.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
            }
            for (int l = 0; l < num_weight_matrices_mlp; ++l) { // All weight/gradient/delta matrices
                region = { (head_idx * num_weight_matrices_mlp + l) * mlp_weights_bytes, mlp_weights_bytes };
                head_gpu_data_cl[head_idx].d_ver_weights[l] = agg_d_ver_weights_storage_buf.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);

                region = { (head_idx * num_weight_matrices_mlp + l) * mlp_weights_bytes, mlp_weights_bytes }; // Added
                head_gpu_data_cl[head_idx].d_ver_gweights[l] = agg_d_ver_gweights_storage_buf.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err); // Added

                region = { (head_idx * num_weight_matrices_mlp + l) * embed_bytes, embed_bytes }; // Assuming deltas are vector of size embed_bytes
                head_gpu_data_cl[head_idx].d_ver_deltas[l] = agg_d_ver_deltas_storage_buf.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
            }
        } // End of per-head command queue and sub-buffer creation loop

        // Iterate through the rows (parallels/heads) in the specified column
        for (int head_idx = 0; head_idx < num_heads_to_process; ++head_idx) {
            attention& head_obj = b[head_idx][k]; // Reference to the current attention head object
            bool attention_type = head_obj.isSelfAttention;
            std::vector<std::vector<float>>& expectedV_head = expectedV[head_idx]; // Expected output for this specific head
            HeadDeviceSubBuffersV& device_ptrs_cl = head_gpu_data_cl[head_idx]; // Sub-buffers for this head
            cl::CommandQueue& current_stream_cl = streams_cl[head_idx]; // Command queue for this head

            const int token_count = head_obj.tokenCount; // tokenCount is specific to the current head
            const size_t active_head_elements = static_cast<size_t>(token_count) * token_count;
            const size_t active_head_bytes = active_head_elements * sizeof(float);
            const size_t active_k_q_elements = static_cast<size_t>(token_count) * mat_heights;
            const size_t active_k_q_bytes = active_k_q_elements * sizeof(float);

            // Re-calculate global work sizes for kernels that depend on `token_count`
            auto calculate_global_1d = [&](size_t total_size) {
                return ((total_size + local_work_size_1d - 1) / local_work_size_1d) * local_work_size_1d;
            };
            auto calculate_global_2d = [&](size_t dim0, size_t dim1) {
                size_t global0 = ((dim0 + local_work_size_2d[0] - 1) / local_work_size_2d[0]) * local_work_size_2d[0];
                size_t global1 = ((dim1 + local_work_size_2d[1] - 1) / local_work_size_2d[1]) * local_work_size_2d[1];
                return cl::NDRange(global0, global1);
            };

            // Global work sizes for kernels
            cl::NDRange global_embed(calculate_global_1d(embedding_dim)); // For MLP 1D kernels (size embedding_dim)
            cl::NDRange global_matrix_proj(calculate_global_1d(proj_mat_elements)); // For MV, MQ, MK update kernels (size proj_mat_elements)
            cl::NDRange global_mat_heights(calculate_global_1d(mat_heights)); // For PreMV_V (size mat_heights)
            cl::NDRange global_ev(calculate_global_1d(ev_total_elements)); // For full EV updates (size ev_total_elements)
            cl::NDRange global_head_1d(calculate_global_1d(active_head_elements)); // For LOTA 1D kernels (size active_head_elements)

            cl::NDRange global_embed_2d = calculate_global_2d(embedding_dim, embedding_dim); // For MLP 2D weights (embed x embed)
            cl::NDRange global_matrix_2d = calculate_global_2d(mat_heights, embedding_dim); // For MV, MQ, MK matrix ops (height x embedding)
            cl::NDRange global_head_2d = calculate_global_2d(token_count, token_count); // For attention head matrix ops (token x token)
            cl::NDRange global_kq_grad_2d = calculate_global_2d(mat_heights, token_count); // For KQ gradients (height x token)

            // bool is_first_head_in_block = (head_idx == 0 && k == 0); // Condition from original CUDA code

            // --- Data Transfer Host -> Device (Asynchronous) ---
            std::vector<float> flat_expectedV_head = flatten(expectedV_head);
            // Validation (from original CUDA code)
            if (!head_obj.EV.mapped_data || (token_count > 0 && (!head_obj.K.mapped_data || !head_obj.Q.mapped_data || !head_obj.KdotQ.mapped_data)) ||
                !head_obj.MV.mapped_data || !head_obj.MQ.mapped_data || !head_obj.MK.mapped_data) {
                throw std::runtime_error("One or more attention mat objects have null mapped_data for head [" + std::to_string(head_idx) + "][" + std::to_string(k) + "]");
            }

            CL_CHECK(current_stream_cl.enqueueWriteBuffer(device_ptrs_cl.d_expected_v, CL_FALSE, 0, ev_total_bytes, flat_expectedV_head.data()));
            CL_CHECK(current_stream_cl.enqueueWriteBuffer(device_ptrs_cl.d_EV, CL_FALSE, 0, ev_total_bytes, head_obj.EV.mapped_data));
            if (token_count > 0) {
                CL_CHECK(current_stream_cl.enqueueWriteBuffer(device_ptrs_cl.d_KdotQ, CL_FALSE, 0, active_head_bytes, head_obj.KdotQ.mapped_data));
                CL_CHECK(current_stream_cl.enqueueWriteBuffer(device_ptrs_cl.d_K, CL_FALSE, 0, active_k_q_bytes, head_obj.K.mapped_data));
                CL_CHECK(current_stream_cl.enqueueWriteBuffer(device_ptrs_cl.d_Q, CL_FALSE, 0, active_k_q_bytes, head_obj.Q.mapped_data));
            }
            CL_CHECK(current_stream_cl.enqueueWriteBuffer(device_ptrs_cl.d_MV_a, CL_FALSE, 0, proj_mat_bytes, head_obj.MV.mapped_data));
            CL_CHECK(current_stream_cl.enqueueWriteBuffer(device_ptrs_cl.d_MQ_a, CL_FALSE, 0, proj_mat_bytes, head_obj.MQ.mapped_data));
            CL_CHECK(current_stream_cl.enqueueWriteBuffer(device_ptrs_cl.d_MK_a, CL_FALSE, 0, proj_mat_bytes, head_obj.MK.mapped_data));

            // --- Data Transfer Host -> Device (Async ver MLP Internals) ---
            if (head_obj.ver.activations.size() != static_cast<size_t>(num_neuron_layers_mlp) ||
                head_obj.ver.weights.size() != static_cast<size_t>(num_weight_matrices_mlp)) {
                 throw std::runtime_error("MLP host ver vector size mismatch for head [" + std::to_string(head_idx) + "][" + std::to_string(k) + "]");
            }
            for (int l = 0; l < num_neuron_layers_mlp; ++l) {
                if (head_obj.ver.activations[l].empty()) {
                    throw std::runtime_error("MLP ver.activations vector is empty for head [" + std::to_string(head_idx) + "][" + std::to_string(k) + "], layer " + std::to_string(l));
                }
                CL_CHECK(current_stream_cl.enqueueWriteBuffer(device_ptrs_cl.d_ver_activations[l], CL_FALSE, 0, embed_bytes, head_obj.ver.activations[l].data())); // Corrected from head_obj.ver.activations[l].data()
            }
            for (int l = 0; l < num_weight_matrices_mlp; ++l) {
                if (!head_obj.ver.weights[l].mapped_data || head_obj.ver.weights[l].row * head_obj.ver.weights[l].col != mlp_weights_elements) {
                    throw std::runtime_error("Invalid ver.weights/gweights mat for head [" + std::to_string(head_idx) + "][" + std::to_string(k) + "], layer " + std::to_string(l));
                }
                CL_CHECK(current_stream_cl.enqueueWriteBuffer(device_ptrs_cl.d_ver_weights[l], CL_FALSE, 0, mlp_weights_bytes, head_obj.ver.weights[l].mapped_data));
            }
            current_stream_cl.flush(); // Flush commands to the device

            // --- Backpropagation Steps (Asynchronous Kernel Launches) ---

            // Step 1: Compute grad_EV (full, summed, scaled)
            cl::Kernel kernelComputeGradientsEV_V_cl = clcontext.kernels.at("kernelComputeGradientsEV_V");
            CL_CHECK(kernelComputeGradientsEV_V_cl.setArg(0, device_ptrs_cl.d_EV));
            CL_CHECK(kernelComputeGradientsEV_V_cl.setArg(1, device_ptrs_cl.d_expected_v));
            CL_CHECK(kernelComputeGradientsEV_V_cl.setArg(2, device_ptrs_cl.d_grad_EV_full));
            CL_CHECK(kernelComputeGradientsEV_V_cl.setArg(3, device_ptrs_cl.d_grad_EV_summed));
            CL_CHECK(kernelComputeGradientsEV_V_cl.setArg(4, device_ptrs_cl.d_grad_EV_scaled));
            CL_CHECK(kernelComputeGradientsEV_V_cl.setArg(5, learning_rate));
            CL_CHECK(kernelComputeGradientsEV_V_cl.setArg(6, context_win));
            CL_CHECK(kernelComputeGradientsEV_V_cl.setArg(7, embedding_dim));
            CL_CHECK(current_stream_cl.enqueueNDRangeKernel(kernelComputeGradientsEV_V_cl, cl::NullRange, global_ev, local_1d));

            // --- Step 2: Backprop through ver MLP ---
            cl::Kernel kernelLastLayerDelta_cl = clcontext.kernels.at("kernelLastLayerDelta");
            CL_CHECK(kernelLastLayerDelta_cl.setArg(0, device_ptrs_cl.d_grad_EV_scaled));
            CL_CHECK(kernelLastLayerDelta_cl.setArg(1, device_ptrs_cl.d_ver_activations[num_neuron_layers_mlp - 1]));
            CL_CHECK(kernelLastLayerDelta_cl.setArg(2, device_ptrs_cl.d_ver_deltas[num_weight_matrices_mlp - 1]));
            CL_CHECK(kernelLastLayerDelta_cl.setArg(3, embedding_dim));
            CL_CHECK(current_stream_cl.enqueueNDRangeKernel(kernelLastLayerDelta_cl, cl::NullRange, global_embed, local_1d));

            cl::Kernel hiddenDeltaKernel_cl = clcontext.kernels.at("hiddenDeltaKernel");
            for (int l = num_weight_matrices_mlp - 1; l >= 1; --l) {
                CL_CHECK(hiddenDeltaKernel_cl.setArg(0, device_ptrs_cl.d_ver_deltas[l]));
                CL_CHECK(hiddenDeltaKernel_cl.setArg(1, device_ptrs_cl.d_ver_weights[l]));
                CL_CHECK(hiddenDeltaKernel_cl.setArg(2, device_ptrs_cl.d_ver_activations[l]));
                CL_CHECK(hiddenDeltaKernel_cl.setArg(3, device_ptrs_cl.d_ver_deltas[l-1]));
                CL_CHECK(hiddenDeltaKernel_cl.setArg(4, embedding_dim));
                CL_CHECK(hiddenDeltaKernel_cl.setArg(5, embedding_dim));
                CL_CHECK(current_stream_cl.enqueueNDRangeKernel(hiddenDeltaKernel_cl, cl::NullRange, global_embed, local_1d));
            }
            cl::Kernel k_update_weights_v = clcontext.kernels.at("kernelUpdateWeightsL2");
            k_update_weights_v = clcontext.kernels.at("kernelUpdateElasticNet"); // Changed to ElasticNet
            for (int l = 0; l < num_weight_matrices_mlp; ++l) {
                CL_CHECK(k_update_weights_v.setArg(0, device_ptrs_cl.d_ver_deltas[l])); // deltas
                CL_CHECK(k_update_weights_v.setArg(1, device_ptrs_cl.d_ver_activations[l])); // prev_activations
                CL_CHECK(k_update_weights_v.setArg(2, device_ptrs_cl.d_ver_weights[l])); // weights
                CL_CHECK(k_update_weights_v.setArg(3, device_ptrs_cl.d_ver_gweights[l])); // gweights
                CL_CHECK(k_update_weights_v.setArg(4, learning_rate)); // learning_rate
                CL_CHECK(k_update_weights_v.setArg(5, lambda_l1)); // lambda_l1
                CL_CHECK(k_update_weights_v.setArg(6, lambda_l2)); // lambda_l2
                CL_CHECK(k_update_weights_v.setArg(7, embedding_dim)); // current_layer_size
                CL_CHECK(k_update_weights_v.setArg(8, embedding_dim)); // prev_layer_size
                CL_CHECK(current_stream_cl.enqueueNDRangeKernel(k_update_weights_v, cl::NullRange, global_embed_2d, local_2d));
            }

            // --- Step 3: Compute grad_dv ---
            cl::Kernel kernelComputeGradMLPInput_cl = clcontext.kernels.at("kernelComputeGradMLPInput");
            CL_CHECK(kernelComputeGradMLPInput_cl.setArg(0, device_ptrs_cl.d_ver_deltas[0]));
            CL_CHECK(kernelComputeGradMLPInput_cl.setArg(1, device_ptrs_cl.d_ver_weights[0]));
            CL_CHECK(kernelComputeGradMLPInput_cl.setArg(2, device_ptrs_cl.d_grad_dv));
            CL_CHECK(kernelComputeGradMLPInput_cl.setArg(3, embedding_dim));
            CL_CHECK(kernelComputeGradMLPInput_cl.setArg(4, embedding_dim));
            CL_CHECK(current_stream_cl.enqueueNDRangeKernel(kernelComputeGradMLPInput_cl, cl::NullRange, global_embed, local_1d));

            if (token_count > 0) {
                // --- Step 4: Compute grad_MV ---
                cl::Kernel cuLOTA_cl = clcontext.kernels.at("clLOTA2dmasking");
                CL_CHECK(cuLOTA_cl.setArg(0, device_ptrs_cl.d_KdotQ));
                CL_CHECK(cuLOTA_cl.setArg(1, device_ptrs_cl.d_head));
                CL_CHECK(cuLOTA_cl.setArg(2, context_win)); // M (rows)
                CL_CHECK(cuLOTA_cl.setArg(3, context_win)); // N (cols)
                CL_CHECK(cuLOTA_cl.setArg(4, token_count)); // active_dim
                cl_int cl_att_is_self_lota_der = attention_type ? 1 : 0;
                CL_CHECK(cuLOTA_cl.setArg(5, cl_att_is_self_lota_der));
                CL_CHECK(current_stream_cl.enqueueNDRangeKernel(cuLOTA_cl, cl::NullRange, global_head_2d, local_2d));

                cl::Kernel kernelComputePreMV_V_cl = clcontext.kernels.at("kernelComputePreMV_V");
                CL_CHECK(kernelComputePreMV_V_cl.setArg(0, device_ptrs_cl.d_head));
                CL_CHECK(kernelComputePreMV_V_cl.setArg(1, device_ptrs_cl.d_Q));
                CL_CHECK(kernelComputePreMV_V_cl.setArg(2, device_ptrs_cl.d_pre_MV));
                CL_CHECK(kernelComputePreMV_V_cl.setArg(3, token_count));
                CL_CHECK(kernelComputePreMV_V_cl.setArg(4, mat_heights));
                CL_CHECK(current_stream_cl.enqueueNDRangeKernel(kernelComputePreMV_V_cl, cl::NullRange, global_mat_heights, local_1d));

                cl::Kernel kernelComputeGradMV_V_cl = clcontext.kernels.at("kernelComputeGradMV_V");
                CL_CHECK(kernelComputeGradMV_V_cl.setArg(0, device_ptrs_cl.d_pre_MV));
                CL_CHECK(kernelComputeGradMV_V_cl.setArg(1, device_ptrs_cl.d_grad_dv));
                CL_CHECK(kernelComputeGradMV_V_cl.setArg(2, device_ptrs_cl.d_grad_MV));
                CL_CHECK(kernelComputeGradMV_V_cl.setArg(3, mat_heights));
                CL_CHECK(kernelComputeGradMV_V_cl.setArg(4, embedding_dim));
                CL_CHECK(current_stream_cl.enqueueNDRangeKernel(kernelComputeGradMV_V_cl, cl::NullRange, global_matrix_2d, local_2d));

                // --- Step 5: Compute grad_head (V path only) ---
                cl::Kernel kernelComputeGradHead_V_cl = clcontext.kernels.at("kernelComputeGradHead_V");
                CL_CHECK(kernelComputeGradHead_V_cl.setArg(0, device_ptrs_cl.d_Q));
                CL_CHECK(kernelComputeGradHead_V_cl.setArg(1, device_ptrs_cl.d_MV_a));
                CL_CHECK(kernelComputeGradHead_V_cl.setArg(2, device_ptrs_cl.d_grad_dv));
                CL_CHECK(kernelComputeGradHead_V_cl.setArg(3, device_ptrs_cl.d_grad_head));
                CL_CHECK(kernelComputeGradHead_V_cl.setArg(4, token_count));
                CL_CHECK(kernelComputeGradHead_V_cl.setArg(5, mat_heights));
                CL_CHECK(kernelComputeGradHead_V_cl.setArg(6, embedding_dim));
                CL_CHECK(current_stream_cl.enqueueNDRangeKernel(kernelComputeGradHead_V_cl, cl::NullRange, global_head_2d, local_2d));

                // --- Step 6: Backprop through LOTA ---
                cl::Kernel cuLOTAder_cl = clcontext.kernels.at("clLOTA2ddermasking");
                CL_CHECK(cuLOTAder_cl.setArg(0, device_ptrs_cl.d_KdotQ));
                CL_CHECK(cuLOTAder_cl.setArg(1, device_ptrs_cl.d_lota_deriv));
                CL_CHECK(cuLOTAder_cl.setArg(2, context_win)); // M (rows)
                CL_CHECK(cuLOTAder_cl.setArg(3, context_win)); // N (cols)
                CL_CHECK(cuLOTAder_cl.setArg(4, token_count)); // active_dim
                // cl_int cl_att_is_self_lota_der = attention_type ? 1 : 0;
                CL_CHECK(cuLOTAder_cl.setArg(5, cl_att_is_self_lota_der));
                CL_CHECK(current_stream_cl.enqueueNDRangeKernel(cuLOTAder_cl, cl::NullRange, global_head_2d, local_2d));

                cl::Kernel kernelComputeGradKdotQ_LOTA_cl = clcontext.kernels.at("kernelComputeGradKdotQ_LOTA");
                CL_CHECK(kernelComputeGradKdotQ_LOTA_cl.setArg(0, device_ptrs_cl.d_grad_head));
                CL_CHECK(kernelComputeGradKdotQ_LOTA_cl.setArg(1, device_ptrs_cl.d_lota_deriv));
                CL_CHECK(kernelComputeGradKdotQ_LOTA_cl.setArg(2, device_ptrs_cl.d_grad_KdotQ));
                CL_CHECK(kernelComputeGradKdotQ_LOTA_cl.setArg(3, scaling_factor));
                CL_CHECK(kernelComputeGradKdotQ_LOTA_cl.setArg(4, static_cast<int>(active_head_elements)));
                CL_CHECK(current_stream_cl.enqueueNDRangeKernel(kernelComputeGradKdotQ_LOTA_cl, cl::NullRange, global_head_1d, local_1d));

                // --- Step 7: Compute grad_Q ---
                cl::Kernel kernelComputeGradQ_V_cl = clcontext.kernels.at("kernelComputeGradQ_V");
                CL_CHECK(kernelComputeGradQ_V_cl.setArg(0, device_ptrs_cl.d_grad_KdotQ));
                CL_CHECK(kernelComputeGradQ_V_cl.setArg(1, device_ptrs_cl.d_K));
                CL_CHECK(kernelComputeGradQ_V_cl.setArg(2, device_ptrs_cl.d_grad_Q));
                CL_CHECK(kernelComputeGradQ_V_cl.setArg(3, token_count));
                CL_CHECK(kernelComputeGradQ_V_cl.setArg(4, mat_heights));
                CL_CHECK(current_stream_cl.enqueueNDRangeKernel(kernelComputeGradQ_V_cl, cl::NullRange, global_kq_grad_2d, local_2d));

                // --- Step 8: Compute grad_MQ and grad_MK_correction ---
                cl::Kernel kernelComputeGradMQ_V_cl = clcontext.kernels.at("kernelComputeGradMQ_V");
                // For the `nullptr` argument in CUDA, use a 0-sized `cl::Buffer` in OpenCL
                cl::Buffer d_null_buffer(clcontext.context, 0, 0, nullptr, &cl_err);
                if (cl_err != CL_SUCCESS && cl_err != CL_INVALID_BUFFER_SIZE) CL_CHECK(cl_err);
                CL_CHECK(kernelComputeGradMQ_V_cl.setArg(0, device_ptrs_cl.d_grad_Q));
                CL_CHECK(kernelComputeGradMQ_V_cl.setArg(1, d_null_buffer)); // d_Q_embed (nullptr equivalent)
                CL_CHECK(kernelComputeGradMQ_V_cl.setArg(2, device_ptrs_cl.d_grad_MQ));
                CL_CHECK(kernelComputeGradMQ_V_cl.setArg(3, token_count));
                CL_CHECK(kernelComputeGradMQ_V_cl.setArg(4, mat_heights));
                CL_CHECK(kernelComputeGradMQ_V_cl.setArg(5, embedding_dim));
                CL_CHECK(current_stream_cl.enqueueNDRangeKernel(kernelComputeGradMQ_V_cl, cl::NullRange, global_matrix_2d, local_2d));

                cl::Kernel kernelComputeGradMKCorrection_cl = clcontext.kernels.at("kernelComputeGradMKCorrection");
                CL_CHECK(kernelComputeGradMKCorrection_cl.setArg(0, device_ptrs_cl.d_grad_MQ));
                CL_CHECK(kernelComputeGradMKCorrection_cl.setArg(1, device_ptrs_cl.d_Q));
                CL_CHECK(kernelComputeGradMKCorrection_cl.setArg(2, device_ptrs_cl.d_K));
                CL_CHECK(kernelComputeGradMKCorrection_cl.setArg(3, device_ptrs_cl.d_grad_MK_correction));
                CL_CHECK(kernelComputeGradMKCorrection_cl.setArg(4, token_count));
                CL_CHECK(kernelComputeGradMKCorrection_cl.setArg(5, mat_heights));
                CL_CHECK(kernelComputeGradMKCorrection_cl.setArg(6, embedding_dim));
                CL_CHECK(current_stream_cl.enqueueNDRangeKernel(kernelComputeGradMKCorrection_cl, cl::NullRange, global_matrix_2d, local_2d));
            }
            else {
                // If token_count is 0, CUDA memsets. OpenCL `enqueueFillBuffer` is equivalent.
                CL_CHECK(current_stream_cl.enqueueFillBuffer(device_ptrs_cl.d_grad_MV, 0.0f, 0, proj_mat_bytes));
                CL_CHECK(current_stream_cl.enqueueFillBuffer(device_ptrs_cl.d_grad_MQ, 0.0f, 0, proj_mat_bytes));
                CL_CHECK(current_stream_cl.enqueueFillBuffer(device_ptrs_cl.d_grad_MK_correction, 0.0f, 0, proj_mat_bytes));
            }

            // --- Step 9 & 10: Update Weights ---
            cl::Kernel kernelUpdateWeights_1stHead_V_cl = clcontext.kernels.at("kernelUpdateWeights_1stHead_V");
            cl::Kernel kernelUpdateSimple_cl = clcontext.kernels.at("kernelUpdateSimple");

            // Combined update for MV, MQ, MK for the very first head
            CL_CHECK(kernelUpdateWeights_1stHead_V_cl.setArg(0, device_ptrs_cl.d_MV_a));
            CL_CHECK(kernelUpdateWeights_1stHead_V_cl.setArg(1, device_ptrs_cl.d_MQ_a));
            CL_CHECK(kernelUpdateWeights_1stHead_V_cl.setArg(2, device_ptrs_cl.d_MK_a));
            CL_CHECK(kernelUpdateWeights_1stHead_V_cl.setArg(3, device_ptrs_cl.d_grad_MV));
            CL_CHECK(kernelUpdateWeights_1stHead_V_cl.setArg(4, device_ptrs_cl.d_grad_MQ));
            CL_CHECK(kernelUpdateWeights_1stHead_V_cl.setArg(5, device_ptrs_cl.d_grad_MK_correction));
            CL_CHECK(kernelUpdateWeights_1stHead_V_cl.setArg(6, learning_rate));
            CL_CHECK(kernelUpdateWeights_1stHead_V_cl.setArg(7, mat_heights));
            CL_CHECK(kernelUpdateWeights_1stHead_V_cl.setArg(8, embedding_dim));
            CL_CHECK(current_stream_cl.enqueueNDRangeKernel(kernelUpdateWeights_1stHead_V_cl, cl::NullRange, global_matrix_proj, local_1d));

            // --- Data Transfer Device -> Host (Asynchronous) ---
            for (int l = 0; l < num_weight_matrices_mlp; ++l) {
                CL_CHECK(current_stream_cl.enqueueReadBuffer(device_ptrs_cl.d_ver_weights[l], CL_FALSE, 0, mlp_weights_bytes, head_obj.ver.weights[l].mapped_data));
            }
            CL_CHECK(current_stream_cl.enqueueReadBuffer(device_ptrs_cl.d_EV, CL_FALSE, 0, ev_total_bytes, head_obj.EV.mapped_data));
            CL_CHECK(current_stream_cl.enqueueReadBuffer(device_ptrs_cl.d_MV_a, CL_FALSE, 0, proj_mat_bytes, head_obj.MV.mapped_data));
            CL_CHECK(current_stream_cl.enqueueReadBuffer(device_ptrs_cl.d_MQ_a, CL_FALSE, 0, proj_mat_bytes, head_obj.MQ.mapped_data));
            CL_CHECK(current_stream_cl.enqueueReadBuffer(device_ptrs_cl.d_MK_a, CL_FALSE, 0, proj_mat_bytes, head_obj.MK.mapped_data));
            CL_CHECK(current_stream_cl.finish());
        } // End loop over heads

        // --- Synchronize all OpenCL command queues ---
        // This ensures all device operations launched asynchronously are completed before function returns.
        // Explicitly release all aggregate OpenCL buffers before final sync
        agg_d_expected_v_buf = cl::Buffer();
        agg_d_EV_buf = cl::Buffer();
        agg_d_grad_EV_full_buf = cl::Buffer();
        agg_d_grad_EV_summed_buf = cl::Buffer();
        agg_d_grad_EV_scaled_buf = cl::Buffer();
        agg_d_grad_dv_buf = cl::Buffer();
        agg_d_KdotQ_buf = cl::Buffer();
        agg_d_head_storage_buf = cl::Buffer();
        agg_d_K_buf = cl::Buffer();
        agg_d_Q_buf = cl::Buffer();
        agg_d_pre_MV_buf = cl::Buffer();
        agg_d_MV_a_buf = cl::Buffer();
        agg_d_MQ_a_buf = cl::Buffer();
        agg_d_MK_a_buf = cl::Buffer();
        agg_d_grad_MV_buf = cl::Buffer();
        agg_d_grad_head_storage_buf = cl::Buffer();
        agg_d_lota_deriv_buf = cl::Buffer();
        agg_d_grad_KdotQ_buf = cl::Buffer();
        agg_d_grad_Q_storage_buf = cl::Buffer();
        agg_d_grad_MQ_buf = cl::Buffer();
        agg_d_grad_MK_correction_buf = cl::Buffer();
        agg_d_ver_activations_storage_buf = cl::Buffer();
        agg_d_ver_gweights_storage_buf = cl::Buffer(); // Added
        agg_d_ver_weights_storage_buf = cl::Buffer();
        agg_d_ver_deltas_storage_buf = cl::Buffer();

        for (int head_idx = 0; head_idx < num_heads_to_process; ++head_idx) {
            CL_CHECK(streams_cl[head_idx].finish());
        }
    } // End try block
      catch (const std::exception& e) {
        std::cerr << "Standard Exception during clpartialbackward1stBlock (V) for column " << k << ": " << e.what() << std::endl;
        throw std::runtime_error("Exception during clpartialbackward1stBlock (V) for column " + std::to_string(k) + ": " + e.what());
    }

    // No explicit cleanup calls like cudaFree/cudaStreamDestroy are needed here
    // because cl::Buffer and cl::CommandQueue objects manage their resources via RAII.
}


/**
 * @brief backpropagation via vertical retention vectors for ith block
 * @param expectedV vertical retention vectors for each head of column
 * @param in embedding dimension and mlp input-output vector dimension
 * @param layers number of mlp activations layers
 * @param k column number
 * @param blocknumber current block position (1-based index)
 */
void block::clpartialbackward(std::vector<std::vector<std::vector<float>>>& expectedV, int& in, int& layers, int& k_col_idx, int& blocknumber_param, float& learning)
{
    cl_int cl_err; // For OpenCL error codes
    const int num_heads_to_process = x; // 'x' is the number of rows/heads in this column

    // Validate column number and input shape
    if (k_col_idx < 0 || k_col_idx >= y) {
        throw std::out_of_range("clpartialbackward(V): Column index 'k_col_idx' (" + std::to_string(k_col_idx) + ") is out of range [0, " + std::to_string(y - 1) + "].");
    }
    if (expectedV.size() != static_cast<size_t>(x)) {
        throw std::runtime_error("ExpectedV size mismatch (number of rows) in clpartialbackward(V). Expected " + std::to_string(x) + ", got " + std::to_string(expectedV.size()));
    }
    if (!expectedV.empty() && (expectedV[0].size() != CONTEXT_WIN || (!expectedV[0].empty() && expectedV[0][0].size() != EMBEDDING))) {
        throw std::runtime_error("ExpectedV dimensions mismatch (context/embedding) in clpartialbackward(V).");
    }
    if (EMBEDDING != in) {
        throw std::runtime_error("Embedding dimension mismatch: EMBEDDING vs in");
    }

    // Constants
    const int embedding_dim = EMBEDDING;
    const int mat_heights = MATHEIGHTS;
    const int context_win = CONTEXT_WIN;
    const float learning_rate = learning;
    const float scaling_factor = SCALING;
    const float lambda_l1 = 0.001f; // L1 regularization parameter
    const float lambda_l2 = 0.001f; // L2 regularization parameter

    // MLP structure parameters
    const int num_total_layers_mlp = layers;
    const int num_neuron_layers_mlp = num_total_layers_mlp;
    const int num_weight_matrices_mlp = num_total_layers_mlp - 1;

    const size_t embed_bytes = embedding_dim * sizeof(float);
    const size_t mlp_weights_elements = static_cast<size_t>(embedding_dim) * embedding_dim;
    const size_t mlp_weights_bytes = mlp_weights_elements * sizeof(float);
    const size_t proj_mat_elements = static_cast<size_t>(mat_heights) * embedding_dim;
    const size_t proj_mat_bytes = proj_mat_elements * sizeof(float);
    const size_t ev_total_elements = static_cast<size_t>(context_win) * embedding_dim;
    const size_t ev_total_bytes = ev_total_elements * sizeof(float);
    const size_t max_head_elements_per_head = static_cast<size_t>(context_win) * context_win;
    const size_t max_head_bytes_per_head = max_head_elements_per_head * sizeof(float);
    const size_t max_k_q_elements_per_head = static_cast<size_t>(context_win) * mat_heights;
    const size_t max_k_q_bytes_per_head = max_k_q_elements_per_head * sizeof(float);
    const size_t pre_mv_elements_per_head = mat_heights;
    const size_t pre_mv_bytes_per_head = pre_mv_elements_per_head * sizeof(float);

    const size_t local_work_size_1d = 256;
    cl::NDRange local_1d(local_work_size_1d);
    const size_t local_work_size_2d[2] = { 16, 16 };
    cl::NDRange local_2d(local_work_size_2d[0], local_work_size_2d[1]);

    cl::Buffer agg_d_expected_v_buf, agg_d_EV_buf;
    cl::Buffer agg_d_grad_EV_full_buf, agg_d_grad_EV_summed_buf, agg_d_grad_EV_scaled_buf;
    cl::Buffer agg_d_grad_dv_buf;
    cl::Buffer agg_d_KdotQ_buf, agg_d_head_storage_buf;
    cl::Buffer agg_d_K_buf, agg_d_Q_buf;
    cl::Buffer agg_d_pre_MV_buf;
    cl::Buffer agg_d_MV_a_buf, agg_d_MQ_a_buf, agg_d_MK_a_buf;
    cl::Buffer agg_d_grad_MV_buf;
    cl::Buffer agg_d_grad_head_storage_buf;
    cl::Buffer agg_d_lota_deriv_buf;
    cl::Buffer agg_d_grad_KdotQ_buf;
    cl::Buffer agg_d_grad_Q_storage_buf;
    cl::Buffer agg_d_grad_MQ_buf, agg_d_grad_MK_correction_buf;
    cl::Buffer agg_d_ver_activations_storage_buf;
    cl::Buffer agg_d_ver_weights_storage_buf;
    cl::Buffer agg_d_ver_deltas_storage_buf;

    std::vector<cl::CommandQueue> streams_cl(num_heads_to_process);
    std::vector<HeadDeviceSubBuffersV> head_gpu_data_cl(num_heads_to_process);

    try {
        OpenCLContext& clcontext = this->clcontext;
        agg_d_expected_v_buf = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * ev_total_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_EV_buf = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * ev_total_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        // ... (all other aggregate buffer allocations, same as clpartialbackward1stBlock)
        agg_d_grad_EV_full_buf = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * ev_total_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_grad_EV_summed_buf = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_grad_EV_scaled_buf = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_grad_dv_buf = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_KdotQ_buf = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * max_head_bytes_per_head, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_head_storage_buf = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * max_head_bytes_per_head, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_K_buf = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * max_k_q_bytes_per_head, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_Q_buf = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * max_k_q_bytes_per_head, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_pre_MV_buf = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * pre_mv_bytes_per_head, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_MV_a_buf = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * proj_mat_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_MQ_a_buf = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * proj_mat_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_MK_a_buf = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * proj_mat_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_grad_MV_buf = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * proj_mat_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_grad_head_storage_buf = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * max_head_bytes_per_head, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_lota_deriv_buf = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * max_head_bytes_per_head, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_grad_KdotQ_buf = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * max_head_bytes_per_head, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_grad_Q_storage_buf = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * max_k_q_bytes_per_head, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_grad_MQ_buf = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * proj_mat_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_grad_MK_correction_buf = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * proj_mat_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_ver_activations_storage_buf = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * num_neuron_layers_mlp * embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_ver_weights_storage_buf = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * num_weight_matrices_mlp * mlp_weights_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_ver_deltas_storage_buf = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, num_heads_to_process * num_weight_matrices_mlp * embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);


        for (int head_idx = 0; head_idx < num_heads_to_process; ++head_idx) {
            streams_cl[head_idx] = cl::CommandQueue(clcontext.context, clcontext.device, 0, &cl_err); CL_CHECK(cl_err);
            head_gpu_data_cl[head_idx].d_ver_activations.resize(num_neuron_layers_mlp);
            head_gpu_data_cl[head_idx].d_ver_weights.resize(num_weight_matrices_mlp);
            head_gpu_data_cl[head_idx].d_ver_gweights.resize(num_weight_matrices_mlp); // Added
            head_gpu_data_cl[head_idx].d_ver_deltas.resize(num_weight_matrices_mlp);

            cl_buffer_region region;
            // ... (all sub-buffer creations, same as clpartialbackward1stBlock)
            region = { head_idx * ev_total_bytes, ev_total_bytes };
            head_gpu_data_cl[head_idx].d_expected_v = agg_d_expected_v_buf.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
            head_gpu_data_cl[head_idx].d_EV = agg_d_EV_buf.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
            head_gpu_data_cl[head_idx].d_grad_EV_full = agg_d_grad_EV_full_buf.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);

            region = { head_idx * embed_bytes, embed_bytes };
            head_gpu_data_cl[head_idx].d_grad_EV_summed = agg_d_grad_EV_summed_buf.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
            head_gpu_data_cl[head_idx].d_grad_EV_scaled = agg_d_grad_EV_scaled_buf.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
            head_gpu_data_cl[head_idx].d_grad_dv = agg_d_grad_dv_buf.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);

            region = { head_idx * max_head_bytes_per_head, max_head_bytes_per_head };
            head_gpu_data_cl[head_idx].d_KdotQ = agg_d_KdotQ_buf.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
            head_gpu_data_cl[head_idx].d_head = agg_d_head_storage_buf.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
            head_gpu_data_cl[head_idx].d_grad_head = agg_d_grad_head_storage_buf.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
            head_gpu_data_cl[head_idx].d_lota_deriv = agg_d_lota_deriv_buf.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
            head_gpu_data_cl[head_idx].d_grad_KdotQ = agg_d_grad_KdotQ_buf.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);

            region = { head_idx * max_k_q_bytes_per_head, max_k_q_bytes_per_head };
            head_gpu_data_cl[head_idx].d_K = agg_d_K_buf.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
            head_gpu_data_cl[head_idx].d_Q = agg_d_Q_buf.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
            head_gpu_data_cl[head_idx].d_grad_Q = agg_d_grad_Q_storage_buf.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);

            region = { head_idx * pre_mv_bytes_per_head, pre_mv_bytes_per_head };
            head_gpu_data_cl[head_idx].d_pre_MV = agg_d_pre_MV_buf.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);

            region = { head_idx * proj_mat_bytes, proj_mat_bytes };
            head_gpu_data_cl[head_idx].d_MV_a = agg_d_MV_a_buf.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
            head_gpu_data_cl[head_idx].d_MQ_a = agg_d_MQ_a_buf.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
            head_gpu_data_cl[head_idx].d_MK_a = agg_d_MK_a_buf.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
            head_gpu_data_cl[head_idx].d_grad_MV = agg_d_grad_MV_buf.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
            head_gpu_data_cl[head_idx].d_grad_MQ = agg_d_grad_MQ_buf.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
            head_gpu_data_cl[head_idx].d_grad_MK_correction = agg_d_grad_MK_correction_buf.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);

            for (int l = 0; l < num_neuron_layers_mlp; ++l) {
                region = { (head_idx * num_neuron_layers_mlp + l) * embed_bytes, embed_bytes };
                head_gpu_data_cl[head_idx].d_ver_activations[l] = agg_d_ver_activations_storage_buf.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
            }
            for (int l = 0; l < num_weight_matrices_mlp; ++l) {
                region = { (head_idx * num_weight_matrices_mlp + l) * mlp_weights_bytes, mlp_weights_bytes };
                head_gpu_data_cl[head_idx].d_ver_weights[l] = agg_d_ver_weights_storage_buf.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
                region = { (head_idx * num_weight_matrices_mlp + l) * mlp_weights_bytes, mlp_weights_bytes }; // Added
                head_gpu_data_cl[head_idx].d_ver_gweights[l] = agg_d_ver_gweights_storage_buf.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err); // Added
                region = { (head_idx * num_weight_matrices_mlp + l) * embed_bytes, embed_bytes }; // Assuming deltas are vector of size embed_bytes
                head_gpu_data_cl[head_idx].d_ver_deltas[l] = agg_d_ver_deltas_storage_buf.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
            }
        }

        for (int head_idx = 0; head_idx < num_heads_to_process; ++head_idx) {
            attention& head_obj = b[head_idx][k_col_idx];
            bool attention_type = head_obj.isSelfAttention;
            std::vector<std::vector<float>>& expectedV_head = expectedV[head_idx];
            HeadDeviceSubBuffersV& device_ptrs_cl = head_gpu_data_cl[head_idx];
            cl::CommandQueue& current_stream_cl = streams_cl[head_idx];

            const int token_count = head_obj.tokenCount;
            const size_t active_head_elements = static_cast<size_t>(token_count) * token_count;
            const size_t active_head_bytes = active_head_elements * sizeof(float);
            const size_t active_k_q_elements = static_cast<size_t>(token_count) * mat_heights;
            const size_t active_k_q_bytes = active_k_q_elements * sizeof(float);

            auto calculate_global_1d = [&](size_t total_size) { return ((total_size + local_work_size_1d - 1) / local_work_size_1d) * local_work_size_1d; };
            auto calculate_global_2d = [&](size_t dim0, size_t dim1) { size_t g0 = ((dim0 + local_work_size_2d[0] - 1) / local_work_size_2d[0]) * local_work_size_2d[0]; size_t g1 = ((dim1 + local_work_size_2d[1] - 1) / local_work_size_2d[1]) * local_work_size_2d[1]; return cl::NDRange(g0, g1); };

            cl::NDRange global_embed(calculate_global_1d(embedding_dim));
            cl::NDRange global_matrix_proj(calculate_global_1d(proj_mat_elements));
            cl::NDRange global_mat_heights(calculate_global_1d(mat_heights));
            cl::NDRange global_ev(calculate_global_1d(ev_total_elements));
            cl::NDRange global_head_1d(calculate_global_1d(active_head_elements));
            cl::NDRange global_embed_2d = calculate_global_2d(embedding_dim, embedding_dim);
            cl::NDRange global_matrix_2d = calculate_global_2d(mat_heights, embedding_dim);
            cl::NDRange global_head_2d = calculate_global_2d(token_count, token_count);
            cl::NDRange global_kq_grad_2d = calculate_global_2d(mat_heights, token_count);

            std::vector<float> flat_expectedV_head = flatten(expectedV_head);
            if (!head_obj.EV.mapped_data || (token_count > 0 && (!head_obj.K.mapped_data || !head_obj.Q.mapped_data || !head_obj.KdotQ.mapped_data)) || !head_obj.MV.mapped_data || !head_obj.MQ.mapped_data || !head_obj.MK.mapped_data) {
                throw std::runtime_error("Null mapped_data for head [" + std::to_string(head_idx) + "][" + std::to_string(k_col_idx) + "]");
            }

            CL_CHECK(current_stream_cl.enqueueWriteBuffer(device_ptrs_cl.d_expected_v, CL_FALSE, 0, ev_total_bytes, flat_expectedV_head.data()));
            // ... (all other H->D transfers, same as clpartialbackward1stBlock)
            CL_CHECK(current_stream_cl.enqueueWriteBuffer(device_ptrs_cl.d_EV, CL_FALSE, 0, ev_total_bytes, head_obj.EV.mapped_data));
            if (token_count > 0) { CL_CHECK(current_stream_cl.enqueueWriteBuffer(device_ptrs_cl.d_KdotQ, CL_FALSE, 0, active_head_bytes, head_obj.KdotQ.mapped_data)); CL_CHECK(current_stream_cl.enqueueWriteBuffer(device_ptrs_cl.d_K, CL_FALSE, 0, active_k_q_bytes, head_obj.K.mapped_data)); CL_CHECK(current_stream_cl.enqueueWriteBuffer(device_ptrs_cl.d_Q, CL_FALSE, 0, active_k_q_bytes, head_obj.Q.mapped_data));}
            CL_CHECK(current_stream_cl.enqueueWriteBuffer(device_ptrs_cl.d_MV_a, CL_FALSE, 0, proj_mat_bytes, head_obj.MV.mapped_data)); CL_CHECK(current_stream_cl.enqueueWriteBuffer(device_ptrs_cl.d_MQ_a, CL_FALSE, 0, proj_mat_bytes, head_obj.MQ.mapped_data)); CL_CHECK(current_stream_cl.enqueueWriteBuffer(device_ptrs_cl.d_MK_a, CL_FALSE, 0, proj_mat_bytes, head_obj.MK.mapped_data));
            if (head_obj.ver.activations.size()!=static_cast<size_t>(num_neuron_layers_mlp) || head_obj.ver.weights.size()!=static_cast<size_t>(num_weight_matrices_mlp)) { throw std::runtime_error("MLP host ver vector size mismatch for head [" + std::to_string(head_idx) + "][" + std::to_string(k_col_idx) + "]"); }
            for (int l=0; l<num_neuron_layers_mlp; ++l) { 
                if(head_obj.ver.activations[l].empty()) { 
                    throw std::runtime_error("MLP ver.activations empty for head [" + std::to_string(head_idx) + "][" + std::to_string(k_col_idx) + "], layer " + std::to_string(l)); 
                }
                CL_CHECK(current_stream_cl.enqueueWriteBuffer(device_ptrs_cl.d_ver_activations[l], CL_FALSE, 0, embed_bytes, head_obj.ver.activations[l].data())); 
            }
            for (int l=0; l<num_weight_matrices_mlp; ++l) {
                if(!head_obj.ver.weights[l].mapped_data || head_obj.ver.weights[l].row*head_obj.ver.weights[l].col!=mlp_weights_elements) { 
                    throw std::runtime_error("Invalid ver.weights/gweights mat for head [" + std::to_string(head_idx) + "][" + std::to_string(k_col_idx) + "], layer " + std::to_string(l)); 
                } 
                CL_CHECK(current_stream_cl.enqueueWriteBuffer(device_ptrs_cl.d_ver_weights[l], CL_FALSE, 0, mlp_weights_bytes, head_obj.ver.weights[l].mapped_data)); 
            }
            current_stream_cl.flush();

            // --- Kernel Launches for steps 1-8 (same as clpartialbackward1stBlock) ---
            cl::Kernel k_grad_ev_v = clcontext.kernels.at("kernelComputeGradientsEV_V"); CL_CHECK(k_grad_ev_v.setArg(0, device_ptrs_cl.d_EV)); CL_CHECK(k_grad_ev_v.setArg(1, device_ptrs_cl.d_expected_v)); CL_CHECK(k_grad_ev_v.setArg(2, device_ptrs_cl.d_grad_EV_full)); CL_CHECK(k_grad_ev_v.setArg(3, device_ptrs_cl.d_grad_EV_summed)); CL_CHECK(k_grad_ev_v.setArg(4, device_ptrs_cl.d_grad_EV_scaled)); CL_CHECK(k_grad_ev_v.setArg(5, learning_rate)); CL_CHECK(k_grad_ev_v.setArg(6, context_win)); CL_CHECK(k_grad_ev_v.setArg(7, embedding_dim)); CL_CHECK(current_stream_cl.enqueueNDRangeKernel(k_grad_ev_v, cl::NullRange, global_ev, local_1d));
            cl::Kernel k_last_delta = clcontext.kernels.at("kernelLastLayerDelta"); CL_CHECK(k_last_delta.setArg(0, device_ptrs_cl.d_grad_EV_scaled)); CL_CHECK(k_last_delta.setArg(1, device_ptrs_cl.d_ver_activations[num_neuron_layers_mlp - 1])); CL_CHECK(k_last_delta.setArg(2, device_ptrs_cl.d_ver_deltas[num_weight_matrices_mlp - 1])); CL_CHECK(k_last_delta.setArg(3, embedding_dim)); CL_CHECK(current_stream_cl.enqueueNDRangeKernel(k_last_delta, cl::NullRange, global_embed, local_1d));
            cl::Kernel k_hidden_delta = clcontext.kernels.at("hiddenDeltaKernel"); 
            for (int l = num_weight_matrices_mlp - 1; l >= 1; --l) { 
                CL_CHECK(k_hidden_delta.setArg(0, device_ptrs_cl.d_ver_deltas[l])); 
                CL_CHECK(k_hidden_delta.setArg(1, device_ptrs_cl.d_ver_weights[l])); 
                CL_CHECK(k_hidden_delta.setArg(2, device_ptrs_cl.d_ver_activations[l])); 
                CL_CHECK(k_hidden_delta.setArg(3, device_ptrs_cl.d_ver_deltas[l-1])); 
                CL_CHECK(k_hidden_delta.setArg(4, embedding_dim)); 
                CL_CHECK(k_hidden_delta.setArg(5, embedding_dim)); 
                CL_CHECK(current_stream_cl.enqueueNDRangeKernel(k_hidden_delta, cl::NullRange, global_embed, local_1d)); 
            }
            cl::Kernel k_update_weights_v = clcontext.kernels.at("kernelUpdateElasticNet"); // Changed to ElasticNet
            for (int l = 0; l < num_weight_matrices_mlp; ++l) {
                CL_CHECK(k_update_weights_v.setArg(0, device_ptrs_cl.d_ver_deltas[l])); // deltas
                CL_CHECK(k_update_weights_v.setArg(1, device_ptrs_cl.d_ver_activations[l])); // prev_activations
                CL_CHECK(k_update_weights_v.setArg(2, device_ptrs_cl.d_ver_weights[l])); // weights
                CL_CHECK(k_update_weights_v.setArg(3, device_ptrs_cl.d_ver_gweights[l])); // gweights
                CL_CHECK(k_update_weights_v.setArg(4, learning_rate)); // learning_rate
                CL_CHECK(k_update_weights_v.setArg(5, lambda_l1)); // lambda_l1
                CL_CHECK(k_update_weights_v.setArg(6, lambda_l2)); // lambda_l2
                CL_CHECK(k_update_weights_v.setArg(7, embedding_dim)); // current_layer_size
                CL_CHECK(k_update_weights_v.setArg(8, embedding_dim)); // prev_layer_size
                CL_CHECK(current_stream_cl.enqueueNDRangeKernel(k_update_weights_v, cl::NullRange, global_embed_2d, local_2d));
            }
            cl::Kernel k_grad_mlp_input = clcontext.kernels.at("kernelComputeGradMLPInput"); CL_CHECK(k_grad_mlp_input.setArg(0, device_ptrs_cl.d_ver_deltas[0])); CL_CHECK(k_grad_mlp_input.setArg(1, device_ptrs_cl.d_ver_weights[0])); CL_CHECK(k_grad_mlp_input.setArg(2, device_ptrs_cl.d_grad_dv)); CL_CHECK(k_grad_mlp_input.setArg(3, embedding_dim)); CL_CHECK(k_grad_mlp_input.setArg(4, embedding_dim)); CL_CHECK(current_stream_cl.enqueueNDRangeKernel(k_grad_mlp_input, cl::NullRange, global_embed, local_1d));

            if (token_count > 0) {
                cl::Kernel k_lota = clcontext.kernels.at("clLOTA2dmasking"); 
                CL_CHECK(k_lota.setArg(0, device_ptrs_cl.d_KdotQ));
                CL_CHECK(k_lota.setArg(1, device_ptrs_cl.d_head));
                CL_CHECK(k_lota.setArg(2, context_win));
                CL_CHECK(k_lota.setArg(3, context_win));
                CL_CHECK(k_lota.setArg(4, token_count));
                cl_int cl_att_is_self_lota_der = attention_type ? 1 : 0;
                CL_CHECK(k_lota.setArg(5, cl_att_is_self_lota_der));
                CL_CHECK(current_stream_cl.enqueueNDRangeKernel(k_lota, cl::NullRange, global_head_2d, local_2d));
                
                cl::Kernel k_pre_mv_v = clcontext.kernels.at("kernelComputePreMV_V"); 
                CL_CHECK(k_pre_mv_v.setArg(0, device_ptrs_cl.d_head));
                CL_CHECK(k_pre_mv_v.setArg(1, device_ptrs_cl.d_Q));
                CL_CHECK(k_pre_mv_v.setArg(2, device_ptrs_cl.d_pre_MV));
                CL_CHECK(k_pre_mv_v.setArg(3, token_count));
                CL_CHECK(k_pre_mv_v.setArg(4, mat_heights));
                CL_CHECK(current_stream_cl.enqueueNDRangeKernel(k_pre_mv_v, cl::NullRange, global_mat_heights, local_1d));
                
                cl::Kernel k_grad_mv_v = clcontext.kernels.at("kernelComputeGradMV_V"); 
                CL_CHECK(k_grad_mv_v.setArg(0, device_ptrs_cl.d_pre_MV));
                CL_CHECK(k_grad_mv_v.setArg(1, device_ptrs_cl.d_grad_dv));
                CL_CHECK(k_grad_mv_v.setArg(2, device_ptrs_cl.d_grad_MV));
                CL_CHECK(k_grad_mv_v.setArg(3, mat_heights));
                CL_CHECK(k_grad_mv_v.setArg(4, embedding_dim));
                CL_CHECK(current_stream_cl.enqueueNDRangeKernel(k_grad_mv_v, cl::NullRange, global_matrix_2d, local_2d));
                
                cl::Kernel k_grad_head_v = clcontext.kernels.at("kernelComputeGradHead_V"); 
                CL_CHECK(k_grad_head_v.setArg(0, device_ptrs_cl.d_Q));
                CL_CHECK(k_grad_head_v.setArg(1, device_ptrs_cl.d_MV_a));
                CL_CHECK(k_grad_head_v.setArg(2, device_ptrs_cl.d_grad_dv));
                CL_CHECK(k_grad_head_v.setArg(3, device_ptrs_cl.d_grad_head));
                CL_CHECK(k_grad_head_v.setArg(4, token_count));
                CL_CHECK(k_grad_head_v.setArg(5, mat_heights));
                CL_CHECK(k_grad_head_v.setArg(6, embedding_dim));
                CL_CHECK(current_stream_cl.enqueueNDRangeKernel(k_grad_head_v, cl::NullRange, global_head_2d, local_2d));
                
                cl::Kernel k_lota_der = clcontext.kernels.at("clLOTA2ddermasking"); 
                CL_CHECK(k_lota_der.setArg(0, device_ptrs_cl.d_KdotQ));
                CL_CHECK(k_lota_der.setArg(1, device_ptrs_cl.d_lota_deriv));
                CL_CHECK(k_lota_der.setArg(2, context_win));
                CL_CHECK(k_lota_der.setArg(3, context_win));
                CL_CHECK(k_lota_der.setArg(4, token_count));
                // cl_int cl_att_is_self_lota_der = attention_type ? 1 : 0;
                CL_CHECK(k_lota_der.setArg(5, cl_att_is_self_lota_der));
                CL_CHECK(current_stream_cl.enqueueNDRangeKernel(k_lota_der, cl::NullRange, global_head_2d, local_2d));
                
                cl::Kernel k_grad_kdotq_lota = clcontext.kernels.at("kernelComputeGradKdotQ_LOTA"); 
                CL_CHECK(k_grad_kdotq_lota.setArg(0, device_ptrs_cl.d_grad_head));
                CL_CHECK(k_grad_kdotq_lota.setArg(1, device_ptrs_cl.d_lota_deriv));
                CL_CHECK(k_grad_kdotq_lota.setArg(2, device_ptrs_cl.d_grad_KdotQ));
                CL_CHECK(k_grad_kdotq_lota.setArg(3, scaling_factor));
                CL_CHECK(k_grad_kdotq_lota.setArg(4, static_cast<int>(active_head_elements)));
                CL_CHECK(current_stream_cl.enqueueNDRangeKernel(k_grad_kdotq_lota, cl::NullRange, global_head_1d, local_1d));
                
                cl::Kernel k_grad_q_v = clcontext.kernels.at("kernelComputeGradQ_V"); 
                CL_CHECK(k_grad_q_v.setArg(0, device_ptrs_cl.d_grad_KdotQ));
                CL_CHECK(k_grad_q_v.setArg(1, device_ptrs_cl.d_K));
                CL_CHECK(k_grad_q_v.setArg(2, device_ptrs_cl.d_grad_Q));
                CL_CHECK(k_grad_q_v.setArg(3, token_count));
                CL_CHECK(k_grad_q_v.setArg(4, mat_heights));
                CL_CHECK(current_stream_cl.enqueueNDRangeKernel(k_grad_q_v, cl::NullRange, global_kq_grad_2d, local_2d));
                
                cl::Kernel k_grad_mq_v = clcontext.kernels.at("kernelComputeGradMQ_V"); 
                cl::Buffer d_null_buf(clcontext.context,0,0,nullptr,&cl_err); 
                if(cl_err!=CL_SUCCESS && cl_err!=CL_INVALID_BUFFER_SIZE) 
                    CL_CHECK(cl_err); CL_CHECK(k_grad_mq_v.setArg(0, device_ptrs_cl.d_grad_Q));
                CL_CHECK(k_grad_mq_v.setArg(1, d_null_buf));
                CL_CHECK(k_grad_mq_v.setArg(2, device_ptrs_cl.d_grad_MQ));
                CL_CHECK(k_grad_mq_v.setArg(3, token_count));
                CL_CHECK(k_grad_mq_v.setArg(4, mat_heights));
                CL_CHECK(k_grad_mq_v.setArg(5, embedding_dim));
                CL_CHECK(current_stream_cl.enqueueNDRangeKernel(k_grad_mq_v, cl::NullRange, global_matrix_2d, local_2d));
                
                cl::Kernel k_grad_mk_corr = clcontext.kernels.at("kernelComputeGradMKCorrection"); 
                CL_CHECK(k_grad_mk_corr.setArg(0, device_ptrs_cl.d_grad_MQ));
                CL_CHECK(k_grad_mk_corr.setArg(1, device_ptrs_cl.d_Q));
                CL_CHECK(k_grad_mk_corr.setArg(2, device_ptrs_cl.d_K));
                CL_CHECK(k_grad_mk_corr.setArg(3, device_ptrs_cl.d_grad_MK_correction));
                CL_CHECK(k_grad_mk_corr.setArg(4, token_count));
                CL_CHECK(k_grad_mk_corr.setArg(5, mat_heights));
                CL_CHECK(k_grad_mk_corr.setArg(6, embedding_dim));
                CL_CHECK(current_stream_cl.enqueueNDRangeKernel(k_grad_mk_corr, cl::NullRange, global_matrix_2d, local_2d));
            }
            else {
                CL_CHECK(current_stream_cl.enqueueFillBuffer(device_ptrs_cl.d_grad_MV, 0.0f, 0, proj_mat_bytes));
                CL_CHECK(current_stream_cl.enqueueFillBuffer(device_ptrs_cl.d_grad_MQ, 0.0f, 0, proj_mat_bytes));
                CL_CHECK(current_stream_cl.enqueueFillBuffer(device_ptrs_cl.d_grad_MK_correction, 0.0f, 0, proj_mat_bytes));
            }

            // --- Step 9 & 10: Update Weights (MV, MQ, MK_correction) and EV ---
            // For non-first blocks, EV is updated.
            cl::Kernel kernelUpdateWeights_EV_V_cl = clcontext.kernels.at("kernelUpdateWeights_EV_V");
            cl_int cl_update_ev = (blocknumber_param > 1) ? 1 : 0; // Always update EV for non-first blocks
            CL_CHECK(kernelUpdateWeights_EV_V_cl.setArg(0, device_ptrs_cl.d_MV_a));
            CL_CHECK(kernelUpdateWeights_EV_V_cl.setArg(1, device_ptrs_cl.d_MQ_a));
            CL_CHECK(kernelUpdateWeights_EV_V_cl.setArg(2, device_ptrs_cl.d_MK_a));
            CL_CHECK(kernelUpdateWeights_EV_V_cl.setArg(3, device_ptrs_cl.d_EV));
            CL_CHECK(kernelUpdateWeights_EV_V_cl.setArg(4, device_ptrs_cl.d_grad_MV));
            CL_CHECK(kernelUpdateWeights_EV_V_cl.setArg(5, device_ptrs_cl.d_grad_MQ));
            CL_CHECK(kernelUpdateWeights_EV_V_cl.setArg(6, device_ptrs_cl.d_grad_MK_correction));
            CL_CHECK(kernelUpdateWeights_EV_V_cl.setArg(7, device_ptrs_cl.d_grad_EV_full)); // Use full EV gradient
            CL_CHECK(kernelUpdateWeights_EV_V_cl.setArg(8, learning_rate));
            CL_CHECK(kernelUpdateWeights_EV_V_cl.setArg(9, cl_update_ev));
            CL_CHECK(kernelUpdateWeights_EV_V_cl.setArg(10, mat_heights));
            CL_CHECK(kernelUpdateWeights_EV_V_cl.setArg(11, embedding_dim));
            CL_CHECK(kernelUpdateWeights_EV_V_cl.setArg(12, context_win));
            CL_CHECK(current_stream_cl.enqueueNDRangeKernel(kernelUpdateWeights_EV_V_cl, cl::NullRange, global_ev, local_1d)); // Launch with grid size covering largest update target (EV)

            // --- Data Transfer Device -> Host (Asynchronous) ---
            for (int l = 0; l < num_weight_matrices_mlp; ++l) {
                CL_CHECK(current_stream_cl.enqueueReadBuffer(device_ptrs_cl.d_ver_weights[l], CL_FALSE, 0, mlp_weights_bytes, head_obj.ver.weights[l].mapped_data));
            }
            CL_CHECK(current_stream_cl.enqueueReadBuffer(device_ptrs_cl.d_EV, CL_FALSE, 0, ev_total_bytes, head_obj.EV.mapped_data));
            CL_CHECK(current_stream_cl.enqueueReadBuffer(device_ptrs_cl.d_MV_a, CL_FALSE, 0, proj_mat_bytes, head_obj.MV.mapped_data));
            CL_CHECK(current_stream_cl.enqueueReadBuffer(device_ptrs_cl.d_MQ_a, CL_FALSE, 0, proj_mat_bytes, head_obj.MQ.mapped_data));
            CL_CHECK(current_stream_cl.enqueueReadBuffer(device_ptrs_cl.d_MK_a, CL_FALSE, 0, proj_mat_bytes, head_obj.MK.mapped_data));
            CL_CHECK(current_stream_cl.finish());
        }

        // Explicitly release all aggregate OpenCL buffers before final sync
        agg_d_expected_v_buf = cl::Buffer();
        agg_d_EV_buf = cl::Buffer();
        agg_d_grad_EV_full_buf = cl::Buffer();
        agg_d_grad_EV_summed_buf = cl::Buffer();
        agg_d_grad_EV_scaled_buf = cl::Buffer();
        agg_d_grad_dv_buf = cl::Buffer();
        agg_d_KdotQ_buf = cl::Buffer();
        agg_d_head_storage_buf = cl::Buffer();
        agg_d_K_buf = cl::Buffer();
        agg_d_Q_buf = cl::Buffer();
        agg_d_pre_MV_buf = cl::Buffer();
        agg_d_MV_a_buf = cl::Buffer();
        agg_d_MQ_a_buf = cl::Buffer();
        agg_d_MK_a_buf = cl::Buffer();
        agg_d_grad_MV_buf = cl::Buffer();
        agg_d_grad_head_storage_buf = cl::Buffer();
        agg_d_lota_deriv_buf = cl::Buffer();
        agg_d_grad_KdotQ_buf = cl::Buffer();
        agg_d_grad_Q_storage_buf = cl::Buffer();
        agg_d_grad_MQ_buf = cl::Buffer();
        agg_d_grad_MK_correction_buf = cl::Buffer();
        agg_d_ver_activations_storage_buf = cl::Buffer();
        agg_d_ver_gweights_storage_buf = cl::Buffer(); // Added
        agg_d_ver_weights_storage_buf = cl::Buffer();
        agg_d_ver_deltas_storage_buf = cl::Buffer();

        for (int head_idx = 0; head_idx < num_heads_to_process; ++head_idx) {
            CL_CHECK(streams_cl[head_idx].finish());
        }
    } 
    catch (const std::exception& e) {
        std::cerr << "Standard Exception during clpartialbackward(V) for column " << k_col_idx << ": " << e.what() << std::endl;
        throw std::runtime_error("Exception during clpartialbackward(V) for column " + std::to_string(k_col_idx) + ": " + e.what());
    }
}

#endif