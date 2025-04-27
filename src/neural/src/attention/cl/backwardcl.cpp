#ifdef USE_OPENCL

#include "include/attention.hpp" // Includes mlp.hpp and maths.hpp indirectly or directly
#include <vector>
#include <string>
#include <stdexcept>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <map>
#include <CL/cl.hpp> // Or <CL/cl.h>

// Assume these OpenCL utilities are defined elsewhere
extern void CL_CHECK(cl_int err, const char* file, int line);
#define CL_CHECK(err) CL_CHECK(err, __FILE__, __LINE__)
extern cl_mem cl_create_buffer(cl_context context, cl_mem_flags flags, size_t size, void* host_ptr, cl_int& err);
extern void cl_write_buffer(cl_command_queue queue, cl_mem buffer, size_t size, const void* ptr, cl_bool blocking = CL_TRUE);
extern void cl_read_buffer(cl_command_queue queue, cl_mem buffer, size_t size, void* ptr, cl_bool blocking = CL_TRUE);
extern void cl_fill_buffer(cl_command_queue queue, cl_mem buffer, const void* pattern, size_t pattern_size, size_t offset, size_t size);
extern void cl_set_kernel_arg(cl_kernel kernel, cl_uint arg_index, size_t arg_size, const void* arg_value);
extern void cl_enqueue_nd_range_kernel(cl_command_queue queue, cl_kernel kernel, cl_uint work_dim, const size_t* global_work_offset, const size_t* global_work_size, const size_t* local_work_size);
extern void cl_finish(cl_command_queue queue);
extern void cl_release_mem_object(cl_mem memobj);
// Assume flatten and unflatten are available
extern std::vector<float> flatten(const std::vector<std::vector<float>>& vec2d);
extern void unflatten(const std::vector<float>& flat, std::vector<std::vector<float>>& vec2d, size_t rows, size_t cols);


/**
 * @brief OpenCL Backward Propagation using gradients from expected Horizontal output.
 *      Updates MH, MV, MQ, MK, EH, and EV.
 * @param context OpenCL context
 * @param queue OpenCL command queue
 * @param kernels Map of compiled OpenCL kernels
 * @param expected Expected output vector (target embedding for next token prediction)
 * @param in Input size (embedding dimension)
 * @param layers Number of layers in the MLPs
 */
void attention::clbackward(std::vector<float>& expected, int& in, int& layers)
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
    const int ev_size = context_win * embedding_dim; // Size of EV buffer
    const size_t embed_bytes = embedding_dim * sizeof(float);
    const size_t weights_bytes = embedding_dim * embedding_dim * sizeof(float); // Assuming square MLP layers

    cl_int err;

    // Validation
    if (embedding_dim != in) throw std::runtime_error("Embedding dimension mismatch");
    if (expected.size() != embedding_dim) throw std::runtime_error("Expected vector size mismatch");
    if (token_count <= 0) {
        std::cerr << "Warning: clbackward(expected,...) called with token_count <= 0. Skipping." << std::endl;
        return;
    }
    // Add other necessary validation checks...

    // --- Device Buffers (Attention) ---
    cl_mem d_expected_h = nullptr, d_EH = nullptr, d_EV = nullptr;
    cl_mem d_grad_EH = nullptr, d_grad_EV_scaled = nullptr;
    cl_mem d_grad_dh = nullptr, d_grad_dv = nullptr;
    cl_mem d_KdotQ = nullptr, d_head = nullptr;
    cl_mem d_K = nullptr, d_Q = nullptr;
    cl_mem d_pre_MH = nullptr, d_pre_MV = nullptr;
    cl_mem d_MH_a = nullptr, d_MV_a = nullptr, d_MQ_a = nullptr, d_MK_a = nullptr;
    cl_mem d_grad_MH = nullptr, d_grad_MV = nullptr;
    cl_mem d_grad_head = nullptr;
    cl_mem d_lota_deriv = nullptr;
    cl_mem d_grad_KdotQ = nullptr;
    cl_mem d_grad_K = nullptr, d_grad_Q = nullptr;
    cl_mem d_grad_MQ = nullptr, d_grad_MK = nullptr; // Simplified gradients

    // --- Device Buffers (MLP Internals) ---
    std::vector<cl_mem> d_hor_activations(layers, nullptr);
    std::vector<cl_mem> d_hor_weights(layers, nullptr);
    std::vector<cl_mem> d_hor_gweights(layers, nullptr);
    std::vector<cl_mem> d_hor_deltas(layers, nullptr);
    std::vector<cl_mem> d_ver_activations(layers, nullptr);
    std::vector<cl_mem> d_ver_weights(layers, nullptr);
    std::vector<cl_mem> d_ver_gweights(layers, nullptr);
    std::vector<cl_mem> d_ver_deltas(layers, nullptr);

    // Temporary flat vectors for H->D copy
    std::vector<std::vector<float>> flat_hor_weights(layers);
    std::vector<std::vector<float>> flat_ver_weights(layers);

    try {
        // --- Allocate Memory (Attention) ---
        d_expected_h = cl_create_buffer(context, CL_MEM_READ_ONLY, embed_bytes, nullptr, err); CL_CHECK(err);
        d_EH = cl_create_buffer(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, err); CL_CHECK(err); // Read/Write for update
        d_EV = cl_create_buffer(context, CL_MEM_READ_WRITE, ev_size * sizeof(float), nullptr, err); CL_CHECK(err); // Read/Write for update
        d_grad_EH = cl_create_buffer(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, err); CL_CHECK(err);
        d_grad_EV_scaled = cl_create_buffer(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, err); CL_CHECK(err);
        d_grad_dh = cl_create_buffer(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, err); CL_CHECK(err);
        d_grad_dv = cl_create_buffer(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, err); CL_CHECK(err);
        d_KdotQ = cl_create_buffer(context, CL_MEM_READ_ONLY, head_size * sizeof(float), nullptr, err); CL_CHECK(err);
        d_head = cl_create_buffer(context, CL_MEM_READ_WRITE, head_size * sizeof(float), nullptr, err); CL_CHECK(err);
        d_K = cl_create_buffer(context, CL_MEM_READ_ONLY, k_q_size * sizeof(float), nullptr, err); CL_CHECK(err);
        d_Q = cl_create_buffer(context, CL_MEM_READ_ONLY, k_q_size * sizeof(float), nullptr, err); CL_CHECK(err);
        d_pre_MH = cl_create_buffer(context, CL_MEM_READ_WRITE, mat_heights * sizeof(float), nullptr, err); CL_CHECK(err);
        d_pre_MV = cl_create_buffer(context, CL_MEM_READ_WRITE, mat_heights * sizeof(float), nullptr, err); CL_CHECK(err);
        d_MH_a = cl_create_buffer(context, CL_MEM_READ_WRITE, mh_mv_mq_mk_size * sizeof(float), nullptr, err); CL_CHECK(err);
        d_MV_a = cl_create_buffer(context, CL_MEM_READ_WRITE, mh_mv_mq_mk_size * sizeof(float), nullptr, err); CL_CHECK(err);
        d_MQ_a = cl_create_buffer(context, CL_MEM_READ_WRITE, mh_mv_mq_mk_size * sizeof(float), nullptr, err); CL_CHECK(err);
        d_MK_a = cl_create_buffer(context, CL_MEM_READ_WRITE, mh_mv_mq_mk_size * sizeof(float), nullptr, err); CL_CHECK(err);
        d_grad_MH = cl_create_buffer(context, CL_MEM_READ_WRITE, mh_mv_mq_mk_size * sizeof(float), nullptr, err); CL_CHECK(err);
        d_grad_MV = cl_create_buffer(context, CL_MEM_READ_WRITE, mh_mv_mq_mk_size * sizeof(float), nullptr, err); CL_CHECK(err);
        d_grad_head = cl_create_buffer(context, CL_MEM_READ_WRITE, head_size * sizeof(float), nullptr, err); CL_CHECK(err);
        d_lota_deriv = cl_create_buffer(context, CL_MEM_READ_WRITE, head_size * sizeof(float), nullptr, err); CL_CHECK(err);
        d_grad_KdotQ = cl_create_buffer(context, CL_MEM_READ_WRITE, head_size * sizeof(float), nullptr, err); CL_CHECK(err);
        d_grad_K = cl_create_buffer(context, CL_MEM_READ_WRITE, k_q_size * sizeof(float), nullptr, err); CL_CHECK(err);
        d_grad_Q = cl_create_buffer(context, CL_MEM_READ_WRITE, k_q_size * sizeof(float), nullptr, err); CL_CHECK(err);
        d_grad_MQ = cl_create_buffer(context, CL_MEM_READ_WRITE, mh_mv_mq_mk_size * sizeof(float), nullptr, err); CL_CHECK(err);
        d_grad_MK = cl_create_buffer(context, CL_MEM_READ_WRITE, mh_mv_mq_mk_size * sizeof(float), nullptr, err); CL_CHECK(err);

        // --- Allocate Memory (MLP Internals) ---
        for (int l = 0; l < layers; ++l) {
            d_hor_activations[l] = cl_create_buffer(context, CL_MEM_READ_ONLY, embed_bytes, nullptr, err); CL_CHECK(err);
            d_hor_weights[l] = cl_create_buffer(context, CL_MEM_READ_WRITE, weights_bytes, nullptr, err); CL_CHECK(err);
            d_hor_gweights[l] = cl_create_buffer(context, CL_MEM_READ_WRITE, weights_bytes, nullptr, err); CL_CHECK(err);
            d_hor_deltas[l] = cl_create_buffer(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, err); CL_CHECK(err);
            d_ver_activations[l] = cl_create_buffer(context, CL_MEM_READ_ONLY, embed_bytes, nullptr, err); CL_CHECK(err);
            d_ver_weights[l] = cl_create_buffer(context, CL_MEM_READ_WRITE, weights_bytes, nullptr, err); CL_CHECK(err);
            d_ver_gweights[l] = cl_create_buffer(context, CL_MEM_READ_WRITE, weights_bytes, nullptr, err); CL_CHECK(err);
            d_ver_deltas[l] = cl_create_buffer(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, err); CL_CHECK(err);
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

        cl_write_buffer(queue, d_expected_h, embed_bytes, expected.data());
        cl_write_buffer(queue, d_EH, embed_bytes, this->EH.data());
        cl_write_buffer(queue, d_EV, ev_size * sizeof(float), flat_EV.data());
        cl_write_buffer(queue, d_KdotQ, head_size * sizeof(float), flat_KdotQ.data());
        cl_write_buffer(queue, d_K, k_q_size * sizeof(float), flat_K.data());
        cl_write_buffer(queue, d_Q, k_q_size * sizeof(float), flat_Q.data());
        cl_write_buffer(queue, d_MH_a, mh_mv_mq_mk_size * sizeof(float), flat_MH_a.data());
        cl_write_buffer(queue, d_MV_a, mh_mv_mq_mk_size * sizeof(float), flat_MV_a.data());
        cl_write_buffer(queue, d_MQ_a, mh_mv_mq_mk_size * sizeof(float), flat_MQ_a.data());
        cl_write_buffer(queue, d_MK_a, mh_mv_mq_mk_size * sizeof(float), flat_MK_a.data());

        // --- Data Transfer H->D (MLP Internals) ---
        for (int l = 0; l < layers; ++l) {
            flat_hor_weights[l] = flatten(this->hor.weights[l]);
            flat_ver_weights[l] = flatten(this->ver.weights[l]);
            cl_write_buffer(queue, d_hor_activations[l], embed_bytes, this->hor.activations[l].data());
            cl_write_buffer(queue, d_hor_weights[l], weights_bytes, flat_hor_weights[l].data());
            cl_write_buffer(queue, d_ver_activations[l], embed_bytes, this->ver.activations[l].data());
            cl_write_buffer(queue, d_ver_weights[l], weights_bytes, flat_ver_weights[l].data());
        }

        // --- Kernel Launch Config ---
        const size_t local_work_size_1d = 256;
        size_t global_work_size_embed[1] = { (static_cast<size_t>(embedding_dim) + local_work_size_1d - 1) / local_work_size_1d * local_work_size_1d };
        size_t global_work_size_head[1] = { (static_cast<size_t>(head_size) + local_work_size_1d - 1) / local_work_size_1d * local_work_size_1d };
        size_t global_work_size_mat_heights[1] = { (static_cast<size_t>(mat_heights) + local_work_size_1d - 1) / local_work_size_1d * local_work_size_1d };
        size_t global_work_size_matrix[1] = { (static_cast<size_t>(mh_mv_mq_mk_size) + local_work_size_1d - 1) / local_work_size_1d * local_work_size_1d };
        size_t global_work_size_ev[1] = { (static_cast<size_t>(ev_size) + local_work_size_1d - 1) / local_work_size_1d * local_work_size_1d };

        size_t local_work_size_2d[2] = { 16, 16 };
        size_t global_work_size_embed_2d[2] = { (static_cast<size_t>(embedding_dim) + local_work_size_2d[0] - 1) / local_work_size_2d[0] * local_work_size_2d[0],
                                                (static_cast<size_t>(embedding_dim) + local_work_size_2d[1] - 1) / local_work_size_2d[1] * local_work_size_2d[1] };
        size_t global_work_size_head_2d[2] = { (static_cast<size_t>(token_count) + local_work_size_2d[0] - 1) / local_work_size_2d[0] * local_work_size_2d[0],
                                               (static_cast<size_t>(token_count) + local_work_size_2d[1] - 1) / local_work_size_2d[1] * local_work_size_2d[1] };
        size_t global_work_size_matrix_2d[2] = { (static_cast<size_t>(embedding_dim) + local_work_size_2d[0] - 1) / local_work_size_2d[0] * local_work_size_2d[0],
                                                 (static_cast<size_t>(mat_heights) + local_work_size_2d[1] - 1) / local_work_size_2d[1] * local_work_size_2d[1] };
        size_t global_work_size_kq_grad_2d[2] = { (static_cast<size_t>(mat_heights) + local_work_size_2d[0] - 1) / local_work_size_2d[0] * local_work_size_2d[0],
                                                  (static_cast<size_t>(token_count) + local_work_size_2d[1] - 1) / local_work_size_2d[1] * local_work_size_2d[1] };

        // --- Backpropagation Steps ---

        // Step 1: Compute grad_EH and grad_EV_scaled
        cl_kernel grad_eh_ev_kernel = kernels.at("kernelComputeGradientsEH_EV");
        cl_set_kernel_arg(grad_eh_ev_kernel, 0, sizeof(cl_mem), &d_EH);
        cl_set_kernel_arg(grad_eh_ev_kernel, 1, sizeof(cl_mem), &d_expected_h);
        cl_set_kernel_arg(grad_eh_ev_kernel, 2, sizeof(cl_mem), &d_grad_EH);
        cl_set_kernel_arg(grad_eh_ev_kernel, 3, sizeof(cl_mem), &d_grad_EV_scaled);
        cl_set_kernel_arg(grad_eh_ev_kernel, 4, sizeof(cl_int), &embedding_dim);
        cl_enqueue_nd_range_kernel(queue, grad_eh_ev_kernel, 1, nullptr, global_work_size_embed, local_work_size_1d);
        cl_finish(queue);

        // --- Step 2: Backprop through MLPs ---
        cl_kernel last_delta_kernel = kernels.at("kernelLastLayerDeltaSigmoid");
        cl_kernel hidden_delta_kernel = kernels.at("kernelHiddenDeltaSigmoid");
        cl_kernel update_weights_kernel = kernels.at("kernelUpdateWeightsAndGradients");

        // --- 2a: Backprop through hor MLP ---
        cl_set_kernel_arg(last_delta_kernel, 0, sizeof(cl_mem), &d_grad_EH);
        cl_set_kernel_arg(last_delta_kernel, 1, sizeof(cl_mem), &d_hor_activations[layers - 1]);
        cl_set_kernel_arg(last_delta_kernel, 2, sizeof(cl_mem), &d_hor_deltas[layers - 1]);
        cl_set_kernel_arg(last_delta_kernel, 3, sizeof(cl_int), &embedding_dim);
        cl_enqueue_nd_range_kernel(queue, last_delta_kernel, 1, nullptr, global_work_size_embed, local_work_size_1d);
        cl_finish(queue);

        for (int l = layers - 2; l >= 0; --l) {
            cl_set_kernel_arg(hidden_delta_kernel, 0, sizeof(cl_mem), &d_hor_deltas[l + 1]);
            cl_set_kernel_arg(hidden_delta_kernel, 1, sizeof(cl_mem), &d_hor_weights[l + 1]);
            cl_set_kernel_arg(hidden_delta_kernel, 2, sizeof(cl_mem), &d_hor_activations[l]);
            cl_set_kernel_arg(hidden_delta_kernel, 3, sizeof(cl_mem), &d_hor_deltas[l]);
            cl_set_kernel_arg(hidden_delta_kernel, 4, sizeof(cl_int), &embedding_dim);
            cl_set_kernel_arg(hidden_delta_kernel, 5, sizeof(cl_int), &embedding_dim);
            cl_enqueue_nd_range_kernel(queue, hidden_delta_kernel, 1, nullptr, global_work_size_embed, local_work_size_1d);
            cl_finish(queue);
        }

        for (int l = 0; l < layers; ++l) {
            cl_mem d_prev_activations = (l == 0) ? d_EH : d_hor_activations[l - 1];
            cl_set_kernel_arg(update_weights_kernel, 0, sizeof(cl_mem), &d_hor_deltas[l]);
            cl_set_kernel_arg(update_weights_kernel, 1, sizeof(cl_mem), &d_prev_activations);
            cl_set_kernel_arg(update_weights_kernel, 2, sizeof(cl_mem), &d_hor_weights[l]);
            cl_set_kernel_arg(update_weights_kernel, 3, sizeof(cl_mem), &d_hor_gweights[l]);
            cl_set_kernel_arg(update_weights_kernel, 4, sizeof(cl_float), &learning_rate);
            cl_set_kernel_arg(update_weights_kernel, 5, sizeof(cl_int), &embedding_dim);
            cl_set_kernel_arg(update_weights_kernel, 6, sizeof(cl_int), &embedding_dim);
            cl_enqueue_nd_range_kernel(queue, update_weights_kernel, 2, nullptr, global_work_size_embed_2d, local_work_size_2d);
            cl_finish(queue);
        }

        // --- 2b: Backprop through ver MLP ---
        cl_set_kernel_arg(last_delta_kernel, 0, sizeof(cl_mem), &d_grad_EV_scaled);
        cl_set_kernel_arg(last_delta_kernel, 1, sizeof(cl_mem), &d_ver_activations[layers - 1]);
        cl_set_kernel_arg(last_delta_kernel, 2, sizeof(cl_mem), &d_ver_deltas[layers - 1]);
        // Arg 3 (embedding_dim) is already set
        cl_enqueue_nd_range_kernel(queue, last_delta_kernel, 1, nullptr, global_work_size_embed, local_work_size_1d);
        cl_finish(queue);

        for (int l = layers - 2; l >= 0; --l) {
            cl_set_kernel_arg(hidden_delta_kernel, 0, sizeof(cl_mem), &d_ver_deltas[l + 1]);
            cl_set_kernel_arg(hidden_delta_kernel, 1, sizeof(cl_mem), &d_ver_weights[l + 1]);
            cl_set_kernel_arg(hidden_delta_kernel, 2, sizeof(cl_mem), &d_ver_activations[l]);
            cl_set_kernel_arg(hidden_delta_kernel, 3, sizeof(cl_mem), &d_ver_deltas[l]);
            // Args 4, 5 (embedding_dim) are already set
            cl_enqueue_nd_range_kernel(queue, hidden_delta_kernel, 1, nullptr, global_work_size_embed, local_work_size_1d);
            cl_finish(queue);
        }

        for (int l = 0; l < layers; ++l) {
            // CAUTION: Assuming input to ver MLP layer 0 is size embedding_dim. See clbackward1stHead comments.
            cl_mem d_prev_activations = (l == 0) ? d_EV : d_ver_activations[l - 1];
            int prev_layer_size = (l == 0) ? embedding_dim : embedding_dim;
            size_t current_global_work_size_2d[2] = { global_work_size_embed_2d[0], global_work_size_embed_2d[1] };

            cl_set_kernel_arg(update_weights_kernel, 0, sizeof(cl_mem), &d_ver_deltas[l]);
            cl_set_kernel_arg(update_weights_kernel, 1, sizeof(cl_mem), &d_prev_activations);
            cl_set_kernel_arg(update_weights_kernel, 2, sizeof(cl_mem), &d_ver_weights[l]);
            cl_set_kernel_arg(update_weights_kernel, 3, sizeof(cl_mem), &d_ver_gweights[l]);
            // Arg 4 (learning_rate) is already set
            cl_set_kernel_arg(update_weights_kernel, 5, sizeof(cl_int), &embedding_dim);
            cl_set_kernel_arg(update_weights_kernel, 6, sizeof(cl_int), &prev_layer_size);
            cl_enqueue_nd_range_kernel(queue, update_weights_kernel, 2, nullptr, current_global_work_size_2d, local_work_size_2d);
            cl_finish(queue);
        }
        // --- End of MLP Backprop ---

        // --- Step 3: Compute grad_dh and grad_dv ---
        // Use the kernel that sums gradients, NOT _1stHead version
        cl_kernel grad_mlp_input_kernel = kernels.at("kernelComputeGradMLPInput");
        cl_set_kernel_arg(grad_mlp_input_kernel, 0, sizeof(cl_mem), &d_hor_deltas[0]);
        cl_set_kernel_arg(grad_mlp_input_kernel, 1, sizeof(cl_mem), &d_hor_weights[0]);
        cl_set_kernel_arg(grad_mlp_input_kernel, 2, sizeof(cl_mem), &d_grad_dh);
        cl_set_kernel_arg(grad_mlp_input_kernel, 3, sizeof(cl_int), &embedding_dim); // current_layer_size
        cl_set_kernel_arg(grad_mlp_input_kernel, 4, sizeof(cl_int), &embedding_dim); // input_size
        cl_enqueue_nd_range_kernel(queue, grad_mlp_input_kernel, 1, nullptr, global_work_size_embed, local_work_size_1d);

        cl_set_kernel_arg(grad_mlp_input_kernel, 0, sizeof(cl_mem), &d_ver_deltas[0]);
        cl_set_kernel_arg(grad_mlp_input_kernel, 1, sizeof(cl_mem), &d_ver_weights[0]);
        cl_set_kernel_arg(grad_mlp_input_kernel, 2, sizeof(cl_mem), &d_grad_dv);
        // Args 3, 4 (embedding_dim) are already set
        cl_enqueue_nd_range_kernel(queue, grad_mlp_input_kernel, 1, nullptr, global_work_size_embed, local_work_size_1d);
        cl_finish(queue);

        // --- Step 4: Compute grad_MH and grad_MV ---
        cl_kernel lota_kernel = kernels.at("clLOTA2d");
        cl_set_kernel_arg(lota_kernel, 0, sizeof(cl_mem), &d_KdotQ);
        cl_set_kernel_arg(lota_kernel, 1, sizeof(cl_mem), &d_head);
        cl_set_kernel_arg(lota_kernel, 2, sizeof(cl_int), &token_count);
        cl_set_kernel_arg(lota_kernel, 3, sizeof(cl_int), &token_count);
        size_t lota_global[1] = { (static_cast<size_t>(head_size) + local_work_size_1d - 1) / local_work_size_1d * local_work_size_1d };
        size_t lota_local[1] = { local_work_size_1d };
         if (head_size > 0) {
             if (head_size > local_work_size_1d) { /* Adjust if needed */ lota_global[0] = local_work_size_1d; }
             else { lota_global[0] = head_size; lota_local[0] = head_size; }
             cl_enqueue_nd_range_kernel(queue, lota_kernel, 1, nullptr, lota_global, lota_local);
         }

        cl_kernel pre_mh_mv_kernel = kernels.at("kernelComputePreMH_MV");
        cl_set_kernel_arg(pre_mh_mv_kernel, 0, sizeof(cl_mem), &d_head);
        cl_set_kernel_arg(pre_mh_mv_kernel, 1, sizeof(cl_mem), &d_K);
        cl_set_kernel_arg(pre_mh_mv_kernel, 2, sizeof(cl_mem), &d_Q);
        cl_set_kernel_arg(pre_mh_mv_kernel, 3, sizeof(cl_mem), &d_pre_MH);
        cl_set_kernel_arg(pre_mh_mv_kernel, 4, sizeof(cl_mem), &d_pre_MV);
        cl_set_kernel_arg(pre_mh_mv_kernel, 5, sizeof(cl_int), &token_count);
        cl_set_kernel_arg(pre_mh_mv_kernel, 6, sizeof(cl_int), &mat_heights);
        cl_enqueue_nd_range_kernel(queue, pre_mh_mv_kernel, 1, nullptr, global_work_size_mat_heights, local_work_size_1d);

        cl_kernel grad_mh_mv_kernel = kernels.at("kernelComputeGradMH_MV");
        cl_set_kernel_arg(grad_mh_mv_kernel, 0, sizeof(cl_mem), &d_pre_MH);
        cl_set_kernel_arg(grad_mh_mv_kernel, 1, sizeof(cl_mem), &d_pre_MV);
        cl_set_kernel_arg(grad_mh_mv_kernel, 2, sizeof(cl_mem), &d_grad_dh);
        cl_set_kernel_arg(grad_mh_mv_kernel, 3, sizeof(cl_mem), &d_grad_dv);
        cl_set_kernel_arg(grad_mh_mv_kernel, 4, sizeof(cl_mem), &d_grad_MH);
        cl_set_kernel_arg(grad_mh_mv_kernel, 5, sizeof(cl_mem), &d_grad_MV);
        cl_set_kernel_arg(grad_mh_mv_kernel, 6, sizeof(cl_int), &mat_heights);
        cl_set_kernel_arg(grad_mh_mv_kernel, 7, sizeof(cl_int), &embedding_dim);
        cl_enqueue_nd_range_kernel(queue, grad_mh_mv_kernel, 2, nullptr, global_work_size_matrix_2d, local_work_size_2d);

        // --- Step 5: Compute grad_head ---
        cl_kernel grad_head_kernel = kernels.at("kernelComputeGradHead");
        cl_set_kernel_arg(grad_head_kernel, 0, sizeof(cl_mem), &d_K);
        cl_set_kernel_arg(grad_head_kernel, 1, sizeof(cl_mem), &d_Q);
        cl_set_kernel_arg(grad_head_kernel, 2, sizeof(cl_mem), &d_MH_a);
        cl_set_kernel_arg(grad_head_kernel, 3, sizeof(cl_mem), &d_MV_a);
        cl_set_kernel_arg(grad_head_kernel, 4, sizeof(cl_mem), &d_grad_dh);
        cl_set_kernel_arg(grad_head_kernel, 5, sizeof(cl_mem), &d_grad_dv);
        cl_set_kernel_arg(grad_head_kernel, 6, sizeof(cl_mem), &d_grad_head);
        cl_set_kernel_arg(grad_head_kernel, 7, sizeof(cl_int), &token_count);
        cl_set_kernel_arg(grad_head_kernel, 8, sizeof(cl_int), &mat_heights);
        cl_set_kernel_arg(grad_head_kernel, 9, sizeof(cl_int), &embedding_dim);
        cl_enqueue_nd_range_kernel(queue, grad_head_kernel, 2, nullptr, global_work_size_head_2d, local_work_size_2d);

        // --- Step 6: Backprop through LOTA ---
        cl_kernel lota_deriv_kernel = kernels.at("clLOTA2dder"); // Or kernelComputeSimpleLOTAder
        cl_set_kernel_arg(lota_deriv_kernel, 0, sizeof(cl_mem), &d_KdotQ);
        cl_set_kernel_arg(lota_deriv_kernel, 1, sizeof(cl_mem), &d_lota_deriv);
        cl_set_kernel_arg(lota_deriv_kernel, 2, sizeof(cl_int), &token_count);
        cl_set_kernel_arg(lota_deriv_kernel, 3, sizeof(cl_int), &token_count);
         if (head_size > 0) {
             if (head_size > local_work_size_1d) { /* Adjust if needed */ lota_global[0] = local_work_size_1d; }
             else { lota_global[0] = head_size; lota_local[0] = head_size; }
             cl_enqueue_nd_range_kernel(queue, lota_deriv_kernel, 1, nullptr, lota_global, lota_local);
         }

        cl_kernel grad_kdotq_kernel = kernels.at("kernelComputeGradKdotQ_LOTA");
        cl_set_kernel_arg(grad_kdotq_kernel, 0, sizeof(cl_mem), &d_grad_head);
        cl_set_kernel_arg(grad_kdotq_kernel, 1, sizeof(cl_mem), &d_lota_deriv);
        cl_set_kernel_arg(grad_kdotq_kernel, 2, sizeof(cl_mem), &d_grad_KdotQ);
        cl_set_kernel_arg(grad_kdotq_kernel, 3, sizeof(cl_float), &scaling_factor);
        cl_set_kernel_arg(grad_kdotq_kernel, 4, sizeof(cl_int), &head_size);
        cl_enqueue_nd_range_kernel(queue, grad_kdotq_kernel, 1, nullptr, global_work_size_head, local_work_size_1d);

        // --- Step 7: Compute grad_K and grad_Q ---
        cl_kernel grad_k_q_kernel = kernels.at("kernelComputeGradK_Q");
        cl_set_kernel_arg(grad_k_q_kernel, 0, sizeof(cl_mem), &d_grad_KdotQ);
        cl_set_kernel_arg(grad_k_q_kernel, 1, sizeof(cl_mem), &d_K);
        cl_set_kernel_arg(grad_k_q_kernel, 2, sizeof(cl_mem), &d_Q);
        cl_set_kernel_arg(grad_k_q_kernel, 3, sizeof(cl_mem), &d_grad_K);
        cl_set_kernel_arg(grad_k_q_kernel, 4, sizeof(cl_mem), &d_grad_Q);
        cl_set_kernel_arg(grad_k_q_kernel, 5, sizeof(cl_int), &token_count);
        cl_set_kernel_arg(grad_k_q_kernel, 6, sizeof(cl_int), &mat_heights);
        cl_enqueue_nd_range_kernel(queue, grad_k_q_kernel, 2, nullptr, global_work_size_kq_grad_2d, local_work_size_2d);

        // --- Step 8: Compute grad_MK and grad_MQ (Simplified) ---
        cl_kernel grad_mk_mq_kernel = kernels.at("kernelComputeGradMK_MQ_Simplified");
        cl_mem d_null = nullptr; // Pass null pointers if K/Q embeds not used/available
        cl_set_kernel_arg(grad_mk_mq_kernel, 0, sizeof(cl_mem), &d_grad_K);
        cl_set_kernel_arg(grad_mk_mq_kernel, 1, sizeof(cl_mem), &d_grad_Q);
        cl_set_kernel_arg(grad_mk_mq_kernel, 2, sizeof(cl_mem), &d_null); // d_K_embed
        cl_set_kernel_arg(grad_mk_mq_kernel, 3, sizeof(cl_mem), &d_null); // d_Q_embed
        cl_set_kernel_arg(grad_mk_mq_kernel, 4, sizeof(cl_mem), &d_grad_MK);
        cl_set_kernel_arg(grad_mk_mq_kernel, 5, sizeof(cl_mem), &d_grad_MQ);
        cl_set_kernel_arg(grad_mk_mq_kernel, 6, sizeof(cl_int), &token_count);
        cl_set_kernel_arg(grad_mk_mq_kernel, 7, sizeof(cl_int), &mat_heights);
        cl_set_kernel_arg(grad_mk_mq_kernel, 8, sizeof(cl_int), &embedding_dim);
        cl_enqueue_nd_range_kernel(queue, grad_mk_mq_kernel, 2, nullptr, global_work_size_matrix_2d, local_work_size_2d);

        // --- Step 9 & 10: Update Weights (MH, MV, MQ, MK) and Embeddings (EH, EV) ---
        cl_kernel update_weights_eh_ev_kernel = kernels.at("kernelUpdateWeights_EH_EV");
        cl_set_kernel_arg(update_weights_eh_ev_kernel, 0, sizeof(cl_mem), &d_MH_a);
        cl_set_kernel_arg(update_weights_eh_ev_kernel, 1, sizeof(cl_mem), &d_MV_a);
        cl_set_kernel_arg(update_weights_eh_ev_kernel, 2, sizeof(cl_mem), &d_MQ_a);
        cl_set_kernel_arg(update_weights_eh_ev_kernel, 3, sizeof(cl_mem), &d_MK_a);
        cl_set_kernel_arg(update_weights_eh_ev_kernel, 4, sizeof(cl_mem), &d_EH);
        cl_set_kernel_arg(update_weights_eh_ev_kernel, 5, sizeof(cl_mem), &d_EV);
        cl_set_kernel_arg(update_weights_eh_ev_kernel, 6, sizeof(cl_mem), &d_grad_MH);
        cl_set_kernel_arg(update_weights_eh_ev_kernel, 7, sizeof(cl_mem), &d_grad_MV);
        cl_set_kernel_arg(update_weights_eh_ev_kernel, 8, sizeof(cl_mem), &d_grad_MQ);
        cl_set_kernel_arg(update_weights_eh_ev_kernel, 9, sizeof(cl_mem), &d_grad_MK);
        cl_set_kernel_arg(update_weights_eh_ev_kernel, 10, sizeof(cl_mem), &d_grad_EH); // Use original grad_EH
        cl_set_kernel_arg(update_weights_eh_ev_kernel, 11, sizeof(cl_mem), &d_grad_EV_scaled); // Use scaled grad for EV update
        cl_set_kernel_arg(update_weights_eh_ev_kernel, 12, sizeof(cl_float), &learning_rate);
        cl_set_kernel_arg(update_weights_eh_ev_kernel, 13, sizeof(cl_int), &mat_heights);
        cl_set_kernel_arg(update_weights_eh_ev_kernel, 14, sizeof(cl_int), &embedding_dim);
        cl_set_kernel_arg(update_weights_eh_ev_kernel, 15, sizeof(cl_int), &context_win);
        // Launch with a grid size covering the largest update target (likely EV)
        cl_enqueue_nd_range_kernel(queue, update_weights_eh_ev_kernel, 1, nullptr, global_work_size_ev, local_work_size_1d);
        cl_finish(queue);

        // --- Data Transfer D->H ---
        // Copy updated MLP weights and gradients back
        for (int l = 0; l < layers; ++l) {
            std::vector<float> updated_flat_weights(embedding_dim * embedding_dim);
            std::vector<float> calculated_flat_gradients(embedding_dim * embedding_dim);
            // Hor MLP
            cl_read_buffer(queue, d_hor_weights[l], weights_bytes, updated_flat_weights.data());
            cl_read_buffer(queue, d_hor_gweights[l], weights_bytes, calculated_flat_gradients.data());
            unflatten(updated_flat_weights, this->hor.weights[l], embedding_dim, embedding_dim);
            unflatten(calculated_flat_gradients, this->hor.gweights[l], embedding_dim, embedding_dim);
            // Ver MLP
            cl_read_buffer(queue, d_ver_weights[l], weights_bytes, updated_flat_weights.data());
            cl_read_buffer(queue, d_ver_gweights[l], weights_bytes, calculated_flat_gradients.data());
            unflatten(updated_flat_weights, this->ver.weights[l], embedding_dim, embedding_dim);
            unflatten(calculated_flat_gradients, this->ver.gweights[l], embedding_dim, embedding_dim);
        }

        // Copy updated Attention parameters back
        std::vector<float> updated_MH_a(mh_mv_mq_mk_size);
        std::vector<float> updated_MV_a(mh_mv_mq_mk_size);
        std::vector<float> updated_MQ_a(mh_mv_mq_mk_size);
        std::vector<float> updated_MK_a(mh_mv_mq_mk_size);
        std::vector<float> updated_EV(ev_size);

        cl_read_buffer(queue, d_EH, embed_bytes, this->EH.data());
        cl_read_buffer(queue, d_EV, ev_size * sizeof(float), updated_EV.data()); // Read full EV
        cl_read_buffer(queue, d_MH_a, mh_mv_mq_mk_size * sizeof(float), updated_MH_a.data());
        cl_read_buffer(queue, d_MV_a, mh_mv_mq_mk_size * sizeof(float), updated_MV_a.data());
        cl_read_buffer(queue, d_MQ_a, mh_mv_mq_mk_size * sizeof(float), updated_MQ_a.data());
        cl_read_buffer(queue, d_MK_a, mh_mv_mq_mk_size * sizeof(float), updated_MK_a.data());

        unflatten(updated_EV, this->EV, context_win, embedding_dim); // Unflatten full EV
        unflatten(updated_MH_a, this->MH.a, mat_heights, embedding_dim);
        unflatten(updated_MV_a, this->MV.a, mat_heights, embedding_dim);
        unflatten(updated_MQ_a, this->MQ.a, mat_heights, embedding_dim);
        unflatten(updated_MK_a, this->MK.a, mat_heights, embedding_dim);

    } catch (const std::exception& e) {
        std::cerr << "OpenCL Exception in clbackward(expected): " << e.what() << std::endl;
        // Cleanup (Attention)
        cl_release_mem_object(d_expected_h); cl_release_mem_object(d_EH); cl_release_mem_object(d_EV);
        cl_release_mem_object(d_grad_EH); cl_release_mem_object(d_grad_EV_scaled);
        cl_release_mem_object(d_grad_dh); cl_release_mem_object(d_grad_dv);
        cl_release_mem_object(d_KdotQ); cl_release_mem_object(d_head); cl_release_mem_object(d_K); cl_release_mem_object(d_Q);
        cl_release_mem_object(d_pre_MH); cl_release_mem_object(d_pre_MV);
        cl_release_mem_object(d_MH_a); cl_release_mem_object(d_MV_a); cl_release_mem_object(d_MQ_a); cl_release_mem_object(d_MK_a);
        cl_release_mem_object(d_grad_MH); cl_release_mem_object(d_grad_MV); cl_release_mem_object(d_grad_head);
        cl_release_mem_object(d_lota_deriv); cl_release_mem_object(d_grad_KdotQ);
        cl_release_mem_object(d_grad_K); cl_release_mem_object(d_grad_Q);
        cl_release_mem_object(d_grad_MQ); cl_release_mem_object(d_grad_MK);
        // Cleanup (MLP Internals)
        for (int l = 0; l < layers; ++l) {
            cl_release_mem_object(d_hor_activations[l]); cl_release_mem_object(d_hor_weights[l]); cl_release_mem_object(d_hor_gweights[l]); cl_release_mem_object(d_hor_deltas[l]);
            cl_release_mem_object(d_ver_activations[l]); cl_release_mem_object(d_ver_weights[l]); cl_release_mem_object(d_ver_gweights[l]); cl_release_mem_object(d_ver_deltas[l]);
        }
        throw;
    }

    // --- Cleanup (Success Case) ---
    cl_release_mem_object(d_expected_h); cl_release_mem_object(d_EH); cl_release_mem_object(d_EV);
    cl_release_mem_object(d_grad_EH); cl_release_mem_object(d_grad_EV_scaled);
    cl_release_mem_object(d_grad_dh); cl_release_mem_object(d_grad_dv);
    cl_release_mem_object(d_KdotQ); cl_release_mem_object(d_head); cl_release_mem_object(d_K); cl_release_mem_object(d_Q);
    cl_release_mem_object(d_pre_MH); cl_release_mem_object(d_pre_MV);
    cl_release_mem_object(d_MH_a); cl_release_mem_object(d_MV_a); cl_release_mem_object(d_MQ_a); cl_release_mem_object(d_MK_a);
    cl_release_mem_object(d_grad_MH); cl_release_mem_object(d_grad_MV); cl_release_mem_object(d_grad_head);
    cl_release_mem_object(d_lota_deriv); cl_release_mem_object(d_grad_KdotQ);
    cl_release_mem_object(d_grad_K); cl_release_mem_object(d_grad_Q);
    cl_release_mem_object(d_grad_MQ); cl_release_mem_object(d_grad_MK);
    for (int l = 0; l < layers; ++l) {
        cl_release_mem_object(d_hor_activations[l]); cl_release_mem_object(d_hor_weights[l]); cl_release_mem_object(d_hor_gweights[l]); cl_release_mem_object(d_hor_deltas[l]);
        cl_release_mem_object(d_ver_activations[l]); cl_release_mem_object(d_ver_weights[l]); cl_release_mem_object(d_ver_gweights[l]); cl_release_mem_object(d_ver_deltas[l]);
    }
}


/**
 * @brief OpenCL Backward Propagation using gradients from expected Vertical output only.
 *      Updates MV, MQ, MK (correction), and EV.
 * @param context OpenCL context
 * @param queue OpenCL command queue
 * @param kernels Map of compiled OpenCL kernels
 * @param expectedV vertical retention vector (host)
 * @param in Input size (embedding dimension - used for MLP)
 * @param layers Number of layers in the MLPs
 */
void attention::clbackward(std::vector<std::vector<float>>& expectedV, int& in, int& layers)
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

    cl_int err;

    // Validation
    if (expectedV.size() != context_win || (!expectedV.empty() && expectedV[0].size() != embedding_dim)) {
        throw std::runtime_error("ExpectedV dimensions mismatch");
    }
     if (token_count <= 0) {
        std::cerr << "Warning: clbackward(expectedV,...) called with token_count <= 0. Skipping." << std::endl;
        return;
    }
    // Add other necessary validation checks...

    // --- Device Buffers (Attention) ---
    cl_mem d_expected_v = nullptr, d_EV = nullptr;
    cl_mem d_grad_EV_full = nullptr, d_grad_EV_summed = nullptr, d_grad_EV_scaled = nullptr;
    cl_mem d_grad_dv = nullptr;
    cl_mem d_KdotQ = nullptr, d_head = nullptr;
    cl_mem d_K = nullptr, d_Q = nullptr;
    cl_mem d_pre_MV = nullptr;
    cl_mem d_MV_a = nullptr, d_MQ_a = nullptr, d_MK_a = nullptr;
    cl_mem d_grad_MV = nullptr;
    cl_mem d_grad_head = nullptr;
    cl_mem d_lota_deriv = nullptr;
    cl_mem d_grad_KdotQ = nullptr;
    cl_mem d_grad_Q = nullptr;
    cl_mem d_grad_MQ = nullptr, d_grad_MK_correction = nullptr;

    // --- Device Buffers (ver MLP Internals) ---
    std::vector<cl_mem> d_ver_activations(layers, nullptr);
    std::vector<cl_mem> d_ver_weights(layers, nullptr);
    std::vector<cl_mem> d_ver_gweights(layers, nullptr);
    std::vector<cl_mem> d_ver_deltas(layers, nullptr);

    // Temporary flat vectors for H->D copy
    std::vector<std::vector<float>> flat_ver_weights(layers);

    try {
        // --- Allocate Memory (Attention) ---
        d_expected_v = cl_create_buffer(context, CL_MEM_READ_ONLY, ev_size * sizeof(float), nullptr, err); CL_CHECK(err);
        d_EV = cl_create_buffer(context, CL_MEM_READ_WRITE, ev_size * sizeof(float), nullptr, err); CL_CHECK(err); // Read/Write for update
        d_grad_EV_full = cl_create_buffer(context, CL_MEM_READ_WRITE, ev_size * sizeof(float), nullptr, err); CL_CHECK(err);
        d_grad_EV_summed = cl_create_buffer(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, err); CL_CHECK(err);
        d_grad_EV_scaled = cl_create_buffer(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, err); CL_CHECK(err);
        d_grad_dv = cl_create_buffer(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, err); CL_CHECK(err);
        d_KdotQ = cl_create_buffer(context, CL_MEM_READ_ONLY, head_size * sizeof(float), nullptr, err); CL_CHECK(err);
        d_head = cl_create_buffer(context, CL_MEM_READ_WRITE, head_size * sizeof(float), nullptr, err); CL_CHECK(err);
        d_K = cl_create_buffer(context, CL_MEM_READ_ONLY, k_q_size * sizeof(float), nullptr, err); CL_CHECK(err);
        d_Q = cl_create_buffer(context, CL_MEM_READ_ONLY, k_q_size * sizeof(float), nullptr, err); CL_CHECK(err);
        d_pre_MV = cl_create_buffer(context, CL_MEM_READ_WRITE, mat_heights * sizeof(float), nullptr, err); CL_CHECK(err);
        d_MV_a = cl_create_buffer(context, CL_MEM_READ_WRITE, mh_mv_mq_mk_size * sizeof(float), nullptr, err); CL_CHECK(err);
        d_MQ_a = cl_create_buffer(context, CL_MEM_READ_WRITE, mh_mv_mq_mk_size * sizeof(float), nullptr, err); CL_CHECK(err);
        d_MK_a = cl_create_buffer(context, CL_MEM_READ_WRITE, mh_mv_mq_mk_size * sizeof(float), nullptr, err); CL_CHECK(err);
        d_grad_MV = cl_create_buffer(context, CL_MEM_READ_WRITE, mh_mv_mq_mk_size * sizeof(float), nullptr, err); CL_CHECK(err);
        d_grad_head = cl_create_buffer(context, CL_MEM_READ_WRITE, head_size * sizeof(float), nullptr, err); CL_CHECK(err);
        d_lota_deriv = cl_create_buffer(context, CL_MEM_READ_WRITE, head_size * sizeof(float), nullptr, err); CL_CHECK(err);
        d_grad_KdotQ = cl_create_buffer(context, CL_MEM_READ_WRITE, head_size * sizeof(float), nullptr, err); CL_CHECK(err);
        d_grad_Q = cl_create_buffer(context, CL_MEM_READ_WRITE, k_q_size * sizeof(float), nullptr, err); CL_CHECK(err);
        d_grad_MQ = cl_create_buffer(context, CL_MEM_READ_WRITE, mh_mv_mq_mk_size * sizeof(float), nullptr, err); CL_CHECK(err);
        d_grad_MK_correction = cl_create_buffer(context, CL_MEM_READ_WRITE, mh_mv_mq_mk_size * sizeof(float), nullptr, err); CL_CHECK(err);

        // --- Allocate Memory (ver MLP Internals) ---
        for (int l = 0; l < layers; ++l) {
            d_ver_activations[l] = cl_create_buffer(context, CL_MEM_READ_ONLY, embed_bytes, nullptr, err); CL_CHECK(err);
            d_ver_weights[l] = cl_create_buffer(context, CL_MEM_READ_WRITE, weights_bytes, nullptr, err); CL_CHECK(err);
            d_ver_gweights[l] = cl_create_buffer(context, CL_MEM_READ_WRITE, weights_bytes, nullptr, err); CL_CHECK(err);
            d_ver_deltas[l] = cl_create_buffer(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, err); CL_CHECK(err);
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

        cl_write_buffer(queue, d_expected_v, ev_size * sizeof(float), flat_expectedV.data());
        cl_write_buffer(queue, d_EV, ev_size * sizeof(float), flat_EV.data());
        cl_write_buffer(queue, d_KdotQ, head_size * sizeof(float), flat_KdotQ.data());
        cl_write_buffer(queue, d_K, k_q_size * sizeof(float), flat_K.data());
        cl_write_buffer(queue, d_Q, k_q_size * sizeof(float), flat_Q.data());
        cl_write_buffer(queue, d_MV_a, mh_mv_mq_mk_size * sizeof(float), flat_MV_a.data());
        cl_write_buffer(queue, d_MQ_a, mh_mv_mq_mk_size * sizeof(float), flat_MQ_a.data());
        cl_write_buffer(queue, d_MK_a, mh_mv_mq_mk_size * sizeof(float), flat_MK_a.data());

        // --- Data Transfer H->D (ver MLP Internals) ---
        for (int l = 0; l < layers; ++l) {
            flat_ver_weights[l] = flatten(this->ver.weights[l]);
            cl_write_buffer(queue, d_ver_activations[l], embed_bytes, this->ver.activations[l].data());
            cl_write_buffer(queue, d_ver_weights[l], weights_bytes, flat_ver_weights[l].data());
        }

        // --- Kernel Launch Config ---
        const size_t local_work_size_1d = 256;
        size_t global_work_size_embed[1] = { (static_cast<size_t>(embedding_dim) + local_work_size_1d - 1) / local_work_size_1d * local_work_size_1d };
        size_t global_work_size_head[1] = { (static_cast<size_t>(head_size) + local_work_size_1d - 1) / local_work_size_1d * local_work_size_1d };
        size_t global_work_size_mat_heights[1] = { (static_cast<size_t>(mat_heights) + local_work_size_1d - 1) / local_work_size_1d * local_work_size_1d };
        size_t global_work_size_matrix[1] = { (static_cast<size_t>(mh_mv_mq_mk_size) + local_work_size_1d - 1) / local_work_size_1d * local_work_size_1d };
        size_t global_work_size_ev[1] = { (static_cast<size_t>(ev_size) + local_work_size_1d - 1) / local_work_size_1d * local_work_size_1d };

        size_t local_work_size_2d[2] = { 16, 16 };
        size_t global_work_size_embed_2d[2] = { (static_cast<size_t>(embedding_dim) + local_work_size_2d[0] - 1) / local_work_size_2d[0] * local_work_size_2d[0],
                                                (static_cast<size_t>(embedding_dim) + local_work_size_2d[1] - 1) / local_work_size_2d[1] * local_work_size_2d[1] };
        size_t global_work_size_head_2d[2] = { (static_cast<size_t>(token_count) + local_work_size_2d[0] - 1) / local_work_size_2d[0] * local_work_size_2d[0],
                                               (static_cast<size_t>(token_count) + local_work_size_2d[1] - 1) / local_work_size_2d[1] * local_work_size_2d[1] };
        size_t global_work_size_matrix_2d[2] = { (static_cast<size_t>(embedding_dim) + local_work_size_2d[0] - 1) / local_work_size_2d[0] * local_work_size_2d[0],
                                                 (static_cast<size_t>(mat_heights) + local_work_size_2d[1] - 1) / local_work_size_2d[1] * local_work_size_2d[1] };
        size_t global_work_size_kq_grad_2d[2] = { (static_cast<size_t>(mat_heights) + local_work_size_2d[0] - 1) / local_work_size_2d[0] * local_work_size_2d[0],
                                                  (static_cast<size_t>(token_count) + local_work_size_2d[1] - 1) / local_work_size_2d[1] * local_work_size_2d[1] };

        // --- Backpropagation Steps ---

        // Step 1: Compute grad_EV (full, summed, scaled)
        cl_kernel grad_ev_v_kernel = kernels.at("kernelComputeGradientsEV_V");
        cl_set_kernel_arg(grad_ev_v_kernel, 0, sizeof(cl_mem), &d_EV);
        cl_set_kernel_arg(grad_ev_v_kernel, 1, sizeof(cl_mem), &d_expected_v);
        cl_set_kernel_arg(grad_ev_v_kernel, 2, sizeof(cl_mem), &d_grad_EV_full);
        cl_set_kernel_arg(grad_ev_v_kernel, 3, sizeof(cl_mem), &d_grad_EV_summed);
        cl_set_kernel_arg(grad_ev_v_kernel, 4, sizeof(cl_mem), &d_grad_EV_scaled);
        cl_set_kernel_arg(grad_ev_v_kernel, 5, sizeof(cl_float), &learning_rate);
        cl_set_kernel_arg(grad_ev_v_kernel, 6, sizeof(cl_int), &context_win);
        cl_set_kernel_arg(grad_ev_v_kernel, 7, sizeof(cl_int), &embedding_dim);
        cl_enqueue_nd_range_kernel(queue, grad_ev_v_kernel, 1, nullptr, global_work_size_embed, local_work_size_1d);
        cl_finish(queue);

        // --- Step 2: Backprop through ver MLP ---
        cl_kernel last_delta_kernel = kernels.at("kernelLastLayerDeltaSigmoid");
        cl_kernel hidden_delta_kernel = kernels.at("kernelHiddenDeltaSigmoid");
        cl_kernel update_weights_kernel = kernels.at("kernelUpdateWeightsAndGradients");

        cl_set_kernel_arg(last_delta_kernel, 0, sizeof(cl_mem), &d_grad_EV_scaled);
        cl_set_kernel_arg(last_delta_kernel, 1, sizeof(cl_mem), &d_ver_activations[layers - 1]);
        cl_set_kernel_arg(last_delta_kernel, 2, sizeof(cl_mem), &d_ver_deltas[layers - 1]);
        cl_set_kernel_arg(last_delta_kernel, 3, sizeof(cl_int), &embedding_dim);
        cl_enqueue_nd_range_kernel(queue, last_delta_kernel, 1, nullptr, global_work_size_embed, local_work_size_1d);
        cl_finish(queue);

        for (int l = layers - 2; l >= 0; --l) {
            cl_set_kernel_arg(hidden_delta_kernel, 0, sizeof(cl_mem), &d_ver_deltas[l + 1]);
            cl_set_kernel_arg(hidden_delta_kernel, 1, sizeof(cl_mem), &d_ver_weights[l + 1]);
            cl_set_kernel_arg(hidden_delta_kernel, 2, sizeof(cl_mem), &d_ver_activations[l]);
            cl_set_kernel_arg(hidden_delta_kernel, 3, sizeof(cl_mem), &d_ver_deltas[l]);
            cl_set_kernel_arg(hidden_delta_kernel, 4, sizeof(cl_int), &embedding_dim);
            cl_set_kernel_arg(hidden_delta_kernel, 5, sizeof(cl_int), &embedding_dim);
            cl_enqueue_nd_range_kernel(queue, hidden_delta_kernel, 1, nullptr, global_work_size_embed, local_work_size_1d);
            cl_finish(queue);
        }

        for (int l = 0; l < layers; ++l) {
            // CAUTION: Assuming input to ver MLP layer 0 is size embedding_dim.
            cl_mem d_prev_activations = (l == 0) ? d_EV : d_ver_activations[l - 1];
            int prev_layer_size = (l == 0) ? embedding_dim : embedding_dim;
            size_t current_global_work_size_2d[2] = { global_work_size_embed_2d[0], global_work_size_embed_2d[1] };

            cl_set_kernel_arg(update_weights_kernel, 0, sizeof(cl_mem), &d_ver_deltas[l]);
            cl_set_kernel_arg(update_weights_kernel, 1, sizeof(cl_mem), &d_prev_activations);
            cl_set_kernel_arg(update_weights_kernel, 2, sizeof(cl_mem), &d_ver_weights[l]);
            cl_set_kernel_arg(update_weights_kernel, 3, sizeof(cl_mem), &d_ver_gweights[l]);
            cl_set_kernel_arg(update_weights_kernel, 4, sizeof(cl_float), &learning_rate);
            cl_set_kernel_arg(update_weights_kernel, 5, sizeof(cl_int), &embedding_dim);
            cl_set_kernel_arg(update_weights_kernel, 6, sizeof(cl_int), &prev_layer_size);
            cl_enqueue_nd_range_kernel(queue, update_weights_kernel, 2, nullptr, current_global_work_size_2d, local_work_size_2d);
            cl_finish(queue);
        }
        // --- End of MLP Backprop ---

        // --- Step 3: Compute grad_dv ---
        cl_kernel grad_mlp_input_kernel = kernels.at("kernelComputeGradMLPInput");
        cl_set_kernel_arg(grad_mlp_input_kernel, 0, sizeof(cl_mem), &d_ver_deltas[0]);
        cl_set_kernel_arg(grad_mlp_input_kernel, 1, sizeof(cl_mem), &d_ver_weights[0]);
        cl_set_kernel_arg(grad_mlp_input_kernel, 2, sizeof(cl_mem), &d_grad_dv);
        cl_set_kernel_arg(grad_mlp_input_kernel, 3, sizeof(cl_int), &embedding_dim);
        cl_set_kernel_arg(grad_mlp_input_kernel, 4, sizeof(cl_int), &embedding_dim);
        cl_enqueue_nd_range_kernel(queue, grad_mlp_input_kernel, 1, nullptr, global_work_size_embed, local_work_size_1d);
        cl_finish(queue);

        // --- Step 4: Compute grad_MV ---
        cl_kernel lota_kernel = kernels.at("clLOTA2d");
        cl_set_kernel_arg(lota_kernel, 0, sizeof(cl_mem), &d_KdotQ);
        cl_set_kernel_arg(lota_kernel, 1, sizeof(cl_mem), &d_head);
        cl_set_kernel_arg(lota_kernel, 2, sizeof(cl_int), &token_count);
        cl_set_kernel_arg(lota_kernel, 3, sizeof(cl_int), &token_count);
        size_t lota_global[1] = { (static_cast<size_t>(head_size) + local_work_size_1d - 1) / local_work_size_1d * local_work_size_1d };
        size_t lota_local[1] = { local_work_size_1d };
         if (head_size > 0) {
             if (head_size > local_work_size_1d) { /* Adjust if needed */ lota_global[0] = local_work_size_1d; }
             else { lota_global[0] = head_size; lota_local[0] = head_size; }
             cl_enqueue_nd_range_kernel(queue, lota_kernel, 1, nullptr, lota_global, lota_local);
         }

        cl_kernel pre_mv_v_kernel = kernels.at("kernelComputePreMV_V");
        cl_set_kernel_arg(pre_mv_v_kernel, 0, sizeof(cl_mem), &d_head);
        cl_set_kernel_arg(pre_mv_v_kernel, 1, sizeof(cl_mem), &d_Q);
        cl_set_kernel_arg(pre_mv_v_kernel, 2, sizeof(cl_mem), &d_pre_MV);
        cl_set_kernel_arg(pre_mv_v_kernel, 3, sizeof(cl_int), &token_count);
        cl_set_kernel_arg(pre_mv_v_kernel, 4, sizeof(cl_int), &mat_heights);
        cl_enqueue_nd_range_kernel(queue, pre_mv_v_kernel, 1, nullptr, global_work_size_mat_heights, local_work_size_1d);

        cl_kernel grad_mv_v_kernel = kernels.at("kernelComputeGradMV_V");
        cl_set_kernel_arg(grad_mv_v_kernel, 0, sizeof(cl_mem), &d_pre_MV);
        cl_set_kernel_arg(grad_mv_v_kernel, 1, sizeof(cl_mem), &d_grad_dv);
        cl_set_kernel_arg(grad_mv_v_kernel, 2, sizeof(cl_mem), &d_grad_MV);
        cl_set_kernel_arg(grad_mv_v_kernel, 3, sizeof(cl_int), &mat_heights);
        cl_set_kernel_arg(grad_mv_v_kernel, 4, sizeof(cl_int), &embedding_dim);
        cl_enqueue_nd_range_kernel(queue, grad_mv_v_kernel, 2, nullptr, global_work_size_matrix_2d, local_work_size_2d);

        // --- Step 5: Compute grad_head (dv part only) ---
        cl_kernel grad_head_v_kernel = kernels.at("kernelComputeGradHead_V");
        cl_set_kernel_arg(grad_head_v_kernel, 0, sizeof(cl_mem), &d_Q);
        cl_set_kernel_arg(grad_head_v_kernel, 1, sizeof(cl_mem), &d_MV_a);
        cl_set_kernel_arg(grad_head_v_kernel, 2, sizeof(cl_mem), &d_grad_dv);
        cl_set_kernel_arg(grad_head_v_kernel, 3, sizeof(cl_mem), &d_grad_head);
        cl_set_kernel_arg(grad_head_v_kernel, 4, sizeof(cl_int), &token_count);
        cl_set_kernel_arg(grad_head_v_kernel, 5, sizeof(cl_int), &mat_heights);
        cl_set_kernel_arg(grad_head_v_kernel, 6, sizeof(cl_int), &embedding_dim);
        cl_enqueue_nd_range_kernel(queue, grad_head_v_kernel, 2, nullptr, global_work_size_head_2d, local_work_size_2d);

        // --- Step 6: Backprop through LOTA ---
        cl_kernel lota_deriv_kernel = kernels.at("clLOTA2dder");
        cl_set_kernel_arg(lota_deriv_kernel, 0, sizeof(cl_mem), &d_KdotQ);
        cl_set_kernel_arg(lota_deriv_kernel, 1, sizeof(cl_mem), &d_lota_deriv);
        cl_set_kernel_arg(lota_deriv_kernel, 2, sizeof(cl_int), &token_count);
        cl_set_kernel_arg(lota_deriv_kernel, 3, sizeof(cl_int), &token_count);
         if (head_size > 0) {
             if (head_size > local_work_size_1d) { /* Adjust if needed */ lota_global[0] = local_work_size_1d; }
             else { lota_global[0] = head_size; lota_local[0] = head_size; }
             cl_enqueue_nd_range_kernel(queue, lota_deriv_kernel, 1, nullptr, lota_global, lota_local);
         }

        cl_kernel grad_kdotq_kernel = kernels.at("kernelComputeGradKdotQ_LOTA");
        cl_set_kernel_arg(grad_kdotq_kernel, 0, sizeof(cl_mem), &d_grad_head);
        cl_set_kernel_arg(grad_kdotq_kernel, 1, sizeof(cl_mem), &d_lota_deriv);
        cl_set_kernel_arg(grad_kdotq_kernel, 2, sizeof(cl_mem), &d_grad_KdotQ);
        cl_set_kernel_arg(grad_kdotq_kernel, 3, sizeof(cl_float), &scaling_factor);
        cl_set_kernel_arg(grad_kdotq_kernel, 4, sizeof(cl_int), &head_size);
        cl_enqueue_nd_range_kernel(queue, grad_kdotq_kernel, 1, nullptr, global_work_size_head, local_work_size_1d);

        // --- Step 7: Compute grad_Q ---
        cl_kernel grad_q_v_kernel = kernels.at("kernelComputeGradQ_V");
        cl_set_kernel_arg(grad_q_v_kernel, 0, sizeof(cl_mem), &d_grad_KdotQ);
        cl_set_kernel_arg(grad_q_v_kernel, 1, sizeof(cl_mem), &d_K);
        cl_set_kernel_arg(grad_q_v_kernel, 2, sizeof(cl_mem), &d_grad_Q);
        cl_set_kernel_arg(grad_q_v_kernel, 3, sizeof(cl_int), &token_count);
        cl_set_kernel_arg(grad_q_v_kernel, 4, sizeof(cl_int), &mat_heights);
        cl_enqueue_nd_range_kernel(queue, grad_q_v_kernel, 2, nullptr, global_work_size_kq_grad_2d, local_work_size_2d);

        // --- Step 8: Compute grad_MQ and grad_MK_correction (Complex) ---
        cl_kernel grad_mq_v_kernel = kernels.at("kernelComputeGradMQ_V");
        cl_mem d_null = nullptr; // Assuming Q embed not available/needed
        cl_set_kernel_arg(grad_mq_v_kernel, 0, sizeof(cl_mem), &d_grad_Q);
        cl_set_kernel_arg(grad_mq_v_kernel, 1, sizeof(cl_mem), &d_null); // d_Q_embed
        cl_set_kernel_arg(grad_mq_v_kernel, 2, sizeof(cl_mem), &d_grad_MQ);
        cl_set_kernel_arg(grad_mq_v_kernel, 3, sizeof(cl_int), &token_count);
        cl_set_kernel_arg(grad_mq_v_kernel, 4, sizeof(cl_int), &mat_heights);
        cl_set_kernel_arg(grad_mq_v_kernel, 5, sizeof(cl_int), &embedding_dim);
        cl_enqueue_nd_range_kernel(queue, grad_mq_v_kernel, 2, nullptr, global_work_size_matrix_2d, local_work_size_2d);

        cl_kernel grad_mk_corr_kernel = kernels.at("kernelComputeGradMKCorrection");
        cl_set_kernel_arg(grad_mk_corr_kernel, 0, sizeof(cl_mem), &d_grad_MQ);
        cl_set_kernel_arg(grad_mk_corr_kernel, 1, sizeof(cl_mem), &d_Q);
        cl_set_kernel_arg(grad_mk_corr_kernel, 2, sizeof(cl_mem), &d_K);
        cl_set_kernel_arg(grad_mk_corr_kernel, 3, sizeof(cl_mem), &d_grad_MK_correction);
        cl_set_kernel_arg(grad_mk_corr_kernel, 4, sizeof(cl_int), &token_count);
        cl_set_kernel_arg(grad_mk_corr_kernel, 5, sizeof(cl_int), &mat_heights);
        cl_set_kernel_arg(grad_mk_corr_kernel, 6, sizeof(cl_int), &embedding_dim);
        cl_enqueue_nd_range_kernel(queue, grad_mk_corr_kernel, 2, nullptr, global_work_size_matrix_2d, local_work_size_2d);

        // --- Step 9 & 10: Update Weights (MV, MQ, MK_correction) and EV ---
        cl_kernel update_weights_ev_v_kernel = kernels.at("kernelUpdateWeights_EV_V");
        cl_set_kernel_arg(update_weights_ev_v_kernel, 0, sizeof(cl_mem), &d_MV_a);
        cl_set_kernel_arg(update_weights_ev_v_kernel, 1, sizeof(cl_mem), &d_MQ_a);
        cl_set_kernel_arg(update_weights_ev_v_kernel, 2, sizeof(cl_mem), &d_MK_a);
        cl_set_kernel_arg(update_weights_ev_v_kernel, 3, sizeof(cl_mem), &d_EV);
        cl_set_kernel_arg(update_weights_ev_v_kernel, 4, sizeof(cl_mem), &d_grad_MV);
        cl_set_kernel_arg(update_weights_ev_v_kernel, 5, sizeof(cl_mem), &d_grad_MQ);
        cl_set_kernel_arg(update_weights_ev_v_kernel, 6, sizeof(cl_mem), &d_grad_MK_correction);
        cl_set_kernel_arg(update_weights_ev_v_kernel, 7, sizeof(cl_mem), &d_grad_EV_full); // Use full EV gradient
        cl_set_kernel_arg(update_weights_ev_v_kernel, 8, sizeof(cl_float), &learning_rate);
        cl_set_kernel_arg(update_weights_ev_v_kernel, 9, sizeof(cl_int), &mat_heights);
        cl_set_kernel_arg(update_weights_ev_v_kernel, 10, sizeof(cl_int), &embedding_dim);
        cl_set_kernel_arg(update_weights_ev_v_kernel, 11, sizeof(cl_int), &context_win);
        // Launch with grid size covering largest update target (EV)
        cl_enqueue_nd_range_kernel(queue, update_weights_ev_v_kernel, 1, nullptr, global_work_size_ev, local_work_size_1d);
        cl_finish(queue);

        // --- Data Transfer D->H ---
        // Copy updated ver MLP weights and gradients back
        for (int l = 0; l < layers; ++l) {
            std::vector<float> updated_flat_weights(embedding_dim * embedding_dim);
            std::vector<float> calculated_flat_gradients(embedding_dim * embedding_dim);
            cl_read_buffer(queue, d_ver_weights[l], weights_bytes, updated_flat_weights.data());
            cl_read_buffer(queue, d_ver_gweights[l], weights_bytes, calculated_flat_gradients.data());
            unflatten(updated_flat_weights, this->ver.weights[l], embedding_dim, embedding_dim);
            unflatten(calculated_flat_gradients, this->ver.gweights[l], embedding_dim, embedding_dim);
        }

        // Copy updated Attention parameters back
        std::vector<float> updated_MV_a(mh_mv_mq_mk_size);
        std::vector<float> updated_MQ_a(mh_mv_mq_mk_size);
        std::vector<float> updated_MK_a(mh_mv_mq_mk_size);
        std::vector<float> updated_EV(ev_size);

        cl_read_buffer(queue, d_EV, ev_size * sizeof(float), updated_EV.data()); // Read full EV
        cl_read_buffer(queue, d_MV_a, mh_mv_mq_mk_size * sizeof(float), updated_MV_a.data());
        cl_read_buffer(queue, d_MQ_a, mh_mv_mq_mk_size * sizeof(float), updated_MQ_a.data());
        cl_read_buffer(queue, d_MK_a, mh_mv_mq_mk_size * sizeof(float), updated_MK_a.data());

        unflatten(updated_EV, this->EV, context_win, embedding_dim); // Unflatten full EV
        unflatten(updated_MV_a, this->MV.a, mat_heights, embedding_dim);
        unflatten(updated_MQ_a, this->MQ.a, mat_heights, embedding_dim);
        unflatten(updated_MK_a, this->MK.a, mat_heights, embedding_dim);

    } catch (const std::exception& e) {
        std::cerr << "OpenCL Exception in clbackward(expectedV): " << e.what() << std::endl;
        // Cleanup (Attention)
        cl_release_mem_object(d_expected_v); cl_release_mem_object(d_EV); cl_release_mem_object(d_grad_EV_full); cl_release_mem_object(d_grad_EV_summed); cl_release_mem_object(d_grad_EV_scaled);
        cl_release_mem_object(d_grad_dv);
        cl_release_mem_object(d_KdotQ); cl_release_mem_object(d_head); cl_release_mem_object(d_K); cl_release_mem_object(d_Q);
        cl_release_mem_object(d_pre_MV); cl_release_mem_object(d_MV_a); cl_release_mem_object(d_MQ_a); cl_release_mem_object(d_MK_a);
        cl_release_mem_object(d_grad_MV); cl_release_mem_object(d_grad_head);
        cl_release_mem_object(d_lota_deriv); cl_release_mem_object(d_grad_KdotQ); cl_release_mem_object(d_grad_Q);
        cl_release_mem_object(d_grad_MQ); cl_release_mem_object(d_grad_MK_correction);
        // Cleanup (ver MLP Internals)
        for (int l = 0; l < layers; ++l) {
            cl_release_mem_object(d_ver_activations[l]); cl_release_mem_object(d_ver_weights[l]); cl_release_mem_object(d_ver_gweights[l]); cl_release_mem_object(d_ver_deltas[l]);
        }
        throw;
    }

    // --- Cleanup (Success Case) ---
    cl_release_mem_object(d_expected_v); cl_release_mem_object(d_EV); cl_release_mem_object(d_grad_EV_full); cl_release_mem_object(d_grad_EV_summed); cl_release_mem_object(d_grad_EV_scaled);
    cl_release_mem_object(d_grad_dv);
    cl_release_mem_object(d_KdotQ); cl_release_mem_object(d_head); cl_release_mem_object(d_K); cl_release_mem_object(d_Q);
    cl_release_mem_object(d_pre_MV); cl_release_mem_object(d_MV_a); cl_release_mem_object(d_MQ_a); cl_release_mem_object(d_MK_a);
    cl_release_mem_object(d_grad_MV); cl_release_mem_object(d_grad_head);
    cl_release_mem_object(d_lota_deriv); cl_release_mem_object(d_grad_KdotQ); cl_release_mem_object(d_grad_Q);
    cl_release_mem_object(d_grad_MQ); cl_release_mem_object(d_grad_MK_correction);
    for (int l = 0; l < layers; ++l) {
        cl_release_mem_object(d_ver_activations[l]); cl_release_mem_object(d_ver_weights[l]); cl_release_mem_object(d_ver_gweights[l]); cl_release_mem_object(d_ver_deltas[l]);
    }
}

#endif // USE_OPENCL
