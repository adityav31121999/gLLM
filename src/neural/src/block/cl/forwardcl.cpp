#ifdef USE_OPENCL
#if defined(_WIN64) 
    #include <CL/cl.hpp>
#elif defined(__linux__)
    #include <CL/opencl.hpp>
#endif
#include "include/block.hpp"
#include <vector>
#include <stdexcept>
#include <string>
#include <map>
#include <maths.hpp>

// Helper macro for OpenCL error checking (copied from attention/cl/forwardcl.cpp for self-containment)
#ifndef CL_CHECK
    #define CL_CHECK(call) do { \
        cl_int err = call; \
        if (err != CL_SUCCESS) { \
            fprintf(stderr, "OpenCL Error in %s at line %d (%s): %s\n", __FILE__, __LINE__, #call, getCLErrorString(err)); \
            throw std::runtime_error(getCLErrorString(err)); \
        } \
    } while (0)
#endif

// Helper function to safely get a kernel from the context's map
static cl::Kernel get_kernel_with_check(OpenCLContext& context_obj, const std::string& kernel_name) {
    auto it = context_obj.kernels.find(kernel_name);
    if (it == context_obj.kernels.end()) {
        throw std::runtime_error("OpenCL kernel not found in context: '" + kernel_name +
                                 "'. Check OpenCLContext initialization and kernel compilation/naming.");
    }
    return it->second;
}

// Local struct to manage device buffers for one head's forward pass
struct HeadForwardDeviceBuffersCL {
    cl::Buffer d_K, d_Q, d_KdotQ, d_head_attention;
    cl::Buffer d_row_sums, d_col_sums;
    cl::Buffer d_dh_accum, d_dv_accum;
    cl::Buffer d_MH_hxd, d_MV_hxd;
    cl::Buffer d_dh, d_dv, d_EH;
    cl::Buffer d_EV_processed_data;
    cl::Buffer d_ver_accumulated_ev;
    cl::Buffer d_hor_inputs, d_ver_inputs;
    cl::Buffer d_hor_output, d_ver_output;
    cl::Buffer d_relu_hor_output, d_relu_ver_output;
    cl::Buffer d_mlp_bufferA_hor, d_mlp_bufferB_hor;
    cl::Buffer d_mlp_bufferA_ver, d_mlp_bufferB_ver;
    cl::Buffer d_mlp_pre_activation;

    HeadForwardDeviceBuffersCL() = default;
};

// Local struct to manage device buffers for one head's forward pass
struct HeadForwardDeviceBuffersEVCL {
    cl::Buffer d_K, d_Q, d_KdotQ, d_head_attention;
    cl::Buffer d_col_sums;
    cl::Buffer d_dv_accum;
    cl::Buffer d_MV_hxd;
    cl::Buffer d_dv;
    cl::Buffer d_EV_processed_data;
    cl::Buffer d_ver_accumulated_ev;
    cl::Buffer d_ver_inputs;
    cl::Buffer d_ver_output;
    cl::Buffer d_relu_ver_output;
    cl::Buffer d_mlp_bufferA_ver, d_mlp_bufferB_ver;
    cl::Buffer d_mlp_pre_activation;

    HeadForwardDeviceBuffersEVCL() = default;
};


/**
 * @brief OpenCL forward propagation on single ith column of the FIRST block.
 * @param context OpenCL context.
 * @param queue OpenCL command queue.
 * @param kernels Map of compiled OpenCL kernels.
 * @param in dimension size (maps to d_embedding in attention::clforprop)
 * @param tokenCount number of tokens (maps to currentTokenCount in attention::clforprop)
 * @param i column index (0-based)
 * @param layers number of layers in mlp (maps to layers_mlp in attention::clforprop)
 */
void block::cl1parallelForprop(int& in_dim_param, int& tokenCount_param, int col_idx_param, int& layers_mlp_param)
{
    if (col_idx_param < 0 || col_idx_param >= y) {
        throw std::out_of_range("cl1parallelForprop (first block): column index 'i' (" + std::to_string(col_idx_param) + ") is out of range [0, " + std::to_string(y - 1) + "].");
    }

    cl_int cl_err; // For OpenCL error codes
    OpenCLContext& context_obj = clcontext;
    cl::Context context = context_obj.context;

    const int num_heads_in_col = x;
    const int d_embedding = EMBEDDING;
    const int h_attention = CONTEXT_WIN;
    const int n_tokens = tokenCount_param;

    // Per-head byte sizes (assuming n_tokens is fixed for all heads in this call)
    size_t k_bytes_ph = static_cast<size_t>(CONTEXT_WIN) * CONTEXT_WIN * sizeof(float);
    size_t q_bytes_ph = static_cast<size_t>(CONTEXT_WIN) * CONTEXT_WIN * sizeof(float);
    size_t kdotq_bytes_ph = static_cast<size_t>(CONTEXT_WIN) * CONTEXT_WIN * sizeof(float);
    size_t head_bytes_ph = static_cast<size_t>(CONTEXT_WIN) * CONTEXT_WIN * sizeof(float);
    size_t sums_bytes_ph = static_cast<size_t>(CONTEXT_WIN) * sizeof(float);
    size_t accum_bytes_ph = static_cast<size_t>(CONTEXT_WIN) * sizeof(float);
    size_t proj_mat_bytes_ph = static_cast<size_t>(EMBEDDING) * CONTEXT_WIN * sizeof(float);
    size_t ev_processed_bytes_ph = static_cast<size_t>(CONTEXT_WIN) * EMBEDDING * sizeof(float);
    size_t embed_bytes_ph = static_cast<size_t>(EMBEDDING) * sizeof(float);

    // --- Aggregate Buffer Allocation ---
    cl::Buffer agg_d_K, agg_d_Q, agg_d_KdotQ, agg_d_head_attention;
    cl::Buffer agg_d_row_sums, agg_d_col_sums;
    cl::Buffer agg_d_dh_accum, agg_d_dv_accum;
    cl::Buffer agg_d_MH_hxd, agg_d_MV_hxd;
    cl::Buffer agg_d_dh, agg_d_dv, agg_d_EH;
    cl::Buffer agg_d_EV_processed_data;
    cl::Buffer agg_d_ver_accumulated_ev;
    cl::Buffer agg_d_hor_inputs, agg_d_ver_inputs;
    cl::Buffer agg_d_hor_output, agg_d_ver_output;
    cl::Buffer agg_d_relu_hor_output, agg_d_relu_ver_output;
    cl::Buffer agg_d_mlp_bufferA_hor, agg_d_mlp_bufferB_hor;
    cl::Buffer agg_d_mlp_bufferA_ver, agg_d_mlp_bufferB_ver;
    cl::Buffer agg_d_mlp_pre_activation;

    if (n_tokens > 0) {
        agg_d_K = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col * k_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_Q = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col * q_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_KdotQ = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col * kdotq_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_head_attention = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col * head_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_row_sums = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col * sums_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_col_sums = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col * sums_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_EV_processed_data = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col * ev_processed_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
    }
    agg_d_dh_accum = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col * accum_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
    agg_d_dv_accum = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col * accum_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
    agg_d_MH_hxd = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col * proj_mat_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
    agg_d_MV_hxd = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col * proj_mat_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
    agg_d_dh = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col * embed_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
    agg_d_dv = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col * embed_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
    agg_d_EH = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col * embed_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
    agg_d_ver_accumulated_ev = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col * embed_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
    agg_d_hor_inputs = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col * embed_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
    agg_d_ver_inputs = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col * embed_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
    agg_d_hor_output = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col * embed_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
    agg_d_ver_output = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col * embed_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
    agg_d_relu_hor_output = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col * embed_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
    agg_d_relu_ver_output = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col * embed_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
    agg_d_mlp_bufferA_hor = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col * embed_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
    agg_d_mlp_bufferB_hor = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col * embed_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
    agg_d_mlp_bufferA_ver = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col * embed_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
    agg_d_mlp_bufferB_ver = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col * embed_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
    agg_d_mlp_pre_activation = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col * embed_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
    
    cl::CommandQueue queue(context, context_obj.device, 0, &cl_err); CL_CHECK(cl_err);
    std::vector<HeadForwardDeviceBuffersCL> head_gpu_data(num_heads_in_col);

    // --- Pre-allocate MLP weight buffers to avoid creation in the loop ---
    std::vector<std::vector<cl::Buffer>> all_d_hor_mlp_weights(num_heads_in_col);
    std::vector<std::vector<cl::Buffer>> all_d_ver_mlp_weights(num_heads_in_col);
    for (int layer_idx = 0; layer_idx < num_heads_in_col; ++layer_idx) {
        attention& head_cpu = b[layer_idx][col_idx_param];
        size_t num_weight_matrices = head_cpu.hor.weights.size();
        all_d_hor_mlp_weights[layer_idx].resize(num_weight_matrices);
        all_d_ver_mlp_weights[layer_idx].resize(num_weight_matrices);

        for (size_t mlp_layer = 0; mlp_layer < num_weight_matrices; ++mlp_layer) {
            mat& hor_weights_mat = head_cpu.hor.weights[mlp_layer];
            size_t weights_bytes = static_cast<size_t>(hor_weights_mat.row) * hor_weights_mat.col * sizeof(float);
            all_d_hor_mlp_weights[layer_idx][mlp_layer] = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, weights_bytes, hor_weights_mat.mapped_data, &cl_err); CL_CHECK(cl_err);

            mat& ver_weights_mat = head_cpu.ver.weights[mlp_layer];
            weights_bytes = static_cast<size_t>(ver_weights_mat.row) * ver_weights_mat.col * sizeof(float);
            all_d_ver_mlp_weights[layer_idx][mlp_layer] = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, weights_bytes, ver_weights_mat.mapped_data, &cl_err); CL_CHECK(cl_err);
        }
    }

    for (int layer_idx = 0; layer_idx < num_heads_in_col; ++layer_idx) {
        attention& head_cpu = b[layer_idx][col_idx_param];
        cl::CommandQueue& current_queue = queue;
        HeadForwardDeviceBuffersCL& current_gpu_bufs = head_gpu_data[layer_idx];

        const int num_ev_rows_to_process = CONTEXT_WIN;

        if (n_tokens <= 0) {
            std::fill(head_cpu.EH.begin(), head_cpu.EH.end(), 0.0f);
            if (n_tokens == 0 && head_cpu.EV.mapped_data && head_cpu.EV.row > 0 && head_cpu.EV.col == d_embedding) {
                std::fill_n(head_cpu.EV.mapped_data, head_cpu.EV.col, 0.0f);
            }
            continue;
        }

        // --- Basic Validation ---
        if (head_cpu.EV.row < n_tokens || head_cpu.EV.col != d_embedding) {
            throw std::runtime_error("EV matrix not properly sized for current tokenCount n in cl1parallelForprop (first block) for head [" + std::to_string(layer_idx) + "][" + std::to_string(col_idx_param) + "]. Expected rows >= " + std::to_string(n_tokens) + " and cols = " + std::to_string(d_embedding) +
                                     ", but got rows = " + std::to_string(head_cpu.EV.row) + " and cols = " + std::to_string(head_cpu.EV.col));
        }
        if (head_cpu.K.row != CONTEXT_WIN || head_cpu.K.col != CONTEXT_WIN ||
            head_cpu.Q.row != CONTEXT_WIN || head_cpu.Q.col != CONTEXT_WIN ||
            head_cpu.KdotQ.row != CONTEXT_WIN || head_cpu.KdotQ.col != CONTEXT_WIN ||
            head_cpu.MH.row != EMBEDDING || head_cpu.MH.col != CONTEXT_WIN ||
            head_cpu.MV.row != EMBEDDING || head_cpu.MV.col != CONTEXT_WIN ||
            head_cpu.EH.size() != static_cast<size_t>(d_embedding) ||
            head_cpu.hor.hlayers.empty() || head_cpu.ver.hlayers.empty() || head_cpu.hor.weights.empty() || head_cpu.ver.weights.empty())
        {
            throw std::runtime_error("Attention component dimension mismatch or uninitialized member in cl1parallelForprop (first block) for head [" +
                                     std::to_string(layer_idx) + "][" + std::to_string(col_idx_param) + "]. K.row=" + std::to_string(head_cpu.K.row) + ", n=" + std::to_string(n_tokens));
        }
        if (d_embedding != in_dim_param) {
            throw std::runtime_error("Embedding dimension mismatch (EMBEDDING vs in) for head [" + std::to_string(layer_idx) + "][" + std::to_string(col_idx_param) + "].");
        }
        if (head_cpu.hor.hlayers[0].size() != static_cast<size_t>(d_embedding) || head_cpu.ver.hlayers[0].size() != static_cast<size_t>(d_embedding) ||
            head_cpu.hor.weights.back().row != static_cast<size_t>(d_embedding) || head_cpu.ver.weights.back().row != static_cast<size_t>(d_embedding)) {
             throw std::runtime_error("MLP input/output layer dimension mismatch with 'd' for head [" + std::to_string(layer_idx) + "][" + std::to_string(col_idx_param) + "].");
        }

        try {
            auto create_sub_buffer = [&](cl::Buffer& parent_agg_buf, size_t per_head_bytes) {
                cl_buffer_region region = {static_cast<size_t>(layer_idx) * per_head_bytes, per_head_bytes};
                return parent_agg_buf.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err);
            };
            HeadForwardDeviceBuffersCL current_gpu_bufs;

            if (n_tokens > 0) {
                current_gpu_bufs.d_K = create_sub_buffer(agg_d_K, k_bytes_ph); CL_CHECK(cl_err);
                current_gpu_bufs.d_Q = create_sub_buffer(agg_d_Q, q_bytes_ph); CL_CHECK(cl_err);
                current_gpu_bufs.d_KdotQ = create_sub_buffer(agg_d_KdotQ, kdotq_bytes_ph); CL_CHECK(cl_err);
                current_gpu_bufs.d_head_attention = create_sub_buffer(agg_d_head_attention, head_bytes_ph); CL_CHECK(cl_err);
                current_gpu_bufs.d_row_sums = create_sub_buffer(agg_d_row_sums, sums_bytes_ph); CL_CHECK(cl_err);
                current_gpu_bufs.d_col_sums = create_sub_buffer(agg_d_col_sums, sums_bytes_ph); CL_CHECK(cl_err);
                current_gpu_bufs.d_EV_processed_data = create_sub_buffer(agg_d_EV_processed_data, ev_processed_bytes_ph); CL_CHECK(cl_err);
            }
            current_gpu_bufs.d_dh_accum = create_sub_buffer(agg_d_dh_accum, accum_bytes_ph); CL_CHECK(cl_err);
            current_gpu_bufs.d_dv_accum = create_sub_buffer(agg_d_dv_accum, accum_bytes_ph); CL_CHECK(cl_err);
            current_gpu_bufs.d_MH_hxd = create_sub_buffer(agg_d_MH_hxd, proj_mat_bytes_ph); CL_CHECK(cl_err);
            current_gpu_bufs.d_MV_hxd = create_sub_buffer(agg_d_MV_hxd, proj_mat_bytes_ph); CL_CHECK(cl_err);
            current_gpu_bufs.d_dh = create_sub_buffer(agg_d_dh, embed_bytes_ph); CL_CHECK(cl_err);
            current_gpu_bufs.d_dv = create_sub_buffer(agg_d_dv, embed_bytes_ph); CL_CHECK(cl_err);
            current_gpu_bufs.d_EH = create_sub_buffer(agg_d_EH, embed_bytes_ph); CL_CHECK(cl_err);
            current_gpu_bufs.d_ver_accumulated_ev = create_sub_buffer(agg_d_ver_accumulated_ev, embed_bytes_ph); CL_CHECK(cl_err);
            current_gpu_bufs.d_hor_inputs = create_sub_buffer(agg_d_hor_inputs, embed_bytes_ph); CL_CHECK(cl_err);
            current_gpu_bufs.d_ver_inputs = create_sub_buffer(agg_d_ver_inputs, embed_bytes_ph); CL_CHECK(cl_err);
            current_gpu_bufs.d_hor_output = create_sub_buffer(agg_d_hor_output, embed_bytes_ph); CL_CHECK(cl_err);
            current_gpu_bufs.d_ver_output = create_sub_buffer(agg_d_ver_output, embed_bytes_ph); CL_CHECK(cl_err);
            current_gpu_bufs.d_relu_hor_output = create_sub_buffer(agg_d_relu_hor_output, embed_bytes_ph); CL_CHECK(cl_err);
            current_gpu_bufs.d_relu_ver_output = create_sub_buffer(agg_d_relu_ver_output, embed_bytes_ph); CL_CHECK(cl_err);
            current_gpu_bufs.d_mlp_bufferA_hor = create_sub_buffer(agg_d_mlp_bufferA_hor, embed_bytes_ph); CL_CHECK(cl_err);
            current_gpu_bufs.d_mlp_bufferB_hor = create_sub_buffer(agg_d_mlp_bufferB_hor, embed_bytes_ph); CL_CHECK(cl_err);
            current_gpu_bufs.d_mlp_bufferA_ver = create_sub_buffer(agg_d_mlp_bufferA_ver, embed_bytes_ph); CL_CHECK(cl_err);
            current_gpu_bufs.d_mlp_bufferB_ver = create_sub_buffer(agg_d_mlp_bufferB_ver, embed_bytes_ph); CL_CHECK(cl_err);
            current_gpu_bufs.d_mlp_pre_activation = create_sub_buffer(agg_d_mlp_pre_activation, embed_bytes_ph); CL_CHECK(cl_err);

            float zero = 0.0f;
            CL_CHECK(current_queue.enqueueFillBuffer(current_gpu_bufs.d_dh_accum, zero, 0, accum_bytes_ph));
            CL_CHECK(current_queue.enqueueFillBuffer(current_gpu_bufs.d_dv_accum, zero, 0, accum_bytes_ph));

            if (n_tokens > 0) {
                CL_CHECK(current_queue.enqueueWriteBuffer(current_gpu_bufs.d_K, CL_FALSE, 0, k_bytes_ph, head_cpu.K.mapped_data));
                CL_CHECK(current_queue.enqueueWriteBuffer(current_gpu_bufs.d_Q, CL_FALSE, 0, q_bytes_ph, head_cpu.Q.mapped_data));
                CL_CHECK(current_queue.enqueueWriteBuffer(current_gpu_bufs.d_KdotQ, CL_FALSE, 0, kdotq_bytes_ph, head_cpu.KdotQ.mapped_data));
                CL_CHECK(current_queue.enqueueWriteBuffer(current_gpu_bufs.d_EV_processed_data, CL_FALSE, 0, ev_processed_bytes_ph, head_cpu.EV.mapped_data));
            }
            CL_CHECK(current_queue.enqueueWriteBuffer(current_gpu_bufs.d_MH_hxd, CL_FALSE, 0, proj_mat_bytes_ph, head_cpu.MH.mapped_data));
            CL_CHECK(current_queue.enqueueWriteBuffer(current_gpu_bufs.d_MV_hxd, CL_FALSE, 0, proj_mat_bytes_ph, head_cpu.MV.mapped_data));
            CL_CHECK(current_queue.enqueueWriteBuffer(current_gpu_bufs.d_EH, CL_FALSE, 0, embed_bytes_ph, head_cpu.EH.data()));

            // score normalisation
            const size_t local_work_size_1d = 256;
            cl::Kernel lota_kernel = get_kernel_with_check(context_obj, "clLOTA2dmasking");
            size_t totalElementsLOTA = static_cast<size_t>(n_tokens);
            if (totalElementsLOTA > 0) {
                size_t global_lota_raw = totalElementsLOTA;
                size_t local_lota_clamped = (std::min)(global_lota_raw, local_work_size_1d);
                if (local_lota_clamped == 0) local_lota_clamped = 1;
                size_t global_lota_padded = ((global_lota_raw + local_lota_clamped - 1) / local_lota_clamped) * local_lota_clamped;
                cl::NDRange global_lota(global_lota_padded);
                cl::NDRange local_lota(local_lota_clamped);
                CL_CHECK(lota_kernel.setArg(0, current_gpu_bufs.d_KdotQ));
                CL_CHECK(lota_kernel.setArg(1, current_gpu_bufs.d_head_attention));
                CL_CHECK(lota_kernel.setArg(2, CONTEXT_WIN));
                CL_CHECK(lota_kernel.setArg(3, CONTEXT_WIN));
                CL_CHECK(lota_kernel.setArg(4, n_tokens));
                cl_int cl_att_is_self_lota = isSelfAttention ? 1 : 0;
                CL_CHECK(lota_kernel.setArg(5, cl_att_is_self_lota));
                CL_CHECK(current_queue.enqueueNDRangeKernel(lota_kernel, cl::NullRange, global_lota, local_lota));
            }

            // accumulate weights row and column wise
            cl::Kernel sums_kernel = get_kernel_with_check(context_obj, "computeHeadSumsMaskedKernel");
            size_t global_sums_raw = static_cast<size_t>(n_tokens); 
            size_t global_sums_padded = ((global_sums_raw + local_work_size_1d - 1) / local_work_size_1d) * local_work_size_1d; 
            cl::NDRange global_sums(global_sums_padded); cl::NDRange local_sums(local_work_size_1d); 
            cl_int cl_isSelfAttention = head_cpu.isSelfAttention;
            CL_CHECK(sums_kernel.setArg(0, current_gpu_bufs.d_head_attention)); 
            CL_CHECK(sums_kernel.setArg(1, current_gpu_bufs.d_row_sums)); 
            CL_CHECK(sums_kernel.setArg(2, current_gpu_bufs.d_col_sums)); 
            CL_CHECK(sums_kernel.setArg(3, n_tokens)); 
            CL_CHECK(sums_kernel.setArg(4, cl_isSelfAttention));
            CL_CHECK(current_queue.enqueueNDRangeKernel(sums_kernel, cl::NullRange, global_sums, local_sums));

            // weighted pool
            cl::Kernel accum_kernel = get_kernel_with_check(context_obj, "accumulateWeightedVectorsKernel");
            size_t global_accum_raw = static_cast<size_t>(n_tokens); 
            size_t global_accum_padded = ((global_accum_raw + local_work_size_1d - 1) / local_work_size_1d) * local_work_size_1d; 
            cl::NDRange global_accum(global_accum_padded); cl::NDRange local_accum(local_work_size_1d);
            CL_CHECK(accum_kernel.setArg(0, current_gpu_bufs.d_row_sums)); 
            CL_CHECK(accum_kernel.setArg(1, current_gpu_bufs.d_col_sums)); 
            CL_CHECK(accum_kernel.setArg(2, current_gpu_bufs.d_K)); 
            CL_CHECK(accum_kernel.setArg(3, current_gpu_bufs.d_Q)); 
            CL_CHECK(accum_kernel.setArg(4, current_gpu_bufs.d_dh_accum)); 
            CL_CHECK(accum_kernel.setArg(5, current_gpu_bufs.d_dv_accum)); 
            CL_CHECK(accum_kernel.setArg(6, n_tokens)); 
            CL_CHECK(accum_kernel.setArg(7, h_attention));
            CL_CHECK(current_queue.enqueueNDRangeKernel(accum_kernel, cl::NullRange, global_accum, local_accum));

            // value propagation via EH and EV
            cl::Kernel proj_kernel = get_kernel_with_check(context_obj, "kernelLayerForward");
            size_t global_proj_raw = static_cast<size_t>(d_embedding); 
            size_t global_proj_padded = ((global_proj_raw + local_work_size_1d - 1) / local_work_size_1d) * local_work_size_1d; 
            cl::NDRange global_proj(global_proj_padded); cl::NDRange local_proj(local_work_size_1d);
            CL_CHECK(proj_kernel.setArg(3, h_attention)); 
            CL_CHECK(proj_kernel.setArg(4, d_embedding));
            CL_CHECK(proj_kernel.setArg(0, current_gpu_bufs.d_dh_accum)); 
            CL_CHECK(proj_kernel.setArg(1, current_gpu_bufs.d_MH_hxd)); 
            CL_CHECK(proj_kernel.setArg(2, current_gpu_bufs.d_dh)); 
            CL_CHECK(current_queue.enqueueNDRangeKernel(proj_kernel, cl::NullRange, global_proj, local_proj));
            CL_CHECK(proj_kernel.setArg(0, current_gpu_bufs.d_dv_accum)); 
            CL_CHECK(proj_kernel.setArg(1, current_gpu_bufs.d_MV_hxd)); 
            CL_CHECK(proj_kernel.setArg(2, current_gpu_bufs.d_dv)); 
            CL_CHECK(current_queue.enqueueNDRangeKernel(proj_kernel, cl::NullRange, global_proj, local_proj));

            // first residual connection
            cl::Kernel add_kernel = get_kernel_with_check(context_obj, "vectorAddKernel");
            size_t global_add_raw = static_cast<size_t>(d_embedding); 
            size_t global_add_padded = ((global_add_raw + local_work_size_1d - 1) / local_work_size_1d) * local_work_size_1d; 
            cl::NDRange global_add(global_add_padded); cl::NDRange local_add(local_work_size_1d); 
            CL_CHECK(add_kernel.setArg(3, d_embedding));
            CL_CHECK(add_kernel.setArg(0, current_gpu_bufs.d_EH)); 
            CL_CHECK(add_kernel.setArg(1, current_gpu_bufs.d_dh)); 
            CL_CHECK(add_kernel.setArg(2, current_gpu_bufs.d_hor_inputs)); 
            CL_CHECK(current_queue.enqueueNDRangeKernel(add_kernel, cl::NullRange, global_add, local_add));
            cl::Kernel accum_ev_kernel = get_kernel_with_check(context_obj, "accumulateEVRowsKernelCL");
            CL_CHECK(accum_ev_kernel.setArg(0, current_gpu_bufs.d_EV_processed_data)); 
            CL_CHECK(accum_ev_kernel.setArg(1, current_gpu_bufs.d_ver_accumulated_ev)); 
            CL_CHECK(accum_ev_kernel.setArg(2, n_tokens)); 
            CL_CHECK(accum_ev_kernel.setArg(3, d_embedding));
            CL_CHECK(current_queue.enqueueNDRangeKernel(accum_ev_kernel, cl::NullRange, global_add, local_add));
            CL_CHECK(add_kernel.setArg(0, current_gpu_bufs.d_ver_accumulated_ev)); 
            CL_CHECK(add_kernel.setArg(1, current_gpu_bufs.d_dv)); 
            CL_CHECK(add_kernel.setArg(2, current_gpu_bufs.d_ver_inputs)); 
            CL_CHECK(current_queue.enqueueNDRangeKernel(add_kernel, cl::NullRange, global_add, local_add));

            // mlp forprop
            cl::Kernel mlp_fwd_kernel = context_obj.kernels.at("kernelLayerForward");
            cl::Kernel sigmoid_kernel = get_kernel_with_check(context_obj, "clSigmoid1d");
            CL_CHECK(current_queue.enqueueCopyBuffer(current_gpu_bufs.d_hor_inputs, current_gpu_bufs.d_mlp_bufferA_hor, 0, 0, embed_bytes_ph));
            CL_CHECK(current_queue.enqueueCopyBuffer(current_gpu_bufs.d_ver_inputs, current_gpu_bufs.d_mlp_bufferA_ver, 0, 0, embed_bytes_ph));
            cl::Buffer& current_in_hor = current_gpu_bufs.d_mlp_bufferA_hor; 
            cl::Buffer& current_out_hor = current_gpu_bufs.d_mlp_bufferB_hor;
            cl::Buffer& current_in_ver = current_gpu_bufs.d_mlp_bufferA_ver; 
            cl::Buffer& current_out_ver = current_gpu_bufs.d_mlp_bufferB_ver;
            size_t num_weight_matrices = head_cpu.hor.weights.size();
            size_t global_mlp_raw = static_cast<size_t>(d_embedding); 
            size_t global_mlp_padded = ((global_mlp_raw + local_work_size_1d - 1) / local_work_size_1d) * local_work_size_1d; 
            cl::NDRange global_mlp(global_mlp_padded); cl::NDRange local_mlp(local_work_size_1d);

            for (size_t layer_idx_mlp = 0; layer_idx_mlp < num_weight_matrices; ++layer_idx_mlp) {
                bool is_last_layer_mlp = (layer_idx_mlp == num_weight_matrices - 1); 
                int input_size_mlp = d_embedding; int output_size_mlp = d_embedding;
                CL_CHECK(mlp_fwd_kernel.setArg(3, input_size_mlp)); 
                CL_CHECK(mlp_fwd_kernel.setArg(4, output_size_mlp));

                { 
                    mat& current_weights_mat = head_cpu.hor.weights[layer_idx_mlp]; 
                    if (static_cast<int>(current_weights_mat.row) != output_size_mlp || static_cast<int>(current_weights_mat.col) != input_size_mlp) {
                        throw std::runtime_error("FATAL ERROR in cl1parallelForprop (first block): head_cpu.hor.weights[" + std::to_string(layer_idx_mlp) +
                                                 "] dimension mismatch for head [" + std::to_string(layer_idx) + "][" + std::to_string(col_idx_param) + "]." +
                                                 " Expected " + std::to_string(output_size_mlp) + "x" + std::to_string(input_size_mlp) +
                                                 ", got " + std::to_string(current_weights_mat.row) + "x" + std::to_string(current_weights_mat.col));
                    }

                    if (current_weights_mat.mapped_data == nullptr) {
                        std::cerr << "FATAL ERROR in cl1parallelForprop (first block): head_cpu.hor.weights[" << layer_idx_mlp 
                                  << "].mapped_data is nullptr for head [" << layer_idx << "][" << col_idx_param << "]."
                                  << std::endl;
                        std::cerr << "  Head Index (layer_idx): " << layer_idx << ", Column Index (col_idx_param): " << col_idx_param << std::endl;
                        std::cerr << "  MLP Layer (layer_idx_mlp): " << layer_idx_mlp << std::endl;
                        std::cerr << "  Mat Details: rows=" << current_weights_mat.row << ", cols=" << current_weights_mat.col;
                        std::cerr << std::endl;
                        throw std::runtime_error("Null mapped_data encountered for MLP weights.");
                    }

                    size_t weights_bytes_mlp = static_cast<size_t>(current_weights_mat.row) * current_weights_mat.col * sizeof(float); 
                    cl::Buffer d_mlp_weights(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, weights_bytes_mlp, current_weights_mat.mapped_data, &cl_err); CL_CHECK(cl_err); 
                    cl::Buffer& target_output_buffer = is_last_layer_mlp ? current_gpu_bufs.d_hor_output : current_gpu_bufs.d_mlp_pre_activation; 
                    CL_CHECK(mlp_fwd_kernel.setArg(0, current_in_hor)); 
                    CL_CHECK(mlp_fwd_kernel.setArg(1, d_mlp_weights)); 
                    CL_CHECK(mlp_fwd_kernel.setArg(2, target_output_buffer));
                    CL_CHECK(current_queue.enqueueNDRangeKernel(mlp_fwd_kernel, cl::NullRange, global_mlp, local_mlp)); 

                    if (!is_last_layer_mlp) {
                        CL_CHECK(sigmoid_kernel.setArg(0, current_gpu_bufs.d_mlp_pre_activation)); 
                        CL_CHECK(sigmoid_kernel.setArg(1, current_out_hor)); 
                        CL_CHECK(sigmoid_kernel.setArg(2, output_size_mlp)); 
                        CL_CHECK(current_queue.enqueueNDRangeKernel(sigmoid_kernel, cl::NullRange, global_mlp, local_mlp)); 
                        std::swap(current_in_hor, current_out_hor); 
                    }
                }

                { 
                    mat& current_weights_mat = head_cpu.ver.weights[layer_idx_mlp]; 
                    if (static_cast<int>(current_weights_mat.row) != output_size_mlp || static_cast<int>(current_weights_mat.col) != input_size_mlp) {
                        throw std::runtime_error("FATAL ERROR in cl1parallelForprop (first block): head_cpu.ver.weights[" + std::to_string(layer_idx_mlp) +
                                                 "] dimension mismatch for head [" + std::to_string(layer_idx) + "][" + std::to_string(col_idx_param) + "]." +
                                                 " Expected " + std::to_string(output_size_mlp) + "x" + std::to_string(input_size_mlp) +
                                                 ", got " + std::to_string(current_weights_mat.row) + "x" + std::to_string(current_weights_mat.col));
                    }

                    if (current_weights_mat.mapped_data == nullptr) {
                        std::cerr << "FATAL ERROR in cl1parallelForprop (first block): head_cpu.ver.weights[" << layer_idx_mlp 
                                  << "].mapped_data is nullptr for head [" << layer_idx << "][" << col_idx_param << "]."
                                  << std::endl;
                        std::cerr << "  Head Index (layer_idx): " << layer_idx << ", Column Index (col_idx_param): " << col_idx_param << std::endl;
                        std::cerr << "  MLP Layer (layer_idx_mlp): " << layer_idx_mlp << std::endl;
                        std::cerr << "  Mat Details: rows=" << current_weights_mat.row << ", cols=" << current_weights_mat.col;
                        std::cerr << std::endl;
                        throw std::runtime_error("Null mapped_data encountered for MLP weights (vertical).");
                    }

                    cl::Buffer& d_mlp_weights = all_d_ver_mlp_weights[layer_idx][layer_idx_mlp];
                    cl::Buffer& target_output_buffer = is_last_layer_mlp ? current_gpu_bufs.d_ver_output : current_gpu_bufs.d_mlp_pre_activation; 
                    CL_CHECK(mlp_fwd_kernel.setArg(0, current_in_ver)); 
                    CL_CHECK(mlp_fwd_kernel.setArg(1, d_mlp_weights)); 
                    CL_CHECK(mlp_fwd_kernel.setArg(2, target_output_buffer)); 
                    CL_CHECK(current_queue.enqueueNDRangeKernel(mlp_fwd_kernel, cl::NullRange, global_mlp, local_mlp));

                    if (!is_last_layer_mlp) { 
                        CL_CHECK(sigmoid_kernel.setArg(0, current_gpu_bufs.d_mlp_pre_activation)); 
                        CL_CHECK(sigmoid_kernel.setArg(1, current_out_ver)); 
                        CL_CHECK(sigmoid_kernel.setArg(2, output_size_mlp)); 
                        CL_CHECK(current_queue.enqueueNDRangeKernel(sigmoid_kernel, cl::NullRange, global_mlp, local_mlp)); 
                        std::swap(current_in_ver, current_out_ver); 
                    }
                }
            }

            // relu mlp outputs
            cl::Kernel relu_kernel = get_kernel_with_check(context_obj, "clReLU1d"); 
            CL_CHECK(relu_kernel.setArg(2, d_embedding));
            CL_CHECK(relu_kernel.setArg(0, current_gpu_bufs.d_hor_output)); CL_CHECK(relu_kernel.setArg(1, current_gpu_bufs.d_relu_hor_output)); 
            CL_CHECK(current_queue.enqueueNDRangeKernel(relu_kernel, cl::NullRange, global_add, local_add));
            CL_CHECK(relu_kernel.setArg(0, current_gpu_bufs.d_ver_output)); CL_CHECK(relu_kernel.setArg(1, current_gpu_bufs.d_relu_ver_output)); 
            CL_CHECK(current_queue.enqueueNDRangeKernel(relu_kernel, cl::NullRange, global_add, local_add));

            // Second Residual Connection
            // EH
            CL_CHECK(add_kernel.setArg(0, current_gpu_bufs.d_EH)); 
            CL_CHECK(add_kernel.setArg(1, current_gpu_bufs.d_relu_hor_output)); 
            CL_CHECK(add_kernel.setArg(2, current_gpu_bufs.d_EH)); 
            CL_CHECK(current_queue.enqueueNDRangeKernel(add_kernel, cl::NullRange, global_add, local_add));
            // EV
            cl::Kernel update_ev_kernel = get_kernel_with_check(context_obj, "updateEVRowsKernelCL");
            size_t global_update_ev_raw = static_cast<size_t>(n_tokens);
            size_t global_update_ev_padded = ((global_update_ev_raw + local_work_size_1d - 1) / local_work_size_1d) * local_work_size_1d; 
            cl::NDRange global_update_ev(global_update_ev_padded);
            cl::NDRange local_update_ev(local_work_size_1d);
            CL_CHECK(update_ev_kernel.setArg(0, current_gpu_bufs.d_EV_processed_data)); 
            CL_CHECK(update_ev_kernel.setArg(1, current_gpu_bufs.d_relu_ver_output)); 
            CL_CHECK(update_ev_kernel.setArg(2, n_tokens)); 
            CL_CHECK(update_ev_kernel.setArg(3, d_embedding));
            CL_CHECK(current_queue.enqueueNDRangeKernel(update_ev_kernel, cl::NullRange, global_update_ev, local_update_ev));

            CL_CHECK(current_queue.enqueueReadBuffer(current_gpu_bufs.d_EH, CL_FALSE, 0, embed_bytes_ph, head_cpu.EH.data()));
            if (n_tokens > 0) {
                CL_CHECK(current_queue.enqueueReadBuffer(current_gpu_bufs.d_EV_processed_data, CL_FALSE, 0, ev_processed_bytes_ph, head_cpu.EV.mapped_data));
            }
            current_gpu_bufs.d_EH = cl::Buffer();
            current_gpu_bufs.d_EV_processed_data = cl::Buffer();

            CL_CHECK(current_queue.finish());
        }
        catch (const std::exception& e) {
            // Add context to the exception before re-throwing
            throw std::runtime_error("Exception in attention::clforprop (first block overload) for head ["
                                     + std::to_string(layer_idx) + "][" + std::to_string(col_idx_param) + "]: " + e.what());
        }
    }
    // Final synchronization for all heads in this column
    CL_CHECK(queue.finish());

    agg_d_K = cl::Buffer();
    agg_d_Q = cl::Buffer();
    agg_d_KdotQ = cl::Buffer();
    agg_d_head_attention = cl::Buffer();
    agg_d_row_sums = cl::Buffer();
    agg_d_col_sums = cl::Buffer();
    agg_d_EV_processed_data = cl::Buffer();
    agg_d_dh_accum = cl::Buffer();
    agg_d_dv_accum = cl::Buffer();
    agg_d_MH_hxd = cl::Buffer();
    agg_d_MV_hxd = cl::Buffer();
    agg_d_dh = cl::Buffer();
    agg_d_dv = cl::Buffer();
    agg_d_EH = cl::Buffer();
    agg_d_ver_accumulated_ev = cl::Buffer();
    agg_d_hor_inputs = cl::Buffer();
    agg_d_ver_inputs = cl::Buffer();
    agg_d_hor_output = cl::Buffer();
    agg_d_ver_output = cl::Buffer();
    agg_d_relu_hor_output = cl::Buffer();
    agg_d_relu_ver_output = cl::Buffer();
    agg_d_mlp_bufferA_hor = cl::Buffer();
    agg_d_mlp_bufferB_hor = cl::Buffer();
    agg_d_mlp_bufferA_ver = cl::Buffer();
    agg_d_mlp_bufferB_ver = cl::Buffer();
    agg_d_mlp_pre_activation = cl::Buffer();
}



/**
 * @brief OpenCL forward propagation on single ith column of the FIRST block.
 * @param context OpenCL context.
 * @param queue OpenCL command queue.
 * @param kernels Map of compiled OpenCL kernels.
 * @param in dimension size (maps to d_embedding in attention::clforprop)
 * @param tokenCount number of tokens (maps to currentTokenCount in attention::clforprop)
 * @param i column index (0-based)
 * @param layers number of layers in mlp (maps to layers_mlp in attention::clforprop)
 */
void block::cl1parallelForpropev(int& in_dim_param, int& tokenCount_param, int col_idx_param, int& layers_mlp_param)
{
    if (col_idx_param < 0 || col_idx_param >= y) {
        throw std::out_of_range("cl1parallelForprop (first block): column index 'i' (" + std::to_string(col_idx_param) + ") is out of range [0, " + std::to_string(y - 1) + "].");
    }

    cl_int cl_err; // For OpenCL error codes
    OpenCLContext& context_obj = clcontext;
    cl::Context context = context_obj.context;

    const int num_heads_in_col = x;
    const int d_embedding = EMBEDDING;
    const int h_attention = CONTEXT_WIN;
    const int n_tokens = tokenCount_param;

    // Per-head byte sizes (assuming n_tokens is fixed for all heads in this call)
    size_t k_bytes_ph = static_cast<size_t>(CONTEXT_WIN) * CONTEXT_WIN * sizeof(float);
    size_t q_bytes_ph = static_cast<size_t>(CONTEXT_WIN) * CONTEXT_WIN * sizeof(float);
    size_t kdotq_bytes_ph = static_cast<size_t>(CONTEXT_WIN) * CONTEXT_WIN * sizeof(float);
    size_t head_bytes_ph = static_cast<size_t>(CONTEXT_WIN) * CONTEXT_WIN * sizeof(float);
    size_t sums_bytes_ph = static_cast<size_t>(CONTEXT_WIN) * sizeof(float);
    size_t accum_bytes_ph = static_cast<size_t>(CONTEXT_WIN) * sizeof(float);
    size_t proj_mat_bytes_ph = static_cast<size_t>(EMBEDDING) * CONTEXT_WIN * sizeof(float);
    size_t ev_processed_bytes_ph = static_cast<size_t>(CONTEXT_WIN) * EMBEDDING * sizeof(float);
    size_t embed_bytes_ph = static_cast<size_t>(EMBEDDING) * sizeof(float);

    // --- Aggregate Buffer Allocation ---
    cl::Buffer agg_d_K, agg_d_Q, agg_d_KdotQ, agg_d_head_attention;
    cl::Buffer agg_d_col_sums;
    cl::Buffer agg_d_dv_accum;
    cl::Buffer agg_d_MV_hxd;
    cl::Buffer agg_d_dv;
    cl::Buffer agg_d_EV_processed_data;
    cl::Buffer agg_d_ver_accumulated_ev;
    cl::Buffer agg_d_ver_inputs;
    cl::Buffer agg_d_ver_output;
    cl::Buffer agg_d_relu_ver_output;
    cl::Buffer agg_d_mlp_bufferA_ver, agg_d_mlp_bufferB_ver;
    cl::Buffer agg_d_mlp_pre_activation;

    if (n_tokens > 0) {
        agg_d_K = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col * k_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_Q = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col * q_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_KdotQ = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col * kdotq_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_head_attention = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col * head_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_col_sums = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col * sums_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_EV_processed_data = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col * ev_processed_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
    }
    agg_d_dv_accum = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col * accum_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
    agg_d_MV_hxd = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col * proj_mat_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
    agg_d_dv = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col * embed_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
    agg_d_ver_accumulated_ev = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col * embed_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
    agg_d_ver_inputs = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col * embed_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
    agg_d_ver_output = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col * embed_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
    agg_d_relu_ver_output = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col * embed_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
    agg_d_mlp_bufferA_ver = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col * embed_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
    agg_d_mlp_bufferB_ver = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col * embed_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
    agg_d_mlp_pre_activation = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col * embed_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
    
    cl::CommandQueue queue(context, context_obj.device, 0, &cl_err); CL_CHECK(cl_err);
    std::vector<HeadForwardDeviceBuffersEVCL> head_gpu_data(num_heads_in_col);

    // --- Pre-allocate MLP weight buffers to avoid creation in the loop ---
    std::vector<std::vector<cl::Buffer>> all_d_hor_mlp_weights(num_heads_in_col);
    std::vector<std::vector<cl::Buffer>> all_d_ver_mlp_weights(num_heads_in_col);
    for (int layer_idx = 0; layer_idx < num_heads_in_col; ++layer_idx) {
        attention& head_cpu = b[layer_idx][col_idx_param];
        size_t num_weight_matrices = head_cpu.hor.weights.size();
        all_d_hor_mlp_weights[layer_idx].resize(num_weight_matrices);
        all_d_ver_mlp_weights[layer_idx].resize(num_weight_matrices);

        for (size_t mlp_layer = 0; mlp_layer < num_weight_matrices; ++mlp_layer) {
            mat& hor_weights_mat = head_cpu.hor.weights[mlp_layer];
            size_t weights_bytes = static_cast<size_t>(hor_weights_mat.row) * hor_weights_mat.col * sizeof(float);
            all_d_hor_mlp_weights[layer_idx][mlp_layer] = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, weights_bytes, hor_weights_mat.mapped_data, &cl_err); CL_CHECK(cl_err);

            mat& ver_weights_mat = head_cpu.ver.weights[mlp_layer];
            weights_bytes = static_cast<size_t>(ver_weights_mat.row) * ver_weights_mat.col * sizeof(float);
            all_d_ver_mlp_weights[layer_idx][mlp_layer] = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, weights_bytes, ver_weights_mat.mapped_data, &cl_err); CL_CHECK(cl_err);
        }
    }

    for (int layer_idx = 0; layer_idx < num_heads_in_col; ++layer_idx) {
        attention& head_cpu = b[layer_idx][col_idx_param];
        cl::CommandQueue& current_queue = queue;
        HeadForwardDeviceBuffersEVCL& current_gpu_bufs = head_gpu_data[layer_idx];

        const int num_ev_rows_to_process = CONTEXT_WIN;

        if (n_tokens <= 0) {
            std::fill(head_cpu.EH.begin(), head_cpu.EH.end(), 0.0f);
            if (n_tokens == 0 && head_cpu.EV.mapped_data && head_cpu.EV.row > 0 && head_cpu.EV.col == d_embedding) {
                std::fill_n(head_cpu.EV.mapped_data, head_cpu.EV.col, 0.0f);
            }
            continue;
        }

        // --- Basic Validation ---
        if (head_cpu.EV.row < n_tokens || head_cpu.EV.col != d_embedding) {
            throw std::runtime_error("EV matrix not properly sized for current tokenCount n in cl1parallelForprop (first block) for head [" + std::to_string(layer_idx) + "][" + std::to_string(col_idx_param) + "]. Expected rows >= " + std::to_string(n_tokens) + " and cols = " + std::to_string(d_embedding) +
                                     ", but got rows = " + std::to_string(head_cpu.EV.row) + " and cols = " + std::to_string(head_cpu.EV.col));
        }
        if (head_cpu.K.row != CONTEXT_WIN || head_cpu.K.col != CONTEXT_WIN ||
            head_cpu.Q.row != CONTEXT_WIN || head_cpu.Q.col != CONTEXT_WIN ||
            head_cpu.KdotQ.row != CONTEXT_WIN || head_cpu.KdotQ.col != CONTEXT_WIN ||
            head_cpu.MH.row != EMBEDDING || head_cpu.MH.col != CONTEXT_WIN ||
            head_cpu.MV.row != EMBEDDING || head_cpu.MV.col != CONTEXT_WIN ||
            head_cpu.EH.size() != static_cast<size_t>(d_embedding) ||
            head_cpu.hor.hlayers.empty() || head_cpu.ver.hlayers.empty() || head_cpu.hor.weights.empty() || head_cpu.ver.weights.empty())
        {
            throw std::runtime_error("Attention component dimension mismatch or uninitialized member in cl1parallelForprop (first block) for head [" +
                                     std::to_string(layer_idx) + "][" + std::to_string(col_idx_param) + "]. K.row=" + std::to_string(head_cpu.K.row) + ", n=" + std::to_string(n_tokens));
        }
        if (d_embedding != in_dim_param) {
            throw std::runtime_error("Embedding dimension mismatch (EMBEDDING vs in) for head [" + std::to_string(layer_idx) + "][" + std::to_string(col_idx_param) + "].");
        }
        if (head_cpu.hor.hlayers[0].size() != static_cast<size_t>(d_embedding) || head_cpu.ver.hlayers[0].size() != static_cast<size_t>(d_embedding) ||
            head_cpu.hor.weights.back().row != static_cast<size_t>(d_embedding) || head_cpu.ver.weights.back().row != static_cast<size_t>(d_embedding)) {
             throw std::runtime_error("MLP input/output layer dimension mismatch with 'd' for head [" + std::to_string(layer_idx) + "][" + std::to_string(col_idx_param) + "].");
        }

        try {
            auto create_sub_buffer = [&](cl::Buffer& parent_agg_buf, size_t per_head_bytes) {
                cl_buffer_region region = {static_cast<size_t>(layer_idx) * per_head_bytes, per_head_bytes};
                return parent_agg_buf.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err);
            };
            HeadForwardDeviceBuffersCL current_gpu_bufs;

            if (n_tokens > 0) {
                current_gpu_bufs.d_K = create_sub_buffer(agg_d_K, k_bytes_ph); CL_CHECK(cl_err);
                current_gpu_bufs.d_Q = create_sub_buffer(agg_d_Q, q_bytes_ph); CL_CHECK(cl_err);
                current_gpu_bufs.d_KdotQ = create_sub_buffer(agg_d_KdotQ, kdotq_bytes_ph); CL_CHECK(cl_err);
                current_gpu_bufs.d_head_attention = create_sub_buffer(agg_d_head_attention, head_bytes_ph); CL_CHECK(cl_err);
                current_gpu_bufs.d_col_sums = create_sub_buffer(agg_d_col_sums, sums_bytes_ph); CL_CHECK(cl_err);
                current_gpu_bufs.d_EV_processed_data = create_sub_buffer(agg_d_EV_processed_data, ev_processed_bytes_ph); CL_CHECK(cl_err);
            }
            current_gpu_bufs.d_dv_accum = create_sub_buffer(agg_d_dv_accum, accum_bytes_ph); CL_CHECK(cl_err);
            current_gpu_bufs.d_MV_hxd = create_sub_buffer(agg_d_MV_hxd, proj_mat_bytes_ph); CL_CHECK(cl_err);
            current_gpu_bufs.d_dv = create_sub_buffer(agg_d_dv, embed_bytes_ph); CL_CHECK(cl_err);
            current_gpu_bufs.d_ver_accumulated_ev = create_sub_buffer(agg_d_ver_accumulated_ev, embed_bytes_ph); CL_CHECK(cl_err);
            current_gpu_bufs.d_ver_inputs = create_sub_buffer(agg_d_ver_inputs, embed_bytes_ph); CL_CHECK(cl_err);
            current_gpu_bufs.d_ver_output = create_sub_buffer(agg_d_ver_output, embed_bytes_ph); CL_CHECK(cl_err);
            current_gpu_bufs.d_relu_ver_output = create_sub_buffer(agg_d_relu_ver_output, embed_bytes_ph); CL_CHECK(cl_err);
            current_gpu_bufs.d_mlp_bufferA_ver = create_sub_buffer(agg_d_mlp_bufferA_ver, embed_bytes_ph); CL_CHECK(cl_err);
            current_gpu_bufs.d_mlp_bufferB_ver = create_sub_buffer(agg_d_mlp_bufferB_ver, embed_bytes_ph); CL_CHECK(cl_err);
            current_gpu_bufs.d_mlp_pre_activation = create_sub_buffer(agg_d_mlp_pre_activation, embed_bytes_ph); CL_CHECK(cl_err);

            float zero = 0.0f;
            CL_CHECK(current_queue.enqueueFillBuffer(current_gpu_bufs.d_dh_accum, zero, 0, accum_bytes_ph));
            CL_CHECK(current_queue.enqueueFillBuffer(current_gpu_bufs.d_dv_accum, zero, 0, accum_bytes_ph));

            if (n_tokens > 0) {
                CL_CHECK(current_queue.enqueueWriteBuffer(current_gpu_bufs.d_K, CL_FALSE, 0, k_bytes_ph, head_cpu.K.mapped_data));
                CL_CHECK(current_queue.enqueueWriteBuffer(current_gpu_bufs.d_Q, CL_FALSE, 0, q_bytes_ph, head_cpu.Q.mapped_data));
                CL_CHECK(current_queue.enqueueWriteBuffer(current_gpu_bufs.d_KdotQ, CL_FALSE, 0, kdotq_bytes_ph, head_cpu.KdotQ.mapped_data));
                CL_CHECK(current_queue.enqueueWriteBuffer(current_gpu_bufs.d_EV_processed_data, CL_FALSE, 0, ev_processed_bytes_ph, head_cpu.EV.mapped_data));
            }
            CL_CHECK(current_queue.enqueueWriteBuffer(current_gpu_bufs.d_MH_hxd, CL_FALSE, 0, proj_mat_bytes_ph, head_cpu.MH.mapped_data));
            CL_CHECK(current_queue.enqueueWriteBuffer(current_gpu_bufs.d_MV_hxd, CL_FALSE, 0, proj_mat_bytes_ph, head_cpu.MV.mapped_data));
            CL_CHECK(current_queue.enqueueWriteBuffer(current_gpu_bufs.d_EH, CL_FALSE, 0, embed_bytes_ph, head_cpu.EH.data()));

            // score normalisation
            const size_t local_work_size_1d = 256;
            cl::Kernel lota_kernel = get_kernel_with_check(context_obj, "clLOTA2dmasking");
            size_t totalElementsLOTA = static_cast<size_t>(n_tokens);
            if (totalElementsLOTA > 0) {
                size_t global_lota_raw = totalElementsLOTA;
                size_t local_lota_clamped = (std::min)(global_lota_raw, local_work_size_1d);
                if (local_lota_clamped == 0) local_lota_clamped = 1;
                size_t global_lota_padded = ((global_lota_raw + local_lota_clamped - 1) / local_lota_clamped) * local_lota_clamped;
                cl::NDRange global_lota(global_lota_padded);
                cl::NDRange local_lota(local_lota_clamped);
                CL_CHECK(lota_kernel.setArg(0, current_gpu_bufs.d_KdotQ));
                CL_CHECK(lota_kernel.setArg(1, current_gpu_bufs.d_head_attention));
                CL_CHECK(lota_kernel.setArg(2, CONTEXT_WIN));
                CL_CHECK(lota_kernel.setArg(3, CONTEXT_WIN));
                CL_CHECK(lota_kernel.setArg(4, n_tokens));
                cl_int cl_att_is_self_lota = isSelfAttention ? 1 : 0;
                CL_CHECK(lota_kernel.setArg(5, cl_att_is_self_lota));
                CL_CHECK(current_queue.enqueueNDRangeKernel(lota_kernel, cl::NullRange, global_lota, local_lota));
            }

            // accumulate weights row and column wise
            cl::Kernel sums_kernel = get_kernel_with_check(context_obj, "computeHeadSumsMaskedKernel");
            size_t global_sums_raw = static_cast<size_t>(n_tokens); 
            size_t global_sums_padded = ((global_sums_raw + local_work_size_1d - 1) / local_work_size_1d) * local_work_size_1d; 
            cl::NDRange global_sums(global_sums_padded); cl::NDRange local_sums(local_work_size_1d); 
            cl_int cl_isSelfAttention = head_cpu.isSelfAttention;
            CL_CHECK(sums_kernel.setArg(0, current_gpu_bufs.d_head_attention)); 
            CL_CHECK(sums_kernel.setArg(1, current_gpu_bufs.d_row_sums)); 
            CL_CHECK(sums_kernel.setArg(2, current_gpu_bufs.d_col_sums)); 
            CL_CHECK(sums_kernel.setArg(3, n_tokens)); 
            CL_CHECK(sums_kernel.setArg(4, cl_isSelfAttention));
            CL_CHECK(current_queue.enqueueNDRangeKernel(sums_kernel, cl::NullRange, global_sums, local_sums));

            // weighted pool
            cl::Kernel accum_kernel = get_kernel_with_check(context_obj, "accumulateWeightedVectorsKernel");
            size_t global_accum_raw = static_cast<size_t>(n_tokens); 
            size_t global_accum_padded = ((global_accum_raw + local_work_size_1d - 1) / local_work_size_1d) * local_work_size_1d; 
            cl::NDRange global_accum(global_accum_padded); cl::NDRange local_accum(local_work_size_1d);
            CL_CHECK(accum_kernel.setArg(0, current_gpu_bufs.d_row_sums)); 
            CL_CHECK(accum_kernel.setArg(1, current_gpu_bufs.d_col_sums)); 
            CL_CHECK(accum_kernel.setArg(2, current_gpu_bufs.d_K)); 
            CL_CHECK(accum_kernel.setArg(3, current_gpu_bufs.d_Q)); 
            CL_CHECK(accum_kernel.setArg(4, current_gpu_bufs.d_dh_accum)); 
            CL_CHECK(accum_kernel.setArg(5, current_gpu_bufs.d_dv_accum)); 
            CL_CHECK(accum_kernel.setArg(6, n_tokens)); 
            CL_CHECK(accum_kernel.setArg(7, h_attention));
            CL_CHECK(current_queue.enqueueNDRangeKernel(accum_kernel, cl::NullRange, global_accum, local_accum));

            // value propagation via EH and EV
            cl::Kernel proj_kernel = get_kernel_with_check(context_obj, "kernelLayerForward");
            size_t global_proj_raw = static_cast<size_t>(d_embedding); 
            size_t global_proj_padded = ((global_proj_raw + local_work_size_1d - 1) / local_work_size_1d) * local_work_size_1d; 
            cl::NDRange global_proj(global_proj_padded); cl::NDRange local_proj(local_work_size_1d);
            CL_CHECK(proj_kernel.setArg(3, h_attention)); 
            CL_CHECK(proj_kernel.setArg(4, d_embedding));
            CL_CHECK(proj_kernel.setArg(0, current_gpu_bufs.d_dh_accum)); 
            CL_CHECK(proj_kernel.setArg(1, current_gpu_bufs.d_MH_hxd)); 
            CL_CHECK(proj_kernel.setArg(2, current_gpu_bufs.d_dh)); 
            CL_CHECK(current_queue.enqueueNDRangeKernel(proj_kernel, cl::NullRange, global_proj, local_proj));
            CL_CHECK(proj_kernel.setArg(0, current_gpu_bufs.d_dv_accum)); 
            CL_CHECK(proj_kernel.setArg(1, current_gpu_bufs.d_MV_hxd)); 
            CL_CHECK(proj_kernel.setArg(2, current_gpu_bufs.d_dv)); 
            CL_CHECK(current_queue.enqueueNDRangeKernel(proj_kernel, cl::NullRange, global_proj, local_proj));

            // first residual connection
            cl::Kernel add_kernel = get_kernel_with_check(context_obj, "vectorAddKernel");
            size_t global_add_raw = static_cast<size_t>(d_embedding); 
            size_t global_add_padded = ((global_add_raw + local_work_size_1d - 1) / local_work_size_1d) * local_work_size_1d; 
            cl::NDRange global_add(global_add_padded); cl::NDRange local_add(local_work_size_1d); 
            CL_CHECK(add_kernel.setArg(3, d_embedding));
            CL_CHECK(add_kernel.setArg(0, current_gpu_bufs.d_EH)); 
            CL_CHECK(add_kernel.setArg(1, current_gpu_bufs.d_dh)); 
            CL_CHECK(add_kernel.setArg(2, current_gpu_bufs.d_hor_inputs)); 
            CL_CHECK(current_queue.enqueueNDRangeKernel(add_kernel, cl::NullRange, global_add, local_add));
            cl::Kernel accum_ev_kernel = get_kernel_with_check(context_obj, "accumulateEVRowsKernelCL");
            CL_CHECK(accum_ev_kernel.setArg(0, current_gpu_bufs.d_EV_processed_data)); 
            CL_CHECK(accum_ev_kernel.setArg(1, current_gpu_bufs.d_ver_accumulated_ev)); 
            CL_CHECK(accum_ev_kernel.setArg(2, n_tokens)); 
            CL_CHECK(accum_ev_kernel.setArg(3, d_embedding));
            CL_CHECK(current_queue.enqueueNDRangeKernel(accum_ev_kernel, cl::NullRange, global_add, local_add));
            CL_CHECK(add_kernel.setArg(0, current_gpu_bufs.d_ver_accumulated_ev)); 
            CL_CHECK(add_kernel.setArg(1, current_gpu_bufs.d_dv)); 
            CL_CHECK(add_kernel.setArg(2, current_gpu_bufs.d_ver_inputs)); 
            CL_CHECK(current_queue.enqueueNDRangeKernel(add_kernel, cl::NullRange, global_add, local_add));

            // mlp forprop
            cl::Kernel mlp_fwd_kernel = context_obj.kernels.at("kernelLayerForward");
            cl::Kernel sigmoid_kernel = get_kernel_with_check(context_obj, "clSigmoid1d");
            CL_CHECK(current_queue.enqueueCopyBuffer(current_gpu_bufs.d_hor_inputs, current_gpu_bufs.d_mlp_bufferA_hor, 0, 0, embed_bytes_ph));
            CL_CHECK(current_queue.enqueueCopyBuffer(current_gpu_bufs.d_ver_inputs, current_gpu_bufs.d_mlp_bufferA_ver, 0, 0, embed_bytes_ph));
            cl::Buffer& current_in_hor = current_gpu_bufs.d_mlp_bufferA_hor; 
            cl::Buffer& current_out_hor = current_gpu_bufs.d_mlp_bufferB_hor;
            cl::Buffer& current_in_ver = current_gpu_bufs.d_mlp_bufferA_ver; 
            cl::Buffer& current_out_ver = current_gpu_bufs.d_mlp_bufferB_ver;
            size_t num_weight_matrices = head_cpu.hor.weights.size();
            size_t global_mlp_raw = static_cast<size_t>(d_embedding); 
            size_t global_mlp_padded = ((global_mlp_raw + local_work_size_1d - 1) / local_work_size_1d) * local_work_size_1d; 
            cl::NDRange global_mlp(global_mlp_padded); cl::NDRange local_mlp(local_work_size_1d);

            for (size_t layer_idx_mlp = 0; layer_idx_mlp < num_weight_matrices; ++layer_idx_mlp) {
                bool is_last_layer_mlp = (layer_idx_mlp == num_weight_matrices - 1); 
                int input_size_mlp = d_embedding; int output_size_mlp = d_embedding;
                CL_CHECK(mlp_fwd_kernel.setArg(3, input_size_mlp)); 
                CL_CHECK(mlp_fwd_kernel.setArg(4, output_size_mlp));
                { 
                    mat& current_weights_mat = head_cpu.ver.weights[layer_idx_mlp]; 
                    if (static_cast<int>(current_weights_mat.row) != output_size_mlp || static_cast<int>(current_weights_mat.col) != input_size_mlp) {
                        throw std::runtime_error("FATAL ERROR in cl1parallelForprop (first block): head_cpu.ver.weights[" + std::to_string(layer_idx_mlp) +
                                                 "] dimension mismatch for head [" + std::to_string(layer_idx) + "][" + std::to_string(col_idx_param) + "]." +
                                                 " Expected " + std::to_string(output_size_mlp) + "x" + std::to_string(input_size_mlp) +
                                                 ", got " + std::to_string(current_weights_mat.row) + "x" + std::to_string(current_weights_mat.col));
                    }

                    if (current_weights_mat.mapped_data == nullptr) {
                        std::cerr << "FATAL ERROR in cl1parallelForprop (first block): head_cpu.ver.weights[" << layer_idx_mlp 
                                  << "].mapped_data is nullptr for head [" << layer_idx << "][" << col_idx_param << "]."
                                  << std::endl;
                        std::cerr << "  Head Index (layer_idx): " << layer_idx << ", Column Index (col_idx_param): " << col_idx_param << std::endl;
                        std::cerr << "  MLP Layer (layer_idx_mlp): " << layer_idx_mlp << std::endl;
                        std::cerr << "  Mat Details: rows=" << current_weights_mat.row << ", cols=" << current_weights_mat.col;
                        std::cerr << std::endl;
                        throw std::runtime_error("Null mapped_data encountered for MLP weights (vertical).");
                    }

                    cl::Buffer& d_mlp_weights = all_d_ver_mlp_weights[layer_idx][layer_idx_mlp];
                    cl::Buffer& target_output_buffer = is_last_layer_mlp ? current_gpu_bufs.d_ver_output : current_gpu_bufs.d_mlp_pre_activation; 
                    CL_CHECK(mlp_fwd_kernel.setArg(0, current_in_ver)); 
                    CL_CHECK(mlp_fwd_kernel.setArg(1, d_mlp_weights)); 
                    CL_CHECK(mlp_fwd_kernel.setArg(2, target_output_buffer)); 
                    CL_CHECK(current_queue.enqueueNDRangeKernel(mlp_fwd_kernel, cl::NullRange, global_mlp, local_mlp));

                    if (!is_last_layer_mlp) { 
                        CL_CHECK(sigmoid_kernel.setArg(0, current_gpu_bufs.d_mlp_pre_activation)); 
                        CL_CHECK(sigmoid_kernel.setArg(1, current_out_ver)); 
                        CL_CHECK(sigmoid_kernel.setArg(2, output_size_mlp)); 
                        CL_CHECK(current_queue.enqueueNDRangeKernel(sigmoid_kernel, cl::NullRange, global_mlp, local_mlp)); 
                        std::swap(current_in_ver, current_out_ver); 
                    }
                }
            }

            // relu mlp outputs
            cl::Kernel relu_kernel = get_kernel_with_check(context_obj, "clReLU1d"); 
            CL_CHECK(relu_kernel.setArg(2, d_embedding));
            CL_CHECK(relu_kernel.setArg(0, current_gpu_bufs.d_hor_output)); CL_CHECK(relu_kernel.setArg(1, current_gpu_bufs.d_relu_hor_output)); 
            CL_CHECK(current_queue.enqueueNDRangeKernel(relu_kernel, cl::NullRange, global_add, local_add));
            CL_CHECK(relu_kernel.setArg(0, current_gpu_bufs.d_ver_output)); CL_CHECK(relu_kernel.setArg(1, current_gpu_bufs.d_relu_ver_output)); 
            CL_CHECK(current_queue.enqueueNDRangeKernel(relu_kernel, cl::NullRange, global_add, local_add));

            // Second Residual Connection
            // EH
            CL_CHECK(add_kernel.setArg(0, current_gpu_bufs.d_EH)); 
            CL_CHECK(add_kernel.setArg(1, current_gpu_bufs.d_relu_hor_output)); 
            CL_CHECK(add_kernel.setArg(2, current_gpu_bufs.d_EH)); 
            CL_CHECK(current_queue.enqueueNDRangeKernel(add_kernel, cl::NullRange, global_add, local_add));
            // EV
            cl::Kernel update_ev_kernel = get_kernel_with_check(context_obj, "updateEVRowsKernelCL");
            size_t global_update_ev_raw = static_cast<size_t>(n_tokens);
            size_t global_update_ev_padded = ((global_update_ev_raw + local_work_size_1d - 1) / local_work_size_1d) * local_work_size_1d; 
            cl::NDRange global_update_ev(global_update_ev_padded);
            cl::NDRange local_update_ev(local_work_size_1d);
            CL_CHECK(update_ev_kernel.setArg(0, current_gpu_bufs.d_EV_processed_data)); 
            CL_CHECK(update_ev_kernel.setArg(1, current_gpu_bufs.d_relu_ver_output)); 
            CL_CHECK(update_ev_kernel.setArg(2, n_tokens)); 
            CL_CHECK(update_ev_kernel.setArg(3, d_embedding));
            CL_CHECK(current_queue.enqueueNDRangeKernel(update_ev_kernel, cl::NullRange, global_update_ev, local_update_ev));

            CL_CHECK(current_queue.enqueueReadBuffer(current_gpu_bufs.d_EH, CL_FALSE, 0, embed_bytes_ph, head_cpu.EH.data()));
            if (n_tokens > 0) {
                CL_CHECK(current_queue.enqueueReadBuffer(current_gpu_bufs.d_EV_processed_data, CL_FALSE, 0, ev_processed_bytes_ph, head_cpu.EV.mapped_data));
            }
            current_gpu_bufs.d_EH = cl::Buffer();
            current_gpu_bufs.d_EV_processed_data = cl::Buffer();

            CL_CHECK(current_queue.finish());
        }
        catch (const std::exception& e) {
            // Add context to the exception before re-throwing
            throw std::runtime_error("Exception in attention::clforprop (first block overload) for head ["
                                     + std::to_string(layer_idx) + "][" + std::to_string(col_idx_param) + "]: " + e.what());
        }
    }
    // Final synchronization for all heads in this column
    CL_CHECK(queue.finish());

    agg_d_K = cl::Buffer();
    agg_d_Q = cl::Buffer();
    agg_d_KdotQ = cl::Buffer();
    agg_d_head_attention = cl::Buffer();
    agg_d_col_sums = cl::Buffer();
    agg_d_EV_processed_data = cl::Buffer();
    agg_d_dv_accum = cl::Buffer();
    agg_d_MV_hxd = cl::Buffer();
    agg_d_dv = cl::Buffer();
    agg_d_ver_accumulated_ev = cl::Buffer();
    agg_d_ver_inputs = cl::Buffer();
    agg_d_ver_output = cl::Buffer();
    agg_d_relu_ver_output = cl::Buffer();
    agg_d_mlp_bufferA_ver = cl::Buffer();
    agg_d_mlp_bufferB_ver = cl::Buffer();
    agg_d_mlp_pre_activation = cl::Buffer();
}


/**
 * @brief OpenCL forward propagation on the FIRST block.
 */
void block::clForprop(int& in_dim_param, int& tokenCount_param, int& layers_mlp_param)
{
    for (int j = 0; j < y; ++j) { // j is col_idx_param
        try {
            cl1parallelForprop(in_dim_param, tokenCount_param, j, layers_mlp_param);
        }
        catch (const std::exception& e) {
            throw std::runtime_error("Exception in cl1parallelForprop (first block) for column ["
                                     + std::to_string(j) + "]: " + e.what());
        }
    }
}


/**
 * @brief OpenCL forward propagation on the FIRST block.
 */
void block::clForpropev(int& in_dim_param, int& tokenCount_param, int& layers_mlp_param)
{
    for (int j = 0; j < y; ++j) { // j is col_idx_param
        try {
            cl1parallelForpropev(in_dim_param, tokenCount_param, j, layers_mlp_param);
        }
        catch (const std::exception& e) {
            throw std::runtime_error("Exception in cl1parallelForprop (first block) for column ["
                                     + std::to_string(j) + "]: " + e.what());
        }
    }
}

#endif // USE_OPENCL