
#ifdef USE_OPENCL
#include <CL/cl.hpp>
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
    cl::Buffer d_EV_processed_data; // For the current block's EV data (first 'n' rows of head.EV or EVp)
    cl::Buffer d_ver_accumulated_ev;
    cl::Buffer d_hor_inputs, d_ver_inputs;
    cl::Buffer d_hor_output, d_ver_output;
    cl::Buffer d_relu_hor_output, d_relu_ver_output;
    cl::Buffer d_mlp_bufferA_hor, d_mlp_bufferB_hor;
    cl::Buffer d_mlp_bufferA_ver, d_mlp_bufferB_ver;
    cl::Buffer d_mlp_pre_activation;

    // Note: For the second overload (subsequent blocks), d_EV_processed_data will hold
    // the data from EVp_layer after transfer to the device.

    HeadForwardDeviceBuffersCL() = default;
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
    if (col_idx_param < 0 || col_idx_param >= this->y) {
        throw std::out_of_range("cl1parallelForprop (first block): column index 'i' (" + std::to_string(col_idx_param) + ") is out of range [0, " + std::to_string(this->y - 1) + "].");
    }

    cl_int cl_err; // For OpenCL error codes
    OpenCLContext& context_obj = this->clcontext; // Use the member reference
    cl::Context context = context_obj.context;

    const int num_heads_in_col = this->x;
    const int d_embedding = EMBEDDING;
    const int h_attention = MATHEIGHTS;
    const int n_tokens = tokenCount_param; // Number of tokens for current processing

    // Per-head byte sizes (assuming n_tokens is fixed for all heads in this call)
    size_t k_bytes_ph = static_cast<size_t>(CONTEXT_WIN) * EMBEDDING * sizeof(float);
    size_t q_bytes_ph = static_cast<size_t>(CONTEXT_WIN) * EMBEDDING * sizeof(float);
    size_t kdotq_bytes_ph = static_cast<size_t>(CONTEXT_WIN) * CONTEXT_WIN * sizeof(float);
    size_t head_bytes_ph = static_cast<size_t>(CONTEXT_WIN) * CONTEXT_WIN * sizeof(float);
    size_t sums_bytes_ph = static_cast<size_t>(CONTEXT_WIN) * sizeof(float);
    size_t accum_bytes_ph = static_cast<size_t>(MATHEIGHTS) * sizeof(float);
    size_t proj_mat_bytes_ph = static_cast<size_t>(EMBEDDING) * MATHEIGHTS * sizeof(float);
    size_t ev_processed_bytes_ph = static_cast<size_t>(CONTEXT_WIN) * EMBEDDING * sizeof(float); // For first 'n_tokens' rows of EV
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

    if (n_tokens > 0) { // Only allocate if there are tokens to process
        agg_d_K = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col * k_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_Q = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col * q_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_KdotQ = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col * kdotq_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_head_attention = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col * head_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_row_sums = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col * sums_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_col_sums = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col * sums_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_EV_processed_data = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col * ev_processed_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
    }
    // These buffers are always needed, even if n_tokens is 0, for EH, MLPs etc.
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

    std::vector<cl::CommandQueue> queues(num_heads_in_col);
    std::vector<HeadForwardDeviceBuffersCL> head_gpu_data(num_heads_in_col);

    for (int k = 0; k < num_heads_in_col; ++k) {
        queues[k] = cl::CommandQueue(context, context_obj.device, 0, &cl_err);
        CL_CHECK(cl_err);
    }

    // Iterate through the layers (rows) of attention heads in the specified column 'i'
    for (int layer_idx = 0; layer_idx < num_heads_in_col; ++layer_idx) {
        attention& head_cpu = this->b[layer_idx][col_idx_param];
        cl::CommandQueue& current_queue = queues[layer_idx];
        HeadForwardDeviceBuffersCL& current_gpu_bufs = head_gpu_data[layer_idx];

        const int num_ev_rows_to_process = CONTEXT_WIN; // For EV.sumRows/addToRows

        if (n_tokens <= 0) {
            std::fill(head_cpu.EH.begin(), head_cpu.EH.end(), 0.0f);
            if (n_tokens == 0 && head_cpu.EV.mapped_data && head_cpu.EV.row > 0 && head_cpu.EV.col == d_embedding) {
                std::fill_n(head_cpu.EV.mapped_data, head_cpu.EV.col, 0.0f);
            }
            continue; // Skip to next head in the column
        }

        // --- Basic Validation (Copied from attention::clforprop) ---
        if (head_cpu.EV.row < n_tokens || head_cpu.EV.col != d_embedding) {
            throw std::runtime_error("EV matrix not properly sized for current tokenCount n in cl1parallelForprop (first block) for head [" + std::to_string(layer_idx) + "][" + std::to_string(col_idx_param) + "]. Expected rows >= " + std::to_string(n_tokens) + " and cols = " + std::to_string(d_embedding) +
                                     ", but got rows = " + std::to_string(head_cpu.EV.row) + " and cols = " + std::to_string(head_cpu.EV.col));
        }
        if (head_cpu.K.row != CONTEXT_WIN || head_cpu.K.col != MATHEIGHTS ||
            head_cpu.Q.row != CONTEXT_WIN || head_cpu.Q.col != MATHEIGHTS ||
            head_cpu.KdotQ.row != CONTEXT_WIN || head_cpu.KdotQ.col != CONTEXT_WIN ||
            head_cpu.MH.row != EMBEDDING || head_cpu.MH.col != MATHEIGHTS ||
            head_cpu.MV.row != EMBEDDING || head_cpu.MV.col != MATHEIGHTS ||
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
            // --- Create Sub-Buffers for the current head ---
            auto create_sub_buffer = [&](cl::Buffer& parent_agg_buf, size_t per_head_bytes) {
                cl_buffer_region region = {static_cast<size_t>(layer_idx) * per_head_bytes, per_head_bytes};
                return parent_agg_buf.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err);
            };

            if (n_tokens > 0) { // Create these sub-buffers only if n_tokens > 0 and their parent aggregate buffers exist
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

            // Initialize accumulators to zero
            float zero = 0.0f;
            CL_CHECK(current_queue.enqueueFillBuffer(current_gpu_bufs.d_dh_accum, zero, 0, accum_bytes_ph));
            CL_CHECK(current_queue.enqueueFillBuffer(current_gpu_bufs.d_dv_accum, zero, 0, accum_bytes_ph));

            // --- Data Transfer H->D (Copied from attention::clforprop) ---
            if (n_tokens > 0) { // Transfer these only if n_tokens > 0
                CL_CHECK(current_queue.enqueueWriteBuffer(current_gpu_bufs.d_K, CL_FALSE, 0, k_bytes_ph, head_cpu.K.mapped_data));
                CL_CHECK(current_queue.enqueueWriteBuffer(current_gpu_bufs.d_Q, CL_FALSE, 0, q_bytes_ph, head_cpu.Q.mapped_data));
                CL_CHECK(current_queue.enqueueWriteBuffer(current_gpu_bufs.d_KdotQ, CL_FALSE, 0, kdotq_bytes_ph, head_cpu.KdotQ.mapped_data));
                CL_CHECK(current_queue.enqueueWriteBuffer(current_gpu_bufs.d_EV_processed_data, CL_FALSE, 0, ev_processed_bytes_ph, head_cpu.EV.mapped_data));
            }
            CL_CHECK(current_queue.enqueueWriteBuffer(current_gpu_bufs.d_MH_hxd, CL_FALSE, 0, proj_mat_bytes_ph, head_cpu.MH.mapped_data));
            CL_CHECK(current_queue.enqueueWriteBuffer(current_gpu_bufs.d_MV_hxd, CL_FALSE, 0, proj_mat_bytes_ph, head_cpu.MV.mapped_data));
            CL_CHECK(current_queue.enqueueWriteBuffer(current_gpu_bufs.d_EH, CL_FALSE, 0, embed_bytes_ph, head_cpu.EH.data()));

            // --- Kernel Launches (Copied from attention::clforprop) ---
            const size_t local_work_size_1d = 256;
            cl::Kernel lota_kernel = get_kernel_with_check(context_obj, "clLOTA2dmasking");
            size_t totalElementsLOTA = static_cast<size_t>(n_tokens) * n_tokens; // Use n_tokens for actual KdotQ size
            if (totalElementsLOTA > 0) {
                size_t global_lota_raw = totalElementsLOTA; //NOLINT
                size_t local_lota_clamped = (std::min)(global_lota_raw, local_work_size_1d);
                if (local_lota_clamped == 0) local_lota_clamped = 1;
                size_t global_lota_padded = ((global_lota_raw + local_lota_clamped - 1) / local_lota_clamped) * local_lota_clamped;
                cl::NDRange global_lota(global_lota_padded);
                cl::NDRange local_lota(local_lota_clamped);
                CL_CHECK(lota_kernel.setArg(0, current_gpu_bufs.d_KdotQ));
                CL_CHECK(lota_kernel.setArg(1, current_gpu_bufs.d_head_attention));
                CL_CHECK(lota_kernel.setArg(2, CONTEXT_WIN)); // rows of KdotQ
                CL_CHECK(lota_kernel.setArg(3, CONTEXT_WIN)); // cols of KdotQ
                CL_CHECK(lota_kernel.setArg(4, n_tokens));
                cl_int cl_att_is_self_lota = this->isSelfAttention ? 1 : 0;
                CL_CHECK(lota_kernel.setArg(5, cl_att_is_self_lota));
                CL_CHECK(current_queue.enqueueNDRangeKernel(lota_kernel, cl::NullRange, global_lota, local_lota));
            } //NOLINT

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
            // CL_CHECK(current_queue.finish()); // Intermediate sync for this head

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
            // CL_CHECK(current_queue.finish()); // Intermediate sync for this head

            // --- Run MLPs Forward (Copied from attention::clforprop) ---
            cl::Kernel mlp_fwd_kernel = context_obj.kernels.at("kernelLayerForward"); // Re-fetch or ensure it's the same as proj_kernel
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
                // Enhanced debug:
                // std::cout << "Head [" << layer_idx << "], MLP Layer [" << layer_idx_mlp << "] HOR: MLP kernel args 3,4 set." << std::endl;
                { 
                    mat& current_weights_mat = head_cpu.hor.weights[layer_idx_mlp]; 
                    // *** ADDED DIMENSION CHECK ***
                    if (static_cast<int>(current_weights_mat.row) != output_size_mlp || static_cast<int>(current_weights_mat.col) != input_size_mlp) {
                        throw std::runtime_error("FATAL ERROR in cl1parallelForprop (first block): head_cpu.hor.weights[" + std::to_string(layer_idx_mlp) +
                                                 "] dimension mismatch for head [" + std::to_string(layer_idx) + "][" + std::to_string(col_idx_param) + "]." +
                                                 " Expected " + std::to_string(output_size_mlp) + "x" + std::to_string(input_size_mlp) +
                                                 ", got " + std::to_string(current_weights_mat.row) + "x" + std::to_string(current_weights_mat.col));
                    }

                    // --- DIAGNOSTIC START ---
                    if (current_weights_mat.mapped_data == nullptr) {
                        std::cerr << "FATAL ERROR in cl1parallelForprop (first block): head_cpu.hor.weights[" << layer_idx_mlp 
                                  << "].mapped_data is nullptr for head [" << layer_idx << "][" << col_idx_param << "]."
                                  << std::endl;
                        std::cerr << "  Head Index (layer_idx): " << layer_idx << ", Column Index (col_idx_param): " << col_idx_param << std::endl;
                        std::cerr << "  MLP Layer (layer_idx_mlp): " << layer_idx_mlp << std::endl;
                        std::cerr << "  Mat Details: rows=" << current_weights_mat.row << ", cols=" << current_weights_mat.col;
                        // Assuming mat class has a way to get its source file path, e.g., a 'path_to_file' member or method
                        // If not, this part might need adjustment based on 'mat' class definition.
                        // std::cerr << ", path=" << current_weights_mat.path_to_file;
                        std::cerr << std::endl;
                        throw std::runtime_error("Null mapped_data encountered for MLP weights.");
                    }
                    // --- DIAGNOSTIC END ---
                    size_t weights_bytes_mlp = static_cast<size_t>(current_weights_mat.row) * current_weights_mat.col * sizeof(float); 
                    cl::Buffer d_mlp_weights(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, weights_bytes_mlp, current_weights_mat.mapped_data, &cl_err); CL_CHECK(cl_err); 
                    cl::Buffer& target_output_buffer = is_last_layer_mlp ? current_gpu_bufs.d_hor_output : current_gpu_bufs.d_mlp_pre_activation; 
                    CL_CHECK(mlp_fwd_kernel.setArg(0, current_in_hor)); 
                    CL_CHECK(mlp_fwd_kernel.setArg(1, d_mlp_weights)); 
                    CL_CHECK(mlp_fwd_kernel.setArg(2, target_output_buffer));
                    // Enhanced debug:
                    // std::cout << "Head [" << layer_idx << "], MLP Layer [" << layer_idx_mlp << "] HOR: MLP kernel args 0,1,2 set." << std::endl;
                    CL_CHECK(current_queue.enqueueNDRangeKernel(mlp_fwd_kernel, cl::NullRange, global_mlp, local_mlp)); 
                    // DEBUG: Add finish to isolate hang
                    // CL_CHECK(current_queue.finish()); 
                    // std::cout <<"Head [" << layer_idx << "], MLP Layer [" << layer_idx_mlp << "] HOR: layer forward enqueued."<< std::endl;
                    if (!is_last_layer_mlp) {
                        CL_CHECK(sigmoid_kernel.setArg(0, current_gpu_bufs.d_mlp_pre_activation)); 
                        CL_CHECK(sigmoid_kernel.setArg(1, current_out_hor)); 
                        CL_CHECK(sigmoid_kernel.setArg(2, output_size_mlp)); 
                        CL_CHECK(current_queue.enqueueNDRangeKernel(sigmoid_kernel, cl::NullRange, global_mlp, local_mlp)); 
                        std::swap(current_in_hor, current_out_hor); 
                    }
                    // Enhanced debug:
                    // std::cout << "Head [" << layer_idx << "], MLP Layer [" << layer_idx_mlp << "] HOR: sigmoid enqueued (or skipped)." << std::endl;
                }
                // Enhanced debug:
                // std::cout << "Head [" << layer_idx << "] HOR MLP processing for layer_idx_mlp " << layer_idx_mlp << " done." << std::endl;
                { 
                    mat& current_weights_mat = head_cpu.ver.weights[layer_idx_mlp]; 
                    // *** ADDED DIMENSION CHECK ***
                    if (static_cast<int>(current_weights_mat.row) != output_size_mlp || static_cast<int>(current_weights_mat.col) != input_size_mlp) {
                        throw std::runtime_error("FATAL ERROR in cl1parallelForprop (first block): head_cpu.ver.weights[" + std::to_string(layer_idx_mlp) +
                                                 "] dimension mismatch for head [" + std::to_string(layer_idx) + "][" + std::to_string(col_idx_param) + "]." +
                                                 " Expected " + std::to_string(output_size_mlp) + "x" + std::to_string(input_size_mlp) +
                                                 ", got " + std::to_string(current_weights_mat.row) + "x" + std::to_string(current_weights_mat.col));
                    }

                    // --- DIAGNOSTIC START ---
                    if (current_weights_mat.mapped_data == nullptr) {
                        std::cerr << "FATAL ERROR in cl1parallelForprop (first block): head_cpu.ver.weights[" << layer_idx_mlp 
                                  << "].mapped_data is nullptr for head [" << layer_idx << "][" << col_idx_param << "]."
                                  << std::endl;
                        std::cerr << "  Head Index (layer_idx): " << layer_idx << ", Column Index (col_idx_param): " << col_idx_param << std::endl;
                        std::cerr << "  MLP Layer (layer_idx_mlp): " << layer_idx_mlp << std::endl;
                        std::cerr << "  Mat Details: rows=" << current_weights_mat.row << ", cols=" << current_weights_mat.col;
                        // std::cerr << ", path=" << current_weights_mat.path_to_file;
                        std::cerr << std::endl;
                        throw std::runtime_error("Null mapped_data encountered for MLP weights (vertical).");
                    }
                    // --- DIAGNOSTIC END ---
                    size_t weights_bytes_mlp = static_cast<size_t>(current_weights_mat.row) * current_weights_mat.col * sizeof(float); 
                    cl::Buffer d_mlp_weights(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, weights_bytes_mlp, current_weights_mat.mapped_data, &cl_err); CL_CHECK(cl_err); 
                    cl::Buffer& target_output_buffer = is_last_layer_mlp ? current_gpu_bufs.d_ver_output : current_gpu_bufs.d_mlp_pre_activation; 
                    CL_CHECK(mlp_fwd_kernel.setArg(0, current_in_ver)); 
                    CL_CHECK(mlp_fwd_kernel.setArg(1, d_mlp_weights)); 
                    CL_CHECK(mlp_fwd_kernel.setArg(2, target_output_buffer)); 
                    CL_CHECK(current_queue.enqueueNDRangeKernel(mlp_fwd_kernel, cl::NullRange, global_mlp, local_mlp));
                    // DEBUG: Add finish to isolate hang
                    // CL_CHECK(current_queue.finish());
                    if (!is_last_layer_mlp) { 
                        CL_CHECK(sigmoid_kernel.setArg(0, current_gpu_bufs.d_mlp_pre_activation)); 
                        CL_CHECK(sigmoid_kernel.setArg(1, current_out_ver)); 
                        CL_CHECK(sigmoid_kernel.setArg(2, output_size_mlp)); 
                        CL_CHECK(current_queue.enqueueNDRangeKernel(sigmoid_kernel, cl::NullRange, global_mlp, local_mlp)); 
                        std::swap(current_in_ver, current_out_ver); 
                    }
                }
                // Enhanced debug:
                // std::cout << "Head [" << layer_idx << "] VER MLP processing for layer_idx_mlp " << layer_idx_mlp << " done." << std::endl;
            }
            // DEBUG: Sync after entire MLP loop for this head
            // CL_CHECK(current_queue.finish());
            // std::cout << "Head [" << layer_idx << "]: All MLP layers processed." << std::endl;

            // 7. Apply ReLU (Copied from attention::clforprop)
            cl::Kernel relu_kernel = get_kernel_with_check(context_obj, "clReLU1d"); 
            CL_CHECK(relu_kernel.setArg(2, d_embedding));
            CL_CHECK(relu_kernel.setArg(0, current_gpu_bufs.d_hor_output)); CL_CHECK(relu_kernel.setArg(1, current_gpu_bufs.d_relu_hor_output)); 
            CL_CHECK(current_queue.enqueueNDRangeKernel(relu_kernel, cl::NullRange, global_add, local_add));
            CL_CHECK(relu_kernel.setArg(0, current_gpu_bufs.d_ver_output)); CL_CHECK(relu_kernel.setArg(1, current_gpu_bufs.d_relu_ver_output)); 
            CL_CHECK(current_queue.enqueueNDRangeKernel(relu_kernel, cl::NullRange, global_add, local_add));
            // DEBUG: Sync after ReLU
            // CL_CHECK(current_queue.finish());
            // std::cout << "Head [" << layer_idx << "]: ReLU applied." << std::endl;

            // 8. Final Residual Update (Copied from attention::clforprop)
            CL_CHECK(add_kernel.setArg(0, current_gpu_bufs.d_EH)); 
            CL_CHECK(add_kernel.setArg(1, current_gpu_bufs.d_relu_hor_output)); 
            CL_CHECK(add_kernel.setArg(2, current_gpu_bufs.d_EH)); 
            CL_CHECK(current_queue.enqueueNDRangeKernel(add_kernel, cl::NullRange, global_add, local_add));
            cl::Kernel update_ev_kernel = get_kernel_with_check(context_obj, "updateEVRowsKernelCL");
            size_t global_update_ev_raw = static_cast<size_t>(n_tokens); size_t global_update_ev_padded = ((global_update_ev_raw + local_work_size_1d - 1) / local_work_size_1d) * local_work_size_1d; 
            cl::NDRange global_update_ev(global_update_ev_padded); cl::NDRange local_update_ev(local_work_size_1d);
            CL_CHECK(update_ev_kernel.setArg(0, current_gpu_bufs.d_EV_processed_data)); 
            CL_CHECK(update_ev_kernel.setArg(1, current_gpu_bufs.d_relu_ver_output)); 
            CL_CHECK(update_ev_kernel.setArg(2, n_tokens)); 
            CL_CHECK(update_ev_kernel.setArg(3, d_embedding));
            CL_CHECK(current_queue.enqueueNDRangeKernel(update_ev_kernel, cl::NullRange, global_update_ev, local_update_ev));
            // DEBUG: Sync after final updates
            // CL_CHECK(current_queue.finish());
            // std::cout << "Head [" << layer_idx << "]: Final residual updates done." << std::endl;

            // --- Copy Results D->H (Copied from attention::clforprop) ---
            CL_CHECK(current_queue.enqueueReadBuffer(current_gpu_bufs.d_EH, CL_FALSE, 0, embed_bytes_ph, head_cpu.EH.data()));
            if (n_tokens > 0) {
                CL_CHECK(current_queue.enqueueReadBuffer(current_gpu_bufs.d_EV_processed_data, CL_FALSE, 0, ev_processed_bytes_ph, head_cpu.EV.mapped_data));
            }
            // Explicitly release these specific sub-buffer objects now that their data is copied.
            // Their OpenCL resources are released; the aggregate buffer memory is managed separately.
            current_gpu_bufs.d_EH = cl::Buffer();
            current_gpu_bufs.d_EV_processed_data = cl::Buffer();

            // std::cout << "Head [" << layer_idx << "]: D->H transfers enqueued." << std::endl;
            // clean all buffers
            CL_CHECK(current_queue.finish());
        }
        catch (const std::exception& e) {
            // Add context to the exception before re-throwing
            throw std::runtime_error("Exception in attention::clforprop (first block overload) for head ["
                                     + std::to_string(layer_idx) + "][" + std::to_string(col_idx_param) + "]: " + e.what());
        }
    }
    // Final synchronization for all heads in this column
    for (int k = 0; k < num_heads_in_col; ++k) {
        CL_CHECK(queues[k].finish());
    }

    // Explicitly release all aggregate OpenCL buffers
    // If a buffer was conditionally not allocated (e.g., n_tokens == 0),
    // assigning cl::Buffer() to it is safe (it's already default-constructed).
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
 * @brief OpenCL forward propagation on single ith column of a SUBSEQUENT block (blockCount > 0).
 * @param context OpenCL context.
 * @param queue OpenCL command queue.
 * @param kernels Map of compiled OpenCL kernels.
 * @param EVp vertical retention vectors from previous block for THIS COLUMN (shape [layer][token][embedding]).
 * @param in dimension size (maps to d_embedding in attention::clforprop)
 * @param tokenCount number of tokens in full context (maps to totalTokenCount in attention::clforprop)
 * @param blockCount position of block in full context (1-based, maps to blockIdx in attention::clforprop)
 * @param i column index (0-based)
 * @param layers number of layers in mlp (maps to layers_mlp in attention::clforprop)
 * @param n context window size (maps to contextWindowSize in attention::clforprop)
 */
void block::cl1ParallelForprop(std::vector<std::vector<std::vector<float>>>& EVp_col, int& in_dim_param, int& totalTokenCount_param, int& blockIdx_param,
                               int col_idx_param, int& layers_mlp_param, int& contextWindowSize_param)
{
    if (col_idx_param < 0 || col_idx_param >= this->y) {
        throw std::out_of_range("cl1ParallelForprop (subsequent block): column index 'i' (" + std::to_string(col_idx_param) + ") is out of range [0, " + std::to_string(this->y - 1) + "].");
    }

    cl_int cl_err; // For OpenCL error codes
    OpenCLContext& context_obj = this->clcontext; // Use the member reference
    cl::Context context = context_obj.context;

    // Validate the incoming EVp for this column
    if (EVp_col.size() != static_cast<size_t>(this->x)) {
         throw std::runtime_error("cl1ParallelForprop (subsequent block): EVp_col layer dimension mismatch for column " + std::to_string(col_idx_param)
                                  + ". Expected " + std::to_string(this->x) + " layers, got " + std::to_string(EVp_col.size()) + ".");
    }

    const int num_heads_in_col = this->x;
    const int d_embedding = EMBEDDING;
    const int h_attention = MATHEIGHTS;
    // `count_tokens_in_block` is the number of tokens relevant to K,Q,KdotQ for this block
    int start_idx_in_full_context_calc = (blockIdx_param - 1) * contextWindowSize_param;
    int end_idx_in_full_context_calc = std::min<int>(totalTokenCount_param, blockIdx_param * contextWindowSize_param);
    const int count_tokens_in_block = std::max<int>(0, end_idx_in_full_context_calc - start_idx_in_full_context_calc);

    // `num_ev_rows_to_process_for_evp` is the number of rows in EVp_layer (typically CONTEXT_WIN or totalTokenCount_param)
    // Based on how EVp is structured and passed, it seems to represent the full context window from the previous block.
    const int num_ev_rows_to_process_for_evp = CONTEXT_WIN;

    // Per-head byte sizes
    size_t k_bytes_ph = static_cast<size_t>(CONTEXT_WIN) * EMBEDDING * sizeof(float);
    size_t q_bytes_ph = static_cast<size_t>(CONTEXT_WIN) * EMBEDDING * sizeof(float);
    size_t kdotq_bytes_ph = static_cast<size_t>(CONTEXT_WIN) * CONTEXT_WIN * sizeof(float);
    size_t head_bytes_ph = static_cast<size_t>(count_tokens_in_block) * count_tokens_in_block * sizeof(float);
    size_t sums_bytes_ph = static_cast<size_t>(count_tokens_in_block) * sizeof(float);

    size_t accum_bytes_ph = static_cast<size_t>(h_attention) * sizeof(float);
    size_t proj_mat_bytes_ph = static_cast<size_t>(d_embedding) * h_attention * sizeof(float);
    size_t ev_from_prev_block_bytes_ph = static_cast<size_t>(num_ev_rows_to_process_for_evp) * d_embedding * sizeof(float); // For EVp_layer
    size_t embed_bytes_ph = static_cast<size_t>(d_embedding) * sizeof(float);

    // --- Aggregate Buffer Allocation ---
    cl::Buffer agg_d_K, agg_d_Q, agg_d_KdotQ, agg_d_head_attention;
    cl::Buffer agg_d_row_sums, agg_d_col_sums;
    cl::Buffer agg_d_dh_accum, agg_d_dv_accum;
    cl::Buffer agg_d_MH_hxd, agg_d_MV_hxd;
    cl::Buffer agg_d_dh, agg_d_dv, agg_d_EH;
    cl::Buffer agg_d_EV_processed_data; // This will hold EVp data from host
    cl::Buffer agg_d_ver_accumulated_ev;
    cl::Buffer agg_d_hor_inputs, agg_d_ver_inputs;
    cl::Buffer agg_d_hor_output, agg_d_ver_output;
    cl::Buffer agg_d_relu_hor_output, agg_d_relu_ver_output;
    cl::Buffer agg_d_mlp_bufferA_hor, agg_d_mlp_bufferB_hor;
    cl::Buffer agg_d_mlp_bufferA_ver, agg_d_mlp_bufferB_ver;
    cl::Buffer agg_d_mlp_pre_activation;

    if (count_tokens_in_block > 0) { // Allocate these only if count_tokens_in_block > 0
        agg_d_K = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col * k_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_Q = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col * q_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_KdotQ = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col * kdotq_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_head_attention = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col * head_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_row_sums = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col * sums_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_col_sums = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col * sums_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
    }
    agg_d_EV_processed_data = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col * ev_from_prev_block_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
    // These buffers are always needed
    agg_d_dh_accum = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col * accum_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
    agg_d_dv_accum = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col * accum_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
    // ... (complete all aggregate allocations as in the first cl1parallelForprop overload, using _ph sizes)
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

    std::vector<cl::CommandQueue> queues(num_heads_in_col);
    std::vector<HeadForwardDeviceBuffersCL> head_gpu_data(num_heads_in_col); // Using the same struct

    for (int k = 0; k < num_heads_in_col; ++k) {
        queues[k] = cl::CommandQueue(context, context_obj.device, 0, &cl_err); CL_CHECK(cl_err);
    }

    // Iterate through the layers (rows) of attention heads in the specified column 'i'
    for (int layer_idx = 0; layer_idx < num_heads_in_col; ++layer_idx) {
        attention& head_cpu = this->b[layer_idx][col_idx_param];
        cl::CommandQueue& current_queue = queues[layer_idx];
        HeadForwardDeviceBuffersCL& current_gpu_bufs = head_gpu_data[layer_idx];
        // EVp[layer_idx] contains the [token][embedding] data for this specific head from the previous block
        std::vector<std::vector<float>>& EVp_layer = EVp_col[layer_idx];

        if (blockIdx_param == 0) { // Should be blockIdx_param > 0 for this overload
            throw std::logic_error("cl1ParallelForprop (subsequent) called with blockCount <= 0");
        }

        if (count_tokens_in_block <= 0) {
            std::fill(head_cpu.EH.begin(), head_cpu.EH.end(), 0.0f);
            // EV might not need zeroing if count is 0, depends on overall logic.
            continue; // Skip to next head
        }

        // --- Basic Validation
        // head_cpu.EV stores the *output* EV for this block, which will be based on EVp_layer
        if (head_cpu.EV.row < num_ev_rows_to_process_for_evp || head_cpu.EV.col != d_embedding) {
            throw std::runtime_error("EV matrix (output) not properly sized for EVp in cl1ParallelForprop (subsequent block) for head [" + std::to_string(layer_idx) + "][" + std::to_string(col_idx_param) + "]. Expected rows >= " + std::to_string(num_ev_rows_to_process_for_evp) + " and cols = " + std::to_string(d_embedding) +
                                     ", but got rows = " + std::to_string(head_cpu.EV.row) + " and cols = " + std::to_string(head_cpu.EV.col));
        }
        if (head_cpu.K.row != CONTEXT_WIN || head_cpu.K.col != MATHEIGHTS || // K,Q,KdotQ are sized for CONTEXT_WIN but operate on count_tokens_in_block
            head_cpu.Q.row != CONTEXT_WIN || head_cpu.Q.col != MATHEIGHTS ||
            head_cpu.KdotQ.row != CONTEXT_WIN || head_cpu.KdotQ.col != CONTEXT_WIN ||
            head_cpu.MH.row != EMBEDDING || head_cpu.MH.col != MATHEIGHTS ||
            head_cpu.MV.row != EMBEDDING || head_cpu.MV.col != MATHEIGHTS ||
            head_cpu.EH.size() != static_cast<size_t>(EMBEDDING) ||
            head_cpu.hor.hlayers.empty() || head_cpu.ver.hlayers.empty() || head_cpu.hor.weights.empty() || head_cpu.ver.weights.empty())
        {
            throw std::runtime_error("Attention component dimension mismatch or uninitialized member in cl1ParallelForprop (subsequent block) for head [" +
                                     std::to_string(layer_idx) + "][" + std::to_string(col_idx_param) + "]. K.row=" + std::to_string(head_cpu.K.row) + ", count=" + std::to_string(count_tokens_in_block));
        }
        if (d_embedding != in_dim_param) {
            throw std::runtime_error("Embedding dimension mismatch (EMBEDDING vs in) for head [" + std::to_string(layer_idx) + "][" + std::to_string(col_idx_param) + "].");
        }
        if (EVp_layer.size() != static_cast<size_t>(num_ev_rows_to_process_for_evp) || (!EVp_layer.empty() && EVp_layer[0].size() != static_cast<size_t>(d_embedding)) ) {
            throw std::runtime_error("EVp_layer dimension mismatch for head [" + std::to_string(layer_idx) + "][" + std::to_string(col_idx_param) +
                                     "]. Expected rows " + std::to_string(num_ev_rows_to_process_for_evp) + ", got " + std::to_string(EVp_layer.size()) +
                                     ". Expected cols " + std::to_string(d_embedding) + ", got " + (EVp_layer.empty() ? "N/A" : std::to_string(EVp_layer[0].size())) );
        }
        if (head_cpu.hor.hlayers[0].size() != static_cast<size_t>(d_embedding) || head_cpu.ver.hlayers[0].size() != static_cast<size_t>(d_embedding) ||
            head_cpu.hor.weights.back().row != static_cast<size_t>(d_embedding) || head_cpu.ver.weights.back().row != static_cast<size_t>(d_embedding)) {
            throw std::runtime_error("MLP input/output layer dimension mismatch with embedding dimension 'd' for head [" + std::to_string(layer_idx) + "][" + std::to_string(col_idx_param) + "].");
        }

        try {
            // --- Create Sub-Buffers for the current head ---
            auto create_sub_buffer = [&](cl::Buffer& parent_agg_buf, size_t per_head_bytes) {
                cl_buffer_region region = {static_cast<size_t>(layer_idx) * per_head_bytes, per_head_bytes};
                return parent_agg_buf.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err);
            };

            if (count_tokens_in_block > 0) { // Create these sub-buffers only if count_tokens_in_block > 0
                current_gpu_bufs.d_K = create_sub_buffer(agg_d_K, k_bytes_ph); CL_CHECK(cl_err);
                current_gpu_bufs.d_Q = create_sub_buffer(agg_d_Q, q_bytes_ph); CL_CHECK(cl_err);
                current_gpu_bufs.d_KdotQ = create_sub_buffer(agg_d_KdotQ, kdotq_bytes_ph); CL_CHECK(cl_err);
                current_gpu_bufs.d_head_attention = create_sub_buffer(agg_d_head_attention, head_bytes_ph); CL_CHECK(cl_err);
                current_gpu_bufs.d_row_sums = create_sub_buffer(agg_d_row_sums, sums_bytes_ph); CL_CHECK(cl_err);
                current_gpu_bufs.d_col_sums = create_sub_buffer(agg_d_col_sums, sums_bytes_ph); CL_CHECK(cl_err);
            }
            current_gpu_bufs.d_EV_processed_data = create_sub_buffer(agg_d_EV_processed_data, ev_from_prev_block_bytes_ph); CL_CHECK(cl_err);
            current_gpu_bufs.d_dh_accum = create_sub_buffer(agg_d_dh_accum, accum_bytes_ph); CL_CHECK(cl_err);
            current_gpu_bufs.d_EH = create_sub_buffer(agg_d_EH, embed_bytes_ph); CL_CHECK(cl_err);
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

            // Initialize accumulators to zero
            float zero = 0.0f;
            CL_CHECK(current_queue.enqueueFillBuffer(current_gpu_bufs.d_dh_accum, zero, 0, accum_bytes_ph));
            CL_CHECK(current_queue.enqueueFillBuffer(current_gpu_bufs.d_dv_accum, zero, 0, accum_bytes_ph));

            // --- Data Transfer H->D
            if (count_tokens_in_block > 0) { // Transfer these only if count_tokens_in_block > 0
                CL_CHECK(current_queue.enqueueWriteBuffer(current_gpu_bufs.d_K, CL_FALSE, 0, k_bytes_ph, head_cpu.K.mapped_data));
                CL_CHECK(current_queue.enqueueWriteBuffer(current_gpu_bufs.d_Q, CL_FALSE, 0, q_bytes_ph, head_cpu.Q.mapped_data));
                CL_CHECK(current_queue.enqueueWriteBuffer(current_gpu_bufs.d_KdotQ, CL_FALSE, 0, kdotq_bytes_ph, head_cpu.KdotQ.mapped_data));
            }
            CL_CHECK(current_queue.enqueueWriteBuffer(current_gpu_bufs.d_MH_hxd, CL_FALSE, 0, proj_mat_bytes_ph, head_cpu.MH.mapped_data));
            CL_CHECK(current_queue.enqueueWriteBuffer(current_gpu_bufs.d_MV_hxd, CL_FALSE, 0, proj_mat_bytes_ph, head_cpu.MV.mapped_data));
            CL_CHECK(current_queue.enqueueWriteBuffer(current_gpu_bufs.d_EH, CL_FALSE, 0, embed_bytes_ph, head_cpu.EH.data()));

            // Copy EVp_layer data to d_EV_processed_data_from_prev_block
            std::vector<float> flat_EVp_layer;
            flatten2DVector(EVp_layer, flat_EVp_layer, num_ev_rows_to_process_for_evp, d_embedding);
            CL_CHECK(current_queue.enqueueWriteBuffer(current_gpu_bufs.d_EV_processed_data, CL_FALSE, 0, ev_from_prev_block_bytes_ph, flat_EVp_layer.data()));

            // --- Kernel Launches
            const size_t local_work_size_1d = 256;
            cl::Kernel lota_kernel = get_kernel_with_check(context_obj, "clLOTA2dmasking");
            size_t totalElementsLOTA = static_cast<size_t>(count_tokens_in_block) * count_tokens_in_block; // Use count_tokens_in_block
            if (totalElementsLOTA > 0) {
                size_t global_lota_raw = totalElementsLOTA; //NOLINT
                size_t local_lota_clamped = (std::min)(global_lota_raw, local_work_size_1d);
                if (local_lota_clamped == 0) local_lota_clamped = 1;
                size_t global_lota_padded = ((global_lota_raw + local_lota_clamped - 1) / local_lota_clamped) * local_lota_clamped;
                cl::NDRange global_lota(global_lota_padded);
                cl::NDRange local_lota(local_lota_clamped);
                CL_CHECK(lota_kernel.setArg(0, current_gpu_bufs.d_KdotQ));
                CL_CHECK(lota_kernel.setArg(1, current_gpu_bufs.d_head_attention)); 
                CL_CHECK(lota_kernel.setArg(2, count_tokens_in_block)); // rows of KdotQ
                CL_CHECK(lota_kernel.setArg(3, count_tokens_in_block)); // cols of KdotQ
                CL_CHECK(lota_kernel.setArg(4, count_tokens_in_block)); // actual_tokens_in_window for masking
                cl_int cl_att_is_self_lota = this->isSelfAttention ? 1 : 0;
                CL_CHECK(lota_kernel.setArg(5, cl_att_is_self_lota));
                CL_CHECK(current_queue.enqueueNDRangeKernel(lota_kernel, cl::NullRange, global_lota, local_lota));
            } //NOLINT

            cl::Kernel sums_kernel = get_kernel_with_check(context_obj, "computeHeadSumsMaskedKernel");
            size_t global_sums_raw = static_cast<size_t>(count_tokens_in_block); 
            size_t global_sums_padded = ((global_sums_raw + local_work_size_1d - 1) / local_work_size_1d) * local_work_size_1d; 
            cl::NDRange global_sums(global_sums_padded); cl::NDRange local_sums(local_work_size_1d); 
            cl_int cl_isSelfAttention = head_cpu.isSelfAttention;
            CL_CHECK(sums_kernel.setArg(0, current_gpu_bufs.d_head_attention)); 
            CL_CHECK(sums_kernel.setArg(1, current_gpu_bufs.d_row_sums)); 
            CL_CHECK(sums_kernel.setArg(2, current_gpu_bufs.d_col_sums)); 
            CL_CHECK(sums_kernel.setArg(3, count_tokens_in_block)); 
            CL_CHECK(sums_kernel.setArg(4, cl_isSelfAttention));
            CL_CHECK(current_queue.enqueueNDRangeKernel(sums_kernel, cl::NullRange, global_sums, local_sums));

            cl::Kernel accum_kernel = get_kernel_with_check(context_obj, "accumulateWeightedVectorsKernel");
            size_t global_accum_raw = static_cast<size_t>(count_tokens_in_block); size_t global_accum_padded = ((global_accum_raw + local_work_size_1d - 1) / local_work_size_1d) * local_work_size_1d; cl::NDRange global_accum(global_accum_padded); cl::NDRange local_accum(local_work_size_1d);
            CL_CHECK(accum_kernel.setArg(0, current_gpu_bufs.d_row_sums)); 
            CL_CHECK(accum_kernel.setArg(1, current_gpu_bufs.d_col_sums)); 
            CL_CHECK(accum_kernel.setArg(2, current_gpu_bufs.d_K)); 
            CL_CHECK(accum_kernel.setArg(3, current_gpu_bufs.d_Q)); 
            CL_CHECK(accum_kernel.setArg(4, current_gpu_bufs.d_dh_accum)); 
            CL_CHECK(accum_kernel.setArg(5, current_gpu_bufs.d_dv_accum)); 
            CL_CHECK(accum_kernel.setArg(6, count_tokens_in_block)); 
            CL_CHECK(accum_kernel.setArg(7, h_attention));
            CL_CHECK(current_queue.enqueueNDRangeKernel(accum_kernel, cl::NullRange, global_accum, local_accum));

            cl::Kernel proj_kernel = get_kernel_with_check(context_obj, "kernelLayerForward");
            size_t global_proj_raw = static_cast<size_t>(d_embedding); 
            size_t global_proj_padded = ((global_proj_raw + local_work_size_1d - 1) / local_work_size_1d) * local_work_size_1d; 
            cl::NDRange global_proj(global_proj_padded); cl::NDRange local_proj(local_work_size_1d);
            CL_CHECK(proj_kernel.setArg(3, h_attention)); 
            CL_CHECK(proj_kernel.setArg(4, d_embedding));
            CL_CHECK(proj_kernel.setArg(0, current_gpu_bufs.d_dh_accum)); CL_CHECK(proj_kernel.setArg(1, current_gpu_bufs.d_MH_hxd)); CL_CHECK(proj_kernel.setArg(2, current_gpu_bufs.d_dh)); 
            CL_CHECK(current_queue.enqueueNDRangeKernel(proj_kernel, cl::NullRange, global_proj, local_proj));
            CL_CHECK(proj_kernel.setArg(0, current_gpu_bufs.d_dv_accum)); CL_CHECK(proj_kernel.setArg(1, current_gpu_bufs.d_MV_hxd)); CL_CHECK(proj_kernel.setArg(2, current_gpu_bufs.d_dv)); 
            CL_CHECK(current_queue.enqueueNDRangeKernel(proj_kernel, cl::NullRange, global_proj, local_proj));
            // CL_CHECK(current_queue.finish()); // Intermediate sync

            cl::Kernel add_kernel = get_kernel_with_check(context_obj, "vectorAddKernel");
            size_t global_add_raw = static_cast<size_t>(d_embedding); 
            size_t global_add_padded = ((global_add_raw + local_work_size_1d - 1) / local_work_size_1d) * local_work_size_1d; 
            cl::NDRange global_add(global_add_padded); cl::NDRange local_add(local_work_size_1d); CL_CHECK(add_kernel.setArg(3, d_embedding));
            CL_CHECK(add_kernel.setArg(0, current_gpu_bufs.d_EH)); CL_CHECK(add_kernel.setArg(1, current_gpu_bufs.d_dh)); CL_CHECK(add_kernel.setArg(2, current_gpu_bufs.d_hor_inputs)); 
            CL_CHECK(current_queue.enqueueNDRangeKernel(add_kernel, cl::NullRange, global_add, local_add));
            cl::Kernel accum_ev_kernel = get_kernel_with_check(context_obj, "accumulateEVRowsKernelCL");
            CL_CHECK(accum_ev_kernel.setArg(0, current_gpu_bufs.d_EV_processed_data)); // This is EVp data
            CL_CHECK(accum_ev_kernel.setArg(1, current_gpu_bufs.d_ver_accumulated_ev));
            CL_CHECK(accum_ev_kernel.setArg(2, num_ev_rows_to_process_for_evp)); // Iterate over all rows of EVp
            CL_CHECK(accum_ev_kernel.setArg(3, d_embedding));
            CL_CHECK(current_queue.enqueueNDRangeKernel(accum_ev_kernel, cl::NullRange, global_add, local_add)); // global_add is for d_embedding size
            CL_CHECK(add_kernel.setArg(0, current_gpu_bufs.d_ver_accumulated_ev)); CL_CHECK(add_kernel.setArg(1, current_gpu_bufs.d_dv)); CL_CHECK(add_kernel.setArg(2, current_gpu_bufs.d_ver_inputs)); CL_CHECK(current_queue.enqueueNDRangeKernel(add_kernel, cl::NullRange, global_add, local_add));
            // CL_CHECK(current_queue.finish()); // Intermediate sync
            // --- Run MLPs Forward
            cl::Kernel mlp_fwd_kernel = get_kernel_with_check(context_obj, "kernelLayerForward");
            cl::Kernel sigmoid_kernel = get_kernel_with_check(context_obj, "clSigmoid1d");
            CL_CHECK(current_queue.enqueueCopyBuffer(current_gpu_bufs.d_hor_inputs, current_gpu_bufs.d_mlp_bufferA_hor, 0, 0, embed_bytes_ph));
            CL_CHECK(current_queue.enqueueCopyBuffer(current_gpu_bufs.d_ver_inputs, current_gpu_bufs.d_mlp_bufferA_ver, 0, 0, embed_bytes_ph));
            cl::Buffer& current_in_hor_mlp = current_gpu_bufs.d_mlp_bufferA_hor; // Renamed to avoid conflict with first overload's current_in_hor
            cl::Buffer& current_out_hor_mlp = current_gpu_bufs.d_mlp_bufferB_hor; // Renamed
            cl::Buffer& current_in_ver_mlp = current_gpu_bufs.d_mlp_bufferA_ver; // Renamed
            cl::Buffer& current_out_ver_mlp = current_gpu_bufs.d_mlp_bufferB_ver; // Renamed
            size_t num_weight_matrices_mlp = head_cpu.hor.weights.size(); // Assuming hor and ver have same num_weight_matrices
            size_t global_mlp_raw_mlp = static_cast<size_t>(d_embedding);
            size_t global_mlp_padded_mlp = ((global_mlp_raw_mlp + local_work_size_1d - 1) / local_work_size_1d) * local_work_size_1d;
            cl::NDRange global_mlp_ndr(global_mlp_padded_mlp); cl::NDRange local_mlp_ndr(local_work_size_1d);

            for (size_t layer_idx_mlp = 0; layer_idx_mlp < num_weight_matrices_mlp; ++layer_idx_mlp) {
                bool is_last_layer_mlp = (layer_idx_mlp == num_weight_matrices_mlp - 1);
                int input_size_mlp = d_embedding; int output_size_mlp = d_embedding;
                CL_CHECK(mlp_fwd_kernel.setArg(3, input_size_mlp));
                CL_CHECK(mlp_fwd_kernel.setArg(4, output_size_mlp));
                { // Horizontal MLP
                    mat& current_weights_mat = head_cpu.hor.weights[layer_idx_mlp];
                    if (static_cast<int>(current_weights_mat.row) != output_size_mlp || static_cast<int>(current_weights_mat.col) != input_size_mlp) {
                        throw std::runtime_error("FATAL ERROR in cl1ParallelForprop (subsequent block): head_cpu.hor.weights[" + std::to_string(layer_idx_mlp) +
                                                 "] dimension mismatch for head [" + std::to_string(layer_idx) + "][" + std::to_string(col_idx_param) + "]." +
                                                 " Expected " + std::to_string(output_size_mlp) + "x" + std::to_string(input_size_mlp) +
                                                 ", got " + std::to_string(current_weights_mat.row) + "x" + std::to_string(current_weights_mat.col));
                    }
                    if (current_weights_mat.mapped_data == nullptr) {
                         throw std::runtime_error("Null mapped_data for hor.weights[" + std::to_string(layer_idx_mlp) + "] in cl1ParallelForprop (subsequent block).");
                    }
                    size_t weights_bytes_mlp = static_cast<size_t>(current_weights_mat.row) * current_weights_mat.col * sizeof(float);
                    cl::Buffer d_mlp_weights(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, weights_bytes_mlp, current_weights_mat.mapped_data, &cl_err); CL_CHECK(cl_err);
                    cl::Buffer& target_output_buffer = is_last_layer_mlp ? current_gpu_bufs.d_hor_output : current_gpu_bufs.d_mlp_pre_activation;
                    CL_CHECK(mlp_fwd_kernel.setArg(0, current_in_hor_mlp));
                    CL_CHECK(mlp_fwd_kernel.setArg(1, d_mlp_weights));
                    CL_CHECK(mlp_fwd_kernel.setArg(2, target_output_buffer));
                    CL_CHECK(current_queue.enqueueNDRangeKernel(mlp_fwd_kernel, cl::NullRange, global_mlp_ndr, local_mlp_ndr));
                    if (!is_last_layer_mlp) {
                        CL_CHECK(sigmoid_kernel.setArg(0, current_gpu_bufs.d_mlp_pre_activation));
                        CL_CHECK(sigmoid_kernel.setArg(1, current_out_hor_mlp));
                        CL_CHECK(sigmoid_kernel.setArg(2, output_size_mlp));
                        CL_CHECK(current_queue.enqueueNDRangeKernel(sigmoid_kernel, cl::NullRange, global_mlp_ndr, local_mlp_ndr));
                        std::swap(current_in_hor_mlp, current_out_hor_mlp);
                    }
                }
                { // Vertical MLP
                    mat& current_weights_mat = head_cpu.ver.weights[layer_idx_mlp];
                    if (static_cast<int>(current_weights_mat.row) != output_size_mlp || static_cast<int>(current_weights_mat.col) != input_size_mlp) {
                        throw std::runtime_error("FATAL ERROR in cl1ParallelForprop (subsequent block): head_cpu.ver.weights[" + std::to_string(layer_idx_mlp) +
                                                 "] dimension mismatch for head [" + std::to_string(layer_idx) + "][" + std::to_string(col_idx_param) + "]." +
                                                 " Expected " + std::to_string(output_size_mlp) + "x" + std::to_string(input_size_mlp) +
                                                 ", got " + std::to_string(current_weights_mat.row) + "x" + std::to_string(current_weights_mat.col));
                    }
                     if (current_weights_mat.mapped_data == nullptr) {
                         throw std::runtime_error("Null mapped_data for ver.weights[" + std::to_string(layer_idx_mlp) + "] in cl1ParallelForprop (subsequent block).");
                    }
                    size_t weights_bytes_mlp = static_cast<size_t>(current_weights_mat.row) * current_weights_mat.col * sizeof(float);
                    cl::Buffer d_mlp_weights(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, weights_bytes_mlp, current_weights_mat.mapped_data, &cl_err); CL_CHECK(cl_err);
                    cl::Buffer& target_output_buffer = is_last_layer_mlp ? current_gpu_bufs.d_ver_output : current_gpu_bufs.d_mlp_pre_activation;
                    CL_CHECK(mlp_fwd_kernel.setArg(0, current_in_ver_mlp));
                    CL_CHECK(mlp_fwd_kernel.setArg(1, d_mlp_weights));
                    CL_CHECK(mlp_fwd_kernel.setArg(2, target_output_buffer));
                    CL_CHECK(current_queue.enqueueNDRangeKernel(mlp_fwd_kernel, cl::NullRange, global_mlp_ndr, local_mlp_ndr));
                    if (!is_last_layer_mlp) {
                        CL_CHECK(sigmoid_kernel.setArg(0, current_gpu_bufs.d_mlp_pre_activation));
                        CL_CHECK(sigmoid_kernel.setArg(1, current_out_ver_mlp));
                        CL_CHECK(sigmoid_kernel.setArg(2, output_size_mlp));
                        CL_CHECK(current_queue.enqueueNDRangeKernel(sigmoid_kernel, cl::NullRange, global_mlp_ndr, local_mlp_ndr));
                        std::swap(current_in_ver_mlp, current_out_ver_mlp);
                    }
                }
            }
            // CL_CHECK(current_queue.finish()); // DEBUG: Sync after entire MLP loop

            // 7. Apply ReLU
            cl::Kernel relu_kernel = get_kernel_with_check(context_obj, "clReLU1d");
            CL_CHECK(relu_kernel.setArg(2, d_embedding));
            CL_CHECK(relu_kernel.setArg(0, current_gpu_bufs.d_hor_output)); CL_CHECK(relu_kernel.setArg(1, current_gpu_bufs.d_relu_hor_output));
            CL_CHECK(current_queue.enqueueNDRangeKernel(relu_kernel, cl::NullRange, global_add, local_add)); // global_add is for d_embedding size
            CL_CHECK(relu_kernel.setArg(0, current_gpu_bufs.d_ver_output)); CL_CHECK(relu_kernel.setArg(1, current_gpu_bufs.d_relu_ver_output));
            CL_CHECK(current_queue.enqueueNDRangeKernel(relu_kernel, cl::NullRange, global_add, local_add));
            CL_CHECK(current_queue.finish()); // Intermediate sync

            // 8. Final Residual Update
            CL_CHECK(add_kernel.setArg(0, current_gpu_bufs.d_EH));
            CL_CHECK(add_kernel.setArg(1, current_gpu_bufs.d_relu_hor_output));
            CL_CHECK(add_kernel.setArg(2, current_gpu_bufs.d_EH));
            CL_CHECK(current_queue.enqueueNDRangeKernel(add_kernel, cl::NullRange, global_add, local_add));

            cl::Kernel update_ev_kernel = get_kernel_with_check(context_obj, "updateEVRowsKernelCL");
            // Note: For subsequent blocks, the output EV (head_cpu.EV) is updated based on the input EVp (now in current_gpu_bufs.d_EV_processed_data)
            // The number of rows to update in head_cpu.EV corresponds to num_ev_rows_to_process_for_evp.
            size_t global_update_ev_raw = static_cast<size_t>(num_ev_rows_to_process_for_evp);
            size_t global_update_ev_padded = ((global_update_ev_raw + local_work_size_1d - 1) / local_work_size_1d) * local_work_size_1d;
            cl::NDRange global_update_ev(global_update_ev_padded); cl::NDRange local_update_ev(local_work_size_1d);

            CL_CHECK(update_ev_kernel.setArg(0, current_gpu_bufs.d_EV_processed_data)); // Source: EVp data
            CL_CHECK(update_ev_kernel.setArg(1, current_gpu_bufs.d_relu_ver_output));
            CL_CHECK(update_ev_kernel.setArg(2, num_ev_rows_to_process_for_evp)); // Number of rows in EVp
            CL_CHECK(update_ev_kernel.setArg(3, d_embedding));
            CL_CHECK(current_queue.enqueueNDRangeKernel(update_ev_kernel, cl::NullRange, global_update_ev, local_update_ev));
            CL_CHECK(current_queue.finish()); // Intermediate sync

            // --- Copy Results D->H
            CL_CHECK(current_queue.enqueueReadBuffer(current_gpu_bufs.d_EH, CL_FALSE, 0, embed_bytes_ph, head_cpu.EH.data()));
            // Copy the updated EV (which originated from EVp_layer and was updated in place on device) back to head_cpu.EV
            // The size of this read should match ev_from_prev_block_bytes_ph
            CL_CHECK(current_queue.enqueueReadBuffer(current_gpu_bufs.d_EV_processed_data, CL_FALSE, 0, ev_from_prev_block_bytes_ph, head_cpu.EV.mapped_data));

            // Explicitly release these specific sub-buffer objects now that their data is copied.
            current_gpu_bufs.d_EH = cl::Buffer();
            current_gpu_bufs.d_EV_processed_data = cl::Buffer();
            CL_CHECK(current_queue.finish());
        }
        catch (const std::exception& e) {
            // Add context to the exception before re-throwing
            throw std::runtime_error("Exception in attention::clforprop (subsequent block overload) for head ["
                                     + std::to_string(layer_idx) + "][" + std::to_string(col_idx_param) + "]: " + e.what());
        }
    }

    // Final synchronization for all heads in this column
    for (int k = 0; k < num_heads_in_col; ++k) {
        CL_CHECK(queues[k].finish());
    }

    // Explicitly release all aggregate OpenCL buffers
    // If a buffer was conditionally not allocated (e.g., count_tokens_in_block == 0),
    // assigning cl::Buffer() to it is safe.
    agg_d_K = cl::Buffer();
    agg_d_Q = cl::Buffer();
    agg_d_KdotQ = cl::Buffer();
    agg_d_head_attention = cl::Buffer();
    agg_d_row_sums = cl::Buffer();
    agg_d_col_sums = cl::Buffer();
    agg_d_EV_processed_data = cl::Buffer(); // This holds EVp data from host
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
 * @brief OpenCL forward propagation on the FIRST block.
 */
void block::clForprop(int& in_dim_param, int& tokenCount_param, int& layers_mlp_param)
{
    for (int j = 0; j < this->y; ++j) { // j is col_idx_param
        try {
            this->cl1parallelForprop(in_dim_param, tokenCount_param, j, layers_mlp_param);
        }
        catch (const std::exception& e) {
            throw std::runtime_error("Exception in cl1parallelForprop (first block) for column ["
                                     + std::to_string(j) + "]: " + e.what());
        }
    }
}


/**
 * @brief OpenCL forward propagation of a SUBSEQUENT block (blockCount > 0).
 */
void block::clForprop(std::vector<std::vector<std::vector<std::vector<float>>>>& EVp_full, 
                      int& in_dim_param, int& totalTokenCount_param, int& blockIdx_param, 
                      int& layers_mlp_param, int& contextWindowSize_param)
{
    if (EVp_full.size() != static_cast<size_t>(x)) {
        throw std::runtime_error("clForprop (subsequent block): EVp_full layer dimension mismatch. Expected "
                                 + std::to_string(x) + " layers, got " + std::to_string(EVp_full.size()) + ".");
    }
    if(!EVp_full.empty() && EVp_full[0].size() != static_cast<size_t>(y)) {
        throw std::runtime_error("clForprop (subsequent block): EVp_full column dimension mismatch. Expected "
                                 + std::to_string(this->y) + " columns, got " + std::to_string(EVp_full[0].size()) + ".");
    }
    
    for (int j = 0; j < this->y; ++j) { // j is col_idx_param
        std::vector<std::vector<std::vector<float>>> EVp_col_j(this->x);
        for (int layer_idx = 0; layer_idx < this->x; ++layer_idx) {
            EVp_col_j[layer_idx] = EVp_full[layer_idx][j];
        }

        try {
            this->cl1ParallelForprop(EVp_col_j, in_dim_param, totalTokenCount_param, blockIdx_param, j, layers_mlp_param, contextWindowSize_param);
        }
        catch (const std::exception& e) {
            throw std::runtime_error("Exception in cl1ParallelForprop (subsequent block) for column ["
                                     + std::to_string(j) + "]: " + e.what());
        }
    }
}

#endif // USE_OPENCL