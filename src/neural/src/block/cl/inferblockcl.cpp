
#ifdef USE_OPENCL

#include <vector>
#include <string>
#include <stdexcept>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <map>
#include <CL/cl.hpp>
#include <maths.hpp>
#include <include/block.hpp>

// Helper macro for OpenCL error checking
#ifndef CL_CHECK
#define CL_CHECK(call) do { \
    cl_int err = call; \
    if (err != CL_SUCCESS) { \
        throw std::runtime_error("OpenCL Error: " + std::to_string(err) + " in " #call + " at " __FILE__ ":" + std::to_string(__LINE__)); \
    } \
} while (0)
#endif

// Helper function to safely get a kernel
static cl::Kernel get_kernel_with_check(OpenCLContext& context_obj, const std::string& kernel_name) {
    auto it = context_obj.kernels.find(kernel_name);
    if (it == context_obj.kernels.end()) {
        throw std::runtime_error("OpenCL kernel not found: '" + kernel_name + "'");
    }
    return it->second;
}

// Struct to manage device buffers for one head's inference pass
struct HeadInferDeviceBuffersCL {
    cl::Buffer d_K, d_Q, d_KdotQ;
    cl::Buffer d_head_attention; 
    cl::Buffer d_row_sums, d_col_sums;
    cl::Buffer d_dh_accum, d_dv_accum;
    cl::Buffer d_khCache, d_qvCache; 
    cl::Buffer d_dh, d_dv;
    cl::Buffer d_EH_head;        
    cl::Buffer d_EV_data_src;      // Source EV data (either head.EV or EVp_head_cpu.EV)
    cl::Buffer d_EV_data_dst;      // Destination EV data (head.EV for output)
    cl::Buffer d_ver_accumulated_ev;
    cl::Buffer d_hor_inputs, d_ver_inputs;
    cl::Buffer d_hor_output, d_ver_output; 
    cl::Buffer d_relu_hor_output, d_relu_ver_output;

    cl::Buffer d_mlp_bufferA_hor, d_mlp_bufferB_hor;
    cl::Buffer d_mlp_bufferA_ver, d_mlp_bufferB_ver;
    cl::Buffer d_mlp_pre_activation; 
};


void block::clInferParallel(const mat& tokens, int& in_dim, int& tokenCount_col, int& layers_mlp, int& col_idx) 
{
    if (col_idx < 0 || col_idx >= this->y) {
        throw std::out_of_range("clInferParallel (first block): column index 'col_idx' (" + std::to_string(col_idx) + ") is out of range [0, " + std::to_string(this->y - 1) + "].");
    }

    cl_int cl_err;
    OpenCLContext& context_obj = this->clcontext;
    cl::Context context = context_obj.context;

    const int num_heads_in_col = this->x;
    const int d_embedding = EMBEDDING; 
    const int n_tokens_current_block = tokenCount_col; 

    if (d_embedding != in_dim) {
        throw std::runtime_error("Embedding dimension mismatch (EMBEDDING vs in_dim).");
    }

    if (n_tokens_current_block <= 0) {
        for (int head_idx = 0; head_idx < num_heads_in_col; ++head_idx) {
            attention& head_cpu = this->b[head_idx][col_idx];
            std::fill(head_cpu.EH.begin(), head_cpu.EH.end(), 0.0f);
        }
        return;
    }

    size_t k_q_bytes_ph = static_cast<size_t>(n_tokens_current_block) * d_embedding * sizeof(float);
    size_t kdotq_bytes_ph = static_cast<size_t>(n_tokens_current_block) * n_tokens_current_block * sizeof(float);
    size_t head_bytes_ph = static_cast<size_t>(n_tokens_current_block) * n_tokens_current_block * sizeof(float);
    size_t sums_bytes_ph = static_cast<size_t>(n_tokens_current_block) * sizeof(float);

    const size_t accum_bytes_ph = static_cast<size_t>(d_embedding) * sizeof(float);
    const size_t cache_mat_bytes_ph = static_cast<size_t>(d_embedding) * d_embedding * sizeof(float);
    const size_t embed_bytes_ph = static_cast<size_t>(d_embedding) * sizeof(float);
    const size_t ev_block_bytes_ph = static_cast<size_t>(CONTEXT_WIN) * d_embedding * sizeof(float);

    cl::Buffer agg_d_K, agg_d_Q, agg_d_KdotQ, agg_d_head_attention, agg_d_row_sums, agg_d_col_sums;
    cl::Buffer agg_d_dh_accum, agg_d_dv_accum, agg_d_khCache, agg_d_qvCache, agg_d_dh, agg_d_dv, agg_d_EH_head;
    cl::Buffer agg_d_EV_data_src, agg_d_EV_data_dst, agg_d_ver_accumulated_ev;
    cl::Buffer agg_d_hor_inputs, agg_d_ver_inputs, agg_d_hor_output, agg_d_ver_output;
    cl::Buffer agg_d_relu_hor_output, agg_d_relu_ver_output;
    cl::Buffer agg_d_mlp_bufferA_hor, agg_d_mlp_bufferB_hor, agg_d_mlp_bufferA_ver, agg_d_mlp_bufferB_ver, agg_d_mlp_pre_activation;

    if (n_tokens_current_block > 0) {
        agg_d_K = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col * k_q_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_Q = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col * k_q_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_KdotQ = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col * kdotq_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_head_attention = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col * head_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_row_sums = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col * sums_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_col_sums = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col * sums_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
    }
    agg_d_dh_accum = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col*accum_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
    agg_d_dv_accum = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col*accum_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
    agg_d_khCache = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col*cache_mat_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
    agg_d_qvCache = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col*cache_mat_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
    agg_d_dh = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col*embed_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
    agg_d_dv = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col*embed_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
    agg_d_EH_head = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col*embed_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
    agg_d_EV_data_src = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col*ev_block_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err); // Source is head.EV
    agg_d_EV_data_dst = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col*ev_block_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err); // Destination is also head.EV (updated)
    agg_d_ver_accumulated_ev = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col*embed_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
    agg_d_hor_inputs = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col*embed_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
    agg_d_ver_inputs = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col*embed_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
    agg_d_hor_output = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col*embed_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
    agg_d_ver_output = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col*embed_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
    agg_d_relu_hor_output = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col*embed_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
    agg_d_relu_ver_output = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col*embed_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
    agg_d_mlp_bufferA_hor = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col*embed_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
    agg_d_mlp_bufferB_hor = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col*embed_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
    agg_d_mlp_bufferA_ver = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col*embed_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
    agg_d_mlp_bufferB_ver = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col*embed_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
    agg_d_mlp_pre_activation = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col*embed_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);

    std::vector<cl::CommandQueue> queues(num_heads_in_col);
    std::vector<HeadInferDeviceBuffersCL> head_gpu_data(num_heads_in_col);

    for (int k = 0; k < num_heads_in_col; ++k) {
        queues[k] = cl::CommandQueue(context, context_obj.device, 0, &cl_err); CL_CHECK(cl_err);
    }

    for (int head_idx = 0; head_idx < num_heads_in_col; ++head_idx) {
        attention& head_cpu = this->b[head_idx][col_idx];
        cl::CommandQueue& current_queue = queues[head_idx];
        HeadInferDeviceBuffersCL& current_gpu_bufs = head_gpu_data[head_idx];

        if ( (n_tokens_current_block > 0 && (head_cpu.K.row != n_tokens_current_block || head_cpu.K.col != d_embedding ||
             head_cpu.Q.row != n_tokens_current_block || head_cpu.Q.col != d_embedding ||
             head_cpu.KdotQ.row != n_tokens_current_block || head_cpu.KdotQ.col != n_tokens_current_block)) ||
            head_cpu.khCache.row != d_embedding || head_cpu.khCache.col != d_embedding ||
            head_cpu.qvCache.row != d_embedding || head_cpu.qvCache.col != d_embedding ||
            head_cpu.EH.size() != static_cast<size_t>(d_embedding) ||
            (!head_cpu.EV.mapped_data || head_cpu.EV.row != CONTEXT_WIN || head_cpu.EV.col != d_embedding) ||
            head_cpu.hor.hlayers.empty() || head_cpu.ver.hlayers.empty() || head_cpu.hor.weights.empty() || head_cpu.ver.weights.empty())
        {
            throw std::runtime_error("Attention component dimension mismatch or uninitialized member in clInferParallel (first block) for head [" +
                                     std::to_string(head_idx) + "][" + std::to_string(col_idx) + "]. K.row=" + std::to_string(head_cpu.K.row) +
                                     ", n_tokens_current_block=" + std::to_string(n_tokens_current_block));
        }

        try {
            auto create_sub_buf = [&](cl::Buffer& parent, size_t offset_multiplier, size_t per_head_bytes) {
                cl_buffer_region region = {offset_multiplier * per_head_bytes, per_head_bytes};
                return parent.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err);
            };

            if (n_tokens_current_block > 0) {
                current_gpu_bufs.d_K = create_sub_buf(agg_d_K, head_idx, k_q_bytes_ph); CL_CHECK(cl_err);
                current_gpu_bufs.d_Q = create_sub_buf(agg_d_Q, head_idx, k_q_bytes_ph); CL_CHECK(cl_err);
                current_gpu_bufs.d_KdotQ = create_sub_buf(agg_d_KdotQ, head_idx, kdotq_bytes_ph); CL_CHECK(cl_err);
                current_gpu_bufs.d_head_attention = create_sub_buf(agg_d_head_attention, head_idx, head_bytes_ph); CL_CHECK(cl_err);
                current_gpu_bufs.d_row_sums = create_sub_buf(agg_d_row_sums, head_idx, sums_bytes_ph); CL_CHECK(cl_err);
                current_gpu_bufs.d_col_sums = create_sub_buf(agg_d_col_sums, head_idx, sums_bytes_ph); CL_CHECK(cl_err);
            }
            current_gpu_bufs.d_dh_accum = create_sub_buf(agg_d_dh_accum, head_idx, accum_bytes_ph); CL_CHECK(cl_err);
            current_gpu_bufs.d_dv_accum = create_sub_buf(agg_d_dv_accum, head_idx, accum_bytes_ph); CL_CHECK(cl_err);
            current_gpu_bufs.d_khCache = create_sub_buf(agg_d_khCache, head_idx, cache_mat_bytes_ph); CL_CHECK(cl_err);
            current_gpu_bufs.d_qvCache = create_sub_buf(agg_d_qvCache, head_idx, cache_mat_bytes_ph); CL_CHECK(cl_err);
            current_gpu_bufs.d_dh = create_sub_buf(agg_d_dh, head_idx, embed_bytes_ph); CL_CHECK(cl_err);
            current_gpu_bufs.d_dv = create_sub_buf(agg_d_dv, head_idx, embed_bytes_ph); CL_CHECK(cl_err);
            current_gpu_bufs.d_EH_head = create_sub_buf(agg_d_EH_head, head_idx, embed_bytes_ph); CL_CHECK(cl_err);
            current_gpu_bufs.d_EV_data_src = create_sub_buf(agg_d_EV_data_src, head_idx, ev_block_bytes_ph); CL_CHECK(cl_err);
            current_gpu_bufs.d_EV_data_dst = create_sub_buf(agg_d_EV_data_dst, head_idx, ev_block_bytes_ph); CL_CHECK(cl_err); // dst is distinct for clarity
            current_gpu_bufs.d_ver_accumulated_ev = create_sub_buf(agg_d_ver_accumulated_ev, head_idx, embed_bytes_ph); CL_CHECK(cl_err);
            current_gpu_bufs.d_hor_inputs = create_sub_buf(agg_d_hor_inputs, head_idx, embed_bytes_ph); CL_CHECK(cl_err);
            current_gpu_bufs.d_ver_inputs = create_sub_buf(agg_d_ver_inputs, head_idx, embed_bytes_ph); CL_CHECK(cl_err);
            current_gpu_bufs.d_hor_output = create_sub_buf(agg_d_hor_output, head_idx, embed_bytes_ph); CL_CHECK(cl_err);
            current_gpu_bufs.d_ver_output = create_sub_buf(agg_d_ver_output, head_idx, embed_bytes_ph); CL_CHECK(cl_err);
            current_gpu_bufs.d_relu_hor_output = create_sub_buf(agg_d_relu_hor_output, head_idx, embed_bytes_ph); CL_CHECK(cl_err);
            current_gpu_bufs.d_relu_ver_output = create_sub_buf(agg_d_relu_ver_output, head_idx, embed_bytes_ph); CL_CHECK(cl_err);
            current_gpu_bufs.d_mlp_bufferA_hor = create_sub_buf(agg_d_mlp_bufferA_hor, head_idx, embed_bytes_ph); CL_CHECK(cl_err);
            current_gpu_bufs.d_mlp_bufferB_hor = create_sub_buf(agg_d_mlp_bufferB_hor, head_idx, embed_bytes_ph); CL_CHECK(cl_err);
            current_gpu_bufs.d_mlp_bufferA_ver = create_sub_buf(agg_d_mlp_bufferA_ver, head_idx, embed_bytes_ph); CL_CHECK(cl_err);
            current_gpu_bufs.d_mlp_bufferB_ver = create_sub_buf(agg_d_mlp_bufferB_ver, head_idx, embed_bytes_ph); CL_CHECK(cl_err);
            current_gpu_bufs.d_mlp_pre_activation = create_sub_buf(agg_d_mlp_pre_activation, head_idx, embed_bytes_ph); CL_CHECK(cl_err);

            float zero_val = 0.0f;
            CL_CHECK(current_queue.enqueueFillBuffer(current_gpu_bufs.d_dh_accum, zero_val, 0, accum_bytes_ph));
            CL_CHECK(current_queue.enqueueFillBuffer(current_gpu_bufs.d_dv_accum, zero_val, 0, accum_bytes_ph));

            if (n_tokens_current_block > 0) {
                CL_CHECK(current_queue.enqueueWriteBuffer(current_gpu_bufs.d_K, CL_FALSE, 0, k_q_bytes_ph, head_cpu.K.mapped_data));
                CL_CHECK(current_queue.enqueueWriteBuffer(current_gpu_bufs.d_Q, CL_FALSE, 0, k_q_bytes_ph, head_cpu.Q.mapped_data));
                CL_CHECK(current_queue.enqueueWriteBuffer(current_gpu_bufs.d_KdotQ, CL_FALSE, 0, kdotq_bytes_ph, head_cpu.KdotQ.mapped_data));
            }
            CL_CHECK(current_queue.enqueueWriteBuffer(current_gpu_bufs.d_khCache, CL_FALSE, 0, cache_mat_bytes_ph, head_cpu.khCache.mapped_data));
            CL_CHECK(current_queue.enqueueWriteBuffer(current_gpu_bufs.d_qvCache, CL_FALSE, 0, cache_mat_bytes_ph, head_cpu.qvCache.mapped_data));
            CL_CHECK(current_queue.enqueueWriteBuffer(current_gpu_bufs.d_EH_head, CL_FALSE, 0, embed_bytes_ph, head_cpu.EH.data()));
            CL_CHECK(current_queue.enqueueWriteBuffer(current_gpu_bufs.d_EV_data_src, CL_FALSE, 0, ev_block_bytes_ph, head_cpu.EV.mapped_data));
            // Copy to d_EV_data_dst as well, as it will be updated in place by updateEVRowsKernelCL
            CL_CHECK(current_queue.enqueueCopyBuffer(current_gpu_bufs.d_EV_data_src, current_gpu_bufs.d_EV_data_dst, 0, 0, ev_block_bytes_ph));


            const size_t local_work_size_1d = 256;
            cl::Kernel lota_kernel = get_kernel_with_check(context_obj, "clLOTA2dmasking");
            size_t totalElementsLOTA = static_cast<size_t>(CONTEXT_WIN) * CONTEXT_WIN;
            if (totalElementsLOTA > 0 && n_tokens_current_block > 0) {
                size_t global_lota_raw = totalElementsLOTA;
                size_t local_lota_clamped = std::min<size_t>(global_lota_raw, local_work_size_1d);
                if (local_lota_clamped == 0) local_lota_clamped = 1;
                size_t global_lota_padded = ((global_lota_raw + local_lota_clamped - 1) / local_lota_clamped) * local_lota_clamped;
                cl::NDRange global_lota(global_lota_padded); 
                cl::NDRange local_lota(local_lota_clamped);
                CL_CHECK(lota_kernel.setArg(0, current_gpu_bufs.d_KdotQ)); 
                CL_CHECK(lota_kernel.setArg(1, current_gpu_bufs.d_head_attention));
                CL_CHECK(lota_kernel.setArg(2, CONTEXT_WIN)); 
                CL_CHECK(lota_kernel.setArg(3, CONTEXT_WIN));
                CL_CHECK(lota_kernel.setArg(4, n_tokens_current_block));
                cl_int cl_att_is_self_lota = head_cpu.isSelfAttention ? 1 : 0;
                CL_CHECK(lota_kernel.setArg(5, cl_att_is_self_lota));
                CL_CHECK(current_queue.enqueueNDRangeKernel(lota_kernel, cl::NullRange, global_lota, local_lota));
            }

            if (n_tokens_current_block > 0) { // Sums, Accum, Proj, MLP only if tokens
                cl::Kernel sums_kernel = get_kernel_with_check(context_obj, "computeHeadSumsMaskedKernel");
                size_t global_sums_raw = static_cast<size_t>(n_tokens_current_block);
                size_t global_sums_padded = ((global_sums_raw + local_work_size_1d - 1) / local_work_size_1d) * local_work_size_1d;
                cl::NDRange global_sums(global_sums_padded); 
                cl::NDRange local_sums(local_work_size_1d);
                cl_int cl_isSelfAttention_sums = head_cpu.isSelfAttention;
                CL_CHECK(sums_kernel.setArg(0, current_gpu_bufs.d_head_attention));
                CL_CHECK(sums_kernel.setArg(1, current_gpu_bufs.d_row_sums));
                CL_CHECK(sums_kernel.setArg(2, current_gpu_bufs.d_col_sums)); 
                CL_CHECK(sums_kernel.setArg(3, n_tokens_current_block));
                CL_CHECK(sums_kernel.setArg(4, cl_isSelfAttention_sums));
                CL_CHECK(current_queue.enqueueNDRangeKernel(sums_kernel, cl::NullRange, global_sums, local_sums));

                cl::Kernel accum_kernel = get_kernel_with_check(context_obj, "accumulateWeightedVectorsKernel");
                size_t global_accum_raw_d = static_cast<size_t>(d_embedding);
                size_t global_accum_padded_d = ((global_accum_raw_d + local_work_size_1d - 1) / local_work_size_1d) * local_work_size_1d;
                cl::NDRange global_accum_d(global_accum_padded_d); cl::NDRange local_accum_d(local_work_size_1d);
                CL_CHECK(accum_kernel.setArg(0, current_gpu_bufs.d_row_sums)); 
                CL_CHECK(accum_kernel.setArg(1, current_gpu_bufs.d_col_sums));
                CL_CHECK(accum_kernel.setArg(2, current_gpu_bufs.d_K)); 
                CL_CHECK(accum_kernel.setArg(3, current_gpu_bufs.d_Q));
                CL_CHECK(accum_kernel.setArg(4, current_gpu_bufs.d_dh_accum)); 
                CL_CHECK(accum_kernel.setArg(5, current_gpu_bufs.d_dv_accum));
                CL_CHECK(accum_kernel.setArg(6, n_tokens_current_block)); 
                CL_CHECK(accum_kernel.setArg(7, d_embedding));
                CL_CHECK(current_queue.enqueueNDRangeKernel(accum_kernel, cl::NullRange, global_accum_d, local_accum_d));

                cl::Kernel proj_kernel = get_kernel_with_check(context_obj, "kernelLayerForward");
                cl::NDRange global_proj = global_accum_d; cl::NDRange local_proj = local_accum_d;
                CL_CHECK(proj_kernel.setArg(3, d_embedding)); 
                CL_CHECK(proj_kernel.setArg(4, d_embedding));
                CL_CHECK(proj_kernel.setArg(0, current_gpu_bufs.d_dh_accum)); 
                CL_CHECK(proj_kernel.setArg(1, current_gpu_bufs.d_khCache)); 
                CL_CHECK(proj_kernel.setArg(2, current_gpu_bufs.d_dh));
                CL_CHECK(current_queue.enqueueNDRangeKernel(proj_kernel, cl::NullRange, global_proj, local_proj));
                CL_CHECK(proj_kernel.setArg(0, current_gpu_bufs.d_dv_accum)); 
                CL_CHECK(proj_kernel.setArg(1, current_gpu_bufs.d_qvCache)); 
                CL_CHECK(proj_kernel.setArg(2, current_gpu_bufs.d_dv));
                CL_CHECK(current_queue.enqueueNDRangeKernel(proj_kernel, cl::NullRange, global_proj, local_proj));

                cl::Kernel add_kernel = get_kernel_with_check(context_obj, "vectorAddKernel");
                cl::NDRange global_add = global_proj; cl::NDRange local_add = local_proj;
                CL_CHECK(add_kernel.setArg(0, current_gpu_bufs.d_EH_head)); 
                CL_CHECK(add_kernel.setArg(1, current_gpu_bufs.d_dh)); 
                CL_CHECK(add_kernel.setArg(2, current_gpu_bufs.d_hor_inputs));
                CL_CHECK(add_kernel.setArg(3, d_embedding));
                CL_CHECK(current_queue.enqueueNDRangeKernel(add_kernel, cl::NullRange, global_add, local_add));

                cl::Kernel accum_ev_kernel = get_kernel_with_check(context_obj, "accumulateEVRowsKernelCL");
                CL_CHECK(accum_ev_kernel.setArg(0, current_gpu_bufs.d_EV_data_src)); 
                CL_CHECK(accum_ev_kernel.setArg(1, current_gpu_bufs.d_ver_accumulated_ev));
                CL_CHECK(accum_ev_kernel.setArg(2, n_tokens_current_block)); // Sum first n_tokens_current_block rows of EV
                CL_CHECK(accum_ev_kernel.setArg(3, d_embedding));
                CL_CHECK(current_queue.enqueueNDRangeKernel(accum_ev_kernel, cl::NullRange, global_add, local_add));
                CL_CHECK(add_kernel.setArg(0, current_gpu_bufs.d_ver_accumulated_ev)); 
                CL_CHECK(add_kernel.setArg(1, current_gpu_bufs.d_dv)); 
                CL_CHECK(add_kernel.setArg(2, current_gpu_bufs.d_ver_inputs));
                CL_CHECK(current_queue.enqueueNDRangeKernel(add_kernel, cl::NullRange, global_add, local_add));

                cl::Kernel mlp_fwd_kernel = get_kernel_with_check(context_obj, "kernelLayerForward");
                cl::Kernel sigmoid_kernel = get_kernel_with_check(context_obj, "clSigmoid1d");
                CL_CHECK(current_queue.enqueueCopyBuffer(current_gpu_bufs.d_hor_inputs, current_gpu_bufs.d_mlp_bufferA_hor, 0, 0, embed_bytes_ph));
                CL_CHECK(current_queue.enqueueCopyBuffer(current_gpu_bufs.d_ver_inputs, current_gpu_bufs.d_mlp_bufferA_ver, 0, 0, embed_bytes_ph));
                cl::Buffer* p_current_in_hor = &current_gpu_bufs.d_mlp_bufferA_hor; 
                cl::Buffer* p_current_out_hor = &current_gpu_bufs.d_mlp_bufferB_hor;
                cl::Buffer* p_current_in_ver = &current_gpu_bufs.d_mlp_bufferA_ver; 
                cl::Buffer* p_current_out_ver = &current_gpu_bufs.d_mlp_bufferB_ver;
                size_t num_weight_matrices = head_cpu.hor.weights.size();
                for (size_t layer_idx_mlp = 0; layer_idx_mlp < num_weight_matrices; ++layer_idx_mlp) {
                    bool is_last_layer_mlp = (layer_idx_mlp == num_weight_matrices - 1); int mlp_io_size = d_embedding;
                    mat& weights_hor_mat = head_cpu.hor.weights[layer_idx_mlp]; 
                    mat& weights_ver_mat = head_cpu.ver.weights[layer_idx_mlp];
                    size_t mlp_weights_bytes = static_cast<size_t>(mlp_io_size) * mlp_io_size * sizeof(float);
                    cl::Buffer d_mlp_weights_hor(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, mlp_weights_bytes, weights_hor_mat.mapped_data, &cl_err); CL_CHECK(cl_err);
                    cl::Buffer d_mlp_weights_ver(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, mlp_weights_bytes, weights_ver_mat.mapped_data, &cl_err); CL_CHECK(cl_err);
                    cl::Buffer& target_hor_mlp = is_last_layer_mlp ? current_gpu_bufs.d_hor_output : current_gpu_bufs.d_mlp_pre_activation;
                    CL_CHECK(mlp_fwd_kernel.setArg(0, *p_current_in_hor)); 
                    CL_CHECK(mlp_fwd_kernel.setArg(1, d_mlp_weights_hor)); 
                    CL_CHECK(mlp_fwd_kernel.setArg(2, target_hor_mlp)); 
                    CL_CHECK(mlp_fwd_kernel.setArg(3, mlp_io_size));
                    CL_CHECK(mlp_fwd_kernel.setArg(4, mlp_io_size));
                    CL_CHECK(current_queue.enqueueNDRangeKernel(mlp_fwd_kernel, cl::NullRange, global_proj, local_proj));
                    if (!is_last_layer_mlp) { 
                        CL_CHECK(sigmoid_kernel.setArg(0, current_gpu_bufs.d_mlp_pre_activation)); 
                        CL_CHECK(sigmoid_kernel.setArg(1, *p_current_out_hor)); 
                        CL_CHECK(sigmoid_kernel.setArg(2, mlp_io_size)); 
                        CL_CHECK(current_queue.enqueueNDRangeKernel(sigmoid_kernel, cl::NullRange, global_proj, local_proj)); 
                        std::swap(p_current_in_hor, p_current_out_hor); 
                    }
                    cl::Buffer& target_ver_mlp = is_last_layer_mlp ? current_gpu_bufs.d_ver_output : current_gpu_bufs.d_mlp_pre_activation;
                    CL_CHECK(mlp_fwd_kernel.setArg(0, *p_current_in_ver)); 
                    CL_CHECK(mlp_fwd_kernel.setArg(1, d_mlp_weights_ver)); 
                    CL_CHECK(mlp_fwd_kernel.setArg(2, target_ver_mlp));
                    CL_CHECK(current_queue.enqueueNDRangeKernel(mlp_fwd_kernel, cl::NullRange, global_proj, local_proj));
                    if (!is_last_layer_mlp) { 
                        CL_CHECK(sigmoid_kernel.setArg(0, current_gpu_bufs.d_mlp_pre_activation)); 
                        CL_CHECK(sigmoid_kernel.setArg(1, *p_current_out_ver)); 
                        CL_CHECK(current_queue.enqueueNDRangeKernel(sigmoid_kernel, cl::NullRange, global_proj, local_proj)); 
                        std::swap(p_current_in_ver, p_current_out_ver); 
                    }
                }

                cl::Kernel relu_kernel = get_kernel_with_check(context_obj, "clReLU1d");
                CL_CHECK(relu_kernel.setArg(0, current_gpu_bufs.d_hor_output)); 
                CL_CHECK(relu_kernel.setArg(1, current_gpu_bufs.d_relu_hor_output));
                CL_CHECK(relu_kernel.setArg(2, d_embedding));
                CL_CHECK(current_queue.enqueueNDRangeKernel(relu_kernel, cl::NullRange, global_add, local_add));
                CL_CHECK(relu_kernel.setArg(0, current_gpu_bufs.d_ver_output)); 
                CL_CHECK(relu_kernel.setArg(1, current_gpu_bufs.d_relu_ver_output));
                CL_CHECK(current_queue.enqueueNDRangeKernel(relu_kernel, cl::NullRange, global_add, local_add));
                // EH += hor.output
                CL_CHECK(add_kernel.setArg(0, current_gpu_bufs.d_EH_head)); 
                CL_CHECK(add_kernel.setArg(1, current_gpu_bufs.d_relu_hor_output)); 
                CL_CHECK(add_kernel.setArg(2, current_gpu_bufs.d_EH_head));
                CL_CHECK(current_queue.enqueueNDRangeKernel(add_kernel, cl::NullRange, global_add, local_add));

                cl::Kernel update_ev_kernel = get_kernel_with_check(context_obj, "updateEVRowsKernelCL");
                size_t global_update_ev_raw = static_cast<size_t>(CONTEXT_WIN); // Update full EV window
                size_t global_update_ev_padded = ((global_update_ev_raw + local_work_size_1d - 1) / local_work_size_1d) * local_work_size_1d;
                cl::NDRange global_update_ev(global_update_ev_padded); cl::NDRange local_update_ev(local_work_size_1d);
                CL_CHECK(update_ev_kernel.setArg(0, current_gpu_bufs.d_EV_data_dst)); // Update the destination EV buffer
                CL_CHECK(update_ev_kernel.setArg(1, current_gpu_bufs.d_relu_ver_output));
                CL_CHECK(update_ev_kernel.setArg(2, CONTEXT_WIN)); 
                CL_CHECK(update_ev_kernel.setArg(3, d_embedding));
                CL_CHECK(current_queue.enqueueNDRangeKernel(update_ev_kernel, cl::NullRange, global_update_ev, local_update_ev));
            } // end if (n_tokens_current_block > 0)

            CL_CHECK(current_queue.enqueueReadBuffer(current_gpu_bufs.d_EH_head, CL_FALSE, 0, embed_bytes_ph, head_cpu.EH.data()));
            CL_CHECK(current_queue.enqueueReadBuffer(current_gpu_bufs.d_EV_data_dst, CL_FALSE, 0, ev_block_bytes_ph, head_cpu.EV.mapped_data));

        } 
        catch (const std::exception& e) {
            throw std::runtime_error("Exception in clInferParallel (first block) for head [" +
                                     std::to_string(head_idx) + "][" + std::to_string(col_idx) + "]: " + e.what());
        }
    } 

    for (int k = 0; k < num_heads_in_col; ++k) {
        CL_CHECK(queues[k].finish());
    }
}


void block::clInferParallel(std::vector<mat>& EVp_col, const mat& tokForBlock, int& in_dim, int& totalTokenCount, 
    int& blockIdx, int& layers_mlp, int& contextWindowSize, int& col_idx) 
{
    if (col_idx < 0 || col_idx >= this->y) {
        throw std::out_of_range("clInferParallel (subsequent block): column index 'col_idx' (" + std::to_string(col_idx) + 
                                ") is out of range [0, " + std::to_string(this->y - 1) + "].");
    }
    if (blockIdx == 0) {
        throw std::logic_error("clInferParallel (subsequent block) called with blockIdx=0. Use the first overload with appropriate tokens.");
        return;
    }

    cl_int cl_err;
    OpenCLContext& context_obj = this->clcontext;
    cl::Context context = context_obj.context;

    const int num_heads_in_col = this->x;
    const int d_embedding = EMBEDDING;

    if (d_embedding != in_dim) {
        throw std::runtime_error("Embedding dimension mismatch (EMBEDDING vs in_dim).");
    }
    if (EVp_col.size() != static_cast<size_t>(num_heads_in_col)) {
        throw std::runtime_error("EVp_col size mismatch. Expected " + std::to_string(num_heads_in_col) + 
                                 " heads, got " + std::to_string(EVp_col.size()));
    }

    int start_idx_in_full_context = (blockIdx - 1) * contextWindowSize;
    int end_idx_in_full_context = std::min<int>(totalTokenCount, blockIdx * contextWindowSize);
    const int n_tokens_current_block = std::max<int>(0, end_idx_in_full_context - start_idx_in_full_context);

    if (n_tokens_current_block <= 0) {
        for (int head_idx = 0; head_idx < num_heads_in_col; ++head_idx) {
            attention& head_cpu = this->b[head_idx][col_idx];
            std::fill(head_cpu.EH.begin(), head_cpu.EH.end(), 0.0f);
            // EV for this block (head_cpu.EV) will be an output, so no need to zero if not written to.
        }
        return;
    }
    
    if (tokForBlock.row != n_tokens_current_block || tokForBlock.col != d_embedding) {
         throw std::runtime_error("tokForBlock dimension mismatch for subsequent block. Expected rows=" + std::to_string(n_tokens_current_block) + 
                                 ", cols=" + std::to_string(d_embedding) + ". Got " + std::to_string(tokForBlock.row) + "x" + std::to_string(tokForBlock.col));
    }

    // --- Buffer sizes and Aggregate Allocations (similar to the first overload, with EVp specific buffer) ---
    // (Sizes are identical to the first overload, as K,Q,KdotQ are based on n_tokens_current_block)
    size_t k_q_bytes_ph = static_cast<size_t>(n_tokens_current_block) * d_embedding * sizeof(float);
    size_t kdotq_bytes_ph = static_cast<size_t>(n_tokens_current_block) * n_tokens_current_block * sizeof(float);
    size_t head_bytes_ph = static_cast<size_t>(n_tokens_current_block) * n_tokens_current_block * sizeof(float);
    size_t sums_bytes_ph = static_cast<size_t>(n_tokens_current_block) * sizeof(float);
    const size_t accum_bytes_ph = static_cast<size_t>(d_embedding) * sizeof(float);
    const size_t cache_mat_bytes_ph = static_cast<size_t>(d_embedding) * d_embedding * sizeof(float);
    const size_t embed_bytes_ph = static_cast<size_t>(d_embedding) * sizeof(float);
    const size_t evp_src_bytes_ph = static_cast<size_t>(CONTEXT_WIN) * d_embedding * sizeof(float); // EVp from prev block
    const size_t this_ev_dst_bytes_ph = static_cast<size_t>(CONTEXT_WIN) * d_embedding * sizeof(float); // This head's EV output

    cl::Buffer agg_d_K, agg_d_Q, agg_d_KdotQ, agg_d_head_attention, agg_d_row_sums, agg_d_col_sums;
    cl::Buffer agg_d_dh_accum, agg_d_dv_accum, agg_d_khCache, agg_d_qvCache, agg_d_dh, agg_d_dv, agg_d_EH_head;
    cl::Buffer agg_d_EVp_data_src, agg_d_this_EV_data_dst, agg_d_ver_accumulated_ev; // Changed EV_data_src to EVp_data_src
    cl::Buffer agg_d_hor_inputs, agg_d_ver_inputs, agg_d_hor_output, agg_d_ver_output;
    cl::Buffer agg_d_relu_hor_output, agg_d_relu_ver_output;
    cl::Buffer agg_d_mlp_bufferA_hor, agg_d_mlp_bufferB_hor, agg_d_mlp_bufferA_ver, agg_d_mlp_bufferB_ver, agg_d_mlp_pre_activation;

    if (n_tokens_current_block > 0) {
        agg_d_K = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col * k_q_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_Q = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col * k_q_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_KdotQ = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col * kdotq_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_head_attention = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col * head_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_row_sums = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col * sums_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_col_sums = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col * sums_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
    }
    agg_d_dh_accum = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col*accum_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
    agg_d_dv_accum = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col*accum_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
    agg_d_khCache = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col*cache_mat_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
    agg_d_qvCache = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col*cache_mat_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
    agg_d_dh = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col*embed_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
    agg_d_dv = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col*embed_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
    agg_d_EH_head = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col*embed_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
    agg_d_EVp_data_src = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col*evp_src_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err); // Source is EVp
    agg_d_this_EV_data_dst = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col*this_ev_dst_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err); // Destination is this head's EV
    agg_d_ver_accumulated_ev = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col*embed_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
    agg_d_hor_inputs = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col*embed_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
    agg_d_ver_inputs = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col*embed_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
    agg_d_hor_output = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col*embed_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
    agg_d_ver_output = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col*embed_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
    agg_d_relu_hor_output = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col*embed_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
    agg_d_relu_ver_output = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col*embed_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
    agg_d_mlp_bufferA_hor = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col*embed_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
    agg_d_mlp_bufferB_hor = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col*embed_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
    agg_d_mlp_bufferA_ver = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col*embed_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
    agg_d_mlp_bufferB_ver = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col*embed_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);
    agg_d_mlp_pre_activation = cl::Buffer(context, CL_MEM_READ_WRITE, num_heads_in_col*embed_bytes_ph, nullptr, &cl_err); CL_CHECK(cl_err);

    std::vector<cl::CommandQueue> queues(num_heads_in_col);
    std::vector<HeadInferDeviceBuffersCL> head_gpu_data(num_heads_in_col);

    for (int k = 0; k < num_heads_in_col; ++k) {
        queues[k] = cl::CommandQueue(context, context_obj.device, 0, &cl_err); CL_CHECK(cl_err);
    }

    for (int head_idx = 0; head_idx < num_heads_in_col; ++head_idx) {
        attention& head_cpu = this->b[head_idx][col_idx];
        mat& EVp_head_cpu = EVp_col[head_idx]; // EV from previous block for this head
        cl::CommandQueue& current_queue = queues[head_idx];
        HeadInferDeviceBuffersCL& current_gpu_bufs = head_gpu_data[head_idx];

        if ( (n_tokens_current_block > 0 && (head_cpu.K.row != n_tokens_current_block || head_cpu.K.col != d_embedding ||
             head_cpu.Q.row != n_tokens_current_block || head_cpu.Q.col != d_embedding ||
             head_cpu.KdotQ.row != n_tokens_current_block || head_cpu.KdotQ.col != n_tokens_current_block)) ||
            head_cpu.khCache.row != d_embedding || head_cpu.khCache.col != d_embedding ||
            head_cpu.qvCache.row != d_embedding || head_cpu.qvCache.col != d_embedding ||
            head_cpu.EH.size() != static_cast<size_t>(d_embedding) ||
            (!EVp_head_cpu.mapped_data || EVp_head_cpu.row != CONTEXT_WIN || EVp_head_cpu.col != d_embedding) || // Validate EVp
            (!head_cpu.EV.mapped_data || head_cpu.EV.row != CONTEXT_WIN || head_cpu.EV.col != d_embedding) || // Validate this head's EV (output)
            head_cpu.hor.hlayers.empty() || head_cpu.ver.hlayers.empty() || head_cpu.hor.weights.empty() || head_cpu.ver.weights.empty())
        {
            throw std::runtime_error("Attention component dimension mismatch or uninitialized member in clInferParallel (subsequent block) for head [" +
                                     std::to_string(head_idx) + "][" + std::to_string(col_idx) + "]. K.row=" + std::to_string(head_cpu.K.row) +
                                     ", n_tokens_current_block=" + std::to_string(n_tokens_current_block) +
                                     ", EVp.row=" + std::to_string(EVp_head_cpu.row));
        }

        try {
            auto create_sub_buf = [&](cl::Buffer& parent, size_t offset_multiplier, size_t per_head_bytes) {
                cl_buffer_region region = {offset_multiplier * per_head_bytes, per_head_bytes};
                return parent.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err);
            };

            if (n_tokens_current_block > 0) {
                current_gpu_bufs.d_K = create_sub_buf(agg_d_K, head_idx, k_q_bytes_ph); CL_CHECK(cl_err);
                current_gpu_bufs.d_Q = create_sub_buf(agg_d_Q, head_idx, k_q_bytes_ph); CL_CHECK(cl_err);
                current_gpu_bufs.d_KdotQ = create_sub_buf(agg_d_KdotQ, head_idx, kdotq_bytes_ph); CL_CHECK(cl_err);
                current_gpu_bufs.d_head_attention = create_sub_buf(agg_d_head_attention, head_idx, head_bytes_ph); CL_CHECK(cl_err);
                current_gpu_bufs.d_row_sums = create_sub_buf(agg_d_row_sums, head_idx, sums_bytes_ph); CL_CHECK(cl_err);
                current_gpu_bufs.d_col_sums = create_sub_buf(agg_d_col_sums, head_idx, sums_bytes_ph); CL_CHECK(cl_err);
            }
            current_gpu_bufs.d_dh_accum = create_sub_buf(agg_d_dh_accum, head_idx, accum_bytes_ph); CL_CHECK(cl_err);
            current_gpu_bufs.d_dv_accum = create_sub_buf(agg_d_dv_accum, head_idx, accum_bytes_ph); CL_CHECK(cl_err);
            current_gpu_bufs.d_khCache = create_sub_buf(agg_d_khCache, head_idx, cache_mat_bytes_ph); CL_CHECK(cl_err);
            current_gpu_bufs.d_qvCache = create_sub_buf(agg_d_qvCache, head_idx, cache_mat_bytes_ph); CL_CHECK(cl_err);
            current_gpu_bufs.d_dh = create_sub_buf(agg_d_dh, head_idx, embed_bytes_ph); CL_CHECK(cl_err);
            current_gpu_bufs.d_dv = create_sub_buf(agg_d_dv, head_idx, embed_bytes_ph); CL_CHECK(cl_err);
            current_gpu_bufs.d_EH_head = create_sub_buf(agg_d_EH_head, head_idx, embed_bytes_ph); CL_CHECK(cl_err);
            current_gpu_bufs.d_EV_data_src = create_sub_buf(agg_d_EVp_data_src, head_idx, evp_src_bytes_ph); CL_CHECK(cl_err);
            current_gpu_bufs.d_EV_data_dst = create_sub_buf(agg_d_this_EV_data_dst, head_idx, this_ev_dst_bytes_ph); CL_CHECK(cl_err);
            current_gpu_bufs.d_ver_accumulated_ev = create_sub_buf(agg_d_ver_accumulated_ev, head_idx, embed_bytes_ph); CL_CHECK(cl_err);
            current_gpu_bufs.d_hor_inputs = create_sub_buf(agg_d_hor_inputs, head_idx, embed_bytes_ph); CL_CHECK(cl_err);
            current_gpu_bufs.d_ver_inputs = create_sub_buf(agg_d_ver_inputs, head_idx, embed_bytes_ph); CL_CHECK(cl_err);
            current_gpu_bufs.d_hor_output = create_sub_buf(agg_d_hor_output, head_idx, embed_bytes_ph); CL_CHECK(cl_err);
            current_gpu_bufs.d_ver_output = create_sub_buf(agg_d_ver_output, head_idx, embed_bytes_ph); CL_CHECK(cl_err);
            current_gpu_bufs.d_relu_hor_output = create_sub_buf(agg_d_relu_hor_output, head_idx, embed_bytes_ph); CL_CHECK(cl_err);
            current_gpu_bufs.d_relu_ver_output = create_sub_buf(agg_d_relu_ver_output, head_idx, embed_bytes_ph); CL_CHECK(cl_err);
            current_gpu_bufs.d_mlp_bufferA_hor = create_sub_buf(agg_d_mlp_bufferA_hor, head_idx, embed_bytes_ph); CL_CHECK(cl_err);
            current_gpu_bufs.d_mlp_bufferB_hor = create_sub_buf(agg_d_mlp_bufferB_hor, head_idx, embed_bytes_ph); CL_CHECK(cl_err);
            current_gpu_bufs.d_mlp_bufferA_ver = create_sub_buf(agg_d_mlp_bufferA_ver, head_idx, embed_bytes_ph); CL_CHECK(cl_err);
            current_gpu_bufs.d_mlp_bufferB_ver = create_sub_buf(agg_d_mlp_bufferB_ver, head_idx, embed_bytes_ph); CL_CHECK(cl_err);
            current_gpu_bufs.d_mlp_pre_activation = create_sub_buf(agg_d_mlp_pre_activation, head_idx, embed_bytes_ph); CL_CHECK(cl_err);

            float zero_val = 0.0f;
            CL_CHECK(current_queue.enqueueFillBuffer(current_gpu_bufs.d_dh_accum, zero_val, 0, accum_bytes_ph));
            CL_CHECK(current_queue.enqueueFillBuffer(current_gpu_bufs.d_dv_accum, zero_val, 0, accum_bytes_ph));

            if (n_tokens_current_block > 0) {
                CL_CHECK(current_queue.enqueueWriteBuffer(current_gpu_bufs.d_K, CL_FALSE, 0, k_q_bytes_ph, head_cpu.K.mapped_data));
                CL_CHECK(current_queue.enqueueWriteBuffer(current_gpu_bufs.d_Q, CL_FALSE, 0, k_q_bytes_ph, head_cpu.Q.mapped_data));
                CL_CHECK(current_queue.enqueueWriteBuffer(current_gpu_bufs.d_KdotQ, CL_FALSE, 0, kdotq_bytes_ph, head_cpu.KdotQ.mapped_data));
            }
            CL_CHECK(current_queue.enqueueWriteBuffer(current_gpu_bufs.d_khCache, CL_FALSE, 0, cache_mat_bytes_ph, head_cpu.khCache.mapped_data));
            CL_CHECK(current_queue.enqueueWriteBuffer(current_gpu_bufs.d_qvCache, CL_FALSE, 0, cache_mat_bytes_ph, head_cpu.qvCache.mapped_data));
            CL_CHECK(current_queue.enqueueWriteBuffer(current_gpu_bufs.d_EH_head, CL_FALSE, 0, embed_bytes_ph, head_cpu.EH.data()));
            CL_CHECK(current_queue.enqueueWriteBuffer(current_gpu_bufs.d_EV_data_src, CL_FALSE, 0, evp_src_bytes_ph, EVp_head_cpu.mapped_data));
            CL_CHECK(current_queue.enqueueWriteBuffer(current_gpu_bufs.d_EV_data_dst, CL_FALSE, 0, this_ev_dst_bytes_ph, head_cpu.EV.mapped_data)); // Initial state of output EV

            // Kernel launch sequence (mirrors the first overload, with adjustments for EV handling)
            const size_t local_work_size_1d = 256;
            cl::Kernel lota_kernel = get_kernel_with_check(context_obj, "clLOTA2dmasking");
            size_t totalElementsLOTA = static_cast<size_t>(CONTEXT_WIN) * CONTEXT_WIN;
            if (totalElementsLOTA > 0 && n_tokens_current_block > 0) {
                size_t global_lota_raw = totalElementsLOTA;
                size_t local_lota_clamped = std::min<size_t>(global_lota_raw, local_work_size_1d);
                if (local_lota_clamped == 0) local_lota_clamped = 1;
                size_t global_lota_padded = ((global_lota_raw + local_lota_clamped - 1) / local_lota_clamped) * local_lota_clamped;
                cl::NDRange global_lota(global_lota_padded); cl::NDRange local_lota(local_lota_clamped);
                CL_CHECK(lota_kernel.setArg(0, current_gpu_bufs.d_KdotQ)); 
                CL_CHECK(lota_kernel.setArg(1, current_gpu_bufs.d_head_attention));
                CL_CHECK(lota_kernel.setArg(2, CONTEXT_WIN)); 
                CL_CHECK(lota_kernel.setArg(3, CONTEXT_WIN));
                CL_CHECK(lota_kernel.setArg(4, n_tokens_current_block));
                cl_int cl_att_is_self_lota = head_cpu.isSelfAttention ? 1 : 0;
                CL_CHECK(lota_kernel.setArg(5, cl_att_is_self_lota));
                CL_CHECK(current_queue.enqueueNDRangeKernel(lota_kernel, cl::NullRange, global_lota, local_lota));
            }

            if (n_tokens_current_block > 0) {
                cl::Kernel sums_kernel = get_kernel_with_check(context_obj, "computeHeadSumsMaskedKernel");
                size_t global_sums_raw = static_cast<size_t>(n_tokens_current_block);
                size_t global_sums_padded = ((global_sums_raw + local_work_size_1d - 1) / local_work_size_1d) * local_work_size_1d;
                cl::NDRange global_sums(global_sums_padded); cl::NDRange local_sums(local_work_size_1d);
                cl_int cl_isSelfAttention_sums = head_cpu.isSelfAttention;
                CL_CHECK(sums_kernel.setArg(0, current_gpu_bufs.d_head_attention)); 
                CL_CHECK(sums_kernel.setArg(1, current_gpu_bufs.d_row_sums));
                CL_CHECK(sums_kernel.setArg(2, current_gpu_bufs.d_col_sums)); 
                CL_CHECK(sums_kernel.setArg(3, n_tokens_current_block));
                CL_CHECK(sums_kernel.setArg(4, cl_isSelfAttention_sums));
                CL_CHECK(current_queue.enqueueNDRangeKernel(sums_kernel, cl::NullRange, global_sums, local_sums));

                cl::Kernel accum_kernel = get_kernel_with_check(context_obj, "accumulateWeightedVectorsKernel");
                size_t global_accum_raw_d = static_cast<size_t>(d_embedding);
                size_t global_accum_padded_d = ((global_accum_raw_d + local_work_size_1d - 1) / local_work_size_1d) * local_work_size_1d;
                cl::NDRange global_accum_d(global_accum_padded_d); cl::NDRange local_accum_d(local_work_size_1d);
                CL_CHECK(accum_kernel.setArg(0, current_gpu_bufs.d_row_sums)); 
                CL_CHECK(accum_kernel.setArg(1, current_gpu_bufs.d_col_sums));
                CL_CHECK(accum_kernel.setArg(2, current_gpu_bufs.d_K));
                CL_CHECK(accum_kernel.setArg(3, current_gpu_bufs.d_Q));
                CL_CHECK(accum_kernel.setArg(4, current_gpu_bufs.d_dh_accum)); 
                CL_CHECK(accum_kernel.setArg(5, current_gpu_bufs.d_dv_accum));
                CL_CHECK(accum_kernel.setArg(6, n_tokens_current_block)); 
                CL_CHECK(accum_kernel.setArg(7, d_embedding));
                CL_CHECK(current_queue.enqueueNDRangeKernel(accum_kernel, cl::NullRange, global_accum_d, local_accum_d));

                cl::Kernel proj_kernel = get_kernel_with_check(context_obj, "kernelLayerForward");
                cl::NDRange global_proj = global_accum_d; cl::NDRange local_proj = local_accum_d;
                CL_CHECK(proj_kernel.setArg(0, current_gpu_bufs.d_dh_accum)); 
                CL_CHECK(proj_kernel.setArg(1, current_gpu_bufs.d_khCache)); 
                CL_CHECK(proj_kernel.setArg(2, current_gpu_bufs.d_dh));
                CL_CHECK(proj_kernel.setArg(3, d_embedding)); 
                CL_CHECK(proj_kernel.setArg(4, d_embedding));
                CL_CHECK(current_queue.enqueueNDRangeKernel(proj_kernel, cl::NullRange, global_proj, local_proj));
                CL_CHECK(proj_kernel.setArg(0, current_gpu_bufs.d_dv_accum)); 
                CL_CHECK(proj_kernel.setArg(1, current_gpu_bufs.d_qvCache)); 
                CL_CHECK(proj_kernel.setArg(2, current_gpu_bufs.d_dv));
                CL_CHECK(current_queue.enqueueNDRangeKernel(proj_kernel, cl::NullRange, global_proj, local_proj));

                cl::Kernel add_kernel = get_kernel_with_check(context_obj, "vectorAddKernel");
                cl::NDRange global_add = global_proj; cl::NDRange local_add = local_proj;
                CL_CHECK(add_kernel.setArg(3, d_embedding));
                CL_CHECK(add_kernel.setArg(0, current_gpu_bufs.d_EH_head)); 
                CL_CHECK(add_kernel.setArg(1, current_gpu_bufs.d_dh)); 
                CL_CHECK(add_kernel.setArg(2, current_gpu_bufs.d_hor_inputs));
                CL_CHECK(current_queue.enqueueNDRangeKernel(add_kernel, cl::NullRange, global_add, local_add));

                cl::Kernel accum_ev_kernel = get_kernel_with_check(context_obj, "accumulateEVRowsKernelCL");
                CL_CHECK(accum_ev_kernel.setArg(0, current_gpu_bufs.d_EV_data_src)); // Use EVp data from previous block
                CL_CHECK(accum_ev_kernel.setArg(1, current_gpu_bufs.d_ver_accumulated_ev));
                CL_CHECK(accum_ev_kernel.setArg(2, CONTEXT_WIN)); // Sum all rows of EVp (CONTEXT_WIN rows)
                CL_CHECK(accum_ev_kernel.setArg(3, d_embedding));
                CL_CHECK(current_queue.enqueueNDRangeKernel(accum_ev_kernel, cl::NullRange, global_add, local_add));
                
                CL_CHECK(add_kernel.setArg(0, current_gpu_bufs.d_ver_accumulated_ev)); 
                CL_CHECK(add_kernel.setArg(1, current_gpu_bufs.d_dv)); 
                CL_CHECK(add_kernel.setArg(2, current_gpu_bufs.d_ver_inputs));
                CL_CHECK(current_queue.enqueueNDRangeKernel(add_kernel, cl::NullRange, global_add, local_add));

                cl::Kernel mlp_fwd_kernel = get_kernel_with_check(context_obj, "kernelLayerForward");
                cl::Kernel sigmoid_kernel = get_kernel_with_check(context_obj, "clSigmoid1d");
                CL_CHECK(current_queue.enqueueCopyBuffer(current_gpu_bufs.d_hor_inputs, current_gpu_bufs.d_mlp_bufferA_hor, 0, 0, embed_bytes_ph));
                CL_CHECK(current_queue.enqueueCopyBuffer(current_gpu_bufs.d_ver_inputs, current_gpu_bufs.d_mlp_bufferA_ver, 0, 0, embed_bytes_ph));
                cl::Buffer* p_current_in_hor = &current_gpu_bufs.d_mlp_bufferA_hor; cl::Buffer* p_current_out_hor = &current_gpu_bufs.d_mlp_bufferB_hor;
                cl::Buffer* p_current_in_ver = &current_gpu_bufs.d_mlp_bufferA_ver; cl::Buffer* p_current_out_ver = &current_gpu_bufs.d_mlp_bufferB_ver;
                size_t num_weight_matrices = head_cpu.hor.weights.size();
                for (size_t layer_idx_mlp = 0; layer_idx_mlp < num_weight_matrices; ++layer_idx_mlp) {
                    bool is_last_layer_mlp = (layer_idx_mlp == num_weight_matrices - 1); int mlp_io_size = d_embedding;
                    mat& weights_hor_mat = head_cpu.hor.weights[layer_idx_mlp]; mat& weights_ver_mat = head_cpu.ver.weights[layer_idx_mlp];
                    size_t mlp_weights_bytes = static_cast<size_t>(mlp_io_size) * mlp_io_size * sizeof(float);
                    
                    cl::Buffer d_mlp_weights_hor(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, mlp_weights_bytes, weights_hor_mat.mapped_data, &cl_err); CL_CHECK(cl_err);
                    cl::Buffer d_mlp_weights_ver(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, mlp_weights_bytes, weights_ver_mat.mapped_data, &cl_err); CL_CHECK(cl_err);
                    cl::Buffer& target_hor_mlp = is_last_layer_mlp ? current_gpu_bufs.d_hor_output : current_gpu_bufs.d_mlp_pre_activation;
                    CL_CHECK(mlp_fwd_kernel.setArg(0, *p_current_in_hor));
                    CL_CHECK(mlp_fwd_kernel.setArg(1, d_mlp_weights_hor)); 
                    CL_CHECK(mlp_fwd_kernel.setArg(2, target_hor_mlp)); 
                    CL_CHECK(mlp_fwd_kernel.setArg(3, mlp_io_size)); 
                    CL_CHECK(mlp_fwd_kernel.setArg(4, mlp_io_size));
                    
                    CL_CHECK(current_queue.enqueueNDRangeKernel(mlp_fwd_kernel, cl::NullRange, global_proj, local_proj));
                    if (!is_last_layer_mlp) { 
                        CL_CHECK(sigmoid_kernel.setArg(0, current_gpu_bufs.d_mlp_pre_activation));
                        CL_CHECK(sigmoid_kernel.setArg(1, *p_current_out_hor)); 
                        CL_CHECK(sigmoid_kernel.setArg(2, mlp_io_size)); 
                        CL_CHECK(current_queue.enqueueNDRangeKernel(sigmoid_kernel, cl::NullRange, global_proj, local_proj)); 
                        std::swap(p_current_in_hor, p_current_out_hor); 
                    }
                    
                    cl::Buffer& target_ver_mlp = is_last_layer_mlp ? current_gpu_bufs.d_ver_output : current_gpu_bufs.d_mlp_pre_activation;
                    CL_CHECK(mlp_fwd_kernel.setArg(0, *p_current_in_ver)); 
                    CL_CHECK(mlp_fwd_kernel.setArg(1, d_mlp_weights_ver)); 
                    CL_CHECK(mlp_fwd_kernel.setArg(2, target_ver_mlp));
                    CL_CHECK(current_queue.enqueueNDRangeKernel(mlp_fwd_kernel, cl::NullRange, global_proj, local_proj));
                    if (!is_last_layer_mlp) { 
                        CL_CHECK(sigmoid_kernel.setArg(0, current_gpu_bufs.d_mlp_pre_activation)); 
                        CL_CHECK(sigmoid_kernel.setArg(1, *p_current_out_ver)); 
                        CL_CHECK(current_queue.enqueueNDRangeKernel(sigmoid_kernel, cl::NullRange, global_proj, local_proj)); 
                        std::swap(p_current_in_ver, p_current_out_ver); 
                    }
                }

                cl::Kernel relu_kernel = get_kernel_with_check(context_obj, "clReLU1d");
                CL_CHECK(relu_kernel.setArg(2, d_embedding));
                CL_CHECK(relu_kernel.setArg(0, current_gpu_bufs.d_hor_output)); 
                CL_CHECK(relu_kernel.setArg(1, current_gpu_bufs.d_relu_hor_output));
                CL_CHECK(current_queue.enqueueNDRangeKernel(relu_kernel, cl::NullRange, global_add, local_add));
                CL_CHECK(relu_kernel.setArg(0, current_gpu_bufs.d_ver_output)); 
                CL_CHECK(relu_kernel.setArg(1, current_gpu_bufs.d_relu_ver_output));
                CL_CHECK(current_queue.enqueueNDRangeKernel(relu_kernel, cl::NullRange, global_add, local_add));

                // EH += ReLU(hor.output)
                CL_CHECK(add_kernel.setArg(0, current_gpu_bufs.d_EH_head)); 
                CL_CHECK(add_kernel.setArg(1, current_gpu_bufs.d_relu_hor_output)); 
                CL_CHECK(add_kernel.setArg(2, current_gpu_bufs.d_EH_head));
                CL_CHECK(current_queue.enqueueNDRangeKernel(add_kernel, cl::NullRange, global_add, local_add));

                cl::Kernel update_ev_kernel = get_kernel_with_check(context_obj, "updateEVRowsKernelCL");
                size_t global_update_ev_raw = static_cast<size_t>(CONTEXT_WIN); // Update full EV window for this head's EV
                size_t global_update_ev_padded = ((global_update_ev_raw + local_work_size_1d - 1) / local_work_size_1d) * local_work_size_1d;
                cl::NDRange global_update_ev(global_update_ev_padded); cl::NDRange local_update_ev(local_work_size_1d);
                CL_CHECK(update_ev_kernel.setArg(0, current_gpu_bufs.d_EV_data_dst)); // Update this head's EV output buffer
                CL_CHECK(update_ev_kernel.setArg(1, current_gpu_bufs.d_relu_ver_output));
                CL_CHECK(update_ev_kernel.setArg(2, CONTEXT_WIN)); 
                CL_CHECK(update_ev_kernel.setArg(3, d_embedding));
                CL_CHECK(current_queue.enqueueNDRangeKernel(update_ev_kernel, cl::NullRange, global_update_ev, local_update_ev));
            } // end if (n_tokens_current_block > 0)

            CL_CHECK(current_queue.enqueueReadBuffer(current_gpu_bufs.d_EH_head, CL_FALSE, 0, embed_bytes_ph, head_cpu.EH.data()));
            CL_CHECK(current_queue.enqueueReadBuffer(current_gpu_bufs.d_EV_data_dst, CL_FALSE, 0, this_ev_dst_bytes_ph, head_cpu.EV.mapped_data));

        } 
        catch (const std::exception& e) {
            throw std::runtime_error("Exception in clInferParallel (subsequent block) for head [" +
                                     std::to_string(head_idx) + "][" + std::to_string(col_idx) + "]: " + e.what());
        }
    } 

    for (int k = 0; k < num_heads_in_col; ++k) {
        CL_CHECK(queues[k].finish());
    }
}


void block::clInfer(const mat& tokens, int& in, int& tokenCount, int& layers) 
{
    for(int j = 0; j < y-1; j++) {
        // parallel execution
        clInferParallel(tokens, in, tokenCount, layers, j);
        if(j == y-1) break;
        for(int i = 0; i < x-1; i++) {
            b[i][j+1].EH = b[i][j].EH;
        }
    }
}


void block::clInfer(const std::vector<std::vector<mat>>& expectedV, const mat& tokForBlock, int& in, 
    int& tokenCount, int& blockCount, int& layers, int& n, int& parallelNumber) 
{
    for(int j = 0; j < y-1; j++) {
        std::vector<mat> eV(x, mat(CONTEXT_WIN, EMBEDDING));
        for(int i = 0; i < x-1; i++) {
            eV[i] = expectedV[i][j];
        }
        // parallel execution
        clInferParallel(eV, tokForBlock, in, tokenCount, blockCount, layers, n, j);
        if(j == y-1) break;
        for(int i = 0; i < x-1; i++) {
            b[i][j+1].EH = b[i][j].EH;
        }
    }
}

#endif
/*    
    // ... (Declare all aggregate cl::Buffer objects as in the first clInferParallel, plus one for EVp source)
    // Example: cl::Buffer agg_d_EVp_data_src = cl::Buffer(context, ..., num_heads_in_col * evp_src_bytes_ph, ...);
    //          cl::Buffer agg_d_this_EV_data_dst = cl::Buffer(context, ..., num_heads_in_col * this_ev_dst_bytes_ph, ...);
    // For brevity, the full list of declarations and allocations is omitted but follows the pattern of the first overload.
    // The implementation would be very similar to the first clInferParallel, with these key changes:
    // 1. `current_gpu_bufs.d_EV_data_src` would be a sub-buffer of `agg_d_EVp_data_src`.
    // 2. `current_gpu_bufs.d_EV_data_dst` would be a sub-buffer of `agg_d_this_EV_data_dst`.
    // 3. H->D: `EVp_head_cpu.mapped_data` -> `current_gpu_bufs.d_EV_data_src`.
    // 4. H->D: `head_cpu.EV.mapped_data` (initial state for output) -> `current_gpu_bufs.d_EV_data_dst`.
    // 5. `accum_ev_kernel` uses `current_gpu_bufs.d_EV_data_src` as input and sums `CONTEXT_WIN` rows (as EVp is full).
    // 6. `update_ev_kernel` updates `current_gpu_bufs.d_EV_data_dst` for `CONTEXT_WIN` rows.
    // 7. D->H: `current_gpu_bufs.d_EV_data_dst` -> `head_cpu.EV.mapped_data`.

    // Due to the significant overlap and length, I'll sketch the main differences rather than repeating all buffer code.
    // The actual implementation would require careful duplication and modification of the first overload's structure.

    // --- (Assume all aggregate buffers are allocated similarly to the first overload) ---
    // --- (Assume HeadInferDeviceBuffersCL struct is used and sub-buffers are created) ---
    // --- (Assume queues are created) ---

    // --- Loop over heads ---
    // for (int head_idx = 0; head_idx < num_heads_in_col; ++head_idx) {
    //     attention& head_cpu = this->b[head_idx][col_idx];
    //     mat& EVp_head_cpu = EVp_col[head_idx];
    //     ... // Get current_queue, current_gpu_bufs

    //     // --- Validate head_cpu and EVp_head_cpu (similar to attention::clInferHead second overload) ---

    //     try {
    //         // --- Create Sub-Buffers (including for EVp_data_src and this_EV_data_dst) ---

    //         // --- H->D Transfers ---
    //         // ... (K, Q, KdotQ, caches, EH as before)
    //         // CL_CHECK(current_queue.enqueueWriteBuffer(current_gpu_bufs.d_EV_data_src, CL_FALSE, 0, evp_src_bytes_ph, EVp_head_cpu.mapped_data));
    //         // CL_CHECK(current_queue.enqueueWriteBuffer(current_gpu_bufs.d_EV_data_dst, CL_FALSE, 0, this_ev_dst_bytes_ph, head_cpu.EV.mapped_data)); // Initial state of output EV

    //         // --- Kernel Launches ---
    //         // ... (LOTA, Sums, Accum K/Q, Proj, Add for hor_inputs - same as first overload using n_tokens_current_block)

    //         // accum_ev_kernel for ver_inputs:
    //         // CL_CHECK(accum_ev_kernel.setArg(0, current_gpu_bufs.d_EV_data_src)); // Use EVp
    //         // CL_CHECK(accum_ev_kernel.setArg(1, current_gpu_bufs.d_ver_accumulated_ev));
    //         // CL_CHECK(accum_ev_kernel.setArg(2, CONTEXT_WIN)); // Sum all rows of EVp
    //         // CL_CHECK(accum_ev_kernel.setArg(3, d_embedding));
    //         // CL_CHECK(current_queue.enqueueNDRangeKernel(accum_ev_kernel, ...));
    //         // ... (Add for ver_inputs, MLPs, ReLU - same as first overload)

    //         // update_ev_kernel for final EV update:
    //         // CL_CHECK(update_ev_kernel.setArg(0, current_gpu_bufs.d_EV_data_dst)); // Update this head's EV output buffer
    //         // CL_CHECK(update_ev_kernel.setArg(1, current_gpu_bufs.d_relu_ver_output));
    //         // CL_CHECK(update_ev_kernel.setArg(2, CONTEXT_WIN)); // Update all rows
    //         // CL_CHECK(update_ev_kernel.setArg(3, d_embedding));
    //         // CL_CHECK(current_queue.enqueueNDRangeKernel(update_ev_kernel, ...));

    //         // --- D->H Transfers ---
    //         // ... (EH as before)
    //         // CL_CHECK(current_queue.enqueueReadBuffer(current_gpu_bufs.d_EV_data_dst, CL_FALSE, 0, this_ev_dst_bytes_ph, head_cpu.EV.mapped_data));

    //     } catch (const std::exception& e) 
    // } // End loop over heads

    // --- Final synchronization ---
    // for (int k = 0; k < num_heads_in_col; ++k) { CL_CHECK(queues[k].finish()); }

    // Placeholder: The full implementation for the second overload is extensive and highly similar in structure
    // to the first one, with the key differences noted above regarding EVp handling.
    // For a complete solution, the first overload's structure should be duplicated and adapted.
*/
