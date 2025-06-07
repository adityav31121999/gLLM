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
 * @brief OpenCL inference for first block's attention head.
 * Mirrors: void attention::inferHead(int &in, int &layers, int &tokenCount)
 * @param d_embedding embedding dimension (in)
 * @param layers_mlp layers of hidden weights in mlp (layers)
 * @param currentTokenCount token count for this attention head (tokenCount)
 */
void attention::clInferHead(const mat& tokens, int& d_embedding, int& layers_mlp, int& currentTokenCount)
{
    const int d = EMBEDDING;        // Embedding dimension. MATHEIGHTS (h) is assumed d.
    const int n = currentTokenCount;// Number of tokens for K, Q, KdotQ
    const int context_win_size = EV.row; // Number of rows in EV, should be CONTEXT_WIN

    if (d != d_embedding) {
        throw std::runtime_error("Embedding dimension mismatch (EMBEDDING vs d_embedding).");
    }

    if (n <= 0) {
        std::cerr << "Warning: clInferHead(..., tokenCount=" << n << ") called with tokenCount <= 0. Skipping." << std::endl;
        std::fill(EH.begin(), EH.end(), 0.0f);
        return;
    }

    if (K.row != n || K.col != d ||
        Q.row != n || Q.col != d ||
        KdotQ.row != n || KdotQ.col != n ||
        khCache.row != d || khCache.col != d ||
        qvCache.row != d || qvCache.col != d ||
        EH.size() != static_cast<size_t>(d) ||
        (!EV.mapped_data || EV.row != context_win_size || EV.col != d) ||
        hor.hlayers.empty() || ver.hlayers.empty() || hor.weights.empty() || ver.weights.empty())
    {
        throw std::runtime_error("Attention component dimension mismatch or uninitialized member in clInferHead(..., tokenCount).");
    }

    try {
        cl_int cl_err;
        OpenCLContext& context_obj = this->clcontext;
        cl::Context context = context_obj.context;
        cl::CommandQueue queue = context_obj.queue;

        size_t k_q_bytes = static_cast<size_t>(n) * d * sizeof(float);
        size_t kdotq_bytes = static_cast<size_t>(n) * n * sizeof(float);
        size_t head_bytes = static_cast<size_t>(n) * n * sizeof(float);
        size_t sums_bytes = static_cast<size_t>(n) * sizeof(float);
        size_t accum_bytes = static_cast<size_t>(d) * sizeof(float); // dh_accum, dv_accum are size d
        size_t cache_mat_bytes = static_cast<size_t>(d) * d * sizeof(float);
        size_t ev_full_bytes = static_cast<size_t>(context_win_size) * d * sizeof(float);
        size_t embed_bytes = static_cast<size_t>(d) * sizeof(float);

        cl::Buffer d_K(context, CL_MEM_READ_ONLY, k_q_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        cl::Buffer d_Q(context, CL_MEM_READ_ONLY, k_q_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        // cl::Buffer d_Q(context, CL_MEM_READ_ONLY, q_q_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        cl::Buffer d_KdotQ(context, CL_MEM_READ_ONLY, kdotq_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        cl::Buffer d_head(context, CL_MEM_READ_WRITE, head_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        cl::Buffer d_row_sums(context, CL_MEM_READ_WRITE, sums_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        cl::Buffer d_col_sums(context, CL_MEM_READ_WRITE, sums_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        cl::Buffer d_dh_accum(context, CL_MEM_READ_WRITE, accum_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        cl::Buffer d_dv_accum(context, CL_MEM_READ_WRITE, accum_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        cl::Buffer d_khCache(context, CL_MEM_READ_ONLY, cache_mat_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        cl::Buffer d_qvCache(context, CL_MEM_READ_ONLY, cache_mat_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        cl::Buffer d_dh(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        cl::Buffer d_dv(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        cl::Buffer d_EH(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        cl::Buffer d_EV_buffer(context, CL_MEM_READ_WRITE, ev_full_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        cl::Buffer d_ver_accumulated_ev(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        cl::Buffer d_hor_inputs(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        cl::Buffer d_ver_inputs(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        cl::Buffer d_hor_output(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        cl::Buffer d_ver_output(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        cl::Buffer d_relu_hor_output(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        cl::Buffer d_relu_ver_output(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);

        cl::Buffer d_mlp_bufferA_hor(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        cl::Buffer d_mlp_bufferB_hor(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        cl::Buffer d_mlp_bufferA_ver(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        cl::Buffer d_mlp_bufferB_ver(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        cl::Buffer d_mlp_pre_activation(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);

        float zero = 0.0f;
        CL_CHECK(queue.enqueueFillBuffer(d_dh_accum, zero, 0, accum_bytes));
        CL_CHECK(queue.enqueueFillBuffer(d_dv_accum, zero, 0, accum_bytes));

        CL_CHECK(queue.enqueueWriteBuffer(d_K, CL_TRUE, 0, k_q_bytes, K.mapped_data));
        CL_CHECK(queue.enqueueWriteBuffer(d_Q, CL_TRUE, 0, k_q_bytes, Q.mapped_data));
        CL_CHECK(queue.enqueueWriteBuffer(d_KdotQ, CL_TRUE, 0, kdotq_bytes, KdotQ.mapped_data));
        CL_CHECK(queue.enqueueWriteBuffer(d_khCache, CL_TRUE, 0, cache_mat_bytes, khCache.mapped_data));
        CL_CHECK(queue.enqueueWriteBuffer(d_qvCache, CL_TRUE, 0, cache_mat_bytes, qvCache.mapped_data));
        CL_CHECK(queue.enqueueWriteBuffer(d_EH, CL_TRUE, 0, embed_bytes, EH.data()));
        CL_CHECK(queue.enqueueWriteBuffer(d_EV_buffer, CL_TRUE, 0, ev_full_bytes, EV.mapped_data));

        const size_t local_work_size_1d = 256;

        cl::Kernel lota_kernel = context_obj.kernels.at("clLOTA2d");
        size_t totalElementsLOTA = static_cast<size_t>(n) * n;
        if (totalElementsLOTA > 0) {
            size_t global_lota_raw = totalElementsLOTA;
            size_t local_lota_clamped = std::min<size_t>(global_lota_raw, local_work_size_1d);
            if (local_lota_clamped == 0) local_lota_clamped = 1;
            size_t global_lota_padded = ((global_lota_raw + local_lota_clamped - 1) / local_lota_clamped) * local_lota_clamped;
            cl::NDRange global_lota(global_lota_padded);
            cl::NDRange local_lota(local_lota_clamped);
            CL_CHECK(lota_kernel.setArg(0, d_KdotQ)); CL_CHECK(lota_kernel.setArg(1, d_head));
            CL_CHECK(lota_kernel.setArg(2, n)); CL_CHECK(lota_kernel.setArg(3, n));
            CL_CHECK(queue.enqueueNDRangeKernel(lota_kernel, cl::NullRange, global_lota, local_lota));
        }

        cl::Kernel sums_kernel = context_obj.kernels.at("computeHeadSumsMaskedKernel");
        size_t global_sums_raw = static_cast<size_t>(n);
        size_t global_sums_padded = ((global_sums_raw + local_work_size_1d - 1) / local_work_size_1d) * local_work_size_1d;
        cl::NDRange global_sums(global_sums_padded);
        cl::NDRange local_sums(local_work_size_1d);
        cl_int cl_isSelfAttention = isSelfAttention;
        CL_CHECK(sums_kernel.setArg(0, d_head)); CL_CHECK(sums_kernel.setArg(1, d_row_sums));
        CL_CHECK(sums_kernel.setArg(2, d_col_sums)); CL_CHECK(sums_kernel.setArg(3, n));
        CL_CHECK(sums_kernel.setArg(4, cl_isSelfAttention));
        CL_CHECK(queue.enqueueNDRangeKernel(sums_kernel, cl::NullRange, global_sums, local_sums));

        cl::Kernel accum_kernel = context_obj.kernels.at("accumulateWeightedVectorsKernel");
        size_t global_accum_raw = static_cast<size_t>(d); // Launch per element of dh_accum (size d)
        size_t global_accum_padded = ((global_accum_raw + local_work_size_1d - 1) / local_work_size_1d) * local_work_size_1d;
        cl::NDRange global_accum_d(global_accum_padded); // For d-sized outputs
        cl::NDRange local_accum_d(local_work_size_1d);
        CL_CHECK(accum_kernel.setArg(0, d_row_sums)); CL_CHECK(accum_kernel.setArg(1, d_col_sums));
        CL_CHECK(accum_kernel.setArg(2, d_K)); CL_CHECK(accum_kernel.setArg(3, d_Q));
        CL_CHECK(accum_kernel.setArg(4, d_dh_accum)); CL_CHECK(accum_kernel.setArg(5, d_dv_accum));
        CL_CHECK(accum_kernel.setArg(6, n)); CL_CHECK(accum_kernel.setArg(7, d)); // h_dim is d
        CL_CHECK(queue.enqueueNDRangeKernel(accum_kernel, cl::NullRange, global_accum_d, local_accum_d));

        cl::Kernel proj_kernel = context_obj.kernels.at("kernelLayerForward");
        size_t global_proj_raw = static_cast<size_t>(d);
        size_t global_proj_padded = ((global_proj_raw + local_work_size_1d - 1) / local_work_size_1d) * local_work_size_1d;
        cl::NDRange global_proj(global_proj_padded);
        cl::NDRange local_proj(local_work_size_1d);
        CL_CHECK(proj_kernel.setArg(0, d_dh_accum)); CL_CHECK(proj_kernel.setArg(1, d_khCache));
        CL_CHECK(proj_kernel.setArg(2, d_dh)); CL_CHECK(proj_kernel.setArg(3, d)); // input_size d
        CL_CHECK(proj_kernel.setArg(4, d)); // output_size d
        CL_CHECK(queue.enqueueNDRangeKernel(proj_kernel, cl::NullRange, global_proj, local_proj));
        CL_CHECK(proj_kernel.setArg(0, d_dv_accum)); CL_CHECK(proj_kernel.setArg(1, d_qvCache));
        CL_CHECK(proj_kernel.setArg(2, d_dv)); // Args 3,4 already d
        CL_CHECK(queue.enqueueNDRangeKernel(proj_kernel, cl::NullRange, global_proj, local_proj));

        cl::Kernel add_kernel = context_obj.kernels.at("vectorAddKernel");
        size_t global_add_raw = static_cast<size_t>(d);
        size_t global_add_padded = ((global_add_raw + local_work_size_1d - 1) / local_work_size_1d) * local_work_size_1d;
        cl::NDRange global_add(global_add_padded);
        cl::NDRange local_add(local_work_size_1d);
        CL_CHECK(add_kernel.setArg(0, d_EH)); CL_CHECK(add_kernel.setArg(1, d_dh));
        CL_CHECK(add_kernel.setArg(2, d_hor_inputs)); CL_CHECK(add_kernel.setArg(3, d));
        CL_CHECK(queue.enqueueNDRangeKernel(add_kernel, cl::NullRange, global_add, local_add));

        cl::Kernel accum_ev_kernel = context_obj.kernels.at("accumulateEVRowsKernelCL");
        CL_CHECK(accum_ev_kernel.setArg(0, d_EV_buffer)); CL_CHECK(accum_ev_kernel.setArg(1, d_ver_accumulated_ev));
        CL_CHECK(accum_ev_kernel.setArg(2, n)); CL_CHECK(accum_ev_kernel.setArg(3, d)); // Sum first n rows
        CL_CHECK(queue.enqueueNDRangeKernel(accum_ev_kernel, cl::NullRange, global_add, local_add)); // global_add is for size d

        CL_CHECK(add_kernel.setArg(0, d_ver_accumulated_ev)); CL_CHECK(add_kernel.setArg(1, d_dv));
        CL_CHECK(add_kernel.setArg(2, d_ver_inputs)); // Arg 3 already d
        CL_CHECK(queue.enqueueNDRangeKernel(add_kernel, cl::NullRange, global_add, local_add));
        CL_CHECK(queue.finish());

        // --- MLPs Forward ( 그대로 사용 ) ---
        cl::Kernel mlp_fwd_kernel = context_obj.kernels.at("kernelLayerForward");
        cl::Kernel sigmoid_kernel = context_obj.kernels.at("clSigmoid1d");
        CL_CHECK(queue.enqueueCopyBuffer(d_hor_inputs, d_mlp_bufferA_hor, 0, 0, embed_bytes));
        CL_CHECK(queue.enqueueCopyBuffer(d_ver_inputs, d_mlp_bufferA_ver, 0, 0, embed_bytes));
        cl::Buffer& current_in_hor = d_mlp_bufferA_hor; cl::Buffer& current_out_hor = d_mlp_bufferB_hor;
        cl::Buffer& current_in_ver = d_mlp_bufferA_ver; cl::Buffer& current_out_ver = d_mlp_bufferB_ver;
        size_t num_weight_matrices = hor.weights.size();

        for (size_t layer_idx = 0; layer_idx < num_weight_matrices; ++layer_idx) {
            bool is_last_layer = (layer_idx == num_weight_matrices - 1);
            int mlp_io_size = d;
            mat& weights_hor_mat = hor.weights[layer_idx];
            mat& weights_ver_mat = ver.weights[layer_idx];
            size_t mlp_weights_bytes = static_cast<size_t>(mlp_io_size) * mlp_io_size * sizeof(float);
            cl::Buffer d_mlp_weights(context, CL_MEM_READ_ONLY , mlp_weights_bytes, nullptr, &cl_err); CL_CHECK(cl_err); // Temp buffer

            // Hor
            CL_CHECK(queue.enqueueWriteBuffer(d_mlp_weights, CL_FALSE, 0, mlp_weights_bytes, weights_hor_mat.mapped_data));
            cl::Buffer& target_hor = is_last_layer ? d_hor_output : d_mlp_pre_activation;
            CL_CHECK(mlp_fwd_kernel.setArg(0, current_in_hor)); CL_CHECK(mlp_fwd_kernel.setArg(1, d_mlp_weights));
            CL_CHECK(mlp_fwd_kernel.setArg(2, target_hor)); CL_CHECK(mlp_fwd_kernel.setArg(3, mlp_io_size)); CL_CHECK(mlp_fwd_kernel.setArg(4, mlp_io_size));
            CL_CHECK(queue.enqueueNDRangeKernel(mlp_fwd_kernel, cl::NullRange, global_proj, local_proj));
            if (!is_last_layer) {
                CL_CHECK(sigmoid_kernel.setArg(0, d_mlp_pre_activation)); CL_CHECK(sigmoid_kernel.setArg(1, current_out_hor)); CL_CHECK(sigmoid_kernel.setArg(2, mlp_io_size));
                CL_CHECK(queue.enqueueNDRangeKernel(sigmoid_kernel, cl::NullRange, global_proj, local_proj));
                std::swap(current_in_hor, current_out_hor);
            }
            // Ver
            CL_CHECK(queue.enqueueWriteBuffer(d_mlp_weights, CL_FALSE, 0, mlp_weights_bytes, weights_ver_mat.mapped_data)); // Non-blocking
            cl::Buffer& target_ver = is_last_layer ? d_ver_output : d_mlp_pre_activation; // Reuse d_mlp_pre_activation
            CL_CHECK(mlp_fwd_kernel.setArg(0, current_in_ver)); CL_CHECK(mlp_fwd_kernel.setArg(1, d_mlp_weights)); CL_CHECK(mlp_fwd_kernel.setArg(2, target_ver));
            CL_CHECK(queue.enqueueNDRangeKernel(mlp_fwd_kernel, cl::NullRange, global_proj, local_proj));
            if (!is_last_layer) {
                CL_CHECK(sigmoid_kernel.setArg(0, d_mlp_pre_activation)); CL_CHECK(sigmoid_kernel.setArg(1, current_out_ver));
                CL_CHECK(queue.enqueueNDRangeKernel(sigmoid_kernel, cl::NullRange, global_proj, local_proj));
                std::swap(current_in_ver, current_out_ver);
            }
        }
        CL_CHECK(queue.finish());
        // --- End MLPs ---

        cl::Kernel relu_kernel = context_obj.kernels.at("clReLU1d");
        CL_CHECK(relu_kernel.setArg(0, d_hor_output)); CL_CHECK(relu_kernel.setArg(1, d_relu_hor_output));
        CL_CHECK(relu_kernel.setArg(2, d));
        CL_CHECK(queue.enqueueNDRangeKernel(relu_kernel, cl::NullRange, global_add, local_add));
        CL_CHECK(relu_kernel.setArg(0, d_ver_output)); CL_CHECK(relu_kernel.setArg(1, d_relu_ver_output));
        CL_CHECK(queue.enqueueNDRangeKernel(relu_kernel, cl::NullRange, global_add, local_add));
        CL_CHECK(queue.finish());

        CL_CHECK(add_kernel.setArg(0, d_EH)); CL_CHECK(add_kernel.setArg(1, d_relu_hor_output));
        CL_CHECK(add_kernel.setArg(2, d_EH)); // Update d_EH in-place
        CL_CHECK(queue.enqueueNDRangeKernel(add_kernel, cl::NullRange, global_add, local_add));

        cl::Kernel update_ev_kernel = context_obj.kernels.at("updateEVRowsKernelCL");
        size_t global_update_ev_raw = static_cast<size_t>(context_win_size);
        size_t global_update_ev_padded = ((global_update_ev_raw + local_work_size_1d - 1) / local_work_size_1d) * local_work_size_1d;
        cl::NDRange global_update_ev(global_update_ev_padded);
        cl::NDRange local_update_ev(local_work_size_1d);
        CL_CHECK(update_ev_kernel.setArg(0, d_EV_buffer)); CL_CHECK(update_ev_kernel.setArg(1, d_relu_ver_output));
        CL_CHECK(update_ev_kernel.setArg(2, context_win_size)); CL_CHECK(update_ev_kernel.setArg(3, d));
        CL_CHECK(queue.enqueueNDRangeKernel(update_ev_kernel, cl::NullRange, global_update_ev, local_update_ev));
        CL_CHECK(queue.finish());

        CL_CHECK(queue.enqueueReadBuffer(d_EH, CL_TRUE, 0, embed_bytes, EH.data()));
        CL_CHECK(queue.enqueueReadBuffer(d_EV_buffer, CL_TRUE, 0, ev_full_bytes, EV.mapped_data));

    }
    catch (const std::runtime_error& e) { // Catch runtime errors from CL_CHECK
        std::cerr << "OpenCL Error in clInferHead(..., tokenCount): " << e.what() << std::endl;
        throw;
    } 
    /*catch (const cl::Error& err) {
        std::cerr << "OpenCL Error in clInferHead(..., tokenCount): " << err.what() << " (" << err.err() << ")" << std::endl;
        throw;
    } */
    catch (const std::exception& e) {
        std::cerr << "Standard Exception in clInferHead(..., tokenCount): " << e.what() << std::endl;
        throw;
    }
}

/**
 * @brief OpenCL inference for a 2nd to last block's attention head.
 * Mirrors: void attention::inferHead(mat& EVp, int& in, int& layers, int& totalTokenCount, int& blockCount, int& n_win)
 * @param EVp_mat EV matrix from previous block
 * @param d_embedding input embedding dimension (in)
 * @param layers_mlp layers of MLPs (layers)
 * @param totalTokenCount total number of tokens processed so far (tokenCount in C++)
 * @param blockIdx which block is being processed (blockCount in C++)
 * @param contextWindowSize number of tokens for each attention head (n_win in C++)
 */
void attention::clInferHead(mat& EVp_mat, const mat& tokForBlock, int& d_embedding, int& layers_mlp, int& totalTokenCount,
    int& blockIdx, int& contextWindowSize)
{
    if (blockIdx == 0) {
        int firstBlockTokenCount = std::min<int>(totalTokenCount, contextWindowSize);
        clInferHead(tokForBlock, d_embedding, layers_mlp, firstBlockTokenCount);
        return;
    }

    const int d = EMBEDDING;
    const int context_win_size_this_ev = EV.row; // For this->EV update

    int start_idx_in_full_context = (blockIdx - 1) * contextWindowSize;
    int end_idx_in_full_context = std::min<int>(totalTokenCount, blockIdx * contextWindowSize);
    const int currentBlockTokenCount = std::max<int>(0, end_idx_in_full_context - start_idx_in_full_context);

    if (d != d_embedding) {
        throw std::runtime_error("Embedding dimension mismatch (EMBEDDING vs d_embedding).");
    }

    if (currentBlockTokenCount <= 0) {
        std::cerr << "Warning: clInferHead(..., blockIdx=" << blockIdx << ") with currentBlockTokenCount <= 0. Skipping." << std::endl;
        std::fill(EH.begin(), EH.end(), 0.0f);
        return;
    }

    if (K.row != currentBlockTokenCount || K.col != d ||
        Q.row != currentBlockTokenCount || Q.col != d ||
        KdotQ.row != currentBlockTokenCount || KdotQ.col != currentBlockTokenCount ||
        khCache.row != d || khCache.col != d ||
        qvCache.row != d || qvCache.col != d ||
        EH.size() != static_cast<size_t>(d) ||
        (!EVp_mat.mapped_data || EVp_mat.row != totalTokenCount || EVp_mat.col != d) ||
        (!EV.mapped_data || EV.row != context_win_size_this_ev || EV.col != d) ||
        hor.hlayers.empty() || ver.hlayers.empty() || hor.weights.empty() || ver.weights.empty())
    {
        throw std::runtime_error("Attention component dimension mismatch or uninitialized member in clInferHead(..., blockIdx).");
    }

    try {
        cl_int cl_err;
        OpenCLContext& context_obj = this->clcontext;
        cl::Context context = context_obj.context;
        cl::CommandQueue queue = context_obj.queue;

        // Sizes
        size_t k_q_bytes = static_cast<size_t>(currentBlockTokenCount) * d * sizeof(float);
        size_t kdotq_bytes = static_cast<size_t>(currentBlockTokenCount) * currentBlockTokenCount * sizeof(float);
        size_t head_bytes = static_cast<size_t>(currentBlockTokenCount) * currentBlockTokenCount * sizeof(float);
        size_t sums_bytes = static_cast<size_t>(currentBlockTokenCount) * sizeof(float);
        size_t accum_bytes = static_cast<size_t>(d) * sizeof(float);
        size_t cache_mat_bytes = static_cast<size_t>(d) * d * sizeof(float);
        size_t evp_bytes = static_cast<size_t>(totalTokenCount) * d * sizeof(float);
        size_t this_ev_bytes = static_cast<size_t>(context_win_size_this_ev) * d * sizeof(float);
        size_t embed_bytes = static_cast<size_t>(d) * sizeof(float);

        // Buffers (similar to first overload, plus d_EVp_data and d_this_EV_buffer)
        cl::Buffer d_K(context, CL_MEM_READ_ONLY, k_q_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        cl::Buffer d_Q(context, CL_MEM_READ_ONLY, k_q_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        cl::Buffer d_KdotQ(context, CL_MEM_READ_ONLY, kdotq_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        cl::Buffer d_head(context, CL_MEM_READ_WRITE, head_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        cl::Buffer d_row_sums(context, CL_MEM_READ_WRITE, sums_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        cl::Buffer d_col_sums(context, CL_MEM_READ_WRITE, sums_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        cl::Buffer d_dh_accum(context, CL_MEM_READ_WRITE, accum_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        cl::Buffer d_dv_accum(context, CL_MEM_READ_WRITE, accum_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        cl::Buffer d_khCache(context, CL_MEM_READ_ONLY, cache_mat_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        cl::Buffer d_qvCache(context, CL_MEM_READ_ONLY, cache_mat_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        cl::Buffer d_dh(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        cl::Buffer d_dv(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        cl::Buffer d_EH(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        cl::Buffer d_EVp_data(context, CL_MEM_READ_ONLY, evp_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        cl::Buffer d_this_EV_buffer(context, CL_MEM_READ_WRITE, this_ev_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        cl::Buffer d_ver_accumulated_ev(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        cl::Buffer d_hor_inputs(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        cl::Buffer d_ver_inputs(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        cl::Buffer d_hor_output(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        cl::Buffer d_ver_output(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        cl::Buffer d_relu_hor_output(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        cl::Buffer d_relu_ver_output(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        cl::Buffer d_mlp_bufferA_hor(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        cl::Buffer d_mlp_bufferB_hor(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        cl::Buffer d_mlp_bufferA_ver(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        cl::Buffer d_mlp_bufferB_ver(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        cl::Buffer d_mlp_pre_activation(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);

        float zero = 0.0f;
        CL_CHECK(queue.enqueueFillBuffer(d_dh_accum, zero, 0, accum_bytes));
        CL_CHECK(queue.enqueueFillBuffer(d_dv_accum, zero, 0, accum_bytes));

        CL_CHECK(queue.enqueueWriteBuffer(d_K, CL_TRUE, 0, k_q_bytes, K.mapped_data));
        CL_CHECK(queue.enqueueWriteBuffer(d_Q, CL_TRUE, 0, k_q_bytes, Q.mapped_data));
        CL_CHECK(queue.enqueueWriteBuffer(d_KdotQ, CL_TRUE, 0, kdotq_bytes, KdotQ.mapped_data));
        CL_CHECK(queue.enqueueWriteBuffer(d_khCache, CL_TRUE, 0, cache_mat_bytes, khCache.mapped_data));
        CL_CHECK(queue.enqueueWriteBuffer(d_qvCache, CL_TRUE, 0, cache_mat_bytes, qvCache.mapped_data));
        CL_CHECK(queue.enqueueWriteBuffer(d_EH, CL_TRUE, 0, embed_bytes, EH.data()));
        CL_CHECK(queue.enqueueWriteBuffer(d_EVp_data, CL_TRUE, 0, evp_bytes, EVp_mat.mapped_data));
        CL_CHECK(queue.enqueueWriteBuffer(d_this_EV_buffer, CL_TRUE, 0, this_ev_bytes, EV.mapped_data));

        const size_t local_work_size_1d = 256;

        // Kernels for LOTA, sums, accumulation (use currentBlockTokenCount for K,Q,KdotQ related sizes)
        cl::Kernel lota_kernel = context_obj.kernels.at("clLOTA2d");
        size_t totalElementsLOTA = static_cast<size_t>(currentBlockTokenCount) * currentBlockTokenCount;
         if (totalElementsLOTA > 0) {
            size_t global_lota_raw = totalElementsLOTA;
            size_t local_lota_clamped = std::min<int>(global_lota_raw, local_work_size_1d);
            if (local_lota_clamped == 0) local_lota_clamped = 1;
            size_t global_lota_padded = ((global_lota_raw + local_lota_clamped - 1) / local_lota_clamped) * local_lota_clamped;
            cl::NDRange global_lota(global_lota_padded); cl::NDRange local_lota(local_lota_clamped);
            CL_CHECK(lota_kernel.setArg(0, d_KdotQ)); CL_CHECK(lota_kernel.setArg(1, d_head));
            CL_CHECK(lota_kernel.setArg(2, currentBlockTokenCount)); CL_CHECK(lota_kernel.setArg(3, currentBlockTokenCount));
            CL_CHECK(queue.enqueueNDRangeKernel(lota_kernel, cl::NullRange, global_lota, local_lota));
        }

        cl::Kernel sums_kernel = context_obj.kernels.at("computeHeadSumsMaskedKernel");
        size_t global_sums_raw = static_cast<size_t>(currentBlockTokenCount);
        size_t global_sums_padded = ((global_sums_raw + local_work_size_1d - 1) / local_work_size_1d) * local_work_size_1d;
        cl::NDRange global_sums(global_sums_padded); cl::NDRange local_sums(local_work_size_1d);
        cl_int cl_isSelfAttention = isSelfAttention;
        CL_CHECK(sums_kernel.setArg(0, d_head)); CL_CHECK(sums_kernel.setArg(1, d_row_sums));
        CL_CHECK(sums_kernel.setArg(2, d_col_sums)); CL_CHECK(sums_kernel.setArg(3, currentBlockTokenCount));
        CL_CHECK(sums_kernel.setArg(4, cl_isSelfAttention));
        CL_CHECK(queue.enqueueNDRangeKernel(sums_kernel, cl::NullRange, global_sums, local_sums));

        cl::Kernel accum_kernel = context_obj.kernels.at("accumulateWeightedVectorsKernel");
        size_t global_accum_raw_d = static_cast<size_t>(d);
        size_t global_accum_padded_d = ((global_accum_raw_d + local_work_size_1d - 1) / local_work_size_1d) * local_work_size_1d;
        cl::NDRange global_accum_d(global_accum_padded_d); cl::NDRange local_accum_d(local_work_size_1d);
        CL_CHECK(accum_kernel.setArg(0, d_row_sums)); CL_CHECK(accum_kernel.setArg(1, d_col_sums));
        CL_CHECK(accum_kernel.setArg(2, d_K)); CL_CHECK(accum_kernel.setArg(3, d_Q));
        CL_CHECK(accum_kernel.setArg(4, d_dh_accum)); CL_CHECK(accum_kernel.setArg(5, d_dv_accum));
        CL_CHECK(accum_kernel.setArg(6, currentBlockTokenCount)); CL_CHECK(accum_kernel.setArg(7, d)); // h_dim is d
        CL_CHECK(queue.enqueueNDRangeKernel(accum_kernel, cl::NullRange, global_accum_d, local_accum_d));

        // Projection, Add kernels (global_proj, global_add are d-sized)
        cl::Kernel proj_kernel = context_obj.kernels.at("kernelLayerForward");
        size_t global_proj_raw = static_cast<size_t>(d);
        size_t global_proj_padded = ((global_proj_raw + local_work_size_1d - 1) / local_work_size_1d) * local_work_size_1d;
        cl::NDRange global_proj(global_proj_padded); cl::NDRange local_proj(local_work_size_1d);
        CL_CHECK(proj_kernel.setArg(0, d_dh_accum)); CL_CHECK(proj_kernel.setArg(1, d_khCache));
        CL_CHECK(proj_kernel.setArg(2, d_dh)); CL_CHECK(proj_kernel.setArg(3, d)); CL_CHECK(proj_kernel.setArg(4, d));
        CL_CHECK(queue.enqueueNDRangeKernel(proj_kernel, cl::NullRange, global_proj, local_proj));
        CL_CHECK(proj_kernel.setArg(0, d_dv_accum)); CL_CHECK(proj_kernel.setArg(1, d_qvCache)); CL_CHECK(proj_kernel.setArg(2, d_dv));
        CL_CHECK(queue.enqueueNDRangeKernel(proj_kernel, cl::NullRange, global_proj, local_proj));

        cl::Kernel add_kernel = context_obj.kernels.at("vectorAddKernel");
        cl::NDRange global_add = global_proj; cl::NDRange local_add = local_proj; // Same dimensions
        CL_CHECK(add_kernel.setArg(0, d_EH)); CL_CHECK(add_kernel.setArg(1, d_dh));
        CL_CHECK(add_kernel.setArg(2, d_hor_inputs)); CL_CHECK(add_kernel.setArg(3, d));
        CL_CHECK(queue.enqueueNDRangeKernel(add_kernel, cl::NullRange, global_add, local_add));

        cl::Kernel accum_ev_kernel = context_obj.kernels.at("accumulateEVRowsKernelCL");
        CL_CHECK(accum_ev_kernel.setArg(0, d_EVp_data)); CL_CHECK(accum_ev_kernel.setArg(1, d_ver_accumulated_ev));
        CL_CHECK(accum_ev_kernel.setArg(2, totalTokenCount)); CL_CHECK(accum_ev_kernel.setArg(3, d)); // Sum totalTokenCount from EVp
        CL_CHECK(queue.enqueueNDRangeKernel(accum_ev_kernel, cl::NullRange, global_add, local_add));

        CL_CHECK(add_kernel.setArg(0, d_ver_accumulated_ev)); CL_CHECK(add_kernel.setArg(1, d_dv)); CL_CHECK(add_kernel.setArg(2, d_ver_inputs));
        CL_CHECK(queue.enqueueNDRangeKernel(add_kernel, cl::NullRange, global_add, local_add));
        CL_CHECK(queue.finish());

        // --- MLPs Forward ( 그대로 사용, global_proj for d-sized MLP I/O ) ---
        // (Same MLP logic as first overload)
        cl::Kernel mlp_fwd_kernel = context_obj.kernels.at("kernelLayerForward");
        cl::Kernel sigmoid_kernel = context_obj.kernels.at("clSigmoid1d");
        CL_CHECK(queue.enqueueCopyBuffer(d_hor_inputs, d_mlp_bufferA_hor, 0, 0, embed_bytes));
        CL_CHECK(queue.enqueueCopyBuffer(d_ver_inputs, d_mlp_bufferA_ver, 0, 0, embed_bytes));
        cl::Buffer& current_in_hor = d_mlp_bufferA_hor; cl::Buffer& current_out_hor = d_mlp_bufferB_hor;
        cl::Buffer& current_in_ver = d_mlp_bufferA_ver; cl::Buffer& current_out_ver = d_mlp_bufferB_ver;
        size_t num_weight_matrices = hor.weights.size();
        for (size_t layer_idx = 0; layer_idx < num_weight_matrices; ++layer_idx) {
            bool is_last_layer = (layer_idx == num_weight_matrices - 1); int mlp_io_size = d;
            mat& weights_hor_mat = hor.weights[layer_idx]; mat& weights_ver_mat = ver.weights[layer_idx];
            size_t mlp_weights_bytes = static_cast<size_t>(mlp_io_size) * mlp_io_size * sizeof(float);
            cl::Buffer d_mlp_weights(context, CL_MEM_READ_ONLY, mlp_weights_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
            CL_CHECK(queue.enqueueWriteBuffer(d_mlp_weights, CL_FALSE, 0, mlp_weights_bytes, weights_hor_mat.mapped_data));
            cl::Buffer& target_hor = is_last_layer ? d_hor_output : d_mlp_pre_activation;
            CL_CHECK(mlp_fwd_kernel.setArg(0, current_in_hor)); CL_CHECK(mlp_fwd_kernel.setArg(1, d_mlp_weights)); CL_CHECK(mlp_fwd_kernel.setArg(2, target_hor)); CL_CHECK(mlp_fwd_kernel.setArg(3, mlp_io_size)); CL_CHECK(mlp_fwd_kernel.setArg(4, mlp_io_size));
            CL_CHECK(queue.enqueueNDRangeKernel(mlp_fwd_kernel, cl::NullRange, global_proj, local_proj));
            if (!is_last_layer) { CL_CHECK(sigmoid_kernel.setArg(0, d_mlp_pre_activation)); CL_CHECK(sigmoid_kernel.setArg(1, current_out_hor)); CL_CHECK(sigmoid_kernel.setArg(2, mlp_io_size)); CL_CHECK(queue.enqueueNDRangeKernel(sigmoid_kernel, cl::NullRange, global_proj, local_proj)); std::swap(current_in_hor, current_out_hor); }
            CL_CHECK(queue.enqueueWriteBuffer(d_mlp_weights, CL_FALSE, 0, mlp_weights_bytes, weights_ver_mat.mapped_data));
            cl::Buffer& target_ver = is_last_layer ? d_ver_output : d_mlp_pre_activation;
            CL_CHECK(mlp_fwd_kernel.setArg(0, current_in_ver)); CL_CHECK(mlp_fwd_kernel.setArg(1, d_mlp_weights)); CL_CHECK(mlp_fwd_kernel.setArg(2, target_ver));
            CL_CHECK(queue.enqueueNDRangeKernel(mlp_fwd_kernel, cl::NullRange, global_proj, local_proj));
            if (!is_last_layer) { CL_CHECK(sigmoid_kernel.setArg(0, d_mlp_pre_activation)); CL_CHECK(sigmoid_kernel.setArg(1, current_out_ver)); CL_CHECK(queue.enqueueNDRangeKernel(sigmoid_kernel, cl::NullRange, global_proj, local_proj)); std::swap(current_in_ver, current_out_ver); }
        }
        CL_CHECK(queue.finish());
        // --- End MLPs ---

        cl::Kernel relu_kernel = context_obj.kernels.at("clReLU1d");
        CL_CHECK(relu_kernel.setArg(0, d_hor_output)); CL_CHECK(relu_kernel.setArg(1, d_relu_hor_output));
        CL_CHECK(relu_kernel.setArg(2, d));
        CL_CHECK(queue.enqueueNDRangeKernel(relu_kernel, cl::NullRange, global_add, local_add));
        CL_CHECK(relu_kernel.setArg(0, d_ver_output)); CL_CHECK(relu_kernel.setArg(1, d_relu_ver_output));
        CL_CHECK(queue.enqueueNDRangeKernel(relu_kernel, cl::NullRange, global_add, local_add));
        CL_CHECK(queue.finish());

        CL_CHECK(add_kernel.setArg(0, d_EH)); CL_CHECK(add_kernel.setArg(1, d_relu_hor_output));
        CL_CHECK(add_kernel.setArg(2, d_EH)); // Update d_EH in-place
        CL_CHECK(queue.enqueueNDRangeKernel(add_kernel, cl::NullRange, global_add, local_add));

        // Update this->EV (d_this_EV_buffer) for context_win_size_this_ev rows
        cl::Kernel update_ev_kernel = context_obj.kernels.at("updateEVRowsKernelCL");
        size_t global_update_ev_raw = static_cast<size_t>(context_win_size_this_ev);
        size_t global_update_ev_padded = ((global_update_ev_raw + local_work_size_1d - 1) / local_work_size_1d) * local_work_size_1d;
        cl::NDRange global_update_ev(global_update_ev_padded);
        cl::NDRange local_update_ev(local_work_size_1d);
        CL_CHECK(update_ev_kernel.setArg(0, d_this_EV_buffer)); CL_CHECK(update_ev_kernel.setArg(1, d_relu_ver_output));
        CL_CHECK(update_ev_kernel.setArg(2, context_win_size_this_ev)); CL_CHECK(update_ev_kernel.setArg(3, d));
        CL_CHECK(queue.enqueueNDRangeKernel(update_ev_kernel, cl::NullRange, global_update_ev, local_update_ev));
        CL_CHECK(queue.finish());

        CL_CHECK(queue.enqueueReadBuffer(d_EH, CL_TRUE, 0, embed_bytes, EH.data()));
        CL_CHECK(queue.enqueueReadBuffer(d_this_EV_buffer, CL_TRUE, 0, this_ev_bytes, EV.mapped_data));

    } 
    catch (const std::exception& e) {
        std::cerr << "Standard Exception in clInferHead(..., blockIdx): " << e.what() << std::endl;
        throw;
    }
}

#endif // USE_OPENCL