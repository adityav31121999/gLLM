
#ifdef USE_OPENCL

#include "include/block.hpp"
#include <vector>
#include <stdexcept>
#include <string>
#include <map>
#include <maths.hpp>
#include <CL/cl.hpp>


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

// Helper function to flatten a 2D vector<vector<float>> into a flat vector (row-major)
// (Copied from previous CUDA diff for consistency, assuming needed for EVp)
static void flatten2DVector(const std::vector<std::vector<float>>& vec2d, std::vector<float>& output_flat, size_t expected_rows, size_t expected_cols) {
    if (vec2d.empty()) {
        output_flat.clear();
        if (expected_rows != 0) { // Only throw if rows were expected but vec2d is empty
            throw std::runtime_error("Input 2D vector is empty but expected " + std::to_string(expected_rows) + " rows.");
        }
        return; // Valid empty if 0 rows expected
    }
    size_t R = vec2d.size();
    if (R != expected_rows) {
        throw std::runtime_error("Row count mismatch in flatten2DVector. Expected " + std::to_string(expected_rows) + ", got " + std::to_string(R));
    }

    size_t C = 0;
    if (R > 0) {
        C = vec2d[0].size();
        if (C != expected_cols) {
             throw std::runtime_error("Column count mismatch in flatten2DVector for row 0. Expected " + std::to_string(expected_cols) + ", got " + std::to_string(C));
        }
    } 
    else if (expected_cols != 0) { // R is 0, but expected_cols is not.
         throw std::runtime_error("Column count mismatch in flatten2DVector: 0 rows but expected " + std::to_string(expected_cols) + " columns.");
    }
    output_flat.resize(R * C);
    for (size_t r_idx = 0; r_idx < R; ++r_idx) {
        if (vec2d[r_idx].size() != C) {
            throw std::runtime_error("Inconsistent column count in flatten2DVector at row " + std::to_string(r_idx) + ". Expected " + std::to_string(C) + ", got " + std::to_string(vec2d[r_idx].size()));
        }
        for (size_t c_idx = 0; c_idx < C; ++c_idx) {
            output_flat[r_idx * C + c_idx] = vec2d[r_idx][c_idx];
        }
    }
}

// Helper function to safely get a kernel from the context's map
static cl::Kernel get_kernel_with_check(OpenCLContext& context_obj, const std::string& kernel_name) {
    auto it = context_obj.kernels.find(kernel_name);
    if (it == context_obj.kernels.end()) {
        throw std::runtime_error("OpenCL kernel not found in context: '" + kernel_name +
                                 "'. Check OpenCLContext initialization and kernel compilation/naming.");
    }
    return it->second;
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
void block::cl1parallelForprop(int& in, int& tokenCount, int i, int& layers)
{
    if (i < 0 || i >= this->y) {
        throw std::out_of_range("cl1parallelForprop (first block): column index 'i' (" + std::to_string(i) + ") is out of range [0, " + std::to_string(this->y - 1) + "].");
    }

    cl_int cl_err; // For OpenCL error codes
    OpenCLContext& context_obj = this->clcontext; // Use the member reference
    cl::Context context = context_obj.context;
    cl::CommandQueue queue = context_obj.queue;
    // Use std::map<std::string, cl::Kernel> kernels = context_obj.kernels;

    // Iterate through the layers (rows) of attention heads in the specified column 'i'
    for (int layer_idx = 0; layer_idx < this->x; ++layer_idx) {
        attention& head = this->b[layer_idx][i];

        const int d = EMBEDDING;        // Embedding dimension
        const int h = MATHEIGHTS;       // Height dimension of K/Q, projection matrices
        const int n = tokenCount;       // Number of tokens
        const int num_ev_rows_to_process = tokenCount; // For EV.sumRows/addToRows

        if (n <= 0) {
            // std::cerr << "Warning: cl1parallelForprop (first block) for head [" << layer_idx << "][" << i << "] called with tokenCount <= 0. Skipping." << std::endl;
            std::fill(head.EH.begin(), head.EH.end(), 0.0f);
            if (n == 0 && head.EV.mapped_data && head.EV.row > 0 && head.EV.col == d) {
                std::fill_n(head.EV.mapped_data, head.EV.col, 0.0f);
            }
            continue; // Skip to next head in the column
        }

        // --- Basic Validation (Copied from attention::clforprop) ---
        if (head.EV.row < n || head.EV.col != d) {
            throw std::runtime_error("EV matrix not properly sized for current tokenCount n in cl1parallelForprop (first block) for head [" + std::to_string(layer_idx) + "][" + std::to_string(i) + "]. Expected rows >= " + std::to_string(n) + " and cols = " + std::to_string(d) +
                                     ", but got rows = " + std::to_string(head.EV.row) + " and cols = " + std::to_string(head.EV.col));
        }
        if (head.K.row != CONTEXT_WIN || head.K.col != EMBEDDING ||
            head.Q.row != CONTEXT_WIN || head.Q.col != EMBEDDING ||
            head.KdotQ.row != CONTEXT_WIN || head.KdotQ.col != CONTEXT_WIN ||
            head.MH.row != EMBEDDING || head.MH.col != MATHEIGHTS ||
            head.MV.row != EMBEDDING || head.MV.col != MATHEIGHTS ||
            head.EH.size() != static_cast<size_t>(d) ||
            head.hor.hlayers.empty() || head.ver.hlayers.empty() || head.hor.weights.empty() || head.ver.weights.empty())
        {
            throw std::runtime_error("Attention component dimension mismatch or uninitialized member in cl1parallelForprop (first block) for head [" +
                                     std::to_string(layer_idx) + "][" + std::to_string(i) + "]. K.row=" + std::to_string(head.K.row) + ", n=" + std::to_string(n));
        }
        if (d != in) {
            throw std::runtime_error("Embedding dimension mismatch (EMBEDDING vs in) for head [" + std::to_string(layer_idx) + "][" + std::to_string(i) + "].");
        }
        if (head.hor.hlayers[0].size() != static_cast<size_t>(d) || head.ver.hlayers[0].size() != static_cast<size_t>(d) ||
            head.hor.weights.back().row != static_cast<size_t>(d) || head.ver.weights.back().row != static_cast<size_t>(d)) {
             throw std::runtime_error("MLP input/output layer dimension mismatch with 'd' for head [" + std::to_string(layer_idx) + "][" + std::to_string(i) + "].");
        }

        // --- Allocate Device Memory (Copied from attention::clforprop) ---
        size_t k_bytes = static_cast<size_t>(n) * d * sizeof(float);
        size_t q_bytes = static_cast<size_t>(n) * d * sizeof(float);
        size_t kdotq_bytes = static_cast<size_t>(n) * n * sizeof(float);
        size_t head_bytes = static_cast<size_t>(n) * n * sizeof(float);
        size_t sums_bytes = static_cast<size_t>(n) * sizeof(float);
        size_t accum_bytes = static_cast<size_t>(h) * sizeof(float);
        size_t proj_mat_bytes = static_cast<size_t>(d) * h * sizeof(float); // MH/MV are d x h
        size_t ev_processed_bytes = static_cast<size_t>(n) * d * sizeof(float); // For first 'n' rows of EV
        size_t embed_bytes = static_cast<size_t>(d) * sizeof(float);        // d

        try {
            cl::Buffer d_K(context, CL_MEM_READ_ONLY, k_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
            cl::Buffer d_Q(context, CL_MEM_READ_ONLY, q_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
            cl::Buffer d_KdotQ(context, CL_MEM_READ_ONLY, kdotq_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
            cl::Buffer d_head_attention(context, CL_MEM_READ_WRITE, head_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
            cl::Buffer d_row_sums(context, CL_MEM_READ_WRITE, sums_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
            cl::Buffer d_col_sums(context, CL_MEM_READ_WRITE, sums_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
            cl::Buffer d_dh_accum(context, CL_MEM_READ_WRITE, accum_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
            cl::Buffer d_dv_accum(context, CL_MEM_READ_WRITE, accum_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
            cl::Buffer d_MH_hxd(context, CL_MEM_READ_ONLY, proj_mat_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
            cl::Buffer d_MV_hxd(context, CL_MEM_READ_ONLY, proj_mat_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
            cl::Buffer d_dh (context, CL_MEM_READ_WRITE, embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
            cl::Buffer d_dv(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
            cl::Buffer d_EH(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
            cl::Buffer d_EV_processed_data(context, CL_MEM_READ_WRITE, ev_processed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
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

            // Initialize accumulators to zero
            float zero = 0.0f;
            CL_CHECK(queue.enqueueFillBuffer(d_dh_accum, zero, 0, accum_bytes));
            CL_CHECK(queue.enqueueFillBuffer(d_dv_accum, zero, 0, accum_bytes));

            // --- Data Transfer H->D (Copied from attention::clforprop) ---
            CL_CHECK(queue.enqueueWriteBuffer(d_K, CL_TRUE, 0, k_bytes, head.K.mapped_data));
            CL_CHECK(queue.enqueueWriteBuffer(d_Q, CL_TRUE, 0, q_bytes, head.Q.mapped_data));
            CL_CHECK(queue.enqueueWriteBuffer(d_KdotQ, CL_TRUE, 0, kdotq_bytes, head.KdotQ.mapped_data));
            CL_CHECK(queue.enqueueWriteBuffer(d_MH_hxd, CL_TRUE, 0, proj_mat_bytes, head.MH.mapped_data)); // MH is mat(d,h)
            CL_CHECK(queue.enqueueWriteBuffer(d_MV_hxd, CL_TRUE, 0, proj_mat_bytes, head.MV.mapped_data)); // MV is mat(d,h)
            CL_CHECK(queue.enqueueWriteBuffer(d_EH, CL_TRUE, 0, embed_bytes, head.EH.data()));
            CL_CHECK(queue.enqueueWriteBuffer(d_EV_processed_data, CL_TRUE, 0, ev_processed_bytes, head.EV.mapped_data));

            // --- Kernel Launches (Copied from attention::clforprop) ---
            const size_t local_work_size_1d = 256;
            cl::Kernel lota_kernel = get_kernel_with_check(context_obj, "clLOTA2d");
            size_t totalElementsLOTA = static_cast<size_t>(n) * n;
            if (totalElementsLOTA > 0) {
                size_t global_lota_raw = totalElementsLOTA; //NOLINT
                size_t local_lota_clamped = (std::min)(global_lota_raw, local_work_size_1d);
                if (local_lota_clamped == 0) local_lota_clamped = 1;
                size_t global_lota_padded = ((global_lota_raw + local_lota_clamped - 1) / local_lota_clamped) * local_lota_clamped;
                cl::NDRange global_lota(global_lota_padded);
                cl::NDRange local_lota(local_lota_clamped);
                CL_CHECK(lota_kernel.setArg(0, d_KdotQ)); 
                CL_CHECK(lota_kernel.setArg(1, d_head_attention)); 
                CL_CHECK(lota_kernel.setArg(2, n));
                CL_CHECK(lota_kernel.setArg(3, n));
                CL_CHECK(queue.enqueueNDRangeKernel(lota_kernel, cl::NullRange, global_lota, local_lota));
            } //NOLINT
            cl::Kernel sums_kernel = get_kernel_with_check(context_obj, "computeHeadSumsMaskedKernel");
            size_t global_sums_raw = static_cast<size_t>(n); size_t global_sums_padded = ((global_sums_raw + local_work_size_1d - 1) / local_work_size_1d) * local_work_size_1d; cl::NDRange global_sums(global_sums_padded); cl::NDRange local_sums(local_work_size_1d); cl_int cl_isSelfAttention = head.isSelfAttention;
            CL_CHECK(sums_kernel.setArg(0, d_head_attention)); CL_CHECK(sums_kernel.setArg(1, d_row_sums)); CL_CHECK(sums_kernel.setArg(2, d_col_sums)); CL_CHECK(sums_kernel.setArg(3, n)); CL_CHECK(sums_kernel.setArg(4, cl_isSelfAttention));
            CL_CHECK(queue.enqueueNDRangeKernel(sums_kernel, cl::NullRange, global_sums, local_sums));
            cl::Kernel accum_kernel = get_kernel_with_check(context_obj, "accumulateWeightedVectorsKernel");
            size_t global_accum_raw = static_cast<size_t>(n); size_t global_accum_padded = ((global_accum_raw + local_work_size_1d - 1) / local_work_size_1d) * local_work_size_1d; cl::NDRange global_accum(global_accum_padded); cl::NDRange local_accum(local_work_size_1d);
            CL_CHECK(accum_kernel.setArg(0, d_row_sums)); CL_CHECK(accum_kernel.setArg(1, d_col_sums)); CL_CHECK(accum_kernel.setArg(2, d_K)); CL_CHECK(accum_kernel.setArg(3, d_Q)); CL_CHECK(accum_kernel.setArg(4, d_dh_accum)); CL_CHECK(accum_kernel.setArg(5, d_dv_accum)); CL_CHECK(accum_kernel.setArg(6, n)); CL_CHECK(accum_kernel.setArg(7, h));
            CL_CHECK(queue.enqueueNDRangeKernel(accum_kernel, cl::NullRange, global_accum, local_accum));
            cl::Kernel proj_kernel = get_kernel_with_check(context_obj, "kernelLayerForward");
            size_t global_proj_raw = static_cast<size_t>(d); size_t global_proj_padded = ((global_proj_raw + local_work_size_1d - 1) / local_work_size_1d) * local_work_size_1d; cl::NDRange global_proj(global_proj_padded); cl::NDRange local_proj(local_work_size_1d);
            CL_CHECK(proj_kernel.setArg(3, h)); CL_CHECK(proj_kernel.setArg(4, d));
            CL_CHECK(proj_kernel.setArg(0, d_dh_accum)); CL_CHECK(proj_kernel.setArg(1, d_MH_hxd)); CL_CHECK(proj_kernel.setArg(2, d_dh)); CL_CHECK(queue.enqueueNDRangeKernel(proj_kernel, cl::NullRange, global_proj, local_proj));
            CL_CHECK(proj_kernel.setArg(0, d_dv_accum)); CL_CHECK(proj_kernel.setArg(1, d_MV_hxd)); CL_CHECK(proj_kernel.setArg(2, d_dv)); CL_CHECK(queue.enqueueNDRangeKernel(proj_kernel, cl::NullRange, global_proj, local_proj));
            CL_CHECK(queue.finish());
            cl::Kernel add_kernel = get_kernel_with_check(context_obj, "vectorAddKernel");
            size_t global_add_raw = static_cast<size_t>(d); size_t global_add_padded = ((global_add_raw + local_work_size_1d - 1) / local_work_size_1d) * local_work_size_1d; cl::NDRange global_add(global_add_padded); cl::NDRange local_add(local_work_size_1d); CL_CHECK(add_kernel.setArg(3, d));
            CL_CHECK(add_kernel.setArg(0, d_EH)); CL_CHECK(add_kernel.setArg(1, d_dh)); CL_CHECK(add_kernel.setArg(2, d_hor_inputs)); CL_CHECK(queue.enqueueNDRangeKernel(add_kernel, cl::NullRange, global_add, local_add));
            cl::Kernel accum_ev_kernel = get_kernel_with_check(context_obj, "accumulateEVRowsKernelCL");
            CL_CHECK(accum_ev_kernel.setArg(0, d_EV_processed_data)); CL_CHECK(accum_ev_kernel.setArg(1, d_ver_accumulated_ev)); CL_CHECK(accum_ev_kernel.setArg(2, n)); CL_CHECK(accum_ev_kernel.setArg(3, d));
            CL_CHECK(queue.enqueueNDRangeKernel(accum_ev_kernel, cl::NullRange, global_add, local_add));
            CL_CHECK(add_kernel.setArg(0, d_ver_accumulated_ev)); CL_CHECK(add_kernel.setArg(1, d_dv)); CL_CHECK(add_kernel.setArg(2, d_ver_inputs)); CL_CHECK(queue.enqueueNDRangeKernel(add_kernel, cl::NullRange, global_add, local_add));
            CL_CHECK(queue.finish());

            // --- Run MLPs Forward (Copied from attention::clforprop) ---
            cl::Kernel mlp_fwd_kernel = get_kernel_with_check(context_obj, "kernelLayerForward"); // Re-fetch or ensure it's the same as proj_kernel
            cl::Kernel sigmoid_kernel = get_kernel_with_check(context_obj, "clSigmoid1d");
            CL_CHECK(queue.enqueueCopyBuffer(d_hor_inputs, d_mlp_bufferA_hor, 0, 0, embed_bytes));
            CL_CHECK(queue.enqueueCopyBuffer(d_ver_inputs, d_mlp_bufferA_ver, 0, 0, embed_bytes));
            cl::Buffer& current_in_hor = d_mlp_bufferA_hor; cl::Buffer& current_out_hor = d_mlp_bufferB_hor;
            cl::Buffer& current_in_ver = d_mlp_bufferA_ver; cl::Buffer& current_out_ver = d_mlp_bufferB_ver;
            size_t num_weight_matrices = head.hor.weights.size();
            size_t global_mlp_raw = static_cast<size_t>(d); size_t global_mlp_padded = ((global_mlp_raw + local_work_size_1d - 1) / local_work_size_1d) * local_work_size_1d; cl::NDRange global_mlp(global_mlp_padded); cl::NDRange local_mlp(local_work_size_1d);
            for (size_t layer_idx_mlp = 0; layer_idx_mlp < num_weight_matrices; ++layer_idx_mlp) {
                bool is_last_layer_mlp = (layer_idx_mlp == num_weight_matrices - 1); 
                int input_size_mlp = d; int output_size_mlp = d;
                { 
                    mat& current_weights_mat = head.hor.weights[layer_idx_mlp]; 
                    size_t weights_bytes_mlp = static_cast<size_t>(current_weights_mat.row) * current_weights_mat.col * sizeof(float); 
                    cl::Buffer d_mlp_weights(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, weights_bytes_mlp, current_weights_mat.mapped_data, &cl_err); CL_CHECK(cl_err); 
                    cl::Buffer& target_output_buffer = is_last_layer_mlp ? d_hor_output : d_mlp_pre_activation; 
                    CL_CHECK(mlp_fwd_kernel.setArg(0, current_in_hor)); CL_CHECK(mlp_fwd_kernel.setArg(1, d_mlp_weights)); 
                    CL_CHECK(mlp_fwd_kernel.setArg(2, target_output_buffer)); CL_CHECK(mlp_fwd_kernel.setArg(3, input_size_mlp)); 
                    CL_CHECK(mlp_fwd_kernel.setArg(4, output_size_mlp)); 
                    CL_CHECK(queue.enqueueNDRangeKernel(mlp_fwd_kernel, cl::NullRange, global_mlp, local_mlp)); 
                    if (!is_last_layer_mlp) { 
                        CL_CHECK(sigmoid_kernel.setArg(0, d_mlp_pre_activation)); CL_CHECK(sigmoid_kernel.setArg(1, current_out_hor)); 
                        CL_CHECK(sigmoid_kernel.setArg(2, output_size_mlp)); 
                        CL_CHECK(queue.enqueueNDRangeKernel(sigmoid_kernel, cl::NullRange, global_mlp, local_mlp)); 
                        std::swap(current_in_hor, current_out_hor); 
                    } 
                }
                { 
                    mat& current_weights_mat = head.ver.weights[layer_idx_mlp]; 
                    size_t weights_bytes_mlp = static_cast<size_t>(current_weights_mat.row) * current_weights_mat.col * sizeof(float); 
                    cl::Buffer d_mlp_weights(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, weights_bytes_mlp, current_weights_mat.mapped_data, &cl_err); CL_CHECK(cl_err); 
                    cl::Buffer& target_output_buffer = is_last_layer_mlp ? d_ver_output : d_mlp_pre_activation; 
                    CL_CHECK(mlp_fwd_kernel.setArg(0, current_in_ver)); CL_CHECK(mlp_fwd_kernel.setArg(1, d_mlp_weights)); 
                    CL_CHECK(mlp_fwd_kernel.setArg(2, target_output_buffer)); 
                    CL_CHECK(queue.enqueueNDRangeKernel(mlp_fwd_kernel, cl::NullRange, global_mlp, local_mlp)); 
                    if (!is_last_layer_mlp) { 
                        CL_CHECK(sigmoid_kernel.setArg(0, d_mlp_pre_activation)); CL_CHECK(sigmoid_kernel.setArg(1, current_out_ver)); 
                        CL_CHECK(sigmoid_kernel.setArg(2, output_size_mlp)); 
                        CL_CHECK(queue.enqueueNDRangeKernel(sigmoid_kernel, cl::NullRange, global_mlp, local_mlp)); 
                        std::swap(current_in_ver, current_out_ver); 
                    } 
                }
            }
            CL_CHECK(queue.finish());

            // 7. Apply ReLU (Copied from attention::clforprop)
            cl::Kernel relu_kernel = get_kernel_with_check(context_obj, "clReLU1d"); CL_CHECK(relu_kernel.setArg(2, d));
            CL_CHECK(relu_kernel.setArg(0, d_hor_output)); CL_CHECK(relu_kernel.setArg(1, d_relu_hor_output)); CL_CHECK(queue.enqueueNDRangeKernel(relu_kernel, cl::NullRange, global_add, local_add));
            CL_CHECK(relu_kernel.setArg(0, d_ver_output)); CL_CHECK(relu_kernel.setArg(1, d_relu_ver_output)); CL_CHECK(queue.enqueueNDRangeKernel(relu_kernel, cl::NullRange, global_add, local_add));
            CL_CHECK(queue.finish());

            // 8. Final Residual Update (Copied from attention::clforprop)
            CL_CHECK(add_kernel.setArg(0, d_EH)); CL_CHECK(add_kernel.setArg(1, d_relu_hor_output)); CL_CHECK(add_kernel.setArg(2, d_EH)); CL_CHECK(queue.enqueueNDRangeKernel(add_kernel, cl::NullRange, global_add, local_add));
            cl::Kernel update_ev_kernel = get_kernel_with_check(context_obj, "updateEVRowsKernelCL");
            size_t global_update_ev_raw = static_cast<size_t>(n); size_t global_update_ev_padded = ((global_update_ev_raw + local_work_size_1d - 1) / local_work_size_1d) * local_work_size_1d; cl::NDRange global_update_ev(global_update_ev_padded); cl::NDRange local_update_ev(local_work_size_1d);
            CL_CHECK(update_ev_kernel.setArg(0, d_EV_processed_data)); CL_CHECK(update_ev_kernel.setArg(1, d_relu_ver_output)); CL_CHECK(update_ev_kernel.setArg(2, n)); CL_CHECK(update_ev_kernel.setArg(3, d));
            CL_CHECK(queue.enqueueNDRangeKernel(update_ev_kernel, cl::NullRange, global_update_ev, local_update_ev));
            CL_CHECK(queue.finish());

            // --- Copy Results D->H (Copied from attention::clforprop) ---
            CL_CHECK(queue.enqueueReadBuffer(d_EH, CL_TRUE, 0, embed_bytes, head.EH.data()));
            CL_CHECK(queue.enqueueReadBuffer(d_EV_processed_data, CL_TRUE, 0, ev_processed_bytes, head.EV.mapped_data));

        }
        catch (const std::exception& e) {
            // Add context to the exception before re-throwing
            throw std::runtime_error("Exception in attention::clforprop (first block overload) for head ["
                                     + std::to_string(layer_idx) + "][" + std::to_string(i) + "]: " + e.what());
        }
    }
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
void block::cl1ParallelForprop(std::vector<std::vector<std::vector<float>>>& EVp, int& in, int& tokenCount, int& blockCount,
                               int i, int& layers, int& n)
{
    if (i < 0 || i >= this->y) {
        throw std::out_of_range("cl1ParallelForprop (subsequent block): column index 'i' (" + std::to_string(i) + ") is out of range [0, " + std::to_string(this->y - 1) + "].");
    }

    cl_int cl_err; // For OpenCL error codes
    OpenCLContext& context_obj = this->clcontext; // Use the member reference
    cl::Context context = context_obj.context;
    cl::CommandQueue queue = context_obj.queue;

    // Validate the incoming EVp for this column
    if (EVp.size() != static_cast<size_t>(this->x)) {
         throw std::runtime_error("cl1ParallelForprop (subsequent block): EVp layer dimension mismatch for column " + std::to_string(i)
                                  + ". Expected " + std::to_string(this->x) + " layers, got " + std::to_string(EVp.size()) + ".");
    }

    // Iterate through the layers (rows) of attention heads in the specified column 'i'
    for (int layer_idx = 0; layer_idx < this->x; ++layer_idx) {
        attention& head = this->b[layer_idx][i];
        // EVp[layer_idx] contains the [token][embedding] data for this specific head from the previous block
        std::vector<std::vector<float>>& EVp_layer = EVp[layer_idx];

        if (blockCount == 0) {
            // This case should ideally be routed to the other block::cl1parallelForprop or this function should not be called with blockCount=0
            // Assuming blockCount > 0 as per function brief.
            throw std::logic_error("cl1ParallelForprop (subsequent) called with blockCount <= 0");
        }

        const int d = EMBEDDING;        // Embedding dimension
        const int h = MATHEIGHTS;       // Height dimension of K/Q, projection matrices
        const int num_ev_rows_to_process = tokenCount; // This is totalTokenCount

        int start_idx_in_full_context = (blockCount - 1) * n; // n is contextWindowSize
        int end_idx_in_full_context = std::min<int>(tokenCount, blockCount * n); // tokenCount is totalTokenCount
        const int count = std::max<int>(0, end_idx_in_full_context - start_idx_in_full_context); // count for K, Q, KdotQ

        if (count <= 0) {
            // std::cerr << "Warning: cl1ParallelForprop (subsequent block) for head [" << layer_idx << "][" << i << "] with calculated count <= 0. Skipping." << std::endl;
            std::fill(head.EH.begin(), head.EH.end(), 0.0f);
            // EV might not need zeroing if count is 0, depends on overall logic.
            continue; // Skip to next head
        }

        // --- Basic Validation
        if (head.EV.row < num_ev_rows_to_process || head.EV.col != d) {
            throw std::runtime_error("EV matrix not properly sized for totalTokenCount in cl1ParallelForprop (subsequent block) for head [" + std::to_string(layer_idx) + "][" + std::to_string(i) + "]. Expected rows >= " + std::to_string(num_ev_rows_to_process) + " and cols = " + std::to_string(d) +
                                     ", but got rows = " + std::to_string(head.EV.row) + " and cols = " + std::to_string(head.EV.col));
        }
        if (head.K.row != CONTEXT_WIN || head.K.col != EMBEDDING ||
            head.Q.row != CONTEXT_WIN || head.Q.col != EMBEDDING ||
            head.KdotQ.row != CONTEXT_WIN || head.KdotQ.col != CONTEXT_WIN ||
            head.MH.row != EMBEDDING || head.MH.col != MATHEIGHTS ||
            head.MV.row != EMBEDDING || head.MV.col != MATHEIGHTS ||
            head.EH.size() != static_cast<size_t>(EMBEDDING) ||
            head.hor.hlayers.empty() || head.ver.hlayers.empty() || head.hor.weights.empty() || head.ver.weights.empty())
        {
            throw std::runtime_error("Attention component dimension mismatch or uninitialized member in cl1ParallelForprop (subsequent block) for head [" +
                                     std::to_string(layer_idx) + "][" + std::to_string(i) + "]. K.row=" + std::to_string(head.K.row) + ", count=" + std::to_string(count));
        }
        if (d != in) {
            throw std::runtime_error("Embedding dimension mismatch (EMBEDDING vs in) for head [" + std::to_string(layer_idx) + "][" + std::to_string(i) + "].");
        }
        if (EVp_layer.size() != static_cast<size_t>(num_ev_rows_to_process) || (!EVp_layer.empty() && EVp_layer[0].size() != static_cast<size_t>(d)) ) {
            throw std::runtime_error("EVp_layer dimension mismatch for head [" + std::to_string(layer_idx) + "][" + std::to_string(i) +
                                     "]. Expected rows " + std::to_string(num_ev_rows_to_process) + " (totalTokenCount), got " + std::to_string(EVp_layer.size()) +
                                     ". Expected cols " + std::to_string(d) + ", got " + (EVp_layer.empty() ? "N/A" : std::to_string(EVp_layer[0].size())) );
        }
        if (head.hor.hlayers[0].size() != static_cast<size_t>(d) || head.ver.hlayers[0].size() != static_cast<size_t>(d) || // Corrected ver.hlayers
            head.hor.weights.back().row != static_cast<size_t>(d) || head.ver.weights.back().row != static_cast<size_t>(d)) {
            throw std::runtime_error("MLP input/output layer dimension mismatch with embedding dimension 'd' for head [" + std::to_string(layer_idx) + "][" + std::to_string(i) + "].");
        }

        // --- Allocate Device Memory
        size_t k_bytes = static_cast<size_t>(count) * d * sizeof(float);
        size_t q_bytes = static_cast<size_t>(count) * d * sizeof(float);
        size_t kdotq_bytes = static_cast<size_t>(count) * count * sizeof(float);
        size_t head_bytes = static_cast<size_t>(count) * count * sizeof(float);
        size_t sums_bytes = static_cast<size_t>(count) * sizeof(float);
        size_t accum_bytes = static_cast<size_t>(h) * sizeof(float);
        size_t proj_mat_bytes = static_cast<size_t>(d) * h * sizeof(float); // MH/MV are d x h
        size_t ev_processed_bytes = static_cast<size_t>(num_ev_rows_to_process) * d * sizeof(float); // For EVp_layer
        size_t embed_bytes = static_cast<size_t>(d) * sizeof(float);

        try {
            cl::Buffer d_K(context, CL_MEM_READ_ONLY, k_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
            cl::Buffer d_Q(context, CL_MEM_READ_ONLY, q_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
            cl::Buffer d_KdotQ(context, CL_MEM_READ_ONLY, kdotq_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
            cl::Buffer d_head_attention(context, CL_MEM_READ_WRITE, head_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
            cl::Buffer d_row_sums(context, CL_MEM_READ_WRITE, sums_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
            cl::Buffer d_col_sums(context, CL_MEM_READ_WRITE, sums_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
            cl::Buffer d_dh_accum(context, CL_MEM_READ_WRITE, accum_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
            cl::Buffer d_dv_accum(context, CL_MEM_READ_WRITE, accum_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
            cl::Buffer d_MH_hxd(context, CL_MEM_READ_ONLY, proj_mat_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
            cl::Buffer d_MV_hxd(context, CL_MEM_READ_ONLY, proj_mat_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
            cl::Buffer d_dh(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
            cl::Buffer d_dv(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
            cl::Buffer d_EH(context, CL_MEM_READ_WRITE, embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
            cl::Buffer d_EV_processed_data_from_prev_block(context, CL_MEM_READ_WRITE, ev_processed_bytes, nullptr, &cl_err); CL_CHECK(cl_err); // Data from EVp_layer
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

            // Initialize accumulators to zero
            float zero = 0.0f;
            CL_CHECK(queue.enqueueFillBuffer(d_dh_accum, zero, 0, accum_bytes));
            CL_CHECK(queue.enqueueFillBuffer(d_dv_accum, zero, 0, accum_bytes));

            // --- Data Transfer H->D
            CL_CHECK(queue.enqueueWriteBuffer(d_K, CL_TRUE, 0, k_bytes, head.K.mapped_data)); // K, Q, KdotQ are for 'count' tokens
            CL_CHECK(queue.enqueueWriteBuffer(d_Q, CL_TRUE, 0, q_bytes, head.Q.mapped_data));
            CL_CHECK(queue.enqueueWriteBuffer(d_KdotQ, CL_TRUE, 0, kdotq_bytes, head.KdotQ.mapped_data));
            CL_CHECK(queue.enqueueWriteBuffer(d_MH_hxd, CL_TRUE, 0, proj_mat_bytes, head.MH.mapped_data));
            CL_CHECK(queue.enqueueWriteBuffer(d_MV_hxd, CL_TRUE, 0, proj_mat_bytes, head.MV.mapped_data));
            CL_CHECK(queue.enqueueWriteBuffer(d_EH, CL_TRUE, 0, embed_bytes, head.EH.data()));

            // Copy EVp_layer data to d_EV_processed_data_from_prev_block
            std::vector<float> flat_EVp_layer;
            flatten2DVector(EVp_layer, flat_EVp_layer, num_ev_rows_to_process, d);
            CL_CHECK(queue.enqueueWriteBuffer(d_EV_processed_data_from_prev_block, CL_TRUE, 0, ev_processed_bytes, flat_EVp_layer.data()));

            // --- Kernel Launches
            const size_t local_work_size_1d = 256;
            cl::Kernel lota_kernel = get_kernel_with_check(context_obj, "clLOTA2d");
            size_t totalElementsLOTA = static_cast<size_t>(count) * count;
            if (totalElementsLOTA > 0) {
                size_t global_lota_raw = totalElementsLOTA; //NOLINT
                size_t local_lota_clamped = (std::min)(global_lota_raw, local_work_size_1d);
                if (local_lota_clamped == 0) local_lota_clamped = 1;
                size_t global_lota_padded = ((global_lota_raw + local_lota_clamped - 1) / local_lota_clamped) * local_lota_clamped;
                cl::NDRange global_lota(global_lota_padded);
                cl::NDRange local_lota(local_lota_clamped);
                CL_CHECK(lota_kernel.setArg(0, d_KdotQ)); CL_CHECK(lota_kernel.setArg(1, d_head_attention)); CL_CHECK(lota_kernel.setArg(2, count)); CL_CHECK(lota_kernel.setArg(3, count));
                CL_CHECK(queue.enqueueNDRangeKernel(lota_kernel, cl::NullRange, global_lota, local_lota));
            } //NOLINT
            cl::Kernel sums_kernel = get_kernel_with_check(context_obj, "computeHeadSumsMaskedKernel");
            size_t global_sums_raw = static_cast<size_t>(count); size_t global_sums_padded = ((global_sums_raw + local_work_size_1d - 1) / local_work_size_1d) * local_work_size_1d; cl::NDRange global_sums(global_sums_padded); cl::NDRange local_sums(local_work_size_1d); cl_int cl_isSelfAttention = head.isSelfAttention;
            CL_CHECK(sums_kernel.setArg(0, d_head_attention)); CL_CHECK(sums_kernel.setArg(1, d_row_sums)); CL_CHECK(sums_kernel.setArg(2, d_col_sums)); CL_CHECK(sums_kernel.setArg(3, count)); CL_CHECK(sums_kernel.setArg(4, cl_isSelfAttention));
            CL_CHECK(queue.enqueueNDRangeKernel(sums_kernel, cl::NullRange, global_sums, local_sums));
            cl::Kernel accum_kernel = get_kernel_with_check(context_obj, "accumulateWeightedVectorsKernel");
            size_t global_accum_raw = static_cast<size_t>(count); size_t global_accum_padded = ((global_accum_raw + local_work_size_1d - 1) / local_work_size_1d) * local_work_size_1d; cl::NDRange global_accum(global_accum_padded); cl::NDRange local_accum(local_work_size_1d);
            CL_CHECK(accum_kernel.setArg(0, d_row_sums)); CL_CHECK(accum_kernel.setArg(1, d_col_sums)); CL_CHECK(accum_kernel.setArg(2, d_K)); CL_CHECK(accum_kernel.setArg(3, d_Q)); CL_CHECK(accum_kernel.setArg(4, d_dh_accum)); CL_CHECK(accum_kernel.setArg(5, d_dv_accum)); CL_CHECK(accum_kernel.setArg(6, count)); CL_CHECK(accum_kernel.setArg(7, h));
            CL_CHECK(queue.enqueueNDRangeKernel(accum_kernel, cl::NullRange, global_accum, local_accum));
            cl::Kernel proj_kernel = get_kernel_with_check(context_obj, "kernelLayerForward");
            size_t global_proj_raw = static_cast<size_t>(d); size_t global_proj_padded = ((global_proj_raw + local_work_size_1d - 1) / local_work_size_1d) * local_work_size_1d; cl::NDRange global_proj(global_proj_padded); cl::NDRange local_proj(local_work_size_1d);
            CL_CHECK(proj_kernel.setArg(3, h)); CL_CHECK(proj_kernel.setArg(4, d));
            CL_CHECK(proj_kernel.setArg(0, d_dh_accum)); CL_CHECK(proj_kernel.setArg(1, d_MH_hxd)); CL_CHECK(proj_kernel.setArg(2, d_dh)); CL_CHECK(queue.enqueueNDRangeKernel(proj_kernel, cl::NullRange, global_proj, local_proj));
            CL_CHECK(proj_kernel.setArg(0, d_dv_accum)); CL_CHECK(proj_kernel.setArg(1, d_MV_hxd)); CL_CHECK(proj_kernel.setArg(2, d_dv)); CL_CHECK(queue.enqueueNDRangeKernel(proj_kernel, cl::NullRange, global_proj, local_proj));
            CL_CHECK(queue.finish());
            cl::Kernel add_kernel = get_kernel_with_check(context_obj, "vectorAddKernel");
            size_t global_add_raw = static_cast<size_t>(d); size_t global_add_padded = ((global_add_raw + local_work_size_1d - 1) / local_work_size_1d) * local_work_size_1d; cl::NDRange global_add(global_add_padded); cl::NDRange local_add(local_work_size_1d); CL_CHECK(add_kernel.setArg(3, d));
            CL_CHECK(add_kernel.setArg(0, d_EH)); CL_CHECK(add_kernel.setArg(1, d_dh)); CL_CHECK(add_kernel.setArg(2, d_hor_inputs)); CL_CHECK(queue.enqueueNDRangeKernel(add_kernel, cl::NullRange, global_add, local_add));
            cl::Kernel accum_ev_kernel = get_kernel_with_check(context_obj, "accumulateEVRowsKernelCL");
            CL_CHECK(accum_ev_kernel.setArg(0, d_EV_processed_data_from_prev_block)); CL_CHECK(accum_ev_kernel.setArg(1, d_ver_accumulated_ev)); CL_CHECK(accum_ev_kernel.setArg(2, num_ev_rows_to_process)); CL_CHECK(accum_ev_kernel.setArg(3, d));
            CL_CHECK(queue.enqueueNDRangeKernel(accum_ev_kernel, cl::NullRange, global_add, local_add));
            CL_CHECK(add_kernel.setArg(0, d_ver_accumulated_ev)); CL_CHECK(add_kernel.setArg(1, d_dv)); CL_CHECK(add_kernel.setArg(2, d_ver_inputs)); CL_CHECK(queue.enqueueNDRangeKernel(add_kernel, cl::NullRange, global_add, local_add));
            CL_CHECK(queue.finish());

            // --- Run MLPs Forward
            cl::Kernel mlp_fwd_kernel = get_kernel_with_check(context_obj, "kernelLayerForward");
            cl::Kernel sigmoid_kernel = get_kernel_with_check(context_obj, "clSigmoid1d");
            CL_CHECK(queue.enqueueCopyBuffer(d_hor_inputs, d_mlp_bufferA_hor, 0, 0, embed_bytes));
            CL_CHECK(queue.enqueueCopyBuffer(d_ver_inputs, d_mlp_bufferA_ver, 0, 0, embed_bytes));
            cl::Buffer& current_in_hor = d_mlp_bufferA_hor; cl::Buffer& current_out_hor = d_mlp_bufferB_hor;
            cl::Buffer& current_in_ver = d_mlp_bufferA_ver; cl::Buffer& current_out_ver = d_mlp_bufferB_ver;
            size_t num_weight_matrices = head.hor.weights.size();
            size_t num_hlayers = head.hor.hlayers.size()-1;
            size_t global_mlp_raw = static_cast<size_t>(d); size_t global_mlp_padded = ((global_mlp_raw + local_work_size_1d - 1) / local_work_size_1d) * local_work_size_1d; cl::NDRange global_mlp(global_mlp_padded); cl::NDRange local_mlp(local_work_size_1d);
            for (size_t layer_idx_mlp = 0; layer_idx_mlp < num_weight_matrices; ++layer_idx_mlp) {
                bool is_last_layer_mlp = (layer_idx_mlp == num_weight_matrices - 1); 
                int input_size_mlp = d; int output_size_mlp = d;
                { 
                    mat& current_weights_mat = head.hor.weights[layer_idx_mlp]; 
                    size_t weights_bytes_mlp = static_cast<size_t>(current_weights_mat.row) * current_weights_mat.col * sizeof(float); 
                    cl::Buffer d_mlp_weights(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, weights_bytes_mlp, current_weights_mat.mapped_data, &cl_err); CL_CHECK(cl_err); 
                    cl::Buffer& target_output_buffer = is_last_layer_mlp ? d_hor_output : d_mlp_pre_activation; 
                    CL_CHECK(mlp_fwd_kernel.setArg(0, current_in_hor)); CL_CHECK(mlp_fwd_kernel.setArg(1, d_mlp_weights)); 
                    CL_CHECK(mlp_fwd_kernel.setArg(2, target_output_buffer)); CL_CHECK(mlp_fwd_kernel.setArg(3, input_size_mlp)); 
                    CL_CHECK(mlp_fwd_kernel.setArg(4, output_size_mlp)); 
                    CL_CHECK(queue.enqueueNDRangeKernel(mlp_fwd_kernel, cl::NullRange, global_mlp, local_mlp)); 
                    if (!is_last_layer_mlp) { 
                        CL_CHECK(sigmoid_kernel.setArg(0, d_mlp_pre_activation)); CL_CHECK(sigmoid_kernel.setArg(1, current_out_hor)); 
                        CL_CHECK(sigmoid_kernel.setArg(2, output_size_mlp)); 
                        CL_CHECK(queue.enqueueNDRangeKernel(sigmoid_kernel, cl::NullRange, global_mlp, local_mlp)); std::swap(current_in_hor, current_out_hor); 
                    } 
                }
                { 
                    mat& current_weights_mat = head.ver.weights[layer_idx_mlp]; 
                    size_t weights_bytes_mlp = static_cast<size_t>(current_weights_mat.row) * current_weights_mat.col * sizeof(float); 
                    cl::Buffer d_mlp_weights(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, weights_bytes_mlp, current_weights_mat.mapped_data, &cl_err); CL_CHECK(cl_err); 
                    cl::Buffer& target_output_buffer = is_last_layer_mlp ? d_ver_output : d_mlp_pre_activation; 
                    CL_CHECK(mlp_fwd_kernel.setArg(0, current_in_ver)); CL_CHECK(mlp_fwd_kernel.setArg(1, d_mlp_weights)); 
                    CL_CHECK(mlp_fwd_kernel.setArg(2, target_output_buffer)); 
                    CL_CHECK(queue.enqueueNDRangeKernel(mlp_fwd_kernel, cl::NullRange, global_mlp, local_mlp)); 
                    if (!is_last_layer_mlp) { 
                        CL_CHECK(sigmoid_kernel.setArg(0, d_mlp_pre_activation)); CL_CHECK(sigmoid_kernel.setArg(1, current_out_ver)); 
                        CL_CHECK(sigmoid_kernel.setArg(2, output_size_mlp)); 
                        CL_CHECK(queue.enqueueNDRangeKernel(sigmoid_kernel, cl::NullRange, global_mlp, local_mlp)); 
                        std::swap(current_in_ver, current_out_ver); 
                    } 
                }
            }
            CL_CHECK(queue.finish());

            // 7. Apply ReLU
            cl::Kernel relu_kernel = get_kernel_with_check(context_obj, "clReLU1d"); CL_CHECK(relu_kernel.setArg(2, d));
            CL_CHECK(relu_kernel.setArg(0, d_hor_output)); CL_CHECK(relu_kernel.setArg(1, d_relu_hor_output)); CL_CHECK(queue.enqueueNDRangeKernel(relu_kernel, cl::NullRange, global_add, local_add));
            CL_CHECK(relu_kernel.setArg(0, d_ver_output)); CL_CHECK(relu_kernel.setArg(1, d_relu_ver_output)); CL_CHECK(queue.enqueueNDRangeKernel(relu_kernel, cl::NullRange, global_add, local_add));
            CL_CHECK(queue.finish());

            // 8. Final Residual Update
            CL_CHECK(add_kernel.setArg(0, d_EH)); CL_CHECK(add_kernel.setArg(1, d_relu_hor_output)); CL_CHECK(add_kernel.setArg(2, d_EH)); CL_CHECK(queue.enqueueNDRangeKernel(add_kernel, cl::NullRange, global_add, local_add));
            cl::Kernel update_ev_kernel = get_kernel_with_check(context_obj, "updateEVRowsKernelCL");
            size_t global_update_ev_raw = static_cast<size_t>(num_ev_rows_to_process); size_t global_update_ev_padded = ((global_update_ev_raw + local_work_size_1d - 1) / local_work_size_1d) * local_work_size_1d; cl::NDRange global_update_ev(global_update_ev_padded); cl::NDRange local_update_ev(local_work_size_1d);
            // Update d_EV_processed_data_from_prev_block with ReLU(ver_output)
            CL_CHECK(update_ev_kernel.setArg(0, d_EV_processed_data_from_prev_block)); CL_CHECK(update_ev_kernel.setArg(1, d_relu_ver_output)); CL_CHECK(update_ev_kernel.setArg(2, num_ev_rows_to_process)); CL_CHECK(update_ev_kernel.setArg(3, d));
            CL_CHECK(queue.enqueueNDRangeKernel(update_ev_kernel, cl::NullRange, global_update_ev, local_update_ev));
            CL_CHECK(queue.finish());

            // --- Copy Results D->H
            CL_CHECK(queue.enqueueReadBuffer(d_EH, CL_TRUE, 0, embed_bytes, head.EH.data()));
            // Copy the updated EV (which originated from EVp_layer) back to head.EV
            if (static_cast<size_t>(head.EV.row) < static_cast<size_t>(num_ev_rows_to_process) || static_cast<size_t>(head.EV.col) != static_cast<size_t>(d)) {
                 throw std::runtime_error("head.EV dimensions are insufficient to store updated EV from previous block for head [" + std::to_string(layer_idx) + "][" + std::to_string(i) + "].");
            }
            CL_CHECK(queue.enqueueReadBuffer(d_EV_processed_data_from_prev_block, CL_TRUE, 0, ev_processed_bytes, head.EV.mapped_data));
        }
        catch (const std::exception& e) {
            // Add context to the exception before re-throwing
            throw std::runtime_error("Exception in attention::clforprop (subsequent block overload) for head ["
                                     + std::to_string(layer_idx) + "][" + std::to_string(i) + "]: " + e.what());
        }
    }
}


/**
 * @brief OpenCL forward propagation on the FIRST block.
 * @param context OpenCL context.
 * @param queue OpenCL command queue.
 * @param kernels Map of compiled OpenCL kernels.
 * @param in dimension size (maps to d_embedding in attention::clforprop)
 * @param tokenCount number of tokens (maps to currentTokenCount in attention::clforprop)
 * @param layers number of layers in mlp (maps to layers_mlp in attention::clforprop)
 */
void block::clForprop(int& in, int& tokenCount, int& layers)
{
    // deserialise(blockFilePath);
    // std::cout << "Forward Propagation for first block" << std::endl;
    // Iterate through all columns (parallels)
    for (int j = 0; j < this->y; ++j) {
        try {
            // Call the first block version for the column j
            this->cl1parallelForprop(in, tokenCount, j, layers);
        }
        catch (const std::exception& e) {
            // Add context to the exception before re-throwing
            throw std::runtime_error("Exception in cl1parallelForprop (first block) for column ["
                                     + std::to_string(j) + "]: " + e.what());
        }
    }
}


/**
 * @brief OpenCL forward propagation of a SUBSEQUENT block (blockCount > 0).
 * @param context OpenCL context.
 * @param queue OpenCL command queue.
 * @param kernels Map of compiled OpenCL kernels.
 * @param EVp vertical retention vectors from previous blocks (shape [x][y][token][embedding]).
 * @param in dimension size (maps to d_embedding in attention::clforprop)
 * @param tokenCount number of tokens in full context (maps to totalTokenCount in attention::clforprop)
 * @param blockCount position of block in full context (1-based, maps to blockIdx in attention::clforprop)
 * @param layers number of layers in mlp (maps to layers_mlp in attention::clforprop)
 * @param n context window size (maps to contextWindowSize in attention::clforprop)
 */
void block::clForprop(std::vector<std::vector<std::vector<std::vector<float>>>>& EVp, int& in, int& tokenCount, int& blockCount, 
    int& layers, int& n)
{
    // deserialise(blockFilePath);
    // Validate the overall structure of EVp
    if (EVp.size() != static_cast<size_t>(x)) {
        throw std::runtime_error("clForprop (subsequent block): EVp layer dimension mismatch. Expected "
                                 + std::to_string(x) + " layers, got " + std::to_string(EVp.size()) + ".");
    }
    if(!EVp.empty() && EVp[0].size() != static_cast<size_t>(y)) {
        throw std::runtime_error("clForprop (subsequent block): EVp column dimension mismatch. Expected "
                                 + std::to_string(this->y) + " columns, got " + std::to_string(EVp[0].size()) + ".");
    }
    
    // Iterate through all columns (parallels)
    for (int j = 0; j < this->y; ++j) {
        // Create the slice of EVp specific to this column j
        // EVp_col_j will have shape [x][token][embedding]
        std::vector<std::vector<std::vector<float>>> EVp_col_j(this->x);
        for (int layer_idx = 0; layer_idx < this->x; ++layer_idx) {
            EVp_col_j[layer_idx] = EVp[layer_idx][j];
        }

        try {
            // Call the subsequent block version for the column j, passing the column-specific EVp
            this->cl1ParallelForprop(EVp_col_j, in, tokenCount, blockCount, j, layers, n);
        }
        catch (const std::exception& e) {
            // Add context to the exception before re-throwing
            throw std::runtime_error("Exception in cl1ParallelForprop (subsequent block) for column ["
                                     + std::to_string(j) + "]: " + e.what());
        }
    }
}

#endif // USE_OPENCL
