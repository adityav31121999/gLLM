#ifdef USE_OPENCL
#include "include/block.hpp" // Assumed to contain block, attention, mlp, mat, OpenCLContext
#include <maths.hpp>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <CL/cl.hpp> // For cl::Buffer, cl::CommandQueue, etc.

void block::clAdamUpdate(int layers_mlp, unsigned long long t_adam, float beta1, float beta2, float epsilon, float learning_rate)
{
    for(int i = 0; i < x; i++) {
        for(int j = 0; j < y; j++) {
            b[i][j].clAdamUpdate(clcontext, t_adam, beta1, beta2, epsilon, learning_rate);
        }
    }
}

/*
// struct parallelHeadAdam (now holds device sub-buffers)
struct parallelHeadAdam {
    cl::Buffer mQ, mK, mH, mV;          // attention weights
    cl::Buffer mQg, mKg, mHg, mVg;      // attention gradients
    cl::Buffer mQm, mKm, mHm, mVm;      // attention momentum
    cl::Buffer mQv, mKv, mHv, mVv;      // attention velocity

    std::vector<cl::Buffer> horW, verW; // MLP weights
    std::vector<cl::Buffer> horG, verG; // MLP gradients
    std::vector<cl::Buffer> horM, verM; // MLP momentum
    std::vector<cl::Buffer> horV, verV; // MLP velocity
};

// block::clParallelAdamUpdate implementation
void block::clParallelAdamUpdate(int layers_mlp, unsigned long long t_adam, int columnNumber, float beta1, float beta2, float epsilon, float learning_rate_param)
{
    cl_int cl_err;
    const int num_heads_to_process = x; // 'x' is the number of rows/heads in this column

    // Corrected columnNumber range check (0-indexed)
    if (columnNumber < 0 || columnNumber >= y) {
        throw std::out_of_range("clParallelAdamUpdate: Column index 'columnNumber' (" + std::to_string(columnNumber) +
                                ") is out of range [0, " + std::to_string(y - 1) + "].");
    }

    // Early exit if no heads to process in this column (num_heads_to_process is 'x')
    if (num_heads_to_process == 0) {
        std::cerr << "Warning: num_heads_to_process (x) is 0. No Adam update for this column." << std::endl;
        return; // No work to do
    }

    cl::Kernel adam_kernel;
    try {
        adam_kernel = clcontext.kernels.at("adam_optimizer_kernel");
    } catch (const std::out_of_range& e) {
        std::cerr << "Error: OpenCL kernel 'adam_optimizer_kernel' not found in context.kernels map: " << e.what() << std::endl;
        throw; // Re-throw the exception after logging
    }

    const int embedding_dim = EMBEDDING;
    const int mat_heights = MATHEIGHTS; // Assuming this is attention projection matrix height
    const float learning_rate = learning_rate_param; // Use the passed learning rate parameter
    const int num_total_layers_mlp = layers_mlp;
    const int num_weight_matrices_mlp = num_total_layers_mlp - 1; // Number of weight matrices in an MLP

    // IMPORTANT: Check for zero dimensions that would lead to zero-sized buffers
    if (embedding_dim <= 0) {
        throw std::runtime_error("EMBEDDING (embedding_dim) must be greater than 0 for OpenCL operations.");
    }
    // If mat_heights is 0, matValuesCount will be 0. We'll handle 0-sized matrix updates separately.
    if (num_total_layers_mlp < 1) { // A model should have at least 1 layer (input layer conceptually)
        throw std::runtime_error("MLP layers_mlp must be at least 1.");
    }

    // Sizes for individual matrices (in number of floats)
    const size_t matValuesCount = static_cast<size_t>(mat_heights) * embedding_dim; // For MQ, MK, MV, MH
    const size_t mlpLayerValuesCount = static_cast<size_t>(embedding_dim) * embedding_dim; // For single MLP weight matrix


    const size_t local_work_size_1d = 256;
    cl::NDRange local_1d(local_work_size_1d);
    
    OpenCLContext& clcontext = this->clcontext; // Use the clcontext member from block

    // Aggregate Buffers: attention matrices (weights, gradients, momentum, velocity)
    cl::Buffer aggMQ_W, aggMK_W, aggMH_W, aggMV_W;          // weights
    cl::Buffer aggMQ_G, aggMK_G, aggMH_G, aggMV_G;      // gradients
    cl::Buffer aggMQ_M, aggMK_M, aggMH_M, aggMV_M;      // momentum
    cl::Buffer aggMQ_V, aggMK_V, aggMH_V, aggMV_V;      // velocity

    // Aggregate Buffers: MLP matrices (weights, gradients, momentum, velocity)
    cl::Buffer aggHor_W, aggVer_W;                    // weights
    cl::Buffer aggHor_G, aggVer_G;                    // gradients
    cl::Buffer aggHor_M, aggVer_M;                    // momentum
    cl::Buffer aggHor_V, aggVer_V;                    // velocity

    std::vector<cl::CommandQueue> streams_cl(num_heads_to_process);
    std::vector<parallelHeadAdam> head_gpu_data_cl(num_heads_to_process); // Holds sub-buffers for each head

    // Lambda to calculate global work size
    auto calculate_global_1d = [&](size_t total_size) {
        if (total_size == 0) return (size_t)0; // Handle zero size gracefully
        return ((total_size + local_work_size_1d - 1) / local_work_size_1d) * local_work_size_1d;
    };

    // --- Core Adam Update Lambda ---
    // This lambda now operates on PRE-CREATED device buffers (sub-buffers) and a specific queue.
    // It does NOT create new cl::Buffer objects or perform host-to-device copies itself.
    auto apply_adam_kernel_on_buffers = [&](cl::Buffer& d_weights, cl::Buffer& d_gradients,
                                             cl::Buffer& d_moments, cl::Buffer& d_velocity,
                                             size_t num_elements, // Number of FLOAT elements
                                             cl::CommandQueue& queue) {

        if (num_elements == 0) return; // Skip empty matrices for kernel launch
        // Set kernel arguments
        CL_CHECK(adam_kernel.setArg(0, d_weights));
        CL_CHECK(adam_kernel.setArg(1, d_gradients));
        CL_CHECK(adam_kernel.setArg(2, d_moments));
        CL_CHECK(adam_kernel.setArg(3, d_velocity));
        CL_CHECK(adam_kernel.setArg(4, learning_rate));
        CL_CHECK(adam_kernel.setArg(5, beta1));
        CL_CHECK(adam_kernel.setArg(6, beta2));
        CL_CHECK(adam_kernel.setArg(7, epsilon));
        CL_CHECK(adam_kernel.setArg(8, t_adam));
        CL_CHECK(adam_kernel.setArg(9, static_cast<cl_int>(num_elements)));

        // Enqueue kernel on the provided queue
        size_t global_work_size = calculate_global_1d(num_elements);
        // Only enqueue if there's actual work to do
        if (global_work_size > 0) {
            CL_CHECK(queue.enqueueNDRangeKernel(adam_kernel, cl::NullRange, cl::NDRange(global_work_size), local_1d));
        }
    };


    try {
        // Calculate aggregate buffer sizes in BYTES
        const size_t agg_mat_bytes_total = num_heads_to_process * matValuesCount * sizeof(float);
        // Correct calculation for MLP aggregate buffer sizes
        const size_t agg_mlp_bytes_total = num_heads_to_process * num_weight_matrices_mlp * mlpLayerValuesCount * sizeof(float);

        // --- Allocate All Aggregate Device Buffers ---
        // Only allocate if size is non-zero to avoid CL_INVALID_BUFFER_SIZE on createSubBuffer if total size is 0
        // (even if OpenCL 1.1 spec says 0-sized buffers are allowed, sub-buffers might have stricter rules)

        // Weights
        if (agg_mat_bytes_total > 0) {
            aggMQ_W = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, agg_mat_bytes_total, nullptr, &cl_err); CL_CHECK(cl_err);
            aggMK_W = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, agg_mat_bytes_total, nullptr, &cl_err); CL_CHECK(cl_err);
            aggMH_W = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, agg_mat_bytes_total, nullptr, &cl_err); CL_CHECK(cl_err);
            aggMV_W = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, agg_mat_bytes_total, nullptr, &cl_err); CL_CHECK(cl_err);
        }
        if (agg_mlp_bytes_total > 0) {
            aggHor_W = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, agg_mlp_bytes_total, nullptr, &cl_err); CL_CHECK(cl_err);
            aggVer_W = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, agg_mlp_bytes_total, nullptr, &cl_err); CL_CHECK(cl_err);
        }
        // Gradients (and Momentum, Velocity - similar size checks)
        if (agg_mat_bytes_total > 0) {
            aggMQ_G = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, agg_mat_bytes_total, nullptr, &cl_err); CL_CHECK(cl_err);
            aggMK_G = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, agg_mat_bytes_total, nullptr, &cl_err); CL_CHECK(cl_err);
            aggMH_G = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, agg_mat_bytes_total, nullptr, &cl_err); CL_CHECK(cl_err);
            aggMV_G = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, agg_mat_bytes_total, nullptr, &cl_err); CL_CHECK(cl_err);
        }
        if (agg_mlp_bytes_total > 0) {
            aggHor_G = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, agg_mlp_bytes_total, nullptr, &cl_err); CL_CHECK(cl_err);
            aggVer_G = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, agg_mlp_bytes_total, nullptr, &cl_err); CL_CHECK(cl_err);
        }
        // Momentum
        if (agg_mat_bytes_total > 0) {
            aggMQ_M = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, agg_mat_bytes_total, nullptr, &cl_err); CL_CHECK(cl_err);
            aggMK_M = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, agg_mat_bytes_total, nullptr, &cl_err); CL_CHECK(cl_err);
            aggMH_M = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, agg_mat_bytes_total, nullptr, &cl_err); CL_CHECK(cl_err);
            aggMV_M = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, agg_mat_bytes_total, nullptr, &cl_err); CL_CHECK(cl_err);
        }
        if (agg_mlp_bytes_total > 0) {
            aggHor_M = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, agg_mlp_bytes_total, nullptr, &cl_err); CL_CHECK(cl_err);
            aggVer_M = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, agg_mlp_bytes_total, nullptr, &cl_err); CL_CHECK(cl_err);
        }
        // Velocity
        if (agg_mat_bytes_total > 0) {
            aggMQ_V = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, agg_mat_bytes_total, nullptr, &cl_err); CL_CHECK(cl_err);
            aggMK_V = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, agg_mat_bytes_total, nullptr, &cl_err); CL_CHECK(cl_err);
            aggMH_V = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, agg_mat_bytes_total, nullptr, &cl_err); CL_CHECK(cl_err);
            aggMV_V = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, agg_mat_bytes_total, nullptr, &cl_err); CL_CHECK(cl_err);
        }
        if (agg_mlp_bytes_total > 0) {
            aggHor_V = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, agg_mlp_bytes_total, nullptr, &cl_err); CL_CHECK(cl_err);
            aggVer_V = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, agg_mlp_bytes_total, nullptr, &cl_err); CL_CHECK(cl_err);
        }


        // --- Create Sub-buffers for Each Head & Enqueue Initial Host->Device Copies ---
        for (int head_idx = 0; head_idx < num_heads_to_process; ++head_idx) {
            streams_cl[head_idx] = cl::CommandQueue(clcontext.context, clcontext.device, 0, &cl_err); CL_CHECK(cl_err);

            // Resize MLP buffer vectors within parallelHeadAdam struct
            // These resizes are fine as they only allocate vector storage, not OpenCL buffers yet.
            head_gpu_data_cl[head_idx].horW.resize(num_weight_matrices_mlp);
            head_gpu_data_cl[head_idx].horG.resize(num_weight_matrices_mlp);
            head_gpu_data_cl[head_idx].horM.resize(num_weight_matrices_mlp);
            head_gpu_data_cl[head_idx].horV.resize(num_weight_matrices_mlp);
            head_gpu_data_cl[head_idx].verW.resize(num_weight_matrices_mlp);
            head_gpu_data_cl[head_idx].verG.resize(num_weight_matrices_mlp);
            head_gpu_data_cl[head_idx].verM.resize(num_weight_matrices_mlp);
            head_gpu_data_cl[head_idx].verV.resize(num_weight_matrices_mlp);

            cl_buffer_region region; // Use a temporary region struct

            // Access the specific attention head and its matrices on the host
            attention& head_obj = b[head_idx][columnNumber];

            // Attention Matrices - Create sub-buffers only if aggregate buffer is valid (non-zero size)
            size_t offset_mat_bytes = head_idx * matValuesCount * sizeof(float);
            size_t size_mat_bytes = matValuesCount * sizeof(float);

            if (agg_mat_bytes_total > 0 && size_mat_bytes > 0) { // Only attempt sub-buffer creation if meaningful
                region = {offset_mat_bytes, size_mat_bytes};
                // Weights
                head_gpu_data_cl[head_idx].mQ = aggMQ_W.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
                head_gpu_data_cl[head_idx].mK = aggMK_W.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
                head_gpu_data_cl[head_idx].mH = aggMH_W.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
                head_gpu_data_cl[head_idx].mV = aggMV_W.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
                // Gradients
                head_gpu_data_cl[head_idx].mQg = aggMQ_G.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
                head_gpu_data_cl[head_idx].mKg = aggMK_G.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
                head_gpu_data_cl[head_idx].mHg = aggMH_G.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
                head_gpu_data_cl[head_idx].mVg = aggMV_G.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
                // Momentum
                head_gpu_data_cl[head_idx].mQm = aggMQ_M.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
                head_gpu_data_cl[head_idx].mKm = aggMK_M.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
                head_gpu_data_cl[head_idx].mHm = aggMH_M.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
                head_gpu_data_cl[head_idx].mVm = aggMV_M.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
                // Velocity
                head_gpu_data_cl[head_idx].mQv = aggMQ_V.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
                head_gpu_data_cl[head_idx].mKv = aggMK_V.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
                head_gpu_data_cl[head_idx].mHv = aggMH_V.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
                head_gpu_data_cl[head_idx].mVv = aggMV_V.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);

                // Enqueue initial host->device copies for attention matrices for this head (using CL_FALSE for async)
                CL_CHECK(streams_cl[head_idx].enqueueWriteBuffer(head_gpu_data_cl[head_idx].mQ, CL_FALSE, 0, size_mat_bytes, head_obj.MQ.mapped_data));
                CL_CHECK(streams_cl[head_idx].enqueueWriteBuffer(head_gpu_data_cl[head_idx].mK, CL_FALSE, 0, size_mat_bytes, head_obj.MK.mapped_data));
                CL_CHECK(streams_cl[head_idx].enqueueWriteBuffer(head_gpu_data_cl[head_idx].mH, CL_FALSE, 0, size_mat_bytes, head_obj.MH.mapped_data));
                CL_CHECK(streams_cl[head_idx].enqueueWriteBuffer(head_gpu_data_cl[head_idx].mV, CL_FALSE, 0, size_mat_bytes, head_obj.MV.mapped_data));

                CL_CHECK(streams_cl[head_idx].enqueueWriteBuffer(head_gpu_data_cl[head_idx].mQg, CL_FALSE, 0, size_mat_bytes, head_obj.gMQ.mapped_data));
                CL_CHECK(streams_cl[head_idx].enqueueWriteBuffer(head_gpu_data_cl[head_idx].mKg, CL_FALSE, 0, size_mat_bytes, head_obj.gMK.mapped_data));
                CL_CHECK(streams_cl[head_idx].enqueueWriteBuffer(head_gpu_data_cl[head_idx].mHg, CL_FALSE, 0, size_mat_bytes, head_obj.gMH.mapped_data));
                CL_CHECK(streams_cl[head_idx].enqueueWriteBuffer(head_gpu_data_cl[head_idx].mVg, CL_FALSE, 0, size_mat_bytes, head_obj.gMV.mapped_data));

                CL_CHECK(streams_cl[head_idx].enqueueWriteBuffer(head_gpu_data_cl[head_idx].mQm, CL_FALSE, 0, size_mat_bytes, head_obj.m_MQ.mapped_data));
                CL_CHECK(streams_cl[head_idx].enqueueWriteBuffer(head_gpu_data_cl[head_idx].mKm, CL_FALSE, 0, size_mat_bytes, head_obj.m_MK.mapped_data));
                CL_CHECK(streams_cl[head_idx].enqueueWriteBuffer(head_gpu_data_cl[head_idx].mHm, CL_FALSE, 0, size_mat_bytes, head_obj.m_MH.mapped_data));
                CL_CHECK(streams_cl[head_idx].enqueueWriteBuffer(head_gpu_data_cl[head_idx].mVm, CL_FALSE, 0, size_mat_bytes, head_obj.m_MV.mapped_data));

                CL_CHECK(streams_cl[head_idx].enqueueWriteBuffer(head_gpu_data_cl[head_idx].mQv, CL_FALSE, 0, size_mat_bytes, head_obj.v_MQ.mapped_data));
                CL_CHECK(streams_cl[head_idx].enqueueWriteBuffer(head_gpu_data_cl[head_idx].mKv, CL_FALSE, 0, size_mat_bytes, head_obj.v_MK.mapped_data));
                CL_CHECK(streams_cl[head_idx].enqueueWriteBuffer(head_gpu_data_cl[head_idx].mHv, CL_FALSE, 0, size_mat_bytes, head_obj.v_MH.mapped_data));
                CL_CHECK(streams_cl[head_idx].enqueueWriteBuffer(head_gpu_data_cl[head_idx].mVv, CL_FALSE, 0, size_mat_bytes, head_obj.v_MV.mapped_data));
            }

            // MLP Weight Matrices - Create sub-buffers only if aggregate buffer is valid (non-zero size)
            size_t size_mlp_layer_bytes = mlpLayerValuesCount * sizeof(float);
            // This is the total size of MLP weights *for one head* in bytes
            size_t total_mlp_bytes_per_head = static_cast<size_t>(num_weight_matrices_mlp) * mlpLayerValuesCount * sizeof(float);
            // This is the offset for the beginning of this head's entire MLP data chunk within the aggregate buffer
            size_t offset_mlp_per_head_bytes = head_idx * total_mlp_bytes_per_head;
            
            // Only proceed with MLP sub-buffers if there are actual weight matrices and aggregate buffers are valid
            if (num_weight_matrices_mlp > 0 && agg_mlp_bytes_total > 0 && size_mlp_layer_bytes > 0) {
                for (int l = 0; l < num_weight_matrices_mlp; ++l) {
                    // Calculate offset for this specific layer's matrix within this head's MLP data region
                    region = {offset_mlp_per_head_bytes + static_cast<size_t>(l) * size_mlp_layer_bytes, size_mlp_layer_bytes};

                    // Weights
                    head_gpu_data_cl[head_idx].horW[l] = aggHor_W.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
                    head_gpu_data_cl[head_idx].verW[l] = aggVer_W.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
                    // Gradients
                    head_gpu_data_cl[head_idx].horG[l] = aggHor_G.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
                    head_gpu_data_cl[head_idx].verG[l] = aggVer_G.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
                    // Momentum
                    head_gpu_data_cl[head_idx].horM[l] = aggHor_M.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
                    head_gpu_data_cl[head_idx].verM[l] = aggVer_M.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
                    // Velocity
                    head_gpu_data_cl[head_idx].horV[l] = aggHor_V.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
                    head_gpu_data_cl[head_idx].verV[l] = aggVer_V.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);

                    // Enqueue initial host->device copies for MLP matrices for this head and layer
                    CL_CHECK(streams_cl[head_idx].enqueueWriteBuffer(head_gpu_data_cl[head_idx].horW[l], CL_FALSE, 0, size_mlp_layer_bytes, head_obj.hor.weights[l].mapped_data));
                    CL_CHECK(streams_cl[head_idx].enqueueWriteBuffer(head_gpu_data_cl[head_idx].horG[l], CL_FALSE, 0, size_mlp_layer_bytes, head_obj.hor.gweights[l].mapped_data));
                    CL_CHECK(streams_cl[head_idx].enqueueWriteBuffer(head_gpu_data_cl[head_idx].horM[l], CL_FALSE, 0, size_mlp_layer_bytes, head_obj.hor.moments[l].mapped_data));
                    CL_CHECK(streams_cl[head_idx].enqueueWriteBuffer(head_gpu_data_cl[head_idx].horV[l], CL_FALSE, 0, size_mlp_layer_bytes, head_obj.hor.velocity[l].mapped_data));

                    CL_CHECK(streams_cl[head_idx].enqueueWriteBuffer(head_gpu_data_cl[head_idx].verW[l], CL_FALSE, 0, size_mlp_layer_bytes, head_obj.ver.weights[l].mapped_data));
                    CL_CHECK(streams_cl[head_idx].enqueueWriteBuffer(head_gpu_data_cl[head_idx].verG[l], CL_FALSE, 0, size_mlp_layer_bytes, head_obj.ver.gweights[l].mapped_data));
                    CL_CHECK(streams_cl[head_idx].enqueueWriteBuffer(head_gpu_data_cl[head_idx].verM[l], CL_FALSE, 0, size_mlp_layer_bytes, head_obj.ver.moments[l].mapped_data));
                    CL_CHECK(streams_cl[head_idx].enqueueWriteBuffer(head_gpu_data_cl[head_idx].verV[l], CL_FALSE, 0, size_mlp_layer_bytes, head_obj.ver.velocity[l].mapped_data));
                }
            } // End if num_weight_matrices_mlp > 0
        } // End of per-head sub-buffer creation and initial data transfer

        // --- Enqueue Adam Kernel Launches for Each Head ---
        for (int head_idx = 0; head_idx < num_heads_to_process; ++head_idx) {
            parallelHeadAdam& device_ptrs = head_gpu_data_cl[head_idx];
            cl::CommandQueue& current_queue = streams_cl[head_idx];

            // Attention Matrix Updates - Only apply if matrices are present (non-zero size)
            if (matValuesCount > 0) {
                apply_adam_kernel_on_buffers(device_ptrs.mQ, device_ptrs.mQg, device_ptrs.mQm, device_ptrs.mQv,
                                            matValuesCount, current_queue);
                apply_adam_kernel_on_buffers(device_ptrs.mK, device_ptrs.mKg, device_ptrs.mKm, device_ptrs.mKv,
                                            matValuesCount, current_queue);
                apply_adam_kernel_on_buffers(device_ptrs.mH, device_ptrs.mHg, device_ptrs.mHm, device_ptrs.mHv,
                                            matValuesCount, current_queue);
                apply_adam_kernel_on_buffers(device_ptrs.mV, device_ptrs.mVg, device_ptrs.mVm, device_ptrs.mVv,
                                            matValuesCount, current_queue);
            }

            // MLP Weight Matrix Updates - Only apply if matrices are present
            if (num_weight_matrices_mlp > 0 && mlpLayerValuesCount > 0) {
                for(int i = 0; i < num_weight_matrices_mlp; i++) {
                    apply_adam_kernel_on_buffers(device_ptrs.horW[i], device_ptrs.horG[i], device_ptrs.horM[i], device_ptrs.horV[i],
                                                mlpLayerValuesCount, current_queue);
                    apply_adam_kernel_on_buffers(device_ptrs.verW[i], device_ptrs.verG[i], device_ptrs.verM[i], device_ptrs.verV[i],
                                                mlpLayerValuesCount, current_queue);
                }
            }
        }

        // --- Enqueue Final Device->Host Copies for Each Head ---
        for (int head_idx = 0; head_idx < num_heads_to_process; ++head_idx) {
            attention& head_obj = b[head_idx][columnNumber]; // Get host object reference
            parallelHeadAdam& device_ptrs = head_gpu_data_cl[head_idx];
            cl::CommandQueue& current_queue = streams_cl[head_idx];

            size_t size_mat_bytes = matValuesCount * sizeof(float);
            size_t size_mlp_layer_bytes = mlpLayerValuesCount * sizeof(float);

            // Attention Matrices - Only read back if they exist
            if (matValuesCount > 0) {
                CL_CHECK(current_queue.enqueueReadBuffer(device_ptrs.mQ, CL_FALSE, 0, size_mat_bytes, head_obj.MQ.mapped_data));
                CL_CHECK(current_queue.enqueueReadBuffer(device_ptrs.mK, CL_FALSE, 0, size_mat_bytes, head_obj.MK.mapped_data));
                CL_CHECK(current_queue.enqueueReadBuffer(device_ptrs.mH, CL_FALSE, 0, size_mat_bytes, head_obj.MH.mapped_data));
                CL_CHECK(current_queue.enqueueReadBuffer(device_ptrs.mV, CL_FALSE, 0, size_mat_bytes, head_obj.MV.mapped_data));

                CL_CHECK(current_queue.enqueueReadBuffer(device_ptrs.mQm, CL_FALSE, 0, size_mat_bytes, head_obj.m_MQ.mapped_data));
                CL_CHECK(current_queue.enqueueReadBuffer(device_ptrs.mKm, CL_FALSE, 0, size_mat_bytes, head_obj.m_MK.mapped_data));
                CL_CHECK(current_queue.enqueueReadBuffer(device_ptrs.mHm, CL_FALSE, 0, size_mat_bytes, head_obj.m_MH.mapped_data));
                CL_CHECK(current_queue.enqueueReadBuffer(device_ptrs.mVm, CL_FALSE, 0, size_mat_bytes, head_obj.m_MV.mapped_data));

                CL_CHECK(current_queue.enqueueReadBuffer(device_ptrs.mQv, CL_FALSE, 0, size_mat_bytes, head_obj.v_MQ.mapped_data));
                CL_CHECK(current_queue.enqueueReadBuffer(device_ptrs.mKv, CL_FALSE, 0, size_mat_bytes, head_obj.v_MK.mapped_data));
                CL_CHECK(current_queue.enqueueReadBuffer(device_ptrs.mHv, CL_FALSE, 0, size_mat_bytes, head_obj.v_MH.mapped_data));
                CL_CHECK(current_queue.enqueueReadBuffer(device_ptrs.mVv, CL_FALSE, 0, size_mat_bytes, head_obj.v_MV.mapped_data));
            }

            // MLP Matrices - Only read back if they exist
            if (num_weight_matrices_mlp > 0 && mlpLayerValuesCount > 0) {
                for(int l = 0; l < num_weight_matrices_mlp; ++l) {
                    CL_CHECK(current_queue.enqueueReadBuffer(device_ptrs.horW[l], CL_FALSE, 0, size_mlp_layer_bytes, head_obj.hor.weights[l].mapped_data));
                    CL_CHECK(current_queue.enqueueReadBuffer(device_ptrs.horM[l], CL_FALSE, 0, size_mlp_layer_bytes, head_obj.hor.moments[l].mapped_data));
                    CL_CHECK(current_queue.enqueueReadBuffer(device_ptrs.horV[l], CL_FALSE, 0, size_mlp_layer_bytes, head_obj.hor.velocity[l].mapped_data));

                    CL_CHECK(current_queue.enqueueReadBuffer(device_ptrs.verW[l], CL_FALSE, 0, size_mlp_layer_bytes, head_obj.ver.weights[l].mapped_data));
                    CL_CHECK(current_queue.enqueueReadBuffer(device_ptrs.verM[l], CL_FALSE, 0, size_mlp_layer_bytes, head_obj.ver.moments[l].mapped_data));
                    CL_CHECK(current_queue.enqueueReadBuffer(device_ptrs.verV[l], CL_FALSE, 0, size_mlp_layer_bytes, head_obj.ver.velocity[l].mapped_data));
                }
            }
        }

        // --- Final Synchronization and Cleanup ---
        // Finish all the queue operations to ensure all data is read back to host
        for (int head_idx = 0; head_idx < num_heads_to_process; ++head_idx) {
            CL_CHECK(streams_cl[head_idx].finish());
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Standard Exception during clParallelAdamUpdate for column " << columnNumber << ": " << e.what() << std::endl;
        // In case of an exception, ensure streams are finished to prevent resource leaks
        for (int head_idx = 0; head_idx < num_heads_to_process; ++head_idx) {
            if (streams_cl[head_idx]()) { // Check if the queue object holds a valid cl_command_queue handle
                cl_int finish_err = streams_cl[head_idx].finish();
                if (finish_err != CL_SUCCESS) {
                    std::cerr << "Warning: Error during stream finish in exception handler: " << oclErrorString(finish_err) << std::endl;
                }
            }
        }
        throw; // Re-throw the exception
    }
}

// single block adam optimiser (this function still needs to be adjusted based on the new structure)
void block::clAdamUpdate(int layers_mlp, unsigned long long t_adam, float beta1, float beta2, float epsilon, float learning_rate)
{
    // This function acts as an orchestrator for all columns in a block.
    // Each column's update is now a 'parallel' operation (due to per-head streams).
    // The `t_adam` should be incremented at the transformer level, and passed down.
    // This function itself does not increment t_adam.

    for (int col_idx = 0; col_idx < y; ++col_idx) {
        clParallelAdamUpdate(
            layers_mlp, // layers_mlp
            t_adam,     // global time step
            col_idx,    // columnNumber (0-indexed)
            beta1,
            beta2,
            epsilon,
            learning_rate // Use the dynamically adjusted learning rate
        );
    }
    // Removed: t_adam += 1; // This should happen at the transformer level (or wherever the global_adam_t lives)
}

*/
#endif