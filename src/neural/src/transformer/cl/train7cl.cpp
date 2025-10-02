#ifdef USE_OPENCL
// all buffers in contextualised training function
#include <iomanip>
#include <maths.hpp>
#include "include/transformer.hpp"

// Helper struct to hold per-head sub-buffers for training
struct allBuffers {
    cl::Buffer d_K, d_Q, d_KdotQ, d_head;
    cl::Buffer d_row_sums, d_col_sums;
    cl::Buffer d_dh_accum, d_dv_accum;
    cl::Buffer d_MH_hxd, d_MV_hxd;
    cl::Buffer d_h, d_v, d_EH, d_EV;
    cl::Buffer d_EV_processed_data;
    cl::Buffer d_ver_accumulated_ev;
    cl::Buffer d_hor_inputs, d_ver_inputs;
    cl::Buffer d_hor_output, d_ver_output;
    cl::Buffer d_relu_hor_output, d_relu_ver_output;
    cl::Buffer d_hor_a, d_hor_b;
    cl::Buffer d_ver_a, d_ver_b;
    cl::Buffer d_mlp_pre_activation;

    cl::Buffer d_expected_h;
    cl::Buffer d_grad_EH, d_grad_EV_scaled;
    cl::Buffer d_grad_dh, d_grad_dv;
    cl::Buffer d_pre_MH, d_pre_MV;
    cl::Buffer d_MH, d_MV, d_MQ, d_MK;
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
 * @brief training of transformer by loading all the data block-wise on gpu buffers.
 *      This works on multiple gpu devices and cpus
 * @param sentence sentence embeddings
 * @param rString token strings
 */
void transformer::clBufferTrainContext(std::vector<std::vector<float>> &sentence, std::vector<std::string> &rString)
{
    // --- Basic validation ---
    if (sentence.size() > FULL_CONTEXT) {
        throw std::runtime_error("clTrain(sentence): Sentence size (" + std::to_string(sentence.size()) + ") exceeds FULL_CONTEXT (" + std::to_string(FULL_CONTEXT) + ").");
    }
    if (sentence.empty() || sentence.size() != rString.size()) {
        std::cout << "sentence.size(): " << sentence.size() << ", rString.size(): " << rString.size() << std::endl;
        throw std::runtime_error("clTrain(sentence): Sentence embeddings/strings mismatch or empty.");
    }
    if (!sentence.empty() && sentence[0].size() != static_cast<size_t>(d)) {
        throw std::runtime_error("clTrain(sentence): Sentence embedding dimension mismatch. Expected " + std::to_string(d) + ", got " + std::to_string(sentence[0].size()));
    }

    cl_int cl_err;          // error
    const int nHead = x;    // heads per column
    const int nPA = y;      // total columns
    const int embedding_dim = EMBEDDING;
    const int mat_heights = CONTEXT_WIN;
    const int context_win = CONTEXT_WIN;
    const int layers_mlp = LAYERS_MLP;
    const float learning_rate = learning;
    const float scaling_factor = 1.0f/(sqrt(static_cast<float>(embedding_dim)));
    int currentBlock = blockCount - 1;

    const int num_weight_matrices_mlp = layers_mlp - 1;

    const size_t embed_bytes = embedding_dim * sizeof(float);
    const size_t embedMatrix_bytes = vocabsize * embed_bytes * sizeof(float);
    const size_t context_bytes = static_cast<size_t>(context_win) * sizeof(float);
    const size_t mlp_weights_elements = static_cast<size_t>(embedding_dim) * embedding_dim;
    const size_t mlp_weights_bytes = mlp_weights_elements * sizeof(float);
    const size_t proj_mat_elements = static_cast<size_t>(mat_heights) * embedding_dim;
    const size_t proj_mat_bytes = proj_mat_elements * sizeof(float);
    const size_t ev_elements = static_cast<size_t>(context_win) * embedding_dim;
    const size_t ev_bytes = ev_elements * sizeof(float);
    const size_t kdotq_elements = static_cast<size_t>(context_win) * context_win;
    const size_t kdotq_bytes = kdotq_elements * sizeof(float);
    const size_t pre_mh_mv_elements_per_head = mat_heights;
    const size_t pre_mh_mv_bytes_per_head = pre_mh_mv_elements_per_head * sizeof(float);
    const size_t token_embed_bytes = static_cast<size_t>(m) * context_win * embedding_dim * sizeof(float);

    const size_t local_work_size_1d = 256;
    cl::NDRange local_1d(local_work_size_1d);
    const size_t local_work_size_2d_arr[2] = { 16, 16 };
    cl::NDRange local_2d(local_work_size_2d_arr[0], local_work_size_2d_arr[1]);

    OpenCLContext& context = clcontext;
    // single buffer for common use
    cl::Buffer d_embeddings;            // single buffer for embedding matrix
    cl::Buffer d_deEmbeddings;          // single buffer for de-embedding matrix
    cl::Buffer d_tokenEmbed;            // all token embeddings from chat
    cl::Buffer d_tokForBlock;           // token embedding for local context
    cl::Buffer d_otok_buffer;           // for accumulated EH from last column
    cl::Buffer predictionLogits;        // predicted scores
    cl::Buffer d_result_index_buffer;   // predicted index from maximum predicted score

    // define all agregated buffers
    cl::Buffer all_MH, all_MV, all_MQ, all_MK;
    cl::Buffer all_K, all_Q, all_KdotQ, all_head;
    cl::Buffer all_row_sums, all_col_sums;
    cl::Buffer all_h_accum, all_v_accum;
    cl::Buffer all_h, all_v, all_EH;
    cl::Buffer all_EV;
    cl::Buffer all_hin, all_vin;
    cl::Buffer all_hout, all_vout;
    cl::Buffer all_relu_hout, all_relu_vout;
    cl::Buffer all_hor_a, all_hor_b;
    cl::Buffer all_ver_a, all_ver_b;
    cl::Buffer all_mlp_pre_activation;

    cl::Buffer all_expected_h;
    cl::Buffer all_grad_EH, all_grad_EV;
    cl::Buffer all_grad_dh, all_grad_dv;
    cl::Buffer all_pre_MH, all_pre_MV;
    cl::Buffer all_grad_MH, all_grad_MV;
    cl::Buffer all_grad_head;
    cl::Buffer all_lota_deriv;
    cl::Buffer all_grad_KdotQ;
    cl::Buffer all_grad_K, all_grad_Q;
    cl::Buffer all_grad_MQ, all_grad_MK;

    cl::Buffer agg_d_MQt, agg_d_MKt;
    cl::Buffer agg_d_head_MKt, agg_d_head_MQt;
    cl::Buffer agg_d_head_MKt_t, agg_d_head_MQt_t;
    cl::Buffer agg_d_hMKt_t, agg_d_hMQt_t;
    cl::Buffer agg_d_grad_T_Q, agg_d_grad_T_K, agg_d_grad_T_h, agg_d_grad_T_v;
    cl::Buffer agg_d_grad_T;

    cl::Buffer all_hor_activations;
    cl::Buffer all_hor_weights, all_hor_gweights, all_hor_deltas;
    cl::Buffer all_ver_activations;
    cl::Buffer all_ver_weights, all_ver_gweights, all_ver_deltas;

    std::vector<cl::CommandQueue> streams_cl(nHead);
    std::vector<std::vector<allBuffers>> head_gpu_data(nPA, std::vector<allBuffers>(nHead));

    float current_error = 0.0f;
    float prev_error = 0.0f;
    int initial_epochs = epochs;
    int initial_token_count = currentTokenCount;    // Store initial count
    float initial_learning_rate = learning;         // Store initial learning rate
    bool blockIncremented = false;                  // check if block to be changed after local context reached

    try {
    // allocate memory to all buffers
        d_embeddings = cl::Buffer(clcontext.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, embed_bytes * vocabsize, embeddings.mapped_data, &cl_err); CL_CHECK(cl_err);
        d_tokForBlock = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, nHead * nPA * proj_mat_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        // forprop
        all_MH = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, nHead * nPA * proj_mat_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        all_MV = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, nHead * nPA * proj_mat_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        all_MQ = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, nHead * nPA * proj_mat_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        all_MK = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, nHead * nPA * proj_mat_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        all_K = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, nHead * nPA * kdotq_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        all_Q = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, nHead * nPA * kdotq_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        all_KdotQ = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, nHead * nPA * kdotq_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        all_head = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, nHead * nPA * kdotq_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        all_row_sums = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, nHead * nPA * context_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        all_col_sums = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, nHead * nPA * context_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        all_EH = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, nHead * nPA * embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        all_EV = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, nHead * nPA * ev_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        all_h_accum = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, nHead * nPA * context_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        all_v_accum = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, nHead * nPA * context_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        all_h = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, nHead * nPA * embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        all_v = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, nHead * nPA * embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        all_hin = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, nHead * nPA * embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        all_vin = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, nHead * nPA * embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        all_hout = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, nHead * nPA * embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        all_vout = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, nHead * nPA * embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        all_relu_hout = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, nHead * nPA * embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        all_relu_vout = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, nHead * nPA * embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        all_hor_a = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, nHead * nPA * embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        all_hor_b = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, nHead * nPA * embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        all_ver_a = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, nHead * nPA * embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        all_ver_b = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, nHead * nPA * embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        all_mlp_pre_activation = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, nHead * nPA * embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        all_hor_activations = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, nHead * nPA * layers_mlp * embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        all_ver_activations = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, nHead * nPA * layers_mlp * embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        all_hor_weights = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, nHead * nPA * num_weight_matrices_mlp * mlp_weights_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        all_ver_weights = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, nHead * nPA * num_weight_matrices_mlp * mlp_weights_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        // all backprop
        all_expected_h = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, nHead * nPA * embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        all_grad_EH = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, nHead * nPA * embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        all_grad_EV = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, nHead * nPA * embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        all_grad_dh = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, nHead * nPA * embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        all_grad_dv = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, nHead * nPA * embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        all_pre_MH = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, nHead * nPA * pre_mh_mv_bytes_per_head, nullptr, &cl_err); CL_CHECK(cl_err);
        all_pre_MV = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, nHead * nPA * pre_mh_mv_bytes_per_head, nullptr, &cl_err); CL_CHECK(cl_err);
        all_grad_MH = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, nHead * nPA * proj_mat_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        all_grad_MV = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, nHead * nPA * proj_mat_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        all_grad_head = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, nHead * nPA * kdotq_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        all_lota_deriv = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, nHead * nPA * kdotq_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        all_grad_KdotQ = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, nHead * nPA * kdotq_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        all_grad_K = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, nHead * nPA * kdotq_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        all_grad_Q = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, nHead * nPA * kdotq_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        all_grad_MQ = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, nHead * nPA * proj_mat_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        all_grad_MK = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, nHead * nPA * proj_mat_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        all_hor_gweights = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, nHead * nPA * num_weight_matrices_mlp * mlp_weights_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        all_ver_gweights = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, nHead * nPA * num_weight_matrices_mlp * mlp_weights_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        all_hor_deltas = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, nHead * nPA * layers_mlp * embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        all_ver_deltas = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, nHead * nPA * layers_mlp * embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_MQt = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, nHead * proj_mat_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_MKt = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, nHead * proj_mat_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_head_MKt = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, nHead * proj_mat_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_head_MQt = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, nHead * proj_mat_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_head_MKt_t = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, nHead * proj_mat_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_head_MQt_t = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, nHead * proj_mat_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_hMKt_t = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, nHead * proj_mat_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_hMQt_t = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, nHead * proj_mat_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_grad_T_Q = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, nHead * embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_grad_T_K = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, nHead * embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_grad_T_h = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, nHead * embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_grad_T_v = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, nHead * embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        agg_d_grad_T = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, nHead * embed_bytes, nullptr, &cl_err); CL_CHECK(cl_err);

        cl::Kernel sigmoid      = clcontext.kernels.at("clSigmoid1d");
        cl::Kernel relu         = clcontext.kernels.at("clReLU1d");
        cl::Kernel kqAll        = clcontext.kernels.at("kernelComputeKQall");
        cl::Kernel kdotq        = clcontext.kernels.at((isSelf == 0) ? "kernelKdotQforSelf_train" : "kernelKdotQforCross_train");
        cl::Kernel lota2d       = clcontext.kernels.at("clLOTA2dmasking");
        cl::Kernel rowSums      = clcontext.kernels.at("computeHeadSumsMaskedKernel");
        cl::Kernel vecAccum     = clcontext.kernels.at("accumulateWeightedVectorsKernel");
        cl::Kernel hv           = clcontext.kernels.at("kernelLayerForward");
        cl::Kernel addVec       = clcontext.kernels.at("vectorAddKernel");
        cl::Kernel evUpdate     = clcontext.kernels.at("updateEVRowsKernelCL");
        cl::Kernel predict      = clcontext.kernels.at("kernelComputePredictionWithScores");
        cl::Kernel lota1d       = clcontext.kernels.at("clLOTA1d");
        cl::Kernel predGrad     = clcontext.kernels.at("kernelComputeGradpred");
        cl::Kernel deEmbedGrad  = clcontext.kernels.at("KernelComputeGradDeEmbeddings");
        cl::Kernel updateWeight = clcontext.kernels.at("kernelUpdateWeightsGeneral_f4");
        cl::Kernel gradForAttEH = clcontext.kernels.at("kernelGradForAttentionOutput");
        cl::Kernel gradEHEV     = clcontext.kernels.at("kernelComputeGradientsEH_EV");
        cl::Kernel hidDelsig    = clcontext.kernels.at("kernelHiddenDeltaSigmoid");
        cl::Kernel updateMLP    = clcontext.kernels.at("kernelUpdateElasticNet");
        cl::Kernel mlpInGrad    = clcontext.kernels.at("kernelComputeGradMLPInput");
        cl::Kernel preMHMV      = clcontext.kernels.at("kernelComputePreMH_MV");
        cl::Kernel gradMHMV     = clcontext.kernels.at("kernelComputeGradMH_MV");
        cl::Kernel gradHead     = clcontext.kernels.at("kernelComputeGradHead");
        cl::Kernel gradKdotQ    = clcontext.kernels.at("kernelComputeGradKdotQ_LOTA");
        cl::Kernel kqGrad       = clcontext.kernels.at("kernelComputeGradK_Q");
        cl::Kernel gradMKMQ     = clcontext.kernels.at("kernelComputeGradMK_MQ");
        cl::Kernel updateHead   = clcontext.kernels.at("kernelUpdateWeightsHeadElastic");
        cl::Kernel transpose    = clcontext.kernels.at("kernelTransposeMatrix");
        cl::Kernel matmul       = clcontext.kernels.at("matrix_multiply_f4");
        cl::Kernel vecmatmul    = clcontext.kernels.at("vector_matrix_multiply_f4");
        cl::Kernel addAllVec    = clcontext.kernels.at("vectorsAddKernel");

    // training loop from token index 1 to last
        for(int i = 1; i < sentence.size(); i++) {
        // copy DATA to all buffers
            if(blockIncremented == 1 || blockCount == 1) {
                // only when its first block, or shifted to next block
                for(int i = 0; i < nPA; i++) {
                    for(int j = 0; j < nHead; j++) {
                        attention& head_obj = blocks[currentBlock].b[j][i];
                        streams_cl[j] = cl::CommandQueue(clcontext.context, clcontext.device, 0, &cl_err); CL_CHECK(cl_err);
                        head_gpu_data[i][j].d_hor_activations.resize(layers_mlp);
                        head_gpu_data[i][j].d_hor_weights.resize(num_weight_matrices_mlp);
                        head_gpu_data[i][j].d_hor_gweights.resize(num_weight_matrices_mlp);
                        head_gpu_data[i][j].d_hor_deltas.resize(layers_mlp);
                        head_gpu_data[i][j].d_ver_activations.resize(layers_mlp);
                        head_gpu_data[i][j].d_ver_weights.resize(num_weight_matrices_mlp);
                        head_gpu_data[i][j].d_ver_gweights.resize(num_weight_matrices_mlp);
                        head_gpu_data[i][j].d_ver_deltas.resize(layers_mlp);

                        cl_buffer_region region;
                        region = { i * j * embed_bytes, embed_bytes }; 
                        head_gpu_data[i][j].d_expected_h = all_expected_h.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
                        head_gpu_data[i][j].d_EH = all_EH.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
                        head_gpu_data[i][j].d_grad_EH = all_grad_EH.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
                        head_gpu_data[i][j].d_grad_EV_scaled = all_grad_EV.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
                        head_gpu_data[i][j].d_grad_dh = all_grad_dh.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
                        head_gpu_data[i][j].d_grad_dv = all_grad_dv.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
                        region = { i * j * ev_bytes, ev_bytes }; 
                        head_gpu_data[i][j].d_EV = all_EV.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
                        region = { i * j * kdotq_bytes, kdotq_bytes }; 
                        head_gpu_data[i][j].d_KdotQ = all_KdotQ.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
                        head_gpu_data[i][j].d_head = all_head.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
                        head_gpu_data[i][j].d_grad_head = all_grad_head.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
                        head_gpu_data[i][j].d_lota_deriv = all_lota_deriv.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
                        head_gpu_data[i][j].d_grad_KdotQ = all_grad_KdotQ.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
                        region = { i * j * kdotq_bytes, kdotq_bytes }; 
                        head_gpu_data[i][j].d_K = all_K.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
                        head_gpu_data[i][j].d_Q = all_Q.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
                        head_gpu_data[i][j].d_grad_K = all_grad_K.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
                        head_gpu_data[i][j].d_grad_Q = all_grad_Q.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
                        region = { i * j * pre_mh_mv_bytes_per_head, pre_mh_mv_bytes_per_head }; 
                        head_gpu_data[i][j].d_pre_MH = all_pre_MH.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
                        head_gpu_data[i][j].d_pre_MV = all_pre_MV.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
                        region = { i * j * proj_mat_bytes, proj_mat_bytes }; 
                        head_gpu_data[i][j].d_MH = all_MH.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
                        head_gpu_data[i][j].d_MV = all_MV.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
                        head_gpu_data[i][j].d_MQ = all_MQ.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
                        head_gpu_data[i][j].d_MK = all_MK.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
                        head_gpu_data[i][j].d_grad_MH = all_grad_MH.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
                        head_gpu_data[i][j].d_grad_MV = all_grad_MV.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
                        head_gpu_data[i][j].d_grad_MQ = all_grad_MQ.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
                        head_gpu_data[i][j].d_grad_MK = all_grad_MK.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
                        region = { 0, token_embed_bytes };
                        for (int l = 0; l < layers_mlp; ++l) { 
                            region = { (j * layers_mlp + l) * embed_bytes, embed_bytes }; 
                            head_gpu_data[i][j].d_hor_activations[l] = all_hor_activations.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
                            head_gpu_data[i][j].d_ver_activations[l] = all_ver_activations.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
                            head_gpu_data[i][j].d_hor_deltas[l] = all_hor_deltas.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
                            head_gpu_data[i][j].d_ver_deltas[l] = all_ver_deltas.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
                        }
                        for (int l = 0; l < num_weight_matrices_mlp; ++l) { 
                            region = { (j * num_weight_matrices_mlp + l) * mlp_weights_bytes, mlp_weights_bytes }; 
                            head_gpu_data[i][j].d_hor_weights[l] = all_hor_weights.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
                            head_gpu_data[i][j].d_ver_weights[l] = all_ver_weights.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
                            head_gpu_data[i][j].d_hor_gweights[l] = all_hor_gweights.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
                            head_gpu_data[i][j].d_ver_gweights[l] = all_ver_gweights.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
                        }
                    }
                }
            }

            // set expected values
            std::vector<float> expected_vec = sentence[i];
            std::string expected_str = rString[i];
            // tokens in local context
            unsigned int effectiveContext = (currentTokenCount + 1) % CONTEXT_WIN;

            int j = 0;          // epoch counter
            while (j < epochs) {
            // forward propagation: parallel operation of all column heads from first to last column
                // column-wise: nHead heads per column
                for(int i = 0; i < nPA; i++) {
                    // run parallels
                    for(int j = 0; j < nHead; j++) {
                        allBuffers& device_ptrs = head_gpu_data[j][i];
                        cl::CommandQueue& current_stream = streams_cl[j];

                        // query-key calculation

                        // scaled dot product, LoTA

                        // h and v and residual connection, EV calculation

                        // mlp hor and ver forprop

                        // residual connection and EV calculation
                    }
                }

                // accumulate the EH
                if (contextTrain == 0 && otok.size() != static_cast<size_t>(d)) {
                    throw std::runtime_error("clTrain(sentence): otok from clForward has incorrect size: " + std::to_string(otok.size()) + " != " + std::to_string(d) + ".");
                }
                if (y > 0) {
                    for (int j = 0; j < x; ++j) {
                        const std::vector<float>& eh_vector = blocks[blockCount-1].b[j][y - 1].EH;
                        if (eh_vector.size() != static_cast<size_t>(d)) {
                            throw std::runtime_error("clForward: EH vector size mismatch during host accumulation for head ["
                                                    + std::to_string(j) + "][" + std::to_string(y - 1) + "]. Expected "
                                                    + std::to_string(d) + ", got " + std::to_string(eh_vector.size()));
                        }
                        for (int k = 0; k < d; ++k) {
                            otok[k] += eh_vector[k];
                        }
                    }
                } 
                else {
                    std::cerr << "Warning: clForward called with y = 0 columns. Cannot accumulate EH." << std::endl;
                }
                // check for nan and inf
                for(size_t k_dim = 0; k_dim < static_cast<size_t>(d); k_dim++) {
                    if (std::isnan(otok[k_dim])) { otok[k_dim] = 0.00001f; }
                    else if (std::isinf(otok[k_dim])) { otok[k_dim] = std::copysign((std::numeric_limits<float>::max)(), otok[k_dim]); }
                }

            // token prediction and adaptive learning
                int result_idx = -1;
                size_t otok_bytes = otok.size() * sizeof(float);
                d_otok_buffer = cl::Buffer(clcontext.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, otok_bytes, otok.data(), &cl_err); CL_CHECK(cl_err);
                d_result_index_buffer = cl::Buffer(clcontext.context, CL_MEM_WRITE_ONLY, sizeof(cl_int), nullptr, &cl_err); CL_CHECK(cl_err);
                // use kernelComputePrediction for output prediction
                CL_CHECK(predict.setArg(0, d_otok_buffer));
                CL_CHECK(predict.setArg(1, d_embeddings));
                CL_CHECK(predict.setArg(2, d_result_index_buffer));
                CL_CHECK(predict.setArg(3, static_cast<cl_int>(d)));
                CL_CHECK(predict.setArg(4, static_cast<cl_int>(vocabsize)));
                cl::NDRange global(1);
                cl::NDRange local(1);
                CL_CHECK(clcontext.queue.enqueueNDRangeKernel(predict, cl::NullRange, global, local));
                CL_CHECK(clcontext.queue.enqueueReadBuffer(d_result_index_buffer, CL_TRUE, 0, sizeof(cl_int), &result_idx));
                indexForToken = result_idx; // Also update the class member

                // get error from otok and expected vector
                current_error = crossEntropy(oneHotEncode, pred);
                std::string predicted_token_str = (indexForToken >= 0 && indexForToken < static_cast<int>(tokens.size()))
                                                    ? tokens[indexForToken] : "INVALID_INDEX";
                std::cout << predicted_token_str << "\t: " << indexForToken << " | " 
                            << current_error << " | " 
                            << std::fixed << std::setprecision(7) << std::exp(current_error) << " | " 
                            << current_error - prev_error << " | " 
                            << j+1 << " | " << learning << std::endl;

                // --- Early exit if token is predicted correctly ---
                if (predicted_token_str == expected_str && predicted_token_str != "INVALID_INDEX") {
                    std::cout << "Token '" << expected_str << "' predicted correctly after " 
                                << j+1 << " epochs. Moving to next token." << std::endl;
                    totalLearning += learning;
                    if(predicted_token_str != "</s>")
                        std::cout << "              --------------- To Next Token -------------              " << std::endl;
                    break;
                }
                if(j == epochs - 1) {
                    std::cout << "Reached maximum epochs (" << epochs << ") for current token without correct prediction." << std::endl;
                    std::cout << "Increasing Epochs by 10." << std::endl;
                    epochs += 10;
                }

                // learning = softAdaptiveLearning(prev_error, current_error, learning, j);

            // backward propagation from last to first column
                for(int i = 0; i < nPA; i++) {
                    for(int j = 0; j < nHead; j++) {
                        attention& head_obj = blocks[currentBlock].b[j][i];
                        allBuffers& device_ptrs = head_gpu_data[i][j];
                        cl::CommandQueue& current_stream = streams_cl[j];
                    }
                }            
            }
            // --- Update Host State ---
            trainCount++;
            epochCount += j;

            if (tokenEmbed.mapped_data && static_cast<size_t>(currentTokenCount) < tokenEmbed.row && tokenEmbed.col == static_cast<size_t>(d)) {
                float* dest_ptr = tokenEmbed.mapped_data + (static_cast<size_t>(currentTokenCount) * d);
                if (expected_vec.size() == static_cast<size_t>(d)) {
                    memcpy(dest_ptr, expected_vec.data(), embed_bytes);
                }
                else {
                    std::cerr << "Error: expected_vec size mismatch for host tokenEmbed (mat) update in clTrain(sentence)." << std::endl;
                }
            }
            else {
                std::cerr << "Error: Host tokenEmbed (mat) not properly initialized or out of bounds for expected_vec update in clTrain(sentence)." << std::endl;
            }
            currentTokenCount++; // Increment after successful or attempted copy

            // Update blockCount for the *next* iteration
            if(currentTokenCount % CONTEXT_WIN == 0) {
                blocks[currentBlock].serialise(blocks[currentBlock].blockFilePath);
                blockIncremented = 1;
                blockCount++;
                currentBlock++;
            }
            else {
                blockIncremented = 0;
            }
        }
        // Reset for next line
        learning = initial_learning_rate;
    }
    catch (const std::runtime_error& e) {
        // Catches std::runtime_error from CL_CHECK
        std::cerr << "Standard Exception in clTrain(sentence): " << e.what() << std::endl;
        epochs = initial_epochs;
        throw;
    }
    // Buffers released by RAII
}

#endif