#ifdef USE_OPENCL
// all buffers in static training function
#include <iomanip>
#include <maths.hpp>
#include "include/transformer.hpp"

// Helper struct to hold per-head sub-buffers for training
struct allBuffers {
    // imp: context_win = matheights
    // imp: full_context = context_win x number_of_blocks
    cl::Buffer d_MH, d_MV, d_MQ, d_MK;
    cl::Buffer d_K, d_Q, d_KdotQ, d_head;       // context_win x context_win
    cl::Buffer d_row_sums, d_col_sums;          // context_win (sum(Kdot[i,j]) for i row or col)
    cl::Buffer d_dh_accum, d_dv_accum;          // d (sum(accum[i]*d_K(i)))
    cl::Buffer d_h, d_v, d_EH;                  // d
    cl::Buffer d_EV;                            // context_win x d
    cl::Buffer d_hin, d_vin;                    // d
    cl::Buffer d_hout, d_vout;                  // d
    cl::Buffer d_relu_hout, d_relu_vout;        // d
    cl::Buffer d_hor_a, d_hor_b;                // d
    cl::Buffer d_ver_a, d_ver_b;                // d
    cl::Buffer d_mlp_pre_activation;            // d

    cl::Buffer d_expected_h;                    // d
    cl::Buffer d_grad_EH, d_grad_EV;            // d
    cl::Buffer d_grad_dh, d_grad_dv;            // d
    cl::Buffer d_pre_MH, d_pre_MV;              // context_win
    cl::Buffer d_grad_MH, d_grad_MV;            // context_win x d
    cl::Buffer d_grad_head;                     // context_win x context_win
    cl::Buffer d_lota_deriv, d_grad_KdotQ;      // context_win x context_win
    cl::Buffer d_grad_K, d_grad_Q;              // context_win x context_win
    cl::Buffer d_grad_MQ, d_grad_MK;            // context_win x d

    std::vector<cl::Buffer> d_hor_activations;  // layers x d
    std::vector<cl::Buffer> d_ver_activations;  // layers x d
    std::vector<cl::Buffer> d_hor_weights, d_hor_gweights, d_hor_deltas;    // layers x (d x d)
    std::vector<cl::Buffer> d_ver_weights, d_ver_gweights, d_ver_deltas;    // layers x (d x d)
};


/**
 * @brief training of transformer by loading all the data block-wise on gpu buffers.
 *      This works on multiple gpu devices and cpus
 * @param sentence sentence embeddings
 * @param rString token strings
 */
void transformer::clBufferTrain(std::vector<std::vector<float>> &sentence, std::vector<std::string> &rString)
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
    const int layers_mlp = LAYERS_MLP;
    const int num_weight_matrices_mlp = layers_mlp - 1;
    const int nHead = x;    // heads per column
    const int nPA = y;      // total columns
    const int embedding_dim = EMBEDDING;
    const int mat_heights = CONTEXT_WIN;
    const int context_win = CONTEXT_WIN;
    const float learning_rate = learning;
    const float scaling_factor = 1.0f/(sqrt(static_cast<float>(embedding_dim)));
    int currentBlock = blockCount - 1;
    int effective_context_size = 0;

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
    const size_t totalTokenEmbedFloats = static_cast<size_t>(context_win) * d * m;
    const size_t token_embed_bytes = static_cast<size_t>(m) * context_win * embedding_dim * sizeof(float);

    const size_t local_work_size_1d = 256;
    cl::NDRange local_1d(local_work_size_1d);
    const size_t local_work_size_2d_arr[2] = { 16, 16 };
    cl::NDRange local_2d(local_work_size_2d_arr[0], local_work_size_2d_arr[1]);
    cl::NDRange global_work_size;

    OpenCLContext& context = clcontext;
    // define buffers
    cl::Buffer d_embeddings;            // single buffer for embeddings matrix
    cl::Buffer d_tokenEmbed;            // all token embeddings from chat
    cl::Buffer d_tokForBlock;           // token embedding for local context
    cl::Buffer d_otok_buffer;           // for accumulated EH from last column
    cl::Buffer d_result_index_buffer;   // predicted index
    // aggregated buffers
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

    // kernels

        cl::Kernel sigmoidk     = clcontext.kernels.at("clSigmoid1d");
        cl::Kernel relu         = clcontext.kernels.at("clReLU1d");
        cl::Kernel kqAll        = clcontext.kernels.at("kernelComputeKQall");
        cl::Kernel kdotq        = clcontext.kernels.at((isSelf == 0) ? "kernelKdotQforSelf_train" : "kernelKdotQforCross_train");
        cl::Kernel lota         = clcontext.kernels.at("clLOTA2dmasking");
        cl::Kernel weightSums   = clcontext.kernels.at("computeHeadSumsMaskedKernel");
        cl::Kernel vecAccum     = clcontext.kernels.at("accumulateWeightedVectorsKernel");
        cl::Kernel hv           = clcontext.kernels.at("kernelLayerForward");
        cl::Kernel addVec       = clcontext.kernels.at("vectorAddKernel");
        cl::Kernel accum_ev     = clcontext.kernels.at("accumulateEVRowsKernelCL");
        cl::Kernel evUpdate     = clcontext.kernels.at("updateEVRowsKernelCL");
        cl::Kernel predict      = clcontext.kernels.at("kernelComputePrediction");
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

    // add all tokens context-wise
        int start = 0;      // index of sentence to start training with
        // check for currentTokenCount : if 0, start from original, else, continue from where left
        std::vector<float> flat_host_tokenEmbed(totalTokenEmbedFloats, 0.0f);
        // start training from first
        if(currentTokenCount == 0) {
            // set tokenEmbed
            blockCount = 1;
            tokenEmbed.addRow(sentence[0], 0);
            effective_context_size = 1;
            currentTokenCount += 1;
            start = 1;
        }
        // continue training in first block
        else if(currentTokenCount > 0 && currentTokenCount < CONTEXT_WIN) {
            // set tokens from currentTokenCount index
            blockCount = 1;
            effective_context_size = currentTokenCount;
            for (int tk = 0; tk < currentTokenCount; ++tk) { // Copy existing context
                float* row_ptr = tokenEmbed.mapped_data + (static_cast<size_t>(tk) * d);
                flat_host_tokenEmbed.insert(flat_host_tokenEmbed.end(), row_ptr, row_ptr + d);
            }
            start = 0;
        }
        // non-first block training
        else {
            effective_context_size = currentTokenCount % CONTEXT_WIN;
            blockCount = (currentTokenCount / CONTEXT_WIN) + 1;
            for (int tk = 0; tk < currentTokenCount; ++tk) {
                // Copy existing context from block specific tokForBlock
                float* row_ptr = tokenEmbed.mapped_data + (static_cast<size_t>(tk) * d);
                flat_host_tokenEmbed.insert(flat_host_tokenEmbed.end(), row_ptr, row_ptr + d);
            }
            start = 0;
        }

    // training loop from token index 1 to last
        for(int i = 1; i < sentence.size(); i++) {
        // copy DATA to all buffers
            // add sentence[i] in d_
            if(blockIncremented == 1) {
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
                        head_gpu_data[i][j].d_grad_EV = all_grad_EV.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err); CL_CHECK(cl_err);
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
            unsigned int current_block_idx = blockCount;

            int j = 0;          // epoch counter
            while (j < epochs) {
            // forward propagation: parallel operation of all column heads from first to last column
                size_t currentBytes = static_cast<size_t>(EMBEDDING) * effectiveContext * sizeof(float);
                CL_CHECK(clcontext.queue.enqueueWriteBuffer(d_tokenEmbed, CL_TRUE, 0, currentBytes, tokenEmbed.mapped_data));

                for(int i = 0; i < nPA; i++) {
                    // run parallels
                    for(int j = 0; j < nHead; j++) {
                        attention& head_obj = blocks[currentBlock].b[j][i];
                        allBuffers& device_ptrs = head_gpu_data[j][i];
                        cl::CommandQueue& current_stream = streams_cl[j];
                        // query-key calculation
                        if(current_block_idx == 1) {
                            // keys and queries for each head of first block
                            int tokInContext = i;
                            for (int layer_idx = 0; layer_idx < x; ++layer_idx) {
                                for (int parallel_idx = 0; parallel_idx < y; ++parallel_idx) {
                                    auto& qMat = blocks[0].b[layer_idx][parallel_idx].MQ;
                                    auto& kMat = blocks[0].b[layer_idx][parallel_idx].MK;
                                    // queries <- tokenEmbed
                                    kdotq.setArg(0, d_tokenEmbed); kdotq.setArg(1, head_gpu_data[j][i].d_MQ); kdotq.setArg(2, head_gpu_data[j][i].d_Q);
                                    kdotq.setArg(3, effectiveContext); kdotq.setArg(4, EMBEDDING); kdotq.setArg(5, CONTEXT_WIN);
                                    CL_CHECK(clcontext.queue.enqueueNDRangeKernel(kdotq, cl::NullRange, cl::NDRange(2), cl::NullRange));
                                    CL_CHECK(clcontext.queue.enqueueReadBuffer(head_gpu_data[j][i].d_Q, CL_TRUE, 0, proj_mat_bytes,
                                             blocks[0].b[layer_idx][parallel_idx].Q.mapped_data));

                                    // keys <- tokenEmbed
                                    kdotq.setArg(0, d_tokenEmbed); kdotq.setArg(1, head_gpu_data[j][i].d_MK); kdotq.setArg(2, head_gpu_data[j][i].d_K);
                                    kdotq.setArg(3, effectiveContext); kdotq.setArg(4, EMBEDDING); kdotq.setArg(5, CONTEXT_WIN);
                                    CL_CHECK(clcontext.queue.enqueueNDRangeKernel(kdotq, cl::NullRange, cl::NDRange(2), cl::NullRange));
                                    CL_CHECK(clcontext.queue.enqueueReadBuffer(head_gpu_data[j][i].d_K, CL_TRUE, 0, proj_mat_bytes,
                                             blocks[0].b[layer_idx][parallel_idx].K.mapped_data));

                                    // kdotq
                                    CL_CHECK(kdotq.setArg(0, head_gpu_data[j][i].d_KdotQ));
                                    CL_CHECK(kdotq.setArg(1, head_gpu_data[j][i].d_K));
                                    CL_CHECK(kdotq.setArg(2, head_gpu_data[j][i].d_Q));
                                    CL_CHECK(kdotq.setArg(3, static_cast<cl_int>(effective_context_size)));
                                    CL_CHECK(kdotq.setArg(4, static_cast<cl_int>(effective_context_size)));
                                    CL_CHECK(kdotq.setArg(5, static_cast<cl_int>(context_win)));
                                    CL_CHECK(kdotq.setArg(6, static_cast<cl_int>(embedding_dim)));
                                    CL_CHECK(kdotq.setArg(7, scaling_factor));
                                    CL_CHECK(current_stream.enqueueNDRangeKernel(kdotq, cl::NullRange, cl::NDRange(effective_context_size, effective_context_size),
                                                                                                       cl::NDRange(16, 16)));
                                    // lota normalisation
                                    CL_CHECK(lota.setArg(0, head_gpu_data[j][i].d_KdotQ));
                                    CL_CHECK(lota.setArg(1, head_gpu_data[j][i].d_head));
                                    CL_CHECK(lota.setArg(2, CONTEXT_WIN));
                                    CL_CHECK(lota.setArg(3, CONTEXT_WIN));
                                    CL_CHECK(lota.setArg(4, effective_context_size));
                                    cl_int cl_att_is_self_lota = isSelf ? 1 : 0;
                                    CL_CHECK(lota.setArg(5, cl_att_is_self_lota));
                                    size_t total_elements = static_cast<size_t>(CONTEXT_WIN) * CONTEXT_WIN;
                                    size_t global_work_items = (total_elements + 3) / 4;
                                    size_t local_work_items = std::min<size_t>(256, global_work_items);
                                    size_t padded_global_size = ((global_work_items + local_work_items - 1) / local_work_items) * local_work_items;
                                    CL_CHECK(current_stream.enqueueNDRangeKernel(lota, cl::NullRange, cl::NDRange(padded_global_size), cl::NDRange(local_work_items)));
                                }
                            }
                            clcontext.queue.finish();
                        }
                        else {
                            int tokInContext = currentTokenCount % CONTEXT_WIN;
                            cl::Buffer pEV = cl::Buffer(clcontext.context, CL_MEM_READ_ONLY, ev_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
                            // start from last token of previous local context
                            size_t fromHereInTokenEmbed = static_cast<size_t>((CONTEXT_WIN) * (blockCount - 1) - 1) * sizeof(float);
                            const float* host_src_ptr = tokenEmbed.mapped_data + (fromHereInTokenEmbed / sizeof(float));
                            CL_CHECK(clcontext.queue.enqueueWriteBuffer(d_tokenEmbed, CL_TRUE, 0, currentBytes, host_src_ptr));

                            // keys and queries for each head of non-first block
                            for (int layer_idx = 0; layer_idx < x; ++layer_idx) {
                                for (int parallel_idx = 0; parallel_idx < y; ++parallel_idx) {
                                    auto& prevEV = blocks[blockCount - 2].b[layer_idx][parallel_idx].EV;
                                    CL_CHECK(clcontext.queue.enqueueWriteBuffer(pEV, CL_TRUE, 0, currentBytes, prevEV.mapped_data));

                                    // queries <- EVs of previous block
                                    kdotq.setArg(0, pEV); kdotq.setArg(1, head_gpu_data[j][i].d_MQ); kdotq.setArg(2, head_gpu_data[j][i].d_Q);
                                    kdotq.setArg(3, effectiveContext); kdotq.setArg(4, EMBEDDING); kdotq.setArg(5, CONTEXT_WIN);
                                    CL_CHECK(clcontext.queue.enqueueNDRangeKernel(kdotq, cl::NullRange, cl::NDRange(2), cl::NullRange));
                                    CL_CHECK(clcontext.queue.enqueueReadBuffer(head_gpu_data[j][i].d_Q, CL_TRUE, 0, proj_mat_bytes,
                                             blocks[blockCount - 1].b[layer_idx][parallel_idx].Q.mapped_data));

                                    // keys <- TokenEmbed from CONTEXT_WIN*(blockCount-1) - 1
                                    kdotq.setArg(0, d_tokenEmbed); kdotq.setArg(1, head_gpu_data[j][i].d_MK); kdotq.setArg(2, head_gpu_data[j][i].d_K);
                                    kdotq.setArg(3, effectiveContext); kdotq.setArg(4, EMBEDDING); kdotq.setArg(5, CONTEXT_WIN);
                                    CL_CHECK(clcontext.queue.enqueueNDRangeKernel(kdotq, cl::NullRange, cl::NDRange(2), cl::NullRange));
                                    CL_CHECK(clcontext.queue.enqueueReadBuffer(head_gpu_data[j][i].d_K, CL_TRUE, 0, proj_mat_bytes,
                                             blocks[blockCount - 1].b[layer_idx][parallel_idx].K.mapped_data));
                                    
                                    // kdotq
                                    CL_CHECK(kdotq.setArg(0, head_gpu_data[j][i].d_KdotQ));
                                    CL_CHECK(kdotq.setArg(1, head_gpu_data[j][i].d_K));
                                    CL_CHECK(kdotq.setArg(2, head_gpu_data[j][i].d_Q));
                                    CL_CHECK(kdotq.setArg(3, static_cast<cl_int>(effective_context_size)));
                                    CL_CHECK(kdotq.setArg(4, static_cast<cl_int>(effective_context_size)));
                                    CL_CHECK(kdotq.setArg(5, static_cast<cl_int>(context_win)));
                                    CL_CHECK(kdotq.setArg(6, static_cast<cl_int>(embedding_dim)));
                                    CL_CHECK(kdotq.setArg(7, scaling_factor));
                                    CL_CHECK(current_stream.enqueueNDRangeKernel(kdotq, cl::NullRange, cl::NDRange(effective_context_size, effective_context_size),
                                                                                                       cl::NDRange(16, 16)));
                                    
                                    // lota normalisation
                                    CL_CHECK(lota.setArg(0, head_gpu_data[j][i].d_KdotQ));
                                    CL_CHECK(lota.setArg(1, head_gpu_data[j][i].d_head));
                                    CL_CHECK(lota.setArg(2, CONTEXT_WIN));
                                    CL_CHECK(lota.setArg(3, CONTEXT_WIN));
                                    CL_CHECK(lota.setArg(4, effective_context_size));
                                    cl_int cl_att_is_self_lota = isSelf ? 1 : 0;
                                    CL_CHECK(lota.setArg(5, cl_att_is_self_lota));
                                    size_t total_elements = static_cast<size_t>(CONTEXT_WIN) * CONTEXT_WIN;
                                    size_t global_work_items = (total_elements + 3) / 4;
                                    size_t local_work_items = std::min<size_t>(256, global_work_items);
                                    size_t padded_global_size = ((global_work_items + local_work_items - 1) / local_work_items) * local_work_items;
                                    CL_CHECK(current_stream.enqueueNDRangeKernel(lota, cl::NullRange, cl::NDRange(padded_global_size), cl::NDRange(local_work_items)));
                                }
                            }
                            clcontext.queue.finish();
                        }
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
                    if (std::isnan(otok[k_dim])) { otok[k_dim] = 0.0f; }
                    else if (std::isinf(otok[k_dim])) { otok[k_dim] = std::copysign((std::numeric_limits<float>::max)(), otok[k_dim]); }
                }

            // token prediction and adaptive learning
                int result_idx = -1;
                size_t otok_bytes = otok.size() * sizeof(float);
                d_otok_buffer = cl::Buffer(clcontext.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, otok_bytes, otok.data(), &cl_err); CL_CHECK(cl_err);
                d_result_index_buffer = cl::Buffer(clcontext.context, CL_MEM_WRITE_ONLY, sizeof(cl_int), nullptr, &cl_err); CL_CHECK(cl_err);
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
                std::vector<float> expv = sigmoid(expected_vec);
                current_error = binaryCrossEntropy(expv, otok);
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
                        allBuffers& device_ptrs = head_gpu_data[j][i];
                        cl::CommandQueue& current_stream = streams_cl[j];

                    }
                }

                totalLearning += learning;
                prev_error = current_error;
                totalBCELoss += current_error;
                totalBCEPerplexity += std::exp(current_error);
                j++;
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