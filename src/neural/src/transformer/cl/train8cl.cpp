#ifdef USE_OPENCL
// all buffers in contextualised training function
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
    cl::Buffer d_mlp_bufferA_hor, d_mlp_bufferB_hor;
    cl::Buffer d_mlp_bufferA_ver, d_mlp_bufferB_ver;
    cl::Buffer d_mlp_pre_activation;
    cl::Buffer d_tokenEmbed;
    cl::Buffer d_embedding;
    cl::Buffer d_deEmbedding;

    cl::Buffer d_expected_h;
    cl::Buffer d_grad_EH, d_grad_EV_scaled;
    cl::Buffer d_grad_dh, d_grad_dv;
    cl::Buffer d_pre_MH, d_pre_MV;
    cl::Buffer d_MH_a, d_MV_a, d_MQ_a, d_MK_a;
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
 * @brief contextualised training on all gpu-buffers for sequence-2-sequence
 */
void transformer::clBufferTrainContext(std::vector<std::vector<float>> &seq1, std::vector<std::vector<float>> &seq2, std::vector<std::string> &seq2String)
{
    // --- Basic validation ---

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

    cl::Buffer all_MQt, all_MKt;
    cl::Buffer all_head_MKt, all_head_MQt;
    cl::Buffer all_head_MKt_t, all_head_MQt_t;
    cl::Buffer all_hMKt_t, all_hMQt_t;
    cl::Buffer all_grad_T_Q, all_grad_T_K, all_grad_T_h, all_grad_T_v;
    cl::Buffer all_grad_T;

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
        // 
    }
    catch (const std::exception& e) {
        std::cerr << "Standard Exception in clTrain(sentence): " << e.what() << std::endl;
    }
}

#endif