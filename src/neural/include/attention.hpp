// attention.hpp: header source for attention class
#ifndef ATTENTION_HPP
#define ATTENTION_HPP 1

/**
 * Attention Mechanism for SHADY ATTENTION ARCHITECTURE
 * ---------------------------------------------------------------------
 * K[i] = T[i] * MK, Q[i] = T[i] * MQ, M = MQ x MK'
 * KdotQ[i][j] = (T[i] x MK) x (T[i] x MQ)' = K[i] x Q'[j] 
 *             = T[i] x MQ x MK' x T'[j]  = T[i] x M x T'[j]
 * head = LOTA(KdotQ) OR LOTA(ReLU(KdotQ)) OR Softmax(KdotQ)
 * dh = sum(head[i][j] * Ki.MH), dv = sum(head[i][j] * Qi.MV)
 * Input(EH + dh) -> MLP(hor) -> ReLU(output) -> mH -> EH = EH + mH
 * Input(EV + dv) -> MLP(ver) -> ReLU(output) -> mV -> EV(i) = EV(i) + mV
 * MQ, MK, MV, MH => MATHEIGHTS x EMBEDDING
 * K, Q, EV => CONTEXT_WIN x EMBEDDING
 * KdotQ => CONTEXT_WIN x CONTEXT_WIN
 * qkCache, khCache, qvcache => EMBEDDING x EMBEDDING
 */

#include <vector>
#include <maths.hpp>
#include "mlp.hpp"

// macros for models
#define NUMBER_OF_PA 8                      // number of Partial Attentions in one Block
#define NUMBER_OF_HEADS 12                  // number of heads in each layer (partial attention)
#define NUMBER_OF_BLOCKS 4                  // number of blocks in transformer
#define EMBEDDING 128                       // embedding dimension for each token
#define LAYERS_MLP 4                        // layers of mlp
#define CONTEXT_WIN 1024                    // context window or number of tokens for each head (or number of PA * embedding)
#define PROMPT_THRESHOLD CONTEXT_WIN/4      // token limit for prompt
#define MATHEIGHTS 1024                     // weight matrix heights
#define FULL_CONTEXT CONTEXT_WIN*NUMBER_OF_BLOCKS               // maximum tokens for full context
#define SCALING std::sqrt(static_cast<float>(EMBEDDING))        // SCALING FACTOR for ATTENTION HEAD
#define DEEMBEDDING EMBEDDING*NUMBER_OF_PA  // embedding dimension for each token

/**
 * @brief ATTENTION CLASS for calculating attention head and Embeddings.
 * An array of attention head is Partial Attention (LAYER) and an array 
 * of partial attention (BLOCK) is complete attention. Attention head in 
 * complete attention working in parallel are referred as PARALLELs.
 */
class attention {
public:
    bool isSelfAttention;       // = 0 if cross attention else = 1 for self attention
    bool inTraining;            // = 1 for training, = 0 for inference
    int tokenCount;             // current token count for this head
    mat MQ;                     // query matrix
    mat MK;                     // key matrix
    mat MV;                     // vertical retention matrix
    mat MH;                     // horizontal retention matrix
    mlp ver;                    // vertical propagation and next block transfer
    mlp hor;                    // horizontal transfer to next head
    mat qkCache;                // QK' cache = MQ x MK' -> inference only
    mat qvCache;                // QH' cache = MQ x MV' -> inference only
    mat khCache;                // KV' cache = MK x MH' -> inference only
    mat K;                      // keys = Tokens x MK (Mapped)
    mat Q;                      // Querys = Tokens x MQ (Mapped)
    mat KdotQ;                  // attention head matrix -> Keys x Querys -> [K(i).Q(j)] <- scalar (Mapped)
    std::vector<float> EH;      // horizontal retention vector (Next Embedding in same block)
    mat EV;                     // vertical retention vectors (Context retention for next block)
    std::vector<float> dh;      // delta for EH: sum of (KdotQ[i][j] * Keys[i] * MH) (row wise)
    std::vector<float> dv;      // delta for EV[i]: sum of (KdotQ[j][i] * Keys[j] * MV) (column wise)
    float learning_rate;        // learning rate for attention
    // float lambda_L1;            // L1 regularization strength
    // float lambda_L2;            // L2 regularization strength
    unsigned long long params;          // parameters in each attention head
    unsigned long long attOffset;       // attention offset

#ifdef USE_OPENCL
    // Default constructor deleted when OpenCL is enabled because reference member clContext needs initialization.
    OpenCLContext& clcontext;
    attention(OpenCLContext& context, int n, int d, int h, int l, bool attentionType, bool inTraining, float& learning);
#elif USE_CUDA || USE_CPU
    // Constructors without OpenCLContext
    attention(int n, int d, int h, int l, bool attentionType, bool inTraining, float& learning);
#endif // USE_OPENCL

    // Explicitly define copy constructor and copy assignment operator
    attention(const attention& other);
    attention& operator=(const attention& other);

    void serialise(int offset, const std::string& locationWithFilename);
    void deserialise(int offset, const std::string& locationWithFilename);

#ifdef USE_CUDA

    float* d_EV; // Device pointer for Vertical Retention
    float* getDeviceEVPointer();

    void cuforprop(int& in, int& layers, int& tokenCount);
    void cuforprop(std::vector<std::vector<float>> EVp, int& in, int& layers, int& tokenCount, int& blockCount, int& n);
    void cuBackward1stHead(std::vector<float>& expected, int& in, int& layers, int headnumber, float& learning);
    void cuBackward1stHead(std::vector<std::vector<float>>& expectedV, int& in, int& layers, float& learning);
    void cuBackward(std::vector<float>& expected, int& in, int& layers, int headnumber, float& learning);
    void cuBackward(std::vector<std::vector<float>>& expectedV, int& layers, int blocknumber, float& learning);

#elif USE_OPENCL

    cl::Buffer d_EV; // Device buffer for Vertical Retention
    cl::Buffer& getDeviceEVBuffer() {
        if (!d_EV()) {
            throw std::runtime_error("attention::getDeviceEVBuffer(): d_EV is not a valid OpenCL buffer.");
        }
        return d_EV;
    } // Getter for the device buffer

    void clforprop(int& in, int& layers, int& tokenCount);
    void clforprop(std::vector<std::vector<float>> EVp, int& in, int& layers, int& tokenCount, int& blockCount, int& n);
    void clbackward1stHead(std::vector<float>& expected, int& in, int& layers, int headnumber, float& learning);
    void clbackward1stHead(std::vector<std::vector<float>>& expectedV, int& in, int& layers, float& learning);
    void clbackward(std::vector<float>& expected, int& in, int& layers, int& headnumber, float& learning);
    void clbackward(std::vector<std::vector<float>>& expectedV, int& layers, int& blocknumber, float& learning);

#else

    // cpp functions for cpu
    // forward propagation for both first and specific block's attention
    void forprop(int& in, int& layers, int& tokenCount);
    void forprop(const mat& EVp, int& in, int& layers, int& tokenCount, int& blockCount, int& n);
    // backward propagation
    void backward1stHead(std::vector<float>& expected, int& in, int& layers, int headnumber, float& learning);
    void backward1stHead(std::vector<std::vector<float>>& expectedV, int& in, int& layers, float& learning);
    void backward(std::vector<float>& expected, int& in, int& layers, int headnumber, float& learning);
    void backward(std::vector<std::vector<float>>& expectedV, int& layers, int blocknumber, float& learning);

#endif

    void setAttentionType(bool attentionType);
    void clearValues();
    void serialiseattention4train(const std::string& locationWithFileName);
    void serialiseattention4use(const std::string& binDirectory, const std::string& locationWithFileName);
    void computeCache();
    ~attention() = default;
};


// Inline implementations for copy constructor and copy assignment operator
inline attention::attention(const attention& other) :
#ifdef USE_OPENCL
    clcontext(other.clcontext),
    d_EV(other.d_EV),
#elif USE_CUDA
    d_EV(other.d_EV),
#endif
    // Initialize common members
    isSelfAttention(other.isSelfAttention),
    inTraining(other.inTraining),
    tokenCount(other.tokenCount),
    MQ(other.MQ),
    MK(other.MK),
    MV(other.MV),
    MH(other.MH),
    ver(other.ver), // mlp copy constructor
    hor(other.hor), // mlp copy constructor
    qkCache(other.qkCache),
    qvCache(other.qvCache),
    khCache(other.khCache),
    K(other.K),
    Q(other.Q),
    KdotQ(other.KdotQ),
    EH(other.EH),
    EV(other.EV),
    dh(other.dh),
    dv(other.dv),
    params(other.params)
{
}

inline attention& attention::operator=(const attention& other) {
    if (this == &other) {
        return *this; // Self-assignment check
    }

    #ifdef USE_OPENCL
        clcontext = other.clcontext;
        d_EV = other.d_EV;
    #elif USE_CUDA
        d_EV = other.d_EV;
    #endif

    // Assign common members
    isSelfAttention = other.isSelfAttention;
    inTraining = other.inTraining;
    tokenCount = other.tokenCount;
    MQ = other.MQ; // mat assignment
    MK = other.MK;
    MV = other.MV;
    MH = other.MH;
    ver = other.ver; // mlp assignment
    hor = other.hor;
    qkCache = other.qkCache;
    qvCache = other.qvCache;
    khCache = other.khCache;
    K = other.K;
    Q = other.Q;
    KdotQ = other.KdotQ;
    EH = other.EH; // std::vector assignment
    EV = other.EV;
    dh = other.dh;
    dv = other.dv;
    params = other.params;

    return *this;
}

#ifdef USE_CUDA

#include <cuda.h>
#include <cuda_runtime.h>

// dot product and multiplication
    __global__ void compute_single_kq_vector_kernel( const float* d_token_embedding, const float* d_projection_matrix, 
                float* d_output_kq_vector, int embedding_dim, int mat_heights);
    __device__ void cuComputeKorQ(const float* tokenEmbed, const float* matrix, float* KorQ, int dim, int height);
    __device__ int compute_prediction(const float* EH, const float* embeddings, int dim, int voc);
    __device__ float compute_dot_product(const float* vec1, const float* vec2, int dim);
    __device__ float compute_dot_product(const float* vec1, const float* vec2, const float* matrix, int dim);
    __global__ void computeAllDotsKernel(const float* vector, const float* matrix, float* results, int num_rows, int vector_dim);
    __global__ void kernelElementwiseMultiply(float* target_and_output, const float* factor, int size);
    // forward propagation
    __global__ void computeHeadSumsMaskedKernel(const float* d_head, float* d_row_sums, float* d_col_sums, 
        int num_tokens, bool isSelfAttention);
    __global__ void accumulateWeightedVectorsKernel(const float* d_row_sums, const float* d_col_sums,
        const float* d_K, const float* d_Q, float* d_dh_accum, float* d_dv_accum, int num_tokens, int h_dim);
    __global__ void accumulateEVRowsKernel(const float* d_EV, float* d_output, int num_rows, int col_size);
    __global__ void updateEVRowsKernel(float* d_EV_rows, const float* d_vector_to_add, int num_rows_to_update, int num_cols);
    // training with forward propagation
    __global__ void kernelKdotQforSelf_train(float* d_kdotq, const float* d_keys, const float* d_querys, int num_queries_eff, 
                int num_keys_eff, int kdotq_width, int embedding_dim, float inv_scaling);
    __global__ void kernelKdotQforCross_train(float* d_kdotq, const float* d_keys, const float* d_querys, int num_queries_eff,
                int num_keys_eff, int kdotq_width, int embedding_dim, float inv_scaling);
    // backprop
    __global__ void kernelComputeGradDhDv_1stHead(const float* d_hor_gweights0, const float* d_ver_gweights0,
        float* grad_dh, float* grad_dv, int embedding_dim);
    __global__ void kernelComputeGradientsEH_EV(const float* eh, const float* expected_h,
        float* grad_eh, float* grad_ev_scaled, int embedding_dim);
    __global__ void kernelComputeGradDhDv(const float* d_hor_gweights0, const float* d_ver_gweights0,
        float* grad_dh, float* grad_dv, int embedding_dim);
    __global__ void kernelComputePreMH_MV(const float* head, const float* k, const float* q,
        float* pre_mh, float* pre_mv, int token_count, int mat_heights);
    __global__ void kernelComputeGradMH_MV(const float* pre_mh, const float* pre_mv, const float* grad_dh, 
        const float* grad_dv, float* grad_mh, float* grad_mv, int mat_heights, int embedding_dim);
    __global__ void kernelComputeGradHead(const float* k, const float* q, const float* mh_a, const float* mv_a,
        const float* grad_dh, const float* grad_dv, float* grad_head, int token_count, int mat_heights, 
        int embedding_dim);
    __global__ void kernelComputeGradKdotQ_LOTA(const float* grad_head, const float* lota_derivative,
        float* grad_kdotq, float scaling_factor, int size);
    __global__ void kernelComputeGradK_Q(const float* grad_kdotq, const float* k, const float* q,
        float* grad_k, float* grad_q, int token_count, int mat_heights);
    __global__ void kernelComputeGradMK_MQ(const float* grad_k, const float* grad_q, const float* k, 
        const float* q, float* grad_mk, float* grad_mq, int token_count, int mat_heights, int embedding_dim);
    __global__ void kernelUpdateWeights_EH_EV(float* mh_a, float* mv_a, float* mq_a, float* mk_a, float* eh, 
        float* ev, const float* grad_mh, const float* grad_mv, const float* grad_mq, const float* grad_mk,
        const float* grad_eh, const float* grad_ev_scaled, float learning_rate, int mat_heights, int embedding_dim, 
        int context_win);
    __global__ void kernelComputeGradientsEV_V(const float* ev, const float* expected_v, float* grad_ev_full, 
        float* grad_ev_summed, float* grad_ev_scaled, float learning_rate, int context_win, int embedding_dim);
    __global__ void kernelComputeGradDv_V(const float* d_ver_gweights0, float* grad_dv, int embedding_dim);
    __global__ void kernelComputePreMV_V(const float* head, const float* q, float* pre_mv, int token_count, int mat_heights);
    __global__ void kernelComputeGradMV_V(const float* pre_mv, const float* grad_dv, float* grad_mv, int mat_heights, 
        int embedding_dim);
    __global__ void kernelComputeGradHead_V(const float* q, const float* mv_a, const float* grad_dv, float* grad_head,
        int token_count, int mat_heights, int embedding_dim);
    __global__ void kernelComputeGradQ_V(const float* grad_kdotq, const float* k, float* grad_q, int token_count, 
        int mat_heights);
    __global__ void kernelComputeGradMQ_V(const float* grad_q, const float* q, float* grad_mq, int token_count, int mat_heights, 
        int embedding_dim);
    __global__ void kernelComputeGradMKCorrection(const float* grad_mq, const float* q, const float* k, float* grad_mk_correction,
        int token_count, int mat_heights, int embedding_dim);
    __global__ void kernelUpdateWeights_EV_V(float* mv_a, float* mq_a, float* mk_a, float* ev, const float* grad_mv, const float* grad_mq,
        const float* grad_mk_correction, const float* grad_ev_full, float learning_rate, int mat_heights, int embedding_dim, 
        int context_win);
    __global__ void kernelComputeGradMK_MQ_Simplified(const float* grad_k, const float* grad_q, const float* k_embed, const float* q_embed,
        float* grad_mk, float* grad_mq, int token_count, int mat_heights, int embedding_dim);
    __global__ void kernelUpdateWeights_1stHead_H(float* mh_a, float* mv_a, float* mq_a, float* mk_a, float* eh, const float* grad_mh, 
        const float* grad_mv, const float* grad_mq, const float* grad_mk, const float* grad_eh, float learning_rate, bool update_eh,
        int mat_heights, int embedding_dim);    
    __global__ void kernelUpdateWeights_1stHead_V(float* mv_a, float* mq_a, float* mk_a, const float* grad_mv, const float* grad_mq,
        const float* grad_mk_correction, float learning_rate, int mat_heights, int embedding_dim);
    __global__ void kernelUpdateWeights_1stHead_HV(float* mh_a, float* mv_a, float* mq_a, float* mk_a, const float* grad_mh, 
        const float* grad_mv, const float* grad_mq, const float* grad_mk, float learning_rate, int mat_heights, int embedding_dim);
    // inference 
    __global__ void kernelKdotQ_Block1_Self_Inference(float* d_kdotq, const float* d_tokenEmbed, const float* d_M, 
                int prompt_start_index, int prompt_len, int context_len, int kdotq_width, int embedding_dim, float inv_scaling);            
    __global__ void kernelKdotQ_Block1_Cross_Inference(float* d_kdotq, const float* d_tokenEmbed, const float* d_M,
                int prompt_start_index, int prompt_len, int context_len, int kdotq_width, int embedding_dim, float inv_scaling);
    __global__ void kernelKdotQ_BlockN_Self_Inference(float* d_kdotq, const float* d_tokForBlock, const float* d_EVp, 
                const float* d_M, int prompt_start_index_in_block, int prompt_len, int context_len_in_block, int kdotq_width, 
                int embedding_dim, float inv_scaling);
    __global__ void kernelKdotQ_BlockN_Cross_Inference(float* d_kdotq, const float* d_tokForBlock, const float* d_EVp, 
                const float* d_M, int prompt_start_index_in_block, int prompt_len, int context_len_in_block, int kdotq_width, 
                int embedding_dim, float inv_scaling);
    __global__ void kernelComputeGradDhDv_1stHead(const float* d_hor_gweights0, const float* d_ver_gweights0,
                float* grad_dh, float* grad_dv, int embedding_dim);
    __global__ void kernelUpdateSimple(float* weights_to_update, const float* gradients, float lr, size_t n_elements);
    __global__ void kernelUpdateEVBroadcasted(float* d_EV, const float* d_grad_EV_scaled, float learning_rate, 
                int context_win, int embedding_dim);

#endif

#endif
