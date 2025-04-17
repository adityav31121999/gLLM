
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
 * Input(EV + dv) -> MLP(ver) -> ReLU(output) -> mV -> EV = EV + mV
 */

#include <vector>
#include <maths.hpp>
#include "mlp.hpp"

// macros for models
#define TERMINATE "@#O"                     // end of conversation (And Its Over)
#define LEARNING 0.01                       // learning rate for MLPs
#define MATHEIGHTS 4096                     // weight matrix heights
#define CONTEXT_WIN 4096                    // context window or number of tokens for each head
#define PROMPT_THRESHOLD CONTEXT_WIN/4      // token limit for prompt
#define EMBEDDING 64                        // embedding dimension for each token
#define SCALING std::sqrt(EMBEDDING)        // SCALING FACTOR for ATTENTION HEAD
#define LAYERS_MLP 16                       // layers of mlp
#define EPOCHS 10                           // number of epochs for MLPs
#define NUMBER_OF_PA 8                      // number of Partial Attentions in one Block
#define NUMBER_OF_HEADS 32                  // number of heads in each layer (partial attention)
#define NUMBER_OF_BLOCKS 8                  // number of blocks in transformer
#define FULL_CONTEXT CONTEXT_WIN*NUMBER_OF_BLOCKS       // maximum tokens for full context


/**
 * @brief ATTENTION CLASS for calculating attention head and Embeddings.
 * An array of attention head is Partial Attention (LAYER) and an array 
 * of partial attention (BLOCK) is complete attention. Attention head in 
 * complete attention working in parallel are referred as PARALLELs.
 */
class attention {
public:
    bool isSelfAttention;   // = 0 if cross attention else = 1 for self attention
    bool inTraining;        // = 1 for training, = 0 for in use
    int tokenCount;         // current token count for this head
    mlp ver;                // vertical propagation and next block transfer
    mlp hor;                // horizontal transfer to next head
    mat MQ;                 // query matrix
    mat MK;                 // key matrix
    mat MV;                 // vertical retention matrix
    mat MH;                 // horizontal retention matrix
    mat qkCache;            // QK' cache = MQ x MK'
    mat qvCache;            // QH' cache = MQ x MV'
    mat khCache;            // KV' cache = MK x MH'
// containers
    std::vector<std::vector<float>> K;          // keys = Tokens x MK
    std::vector<std::vector<float>> Q;          // Querys = Tokens x MQ
    std::vector<std::vector<float>> KdotQ;      // attention head matrix -> Keys x Querys -> [K(i).Q(j)] <- scalar
    std::vector<float> EH;      // horizontal retention vector (Next Embedding in same block)
    std::vector<std::vector<float>> EV;         // vertical retention vectors (Context retention for next block)
    std::vector<float> dh;      // delta for EH: sum of (KdotQ[i][j] * Keys[i] * MH) (row wise)
    std::vector<float> dv;      // delta for EV[i]: sum of (KdotQ[j][i] * Keys[j] * MV) (column wise)

// functions
    // default constructor
    attention() = default;
    attention(int n, int d, int h, int l);
    attention(int n, int d, int h, int l, bool attentionType);
    attention(int n, int d, int h, int l, bool attentionType, bool inTraining);
    void setAttentionType(bool attentionType);
// cpp functions for cpu
    // forward propagation for both first and specific block's attention
    void forprop(int& in, int& layers, int& tokenCount);
    void forprop(std::vector<std::vector<float>> EVp, int& in, int& layers, int& tokenCount, int& blockCount, int& n);
    // backward propagation
    void backward(std::vector<float>& expected, int& in, int& layers);
    void backward(std::vector<std::vector<float>>& expectedV, int& in, int& layers);
    void backward1stHead(std::vector<float>& expected, int& in, int& layers, bool& first);
    void backward1stHead(std::vector<std::vector<float>>& expectedV, int& in, int& layers);
    void backward1stHead(std::vector<float>& expected, std::vector<std::vector<float>>& expectedV, int& in, int& layers);

#ifdef USE_CUDA
// cuda equivalent functions for attention
    void cuforprop(int& in, int& layers, int& tokenCount);
    void cuforprop(std::vector<std::vector<float>> EVp, int& in, int& layers, int& tokenCount, int& blockCount, int& n);
    void cubackward(std::vector<float>& expected, int& in, int& layers);
    void cubackward(std::vector<std::vector<float>>& expectedV, int& in, int& layers);
    void cubackward1stHead(std::vector<float>& expected, int& in, int& layers);
    void cubackward1stHead(std::vector<std::vector<float>>& expectedV, int& in, int& layers);
    void cubackward1stHead(std::vector<float>& expectedH, std::vector<std::vector<float>>& expectedV, int& in, int& layers);
#elif USE_OPENCL
// opencl equivalent functions for attention
    void clforprop(int& in, int& layers, int& tokenCount);
    void clforprop(std::vector<std::vector<float>> EVp, int& in, int& layers, int& tokenCount, int& blockCount, int& n);
    void clbackward(std::vector<float>& expected, int& in, int& layers);
    void clbackward(std::vector<std::vector<float>>& expectedV, int& in, int& layers);
    void clbackward1stHead(std::vector<float>& expected, int& in, int& layers);
    void clbackward1stHead(std::vector<std::vector<float>>& expectedV, int& in, int& layers);
    void clbackward1stHead(std::vector<float>& expectedH, std::vector<std::vector<float>>& expectedV, int& in, int& layers);
#endif

    // default destructor
    ~attention() = default;
};

#endif
