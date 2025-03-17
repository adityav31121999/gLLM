
// attention.hpp: header source for attention class
#ifndef ATTENTION_HPP
#define ATTENTION_HPP 1

/**
 * K[i] = T[i] * MK, Q[i] = T[i] * MQ
 * KdotQ[i][j] = K[i].Q[j]
 * head = LOTA(KdotQ) OR LOTA(ReLU(KdotQ)) OR Softmax(KdotQ)
 * dh = sum(head[i][j] * Ki.MH), dv = sum(head[i][j] * Qi.MV)
 * Input(EH + dh) -> MLP(hor) -> ReLU(output) -> mH -> EH = EH + mH
 * Input(EV + dv) -> MLP(ver) -> ReLU(output) -> mV -> EV = EV + mV
 */

#include <vector>
#include <maths.hpp>
#include "mlp.hpp"

#define TERMINATE "@#O"     // end of conversation (And Its Over)
#define LEARNING 0.01       // learning rate for MLPs
#define EMBEDDING 64        // embedding dimension for each token
#define SCALING std::sqrt(EMBEDDING)    // SCALING FACTOR for ATTENTION HEAD

/**
 * @brief ATTENTION CLASS for calculating attention head and Embeddings.
 * An array of incomplete attention is Partial Attention (LAYER) and an 
 * array of partial attention (BLOCK) is complete attention.
 */
class attention {
public:
// operands
    mlp ver;            // next block transfer
    mlp hor;            // horizontal transfer
    mat MQ;             // query matrix
    mat MK;             // key matrix
    mat MV;             // vertical value for deltas
    mat MH;             // horizontal value for deltas
// containers
    std::vector<std::vector<double>> K;         // keys = Tokens x MK
    std::vector<std::vector<double>> Q;         // Querys = Tokens x MQ
    std::vector<std::vector<double>> head;      // attention head matrix -> Keys x Querys -> [K(i).Q(j)] <- scalar
    std::vector<std::vector<double>> KdotQ;     // = LOTA(head, CurrentTokenCount) -> probability distribution of relation between tokens
    std::vector<double> EH;         // Next Embedding in same block
    std::vector<double> dh;         // sum of (KdotQ[i][j] * Keys[i] * MH)
    std::vector<double> mh;         // ReLU of hor output
    std::vector<double> changeH;    // change in Horizontal process as expected vector for backpropagation in hor mlp

// functions
    // default constructor
    attention() = default;
    attention(int n, int d, int h, int l);
    // compute attention
    void computeAttention(std::vector<std::vector<double>>& KdotQ, std::vector<std::vector<double>>& Keys, std::vector<std::vector<double>>& Queries, int count);
    // forward propagation for both first and specific block's attention
    void forprop(std::vector<std::vector<double>>& tokenEmbed, std::vector<std::vector<double>>& KdotQ, std::vector<std::vector<double>>& K, 
                    std::vector<std::vector<double>>& Q, std::vector<double>& dv, std::vector<double>& EV, std::vector<double>& changeV, int& in, 
                    int& layers, int& tokenCount);
    void forprop(std::vector<std::vector<double>>& tokenEmbed, std::vector<std::vector<double>>& KdotQ, std::vector<std::vector<double>>& K, 
                    std::vector<std::vector<double>>& Q, std::vector<std::vector<double>>& EVp, std::vector<double>& dv, std::vector<double>& EVc, 
                    std::vector<double>& changeV, int& in, int& layers, int& tokenCount, int& blockCount, int& n);
    // backward propagation
    void backward(std::vector<double>& expected, std::vector<double>& changeV, std::vector<double>& dv, std::vector<double>& EV,
                    int& in, int& layers);
    // train the attention class
    void train(std::vector<std::vector<double>>& tokenEmded, std::vector<std::vector<double>>& KdotQ, std::vector<std::vector<double>>& K, 
                    std::vector<std::vector<double>>& Q, std::vector<double>& dv, std::vector<double>& EV, std::vector<double>& changeV, int& in, 
                    int& layers, int& tokenCount, double& learning, double& error);

    // default destructor
    ~attention() = default;
};

#endif
