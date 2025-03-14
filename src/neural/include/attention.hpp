
// attention.hpp: header source for attention class
#ifndef ATTENTION_HPP
#define ATTENTION_HPP 1

/**
 * K[i] = T[i] * MK, Q[i] = T[i] * MQ
 * KdotQ[i][j] = K[i].Q[j]
 * head = LOTA(KdotQ)
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
    std::vector<std::vector<double>> K;         // keys = TxMK
    std::vector<std::vector<double>> Q;         // Querys = TXMQ
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

    void forprop(std::vector<std::vector<double>>& tokenEmbed, std::vector<double>& dv, std::vector<double>& EV, std::vector<double>& changeV,
                    int& in, int& layers, int& tokenCount);
    void forprop(std::vector<std::vector<double>>& tokenEmbed, std::vector<std::vector<double>>& EVp, std::vector<double>& dv, std::vector<double>& EVc,
                    std::vector<double>& changeV, int& in, int& layers, int& tokenCount, int& blockCount, int& n);
    void backward(std::vector<double>& expected, std::vector<double>& changeV, std::vector<double>& dv, std::vector<double>& EV,
                    int& in, int& layers);
    void train(std::vector<std::vector<double>>& tokenEmded, std::vector<double>& dv, std::vector<double>& EV, std::vector<double>& changeV,
                    int& in, int& layers, int& tokenCount, double& learning, double& error);

    // default destructor
    ~attention() = default;
};

#endif


/**
 *                              DOT PRODUCT MATRIX
 * ------------------------------------------------------------------------------------
 *                       T1      T2      T3      T4  ..........................  Tn
 *                       E1      E2      E3      E4  ..........................  En
 *                        x       x       x       x                               x
 *                       MQ      MQ      MQ      MQ            -------           MQ
 *                       ->      ->      ->      ->                              ->
 *                       Q1      Q2      Q3      Q4  ..........................  Qn
 * ------------------------------------------------------------------------------------
 * T1 E1 x MK -> K1  | K1.Q1   K1.Q2   K1.Q3   K1.Q4  ........................  K1.Qn |
 * T2 E2 x MK -> K2  | K2.Q1   K2.Q2   K2.Q3   K2.Q4  ........................  K2.Qn |
 * T3 E3 x MK -> K3  | K3.Q1   K3.Q2   K3.Q3   K3.Q4  ........................  K3.Qn |
 * T4 E4 x MK -> K4  | K4.Q1   K4.Q2   K4.Q3   K4.Q4  ........................  K4.Qn |
 * T5 E5 x MK -> K5  | K5.Q1   K5.Q2   K5.Q3   K5.Q4  ........................  K5.Qn |
 * T6 E6 x MK -> K6  | K6.Q1   K6.Q2   K6.Q3   K6.Q4  ........................  K6.Qn | <--- KdotQ
 * T7 E7 x MK -> K7  | K7.Q1   K7.Q2   K7.Q3   K7.Q4  ........................  K7.Qn |    
 * :  :              |  :      :       :       :                                :     |
 * :  :              |  :      :       :       :                                :     |
 * :  :              |  :      :       :       :                                :     |
 * Tn En x MK -> Kn  | Kn.Q1   Kn.Q2   Kn.Q3   Kn.Q4  ........................  Kn.Qn |
 * ------------------------------------------------------------------------------------
 * KdotQ = matrix(Ki.Qj)
 * head = LOTA(KdotQ) = softmax(KdotQ) = LOTA(ReLU(KdotQ)) = softmax(ReLU(KdotQ)) = inverse(KdotQ)
 * dh = sum(head[i][j] * Ki.MH) = sum(sum(row[i]) * Ki) *MH
 * dv = sum(head[i][j] * Qi.MV) = sum(sum(col[j]) * Qi) *MV
 * where row[i] = ith row of head and col[j] = jth column of head
 * EH + dh -> MLP(hor) -> ReLU(output) -> mH -> EH = EH + mH
 * EV + dv -> MLP(ver) -> ReLU(output) -> mV -> EV = EV + mV
 * ------------------------------------------------------------------------------------
 * MQ and MK for each attention class is unique and different
 */
