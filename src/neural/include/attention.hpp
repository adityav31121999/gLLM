
/**
 * @file Attention class for calculating attention in neural networks. Attention is 
 * a mechanism that allows a neural network to focus on a specific part of the input 
 * sequence.
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
 * T6 E6 x MK -> K6  | K6.Q1   K6.Q2   K6.Q3   K6.Q4  ........................  K6.Qn | <--- HEAD
 * T7 E7 x MK -> K7  | K7.Q1   K7.Q2   K7.Q3   K7.Q4  ........................  K7.Qn |    
 * :  :              |  :      :       :       :                                :     |
 * :  :              |  :      :       :       :                                :     |
 * :  :              |  :      :       :       :                                :     |
 * Tn En x MK -> Kn  | Kn.Q1   Kn.Q2   Kn.Q3   Kn.Q4  ........................  Kn.Qn |
 * ------------------------------------------------------------------------------------
 *   dh = sum(head[i][j] * Ki.MH)
 *   dv = sum(head[i][j] * Qi.MV)
 *   EH + dh -> MLP(hor) -> ReLU(output) -> mH -> EH = EH + mH
 *   EV + dv -> MLP(ver) -> ReLU(output) -> mV -> EV = EV + mV
 * ------------------------------------------------------------------------------------
 * MQ and MK for each attention class is unique and different
 */

#ifndef ATTENTION_HPP
#define ATTENTION_HPP 1

#include <vector>
#include <maths.hpp>
#include "mlp.hpp"

#define TERMINATE "@#O"     // end of conversation (And Its Over)
#define LEARNING 0.01       // learning rate for MLPs

/**
 * @brief ATTENTION CLASS for calculating incomplete attention.
 * An array of incomplete attention is Partial Attention (LAYER) 
 * and an array of partial attention (BLOCK) is complete attention.
 */
class attention {
public:
// variables
    double error;       // error for attention
// operands
    mlp ver;            // next block transfer
    mlp hor;            // horizontal transfer
    mat MQ;             // query matrix
    mat MK;             // key matrix
    mat MV;             // vertical value for deltas
    mat MH;             // horizontal value for deltas
// containers
    std::vector<std::vector<double>> head;      // attention head matrix -> KEYs x QUERYs -> [K(i).Q(j)] <- scalar
    std::vector<std::vector<double>> KdotQ;     // = LOTA(head, CurrentTokenCount) -> probability distribution of relation between tokens
    std::vector<double> EH;         // Next Embedding in same block
    std::vector<double> dh;         // sum of (KdotQ[i][j] * Keys[i] * MH)
    std::vector<double> mh;         // ReLU of hor output
    std::vector<double> changeH;    // change in Horizontal process as expected vector for backpropagation in hor mlp

// functions
    // default constructor
    attention() = default;
    attention(int n, int d, int h, int l);

    void forprop(std::vector<std::vector<double>>&, std::vector<double>&, std::vector<double>&, std::vector<double>&, int&, int&);
    void forprop(std::vector<std::vector<double>>&, std::vector<std::vector<double>>&, std::vector<double>&, std::vector<double>&, std::vector<double>&, int&, int&, int&, int&);
    void backward(std::vector<double>&, std::vector<double>&, std::vector<double>&, std::vector<double>&);
    void train(std::vector<std::vector<double>>&, std::vector<double>&, std::vector<double>&, std::vector<double>&, int, int);

    // default destructor
    ~attention() = default;
};

#endif
