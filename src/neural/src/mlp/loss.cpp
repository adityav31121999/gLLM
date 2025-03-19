
// loss.cpp: calculate losses and penalties required for mlp
#include "include/mlp.hpp"
#include <cmath>

/**
 * @brief Calculates the L1 penalty for all weights in the network.
 * The L1 penalty is the sum of the absolute value of all the weights in the network.
 * @param 3d matrix
 * @return The L1 penalty for the network.
 */
float getL1Penalty(std::vector<std::vector<std::vector<float>>>& weights) {
    float penalty = 0;
    for (const auto& layer : weights) {
        for (const auto& neuron : layer) {
            for (const auto& w : neuron) {
                penalty += std::abs(w);
            }
        }
    }
    return penalty;
}


/**
 * @brief Calculates the L2 penalty for all weights in the network.
 * The L2 penalty is the sum of the squares of all the weights in the network.
 * @param weights 3d matrix
 * @return The L2 penalty for the network.
 */
float getL2Penalty(std::vector<std::vector<std::vector<float>>>& weights) {
    float penalty = 0;
    for (const auto& layer : weights) {
        for (const auto& neuron : layer) {
            for (const auto& w : neuron) {
                penalty += w * w;
            }
        }
    }
    return penalty;
}


/**
 * @brief Calculates the LN penalty for all weights in the network.
 * The L2 penalty is the sum of the squares of all the weights in the network.
 * @param weights 3d matrix
 * @return The LN penalty for the network.
 */
float get_LN_Penalty(std::vector<std::vector<std::vector<float>>>& weights, int N) {
    float penalty = 0;
    if(N%2 == 0) {
        for (const auto& layer : weights) {
            for (const auto& neuron : layer) {
                for (const auto& w : neuron) {
                    penalty += std::pow(w, N);
                }
            }
        }
    }
    else {
        for (const auto& layer : weights) {
            for (const auto& neuron : layer) {
                for (const auto& w : neuron) {
                    penalty += std::abs(w) * std::pow(w, N-1);
                }
            }
        }
    }
    return penalty;
}


/**
 * @brief Calculates the Lxyz penalty for all weights in the network.
 * The L2 penalty is the sum of the squares of all the weights in the network.
 * @param weights 3d matrix
 * @return The Lxyz penalty for the network.
 */
float getLxyzPenalty(std::vector<std::vector<std::vector<float>>>& weights) {
    float penalty = 0;
    for (int i = 0; i < weights.size(); i++) {
        for (int j = 0; j < weights.size(); j++) {
            for (int k = 0; k < weights.size(); k++) {
                if(i*j*k % 2 == 0) 
                    penalty += std::pow(weights[i][j][k], i * j * k);
                else
                    penalty += std::abs(weights[i][j][k]) * std::pow(weights[i][j][k], i * j * k - 1);
            }
        }
    }
    return penalty;
}


/**
 * @brief Calculates the LN penalty for all weights in the network.
 * The L2 penalty is the sum of the squares of all the weights in the network.
 * @param weights 3d matrix
 * @return The LN penalty for the network.
 */
float get_LNs_Penalty(std::vector<std::vector<std::vector<float>>>& weights, int N) {
    float penalty = 0;
    if(N % 2 == 0) {
        for (int i = 0; i < weights.size(); i++) {
            for (int j = 0; j < weights.size(); j++) {
                for (int k = 0; k < weights.size(); k++) {
                    penalty += std::pow(weights[i][j][k], i+j+k);
                }
            }
        }
    }
    else {
        for (int i = 0; i < weights.size(); i++) {
            for (int j = 0; j < weights.size(); j++) {
                for (int k = 0; k < weights.size(); k++) {
                    penalty += std::pow(weights[i][j][k], i*j*k);
                }
            }
        }
    }
    return penalty;
}


/**
 * @brief Computes the loss with L1 regularization. The loss is the sum of 
 * the absolute difference between the predicted output and the target output.
 * The L1 regularization term is added to the loss.
 * @param outputs The predicted output of the network.
 * @param targets The target output of the network.
 * @param network The network to compute the loss for.
 * @param lambda The regularization parameter.
 * @return The loss with L1 regularization.
 */
float computeLossWithL1(std::vector<float>& outputs, std::vector<float>& targets, mlp& network, float lambda) {
    float loss = 0;
    for (size_t i = 0; i < outputs.size(); ++i) {
        loss += std::abs(outputs[i] - targets[i]);
    }
    return loss + 0.5 * lambda * getL1Penalty(network.weights);
}


/**
 * @brief Computes the loss with L2 regularization. The loss is the sum of the 
 * squared difference between the predicted output and the target output.
 * The L2 regularization term is added to the loss.
 * @param outputs The predicted output of the network.
 * @param targets The target output of the network.
 * @param network The network to compute the loss for.
 * @param lambda The regularization parameter.
 * @return The loss with L2 regularization.
 */
float computeLossWithL2(std::vector<float>& outputs, std::vector<float>& targets, mlp& network, float lambda) {
    float loss = 0;
    for (size_t i = 0; i < outputs.size(); ++i) {
        loss += std::pow(outputs[i] - targets[i], 2);
    }
    return 0.5 * loss + 0.5 * lambda * getL2Penalty(network.weights);
}


/**
 * @brief Computes the loss with dropout generalization. The loss is the sum of 
 * the squared difference between the predicted output and the target output.
 * The dropout generalization term is added to the loss.
 * @param outputs The predicted output of the network.
 * @param targets The target output of the network.
 * @param network The network to compute the loss for.
 * @param p The dropout probability.
 * @return The loss with dropout generalization.
 */
float dropoutGeneralisation(std::vector<float>& outputs, std::vector<float>& targets, mlp& network, float p) {
    float loss = 0;
    for (size_t i = 0; i < outputs.size(); ++i) {
        loss += std::pow(outputs[i] - targets[i], 2);
    }
    return loss / (1 - p);
}
