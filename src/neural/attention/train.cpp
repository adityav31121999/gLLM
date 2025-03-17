
#include "include/attention.hpp"

/**
 * @brief train the attention mechanism
 * @param tokens token embeddings
 * @param holddv dv vector for next block
 * @param holdEV EV vector for next block
 * @param changeV vertical change vector for next block
 * @param in input token count
 * @param tokenCount token count for each attention head (how many tokens have been generated or taken as input)
 * @param learning learning rate for MLPs
 */
void attention::train(std::vector<std::vector<double>> &tokenEmded, std::vector<std::vector<double>> &KdotQ, std::vector<std::vector<double>> &K, 
    std::vector<std::vector<double>> &Q, std::vector<double> &dv, std::vector<double> &EV, std::vector<double> &changeV, int &in, int &layers, 
    int &tokenCount, double &learning, double &error)
{
    // 
}