
#include "include/attention.hpp"
#include "include/block.hpp"

/**
 * @brief partial attention backpropagation on the ith block
 * @param tExp expected token
 * @param tokenCount number of tokens predicted/generated or provided as input
 * @param layer layer number
 */
void block::partialbackward(std::vector<double>& expected, std::vector<std::vector<double>>& changeV, 
    std::vector<std::vector<double>>& dv, std::vector<std::vector<double>>& EV, int& layer) 
{
    for(int i = b[layer].size(); i >= 0; i++) {
        b[layer][i].backward(expected, changeV[i], dv[i], EV[i]);            
        expected = b[layer][i].EH;
    }
}

/**
 * @brief complete attention backward propagation
 * @param tExp expected token
 * @param tokenCount number of tokens predicted/generated or provided as input
 */
void block::backward(std::vector<double>& expected, std::vector<std::vector<std::vector<double>>>& changeV, 
    std::vector<std::vector<std::vector<double>>>& dv, std::vector<std::vector<std::vector<double>>>& EV) 
{
    for(int i = b.size(); i >= 0; i++) {
        partialbackward(expected, changeV[i], dv[i], EV[i], i);
    }
}
