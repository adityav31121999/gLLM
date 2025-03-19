
#include "include/attention.hpp"
#include "include/block.hpp"

/**
 * @brief partial attention backpropagation on the ith block
 * @param tExp expected token
 * @param tokenCount number of tokens predicted/generated or provided as input
 * @param layer layer number
 */
void block::partialbackward(std::vector<float>& expected, std::vector<std::vector<float>>& changeV, 
    std::vector<std::vector<float>>& dv, std::vector<std::vector<float>>& EV, int& in, int& layers, int layno) 
{
    for(int i = b[layno].size(); i >= 0; i++) {
        b[layno][i].backward(expected, changeV[i], dv[i], EV[i], in, layers);            
        expected = b[layno][i].EH;
    }
}


/**
 * @brief complete attention backward propagation
 * @param tExp expected token
 * @param tokenCount number of tokens predicted/generated or provided as input
 */
void block::backward(std::vector<float>& expected, std::vector<std::vector<std::vector<float>>>& changeV, 
    std::vector<std::vector<std::vector<float>>>& dv, std::vector<std::vector<std::vector<float>>>& EV, int& in,
    int& layers) 
{
    for(int i = b.size(); i >= 0; i++) {
        partialbackward(expected, changeV[i], dv[i], EV[i], in, layers, i);
    }
}
