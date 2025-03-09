
#include "include/attention.hpp"
#include "include/block.hpp"

/**
 * @brief partial attention backpropagation on the ith block
 * @param tExp expected token
 * @param tokenCount number of tokens predicted/generated or provided as input
 * @param layer layer number
 */
void block::partialbackward(std::vector<double> tExp, int tokenCount, int layer, double learning) {
    std::vector<double> p(tExp);
    for(int i = b[layer].size(); i >= 0; i++) {
        b[layer][i].backward(p, learning);
        p = b[layer][i].EH;
    }
}

/**
 * @brief complete attention backward propagation
 * @param tExp expected token
 * @param tokenCount number of tokens predicted/generated or provided as input
 */
void block::backward(std::vector<double> tExp, int tokenCount, double learning) {
    //
}
