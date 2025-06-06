
// backward propagation for blocks
#include "include/attention.hpp"
#include "include/block.hpp"

#ifdef USE_CPU

/**
 * @brief backward propagation for kth layer (when expected EV is not known, for block where backprop begins)
 * @param expectedH expected token vector
 * @param in dimension of embedding = EMBEDDING
 * @param layers layers of MLP
 * @param k layer number
 */
void block::partialbackward(std::vector<float>& expectedH, int& in, int& layers, int k) {
    bool first = 0;
    std::vector<float> ex = expectedH;
    for(int i = y-1; i >= 1; i ++) {
        b[k][i].backward(ex, in, layers);
        ex = b[k][0].EH;
    }
    // need not to update the EH
    b[k][0].backward1stHead(ex, in, layers, first);
}


/**
 * @brief backward propagation for kth layer (when EVs need to be corrected only, for intermediate blocks)
 * @param expectedV expected retention vectors
 * @param in dimension of embedding = EMBEDDING
 * @param layers layers of MLP
 * @param k layer number
 */
void block::partialbackward(std::vector<std::vector<std::vector<float>>>& expectedV, int& in, int& layers, int k) {
    for(int i = y-1; i >= 0; i ++) {
        b[k][i].backward(expectedV[i], layers);
    }
}


/**
 * @brief backward propagation for kth layer (when expected EV is not known, for first block)
 * @param expectedH expected token vector
 * @param in dimension of embedding
 * @param layers layers of MLP
 * @param k layer number
 */
void block::partialbackward1stBlock(std::vector<float>& expectedH, int& in, int& layers, int k) {
    bool first = 0;
    std::vector<float> ex = expectedH;
    for(int i = y-1; i >= 1; i ++) { 
        b[k][i].backward1stHead(ex, in, layers, first);
        ex = b[k][0].EH;
    }
    first = 1;
    b[k][0].backward1stHead(ex, in, layers, first);
}


/**
 * @brief backward propagation for kth layer (when EVs need to be corrected only, backropagation is continued here to end)
 * @param expectedV expected retention vectors
 * @param in dimension of embedding = EMBEDDING
 * @param layers layers of MLP
 * @param k layer number
 */
void block::partialbackward1stBlock(std::vector<std::vector<std::vector<float>>>& expectedV, int& in, int& layers, int k) {
    for(int i = y-1; i >= 0; i ++) {
        b[k][i].backward1stHead(expectedV[i], in, layers);
    }
}

#endif