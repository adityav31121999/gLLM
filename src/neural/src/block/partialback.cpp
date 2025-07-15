#ifdef USE_CPU

// backward propagation for blocks
#include "include/attention.hpp"
#include "include/block.hpp"

/**
 * @brief backward propagation for last column of first block
 * @param expectedH expected token vector
 * @param in dimension of embedding
 * @param layers layers of MLP
 * @param k column numebr
 */
void block::partialbackward1stBlock(std::vector<float>& expectedH, int& in, int& layers, int k, float& learning) {
    for(int i = 0; i < x; i ++) {
        // for last column
        b[i][k].backward1stHead(expectedH, in, layers, k, learning);
    }
}


/**
 * @brief backward propagation for first and last second layer, last if training for multiple outputs
 * @param expectedH expected horizontal retention vectors
 * @param in dimension of embedding
 * @param layers layers of MLP
 * @param k column number
 */
void block::partialbackward1stBlock(std::vector<std::vector<float>>& expectedH, int& in, int& layers, int k, float& learning) {
    for(int i = 0; i < x; i ++) {
        // for ith column from last second to first column
        b[i][k].backward1stHead(expectedH[i], in, layers, k, learning);
    }
}


/**
 * @brief backward propagation for first block
 * @param expectedV expected vertical retention vectors for each head of vertical vectors
 * @param in dimension of embedding = EMBEDDING
 * @param layers layers of MLP
 * @param k column number
 */
void block::partialbackward1stBlock(std::vector<std::vector<std::vector<float>>>& expectedV, int& in, int& layers, int k, float& learning) {
    for(int i = 0; i < x; i ++) {
        // for all columns of block
        b[i][k].backward1stHead(expectedV[i], in, layers, learning);
    }
}


/**
 * @brief backward propagation for last column of kth block
 * @param expectedH expected token vector
 * @param in dimension of embedding = EMBEDDING and input-output vector of mlp
 * @param layers layers of MLP
 * @param k column number (b[i][k])
 */
void block::partialbackward(std::vector<float>& expectedH, int& in, int& layers, int k, float& learning) {
    for(int i = 0; i < x; i ++) {
        // for last column of block
        b[i][k].backward(expectedH, in, layers, k, learning);
    }
}


/**
 * @brief backward propagation for first and last second layer, last if training for multiple outputs
 * @param expectedH expected horizontal retention vectors
 * @param in dimension of embedding = EMBEDDING and input-output vector of mlp
 * @param layers layers of MLP
 * @param k column number (b[i][k])
 */
void block::partialbackward(std::vector<std::vector<float>>& expectedH, int& in, int& layers, int k, float& learning) {
    for(int i = 0; i < x; i ++) {
        // for last second to first column
        b[i][k].backward(expectedH[i], in, layers, k, learning);
    }
}


/**
 * @brief backward propagation for kth layer of ith block
 * @param expectedV expected vertical retention vectors for each head of vertical vectors
 * @param in dimension of embedding = EMBEDDING
 * @param layers layers of MLP
 * @param k column number
 * @param i current block index 1-based
 */
void block::partialbackward(std::vector<std::vector<std::vector<float>>>& expectedV, int& layers, int k, int i, float& learning) {
    for(int i = 0; i < x; i ++) {
        // for all columns of block
        b[i][k].backward(expectedV[i], layers, i, learning);
    }
}

#endif
