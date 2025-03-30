
// backward propagation of all types here
#include "include/attention.hpp"
#include "include/block.hpp"

/**
 * @brief backward propagation for kth layer
 * @param expectedH expected token vector
 * @param in dimension of embedding = EMBEDDING
 * @param layers layers of MLP
 * @param k layer number
 */
void block::partialbackward(std::vector<float>& expectedH, int& in, int& layers, int k) {
    std::vector<float> ex = expectedH;
    for(int i = y-1; i >= 0; i ++) {
        b[k][i].backward(ex, in, layers);
        if(i == 0)
            break;
        ex = b[k][0].EH;
    }
}


/**
 * @brief backward propagation for kth layer
 * @param expectedV expected retention vectors
 * @param expectedH expected token vector
 * @param in dimension of embedding = EMBEDDING
 * @param layers layers of MLP
 * @param k layer number
 */
void block::partialbackward(std::vector<std::vector<float>>& expectedV, std::vector<float>& expectedH, int& in, int& layers, int k) {
    std::vector<float> ex = expectedH;
    for(int i = y-1; i >= 0; i ++) {
        b[k][i].backward(ex, expectedV[i], in, layers);
        if(i == 0)
            break;
        ex = b[k][i].EH;
    }
}


/**
 * @brief backward propagation for kth layer
 * @param expectedV expected retention vectors
 * @param in dimension of embedding = EMBEDDING
 * @param layers layers of MLP
 * @param k layer number
 */
void block::partialbackward(std::vector<std::vector<float>>& expectedV, int& in, int& layers, int k) {
    std::vector<float> nil = std::vector<float>(EMBEDDING, 0.0f);       // nill vector
    for(int i = y-1; i >= 0; i ++) {
        b[k][i].backward(nil, expectedV[i], in, layers);
        if(i == 0)
            break;
        nil = b[k][0].EH;
    }
}


/**
 * @brief complete attention backward propagation
 * @param tExp expected token
 * @param tokenCount number of tokens predicted/generated or provided as input
 */
void block::backward(std::vector<float>& expectedH, int& in, int& layers) {
    // run partial backpropagation in parallel
    std::vector<float> ex = expectedH;
    for(int i = x-1; i >= 0; i++) {
        partialbackward(ex, in, layers, i);
        if(i == 0)
            break;
        ex = b[i][0].EH;
    }
}

void block::backward(std::vector<std::vector<float>>& expectedV, std::vector<float>& expectedH, int& in, int& layers) {
    //
}


void block::backward(std::vector<std::vector<float>>& expected, int& in, int& layers) {
    //
}


/**
    for(int i = b.size()-1; i >= 0; i--) {
        std::vector<float> ex = expected;
        for(int j = b[0].size()-1; j >=0; j--) {
            b[i][j].backward(expected[i][j], in, layers);
            if(i == 0)
                break;
            ex = b[i][0].EH;
        }
    }
     */
