
// backward propagation for blocks
#include "include/attention.hpp"
#include "include/block.hpp"


/**
 * @brief backward propagation for kth layer (when expected EV is not known)
 * @param expectedH expected token vector
 * @param in dimension of embedding = EMBEDDING
 * @param layers layers of MLP
 * @param k layer number
 */
void block::partialbackward(std::vector<float>& expectedH, int& in, int& layers, int k) {
    std::vector<float> ex = expectedH;
    for(int i = y-1; i >= 0; i ++) {
        // 
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
void block::partialbackward(std::vector<std::vector<std::vector<float>>>& expectedV, std::vector<float>& expectedH, int& in, int& layers, 
    int k) 
{
    std::vector<float> ex = expectedH;
    for(int i = y-1; i >= 0; i ++) {
        // 
        b[k][i].backward(ex, expectedV[i], in, layers);
        if(i == 0)
            break;
        ex = b[k][i].EH;
    }
}


/**
 * @brief backward propagation for kth layer (when EVs need to be corrected only)
 * @param expectedV expected retention vectors
 * @param in dimension of embedding = EMBEDDING
 * @param layers layers of MLP
 * @param k layer number
 */
void block::partialbackward(std::vector<std::vector<std::vector<float>>>& expectedV, int& in, int& layers, int k) {
    for(int i = y-1; i >= 0; i ++) {
        // 
        b[k][i].backward(expectedV[i], in, layers);
    }
}
