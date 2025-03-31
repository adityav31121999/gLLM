
// backward propagation of all types here
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
        b[k][i].backward(expectedV[i], in, layers);
    }
}


/**
 * @brief backward propagation for block (when both EH and EV need to be corrected)
 * @param expectedV expected EVs for each head
 * @param expectedH expected EHs for last head of each partial attention
 * @param in dimension of embeddings
 * @param layers layers of MLPs
 */
void block::backward(std::vector<std::vector<std::vector<std::vector<float>>>>& expectedV, std::vector<std::vector<float>>& expectedH,
    int& in, int& layers) 
{
    for(int i = 0; i < x; i++) {
        partialbackward(expectedV[i], expectedH[i], in, layers, i);
    }
}


/**
 * @brief backward propagation for block (when both EH and EV need to be corrected)
 * @param expectedV expected EVs for each head
 * @param expectedH expected EH for last head of each partial attention (common)
 * @param in dimension of embeddings
 * @param layers layers of MLPs
 */
void block::backward(std::vector<std::vector<std::vector<std::vector<float>>>>& expectedV, std::vector<float>& expectedH, int& in, 
    int& layers)
{
    for(int i = 0; i < x; i++) {
        partialbackward(expectedV[i], expectedH, in, layers, i);
    }
}


/**
 * @brief backward propagation for block (when only EV need to be corrected)
 * @param expectedV expected EVs for each head
 * @param in dimension of embeddings
 * @param layers layers of MLPs
 */
void block::backward(std::vector<std::vector<std::vector<std::vector<float>>>>& expectedV, int& in, int& layers) {
    for(int i = 0; i < x; i++) {
        partialbackward(expectedV[i], in, layers, i);
    }
}


/**
 * @brief backward propagation for block (when only EH need to be corrected)
 * @param expectedH expected EH for last head of each partial attention (common)
 * @param in dimension of embeddings
 * @param layers layers of MLPs
 */
void block::backward(std::vector<float>& expectedH, int& in, int& layers) {
    // run partial backpropagation in parallel
    for(int i = x-1; i >= 0; i++) {
        partialbackward(expectedH, in, layers, i);
    }
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
*/
