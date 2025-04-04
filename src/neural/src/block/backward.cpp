
// backward propagation for blocks
#include "include/attention.hpp"
#include "include/block.hpp"


/**
 * @brief backward propagation for block (when only EV need to be corrected)
 * @param expectedV expected EVs for each head
 * @param in dimension of embeddings
 * @param layers layers of MLPs
 */
void block::backward(std::vector<std::vector<std::vector<std::vector<float>>>>& expectedV, int& in, int& layers) {
    for(int i = 0; i < x; i++) {
        // for each layer expectedV is available and the horizontal embedding is accurate
        partialbackward(expectedV[i], in, layers, i);
    }
}


/**
 * @brief backward propagation for block (when only EH need to be corrected), used in 
 *      transformer::backward(std::vector<float>& expected)
 * @param expectedH expected EH for last head of each partial attention (common)
 * @param in dimension of embeddings
 * @param layers layers of MLPs
 */
void block::backward(std::vector<std::vector<float>>& expectedH, int& in, int& layers) {
    // run partial backpropagation in parallel
    for(int i = x-1; i >= 0; i++) {
        // expectedV for each is not considered since the horizontal pass ended in this block
        partialbackward(expectedH[i], in, layers, i);
    }
}


/**
 * @brief backward propagation for block (when only EH need to be corrected), used in 
 *      transformer::backward(std::vector<float>& expected)
 * @param expectedH expected EH for last head of each partial attention (common)
 * @param in dimension of embeddings
 * @param layers layers of MLPs
 */
void block::backward(std::vector<float>& expectedH, int& in, int& layers) {
    // run partial backpropagation in parallel
    for(int i = x-1; i >= 0; i++) {
        // expectedV for each is not considered since the horizontal pass ended in this block
        partialbackward(expectedH, in, layers, i);
    }
}


/**
 * @brief backward propagation for block (when only EV need to be corrected)
 * @param expectedV expected EVs for each head
 * @param in dimension of embeddings
 * @param layers layers of MLPs
 */
void block::backward1stBlock(std::vector<std::vector<std::vector<std::vector<float>>>>& expectedV, int& in, int& layers) {
    for(int i = 0; i < x; i++) {
        // for each layer expectedV is available and the horizontal embedding is accurate
        partialbackward1stBlock(expectedV[i], in, layers, i);
    }
}


/**
 * @brief backward propagation for block (when only EH need to be corrected), used in 
 *      transformer::backward(std::vector<float>& expected)
 * @param expectedH expected EH for last head of each partial attention (common)
 * @param in dimension of embeddings
 * @param layers layers of MLPs
 */
void block::backward1stBlock(std::vector<std::vector<float>>& expectedH, int& in, int& layers) {
    // run partial backpropagation in parallel
    for(int i = x-1; i >= 0; i++) {
        // expectedV for each is not considered since the horizontal pass ended in this block
        partialbackward1stBlock(expectedH[i], in, layers, i);
    }
}


/**
 * @brief backward propagation for block (when only EH need to be corrected), used in 
 *      transformer::backward(std::vector<float>& expected)
 * @param expectedH expected EH for last head of each partial attention (common)
 * @param in dimension of embeddings
 * @param layers layers of MLPs
 */
void block::backward1stBlock(std::vector<float>& expectedH, int& in, int& layers) {
    // run partial backpropagation in parallel
    for(int i = x-1; i >= 0; i++) {
        // expectedV for each is not considered since the horizontal pass ended in this block
        partialbackward1stBlock(expectedH, in, layers, i);
    }
}
