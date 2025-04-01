
// backward propagation for blocks
#include "include/attention.hpp"
#include "include/block.hpp"

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
        // for each layer expectedH is available and for each head expectedV is available
        partialbackward(expectedV[i], expectedH[i], in, layers, i);
    }
}


/**
 * @brief backward propagation for block (when both EH and EV need to be corrected), use 
 *          in transformer::backward(std::vector<float>& expected, int& k)
 * @param expectedV expected EVs for each head
 * @param expectedH expected EH for last head of each partial attention (common)
 * @param in dimension of embeddings
 * @param layers layers of MLPs
 */
void block::backward(std::vector<std::vector<std::vector<std::vector<float>>>>& expectedV, std::vector<float>& expectedH, int& in, 
    int& layers)
{
    for(int i = 0; i < x; i++) {
        // common expectedH and for each head expectedV is available
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
 * @brief backward propagation for block (when both EH and EV need to be corrected)
 * @param expectedV expected EVs for each head
 * @param expectedH expected EHs for last head of each partial attention
 * @param in dimension of embeddings
 * @param layers layers of MLPs
 */
void block::backward1stBlock(std::vector<std::vector<std::vector<std::vector<float>>>>& expectedV, std::vector<std::vector<float>>& expectedH,
    int& in, int& layers) 
{
    for(int i = 0; i < x; i++) {
        // for each layer expectedH is available and for each head expectedV is available
        partialbackward1stBlock(expectedV[i], expectedH[i], in, layers, i);
    }
}


/**
 * @brief backward propagation for block (when both EH and EV need to be corrected), use 
 *          in transformer::backward(std::vector<float>& expected, int& k)
 * @param expectedV expected EVs for each head
 * @param expectedH expected EH for last head of each partial attention (common)
 * @param in dimension of embeddings
 * @param layers layers of MLPs
 */
void block::backward1stBlock(std::vector<std::vector<std::vector<std::vector<float>>>>& expectedV, std::vector<float>& expectedH, int& in, 
    int& layers)
{
    for(int i = 0; i < x; i++) {
        // common expectedH and for each head expectedV is available
        partialbackward1stBlock(expectedV[i], expectedH, in, layers, i);
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
