
#ifdef USE_CPU

// backward propagation for blocks
#include "include/attention.hpp"
#include "include/block.hpp"

/**
 * @brief backward propagation for block (when only EH need to be corrected), used in 
 *      transformer::backward(std::vector<float>& expected)
 * @param expectedH expected EH for last head of each partial attention (common)
 * @param in dimension of embeddings
 * @param layers layers of MLPs
 */
void block::backward1stBlock(std::vector<float>& expectedH, int& in, int& layers, float& learning)
{
    // run partial backpropagation in parallel
    for(int i = y; i >= 1; i--) {
        if(i == y) {
            // for last column
            partialbackward1stBlock(expectedH, in, layers, i, learning);
        }
        else if(i < y and i > 1) {
            std::vector<std::vector<float>> h2(x, std::vector<float>(in, 0.0f));
            for(int j = 0; j < x; j++) {
                h2[i] = b[j][i+1].EH;
            }
            // for last second to second block
            partialbackward1stBlock(h2, in, layers, i, learning);
        }
        else if(i == 1) {
            std::vector<std::vector<float>> h2(x, std::vector<float>(in, 0.0f));
            for(int j = 0; j < x; j++) {
                h2[i] = b[j][i+1].EH;
            }
            // for first column
            partialbackward1stBlock(h2, in, layers, i, learning);
        }
        else {
            throw std::runtime_error("invalid index in backward1stBlock (H)");
        }
    }
    // serialise(blockFilePath);
}


/**
 * @brief backward propagation for block (when only EH need to be corrected), used in 
 *      transformer::backward(std::vector<float>& expected)
 * @param expectedH expected EH for last head of each partial attention (common)
 * @param in dimension of embeddings
 * @param layers layers of MLPs
 */
void block::backward1stBlock(std::vector<std::vector<float>>& expectedH, int& in, int& layers, float& learning)
{
    // run partial backpropagation in parallel
    for(int i = y; i >= 1; i--) {
        if(i == y) {
            // for last column
            partialbackward1stBlock(expectedH[i-1], in, layers, i, learning);
        }
        else if(i < y and i > 1) {
            std::vector<std::vector<float>> h2(x, std::vector<float>(in, 0.0f));
            for(int j = 0; j < x; j++) {
                h2[i] = b[j][i+1].EH;
            }
            // for last second to second column
            partialbackward1stBlock(h2, in, layers, i, learning);
        }
        else if(i == 1) {
            std::vector<std::vector<float>> h2(x, std::vector<float>(in, 0.0f));
            for(int j = 0; j < x; j++) {
                h2[i] = b[j][i+1].EH;
            }
            // for first column
            partialbackward1stBlock(h2, in, layers, i, learning);
        }
        else {
            throw std::runtime_error("invalid index in backward1stBlock (H2D)");
        }
    }
    // serialise(blockFilePath);
}


/**
 * @brief backward propagation for block (when only EV need to be corrected)
 * @param expectedV expected EVs for each head
 * @param in dimension of embeddings
 * @param layers layers of MLPs
 */
void block::backward1stBlock(std::vector<std::vector<std::vector<std::vector<float>>>>& expectedV, int& in, int& layers, float& learning)
{
    for(int i = y; i >= 1; i--) {
        partialbackward1stBlock(expectedV[i-1], in, layers, i, learning);
        /*if(i == y) {
            partialbackward1stBlock(expectedV[i-1], in, layers, i);
        }
        else if(i < y and i > 1) {
            partialbackward1stBlock(expectedV[i-1], in, layers, i);
        }
        else if(i == 1) {
            partialbackward1stBlock(expectedV[i-1], in, layers, i);
        }
        else {
            throw std::runtime_error("invalid index in backward1stBlock (V)");
        }*/
    }
    // serialise(blockFilePath);
}


/**
 * @brief backward propagation for block (when only EH need to be corrected), used in 
 *      transformer::backward(std::vector<float>& expected)
 * @param expectedH expected EH for last head of each partial attention (common)
 * @param in dimension of embeddings
 * @param layers layers of MLPs
 */
void block::backward(std::vector<float>& expectedH, int& in, int& layers, int blockCount, float& learning) {
    // run partial backpropagation in parallel
    for(int i = y; i >= 1; i--) {
        if(i == y) {
            // for last column
            partialbackward(expectedH, in, layers, i, learning);
        }
        else if(i < y and i > 1) {
            std::vector<std::vector<float>> h2(x, std::vector<float>(in, 0.0f));
            for(int j = 0; j < x; j++) {
                h2[i] = b[j][i+1].EH;
            }
            // for last second to second column
            partialbackward(h2, in, layers, i, learning);
        }
        else if(i == 1) {
            std::vector<std::vector<float>> h2(x, std::vector<float>(in, 0.0f));
            for(int j = 0; j < x; j++) {
                h2[i] = b[j][i+1].EH;
            }
            // for first column
            partialbackward(h2, in, layers, i, learning);
        }
        else {
            throw std::runtime_error("invalid index in backward (H)");
        }
    }
    // serialise(blockFilePath);
}


/**
 * @brief backward propagation for block (when only EH need to be corrected), used in 
 *      transformer::backward(std::vector<float>& expected)
 * @param expectedH expected EH for last head of each partial attention (common)
 * @param in dimension of embeddings
 * @param layers layers of MLPs
 */
void block::backward(std::vector<std::vector<float>>& expectedH, int& in, int& layers, int blockCount, float& learning) {
    // run partial backpropagation in parallel
    for(int i = y; i >= 1; i--) {
        if(i == y) {
            // for last column
            partialbackward(expectedH, in, layers, i, learning);
        }
        else if(i < y and i > 1) {
            std::vector<std::vector<float>> h2(x, std::vector<float>(in, 0.0f));
            for(int j = 0; j < x; j++) {
                h2[i] = b[j][i+1].EH;
            }
            // for last second to second column
            partialbackward(h2, in, layers, i, learning);
        }
        else if(i == 1) {
            std::vector<std::vector<float>> h2(x, std::vector<float>(in, 0.0f));
            for(int j = 0; j < x; j++) {
                h2[i] = b[j][i+1].EH;
            }
            // for first column
            partialbackward(h2, in, layers, i, learning);
        }
        else {
            throw std::runtime_error("invalid index in backward1stBlock (H)");
        }
    }
    // serialise(blockFilePath);
}


/**
 * @brief backward propagation for block (when only EV need to be corrected)
 * @param expectedV expected EVs for each head
 * @param in dimension of embeddings
 * @param layers layers of MLPs
 */
void block::backward(std::vector<std::vector<std::vector<std::vector<float>>>>& expectedV, int& in, int& layers, int blockCount, float& learning)
{
    for(int i = y; i >= 1; i--) {
        partialbackward(expectedV[i-1], layers, i, blockCount, learning);
        /*if(i == y) {
            partialbackward(expectedV[i-1], in, layers, i);
        }
        else if(i < y and i > 1) {
            partialbackward(expectedV[i-1], in, layers, i);
        }
        else if(i == 1) {
            partialbackward(expectedV[i-1], in, layers, i);
        }
        else {
            throw std::runtime_error("invalid index in backward1stBlock (V)");
        }*/
    }
    // serialise(blockFilePath);
}

#endif
