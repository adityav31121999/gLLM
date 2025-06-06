
#ifdef USE_CPU

#include "include/block.hpp"

/**
 * @brief inference of heads in single column for first block
 * @param parallelNumber column number
 */
void block::inferParallel(const mat& tokens, int &in, int &tokenCount, int &layers, int& parallelNumber)
{
    for(int i = 0; i < x; i++) {
        b[i][parallelNumber].inferHead(tokens, in, layers, tokenCount);
    }
}


/**
 * @brief inference of heads in single column for subsequent blocks
 * @param parallelNumber column number
 */
void block::inferParallel(std::vector<mat>& expectedV, const mat& tokForBlock, int &in, int &tokenCount, int &blockCount, int &layers, int &n, 
        int& parallelNumber)
{
    for(int i = 0; i < x; i++) {
        b[i][parallelNumber].inferHead(expectedV[i], tokForBlock, in, layers, tokenCount, blockCount, n);
    }
}

#endif
