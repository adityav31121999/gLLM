
#include "include/attention.hpp"
#include "include/block.hpp"
#include "include/transformer.hpp"


/**
 * @brief forward propagation for transformers
 * @param blockCount current block index
 * @param currentTokenCount current number of tokens
 * @param promptCount number of tokens in prompt
 */
void transformer::forward(int& blockCount, int& currentTokenCount, int& promptCount)
{
    // perform forward propagation for blocks
    if(blockCount == 0) {
        // first block
        if(currentTokenCount == 0) {
            // compute the KdotQ
            computeKdotQs(promptCount, currentTokenCount, blockCount, isSelf);
            // forward propagation to start conversation
            t[0].forprop(d, currentTokenCount, l);
        }
    }
    else {
        // compute the KdotQ
        computeKdotQs(promptCount, currentTokenCount, blockCount, isSelf);
        //second to last block
        t[blockCount].forprop(t[blockCount-1].EV, d, currentTokenCount, blockCount, l, n);
    }
}
