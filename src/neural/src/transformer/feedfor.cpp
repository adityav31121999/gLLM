
#include "include/attention.hpp"
#include "include/block.hpp"
#include "include/transformer.hpp"

#ifndef USE_CUDA && USE_OPENCL

/**
 * @brief forward propagation for transformers
 * @param blockCount current block index
 * @param currentTokenCount current number of tokens
 * @param promptCount number of tokens in prompt
 */
void transformer::forward(int& blockCount, int& currentTokenCount, int& promptCount)
{
    // compute the KdotQ
    computeKdotQs(promptCount, currentTokenCount, blockCount, isSelf, inTraining);
    // perform forward propagation for first blocks
    if(blockCount == 0) {
        t[0].forprop(d, currentTokenCount, l);
        for(int i = 0; i < d; i++) {
            for(int j = 0; j < x; j++) {
                otok[i] += t[0].b[j][y-1].EH[i];
            }  
        }
        // int index = 0;
        computeOutput(otok, embeddings, vocabsize, indexForToken);
        // tokenEmbed[currentTokenCount] = ;    // extract the token at 'index' and add it to token
    }
    // for ith block
    else {
        // compute the KdotQ
        computeKdotQs(promptCount, currentTokenCount, blockCount, isSelf, inTraining);
        //second to last block
        t[blockCount-1].forprop(t[blockCount-2].EV, d, currentTokenCount, blockCount, l, n);
        for(int i = 0; i < d; i++) {
            for(int j = 0; j < x; j++) {
                otok[i] += t[0].b[j][y-1].EH[i];
            }  
        }
        // int index = 0;
        computeOutput(otok, embeddings, vocabsize, indexForToken);
        // tokenEmbed[currentTokenCount] = ;    // extract the token at 'index' and add it to token
    }
}

#endif