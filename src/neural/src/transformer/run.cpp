
#include "include/transformer.hpp"

/**
 * @brief program to run transformer using model parameters of cache and MLPs for inference
 */
void transformer::run() {
    while(1) {
        // get token embedding for prompts and provide it to block for KdotQs
        // inTraining = 0 for inference
        computeKdotQs(promptCount, currentTokenCount, blockCount, isSelf, inTraining);
        // forward(promptCount, currentTokenCount, blockCount);
        if(blockCount == 1) {
            t[0].forprop(d, currentTokenCount, l);
        }
        else {
            t[blockCount-1].forprop(t[blockCount-2].EV, d, currentTokenCount, blockCount, l, n);
        }
        // compute output
        computeOutput(otok, embeddings, vocabsize, indexForToken);
        
    }
}
