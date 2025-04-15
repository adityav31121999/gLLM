
#include "include/transformer.hpp"

/**
 * @brief Run transformer using model parameters of cache and MLPs for inference, use
 * cache QK' for calculation of KdotQ and then use caches QV' and KH' for EV and EH 
 * calculation.
 */
void transformer::run() {
    // set for inference
    inTraining = 0;
    while (1) {
        // get prompt
        // compute kdotq
        computeKdotQs(promptCount, currentTokenCount, blockCount, isSelf, inTraining);
        // caculate response
        // redo
    }
}
