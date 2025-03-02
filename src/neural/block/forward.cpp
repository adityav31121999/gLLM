
#include "include/attention.hpp"

/**
 * @brief forward propagation of the partial attention.
 */
void block::paForward(int k) {
    // kth partial attention in the block
    std::fill(b[k][0].EV.begin(), b[k][0].EV.end(), 0);
    std::fill(b[k][0].EH.begin(), b[k][0].EH.end(), 0);
    for(int i = 0; i < y; i++) {
        b[k][i].forward();
        holdEVs[k][i] = b[k][i].EV;
    }
}


/**
 * @brief forward propagation of the attention block. Use paForward
 * in parallel for all partial attentions in a block.
 */
void block::forward() {
    // run paForward in parallel with CUDA and OpenCL
    tokenCount = tokenCount + 1;
}
