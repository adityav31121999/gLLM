
// block forward propagation
#include "include/attention.hpp"

/**
 * @brief partial attention forward propagation
 */
void block::partialforprop(int i) {
    // for one partial attention
    for(int j = 0; j < y; j++) {
        b[i][j].forprop(tokens, tokenCount);      // incomplete attention forprop
        holdEVs[i][j] = b[i][j].EV;
        b[i][j + 1].EH = b[i][j].EH;
    }
}

/**
 * @brief forward propagation
 */
void block::forprop() {
    // y partial attention in x layers => x parallel processes
#ifdef HAS_CL
    // run partial forprop in parallel with OpenCL
#elif HAS_CUDA
    // run partial forprop in parallel with CUDA
#endif
}
