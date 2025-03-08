
#include "include/attention.hpp"

/**
 * @brief partial attention training
 */
void block::train() {
#ifdef HAS_CL
    // run partial forprop in parallel with OpenCL
#elif HAS_CUDA
    // run partial forprop in parallel with CUDA
#endif
}
