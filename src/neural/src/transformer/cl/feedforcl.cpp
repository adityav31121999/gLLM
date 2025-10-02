#ifdef USE_OPENCL
#include "include/transformer.hpp"
#include <stdexcept>
#include <iostream>

/**
 * @brief OpenCL forward propagation for transformers.
 *        Mirrors the logic of cuForward using OpenCL kernels and functions provided
 *        by the block, attention, and mlp classes. Uses kernelComputePredictionIndex for prediction.
 * @param blockCount current block index (0-based).
 * @param currentTokenCount current total number of tokens processed (in full cntext, not just the current block).
 * @param sequence1Count number of tokens in the current sequence1 segment being processed
 */
void transformer::clForward(int &blockCount, int &currentTokenCount, int &sequence1Count)
{
    try {
        // Step 1: Compute KdotQ matrices
        // Step 2 & 3: Perform forward propagation for the relevant block using its OpenCL method
        if (blockCount == 1) {
            for(int i = 0; i < x; ++i) {
                clParallelKdotQs(sequence1Count, currentTokenCount, blockCount, i, isSelf, inTraining);
            }
            blocks[0].clForprop(d, currentTokenCount, l);
        }
        else {
            if (static_cast<size_t>(blockCount - 1) >= blocks.size()) {
                throw std::logic_error("clForward: Invalid blockCount logic: trying to access EV from non-existent previous block index " + std::to_string(blockCount - 1));
            }
            for(int i = 0; i < x; ++i) {
                clParallelKdotQs(sequence1Count, currentTokenCount, blockCount, i, isSelf, inTraining);
            }
            blocks[blockCount-1].clForprop(blocks[blockCount - 1].EV, d, currentTokenCount, blockCount, l, n);
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Standard Exception in transformer::clForward: " << e.what() << std::endl;
        throw;
    }
}

#endif // USE_OPENCL