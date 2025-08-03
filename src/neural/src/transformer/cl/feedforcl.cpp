
#ifdef USE_OPENCL
#include "include/transformer.hpp"
#include <vector>
#include <numeric>
#include <stdexcept>
#include <iostream>
#include <limits>
#include <string>

/**
 * @brief OpenCL forward propagation for transformers.
 *        Mirrors the logic of cuForward using OpenCL kernels and functions provided
 *        by the block, attention, and mlp classes. Uses kernelComputePredictionIndex for prediction.
 * @param blockCount current block index (0-based).
 * @param currentTokenCount current total number of tokens processed. (in full cntext, not just the current block).
 * @param promptCount number of tokens in the current prompt segment being processed (unused in this specific logic but kept for signature consistency).
 */
void transformer::clForward(int &blockCount, int &currentTokenCount, int &promptCount)
{
    // --- Basic Validation ---
    if (blockCount < 0 || blockCount >= m) {
        throw std::out_of_range("clForward: blockCount (" + std::to_string(blockCount) + ") is out of range [0, " + std::to_string(m - 1) + "].");
    }
    if (t.empty() || static_cast<size_t>(blockCount) >= t.size()) {
        throw std::runtime_error("clForward: Transformer blocks not initialized or blockCount exceeds allocated blocks.");
    }
    if (t[blockCount].b.empty() || t[blockCount].b[0].empty()) {
        throw std::runtime_error("clForward: Attention heads not initialized for block " + std::to_string(blockCount) + ".");
    }
    if (embeddings.row <= 0 || embeddings.col <= 0 || embeddings.mapped_data == nullptr || vocabsize <= 0) {
        throw std::runtime_error("transformer::forward: Embeddings not loaded/initialized or vocabsize is zero/negative.");
    }
    if (d <= 0 || x <= 0 || y <= 0 || l <= 0) {
        throw std::runtime_error("clForward: Transformer dimensions (d, x, y, l) are not valid.");
    }

    // Ensure output token vector is sized correctly on the host
    if (otok.size() != static_cast<size_t>(d)) {
        otok.resize(d, 0.0f);
        std::cout << "clForward: Resized host otok vector to size " << d * x << std::endl;
    }
    else {
        // Reset host accumulator before accumulating new values
        std::fill(otok.begin(), otok.end(), 0.0f);
    }

    cl_int cl_err; // For OpenCL error codes

    try {
        // Step 1: Compute KdotQ matrices
        // Step 2 & 3: Perform forward propagation for the relevant block using its OpenCL method
        if (blockCount == 1) {
            std::cout << "-> clForward: Executing clForprop for Block 1\n";
            // t[0].deserialise(t[0].blockFilePath);
            for(int i = 0; i < x; ++i) {
                clParallelKdotQs(promptCount, currentTokenCount, blockCount, i, isSelf, inTraining);
            }
            // std::cout << "clForward: Scaled Dot Products Calculated" << std::endl;
            t[0].clForprop(d, currentTokenCount, l);
            // std::cout << "clForward: Block 1 clForprop finished." << std::endl;
        }
        else {
            if (static_cast<size_t>(blockCount - 1) >= t.size()) {
                throw std::logic_error("clForward: Invalid blockCount logic: trying to access EV from non-existent previous block index " + std::to_string(blockCount - 1));
            }
            std::cout << "-> clForward: Executing clForprop for Block " << blockCount << " using EV from Block " << blockCount - 1 << ":- ";
            // t[blockCount-1].deserialise(t[blockCount-1].blockFilePath);
            for(int i = 0; i < x; ++i) {
                clParallelKdotQs(promptCount, currentTokenCount, blockCount, i, isSelf, inTraining);
            }
            // std::cout << "clForward: Scaled Dot Products Calculated" << std::endl;
            t[blockCount-1].clForprop(t[blockCount - 1].EV, d, currentTokenCount, blockCount, l, n);
            // std::cout << "clForward: Block " << blockCount << " clForprop finished." << std::endl;
        }

        // Step 3.5: Accumulate EH from the last column of the current block (HOST-SIDE)
        // std::cout << "clForward: Concatanating EH from last column (y-1=" << y-1 << ") of Block " << blockCount << " on host..." << std::endl;
        if (y > 0) {
            for (int j = 0; j < x; ++j) {
                const std::vector<float>& eh_vector = t[blockCount-1].b[j][y - 1].EH;
                if (eh_vector.size() != static_cast<size_t>(d)) {
                    throw std::runtime_error("clForward: EH vector size mismatch during host accumulation for head ["
                                             + std::to_string(j) + "][" + std::to_string(y - 1) + "]. Expected "
                                             + std::to_string(d) + ", got " + std::to_string(eh_vector.size()));
                }
                for (int k = 0; k < EMBEDDING; ++k) {
                    otok[k] += eh_vector[k];
                }
            }
        }
        else {
            std::cerr << "Warning: clForward called with y=0 columns. Cannot accumulate EH." << std::endl;
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Standard Exception in transformer::clForward: " << e.what() << std::endl;
        throw;
    }
}

#endif // USE_OPENCL
