#ifdef USE_CU
#include "include/transformer.hpp"
#include "include/block.hpp"
#include "include/attention.hpp"
#include "include/mlp.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <stdexcept>
#include <iostream>

// Helper macro for CUDA error checking (assuming it's defined elsewhere or define it here)
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        throw std::runtime_error(cudaGetErrorString(err)); \
    } \
} while (0)


/**
 * @brief CUDA forward propagation for transformers
 * @param blockCount current block index (1-based)
 * @param currentTokenCount current total number of tokens processed
 * @param sequence1Count number of tokens in the current sequence1 segment being processed
 */
void transformer::cuForward(int& blockCount, int& currentTokenCount, int& sequence1Count)
{
    try {
        // Step 1: Compute KdotQ matrices
        // Step 2 & 3: Perform forward propagation for the relevant block using its OpenCL method
        if (blockCount == 1) {
            for(int i = 0; i < x; ++i) {
                cuParallelKdotQs(sequence1Count, currentTokenCount, blockCount, i, isSelf, inTraining);
            }
            blocks[0].cuForprop(d, currentTokenCount, l);
        }
        else {
            if (static_cast<size_t>(blockCount - 1) >= blocks.size()) {
                throw std::logic_error("clForward: Invalid blockCount logic: trying to access EV from non-existent previous block index " + std::to_string(blockCount - 1));
            }
            for(int i = 0; i < x; ++i) {
                cuParallelKdotQs(sequence1Count, currentTokenCount, blockCount, i, isSelf, inTraining);
            }
            blocks[blockCount-1].cuForprop(blocks[blockCount - 1].EV, d, currentTokenCount, blockCount, l, n);
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Standard Exception in transformer::clForward: " << e.what() << std::endl;
        throw;
    }
}


/**
 * @brief CUDA forward propagation for transformers
 * @param blockCount current block index (1-based)
 * @param currentTokenCount current total number of tokens processed
 * @param sequence1Count number of tokens in the current sequence1 segment being processed
 */
void transformer::cuForward_ev(int& blockCount, int& currentTokenCount, int& sequence1Count)
{
    try {
        // Step 1: Compute KdotQ matrices
        // Step 2 & 3: Perform forward propagation for the relevant block using its OpenCL method
        if (blockCount == 1) {
            for(int i = 0; i < x; ++i) {
                cuParallelKdotQs(sequence1Count, currentTokenCount, blockCount, i, isSelf, inTraining);
            }
            blocks[0].cuForprop(d, currentTokenCount, l);
        }
        else {
            if (static_cast<size_t>(blockCount - 1) >= blocks.size()) {
                throw std::logic_error("clForward: Invalid blockCount logic: trying to access EV from non-existent previous block index " + std::to_string(blockCount - 1));
            }
            for(int i = 0; i < x; ++i) {
                cuParallelKdotQs(sequence1Count, currentTokenCount, blockCount, i, isSelf, inTraining);
            }
            blocks[blockCount-1].cuForprop(blocks[blockCount - 1].EV, d, currentTokenCount, blockCount, l, n);
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Standard Exception in transformer::clForward: " << e.what() << std::endl;
        throw;
    }
}

#endif