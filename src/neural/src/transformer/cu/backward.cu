
#include "include/transformer.hpp"
#include "include/block.hpp"
#include "include/attention.hpp" // Include necessary headers
#include "include/mlp.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <string> // For std::to_string

// Helper macro for CUDA error checking
#ifndef CUDA_CHECK
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        throw std::runtime_error(cudaGetErrorString(err)); \
    } \
} while (0)
#endif


/**
 * @brief CUDA backward propagation from the last block (m-1) down to the first block (0).
 *        Uses a single common expected horizontal error vector for the last block.
 * @param expectedH Expected horizontal embedding for the last block's output.
 */
void transformer::cuBackward(std::vector<float>& expectedH) {
    if (m <= 0) {
        std::cerr << "Warning: cuBackward called with no blocks (m=" << m << ")." << std::endl;
        return;
    }
    // Start backprop from the last block (m)
    int start_block_index = m - 1; // 0-based index

    try {
        // --- Starting Block (m-1) ---
        // Receives the external horizontal error signal 'expectedH'.
        // If m=1, this is also the first block.
        if (start_block_index == 0) { // Only one block (m=1)
            t[0].cuBackward1stBlock(expectedH, this->d, this->l);
        } else {
            t[start_block_index].cuBackward(expectedH, this->d, this->l);
        }

        // --- Intermediate Blocks (m-2 down to 1) ---
        // Propagate vertical error (EV) from the next block.
        for (int i = start_block_index - 1; i >= 1; --i) {
            // Block 'i' receives EV from block 'i+1'.
            t[i].cuBackward(t[i + 1].EV, this->d, this->l);
        }

        // --- First Block (0) ---
        // Only if there's more than one block (start_block_index > 0).
        // Receives EV from block 1. Uses the special '1stBlock' function.
        if (start_block_index > 0) {
            t[0].cuBackward1stBlock(t[1].EV, this->d, this->l);
        }
    } catch (const std::exception& e) {
        throw std::runtime_error("Exception during transformer::cuBackward(vector<float>): " + std::string(e.what()));
    }
}

/**
 * @brief CUDA backward propagation from a specific block 'k' down to the first block (0).
 *        Uses a single common expected horizontal error vector for block 'k-1'.
 * @param expectedH Expected horizontal embedding for block 'k-1's output.
 * @param k The block number (1-based index) to start backpropagation from.
 */
void transformer::cuBackward(std::vector<float>& expectedH, int& k) {
    if (k <= 0 || k > m) {
        throw std::out_of_range("cuBackward(vector<float>, k): Block index k=" + std::to_string(k) + " is out of range [1, " + std::to_string(m) + "].");
    }

    int start_block_index = k - 1; // 0-based index

    try {
        // --- Starting Block (k-1) ---
        // Receives the external horizontal error signal 'expectedH'.
        // If k=1, this is the first block.
        if (start_block_index == 0) { // Starting from the first block (k=1)
            t[0].cuBackward1stBlock(expectedH, this->d, this->l);
        } else {
            t[start_block_index].cuBackward(expectedH, this->d, this->l);
        }

        // --- Intermediate Blocks (k-2 down to 1) ---
        // Propagate vertical error (EV) from the next block.
        for (int i = start_block_index - 1; i >= 1; --i) {
            // Block 'i' receives EV from block 'i+1'.
            t[i].cuBackward(t[i + 1].EV, this->d, this->l);
        }

        // --- First Block (0) ---
        // Only if k > 1 (i.e., start_block_index > 0).
        // Receives EV from block 1. Uses the special '1stBlock' function.
        if (start_block_index > 0) {
            t[0].cuBackward1stBlock(t[1].EV, this->d, this->l);
        }
    } catch (const std::exception& e) {
         throw std::runtime_error("Exception during transformer::cuBackward(vector<float>, k=" + std::to_string(k) + "): " + std::string(e.what()));
    }
}

/**
 * @brief CUDA backward propagation from the last block (m-1) down to the first block (0).
 *        Uses distinct expected horizontal error vectors (one per row/parallel) for the last block.
 * @param expectedH Vector of expected horizontal embeddings for the last block's output (shape [x][EMBEDDING]).
 */
void transformer::cuBackward(std::vector<std::vector<float>>& expectedH) {
     if (m <= 0) {
        std::cerr << "Warning: cuBackward called with no blocks (m=" << m << ")." << std::endl;
        return;
    }
    // Validate expectedH shape for the last block
    // The block::cuBackward function expects [y][EMBEDDING] if it's a vector<vector<float>>
    // The CPU code passes [x][EMBEDDING]. Let's assume the block function handles [x][EMBEDDING].
    // **Correction:** Looking at `block::cuBackward(vector<vector<float>>...)`, it expects `[y][EMBEDDING]`.
    // The CPU code `transformer::backward(vector<vector<float>>)` passes `expectedH[i]` to `partialbackward`, implying `expectedH` is `[x][EMBEDDING]`.
    // There's an inconsistency. Assuming the intent is to pass `[x][EMBEDDING]` and the block function should handle it.
    // **Revisiting block::cuBackward:** It iterates `j` from `y-1` to 0 and calls `cu1ParallelBackward(expectedH[j], ...)`. This confirms it expects `[y][EMBEDDING]`.
    // **Conclusion:** The CPU `transformer::backward(vector<vector<float>>)` logic seems incompatible with the CUDA `block::cuBackward(vector<vector<float>>)` signature/implementation.
    // Let's implement based on the CUDA block signature, assuming `expectedH` is `[y][EMBEDDING]`.
     if (expectedH.size() != static_cast<size_t>(this->y)) {
          throw std::runtime_error("cuBackward(vector<vector<float>>): Outer dimension of expectedH (" + std::to_string(expectedH.size()) + ") does not match number of columns y (" + std::to_string(this->y) + ").");
     }
     if (!expectedH.empty() && expectedH[0].size() != EMBEDDING) {
          throw std::runtime_error("cuBackward(vector<vector<float>>): Inner dimension of expectedH (" + std::to_string(expectedH[0].size()) + ") does not match EMBEDDING (" + std::to_string(EMBEDDING) + ").");
     }

    int start_block_index = m - 1; // 0-based index

    try {
        // --- Starting Block (m-1) ---
        // Receives the external horizontal error signal 'expectedH'.
        // If m=1, this is also the first block.
        if (start_block_index == 0) { // Only one block (m=1)
            t[0].cuBackward1stBlock(expectedH, this->d, this->l);
        } else {
            t[start_block_index].cuBackward(expectedH, this->d, this->l);
        }

        // --- Intermediate Blocks (m-2 down to 1) ---
        // Propagate vertical error (EV) from the next block.
        for (int i = start_block_index - 1; i >= 1; --i) {
            // Block 'i' receives EV from block 'i+1'.
            t[i].cuBackward(t[i + 1].EV, this->d, this->l);
        }

        // --- First Block (0) ---
        // Only if there's more than one block (start_block_index > 0).
        // Receives EV from block 1. Uses the special '1stBlock' function.
        if (start_block_index > 0) {
            t[0].cuBackward1stBlock(t[1].EV, this->d, this->l);
        }
    } catch (const std::exception& e) {
        throw std::runtime_error("Exception during transformer::cuBackward(vector<vector<float>>): " + std::string(e.what()));
    }
}

/**
 * @brief CUDA backward propagation from a specific block 'k' down to the first block (0).
 *        Uses distinct expected horizontal error vectors (one per column/parallel) for block 'k-1'.
 * @param expectedH Vector of expected horizontal embeddings for block 'k-1's output (shape [y][EMBEDDING]).
 * @param k The block number (1-based index) to start backpropagation from.
 */
void transformer::cuBackward(std::vector<std::vector<float>>& expectedH, int& k) {
    if (k <= 0 || k > m) {
        throw std::out_of_range("cuBackward(vector<vector<float>>, k): Block index k=" + std::to_string(k) + " is out of range [1, " + std::to_string(m) + "].");
    }
    // Validate expectedH shape for the starting block (assuming [y][EMBEDDING])
    if (expectedH.size() != static_cast<size_t>(this->y)) {
         throw std::runtime_error("cuBackward(vector<vector<float>>, k): Outer dimension of expectedH (" + std::to_string(expectedH.size()) + ") does not match number of columns y (" + std::to_string(this->y) + ").");
    }
    if (!expectedH.empty() && expectedH[0].size() != EMBEDDING) {
         throw std::runtime_error("cuBackward(vector<vector<float>>, k): Inner dimension of expectedH (" + std::to_string(expectedH[0].size()) + ") does not match EMBEDDING (" + std::to_string(EMBEDDING) + ").");
    }

    int start_block_index = k - 1; // 0-based index

    try {
        // --- Starting Block (k-1) ---
        // Receives the external horizontal error signal 'expectedH'.
        // If k=1, this is the first block.
        if (start_block_index == 0) { // Starting from the first block (k=1)
            t[0].cuBackward1stBlock(expectedH, this->d, this->l);
        } else {
            t[start_block_index].cuBackward(expectedH, this->d, this->l);
        }

        // --- Intermediate Blocks (k-2 down to 1) ---
        // Propagate vertical error (EV) from the next block.
        for (int i = start_block_index - 1; i >= 1; --i) {
            // Block 'i' receives EV from block 'i+1'.
            t[i].cuBackward(t[i + 1].EV, this->d, this->l);
        }

        // --- First Block (0) ---
        // Only if k > 1 (i.e., start_block_index > 0).
        // Receives EV from block 1. Uses the special '1stBlock' function.
        if (start_block_index > 0) {
            t[0].cuBackward1stBlock(t[1].EV, this->d, this->l);
        }
    } catch (const std::exception& e) {
         throw std::runtime_error("Exception during transformer::cuBackward(vector<vector<float>>, k=" + std::to_string(k) + "): " + std::string(e.what()));
    }
}


// --- Implementations for expectedV ---
// These follow the same pattern: start at block k-1 (or m-1) with expectedV,
// then propagate EV from block i+1 to block i for i < start_block_index.

/**
 * @brief CUDA backward propagation from the last block (m-1) down to the first block (0).
 *        Driven by expected vertical error vectors for the last block.
 * @param expectedV Expected vertical embeddings for the last block (shape [x][y][CONTEXT_WIN][EMBEDDING]).
void transformer::cuBackward(std::vector<std::vector<std::vector<std::vector<float>>>>& expectedV) {
    if (m <= 0) {
        std::cerr << "Warning: cuBackward called with no blocks (m=" << m << ")." << std::endl;
        return;
    }
    // Basic validation for expectedV shape for the last block
    if (expectedV.size() != static_cast<size_t>(this->x) ||
        (!expectedV.empty() && expectedV[0].size() != static_cast<size_t>(this->y))) {
         throw std::runtime_error("cuBackward(4D vector): expectedV dimensions mismatch for last block. Expected ["
                                  + std::to_string(this->x) + "][" + std::to_string(this->y) + "][...][...]");
    }

    int start_block_index = m - 1; // 0-based index

    try {
        // --- Starting Block (m-1) ---
        // Receives the external vertical error signal 'expectedV'.
        // If m=1, this is also the first block.
        if (start_block_index == 0) { // Only one block (m=1)
            t[0].cuBackward1stBlock(expectedV, this->d, this->l);
        } else {
            t[start_block_index].cuBackward(expectedV, this->d, this->l);
        }

        // --- Intermediate Blocks (m-2 down to 1) ---
        // Propagate vertical error (EV) from the next block.
        for (int i = start_block_index - 1; i >= 1; --i) {
            // Block 'i' receives EV from block 'i+1'.
            t[i].cuBackward(t[i + 1].EV, this->d, this->l);
        }

        // --- First Block (0) ---
        // Only if there's more than one block (start_block_index > 0).
        // Receives EV from block 1. Uses the special '1stBlock' function.
        if (start_block_index > 0) {
            t[0].cuBackward1stBlock(t[1].EV, this->d, this->l);
        }
    } catch (const std::exception& e) {
        throw std::runtime_error("Exception during transformer::cuBackward(4D vector): " + std::string(e.what()));
    }
}

/**
 * @brief CUDA backward propagation from a specific block 'k' down to the first block (0).
 *        Driven by expected vertical error vectors for block 'k-1'.
 * @param expectedV Expected vertical embeddings for block 'k-1' (shape [x][y][CONTEXT_WIN][EMBEDDING]).
 * @param k The block number (1-based index) to start backpropagation from.
void transformer::cuBackward(std::vector<std::vector<std::vector<std::vector<float>>>>& expectedV, int& k) {
    if (k <= 0 || k > m) {
        throw std::out_of_range("cuBackward(4D vector, k): Block index k=" + std::to_string(k) + " is out of range [1, " + std::to_string(m) + "].");
    }
    // Basic validation for expectedV shape for the starting block
    if (expectedV.size() != static_cast<size_t>(this->x) ||
        (!expectedV.empty() && expectedV[0].size() != static_cast<size_t>(this->y))) {
         throw std::runtime_error("cuBackward(4D vector, k): expectedV dimensions mismatch for starting block. Expected ["
                                  + std::to_string(this->x) + "][" + std::to_string(this->y) + "][...][...]");
    }

    int start_block_index = k - 1; // 0-based index

    try {
        // --- Starting Block (k-1) ---
        // Receives the external vertical error signal 'expectedV'.
        // If k=1, this is the first block.
        if (start_block_index == 0) { // Starting from the first block (k=1)
            t[0].cuBackward1stBlock(expectedV, this->d, this->l);
        } else {
            t[start_block_index].cuBackward(expectedV, this->d, this->l);
        }

        // --- Intermediate Blocks (k-2 down to 1) ---
        // Propagate vertical error (EV) from the next block.
        for (int i = start_block_index - 1; i >= 1; --i) {
            // Block 'i' receives EV from block 'i+1'.
            t[i].cuBackward(t[i + 1].EV, this->d, this->l);
        }

        // --- First Block (0) ---
        // Only if k > 1 (i.e., start_block_index > 0).
        // Receives EV from block 1. Uses the special '1stBlock' function.
        if (start_block_index > 0) {
            t[0].cuBackward1stBlock(t[1].EV, this->d, this->l);
        }
    } catch (const std::exception& e) {
         throw std::runtime_error("Exception during transformer::cuBackward(4D vector, k=" + std::to_string(k) + "): " + std::string(e.what()));
    }
}
*/