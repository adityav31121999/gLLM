#ifdef USE_OPENCL
#include "include/transformer.hpp"
#include <vector>
#include <stdexcept>
#include <iostream>
#include <string>

/**
 * @brief CUDA backward propagation from the last block (m-1) down to the first block (0).
 *        Uses a single common expected horizontal error vector for the last block.
 * @param expectedH Expected horizontal embedding for the last block's output.
 */
void transformer::clBackward(std::vector<float>& expectedH) {
    if (m <= 0) {
        std::cerr << "Warning: cuBackward called with no blocks (m=" << m << ")." << std::endl;
        return;
    }
    // Start backprop from the last block (m)
    int start_block_index = m - 1; // 0-based index
    std::cout << "-> clBackward (single expectedH)" << std::endl;

    try {
        // --- Starting Block (m-1) ---
        // Receives the external horizontal error signal 'expectedH'.
        // If m=1, this is also the first block.
        if (start_block_index == 0) { // Only one block (m=1)
            t[0].tokenCount = this->currentTokenCount;
            t[0].clbackward1stBlock(expectedH, this->d, this->l, learning, lambda_L1, lambda_L2);
        } 
        else {
            t[start_block_index].tokenCount = this->currentTokenCount % CONTEXT_WIN;
            t[start_block_index].clbackward(expectedH, start_block_index, this->d, this->l, learning, lambda_L1, lambda_L2);
        }

        // --- Intermediate Blocks (m-2 down to 1) ---
        for (int i = start_block_index - 1; i >= 1; --i) {
            t[i].tokenCount = CONTEXT_WIN;
            // Block 'i' receives EV from block 'i+1'.
            t[i].clbackward(t[i + 1].EV, i, this->d, this->l, learning, lambda_L1, lambda_L2);
        }

        // --- First Block (0) ---
        if (start_block_index > 0) {
            t[0].tokenCount = CONTEXT_WIN;
            t[0].clbackward1stBlock(t[1].EV, this->d, this->l, learning, lambda_L1, lambda_L2);
        }
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Exception during transformer::clBackward(vector<float>): " + std::string(e.what()));
    }
}

/**
 * @brief CUDA backward propagation from a specific block 'k' down to the first block (0).
 *        Uses a single common expected horizontal error vector for block 'k-1'.
 * @param expectedH Expected horizontal embedding for block 'k-1's output.
 * @param k The block number (1-based index) to start backpropagation from.
 */
void transformer::clBackward(std::vector<float>& expectedH, int& k) {
    if (k <= 0 || k > m) {
        throw std::out_of_range("clBackward(vector<float>, k): Block index k=" + std::to_string(k) + " is out of range [1, " + std::to_string(m) + "].");
    }

    int start_block_index = k - 1; // 0-based index
    std::cout << "-> clBackward (single expectedH, start_block = " << k << ")" << std::endl;

    try {
        // --- Starting Block (k-1) ---
        // Receives the external horizontal error signal 'expectedH'.
        // If k=1, this is the first block.
        if (start_block_index == 0) { // Starting from the first block (k=1)
            t[0].tokenCount = this->currentTokenCount;
            t[0].clbackward1stBlock(expectedH, this->d, this->l, learning, lambda_L1, lambda_L2);
        }
        else {
            t[start_block_index].tokenCount = this->currentTokenCount % CONTEXT_WIN;
            t[start_block_index].clbackward(expectedH, start_block_index, this->d, this->l, learning, lambda_L1, lambda_L2);
        }

        // --- Intermediate Blocks (k-2 down to 1) ---
        // Propagate vertical error (EV) from the next block.
        for (int i = start_block_index - 1; i >= 1; --i) {
            t[i].tokenCount = CONTEXT_WIN;
            // Block 'i' receives EV from block 'i+1'.
            t[i].clbackward(t[i + 1].EV, i,  this->d, this->l, learning, lambda_L1, lambda_L2);
        }

        // --- First Block (0) ---
        if (start_block_index > 0) {
            t[0].tokenCount = CONTEXT_WIN;
            t[0].clbackward1stBlock(t[1].EV, this->d, this->l, learning, lambda_L1, lambda_L2);
        }
    } 
    catch (const std::exception& e) {
         throw std::runtime_error("Exception during transformer::clBackward(vector<float>, k=" + std::to_string(k) + "): " + std::string(e.what()));
    }
}

/**
 * @brief CUDA backward propagation from the last block (m-1) down to the first block (0).
 *        Uses distinct expected horizontal error vectors (one per row/parallel) for the last block.
 * @param expectedH Vector of expected horizontal embeddings for the last block's output (shape [x][EMBEDDING]).
 */
void transformer::clBackward(std::vector<std::vector<float>>& expectedH) {
    if (m <= 0) {
        std::cerr << "Warning: clBackward called with no blocks (m=" << m << ")." << std::endl;
        return;
    }
    if (expectedH.size() != static_cast<size_t>(this->y)) {
        throw std::runtime_error("clBackward(vector<vector<float>>): Outer dimension of expectedH (" + std::to_string(expectedH.size()) + ") does not match number of columns y (" + std::to_string(this->y) + ").");
    }
    if (!expectedH.empty() && expectedH[0].size() != EMBEDDING) {
        throw std::runtime_error("clBackward(vector<vector<float>>): Inner dimension of expectedH (" + std::to_string(expectedH[0].size()) + ") does not match EMBEDDING (" + std::to_string(EMBEDDING) + ").");
    }

    int start_block_index = m - 1; // 0-based index

    try {
        // --- Starting Block (m-1) ---
        // Receives the external horizontal error signal 'expectedH'.
        // If m=1, this is also the first block.
        if (start_block_index == 0) { // Only one block (m=1)
            t[0].tokenCount = this->currentTokenCount;
            t[0].clbackward1stBlock(expectedH, this->d, this->l, learning, lambda_L1, lambda_L2);
        }
        else {
            t[start_block_index].tokenCount = this->currentTokenCount % CONTEXT_WIN;
            t[start_block_index].clbackward(expectedH, start_block_index, this->d, this->l, learning, lambda_L1, lambda_L2);
        }

        // --- Intermediate Blocks (m-2 down to 1) ---
        // Propagate vertical error (EV) from the next block.
        for (int i = start_block_index - 1; i >= 1; --i) {
            t[i].tokenCount = CONTEXT_WIN;
            // Block 'i' receives EV from block 'i+1'.
            t[i].clbackward(t[i + 1].EV, i, this->d, this->l, learning, lambda_L1, lambda_L2);
        }

        // --- First Block (0) ---
        // Only if there's more than one block (start_block_index > 0).
        // Receives EV from block 1. Uses the special '1stBlock' function.
        if (start_block_index > 0) {
            t[0].tokenCount = CONTEXT_WIN;
            t[0].clbackward1stBlock(t[1].EV, this->d, this->l, learning, lambda_L1, lambda_L2);
        }
    } catch (const std::exception& e) {
        throw std::runtime_error("Exception during transformer::clBackward(vector<vector<float>>): " + std::string(e.what()));
    }
}

/**
 * @brief CUDA backward propagation from a specific block 'k' down to the first block (0).
 *        Uses distinct expected horizontal error vectors (one per column/parallel) for block 'k-1'.
 * @param expectedH Vector of expected horizontal embeddings for block 'k-1's output (shape [y][EMBEDDING]).
 * @param k The block number (1-based index) to start backpropagation from.
 */
void transformer::clBackward(std::vector<std::vector<float>>& expectedH, int& k) {
    if (k <= 0 || k > m) {
        throw std::out_of_range("clBackward(vector<vector<float>>, k): Block index k=" + std::to_string(k) + " is out of range [1, " + std::to_string(m) + "].");
    }
    // Validate expectedH shape for the starting block (assuming [y][EMBEDDING])
    if (expectedH.size() != static_cast<size_t>(this->y)) {
         throw std::runtime_error("clBackward(vector<vector<float>>, k): Outer dimension of expectedH (" + std::to_string(expectedH.size()) + ") does not match number of columns y (" + std::to_string(this->y) + ").");
    }
    if (!expectedH.empty() && expectedH[0].size() != EMBEDDING) {
         throw std::runtime_error("clBackward(vector<vector<float>>, k): Inner dimension of expectedH (" + std::to_string(expectedH[0].size()) + ") does not match EMBEDDING (" + std::to_string(EMBEDDING) + ").");
    }

    int start_block_index = k - 1; // 0-based index

    try {
        // --- Starting Block (k-1) ---
        // Receives the external horizontal error signal 'expectedH'.
        // If k=1, this is the first block.
        if (start_block_index == 0) { // Starting from the first block (k=1)
            t[0].tokenCount = this->currentTokenCount;
            t[0].clbackward1stBlock(expectedH, this->d, this->l, learning, lambda_L1, lambda_L2);
        } 
        else {
            t[start_block_index].tokenCount = this->currentTokenCount % CONTEXT_WIN;
            t[start_block_index].clbackward(expectedH, start_block_index, this->d, this->l, learning, lambda_L1, lambda_L2);
        }

        // --- Intermediate Blocks (k-2 down to 1) ---
        // Propagate vertical error (EV) from the next block.
        for (int i = start_block_index - 1; i >= 1; --i) {
            t[i].tokenCount = CONTEXT_WIN;
            // Block 'i' receives EV from block 'i+1'.
            t[i].clbackward(t[i + 1].EV, i, this->d, this->l, learning, lambda_L1, lambda_L2);
        }

        // --- First Block (0) ---
        // Only if k > 1 (i.e., start_block_index > 0).
        // Receives EV from block 1. Uses the special '1stBlock' function.
        if (start_block_index > 0) {
            t[0].tokenCount = CONTEXT_WIN;
            t[0].clbackward1stBlock(t[1].EV, this->d, this->l, learning, lambda_L1, lambda_L2);
        }
    } catch (const std::exception& e) {
         throw std::runtime_error("Exception during transformer::clBackward(vector<vector<float>>, k=" + std::to_string(k) + "): " + std::string(e.what()));
    }
}

#endif