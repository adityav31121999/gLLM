
#ifdef USE_OPENCL

#include "include/mlp.hpp"
#include "include/attention.hpp"
#include "include/block.hpp"
#include "include/transformer.hpp"

#include <CL/cl.hpp> // Use cl.hpp for C++ bindings
#include <vector>
#include <stdexcept> // For exceptions
#include <string>    // For std::to_string
#include <iostream>  // For std::cerr

/**
 * @brief OpenCL backward propagation from the last block (m-1) down to the first block (0).
 *        Uses a single common expected horizontal error vector for the last block.
 * @param expectedH Expected horizontal embedding for the last block's output.
 */
void transformer::clBackward(std::vector<float>& expectedH) {
    if (m <= 0) {
        std::cerr << "Warning: clBackward(vector<float>) called with no blocks (m=" << m << ")." << std::endl;
        return;
    }
    if (expectedH.size() != EMBEDDING) {
        throw std::runtime_error("clBackward(vector<float>): expectedH size (" + std::to_string(expectedH.size()) +
                                 ") does not match EMBEDDING (" + std::to_string(EMBEDDING) + ").");
    }

    // Start backprop from the last block (m-1)
    int start_block_index = this->blockCount - 1; // 0-based index

    try {
        // --- Starting Block (m-1) ---
        // Receives the external horizontal error signal 'expectedH'.
        // If m=1, this is also the first block.
        if (start_block_index == 0) { // Only one block (m=1)
            t[0].tokenCount = this->currentTokenCount;
            std::cout << "-> clBackward: First block Backprop" << std::endl;
            // Use the '1stBlock' function as it's the first and only block
            t[0].clBackward1stBlock(expectedH, this->d, this->l);
            // t[0].serialise(t[0].blockFilePath); // Save the state of the first block
        }
        else {
            std::cout << "-> clBackward: Specific block Backprop" << std::endl;
            t[start_block_index].tokenCount = this->currentTokenCount;
            // Use the general 'clBackward' for non-first blocks
            t[start_block_index].clBackward(expectedH, start_block_index, this->d, this->l);
            // t[start_block_index].serialise(t[start_block_index].blockFilePath); // Serialize the block after backward pass

            // --- Intermediate Blocks (m-2 down to 1) ---
            // Propagate vertical error (EV) from the next block.
            for (int i = start_block_index - 1; i >= 1; --i) {
                std::cout << "-> clBackward: ith block Backprop" << std::endl;
                // Block 'i' receives EV from block 'i+1'.
                // Ensure EV from the next block is correctly sized before passing
                if (t[i + 1].EV.size() != static_cast<size_t>(this->x) ||
                    (!t[i + 1].EV.empty() && t[i + 1].EV[0].size() != static_cast<size_t>(this->y))) 
                {
                    throw std::runtime_error("clBackward(vector<float>): EV dimensions from block " + std::to_string(i + 1) + " are incorrect.");
                }
                t[i].clBackward(t[i + 1].EV, i, this->d, this->l);
                // t[i].serialise(t[i].blockFilePath); // Serialize the block after backward pass
            }

            // --- First Block (0) ---
            // Only if there's more than one block (start_block_index > 0).
            // Receives EV from block 1. Uses the special '1stBlock' function.
            if (start_block_index > 0) {
                std::cout << "-> clBackward: First block Backprop" << std::endl;
                if (t[1].EV.size() != static_cast<size_t>(this->x) ||
                    (!t[1].EV.empty() && t[1].EV[0].size() != static_cast<size_t>(this->y))) {
                    throw std::runtime_error("clBackward(vector<float>): EV dimensions from block 1 are incorrect.");
                }
                t[0].clBackward1stBlock(t[1].EV, this->d, this->l);
                // t[0].serialise(t[0].blockFilePath); // Serialize the first block after backward pass
            }
            std::cout << "clBackward Performed." << std::endl;
        }
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Exception during transformer::clBackward(vector<float>): " + std::string(e.what()));
    }
}

/**
 * @brief OpenCL backward propagation from a specific block 'k' down to the first block (0).
 *        Uses a single common expected horizontal error vector for block 'k-1'.
 * @param expectedH Expected horizontal embedding for block 'k-1's output.
 * @param k The block number (1-based index) to start backpropagation from.
 */
void transformer::clBackward(std::vector<float>& expectedH, int& k) {
    if (k <= 0 || k > m) {
        throw std::out_of_range("clBackward(vector<float>, k): Block index k=" + std::to_string(k) + " is out of range [1, " + std::to_string(m) + "].");
    }
    if (expectedH.size() != EMBEDDING) {
        throw std::runtime_error("clBackward(vector<float>, k): expectedH size (" + std::to_string(expectedH.size()) +
                                 ") does not match EMBEDDING (" + std::to_string(EMBEDDING) + ").");
    }

    if(k == m) {
        // use this function
        clBackward(expectedH);
        return;
    }

    int start_block_index = k - 1; // 0-based index

    try {
        // --- Starting Block (k-1) ---
        // Receives the external horizontal error signal 'expectedH'.
        // If k=1, this is the first block.
        if (start_block_index == 0) { // Starting from the first block (k=1)
            std::cout << "-> clBackward: First block Backprop" << std::endl;
            t[0].tokenCount = this->currentTokenCount;
            t[0].clBackward1stBlock(expectedH, this->d, this->l);
            // t[0].serialise(t[0].blockFilePath); // Save the state of the first block
        }
        else {
            std::cout << "-> clBackward: Specific block Backprop" << std::endl;
            t[start_block_index].tokenCount = this->currentTokenCount;
            t[start_block_index].clBackward(expectedH, start_block_index, this->d, this->l);
            t[start_block_index].serialise(t[start_block_index].blockFilePath); // Serialize the block after backward pass
            
            // --- Intermediate Blocks (k-2 down to 1) ---
            // Propagate vertical error (EV) from the next block.
            for (int i = start_block_index - 1; i >= 1; --i) {
                std::cout << "-> clBackward: ith block Backprop" << std::endl;
                // Block 'i' receives EV from block 'i+1'.
                if (t[i + 1].EV.size() != static_cast<size_t>(this->x) ||
                    (!t[i + 1].EV.empty() && t[i + 1].EV[0].size() != static_cast<size_t>(this->y))) {
                    throw std::runtime_error("clBackward(vector<float>, k): EV dimensions from block " + std::to_string(i + 1) + " are incorrect.");
                }
                t[i].clBackward(t[i + 1].EV, i, this->d, this->l);
                // t[i].serialise(t[i].blockFilePath); // Serialize the block after backward pass
            }

            // --- First Block (0) ---
            // Only if k > 1 (i.e., start_block_index > 0).
            // Receives EV from block 1. Uses the special '1stBlock' function.
            if (start_block_index > 0) {
                std::cout << "-> clBackward: First block Backprop" << std::endl;
                if (t[1].EV.size() != static_cast<size_t>(this->x) ||
                    (!t[1].EV.empty() && t[1].EV[0].size() != static_cast<size_t>(this->y))) {
                    throw std::runtime_error("clBackward(vector<float>, k): EV dimensions from block 1 are incorrect.");
                }
                t[0].clBackward1stBlock(t[1].EV, this->d, this->l);
                // t[0].serialise(t[0].blockFilePath); // Serialize the first block after backward pass
            }
        }
    } 
    catch (const std::exception& e) {
        throw std::runtime_error("Exception during transformer::clBackward(vector<float>, k=" + std::to_string(k) + "): " + std::string(e.what()));
    }
}

#endif // USE_OPENCL
