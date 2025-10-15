#ifdef USE_CPU
// backward propagation for transformer
#include "include/attention.hpp"
#include "include/block.hpp"
#include "include/transformer.hpp"


/**
 * @brief backward propagation for kth to first block, Expected EVs are not known
 *          (common expected EH for kth block)
 * @param expected expected token embedding from horizontal pass
 * @param k block number (1-based index)
 */
void transformer::backward(std::vector<float>& expected, int& k) {
    if (k <= 0 || k > m) {
        throw std::out_of_range("clBackward(vector<float>, k): Block index k=" + std::to_string(k) + " is out of range [1, " + std::to_string(m) + "].");
    }

    int start_block_index = k - 1; // 0-based index
    // std::cout << "-> clBackward (H, start_block = " << k << ")" << std::endl;

    try {
        // --- Starting Block (k-1) ---
        // Receives the external horizontal error signal 'expectedH'.
        // If k=1, this is the first block.
        if (start_block_index == 0) { // Starting from the first block (k=1)
            blocks[0].tokenCount = currentTokenCount;
            blocks[0].backward1stBlock(expected, d, l, learning);
        }
        else { // Handles all k > 1
            blocks[start_block_index].tokenCount = currentTokenCount % CONTEXT_WIN;
            blocks[start_block_index].backward(expected, start_block_index, d, l, learning);
        }
    }
    catch (const std::exception& e) {
         throw std::runtime_error("Exception during transformer::backward(vector<float>, k=" + std::to_string(k) + "): " + std::string(e.what()));
    }
}


/**
 * @brief backward propagation for kth to first block, Expected EVs are not known
 *          (distinct expected EH for last block)
 * @param expected expected token embeddings from horizontal pass
 * @param k block number (1-based index)
 */
void transformer::backward(std::vector<std::vector<float>>& expected, int& k) {
    if (k <= 0 || k > m) {
        throw std::out_of_range("clBackward(vector<vector<float>>, k): Block index k=" + std::to_string(k) + " is out of range [1, " + std::to_string(m) + "].");
    }

    int start_block_index = k - 1; // 0-based index
    // std::cout << "-> clBackward (H, start_block = " << k << ")" << std::endl;

    try {
        // --- Starting Block (k-1) ---
        // Receives the external horizontal error signal 'expectedH'.
        // If k=1, this is the first block.
        if (start_block_index == 0) { // Starting from the first block (k=1)
            blocks[0].tokenCount = currentTokenCount;
            blocks[0].backward1stBlock(expected, d, l, learning);
        }
        else { // Handles all k > 1
            blocks[start_block_index].tokenCount = currentTokenCount % CONTEXT_WIN;
            blocks[start_block_index].backward(expected, start_block_index, d, l, learning);
        }
    }
    catch (const std::exception& e) {
         throw std::runtime_error("Exception during transformer::backward(vector<vector<float>>, k=" + std::to_string(k) + "): " + std::string(e.what()));
    }
}


/**
 * @brief backward propagation for kth to first block, Expected EVs are not known
 *          (distinct expected EH for last block)
 * @param expected expected token embeddings from horizontal pass
 * @param k block number (1-based index)
 */
void transformer::backwardContext(std::vector<std::vector<float>>& expected, int& k) {
    if (k <= 0 || k > m) {
        throw std::out_of_range("clBackward(vector<vector<float>>, k): Block index k=" + std::to_string(k) + " is out of range [1, " + std::to_string(m) + "].");
    }

    int start_block_index = k - 1; // 0-based index
    // std::cout << "-> clBackward (H, start_block = " << k << ")" << std::endl;

    try {
        // --- Starting Block (k-1) ---
        // Receives the external horizontal error signal 'expectedH'.
        // If k=1, this is the first block.
        if (start_block_index == 0) { // Starting from the first block (k=1)
            blocks[0].tokenCount = currentTokenCount;
            blocks[0].rbackward1stBlock(expected, d, l, learning);
        }
        else { // Handles all k > 1
            blocks[start_block_index].tokenCount = currentTokenCount % CONTEXT_WIN;
            blocks[start_block_index].backward(expected, start_block_index, d, l, learning);
        }
    }
    catch (const std::exception& e) {
         throw std::runtime_error("Exception during transformer::backward(vector<vector<float>>, k=" + std::to_string(k) + "): " + std::string(e.what()));
    }
}

#endif
