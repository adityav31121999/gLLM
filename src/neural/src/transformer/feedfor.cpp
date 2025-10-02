#ifdef USE_CPU

#include "include/attention.hpp"
#include "include/block.hpp"
#include "include/transformer.hpp"

/**
 * @brief forward propagation for transformers
 * @param blockCount current block index
 * @param currentTokenCount current number of tokens
 * @param sequence1Count number of tokens in sequence1
 */
void transformer::forward(int& blockCount, int& currentTokenCount, int& sequence1Count)
{
    if (blockCount < 0 || blockCount >= m) {
        throw std::out_of_range("transformer::forward: blockCount (" + std::to_string(blockCount) + ") is out of range [0, " + std::to_string(m - 1) + "].");
    }
    if (t.empty() || static_cast<size_t>(blockCount) >= blocks.size()) {
        throw std::runtime_error("transformer::forward: Transformer blocks not initialized or blockCount exceeds allocated blocks.");
    }
    if (blockCount < blocks.size() && (t[blockCount].b.empty() || blocks[blockCount].b[0].empty())) {
        throw std::runtime_error("transformer::forward: Attention heads not initialized for block " + std::to_string(blockCount) + ".");
    }

    if (embeddings.row <= 0 || embeddings.col <= 0 || embeddings.mapped_data == nullptr || vocabsize <= 0) {
        throw std::runtime_error("transformer::forward: Embeddings not loaded/initialized or vocabsize is zero/negative.");
    }
    if (d <= 0 || x <= 0 || y <= 0 || l <= 0) {
        throw std::runtime_error("transformer::forward: Transformer dimensions (d, x, y, l) are not valid (must be positive).");
    }

    if (otok.size() != static_cast<size_t>(d)) {
        otok.resize(d, 0.0f);
    } 
    else {
        std::fill(otok.begin(), otok.end(), 0.0f);
    }

    // compute the KdotQ
    computeKdotQs(sequence1Count, currentTokenCount, blockCount, isSelf, inTraining);
    if(blockCount == 0) {
        blocks[0].forprop(d, currentTokenCount, l, 1);
        for(int i = 0; i < d; i++) {
            for(int j = 0; j < x; j++) {
                otok[i] += blocks[0].b[j][y-1].EH[i];
            }  
        }
        computeOutput(otok, tokenEmbed, vocabsize, indexForToken);
    }
    else {
        if (blockCount > 0 && static_cast<size_t>(blockCount-1) < blocks.size() && (blockCount == 1 || static_cast<size_t>(blockCount-2) < blocks.size())) {
            blocks[blockCount-1].forprop((blockCount > 1 ? blocks[blockCount-2].EV : EVuse), d, currentTokenCount, blockCount, l, n);
        } 
        else if (blockCount < 0 || blockCount > m) {
            throw std::runtime_error("transformer::forward: Invalid block indices for forprop in else branch.");
        }
        for(int i = 0; i < d; i++) {
            for(int j = 0; j < x; j++) {
                otok[i] += blocks[0].b[j][y-1].EH[i];
            }
        }
        computeOutput(otok, embeddings, vocabsize, indexForToken);
    }
}

#endif