
#include "include/attention.hpp"
#include "include/block.hpp"

#ifdef USE_CPU

/**
 * @brief forward propagation for partial attention (layer of block)
 * @param in embedding dimensions
 * @param tokenCount current token count
 * @param i ith layer of block
 * @param layers number of layers of MLP
 */
void block::partialforprop(int& in, int& tokenCount, int i, int& layers) 
{
    // initialize the horizontal embedding vector to 0
    for (int j = 0; j < y; j++) {
        b[i][j].EH = std::vector<float>(in, 0.0f);
    }

    // for one partial attention
    for(int j = 0; j < y; j++) {
        b[i][j].forprop(in, layers, tokenCount);      // incomplete attention forprop
        // break when last head forprop is done
        if(j == (y - 1))
            break;
        b[i][j + 1].EH = b[i][j].EH;
    }
}


/**
 * @brief forward propagation for partial attention (ith layer of kth block)
 * @param EVp EVs of previous blocks ith layer
 * @param in embedding dimensions
 * @param tokenCount current token count
 * @param k current block number
 * @param i ith layer of current block
 * @param layers number of layers of MLP
 * @param n number of tokens for each head (context window)
 */
void block::partialforprop(std::vector<std::vector<std::vector<float>>>& EVp_layer_from_prev_block, int& in, int& tokenCount, int& k, int current_block_layer_idx,
    int& layers, int& n)
{
    // initialize the horizontal embedding vector to 0
    for (int j = 0; j < y; j++) {
        b[current_block_layer_idx][j].EH = std::vector<float>(in, 0.0f);
    }
    if (EVp_layer_from_prev_block.size() != static_cast<size_t>(y)) {
        throw std::runtime_error("EVp_layer_from_prev_block head count mismatch in block::partialforprop. Expected " +
                                 std::to_string(y) + ", got " + std::to_string(EVp_layer_from_prev_block.size()));
    }

    // for one partial attention
    for(int j = 0; j < y; j++) {
        // Create a temporary mat from EVp_layer_from_prev_block[j]
        // EVp_layer_from_prev_block[j] is std::vector<std::vector<float>> [token_idx][embedding_dim]
        // It should have totalTokenCount rows (passed as 'tokenCount' to this function).
        const auto& evp_head_data_2d = EVp_layer_from_prev_block[j];
        if (evp_head_data_2d.empty() && tokenCount > 0) { // tokenCount is totalTokenCount
             throw std::runtime_error("EVp head data is empty but totalTokenCount > 0 for head [" + std::to_string(current_block_layer_idx) + "][" + std::to_string(j) + "]");
        }
        int evp_rows = evp_head_data_2d.size();
        int evp_cols = evp_rows > 0 ? evp_head_data_2d[0].size() : 0;

        // The mat passed to attention::forprop should have EVp.row = totalTokenCount and EVp.col = EMBEDDING
        mat temp_EVp_mat(tokenCount, EMBEDDING); // Expecting totalTokenCount rows
        if (evp_rows == tokenCount && evp_cols == EMBEDDING) { // Check if provided data matches expectation
            for(int r = 0; r < evp_rows; ++r) {
                for(int c = 0; c < evp_cols; ++c) {
                    temp_EVp_mat(r,c) = evp_head_data_2d[r][c];
                }
            }
        } else if (tokenCount > 0) { // If dimensions don't match but we expected data, it's an issue.
            throw std::runtime_error("Dimension mismatch for EVp data for head [" + std::to_string(current_block_layer_idx) + "][" + std::to_string(j) + "]. Expected " + std::to_string(tokenCount) + "x" + std::to_string(EMBEDDING) + ", got " + std::to_string(evp_rows) + "x" + std::to_string(evp_cols));
        }
        // If tokenCount is 0, temp_EVp_mat will be 0xEMBEDDING, which is fine.

        b[current_block_layer_idx][j].forprop(temp_EVp_mat, in, layers, tokenCount, k, n);      // incomplete attention forprop
        // break when last head forprop is done
        if(j == (y - 1))
            break;
        b[current_block_layer_idx][j + 1].EH = b[current_block_layer_idx][j].EH;
    }
}


/**
 * @brief forward propagation for complete attention (for 1st block only)
 * @param in dimension of embedding
 * @param tokenCount current token count
 * @param layers layers of mlp
 */
void block::forprop(int& in, int& tokenCount, int& layers, int blockCount) 
{
    // deserialise(blockFilePath);
    // y partial attention in x layers => x parallel processes
    for(int i = 0; i < x; i++) {
        // for all layers
        partialforprop(in, tokenCount, i, layers);
    }
}


/**
 * @brief forward propagation for complete attention (for kth block)
 * @param EVp previous block context retention vectors
 * @param in dimension of embedding
 * @param currentTokenCount current token count
 * @param layers layers of mlp
 * @param n context window
 * @param blockCount current block position in full context
 */
void block::forprop(std::vector<std::vector<std::vector<std::vector<float>>>>& EVp, int& in, int& tokenCount, int& blockCount, int& layers, 
                int& n)
{
    // deserialise(blockFilePath);
    // y partial attention in x layers => x parallel processes
    // int tokenCount = std::abs(currentTokenCount - (n * k));
    // forward propagation for all layers
    for(int i = 0; i < x; i++) {
        partialforprop(EVp[i], in, tokenCount, blockCount, i, layers, n); // EVp[i] is the data for layer i of previous block
    }
}

#endif
