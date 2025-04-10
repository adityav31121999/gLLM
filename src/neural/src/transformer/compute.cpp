
// compute functions
#include "include/block.hpp"
#include "include/transformer.hpp"


/**
 * @brief compute key or query of token using token embedding and matrix for keys and queries
 * @param[in] tokenEmbed token embedding
 * @param[in] m matrix for keys or queries
 * @param[out] KorQ Key or query vector to be calculated and stored
 */
void computeKQ(std::vector<float>& tokenEmmbed, mat& m, std::vector<float>& KorQ) {
    for(int i = 0; i < tokenEmmbed.size(); i++) {
        KorQ[i] = std::inner_product(tokenEmmbed.begin(), tokenEmmbed.end(), m.a[i].begin(), 0.0f);
    }
}

/**
 * @brief Dot product of Ti x M x Tj' for use
 * @param[in] Ti ith token
 * @param[in] M QK' cache
 * @param[in] Tj jth token (count as transpose) / transpose of vertical retention vector
 * @param[out] dot = Ti x M x Tj'
 */
void computeDot(std::vector<float>& Ti, mat M, std::vector<float>& Tj, float& dot) {
    std::vector<float> vec(Ti.size(), 0.0f);
    for(int i = 0; i < Ti.size(); i++) {
        vec[i] = std::inner_product(Ti.begin(), Ti.end(), M.a[i].begin(), 0.0f);
    }
    dot = std::inner_product(vec.begin(), vec.end(), Tj.begin(), 0.0f);
}

/**
 * @brief Dot product of T1 x M x T2' for use
 * @param[in] T1 token embedding
 * @param[in] T2 token embedding / vertical retention vector
 * @param[in] M matrix for attention head calculation (MQ x MK')
 * @param[out] dot = T1 x M x T2'
 */
void computeDot(std::vector<float>& T1, std::vector<float>& T2, std::vector<std::vector<float>>& M, float& dot) {
    std::vector<double> temp(T1.size(), 0);
    for(int i = 0; i < T1.size(); i++) {
        temp[i] = std::inner_product(T1.begin(), T1.end(), M[i].begin(), 0.0);
    }
    dot = std::inner_product(temp.begin(), temp.end(), T2.begin(), 0.0);
}

///////////////////------------------FOR TRAINING------------------///////////////////

/**
 * @brief KdotQ via QxK (Q[i].K[j]) for training purpose
 * @param[out] KdotQ dot product
 * @param[in] K Keys
 * @param[in] Q Queries
 * @param[in] currentTokenCount number of tokens in full context
 * @param[in] promptCount tokens in prompt
 * @param[in] attentionType attention type, 1 for self, 0 for cross
 */
void computeKdotQ(std::vector<std::vector<float>>& KdotQ, std::vector<std::vector<float>>& K, std::vector<std::vector<float>>& Q, 
    int& currentTokenCount, int& promptCount, int& blockCount, bool& attentionType)
{
    // for first block
    if(blockCount == 1) {
        // first prompt
        if(currentTokenCount == 0) {
            for(int i = 0; i < promptCount; i++) {
                for(int j = 0; j < (attentionType ? i : promptCount); j++) {
                    KdotQ[i][j] = std::inner_product(Q[i].begin(), Q[i].end(), K[j].begin(), 0.0f) / SCALING;
                }
            }
            currentTokenCount += promptCount;
        }
        // new prompt within first block
        else if (currentTokenCount > 0){
            for(int i = 0; i < promptCount; i++) {
                KdotQ[i][i] = std::inner_product(Q[i].begin(), Q[i].end(), K[i].begin(), 0.0f) / SCALING;
                for(int j = 0; j < currentTokenCount; j++) {
                    KdotQ[i][j] = std::inner_product(Q[i].begin(), Q[i].end(), K[j].begin(), 0.0f) / SCALING;
                }
                if(attentionType == 0) {
                    for(int j = 0; j < currentTokenCount; j++) {
                        KdotQ[j][i] = std::inner_product(Q[j].begin(), Q[j].end(), K[i].begin(), 0.0f) / SCALING;
                    }
                }
                currentTokenCount += 1;
            }
        }
        // first block ended
    }
    // for 2nd to last block
    else {
        if(currentTokenCount%CONTEXT_WIN == 0 && promptCount > 0) {
            for(int i = 0; i < promptCount; i++) {
                for(int j = 0; j < (attentionType ? i : promptCount); j++) {
                    KdotQ[i][j] = std::inner_product(Q[i].begin(), Q[i].end(), K[j].begin(), 0.0f) / SCALING;
                }
            }
            currentTokenCount += promptCount;
        }
        else {
            // when prompt in nth block
            if(currentTokenCount%CONTEXT_WIN != 0 && promptCount > 0) {
                int c = currentTokenCount - (blockCount - 1)*CONTEXT_WIN;
                for(int i = 0; i < promptCount; i++) {
                    KdotQ[i][i] = std::inner_product(Q[i].begin(), Q[i].end(), K[i].begin(), 0.0f) / SCALING;
                    for(int j = 0; j < currentTokenCount; j++) {
                        KdotQ[i][j] = std::inner_product(Q[i].begin(), Q[i].end(), K[j].begin(), 0.0f) / SCALING;
                    }
                    if(attentionType == 0) {
                        for(int j = 0; j < currentTokenCount; j++) {
                            KdotQ[j][i] = std::inner_product(Q[j].begin(), Q[j].end(), K[i].begin(), 0.0f) / SCALING;
                        }
                    }
                    c += 1;
                }
                currentTokenCount += promptCount;
            }
        }
    }
}

///////////////////------------------FOR USE------------------///////////////////

/**
 * @brief KdotQ via tokens (TxMxT') where M = MQ x MK' for use cases for first block
 * @param[out] KdotQ dot product
 * @param[in] tokenEmbed tokens
 * @param[in] M QK' cache
 * @param[in] currentTokenCount number of tokens in full context
 * @param[in] promptCount tokens in prompt 
 * @param[in] attentionType attention type, 1 for self, 0 for cross
 */
void computeKdotQ(std::vector<std::vector<float>>& KdotQ, std::vector<std::vector<float>>& tokenEmbed, mat M, int& currentTokenCount,
    int& promptCount, bool& attentionType)
{
    // first single word prompt like 'hi', 'hello', 'hey', etc.
    if (currentTokenCount == 0 && promptCount == 1) {
        computeDot(tokenEmbed[0], M, tokenEmbed[0], KdotQ[0][0]);
        KdotQ[0][0] = KdotQ[0][0] / SCALING;
        currentTokenCount += 1;
        return;
    }
    // long prompts like 'Hi, Obi'Wan Kenobi here.', etc.
    else if (currentTokenCount == 0 && promptCount > 1) {
        for(int i = 0; i < promptCount; i++) {
            for(int j = 0; j < (attentionType ? i : promptCount); j++) {
                computeDot(tokenEmbed[i], M, tokenEmbed[j], KdotQ[i][j]);
                KdotQ[i][j] = KdotQ[i][j] / SCALING;
            }
        }
        currentTokenCount += promptCount;
        return;
    }
    // for next prompt
    else if(currentTokenCount > 0 && promptCount > 1) {
        for(int i = 0; i < promptCount; i++) {
            // diagonal elements
            computeDot(tokenEmbed[i], M, tokenEmbed[i], KdotQ[i][i]);
            // rows and columns (when cross attention)
            for(int j = 0; j < currentTokenCount; j++) {
                // rows
                computeDot(tokenEmbed[i], M, tokenEmbed[j], KdotQ[i][j]);
                KdotQ[i][j] = KdotQ[i][j] / SCALING;
            }
            // when cross attention
            if(attentionType == 0) {
                for(int j = 0; j < currentTokenCount; j++) {
                    // columns
                    computeDot(tokenEmbed[j], M, tokenEmbed[i], KdotQ[j][i]);
                    KdotQ[i][j] = KdotQ[i][j] / SCALING;
                }
            }
            currentTokenCount += 1;
        }
        return;
    }
}


/**
 * @brief KdotQ via tokens (TxMxEVp') where M = MQ x MK' for use cases (for 2nd to last blocks)
 * @param[out] KdotQ dot product
 * @param[in] tokForBlock token embeddings for this block, in context window of head
 * @param[in] EVp vertical retention vectors of previous block's head of same location
 * @param[in] M QK' cache
 * @param[in] currentTokenCount number of tokens in full context
 * @param[in] promptCount tokens in prompt 
 * @param[in] blockCount current block in the transformer
 * @param[in] attentionType attention type, 1 for self, 0 for cross
 */
void computeKdotQ(std::vector<std::vector<float>>& KdotQ, std::vector<std::vector<float>>& tokForBlock, std::vector<std::vector<float>>& EVp,
    mat M, int& currentTokenCount, int& promptCount, int& blockCount, bool& attentionType)
{
    // tokens in current context
    int c = currentTokenCount - (blockCount - 1)*CONTEXT_WIN;
    // first single word prompt like 'hi', 'hello', 'hey', etc.
    if (c == 0 && promptCount == 1) {
        computeDot(tokForBlock[0], M, EVp[0], KdotQ[0][0]);
        KdotQ[0][0] = KdotQ[0][0] / SCALING;
        currentTokenCount += 1;
        return;
    }
    // long prompts like 'Hi, Obi'Wan Kenobi here.', etc.
    else if (c == 0 && promptCount > 1) {
        for(int i = 0; i < promptCount; i++) {
            for(int j = 0; j < (attentionType ? i : promptCount); j++) {
                computeDot(tokForBlock[i], M, EVp[j], KdotQ[i][j]);
                KdotQ[i][j] = KdotQ[i][j] / SCALING;
            }
        }
        currentTokenCount += promptCount;
        return;
    }
    // for next prompt
    else if(c > 0 && promptCount > 1) {
        for(int i = 0; i < promptCount; i++) {
            // diagonal elements
            computeDot(tokForBlock[i], M, EVp[i], KdotQ[i][i]);
            // rows and columns (when cross attention)
            for(int j = 0; j < c; j++) {
                // rows
                computeDot(tokForBlock[i], M, EVp[j], KdotQ[i][j]);
                KdotQ[i][j] = KdotQ[i][j] / SCALING;
            }
            // when cross attention
            if(attentionType == 0) {
                for(int j = 0; j < c; j++) {
                    // columns
                    computeDot(tokForBlock[j], M, EVp[i], KdotQ[j][i]);
                    KdotQ[i][j] = KdotQ[i][j] / SCALING;
                }
                currentTokenCount += 1;
            }
        }
        return;
    }
}
