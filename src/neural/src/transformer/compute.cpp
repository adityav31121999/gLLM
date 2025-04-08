
// compute functions
#include "include/block.hpp"
#include "include/transformer.hpp"

/**
 * @brief Dot product of T1 x M x T2' for use
 * @param T1 token embedding
 * @param T2 token embedding / vertical retention vector
 * @param M matrix for attention head calculation (MQ x MK')
 * @param dot = T1 x M x T2'
 */
void computeDot(std::vector<float>& T1, std::vector<float>& T2, std::vector<std::vector<float>>& M, float& dot) {
    std::vector<double> temp(T1.size(), 0);
    for(int i = 0; i < T1.size(); i++) {
        temp[i] = std::inner_product(T1.begin(), T1.end(), M[i].begin(), 0.0);
    }
    dot = std::inner_product(temp.begin(), temp.end(), T2.begin(), 0.0);
}

/**
 * @brief Dot product of Ti x M x Tj' for use
 * @param Ti ith token
 * @param M QK' cache
 * @param Tj jth token (count as transpose) / transpose of vertical retention vector
 * @param dot = Ti x M x Tj'
 */
void computeDot(std::vector<float>& Ti, mat M, std::vector<float>& Tj, float& dot) {
    std::vector<float> vec(Ti.size(), 0.0f);
    for(int i = 0; i < Ti.size(); i++) {
        vec[i] = std::inner_product(Ti.begin(), Ti.end(), M.a[i].begin(), 0.0f);
    }
    dot = std::inner_product(vec.begin(), vec.end(), Tj.begin(), 0.0f);
}

///////////////////------------------FOR TRAINING------------------///////////////////

/**
 * @brief KdotQ via QxK (Q[i].K[j]) for training purpose
 * @param KdotQ dot product
 * @param K Keys
 * @param Q Queries
 * @param currentTokenCount number of tokens in full context
 * @param promptCount tokens in prompt
 * @param attentionType attention type, 1 for self, 0 for cross
 */
void computeKdotQ(std::vector<std::vector<float>>& KdotQ, std::vector<std::vector<float>>& K, std::vector<std::vector<float>>& Q, 
    int& currentTokenCount, int& promptCount, int& blockCount, bool& attentionType)
{
    // for first user prompt
    if (currentTokenCount == 0 && promptCount <= CONTEXT_WIN) {
        if(promptCount == 1) {
            KdotQ[0][0] = std::inner_product(K[0].begin(), K[0].end(), Q[0].begin(), 0.0f)/SCALING;
            currentTokenCount += 1;
        }
        else {
            for(int i = 0; i < promptCount; i++) {
                for(int j = 0; j < (attentionType ? i : promptCount); j++) {
                    KdotQ[i][j] = std::inner_product(K[i].begin(), K[i].end(), Q[j].begin(), 0.0f)/SCALING;
                }
            }
            currentTokenCount += promptCount;

        }
    }
    // for first user prompt that exceeds context window
    if (currentTokenCount == 0 && promptCount > CONTEXT_WIN) {
        if(promptCount == 1) {
            KdotQ[0][0] = std::inner_product(K[0].begin(), K[0].end(), Q[0].begin(), 0.0f)/SCALING;
            currentTokenCount += 1;
        }
        else {
            for(int i = 0; i < promptCount; i++) {
                for(int j = 0; j < (attentionType ? i : promptCount); j++) {
                    KdotQ[i][j] = std::inner_product(K[i].begin(), K[i].end(), Q[j].begin(), 0.0f)/SCALING;
                }
            }
            currentTokenCount += promptCount;
        }
    }
    // 
    if (currentTokenCount > CONTEXT_WIN) {
        // promptCount >= 1
        int c = currentTokenCount;
        for(int i = 0; i < promptCount; i++) {
            KdotQ[c+i][c+i] = std::inner_product(K[c+i].begin(), K[c+i].end(), Q[c+i].begin(), 0.0f)/SCALING;
            for(int j = 0; j < currentTokenCount; j++) {
                // for row
                KdotQ[c+i][j] = std::inner_product(K[c+i].begin(), K[c+i].end(), Q[j].begin(), 0.0f)/SCALING;
                // for column, FOR CROSS ATTENTION ONLY
                if(attentionType == 0) {
                    KdotQ[j][c+i] = std::inner_product(K[j].begin(), K[j].end(), Q[c+i].begin(), 0.0f)/SCALING;
                }
            }
            currentTokenCount += 1;
        }
    }
}

///////////////////------------------FOR TRAINING------------------///////////////////

/**
 * @brief compute KdotQ for all heads of a block
 * @param promptCount number of tokens in user prompt
 * @param currentTokenCount number of tokens in full context
 * @param blockCount current block number in full context
 * @param isSelf attention type for masking (1 for self, 0 for cross)
 */
void transformer::computeKdotQs(int &promptCount, int &currentTokenCount, int &blockCount, bool &isSelf)
{
    //
}

/**
 * compute KdotQ for model operations using tokens and cache in first block
 */

/**
 * @brief KdotQ via tokens (TxMxT') where M = MQ x MK' for use cases for first block
 * @param KdotQ dot product
 * @param tokenEmbed tokens
 * @param M QK' cache
 * @param currentTokenCount number of tokens in full context
 * @param promptCount tokens in prompt 
 * @param attentionType attention type, 1 for self, 0 for cross
 */
void computeKdotQ(std::vector<std::vector<float>>& KdotQ, std::vector<std::vector<float>>& tokenEmbed, mat M, int& currentTokenCount,
    int& promptCount, bool& attentionType)
{
    // first single word prompt like 'hi', 'hello', 'hey', etc.
    if (currentTokenCount == 0 && promptCount == 1) {
        computeDot(tokenEmbed[0], M, tokenEmbed[0], KdotQ[0][0]);
        KdotQ[0][0] = KdotQ[0][0] / SCALING;
        currentTokenCount += 1;
    }
    // first long prompt input like 'Hello there, Obi'van Kenobi here', 'Are you there?', etc.
    if (currentTokenCount == 0 && promptCount > 1) {
        for(int i = 0; i < promptCount; i++) {
            for(int j = 0; j < (attentionType ? i : promptCount); j++) {
                computeDot(tokenEmbed[i], M, tokenEmbed[j], KdotQ[i][j]);
                KdotQ[i][j] = KdotQ[i][j] / SCALING;
            }
        }
        currentTokenCount += promptCount;
    }
    // for next prompt
    if(currentTokenCount > 0 && promptCount > 1) {
        for(int i = 0; i < promptCount; i++) {
            // diagonal elements
            computeDot(tokenEmbed[i], M, tokenEmbed[i], KdotQ[i][i]);
            // rows and columns (when cross attention)
            for(int j = 0; j < currentTokenCount; j++) {
                // rows
                computeDot(tokenEmbed[i], M, tokenEmbed[j], KdotQ[i][j]);
                KdotQ[i][j] = KdotQ[i][j] / SCALING;
                // when cross attention
                if(attentionType == 0) {
                    // columns
                    computeDot(tokenEmbed[j], M, tokenEmbed[i], KdotQ[j][i]);
                    KdotQ[i][j] = KdotQ[i][j] / SCALING;
                }
            }
            currentTokenCount += 1;
        }
    }
}

/**
 * compute KdotQ for model operations using tokens, cache and vertical retention
 * vectors in 2nd to last blocks
 */

/**
 * @brief KdotQ via tokens (TxMxEVp') where M = MQ x MK' for use cases (for 2nd to last blocks)
 * @param KdotQ dot product
 * @param tokenEmbed token embeddings
 * @param M QK' cache
 * @param currentTokenCount number of tokens in full context
 * @param promptCount tokens in prompt 
 * @param attentionType attention type, 1 for self, 0 for cross
 */
void computeKdotQ(std::vector<std::vector<float>>& KdotQ, std::vector<std::vector<float>>& tokenEmbed, std::vector<std::vector<float>>& EVp,
    mat M, int& currentTokenCount, int& promptCount, int& blockCount, bool& attentionType)
{
    // when KdotQ has no prefilled values
    if (currentTokenCount % CONTEXT_WIN == 0 && promptCount > 0) 
    {
        // for 2nd to last block
        int c = currentTokenCount - (blockCount * CONTEXT_WIN);
        for(int i = 0; i < promptCount; i++)
        {
            // diagonal element
            computeDot(tokenEmbed[currentTokenCount + i], M, EVp[c + i], KdotQ[c + i][c + i]);
            KdotQ[c + i][c + i] = KdotQ[c + i][c + i] / SCALING;
            // for rows and columns
            for(int j = 0; j < c; j++) {
                // for row
                computeDot(tokenEmbed[currentTokenCount + i], M, EVp[j], KdotQ[c + i][j]);
                KdotQ[c + i][j] = KdotQ[c + i][j] / SCALING;
            }
            // in cross attention only
            if(attentionType == 0) {
                for(int j = 0; j < currentTokenCount; j++) {
                    // for columns
                    computeDot(tokenEmbed[currentTokenCount + j], M, EVp[i], KdotQ[j][c + i]);
                    KdotQ[j][c + i] = KdotQ[j][c + i] / SCALING;
                }
            }
            currentTokenCount += 1;
        }
    }
    // when the KdotQ prefilled values
    if (currentTokenCount % CONTEXT_WIN >= 0 && promptCount > 0) 
    {
        // for for 2nd to last block
        int c = currentTokenCount - (blockCount - 1)*CONTEXT_WIN;
        for(int i = 0; i < promptCount; i++)
        {
            // diagonal element
            computeDot(tokenEmbed[currentTokenCount + i], M, EVp[c + i], KdotQ[c + i][c + i]);
            KdotQ[c + i][c + i] = KdotQ[c + i][c + i] / SCALING;
            // for rows and columns
            for(int j = 0; j < c; j++) {
                // for row
                computeDot(tokenEmbed[currentTokenCount + i], M, EVp[j], KdotQ[c + i][j]);
                KdotQ[c + i][j] = KdotQ[c + i][j] / SCALING;
            }
            // in cross attention only
            if(attentionType == 0) {
                for(int j = 0; j < currentTokenCount; j++) {
                    // for columns
                    computeDot(tokenEmbed[currentTokenCount + j], M, EVp[i], KdotQ[j][c + i]);
                    KdotQ[j][c + i] = KdotQ[j][c + i] / SCALING;
                }
            }
            currentTokenCount += 1;
        }
    }
}
