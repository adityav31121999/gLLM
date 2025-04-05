
// compute functions
#include "include/block.hpp"
#include "include/transformer.hpp"

/**
 * @brief compute key vectors
 * @param[in] t token embedding
 * @param[in] m matrix input
 * @param[out] k key vector = t x m
 */
void computeKeys(std::vector<float>& t, std::vector<std::vector<float>>& m, std::vector<float>& k) {
    for(int i = 0; i < m.size(); i++) {
        k[i] = std::inner_product(t.begin(), t.end(), m[i].begin(), 0.0f);
    }
}


/**
 * @brief compute query vectors
 * @param[in] t token embedding
 * @param[in] m query matrix input
 * @param[out] q query vector = t x m
 */
void computeQuerys(std::vector<float>& t, std::vector<std::vector<float>>& m, std::vector<float>& q) {
    for(int i = 0; i < m.size(); i++) {
        q[i] = std::inner_product(t.begin(), t.end(), m[i].begin(), 0.0f);
    }
}


/**
 * @brief Dot product of T x M x T'
 * @param T token embedding
 * @param M matrix for attention head calculation (MQ x MK')
 * @param dot = T x M x T'
 */
void computeDot(std::vector<float>& T, std::vector<std::vector<float>>& M, float& dot) {
    std::vector<double> temp(T.size(), 0);
    for(int i = 0; i < T.size(); i++) {
        temp[i] = std::inner_product(T.begin(), T.end(), M[i].begin(), 0.0);
    }
    dot = std::inner_product(temp.begin(), temp.end(), T.begin(), 0.0);
}


/**
 * @brief Dot product of T1 x M x T2'
 * @param T1 token embedding
 * @param T2 token embedding
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
 * @brief Dot product of Ti x M x Tj'
 * @param Ti ith token
 * @param M QK' cache
 * @param Tj jth token (count as transpose)
 * @param dot = Ti x M x Tj'
 */
void computeDot(std::vector<float>& Ti, mat M, std::vector<float>& Tj, float& dot) {
    std::vector<float> vec(Ti.size(), 0.0f);
    for(int i = 0; i < Ti.size(); i++) {
        vec[i] = std::inner_product(Ti.begin(), Ti.end(), M.a[i].begin(), 0.0f);
    }
    dot = std::inner_product(vec.begin(), vec.end(), Tj.begin(), 0.0f);
}


/**
 * @brief KdotQ via tokens (TxMxT') where M = MQ x MK'
 * @param KdotQ dot product
 * @param tokenEmbed tokens
 * @param M QK' cache
 * @param currentTokenCount number of tokens in full context
 * @param promptCount tokens in prompt 
 * @param attentionType attention type, 1 for self, 0 for cross
 */
void computeKdotQ(std::vector<std::vector<float>>& KdotQ, std::vector<std::vector<float>>& tokenEmbed, mat& M, int& currentTokenCount, 
    int& promptCount, bool& attentionType)
{
    // original input
    if (currentTokenCount == 0) {
        // single word like 'Hey', 'Hi', 'Hello', 'Oye', etc.
        if(promptCount == 1) {
            computeDot(tokenEmbed[0], M, tokenEmbed[0], KdotQ[0][0]);
            KdotQ[0][0] = KdotQ[0][0] / SCALING;
            currentTokenCount += 1;
        }
        // long prompt input like 'Hello there, Obi'van Kenobi here', etc.
        else {
            for(int i = 0; i < promptCount; i++) {
                for(int j = 0; j < (attentionType ? i : promptCount); j++) {
                    computeDot(tokenEmbed[i], M, tokenEmbed[j], KdotQ[i][j]);
                    KdotQ[i][j] = KdotQ[i][j] / SCALING;
                }
            }
            currentTokenCount += promptCount;
        }
    }
    // other prompts
    else {
        // promptCount >= 1
        // for single word prompt like 'Why', 'What', 'Who', 'How', 'seriously': promptCount = 1
        // for long prompt like 'are you serious', 'is this really true fact', etc.
        int c = currentTokenCount;
        for(int i = 0; i < promptCount; i++) {
            // diagonal element
            computeDot(tokenEmbed[c + i], M, tokenEmbed[c + i], KdotQ[c + i][c + i]);
            KdotQ[c + i][c + i] = KdotQ[c + i][c + i] / SCALING;
            // for rows and columns
            for(int j = 0; j < currentTokenCount; j++) {
                // for row
                computeDot(tokenEmbed[c + i], M, tokenEmbed[j], KdotQ[c+i][j]);
                KdotQ[c+i][j] = KdotQ[c+i][j] / SCALING;
                // for column, FOR CROSS ATTENTION ONLY 
                if(attentionType == 0) {
                    computeDot(tokenEmbed[j], M, tokenEmbed[c+i], KdotQ[j][c+i]);
                    KdotQ[j][c+i] = KdotQ[j][c+i] / SCALING;
                }
            }
            currentTokenCount += 1;
        }
    }
}


/**
 * @brief KdotQ via QxK (Q[i].K[j])
 * @param KdotQ dot product
 * @param K Keys
 * @param Q Queries
 * @param currentTokenCount number of tokens in full context
 * @param promptCount tokens in prompt
 * @param attentionType attention type, 1 for self, 0 for cross
 */
void computeKdotQ(std::vector<std::vector<float>>& KdotQ, std::vector<std::vector<float>>& K, std::vector<std::vector<float>>& Q, 
    int& currentTokenCount, int& promptCount, bool& attentionType)
{
    if (currentTokenCount == 0) {
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
    else {
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


/**
 * @brief compute KdotQ of each head in the block
 * @param promptCount number of tokens in prompt
 * @param currentTokenCount current count of tokens in full context
 * @param blockCount current block index
 * @param isSelf attention type
 */
void transformer::computeKdotQs(int& promptCount, int& currentTokenCount, int& blockCount, bool& isSelf) {
    // for first block
    if (blockCount == 1) {
        for (int i = 0; i < x; i++) {
            for (int j = 0; j < y; j++) {
                for (int k = 0; k < promptCount; k++) {
                    computeKeys(input[k], t[0].b[i][j].MK.a, t[0].b[i][j].K[k]);
                    computeQuerys(input[k], t[0].b[i][j].MQ.a, t[0].b[i][j].Q[k]);
                }
                computeKdotQ(t[0].b[i][j].KdotQ, t[0].b[i][j].K, t[0].b[i][j].Q, currentTokenCount, promptCount, isSelf);
            }
        }
    }
    // for second to last block
    else {
        // count -> gives current number of tokens in current context window
        int count = std::abs(currentTokenCount - (blockCount-1)*CONTEXT_WIN);
        for (int i = 0; i < x; i++) {
            for (int j = 0; j < y; j++) {
                for (int k = 0; k < count; k++) {
                    computeKeys(input[currentTokenCount + k - 1], t[blockCount - 1].b[i][j].MK.a, t[blockCount - 1].b[i][j].K[k]);
                    computeQuerys(t[blockCount - 2].b[i][j].EV[k], t[blockCount - 1].b[i][j].MQ.a, t[blockCount - 1].b[i][j].Q[k]);
                }
                computeKdotQ(t[blockCount - 1].b[i][j].KdotQ, t[blockCount - 1].b[i][j].K, t[blockCount - 1].b[i][j].Q, 
                            currentTokenCount, count, isSelf);
            }
        }
    }
}
