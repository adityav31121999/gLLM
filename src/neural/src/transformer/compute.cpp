#ifdef USE_CPU
// compute functions
#include "include/block.hpp"
#include "include/transformer.hpp"

/**
 * @brief compute key or query of token using token embedding and matrix for keys and queries
 * @param[in] tokenEmbed token embedding
 * @param[in] m matrix for keys or queries
 * @param[out] KorQ Key or query vector to be calculated and stored
 */
void computeKorQ(const std::vector<float>& tokenEmbed, const mat& m, std::vector<float>& KorQ) {
    if (tokenEmbed.size() != static_cast<size_t>(m.col) || KorQ.size() != static_cast<size_t>(m.row)) {
        throw std::invalid_argument("computeKorQ: Dimension mismatch. tokenEmbed size (" + std::to_string(tokenEmbed.size()) + ") != m.col (" + std::to_string(m.col) + ") or KorQ size (" + std::to_string(KorQ.size()) + ") != m.row (" + std::to_string(m.row) + ")");
    }
    if (!m.mapped_data) {
        throw std::runtime_error("computeKorQ: Matrix 'm' is not mapped.");
    }
    for(int i = 0; i < m.row; ++i) {
        float sum = 0.0f;
        size_t row_offset = static_cast<size_t>(i) * m.col;
        for(int k = 0; k < m.col; ++k) {
            sum += tokenEmbed[k] * m.mapped_data[row_offset + k];
        }
        KorQ[i] = sum;
    }
}


/**
 * @brief Dot product of Ti x M x Tj' for use
 * @param[in] Ti ith token
 * @param[in] M QK' cache
 * @param[in] Tj jth token (count as transpose) / transpose of vertical retention vector
 * @param[out] dot = Ti x M x Tj'
 */
void computeDot(const std::vector<float>& Ti, const mat& M, const std::vector<float>& Tj, float& dot) {
    if (Ti.size() != static_cast<size_t>(M.col) || Tj.size() != static_cast<size_t>(M.row)) {
         throw std::invalid_argument("computeDot (vec, mat, vec): Dimension mismatch. Ti size (" + std::to_string(Ti.size()) + ") != M.col (" + std::to_string(M.col) + ") or Tj size (" + std::to_string(Tj.size()) + ") != M.row (" + std::to_string(M.row) + ")");
    }
    if (!M.mapped_data) {
        throw std::runtime_error("computeDot (vec, mat, vec): Matrix 'M' is not mapped.");
    }
    std::vector<float> temp_vec(M.row); // Result of Ti * M
    for(int i = 0; i < M.row; ++i) {
        float row_dot = 0.0f;
        size_t row_offset = static_cast<size_t>(i) * M.col;
        for(int k = 0; k < M.col; ++k) {
            row_dot += Ti[k] * M.mapped_data[row_offset + k];
        }
        temp_vec[i] = row_dot;
    }
    // dot = dot product of temp_vec and Tj
    dot = std::inner_product(temp_vec.begin(), temp_vec.end(), Tj.begin(), 0.0f);
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

/**
 * @brief compute the prediction for possible token embedding output
 * @param output forward propagation from block: EH
 * @param embeddings token embeddings
 * @param voc size of token vocabulary
 * @param index position of highest probability token embedding
 * @note it is assumed in this function that the case of "all dot products being zero" will
 *      not occur
 */
void computeOutput(std::vector<float>& output, std::vector<std::vector<float>>& embeddings, unsigned long long& voc, int& index)
{
    std::vector<float> pred(voc, 0.0f);     // hold predictions
    for(int i = 0; i < voc; i++) {
        pred[i] = std::inner_product(output.begin(), output.end(), embeddings[i].begin(), 0.0f); // dot product
    }
    // find the highest value in the pred vector
    float max = pred[0];
    index = 0;
    for(int i = 1; i < voc; i++) {
        if(pred[i] > max) {
            max = pred[i];
            index = i;
        }
    }
}

/**
 * @brief compute the prediction for possible token embedding output
 * @param output forward propagation from block: EH
 * @param embeddings token embeddings
 * @param voc size of token vocabulary
 * @param index position of highest probability token embedding
 * @note it is assumed in this function that the case of "all dot products being zero" will
 *      not occur
 */
void computeOutput(const std::vector<float>& output, const mat& embeddings, unsigned long long& voc, int& index)
{
    if (output.size() != static_cast<size_t>(embeddings.col) || voc != embeddings.row) {
        throw std::invalid_argument("computeOutput: Dimension mismatch. output size (" + std::to_string(output.size()) + ") != embeddings.col (" + std::to_string(embeddings.col) + ") or voc (" + std::to_string(voc) + ") != embeddings.row (" + std::to_string(embeddings.row) + ")");
    }
    if (!embeddings.mapped_data) {
        throw std::runtime_error("computeOutput: Embeddings matrix is not mapped.");
    }
    std::vector<float> pred(voc, 0.0f);     // hold predictions
    // Calculate dot product of 'output' with each row of 'embeddings'
    for(int i = 0; i < voc; i++) {
        float row_dot = 0.0f;
        size_t row_offset = static_cast<size_t>(i) * embeddings.col;
        for(int k = 0; k < embeddings.col; ++k) {
            row_dot += output[k] * embeddings.mapped_data[row_offset + k];
        }
        pred[i] = row_dot; // dot product
    }
    // find the highest value in the pred vector
    float max = pred[0];
    index = 0;
    for(int i = 1; i < voc; i++) {
        if(pred[i] > max) {
            max = pred[i];
            index = i;
        }
    }
}

///////////////////------------------FOR TRAINING------------------///////////////////

/**
 * @brief KdotQ via QxK (Q[i].K[j]) for training purpose
 * @param[out] KdotQ dot product
 * @param[in] K Keys
 * @param[in] Q Queries
 * @param[in] currentTokenCount number of tokens in full context
 * @param[in] sequence1Count tokens in sequence1
 * @param[in] attentionType attention type, 1 for self, 0 for cross
 */
void computeKdotQ(std::vector<std::vector<float>>& KdotQ, std::vector<std::vector<float>>& K, std::vector<std::vector<float>>& Q, 
    int& currentTokenCount, int& sequence1Count, int& blockCount, bool& attentionType)
{
    // for first block
    if(blockCount == 1) {
        // first sequence1
        if(currentTokenCount == 0) {
            for(int i = 0; i < sequence1Count; i++) {
                for(int j = 0; j < (attentionType ? i : sequence1Count); j++) {
                    KdotQ[i][j] = std::inner_product(Q[i].begin(), Q[i].end(), K[j].begin(), 0.0f) / SCALING;
                }
            }
            currentTokenCount += sequence1Count;
        }
        // new sequence1 within first block
        else if (currentTokenCount > 0){
            for(int i = 0; i < sequence1Count; i++) {
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
        if(currentTokenCount%CONTEXT_WIN == 0 && sequence1Count > 0) {
            for(int i = 0; i < sequence1Count; i++) {
                for(int j = 0; j < (attentionType ? i : sequence1Count); j++) {
                    KdotQ[i][j] = std::inner_product(Q[i].begin(), Q[i].end(), K[j].begin(), 0.0f) / SCALING;
                }
            }
            currentTokenCount += sequence1Count;
        }
        else {
            // when sequence1 in nth block
            if(currentTokenCount%CONTEXT_WIN != 0 && sequence1Count > 0) {
                int c = currentTokenCount - (blockCount - 1)*CONTEXT_WIN;
                for(int i = 0; i < sequence1Count; i++) {
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
                currentTokenCount += sequence1Count;
            }
        }
    }
}

///////////////////------------------FOR INFERENCE------------------///////////////////

/**
 * @brief KdotQ via tokens (TxMxT') where M = MQ x MK' for inference for first block
 * @param[out] KdotQ dot product
 * @param[in] tokenEmbed tokens
 * @param[in] M QK' cache
 * @param[in] currentTokenCount number of tokens in full context
 * @param[in] sequence1Count tokens in sequence1 
 * @param[in] attentionType attention type, 1 for self, 0 for cross
 */
void computeKdotQ(std::vector<std::vector<float>>& KdotQ, std::vector<std::vector<float>>& tokenEmbed, mat& M, int& currentTokenCount,
    int& sequence1Count, bool& attentionType)
{
    // first single word sequence1 like 'hi', 'hello', 'hey', etc.
    if (currentTokenCount == 0 && sequence1Count == 1) {
        computeDot(tokenEmbed[0], M, tokenEmbed[0], KdotQ[0][0]);
        KdotQ[0][0] = KdotQ[0][0] / SCALING;
        currentTokenCount += 1;
        return;
    }
    // long sequence1s like 'Hi, Obi'Wan Kenobi here.', etc.
    else if (currentTokenCount == 0 && sequence1Count > 1) {
        for(int i = 0; i < sequence1Count; i++) {
            for(int j = 0; j < (attentionType ? i : sequence1Count); j++) {
                computeDot(tokenEmbed[i], M, tokenEmbed[j], KdotQ[i][j]);
                KdotQ[i][j] = KdotQ[i][j] / SCALING;
            }
        }
        currentTokenCount += sequence1Count;
        return;
    }
    // for next sequence1
    else if(currentTokenCount > 0 && sequence1Count > 1) {
        for(int i = 0; i < sequence1Count; i++) {
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
 * @param[in] sequence1Count tokens in sequence1 
 * @param[in] blockCount current block in the transformer
 * @param[in] attentionType attention type, 1 for self, 0 for cross
 */
void computeKdotQ(std::vector<std::vector<float>>& KdotQ, std::vector<std::vector<float>>& tokForBlock, std::vector<std::vector<float>>& EVp,
    mat& M, int& currentTokenCount, int& sequence1Count, int& blockCount, bool& attentionType)
{
    // tokens in local context
    int c = currentTokenCount - (blockCount - 1)*CONTEXT_WIN;
    // first single word sequence1 like 'hi', 'hello', 'hey', etc.
    if (c == 0 && sequence1Count == 1) {
        computeDot(tokForBlock[0], M, EVp[0], KdotQ[0][0]);
        KdotQ[0][0] = KdotQ[0][0] / SCALING;
        currentTokenCount += 1;
        return;
    }
    // long sequence1s like 'Hi, Obi'Wan Kenobi here.', etc.
    else if (c == 0 && sequence1Count > 1) {
        for(int i = 0; i < sequence1Count; i++) {
            for(int j = 0; j < (attentionType ? i : sequence1Count); j++) {
                computeDot(tokForBlock[i], M, EVp[j], KdotQ[i][j]);
                KdotQ[i][j] = KdotQ[i][j] / SCALING;
            }
        }
        currentTokenCount += sequence1Count;
        return;
    }
    // for next sequence1
    else if(c > 0 && sequence1Count > 1) {
        for(int i = 0; i < sequence1Count; i++) {
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

/**
 * @brief compute parallel KdotQs of column of a block
 * @param sequence1Count number of tokens in sequence1
 * @param currentTokenCount number of tokens in full context
 * @param blockCount current position of block in full context
 * @param column current column in local context or block
 * @param isSelf attention type (self or cross)
 * @param inTraining training or inference
 */
void transformer::parallelKdotQs(int &sequence1Count, int &currentTokenCount, int &blockCount, int &column, bool &isSelf, bool &inTraining)
{
    // first block
    if(blockCount == 1) {
        if(inTraining == 1) {
            // in training
            for(int i = 0; i < x; i++) {
                std::vector<std::vector<float>> KdotQ = blocks[0].b[i][column].KdotQ.make2dVector(t[0].b[i][column].KdotQ, CONTEXT_WIN, CONTEXT_WIN);
                std::vector<std::vector<float>> K = blocks[0].b[i][column].K.make2dVector(t[0].b[i][column].K, CONTEXT_WIN, EMBEDDING);
                std::vector<std::vector<float>> Q = blocks[0].b[i][column].Q.make2dVector(t[0].b[i][column].Q, CONTEXT_WIN, EMBEDDING);
                // compute KdotQ of (i, j) head of first block
                computeKdotQ(KdotQ, K, Q, currentTokenCount, sequence1Count, blockCount, isSelf);
                blocks[0].b[i][column].KdotQ = KdotQ;
            }
        }
        else {
            // for inference
            for(int i = 0; i < x; i++) {
                std::vector<std::vector<float>> KdotQ = blocks[0].b[i][column].KdotQ.make2dVector(t[0].b[i][column].KdotQ, CONTEXT_WIN, CONTEXT_WIN);
                std::vector<std::vector<float>> TokenEmbed = tokenEmbed.make2dVector(tokenEmbed, CONTEXT_WIN, EMBEDDING);
                // compute KdotQ of (i, j) head of first block
                computeKdotQ(KdotQ, TokenEmbed, blocks[0].b[i][column].qkCache, currentTokenCount, sequence1Count, isSelf);
                blocks[0].b[i][column].KdotQ = KdotQ;
            }
        }
    }
    // for ith block
    else {
        if(inTraining == 1) {
            // in training
            for(int i = 0; i < x; i++) {
                std::vector<std::vector<float>> KdotQ = blocks[blockCount-1].b[i][column].KdotQ.make2dVector(t[blockCount-1].b[i][column].KdotQ, CONTEXT_WIN, CONTEXT_WIN);
                std::vector<std::vector<float>> K = blocks[blockCount-1].b[i][column].K.make2dVector(t[blockCount-1].b[i][column].K, CONTEXT_WIN, EMBEDDING);
                std::vector<std::vector<float>> Q = blocks[blockCount-1].b[i][column].Q.make2dVector(t[blockCount-1].b[i][column].Q, CONTEXT_WIN, EMBEDDING);
                computeKdotQ(KdotQ, K, Q, currentTokenCount, sequence1Count, blockCount, isSelf);
                blocks[0].b[i][column].KdotQ = KdotQ;
            }
        }
        else {
            // for inference
            for(int i = 0; i < x; i++) {
                std::vector<std::vector<float>> kdotq = blocks[blockCount-1].b[i][column].KdotQ.make2dVector(t[0].b[i][column].KdotQ, CONTEXT_WIN, CONTEXT_WIN);
                std::vector<std::vector<float>> tokforblock = blocks[blockCount-1].tokForBlock.make2dVector(t[0].tokForBlock, CONTEXT_WIN, EMBEDDING);
                std::vector<std::vector<float>> ev = blocks[blockCount-1].b[i][column].EV.make2dVector(t[0].b[i][column].EV, CONTEXT_WIN, EMBEDDING);
                // compute KdotQ of (i, j) head of first block
                computeKdotQ(kdotq, tokforblock, ev, blocks[blockCount-1].b[i][column].qkCache, currentTokenCount, sequence1Count, blockCount, isSelf);
            }
        }
    }
}


/**
 * @brief compute KdotQ of each head in the block
 * @param sequence1Count number of tokens in the sequence1
 * @param blockCount current block in transformer
 * @param isSelf attention type (= 1 for self, = 0 for cross)
 * @param inTraining 1 for training and 0 for inference
 */
void transformer::computeKdotQs(int &sequence1Count, int &currentTokenCount, int &blockCount, bool &isSelf, bool &inTraining)
{
    // for first block
    if(blockCount == 1) {
        if(inTraining == 1) {
            // in training
            for(int i = 0; i < x; i++) {
                for(int j = 0; j < y; j++) {
                    std::vector<std::vector<float>> KdotQ = blocks[0].b[i][j].KdotQ.make2dVector(t[0].b[i][j].KdotQ, CONTEXT_WIN, CONTEXT_WIN);
                    std::vector<std::vector<float>> K = blocks[0].b[i][j].K.make2dVector(t[0].b[i][j].K, CONTEXT_WIN, EMBEDDING);
                    std::vector<std::vector<float>> Q = blocks[0].b[i][j].Q.make2dVector(t[0].b[i][j].Q, CONTEXT_WIN, EMBEDDING);    
                    // compute KdotQ of (i, j) head of first block
                    computeKdotQ(KdotQ, K, Q, currentTokenCount, sequence1Count, blockCount, isSelf);
                    blocks[0].b[i][j].KdotQ = KdotQ;
                }
            }
        }
        else {
            // in use
            for(int i = 0; i < x; i++) {
                for(int j = 0; j < y; j++) {
                    std::vector<std::vector<float>> KdotQ = blocks[0].b[i][j].KdotQ.make2dVector(t[0].b[i][j].KdotQ, CONTEXT_WIN, CONTEXT_WIN);
                    std::vector<std::vector<float>> ctokenEmbed = tokenEmbed.make2dVector(tokenEmbed, CONTEXT_WIN, EMBEDDING);
                    // compute KdotQ of (i, j) head of first block
                    computeKdotQ(KdotQ, ctokenEmbed, blocks[0].b[i][j].qkCache, currentTokenCount,
                                    sequence1Count, isSelf);
                }
            }
        }
    }
    // for ith block
    else {
        if(inTraining == 1) {
            // in training
            for(int i = 0; i < x; i++) {
                for(int j = 0; j < y; j++) {
                    std::vector<std::vector<float>> kdotq = blocks[blockCount-1].b[i][j].KdotQ.make2dVector(t[blockCount-1].b[i][j].KdotQ, CONTEXT_WIN, CONTEXT_WIN);
                    std::vector<std::vector<float>> K = blocks[blockCount-1].b[i][j].K.make2dVector(t[blockCount-1].b[i][j].K, CONTEXT_WIN, EMBEDDING);
                    std::vector<std::vector<float>> Q = blocks[blockCount-1].b[i][j].Q.make2dVector(t[blockCount-1].b[i][j].Q, CONTEXT_WIN, EMBEDDING);    
                    // compute KdotQ of (i, j) head of first block
                    computeKdotQ(kdotq, K, Q, currentTokenCount, sequence1Count, blockCount, isSelf);
                    blocks[blockCount-1].b[i][j].KdotQ = kdotq;
                }
            }
        }
        else {
            // in use
            for(int i = 0; i < x; i++) {
                for(int j = 0; j < y; j++) {
                    std::vector<std::vector<float>> kdotq = blocks[0].b[i][j].KdotQ.make2dVector(t[0].b[i][j].KdotQ, CONTEXT_WIN, CONTEXT_WIN);
                    std::vector<std::vector<float>> tokForBlock = blocks[blockCount-1].tokForBlock.make2dVector(t[0].tokForBlock, CONTEXT_WIN, EMBEDDING);
                    std::vector<std::vector<float>> EV = blocks[0].b[i][j].EV.make2dVector(t[0].b[i][j].EV, CONTEXT_WIN, EMBEDDING);
                    // compute KdotQ of (i, j) head of first block
                    computeKdotQ(kdotq, tokForBlock, EV, blocks[blockCount-1].b[i][j].qkCache, currentTokenCount, sequence1Count, blockCount, isSelf);
                    blocks[blockCount-1].b[i][j].KdotQ = kdotq;
                }
            }
        }
    }
}

#endif
