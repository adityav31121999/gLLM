#ifdef USE_CPU
// compute functions
#include <numeric>
#include <stdexcept>
#include <algorithm>
#include <thread>
#include <chrono>
#include <mutex>
#include "include/block.hpp"
#include "include/transformer.hpp"

///////////////////------------------FOR Prediction------------------///////////////////

// compute prediction from otok and embeddings/deEmbeddings
void transformer::computePrediction()
{
    pred.clear();
    pred.resize(vocabsize, 0.0f);     // hold predictions
    for(int i = 0; i < vocabsize; i++) {
        pred[i] = std::inner_product(otok.begin(), otok.end(), 
                        (contextTrain == 0) ? embeddings(i).begin() : deEmbeddings(i).begin(), 0.0f);
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
    int numThreads = std::thread::hardware_concurrency();
    if(numThreads == 0) {
        numThreads = 1;
    }

    if(numThreads > 0 && numThreads <= 4) {
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
    else if (numThreads > 4) {
        //
    }
    else {
        throw std::runtime_error("Invalid thread count: " + std::to_string(numThreads) + ".");
    }
}

///////////////////------------------FOR INFERENCE------------------///////////////////

/**
 * @brief Dot product between vec1, M and vec2
 * @param[in] vec1 first vector
 * @param[in] M matrix
 * @param[in] vec2 second vector
 * @param[out] result dot product result
 */
void computeDot(const std::vector<float> &vec1, const mat &M, const std::vector<float> &vec2, float &result)
{
    std::vector<float> temp(M.col, 0.0f);
    // Compute vec1 * M
    for (int j = 0; j < M.col; ++j) {
        for (int k = 0; k < M.row; ++k) {
            temp[j] += vec1[k] * M(k, j);
        }
        result += temp[j] * vec2[j];
    }
}

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
        KdotQ[0][0] /= SCALING;
        currentTokenCount += 1;
        return;
    }
    // long sequence1s like 'Hi, Obi'Wan Kenobi here.', etc.
    else if (currentTokenCount == 0 && sequence1Count > 1) {
        for(int i = 0; i < sequence1Count; i++) {
            for(int j = 0; j < (attentionType ? i : sequence1Count); j++) {
                computeDot(tokenEmbed[i], M, tokenEmbed[j], KdotQ[i][j]);
                KdotQ[i][j] /= SCALING;
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
 * @brief compute KdotQ of each head in the block
 * @param sequence1Count number of tokens in the sequence1
 * @param blockCount current block in transformer
 * @param isSelf attention type (= 1 for self, = 0 for cross)
 * @param inTraining 1 for training and 0 for inference
 */
void transformer::computeKdotQs(int& sequence1Count, int& currentTokenCount, int& blockCount, bool&  isSelf, bool&  inTraining)
{
    // for first block
    if(blockCount == 1) {
        if(inTraining == 1) {
            // in training
            for(int i = 0; i < x; i++) {
                for(int j = 0; j < y; j++) {
                    std::vector<std::vector<float>> KdotQ = blocks[0].b[i][j].KdotQ.make2dVector(blocks[0].b[i][j].KdotQ, CONTEXT_WIN, CONTEXT_WIN);
                    std::vector<std::vector<float>> K = blocks[0].b[i][j].K.make2dVector(blocks[0].b[i][j].K, CONTEXT_WIN, EMBEDDING);
                    std::vector<std::vector<float>> Q = blocks[0].b[i][j].Q.make2dVector(blocks[0].b[i][j].Q, CONTEXT_WIN, EMBEDDING);    
                    // compute KdotQ of (i, j) head of first block
                    computeKdotQ(KdotQ, K, Q, currentTokenCount, sequence1Count, blockCount, isSelf);
                    blocks[0].b[i][j].KdotQ = KdotQ;
                }
            }
        }
        else {
            // inference
            for(int i = 0; i < x; i++) {
                for(int j = 0; j < y; j++) {
                    std::vector<std::vector<float>> KdotQ = blocks[0].b[i][j].KdotQ.make2dVector(blocks[0].b[i][j].KdotQ, CONTEXT_WIN, CONTEXT_WIN);
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
                    std::vector<std::vector<float>> kdotq = blocks[blockCount-1].b[i][j].KdotQ.make2dVector(blocks[blockCount-1].b[i][j].KdotQ, CONTEXT_WIN, CONTEXT_WIN);
                    std::vector<std::vector<float>> K = blocks[blockCount-1].b[i][j].K.make2dVector(blocks[blockCount-1].b[i][j].K, CONTEXT_WIN, EMBEDDING);
                    std::vector<std::vector<float>> Q = blocks[blockCount-1].b[i][j].Q.make2dVector(blocks[blockCount-1].b[i][j].Q, CONTEXT_WIN, EMBEDDING);    
                    // compute KdotQ of (i, j) head of first block
                    computeKdotQ(kdotq, K, Q, currentTokenCount, sequence1Count, blockCount, isSelf);
                    blocks[blockCount-1].b[i][j].KdotQ = kdotq;
                }
            }
        }
        else {
            // inference
            for(int i = 0; i < x; i++) {
                for(int j = 0; j < y; j++) {
                    std::vector<std::vector<float>> kdotq = blocks[0].b[i][j].KdotQ.make2dVector(blocks[0].b[i][j].KdotQ, CONTEXT_WIN, CONTEXT_WIN);
                    std::vector<std::vector<float>> tokForBlock = blocks[blockCount-1].tokForBlock.make2dVector(blocks[0].tokForBlock, CONTEXT_WIN, EMBEDDING);
                    std::vector<std::vector<float>> EV = blocks[0].b[i][j].EV.make2dVector(blocks[0].b[i][j].EV, CONTEXT_WIN, EMBEDDING);
                    // compute KdotQ of (i, j) head of first block
                    computeKdotQ(kdotq, tokForBlock, EV, blocks[blockCount-1].b[i][j].qkCache, currentTokenCount, sequence1Count, blockCount, isSelf);
                    blocks[blockCount-1].b[i][j].KdotQ = kdotq;
                }
            }
        }
    }
}

#endif
