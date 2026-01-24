#ifdef USE_CPU
// transformer training
#include "include/attention.hpp"
#include "include/block.hpp"
#include "include/transformer.hpp"
#include <algorithm>
#include <maths.hpp>

/**
 * @brief train the transformer for full context last token prediction
 * @param expected expected token embedding
 * @param expString expected token
 */
void transformer::train(std::vector<float>& expected, std::string& expString) 
{
    sequence1Count = 1; // Set sequence1Count to 1 for single token training

    // for first block
    if(blockCount == 1 && currentTokenCount < CONTEXT_WIN) {
        computeKdotQs(sequence1Count, currentTokenCount, blockCount, isSelf, inTraining);
        // train from here
        forward(blockCount, currentTokenCount, sequence1Count);
        int i = 0;
        while (i <= epochs) {
            computePrediction();
            auto max_it = std::max_element(pred.begin(), pred.end());
            indexForToken = std::distance(pred.begin(), max_it);

            if((binaryCrossEntropy(expected, otok) < 0.01) || tokens[indexForToken] == expString) {
                break;
            }
            // if error is not corrected even after epochs, then increase epochs
            if(binaryCrossEntropy(expected, otok) > 0.01 && i == epochs) {
                epochs += 10;
            }
            backward(expected, blockCount);
            forward(blockCount, currentTokenCount, sequence1Count);
            i++;
        }
        trainCount++;
        epochCount += i;
        error += binaryCrossEntropy(expected, otok);
        currentTokenCount += 1;
        if(currentTokenCount == CONTEXT_WIN) {
            blockCount += 1;
        }
    }
    // for next blocks
    else if(blockCount > 1 && currentTokenCount >= CONTEXT_WIN) {
        computeKdotQs(sequence1Count, currentTokenCount, blockCount, isSelf, inTraining);
        // train from here
        forward(blockCount, currentTokenCount, sequence1Count);
        int i = 0;
        while (i < epochs) {
            computePrediction();
            auto max_it = std::max_element(pred.begin(), pred.end());
            indexForToken = std::distance(pred.begin(), max_it);

            if(binaryCrossEntropy(expected, otok) < 0.01 || tokens[indexForToken] == expString) {
                break;
            }
            // if error is not corrected even after epochs, then increase epochs
            if(binaryCrossEntropy(expected, otok) > 0.01 && i == epochs) {
                epochs += 10;
            }
            i++;
            backward(expected, blockCount);
            forward(blockCount, currentTokenCount, sequence1Count);
        }
        trainCount++;
        epochCount += i;
        error += binaryCrossEntropy(expected, otok);
        currentTokenCount += 1;
        if(currentTokenCount % CONTEXT_WIN == 0) {
            blockCount += 1;
        }
    }
}

/**
 * @brief train the transformer for full context last token prediction (Contextualised)
 * @param expected expected token embedding
 * @param expString expected token
 */
void transformer::trainContext(std::vector<float>& expected, std::string& expString) 
{
    sequence1Count = 1;
    
    // Find index for expString
    int targetIndex = -1;
    for(size_t k=0; k<tokens.size(); ++k){
        if(tokens[k] == expString) {
            targetIndex = k;
            break;
        }
    }
    if(targetIndex == -1) throw std::runtime_error("Token not found: " + expString);

    std::fill(oneHotEncode.begin(), oneHotEncode.end(), 0.0f);
    oneHotEncode[targetIndex] = 1.0f;

    // for first block
    if(blockCount == 1 && currentTokenCount < CONTEXT_WIN) {
        computeKdotQs(sequence1Count, currentTokenCount, blockCount, isSelf, inTraining);
        // train from here
        forward(blockCount, currentTokenCount, sequence1Count);
        int i = 0;
        while (i <= epochs) {
            computePrediction();
            pred = softmax(pred);
            auto max_it = std::max_element(pred.begin(), pred.end());
            indexForToken = std::distance(pred.begin(), max_it);

            if(tokens[indexForToken] == expString) {
                break;
            }
            // if error is not corrected even after epochs, then increase epochs
            if(i == epochs) {
                epochs += 10;
            }

            // Backprop logic for context
            std::vector<float> gradEH(d * x, 0.0f);
            updateDeEmbeddings(deEmbeddings, pred, oneHotEncode, learning, gradEH);
            
            std::vector<std::vector<float>> targets_for_heads(x, std::vector<float>(EMBEDDING, 0.0f));
            for(int head_idx = 0; head_idx < x; ++head_idx) {
                for(int eidx = 0; eidx < EMBEDDING; ++eidx) {
                    float gradient = learning * (gradEH[(head_idx * EMBEDDING) + eidx]
                                              + (lambda_L1 * embeddings(indexForToken, eidx))
                                              + (2.0f * lambda_L2 * embeddings(indexForToken, eidx)));
                    if (fabs(gradient) >= MAX_GRAD_CLIP) gradient = std::copysign(MAX_GRAD_CLIP, gradient);
                    targets_for_heads[head_idx][eidx] = otok[(head_idx * EMBEDDING) + eidx] - gradient;
                }
            }
            backwardContext(targets_for_heads, blockCount);
            updateEmbeddings(embeddings, blocks[blockCount-1].gradToken, learning, vocabsize);
            updateEmbeddings(tokenEmbed, blocks[blockCount-1].gradToken, learning, currentTokenCount);

            forward(blockCount, currentTokenCount, sequence1Count);
            i++;
        }
        trainCount++;
        epochCount += i;
        currentTokenCount += 1;
        if(currentTokenCount == CONTEXT_WIN) {
            blockCount += 1;
        }
    }
    // for next blocks
    else if(blockCount > 1 && currentTokenCount >= CONTEXT_WIN) {
        computeKdotQs(sequence1Count, currentTokenCount, blockCount, isSelf, inTraining);
        // train from here
        forward(blockCount, currentTokenCount, sequence1Count);
        int i = 0;
        while (i < epochs) {
            computePrediction();
            pred = softmax(pred);
            auto max_it = std::max_element(pred.begin(), pred.end());
            indexForToken = std::distance(pred.begin(), max_it);

            if(tokens[indexForToken] == expString) {
                break;
            }
            if(i == epochs) {
                epochs += 10;
            }
            
            // Backprop logic for context
            std::vector<float> gradEH(d * x, 0.0f);
            updateDeEmbeddings(deEmbeddings, pred, oneHotEncode, learning, gradEH);
            
            std::vector<std::vector<float>> targets_for_heads(x, std::vector<float>(EMBEDDING, 0.0f));
            for(int head_idx = 0; head_idx < x; ++head_idx) {
                for(int eidx = 0; eidx < EMBEDDING; ++eidx) {
                    float gradient = learning * (gradEH[(head_idx * EMBEDDING) + eidx]
                                              + (lambda_L1 * embeddings(indexForToken, eidx))
                                              + (2.0f * lambda_L2 * embeddings(indexForToken, eidx)));
                    if (fabs(gradient) >= MAX_GRAD_CLIP) gradient = std::copysign(MAX_GRAD_CLIP, gradient);
                    targets_for_heads[head_idx][eidx] = otok[(head_idx * EMBEDDING) + eidx] - gradient;
                }
            }
            backwardContext(targets_for_heads, blockCount);
            updateEmbeddings(embeddings, blocks[blockCount-1].gradToken, learning, vocabsize);
            updateEmbeddings(tokenEmbed, blocks[blockCount-1].gradToken, learning, currentTokenCount);

            i++;
            forward(blockCount, currentTokenCount, sequence1Count);
        }
        trainCount++;
        epochCount += i;
        currentTokenCount += 1;
        if(currentTokenCount % CONTEXT_WIN == 0) {
            blockCount += 1;
        }
    }
}

#endif
