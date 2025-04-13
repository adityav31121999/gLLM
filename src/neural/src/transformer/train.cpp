
// transformer training
#include "include/attention.hpp"
#include "include/block.hpp"
#include "include/transformer.hpp"
#include "transformer.hpp"


/**
 * @brief train the transformer for next token prediction (single token training)
 * @param promptCount number of tokens in the prompt
 * @param currentTokenCount number of tokens in the full context
 * @param blockCount current block in full context
 * @param isSelf attention type
 * @param expected expected token embedding
 */
void transformer::train(int& promptCount, int& currentTokenCount, int& blockCount, std::vector<float>& expected) 
{
    // for first block
    if(blockCount == 1 && (currentTokenCount < CONTEXT_WIN-1)) {
        // compute the kdotQ for each head of the block
        computeKdotQs(promptCount, currentTokenCount, blockCount, isSelf, inTraining);
        int i = 0;
        forward(blockCount, currentTokenCount, promptCount);
        while (i <= epochs) {
            computeOutput(otok, embeddings, vocabsize, indexForToken);
            if((errorofv(t[0].EH, expected) < 0.01) || tokens[indexForToken] == expString) {
                input[currentTokenCount] = t[0].EH;
                break;
            }
            // if error is not corrected even after epochs, then increase epochs
            if(errorofv(t[0].EH, expected) > 0.01 && i == epochs) {
                epochs += 10;
            }
            backward(expected);
            forward(blockCount, currentTokenCount, promptCount);
            i++;
        }
        trainCount++;
        epochCount += i;
        error += errorofv(t[0].EH, expected);
        currentTokenCount += 1;
        if(currentTokenCount == CONTEXT_WIN) {
            blockCount += 1;
        }
    }
    // for next blocks
    if(blockCount > 1 && currentTokenCount > CONTEXT_WIN) {
        // compute the KdotQ for each head of block using EVs of previous blocks
        computeKdotQs(promptCount, currentTokenCount, blockCount, isSelf, inTraining);
        forward(blockCount, currentTokenCount, promptCount);
        int i = 0;
        while (i < epochs) {
            if(errorofv(t[blockCount-1].EH, expected) < 0.01 || tokens[indexForToken] == expString) {
                input[currentTokenCount] = t[blockCount-1].EH;
                break;
            }
            // if error is not corrected even after epochs, then increase epochs
            if(errorofv(t[blockCount-1].EH, expected) > 0.01 && i == epochs) {
                epochs += 10;
            }
            i++;
            backward(expected, blockCount);
            forward(blockCount, currentTokenCount, promptCount);
        }
        trainCount++;
        epochCount += i;
        error += errorofv(t[blockCount-1].EH, expected);
        currentTokenCount += 1;
        if(currentTokenCount % CONTEXT_WIN == 0) {
            blockCount += 1;
        }
    }
}


/**
 * @brief train the transformer (single continuous sentece, paragraphs and passages)
 * @param sentence sentence to train
 */
void transformer::train(std::vector<std::vector<float>>& sentence) {
    // compute KdotQ for first element
    input[0] = sentence[0];
    promptCount = 1;
    blockCount = 1;
    // keep this in a loop and train for each token in the sentence starting from second token
    for(int i = 1; i < sentence.size(); i++) {
        // 
        if(blockCount == 1 && (currentTokenCount < CONTEXT_WIN-1)) {
            // compute the kdotQ for each head of the block
            computeKdotQs(promptCount, currentTokenCount, blockCount, isSelf, inTraining);
            int i = 0;
            forward(blockCount, currentTokenCount, promptCount);
            while (i <= epochs) {
                computeOutput(otok, embeddings, vocabsize, indexForToken);
                if((errorofv(t[0].EH, sentence[i]) < 0.01) || tokens[indexForToken] == expString) {
                    input[currentTokenCount] = t[0].EH;
                    break;
                }
                // if error is not corrected even after epochs, then increase epochs
                if(errorofv(t[0].EH, sentence[i]) > 0.01 && i == epochs) {
                    epochs += 10;
                }
                backward(sentence[i]);
                forward(blockCount, currentTokenCount, promptCount);
                i++;
            }
            trainCount++;
            epochCount += i;
            error += errorofv(t[0].EH, sentence[i]);
            currentTokenCount += 1;
            if(currentTokenCount == CONTEXT_WIN) {
                blockCount += 1;
            }
        }
        if(blockCount > 1 && currentTokenCount > CONTEXT_WIN) {
            // compute the KdotQ for each head of block using EVs of previous blocks
            computeKdotQs(promptCount, currentTokenCount, blockCount, isSelf, inTraining);
            forward(blockCount, currentTokenCount, promptCount);
            int i = 0;
            while (i < epochs) {
                if(errorofv(t[blockCount-1].EH, sentence[i]) < 0.01 || tokens[indexForToken] == expString) {
                    input[currentTokenCount] = t[blockCount-1].EH;
                    break;
                }
                // if error is not corrected even after epochs, then increase epochs
                if(errorofv(t[blockCount-1].EH, sentence[i]) > 0.01 && i == epochs) {
                    epochs += 10;
                }
                i++;
                backward(sentence[i], blockCount);
                forward(blockCount, currentTokenCount, promptCount);
            }
            trainCount++;
            epochCount += i;
            error += errorofv(t[blockCount-1].EH, sentence[i]);
            currentTokenCount += 1;
            if(currentTokenCount % CONTEXT_WIN == 0) {
                blockCount += 1;
            }
        }
    }
}


/**
 * @brief train the transformer for prompt and response (single prompt and response)
 * @param prompt prompt to model
 * @param response response from model
 */
void transformer::train(std::vector<std::vector<float>>& prompt, std::vector<std::vector<float>>& response) {
    // for first prompt
    if(currentTokenCount == 0) {
        // token embedding should be divided for specific sizes
        // for smaller prompts
        if(prompt.size() < CONTEXT_WIN) {
            for(int i = 0; i < prompt.size(); i++) {
                // add excess to tokForBlock
                tokenEmbed[i] = prompt[i];
            }
            for(int i = 0; i < x; i++) {
                for(int j = 0; j < y; j++) {
                    for(int k = 0; k < prompt.size(); k++) {
                        t[0].b[i][j].EV[k] = tokenEmbed[k];
                    }
                }
            }
        }
        promptCount = prompt[0].size();
        for(int i = 0; i < response.size(); i++) {
            // for first block
            if(blockCount == 1 && (currentTokenCount < CONTEXT_WIN)) {
                // compute the kdotQ for each head of the block
                computeKdotQs(promptCount, currentTokenCount, blockCount, isSelf, inTraining);
                int j = 0;
                forward(blockCount, currentTokenCount, promptCount);
                while (j <= epochs) {
                    computeOutput(otok, embeddings, vocabsize, indexForToken);
                    if((errorofv(t[0].EH, response[i]) < 0.01) || tokens[indexForToken] == expString) {
                        input[currentTokenCount] = t[0].EH;
                        break;
                    }
                    // if error is not corrected even after epochs, then increase epochs
                    if(errorofv(t[0].EH, response[i]) > 0.01 && j == epochs) {
                        epochs += 10;
                    }
                    backward(response[i]);
                    forward(blockCount, currentTokenCount, promptCount);
                    j++;
                }
                trainCount++;
                epochCount += j;
                error += errorofv(t[0].EH, response[i]);
                currentTokenCount += 1;
                if(currentTokenCount == CONTEXT_WIN) {
                    blockCount += 1;
                }
            }
            // for next blocks
            if(blockCount > 1 && currentTokenCount > CONTEXT_WIN) {
                // compute the KdotQ for each head of block using EVs of previous blocks
                computeKdotQs(promptCount, currentTokenCount, blockCount, isSelf, inTraining);
                forward(blockCount, currentTokenCount, promptCount);
                int j = 0;
                while (j < epochs) {
                    if(errorofv(t[blockCount-1].EH, response[i]) < 0.01 || tokens[indexForToken] == expString) {
                        input[currentTokenCount] = t[blockCount-1].EH;
                        break;
                    }
                    // if error is not corrected even after epochs, then increase epochs
                    if(errorofv(t[blockCount-1].EH, response[i]) > 0.01 && j == epochs) {
                        epochs += 10;
                    }
                    j++;
                    backward(response[i], blockCount);
                    forward(blockCount, currentTokenCount, promptCount);
                }
                trainCount++;
                epochCount += j;
                error += errorofv(t[blockCount-1].EH, response[i]);
                currentTokenCount += 1;
                if(currentTokenCount % CONTEXT_WIN == 0) {
                    blockCount += 1;
                }
            }
        }
    }
    // for next prompts
    else {
        //
    }
}
