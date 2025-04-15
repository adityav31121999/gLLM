
// transformer training
#include "include/attention.hpp"
#include "include/block.hpp"
#include "include/transformer.hpp"

/**
 * @brief train the transformer for next token prediction (single token training)
 * @param promptCount number of tokens in the prompt
 * @param currentTokenCount number of tokens in the full context
 * @param blockCount current block in full context
 * @param expected expected token embedding
 * @param expString expected token
 */
void transformer::train(int& promptCount, int& currentTokenCount, int& blockCount, std::vector<float>& expected,
    std::string& expString) 
{
    // for first block
    if(blockCount == 1 && currentTokenCount < CONTEXT_WIN) {
        // compute the kdotQ for each head of the block
        for(int i = 0; i < x; i++) {
            for(int j = 0; j < y; j++) {
                for(int k = 0; k < currentTokenCount; k++) {
                    computeKorQ(tokenEmbed[k], t[0].b[i][j].MQ, t[0].b[i][j].Q[k]);   // Queries
                    computeKorQ(tokenEmbed[k], t[0].b[i][j].MK, t[0].b[i][j].K[k]);   // Keys
                    t[0].b[i][j].EV[k] = tokenEmbed[k];     // vertical retention vectors
                }
            }
        }
        computeKdotQs(promptCount, currentTokenCount, blockCount, isSelf, inTraining);
        // train from here
        forward(blockCount, currentTokenCount, promptCount);
        int i = 0;
        while (i <= epochs) {
            computeOutput(otok, embeddings, vocabsize, indexForToken);
            if((errorofv(t[0].EH, expected) < 0.01) || tokens[indexForToken] == expString) {
                // input[currentTokenCount] = t[0].EH;
                // tokenEmbed[currentTokenCount] = otok;
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
    else if(blockCount > 1 && currentTokenCount >= CONTEXT_WIN) {
        // assign tokForBlock embeddings
        for(int i = 0; i < currentTokenCount; i++) {
            t[blockCount-1].tokForBlock[i] = tokenEmbed[CONTEXT_WIN*(blockCount-2) + i];
        }
        // compute the KdotQ for each head of block using EVs of previous blocks
        for(int i = 0; i < x; i++) {
            for(int j = 0; j < y; j++) {
                for(int k = 0; k < currentTokenCount; k++) {
                    // blockCount-1 refers to index of current block
                    computeKorQ(t[blockCount-2].b[i][j].EV[k], t[blockCount-1].b[i][j].MQ, t[blockCount-1].b[i][j].Q[k]);   // Queries
                    computeKorQ(t[blockCount-1].tokForBlock[k], t[blockCount-1].b[i][j].MK, t[blockCount-1].b[i][j].K[k]);   // Keys
                    t[0].b[i][j].EV[k] = tokenEmbed[k];     // vertical retention vectors
                }
            }
        }
        computeKdotQs(promptCount, currentTokenCount, blockCount, isSelf, inTraining);
        // train from here
        forward(blockCount, currentTokenCount, promptCount);
        int i = 0;
        while (i < epochs) {
            computeOutput(otok, embeddings, vocabsize, indexForToken);
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
 * @brief train the transformer on sentences (single continuous sentece, paragraphs and passages)
 * @param sentence token embedding of sentence
 * @param rString sentence tokens
 */
void transformer::train(std::vector<std::vector<float>>& sentence, std::vector<std::string>& rString)
{
    // constraints for training data
    if(sentence.size() > FULL_CONTEXT) {
        throw std::runtime_error("Sentence size should not exceed FULL_CONTEXT");
    }
    if(sentence.size() != rString.size()) {
        throw std::runtime_error("Sentence embeddings and sentence strings must be non-empty and have the same size.");
    }
    // compute KdotQ for first element
    tokenEmbed[0] = sentence[0];
    promptCount = 1;
    blockCount = 1;
    // keep this in a loop and train for each token in the sentence starting from second token
    for(int i = 1; i < sentence.size(); i++) {
        // first block
        if(blockCount == 1 && (currentTokenCount < CONTEXT_WIN-1)) {
            // get keys and queries for KdotQ
            for(int j = 0; j < x; j++) {
                for(int k = 0; k < y; k++) {
                    computeKorQ(sentence[i], t[0].b[j][k].MQ, t[0].b[j][k].Q[i]);   // Queries
                    computeKorQ(sentence[i], t[0].b[j][k].MK, t[0].b[j][k].K[i]);   // Keys
                    t[0].b[j][k].EV[j] = sentence[i];
                }
            }
            // compute KdotQ for all heads of this block
            computeKdotQs(promptCount, currentTokenCount, blockCount, isSelf, inTraining);
            int j = 0;
            // TRAIN FOR SENTENCE
            forward(blockCount, currentTokenCount, promptCount);
            while (j <= epochs) {
                computeOutput(otok, embeddings, vocabsize, indexForToken);
                if((errorofv(otok, sentence[i]) < 0.01) || tokens[indexForToken] == rString[i]) {
                    // tokenEmbed[currentTokenCount] = otok;
                    // tokenEmbed[currentTokenCount] = sentence[i];
                    // tokens[i] = rString[i];
                    break;
                }
                // if error is not corrected even after epochs, then increase epochs
                if(errorofv(t[0].EH, sentence[i]) > 0.01 && j == epochs) {
                    epochs += 10;
                }
                backward(sentence[i]);
                forward(blockCount, currentTokenCount, promptCount);
                j++;
            }
            // update variables
            trainCount++;
            epochCount += j;
            error += errorofv(t[0].EH, sentence[i]);
            currentTokenCount += 1;
            if(currentTokenCount == CONTEXT_WIN) {
                blockCount += 1;
            }
        }
        // if local context of previous block is full, shift to next block directly from here,
        // no need to end the loop and start new one
        else if(blockCount > 1 && currentTokenCount > CONTEXT_WIN) {
            // get number of tokens already in this block
            int z = currentTokenCount - (blockCount-1)*CONTEXT_WIN;
            // add new token embedding to tokForBlock for this block
            t[blockCount-1].tokForBlock[z] = sentence[currentTokenCount-1];
            // compute keys and queries
            for(int j = 0; j < x; j++) {
                for(int k = 0; k < y; k++) {
                    // compute Key and Query for this iteration
                    computeKorQ(t[blockCount-2].b[j][k].EV[l], t[blockCount-1].b[j][k].MQ, t[blockCount-1].b[j][k].Q[z]);   // Queries
                    computeKorQ(t[blockCount-1].tokForBlock[l], t[blockCount-1].b[j][k].MK, t[blockCount-1].b[j][k].K[z]);   // Keys
                    t[0].b[j][k].EV[l] = sentence[i];
                }
            }
            // compute the KdotQ for each head of block using EVs of previous blocks
            computeKdotQs(promptCount, currentTokenCount, blockCount, isSelf, inTraining);
            // TRAIN FOE SENTENCE
            forward(blockCount, currentTokenCount, promptCount);
            int j = 0;
            while (j < epochs) {
                computeOutput(otok, embeddings, vocabsize, indexForToken);
                if(errorofv(otok, sentence[i]) < 0.01 || tokens[indexForToken] == rString[i]) {
                    // tokenEmbed[currentTokenCount] = otok;
                    // tokenEmbed[currentTokenCount] = sentence[i];
                    // tokens[i] = rString[i];
                    break;
                }
                // if error is not corrected even after epochs, then increase epochs
                if(errorofv(t[blockCount-1].EH, sentence[i]) > 0.01 && j == epochs) {
                    epochs += 10;
                }
                j++;
                backward(sentence[i], blockCount);
                forward(blockCount, currentTokenCount, promptCount);
            }
            // update variables
            trainCount++;
            epochCount += j;
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
 * @param prompt prompt token embeddings
 * @param response response token embeddings
 * @param rString tokens of response
 */
void transformer::train(std::vector<std::vector<float>>& prompt, std::vector<std::vector<float>>& response, std::vector<std::string>& rString) 
{
    // prompt must not be empty
    if (prompt.empty()) {
        throw std::runtime_error("Initial prompt cannot be empty!");
    }
    // prompt size should not exceed threshold (one fourth of context window)
    if(prompt.size() > PROMPT_THRESHOLD) {
        throw std::runtime_error("Pompt size should not exceed CONTEXT_WIN!");
    }
    // Basic validation
    if (response.empty() || response.size() != rString.size()) {
        throw std::runtime_error("Response embeddings and response strings must be non-empty and have the same size!");
    }
    
    // for first prompt
    if(currentTokenCount == 0) {
        // token embedding should be divided for specific sizes
        // for smaller prompts
        if(prompt.size() < CONTEXT_WIN) {
            // add prompt to continuous tokenEmbed
            for(int i = 0; i < prompt.size(); i++) {
                tokenEmbed[i] = prompt[i];
            }
            // add prompt to vertical retention vectors
            for(int i = 0; i < x; i++) {
                for(int j = 0; j < y; j++) {
                    for(int k = 0; k < prompt.size(); k++) {
                        t[0].b[i][j].EV[k] = prompt[k];
                    }
                }
            }
        }
        // set current token count and prompt count
        currentTokenCount += prompt.size();
        promptCount = prompt.size();
        // TRAIN FOR RESPONSE
        for(int i = 0; i < response.size(); i++) {
            // for ith blocks
            if(blockCount > 1 && currentTokenCount > CONTEXT_WIN) {
                // set tokens for tokForBlock to calculate the KdotQ for this block and response training
                for(int i = 0; i < CONTEXT_WIN; i++) {
                    t[blockCount-1].tokForBlock[i] = tokenEmbed[(blockCount-2)*CONTEXT_WIN - i];
                }
            }
            // compute the KdotQ for each head of block using EVs of previous blocks
            computeKdotQs(promptCount, currentTokenCount, blockCount, isSelf, inTraining);
            forward(blockCount, currentTokenCount, promptCount);
            int j = 0;
            while (j < epochs) {
                computeOutput(t[blockCount-1].EH, embeddings, vocabsize, indexForToken);
                if(errorofv(t[blockCount-1].EH, response[i]) < 0.01 || tokens[indexForToken] == rString[i]) {
                    tokenEmbed[currentTokenCount] = t[blockCount-1].EH;
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
            // update variables
            trainCount++;
            epochCount += j;
            error += errorofv(t[blockCount-1].EH, response[i]);
            currentTokenCount += 1;
            if(currentTokenCount % CONTEXT_WIN == 0) {
                blockCount += 1;
            }
        }
    }
    // for next prompts
    else {
        // available tokens in local context
        int c = currentTokenCount - (blockCount - 1)*CONTEXT_WIN;
        if(prompt.size() <= c) {
            // add prompt to tokenEmbed and tokForBlock
            for(int i = 0; i < prompt.size(); i++) {
                tokenEmbed[currentTokenCount + i] = prompt[i];
                t[blockCount - 1].tokForBlock[c + i] = prompt[i];
            }
            // add prompts to vertical retention vectors
            for(int i = 0; i < x; i++) {
                for(int j = 0; j < y; j++) {
                    for(int k = 0; k < prompt.size(); k++) {
                        t[blockCount - 1].b[i][j].EV[c + k] = prompt[k];
                    }
                }
            }
            promptCount = prompt.size();
            currentTokenCount += prompt.size();
            if(currentTokenCount % CONTEXT_WIN == 0) {
                blockCount += 1;
            }
        }
        // for prompt size larger than available tokens in local context
        else if(prompt.size() > c) {
            // available tokens in local context
            int m1 = prompt.size() - c;
            for(int i = 0; i < m1; i++) {
                tokenEmbed[currentTokenCount + i] = prompt[i];
            }
            currentTokenCount += m1;
            promptCount = m1;
            computeKdotQs(promptCount, currentTokenCount, blockCount, isSelf, inTraining);
            // number of prompt tokens in next block
            int m2 = prompt.size() - m1;
            for(int i = 0; i < m2; i++) {
                t[blockCount].tokForBlock[c + i] = prompt[i];
            }
            // add prompts to tokenEmbed
            for(int i = 0; i < prompt.size(); i++) {
                tokenEmbed[currentTokenCount + i] = prompt[i];
            }
            currentTokenCount += m2;
            promptCount = m2;
            blockCount += 1;
        }
        // TRAIN FOR RESPONSE
        for(int i = 0; i < response.size(); i++) {
            // for first block
            // compute the KdotQ for each head of block using EVs of previous blocks
            computeKdotQs(promptCount, currentTokenCount, blockCount, isSelf, inTraining);
            forward(blockCount, currentTokenCount, promptCount);
            int j = 0;
            while (j < epochs) {
                if(errorofv(t[blockCount-1].EH, response[i]) < 0.01 || tokens[indexForToken] == rString[i]) {
                    tokenEmbed[currentTokenCount] = t[blockCount-1].EH;
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


/**
 * @brief train transformers for continuous chats
 * @param prompts all prompts
 * @param responses token embeddings all responses to the prompts
 * @param rString tokens of responses
 */
void transformer::train(std::vector<std::vector<std::vector<float>>>& prompts, std::vector<std::vector<std::vector<float>>>& responses, 
                        std::vector<std::vector<std::string>>& rString) 
{
    if(prompts.size() != responses.size() || responses.size() != rString.size()) {
        throw std::runtime_error("Rows of all the vectors must match");
    }
    int sum = 0;
    for(int i = 0; i < prompts.size(); i++) {
        sum += prompts[i].size() + responses[i].size();
    }
    if(sum > FULL_CONTEXT) {
        throw std::runtime_error("TOTAL TOKENS SHOULD NOT EXCEED THE FULL CONTEXT");
    }
    // TRAIN FOR CHAT
    for(int i = 0; i < prompts.size(); i++) {
        train(prompts[i], responses[i], rString[i]);
    }
}
