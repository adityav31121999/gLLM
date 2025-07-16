
#ifdef USE_CPU
#include <vector>
#include <stdexcept>
#include <iostream>
#include <string>
#include <cmath>
#include <maths.hpp>
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
    float current_Error = 0.0f;
    float prev_Error = 0.0f;
    // for first block
    if(blockCount == 1 && currentTokenCount < CONTEXT_WIN) {
        computeKdotQs(promptCount, currentTokenCount, blockCount, isSelf, inTraining);
        // train from here
        forward(blockCount, currentTokenCount, promptCount);
        int j = 0;
        prev_Error = 0.0f;
        while (j <= epochs) {
            computeOutput(otok, embeddings, vocabsize, indexForToken);
            current_Error = crossEntropy(otok, expected);
            if (this->tokens[indexForToken] == expString && this->tokens[indexForToken] != "INVALID_INDEX") {
                if(this->tokens[indexForToken] == "@#0") {
                    std::cout << "--------------->>>>>>>>>>>>> To next LINE >>>>>>>>>>>>>>>>-------------" << std::endl;
                }
                else {
                    std::cout << "--------------------- To next token ------------->>>>>>>>>>>>>>>>>" << std::endl;
                    totalLearning += learning;
                    break;
                }
            }
            else if (j == this->epochs - 1) {
                if (this->tokens[indexForToken] != expString) {
                    std::cout << "Increasing Epoch Count by 10 '-'" << std::endl;
                    this->epochs += 10;
                }
            }
            // update learning rate starting from second epoch and specific conditions
            if(j > 0) {
                if(current_Error <= prev_Error) {
                    if(j <= 6)   
                        learning *= 1.05;
                    else if (j % 6 == 0)
                        learning *= (1 + (j/6)*0.05);
                }
                else {
                    if(j <= 6)   
                        learning *= 0.95;
                    else if (j % 6 == 0)
                        learning *= (1 - (j/6)*0.05);
                }
            }
            backward(expected);
            forward(blockCount, currentTokenCount, promptCount);
            prev_Error = current_Error;
            j++;
        }
        trainCount++;
        epochCount += j;
        error += crossEntropy(otok, expected);
        currentTokenCount += 1;
        if(currentTokenCount == CONTEXT_WIN) {
            blockCount += 1;
        }
    }
    // for next blocks
    else if(blockCount > 1 && currentTokenCount >= CONTEXT_WIN) {
        computeKdotQs(promptCount, currentTokenCount, blockCount, isSelf, inTraining);
        // train from here
        forward(blockCount, currentTokenCount, promptCount);
        int j = 0;
        prev_Error = 0.0f;
        while (j < epochs) {
            computeOutput(otok, embeddings, vocabsize, indexForToken);
            current_Error = crossEntropy(otok, expected);
            if (this->tokens[indexForToken] == expString && this->tokens[indexForToken] != "INVALID_INDEX") {
                if(this->tokens[indexForToken] == "@#0") {
                    std::cout << "--------------->>>>>>>>>>>>> To next LINE >>>>>>>>>>>>>>>>-------------" << std::endl;
                }
                else {
                    std::cout << "--------------------- To next token ------------->>>>>>>>>>>>>>>>>" << std::endl;
                    totalLearning += learning;
                    break;
                }
            }
            else if (j == this->epochs - 1) {
                if (this->tokens[indexForToken] != expString) {
                    std::cout << "Increasing Epoch Count by 10 '-'" << std::endl;
                    this->epochs += 10;
                }
            }
            // update learning rate starting from second epoch and specific conditions
            if(j > 0) {
                if(current_Error <= prev_Error) {
                    if(j <= 6)   
                        learning *= 1.05;
                    else if (j % 6 == 0)
                        learning *= (1 + (j/6)*0.05);
                }
                else {
                    if(j <= 6)   
                        learning *= 0.95;
                    else if (j % 6 == 0)
                        learning *= (1 - (j/6)*0.05);
                }
            }
            backward(expected, blockCount);
            forward(blockCount, currentTokenCount, promptCount);
            j++;
        }
        trainCount++;
        epochCount += j;
        error += crossEntropy(otok, expected);
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
    setRow(tokenEmbed, 0, sentence[0]);
    promptCount = 1;
    blockCount = 1;
    float current_Error = 0.0f;
    float prev_Error = 0.0f;
    // keep this in a loop and train for each token in the sentence starting from second token
    for(int token_idx_in_sentence = 1; token_idx_in_sentence < sentence.size(); token_idx_in_sentence++) {
        // first block
        if(blockCount == 1 && currentTokenCount < CONTEXT_WIN) {
            // get keys and queries for KdotQ
            for(int j = 0; j < x; j++) {
                for(int k = 0; k < y; k++) {
                    std::vector<float> q_output_vec(this->h);
                    computeKorQ(sentence[token_idx_in_sentence], t[0].b[j][k].MQ, q_output_vec);
                    setRow(t[0].b[j][k].Q, token_idx_in_sentence, q_output_vec);

                    std::vector<float> k_output_vec(this->h);
                    computeKorQ(sentence[token_idx_in_sentence], t[0].b[j][k].MK, k_output_vec);
                    setRow(t[0].b[j][k].K, token_idx_in_sentence, k_output_vec);

                    setRow(t[0].b[j][k].EV, token_idx_in_sentence, sentence[token_idx_in_sentence]);
                }
            }
            // compute KdotQ for all heads of this block
            computeKdotQs(promptCount, currentTokenCount, blockCount, isSelf, inTraining);
            int j_epoch = 0; // Renamed to avoid conflict with outer loop variable j
            // TRAIN FOR SENTENCE
            forward(blockCount, currentTokenCount, promptCount);
            prev_Error = 0.0f;
            while (j_epoch <= epochs) {
                computeOutput(otok, embeddings, vocabsize, indexForToken);
                current_Error = crossEntropy(otok, sentence[token_idx_in_sentence]);
                if (this->tokens[indexForToken] == rString[token_idx_in_sentence] && this->tokens[indexForToken] != "INVALID_INDEX") {
                    if(this->tokens[indexForToken] == "@#0") {
                        std::cout << "--------------->>>>>>>>>>>>> To next LINE >>>>>>>>>>>>>>>>-------------" << std::endl;
                    }
                    else {
                        std::cout << "--------------------- To next token ------------->>>>>>>>>>>>>>>>>" << std::endl;
                        totalLearning += learning;
                        break;
                    }
                }
                else if (j_epoch == this->epochs - 1) {
                    if (this->tokens[indexForToken] != rString[token_idx_in_sentence]) {
                        std::cout << "Increasing Epoch Count by 10 '-'" << std::endl;
                        this->epochs += 10;
                    }
                }
                // update learning rate starting from second epoch and specific conditions
                if(j_epoch > 0) {
                    if(current_Error <= prev_Error) {
                        if(j_epoch <= 6)   
                            learning *= 1.05;
                        else if (j_epoch % 6 == 0)
                            learning *= (1 + (j_epoch/6)*0.05);
                    }
                    else {
                        if(j_epoch <= 6)   
                            learning *= 0.95;
                        else if (j_epoch % 6 == 0)
                            learning *= (1 - (j_epoch/6)*0.05);
                    }
                }
                backward(sentence[token_idx_in_sentence]);
                forward(blockCount, currentTokenCount, promptCount);
                prev_Error = current_Error;
                j_epoch++;
            }
            // update variables
            trainCount++;
            epochCount += j_epoch;
            error += errorofv(otok, sentence[token_idx_in_sentence]);
            currentTokenCount += 1;
            if(currentTokenCount == CONTEXT_WIN) {
                blockCount += 1;
            }
        }
        // if local context of previous block is full, shift to next block directly from here,
        // no need to end the loop and start new one
        else if(blockCount > 1 && currentTokenCount > CONTEXT_WIN) {
            // token_idx_in_block: index of the current token within the current block's window
            int token_idx_in_block = currentTokenCount % CONTEXT_WIN; // Or currentTokenCount - (blockCount-1)*CONTEXT_WIN if currentTokenCount is global
            if (token_idx_in_block == 0 && currentTokenCount >= CONTEXT_WIN) token_idx_in_block = CONTEXT_WIN -1; // If it's the last token of a full window

            // add new token embedding to tokForBlock for this block
            setRow(t[blockCount-1].tokForBlock, token_idx_in_block, sentence[currentTokenCount-1]); // sentence[currentTokenCount-1] is the latest input token
            // compute keys and queries
            for(int j = 0; j < x; j++) {
                for(int k = 0; k < y; k++) {
                    // compute Key and Query for this iteration
                    std::vector<float> prev_block_ev_row = getRow(t[blockCount-2].b[j][k].EV, token_idx_in_block);
                    std::vector<float> q_output_vec(this->h);
                    computeKorQ(prev_block_ev_row, t[blockCount-1].b[j][k].MQ, q_output_vec);
                    setRow(t[blockCount-1].b[j][k].Q, token_idx_in_block, q_output_vec);

                    std::vector<float> current_block_token_row = getRow(t[blockCount-1].tokForBlock, token_idx_in_block);
                    std::vector<float> k_output_vec(this->h);
                    computeKorQ(current_block_token_row, t[blockCount-1].b[j][k].MK, k_output_vec);
                    setRow(t[blockCount-1].b[j][k].K, token_idx_in_block, k_output_vec);

                    setRow(t[blockCount-1].b[j][k].EV, token_idx_in_block, sentence[token_idx_in_sentence]); // Target token for EV? Or input?
                }
            }
            // compute the KdotQ for each head of block using EVs of previous blocks
            computeKdotQs(promptCount, currentTokenCount, blockCount, isSelf, inTraining);
            // TRAIN FOE SENTENCE
            forward(blockCount, currentTokenCount, promptCount);
            int j_epoch = 0; // Renamed
            while (j_epoch < epochs) {
                computeOutput(otok, embeddings, vocabsize, indexForToken);
                if (this->tokens[indexForToken] == rString[token_idx_in_sentence] && this->tokens[indexForToken] != "INVALID_INDEX") {
                    if(this->tokens[indexForToken] == "@#0") {
                        std::cout << "--------------->>>>>>>>>>>>> To next LINE >>>>>>>>>>>>>>>>-------------" << std::endl;
                    }
                    else {
                        std::cout << "--------------------- To next token ------------->>>>>>>>>>>>>>>>>" << std::endl;
                        totalLearning += learning;
                        break;
                    }
                }
                else if (j_epoch == this->epochs - 1) {
                    if (this->tokens[indexForToken] != rString[token_idx_in_sentence]) {
                        std::cout << "Increasing Epoch Count by 10 '-'" << std::endl;
                        this->epochs += 10;
                    }
                }
                // update learning rate starting from second epoch and specific conditions
                if(j_epoch > 0) {
                    if(current_Error <= prev_Error) {
                        if(j_epoch <= 6)   
                            learning *= 1.05;
                        else if (j_epoch % 6 == 0)
                            learning *= (1 + (j_epoch/6)*0.05);
                    }
                    else {
                        if(j_epoch <= 6)   
                            learning *= 0.95;
                        else if (j_epoch % 6 == 0)
                            learning *= (1 - (j_epoch/6)*0.05);
                    }
                }
                j_epoch++;
                backward(sentence[token_idx_in_sentence], blockCount);
                forward(blockCount, currentTokenCount, promptCount);
            }
            // update variables
            trainCount++;
            epochCount += j_epoch;
            error += errorofv(otok, sentence[token_idx_in_sentence]);
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
void transformer::train(std::vector<std::vector<float>>& prompt, std::vector<std::vector<float>>& response, 
    std::vector<std::string>& rString) 
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
    
    int initialTokCount = this->currentTokenCount;
    float prev_Error = 0.0f;
    float current_Error = 0.0f;

    // Keys and Queries of prompts
    if(blockCount == 1) {
        for(int i_pa = 0; i_pa < x; i_pa++) {
            for(int j_head = 0; j_head < y; j_head++) {
                for(int k = 0; k < prompt.size(); k++) {
                    // make queries using compute KorQ: t[0].b[i][j].Q[currentTokenCount%CONTEXT_WIN] = prompt(i) * t[0].b[i][j].MQ
                    t[0].b[i_pa][j_head].Q(initialTokCount%CONTEXT_WIN + k) = dot(prompt[i_pa], t[0].b[i_pa][j_head].MQ);
                    // make keys using compute KorQ: t[0].b[i][j].K[currentTokenCount%CONTEXT_WIN] = prompt(i) * t[0].b[i][j].MK
                    t[0].b[i_pa][j_head].K(initialTokCount%CONTEXT_WIN + k) = dot(prompt[i_pa], t[0].b[i_pa][j_head].MK);
                }
            }
        }
    }
    else {
        for(int i_pa = 0; i_pa < x; i_pa++) {
            for(int j_head = 0; j_head < y; j_head++) {
                for(int k = 0; k < prompt.size(); k++) {
                    // make queries using compute KorQ: t[0].b[i][j].Q[currentTokenCount%CONTEXT_WIN] = prompt(i) * t[0].b[i][j].MQ
                    t[blockCount-1].b[i_pa][j_head].Q(initialTokCount%CONTEXT_WIN + k) = dot(prompt[i_pa], t[0].b[i_pa][j_head].MQ);
                }
                for(int k = 0; k < CONTEXT_WIN; k++) {
                    // make queries using compute KorQ: t[0].b[i][j].K[i] = prompt(i) * t[blockCount-1].b[i][j].MK
                    t[blockCount-1].b[i_pa][j_head].K(k) = dot(t[blockCount-1].b[i_pa][j_head].EV(k), t[0].b[i_pa][j_head].MQ);
                }
            }
        }
    }

    int resCount = 0;

    for(int i = 0; i < response.size(); i++) {
        // compute the KdotQ for each head of block using EVs of previous blocks
        computeKdotQs(promptCount, currentTokenCount, blockCount, isSelf, inTraining);
        forward(blockCount, currentTokenCount, promptCount);

        int j_epoch = 0;
        prev_Error = 0.0f;
        while (j_epoch < epochs) {
            computeOutput(otok, embeddings, vocabsize, indexForToken);
            current_Error = crossEntropy(otok, response[i]);
            std::string predicted_token_str = (indexForToken >= 0 && (indexForToken < vocabsize)) ? tokens[indexForToken] : "INVALID_INDEX";

            std::cout << "Computed token is -> " << predicted_token_str << " (index: " << indexForToken << ") | with BCE error " << current_Error << " | MAE Error " << MAE(otok, response[i]) << std::endl;
            if (predicted_token_str == rString[i] && predicted_token_str != "INVALID_INDEX") {
                if(predicted_token_str == "@#0"){
                    std::cout << "--------------->>>>>>>>>>>>> To next LINE >>>>>>>>>>>>>>>>-------------" << std::endl;
                }
                else {
                    std::cout << "--------------------- To next token ------------->>>>>>>>>>>>>>>>>" << std::endl;
                    break;
                }
            }
            else if (j_epoch == this->epochs - 1) {
                if (predicted_token_str != rString[i]) {
                    std::cout << "Increasing Epoch Count by 10 '-'" << std::endl;
                    this->epochs += 10;
                }
            }

            j_epoch++;
            backward(response[i], blockCount);
            // Keys and Queries of both prompts and response
            if(blockCount == 1) {
                for(int i_pa = 0; i_pa < x; i_pa++) {
                    for(int j_head = 0; j_head < y; j_head++) {
                        for(int k = 0; k < prompt.size(); k++) {
                            // make queries using compute KorQ: t[0].b[i][j].Q[currentTokenCount%CONTEXT_WIN] = prompt(i) * t[0].b[i][j].MQ
                            t[0].b[i_pa][j_head].Q(initialTokCount%CONTEXT_WIN + k) = dot(prompt[i_pa], t[0].b[i_pa][j_head].MQ);
                            // make keys using compute KorQ: t[0].b[i][j].K[currentTokenCount%CONTEXT_WIN] = prompt(i) * t[0].b[i][j].MK
                            t[0].b[i_pa][j_head].K(initialTokCount%CONTEXT_WIN + k) = dot(prompt[i_pa], t[0].b[i_pa][j_head].MK);
                        }
                        if(resCount > 0) {
                            for(int k = 0; k < resCount; k++) {
                                // make queries using compute KorQ: t[0].b[i][j].Q[currentTokenCount%CONTEXT_WIN + prompt.size() + k] = response(i) * t[0].b[i][j].MQ
                                t[0].b[m][n].Q(initialTokCount%CONTEXT_WIN + prompt.size() + k) = dot(response[k], t[0].b[m][n].MQ);
                                // make keys using compute KorQ: t[0].b[i][j].K[currentTokenCount%CONTEXT_WIN + prompt.size() + k] = response(i) * t[0].b[i][j].MK
                                t[0].b[m][n].K(initialTokCount%CONTEXT_WIN + prompt.size() + k) = dot(response[k], t[0].b[m][n].MK);
                            }
                        }
                    }
                }
            }
            else {
                for(int i_pa = 0; i_pa < x; i_pa++) {
                    for(int j_head = 0; j_head < y; j_head++) {
                        for(int k = 0; k < prompt.size(); k++) {
                            // make queries using compute KorQ: t[0].b[i][j].Q[currentTokenCount%CONTEXT_WIN] = prompt(i) * t[0].b[i][j].MQ
                            t[blockCount-1].b[i_pa][j_head].Q(initialTokCount%CONTEXT_WIN + k) = dot(prompt[i_pa], t[0].b[i_pa][j_head].MQ);
                        }
                        for(int k = 0; k < CONTEXT_WIN; k++) {
                            // make queries using compute KorQ: t[0].b[i][j].K[i] = prompt(i) * t[blockCount-1].b[i][j].MK
                            t[blockCount-1].b[i_pa][j_head].K(k) = dot(t[blockCount-1].b[i_pa][j_head].EV(k), t[0].b[i_pa][j_head].MQ);
                        }
                        if(resCount > 0) {
                            for(int k = 0; k < resCount; k++) {
                                // make queries using compute KorQ: t[blockCount-1].b[i][j].Q[currentTokenCount%CONTEXT_WIN + prompt.size() + k] = response(i) * t[0].b[i][j].MQ
                                t[blockCount-1].b[m][n].Q(initialTokCount%CONTEXT_WIN + prompt.size() + k) = dot(response[k], t[blockCount-1].b[m][n].MQ);
                                // make keys using compute KorQ: t[blockCount-1].b[i][j].K[currentTokenCount%CONTEXT_WIN + prompt.size() + k] = response(i) * t[0].b[i][j].MK
                                t[blockCount-1].b[m][n].K(initialTokCount%CONTEXT_WIN + prompt.size() + k) = dot(response[k], t[blockCount-1].b[m][n].MK);
                            }
                        }
                    }
                }
            }
            // forward(blockCount, currentTokenCount, promptCount);
            prev_Error = current_Error;
        }
        // update variables
        resCount++;
        trainCount++;
        epochCount += j_epoch;
        error += crossEntropy(otok, response[i]);
        currentTokenCount += 1;
        if(currentTokenCount % CONTEXT_WIN == 0) {
            blockCount += 1;
            // t[blockCount-1].deserialise(t[blockCount-1].blockFilePath);
        }
        // Keys and Queries of response
        if(resCount > 0) {
            if(blockCount == 1) {
                for(int m = 0; m < x; m++) {
                    for(int n = 0; n < y; n++) {
                        for(int k = 0; k < resCount; k++) {
                            // make queries using compute KorQ: t[0].b[i][j].Q[currentTokenCount%CONTEXT_WIN + prompt.size() + k] = response(i) * t[0].b[i][j].MQ
                            t[0].b[m][n].Q(initialTokCount%CONTEXT_WIN + prompt.size() + k) = dot(response[k], t[0].b[m][n].MQ);
                            // make keys using compute KorQ: t[0].b[i][j].K[currentTokenCount%CONTEXT_WIN + prompt.size() + k] = response(i) * t[0].b[i][j].MK
                            t[0].b[m][n].K(initialTokCount%CONTEXT_WIN + prompt.size() + k) = dot(response[k], t[0].b[m][n].MK);
                        }
                    }
                }
            }
            else {
                for(int m = 0; m < x; m++) {
                    for(int n = 0; n < y; n++) {
                        for(int k = 0; k < resCount; k++) {
                            // make queries using compute KorQ: t[blockCount-1].b[i][j].Q[currentTokenCount%CONTEXT_WIN + prompt.size() + k] = response(i) * t[0].b[i][j].MQ
                            t[blockCount-1].b[m][n].Q(initialTokCount%CONTEXT_WIN + prompt.size() + k) = dot(response[k], t[blockCount-1].b[m][n].MQ);
                            // make keys using compute KorQ: t[blockCount-1].b[i][j].K[currentTokenCount%CONTEXT_WIN + prompt.size() + k] = response(i) * t[0].b[i][j].MK
                            t[blockCount-1].b[m][n].K(initialTokCount%CONTEXT_WIN + prompt.size() + k) = dot(response[k], t[blockCount-1].b[m][n].MK);
                        }
                    }
                }
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

#endif
