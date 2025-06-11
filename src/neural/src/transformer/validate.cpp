// d:\gLLM\src\neural\src\transformer\validate.cpp
// transformer validation
#include "include/attention.hpp"
#include "include/block.hpp"
#include "include/transformer.hpp"
#include <maths.hpp> // For errorofv, MSE, and computeOutput
#include <vector>
#include <string>
#include <stdexcept> // For runtime_error
#include <iostream>  // For cerr

#ifdef USE_CPU

/**
 * @brief Validate the transformer for next token prediction (single token validation)
 * @param promptCount number of tokens in the prompt
 * @param currentTokenCount number of tokens in the full context
 * @param blockCount current block in full context
 * @param expected expected token embedding
 * @param expString expected token string (currently unused in validation logic, but kept for signature consistency)
 */
void transformer::validate(int& promptCount, int& currentTokenCount, int& blockCount, std::vector<float>& expected,
    std::string& expString)
{
    // For first block
    if(blockCount == 1 && currentTokenCount < CONTEXT_WIN) {
        for(int i = 0; i < x; i++) {
            for(int j = 0; j < y; j++) {
                for(int token_idx = 0; token_idx < currentTokenCount; token_idx++) {
                    // These might not be strictly needed if K/Q caches are managed differently for validation/inference
                    std::vector<float> current_token_embed_vec = getRow(tokenEmbed, token_idx);

                    std::vector<float> q_output_vec(this->h); // Assuming h is the dimension of Q vector
                    computeKorQ(current_token_embed_vec, t[0].b[i][j].MQ, q_output_vec);
                    setRow(t[0].b[i][j].Q, token_idx, q_output_vec);

                    std::vector<float> k_output_vec(this->h); // Assuming h is the dimension of K vector
                    computeKorQ(current_token_embed_vec, t[0].b[i][j].MK, k_output_vec);
                    setRow(t[0].b[i][j].K, token_idx, k_output_vec);

                    // EV update seems incorrect for validation, should likely reflect actual history.
                    // setRow(t[0].b[i][j].EV, token_idx, current_token_embed_vec); // This might need adjustment
                }
            }
        }
        // computeKdotQs might also be part of the forward pass internally
        // computeKdotQs(promptCount, currentTokenCount, blockCount, isSelf, inTraining); // inTraining should be false

        // Perform forward pass
        forward(blockCount, currentTokenCount, promptCount); // Ensure forward uses correct context

        // Compute output and error
        computeOutput(otok, embeddings, vocabsize, indexForToken); // Use otok which should be updated by forward
        float currentError = errorofv(otok, expected); // Compare final output 'otok' with expected using original error func
        float currentMSE = MSE(otok, expected);        // Calculate Mean Squared Error
        validationError += currentError; // Accumulate validation error
        validationMSE += currentMSE;     // Accumulate validation MSE
        validationCount++;               // Increment validation counter

        // Update state for next token (if validating sequentially)
        // This assumes the caller manages adding the *correct* next token embedding to tokenEmbed
        currentTokenCount += 1;
        if(currentTokenCount == CONTEXT_WIN) {
            blockCount += 1;
        }
    }
    // For next blocks
    else if(blockCount > 1 && currentTokenCount >= CONTEXT_WIN) {
        // Prepare context (tokForBlock, K/Q) - Similar concerns as above regarding state updates vs. pure forward pass
        // The logic here heavily mirrors training state updates. Double-check if this is needed for validation.
        int startIdx = CONTEXT_WIN * (blockCount - 2);
        for(int token_idx_in_block = 0; token_idx_in_block < CONTEXT_WIN; token_idx_in_block++) { // Should be CONTEXT_WIN tokens from the previous block's view
             if (startIdx + token_idx_in_block < tokenEmbed.row) { // Boundary check against mat rows
                 std::vector<float> prev_token_embed_row = getRow(tokenEmbed, startIdx + token_idx_in_block);
                 setRow(t[blockCount-1].tokForBlock, token_idx_in_block, prev_token_embed_row);
             } else {
                 // Handle error or fill with padding if necessary
             }
        }
        for(int i = 0; i < x; i++) {
            for(int j = 0; j < y; j++) {
                // Assuming K/Q depend on previous block's state (EV) and current block's input (tokForBlock)
                // This needs careful review based on the model architecture (e.g., RetNet, standard Transformer)
                for(int token_idx_in_block = 0; token_idx_in_block < CONTEXT_WIN; token_idx_in_block++) { // Process CONTEXT_WIN tokens relevant to this block transition
                    // Q derived from previous block's state?
                    std::vector<float> prev_block_ev_row = getRow(t[blockCount-2].b[i][j].EV, token_idx_in_block);
                    std::vector<float> q_output_vec(this->h);
                    computeKorQ(prev_block_ev_row, t[blockCount-1].b[i][j].MQ, q_output_vec);
                    setRow(t[blockCount-1].b[i][j].Q, token_idx_in_block, q_output_vec);

                    // K derived from current block's input?
                    std::vector<float> current_block_token_row = getRow(t[blockCount-1].tokForBlock, token_idx_in_block);
                    std::vector<float> k_output_vec(this->h);
                    computeKorQ(current_block_token_row, t[blockCount-1].b[i][j].MK, k_output_vec);
                    setRow(t[blockCount-1].b[i][j].K, token_idx_in_block, k_output_vec);
                }
            }
        }
        // Perform forward pass
        forward(blockCount, currentTokenCount, promptCount); // Ensure forward uses correct context

        // Compute output and error
        computeOutput(otok, embeddings, vocabsize, indexForToken); // Use otok
        float currentError = errorofv(otok, expected); // Compare final output 'otok'
        float currentMSE = MSE(otok, expected);        // Calculate Mean Squared Error
        validationError += currentError; // Accumulate validation error
        validationMSE += currentMSE;     // Accumulate validation MSE
        validationCount++;               // Increment validation counter

        currentTokenCount += 1;
        if(currentTokenCount % CONTEXT_WIN == 0) {
            blockCount += 1;
        }
    }
    else {
         std::cerr << "Warning: Unexpected state in validate (currentTokenCount=" << currentTokenCount
                   << ", blockCount=" << blockCount << ")" << std::endl;
    }
}

/**
 * @brief Validate the transformer on sentences (single continuous sentence, paragraphs and passages)
 * @param sentence token embedding of sentence
 * @param rString sentence tokens (expected output strings)
 */
void transformer::validate(std::vector<std::vector<float>>& sentence, std::vector<std::string>& rString)
{
    // Constraints check
    if(sentence.size() > FULL_CONTEXT) {
        // Optionally truncate or handle differently instead of throwing
        std::cerr << "Warning: Sentence size (" << sentence.size() << ") exceeds FULL_CONTEXT (" << FULL_CONTEXT << "). Truncating." << std::endl;
    }
    if(sentence.empty() || sentence.size() != rString.size()) {
        throw std::runtime_error("Sentence embeddings and strings must be non-empty and have the same size.");
    }

    currentTokenCount = 0; // Start from the beginning of the context for this sentence
    blockCount = 1;        // Start with the first block
    tokenEmbed = mat(FULL_CONTEXT, this->d); // Clear/Recreate tokenEmbed for this run

    // Add the first token as the initial context
    if (!sentence.empty()) {
        setRow(tokenEmbed, 0, sentence[0]);
        currentTokenCount = 1;
        promptCount = 1; // The first token acts as the initial prompt
    } else {
        return; // Cannot validate an empty sentence
    }


    // Validate for each subsequent token in the sentence (predicting token i based on 0 to i-1)
    for(int i = 1; i < sentence.size(); ++i) {
        if (currentTokenCount >= FULL_CONTEXT) {
             std::cerr << "Warning: Validation context full during sentence validation, stopping early at token " << i << "." << std::endl;
             break;
        }
        // Get expected output for this step
        std::vector<float>& expectedEmbedding = sentence[i];
        std::string& expectedString = rString[i]; // String is currently unused in validate step
        validate(promptCount, currentTokenCount, blockCount, expectedEmbedding, expectedString);

        if (currentTokenCount <= FULL_CONTEXT) { // Check bounds (currentTokenCount was already incremented)
             setRow(tokenEmbed, currentTokenCount - 1, sentence[i]); // Add the actual token embedding for the next step's context
        }

        // After the first prediction, subsequent steps predict one token based on the preceding ones.
        promptCount = 1;
    };
}

/**
 * @brief Validate the transformer for prompt and response (single prompt and response)
 * @param prompt prompt token embeddings
 * @param response response token embeddings (expected outputs)
 * @param rString tokens of response (expected output strings)
 */
void transformer::validate(std::vector<std::vector<float>>& prompt, std::vector<std::vector<float>>& response, 
    std::vector<std::string>& rString)
{
    // Validation checks
    if (prompt.empty()) {
        throw std::runtime_error("Initial prompt cannot be empty for validation!");
    }
    if(prompt.size() > PROMPT_THRESHOLD) { // Use appropriate threshold
        // Handle or warn
        std::cerr << "Warning: Prompt size (" << prompt.size() << ") exceeds threshold (" << PROMPT_THRESHOLD << ")." << std::endl;
        // Optionally truncate prompt
    }
    if (response.empty() || response.size() != rString.size()) {
        throw std::runtime_error("Response embeddings and strings must be non-empty and have the same size for validation!");
    }
    if (prompt.size() + response.size() > FULL_CONTEXT) {
         std::cerr << "Warning: Combined prompt and response size exceeds FULL_CONTEXT. Validation might be incomplete." << std::endl;
    }

    // --- Reset or manage state for this specific prompt-response pair ---
    // Similar state management considerations as in the sentence validation
    currentTokenCount = 0;
    blockCount = 1;
    tokenEmbed = mat(FULL_CONTEXT, this->d); // Clear/Recreate tokenEmbed

    // --- Process the prompt first (no validation here, just setting context) ---
    for(int i = 0; i < prompt.size(); ++i) {
        if (currentTokenCount >= FULL_CONTEXT) {
            std::cerr << "Warning: Context full while processing prompt. Prompt truncated." << std::endl;
            break;
        }
        setRow(tokenEmbed, currentTokenCount, prompt[i]);
        currentTokenCount++;
        // Update blockCount if context window boundary is crossed
        if (currentTokenCount > 0 && currentTokenCount % CONTEXT_WIN == 0) {
             // Avoid incrementing blockCount beyond 'm' if 'm' is fixed
             if (blockCount < m) { // Assuming 'm' is the max number of blocks
                 blockCount++;
             } else {
                 // Handle context overflow if block count is limited
             }
        }
    }
    promptCount = currentTokenCount; // The entire prompt is the initial "prompt count" for the first prediction

    // --- Validate the response tokens sequentially ---
    for(int i = 0; i < response.size(); ++i) {
        if (currentTokenCount >= FULL_CONTEXT) {
             std::cerr << "Warning: Validation context full during response validation, stopping early at response token " << i << "." << std::endl;
             break;
        }
        std::vector<float>& expectedEmbedding = response[i];
        std::string& expectedString = rString[i]; // String unused in validate step

        // Call single-token validate
        // It uses and increments currentTokenCount, blockCount internally
        validate(promptCount, currentTokenCount, blockCount, expectedEmbedding, expectedString);

        // After validating response token 'i', add its *true* embedding for the next prediction step (i+1)
        if (currentTokenCount <= FULL_CONTEXT) { // currentTokenCount was incremented by validate
             setRow(tokenEmbed, currentTokenCount - 1, response[i]);
        }

        // After the first response token prediction, subsequent predictions are based on a single preceding token context shift.
        promptCount = 1;
    }
     // Restore state if needed
}

/**
 * @brief Validate transformers for continuous chats
 * @param prompts all prompts in the chat sequence
 * @param responses all corresponding responses token embeddings
 * @param rString all corresponding response tokens (strings)
 */
void transformer::validate(std::vector<std::vector<std::vector<float>>>& prompts, std::vector<std::vector<std::vector<float>>>& responses,
    std::vector<std::vector<std::string>>& rString)
{
    if(prompts.size() != responses.size() || responses.size() != rString.size()) {
        throw std::runtime_error("Prompts, responses, and response strings must have the same number of turns for chat validation.");
    }

    // --- Reset overall state for the entire chat validation ---
    currentTokenCount = 0;
    blockCount = 1;
    tokenEmbed = mat(FULL_CONTEXT, this->d); // Clear/Recreate tokenEmbed for the chat

    // --- Process each turn (prompt-response pair) sequentially, maintaining context ---
    for(int turn = 0; turn < prompts.size(); ++turn) {
        std::vector<std::vector<float>>& currentPrompt = prompts[turn];
        std::vector<std::vector<float>>& currentResponse = responses[turn];
        std::vector<std::string>& currentRString = rString[turn];

        // --- Add prompt to context ---
        int promptStartTokenCount = currentTokenCount;
        for(int i = 0; i < currentPrompt.size(); ++i) {
            if (currentTokenCount >= FULL_CONTEXT) {
                std::cerr << "Warning: Context full while processing prompt in chat turn " << turn << ". Prompt truncated." << std::endl;
                break;
            }
            setRow(tokenEmbed, currentTokenCount, currentPrompt[i]);
            currentTokenCount++;
            // Update blockCount if context window boundary is crossed
            if (currentTokenCount > 0 && currentTokenCount % CONTEXT_WIN == 0) {
                 if (blockCount < m) blockCount++;
            }
        }
        // Set promptCount for the first prediction of this turn's response
        promptCount = currentTokenCount - promptStartTokenCount;
        if (promptCount == 0 && currentResponse.empty()) continue; // Skip empty turns if needed
        if (promptCount == 0 && !currentResponse.empty()) {
             std::cerr << "Warning: Empty prompt followed by non-empty response in turn " << turn << std::endl;
             promptCount = 1; // Set to 1 to proceed? Or handle as error?
        }


        // --- Validate the response tokens for this turn ---
        for(int i = 0; i < currentResponse.size(); ++i) {
            if (currentTokenCount >= FULL_CONTEXT) {
                std::cerr << "Warning: Validation context full during response validation in chat turn " << turn << ", stopping early." << std::endl;
                break;
            }
            std::vector<float>& expectedEmbedding = currentResponse[i];
            std::string& expectedString = currentRString[i]; // String unused

            // Call single-token validate
            validate(promptCount, currentTokenCount, blockCount, expectedEmbedding, expectedString);

            // Add true response token embedding for next step's context
            if (currentTokenCount <= FULL_CONTEXT) { // currentTokenCount was incremented
                 setRow(tokenEmbed, currentTokenCount - 1, currentResponse[i]);
            }

            // Subsequent predictions in the response sequence have promptCount=1
            promptCount = 1;
        }
    }
}

#endif // USE_CPU
/*
// transformer training
#include "include/attention.hpp"
#include "include/block.hpp"
#include "include/transformer.hpp"

#ifdef USE_CPU

/**
 * @brief train the transformer for next token prediction (single token training)
 * @param promptCount number of tokens in the prompt
 * @param currentTokenCount number of tokens in the full context
 * @param blockCount current block in full context
 * @param expected expected token embedding
 * @param expString expected token

void transformer::train(int& promptCount, int& currentTokenCount, int& blockCount, std::vector<float>& expected,
    std::string& expString) 
{
    // for first block
    if(blockCount == 1 && currentTokenCount < CONTEXT_WIN) {
        /*
        // compute the kdotQ for each head of the block
        for(int i = 0; i < x; i++) {
            for(int j = 0; j < y; j++) {
                for(int k = 0; k < currentTokenCount; k++) {
                    std::vector<float> current_token_embed_vec(this->d);
                    for(int m = 0; m < this->d; m++) {
                        current_token_embed_vec[m] = tokenEmbed(k,m);
                    }

                    std::vector<float> q_output_vec(this->h);
                    computeKorQ(current_token_embed_vec, t[0].b[i][j].MQ, q_output_vec);
                    setRow(t[0].b[i][j].Q, k, q_output_vec);

                    std::vector<float> k_output_vec(this->h);
                    computeKorQ(current_token_embed_vec, t[0].b[i][j].MK, k_output_vec);
                    setRow(t[0].b[i][j].K, k, k_output_vec);

                    // Update EV for the current head with the current token's embedding
                    setRow(t[0].b[i][j].EV, k, current_token_embed_vec);
                }
            }
        }
        computeKdotQs(promptCount, currentTokenCount, blockCount, isSelf, inTraining);
        // train from here
        forward(blockCount, currentTokenCount, promptCount);
        int i = 0;
        while (i <= epochs) {
            computeOutput(otok, embeddings, vocabsize, indexForToken);
            if((errorofv(otok, expected) < 0.01) || tokens[indexForToken] == expString) {
                break;
            }
            // if error is not corrected even after epochs, then increase epochs
            if(errorofv(otok, expected) > 0.01 && i == epochs) {
                epochs += 10;
            }
            backward(expected);
            forward(blockCount, currentTokenCount, promptCount);
            i++;
        }
        trainCount++;
        epochCount += i;
        error += errorofv(otok, expected);
        currentTokenCount += 1;
        if(currentTokenCount == CONTEXT_WIN) {
            blockCount += 1;
        }
    }
    // for next blocks
    else if(blockCount > 1 && currentTokenCount >= CONTEXT_WIN) {
        /*
        // assign tokForBlock embeddings
        // Assuming currentTokenCount here refers to tokens within the current local window (up to CONTEXT_WIN)
        // And tokForBlock is for the current block (blockCount-1)
        for(int token_idx_in_block = 0; token_idx_in_block < CONTEXT_WIN; token_idx_in_block++) {
            int global_token_idx = (blockCount-2)*CONTEXT_WIN + token_idx_in_block; // token from previous block's full context window
            if (global_token_idx < this->currentTokenCount) { // Ensure we don't read past actual currentTokenCount
                std::vector<float> prev_token_embed_row = getRow(tokenEmbed, global_token_idx);
                setRow(t[blockCount-1].tokForBlock, token_idx_in_block, prev_token_embed_row);
            }
        }
        // compute the KdotQ for each head of block using EVs of previous blocks
        for(int i = 0; i < x; i++) {
            for(int j = 0; j < y; j++) {
                for(int k = 0; k < CONTEXT_WIN; k++) { // Iterate through tokens in the window for this block
                    std::vector<float> prev_block_ev_row = getRow(t[blockCount-2].b[i][j].EV, k);
                    std::vector<float> q_output_vec(this->h);
                    computeKorQ(prev_block_ev_row, t[blockCount-1].b[i][j].MQ, q_output_vec);
                    setRow(t[blockCount-1].b[i][j].Q, k, q_output_vec);

                    std::vector<float> current_block_token_row = getRow(t[blockCount-1].tokForBlock, k);
                    std::vector<float> k_output_vec(this->h);
                    computeKorQ(current_block_token_row, t[blockCount-1].b[i][j].MK, k_output_vec);
                    setRow(t[blockCount-1].b[i][j].K, k, k_output_vec);
                    // EV for the current block (blockCount-1) at token k is updated with its own token from tokForBlock
                    setRow(t[blockCount-1].b[i][j].EV, k, current_block_token_row);
                }
            }
        }
        computeKdotQs(promptCount, currentTokenCount, blockCount, isSelf, inTraining);
        // train from here
        forward(blockCount, currentTokenCount, promptCount);
        int i = 0;
        while (i < epochs) {
            computeOutput(otok, embeddings, vocabsize, indexForToken);
            if(errorofv(otok, expected) < 0.01 || tokens[indexForToken] == expString) {
                // tokenEmbed[currentTokenCount] = t[blockCount-1].EH; // This was problematic, EH is vector, tokenEmbed is mat
                // If otok is the predicted embedding, and it's good, this might be where it's stored.
                break;
            }
            // if error is not corrected even after epochs, then increase epochs
            if(errorofv(otok, expected) > 0.01 && i == epochs) {
                epochs += 10;
            }
            i++;
            backward(expected, blockCount);
            forward(blockCount, currentTokenCount, promptCount);
        }
        trainCount++;
        epochCount += i;
        error += errorofv(otok, expected);
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
    // keep this in a loop and train for each token in the sentence starting from second token
    for(int token_idx_in_sentence = 1; token_idx_in_sentence < sentence.size(); token_idx_in_sentence++) {
        // first block
        if(blockCount == 1 && (currentTokenCount < CONTEXT_WIN-1)) {
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
            while (j_epoch <= epochs) {
                computeOutput(otok, embeddings, vocabsize, indexForToken);
                if((errorofv(otok, sentence[token_idx_in_sentence]) < 0.01) || tokens[indexForToken] == rString[token_idx_in_sentence]) {
                    break;
                }
                // if error is not corrected even after epochs, then increase epochs
                if(errorofv(otok, sentence[token_idx_in_sentence]) > 0.01 && j_epoch == epochs) {
                    epochs += 10;
                }
                backward(sentence[token_idx_in_sentence]);
                forward(blockCount, currentTokenCount, promptCount);
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
                if(errorofv(otok, sentence[token_idx_in_sentence]) < 0.01 || tokens[indexForToken] == rString[token_idx_in_sentence]) {
                    break;
                }
                // if error is not corrected even after epochs, then increase epochs
                if(errorofv(otok, sentence[token_idx_in_sentence]) > 0.01 && j_epoch == epochs) {
                    epochs += 10;
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
    
    // for first prompt
    if(currentTokenCount == 0) {
        // token embedding should be divided for specific sizes
        // for smaller prompts
        if(prompt.size() < CONTEXT_WIN) {
            // add prompt to continuous tokenEmbed
            for(int i = 0; i < prompt.size(); i++) {
                setRow(tokenEmbed, i, prompt[i]);
            }
            // add prompt to vertical retention vectors
            for(int i = 0; i < x; i++) {
                for(int j = 0; j < y; j++) {
                    for(int k = 0; k < prompt.size(); k++) {
                        setRow(t[0].b[i][j].EV, k, prompt[k]);
                    }
                }
            }
        }
        // set current token count and prompt count
        currentTokenCount += prompt.size();
        promptCount = prompt.size();
        // TRAIN FOR RESPONSE
        for(int resp_idx = 0; resp_idx < response.size(); resp_idx++) {
            // for ith blocks
            if(blockCount > 1 && currentTokenCount > CONTEXT_WIN) {
                // set tokens for tokForBlock to calculate the KdotQ for this block and response training
                for(int token_idx_in_block = 0; token_idx_in_block < CONTEXT_WIN; token_idx_in_block++) {
                    int global_token_idx = (blockCount-2)*CONTEXT_WIN + token_idx_in_block; // Corrected offset
                    std::vector<float> prev_token_embed_row = getRow(tokenEmbed, global_token_idx);
                    setRow(t[blockCount-1].tokForBlock, token_idx_in_block, prev_token_embed_row);
                }
            }
            if(blockCount == 1) {
                // make queries using compute KorQ: t[0].Q[currentTokenCount%CONTEXT_WIN] = (prompt(i) * t[0].MQ)
                // make keys using compute KorQ: t[0].K[currentTokenCount%CONTEXT_WIN] = (prompt(i) * t[0].MK)
                for(int i = 0; i < x; i++) {
                    for(int j = 0; j < y; j++) {
                        for(int k = 0; k < currentTokenCount; k++) {
                            std::vector<float> current_token_embed_vec(this->d);
                            for(int m = 0; m < this->d; m++) {
                                current_token_embed_vec[m] = tokenEmbed(k,m);
                            }

                            std::vector<float> q_output_vec(this->h);
                            computeKorQ(current_token_embed_vec, t[0].b[i][j].MQ, q_output_vec);
                            setRow(t[0].b[i][j].Q, k, q_output_vec);

                            std::vector<float> k_output_vec(this->h);
                            computeKorQ(current_token_embed_vec, t[0].b[i][j].MK, k_output_vec);
                            setRow(t[0].b[i][j].K, k, k_output_vec);

                            // Update EV for the current head with the current token's embedding
                            setRow(t[0].b[i][j].EV, k, current_token_embed_vec);
                        }
                    }
                }
            }
            else {
                // make queries using compute KorQ: t[blockCount-1].Q = (t[blockCount-2].EV(currentTokenCount%CONTEXT_WIN) * t[blockCount-2].MQ)
                // make keys using compute KorQ: t[blockCount-1].K = (prompt(i) * t[blockCount-2].MK)
                for(int token_idx_in_block = 0; token_idx_in_block < CONTEXT_WIN; token_idx_in_block++) {
                    int global_token_idx = (blockCount-2)*CONTEXT_WIN + token_idx_in_block; // token from previous block's full context window
                    if (global_token_idx < this->currentTokenCount) { // Ensure we don't read past actual currentTokenCount
                        std::vector<float> prev_token_embed_row = getRow(tokenEmbed, global_token_idx);
                        setRow(t[blockCount-1].tokForBlock, token_idx_in_block, prev_token_embed_row);
                    }
                }
                // compute the KdotQ for each head of block using EVs of previous blocks
                for(int i = 0; i < x; i++) {
                    for(int j = 0; j < y; j++) {
                        for(int k = 0; k < CONTEXT_WIN; k++) { // Iterate through tokens in the window for this block
                            std::vector<float> prev_block_ev_row = getRow(t[blockCount-2].b[i][j].EV, k);
                            std::vector<float> q_output_vec(this->h);
                            computeKorQ(prev_block_ev_row, t[blockCount-1].b[i][j].MQ, q_output_vec);
                            setRow(t[blockCount-1].b[i][j].Q, k, q_output_vec);

                            std::vector<float> current_block_token_row = getRow(t[blockCount-1].tokForBlock, k);
                            std::vector<float> k_output_vec(this->h);
                            computeKorQ(current_block_token_row, t[blockCount-1].b[i][j].MK, k_output_vec);
                            setRow(t[blockCount-1].b[i][j].K, k, k_output_vec);
                            // EV for the current block (blockCount-1) at token k is updated with its own token from tokForBlock
                            setRow(t[blockCount-1].b[i][j].EV, k, current_block_token_row);
                        }
                    }
                }
            }
            // compute the KdotQ for each head of block using EVs of previous blocks
            computeKdotQs(promptCount, currentTokenCount, blockCount, isSelf, inTraining);
            forward(blockCount, currentTokenCount, promptCount);
            int j_epoch = 0;
            while (j_epoch < epochs) {
                computeOutput(otok, embeddings, vocabsize, indexForToken);
                if(errorofv(otok, response[resp_idx]) < 0.01 || tokens[indexForToken] == rString[resp_idx]) {
                    setRow(tokenEmbed, currentTokenCount, otok); // EH is std::vector, assign to row
                    break;
                }
                // if error is not corrected even after epochs, then increase epochs
                if(errorofv(otok, response[resp_idx]) > 0.01 && j_epoch == epochs) {
                    epochs += 10;
                }
                j_epoch++;
                backward(response[resp_idx], blockCount);
                forward(blockCount, currentTokenCount, promptCount);
            }
            // update variables
            trainCount++;
            epochCount += j_epoch;
            error += errorofv(otok, response[resp_idx]);
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
                setRow(tokenEmbed, currentTokenCount + i, prompt[i]);
                setRow(t[blockCount - 1].tokForBlock, c + i, prompt[i]);
            }
            // add prompts to vertical retention vectors
            for(int i = 0; i < x; i++) {
                for(int j = 0; j < y; j++) {
                    for(int k = 0; k < prompt.size(); k++) {
                        setRow(t[blockCount - 1].b[i][j].EV, c + k, prompt[k]);
                    }
                }
            }
            promptCount = prompt.size();
            currentTokenCount += prompt.size();
            if(currentTokenCount % CONTEXT_WIN == 0) {
                blockCount += 1;
            }
            if(blockCount == 1) {
                // make queries using compute KorQ: t[0].Q[currentTokenCount%CONTEXT_WIN] = (prompt(i) * t[0].MQ)
                // make keys using compute KorQ: t[0].K[currentTokenCount%CONTEXT_WIN] = (prompt(i) * t[0].MK)
            }
            else {
                // make queries using compute KorQ: t[blockCount-1].Q = (t[blockCount-2].EV(currentTokenCount%CONTEXT_WIN) * t[blockCount-2].MQ)
                // make keys using compute KorQ: t[blockCount-1].K = (prompt(i) * t[blockCount-2].MK)
            }
            if(blockCount == 1) {
                // make queries using compute KorQ: t[0].Q[currentTokenCount%CONTEXT_WIN] = (prompt(i) * t[0].MQ)
                // make keys using compute KorQ: t[0].K[currentTokenCount%CONTEXT_WIN] = (prompt(i) * t[0].MK)
                for(int i = 0; i < x; i++) {
                    for(int j = 0; j < y; j++) {
                        for(int k = 0; k < currentTokenCount; k++) {
                            std::vector<float> current_token_embed_vec(this->d);
                            for(int m = 0; m < this->d; m++) {
                                current_token_embed_vec[m] = tokenEmbed(k,m);
                            }

                            std::vector<float> q_output_vec(this->h);
                            computeKorQ(current_token_embed_vec, t[0].b[i][j].MQ, q_output_vec);
                            setRow(t[0].b[i][j].Q, k, q_output_vec);

                            std::vector<float> k_output_vec(this->h);
                            computeKorQ(current_token_embed_vec, t[0].b[i][j].MK, k_output_vec);
                            setRow(t[0].b[i][j].K, k, k_output_vec);

                            // Update EV for the current head with the current token's embedding
                            setRow(t[0].b[i][j].EV, k, current_token_embed_vec);
                        }
                    }
                }
            }
            else {
                // make queries using compute KorQ: t[blockCount-1].Q = (t[blockCount-2].EV(currentTokenCount%CONTEXT_WIN) * t[blockCount-2].MQ)
                // make keys using compute KorQ: t[blockCount-1].K = (prompt(i) * t[blockCount-2].MK)
                for(int token_idx_in_block = 0; token_idx_in_block < CONTEXT_WIN; token_idx_in_block++) {
                    int global_token_idx = (blockCount-2)*CONTEXT_WIN + token_idx_in_block; // token from previous block's full context window
                    if (global_token_idx < this->currentTokenCount) { // Ensure we don't read past actual currentTokenCount
                        std::vector<float> prev_token_embed_row = getRow(tokenEmbed, global_token_idx);
                        setRow(t[blockCount-1].tokForBlock, token_idx_in_block, prev_token_embed_row);
                    }
                }
                // compute the KdotQ for each head of block using EVs of previous blocks
                for(int i = 0; i < x; i++) {
                    for(int j = 0; j < y; j++) {
                        for(int k = 0; k < CONTEXT_WIN; k++) { // Iterate through tokens in the window for this block
                            std::vector<float> prev_block_ev_row = getRow(t[blockCount-2].b[i][j].EV, k);
                            std::vector<float> q_output_vec(this->h);
                            computeKorQ(prev_block_ev_row, t[blockCount-1].b[i][j].MQ, q_output_vec);
                            setRow(t[blockCount-1].b[i][j].Q, k, q_output_vec);

                            std::vector<float> current_block_token_row = getRow(t[blockCount-1].tokForBlock, k);
                            std::vector<float> k_output_vec(this->h);
                            computeKorQ(current_block_token_row, t[blockCount-1].b[i][j].MK, k_output_vec);
                            setRow(t[blockCount-1].b[i][j].K, k, k_output_vec);
                            // EV for the current block (blockCount-1) at token k is updated with its own token from tokForBlock
                            setRow(t[blockCount-1].b[i][j].EV, k, current_block_token_row);
                        }
                    }
                }
            }
        }
        // for prompt size larger than available tokens in local context
        else if(prompt.size() > c) {
            // available tokens in local context
            int m1 = prompt.size() - c;
            for(int i = 0; i < m1; i++) {
                setRow(tokenEmbed, currentTokenCount + i, prompt[i]);
            }
            currentTokenCount += m1;
            promptCount = m1;
            computeKdotQs(promptCount, currentTokenCount, blockCount, isSelf, inTraining);
            // number of prompt tokens in next block
            int m2 = prompt.size() - m1;
            blockCount += 1; // Increment block count before using it as index for t[blockCount]
            if(blockCount == 1) {
                // make queries using compute KorQ: t[0].Q[currentTokenCount%CONTEXT_WIN] = (prompt(i) * t[0].MQ)
                // make keys using compute KorQ: t[0].K[currentTokenCount%CONTEXT_WIN] = (prompt(i) * t[0].MK)
            }
            else {
                // make queries using compute KorQ: t[blockCount-1].Q = (t[blockCount-2].EV(currentTokenCount%CONTEXT_WIN) * t[blockCount-2].MQ)
                // make keys using compute KorQ: t[blockCount-1].K = (prompt(i) * t[blockCount-2].MK)
            }
            if(blockCount == 1) {
                // make queries using compute KorQ: t[0].Q[currentTokenCount%CONTEXT_WIN] = (prompt(i) * t[0].MQ)
                // make keys using compute KorQ: t[0].K[currentTokenCount%CONTEXT_WIN] = (prompt(i) * t[0].MK)
                for(int i = 0; i < x; i++) {
                    for(int j = 0; j < y; j++) {
                        for(int k = 0; k < currentTokenCount; k++) {
                            std::vector<float> current_token_embed_vec(this->d);
                            for(int m = 0; m < this->d; m++) {
                                current_token_embed_vec[m] = tokenEmbed(k,m);
                            }

                            std::vector<float> q_output_vec(this->h);
                            computeKorQ(current_token_embed_vec, t[0].b[i][j].MQ, q_output_vec);
                            setRow(t[0].b[i][j].Q, k, q_output_vec);

                            std::vector<float> k_output_vec(this->h);
                            computeKorQ(current_token_embed_vec, t[0].b[i][j].MK, k_output_vec);
                            setRow(t[0].b[i][j].K, k, k_output_vec);

                            // Update EV for the current head with the current token's embedding
                            setRow(t[0].b[i][j].EV, k, current_token_embed_vec);
                        }
                    }
                }
            }
            else {
                // make queries using compute KorQ: t[blockCount-1].Q = (t[blockCount-2].EV(currentTokenCount%CONTEXT_WIN) * t[blockCount-2].MQ)
                // make keys using compute KorQ: t[blockCount-1].K = (prompt(i) * t[blockCount-2].MK)
                for(int token_idx_in_block = 0; token_idx_in_block < CONTEXT_WIN; token_idx_in_block++) {
                    int global_token_idx = (blockCount-2)*CONTEXT_WIN + token_idx_in_block; // token from previous block's full context window
                    if (global_token_idx < this->currentTokenCount) { // Ensure we don't read past actual currentTokenCount
                        std::vector<float> prev_token_embed_row = getRow(tokenEmbed, global_token_idx);
                        setRow(t[blockCount-1].tokForBlock, token_idx_in_block, prev_token_embed_row);
                    }
                }
                // compute the KdotQ for each head of block using EVs of previous blocks
                for(int i = 0; i < x; i++) {
                    for(int j = 0; j < y; j++) {
                        for(int k = 0; k < CONTEXT_WIN; k++) { // Iterate through tokens in the window for this block
                            std::vector<float> prev_block_ev_row = getRow(t[blockCount-2].b[i][j].EV, k);
                            std::vector<float> q_output_vec(this->h);
                            computeKorQ(prev_block_ev_row, t[blockCount-1].b[i][j].MQ, q_output_vec);
                            setRow(t[blockCount-1].b[i][j].Q, k, q_output_vec);

                            std::vector<float> current_block_token_row = getRow(t[blockCount-1].tokForBlock, k);
                            std::vector<float> k_output_vec(this->h);
                            computeKorQ(current_block_token_row, t[blockCount-1].b[i][j].MK, k_output_vec);
                            setRow(t[blockCount-1].b[i][j].K, k, k_output_vec);
                            // EV for the current block (blockCount-1) at token k is updated with its own token from tokForBlock
                            setRow(t[blockCount-1].b[i][j].EV, k, current_block_token_row);
                        }
                    }
                }
            }
            for(int i = 0; i < m2; i++) {
                // Assuming tokForBlock for new block is indexed from 0
                setRow(t[blockCount-1].tokForBlock, i, prompt[i]); // prompt[i] or prompt[m1+i]?
            }
            // add prompts to tokenEmbed
            for(int i = 0; i < prompt.size(); i++) {
                // This should be careful about global currentTokenCount
                // setRow(tokenEmbed, currentTokenCount_before_m2_increment + i, prompt[m1+i]); // if adding only m2 part
            }
            currentTokenCount += m2;
            promptCount = m2;
        }
        // TRAIN FOR RESPONSE
        for(int resp_idx = 0; resp_idx < response.size(); resp_idx++) {
            // for first block
            // compute the KdotQ for each head of block using EVs of previous blocks
            computeKdotQs(promptCount, currentTokenCount, blockCount, isSelf, inTraining);
            forward(blockCount, currentTokenCount, promptCount);
            int j_epoch = 0;
            while (j_epoch < epochs) {
                if(errorofv(otok, response[resp_idx]) < 0.01 || tokens[indexForToken] == rString[resp_idx]) {
                    setRow(tokenEmbed, currentTokenCount, otok);
                    break;
                }
                // if error is not corrected even after epochs, then increase epochs
                if(errorofv(otok, response[resp_idx]) > 0.01 && j_epoch == epochs) {
                    epochs += 10;
                }
                j_epoch++;
                backward(response[resp_idx], blockCount);
                forward(blockCount, currentTokenCount, promptCount);
            }
            trainCount++;
            epochCount += j_epoch;
            error += errorofv(otok, response[resp_idx]);
            currentTokenCount += 1;
            if(currentTokenCount % CONTEXT_WIN == 0) {
                blockCount += 1;
                // shift to next block
                // make Query from t[blockCount-2].EV(i): t[blockCount-2].Q[currentTokenCount%CONTEXT_WIN] = t[blockCount-2].EV(currentTokenCount%CONTEXT_WIN) * t[blockCount-1].MQ
                // make key of response(i): t[blockCount-2].K[(currentTokenCount-CONTEXT_WIN)%CONTEXT_WIN] = response(i) * t[blockCount-1].MK
                if(blockCount == 1) {
                    // make queries using compute KorQ: t[0].Q[currentTokenCount%CONTEXT_WIN] = (prompt(i) * t[0].MQ)
                    // make keys using compute KorQ: t[0].K[currentTokenCount%CONTEXT_WIN] = (prompt(i) * t[0].MK)
                    for(int i = 0; i < x; i++) {
                        for(int j = 0; j < y; j++) {
                            for(int k = 0; k < resp_idx+1; k++) {
                                std::vector<float> current_token_embed_vec(this->d);
                                for(int m = 0; m < this->d; m++) {
                                    current_token_embed_vec[m] = response[k][m];
                                }

                                std::vector<float> q_output_vec(this->h);
                                computeKorQ(current_token_embed_vec, t[0].b[i][j].MQ, q_output_vec);
                                setRow(t[0].b[i][j].Q, currentTokenCount + k, q_output_vec);

                                std::vector<float> k_output_vec(this->h);
                                computeKorQ(current_token_embed_vec, t[0].b[i][j].MK, k_output_vec);
                                setRow(t[0].b[i][j].K, currentTokenCount + k, k_output_vec);

                                // Update EV for the current head with the current token's embedding
                                setRow(t[0].b[i][j].EV, currentTokenCount + k, current_token_embed_vec);
                            }
                        }
                    }
                }
                else {
                    // make queries using compute KorQ: t[blockCount-1].Q = (t[blockCount-2].EV(currentTokenCount%CONTEXT_WIN) * t[blockCount-2].MQ)
                    // make keys using compute KorQ: t[blockCount-1].K = (prompt(i) * t[blockCount-2].MK)
                    for(int token_idx_in_block = 0; token_idx_in_block < CONTEXT_WIN; token_idx_in_block++) {
                        int global_token_idx = (blockCount-2)*CONTEXT_WIN + token_idx_in_block; // token from previous block's full context window
                        if (global_token_idx < this->currentTokenCount) { // Ensure we don't read past actual currentTokenCount
                            std::vector<float> prev_token_embed_row = getRow(tokenEmbed, global_token_idx);
                            setRow(t[blockCount-1].tokForBlock, token_idx_in_block, prev_token_embed_row);
                        }
                    }
                    // compute the KdotQ for each head of block using EVs of previous blocks
                    for(int i = 0; i < x; i++) {
                        for(int j = 0; j < y; j++) {
                            for(int k = 0; k < CONTEXT_WIN; k++) { // Iterate through tokens in the window for this block
                                std::vector<float> prev_block_ev_row = getRow(t[blockCount-2].b[i][j].EV, k);
                                std::vector<float> q_output_vec(this->h);
                                computeKorQ(prev_block_ev_row, t[blockCount-1].b[i][j].MQ, q_output_vec);
                                setRow(t[blockCount-1].b[i][j].Q, k, q_output_vec);

                                std::vector<float> current_block_token_row = getRow(t[blockCount-1].tokForBlock, k);
                                std::vector<float> k_output_vec(this->h);
                                computeKorQ(current_block_token_row, t[blockCount-1].b[i][j].MK, k_output_vec);
                                setRow(t[blockCount-1].b[i][j].K, k, k_output_vec);
                                // EV for the current block (blockCount-1) at token k is updated with its own token from tokForBlock
                                setRow(t[blockCount-1].b[i][j].EV, k, current_block_token_row);
                            }
                        }
                    }
                }
                if(blockCount == 1) {
                    // make queries using compute KorQ: t[0].Q[currentTokenCount%CONTEXT_WIN] = (prompt(i) * t[0].MQ)
                    // make keys using compute KorQ: t[0].K[currentTokenCount%CONTEXT_WIN] = (prompt(i) * t[0].MK)
                    for(int i = 0; i < x; i++) {
                        for(int j = 0; j < y; j++) {
                            for(int k = 0; k < currentTokenCount; k++) {
                                std::vector<float> current_token_embed_vec(this->d);
                                for(int m = 0; m < this->d; m++) {
                                    current_token_embed_vec[m] = tokenEmbed(k,m);
                                }

                                std::vector<float> q_output_vec(this->h);
                                computeKorQ(current_token_embed_vec, t[0].b[i][j].MQ, q_output_vec);
                                setRow(t[0].b[i][j].Q, k, q_output_vec);

                                std::vector<float> k_output_vec(this->h);
                                computeKorQ(current_token_embed_vec, t[0].b[i][j].MK, k_output_vec);
                                setRow(t[0].b[i][j].K, k, k_output_vec);

                                // Update EV for the current head with the current token's embedding
                                setRow(t[0].b[i][j].EV, k, current_token_embed_vec);
                            }
                        }
                    }
                }
                else {
                    // make queries using compute KorQ: t[blockCount-1].Q = (t[blockCount-2].EV(currentTokenCount%CONTEXT_WIN) * t[blockCount-2].MQ)
                    // make keys using compute KorQ: t[blockCount-1].K = (prompt(i) * t[blockCount-2].MK)
                    for(int token_idx_in_block = 0; token_idx_in_block < CONTEXT_WIN; token_idx_in_block++) {
                        int global_token_idx = (blockCount-2)*CONTEXT_WIN + token_idx_in_block; // token from previous block's full context window
                        if (global_token_idx < this->currentTokenCount) { // Ensure we don't read past actual currentTokenCount
                            std::vector<float> prev_token_embed_row = getRow(tokenEmbed, global_token_idx);
                            setRow(t[blockCount-1].tokForBlock, token_idx_in_block, prev_token_embed_row);
                        }
                    }
                    // compute the KdotQ for each head of block using EVs of previous blocks
                    for(int i = 0; i < x; i++) {
                        for(int j = 0; j < y; j++) {
                            for(int k = 0; k < CONTEXT_WIN; k++) { // Iterate through tokens in the window for this block
                                std::vector<float> prev_block_ev_row = getRow(t[blockCount-2].b[i][j].EV, k);
                                std::vector<float> q_output_vec(this->h);
                                computeKorQ(prev_block_ev_row, t[blockCount-1].b[i][j].MQ, q_output_vec);
                                setRow(t[blockCount-1].b[i][j].Q, k, q_output_vec);

                                std::vector<float> current_block_token_row = getRow(t[blockCount-1].tokForBlock, k);
                                std::vector<float> k_output_vec(this->h);
                                computeKorQ(current_block_token_row, t[blockCount-1].b[i][j].MK, k_output_vec);
                                setRow(t[blockCount-1].b[i][j].K, k, k_output_vec);
                                // EV for the current block (blockCount-1) at token k is updated with its own token from tokForBlock
                                setRow(t[blockCount-1].b[i][j].EV, k, current_block_token_row);
                            }
                        }
                    }
                }
                continue;
            }
            // same block
            // make Query from t[blockCount-2].EV(i): Q[currentTokenCount%CONTEXT_WIN] = t[blockCount-1].EV(i) * M
            // make key of tokenEmbed(i): K[(currentTokenCount-CONTEXT_WIN)%CONTEXT_WIN] = response(i) * MK
            if(blockCount == 1) {
                // make queries using compute KorQ: t[0].Q[currentTokenCount%CONTEXT_WIN] = (prompt(i) * t[0].MQ)
                // make keys using compute KorQ: t[0].K[currentTokenCount%CONTEXT_WIN] = (prompt(i) * t[0].MK)
                for(int i = 0; i < x; i++) {
                    for(int j = 0; j < y; j++) {
                        for(int k = 0; k < currentTokenCount; k++) {
                            std::vector<float> current_token_embed_vec(this->d);
                            for(int m = 0; m < this->d; m++) {
                                current_token_embed_vec[m] = tokenEmbed(k,m);
                            }

                            std::vector<float> q_output_vec(this->h);
                            computeKorQ(current_token_embed_vec, t[0].b[i][j].MQ, q_output_vec);
                            setRow(t[0].b[i][j].Q, k, q_output_vec);

                            std::vector<float> k_output_vec(this->h);
                            computeKorQ(current_token_embed_vec, t[0].b[i][j].MK, k_output_vec);
                            setRow(t[0].b[i][j].K, k, k_output_vec);

                            // Update EV for the current head with the current token's embedding
                            setRow(t[0].b[i][j].EV, k, current_token_embed_vec);
                        }
                    }
                }
            }
            else {
                // make queries using compute KorQ: t[blockCount-1].Q = (t[blockCount-2].EV(currentTokenCount%CONTEXT_WIN) * t[blockCount-2].MQ)
                // make keys using compute KorQ: t[blockCount-1].K = (prompt(i) * t[blockCount-2].MK)
                for(int token_idx_in_block = 0; token_idx_in_block < CONTEXT_WIN; token_idx_in_block++) {
                    int global_token_idx = (blockCount-2)*CONTEXT_WIN + token_idx_in_block; // token from previous block's full context window
                    if (global_token_idx < this->currentTokenCount) { // Ensure we don't read past actual currentTokenCount
                        std::vector<float> prev_token_embed_row = getRow(tokenEmbed, global_token_idx);
                        setRow(t[blockCount-1].tokForBlock, token_idx_in_block, prev_token_embed_row);
                    }
                }
                // compute the KdotQ for each head of block using EVs of previous blocks
                for(int i = 0; i < x; i++) {
                    for(int j = 0; j < y; j++) {
                        for(int k = 0; k < CONTEXT_WIN; k++) { // Iterate through tokens in the window for this block
                            std::vector<float> prev_block_ev_row = getRow(t[blockCount-2].b[i][j].EV, k);
                            std::vector<float> q_output_vec(this->h);
                            computeKorQ(prev_block_ev_row, t[blockCount-1].b[i][j].MQ, q_output_vec);
                            setRow(t[blockCount-1].b[i][j].Q, k, q_output_vec);

                            std::vector<float> current_block_token_row = getRow(t[blockCount-1].tokForBlock, k);
                            std::vector<float> k_output_vec(this->h);
                            computeKorQ(current_block_token_row, t[blockCount-1].b[i][j].MK, k_output_vec);
                            setRow(t[blockCount-1].b[i][j].K, k, k_output_vec);
                            // EV for the current block (blockCount-1) at token k is updated with its own token from tokForBlock
                            setRow(t[blockCount-1].b[i][j].EV, k, current_block_token_row);
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
*/