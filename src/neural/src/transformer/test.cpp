
#include "include/attention.hpp"
#include "include/block.hpp"
#include "include/transformer.hpp"
#include <maths.hpp>
#include <vector>
#include <string>
#include <stdexcept>
#include <iostream>

#ifdef USE_CPU

/**
 * @brief Test the transformer for next token prediction (single token testing)
 * @param promptCount number of tokens in the prompt
 * @param currentTokenCount number of tokens in the full context
 * @param blockCount current block in full context
 * @param expected expected token embedding
 * @param expString expected token string (currently unused in testing logic, but kept for signature consistency)
 */
void transformer::test(int& promptCount, int& currentTokenCount, int& blockCount, std::vector<float>& expected,
    std::string& expString)
{
    // For first block
    if(blockCount == 1 && currentTokenCount < CONTEXT_WIN) {
        for(int i = 0; i < x; i++) {
            for(int j = 0; j < y; j++) {
                for(int token_idx = 0; token_idx < currentTokenCount; token_idx++) {
                    std::vector<float> current_token_embed_vec = getRow(tokenEmbed, token_idx);

                    std::vector<float> q_output_vec(this->h);
                    computeKorQ(current_token_embed_vec, t[0].b[i][j].MQ, q_output_vec);
                    setRow(t[0].b[i][j].Q, token_idx, q_output_vec);

                    std::vector<float> k_output_vec(this->h);
                    computeKorQ(current_token_embed_vec, t[0].b[i][j].MK, k_output_vec);
                    setRow(t[0].b[i][j].K, token_idx, k_output_vec);
                }
            }
        }
        // Perform forward pass
        forward(blockCount, currentTokenCount, promptCount); // Ensure forward uses correct context

        // Compute output and error
        computeOutput(otok, embeddings, vocabsize, indexForToken); // Use otok which should be updated by forward
        float currentError = errorofv(otok, expected); // Compare final output 'otok' with expected using original error func
        float currentMSE = MSE(otok, expected);        // Calculate Mean Squared Error
        testError += currentError; // Accumulate test error
        testMSE += currentMSE;     // Accumulate test MSE
        testCount++;               // Increment test counter
        currentTokenCount += 1;
        if(currentTokenCount == CONTEXT_WIN) {
            blockCount += 1;
        }
    }
    // For next blocks
    else if(blockCount > 1 && currentTokenCount >= CONTEXT_WIN) {
        int startIdx = CONTEXT_WIN * (blockCount - 2);
        for(int token_idx_in_block = 0; token_idx_in_block < CONTEXT_WIN; token_idx_in_block++) { // Should be CONTEXT_WIN tokens from the previous block's view
             if (startIdx + token_idx_in_block < tokenEmbed.row) { // Boundary check against mat rows
                std::vector<float> prev_token_embed_row = getRow(tokenEmbed, startIdx + token_idx_in_block);
                setRow(t[blockCount-1].tokForBlock, token_idx_in_block, prev_token_embed_row);
            } 
            else { 

            }
        }
        for(int i = 0; i < x; i++) {
            for(int j = 0; j < y; j++) {
                // Review K/Q/EV logic for testing/inference
                for(int token_idx_in_block = 0; token_idx_in_block < CONTEXT_WIN; token_idx_in_block++) {
                    std::vector<float> prev_block_ev_row = getRow(t[blockCount-2].b[i][j].EV, token_idx_in_block);
                    std::vector<float> q_output_vec(this->h);
                    computeKorQ(prev_block_ev_row, t[blockCount-1].b[i][j].MQ, q_output_vec);
                    setRow(t[blockCount-1].b[i][j].Q, token_idx_in_block, q_output_vec);

                    std::vector<float> current_block_token_row = getRow(t[blockCount-1].tokForBlock, token_idx_in_block);
                    std::vector<float> k_output_vec(this->h);
                    computeKorQ(current_block_token_row, t[blockCount-1].b[i][j].MK, k_output_vec);
                    setRow(t[blockCount-1].b[i][j].K, token_idx_in_block, k_output_vec);
                    // setRow(t[blockCount-1].b[i][j].EV, token_idx_in_block, current_block_token_row); // EV update for testing?
                }
            }
        }
        // computeKdotQs(promptCount, currentTokenCount, blockCount, isSelf, inTraining); // inTraining should be false

        // Perform forward pass
        forward(blockCount, currentTokenCount, promptCount); // Ensure forward uses correct context

        // Compute output and error
        computeOutput(otok, embeddings, vocabsize, indexForToken); // Use otok
        float currentError = errorofv(otok, expected); // Compare final output 'otok'
        float currentMSE = MSE(otok, expected);        // Calculate Mean Squared Error
        testError += currentError; // Accumulate test error
        testMSE += currentMSE;     // Accumulate test MSE
        testCount++;               // Increment test counter

        // Update state
        // Assumes caller manages adding the *correct* next token embedding to tokenEmbed
        currentTokenCount += 1;
        if(currentTokenCount % CONTEXT_WIN == 0) {
            blockCount += 1;
        }
    } else {
         // Handle unexpected states
         std::cerr << "Warning: Unexpected state in test (currentTokenCount=" << currentTokenCount
                   << ", blockCount=" << blockCount << ")" << std::endl;
    }
}

/**
 * @brief Test the transformer on sentences (single continuous sentence, paragraphs and passages)
 * @param sentence token embedding of sentence
 * @param rString sentence tokens (expected output strings)
 */
void transformer::test(std::vector<std::vector<float>>& sentence, std::vector<std::string>& rString)
{
    // Constraints check
    if(sentence.size() > FULL_CONTEXT) {
        std::cerr << "Warning: Sentence size (" << sentence.size() << ") exceeds FULL_CONTEXT (" << FULL_CONTEXT << "). Testing might be truncated." << std::endl;
    }
    if(sentence.empty() || sentence.size() != rString.size()) {
        throw std::runtime_error("Sentence embeddings and strings must be non-empty and have the same size for testing.");
    }

    // --- Initialize state for this sentence test ---
    currentTokenCount = 0;
    blockCount = 1;
    tokenEmbed = mat(FULL_CONTEXT, this->d); // Recreate/clear tokenEmbed

    // Add the first token as the initial context
    if (!sentence.empty()) {
        setRow(tokenEmbed, 0, sentence[0]);
        currentTokenCount = 1;
        promptCount = 1;
    } 
    else { 
        return; 
    } // Cannot test an empty sentence

    // Test for each subsequent token
    for(int i = 1; i < sentence.size(); ++i) {
        if (currentTokenCount >= FULL_CONTEXT) {
             std::cerr << "Warning: Test context full during sentence testing, stopping early at token " << i << "." << std::endl;
             break;
        }
        std::vector<float>& expectedEmbedding = sentence[i];
        std::string& expectedString = rString[i]; // String unused

        // Call single-token test function
        test(promptCount, currentTokenCount, blockCount, expectedEmbedding, expectedString);

        // Add the *true* embedding to context for the next step
        if (currentTokenCount <= FULL_CONTEXT) { // currentTokenCount was incremented by test()
             setRow(tokenEmbed, currentTokenCount - 1, sentence[i]);
        }
        promptCount = 1; // Subsequent steps predict one token
    }
}

/**
 * @brief Test the transformer for prompt and response (single prompt and response)
 * @param prompt prompt token embeddings
 * @param response response token embeddings (expected outputs)
 * @param rString tokens of response (expected output strings)
 */
void transformer::test(std::vector<std::vector<float>>& prompt, std::vector<std::vector<float>>& response, 
    std::vector<std::string>& rString)
{
    // Validation checks
    if (prompt.empty()) throw std::runtime_error("Initial prompt cannot be empty for testing!");
    if(prompt.size() > PROMPT_THRESHOLD) std::cerr << "Warning: Prompt size exceeds threshold." << std::endl;
    if (response.empty() || response.size() != rString.size()) throw std::runtime_error("Response embeddings/strings mismatch for testing!");
    if (prompt.size() + response.size() > FULL_CONTEXT) std::cerr << "Warning: Combined prompt/response exceeds FULL_CONTEXT." << std::endl;

    // --- Reset or manage state ---
    currentTokenCount = 0;
    blockCount = 1;
    tokenEmbed = mat(FULL_CONTEXT, this->d); // Recreate/clear

    // --- Process prompt (set context) ---
    for(int i = 0; i < prompt.size(); ++i) {
        if (currentTokenCount >= FULL_CONTEXT) { std::cerr << "Warning: Context full processing prompt." << std::endl; break; }
        // tokenEmbed[currentTokenCount++] = prompt[i]; // Old
        setRow(tokenEmbed, currentTokenCount, prompt[i]);
        currentTokenCount++;
        if (currentTokenCount > 0 && currentTokenCount % CONTEXT_WIN == 0) { if (blockCount < m) blockCount++; }
    }
    promptCount = currentTokenCount;

    // --- Test response tokens ---
    for(int i = 0; i < response.size(); ++i) {
        if (currentTokenCount >= FULL_CONTEXT) { std::cerr << "Warning: Test context full during response, stopping early." << std::endl; break; }
        std::vector<float>& expectedEmbedding = response[i];
        std::string& expectedString = rString[i]; // String unused

        // Call single-token test
        test(promptCount, currentTokenCount, blockCount, expectedEmbedding, expectedString);

        // Add true response token for next step's context
        if (currentTokenCount <= FULL_CONTEXT) { // currentTokenCount was incremented by test()
            setRow(tokenEmbed, currentTokenCount - 1, response[i]);
        }
        promptCount = 1; // Subsequent predictions
    }
}

/**
 * @brief Test transformers for continuous chats
 * @param prompts all prompts in the chat sequence
 * @param responses all corresponding responses token embeddings
 * @param rString all corresponding response tokens (strings)
 */
void transformer::test(std::vector<std::vector<std::vector<float>>>& prompts, std::vector<std::vector<std::vector<float>>>& responses,
    std::vector<std::vector<std::string>>& rString)
{
    if(prompts.size() != responses.size() || responses.size() != rString.size()) {
        throw std::runtime_error("Prompts, responses, and strings must match for chat testing.");
    }

    // --- Reset state for the chat test ---
    currentTokenCount = 0;
    blockCount = 1;
    tokenEmbed = mat(FULL_CONTEXT, this->d); // Recreate/clear

    // --- Process each turn sequentially ---
    for(int turn = 0; turn < prompts.size(); ++turn) {
        std::vector<std::vector<float>>& currentPrompt = prompts[turn];
        std::vector<std::vector<float>>& currentResponse = responses[turn];
        std::vector<std::string>& currentRString = rString[turn];

        // --- Add prompt to context ---
        int promptStartTokenCount = currentTokenCount;
        for(int i = 0; i < currentPrompt.size(); ++i) {
            if (currentTokenCount >= FULL_CONTEXT) { std::cerr << "Warning: Context full processing prompt in chat turn " << turn << "." << std::endl; break; }
            setRow(tokenEmbed, currentTokenCount, currentPrompt[i]);
            currentTokenCount++;
            if (currentTokenCount > 0 && currentTokenCount % CONTEXT_WIN == 0) { if (blockCount < m) blockCount++; }
        }
        promptCount = currentTokenCount - promptStartTokenCount;
        if (promptCount == 0 && currentResponse.empty()) continue;
        if (promptCount == 0 && !currentResponse.empty()) { std::cerr << "Warning: Empty prompt, non-empty response in turn " << turn << std::endl; promptCount = 1; }

        // --- Test response tokens ---
        for(int i = 0; i < currentResponse.size(); ++i) {
            if (currentTokenCount >= FULL_CONTEXT) { 
                std::cerr << "Warning: Test context full during response in chat turn " << turn << "." << std::endl;
                break; 
            }
            std::vector<float>& expectedEmbedding = currentResponse[i];
            std::string& expectedString = currentRString[i]; // String unused

            // Call single-token test
            test(promptCount, currentTokenCount, blockCount, expectedEmbedding, expectedString);

            // Add true response token for next step
            if (currentTokenCount <= FULL_CONTEXT) { // currentTokenCount was incremented by test()
                setRow(tokenEmbed, currentTokenCount - 1, currentResponse[i]);
            }
            promptCount = 1; // Subsequent predictions
        }
    }
}

#endif // USE_CPU
