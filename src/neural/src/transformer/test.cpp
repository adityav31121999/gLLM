
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
 * @brief Test the transformer for next token prediction (single token testing)
 * @param promptCount number of tokens in the prompt
 * @param currentTokenCount number of tokens in the full context
 * @param blockCount current block in full context
 * @param expected expected token embedding
 * @param expString expected token string (currently unused in testing logic, but kept for signature consistency)
 */
void transformer::test(int& promptCount, int& currentTokenCount, int& blockCount, std::vector<float>& expected,
    std::string& expString) // expString is unused here but kept for consistency
{
    // Testing logic mirrors validation/training setup but without backprop/updates

    // For first block
    if(blockCount == 1 && currentTokenCount < CONTEXT_WIN) {
        // Prepare K and Q (similar to validation/training, review necessity for pure inference)
        // Assuming K/Q computation is necessary for the forward pass context
        // Revisit this section based on how context/state is managed during inference.
        for(int i = 0; i < x; i++) {
            for(int j = 0; j < y; j++) {
                for(int k = 0; k < currentTokenCount; k++) {
                    computeKorQ(tokenEmbed[k], t[0].b[i][j].MQ, t[0].b[i][j].Q[k]);
                    computeKorQ(tokenEmbed[k], t[0].b[i][j].MK, t[0].b[i][j].K[k]);
                    // t[0].b[i][j].EV[k] = tokenEmbed[k]; // EV update might need adjustment
                }
            }
        }
        // computeKdotQs(promptCount, currentTokenCount, blockCount, isSelf, inTraining); // inTraining should be false

        // Perform forward pass
        forward(blockCount, currentTokenCount, promptCount); // Ensure forward uses correct context

        // Compute output and error
        computeOutput(otok, embeddings, vocabsize, indexForToken); // Use otok which should be updated by forward
        float currentError = errorofv(otok, expected); // Compare final output 'otok' with expected using original error func
        float currentMSE = MSE(otok, expected);        // Calculate Mean Squared Error
        testError += currentError; // Accumulate test error
        testMSE += currentMSE;     // Accumulate test MSE
        testCount++;               // Increment test counter

        // Update state for next token (if testing sequentially)
        // Assumes caller manages adding the *correct* next token embedding to tokenEmbed
        currentTokenCount += 1;
        if(currentTokenCount == CONTEXT_WIN) {
            blockCount += 1;
        }
    }
    // For next blocks
    else if(blockCount > 1 && currentTokenCount >= CONTEXT_WIN) {
        // Prepare context (tokForBlock, K/Q) - Similar concerns as above regarding state updates vs. pure forward pass
        // Double-check if this is needed for testing.
        int startIdx = CONTEXT_WIN * (blockCount - 2);
        for(int i = 0; i < CONTEXT_WIN; i++) { // Should be CONTEXT_WIN tokens from the previous block's view
             if (startIdx + i < tokenEmbed.size()) { // Boundary check
                 t[blockCount-1].tokForBlock[i] = tokenEmbed[startIdx + i];
             } else { /* Handle error */ }
        }
        for(int i = 0; i < x; i++) {
            for(int j = 0; j < y; j++) {
                // Review K/Q/EV logic for testing/inference
                for(int k = 0; k < CONTEXT_WIN; k++) {
                    // computeKorQ(t[blockCount-2].b[i][j].EV[k], t[blockCount-1].b[i][j].MQ, t[blockCount-1].b[i][j].Q[k]);
                    // computeKorQ(t[blockCount-1].tokForBlock[k], t[blockCount-1].b[i][j].MK, t[blockCount-1].b[i][j].K[k]);
                    // t[blockCount-1].b[i][j].EV[k] = ???; // How is EV state propagated/updated in testing?
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
    tokenEmbed.assign(FULL_CONTEXT, std::vector<float>(d, 0.0f)); // Clear/Resize tokenEmbed

    // Add the first token as the initial context
    if (!sentence.empty()) {
        tokenEmbed[0] = sentence[0];
        currentTokenCount = 1;
        promptCount = 1;
    } else { return; } // Cannot test an empty sentence

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
             tokenEmbed[currentTokenCount - 1] = sentence[i];
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
void transformer::test(std::vector<std::vector<float>>& prompt, std::vector<std::vector<float>>& response, std::vector<std::string>& rString)
{
    // Validation checks
    if (prompt.empty()) throw std::runtime_error("Initial prompt cannot be empty for testing!");
    if(prompt.size() > PROMPT_THRESHOLD) std::cerr << "Warning: Prompt size exceeds threshold." << std::endl;
    if (response.empty() || response.size() != rString.size()) throw std::runtime_error("Response embeddings/strings mismatch for testing!");
    if (prompt.size() + response.size() > FULL_CONTEXT) std::cerr << "Warning: Combined prompt/response exceeds FULL_CONTEXT." << std::endl;

    // --- Reset or manage state ---
    currentTokenCount = 0;
    blockCount = 1;
    tokenEmbed.assign(FULL_CONTEXT, std::vector<float>(d, 0.0f)); // Clear/Resize

    // --- Process prompt (set context) ---
    for(int i = 0; i < prompt.size(); ++i) {
        if (currentTokenCount >= FULL_CONTEXT) { std::cerr << "Warning: Context full processing prompt." << std::endl; break; }
        tokenEmbed[currentTokenCount++] = prompt[i];
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
        if (currentTokenCount <= FULL_CONTEXT) { tokenEmbed[currentTokenCount - 1] = response[i]; }
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
    tokenEmbed.assign(FULL_CONTEXT, std::vector<float>(d, 0.0f)); // Clear/Resize

    // --- Process each turn sequentially ---
    for(int turn = 0; turn < prompts.size(); ++turn) {
        std::vector<std::vector<float>>& currentPrompt = prompts[turn];
        std::vector<std::vector<float>>& currentResponse = responses[turn];
        std::vector<std::string>& currentRString = rString[turn];

        // --- Add prompt to context ---
        int promptStartTokenCount = currentTokenCount;
        for(int i = 0; i < currentPrompt.size(); ++i) {
            if (currentTokenCount >= FULL_CONTEXT) { std::cerr << "Warning: Context full processing prompt in chat turn " << turn << "." << std::endl; break; }
            tokenEmbed[currentTokenCount++] = currentPrompt[i];
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
            if (currentTokenCount <= FULL_CONTEXT) { tokenEmbed[currentTokenCount - 1] = currentResponse[i]; }
            promptCount = 1; // Subsequent predictions
        }
    }
    // Final test metrics (average error, average MSE) can be calculated outside this function
    // using testError, testMSE, and testCount.
}

#endif // USE_CPU
