
#ifdef USE_CPU

#include "include/attention.hpp"
#include "include/block.hpp"
#include "include/transformer.hpp"
#include <maths.hpp>
#include <vector>
#include <string>
#include <stdexcept>
#include <iostream>
#include <numeric> // For std::accumulate if needed
#include <algorithm> // For std::min, std::copy

/**
 * @brief Test the transformer for prompt and response (single prompt and response)
 * @param prompt prompt token embeddings
 * @param response response token embeddings (expected outputs)
 * @param rString tokens of response (expected output strings)
 */
void transformer::test(std::vector<std::vector<float>>& prompt, std::vector<std::vector<float>>& response, 
    std::vector<std::string>& rString)
{
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
    
    bool originalInTraining = this->inTraining;
    this->inTraining = false; // Ensure inference mode for all turns

    // Optionally, save and restore overall stats if this is a "session"
    // float overall_start_testMSE = this->testMSE;
    // float overall_start_testError = this->testError;
    // long long overall_start_testCount = this->testCount;
    // this->testMSE = 0; this->testError = 0; this->testCount = 0; // Reset for this chat test run

    for (size_t i = 0; i < prompts.size(); ++i) {
        std::cout << "--- Testing Chat Turn " << i + 1 << "/" << prompts.size() << " ---" << std::endl;
        test(prompts[i], responses[i], rString[i]); // Call single test, state persists
    }

    std::cout << "\n--- Overall Chat Test Summary (from cumulative transformer stats for this session) ---" << std::endl;
    std::cout << "=========================================" << std::endl;
    this->inTraining = originalInTraining; // Restore original state
}

#endif // USE_CPU
