
#ifdef USE_CPU

#include "include/attention.hpp"
#include "include/block.hpp"
#include "include/transformer.hpp"
#include <maths.hpp> // For errorofv, MSE, and computeOutput
#include <vector>
#include <string>
#include <stdexcept> // For runtime_error
#include <iostream>  // For cerr
#include <numeric>   // For std::accumulate if needed
#include <algorithm> // For std::min, std::copy


/**
 * @brief Validate the transformer for prompt and response (single prompt and response)
 * @param prompt prompt token embeddings
 * @param response response token embeddings (expected outputs)
 * @param rString tokens of response (expected output strings)
 */
void transformer::validate(std::vector<std::vector<float>>& prompt, std::vector<std::vector<float>>& response, 
    std::vector<std::string>& rString)
{
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

    bool originalInTraining = this->inTraining;
    this->inTraining = false; // Ensure inference mode for all turns

    for (size_t i = 0; i < prompts.size(); ++i) {
        std::cout << "--- Validating Chat Turn " << i + 1 << "/" << prompts.size() << " ---" << std::endl;
        validate(prompts[i], responses[i], rString[i]); // Call single validate, state persists
    }

    std::cout << "\n--- Overall Chat Validation Summary (from cumulative transformer stats for this session) ---" << std::endl;
    std::cout << "==================================================" << std::endl;
    this->inTraining = originalInTraining; // Restore original state
}

#endif // USE_CPU
