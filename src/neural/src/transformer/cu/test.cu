
#include "include/transformer.hpp"
#include "include/block.hpp"
#include "include/attention.hpp"
#include "include/mlp.hpp" // For errorofv, MSE
#include <maths.hpp>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <string>
#include <cmath>

// --- CUDA Error Checking Macro ---
#define CUDA_CHECK(call)                                                     \
do {                                                                         \
    cudaError_t err = call;                                                  \
    if (err != cudaSuccess) {                                                \
        fprintf(stderr, "CUDA Error in %s at line %d: %s (%d)\n",            \
                __FILE__, __LINE__, cudaGetErrorString(err), err);           \
        throw std::runtime_error("CUDA Error: " + std::string(cudaGetErrorString(err)));    \
    }                                                                        \
} while (0)


/**
 * @brief (CUDA) Test the transformer for prompt and response.
 * @param prompt Prompt token embeddings (on host).
 * @param response Response token embeddings (on host).
 * @param rString Tokens of the response (on host).
 */
void transformer::cuTest(std::vector<std::vector<float>>& prompt, std::vector<std::vector<float>>& response,
    std::vector<std::string>& rString)
{
}


/**
 * @brief (CUDA) Test transformers for continuous chats.
 * @param prompts All prompts (on host).
 * @param responses Token embeddings of all responses to the prompts (on host).
 * @param rString Tokens of the responses (on host).
 */
void transformer::cuTest(std::vector<std::vector<std::vector<float>>>& prompts, std::vector<std::vector<std::vector<float>>>& responses,
                        std::vector<std::vector<std::string>>& rString)
{
    // Basic Validation
    if (prompts.size() != responses.size() || responses.size() != rString.size()) throw std::runtime_error("cuTest(chat): Mismatch in number of prompts, responses, and response strings.");
    size_t total_tokens_to_add = 0;
    for (size_t i = 0; i < prompts.size(); ++i) {
        if (prompts[i].empty()) throw std::runtime_error("cuTest(chat): Non-empty prompts required.");
        if (responses[i].empty() || responses[i].size() != rString[i].size()) throw std::runtime_error("cuTest(chat): Response mismatch/empty.");
        if (!prompts[i].empty() && prompts[i][0].size() != static_cast<size_t>(d)) throw std::runtime_error("cuTest(chat): Prompt dim mismatch.");
        if (!responses[i].empty() && responses[i][0].size() != static_cast<size_t>(d)) throw std::runtime_error("cuTest(chat): Response dim mismatch.");
        total_tokens_to_add += prompts[i].size() + responses[i].size();
    }
    // If chat test is a session, currentTokenCount should persist.
    // The current stub resets it, implying each call to cuTest (chat) is also self-contained or resets session state.
    // For now, let's assume it's a fresh session for each cuTest(chat,...) call, so internal calls to cuTest(prompt,response,...) will manage their own local state.
    // The main transformer's currentTokenCount, blockCount are effectively reset by the inner cuTest calls.

    bool originalInTraining = this->inTraining;
    this->inTraining = false; // Ensure inference mode

    for (size_t i = 0; i < prompts.size(); ++i) {
        std::cout << "--- Testing Chat Turn (CUDA) " << i + 1 << "/" << prompts.size() << " ---" << std::endl;
        cuTest(prompts[i], responses[i], rString[i]); // Call single test, state is managed locally within that call
    }
    // Overall chat summary would use this->testCount, this->testMSE which are accumulated by inner calls.
    this->inTraining = originalInTraining;
}
