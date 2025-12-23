#ifdef USE_CPU
#include <vector>
#include <iostream>
#include "include/transformer.hpp"

/**
 * @brief test function for transformer
 * @param [in] prompt prompt embeddings
 * @param [out] rString response produced by function
 */
void transformer::test(std::vector<std::vector<float>> &prompt, std::vector<std::string> &rString)
{
    // check for full context
    if(currentTokenCount+sequence1Count >= FULL_CONTEXT) {
        throw std::runtime_error("test: TOKEN LIMIT REACHED AT FULL CONTEXT! FURTHER PROCESS CANNOT TAKE PLACE -_-");
    }

    // prompt should not be empty
    if(prompt.empty()) {
        throw std::runtime_error("test: Prompt cannot be empty.");
    }

    
}

#endif // USE_CPU