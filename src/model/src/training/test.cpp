
#include "include/model.hpp"
#include "include/model_fs.hpp"
#include <neural.hpp>
#include <filesystem> // Required for directory iteration
#include <iostream>   // Required for std::cout, std::cerr
#include <stdexcept>  // Required for std::runtime_error
#include <maths.hpp>

#ifdef USE_CUDA
    #include <cuda.h>
    #include <cuda_runtime.h>

    #define CUDA_CHECK(call)                                                     \
    do {                                                                         \
        cudaError_t err = call;                                                  \
        if (err != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA Error in %s at line %d: %s\n",                 \
                    __FILE__, __LINE__, cudaGetErrorString(err));                \
            /* Consider throwing an exception or exiting */                      \
            throw std::runtime_error("CUDA Error: " + std::string(cudaGetErrorString(err)));    \
        }                                                                        \
    } while (0)

#elif USE_OPENCL
    #include <CL/cl.hpp>
#endif


/**
 * @brief train the first block of transformer
 * @param prompt prompt embeddings for model
 * @param response expected response embeddings from model
 * @param rString tokens of response
 */
void model::test1stBlock(std::vector<std::vector<float>> &prompt, std::vector<std::vector<float>> &response, std::vector<std::string> rString)
{
    if(this->T.currentTokenCount + prompt.size() + response.size() < CONTEXT_WIN) 
    {
        #ifdef USE_CUDA
            this->T.cuTest(prompt, response, rString);
        #elif USE_OPENCL
            this->T.clTest(prompt, response, rString);
        #elif USE_CPU
            this->T.test(prompt, response, rString);
        #endif
    }
    else {
        throw std::runtime_error("LOCAL CONTEXT LIMIT of REACHED");
    }
}



/**
 * @brief test first block using data from folder
 * @param testData path to folder with all testing data
 */
void model::testBlock(const std::string& testData) 
{
    if(testData.empty()) {
        throw std::runtime_error("Test data folder cannot be empty!");
    }
    int numberoffiles = 0;
    for (const auto& entry : std::filesystem::directory_iterator(testData)) {
        if (entry.is_regular_file() && entry.path().extension() == ".txt") {
            numberoffiles++;
        }
    }
    if(numberoffiles == 0) {
        throw std::runtime_error("No test data found in the specified folder!");
    }
    std::cout << "Total testing files: " << numberoffiles << std::endl;
    newChat(); // Reset context before processing files
    for (const auto& entry : std::filesystem::directory_iterator(testData)) {
        if (entry.is_regular_file() && entry.path().extension() == ".txt") {
            std::string path2file = entry.path().string();
            std::cout << "Testing on file: " << path2file << std::endl;
            std::vector<std::string> tokensOfFile;
            std::vector<std::vector<std::string>> oddSentence, evenSentence;
            textSplit(path2file, tokensOfFile, oddSentence, evenSentence);
            // Ensure pairs match
            if(oddSentence.size() != evenSentence.size() && oddSentence.size() > evenSentence.size()) {
                if(!oddSentence.empty() && !evenSentence.empty()) {
                    evenSentence.back().insert(evenSentence.back().end(), oddSentence.back().begin(), oddSentence.back().end());
                    oddSentence.pop_back();
                }
            }
            // Process sentence pairs
            for(int i = 0; i < oddSentence.size(); i++) {
                std::vector<std::vector<float>> promptEmbeddings, responseEmbeddings;
                std::vector<std::string> responseTokens;
                // Get prompt embeddings
                for(int j = 0; j < oddSentence[i].size(); j++) {
                    std::vector<float> embed(d, 0.0f);
                    this->T.getEmbedding(oddSentence[i][j], embed);
                    promptEmbeddings.push_back(embed);
                }
                // Get expected response embeddings and tokens
                for(int j = 0; j < evenSentence[i].size(); j++) {
                    std::vector<float> embed(d, 0.0f);
                    this->T.getEmbedding(evenSentence[i][j], embed);
                    responseEmbeddings.push_back(embed);
                    responseTokens.push_back(evenSentence[i][j]);
                }

                // Note: Testing might not always need the response embeddings/tokens depending on implementation
                // Adjust the check if only prompt size matters for testing context.
                // Call the appropriate test function based on the build configuration
                #ifdef USE_CUDA
                    // Assuming a T.cuTest method exists similar to T.cuTrain
                    this->T.cuTest(promptEmbeddings, responseEmbeddings, responseTokens); // Adjust args as needed for testing
                #elif USE_OPENCL
                    // Assuming a T.clTest method exists similar to T.clTrain
                    this->T.clTest(promptEmbeddings, responseEmbeddings, responseTokens); // Adjust args as needed for testing
                #elif USE_CPU
                    // Assuming a T.test method exists similar to T.train
                    this->T.test(promptEmbeddings, responseEmbeddings, responseTokens); // Adjust args as needed for testing
                #endif
            }
            newChat(); // Reset context after processing each file
        }
    }
}


/**
 * @brief test first block using data from folder
 * @param testData path to folder with all testing data
 */
void model::test(const std::string& testData) 
{
    if(testData.empty()) {
        throw std::runtime_error("Test data folder cannot be empty!");
    }
    int numberoffiles = 0;
    for (const auto& entry : std::filesystem::directory_iterator(testData)) {
        if (entry.is_regular_file() && entry.path().extension() == ".txt") {
            numberoffiles++;
        }
    }
    if(numberoffiles == 0) {
        throw std::runtime_error("No test data found in the specified folder!");
    }
    std::cout << "Total testing files: " << numberoffiles << std::endl;
    newChat(); // Reset context before processing files
    for (const auto& entry : std::filesystem::directory_iterator(testData)) {
        if (entry.is_regular_file() && entry.path().extension() == ".txt") {
            std::string path2file = entry.path().string();
            std::cout << "Testing on file: " << path2file << std::endl;
            std::vector<std::string> tokensOfFile;
            std::vector<std::vector<std::string>> oddSentence, evenSentence;
            textSplit(path2file, tokensOfFile, oddSentence, evenSentence);
            // Ensure pairs match
            if(oddSentence.size() != evenSentence.size() && oddSentence.size() > evenSentence.size()) {
                if(!oddSentence.empty() && !evenSentence.empty()) {
                    evenSentence.back().insert(evenSentence.back().end(), oddSentence.back().begin(), oddSentence.back().end());
                    oddSentence.pop_back();
                }
            }
            // Process sentence pairs
            for(int i = 0; i < oddSentence.size(); i++) {
                std::vector<std::vector<float>> promptEmbeddings, responseEmbeddings;
                std::vector<std::string> responseTokens;
                // Get prompt embeddings
                for(int j = 0; j < oddSentence[i].size(); j++) {
                    std::vector<float> embed(d, 0.0f);
                    this->T.getEmbedding(oddSentence[i][j], embed);
                    promptEmbeddings.push_back(embed);
                }
                // Get expected response embeddings and tokens
                for(int j = 0; j < evenSentence[i].size(); j++) {
                    std::vector<float> embed(d, 0.0f);
                    this->T.getEmbedding(evenSentence[i][j], embed);
                    responseEmbeddings.push_back(embed);
                    responseTokens.push_back(evenSentence[i][j]);
                }

                // Note: Testing might not always need the response embeddings/tokens depending on implementation
                // Adjust the check if only prompt size matters for testing context.
                // Call the appropriate test function based on the build configuration
                #ifdef USE_CUDA
                    // Assuming a T.cuTest method exists similar to T.cuTrain
                    this->T.cuTest(promptEmbeddings, responseEmbeddings, responseTokens); // Adjust args as needed for testing
                #elif USE_OPENCL
                    // Assuming a T.clTest method exists similar to T.clTrain
                    this->T.clTest(promptEmbeddings, responseEmbeddings, responseTokens); // Adjust args as needed for testing
                #elif USE_CPU
                    // Assuming a T.test method exists similar to T.train
                    this->T.test(promptEmbeddings, responseEmbeddings, responseTokens); // Adjust args as needed for testing
                #endif
            }
            newChat(); // Reset context after processing each file
        }
    }
}
