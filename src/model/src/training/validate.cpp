
#include "include/model.hpp"
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

void model::validate1stBlock(std::vector<std::vector<float>> &prompt, std::vector<std::vector<float>> &response, std::vector<std::string> rString)
{
    if(this->T.currentTokenCount + prompt.size() + response.size() < CONTEXT_WIN) 
    {
        #ifdef USE_CUDA
            this->T.cuValidate(prompt, response, rString);
        #elif USE_OPENCL
            this->T.clValidate(prompt, response, rString);
        #elif USE_CPU
            this->T.validate(prompt, response, rString);
        #endif
    }
    else {
        throw std::runtime_error("LOCAL CONTEXT LIMIT of REACHED");
    }
}

/**
 * @brief validate first block using data from folder
 * @param validationData path to folder with all validation data
 */
void model::validateBlock(const std::string& validationData) 
{
    if(validationData.empty()) {
        throw std::runtime_error("Validation data folder cannot be empty!");
    }
    int numberoffiles = 0;
    for (const auto& entry : std::filesystem::directory_iterator(validationData)) {
        if (entry.is_regular_file() && entry.path().extension() == ".txt") {
            numberoffiles++;
        }
    }
    if(numberoffiles == 0) {
        throw std::runtime_error("No validation data found in the specified folder!");
    }
    std::cout << "Total validation files: " << numberoffiles << std::endl;
    newChat(); // Reset context before processing files
    for (const auto& entry : std::filesystem::directory_iterator(validationData)) {
        if (entry.is_regular_file() && entry.path().extension() == ".txt") {
            std::string path2file = entry.path().string();
            std::cout << "Validating on file: " << path2file << std::endl;
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

                // Adjust the check if only prompt size matters for validation context.
                // Call the appropriate validate function based on the build configuration
                #ifdef USE_CUDA
                    // Assuming a T.cuValidate method exists similar to T.cuTrain
                    this->T.cuValidate(promptEmbeddings, responseEmbeddings, responseTokens); // Adjust args as needed
                #elif USE_OPENCL
                    // Assuming a T.clValidate method exists similar to T.clTrain
                    this->T.clValidate(promptEmbeddings, responseEmbeddings, responseTokens); // Adjust args as needed
                #elif USE_CPU
                    // Assuming a T.validate method exists similar to T.train
                    this->T.validate(promptEmbeddings, responseEmbeddings, responseTokens); // Adjust args as needed
                #endif
            }
            newChat(); // Reset context after processing each file
        }
    }
}


/**
 * @brief validate first block using data from folder
 * @param validationData path to folder with all validation data
 */
void model::validate(const std::string& validationData) 
{
    if(validationData.empty()) {
        throw std::runtime_error("Validation data folder cannot be empty!");
    }
    int numberoffiles = 0;
    for (const auto& entry : std::filesystem::directory_iterator(validationData)) {
        if (entry.is_regular_file() && entry.path().extension() == ".txt") {
            numberoffiles++;
        }
    }
    if(numberoffiles == 0) {
        throw std::runtime_error("No validation data found in the specified folder!");
    }
    std::cout << "Total validation files: " << numberoffiles << std::endl;
    newChat(); // Reset context before processing files
    for (const auto& entry : std::filesystem::directory_iterator(validationData)) {
        if (entry.is_regular_file() && entry.path().extension() == ".txt") {
            std::string path2file = entry.path().string();
            std::cout << "Validating on file: " << path2file << std::endl;
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

                // Adjust the check if only prompt size matters for validation context.
                // Call the appropriate validate function based on the build configuration
                #ifdef USE_CUDA
                    // Assuming a T.cuValidate method exists similar to T.cuTrain
                    this->T.cuValidate(promptEmbeddings, responseEmbeddings, responseTokens); // Adjust args as needed
                #elif USE_OPENCL
                    // Assuming a T.clValidate method exists similar to T.clTrain
                    this->T.clValidate(promptEmbeddings, responseEmbeddings, responseTokens); // Adjust args as needed
                #elif USE_CPU
                    // Assuming a T.validate method exists similar to T.train
                    this->T.validate(promptEmbeddings, responseEmbeddings, responseTokens); // Adjust args as needed
                #endif
            }
            newChat(); // Reset context after processing each file
        }
    }
}
