
#include "include/model.hpp"
#include "include/model_fs.hpp"
#include <neural.hpp>
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
void model::train1stBlock(std::vector<std::vector<float>> &prompt, std::vector<std::vector<float>> &response, std::vector<std::string> rString)
{
    if(this->T.currentTokenCount + prompt.size() + response.size() < CONTEXT_WIN) 
    {
        #ifdef USE_CUDA
            this->T.cuTrain(prompt, response, rString);
        #elif USE_OPENCL
            this->T.clTrain(prompt, response, rString);
        #elif USE_CPU
            this->T.train(prompt, response, rString);
        #endif
    }
    else {
        throw std::runtime_error("LOCAL CONTEXT LIMIT of REACHED");
    }
}


/**
 * @brief train first block using training data from folder
 * @param trainingData path to folder with all training data
 */
void model::trainBlock(const std::string& trainingData) 
{
    if(trainingData.empty()) {
        throw std::runtime_error("Training data folder cannot be empty!");
    }
    int numberoffiles = 0;
    for (const auto& entry : std::filesystem::directory_iterator(trainingData)) {
        if (entry.is_regular_file() && entry.path().extension() == ".txt") {
            numberoffiles++;
        }
    }
    if(numberoffiles == 0) {
        throw std::runtime_error("No training data found in the specified folder!");
    }
    std::cout << "Total training files: " << numberoffiles << std::endl;
    newChat();
    for (const auto& entry : std::filesystem::directory_iterator(trainingData)) {
        if (entry.is_regular_file() && entry.path().extension() == ".txt") {
            std::string path2file = entry.path().string();
            std::cout << "Training on file: " << path2file << std::endl;
            std::vector<std::string> tokensOfFile;
            std::vector<std::vector<std::string>> oddSentence, evenSentence;
            textSplit(path2file, tokensOfFile, oddSentence, evenSentence);
            // if number of oddSentence is not equal to number of even sentence, than combine last odd sentence with the last even sentence
            if(oddSentence.size() != evenSentence.size() && oddSentence.size() > evenSentence.size()) {
                if(!oddSentence.empty() && !evenSentence.empty()) {
                    evenSentence.back().insert(evenSentence.back().end(), oddSentence.back().begin(), oddSentence.back().end());
                    oddSentence.pop_back();
                }
            }
            // get embeddings and response
            for(int i = 0; i < oddSentence.size(); i++) {
                std::vector<std::vector<float>> promptEmbeddings, responseEmbeddings;
                std::vector<std::string> responseTokens;
                for(int j = 0; j < oddSentence[i].size(); j++) {
                    std::vector<float> embed(d, 0.0f);
                    this->T.getEmbedding(oddSentence[i][j], embed);
                    promptEmbeddings.push_back(embed);
                }
                for(int j = 0; j < evenSentence[i].size(); j++) {
                    std::vector<float> embed(d, 0.0f);
                    this->T.getEmbedding(evenSentence[i][j], embed);
                    responseEmbeddings.push_back(embed);
                    responseTokens.push_back(evenSentence[i][j]);
                }
                if(this->T.currentTokenCount + promptEmbeddings.size() + responseEmbeddings.size() < CONTEXT_WIN) 
                {
                    #ifdef USE_CUDA
                        this->T.cuTrain(promptEmbeddings, responseEmbeddings, responseTokens);
                    #elif USE_OPENCL
                        this->T.clTrain(promptEmbeddings, responseEmbeddings, responseTokens);
                    #elif USE_CPU
                        this->T.train(promptEmbeddings, responseEmbeddings, responseTokens);
                    #endif
                }
                else {
                    throw std::runtime_error("LOCAL CONTEXT LIMIT of REACHED");
                }
            }
            newChat();
        }
    }
}


/**
 * @brief train first block using training data from folder
 * @param trainingData path to folder with all training data
 */
void model::train(const std::string& trainingData) 
{
    if(trainingData.empty()) {
        throw std::runtime_error("Training data folder cannot be empty!");
    }
    int numberoffiles = 0;
    for (const auto& entry : std::filesystem::directory_iterator(trainingData)) {
        if (entry.is_regular_file() && entry.path().extension() == ".txt") {
            numberoffiles++;
        }
    }
    if(numberoffiles == 0) {
        throw std::runtime_error("No training data found in the specified folder!");
    }
    std::cout << "Total training files: " << numberoffiles << std::endl;
    newChat();
    for (const auto& entry : std::filesystem::directory_iterator(trainingData)) {
        if (entry.is_regular_file() && entry.path().extension() == ".txt") {
            std::string path2file = entry.path().string();
            std::cout << "Training on file: " << path2file << std::endl;
            std::vector<std::string> tokensOfFile;
            std::vector<std::vector<std::string>> oddSentence, evenSentence;
            textSplit(path2file, tokensOfFile, oddSentence, evenSentence);
            // if number of oddSentence is not equal to number of even sentence, than combine last odd sentence with the last even sentence
            if(oddSentence.size() != evenSentence.size() && oddSentence.size() > evenSentence.size()) {
                if(!oddSentence.empty() && !evenSentence.empty()) {
                    evenSentence.back().insert(evenSentence.back().end(), oddSentence.back().begin(), oddSentence.back().end());
                    oddSentence.pop_back();
                }
            }
            // get embeddings and response
            for(int i = 0; i < oddSentence.size(); i++) {
                std::vector<std::vector<float>> promptEmbeddings, responseEmbeddings;
                std::vector<std::string> responseTokens;
                for(int j = 0; j < oddSentence[i].size(); j++) {
                    std::vector<float> embed(d, 0.0f);
                    this->T.getEmbedding(oddSentence[i][j], embed);
                    promptEmbeddings.push_back(embed);
                }
                for(int j = 0; j < evenSentence[i].size(); j++) {
                    std::vector<float> embed(d, 0.0f);
                    this->T.getEmbedding(evenSentence[i][j], embed);
                    responseEmbeddings.push_back(embed);
                    responseTokens.push_back(evenSentence[i][j]);
                }
                #ifdef USE_CUDA
                    this->T.cuTrain(promptEmbeddings, responseEmbeddings, responseTokens);
                #elif USE_OPENCL
                    this->T.clTrain(promptEmbeddings, responseEmbeddings, responseTokens);
                #elif USE_CPU
                    this->T.train(promptEmbeddings, responseEmbeddings, responseTokens);
                #endif
            }
            newChat();
        }
    }
}
