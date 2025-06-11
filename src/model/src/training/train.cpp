
#include "include/model.hpp"
#include <filesystem>
#include <fstream>
#include <neural.hpp>
#include <maths.hpp>
#include <chrono>

#ifdef USE_CUDA
    #include <cuda.h>
    #include <cuda_runtime.h>

    #define CUDA_CHECK(call)                                                        \
    do {                                                                            \
        cudaError_t err = call;                                                     \
        if (err != cudaSuccess) {                                                   \
            fprintf(stderr, "CUDA Error in %s at line %d: %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));                   \
            /* Consider throwing an exception or exiting */                         \
            throw std::runtime_error("CUDA Error: " + std::string(cudaGetErrorString(err)));    \
        }                                                                           \
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
 * @brief train first block using training data from txt file
 * @param txtFileLocation location of txt file (with each line as a sentence, 
 *          between two lines there is an empty line)
 */
void model::trainBlock(const std::string& txtFileLocation) 
{
    // basic checks for file
    if (txtFileLocation.empty()) {
        throw std::runtime_error("Training data file path cannot be empty!");
    }
    if (!std::filesystem::exists(txtFileLocation) || !std::filesystem::is_regular_file(txtFileLocation)) {
        throw std::runtime_error("Training data file not found or is not a regular file: " + txtFileLocation);
    }
    if (std::filesystem::path(txtFileLocation).extension() != ".txt") {
        throw std::runtime_error("Training data file must be a .txt file: " + txtFileLocation);
    }
    // open file and read line by line
    std::ifstream file(txtFileLocation);
    if (!file.is_open()) {
        throw std::runtime_error("Error opening training data file: " + txtFileLocation);
    }
    int numberOfLines = countLineInTXT(txtFileLocation);
    if (numberOfLines <= 0) {
        throw std::runtime_error("No training data found in the specified file!");
    }
    std::cout << "Total training lines: " << numberOfLines << std::endl;
    std::string line;

    // Start timing here
    auto startTime = std::chrono::high_resolution_clock::now();

    // get tokens
    std::vector<std::string> linesOfFile, tokensOfFile;
    while (std::getline(file, line)) {
        if(line.empty() == 1) {
            continue;
        }
        linesOfFile.push_back(line);
    }
    std::vector<std::vector<std::string>> oddSentence, evenSentence;
    for(int k = 0; k < numberOfLines; k++)
    {
        tokensOfFile.clear();
        oddSentence.clear();
        evenSentence.clear();
        T.currentTokenCount = 0;

        splitLine2SubSentences(linesOfFile[k], tokensOfFile);

        // Ensure tokensOfFile contains pairs of sub-sentences (prompt, response)
        if (tokensOfFile.size() % 2 != 0) {
            std::cerr << "Warning: Line " << k << " in file " << txtFileLocation 
                      << " resulted in an odd number of sub-sentences (" << tokensOfFile.size() 
                      << "). Skipping this line." << std::endl;
            continue; // Skip to the next line from the input file
        }

        if (tokensOfFile.empty() && k < numberOfLines) { // k < numberOfLines to ensure it's not just end of file
            std::cout << "Info: Line " << k << " in file " << txtFileLocation 
                      << " produced no tokens. Skipping." << std::endl;
            continue; // Skip to the next line
        }

        int num_pairs = tokensOfFile.size() / 2;
        if (num_pairs > 0) {
            oddSentence.resize(num_pairs);
            evenSentence.resize(num_pairs);
        }
        else if (!tokensOfFile.empty()) {
            std::cerr << "Warning: Line " << k << " has tokens but num_pairs is 0. Skipping." << std::endl;
            continue;
        }
        std::cout << "In line " << k << ". Number of pairs of Odd-Even Lines are: " << num_pairs << "." << std::endl;
        int tok = 0;        // token limit counter
        T.currentTokenCount = 0;      // 
        // Loop based on the number of pairs identified
        for(int i = 0; i < num_pairs; i++) {
            tokenize_with_numbers(tokensOfFile[2*i], oddSentence[i]);    // WILL TAKE EVEN INDEX (odd NUMBERED SENTENCE)
            tokenize_with_numbers(tokensOfFile[2*i+1], evenSentence[i]); // WILL TAKE ODD INDEX (odd NUMBERED SENTENCE)
            evenSentence.resize(evenSentence.size() + 1);
            evenSentence.back().push_back("@#0");
            std::vector<std::vector<float>> promptEmbeddings, responseEmbeddings;
            std::vector<std::string> responseTokens;
            // get embeddings for prompt
            for(int j = 0; j < oddSentence[i].size(); j++) {
                std::vector<float> embed(d, 0.0f);
                this->T.getEmbedding(oddSentence[i][j], embed);
                promptEmbeddings.push_back(embed);
                int actual_row_in_ev = (T.currentTokenCount + j) % CONTEXT_WIN;
                for(int m1 = 0; m1 < x; m1++) {
                    for(int m2 = 0; m2 < y; m2++) {
                        std::vector<float> v(EMBEDDING, 0.0f);
                        v = T.t[0].b[m1][m2].EV(actual_row_in_ev);
                        v += embed;
                        T.t[0].b[m1][m2].EV.addRow(v, actual_row_in_ev);
                    }
                }
            }
            // get embedding for response
            for(int j = 0; j < evenSentence[i].size(); j++) {
                std::vector<float> embed(d, 0.0f);
                this->T.getEmbedding(evenSentence[i][j], embed);
                responseEmbeddings.push_back(embed);
                responseTokens.push_back(evenSentence[i][j]);
            }
            tok += promptEmbeddings.size() + responseEmbeddings.size();
            // train the first block
            if(tok < CONTEXT_WIN)
            {
                #ifdef USE_CUDA
                    std::cout << "Using CUDA Implementation" << std::endl;
                    this->T.cuTrain(promptEmbeddings, responseEmbeddings, responseTokens);
                #elif USE_OPENCL
                    this->T.clTrain(promptEmbeddings, responseEmbeddings, responseTokens);
                #elif USE_CPU
                    this->T.train(promptEmbeddings, responseEmbeddings, responseTokens);
                #endif
            }
            else {
                throw std::runtime_error("LOCAL CONTEXT LIMIT REACHED");
            }
            totalTokens += tok;
        }
        std::cout << "complete " << k << "th part." << std::endl;
        newChat();
    }
    std::cout << "Training complete for file " << txtFileLocation << std::endl;
    
    T.t[0].serialise(T.t[0].blockFilePath);
    // copy to other blocks and serialise
    copy1toOhterBlocks();
    // for(int i = 1; i < m; i++) {
    //     T.t[i].serialise(T.t[i].blockFilePath);
    // }

    // End timing here
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "Total training time for file " << txtFileLocation << ": " << duration.count() << " ms" << std::endl;

    std::cout << "-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-" << std::endl;
}


/**
 * @brief train all blocks using training data from txt file
 * @param txtFile path 2 txt file for training 2 or more or all blocks
 */
void model::trainModel(const std::string& txtFile) 
{
    // basic checks for file
    if (txtFile.empty()) {
        throw std::runtime_error("Training data file path cannot be empty!");
    }
    if (!std::filesystem::exists(txtFile) || !std::filesystem::is_regular_file(txtFile)) {
        throw std::runtime_error("Training data file not found or is not a regular file: " + txtFile);
    }
    if (std::filesystem::path(txtFile).extension() != ".txt") {
        throw std::runtime_error("Training data file must be a .txt file: " + txtFile);
    }

    // open file and read line by line
    std::ifstream file(txtFile);
    if (!file.is_open()) {
        throw std::runtime_error("Error opening training data file: " + txtFile);
    }
    int numberOfLines = countLineInTXT(txtFile);
    if (numberOfLines <= 0) {
        throw std::runtime_error("No training data found in the specified file!");
    }
    std::cout << "Total training lines: " << numberOfLines << std::endl;
    std::string line;
    // Start timing here
    auto startTime = std::chrono::high_resolution_clock::now();

    // get tokens 
    std::vector<std::string> linesOfFile, tokensOfFile;
    while (std::getline(file, line) && !line.empty()) {
        linesOfFile.push_back(line);
    }
    std::vector<std::vector<std::string>> oddSentence, evenSentence;
    // get all values for matrices and mlps from block .bin files
    for(int i = 0; i < m; i++) {
        T.t[i].deserialise(T.t[i].blockFilePath);
    }
    for(int i = 0; i < numberOfLines; i++)
    {
        tokensOfFile.clear();
        oddSentence.clear();
        evenSentence.clear();
        splitLine2SubSentences(linesOfFile[i], tokensOfFile);
        if(oddSentence.size() != evenSentence.size() && oddSentence.size() > evenSentence.size()) {
            if(!oddSentence.empty() && !evenSentence.empty()) {
                evenSentence.back().insert(evenSentence.back().end(), oddSentence.back().begin(), oddSentence.back().end());
                oddSentence.pop_back();
            }
        }
        int tok = 0;
        // get embeddings and response
        for(int i = 0; i < oddSentence.size(); i++) {
            tokenize(tokensOfFile[2*i-1], oddSentence[i]);
            tokenize(tokensOfFile[2*i], evenSentence[i]);
            std::vector<std::vector<float>> promptEmbeddings, responseEmbeddings;
            std::vector<std::string> responseTokens;
            // get embeddings for prompt
            for(int j = 0; j < oddSentence[i].size(); j++) {
                std::vector<float> embed(d, 0.0f);
                this->T.getEmbedding(oddSentence[i][j], embed);
                promptEmbeddings.push_back(embed);
                for(int m1 = 0; m1 < x; m1++) {
                    for(int m2 = 0; m2 < y; m2++) {
                        int currentIdx = T.currentTokenCount - (T.blockCount - 1)*CONTEXT_WIN;
                        std::vector<float> v = T.t[0].b[i][j].EV(currentIdx);
                        v += embed;
                        T.t[0].b[i][j].EV.addRow(v, currentIdx-1);
                    }
                }
            }
            // get embedding for response
            for(int j = 0; j < evenSentence[i].size(); j++) {
                std::vector<float> embed(d, 0.0f);
                this->T.getEmbedding(evenSentence[i][j], embed);
                responseEmbeddings.push_back(embed);
                responseTokens.push_back(evenSentence[i][j]);
            }
            tok += promptEmbeddings[i].size() + responseEmbeddings[i].size();
            // train all blocks
            #ifdef USE_CUDA
                this->T.cuTrain(promptEmbeddings, responseEmbeddings, responseTokens);
            #elif USE_OPENCL
                this->T.clTrain(promptEmbeddings, responseEmbeddings, responseTokens);
            #elif USE_CPU
                this->T.train(promptEmbeddings, responseEmbeddings, responseTokens);
            #endif
            totalTokens += tok;
        }
    }
    // serialise all matrices and mlps to discrete .bin files
    for(int i = 0; i < m; i++) {
        T.t[i].serialise(T.t[i].blockFilePath);
    }
    serialise();
        // End timing here
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "Total training time for file " << txtFile << ": " << duration.count() << " ms" << std::endl;

    std::cout << "-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-" << std::endl;
}
