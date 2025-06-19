
#include "include/model.hpp"
#include <neural.hpp>
#include <filesystem> // Required for directory iteration
#include <iostream>   // Required for std::cout, std::cerr
#include <stdexcept>  // Required for std::runtime_error
#include <maths.hpp>

/**
 * @brief train first block using training data from folder
 * @param txtFileLocation location of txt file (with each line as a sentence, 
 *          between two lines there is an empty line)
 */
void model::testBlock(const std::string& txtFileLocation) 
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

    // get tokens 
    std::vector<std::string> linesOfFile, tokensOfFile;
    while (std::getline(file, line) && !line.empty()) {
        linesOfFile.push_back(line);
    }
    std::vector<std::vector<std::string>> oddSentence, evenSentence;

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
            tokenize_with_numbers(tokensOfFile[2*i-1], oddSentence[i]);
            tokenize_with_numbers(tokensOfFile[2*i], evenSentence[i]);
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
            // train the first block
            if(tok < CONTEXT_WIN) 
            {
                #ifdef USE_CUDA
                    this->T.cuTest(promptEmbeddings, responseEmbeddings, responseTokens);
                #elif USE_OPENCL
                    this->T.clTest(promptEmbeddings, responseEmbeddings, responseTokens);
                #elif USE_CPU
                    this->T.test(promptEmbeddings, responseEmbeddings, responseTokens);
                #endif
            }
            else {
                throw std::runtime_error("LOCAL CONTEXT LIMIT of REACHED");
            }
            totalTokens += tok;
        }
    }
}


/**
 * @brief train all blocks using training data from folder
 * @param trainingData path to folder with all training data
 */
void model::testModel(const std::string& txtFileLocation) 
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

    // get tokens 
    std::vector<std::string> linesOfFile, tokensOfFile;
    while (std::getline(file, line) && !line.empty()) {
        linesOfFile.push_back(line);
    }
    std::vector<std::vector<std::string>> oddSentence, evenSentence;

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
            tokenize_with_numbers(tokensOfFile[2*i-1], oddSentence[i]);
            tokenize_with_numbers(tokensOfFile[2*i], evenSentence[i]);
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
                this->T.cuTest(promptEmbeddings, responseEmbeddings, responseTokens);
            #elif USE_OPENCL
                this->T.clTest(promptEmbeddings, responseEmbeddings, responseTokens);
            #elif USE_CPU
                this->T.test(promptEmbeddings, responseEmbeddings, responseTokens);
            #endif
            totalTokens += tok;
        }
    }
}
