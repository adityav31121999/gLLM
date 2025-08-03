// support functions
#include <iostream>
#include <fstream>
#include <random>       // For std::mt19937 and std::uniform_real_distribution
#include <iomanip>      // For std::fixed and std::setprecision
#include <vector>
#include <algorithm>    // For std::sort
#include <cmath>        // For std::abs
#include "include/tokenise.hpp"


/**
 * @brief multiplicative inverse of a vector
 * @param vec vector input
 * @return inverse of vector
 */
std::vector<float> vectorInverse(const std::vector<float> &vec)
{
    std::vector<float> inverse(vec.size());
    float magnitudeOfVec = 0.0f;
    for (float val : vec) {
        magnitudeOfVec += val * val;
    }
    for (size_t i = 0; i < vec.size(); ++i) {
        inverse[i] = vec[i]/magnitudeOfVec;
    }
    return inverse;
}


/**
 * @brief Generates random seeds and computes embeddings for the current vocabulary.
 * This function populates the internal embedding vectors based on the tokens
 * currently stored in the class. It uses either the CPU, CUDA, or OpenCL
 * implementation to calculate the embeddings and their inverses. Finally,
 * it saves the token-embedding pairs to a specified CSV file.
 * @param outputPath The file path where the token-embedding CSV should be saved.
 * @param r1 The lower bound for the random seed generation.
 * @param r2 The upper bound for the random seed generation.
 * @throws std::runtime_error if the output file cannot be opened.
 */
void tokeniser::generateAndSaveEmbeddings(const std::string& embeddingCSVpath, float r1) {
    if (this->tokens.empty()) {
        throw std::runtime_error("Error: Vocabulary is not trained. Cannot generate embeddings.");
    }
    this->vocSize = this->tokens.size();
    std::cout << "-> Creating a temporary, lexicographically sorted token list for saving..." << std::endl;
    std::vector<std::string> sorted_tokens_for_saving = this->tokens;
    std::string csvEmbeddingOnly = embeddingCSVpath + "/_embeddings_only.csv";
    this->embeddings.resize(this->vocSize, std::vector<float>(this->d));
    
    #ifdef USE_CUDA
        // Call the CUDA kernel wrapper
        cuEmbeddingFormula(this->embeddings, this->seeds, this->d, this->vocSize, r1);
    #elif USE_OPENCL
        // Call the OpenCL kernel wrapper
        clEmbeddingFormula(this->ocl, this->embeddings, this->seeds, this->d, this->vocSize, r1);
    #else
        std::random_device rd;
        std::mt19937 gen(rd());
        // std::uniform_real_distribution<float> dis(r1, r2);
        std::poisson_distribution<int> dis(r1);
        this->embeddings.resize(this->vocSize);
        // resizinig of each row and assigning embeddings
        for (int i = 0; i < this->vocSize; ++i) {
            this->embeddings[i].resize(this->d);
            for (int j = 0; j < this->d; ++j) {
                // random number * (-1)^(i+j) * (sin(i+1) + cos(j-1)) = 0.01
                // possibly no zeroes allowed
                this->embeddings[i][j] = dis(gen) * (std::sin(i+1) + std::cos(j-1)) * 0.1 + 0.01;
            }
        }
    #endif

    std::cout << "-> Embedding generation complete." << std::endl;
    std::cout << "-> Saving only embeddings to: " << csvEmbeddingOnly << std::endl;
    std::ofstream outFile1(csvEmbeddingOnly);
    if (!outFile1.is_open()) {
        std::cerr << "Error: Could not open file to save embeddings: " << csvEmbeddingOnly << std::endl;
        return;
    }

    // Iterate over the *learned tokens* (this->tokens) to ensure consistency
    for (size_t i = 0; i < this->vocSize; ++i) {
        const std::string& token = this->tokens[i];
        std::vector<float> embedding = embeddings[i];
        for (float val : embedding) { 
            outFile1 << "," << val;
        } 
        outFile1 << "\n";
    }
    outFile1.close();
    std::cout << "Successfully saved " << this->tokens.size() << " embeddings to " << embeddingCSVpath << std::endl;
}


/**
 * @brief save deEmbeddings obtained from LLM training
 * @param[out] outputPath output path for deEmbedding csv
 * @param[in] deEmbedding de-embeddings from LLM training
 */
void tokeniser::savedeEmbeddings(const std::string &outputPath, const std::vector<std::vector<float>>& deEmebdding)
{
    // save deEmbeddings that is obtained from LLM training
    std::ofstream file(outputPath);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file for writing: " << outputPath << std::endl;
        return;
    }
    for (const auto& deEmbed : deEmebdding) {
        for (float val : deEmbed) {
            file << val << ",";
        }
        file << "\n";
    }
    file.close();
}