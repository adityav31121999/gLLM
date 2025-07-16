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
void tokeniser::generateAndSaveEmbeddings(const std::string& embeddingCSVpath, float r1, float r2) {
    if (this->tokens.empty()) {
        throw std::runtime_error("Error: Vocabulary is not trained. Cannot generate embeddings.");
    }
    this->vocSize = this->tokens.size();

    std::cout << "-> Creating a temporary, lexicographically sorted token list for saving..." << std::endl;
    std::vector<std::string> sorted_tokens_for_saving = this->tokens;
    // std::sort(sorted_tokens_for_saving.begin(), sorted_tokens_for_saving.end());
    std::unordered_map<std::string, int> token_to_original_index;

    for(int i = 0; i < this->vocSize; ++i) {
        token_to_original_index[this->tokens[i]] = i;
    }

    std::string csvEmbeddingOnly = embeddingCSVpath + "/_embeddings_only.csv";
    std::string tokenEmbeddingcsv = embeddingCSVpath + "/_tokenEmbedding.csv";
    // std::string csvDeEmbeddings = embeddingCSVpath + "/_deEmbedding.csv";
    this->embeddings.resize(this->vocSize, std::vector<float>(this->d));
    // this->deEmbeddings.resize(this->vocSize, std::vector<float>(this->d));   // or multiple of d (m*d)
    
    #ifdef USE_CUDA
        // Call the CUDA kernel wrapper
        cuEmbeddingFormula(this->embeddings, this->seeds, this->d, this->vocSize, r1, r2);
    #elif USE_OPENCL
        // Call the OpenCL kernel wrapper
        clEmbeddingFormula(this->ocl, this->embeddings, this->seeds, this->d, this->vocSize, r1, r2);
    #else
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(r1, r2);

        // Ensure the embeddings vector is correctly sized before populating
        // This is a crucial step if embeddings is not pre-sized in a constructor.
        this->embeddings.resize(this->vocSize);
        for (int i = 0; i < this->vocSize; ++i) {
            this->embeddings[i].resize(this->d);
        }

        // Fallback to CPU implementation
        for (int i = 0; i < this->vocSize; ++i) {
            // Note: `i` here corresponds to the index in the original `this->tokens` list
            for (int j = 0; j < this->d; ++j) {
                this->embeddings[i][j] = dis(gen); // <--- CORRECTED: Call dis with the generator 'gen'
            }
        }
    #endif

    std::cout << "-> Embedding generation complete." << std::endl;
    std::cout << "-> Saving only embeddings to: " << csvEmbeddingOnly << std::endl;
    std::cout << "-> Saving tokens and embeddings to: " << tokenEmbeddingcsv << std::endl;
    std::ofstream outFile1(csvEmbeddingOnly);
    std::ofstream outFile2(tokenEmbeddingcsv);
    if (!outFile1.is_open()) {
        std::cerr << "Error: Could not open file to save embeddings: " << csvEmbeddingOnly << std::endl;
        return;
    }
    if (!outFile2.is_open()) {
        std::cerr << "Error: Could not open file to save tokens and embeddings: " << tokenEmbeddingcsv << std::endl;
        return;
    }

    // Iterate over the *learned tokens* (this->tokens) to ensure consistency
    for (size_t i = 0; i < this->vocSize; ++i) {
        const std::string& token = this->tokens[i];
        std::vector<float> embedding = embeddings[i];

        // Store in mappedEmbeddings for later use (optional, but good practice)
        this->mappedEmbeddings[token] = embedding;

        // Write to CSV, handling quoting for tokens
        std::string escaped_token = token;
        outFile2 << "\"" << escaped_token << "\""; // Always quote the token field

        for (float val : embedding) { outFile1 << "," << val; } outFile1 << "\n";
        for (float val : embedding) { outFile2 << "," << val; } outFile2 << "\n";
    }
    outFile1.close();
    std::cout << "Successfully saved " << this->tokens.size() << " embeddings to " << embeddingCSVpath << std::endl;
}
