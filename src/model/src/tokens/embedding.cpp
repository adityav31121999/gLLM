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
    std::string csvEmbeddingOnly = embeddingCSVpath + "/_embeddings_only.csv";
    this->embeddings.resize(this->vocSize, std::vector<float>(this->d));
    
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
