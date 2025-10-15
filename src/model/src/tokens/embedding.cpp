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
 * @throws std::runtime_error if the output file cannot be opened.
 */
void tokeniser::generateAndSaveEmbeddings(const std::string& embeddingCSVpath, float r1, float r2) {
    if (tokens.empty()) {
        throw std::runtime_error("Error: Vocabulary is not trained. Cannot generate embeddings.");
    }
    vocSize = tokens.size();
    // The `tokens` member is sorted by length for the split algorithm.
    // We must save embeddings in a consistent, lexicographical order to match the stats file.
    std::vector<std::string> sorted_tokens = tokens;
    std::sort(sorted_tokens.begin(), sorted_tokens.end());

    std::string csvEmbeddingOnly = embeddingCSVpath + "/_embeddings_only.csv";
    embeddings.resize(vocSize, std::vector<float>(d));
    
    #ifdef USE_CUDA
        // Call the CUDA kernel wrapper
        cuEmbeddingFormula(embeddings, seeds, d, vocSize, r1, r2);
    #elif USE_OPENCL
        // Call the OpenCL kernel wrapper
        clEmbeddingFormula(ocl, embeddings, seeds, d, vocSize, r1, r2);
    #else
        std::random_device rd;
        std::mt19937 gen(rd());
        // std::uniform_real_distribution<float> dis(r1, r2); // (-2.50f, 5.0f);
        std::uniform_real_distribution<float> dis(r1, r2);
        // The embeddings are generated for the length-sorted `tokens` order.
        for (int i = 0; i < vocSize; ++i) {
            // `embeddings` will be in the same (length-sorted) order as `tokens`
            embeddings[i].resize(d);
            for (int j = 0; j < d; ++j) {
                // random number * (-1)^(i+j) * (sin(i+1) + cos(j-1)) = 0.01
                // possibly no zeroes allowed
                embeddings[i][j] = dis(gen) * (std::sin(i+1) + std::cos(j-1)) * 0.01f + 0.1f;
            }
        }
    #endif

    std::cout << "-> Embedding generation complete." << std::endl;

    // Create a map from token string to its generated embedding for easy lookup.
    // The embeddings in `embeddings` correspond to the length-sorted `tokens`.
    std::unordered_map<std::string, std::vector<float>> token_to_embedding_map;
    token_to_embedding_map.reserve(vocSize);
    for (size_t i = 0; i < vocSize; ++i) {
        token_to_embedding_map[tokens[i]] = embeddings[i];
    }

    std::cout << "-> Saving only embeddings to: " << csvEmbeddingOnly << std::endl;
    std::ofstream outFile1(csvEmbeddingOnly);
    if (!outFile1.is_open()) {
        std::cerr << "Error: Could not open file to save embeddings: " << csvEmbeddingOnly << std::endl;
        return;
    }

    // Now, iterate over the LEXICOGRAPHICALLY sorted tokens and save their embeddings.
    // This ensures the embeddings file has the same token order as the stats file.
    for (const auto& token : sorted_tokens) {
        const auto& embedding = token_to_embedding_map.at(token);
        for (size_t j = 0; j < embedding.size(); ++j) {
            outFile1 << embedding[j] << (j == embedding.size() - 1 ? "" : ",");
        }
        outFile1 << "\n";
    }
    outFile1.close();
    std::cout << "Successfully saved " << sorted_tokens.size() << " embeddings to " << csvEmbeddingOnly << std::endl;
}


/**
 * @brief Generates random seeds and computes embeddings for the current vocabulary.
 * This function populates the internal embedding vectors based on the tokens
 * currently stored in the class. It uses either the CPU, CUDA, or OpenCL
 * implementation to calculate the embeddings and their inverses. Finally,
 * it saves the token-embedding pairs to a specified CSV file.
 * @param outputPath The file path where the token-embedding CSV should be saved.
 * @param r1 The lower bound for the random seed generation.
 * @throws std::runtime_error if the output file cannot be opened.
 */
void tokeniser::generateAndSavedeEmbeddings(const std::string& embeddingCSVpath, float r1, float r2) {
    if (tokens.empty()) {
        throw std::runtime_error("Error: Vocabulary is not trained. Cannot generate embeddings.");
    }
    vocSize = tokens.size();
    // The `tokens` member is sorted by length for the split algorithm.
    // We must save embeddings in a consistent, lexicographical order to match the stats file.
    std::vector<std::string> sorted_tokens = tokens;
    std::sort(sorted_tokens.begin(), sorted_tokens.end());

    std::string csvEmbeddingOnly = embeddingCSVpath + "/_embeddings_only.csv";
    embeddings.resize(vocSize, std::vector<float>(d));
    
    #ifdef USE_CUDA
        // Call the CUDA kernel wrapper
        cudeEmbeddingFormula(embeddings, seeds, d, vocSize, r1, r2);
    #elif USE_OPENCL
        // Call the OpenCL kernel wrapper
        cldeEmbeddingFormula(ocl, embeddings, seeds, d, vocSize, r1, r2);
    #else
        std::random_device rd;
        std::mt19937 gen(rd());
        // std::uniform_real_distribution<float> dis(r1, r2); // (-2.50f, 5.0f);
        std::uniform_real_distribution<float> dis(r1, r2);
        // The embeddings are generated for the length-sorted `tokens` order.
        for (int i = 0; i < vocSize; ++i) {
            // `embeddings` will be in the same (length-sorted) order as `tokens`
            embeddings[i].resize(d);
            for (int j = 0; j < d; ++j) {
                // random number * (-1)^(i+j) * (sin(i+1) + cos(j-1)) = 0.01
                // possibly no zeroes allowed
                embeddings[i][j] = dis(gen) * (std::sin(i+1) + std::cos(j-1)) * 0.01f + 0.1f;
            }
        }
    #endif

    std::cout << "-> Embedding generation complete." << std::endl;

    // Create a map from token string to its generated embedding for easy lookup.
    // The embeddings in `embeddings` correspond to the length-sorted `tokens`.
    std::unordered_map<std::string, std::vector<float>> token_to_embedding_map;
    token_to_embedding_map.reserve(vocSize);
    for (size_t i = 0; i < vocSize; ++i) {
        token_to_embedding_map[tokens[i]] = embeddings[i];
    }

    std::cout << "-> Saving only embeddings to: " << csvEmbeddingOnly << std::endl;
    std::ofstream outFile1(csvEmbeddingOnly);
    if (!outFile1.is_open()) {
        std::cerr << "Error: Could not open file to save embeddings: " << csvEmbeddingOnly << std::endl;
        return;
    }

    // Now, iterate over the LEXICOGRAPHICALLY sorted tokens and save their embeddings.
    // This ensures the embeddings file has the same token order as the stats file.
    for (const auto& token : sorted_tokens) {
        const auto& embedding = token_to_embedding_map.at(token);
        for (size_t j = 0; j < embedding.size(); ++j) {
            outFile1 << embedding[j] << (j == embedding.size() - 1 ? "" : ",");
        }
        outFile1 << "\n";
    }
    outFile1.close();
    std::cout << "Successfully saved " << sorted_tokens.size() << " embeddings to " << csvEmbeddingOnly << std::endl;
}


/**
 * @brief save deEmbeddings obtained from LLM training
 * @param[out] outputPath output path for deEmbedding csv
 * @param[in] embedding embeddings from LLM training
 */
void tokeniser::saveEmbeddings(const std::string &outputPath, const std::vector<std::vector<float>>& embedding)
{
    // save deEmbeddings that is obtained from LLM training
    std::ofstream file(outputPath);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file for writing: " << outputPath << std::endl;
        return;
    }
    for (const auto& embed : embedding) {
        for (size_t j = 0; j < embed.size(); ++j) {
            file << embed[j] << (j == embed.size() - 1 ? "" : ",");
        }
        file << "\n";
    }
    file.close();
    std::cout << "Embeddings Saved :)" << std::endl;
}

/**
 * @brief save deEmbeddings obtained from LLM training
 * @param[out] outputPath output path for deEmbedding csv
 * @param[in] embeddings embeddings from LLM training
 */
void tokeniser::saveEmbeddings(const std::string &outputPath, const float *embedding)
{
    // save deEmbeddings that is obtained from LLM training
    std::ofstream file(outputPath);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file for writing: " << outputPath << std::endl;
        return;
    }
    for (int i = 0; i < vocSize; i++) {
        for (size_t j = 0; j < embeddings[0].size(); j++) {
            file << embedding[i*embeddings[0].size() + j] << (j == embeddings[0].size() - 1 ? "" : ",");
        }
        file << "\n";
    }
    file.close();
    std::cout << "Embeddings Saved :)" << std::endl;
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
        for (size_t j = 0; j < deEmbed.size(); ++j) {
            file << deEmbed[j] << (j == deEmbed.size() - 1 ? "" : ",");
        }
        file << "\n";
    }
    file.close();
    std::cout << "DeEmbeddings Saved :)" << std::endl;
}

/**
 * @brief save deEmbeddings obtained from LLM training
 * @param[out] outputPath output path for deEmbedding csv
 * @param[in] deEmbedding de-embeddings from LLM training
 */
void tokeniser::savedeEmbeddings(const std::string &outputPath, const float *deEmbedding)
{
    // save deEmbeddings that is obtained from LLM training
    std::ofstream file(outputPath);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file for writing: " << outputPath << std::endl;
        return;
    }
    for (int i = 0; i < vocSize; i++) {
        for (size_t j = 0; j < deEmbeddings[0].size(); j++) {
            file << deEmbedding[i*deEmbeddings[0].size() + j] << (j == deEmbeddings[0].size() - 1 ? "" : ",");
        }
        file << "\n";
    }
    file.close();
    std::cout << "DeEmbeddings Saved :)" << std::endl;
}