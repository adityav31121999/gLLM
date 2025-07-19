
#include "include/tokenise.hpp"
#include <algorithm>
#include <iterator>
#include <thread>

/**
 * @brief set dimension of embedding
 * @param d dimension of embedding
 */
void tokeniser::setEmbeddingDimension(int d) {
    // Set the dimension for the embeddings.
    this->d = d;
}

/**
 * @brief set divisor for embedding calculation
 * @param d_val divisor for embedding
 */
void tokeniser::setDval(int d_val) {
    // Set the divisor for the embeddings.
    this->d_val = d_val;
}

/**
 * @brief set total vocabulary size
 * @param vocSize vocabulary size
 */
void tokeniser::setVocabularySize(int vocSize) {
    // Set the vocabulary size.
    this->vocSize = vocSize;
}

void tokeniser::setNumThreads()
{
    num_threads = std::thread::hardware_concurrency();
}


/**
 * @brief Sets the embedding for a given token.
 * This function searches for the specified token within the `tokens` vector.
 * If found, it updates the corresponding embedding in the `embeddings` vector
 * and also updates the `mappedEmbeddings` map to keep it in sync.
 * @param token The string token whose embedding is to be set.
 * @param embedding The vector representing the embedding for the token.
 */
void tokeniser::setEmbedding(const std::string& token, std::vector<float> embedding) {
    auto it = std::find(tokens.begin(), tokens.end(), token);

    // set embedding for this token from this->embeddings
    if (it != tokens.end()) {
        auto index = std::distance(tokens.begin(), it);
        embeddings[index] = std::move(embedding);
    }
}


/**
 * @brief Gets the embedding for a given token.
 * @param token The token to look up.
 * @return The embedding vector for the token. Returns an empty vector if not found.
 */
std::vector<float> tokeniser::getEmbeddingForToken(const std::string& token) const {
    auto it = std::find(tokens.begin(), tokens.end(), token);
    if (it != tokens.end()) {
        auto index = std::distance(tokens.begin(), it);
        return embeddings[index];
    }
    return {}; // Return empty vector if not found
}