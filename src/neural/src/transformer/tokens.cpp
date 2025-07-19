// tokens and their splitting for training
#include "include/transformer.hpp"
#include <sstream>
#include <string>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <algorithm>
#include <iterator>
#include <cctype>
#include <random>
#include <numeric>
#include <cstring>

/**
 * @brief Helper function to convert string to lowercase
 * @param str word to be converted to lower case
 */
inline std::string toLower(const std::string& str) {
    std::string lower_str;
    lower_str.reserve(str.length()); // Avoid reallocations
    std::transform(str.begin(), str.end(), std::back_inserter(lower_str),
                   [](unsigned char c){ return std::tolower(c); });
    return lower_str;
}


/**
 * @brief Get embedding for a given word. Searches case-insensitively.
 *        If the word is not found, it adds the lowercase version to the
 *        vocabulary, generates a new embedding (optionally averaged with neighbors),
 *        and updates internal state.
 * @param[in] word The token (string) to search for.
 * @param[out] embed Output vector where the found or newly generated embedding will be stored.
 */
void transformer::getEmbedding(std::string &word, std::vector<float> &embed) {
    // Ensure the output vector has the correct size
    if (embed.size() != static_cast<size_t>(this->d)) {
        embed.resize(this->d);
    }
    if (!this->embeddings.mapped_data) {
        throw std::runtime_error("Embeddings matrix is not mapped or initialized.");
    }

    std::string lower_word = toLower(word);
    auto it = std::lower_bound(this->tokens.begin(), this->tokens.end(), lower_word,
        [](const std::string& element_in_vocab, const std::string& target_value) {
            return toLower(element_in_vocab) < target_value;
        }
    );
    // Check if the exact lowercase match was found
    bool found = (it != this->tokens.end() && toLower(*it) == lower_word);

    int indexInTokens = std::distance(this->tokens.begin(), it);
    if (indexInTokens < this->vocabsize && indexInTokens < this->embeddings.row) {
        // Copy the embedding row from the mapped memory
        size_t row_offset = static_cast<size_t>(indexInTokens) * this->embeddings.col;
        if (static_cast<size_t>(this->embeddings.col) != embed.size()) {
                throw std::runtime_error("Internal error: Embed vector size mismatch during retrieval.");
        }
        std::memcpy(embed.data(), this->embeddings.mapped_data + row_offset, embed.size() * sizeof(float));
        this->indexForToken = indexInTokens;     // Update the member variable
    }
    else {
        std::cerr << "CRITICAL Error: Word '" << word << "' (lowercase: '" << lower_word << "') found in tokens at index "\
        << indexInTokens << ", but this index is out of bounds for used embeddings (vocabsize " << this->vocabsize << \
        ") or allocated embedding rows (mat.row " << this->embeddings.row << \
        "). Data inconsistency!" << std::endl;
        throw std::out_of_range("Embedding index out of range despite token match. Data corrupted?");
    }
}
