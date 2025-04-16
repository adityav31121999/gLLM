
// tokens and their splitting for training
#include "include/transformer.hpp" // Main header for the class definition
#include <sstream>
#include <string>
#include <vector>
#include <stdexcept> // For std::runtime_error, std::out_of_range
#include <iostream>
#include <algorithm> // For std::lower_bound, std::transform
#include <iterator>  // For std::distance, std::back_inserter
#include <cctype>    // For std::tolower
#include <random>    // For random number generation
#include <numeric>   // For potential accumulation (though manual loop is fine)

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
 * @brief tokenise the user prompt and place tokens into mTokens starting at currentTokenCount
 * @param words user prompt
 * @param mTokens Vector of strings where new tokens will be placed.
 * @param currentTokenCount The index in mTokens where the first new token should be placed.
 * @return The number of *new* tokens added to mTokens during this call. Returns 0 if initial 
 * index is invalid or capacity is exceeded immediately.
 */
int transformer::tokenise(std::string &words, std::vector<std::string>& mTokens, int currentTokenCount)
{
    if (currentTokenCount < 0 || currentTokenCount > FULL_CONTEXT) {
         std::cerr << "Error: Initial currentTokenCount (" << currentTokenCount
                   << ") is out of bounds for mTokens size (" << mTokens.size() << ")." << std::endl;
         return 0; // Cannot proceed
    }

    // stream of string that holds all strings after being separated on the basis of white space between them
    std::stringstream ss(words);
    std::string chunk; // Holds whitespace-separated chunks
    // punctuations
    std::string delimiters = " .,!?;:-~_+=/@#$&*`%^\\|\"\'(){}[]<>";
    int added_count = 0; // Counts only the tokens added in *this* function call
    int write_index = currentTokenCount; // The index in mTokens to write the next token
    // take each string of ss as input and copy it to chunk
    while (ss >> chunk) {
        std::string current_word_part;
        for (char c : chunk) {
            if (delimiters.find(c) != std::string::npos) {
                if (!current_word_part.empty()) {
                    if (write_index >= mTokens.size()) {
                        std::cerr << "Error: mTokens capacity (" << mTokens.size()
                                  << ") exceeded at index " << write_index << " during tokenization." << std::endl;
                        return added_count;
                    }
                    mTokens[write_index] = current_word_part;
                    write_index++;
                    added_count++;
                    current_word_part.clear();
                }
                if (write_index >= mTokens.size()) {
                    std::cerr << "Error: mTokens capacity (" << mTokens.size()
                               << ") exceeded at index " << write_index << " during tokenization." << std::endl;
                    return added_count;
                }
                mTokens[write_index] = std::string(1, c);
                write_index++;
                added_count++;
            } else {
                current_word_part += c;
            }
        }
        if (!current_word_part.empty()) {
             if (write_index >= mTokens.size()) {
                 std::cerr << "Error: mTokens capacity (" << mTokens.size()
                           << ") exceeded at index " << write_index << " during tokenization." << std::endl;
                 return added_count;
             }
            mTokens[write_index] = current_word_part;
            write_index++;
            added_count++;
        }
    }
    return added_count;
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
    std::string lower_word = toLower(word);
    auto it = std::lower_bound(this->tokens.begin(), this->tokens.end(), lower_word,
        [](const std::string& element_in_vocab, const std::string& target_value) {
            return toLower(element_in_vocab) < target_value;
        }
    );

    bool found = (it != this->tokens.end() && toLower(*it) == lower_word);

    if (found) {
        int indexInTokens = std::distance(this->tokens.begin(), it);
        if (static_cast<size_t>(indexInTokens) < this->embeddings.size()) {
            embed = this->embeddings[indexInTokens]; // Retrieve the existing embedding
            this->indexForToken = indexInTokens;     // Update the member variable
        } else {
            std::cerr << "CRITICAL Error: Word '" << word << "' (lowercase: '" << lower_word << "') found in tokens at index "\
            << indexInTokens << ", but this index is out of bounds for embeddings vector (size " << this->embeddings.size() << \
            "). Data inconsistency!" << std::endl;
            throw std::out_of_range("Embedding index out of range despite token match. Data corrupted?");
        }
    } else {
        if (this->d <= 0) {
             throw std::runtime_error("Cannot generate embedding: embedding dimension 'd' is not positive.");
        }
        if (this->tokens.size() != this->embeddings.size()) {
             std::cerr << "Warning: Mismatch between tokens size (" << this->tokens.size() << ") and embeddings size ("\
             << this->embeddings.size() << ") before adding new word '" << word << "' (lowercase: '" << lower_word <<\
             "'). Attempting to proceed." << std::endl;
        }
        if (static_cast<size_t>(this->vocabsize) != this->tokens.size()) {
            std::cerr << "Warning: Mismatch between vocabsize (" << this->vocabsize << ") and actual tokens size (" \
            << this->tokens.size() << ") before adding new word '" << word << "' (lowercase: '" << lower_word << \
            "'). Vocabsize will be corrected." << std::endl;
        }

        int insertIdx = std::distance(this->tokens.begin(), it);
        std::vector<float> newEmbedding(this->d);
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_real_distribution<float> distrib(-1.0f, 1.0f);
        for (int i = 0; i < this->d; ++i) {
            newEmbedding[i] = distrib(gen);
        }
        bool hasPrev = (insertIdx > 0);
        bool hasNext = (static_cast<size_t>(insertIdx) < this->tokens.size());
        if (hasPrev || hasNext) {
            std::vector<float> avgNeighbor(this->d, 0.0f);
            int neighborCount = 0;
            if (hasPrev) {
                 if (static_cast<size_t>(insertIdx - 1) < this->embeddings.size()) {
                    const auto& prevEmbed = this->embeddings[insertIdx - 1];
                    if (prevEmbed.size() == static_cast<size_t>(this->d)) {
                        for(int i = 0; i < this->d; ++i) {
                            avgNeighbor[i] += prevEmbed[i];
                        }
                        neighborCount++;
                    } else {
                        std::cerr << "Warning: Size mismatch for previous neighbor embedding (index " << insertIdx - 1
                                  << ", token '" << this->tokens[insertIdx - 1] << "'). Skipping for average." << std::endl;
                    }
                }
            }
            if (hasNext) {
                const auto& nextEmbed = this->embeddings[insertIdx];
                if (nextEmbed.size() == static_cast<size_t>(this->d)) { // Size check
                    for(int i = 0; i < this->d; ++i) {
                        avgNeighbor[i] += nextEmbed[i];
                    }
                    neighborCount++;
                } else {
                    std::cerr << "Warning: Size mismatch for next neighbor embedding (index " << insertIdx
                            << ", token '" << this->tokens[insertIdx] << "'). Skipping for average." << std::endl;
                }
            }

            if (neighborCount > 0) {
                float invNeighborCount = 1.0f / static_cast<float>(neighborCount);
                float random_weight = 0.5f;
                float neighbor_weight = 0.5f;
                for (int i = 0; i < this->d; ++i) {
                    newEmbedding[i] = (newEmbedding[i] * random_weight) + (avgNeighbor[i] * invNeighborCount * neighbor_weight);
                }
            }
        }
        this->tokens.insert(it, lower_word);
        this->embeddings.insert(this->embeddings.begin() + insertIdx, newEmbedding);
        this->vocabsize++; 
        if (this->tokens.size() != this->embeddings.size() || this->tokens.size() != static_cast<size_t>(this->vocabsize)) {
            std::cerr << "CRITICAL Error: Size mismatch after inserting '" << lower_word << "'! "
                      << "Tokens: " << this->tokens.size()
                      << ", Embeddings: " << this->embeddings.size()
                      << ", Vocabsize: " << this->vocabsize << std::endl;
            throw std::runtime_error("Vocabulary and embedding size inconsistency after insertion.");
        }

        embed = newEmbedding;
        this->indexForToken = insertIdx;
    }
}
