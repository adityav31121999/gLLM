
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
#include <cstring>   // For memcpy, memmove

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
            }
            else {
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

    if (found) {
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
    else {
        // Word not found. Add it to vocabulary and generate/add embedding.

        // 1. Generate random values for the new embedding
        std::vector<float> new_embed_vec(this->d); // Use a temporary vector for the new embedding
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-10.0f, 10.0f);
        std::generate(new_embed_vec.begin(), new_embed_vec.end(), [&]() { return dis(gen); });

        // Determine the index where the new word will be inserted in the sorted tokens vector
        int new_index = static_cast<int>(std::distance(this->tokens.begin(), it));

        // 2. Insert the word into the tokens vector at the correct lexicographical position
        this->tokens.insert(it, word);

        // 3. Increment vocabulary size
        int old_vocab_size = this->vocabsize;
        this->vocabsize++;
        int new_vocab_size = this->vocabsize; // Which is old_vocab_size + 1

        // Resize the embeddings matrix to accommodate the new token
        // This will create a new mapped file and copy existing data
        mat new_embeddings(new_vocab_size, this->d);

        // Ensure the new matrix is successfully mapped
        if (!new_embeddings.mapped_data) {
             throw std::runtime_error("Failed to create or map new embeddings matrix during vocabulary expansion.");
        }

        // 5. Copy data from the old embeddings matrix to the new one, inserting the new embedding
        size_t embedding_size_bytes = static_cast<size_t>(this->d) * sizeof(float); // Size of one embedding row in bytes

        // Copy rows before the insertion point
        if (new_index > 0 && this->embeddings.mapped_data) {
            std::memcpy(new_embeddings.mapped_data, this->embeddings.mapped_data, static_cast<size_t>(new_index) * embedding_size_bytes);
        }

        // Copy the new embedding vector into the new matrix at the insertion point
        std::memcpy(new_embeddings.mapped_data + static_cast<size_t>(new_index) * this->d, new_embed_vec.data(), embedding_size_bytes);

        // Copy rows after the insertion point
        if (new_index < old_vocab_size && this->embeddings.mapped_data) {
            std::memcpy(new_embeddings.mapped_data + static_cast<size_t>(new_index + 1) * this->d, this->embeddings.mapped_data + static_cast<size_t>(new_index) * this->d, static_cast<size_t>(old_vocab_size - new_index) * embedding_size_bytes);
        }

        // 6. Replace the old embeddings matrix with the new one using move assignment
        this->embeddings = std::move(new_embeddings); // Replace old matrix with new one

        // 7. Update the index for the token (which is the insertion index)
        this->indexForToken = new_index;

        // 8. Copy the newly generated embedding to the output vector 'embed'
        embed = new_embed_vec;
    }
}
