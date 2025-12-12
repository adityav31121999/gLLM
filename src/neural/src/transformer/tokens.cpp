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
#include <numeric>
#include <cstring>
#include "transformer.hpp"

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
    if (embed.size() != static_cast<size_t>(d)) {
        embed.resize(d);
    }

    std::string lower_word = toLower(word);
    auto it = std::lower_bound(tokens.begin(), tokens.end(), lower_word);
    // Check if the exact lowercase match was found
    bool found = (it != tokens.end() && *it == lower_word);

    if (found) {
        int indexInTokens = std::distance(tokens.begin(), it);
        if (indexInTokens < vocabsize && indexInTokens < embeddings.row) {
            // Copy the embedding row from the mapped memory
            size_t row_offset = static_cast<size_t>(indexInTokens) * embeddings.col;
            if (static_cast<size_t>(embeddings.col) != embed.size()) {
                 throw std::runtime_error("Internal error: Embed vector size mismatch during retrieval.");
            }
            std::memcpy(embed.data(), embeddings.mapped_data + row_offset, embed.size() * sizeof(float));
            indexForToken = indexInTokens;     // Update the member variable
            std::cout << "Embedding found for token: \t\'" << word << "\' at index \t" << indexInTokens << std::endl;
        }
        else {
            std::cerr << "CRITICAL Error: Word '" << word << "' (lowercase: '" << lower_word << "') found in tokens at index "\
            << indexInTokens << ", but this index is out of bounds for used embeddings (vocabsize " << vocabsize << \
            ") or allocated embedding rows (mat.row " << embeddings.row << \
            "). Data inconsistency!" << std::endl;
            throw std::out_of_range("Embedding index out of range despite token match. Data corrupted?");
        }
    }
    else {
       std::cout << "Word '" << word << "' (lowercase: '" << lower_word << "') not found in vocabulary." << std::endl;
        /*
        // Word not found. Add it to vocabulary and generate/add embedding.
        // 1. Generate random values for the new embedding
        std::vector<float> new_embed_vec(d); // Use a temporary vector for the new embedding
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-10.0f, 10.0f);
        std::generate(new_embed_vec.begin(), new_embed_vec.end(), [&]() { return dis(gen); });

        // Determine the index where the new word will be inserted in the sorted tokens vector
        int new_index = static_cast<int>(std::distance(tokens.begin(), it));

        // 2. Insert the word into the tokens vector at the correct lexicographical position
        tokens.insert(it, word);

        // 3. Increment vocabulary size
        int old_vocab_size = vocabsize;
        vocabsize++;
        int new_vocab_size = vocabsize; // Which is old_vocab_size + 1

        // Resize the embeddings matrix to accommodate the new token
        // This will create a new mapped file and copy existing data
        mat new_embeddings(new_vocab_size, d);

        // Ensure the new matrix is successfully mapped
        if (!new_embeddings.mapped_data) {
             throw std::runtime_error("Failed to create or map new embeddings matrix during vocabulary expansion.");
        }

        // 5. Copy data from the old embeddings matrix to the new one, inserting the new embedding
        size_t embedding_size_bytes = static_cast<size_t>(d) * sizeof(float); // Size of one embedding row in bytes

        // Copy rows before the insertion point
        if (new_index > 0 && embeddings.mapped_data) {
            std::memcpy(new_embeddings.mapped_data, embeddings.mapped_data, static_cast<size_t>(new_index) * embedding_size_bytes);
        }

        // Copy the new embedding vector into the new matrix at the insertion point
        std::memcpy(new_embeddings.mapped_data + static_cast<size_t>(new_index) * d, new_embed_vec.data(), embedding_size_bytes);

        // Copy rows after the insertion point
        if (new_index < old_vocab_size && embeddings.mapped_data) {
            std::memcpy(new_embeddings.mapped_data + static_cast<size_t>(new_index + 1) * d, embeddings.mapped_data + static_cast<size_t>(new_index) * d, static_cast<size_t>(old_vocab_size - new_index) * embedding_size_bytes);
        }

        // 6. Replace the old embeddings matrix with the new one using move assignment
        embeddings = std::move(new_embeddings); // Replace old matrix with new one

        // 7. Update the index for the token (which is the insertion index)
        indexForToken = new_index;

        // 8. Copy the newly generated embedding to the output vector 'embed'
        embed = new_embed_vec;
        */
       std::cout << "Current vocabulary size: " << vocabsize << std::endl;
    }
}
