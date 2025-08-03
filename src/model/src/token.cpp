
#include "include/model.hpp"
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <locale>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <set>
#include <map>
#include <random>

// Helper function to check if a character is a digit
static bool is_digit(char c) {
    return c >= '0' && c <= '9';
}

// Helper function to check if a character is a delimiter for sub-sentences
static bool is_sub_sentence_delimiter(char c) {
    return c == '.' || c == '!' || c == '?' || c == ':';
}


/**
 * @brief Split a line into sub-sentences based on delimiters (., !, ?, :) and store them in a vector.
 * @param line The line to split.
 * @param subSentences The vector to store the sub-sentences.
 */
void splitLine2SubSentences(std::string& line, std::vector<std::string>& subSentences)
{
    if (line.empty()) {
        return;
    }
    subSentences.clear();
    std::string current_sub_sentence;
    const std::locale loc;
    // We use subSentences.size() to determine the 0-based index 
    // of the sub-sentence *before* it's added.
    // 1st sentence: index 0 (subSentences.size() == 0) -> odd-positioned
    // 2nd sentence: index 1 (subSentences.size() == 1) -> even-positioned
    // 3rd sentence: index 2 (subSentences.size() == 2) -> odd-positioned

    for (char c : line) {
        current_sub_sentence += c;
        if (is_sub_sentence_delimiter(c)) {
            // Check if the sub-sentence actually contains non-whitespace content
            bool contains_non_whitespace = false;
            for (char sentence_char : current_sub_sentence) {
                if (!std::isspace(sentence_char, loc)) {
                    contains_non_whitespace = true; // contains characters
                    break;
                }
            }
            if (contains_non_whitespace) {
                if (subSentences.size() % 2 == 1) { // Even-positioned (2nd, 4th, etc.)
                    // current_sub_sentence += "@#0";
                }
                subSentences.push_back(current_sub_sentence);
            }
            current_sub_sentence.clear();
        }
    }
    // Add any remaining part as the last sub-sentence if it's not empty
    // and contains non-whitespace characters.
    if (!current_sub_sentence.empty()) {
        bool tail_contains_non_whitespace = false;
        for (char char_in_remaining : current_sub_sentence) {
            if (!std::isspace(char_in_remaining, loc)) {
                tail_contains_non_whitespace = true;
                break;
            }
        }
        if (tail_contains_non_whitespace) {
            if (subSentences.size() % 2 == 1) { // Even-positioned (2nd, 4th, etc.)
                // current_sub_sentence += "@#0";
            }
            subSentences.push_back(current_sub_sentence);
        }
    }

    // If, after collecting all valid sub-sentences, there's an odd number
    // and more than one, concatenate the last two.
    // This aims to create pairs for prompt/response processing.
    if (subSentences.size() > 1 && subSentences.size() % 2 == 1) {
        std::string last_sentence_to_merge = subSentences.back();
        subSentences.pop_back(); // Remove the original last element
        // Append the (original) last sentence to the (new) last element
        subSentences.back() += last_sentence_to_merge;
    }
}

/**
 * @brief split sentences of txt file based on delimiters (., !, ?, :)
 * @param path2file path of txt file
 * @param tokensOfFile all tokens of file in this vector
 * @param oddSentence sentences which are first, third, etc.
 * @param evenSentence sentences which are second, fourth, etc.
 */
void textSplit(std::string &path2file, std::vector<std::string> &tokensOfFile, std::vector<std::vector<std::string>> &oddSentence, 
                std::vector<std::vector<std::string>> &evenSentence)
{
    tokensOfFile.clear();
    oddSentence.clear();
    evenSentence.clear();

    std::ifstream ifs(path2file);
    if (!ifs.is_open()) {
        std::cerr << "Error: Could not open file: " << path2file << std::endl;
        return; // Or throw an exception
    }

    std::string current_sentence;
    char c;
    int sentence_count = 0;
    std::locale loc; // For checking whitespace

    while (ifs.get(c)) {
        current_sentence += c;

        // Check if the character is a sentence delimiter
        if (c == '.' || c == '!' || c == '?' || c == ':') {
            // Check if the sentence actually contains non-whitespace content
            bool contains_non_whitespace = false;
            for (char sentence_char : current_sentence) {
                if (!std::isspace(sentence_char, loc)) {
                    contains_non_whitespace = true;
                    break;
                }
            }

            if (contains_non_whitespace) {
                std::vector<std::string> sentence_vec = {current_sentence};
                if (sentence_count % 2 == 0) {
                    oddSentence.push_back(sentence_vec);
                } 
                else {
                    evenSentence.push_back(sentence_vec);
                }
                sentence_count++;
            }
            current_sentence.clear();
        }
    }

    bool contains_non_whitespace = false;
    for (char sentence_char : current_sentence) {
        if (!std::isspace(sentence_char, loc)) {
            contains_non_whitespace = true;
            break;
        }
    }
    if (contains_non_whitespace) {
        std::vector<std::string> sentence_vec = {current_sentence};
        if (sentence_count % 2 == 0) {
            oddSentence.push_back(sentence_vec);
        } 
        else {
            evenSentence.push_back(sentence_vec);
        }
    }
}


/**
 * @brief Set the embedding from a CSV file.
 * @param path2file The path to the CSV file.
 */
void model::setEmbeddingFromCSV(const std::string& path2file) {
    if (path2file.empty()) {
        throw std::invalid_argument("Path to CSV file cannot be empty.");
    }

    std::ifstream file(path2file);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open CSV file: " + path2file);
    }

    // Verify that T.vocabsize and T.d are valid for loading embeddings
    if (this->T.vocabsize <= 0 || this->T.d <= 0) {
        file.close();
        throw std::runtime_error("Cannot load embeddings: Transformer vocabsize (" + std::to_string(this->T.vocabsize) +
                                 ") or embedding dimension (" + std::to_string(this->T.d) +
                                 ") is zero or negative.");
    }

    // T.embeddings should have been initialized by the transformer's constructor to these dimensions.
    if (this->T.embeddings.row != this->T.vocabsize || this->T.embeddings.col != this->T.d) {
        file.close();
        std::string error_msg = "T.embeddings dimensions mismatch. ";
        error_msg += "Expected (from T.vocabsize, T.d): " + std::to_string(this->T.vocabsize) + "x" + std::to_string(this->T.d) + ". ";
        error_msg += "Actual T.embeddings: " + std::to_string(this->T.embeddings.row) + "x" + std::to_string(this->T.embeddings.col) + ".";
        throw std::runtime_error(error_msg);
    }

    if (!this->T.embeddings.mapped_data) {
        // This should ideally not happen if row/col are > 0 due to mat constructor logic.
        file.close();
        throw std::runtime_error("T.embeddings is not mapped (mapped_data is null), "
                                 "but dimensions are " + std::to_string(this->T.embeddings.row) + "x" + std::to_string(this->T.embeddings.col) + ".");
    }

    this->T.tokens.clear();
    this->T.tokens.reserve(this->T.vocabsize); // Pre-allocate memory

    std::string line;
    int csv_line_number = 0;
    int embeddings_row_idx = 0; // Index for T.tokens and T.embeddings

    std::cout << "Reading embeddings from " << path2file << " for vocab size " << this->T.vocabsize << " and dimension " << this->T.d << "..." << std::endl;

    while (std::getline(file, line)) {
        csv_line_number++;
        if (embeddings_row_idx >= this->T.vocabsize) {
            std::cerr << "Warning: CSV file '" << path2file << "' (line " << csv_line_number
                      << ") has more lines than expected vocab size (" << this->T.vocabsize
                      << "). Ignoring extra lines." << std::endl;
            break;
        }

        std::stringstream ss(line);
        std::string token_str;
        std::string cell;

        if (!std::getline(ss, token_str, ',')) {
            std::cerr << "Warning: Malformed CSV line " << csv_line_number << " in '" << path2file
                      << "'. Could not read token. Skipping line." << std::endl;
            continue; 
        }

        this->T.tokens.push_back(token_str);
        bool critical_error_in_line = false;

        // Loop to read embedding values for the current token
        for (int j = 0; j < this->T.d; ++j) {
            if (!std::getline(ss, cell, ',')) {
                std::cerr << "Error: CSV line " << csv_line_number << " (token: '" << token_str << "') in '" << path2file
                          << "' does not have enough embedding values. Expected " << this->T.d
                          << ", found " << j << ". Filling rest with 0.0f for this token." << std::endl;
                for (int k = j; k < this->T.d; ++k) {
                    try {
                        this->T.embeddings(embeddings_row_idx, k) = 0.0f;
                    } 
                    catch (const std::out_of_range& oor_ex) {
                         std::cerr << "Critical Error: Out of range access to T.embeddings at (" << embeddings_row_idx << ", " << k
                                  << ") while filling with zeros for missing values. " << oor_ex.what() << std::endl;
                        critical_error_in_line = true; break;
                    }
                }
                break; // Break from the for-j loop (reading columns for current token)
            }
            if(critical_error_in_line) break;

            float val_from_stof = 0.0f;
            try {
                val_from_stof = std::stof(cell);
            } 
            catch (const std::invalid_argument& ia) {
                std::cerr << "Warning: Invalid number format for embedding value '" << cell << "' at CSV line "
                          << csv_line_number << ", column " << j + 1 << " (token: '" << token_str
                          << "') in '" << path2file << "'. Using 0.0f. Error: " << ia.what() << std::endl;
                // val_from_stof remains 0.0f
            } 
            catch (const std::out_of_range& oor_stof) { 
                std::cerr << "Warning: Number out of range for embedding value '" << cell << "' at CSV line "
                          << csv_line_number << ", column " << j + 1 << " (token: '" << token_str
                          << "') in '" << path2file << "' (std::stof). Using 0.0f. Error: " << oor_stof.what() << std::endl;
                // val_from_stof remains 0.0f
            }

            try {
                this->T.embeddings(embeddings_row_idx, j) = val_from_stof;
            }
            catch (const std::out_of_range& oor_mat_assign) { 
                 std::cerr << "Critical Error: Out of range access to T.embeddings at (" << embeddings_row_idx << ", " << j
                           << ") for token '" << token_str << "'. CSV line " 
                           << csv_line_number << ". Error: " << oor_mat_assign.what() << std::endl;
                 critical_error_in_line = true;
            }

            if(critical_error_in_line) break; // Break from for-j loop if matrix assignment failed critically
        } // End of for-j loop (columns)
        
        if (critical_error_in_line) {
            if (!this->T.tokens.empty() && this->T.tokens.back() == token_str) {
                this->T.tokens.pop_back();
            }
            std::cerr << "Skipping CSV line " << csv_line_number << " due to critical error during embedding processing for token '" << token_str << "'." << std::endl;
            continue; // Continue to the next line in the CSV file
        }

        if (std::getline(ss, cell, ',')) {
            std::cerr << "Warning: CSV line " << csv_line_number << " (token: '" << token_str << "') in '" << path2file
                      << "' has more columns than expected embedding dimension (" << this->T.d
                      << "). Ignoring extra columns." << std::endl;
        }
        embeddings_row_idx++; // Successfully processed a row
    }

    file.close();

    if (embeddings_row_idx < this->T.vocabsize) {
        std::cerr << "Warning: Loaded " << embeddings_row_idx << " tokens/embeddings from CSV '" << path2file
                  << "', which is fewer than the expected vocab size (" << this->T.vocabsize
                  << "). The remaining " << (this->T.vocabsize - embeddings_row_idx)
                  << " entries in T.embeddings will be zeroed." << std::endl;
        for (int i = embeddings_row_idx; i < this->T.vocabsize; ++i) {
            for (int j = 0; j < this->T.d; ++j) {
                try {
                    this->T.embeddings(i, j) = 0.0f;
                } 
                catch (const std::out_of_range& oor_ex) {
                    std::cerr << "Critical Error: Out of range access to T.embeddings at (" << i << ", " << j
                              << ") while zeroing remaining entries. " 
                              << oor_ex.what() << std::endl;
                    break; 
                }
            }
        }
    }
    
    if (this->T.tokens.size() != embeddings_row_idx) {
        std::cerr << "Critical Inconsistency: T.tokens.size() (" << this->T.tokens.size()
                  << ") does not match the number of successfully processed embedding rows (" << embeddings_row_idx
                  << "). This indicates a bug in the loading logic." << std::endl;
    }
    
    if (embeddings_row_idx != this->T.vocabsize) {
        std::cout << "Note: Effective vocabulary size after loading is " << embeddings_row_idx
                  << ", while pre-allocated T.vocabsize was " << this->T.vocabsize << "." << std::endl;
    }

    std::cout << "Finished reading embeddings from " << path2file << ". Loaded " << this->T.tokens.size() << " tokens." << std::endl;
    
    if (!this->T.tokens.empty() && this->T.embeddings.row > 0 && this->T.embeddings.col > 0 && embeddings_row_idx > 0) {
        int values_to_print = 0;
        if (this->T.d > 0) { // Ensure T.d is positive before std::min, though earlier checks should guarantee this
            values_to_print = std::min<int>(5, this->T.d);
        }

        std::cout << "First token: '" << this->T.tokens[0] << "' with embedding (first " << values_to_print << " values): ";
        for(int k=0; k < values_to_print; ++k) {
            try {
                std::cout << this->T.embeddings(0,k) << " ";
            } 
            catch (const std::out_of_range&) { 
                std::cout << "[OOB] "; 
            }
        }
        std::cout << (this->T.d > 5 ? "..." : "") << std::endl;
    }
}


/**
 * @brief Function to calculate and apply absolute sinusoidal positional embedding.
 * This function takes an original token embedding, calculates its sinusoidal positional 
 * encoding based on the given 'position', and adds it to create a new embedding vector.
 * @param[in] originalEmbedding original embedding for the token
 * @param[out] newEmbedding new embedding obtained after adding positional embedding
 * @param[in] position The absolute position of the token in the sequence (0-indexed)
 */
void model::positionalEmbedding(const std::vector<float>& originalEmbedding, std::vector<float>& newEmbedding,
    int position) 
{
    // The dimension of the embedding (d_model) is inferred from the size of the original embedding.
    size_t d_model = originalEmbedding.size();
    // Ensure newEmbedding has the correct size.
    if (newEmbedding.size() != d_model) {
        newEmbedding.resize(d_model);
    }

    // Iterate through each dimension of the embedding to calculate the positional encoding.
    for (size_t i = 0; i < d_model; ++i) {
        // The frequency term for the sine/cosine wave depends on the dimension index.
        // The formula uses 10000^(2*k/d_model) where 'k' is floor(dimension_index / 2).
        double div_term_exponent = static_cast<double>(i / 2) * 2.0 / d_model;
        double div_term = std::pow(10000.0, div_term_exponent);

        // Apply sine for even dimensions and cosine for odd dimensions.
        if (i % 2 == 0) { // Even dimension indices (0, 2, 4, ...)
            newEmbedding[i] = originalEmbedding[i] + std::sin(position / div_term);
        }
        else { // Odd dimension indices (1, 3, 5, ...)
            newEmbedding[i] = originalEmbedding[i] + std::cos(position / div_term);
        }
    }
}
