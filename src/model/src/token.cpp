
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
 * @brief Tokenize a string into words, separate numbers (digit by digit), and punctuations.
 * @param str The string to tokenize.
 * @param tokens The vector to store the tokens.
 * @param sortIt when embedding needed (1) sort else don't
 */
void tokenize_with_numbers(const std::string& str, std::vector<std::string>& tokens, bool& sortIt)
{
    tokens.clear(); // Ensure the output vector starts fresh
    std::set<std::string> encountered_tokens; // To keep track of tokens already added

    // Delimiters include common punctuation and symbols (excluding space)
    static const std::string delimiters = ".,!?;:-~_+=/@#$&*`%^\\|\"\'(){}[]<>";

    // Helper lambda to add a token to the 'tokens' vector if it's new
    auto add_unique_token = [&](const std::string& token_val) {
        if (!token_val.empty() && encountered_tokens.find(token_val) == encountered_tokens.end()) {
            tokens.push_back(token_val);
            encountered_tokens.insert(token_val);
        }
    };

    std::string current_part;
    
    // Process each character in the string
    for (char c : str) {
        if (c == ' ') {
            // Space separates tokens but is not included as a token
            add_unique_token(current_part);
            current_part.clear();
        } 
        else if (is_digit(c)) {
            // Add accumulated word part if any
            add_unique_token(current_part);
            current_part.clear();
            // Add digit as a separate token
            add_unique_token(std::string(1, c));
        } 
        else if (delimiters.find(c) != std::string::npos) {
            // Add accumulated word part if any
            add_unique_token(current_part);
            current_part.clear();
            // Add delimiter as a separate token
            add_unique_token(std::string(1, c));
        } 
        else {
            // Accumulate character for a word/token
            current_part.push_back(c);
        }
    }
    
    // Add any remaining part
    add_unique_token(current_part);
    // Sort the tokens lexicographically
    if(sortIt == 1)
        std::sort(tokens.begin(), tokens.end());
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
                    // current_sub_sentence += "</s>";
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
                // current_sub_sentence += "</s>";
            }
            subSentences.push_back(current_sub_sentence);
        }
    }

    // If, after collecting all valid sub-sentences, there's an odd number
    // and more than one, concatenate the last two.
    // This aims to create pairs for sequence1/sequence2 processing.
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
 * @brief Make embedding for the model. This function reads a file, tokenizes its 
 *      content, and initializes or updates the model's vocabulary and embeddings.
 * @param path2file Path to the file containing tokens.
 */
void model::makeEmbedding(std::string &path2file)
{
    bool sortIt = 1;
    const std::string special_token = "</s>";

    std::ifstream file(path2file);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file: " << path2file << std::endl;
        throw std::runtime_error("makeEmbedding: Could not open file: " + path2file);
    }

    std::string line_content;
    std::vector<std::string> all_tokens_from_file_temp;
    while (std::getline(file, line_content)) {
        if(line_content.empty()) {
            continue;
        }
        std::vector<std::string> line_tokens_temp;
        tokenize_with_numbers(line_content, line_tokens_temp, sortIt);
        all_tokens_from_file_temp.insert(all_tokens_from_file_temp.end(), line_tokens_temp.begin(), line_tokens_temp.end());
    }
    file.close();

    if (all_tokens_from_file_temp.empty()) {
        std::cout << "Info: No tokens found in the input file: " << path2file << std::endl;
        return; 
    }

    // Random number generation (seeded once per program execution for consistency within a run)
    static std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist_uniform(-2.5f, 5.0f);

    std::set<std::string> unique_tokens_from_current_file(all_tokens_from_file_temp.begin(), all_tokens_from_file_temp.end());

    if (T.d <= 0) {
        throw std::runtime_error("makeEmbedding: Transformer embedding dimension (T.d) must be positive.");
    }

    if (T.tokens.empty()) {
        std::cout << "Info: Initializing vocabulary from: " << path2file << std::endl;
        
        // Ensure the special token is included during initialization
        unique_tokens_from_current_file.insert(special_token);

        std::vector<std::string> sorted_unique_tokens(unique_tokens_from_current_file.begin(), unique_tokens_from_current_file.end());
        std::sort(sorted_unique_tokens.begin(), sorted_unique_tokens.end());

        T.tokens = sorted_unique_tokens;
        T.vocabsize = T.tokens.size();

        if (T.vocabsize == 0) {
            std::cerr << "Warning: No unique tokens derived from file " << path2file << ", though file was not empty of tokens." << std::endl;
            return;
        }

        T.embeddings = mat(T.vocabsize, T.d); // T.vocabsize and T.d are int

        for (int i = 0; i < T.vocabsize; ++i) {
            for (int j = 0; j < T.d; ++j) {
                T.embeddings(i, j) = dist_uniform(rng);
            }
        }
        std::cout << "Info: Initialized  T.tokens with " << T.vocabsize << " unique tokens and T.embeddings." << std::endl;
    }
    else {
        std::cout << "Info: Updating existing vocabulary using: " << path2file << std::endl;

        // Store current tokens and their embeddings
        std::vector<std::pair<std::string, std::vector<float>>> combined_token_embeddings;
        combined_token_embeddings.reserve(T.vocabsize + unique_tokens_from_current_file.size());

        for (int i = 0; i < T.vocabsize; ++i) {
            std::vector<float> current_embedding_row(T.d);
            for (int j = 0; j < T.d; ++j) {
                try {
                    current_embedding_row[j] = T.embeddings(i, j);
                } 
                catch (const std::exception& e) {
                    std::cerr << "Error accessing T.embeddings(" << i << "," << j << ") for existing token '" << T.tokens[i] << "': " << e.what() << std::endl;
                    current_embedding_row[j] = 0.0f; // Default on error
                }
            }
            combined_token_embeddings.push_back({T.tokens[i], current_embedding_row});
        }

        // Identify genuinely new tokens to add
        // These are tokens in unique_tokens_from_current_file but not in  T.tokens yet
        std::set<std::string> existing_vocab_set_for_check(T.tokens.begin(), T.tokens.end());
        std::vector<std::string> new_tokens_from_file;
        for (const auto& token_from_file : unique_tokens_from_current_file) {
            if (existing_vocab_set_for_check.find(token_from_file) == existing_vocab_set_for_check.end()) {
                new_tokens_from_file.push_back(token_from_file);
            }
        }

        bool new_file_tokens_were_added = !new_tokens_from_file.empty();

        if (new_file_tokens_were_added) {
            std::cout << "Info: Found " << new_tokens_from_file.size() << " new unique tokens from file to add." << std::endl;
            for (const auto& new_token : new_tokens_from_file) {
                std::vector<float> random_embedding_row(T.d);
                for (int j = 0; j < T.d; ++j) {
                    random_embedding_row[j] = dist_uniform(rng);
                }
                combined_token_embeddings.push_back({new_token, random_embedding_row});
            }
        }

        // Ensure special_token "</s>" is present in the combined list
        bool special_token_found_in_combined = false;
        for (const auto& pair : combined_token_embeddings) {
            if (pair.first == special_token) {
                special_token_found_in_combined = true;
                break;
            }
        }

        bool special_token_was_explicitly_added = false;
        if (!special_token_found_in_combined) {
            std::cout << "Info: Special token '" << special_token << "' not found. Adding it to vocabulary." << std::endl;
            std::vector<float> random_embedding_row(T.d);
            for (int j = 0; j < T.d; ++j) {
                random_embedding_row[j] = dist_uniform(rng);
            }
            combined_token_embeddings.push_back({special_token, random_embedding_row});
            special_token_was_explicitly_added = true;
        }

        // If no new tokens from file and special token didn't need explicit adding, vocabulary is unchanged.
        if (!new_file_tokens_were_added && !special_token_was_explicitly_added) {
            std::cout << "Info: No new unique tokens from " << path2file << " and special token '" << special_token << "' already exists or was part of file tokens. Vocabulary structure unchanged." << std::endl;
            return;
        }

        // Sort the combined list by token string if changes were made
        std::sort(combined_token_embeddings.begin(), combined_token_embeddings.end(),
                  [](const auto& a, const auto& b) {
                      return a.first < b.first;
                  });

        // Update  T.tokens, T.vocabsize, and T.embeddings
        T.tokens.clear();
        T.tokens.reserve(combined_token_embeddings.size());
        T.vocabsize = combined_token_embeddings.size();
        T.embeddings = mat(T.vocabsize, T.d); // Re-initialize mat

        for (int i = 0; i < T.vocabsize; ++i) {
            T.tokens.push_back(combined_token_embeddings[i].first);
            for (int j = 0; j < T.d; ++j) {
                try {
                    T.embeddings(i, j) = combined_token_embeddings[i].second[j];
                }
                catch (const std::exception& e) {
                    std::cerr << "Error writing combined embedding to T.embeddings(" << i << "," << j << ") for token '" << T.tokens[i] << "': " << e.what() << std::endl;
                }
            }
        }
        std::cout << "Info: Updated and sorted  T.tokens to " << T.vocabsize << " unique tokens and T.embeddings." << std::endl;
    }
    std::cout << "Embeddings Made" << std::endl;
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
    if (T.vocabsize <= 0 || T.d <= 0) {
        file.close();
        throw std::runtime_error("Cannot load embeddings: Transformer vocabsize (" + std::to_string(T.vocabsize) +
                                 ") or embedding dimension (" + std::to_string(T.d) +
                                 ") is zero or negative.");
    }

    // T.embeddings should have been initialized by the transformer's constructor to these dimensions.
    if (T.embeddings.row != T.vocabsize || T.embeddings.col != T.d) {
        file.close();
        std::string error_msg = "T.embeddings dimensions mismatch. ";
        error_msg += "Expected (from T.vocabsize, T.d): " + std::to_string(T.vocabsize) + "x" + std::to_string(T.d) + ". ";
        error_msg += "Actual T.embeddings: " + std::to_string(T.embeddings.row) + "x" + std::to_string(T.embeddings.col) + ".";
        throw std::runtime_error(error_msg);
    }

    if (!T.embeddings.mapped_data) {
        // This should ideally not happen if row/col are > 0 due to mat constructor logic.
        file.close();
        throw std::runtime_error("T.embeddings is not mapped (mapped_data is null), "
                                 "but dimensions are " + std::to_string(T.embeddings.row) + "x" + std::to_string(T.embeddings.col) + ".");
    }

    T.tokens.clear();
    T.tokens.reserve(T.vocabsize); // Pre-allocate memory

    std::string line;
    int csv_line_number = 0;
    int embeddings_row_idx = 0; // Index for  T.tokens and T.embeddings

    std::cout << "Reading embeddings from " << path2file << " for vocab size " << T.vocabsize << " and dimension " << T.d << "..." << std::endl;

    while (std::getline(file, line)) {
        csv_line_number++;
        if (embeddings_row_idx >= T.vocabsize) {
            std::cerr << "Warning: CSV file '" << path2file << "' (line " << csv_line_number
                      << ") has more lines than expected vocab size (" << T.vocabsize
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

        T.tokens.push_back(token_str);
        bool critical_error_in_line = false;

        // Loop to read embedding values for the current token
        for (int j = 0; j < T.d; ++j) {
            if (!std::getline(ss, cell, ',')) {
                std::cerr << "Error: CSV line " << csv_line_number << " (token: '" << token_str << "') in '" << path2file
                          << "' does not have enough embedding values. Expected " << T.d
                          << ", found " << j << ". Filling rest with 0.0f for this token." << std::endl;
                for (int k = j; k < T.d; ++k) {
                    try {
                        T.embeddings(embeddings_row_idx, k) = 0.0f;
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
                T.embeddings(embeddings_row_idx, j) = val_from_stof;
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
            if (!T.tokens.empty() && T.tokens.back() == token_str) {
                T.tokens.pop_back();
            }
            std::cerr << "Skipping CSV line " << csv_line_number << " due to critical error during embedding processing for token '" << token_str << "'." << std::endl;
            continue; // Continue to the next line in the CSV file
        }

        if (std::getline(ss, cell, ',')) {
            std::cerr << "Warning: CSV line " << csv_line_number << " (token: '" << token_str << "') in '" << path2file
                      << "' has more columns than expected embedding dimension (" << T.d
                      << "). Ignoring extra columns." << std::endl;
        }
        embeddings_row_idx++; // Successfully processed a row
    }

    file.close();

    if (embeddings_row_idx < T.vocabsize) {
        std::cerr << "Warning: Loaded " << embeddings_row_idx << " tokens/embeddings from CSV '" << path2file
                  << "', which is fewer than the expected vocab size (" << T.vocabsize
                  << "). The remaining " << (T.vocabsize - embeddings_row_idx)
                  << " entries in T.embeddings will be zeroed." << std::endl;
        for (int i = embeddings_row_idx; i < T.vocabsize; ++i) {
            for (int j = 0; j < T.d; ++j) {
                try {
                    T.embeddings(i, j) = 0.0f;
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
    
    if (T.tokens.size() != embeddings_row_idx) {
        std::cerr << "Critical Inconsistency:  T.tokens.size() (" << T.tokens.size()
                  << ") does not match the number of successfully processed embedding rows (" << embeddings_row_idx
                  << "). This indicates a bug in the loading logic." << std::endl;
    }
    
    if (embeddings_row_idx != T.vocabsize) {
        std::cout << "Note: Effective vocabulary size after loading is " << embeddings_row_idx
                  << ", while pre-allocated T.vocabsize was " << T.vocabsize << "." << std::endl;
    }

    std::cout << "Finished reading embeddings from " << path2file << ". Loaded " << T.tokens.size() << " tokens." << std::endl;
    
    if (!T.tokens.empty() && T.embeddings.row > 0 && T.embeddings.col > 0 && embeddings_row_idx > 0) {
        int values_to_print = 0;
        if (T.d > 0) { // Ensure T.d is positive before std::min, though earlier checks should guarantee this
            values_to_print = std::min<int>(5, T.d);
        }

        std::cout << "First token: '" << T.tokens[0] << "' with embedding (first " << values_to_print << " values): ";
        for(int k=0; k < values_to_print; ++k) {
            try {
                std::cout << T.embeddings(0,k) << " ";
            } 
            catch (const std::out_of_range&) { 
                std::cout << "[OOB] "; 
            }
        }
        std::cout << (T.d > 5 ? "..." : "") << std::endl;
    }
}
