#include <fstream>
#include <filesystem>
#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <algorithm>
#include <cctype>
#include "include/tokenise.hpp"

// count lines or rows in file
long long count_lines(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Warning: Could not open file to count lines: " << filename << std::endl;
        return -1;
    }
    long long count = 0;
    std::string line;
    while (std::getline(file, line)) {
        count++;
    }
    return count;
}

// Utility function to trim whitespace from both ends of a string
std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    size_t last = str.find_last_not_of(" \t\r\n");
    return str.substr(first, (last - first + 1));
}

// Utility function to remove outer quotes from a string AND unescape internal doubled quotes.
// This function is generally robust for reading CSV fields that were correctly quoted.
std::string removeQuotes(const std::string& str) {
    std::string temp_str = str;
    bool was_quoted = false;

    // First, check and remove outer quotes
    if (temp_str.length() >= 2 && ((temp_str.front() == '"' && temp_str.back() == '"') || (temp_str.front() == '\"' && temp_str.back() == '\"'))) {
        temp_str = temp_str.substr(1, temp_str.length() - 2);
        was_quoted = true;
    }
    // Handle single quotes for consistency, but not for CSV unescaping (CSV uses double quotes)
    else if (temp_str.length() >= 2 && temp_str.front() == '\'' && temp_str.back() == '\'') {
        temp_str = temp_str.substr(1, temp_str.length() - 2);
    }

    // If it was originally quoted, unescape internal double quotes ("" -> ")
    if (was_quoted) {
        std::string unescaped_result;
        unescaped_result.reserve(temp_str.length());
        for (size_t i = 0; i < temp_str.length(); ++i) {
            if (temp_str[i] == '"' && (i + 1 < temp_str.length() && temp_str[i+1] == '"')) {
                unescaped_result += '"'; // Add single quote
                i++; // Skip the next character (the second quote in "")
            } else {
                unescaped_result += temp_str[i];
            }
        }
        return unescaped_result;
    } else {
        // If not originally quoted, return as is (no unescaping needed)
        return temp_str;
    }
}

// Helper function to correctly escape and quote a string for CSV output
// This should be used for *any* field that might contain commas, newlines, or double quotes.
std::string escapeAndQuoteCsvField(const std::string& field) {
    // Check if quoting is necessary
    // Needs quotes if it contains:
    // 1. A comma (,)
    // 2. A double quote (")
    // 3. A newline character (\n or \r)
    // 4. An empty string (to distinguish from missing field if not last)
    // 5. Or if the string is just spaces (to preserve them)
    bool needs_quoting = false;
    if (field.empty() ||
        field.find(',') != std::string::npos ||
        field.find('"') != std::string::npos ||
        field.find('\n') != std::string::npos ||
        field.find('\r') != std::string::npos ||
        (field.find_first_not_of(" \t") == std::string::npos && !field.empty())) // Check for all whitespace string
    {
        needs_quoting = true;
    }

    std::string escaped_field;
    escaped_field.reserve(field.length() + 2); // Reserve space for potential quotes and escaped chars

    for (char c : field) {
        if (c == '"') {
            escaped_field += "\"\""; // Double existing double quotes
        } else {
            escaped_field += c;
        }
    }

    if (needs_quoting) {
        return "\"" + escaped_field + "\""; // Enclose in outer double quotes
    } else {
        return escaped_field; // No quoting needed
    }
}


// Function to check if a line looks like a CSV header
bool isHeaderLine(const std::string& line) {
    std::string trimmed = trim(line);
    std::transform(trimmed.begin(), trimmed.end(), trimmed.begin(), ::tolower);

    // Common header patterns - checks for typical CSV header combinations
    return (trimmed.find("token") != std::string::npos &&
            (trimmed.find("count") != std::string::npos ||
             trimmed.find("repetitions") != std::string::npos)) ||
           (trimmed.find("word") != std::string::npos &&
            trimmed.find("count") != std::string::npos) ||
           (trimmed.find("embedding") != std::string::npos) ||
           (trimmed == "word,count" || trimmed == "token,count" ||
            trimmed == "token,repetitions");
}

// Helper function to read a single CSV field, handling quotes and escaped quotes
std::string readCsvField(std::stringstream& ss) {
    std::string field;
    char c;

    // Consume leading whitespace before the field starts
    while (ss.peek() != EOF && std::isspace(static_cast<unsigned char>(ss.peek()))) {
        ss.get();
    }

    if (ss.peek() == '"') {
        ss.get(); // Consume opening quote
        bool in_quotes = true;
        while (ss.get(c) && in_quotes) { // Loop while we are inside the quoted field
            if (c == '"') {
                if (ss.peek() == '"') { // Escaped double quote ("")
                    field += '"';
                    ss.get(); // Consume the second quote
                } else { // Closing quote
                    in_quotes = false; // Exit loop after consuming closing quote
                }
            } else {
                field += c;
            }
        }
        // After finding the closing quote, consume any trailing whitespace then the comma
        while (ss.peek() != EOF && std::isspace(static_cast<unsigned char>(ss.peek()))) {
            ss.get();
        }
        if (ss.peek() == ',') {
            ss.get(); // Consume the comma
        }
        // If it was a quoted field and no comma followed, it means it's the last field,
        // or a malformed line. No need to consume a comma that's not there.
    } else {
        // Not a quoted field, read until comma or end of line
        std::getline(ss, field, ',');
    }
    return field; // `removeQuotes` will be called on this string later
}

// Function to read a single column CSV into a vector of strings
std::vector<std::string> readSingleColumnCsv(const std::string& filename) {
    std::vector<std::string> columnData;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        std::cerr << "Check if the file exists and has proper read permissions." << std::endl;
        return columnData;
    }

    std::string line;
    int lineNumber = 0;

    while (std::getline(file, line)) {
        lineNumber++;

        if (line.empty()) {
            continue;
        }

        line = trim(line); // Trim whitespace from the entire line

        if (line.empty()) {
            continue;
        }

        std::stringstream ss(line);
        std::string raw_field = readCsvField(ss); // Use the new helper for robustness
        columnData.push_back(removeQuotes(trim(raw_field))); // Process the extracted field
    }

    file.close();

    if (columnData.empty()) {
        std::cerr << "Warning: No data found in file " << filename << std::endl;
    } else {
        std::cout << "Successfully read " << columnData.size()
                  << " entries from file " << filename << std::endl;
    }

    return columnData;
}

// Function to read a specific column (0-based index) from a multi-column CSV
std::vector<std::string> readSpecificColumnFromCsv(const std::string& filename, int targetColumnIndex) {
    std::vector<std::string> columnData;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        std::cerr << "Check if the file exists and has proper read permissions." << std::endl;
        return columnData;
    }

    if (targetColumnIndex < 0) {
        std::cerr << "Error: Invalid column index " << targetColumnIndex
                  << ". Column index must be non-negative." << std::endl;
        file.close();
        return columnData;
    }

    std::string line;
    int lineNumber = 0;
    bool headerSkipped = false;

    while (std::getline(file, line)) {
        lineNumber++;

        if (line.empty()) {
            continue;
        }

        line = trim(line);

        if (line.empty()) {
            continue;
        }

        // Removed header skip logic here, as per previous discussion it was commented out in user's code.
        // If your files legitimately have headers, uncomment this AND ensure it correctly identifies them.
        /*if (!headerSkipped && isHeaderLine(line)) {
            headerSkipped = true;
            std::cout << "Skipping header line: " << line << std::endl;
            continue;
        }*/

        std::stringstream ss(line);
        std::string segment_raw;
        int currentColumnIndex = 0;
        bool columnFoundForThisLine = false;

        while (true) { // Loop to read fields
            segment_raw = readCsvField(ss);

            if (currentColumnIndex == targetColumnIndex) {
                columnData.push_back(removeQuotes(trim(segment_raw)));
                columnFoundForThisLine = true;
                break; // Found the target column, move to next line
            }
            currentColumnIndex++;

            if (ss.eof() && segment_raw.empty() && currentColumnIndex <= targetColumnIndex) {
                break;
            }
            if (ss.eof() && !segment_raw.empty() && currentColumnIndex <= targetColumnIndex) {
                break;
            }
            // If we are here, it means readCsvField successfully consumed a field,
            // and there was a comma, but it was not the target.
            // Check if there are more characters in the stream after consuming the field+comma.
            if (ss.peek() == EOF && !ss.good()) { // Check if stream failed or reached true EOF
                break;
            }
        }

        if (!columnFoundForThisLine) {
            if (lineNumber <= 3) {
                std::cerr << "Warning: Column " << targetColumnIndex
                          << " not found in line " << lineNumber << " of file " << filename
                          << " (line has " << currentColumnIndex << " columns)" << std::endl;
            }
            columnData.push_back(""); // Add empty string to maintain row count
        }
    }

    file.close();

    if (columnData.empty()) {
        std::cerr << "Warning: No data found in column " << targetColumnIndex
                  << " of file " << filename << std::endl;
    } else {
        std::cout << "Successfully read " << columnData.size()
                  << " entries from column " << targetColumnIndex
                  << " of file " << filename << std::endl;
    }

    return columnData;
}

// Function to read an entire CSV file into a 2D vector of strings (row by row)
std::vector<std::vector<std::string>> readCsvTo2DVector(const std::string& filename) {
    std::vector<std::vector<std::string>> csvData;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        std::cerr << "Check if the file exists and has proper read permissions." << std::endl;
        return csvData;
    }

    std::string line;
    int lineNumber = 0;

    while (std::getline(file, line)) {
        lineNumber++;

        if (line.empty()) {
            continue;
        }

        line = trim(line);

        if (line.empty()) {
            continue;
        }

        std::vector<std::string> row;
        std::stringstream ss(line);

        while (true) {
            std::string raw_field = readCsvField(ss);
            row.push_back(removeQuotes(trim(raw_field)));

            // Check if there are more fields to read
            if (ss.eof() && ss.peek() == EOF) {
                break; // No more characters and stream is at end
            }
        }
        if (!row.empty()) {
            csvData.push_back(row);
        }
    }

    file.close();

    if (csvData.empty()) {
        std::cerr << "Warning: No data found in file " << filename << std::endl;
    } else {
        std::cout << "Successfully read " << csvData.size()
                  << " rows from file " << filename << std::endl;
    }

    return csvData;
}


// Function to read a CSV with "word,count" format into an unordered_map<string, int>
std::unordered_map<std::string, int> readUnorderedMap(const std::string& filename) {
    std::unordered_map<std::string, int> corpusWordCount;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        std::cerr << "Check if the file exists and has proper read permissions." << std::endl;
        return corpusWordCount;
    }

    std::string line;
    int lineNumber = 0;
    int successfullyParsed = 0;
    bool headerSkipped = false;

    while (std::getline(file, line)) {
        lineNumber++;

        if (line.empty()) {
            continue;
        }

        line = trim(line);
        if (line.empty()) {
            continue;
        }

        // Removed header skip logic here, as per previous discussion it was commented out in user's code.
        // If your files legitimately have headers, uncomment this AND ensure it correctly identifies them.
        /*if (!headerSkipped && isHeaderLine(line)) {
            headerSkipped = true;
            std::cout << "Skipping header line: " << line << std::endl;
            continue;
        }*/

        std::stringstream ss(line);
        
        // Read the token field using the robust helper
        std::string raw_token_field = readCsvField(ss);
        std::string token = removeQuotes(trim(raw_token_field));

        // Read the count field using the robust helper
        std::string raw_count_field = readCsvField(ss);
        std::string count_str_cleaned = removeQuotes(trim(raw_count_field));

        std::string final_numeric_count_str;
        bool first_char_of_count = true;
        for (char c : count_str_cleaned) {
            if (std::isdigit(static_cast<unsigned char>(c))) {
                final_numeric_count_str += c;
            } else if (c == '-' && first_char_of_count) { // Allow only one leading minus sign
                final_numeric_count_str += c;
            }
            first_char_of_count = false;
        }
        // If after cleaning, the string is empty, AND the original was not just "0", it's invalid.
        if (final_numeric_count_str.empty() && !count_str_cleaned.empty() && count_str_cleaned != "0") {
             std::cerr << "Warning: Count string became empty after cleaning non-numeric characters in line " << lineNumber << ". Original raw: '" << raw_count_field << "'. Cleaned: '" << count_str_cleaned << "'" << std::endl;
             continue;
        }

        if (token.empty()) {
            // This case should ideally result in a valid empty string token `""`.
            // If a genuinely empty token is not expected in your vocabulary,
            // then consider adding a specific check for it here, e.g., if (token != "\"\"") and token.empty().
            // For now, allowing empty string as a key.
            if (lineNumber <= 15) {
                std::cerr << "Warning: Token resolved to empty string in line " << lineNumber
                          << " of file " << filename << ". Raw field: '" << raw_token_field << "'. Processed token: '" << token << "'" << std::endl;
            }
        }

        // Attempt to convert to integer
        try {
            int count = std::stoi(final_numeric_count_str);
            corpusWordCount[token] = count;
            successfullyParsed++;
        } catch (const std::invalid_argument& e) {
            std::cerr << "Warning: Invalid integer format for count '"
                      << raw_count_field << "' (cleaned to '" << final_numeric_count_str << "') in line " << lineNumber
                      << " of file " << filename << ": " << e.what() << std::endl;
        } catch (const std::out_of_range& e) {
            std::cerr << "Warning: Count out of range '"
                      << raw_count_field << "' (cleaned to '" << final_numeric_count_str << "') in line " << lineNumber
                      << " of file " << filename << ": " << e.what() << std::endl;
        }
    }

    file.close();

    if (corpusWordCount.empty()) {
        std::cerr << "Warning: No valid word-count pairs found in file " << filename << std::endl;
    } else {
        std::cout << "Successfully read " << successfullyParsed
                  << " word-count pairs from file " << filename << std::endl;
    }

    return corpusWordCount;
}


// Function to read a CSV with "word,float1,float2,..." format into an unordered_map<string, vector<float>>
std::unordered_map<std::string, std::vector<float>> readMappedEmbeddings(const std::string& filename) {
    std::unordered_map<std::string, std::vector<float>> mappedEmbeddings;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        std::cerr << "Check if the file exists and has proper read permissions." << std::endl;
        return mappedEmbeddings;
    }

    std::string line;
    int lineNumber = 0;
    int successfullyParsed = 0;
    int expectedEmbeddingSize = -1; // To track consistency
    // bool headerSkipped = false; // REMOVED: No header for this file per user feedback
    
    while (std::getline(file, line)) {
        lineNumber++;

        if (line.empty()) {
            continue;
        }

        line = trim(line);

        if (line.empty()) {
            continue;
        }

        std::stringstream ss(line);
        std::string raw_token_field = readCsvField(ss);
        std::string word_str = removeQuotes(trim(raw_token_field));


        if (word_str.empty()) {
            // An empty string as a token for embedding mapping implies a problem
            // unless a specific empty string token `""` is expected to have an embedding.
            // If your CSV outputs `""` for some tokens, and they should have embeddings,
            // then `word_str` should ideally become `"`, not empty.
            if (lineNumber <= 15) {
                std::cerr << "Warning: Empty word in line " << lineNumber
                          << " of file " << filename << ". Raw field: '" << raw_token_field << "'. Processed token: '" << word_str << "'" << std::endl;
            }
            continue; // Skip lines with empty tokens for embeddings, as they can't be mapped meaningfully
        }
        
        std::vector<float> embeddings_vector;
        bool parseError = false;

        // Extract the rest of the floats using the robust helper
        while (!ss.eof() && ss.peek() != EOF) { // Continue as long as there's something to read
            std::string raw_segment = readCsvField(ss);
            std::string segment = removeQuotes(trim(raw_segment));

            if (segment.empty()) {
                // An empty segment for a float value is likely an error.
                // stof will throw std::invalid_argument for "".
                // Allow the try-catch to handle it.
            }
            
            // Remove any remaining quotes that might be embedded for float conversion.
            segment.erase(std::remove(segment.begin(), segment.end(), '"'), segment.end());
            segment.erase(std::remove(segment.begin(), segment.end(), '\''), segment.end());

            try {
                float value = std::stof(segment);
                embeddings_vector.push_back(value);
            } catch (const std::invalid_argument& e) {
                if (lineNumber <= 5) {
                    std::cerr << "Warning: Invalid float format '" << raw_segment
                              << "' (cleaned to '" << segment << "') for word '" << word_str << "' in line " << lineNumber
                              << " of file " << filename << ": " << e.what() << std::endl;
                }
                parseError = true;
                break;
            } catch (const std::out_of_range& e) {
                if (lineNumber <= 5) {
                    std::cerr << "Warning: Float out of range '" << raw_segment
                              << "' (cleaned to '" << segment << "') for word '" << word_str << "' in line " << lineNumber
                              << " of file " << filename << ": " << e.what() << std::endl;
                }
                parseError = true;
                break;
            }
        }
        
        if (parseError) {
            continue;
        }

        // Only add to map if we got at least some embeddings or if empty embeddings are valid.
        mappedEmbeddings[word_str] = embeddings_vector;
        successfullyParsed++;
    }

    file.close();

    if (mappedEmbeddings.empty()) {
        std::cerr << "Warning: No valid word-embedding pairs found in file " << filename << std::endl;
    } else {
        std::cout << "Successfully read " << successfullyParsed
                  << " word-embedding pairs from file " << filename;
        if (expectedEmbeddingSize != -1) {
            std::cout << " (embedding dimension: " << expectedEmbeddingSize << ")";
        }
        std::cout << std::endl;
    }

    return mappedEmbeddings;
}


// Your tokeniser::readFromFiles method remains largely the same,
// as it calls the updated readUnorderedMap and readMappedEmbeddings functions.
void tokeniser::readFromFiles(const std::string& path2ClassDataFolder) {
    // 1. Load the token counts (which contain the vocabulary keys)
    // Assuming statOfTokens.csv is where your trained token counts are saved by calculateTokenStatsFromCounts.
    std::string token_stats_file = path2ClassDataFolder + "/_final_token_stats.csv";

    // Add robust file existence check
    if (!std::filesystem::exists(token_stats_file)) {
        std::cerr << "Error: Token statistics file not found at " << token_stats_file << std::endl;
        throw std::runtime_error("Required token statistics file missing. Ensure training created '_final_token_stats.csv' in the specified path.");
    }
    this->statOfTokens = readUnorderedMap(token_stats_file);

    // 2. Load the mapped embeddings
    std::string embeddings_file = path2ClassDataFolder + "/_final_embeddings.csv";
    if (!std::filesystem::exists(embeddings_file)) {
        std::cerr << "Error: Embeddings file not found at " << embeddings_file << std::endl;
        throw std::runtime_error("Required embeddings file missing. Ensure training created '_final_embeddings.csv' in the specified path.");
    }
    this->mappedEmbeddings = readMappedEmbeddings(embeddings_file);

    // --- CRITICAL STEP: Populate 'this->tokens' from the loaded vocabulary ---
    this->tokens.clear(); // Ensure it's empty before populating
    for (const auto& pair : this->statOfTokens) {
        this->tokens.push_back(pair.first); // Add all unique tokens (including BPE merges) to the vector
    }

    // Optional: Manually add special tokens if they are not guaranteed to be in statOfTokens
    // by the training process, but are expected for tokenization.
    // Example: If "</w>" or "<@#0>" might be missing from statOfTokens.
    // if (std::find(this->tokens.begin(), this->tokens.end(), "<@#0>") == this->tokens.end()) {
    //     this->tokens.push_back("<@#0>");
    // }
    // if (std::find(this->tokens.begin(), this->tokens.end(), "</w>") == this->tokens.end()) {
    //     this->tokens.push_back("</w>");
    // }

    // 3. IMPORTANT: Sort the tokens by length in descending order.
    // This is crucial for the greedy longest-match logic in splitWord.
    std::sort(this->tokens.begin(), this->tokens.end(),
        [](const std::string& a, const std::string& b) {
            // Sort by length descending. For ties, sort alphabetically for consistent behavior.
            if (a.length() != b.length()) {
                return a.length() > b.length();
            }
            return a < b; // Tie-breaking: alphabetical order
        }
    );

    // Update vocabulary size and embedding dimension based on loaded data
    this->vocSize = this->tokens.size();
    if (!this->mappedEmbeddings.empty()) {
        // Set 'd' based on the dimension of the first loaded embedding
        this->d = this->mappedEmbeddings.begin()->second.size();
    }
    else {
        this->d = 0;
        std::cerr << "Warning: No embeddings loaded, embedding dimension (d) set to 0. This might affect subsequent operations." << std::endl;
    }

    // Populate 'embeddings' and 'deEmbeddings' vectors if they are used directly elsewhere
    // and need to be ordered consistently with 'this->tokens'.
    this->embeddings.clear();
    this->embeddings.reserve(this->tokens.size()); // Pre-allocate memory
    for (const auto& token_str : this->tokens) {
        auto it = this->mappedEmbeddings.find(token_str);
        if (it != this->mappedEmbeddings.end()) {
            this->embeddings.push_back(it->second);
        } else {
            // This is the warning that's being triggered.
            std::cerr << "Warning: Token '" << token_str << "' present in vocabulary but missing embedding in '_final_embeddings.csv'. Adding zero vector." << std::endl;
            this->embeddings.push_back(std::vector<float>(this->d, 0.0f)); // Provide a zero vector as a fallback
        }
    }

    // Similarly for deEmbeddings if applicable
    // this->deEmbeddings.clear();
    // this->deEmbeddings.reserve(this->tokens.size());
    // for (const auto& embedding_vec : this->embeddings) {
    //     this->deEmbeddings.push_back(vectorInverse(embedding_vec));
    // }

    std::cout << "Tokenizer initialized successfully:" << std::endl;
    std::cout << "  - Tokens loaded: " << this->tokens.size() << std::endl;
    std::cout << "  - Vocabulary size: " << this->vocSize << std::endl;
    std::cout << "  - Embedding dimension: " << this->d << std::endl;
}