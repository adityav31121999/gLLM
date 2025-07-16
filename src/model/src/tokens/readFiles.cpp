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
// Modified to return the raw field as read, without trimming or quote removal.
// This allows the caller to decide how to process the field.
std::string readCsvField(std::stringstream& ss) {
    std::string field;
    char c;

    // Consume leading whitespace before the field starts (if any)
    while (ss.peek() != EOF && std::isspace(static_cast<unsigned char>(ss.peek()))) {
        ss.get();
    }

    if (ss.peek() == '"') {
        ss.get(); // Consume opening quote
        bool in_quotes = true;
        while (ss.get(c)) { // Loop until the end of the quoted field or stream
            if (c == '"') {
                if (ss.peek() == '"') { // Escaped double quote ("")
                    field += '"';
                    ss.get(); // Consume the second quote
                } else { // Closing quote
                    in_quotes = false;
                    break; // Exit the loop for this field
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
            ss.get(); // Consume the comma delimiter
        }
        // If it was a quoted field and no comma followed, it means it's the last field,
        // or a malformed line. No need to consume a comma that's not there.
    } else {
        // Not a quoted field, read until comma or end of line
        std::getline(ss, field, ',');
    }
    return field;
}


// Function to read an entire CSV file into a 2D vector of floats
std::vector<std::vector<float>> readCsvTo2DVector(const std::string& filename) {
    std::vector<std::vector<float>> csvData;
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
            continue; // Skip truly empty lines
        }
        // Apply global trim to the entire line
        line = trim(line);
        if (line.empty()) {
            continue; // Skip lines that become empty after trimming
        }

        std::vector<float> row;
        std::stringstream ss(line);

        // Loop to read fields using the robust readCsvField
        // Check for success of readCsvField and if there are actual characters processed
        bool has_more_fields_on_line = true;
        while (has_more_fields_on_line) {
            std::string raw_field = readCsvField(ss);
            // After reading the field, if ss is at EOF and the field is empty, it means we hit end of line (possibly after a trailing comma).
            // This is a subtle point: readCsvField will consume a trailing comma. If no chars are left after that, it's EOF.
            // If the last actual field was empty (e.g., "1,2,"), raw_field for the last empty one will be empty.
            if (raw_field.empty() && ss.eof()) {
                has_more_fields_on_line = false; // No more actual fields.
            }

            // Apply trim and removeQuotes to the raw field string before conversion
            std::string cleaned_field_str = removeQuotes(trim(raw_field));

            float value = 0.0f;
            // Attempt to convert the string to float
            try {
                // If the cleaned_field_str is empty, stof will throw invalid_argument, which is desired.
                value = std::stof(cleaned_field_str);
            } catch (const std::invalid_argument& e) {
                // Conversion failed (e.g., empty string, "abc" to float)
                // This means the field was empty or non-numeric. Treat as 0.0f.
                if (!cleaned_field_str.empty()) { // Only warn if it wasn't just an empty field
                    std::cerr << "Warning: Failed to convert field '" << cleaned_field_str
                              << "' to float at line " << lineNumber
                              << " in file " << filename << ". Defaulting to 0.0f." << std::endl;
                }
                value = 0.0f;
            } catch (const std::out_of_range& e) {
                std::cerr << "Warning: Float out of range '" << cleaned_field_str
                          << "' at line " << lineNumber
                          << " in file " << filename << ". Defaulting to 0.0f." << std::endl;
                value = 0.0f;
            }
            row.push_back(value);

            // Check if there are more fields to read in the current line.
            // If ss.peek() returns EOF or fails, or if it's not a comma, we are at the end of fields for this line.
            if (ss.peek() == EOF && !ss.good()) {
                has_more_fields_on_line = false;
            }
        } // End of while(has_more_fields_on_line)

        // Add row only if it's not empty (e.g., not just an empty line that resulted in no fields)
        if (!row.empty() || line.find(',') != std::string::npos) { // Add row even if empty but it indicates fields (e.g. "a,,c")
            csvData.push_back(row);
        }
    }

    file.close();
    if (csvData.empty()) {
        std::cerr << "Warning: No data found in file " << filename << std::endl;
    }
    else {
        std::cout << "Successfully read " << csvData.size()
                  << " rows from file " << filename << std::endl;
    }

    return csvData;
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
    }
    else {
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
        std::string segment_raw;
        int currentColumnIndex = 0;
        bool columnFoundForThisLine = false;

        bool has_more_fields_on_line = true;
        while (has_more_fields_on_line) {
            segment_raw = readCsvField(ss);
            
            if (segment_raw.empty() && ss.eof()) { // Check for empty field at end of line
                 has_more_fields_on_line = false;
            }

            if (currentColumnIndex == targetColumnIndex) {
                columnData.push_back(removeQuotes(trim(segment_raw)));
                columnFoundForThisLine = true;
                break; // Found the target column, move to next line
            }
            currentColumnIndex++;

            if (ss.peek() == EOF && !ss.good()) { // Check if stream failed or reached true EOF
                has_more_fields_on_line = false;
            }
        }

        if (!columnFoundForThisLine) {
            // Only warn if the line genuinely had fewer columns than expected,
            // not if it was just a malformed last line.
            if (currentColumnIndex <= targetColumnIndex) { // Check if we even reached the target column index
                if (lineNumber <= 10) { // Limit warnings for brevity
                    std::cerr << "Warning: Column " << targetColumnIndex
                              << " not found in line " << lineNumber << " of file " << filename
                              << " (line has only " << currentColumnIndex << " columns)." << std::endl;
                }
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
    bool headerSkipped = false; // Keep this for potential future use or if some files truly have headers

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

        // Attempt to convert to integer
        try {
            // std::stoi is quite robust for numeric strings, including leading/trailing spaces
            // and signs. It stops at the first non-numeric character.
            int count = std::stoi(count_str_cleaned);
            corpusWordCount[token] = count; // Allow empty string as a key if it comes from data
            successfullyParsed++;
        } catch (const std::invalid_argument& e) {
            if (!count_str_cleaned.empty() && count_str_cleaned.find_first_not_of(" \t\r\n") != std::string::npos) { // Only warn if it wasn't just empty/whitespace
                std::cerr << "Warning: Invalid integer format for count '"
                          << raw_count_field << "' (cleaned to '" << count_str_cleaned << "') in line " << lineNumber
                          << " of file " << filename << ": " << e.what() << ". Skipping entry." << std::endl;
            }
        } catch (const std::out_of_range& e) {
            std::cerr << "Warning: Count out of range '"
                      << raw_count_field << "' (cleaned to '" << count_str_cleaned << "') in line " << lineNumber
                      << " of file " << filename << ": " << e.what() << ". Skipping entry." << std::endl;
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
    // int expectedEmbeddingSize = -1; // Not strictly needed here, 'd' is set in Tokenizer after loading

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
            if (lineNumber <= 10) { // Limit warnings
                std::cerr << "Warning: Empty token string in line " << lineNumber
                          << " of file " << filename << ". Skipping embedding for this line." << std::endl;
            }
            continue;
        }
        
        std::vector<float> embeddings_vector;
        bool parseError = false;
        
        // Extract the rest of the floats, handling potential consecutive commas (empty segments).
        bool has_more_fields_on_line = true;
        while (has_more_fields_on_line) {
            std::string raw_segment = readCsvField(ss);

            if (raw_segment.empty() && ss.eof()) { // Check for empty field at end of line
                 has_more_fields_on_line = false;
                 // If the last field was truly empty and there was a comma before it, we still want to add a 0.0f.
                 // Otherwise, if it was the end of the line, break.
                 if (line.back() == ',') { // Check if the original line ended with a comma
                     embeddings_vector.push_back(0.0f); // Treat trailing empty field as 0.0f
                 }
                 break;
            }
            if (raw_segment.empty() && ss.peek() == ',') { // Empty field followed by a comma (,,)
                embeddings_vector.push_back(0.0f); // Treat empty intermediate field as 0.0f
                ss.get(); // Consume the comma
                continue; // Continue to next field
            }
            
            std::string segment = trim(removeQuotes(raw_segment));

            try {
                float value = std::stof(segment);
                embeddings_vector.push_back(value);
            } catch (const std::invalid_argument& e) {
                if (!segment.empty()) { // Only warn if it wasn't just an empty field
                    if (lineNumber <= 5) {
                        std::cerr << "Warning: Invalid float format '" << raw_segment
                                  << "' (cleaned to '" << segment << "') for word '" << word_str << "' in line " << lineNumber
                                  << " of file " << filename << ": " << e.what() << ". Skipping line." << std::endl;
                    }
                }
                parseError = true;
                break;
            } catch (const std::out_of_range& e) {
                if (lineNumber <= 5) {
                    std::cerr << "Warning: Float out of range '" << raw_segment
                              << "' (cleaned to '" << segment << "') for word '" << word_str << "' in line " << lineNumber
                              << " of file " << filename << ": " << e.what() << ". Skipping line." << std::endl;
                }
                parseError = true;
                break;
            }
            // Check if there are more fields to read (after consuming current field and its comma)
            if (ss.peek() == EOF && !ss.good()) {
                has_more_fields_on_line = false;
            }
        } // End of while(has_more_fields_on_line)

        if (parseError) {
            continue;
        }

        // Only add to map if we got at least some embeddings
        if (!embeddings_vector.empty()) {
            mappedEmbeddings[word_str] = embeddings_vector;
            successfullyParsed++;
        } else {
            if (lineNumber <= 10) {
                std::cerr << "Warning: No embeddings found for token '" << word_str << "' in line " << lineNumber << ". Skipping." << std::endl;
            }
        }
    }

    file.close();

    if (mappedEmbeddings.empty()) {
        std::cerr << "Warning: No valid word-embedding pairs found in file " << filename << std::endl;
    } else {
        std::cout << "Successfully read " << successfullyParsed
                  << " word-embedding pairs from file " << filename << std::endl;
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

    // 2. Load the mapped embeddings (token -> vector<float>)
    std::string tokenembeddings_file = path2ClassDataFolder + "/_tokenEmbedding.csv";
    if (!std::filesystem::exists(tokenembeddings_file)) {
        std::cerr << "Error: Token embeddings file not found at " << tokenembeddings_file << std::endl;
        throw std::runtime_error("Required token embeddings file missing. Ensure training created '_tokenEmbedding.csv' in the specified path.");
    }
    this->mappedEmbeddings = readMappedEmbeddings(tokenembeddings_file);

    // --- CRITICAL STEP: Populate 'this->tokens' from the loaded vocabulary AND then 'this->embeddings' ---
    this->tokens.clear(); // Ensure it's empty before populating
    this->embeddings.clear(); // Clear existing embeddings
    this->embeddings.reserve(this->statOfTokens.size()); // Pre-allocate space

    // Sort the tokens from statOfTokens to ensure a consistent order for `this->tokens`
    // (and thus for the `embeddings` vector if you populate it in the same order).
    // This is crucial for `splitWord`'s longest-match logic.
    std::vector<std::string> sorted_tokens_from_stats;
    for (const auto& pair : this->statOfTokens) {
        sorted_tokens_from_stats.push_back(pair.first);
    }
    std::sort(sorted_tokens_from_stats.begin(), sorted_tokens_from_stats.end(),
        [](const std::string& a, const std::string& b) {
            if (a.length() != b.length()) {
                return a.length() > b.length(); // Longer tokens first
            }
            return a < b; // Alphabetical for tie-breaking
        }
    );
    this->tokens = sorted_tokens_from_stats; // Populate `this->tokens` with the sorted list

    // Now populate 'this->embeddings' and 'this->deEmbeddings' based on `this->tokens` and `this->mappedEmbeddings`
    this->d = 0; // Initialize embedding dimension
    if (!this->mappedEmbeddings.empty()) {
        this->d = this->mappedEmbeddings.begin()->second.size(); // Set 'd' based on the first loaded embedding
    }

    for (const auto& token_str : this->tokens) {
        auto it = this->mappedEmbeddings.find(token_str);
        if (it != this->mappedEmbeddings.end()) {
            this->embeddings.push_back(it->second);
        }
        else {
            // This warning will trigger if a token in _final_token_stats.csv is not in _tokenEmbedding.csv
            // The original warning for '"' means your BPE process likely produced a token that was not embedded.
            std::cerr << "Warning: Token '" << token_str << "' from '_final_token_stats.csv' missing embedding in '_tokenEmbedding.csv'. Adding zero vector." << std::endl;
            this->embeddings.push_back(std::vector<float>(this->d, 0.0f)); // Provide a zero vector as a fallback
        }
    }

    // Similarly for deEmbeddings if applicable
    this->deEmbeddings.clear();
    this->deEmbeddings.reserve(this->embeddings.size()); // Reserve space
    for (const auto& embedding_vec : this->embeddings) {
        this->deEmbeddings.push_back(vectorInverse(embedding_vec));
    }


    // Update vocabulary size based on loaded data
    this->vocSize = this->tokens.size();


    std::cout << "Tokenizer initialized successfully:" << std::endl;
    std::cout << "  - Tokens loaded: " << this->tokens.size() << std::endl;
    std::cout << "  - Vocabulary size: " << this->vocSize << std::endl;
    std::cout << "  - Embedding dimension: " << this->d << std::endl;
    std::cout << "  - deEmbedding count: " << this->deEmbeddings.size() << std::endl;
}
