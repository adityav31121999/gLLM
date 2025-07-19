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
