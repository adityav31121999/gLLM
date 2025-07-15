#include "include/tokenise.hpp"
#include <string>
#include <sstream>
#include <string_view>
#include <regex>
#include <fstream>
#include <iostream>
#include <filesystem>

#ifdef USE_OPENCL

/**
 * @brief constructor for tokeniser (use dimensions directly)
 * @param d dimension for embedding
 */
tokeniser::tokeniser(int d, OpenCLContext& context) : d(d), bpe_progress(std::make_unique<ProgressData>()), ocl(context) {}

/**
 * @brief constructor for tokeniser (use dimensions directly)
 * @param d dimension for embedding
 * @param vocSize vocabulary size
 */
tokeniser::tokeniser(int d, int d_val, OpenCLContext& context) : d(d), d_val(d_val), bpe_progress(std::make_unique<ProgressData>()), ocl(context) {}


/**
 * @brief constructor for tokeniser (use data set directly) - preserves CSV order
 * @param path2data path to folder with all dataset files
 */
tokeniser::tokeniser(const std::string& path2data, OpenCLContext& context) noexcept : path2data(path2data), bpe_progress(std::make_unique<ProgressData>()), ocl(context)
{
    try {
        // Read the CSV file once and build both data structures
        readFromFiles(path2data);
        /*
        std::ifstream file(path2data + "/_final_token_stats.csv");
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << path2data + "/_final_token_stats.csv" << std::endl;
            return;
        }

        std::string line;
        int lineNumber = 0;
        int successfullyParsed = 0;
        bool headerSkipped = false;
        
        // Clear existing data
        this->tokens.clear();
        this->statOfTokens.clear();
        
        while (std::getline(file, line)) {
            lineNumber++;
            
            if (line.empty()) continue;
            
            line = trim(line);
            if (line.empty()) continue;
            
            // Skip header line
            if (!headerSkipped && isHeaderLine(line)) {
                headerSkipped = true;
                std::cout << "Skipping header line: " << line << std::endl;
                continue;
            }
            
            // Parse the line
            std::vector<std::string> columns;
            std::stringstream ss(line);
            std::string segment;
            
            while (std::getline(ss, segment, ',')) {
                columns.push_back(trim(segment));
            }
            
            if (columns.size() < 2) {
                if (lineNumber <= 5) {
                    std::cerr << "Warning: Insufficient columns in line " << lineNumber << std::endl;
                }
                continue;
            }
            
            std::string token = removeQuotes(columns[0]);
            std::string count_str = removeQuotes(columns[1]);
            
            // Skip empty tokens
            if (token.empty()) {
                if (lineNumber <= 5) {
                    std::cerr << "Warning: Empty token in line " << lineNumber << std::endl;
                }
                continue;
            }
            
            // Clean count string
            count_str.erase(std::remove(count_str.begin(), count_str.end(), '"'), count_str.end());
            if (count_str.empty() || count_str[0] == ',') {
                if (lineNumber <= 5) {
                    std::cerr << "Warning: Invalid count format in line " << lineNumber << std::endl;
                }
                continue;
            }
            
            try {
                int count = std::stoi(count_str);
                
                // Add to both data structures
                this->tokens.push_back(token);
                this->statOfTokens[token] = count;
                successfullyParsed++;
                
            } catch (const std::exception& e) {
                std::cerr << "Warning: Invalid count '" << count_str 
                          << "' in line " << lineNumber << ": " << e.what() << std::endl;
            }
        }
        
        file.close();
        
        std::cout << "Successfully read " << successfullyParsed 
                  << " token-count pairs from file" << std::endl;
        
        // Read embeddings
        this->mappedEmbeddings = readMappedEmbeddings(path2data + "/_final_embeddings.csv");
        
        // Set vocabulary size
        this->vocSize = this->statOfTokens.empty() ? 0 : this->statOfTokens.size() - 1;
        
        // Set embedding dimension
        if (!this->mappedEmbeddings.empty()) {
            this->d = this->mappedEmbeddings.begin()->second.size();
        } else {
            this->d = 0;
            std::cerr << "Warning: No embeddings loaded, setting dimension to 0" << std::endl;
        }
        
        std::cout << "Tokenizer initialized successfully:" << std::endl;
        std::cout << "  - Tokens loaded: " << this->tokens.size() << std::endl;
        std::cout << "  - Vocabulary size: " << this->vocSize << std::endl;
        std::cout << "  - Embedding dimension: " << this->d << std::endl;
        */
    } catch (const std::exception& e) {
        std::cerr << "Error initializing tokenizer: " << e.what() << std::endl;
        this->vocSize = 0;
        this->d = 0;
    }
}


#elif USE_CUDA || USE_CPU

/**
 * @brief constructor for tokeniser (use dimensions directly)
 * @param d dimension for embedding
 */
tokeniser::tokeniser(int d) : d(d), bpe_progress(std::make_unique<ProgressData>()) {}

/**
 * @brief constructor for tokeniser (use dimensions directly)
 * @param d dimension for embedding
 * @param vocSize vocabulary size
 */
tokeniser::tokeniser(int d, int d_val) : d(d), d_val(d_val), bpe_progress(std::make_unique<ProgressData>()) {}

/**
 * @brief constructor for tokeniser (use data set directly) - preserves CSV order
 * @param path2data path to folder with all dataset files
 */
tokeniser::tokeniser(const std::string& path2data) noexcept : path2data(path2data), bpe_progress(std::make_unique<ProgressData>())
{
    try {
        // Read the CSV file once and build both data structures
        readFromFiles(path2data);
        /*
        std::ifstream file(path2data + "/_final_token_stats.csv");
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << path2data + "/_final_token_stats.csv" << std::endl;
            return;
        }

        std::string line;
        int lineNumber = 0;
        int successfullyParsed = 0;
        bool headerSkipped = false;
        
        // Clear existing data
        this->tokens.clear();
        this->statOfTokens.clear();
        
        while (std::getline(file, line)) {
            lineNumber++;
            
            if (line.empty()) continue;
            
            line = trim(line);
            if (line.empty()) continue;
            
            // Skip header line
            if (!headerSkipped && isHeaderLine(line)) {
                headerSkipped = true;
                std::cout << "Skipping header line: " << line << std::endl;
                continue;
            }
            
            // Parse the line
            std::vector<std::string> columns;
            std::stringstream ss(line);
            std::string segment;
            
            while (std::getline(ss, segment, ',')) {
                columns.push_back(trim(segment));
            }
            
            if (columns.size() < 2) {
                if (lineNumber <= 5) {
                    std::cerr << "Warning: Insufficient columns in line " << lineNumber << std::endl;
                }
                continue;
            }
            
            std::string token = removeQuotes(columns[0]);
            std::string count_str = removeQuotes(columns[1]);
            
            // Skip empty tokens
            if (token.empty()) {
                if (lineNumber <= 5) {
                    std::cerr << "Warning: Empty token in line " << lineNumber << std::endl;
                }
                continue;
            }
            
            // Clean count string
            count_str.erase(std::remove(count_str.begin(), count_str.end(), '"'), count_str.end());
            if (count_str.empty() || count_str[0] == ',') {
                if (lineNumber <= 5) {
                    std::cerr << "Warning: Invalid count format in line " << lineNumber << std::endl;
                }
                continue;
            }
            
            try {
                int count = std::stoi(count_str);
                
                // Add to both data structures
                this->tokens.push_back(token);
                this->statOfTokens[token] = count;
                successfullyParsed++;
                
            } catch (const std::exception& e) {
                std::cerr << "Warning: Invalid count '" << count_str 
                          << "' in line " << lineNumber << ": " << e.what() << std::endl;
            }
        }
        
        file.close();
        
        std::cout << "Successfully read " << successfullyParsed 
                  << " token-count pairs from file" << std::endl;
        
        // Read embeddings
        this->mappedEmbeddings = readMappedEmbeddings(path2data + "/_final_embeddings.csv");
        
        // Set vocabulary size
        this->vocSize = this->statOfTokens.empty() ? 0 : this->statOfTokens.size() - 1;
        
        // Set embedding dimension
        if (!this->mappedEmbeddings.empty()) {
            this->d = this->mappedEmbeddings.begin()->second.size();
        } else {
            this->d = 0;
            std::cerr << "Warning: No embeddings loaded, setting dimension to 0" << std::endl;
        }
        
        std::cout << "Tokenizer initialized successfully:" << std::endl;
        std::cout << "  - Tokens loaded: " << this->tokens.size() << std::endl;
        std::cout << "  - Vocabulary size: " << this->vocSize << std::endl;
        std::cout << "  - Embedding dimension: " << this->d << std::endl;
        */
    } catch (const std::exception& e) {
        std::cerr << "Error initializing tokenizer: " << e.what() << std::endl;
        this->vocSize = 0;
        this->d = 0;
    }
}

#endif


// Helper function to allow splitting file_paths among producers
// This is not a member function, can be placed in the .cpp file before buildCorpusWordCountsParallel
std::vector<std::vector<std::string>> split_vector_for_producers(const std::vector<std::string>& files, 
    int num_producers)
{
    std::vector<std::vector<std::string>> splits(num_producers);
    size_t files_per_producer = files.size() / num_producers;
    size_t remainder = files.size() % num_producers;

    size_t current_file_idx = 0;
    for (int i = 0; i < num_producers; ++i) {
        size_t count = files_per_producer + (i < remainder ? 1 : 0);
        for (size_t j = 0; j < count; ++j) {
            splits[i].push_back(files[current_file_idx++]);
        }
    }
    return splits;
}