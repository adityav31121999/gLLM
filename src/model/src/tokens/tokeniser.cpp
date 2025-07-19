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
    }
    catch (const std::exception& e) {
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