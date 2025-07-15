#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include <string>
#include "include/tokenise.hpp"

/**
 * @brief training function for tokenisaton of data
 * @param[in] path2trainData path to training data (txt, csv, etc.)
 * @param[in] num_merges number of merges
 * @param[out] path2tokenData path to tokenised data obtained via training (csv file)
 * for tokens stats, embeddings and unique tokens
 */
void tokeniser::train(const std::string& path2trainData, int num_merges, const std::string& path2tokenData)
{
    setNumThreads(); // Automatically set to use max threads
    std::cout << "-> Number of threads for CPU: " << num_threads << std::endl;

    const std::string unique_tokens_output_path = path2tokenData + "/" + "_unique_initial_tokens.csv";
    const std::string stats_output_path = path2tokenData + "/" + "_final_token_stats.csv";
    const std::string embeddings_output_path = path2tokenData + "/" + "_final_embeddings.csv";

    std::cout << "------------------------ 1. AGGREGATING DATA --------------------------" << std::endl;
    // Step A: Collect all file paths
    std::vector<std::string> all_file_paths;
    for (const auto& entry : std::filesystem::directory_iterator(path2trainData)) {
        if (entry.is_regular_file()) {
            all_file_paths.push_back(entry.path().string());
        }
    }
    std::cout << "-> Found " << all_file_paths.size() << " files for training in: " << path2trainData << std::endl;
    if (all_file_paths.empty()) throw std::runtime_error("No files found in the specified directory.");
    // Step B: Build word counts using the robust producer-consumer model
    std::unordered_map<std::string, int> corpus_word_counts;
    buildCorpusWordCounts(all_file_paths, corpus_word_counts);
    std::cout << "-> Data aggregation complete. Total unique raw tokens: " << corpus_word_counts.size() << std::endl;
    if (corpus_word_counts.empty()) 
        throw std::runtime_error("No data loaded from files. Check file content.");
    // Step C: Save the gathered unique raw tokens to a CSV file.
    saveUniqueTokensToCSV(corpus_word_counts, unique_tokens_output_path);
    std::cout << "-> " << std::filesystem::path(unique_tokens_output_path).filename().string() << " contains " << count_lines(unique_tokens_output_path) << " rows." << std::endl;

    std::cout << "--------------------------- 2. VOCABULARY LEARNING ---------------------------" << std::endl;
    std::vector<std::string> final_vocabulary;
    // Call the new two-stage learning function.
    learn_vocabulary_from_word_counts(corpus_word_counts, num_merges, final_vocabulary);
    std::cout << "-> Vocabulary Learning complete. Final vocabulary size: " << getVocabularySize() << std::endl;

    std::cout << "---------------------- 3. STATS & EMBEDDING GEN -----------------------" << std::endl;
    // Step A: Calculate statistics based on the final BPE vocabulary
    calculateTokenStatsFromCounts(corpus_word_counts, stats_output_path);
    std::cout << "-> " << std::filesystem::path(stats_output_path).filename().string() << " contains " << count_lines(stats_output_path) << " rows." << std::endl;
    // Step B: Generate embeddings using original formula
    generateAndSaveEmbeddings(embeddings_output_path, -10.0f, 10.0f);
    std::cout << "-> " << std::filesystem::path(stats_output_path).filename().string() << " contains " << count_lines(stats_output_path) << " rows." << std::endl;
    std::cout << "-> " << std::filesystem::path(embeddings_output_path).filename().string() << " contains " << count_lines(embeddings_output_path) << " rows." << std::endl;
}