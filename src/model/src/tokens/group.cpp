// group.cpp
#include "include/tokenise.hpp"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <vector>
#include <set>
#include <unordered_set>
#include <unordered_map>
#include <string>
#include <regex>
#include <algorithm>
#include <future>
#include <string_view>
#include <cctype>


/**
 * @brief (INVERTED INDEX OPTIMIZATION) Learns a BPE vocabulary with extreme speed.
 * This version uses an inverted index to track which words are affected by a merge.
 * It includes corrected initialization logic and robust debugging logs to verify
 * each stage of the process.
 * @param corpus_word_counts A map of unique words and their frequencies.
 * @param num_merges The number of merge operations to perform.
 * @param final_vocab Output vector to store the learned vocabulary tokens.
 */
void tokeniser::groupCommonTokens(const std::unordered_map<std::string, int>& corpus_word_counts, int num_merges, 
    std::vector<std::string>& final_vocab) 
{
    // 1. INITIAL SETUP
    std::set<std::string> vocab;
    std::map<std::string, int> bpe_word_counts;
    std::map<std::string, std::vector<std::string>> splits;

    // A robust check for what constitutes a "word" to be split by BPE.
    auto is_word_for_bpe = [](const std::string& s) {
        return !s.empty() && std::isalpha(static_cast<unsigned char>(s[0])) && s.length() > 1;
    };

    // --- Step 1a: Separate raw tokens into BPE candidates and atomic tokens ---
    std::cout << "[DEBUG] Total unique raw tokens received: " << corpus_word_counts.size() << std::endl;
    for (const auto& pair : corpus_word_counts) {
        if (is_word_for_bpe(pair.first)) {
            bpe_word_counts[pair.first] = pair.second;
        } else {
            // This includes punctuation, symbols, and single-letter words.
            vocab.insert(pair.first);
        }
    }
    std::cout << "[DEBUG] Number of words selected for BPE processing: \t\t" << bpe_word_counts.size() << std::endl;
    std::cout << "[DEBUG] Number of initial atomic tokens (punctuation, etc.): \t" << vocab.size() << std::endl;
    std::cout << "[DEBUG] Number of Mergers to be made: \t\t\t\t" << num_merges << std::endl;

    if (bpe_word_counts.empty()) {
        std::cerr << "[WARNING] No words were long enough for BPE splitting. The vocabulary will consist of only initial tokens." << std::endl;
        final_vocab.assign(vocab.begin(), vocab.end());
        this->tokens = final_vocab;
        this->vocSize = this->tokens.size();
        return; // Exit gracefully
    }

    // --- Step 1b: Create initial character-level splits and populate base vocabulary ---
    for (const auto& pair : bpe_word_counts) {
        const std::string& word = pair.first;
        std::vector<std::string> chars;
        for (char c : word) {
            std::string s(1, c);
            chars.push_back(s);
            vocab.insert(s); // Add every single character to the initial vocab
        }
        chars.push_back("</w>"); // Add the essential end-of-word token
        splits[word] = std::move(chars);
    }
    vocab.insert("</w>"); // Ensure end-of-word token is in the vocab

    {
        for(auto& token : vocab) {
            std::cout << token << " ";
        }
    }
    std::cout << std::endl;

    // 2. BUILD INITIAL STATS AND INVERTED INDEX (ONCE!)
    std::cout << "Building initial statistics and inverted index..." << std::endl;
    std::map<std::pair<std::string, std::string>, int> pair_stats;
    std::map<std::pair<std::string, std::string>, std::vector<std::string>> inverted_index;

    for (const auto& p : splits) {
        const std::string& word = p.first;
        const auto& symbols = p.second;
        if (symbols.size() < 2) continue;
        for (size_t i = 0; i < symbols.size() - 1; ++i) {
            const auto current_pair = std::make_pair(symbols[i], symbols[i + 1]);
            pair_stats[current_pair] += bpe_word_counts.at(word);
            inverted_index[current_pair].push_back(word);
        }
    }

    std::cout << "[DEBUG] Size of initial pair_stats map: " << pair_stats.size() << ". Initialization complete. Starting merges." << std::endl;

    // 3. HIGH-SPEED MERGE LOOP
    std::cout << "Merge Count:" << std::endl;
    for (int i = 0; i < num_merges; ++i) {
        if (pair_stats.empty()) {
            std::cout << "[INFO] No more pairs to merge. Stopping at merge " << i + 1 << "." << std::endl;
            break;
        }
        auto best_pair_it = std::max_element(pair_stats.begin(), pair_stats.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });
        const auto best_pair = best_pair_it->first;
        const int best_pair_freq = best_pair_it->second;

        const std::string new_token = best_pair.first + best_pair.second;
        vocab.insert(new_token);

        if (inverted_index.find(best_pair) == inverted_index.end()) {
            pair_stats.erase(best_pair_it);
            continue;
        }
        const auto& affected_words = inverted_index.at(best_pair);

        // This loop is now very fast as it only iterates over a small number of words
        for (const auto& word : affected_words) {
            auto& symbols = splits.at(word);
            int freq = bpe_word_counts.at(word);
            if (symbols.size() < 2) continue; // Skip if already merged into a single token

            std::vector<std::string> new_symbols;
            new_symbols.reserve(symbols.size());

            size_t k = 0;
            while (k < symbols.size()) {
                if (k < symbols.size() - 1 && symbols[k] == best_pair.first && symbols[k + 1] == best_pair.second) {
                    if (k > 0) {
                        auto old_left_pair = std::make_pair(symbols[k - 1], best_pair.first);
                        pair_stats[old_left_pair] -= freq;
                        if (pair_stats[old_left_pair] <= 0) pair_stats.erase(old_left_pair);
                        
                        auto new_left_pair = std::make_pair(symbols[k - 1], new_token);
                        pair_stats[new_left_pair] += freq;
                        inverted_index[new_left_pair].push_back(word);
                    }
                    if (k < symbols.size() - 2) {
                        auto old_right_pair = std::make_pair(best_pair.second, symbols[k + 2]);
                        pair_stats[old_right_pair] -= freq;
                        if (pair_stats[old_right_pair] <= 0) pair_stats.erase(old_right_pair);

                        auto new_right_pair = std::make_pair(new_token, symbols[k + 2]);
                        pair_stats[new_right_pair] += freq;
                        inverted_index[new_right_pair].push_back(word);
                    }
                    new_symbols.push_back(new_token);
                    k += 2;
                } else {
                    new_symbols.push_back(symbols[k]);
                    k += 1;
                }
            }
            symbols = std::move(new_symbols);
        }

        inverted_index.erase(best_pair);
        pair_stats.erase(best_pair);

        if ((i + 1) % 1000 == 0 || i == num_merges - 1) {
            std::cout << "Merge " << i + 1 << "/" << num_merges << ": Merged '" << best_pair.first 
                      << "' and '" << best_pair.second << "' (Frequency: " << best_pair_freq << ")" << std::endl;
        }
    }

    // 4. FINALIZE VOCABULARY
    // final_vocab.push_back("<@#0>");     // add sentence terminator token "<@#0>"
    final_vocab.assign(vocab.begin(), vocab.end());
    std::sort(final_vocab.begin(), final_vocab.end(), [](const auto& a, const auto& b){ return a.length() > b.length(); });

    this->tokens = final_vocab;
    this->vocSize = this->tokens.size();
    std::cout << "BPE training complete. Final vocabulary size: " << this->vocSize << std::endl;
}
