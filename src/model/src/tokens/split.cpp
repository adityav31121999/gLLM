// split.cpp
#include "include/tokenise.hpp"
#include <future>
#include <fstream>
#include <sstream>
#include <regex>
#include <map>
#include <algorithm>
#include <iostream>
#include <cctype>
#include <string_view>
#include <atomic>
#include <thread>
#include <chrono>
#include <iomanip>


/**
 * @brief Pre-tokenizes a single word based on common patterns like camelCase or PascalCase.
 * This version uses a fast, manual character-by-character scan instead of std::regex,
 * which significantly improves performance in multithreaded contexts by avoiding
 * repeated regex compilation and execution overhead.
 * It splits words at transitions from lowercase to uppercase letters, and also handles acronyms
 * like in "MyHTTPRequest" -> "My", "HTTP", "Request".
 * Note: This heuristic will not split already-lowercased concatenated words like
 * "earlychristianwritings". That task is left to the BPE algorithm itself.
 * @param word The word to potentially split (case-sensitive).
 * @return A vector of sub-words. If no split occurs, it returns a vector with the original word.
 */
std::vector<std::string_view> pre_split_word(std::string_view word) {
    if (word.empty()) {
        return {};
    }

    std::vector<std::string_view> subtokens;
    size_t start = 0;

    for (size_t i = 1; i < word.length(); ++i) {
        bool split = false;
        const unsigned char prev_char = word[i-1];
        const unsigned char curr_char = word[i];

        // Condition 1: Lower to Upper (e.g., "camel|Case")
        if (std::islower(prev_char) && std::isupper(curr_char)) {
            split = true;
        }
        // Condition 2: Acronym to Word (e.g., "HTTP|Request")
        // This detects a transition from an uppercase sequence to a new capitalized word.
        else if (i + 1 < word.length() && std::isupper(prev_char) && std::isupper(curr_char) && std::islower(static_cast<unsigned char>(word[i+1]))) {
            split = true;
        }

        if (split) {
            subtokens.emplace_back(word.substr(start, i - start));
            start = i;
        }
    }

    // Add the last or only token.
    subtokens.emplace_back(word.substr(start));
    return subtokens;
}


/**
 * @brief Splits a single word into a sequence of subword tokens using the learned vocabulary.
 * This function implements the tokenization of a single word by greedily matching the
 * longest possible tokens from the vocabulary.
 * @param word The word to be tokenized.
 * @param subwords Output vector to store the resulting subword tokens.
 */
void tokeniser::splitWord(const std::string& word, std::vector<std::string>& subwords) const {
    subwords.clear();
    if (word.empty()) return;

    // Add end-of-word token to handle word boundaries correctly
    std::string current_word = word + "</w>";
    
    while (!current_word.empty()) {
        bool found_match = false;
        // Greedily find the longest token in our vocabulary that is a prefix of the current word.
        // This requires the vocabulary `this->tokens` to be sorted by length, descending.
        for (const auto& token : this->tokens) {
            if (current_word.rfind(token, 0) == 0) { // check if string starts with token
                subwords.push_back(token);
                current_word = current_word.substr(token.length());
                found_match = true;
                break;
            }
        }
        if (!found_match) {
            // Fallback for unknown characters. This should not happen if the initial
            // vocabulary includes all single characters from the training corpus.
            subwords.push_back(current_word.substr(0, 1));
            current_word = current_word.substr(1);
        }
    }
}


/**
 * @brief Tokenizes a full sentence into a sequence of subword tokens.
 * This function splits a sentence into words and punctuation, then tokenizes
 * each word using the `splitWord` method.
 * @param sentence The input sentence string.
 * @param all_subwords Output vector to store the final sequence of tokens.
 */
void tokeniser::splitSentence(const std::string& sentence, std::vector<std::string>& all_subwords) const {
    all_subwords.clear();

    // Regex to split sentence by words and punctuation.
    // It captures sequences of letters OR single non-letter, non-space characters.
    std::regex re(R"([a-zA-Z]+|[^a-zA-Z\s])");
    auto words_begin = std::sregex_iterator(sentence.begin(), sentence.end(), re);
    auto words_end = std::sregex_iterator();

    for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
        std::string token_str = (*i).str();
        
        // If it's a word (starts with a letter), split it using our learned vocabulary
        if (std::isalpha(token_str[0])) {
            std::string lower_token_str = token_str;
            std::transform(lower_token_str.begin(), lower_token_str.end(), lower_token_str.begin(), ::tolower);
            
            std::vector<std::string> word_subwords;
            splitWord(lower_token_str, word_subwords);
            all_subwords.insert(all_subwords.end(), word_subwords.begin(), word_subwords.end());
        } 
        else {
            // It's punctuation or another symbol, keep it as a single token
            all_subwords.push_back(token_str);
        }
    }
}