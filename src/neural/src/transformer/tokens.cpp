
// tokens and their splitting for training
#include "include/transformer.hpp"

/**
 * @brief Split Tokens alternatively in prompt and response
 * @param token chunk of tokens (T1 T2 T3 T4 T5 .....)
 * @param prompt prompt with alternate spaces (T1 _ T3 _ .....)
 * @param response response with alternate spaces (_ T2 _ T4 _ .....)
 */
void alternateSplit(std::vector<std::string>& token, std::vector<std::string>& prompt, std::vector<std::string>& response) {

}


/**
 * @brief split Tokens in contiuous way in prompt and response
 * @param token chunk of tokens (T1 T2 T3 T4 T5 .....)
 * @param prompt prompt with alternate chunks (T1 T2 T3 _ _ _ _ T8 .....)
 * @param response response with alternate chunks ( _ _ _ T4 T5 T6 T7 _ .....)
 */
void continuousSplit(std::vector<std::string>& token, std::vector<std::string>& prompt, std::vector<std::string>& response) {

}


/**
 * @brief Split Tokens in question and answer format (use '?' for ending prompt as question)
 * @param token chunk of tokens (T1 T2 T3 T4 T5 .....)
 * @param prompt prompt with Question (T1 T2 T3 ..... ?)
 * @param response response with alternate (Tn T(n+1) T(n+2) _ .....)
 */
void QNASplit(std::vector<std::string>& token, std::vector<std::string>& prompt, std::vector<std::string>& response) {

}


/**
 * @brief split Tokens as sentences in prompt and response (use '.', '!', '?', ':', ';' etc for sentence ending)
 * @param token chunk of tokens (T1 T2 T3 T4 T5 .....)
 * @param prompt prompt with alternate sentences (T1 T2 T3 _ _ _ _ T8 .....)
 * @param response response with alternate ( _ _ _ T4 T5 T6 T7 _ .....)
 */
void sentenceSplit(std::vector<std::string>& token, std::vector<std::string>& prompt, std::vector<std::string>& response) {

}


/**
 * @brief Split Tokens as words and punctuations in prompt and response
 * @param token chunk of tokens (T1 T2 T3 T4 T5 .....)
 * @param prompt words
 * @param response punctuations
 */
void punctutationSplit(std::vector<std::string>& token, std::vector<std::string>& prompt, std::vector<std::string>& response) {

}


/**
 * @brief Split Tokens randomly
 * @param token chunk of tokens (T1 T2 T3 T4 T5 .....)
 * @param prompt continuous tokens (T1 _ T3 T4 T5 T6 T7 _ T9 .....)
 * @param response random selection (_ T2 _ _ _ _ _ T8 _ .....)
 */
void fillInTheBlanks(std::vector<std::string>& token, std::vector<std::string>& prompt, std::vector<std::string>& response) {

}
