
// model.cpp: implementation of Model class
#include "include/model.hpp"
#include <iostream>
#include <stdexcept>
#include <filesystem>
#include <fstream>
#include <locale> // for isspace

#ifdef USE_OPENCL

/**
 * @brief Constructor for model (training on self attention)
 * @param m number of blocks
 * @param x number of incomplete attentions in each partial attention
 * @param y number of layers of partial attention for complete attention block
 * @param n total tokens for each attention head
 * @param d token dimension
 * @param h height of MQ, MK and columns of MV, MH
 * @param l layers of mlp
 */
model::model(OpenCLContext& context, int m, int x, int y, int n, int d, int h, int l, int vocab) :
    clcontext(context), m(m), x(x), y(y), n(n), d(d), h(h), l(l),
    T(context, m, x, y, n, d, h, l, vocab) // Initialize T in the initializer list
{
    totalParams = m * x * y * ((4 * h * d) + (2 * d * d * l));
    isSelf = 1;
    toTrain = 1;
    // allocate float value block of size totalParams to file
    // allocateMemory();
}


/**
 * @brief Constructor for model with learning rate (training on self attention)
 * @param m number of blocks
 * @param x number of incomplete attentions in each partial attention
 * @param y number of layers of partial attention for complete attention block
 * @param n total tokens for each attention head
 * @param d token dimension
 * @param h height of MQ, MK and columns of MV, MH
 * @param l layers of mlp
 * @param learning learning rate for MLPs
 */
model::model(OpenCLContext& context, int m, int x, int y, int n, int d, int h, int l, float learning, int vocab) :
    clcontext(context), m(m), x(x), y(y), n(n), d(d), h(h), l(l), learning(learning),
    T(context, m, x, y, n, d, h, l, vocab) // Initialize T in the initializer list
{
    totalParams = m * x * y * ((4 * h * d) + (2 * d * d * l));
    T.setLearning(learning);  // Set learning rate for the transformer
    isSelf = 1;
    toTrain = 1;
    // allocate float value block of size totalParams to file
    // allocateMemory();
}


/**
 * @brief Constructor for model
 * @param m number of blocks
 * @param x number of incomplete attentions in each partial attention
 * @param y number of layers of partial attention for complete attention block
 * @param n total tokens for each attention head
 * @param d token dimension
 * @param h height of MQ, MK and columns of MV, MH
 * @param l layers of mlp
 */
model::model(OpenCLContext& context, int m, int x, int y, int n, int d, int h, int l, int vocab, bool isSelfAttention, bool toTrainModel) :
    clcontext(context), m(m), x(x), y(y), n(n), d(d), h(h), l(l), isSelf(isSelfAttention), toTrain(toTrainModel),
    T(context, (toTrainModel ? m : 1), x, y, n, d, h, l, vocab, isSelfAttention, toTrainModel) // Initialize T using the appropriate transformer constructor
{
    totalParams = m * x * y * ((4 * h * d) + (2 * d * d * l));
    // allocate float value block of size totalParams to file
    // allocateMemory();
}


/**
 * @brief Constructor for single transformer model with learning rate
 * @param m number of blocks
 * @param x number of incomplete attentions in each partial attention
 * @param y number of layers of partial attention for complete attention block
 * @param n total tokens for each attention head
 * @param d token dimension
 * @param h height of MQ, MK and columns of MV, MH
 * @param l layers of mlp
 * @param learning learning rate for MLPs
 */
model::model(OpenCLContext& context, int m, int x, int y, int n, int d, int h, int l, float learning, int vocab, bool isSelfAttention, bool toTrainModel) :
    clcontext(context), m(m), x(x), y(y), n(n), d(d), h(h), l(l), learning(learning), isSelf(isSelfAttention), toTrain(toTrainModel),
    T(context, m, x, y, n, d, h, l, vocab, isSelfAttention, toTrainModel) // Initialize T using the appropriate transformer constructor
{
    totalParams = m * x * y * ((4 * h * d) + (2 * d * d * l));
    T.setLearning(learning);  // Set learning rate for the transformer
    // allocate float value block of size totalParams to file
    // allocateMemory();
}

#else 

/**
 * @brief Constructor for model (training on self attention)
 * @param m number of blocks
 * @param x number of incomplete attentions in each partial attention
 * @param y number of layers of partial attention for complete attention block
 * @param n total tokens for each attention head
 * @param d token dimension
 * @param h height of MQ, MK and columns of MV, MH
 * @param l layers of mlp
 */
model::model(int m, int x, int y, int n, int d, int h, int l, int vocab) :
    m(m), x(x), y(y), n(n), d(d), h(h), l(l) {
    totalParams = m * x * y * ((4 * h * d) + (2 * d * d * l));
    T = transformer(m, x, y, n, d, h, l, vocab);  // transformer(int m, int x, int y, int n, int d, int h, int l);
    isSelf = 1;
    toTrain = 1;
    // allocate float value block of size totalParams to file
    // allocateMemory();
}


/**
 * @brief Constructor for model with learning rate (training on self attention)
 * @param m number of blocks
 * @param x number of incomplete attentions in each partial attention
 * @param y number of layers of partial attention for complete attention block
 * @param n total tokens for each attention head
 * @param d token dimension
 * @param h height of MQ, MK and columns of MV, MH
 * @param l layers of mlp
 * @param learning learning rate for MLPs
 */
model::model(int m, int x, int y, int n, int d, int h, int l, float learning, int vocab) :
    m(m), x(x), y(y), n(n), d(d), h(h), l(l), learning(learning) {
    totalParams = m * x * y * ((4 * h * d) + (2 * d * d * l));
    T = transformer(m, x, y, n, d, h, l, vocab);
    T.setLearning(learning);  // Set learning rate for the transformer
    isSelf = 1;
    toTrain = 1;
    // allocate float value block of size totalParams to file
    // allocateMemory();
}


/**
 * @brief Constructor for model
 * @param m number of blocks
 * @param x number of incomplete attentions in each partial attention
 * @param y number of layers of partial attention for complete attention block
 * @param n total tokens for each attention head
 * @param d token dimension
 * @param h height of MQ, MK and columns of MV, MH
 * @param l layers of mlp
 */
model::model(int m, int x, int y, int n, int d, int h, int l, int vocab, bool isSelfAttention, bool toTrainModel) :
    m(m), x(x), y(y), n(n), d(d), h(h), l(l), isSelf(isSelfAttention), toTrain(toTrainModel)
{
    totalParams = m * x * y * ((4 * h * d) + (2 * d * d * l));
    if(toTrainModel == 1) {
        T = transformer(m, x, y, n, d, h, l, vocab, isSelfAttention);
    }
    else {
        T = transformer(x, y, n, d, h, l, vocab, isSelfAttention);
    }
    // allocate float value block of size totalParams to file
    // allocateMemory();
}


/**
 * @brief Constructor for training transformer model with learning rate
 * @param m number of blocks
 * @param x number of incomplete attentions in each partial attention
 * @param y number of layers of partial attention for complete attention block
 * @param n total tokens for each attention head
 * @param d token dimension
 * @param h height of MQ, MK and columns of MV, MH
 * @param l layers of mlp
 * @param learning learning rate for MLPs
 */
model::model(int m, int x, int y, int n, int d, int h, int l, float learning, int vocab, bool isSelfAttention, bool toTrainModel) :
    m(m), x(x), y(y), n(n), d(d), h(h), l(l), learning(learning), isSelf(isSelfAttention), toTrain(toTrainModel)
{
    totalParams = m * x * y * ((4 * h * d) + (2 * d * d * l));
    T = transformer(m, x, y, n, d, h, l, vocab, isSelfAttention);
    T.setLearning(learning);  // Set learning rate for the transformer
    // allocate float value block of size totalParams to file
    // allocateMemory();
}

#endif


void model::setLearning(float learning) {
    this->learning = learning;
    T.setLearning(learning);
}

void model::setVocab(int vocab) {
    info.vocab = vocab;
}

void model::setModelName(std::string& modelName) {
    info.modelName = modelName;
}

void model::setVersion(std::string& version) {
    info.version = version;
}

void model::setAuthor(std::string& author) {
    info.author = author;
}

void model::setDate(std::string& date) {
    info.date = date;
}

void model::setLicense(std::string& license) {
    info.license = license;
}

void model::setInfo(modelDataInfo& info) {
    this->info = info;
}

void model::setInfo(std::string& modelName, std::string& version, std::string& author, 
                   std::string& date, std::string& modelArch, std::string& license, 
                   std::string& trainingData) {
    info.modelName = modelName;
    info.version = version;
    info.author = author;
    info.date = date;
    info.modelArch = modelArch;
    info.license = license;
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
    // Clear the output vectors first
    tokensOfFile.clear(); // Not used in this implementation based on the primary request
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
            // Reset for the next sentence, regardless of whether this one was added
            current_sentence.clear();
        }
    }

    // Handle the last part of the text if it doesn't end with a delimiter
    // Check if the remaining current_sentence contains non-whitespace content
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
        } else {
            evenSentence.push_back(sentence_vec);
        }
    }
    // File is closed automatically when ifs goes out of scope
}
