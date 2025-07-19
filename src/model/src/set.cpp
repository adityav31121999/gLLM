#include "include/model.hpp"
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <iostream>
#include <locale>

// Helper to ensure file has a certain size.
int ensure_file_size_basic(FILE* fp, size_t required_size) {
    if (!fp) return -1;
    long current_pos = ftell(fp);
    if (fseek(fp, 0, SEEK_END) != 0) return -1;
    long current_size = ftell(fp);
    if (current_size < 0) return -1;

    if (static_cast<size_t>(current_size) < required_size) {
        if (fseek(fp, required_size - 1, SEEK_SET) != 0) return -1;
        if (fwrite("", 1, 1, fp) != 1) return -1; // Write a single byte to extend
        if (fflush(fp) != 0) return -1;
    }
    // Restore original position or rewind
    if (current_pos != -1) fseek(fp, current_pos, SEEK_SET);
    else rewind(fp);
    return 0;
}

// set learning for model training
void model::setLearning(float learning) {
    this->learning = learning;
    T.setLearning(learning);
}

// set vocabulary size
void model::setVocab(int vocab) {
    info.vocab = vocab;
}

// set name of model
void model::setModelName(const std::string& modelName) {
    info.modelName = modelName;
}

// set version of model
void model::setVersion(const std::string& version) {
    info.version = version;
}

// set name of developer/author
void model::setAuthor(const std::string& author) {
    info.author = author;
}

// set date of creation/or date of launch
void model::setDate(const std::string& date) {
    info.date = date;
}

// license type
void model::setLicense(const std::string& license) {
    info.license = license;
}

// set model info
void model::setInfo(modelDataInfo& info) {
    this->info = info;
}

// set model info
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


// set embeddings for transformer via tokeniser data
void model::setTokenAndEmbeddingForTransformer(tokeniser &tok) {
    std::cout << "TOK.embeddings dimensions: " << tok.getEmbeddingDimension() << " x " << tok.getVocabularySize() << std::endl;
    // set tokens to T.tokens
    T.tokens = tok.getTokens();         // take tokens from token
    T.tokens.push_back("<@#0>");        // add terminator
    // set embeddings to T.embeddings
    if(T.tokens.size() != T.embeddings.row) {
        throw std::runtime_error("setTokenAndEmbeddingForTransformer: Vocabulary size don't match. Size is " + std::to_string(T.tokens.size()) + "and " + std::to_string(T.embeddings.row) + ".");
    }
    std::cout << "setTokenAndEmbeddingForTransformer: Adding embeddings from index '0' to second last i.e., " << T.vocabsize - 1 << std::endl;
    std::vector<std::vector<float>> emb = tok.getEmbeddings();
    float f = 10.0f;
    int i = 0;
    std::vector<float> vec(EMBEDDING);
    for (int i = 0; i < EMBEDDING; ++i) {
        vec[i] = terminatorEmbed(f, i);
    }
    emb.push_back(vec);
    T.embeddings = emb;
    std::cout << "setTokenAndEmbeddingForTransformer: Adding terminator embedding at index " << T.vocabsize << std::endl;
    // T.embeddings.addRow(vec, T.tokens.size());
    T.vocabsize = T.tokens.size();
    vocabsize = T.tokens.size();
    if(T.d != tok.getEmbeddingDimension()) {
        throw std::runtime_error("setTokenAndEmbeddingForTransformer: Embedding dimension don't match. Size is " + std::to_string(T.d) + " and " + std::to_string(tok.getEmbeddingDimension()) + ".");
    }
    std::cout << "setTokenAndEmbeddingForTransformer: Tokens and Embeddings set to transformer with vocabulary size of " << T.vocabsize << std::endl;
}