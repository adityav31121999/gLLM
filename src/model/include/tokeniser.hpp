#ifndef TOKENISER_HPP
#define TOKENISER_HPP

#include <map>
#include <string>
#include <vector>

class TOKENISER {
private:
    int d;          // embedding dimension
    int vocSize;    // vocaulary size
    int chunkSize;  // chunk sizes for Large Context Models
    int tokenSize;  // maximum number of tokens

public:

    std::string path2data;          // path to data file

    // polynomial for embedding
    // monomial for embedding
    // fft for embedding
    // n-nacci for embedding
    // series for embedding

    std::vector<std::string> tokens;            // tokens (vocSize)
    std::vector<float> seeds;                       // seed for each embeddings (random values)
    std::vector<std::vector<float>> embeddings;     // embeddings for all tokens (vocSize x d)
    std::map<std::string, int> wordMap;         // word and its occurence
    std::map<std::string, int> tokenMap;        // token and its occurence
    std::map<std::string, std::vector<float>> tokenEmbeddings;          // token and its occurence
};

#endif