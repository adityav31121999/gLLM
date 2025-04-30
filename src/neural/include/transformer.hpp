
// transformer.hpp: header source for transformer class
#ifndef TRANSFORMER_HPP
#define TRANSFORMER_HPP 1

#include <string>
#include <cmath>
#include <vector>
#include <iostream>
#include <fstream>
#include <map>

#include <maths.hpp>
#include "mlp.hpp"
#include "attention.hpp"
#include "block.hpp"

/**
 * @brief Transformer (FULL CONTEXT) class for token/chunk prediction and context 
 * retention. This can be single or multi-block architecture based on use case i.e.,
 * single block for operation or application or use and multi-block for training.
 */
class transformer {
public:
// these are constant values, once constructor is called, they cannot be changed
    bool isSelf;            // if self attention or cross attention
    bool inTraining;        // = 1 for training, = 0 for in use
    int m;                  // number of blocks
    int y;                  // number of incomplete attentions in each partial attention
    int x;                  // number of layers of partial attention for complete attention block
    int n;                  // total tokens for each attention head or context window
    int d;                  // token dimension
    int h;                  // height of MQ, MK and columns of MV, MH
    int l;                  // layers of mlp
    int epochs;             // number of epochs for MLPs and Blocks
    int totalParams;        // total parameters of transformer
    float learning;         // learning rate for MLPs
    
    int cacheOffset;        // for extracting caches
    int matOffset;          // for extracting matrices
    int mlpOffset;          // for extracting MLPs

// these are variables that change during training
    int blockCount;         // which block is working
    int epochCount;         // epoch counter
    int promptCount;        // number of tokens in the prompt
    int currentTokenCount;  // current count of tokens in full context
    int indexForToken;      // this is to set token index from embedding list

// these are variables that change during runtime
    float error;            // error for transformer
    int trainCount;         // total training count
    int vocabsize;          // size of vocabulary
    bool isTerminate;       // when '@#0' is calculated, to end the forward propagation

// containers
    std::vector<block> t;               // attention block ('1' for inference and 'm' for training)
    std::vector<std::string> tokens;    // tokens in vocabulary
    std::vector<std::string> mTokens;   // prompts and response tokens
    std::vector<float> otok;            // output token
    std::vector<std::vector<float>> embeddings;         // all glove embeddings with 64D
    std::vector<std::vector<float>> tokenEmbed;         // token embedding (prommpt + response)
    FILE* promptNresponse;              // prompt and response text file
    // when model is in inference, hold EV of ith block here
    std::vector<std::vector<std::vector<std::vector<float>>>> EVuse;
    std::vector<std::vector<float>> tokForBlock;        // token embeddings for local context for inference

#ifdef USE_OPENCL
    // If transformer owns the context, declare the object here.
    // The OpenCLContext class itself needs a default constructor or
    // the transformer constructors need to initialize it in their
    // initializer list or body.
    OpenCLContext& clcontext;
    transformer(OpenCLContext& context, int x, int y, int n, int d, int h, int l, int vocab);
    transformer(OpenCLContext& context, int x, int y, int n, int d, int h, int l, int vocab, bool attentionType);
    transformer(OpenCLContext& context, int m, int x, int y, int n, int d, int h, int l, int vocab);
    transformer(OpenCLContext& context, int m, int x, int y, int n, int d, int h, int l, int vocab, bool attentionType);
    transformer(OpenCLContext& context, int m, int x, int y, int n, int d, int h, int l, int vocab, bool attentionType, bool& inTraining);
#else
    transformer() = default;
    transformer(int x, int y, int n, int d, int h, int l, int vocab);
    transformer(int x, int y, int n, int d, int h, int l, int vocab, bool attentionType);
    transformer(int m, int x, int y, int n, int d, int h, int l, int vocab);
    transformer(int m, int x, int y, int n, int d, int h, int l, int vocab, bool attentionType);
    transformer(int m, int x, int y, int n, int d, int h, int l, int vocab, bool attentionType, bool& inTraining);
#endif

#ifdef USE_CUDA     // cuda implementation

    void cuParallelKdotQs(int& promptCount, int& currentTokenCount, int& blockCount, int& column, bool& isSelf, bool& inTraining);
    void cuForward(int& blockCount, int& currentTokenCount, int& promptCount);
    void cuBackward(std::vector<float>& expectedH);
    void cuBackward(std::vector<float>& expectedH, int& blockCount);
    void cuBackward(std::vector<std::vector<float>>& expectedH);
    void cuBackward(std::vector<std::vector<float>>& expectedH, int& blockCount);
    void cuTrain(int& promptCount, int& currentTokenCount, int& blockCount, std::vector<float>& expected, std::string&);
    void cuTrain(std::vector<std::vector<float>>& sentence, std::vector<std::string>& rString);
    void cuTrain(std::vector<std::vector<float>>& prompt, std::vector<std::vector<float>>& response, std::vector<std::string>& rString);
    void cuTrain(std::vector<std::vector<std::vector<float>>>& prompts, std::vector<std::vector<std::vector<float>>>& responses, 
                std::vector<std::vector<std::string>>& rString);
    void cuRun();

#elif USE_OPENCL    // opencl implementation
    void clParallelKdotQs(int& promptCount, int& currentTokenCount, int& blockCount, int& column, bool& isSelf, bool& inTraining);
    void clForward(int &blockCount, int &currentTokenCount, int &promptCount);
    void clBackward(std::vector<float>& expectedH);
    void clBackward(std::vector<float>& expectedH, int& blockCount);
    void clBackward(std::vector<std::vector<float>>& expectedH);
    void clBackward(std::vector<std::vector<float>>& expectedH, int& blockCount);
    void clTrain(int& promptCount, int& currentTokenCount, int& blockCount, std::vector<float>& expected, std::string&);
    void clTrain(std::vector<std::vector<float>>& sentence, std::vector<std::string>& rString);
    void clTrain(std::vector<std::vector<float>>& prompt, std::vector<std::vector<float>>& response, std::vector<std::string>& rString);
    void clTrain(std::vector<std::vector<std::vector<float>>>& prompts, std::vector<std::vector<std::vector<float>>>& responses, 
                std::vector<std::vector<std::string>>& rString);
    void clComputeOutput(std::vector<float>& output, std::vector<std::vector<float>>& embeddings, int& voc, int& index);
    void clRun();

#else

    void parallelKdotQs(int& promptCount, int& currentTokenCount, int& blockCount, int& column, bool& isSelf, bool& inTraining);
    void computeKdotQs(int& promptCount, int& currentTokenCount, int& blockCount, bool& isSelf, bool& inTraining);
    void forward(int& blockCount, int& currentTokenCount, int& promptCount);
    void backward(std::vector<float>& expectedH);
    void backward(std::vector<float>& expectedH, int& blockCount);
    void backward(std::vector<std::vector<float>>& expectedH);
    void backward(std::vector<std::vector<float>>& expectedH, int& blockCount);
    void train(int& promptCount, int& currentTokenCount, int& blockCount, std::vector<float>& expected, std::string&);
    void train(std::vector<std::vector<float>>& sentence, std::vector<std::string>& rString);
    void train(std::vector<std::vector<float>>& prompt, std::vector<std::vector<float>>& response, std::vector<std::string>& rString);
    void train(std::vector<std::vector<std::vector<float>>>& prompts, std::vector<std::vector<std::vector<float>>>& responses, 
                std::vector<std::vector<std::string>>& rString);
    void run();

#endif

    void setDims(int m, int x, int y, int n, int d, int h, int l);  // set dimension of transformer
    void setLearning(float learning);           // set learning rate for MLPs
    void setEpochs(int epochs);                 // set epochs for MLPs
    void setAttention(bool attentionType);      // set self attention (1) or cross attention (0)

    void getAllValues(int blockCount, std::string path2folderOfAllBins, bool& inTraining);
    void getcache(int blockCount, int i, int j, std::vector<std::vector<float>>& q, std::string path2file);
    void getmat(int blockCount, int i, int j, std::vector<std::vector<float>>& q, std::string path2file, int& row, int& column);
    void getmlp(int blockCount, int i, int j, std::vector<std::vector<std::vector<float>>>& m, std::string path2file);

    int tokenise(std::string &words, std::vector<std::string>& mTokens, int currentTokenCount);
    void getEmbedding(std::string& word, std::vector<float>& embed);

    // default destructor
    ~transformer() = default;
};

std::string toLower(const std::string& str);

// compute functions for dot, KdotQ and other values
void cuComputeOutput(float* d_output, float* d_embeddings, int voc_size, int& index, int embedding_dim);
void computeOutput(std::vector<float>& output, std::vector<std::vector<float>>& embeddings, int& voc, int& index);
void computeKorQ(std::vector<float>& tokenEmmbed, mat& m, std::vector<float>& KorQ);
void computeDot(std::vector<float>& T1, std::vector<float>& T2, std::vector<std::vector<float>>& M, float& dot);
void computeDot(std::vector<float>& Ti, mat M, std::vector<float>& Tj, float& dot);
void computeKdotQ(std::vector<std::vector<float>>& KdotQ, std::vector<std::vector<float>>& K, std::vector<std::vector<float>>& Q, 
    int& currentTokenCount, int& promptCount, int& blockCount, bool& attentionType);
void computeKdotQ(std::vector<std::vector<float>>& KdotQ, std::vector<std::vector<float>>& tokenEmbed, mat M, int& currentTokenCount,
    int& promptCount, bool& attentionType);
void computeKdotQ(std::vector<std::vector<float>>& KdotQ, std::vector<std::vector<float>>& tokenEmbed, std::vector<std::vector<float>>& EVp,
    mat M, int& currentTokenCount, int& promptCount, int& blockCount, bool& attentionType);

#ifdef USE_CUDA
    __global__ void accumulateEH(float** d_eh_pointers, float* d_otok, int num_layers, int embedding_dim);
    __global__ void computeAllDotsKernel(const float* vector, const float* matrix, float* results, int num_rows, int vector_dim);
#endif

#endif

/*
// i dont know why i it is needed, ai suggested it
// Singleton pattern implementation
    static transformer& getInstance() {
        if (instance == nullptr) {
            instance = new transformer();
        }
        return *instance;
    }
*/
