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
#include "attention.hpp"
#include "block.hpp"

/**
 * @brief Transformer (FULL CONTEXT) class for token/chunk prediction and context 
 * retention. This can be single or multi-block architecture based on use case i.e.,
 * single block for operation or application or use and multi-block for training.
 */
class transformer {
public:
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
    int blockCount;         // which block is working
    int epochCount;         // epoch counter
    int promptCount;        // number of tokens in the prompt
    int currentTokenCount;  // current count of tokens in full context
    int indexForToken;      // this is to set token index from embedding list
    int resCount;           // response count for every prompt
    float learning;         // learning rate for MLPs
    double totalLearning;   // total learning for all updates (adaptive learning)
    float lambda_L1;        // lambda for L1 penalty
    float lambda_L2;        // lambda for L2 penalty
    float error;            // error for transformer (after complete trainin)
    float cErr;             // current error for ongoing iteration of training
    float pErr1;            // previous iteration's error
    float pErr2;            // previous to previous iteration's error
    float fErr;             // next iteration's predicted error
    bool isTerminate;       // when '</s>' is calculated, to end the forward propagation
    long long int params;           // parameters in transformer
    long long int trainCount;       // total training count
    long long int vocabsize;        // size of vocabulary
    long long int cacheOffset;      // for extracting caches
    long long int matOffset;        // for extracting matrices
    long long int mlpOffset;        // for extracting MLPs

    std::vector<block> t;               // attention block ('1' for inference and 'm' for training)
    std::vector<std::string> tokens;    // tokens in vocabulary
    std::vector<std::string> mTokens;   // prompts and response tokens
    std::vector<int> indexVec;          // indices vector
    std::vector<int> expIndex;          // expected index vector
    std::vector<float> otok;            // output token (vector, size d * NUMBER_OF_HEADS)
    std::vector<float> pred;            // prediction from forprop
    std::vector<float> oneHotEncode;    // one hot encoding with 1 at predicted token index and rest are set to 0
    mat embeddings;                     // all trained embeddings (Mapped, vocabsize x d)
    mat deEmbeddings;                   // deEmbeddings obtained while training
    mat tokenEmbed;                     // token embedding (prompt + response) (Mapped, currentTokenCount x d)
    FILE* promptNresponse;              // prompt and response text file
    // when model is in inference, hold EV of ith block here
    std::vector<std::vector<std::vector<std::vector<float>>>> EVuse; // Keeping as vector due to complexity
    mat tokForBlock;                    // token embeddings for local context for inference (Mapped, n x d)

#ifdef USE_OPENCL
    OpenCLContext& clcontext;
    // Constructor with explicit learning rate
    transformer(OpenCLContext& context, int m_param, int x_param, int y_param, int n_param, int d_param, int h_param, int l_param, 
        long long int vocab_param, float learning_rate_param, float lambda_L1_param, float lambda_L2_param, bool attentionType_param, 
        bool& inTraining_param, const std::string& modelDir_param);
#elif USE_CUDA || USE_CPU
    transformer() = default;
    // Constructor with explicit learning rate
    transformer(int m_param, int x_param, int y_param, int n_param, int d_param, int h_param, int l_param, long long int vocab_param, 
        float learning_rate_param, float lambda_L1_param, float lambda_L2_param, bool attentionType_param, bool& inTraining_param, 
        const std::string& blockBinPath);
#endif

#ifdef USE_CUDA

// cuda implementation
    void cuParallelKdotQs(int& promptCount, int& currentTokenCount, int& blockCount, int& column, bool& isSelf, bool& inTraining);
    void cuForward(int& blockCount, int& currentTokenCount, int& promptCount);
    void cuBackward(std::vector<float>& expectedH);
    void cuBackward(std::vector<float>& expectedH, int& blockCount);
    void cuBackward(std::vector<std::vector<float>>& expectedH);
    void cuBackward(std::vector<std::vector<float>>& expectedH, int& blockCount);
    void cuTrain(int& promptCount, int& currentTokenCount, int& blockCount, std::vector<float>& expected, std::string& rString);
    void cuTrain(std::vector<std::vector<float>>& sentence, std::vector<std::string>& rString);
    void cuTrain(std::vector<std::vector<float>>& prompt, std::vector<std::vector<float>>& response, std::vector<std::string>& rString);
    void cuTrain(std::vector<std::vector<std::vector<float>>>& prompts, std::vector<std::vector<std::vector<float>>>& responses, 
                std::vector<std::vector<std::string>>& rString);
    void cuTest(std::vector<std::vector<float>>& prompt, std::vector<std::vector<float>>& response, std::vector<std::string>& rString);
    void cuTest(std::vector<std::vector<std::vector<float>>>& prompts, std::vector<std::vector<std::vector<float>>>& responses, 
             std::vector<std::vector<std::string>>& rString);
    void cuRun();

#elif USE_OPENCL

// opencl implementation
    void clParallelKdotQs(int& promptCount, int& currentTokenCount, int& blockCount, int& column, bool& isSelf, bool& inTraining);
    void clForward(int& blockCount, int& currentTokenCount, int& promptCount);
    void clBackward(std::vector<float>& expectedH, int& blockCount);
    void clBackward(std::vector<std::vector<float>>& expectedH, int& blockCount);
    void clBackwardContext(std::vector<std::vector<float>>& expectedH, int& blockCount);
    void clTrain(int& promptCount, int& currentTokenCount, int& blockCount, std::vector<float>& expected, std::string& rString);
    void clTrain(std::vector<std::vector<float>>& sentence, std::vector<std::string>& rString);
    void clTrain(std::vector<std::vector<float>>& prompt, std::vector<std::vector<float>>& response, std::vector<std::string>& rString);
    void clTrain(std::vector<std::vector<std::vector<float>>>& prompts, std::vector<std::vector<std::vector<float>>>& responses, 
                std::vector<std::vector<std::string>>& rString);
    void clTrainContext(int& promptCount, int& currentTokenCount, int& blockCount, std::vector<float>& expected, std::string& rString);
    void clTrainContext(std::vector<std::vector<float>>& sentence, std::vector<std::string>& rString);
    void clTrainContext(std::vector<std::vector<float>>& prompt, std::vector<std::vector<float>>& response, std::vector<std::string>& rString);
    void clTrainContext(std::vector<std::vector<std::vector<float>>>& prompts, std::vector<std::vector<std::vector<float>>>& responses, 
                std::vector<std::vector<std::string>>& rString);
    void clUpdateEmbeddings(std::vector<float>& tokenEmbedding, float learning, float lambda_L1, float lambda_L2,
                            std::vector<float> &gradForEh);
    void clUpdateDeEmbeddings(mat& deEmbeddings, std::vector<float> logit, std::vector<float> calculatedToken,
                            std::vector<float> expected_one_hot_host, int indexForToken, float learning, float lambda_L1,
                            float lambda_L2, std::vector<float> &gradForEh);
    void clTest(std::vector<std::vector<float>>& prompt, std::vector<std::vector<float>>& response, std::vector<std::string>& rString);
    void clTest(std::vector<std::vector<std::vector<float>>>& prompts, std::vector<std::vector<std::vector<float>>>& responses, 
             std::vector<std::vector<std::string>>& rString);
    void clRun();

#else

// c++ implementation
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
    void test(std::vector<std::vector<float>>& prompt, std::vector<std::vector<float>>& response, std::vector<std::string>& rString);
    void test(std::vector<std::vector<std::vector<float>>>& prompts, std::vector<std::vector<std::vector<float>>>& responses, 
                std::vector<std::vector<std::string>>& rString);
    void run();

#endif

    float adaptiveLearningOptimiser(float prev_Error, float current_Error, float learning, int epochCount);
    void setDims(int m, int x, int y, int n, int d, int h, int l);      // set dimension of transformer
    void setLearning(float learning);           // set learning rate for MLPs
    void setEpochs(int epochs);                 // set epochs for MLPs
    void setAttention(bool attentionType);      // set self attention (1) or cross attention (0)
    void getAllValues(int blockCount, std::string path2folderOfAllBins, bool& inTraining);
    void getcache(int blockCount, int i, int j, mat& q, std::string path2file);
    void getmat(int blockCount, int i, int j, mat& q, std::string path2file, int& row, int& column);
    void getmlp(int blockCount, int i, int j, std::vector<mat>& weights, std::string path2file);
    void getEmbedding(std::string& word, std::vector<float>& embed);
    void updateLearning(float& pErr, float& cErr);
    void makeCommon(std::string& path2folderOfAllBins);
    void clearValues();

    ~transformer() = default;
};

std::string toLower(const std::string& str);

// compute functions for dot, KdotQ and other values

void computeOutput(std::vector<float>& output, std::vector<std::vector<float>>& embeddings, long long int& voc, int& index);
void computeOutput(const std::vector<float>& output, mat& embeddings, long long int& voc, int& index);
void computeKorQ(std::vector<float>& tokenEmmbed, mat& m, std::vector<float>& KorQ);
void computeDot(std::vector<float>& T1, std::vector<float>& T2, std::vector<std::vector<float>>& M, float& dot);
void computeDot(std::vector<float>& Ti, mat& M, std::vector<float>& Tj, float& dot);
void computeKdotQ(std::vector<std::vector<float>>& KdotQ, std::vector<std::vector<float>>& K, std::vector<std::vector<float>>& Q, 
    int& currentTokenCount, int& promptCount, int& blockCount, bool& attentionType);
void computeKdotQ(std::vector<std::vector<float>>& KdotQ, std::vector<std::vector<float>>& tokenEmbed, mat& M, int& currentTokenCount,
    int& promptCount, bool& attentionType);
void computeKdotQ(std::vector<std::vector<float>>& KdotQ, std::vector<std::vector<float>>& tokenEmbed, std::vector<std::vector<float>>& EVp,
    mat& M, int& currentTokenCount, int& promptCount, int& blockCount, bool& attentionType);

#ifdef USE_CUDA
    void cuComputeOutput(float* d_output, float* d_embeddings, int voc_size, int& index, int embedding_dim);
    __global__ void accumulateEH(float** d_eh_pointers, float* d_otok, int num_layers, int embedding_dim);
    __global__ void computeAllDotsKernel(const float* vector, const float* matrix, float* results, int num_rows, int vector_dim);
#endif

#endif
