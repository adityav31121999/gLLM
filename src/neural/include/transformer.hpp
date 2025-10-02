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
    bool isSelf;                // if self attention or cross attention
    bool inTraining;            // = 1 for training, = 0 for in use
    bool contextTrain;          // contextualised training = 1 else static training
    int m;                      // number of blocks
    int y;                      // number of incomplete attentions in each partial attention
    int x;                      // number of layers of partial attention for complete attention block
    int n;                      // total tokens for each attention head or context window
    int d;                      // token dimension
    int h;                      // height of MQ, MK and columns of MV, MH
    int l;                      // layers of mlp
    int epochs;                 // number of epochs for MLPs and Blocks

    unsigned int vocabsize;             // size of vocabulary
    unsigned long long params;          // parameters in transformer
    unsigned long long trainCount;      // total training count
    unsigned long long cacheOffset;     // for extracting caches
    unsigned long long matOffset;       // for extracting matrices
    unsigned long long mlpOffset;       // for extracting MLPs

    int blockCount;             // which block is working
    int epochCount;             // epoch counter
    int sequence1Count;            // number of tokens in the sequence1
    int currentTokenCount;      // current count of tokens in full context
    int indexForToken;          // this is to set token index from embedding list
    int resCount;               // sequence2 count for every sequence1
    int plateau_count;          // counter for learning rate on plateau
    int plateau_patience;       // patience for learning rate on plateau

    float lambda_L1;            // lambda for L1 regularisation
    float lambda_L2;            // lambda for L2 regularisation
    float learning;             // learning rate
    float error;                // error for transformer (after complete trainin)
    float perplexityCE;         // currentc CE perplexity
    float perplexityBCE;        // current BCE perplexity

    double totalBCELoss;        // total BCE error for all updates (adaptive learning)
    double totalCELoss;         // total CE error for all updates (adaptive learning)
    double totalLearning;       // total learning for all updates (adaptive learning)
    double totalCEPerplexity;   // total CE perplexity for all updates (adaptive learning)
    double totalBCEPerplexity;  // total BCE perplexity for all updates (adaptive learning)

    float avgBCELoss;           // average bce
    float avgCELoss;            // average ce
    float averagePerplexity;    // average perplexity throughout training
    float averageLearningRate;  // average learning rate throughout training

    std::vector<block> blocks;          // attention block ('1' for inference and 'm' for training)
    std::vector<std::string> tokens;    // tokens in vocabulary
    std::vector<std::string> mTokens;   // sequence1s and sequence2 tokens
    std::vector<int> indexVec;          // indices vector
    std::vector<int> expIndex;          // expected index vector
    std::vector<float> otok;            // output token (vector, size d * NUMBER_OF_HEADS)
    std::vector<float> pred;            // prediction from forprop
    std::vector<float> oneHotEncode;    // one hot encoding with 1 at predicted token index and rest are set to 0
    mat embeddings;                     // all trained embeddings (Mapped, vocabsize x d)
    mat deEmbeddings;                   // deEmbeddings obtained while training
    mat tokenEmbed;                     // token embedding (sequence1 + sequence2) (Mapped, currentTokenCount x d)
    FILE* seqChat;              // sequence1 and sequence2 text file
    // when model is in inference, hold EV of ith block here
    std::vector<std::vector<std::vector<std::vector<float>>>> EVuse; // Keeping as vector due to complexity
    mat tokForBlock;                    // token embeddings for local context for inference (Mapped, n x d)

#ifdef USE_OPENCL
    OpenCLContext& clcontext;
    transformer(OpenCLContext& context, int m_param, int x_param, int y_param, int n_param, int d_param, int h_param, int l_param, 
        unsigned int vocab_param, float learning_rate_param, float lambda_L1_param, float lambda_L2_param, bool attentionType_param, 
        bool& inTraining_param, bool& contextTrain_param, const std::string& modelDir_param);
    transformer(OpenCLContext& context, const std::string& ModelName, int m_param, int x_param, int y_param, int n_param, int d_param, int h_param, int l_param, 
        unsigned int vocab_param, float learning_rate_param, float lambda_L1_param, float lambda_L2_param, bool attentionType_param, 
        bool& inTraining_param, bool& contextTrain_param, const std::string& modelDir_param);
#elif USE_CUDA || USE_CPU
    transformer() = default;
    transformer(int m_param, int x_param, int y_param, int n_param, int d_param, int h_param, int l_param, unsigned int vocab_param, 
        float learning_rate_param, float lambda_L1_param, float lambda_L2_param, bool attentionType_param, bool& inTraining_param, 
        bool& contextTrain_param, const std::string& blockBinPath);
    transformer(const std::string& ModelName, int m_param, int x_param, int y_param, int n_param, int d_param, int h_param, int l_param, unsigned int vocab_param, 
        float learning_rate_param, float lambda_L1_param, float lambda_L2_param, bool attentionType_param, bool& inTraining_param, 
        bool& contextTrain_param, const std::string& blockBinPath);
#endif

#ifdef USE_CUDA

// cuda implementation
    void cuParallelKdotQs(int& sequence1Count, int& currentTokenCount, int& blockCount, int& column, bool& isSelf, bool& inTraining);
    void cuForward(int& blockCount, int& currentTokenCount, int& sequence1Count);
    void cuBackward(std::vector<float>& expectedH, int& blockCount);
    void cuBackward(std::vector<std::vector<float>>& expectedH, int& blockCount);
    void cuBackwardContext(std::vector<std::vector<float>>& expectedH, int& blockCount);
    void cuUpdateEmbeddings(mat tokenEmbedding, std::vector<float>& gradForEh, float learning, float lambda_L1, float lambda_L2, int rows);
    void cuUpdateDeEmbeddings(mat& deEmbeddings, std::vector<float> logit, std::vector<float> calculatedToken, std::vector<float> one_hot_host,
                            int indexForToken, float learning, float lambda_L1, float lambda_L2, std::vector<float> &gradForEh);
    void cuTrain(std::vector<std::vector<float>>& sentence, std::vector<std::string>& rString);
    void cuTrain(std::vector<std::vector<float>>& sequence1, std::vector<std::vector<float>>& sequence2, std::vector<std::string>& rString);
    void cuTrainContext(std::vector<std::vector<float>>& sentence, std::vector<std::string>& rString);
    void cuTrainContext(std::vector<std::vector<float>>& sequence1, std::vector<std::vector<float>>& sequence2, std::vector<std::string>& rString);
    void cuBufferTrain(std::vector<std::vector<float>> &sentence, std::vector<std::string> &rString);
    void cuBufferTrain(std::vector<std::vector<float>>& sequence1, std::vector<std::vector<float>>& sequence2, std::vector<std::string>& rString);
    void cuBufferTrainContext(std::vector<std::vector<float>> &sentence, std::vector<std::string> &rString);
    void cuBufferTrainContext(std::vector<std::vector<float>>& sequence1, std::vector<std::vector<float>>& sequence2, std::vector<std::string>& rString);
    void cuTest(std::vector<std::vector<float>>& sequence1, std::vector<std::vector<float>>& sequence2, std::vector<std::string>& rString);
    void cuBufferTest(std::vector<std::vector<float>>& sequence1, std::vector<std::vector<float>>& sequence2, std::vector<std::string>& rString);
    void cuRun();
    void cuBufferRun();

#elif USE_OPENCL

// opencl implementation
    void clParallelKdotQs(int& sequence1Count, int& currentTokenCount, int& blockCount, int& column, bool& isSelf, bool& inTraining);
    void clForward(int& blockCount, int& currentTokenCount, int& sequence1Count);
    void clBackward(std::vector<float>& expectedH, int& blockCount);
    void clBackward(std::vector<std::vector<float>>& expectedH, int& blockCount);
    void clUpdateEmbeddings(mat tokenEmbedding, std::vector<float>& gradForEh, float learning, float lambda_L1, float lambda_L2, int rows);
    void clUpdateDeEmbeddings(mat& deEmbeddings, std::vector<float> logit, std::vector<float> calculatedToken, std::vector<float> one_hot_host,
                            int indexForToken, float learning, float lambda_L1, float lambda_L2, std::vector<float> &gradForEh);
    void clBackwardContext(std::vector<std::vector<float>>& expectedH, int& blockCount);
    void clTrain(std::vector<std::vector<float>>& sentence, std::vector<std::string>& rString);
    void clTrain(std::vector<std::vector<float>>& sequence1, std::vector<std::vector<float>>& sequence2, std::vector<std::string>& rString);
    void clTrainContext(std::vector<std::vector<float>>& sentence, std::vector<std::string>& rString);
    void clTrainContext(std::vector<std::vector<float>>& sequence1, std::vector<std::vector<float>>& sequence2, std::vector<std::string>& rString);
    void clBufferTrain(std::vector<std::vector<float>> &sentence, std::vector<std::string> &rString);
    void clBufferTrain(std::vector<std::vector<float>>& sequence1, std::vector<std::vector<float>>& sequence2, std::vector<std::string>& rString);
    void clBufferTrainContext(std::vector<std::vector<float>> &sentence, std::vector<std::string> &rString);
    void clBufferTrainContext(std::vector<std::vector<float>>& sequence1, std::vector<std::vector<float>>& sequence2, std::vector<std::string>& rString);
    void clTest(std::vector<std::vector<float>>& sequence1, std::vector<std::vector<float>>& sequence2, std::vector<std::string>& rString);
    void clBufferTest(std::vector<std::vector<float>>& sequence1, std::vector<std::vector<float>>& sequence2, std::vector<std::string>& rString);
    void clRun();
    void clRunContext();
    void clBufferRun();
    void clBufferRunContext();

#else

// c++ implementation
    void parallelKdotQs(int& sequence1Count, int& currentTokenCount, int& blockCount, int& column, bool& isSelf, bool& inTraining);
    void computeKdotQs(int& sequence1Count, int& currentTokenCount, int& blockCount, bool& isSelf, bool& inTraining);
    void forward(int& blockCount, int& currentTokenCount, int& sequence1Count);
    void backward(std::vector<float>& expectedH);
    void backward(std::vector<float>& expectedH, int& blockCount);
    void backward(std::vector<std::vector<float>>& expectedH);
    void backward(std::vector<std::vector<float>>& expectedH, int& blockCount);
    void train(std::vector<std::vector<float>>& sentence, std::vector<std::string>& rString);
    void train(std::vector<std::vector<float>>& sequence1, std::vector<std::vector<float>>& sequence2, std::vector<std::string>& rString);
    void trainOnThreads(std::vector<std::vector<float>>& sequence1, std::vector<std::vector<float>>& sequence2, std::vector<std::string>& rString);
    void test(std::vector<std::vector<float>>& sequence1, std::vector<std::vector<float>>& sequence2, std::vector<std::string>& rString);
    void testOnThreads(std::vector<std::vector<float>>& sequence1, std::vector<std::vector<float>>& sequence2, std::vector<std::string>& rString);
    void run();
    void runOnThreads();

#endif

    float periodicLearning(float x, float u, float v, float value_at_0);
    float cosineAnnealingLR(int current_epoch, int total_epochs, float max_lr, float min_lr);
    float adaptiveLearningRateOnPlateau(float current_error, float previous_error, float& learning_rate, 
                                        float factor, float max_lr, float min_lr, float epsilon, int patience);
    float adaptiveLearningRate(float current_error, float prev_error, int epochs, float current_lr, float min_lr = 1e-5, float max_lr = 0.1);
    float softsignLearning(float errordif, float currentLearning);
    float softsignLearning(float errordif, float currentLearning, float initialLearning);
    float smoothLearningRate(float current_error, float prev_error, float current_lr, float initial_lr, int epochs);
    void setDims(int m, int x, int y, int n, int d, int h, int l);
    void setLearning(float learning);
    void setEpochs(int epochs);
    void getIndexOfAllTokens(std::vector<std::string>& tokensOfLine, std::vector<int>& indexVec);
    void setAttention(bool attentionType);
    void getAllValues(int blockCount, std::string path2folderOfAllBins, bool& inTraining);
    void getcache(int blockCount, int i, int j, mat& q, std::string path2file);
    void getmat(int blockCount, int i, int j, mat& q, std::string path2file, int& row, int& column);
    void getmlp(int blockCount, int i, int j, std::vector<mat>& weights, std::string path2file);
    void getEmbedding(std::string& word, std::vector<float>& embed);
    std::vector<float> positionalEmbeddings(int position, int embeddingDimension);
    void makeCommon(std::string& path2folderOfAllBins);
    void clearValues();

    ~transformer() = default;
};

std::string toLower(const std::string& str);

// compute functions for dot, KdotQ and other values

void computeOutput(std::vector<float>& output, std::vector<std::vector<float>>& embeddings, unsigned long long& voc, int& index);
void computeOutput(const std::vector<float>& output, mat& embeddings, unsigned long long& voc, int& index);
void computeKorQ(std::vector<float>& tokenEmmbed, mat& m, std::vector<float>& KorQ);
void computeDot(std::vector<float>& T1, std::vector<float>& T2, std::vector<std::vector<float>>& M, float& dot);
void computeDot(std::vector<float>& Ti, mat& M, std::vector<float>& Tj, float& dot);
void computeKdotQ(std::vector<std::vector<float>>& KdotQ, std::vector<std::vector<float>>& K, std::vector<std::vector<float>>& Q, 
    int& currentTokenCount, int& sequence1Count, int& blockCount, bool& attentionType);
void computeKdotQ(std::vector<std::vector<float>>& KdotQ, std::vector<std::vector<float>>& tokenEmbed, mat& M, int& currentTokenCount,
    int& sequence1Count, bool& attentionType);
void computeKdotQ(std::vector<std::vector<float>>& KdotQ, std::vector<std::vector<float>>& tokenEmbed, std::vector<std::vector<float>>& EVp,
    mat& M, int& currentTokenCount, int& sequence1Count, int& blockCount, bool& attentionType);

#endif