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
#include "block.hpp"

/**
 * @brief Transformer (FULL CONTEXT) class for token/chunk prediction and context 
 * retention. This can be single or multi-block architecture based on use case i.e.,
 * single block for infernce and multi-block for training.
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
    int expectedIndex;      // expected index
    int resCount;           // response count for every prompt
    float learning;         // learning rate for MLPs
    double totalLearning;   // total learning for all updates (adaptive learning)
    float lambda_L1;        // lambda for L1 penalty
    float lambda_L2;        // lambda for L2 penalty
    float error;            // error for transformer (after complete trainin)
    bool isTerminate;       // when '@#0' is calculated, to end the forward propagation
    unsigned long long params;              // parameters in transformer
    unsigned long long trainCount;          // total training count
    unsigned long long vocabsize;           // size of vocabulary
    unsigned long long cacheOffset;         // for extracting caches
    unsigned long long matOffset;           // for extracting matrices
    unsigned long long mlpOffset;           // for extracting MLPs*

    // Adam Optimizer Hyperparameters and State
    float clip_norm;            // gradient clipping norm
    float beta1;                // decay rate for first moment
    float beta2;                // decay rate for second moment
    float epsilon;              // small value to prevent division by zero
    unsigned long long t_step_adam;         // Global Adam time step (number of updates), initialized to 0. Use unsigned long long for potentially many updates

    // Parameters for ReduceLROnPlateau
    float best_loss_for_lr_schedule;        // Stores the lowest loss seen so far for LR adjustments
    int lr_patience_counter;                // Counts epochs without improvement
    int LR_PATIENCE_CONFIG = 10;            // Increased patience (e.g., 10-20)
    float LR_DECAY_FACTOR_CONFIG = 0.5f;    // Halve the LR (common and gentler)
    float LR_MIN_DELTA_CONFIG = 1e-3f;      // More significant improvement required (e.g., 0.001 to 0.01)

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
    std::vector<std::vector<float>> gradForTokenEmbed;          // gradients for token embeddings in use

#ifdef USE_OPENCL
    OpenCLContext& clcontext;
    // Constructor with explicit learning rate
    transformer(OpenCLContext& context_param, int m_param, int x_param, int y_param, int n_param, 
        int d_param, int h_param, int l_param, unsigned long long vocab_param, float learning_rate_param, 
        float lambda_L1_param, float lambda_L2_param, bool attentionType_param, bool& inTraining_param, 
        int epoch, const std::string& blockBinPath);
#elif USE_CUDA || USE_CPU
    transformer() = default;
    // Constructor with explicit learning rate
    transformer(int m_param, int x_param, int y_param, int n_param, int d_param, int h_param, int l_param, 
        unsigned long long vocab_param, float learning_rate_param, float lambda_L1_param, float lambda_L2_param, 
        bool attentionType_param, bool& inTraining_param, int epoch, const std::string& blockBinPath);
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
    void cuUpdateDeEmbeddings(mat& deEmbeddings, std::vector<float> logit, std::vector<float> calculatedToken,
                                std::vector<float> expected_one_hot_host, float learning, float lambda_L1,
                                float lambda_L2, float clip_norm, std::vector<float> &gradForEh);
    void cuTest(std::vector<std::vector<float>>& prompt, std::vector<std::vector<float>>& response, std::vector<std::string>& rString);
    void cuTest(std::vector<std::vector<std::vector<float>>>& prompts, std::vector<std::vector<std::vector<float>>>& responses, 
             std::vector<std::vector<std::string>>& rString);
    void cuRun();

#elif USE_OPENCL
// opencl implementation
    void clParallelKdotQs(int& promptCount, int& currentTokenCount, int& blockCount, int& column, bool& isSelf, bool& inTraining);
    void clForward(int& blockCount, int& currentTokenCount, int& promptCount);
    void clBackward(std::vector<float>& expectedH, float& clip_norm);
    void clBackward(std::vector<float>& expectedH, int& blockCount, float& clip_norm);
    void clBackward(std::vector<std::vector<float>>& expectedH, float& clip_norm);
    void clBackward(std::vector<std::vector<float>>& expectedH, int& blockCount, float& clip_norm);
    void clTrain(int& promptCount, int& currentTokenCount, int& blockCount, std::vector<float>& expected, std::string& rString);
    void clTrain(std::vector<std::vector<float>>& sentence, std::vector<std::string>& rString);
    void clTrain(std::vector<std::vector<float>>& prompt, std::vector<std::vector<float>>& response, std::vector<std::string>& rString);
    void clTrain(std::vector<std::vector<std::vector<float>>>& prompts, std::vector<std::vector<std::vector<float>>>& responses, 
                std::vector<std::vector<std::string>>& rString);
    void clUpdateEmbeddings(mat& tokenEmbed, std::vector<float> logit, std::vector<float> calculatedToken,
                            std::vector<std::vector<float>> gradForTokenEmbed, float learning, float lambda_L1, float lambda_L2,
                            float clip_norm);
    void clUpdateDeEmbeddings(mat& deEmbeddings, std::vector<float> logit, std::vector<float> calculatedToken,
                                std::vector<float> expected_one_hot_host, int indexForToken, float learning, float lambda_L1,
                                float lambda_L2, float clip_norm, std::vector<float> &gradForEh);
    void clTest(std::vector<std::vector<float>>& prompt, std::vector<std::vector<float>>& response, std::vector<std::string>& rString);
    void clTest(std::vector<std::vector<std::vector<float>>>& prompts, std::vector<std::vector<std::vector<float>>>& responses, 
             std::vector<std::vector<std::string>>& rString);
    void clRun();

#else

    void parallelKdotQs(int &promptCount, int &currentTokenCount, int &blockCount, int &column, bool &isSelf, bool &inTraining);
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
    void updateDeEmbeddings(mat& deEmbeddings, std::vector<float> logit, std::vector<float> calculatedToken,
                                std::vector<float> expected_one_hot_host, float learning, float lambda_L1,
                                float lambda_L2, float clip_norm, std::vector<float> &gradForEh);
    void test(std::vector<std::vector<float>>& prompt, std::vector<std::vector<float>>& response, std::vector<std::string>& rString);
    void test(std::vector<std::vector<std::vector<float>>>& prompts, std::vector<std::vector<std::vector<float>>>& responses, 
                std::vector<std::vector<std::string>>& rString);
    void run();

#endif

    float adaptiveLearningOptimiser(float prev_Error, float current_Error, float learning, int epochCount);
    void adjustLearningRateOnPlateau(float current_loss);
    void setDims(int m, int x, int y, int n, int d, int h, int l);      // set dimension of transformer
    void setLearning(float learning);           // set learning rate for MLPs
    void setEpochs(int epochs);                 // set epochs for MLPs
    void setAttention(bool attentionType);      // set self attention (1) or cross attention (0)
    void getAllValues(int blockCount, std::string path2folderOfAllBins, bool& inTraining);
    void getcache(int blockCount, int i, int j, mat& q, std::string path2file);
    void getmat(int blockCount, int i, int j, mat& q, std::string path2file, int& row, int& column);
    void getmlp(int blockCount, int i, int j, std::vector<mat>& weights, std::string path2file);
    void getEmbedding(std::string& word, std::vector<float>& embed);
    void makeCommon(std::string& path2folderOfAllBins);
    void clearValues();

    ~transformer() = default;
};

std::string toLower(const std::string& str);

// compute functions for dot, KdotQ and other values

void computeOutput(const std::vector<float>& output, const mat& embeddings, unsigned long long& voc, int& index);
void computeKorQ(const std::vector<float>& tokenEmbed, const mat& m, std::vector<float>& KorQ);
void computeDot(std::vector<float>& T1, std::vector<float>& T2, std::vector<std::vector<float>>& M, float& dot);
void computeDot(const std::vector<float>& Ti, const mat& M, const std::vector<float>& Tj, float& dot);
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
