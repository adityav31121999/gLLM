
// transformer.hpp: header source for transformer class
#ifndef TRANSFORMER_HPP
#define TRANSFORMER_HPP 1

#include <string>
#include <cmath>
#include <vector>
#include <iostream>
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

// these are variables that change during training
    int blockCount;         // which block is working
    int epochCount;         // epoch counter
    int promptCount;        // number of tokens in the prompt
    int currentTokenCount;  // current count of tokens in full context

// these are variables that change during runtime
    float error;            // error for transformer
    int trainCount;         // total training count
    bool isTerminate;       // when '@#0' is calculated, to end the forward propagation

    std::vector<block> t;               // attention block (1 or many)
    std::vector<std::vector<float>> tokenEmbed;         // token embedding
    std::vector<std::vector<float>> input;              // input embeddings
    std::vector<std::vector<float>> output;             // output embeddings
    std::vector<std::vector<float>> embeddings;         // all glove embeddings with 64D
    std::vector<std::vector<float>> tokForBlock;        // tokens for kth block for KdotQ
    FILE* promptNresponse;  // prompt and response text file
    // when model is in training, hold EV of all the blocks here
    std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>> EVs;
    // for use in calculating next tokens via next block
    std::vector<std::vector<std::vector<std::vector<float>>>> EVUse;

    // default constructor
    transformer() = default;
    transformer(int x, int y, int n, int d, int h, int l, int vocab);
    transformer(int x, int y, int n, int d, int h, int l, int vocab, bool attentionType);
    transformer(int m, int x, int y, int n, int d, int h, int l, int vocab);
    transformer(int m, int x, int y, int n, int d, int h, int l, int vocab, bool attentionType);
    transformer(int m, int x, int y, int n, int d, int h, int l, int vocab, bool attentionType, bool& inTraining);

    void setDims(int m, int x, int y, int n, int d, int h, int l);      // set dimension of transformer
    void setLearning(float learning);       // set learning rate for MLPs
    void setEpochs(int epochs);             // set epochs for MLPs
    void setAttention(bool attentionType);  // set self attention (1) or cross attention (0)
    void setInput(std::vector<std::vector<float>>& input);      // set input embeddings for running transformer
    void setOutput(std::vector<std::vector<float>>& output);    // set output embeddings for instructing and finetuning
    void setEmbedding(std::string& embedding);      // set token embeddings

    void computeKdotQs(int& promptCount, int& currentTokenCount, int& blockCount, bool& isSelf);
    void forward(int& blockCount, int& currentTokenCount, int& promptCount);
    void backward(std::vector<float>& expectedH);
    void backward(std::vector<float>& expectedH, int& blockCount);
    void backward(std::vector<std::vector<float>>& expectedH);
    void backward(std::vector<std::vector<float>>& expectedH, int& blockCount);
    void train(int& promptCount, int& currentTokenCount, int& blockCount, bool& isSelf, std::vector<float>& expected);
    void train(int& promptCount, int& currentTokenCount, int& blockCount, bool& isSelf, std::vector<std::vector<float>>& expected);
    void train(std::vector<std::vector<float>>& sentence);
    void train(std::vector<std::vector<std::vector<float>>>& sentences);
    void train(std::vector<std::vector<float>>& prompt, std::vector<std::vector<float>>& response);
    void train(std::vector<std::vector<std::vector<float>>>& prompts, std::vector<std::vector<std::vector<float>>>& responses);
    void computeOutput(std::vector<float>& output, std::vector<std::vector<float>>& embeddings, int& voc, int& index);
    void run();

#ifdef USE_CUDA     // cuda implementation
    void cuParallelKdotQs(int& promptCount, int& currentTokenCount, int& blockCount, bool& isSelf);
    void cuForward();
    void cuBackward(std::vector<float>& expectedH);
    void cuBackward(std::vector<float>& expectedH, int& blockCount);
    void cuBackward(std::vector<std::vector<float>>& expectedH);
    void cuBackward(std::vector<std::vector<float>>& expectedH, int& blockCount);
    void cuBackward(std::vector<std::vector<std::vector<float>>>& expectedH);
    void cuBackward(std::vector<std::vector<std::vector<float>>>& expectedH, int& blockCount);
    void cuTrain(int& promptCount, int& currentTokenCount, int& blockCount, bool& isSelf, std::vector<float>& expected);
    void cuTrain(int& promptCount, int& currentTokenCount, int& blockCount, bool& isSelf, std::vector<std::vector<float>>& expected);
    void cuTrain(std::vector<std::vector<float>>& sentence);
    void cuTrain(std::vector<std::vector<std::vector<float>>>& sentences);
    void cuTrain(std::vector<std::vector<float>>& prompt, std::vector<std::vector<float>>& response);
    void cuTrain(std::vector<std::vector<std::vector<float>>>& prompts, std::vector<std::vector<std::vector<float>>>& responses);
    void cuComputeOutput(std::vector<float>& output, std::vector<std::vector<float>>& embeddings, int& voc, int& index);
    void cuRun();
#elif USE_OPENCL    // opencl implementation
    void clParallelKdotQs(int& promptCount, int& currentTokenCount, int& blockCount, bool& isSelf);
    void clForward();
    void clBackward(std::vector<float>& expectedH);
    void clBackward(std::vector<float>& expectedH, int& blockCount);
    void clBackward(std::vector<std::vector<float>>& expectedH);
    void clBackward(std::vector<std::vector<float>>& expectedH, int& blockCount);
    void clBackward(std::vector<std::vector<std::vector<float>>>& expectedH);
    void clBackward(std::vector<std::vector<std::vector<float>>>& expectedH, int& blockCount);
    void clTrain(int& promptCount, int& currentTokenCount, int& blockCount, bool& isSelf, std::vector<float>& expected);
    void clTrain(int& promptCount, int& currentTokenCount, int& blockCount, bool& isSelf, std::vector<std::vector<float>>& expected);
    void clTrain(std::vector<std::vector<float>>& sentence);
    void clTrain(std::vector<std::vector<std::vector<float>>>& sentences);
    void clTrain(std::vector<std::vector<float>>& prompt, std::vector<std::vector<float>>& response);
    void clTrain(std::vector<std::vector<std::vector<float>>>& prompts, std::vector<std::vector<std::vector<float>>>& responses);
    void clComputeOutput(std::vector<float>& output, std::vector<std::vector<float>>& embeddings, int& voc, int& index);
    void clRun();
#endif

    // default destructor
    ~transformer() = default;
};


// compute functions for dot, KdotQ and other values

void computeKQ(std::vector<float>& tokenEmmbed, mat& m, std::vector<float>& KorQ);
void computeDot(std::vector<float>& T1, std::vector<float>& T2, std::vector<std::vector<float>>& M, float& dot);
void computeDot(std::vector<float>& Ti, mat M, std::vector<float>& Tj, float& dot);
void computeKdotQ(std::vector<std::vector<float>>& KdotQ, std::vector<std::vector<float>>& K, std::vector<std::vector<float>>& Q, 
    int& currentTokenCount, int& promptCount, int& blockCount, bool& attentionType);
void computeKdotQ(std::vector<std::vector<float>>& KdotQ, std::vector<std::vector<float>>& tokenEmbed, mat M, int& currentTokenCount,
    int& promptCount, bool& attentionType);
void computeKdotQ(std::vector<std::vector<float>>& KdotQ, std::vector<std::vector<float>>& tokenEmbed, std::vector<std::vector<float>>& EVp,
    mat M, int& currentTokenCount, int& promptCount, int& blockCount, bool& attentionType);

#ifdef USE_CUDA
    // cuda implementation
    __device__ float compute_dot_product(const float* vec1, const float* vec2, int dim);
    __device__ float compute_dot_product(const float* vec1, const float* vec2, const float* matrix, int dim);
    __device__ int compute_prediction(const float* EH, const float* embeddings, int dim, int voc);
    __global__ void kernelKdotQforSelf_train(float* d_kdotq, const float* d_keys, const float* d_querys,
                int num_queries_eff, int num_keys_eff, int kdotq_width, int embedding_dim, float inv_scaling);
    __global__ void kernelKdotQforCross_train(float* d_kdotq, const float* d_keys, const float* d_querys,
                int num_queries_eff, int num_keys_eff, int kdotq_width, int embedding_dim, float inv_scaling);
    __global__ void kernelKdotQ_Block1_Self_Inference(float* d_kdotq, const float* d_tokenEmbed, const float* d_M, 
                int prompt_start_index, int prompt_len, int context_len, int kdotq_width, int embedding_dim, 
                float inv_scaling);
    __global__ void kernelKdotQ_Block1_Cross_Inference(float* d_kdotq, const float* d_tokenEmbed, const float* d_M, 
                int prompt_start_index, int prompt_len, int context_len, int kdotq_width, int embedding_dim, 
                float inv_scaling);
    __global__ void kernelKdotQ_BlockN_Self_Inference(float* d_kdotq, const float* d_tokForBlock, const float* d_EVp, 
                const float* d_M, int prompt_start_index_in_block, int prompt_len, int context_len_in_block, 
                int kdotq_width, int embedding_dim, float inv_scaling);
    __global__ void kernelKdotQ_BlockN_Cross_Inference(float* d_kdotq, const float* d_tokForBlock, const float* d_EVp,
                const float* d_M, int prompt_start_index_in_block, int prompt_len, int context_len_in_block, 
                int kdotq_width, int embedding_dim, float inv_scaling);    
#elif USE_OPENCL
    // opencl implementation
    float compute_dot_product(__global const float* vec1, __global const float* vec2, int dim);
    float compute_dot_product(__global const float* vec1, __global const float* vec2, __global const float* matrix, int dim);
    int compute_prediction(__global const float* EH, __global const float* embeddings, int dim, int voc);
    __kernel void kernelKdotQforSelf_train(__global float* d_kdotq, __global const float* d_keys, __global const float* d_querys,
        int num_queries_eff, int num_keys_eff, int kdotq_width, int embedding_dim, float inv_scaling);
    __kernel void kernelKdotQforCross_train(__global float* d_kdotq, __global const float* d_keys, __global const float* d_querys, 
        int num_queries_eff, int num_keys_eff, int kdotq_width, int embedding_dim, float inv_scaling);
    __kernel void kernelKdotQBlock1Self_Inference(__global float* d_kdotq, __global const float* d_tokenEmbed, __global const float* d_M,
        int prompt_start_index, int prompt_len, int context_len, int kdotq_width,int embedding_dim, float inv_scaling);
    __kernel void kernelKdotQBlock1Cross_Inference(__global float* d_kdotq, __global const float* d_tokenEmbed, __global const float* d_M,
        int prompt_start_index, int prompt_len, int context_len, int kdotq_width, int embedding_dim, float inv_scaling);
    __kernel void kernelKdotQBlockNSelf_Inference(__global float* d_kdotq, __global const float* d_tokForBlock, __global const float* d_EVp,
        __global const float* d_M, int prompt_start_index_in_block, int prompt_len, int context_len_in_block,
        int kdotq_width, int embedding_dim, float inv_scaling);
    __kernel void kernelKdotQBlockNCross_Inference(__global float* d_kdotq, __global const float* d_tokForBlock, __global const float* d_EVp,
        __global const float* d_M, int prompt_start_index_in_block, int prompt_len, int context_len_in_block,
        int kdotq_width, int embedding_dim, float inv_scaling);

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
