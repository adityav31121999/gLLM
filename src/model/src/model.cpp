// model.cpp: implementation of Model class
#include "include/model.hpp"
#include <maths.hpp>
#include <iostream>
#include <stdexcept>
#include <filesystem>
#include <fstream>
#include <locale>
#include <sys/stat.h>
#ifdef USE_OPENCL

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
model::model(OpenCLContext& context, const std::string& baseDirectory, int m_param, int x_param, int y_param, int n_param, int d_param, 
    int matheight_param, int l_param, float learning_param, float lambda_L1, float lambda_L2, bool isSelfAttention_param, 
    bool toTrainModel_param, const std::string& tokeniserPath) :
    baseDir(baseDirectory), clcontext(context), m(toTrainModel_param ? m_param : 1), x(x_param), y(y_param), n(n_param), 
    d(d_param), matheight(matheight_param), l(l_param), learning(learning_param), lambda_L1(lambda_L1), lambda_L2(lambda_L2), 
    total(m * n), isSelf(isSelfAttention_param), toTrain(toTrainModel_param), metadata(nullptr), chat(nullptr), currentChatLogPath(""), 
    TOK(tokeniserPath, context),
    T(context, this->m, x_param, y_param, n_param, d_param, matheight_param, l_param, TOK.getVocabularySize(),
        this->learning, this->lambda_L1, this->lambda_L2, this->isSelf, this->toTrain, baseDirectory)
{
    total = this->m * this->n;
    info = {}; // Zero-initialize info struct
    info.vocab = vocabsize;
    info.m = this->m;
    info.x = this->x;
    info.y = this->y;
    info.n = this->n;
    info.d = this->d;
    info.h = this->matheight;
    info.l = this->l;
    info.learning = this->learning;
    info.attentionMech = MECH;
    info.modelArch = ARCH;
    info.matheight = this->matheight;
    info.attentionType = this->isSelf;
    info.totalContext = this->m * this->n;
    if(toTrain == 1) {
        std::cout << "Setting tokens and embeddings from tokeniser into transformer" << std::endl;
        setTokenAndEmbeddingForTransformer(TOK);
    }
    else {
        // read from csv file
    }
    calculateAndSetLayout();
    std::cout << "MODEL CLASS CREATED with OpenCL device " << context.device.getInfo<CL_DEVICE_NAME>() << std::endl;
    std::cout << "-----------------------------------------------------------------------" << std::endl;
}

#else

#ifdef USE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

void printCudaDeviceName() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    if (err != cudaSuccess) {
        std::cerr << "CUDA Error (cudaGetDeviceCount): "
                  << cudaGetErrorString(err) << std::endl;
        return;
    }

    if (deviceCount == 0) {
        std::cout << "No CUDA-capable devices found." << std::endl;
        return;
    }

    // Get the current device (or use 0 directly if not set)
    int currentDevice = 0;
    err = cudaGetDevice(&currentDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error (cudaGetDevice): "
                  << cudaGetErrorString(err) << std::endl;
        return;
    }

    cudaDeviceProp deviceProp;
    err = cudaGetDeviceProperties(&deviceProp, currentDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error (cudaGetDeviceProperties): "
                  << cudaGetErrorString(err) << std::endl;
        return;
    }

    std::cout << "CUDA Device " << currentDevice << ": "
              << deviceProp.name << std::endl;
}

#else

void printCudaDeviceName() {
    std::cout << "CUDA is not enabled. Cannot get GPU name." << std::endl;
}

#endif


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
model::model(const std::string& baseDirectory, int m_param, int x_param, int y_param, int n_param, int d_param, 
    int matHeightParam, int l_param, float learning_param, float lambda_L1, float lambda_L2, bool isSelfAttention_param, 
    bool toTrainModel_param, const std::string& tokeniserPath) :
    baseDir(baseDirectory), m(toTrainModel_param ? (m_param > 0 ? m_param : 1) : 1), x(x_param), y(y_param), 
    n(n_param), d(d_param), matheight(matHeightParam), l(l_param), learning(learning_param), lambda_L1(lambda_L1), 
    lambda_L2(lambda_L2), total(m * n), isSelf(isSelfAttention_param), toTrain(toTrainModel_param), metadata(nullptr), 
    chat(nullptr), currentChatLogPath(""), TOK(tokeniserPath), 
    T(this->m, x_param, y_param, n_param, d_param, matHeightParam, l_param, TOK.getVocabularySize(), this->learning, lambda_L1, lambda_L2, 
        this->isSelf, this->toTrain, baseDirectory)
{
    total = this->m * this->n;
    info = {}; // Zero-initialize info struct
    info.vocab = vocabsize;
    info.m = this->m;
    info.x = this->x;
    info.y = this->y;
    info.n = this->n;
    info.d = this->d;
    info.h = this->matheight;
    info.l = this->l;
    info.learning = this->learning;
    info.attentionMech = MECH;
    info.modelArch = ARCH;
    info.matheight = this->matheight;
    info.attentionType = this->isSelf;
    info.totalContext = this->m * this->n;
    std::cout << "Setting tokens and embeddings from tokeniser into transformer" << std::endl;
    if(toTrain == 1) {
        setTokenAndEmbeddingForTransformer(TOK);
    }
    else {
        // read from csv file directly
    }
    calculateAndSetLayout();
    #ifdef USE_CUDA
    std::cout << "Model Created. "; printCudaDeviceName();
    #elif USE_CPU
    std::cout << "Model Created using CPU" << std::endl;
    #endif
    std::cout << "-----------------------------------------------------------------------" << std::endl;
}

#endif
