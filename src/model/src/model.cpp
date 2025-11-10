// model.cpp: implementation of Model class
#include "include/model.hpp"
#include <maths.hpp>
#include <iostream>
#include <stdexcept>
#include <filesystem>
#include <fstream>
#include <locale> // for isspace
#include <sys/stat.h> // For stat to check file existence (alternative to std::filesystem::exists)

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
model::model(OpenCLContext& context, const std::string& baseDirectory, const std::string& tokenDirectory, int m, int x, int y, 
    int n, int d, int matheight, int l, float learning, float lambda_L1, float lambda_L2, bool isSelfAttention, 
    bool toTrainModel, bool contextTrainModel) :
    clcontext(context), baseDir(baseDirectory), m(toTrainModel ? m : 1), x(x), y(y), n(n), d(d), 
    matheight(matheight), l(l), learning(learning), lambda_L1(lambda_L1), lambda_L2(lambda_L2), total(m * n),
    isSelf(isSelfAttention), toTrain(toTrainModel), metadata(nullptr), chat(nullptr), currentChatLogPath(""),
    contextTrain(contextTrainModel), TOK(tokenDirectory, contextTrain, clcontext), vocabsize(TOK.getVocabularySize()),
    T(context, m, x, y, n, d, matheight, l, TOK.getVocabularySize(), learning, lambda_L1, lambda_L2, 
        isSelfAttention, toTrainModel, contextTrainModel, baseDirectory)
{
    total = m * n;
    info = {}; // Zero-initialize info struct
    info.m = m;
    info.x = x;
    info.y = y;
    info.n = n;
    info.d = d;
    info.h = matheight;
    info.l = l;
    info.learning = learning;
    info.attentionMech = MECH;
    info.modelArch = ARCH;
    info.matheight = matheight;
    info.attentionType = isSelf;
    info.totalContext = m * n;
    // setTokens2Transformer();
    calculateAndSetLayout();
    std::cout << "MODEL CLASS CREATED with OpenCL device " << context.device.getInfo<CL_DEVICE_NAME>() << std::endl;
    std::cout << "-----------------------------------------------------------------------" << std::endl;
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
model::model(OpenCLContext& context, const std::string& modelName, const std::string& baseDirectory, const std::string& tokenDirectory, 
    int m, int x, int y, int n, int d, int matheight, int l, float learning, float lambda_L1, float lambda_L2, bool isSelfAttention, 
    bool toTrainModel, bool contextTrainModel) :
    clcontext(context), baseDir(baseDirectory), m(toTrainModel ? m : 1), x(x), y(y), n(n), d(d), 
    matheight(matheight), l(l), learning(learning), lambda_L1(lambda_L1), lambda_L2(lambda_L2), total(m * n),
    isSelf(isSelfAttention), toTrain(toTrainModel), metadata(nullptr), chat(nullptr), currentChatLogPath(""),
    contextTrain(contextTrainModel), TOK(tokenDirectory, contextTrain, clcontext), vocabsize(TOK.getVocabularySize()),
    T(context, modelName + "_", m, x, y, n, d, matheight, l, TOK.getVocabularySize(), learning, lambda_L1, lambda_L2, 
        isSelfAttention, toTrainModel, contextTrainModel, baseDirectory)
{
    total = m * n;
    info = {}; // Zero-initialize info struct
    info.m = m;
    info.x = x;
    info.y = y;
    info.n = n;
    info.d = d;
    info.h = matheight;
    info.l = l;
    info.learning = learning;
    info.attentionMech = MECH;
    info.modelArch = ARCH;
    info.matheight = matheight;
    info.attentionType = isSelf;
    info.totalContext = m * n;
    // setTokens2Transformer();
    calculateAndSetLayout();
    std::cout << "MODEL CREATED for " << modelName << " with OpenCL device " << context.device.getInfo<CL_DEVICE_NAME>() << std::endl;
    std::cout << "-----------------------------------------------------------------------" << std::endl;
}

#else

#ifdef USE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>
#include <cuda_runtime.h>

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
model::model(const std::string& baseDirectory, const std::string& tokenDirectory, int m, int x, int y, 
    int n, int d, int matheight, int l, float learning, float lambda_L1, float lambda_L2, bool isSelfAttention, 
    bool toTrainModel, bool contextTrainModel) :
    baseDir(baseDirectory), m(toTrainModel ? m : 1), x(x), y(y), n(n), d(d), matheight(matheight), l(l), 
    learning(learning), lambda_L1(lambda_L1), lambda_L2(lambda_L2), total(m * n), isSelf(isSelfAttention), toTrain(toTrainModel),
    metadata(nullptr), chat(nullptr), currentChatLogPath(""), contextTrain(contextTrainModel), TOK(tokenDirectory, contextTrain), vocabsize(TOK.getVocabularySize()),
    T(m, x, y, n, d, matheight, l, TOK.getVocabularySize(), learning, lambda_L1, lambda_L2, 
        isSelfAttention, toTrainModel, contextTrainModel, baseDirectory)
{
    total = m * n;
    info = {}; // Zero-initialize info struct
    info.vocab = vocabsize;
    info.m = m;
    info.x = x;
    info.y = y;
    info.n = n;
    info.d = d;
    info.h = matheight;
    info.l = l;
    info.learning = learning;
    info.attentionMech = MECH;
    info.modelArch = ARCH;
    info.matheight = matheight;
    info.attentionType = isSelf;
    info.totalContext = m * n;
    calculateAndSetLayout();
    #ifdef USE_CUDA
    std::cout << "Model Created. "; printCudaDeviceName();
    #elif USE_CPU
    std::cout << "Model Created using CPU" << std::endl;
    #endif
    std::cout << "-----------------------------------------------------------------------" << std::endl;
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
model::model(const std::string& modelName, const std::string& baseDirectory, const std::string& tokenDirectory, 
    int m, int x, int y, int n, int d, int matheight, int l, float learning, float lambda_L1, float lambda_L2, bool isSelfAttention, 
    bool toTrainModel, bool contextTrainModel) :
    baseDir(baseDirectory), m(toTrainModel ? m : 1), x(x), y(y), n(n), d(d), matheight(matheight), l(l), 
    learning(learning), lambda_L1(lambda_L1), lambda_L2(lambda_L2), total(m * n), isSelf(isSelfAttention), toTrain(toTrainModel),
    metadata(nullptr), chat(nullptr), currentChatLogPath(""), contextTrain(contextTrainModel), TOK(tokenDirectory, contextTrain), vocabsize(TOK.getVocabularySize()),
    T(modelName + "_", m, x, y, n, d, matheight, l, TOK.getVocabularySize(), learning, lambda_L1, lambda_L2, 
        isSelfAttention, toTrainModel, contextTrainModel, baseDirectory)
{
    total = m * n;
    info = {}; // Zero-initialize info struct
    info.vocab = vocabsize;
    info.m = m;
    info.x = x;
    info.y = y;
    info.n = n;
    info.d = d;
    info.h = matheight;
    info.l = l;
    info.learning = learning;
    info.attentionMech = MECH;
    info.modelArch = ARCH;
    info.matheight = matheight;
    info.attentionType = isSelf;
    info.totalContext = m * n;
    calculateAndSetLayout();
    #ifdef USE_CUDA
    std::cout << "Model Created. "; printCudaDeviceName();
    #elif USE_CPU
    std::cout << "Model Created using CPU" << std::endl;
    #endif
    std::cout << "-----------------------------------------------------------------------" << std::endl;
}

#endif