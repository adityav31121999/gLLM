// model.cpp: implementation of Model class
#include "include/model.hpp"
#include <maths.hpp>
#include <iostream>
#include <stdexcept>
#include <filesystem>
#include <fstream>
#include <locale> // for isspace
#include <sys/stat.h> // For stat to check file existence (alternative to std::filesystem::exists)
#include <thread> // For std::thread::hardware_concurrency
#include <string> // For std::string and std::to_string

#ifdef USE_CL

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
    contextTrain(contextTrainModel), TOK(tokenDirectory, contextTrain, clcontext), vocabsize(0),
    T(context, m, x, y, n, d, matheight, l, TOK.getVocabularySize(), learning, lambda_L1, lambda_L2, 
        isSelfAttention, toTrainModel, contextTrainModel, baseDirectory)
{
    std::cout << "Model class to be created using: " << std::endl;
    std::cout << "\t 1. number of blocks: " << m << std::endl;
    std::cout << "\t 2. number of incomplete attentions in each partial attention: " << x << std::endl;
    std::cout << "\t 3. number of layers of partial attention for complete attention block: " << y << std::endl;
    std::cout << "\t 4. total tokens for each attention head: " << n << std::endl;
    std::cout << "\t 5. token dimension: " << d << std::endl;
    std::cout << "\t 6. height of MQ, MK and columns of MV, MH: " << matheight << std::endl;
    std::cout << "\t 7. layers of mlp: " << l << std::endl;
    std::cout << "\t 8. learning rate for MLPs: " << learning << std::endl;
    std::cout << "\t 9. lambda L1: " << lambda_L1 << std::endl;
    std::cout << "\t10. lambda L2: " << lambda_L2 << std::endl;
    std::cout << "\t11. is self attention: " << isSelf << std::endl;
    std::cout << "\t12. to train model: " << toTrain << std::endl;
    std::cout << "\t13. to train context: " << contextTrain << std::endl;
    std::cout << "\t14. base directory: " << baseDir << std::endl;
    std::cout << "\t15. token directory: " << tokenDirectory << std::endl;
    std::cout << "\t16. vocabulary size: " << vocabsize << std::endl;
    // initialise members
    total = m * n;
    vocabsize = TOK.getVocabularySize();
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
    contextTrain(contextTrainModel), TOK(tokenDirectory, contextTrain, clcontext), vocabsize(0),
    T(context, modelName + "_", m, x, y, n, d, matheight, l, TOK.getVocabularySize(), learning, lambda_L1, lambda_L2, 
        isSelfAttention, toTrainModel, contextTrainModel, baseDirectory)
{
    std::cout << "Model class to be created using: " << std::endl;
    std::cout << "\t 1. model name: " << modelName << std::endl;
    std::cout << "\t 2. number of blocks: " << m << std::endl;
    std::cout << "\t 3. number of incomplete attentions in each partial attention: " << x << std::endl;
    std::cout << "\t 4. number of layers of partial attention for complete attention block: " << y << std::endl;
    std::cout << "\t 5. total tokens for each attention head: " << n << std::endl;
    std::cout << "\t 6. token dimension: " << d << std::endl;
    std::cout << "\t 7. height of MQ, MK and columns of MV, MH: " << matheight << std::endl;
    std::cout << "\t 8. layers of mlp: " << l << std::endl;
    std::cout << "\t 9. learning rate for MLPs: " << learning << std::endl;
    std::cout << "\t10. lambda L1: " << lambda_L1 << std::endl;
    std::cout << "\t11. lambda L2: " << lambda_L2 << std::endl;
    std::cout << "\t12. is self attention: " << isSelf << std::endl;
    std::cout << "\t13. to train model: " << toTrain << std::endl;
    std::cout << "\t14. to train context: " << contextTrain << std::endl;
    std::cout << "\t15. base directory: " << baseDir << std::endl;
    std::cout << "\t16. token directory: " << tokenDirectory << std::endl;
    std::cout << "\t17. vocabulary size: " << vocabsize << std::endl;
    // initialise members
    total = m * n;
    vocabsize = TOK.getVocabularySize();
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

    #ifdef USE_CU
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
    metadata(nullptr), chat(nullptr), currentChatLogPath(""), contextTrain(contextTrainModel), TOK(tokenDirectory, contextTrain), vocabsize(0),
    T(m, x, y, n, d, matheight, l, TOK.getVocabularySize(), learning, lambda_L1, lambda_L2, 
        isSelfAttention, toTrainModel, contextTrainModel, baseDirectory)
{
    std::cout << "Model class to be created using: " << std::endl;
    std::cout << "Model class to be created using: " << std::endl;
    std::cout << "\t 1. number of blocks: " << m << std::endl;
    std::cout << "\t 2. number of incomplete attentions in each partial attention: " << x << std::endl;
    std::cout << "\t 3. number of layers of partial attention for complete attention block: " << y << std::endl;
    std::cout << "\t 4. total tokens for each attention head: " << n << std::endl;
    std::cout << "\t 5. token dimension: " << d << std::endl;
    std::cout << "\t 6. height of MQ, MK and columns of MV, MH: " << matheight << std::endl;
    std::cout << "\t 7. layers of mlp: " << l << std::endl;
    std::cout << "\t 8. learning rate for MLPs: " << learning << std::endl;
    std::cout << "\t 9. lambda L1: " << lambda_L1 << std::endl;
    std::cout << "\t10. lambda L2: " << lambda_L2 << std::endl;
    std::cout << "\t11. is self attention: " << isSelf << std::endl;
    std::cout << "\t12. to train model: " << toTrain << std::endl;
    std::cout << "\t13. to train context: " << contextTrain << std::endl;
    std::cout << "\t14. base directory: " << baseDir << std::endl;
    std::cout << "\t15. token directory: " << tokenDirectory << std::endl;
    std::cout << "\t16. vocabulary size: " << vocabsize << std::endl;
    // initialise members
    total = m * n;
    vocabsize = TOK.getVocabularySize();
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
    #ifdef USE_CU
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
    metadata(nullptr), chat(nullptr), currentChatLogPath(""), contextTrain(contextTrainModel), TOK(tokenDirectory, contextTrain), vocabsize(0),
    T(modelName + "_", m, x, y, n, d, matheight, l, TOK.getVocabularySize(), learning, lambda_L1, lambda_L2, 
        isSelfAttention, toTrainModel, contextTrainModel, baseDirectory)
{
    std::cout << "Model class to be created using: " << std::endl;
    std::cout << "Model class to be created using: " << std::endl;
    std::cout << "\t 1. model name: " << modelName << std::endl;
    std::cout << "\t 2. number of blocks: " << m << std::endl;
    std::cout << "\t 3. number of incomplete attentions in each partial attention: " << x << std::endl;
    std::cout << "\t 4. number of layers of partial attention for complete attention block: " << y << std::endl;
    std::cout << "\t 5. total tokens for each attention head: " << n << std::endl;
    std::cout << "\t 6. token dimension: " << d << std::endl;
    std::cout << "\t 7. height of MQ, MK and columns of MV, MH: " << matheight << std::endl;
    std::cout << "\t 8. layers of mlp: " << l << std::endl;
    std::cout << "\t 9. learning rate for MLPs: " << learning << std::endl;
    std::cout << "\t10. lambda L1: " << lambda_L1 << std::endl;
    std::cout << "\t11. lambda L2: " << lambda_L2 << std::endl;
    std::cout << "\t12. is self attention: " << isSelf << std::endl;
    std::cout << "\t13. to train model: " << toTrain << std::endl;
    std::cout << "\t14. to train context: " << contextTrain << std::endl;
    std::cout << "\t15. base directory: " << baseDir << std::endl;
    std::cout << "\t16. token directory: " << tokenDirectory << std::endl;
    std::cout << "\t17. vocabulary size: " << vocabsize << std::endl;
    // initialise members
    total = m * n;
    vocabsize = TOK.getVocabularySize();
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
    #ifdef USE_CU
    std::cout << "Model Created. "; printCudaDeviceName();
    #elif USE_CPU
    std::cout << "Model Created using CPU" << std::endl;
    #endif
    std::cout << "-----------------------------------------------------------------------" << std::endl;
}

#endif