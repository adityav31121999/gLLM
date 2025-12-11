#include "include/attention.hpp"
#include "include/block.hpp"
#include "include/transformer.hpp"
#ifdef USE_CU
#include <cuda_runtime.h>
#endif
#include <cstring>

#ifndef USE_CL

/**
 * @brief constructor for transformer when case is defined
 * @param m_param Number of blocks in the transformer.
 * @param x_param Number of layers of Partial Attention (PA) within each block.
 * @param y_param Number of attention heads in each PA layer.
 * @param n_param Context window size (number of tokens) for a single attention head.
 * @param d_param Dimension of token embeddings.
 * @param h_param Height of MQ, MK matrices (internal dimension for attention).
 * @param l_param Number of layers in MLPs within attention heads/blocks.
 * @param vocab_param Size of the vocabulary.
 * @param learning_rate_param Learning rate for the transformer's components.
 * @param attentionType_param Type of attention: true for self-attention, false for cross-attention.
 * @param inTraining_param Reference to a boolean indicating if the model is in training (true) or inference (false) mode.
 * @param modelDir_param Base directory path for model files.
 */
transformer::transformer(int m_param, int x_param, int y_param, int n_param, int d_param, int h_param,
    int l_param, unsigned int vocab_param, float learning_rate_param, float lambda_L1_param, float lambda_L2_param,
    bool attentionType_param, bool& inTraining_param, bool& contextTrain_param, const std::string& modelDir_param) :
    m(inTraining_param ? (m_param > 0 ? m_param : 1) : 1), x(x_param), y(y_param), n(n_param), d(d_param),
    h(h_param), l(l_param), vocabsize(vocab_param), isSelf(attentionType_param), inTraining(inTraining_param), 
    learning(learning_rate_param), lambda_L1(lambda_L1_param), lambda_L2(lambda_L2_param), epochs(EPOCHS),
    error(0.0f), trainCount(0), epochCount(0), seqChat(nullptr), embeddings(vocab_param, d_param),
    tokenEmbed(m_param * n_param, d_param), positional(m_param * n_param, d_param), embedPlusPos(m_param * n_param, d_param),
    contextTrain(contextTrain_param)
{
    if(inTraining) {
       blocks.reserve(m); // Reserve space
        for (int i = 0; i < m; ++i) {
            std::cout << "-----------------------------------------------------------------------" << std::endl;
            std::cout << "                   Block " << i+1 << " constructed                     " << std::endl;
           blocks.emplace_back(x_param, y_param, n_param, d_param, h_param, l_param, vocabsize, isSelf, inTraining, i+1, modelDir_param, learning);
        }
        std::cout << "-----------------------------------------------------------------------" << std::endl;
        // training
        std::cout << "block vector t initialised." << std::endl;
        if(contextTrain == 0) {
            otok.resize(d, 0.0f);
        }
        else {
            oneHotEncode.resize(vocabsize, 0.0f);
            pred.resize(vocabsize, 0.0f);
            otok.resize(x * d, 0.0f);
            deEmbeddings = mat(vocabsize, d*x); // Initialize deEmbeddings for context training
        }
        std::cout << "otok initialised. Size: " << otok.size() << std::endl;
        currentTokenCount = 0;
        blockCount = 1;
        sequence1Count = 0;
        indexForToken = 0;
        plateau_patience = LR_PATIENCE;
        plateau_count = 0;
        params = (m * blocks[0].params) + d + (vocabsize * d) + (n*m)*d;
        std::cout << "TRANSFORMER constructed. TOTAL PARAMETERS: " << params << std::endl;
    }
    else {
        // for inference
       blocks.reserve(1); // m is 1 for inference
        std::cout << "-----------------------------------------------------------------------" << std::endl;
        std::cout << "                      Block " << 1 << " constructed                    " << std::endl;
       blocks.emplace_back(x_param, y_param, n_param, d_param, h_param, l_param, vocabsize, isSelf, inTraining, 1, modelDir_param, learning);
        std::cout << "-----------------------------------------------------------------------" << std::endl;
        std::cout << "\nblock vector t initialised." << std::endl;
        if(contextTrain == 0) {
            otok.resize(d, 0.0f);
        }
        else {
            oneHotEncode.resize(vocabsize, 0.0f);
            pred.resize(vocabsize, 0.0f);
            otok.resize(x * d, 0.0f);
            deEmbeddings = mat(vocabsize, d*x); // Initialize deEmbeddings for context training
        }
        std::cout << "otok initialised. Size: " << otok.size() << std::endl;
        EVuse.resize(x, std::vector<std::vector<std::vector<float>>>(y, std::vector<std::vector<float>>(n, std::vector<float>(d, 0))));
        tokForBlock = mat(n, d);
        embedQKed = mat(CONTEXT_WIN, d_param);
        currentTokenCount = 0;
        blockCount = 1;
        sequence1Count = 0;
        indexForToken = 0;
        params = blocks[0].params + d + (vocabsize * d) + (n*m)*d + (x * y * n * d) + n*d;
        std::cout << "TRANSFORMER constructed. TOTAL PARAMETERS: " << params << std::endl;
    }
    #ifdef USE_CU
        int deviceId;
        cudaGetDevice(&deviceId);
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, deviceId);
        std::cout << "TRANSFORMER constructed using CUDA device: " << props.name << std::endl;
    #elif USE_CPU
        std::cout << "TRANSFORMER constructed using CPU" << std::endl;
    #endif
    std::cout << "-----------------------------------------------------------------------" << std::endl;
}


/**
 * @brief constructor for transformer when case is defined
 * @param m_param Number of blocks in the transformer.
 * @param x_param Number of layers of Partial Attention (PA) within each block.
 * @param y_param Number of attention heads in each PA layer.
 * @param n_param Context window size (number of tokens) for a single attention head.
 * @param d_param Dimension of token embeddings.
 * @param h_param Height of MQ, MK matrices (internal dimension for attention).
 * @param l_param Number of layers in MLPs within attention heads/blocks.
 * @param vocab_param Size of the vocabulary.
 * @param learning_rate_param Learning rate for the transformer's components.
 * @param attentionType_param Type of attention: true for self-attention, false for cross-attention.
 * @param inTraining_param Reference to a boolean indicating if the model is in training (true) or inference (false) mode.
 * @param modelDir_param Base directory path for model files.
 */
transformer::transformer(const std::string& modelName, int m_param, int x_param, int y_param, int n_param,
    int d_param, int h_param, int l_param, unsigned int vocab_param, float learning_rate_param,
    float lambda_L1_param, float lambda_L2_param, bool attentionType_param, bool& inTraining_param,
    bool& contextTrain_param, const std::string& modelDir_param) :
    m(inTraining_param ? (m_param > 0 ? m_param : 1) : 1), x(x_param), y(y_param), n(n_param), d(d_param), 
    h(h_param), l(l_param), vocabsize(vocab_param), isSelf(attentionType_param), inTraining(inTraining_param), 
    learning(learning_rate_param), lambda_L1(lambda_L1_param), lambda_L2(lambda_L2_param), epochs(EPOCHS), 
    error(0.0f), trainCount(0), epochCount(0), seqChat(nullptr), embeddings(vocab_param, d_param), 
    tokenEmbed(m_param * n_param, d_param), positional(m_param * n_param, d_param), embedPlusPos(m_param * n_param, d_param),
    contextTrain(contextTrain_param)
{
    if(inTraining) {
       blocks.reserve(m); // Reserve space
        for (int i = 0; i < m; ++i) {
            std::cout << "-----------------------------------------------------------------------" << std::endl;
            std::cout << "                   Block " << i+1 << " constructed                     " << std::endl;
           blocks.emplace_back(modelName, x_param, y_param, n_param, d_param, h_param, l_param, vocabsize, isSelf, inTraining, i+1, modelDir_param, learning);
        }
        std::cout << "-----------------------------------------------------------------------" << std::endl;
        // training
        std::cout << "block vector t initialised." << std::endl;
        if(contextTrain == 0) {
            otok.resize(d, 0.0f);
        }
        else {
            oneHotEncode.resize(vocabsize, 0.0f);
            pred.resize(vocabsize, 0.0f);
            otok.resize(x * d, 0.0f);
            deEmbeddings = mat(vocabsize, d*x); // Initialize deEmbeddings for context training
        }
        std::cout << "otok initialised. Size: " << otok.size() << std::endl;
        currentTokenCount = 0;
        blockCount = 1;
        sequence1Count = 0;
        indexForToken = 0;
        plateau_patience = LR_PATIENCE;
        plateau_count = 0;
        params = (m * blocks[0].params) + d + (vocabsize * d) + (n*m)*d;
        std::cout << "TRANSFORMER constructed. TOTAL PARAMETERS: " << params << std::endl;
    }
    else {
        // for inference
       blocks.reserve(1); // m is 1 for inference
        std::cout << "-----------------------------------------------------------------------" << std::endl;
        std::cout << "                      Block " << 1 << " constructed                    " << std::endl;
       blocks.emplace_back(modelName, x_param, y_param, n_param, d_param, h_param, l_param, vocabsize, isSelf, inTraining, 1, modelDir_param, learning);
        std::cout << "-----------------------------------------------------------------------" << std::endl;
        std::cout << "\nblock vector t initialised." << std::endl;
        if(contextTrain == 0) {
            otok.resize(d, 0.0f);
        }
        else {
            oneHotEncode.resize(vocabsize, 0.0f);
            pred.resize(vocabsize, 0.0f);
            otok.resize(x * d, 0.0f);
            deEmbeddings = mat(vocabsize, d*x); // Initialize deEmbeddings for context training
        }
        std::cout << "otok initialised. Size: " << otok.size() << std::endl;
        EVuse.resize(x, std::vector<std::vector<std::vector<float>>>(y, std::vector<std::vector<float>>(n, std::vector<float>(d, 0))));
        tokForBlock = mat(n, d);
        embedQKed = mat(CONTEXT_WIN, d_param);
        currentTokenCount = 0;
        blockCount = 1;
        sequence1Count = 0;
        indexForToken = 0;
        params = blocks[0].params + d + (vocabsize * d) + (n*m)*d + (x * y * n * d) + n*d;
        std::cout << "TRANSFORMER constructed. TOTAL PARAMETERS: " << params << std::endl;
    }
    #ifdef USE_CU
        int deviceId;
        cudaGetDevice(&deviceId);
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, deviceId);
        std::cout << "TRANSFORMER constructed using CUDA device: " << props.name << std::endl;
    #elif USE_CPU
        std::cout << "TRANSFORMER constructed using CPU" << std::endl;
    #endif
    std::cout << "-----------------------------------------------------------------------" << std::endl;
}

#else

/**
 * @brief constructor for transformer when case is defined
 * @param context_param Reference to the shared OpenCL context.
 * @param m_param Number of blocks in the transformer.
 * @param x_param Number of layers of Partial Attention (PA) within each block.
 * @param y_param Number of attention heads in each PA layer.
 * @param n_param Context window size (number of tokens) for a single attention head.
 * @param d_param Dimension of token embeddings.
 * @param h_param Height of MQ, MK matrices (internal dimension for attention).
 * @param l_param Number of layers in MLPs within attention heads/blocks.
 * @param vocab_param Size of the vocabulary.
 * @param learning_rate_param Learning rate for the transformer's components.
 * @param attentionType_param Type of attention: true for self-attention, false for cross-attention.
 * @param inTraining_param Reference to a boolean indicating if the model is in training (true) or inference (false) mode.
 * @param modelDir_param Base directory path for model files.
 */
transformer::transformer(OpenCLContext& context_param, int m_param, int x_param, int y_param, 
    int n_param, int d_param, int h_param, int l_param, unsigned int vocab_param, float learning_rate_param, 
    float lambda_L1_param, float lambda_L2_param, bool attentionType_param, bool& inTraining_param, 
    bool& contextTrain_param, const std::string& modelDir_param) : clcontext(context_param),
    m(inTraining_param ? (m_param > 0 ? m_param : 1) : 1), x(x_param), y(y_param), n(n_param), d(d_param),
    h(h_param), l(l_param), vocabsize(vocab_param), isSelf(attentionType_param), inTraining(inTraining_param), 
    learning(learning_rate_param), lambda_L1(lambda_L1_param), lambda_L2(lambda_L2_param), epochs(EPOCHS),
    error(0.0f), trainCount(0), epochCount(0), seqChat(nullptr), embeddings(vocab_param, d_param),
    tokenEmbed(m_param * n_param, d_param), positional(m_param * n_param, d_param), embedPlusPos(m_param * n_param, d_param),
    contextTrain(contextTrain_param)
{
    if(inTraining) {
        std::cout << "TRANSFORMER: About to initialize block vector t (training, size " << m << ")..." << std::endl << std::flush;
        // training
        blocks.reserve(m);
        for (int i = 0; i < m; ++i) {
            std::cout << "-----------------------------------------------------------------------" << std::endl;
            std::cout << "                   Block " << i+1 << " constructed                     " << std::endl;
            blocks.emplace_back(clcontext, x, y, n, d, h, l, vocabsize, isSelf, inTraining, i+1, modelDir_param, learning);
        }
        std::cout << "-----------------------------------------------------------------------" << std::endl;
        std::cout << "block vector t initialised." << std::endl;
        if(contextTrain == 0) {
            otok.resize(d, 0.0f);
        }
        else {
            oneHotEncode.resize(vocabsize, 0.0f);
            pred.resize(vocabsize, 0.0f);
            otok.resize(x * d, 0.0f);
            deEmbeddings = mat(vocabsize, d*x); // Initialize deEmbeddings for context training
        }
        std::cout << "otok initialised. Size: " << otok.size() << std::endl;
        currentTokenCount = 0;
        blockCount = 1;
        sequence1Count = 0;
        indexForToken = 0;
        plateau_patience = LR_PATIENCE;
        plateau_count = 0;
        // Corrected params calculation for training
        params = (m * blocks[0].params) + d + (vocabsize * d) + (m * n * d);
        std::cout << "TRANSFORMER constructed. TOTAL PARAMETERS: " << params << std::endl;
    }
    else {
        // m is 1 for inference
        std::cout << "TRANSFORMER: About to initialize block vector t (inference, size " << m << ")..." << std::endl << std::flush; // m is 1
        // for inference
        blocks.reserve(1); // m is 1
        std::cout << "-----------------------------------------------------------------------" << std::endl;
        std::cout << "                     Block " << 1 << " constructed                     " << std::endl;
        blocks.emplace_back(clcontext, x, y, n, d, h, l, vocabsize, isSelf, inTraining, 1, modelDir_param, learning);
        std::cout << "-----------------------------------------------------------------------" << std::endl;
        std::cout << "block vector t initialised." << std::endl;
        if(contextTrain == 0) {
            otok.resize(d, 0.0f);
        }
        else {
            oneHotEncode.resize(vocabsize, 0.0f);
            pred.resize(vocabsize, 0.0f);
            otok.resize(x * d, 0.0f);
            deEmbeddings = mat(vocabsize, d*x); // Initialize deEmbeddings for context training
        }
        std::cout << "otok initialised. Size: " << otok.size() << std::endl;
        embedQKed = mat(CONTEXT_WIN, d_param);
        tokForBlock = mat(n_param, d_param);
        EVuse.resize(x, std::vector<std::vector<std::vector<float>>>(y, std::vector<std::vector<float>>(n, std::vector<float>(d, 0))));
        currentTokenCount = 0;
        blockCount = 1;
        sequence1Count = 0;
        indexForToken = 0;
        // Corrected params calculation for inference (m is 1)
        params = blocks[0].params + d + (vocabsize * d) + (m * n * d) + (x * y * n * d) + (n * d);
        std::cout << "TRANSFORMER constructed. TOTAL PARAMETERS: " << params << std::endl;
    }
    std::cout << "TRANSFORMER constructed with OpenCL context using device: "
              << clcontext.device.getInfo<CL_DEVICE_NAME>() << std::endl << std::flush;
    std::cout << "-----------------------------------------------------------------------" << std::endl;
}


/**
 * @brief constructor for transformer when case is defined
 * @param context_param Reference to the shared OpenCL context.
 * @param m_param Number of blocks in the transformer.
 * @param x_param Number of layers of Partial Attention (PA) within each block.
 * @param y_param Number of attention heads in each PA layer.
 * @param n_param Context window size (number of tokens) for a single attention head.
 * @param d_param Dimension of token embeddings.
 * @param h_param Height of MQ, MK matrices (internal dimension for attention).
 * @param l_param Number of layers in MLPs within attention heads/blocks.
 * @param vocab_param Size of the vocabulary.
 * @param learning_rate_param Learning rate for the transformer's components.
 * @param attentionType_param Type of attention: true for self-attention, false for cross-attention.
 * @param inTraining_param Reference to a boolean indicating if the model is in training (true) or inference (false) mode.
 * @param modelDir_param Base directory path for model files.
 */
transformer::transformer(OpenCLContext& context_param, const std::string& modelName, int m_param, int x_param, int y_param, 
    int n_param, int d_param, int h_param, int l_param, unsigned int vocab_param, float learning_rate_param, 
    float lambda_L1_param, float lambda_L2_param, bool attentionType_param, bool& inTraining_param, 
    bool& contextTrain_param, const std::string& modelDir_param) : clcontext(context_param), 
    m(inTraining_param ? (m_param > 0 ? m_param : 1) : 1), x(x_param), y(y_param), n(n_param), d(d_param), 
    h(h_param), l(l_param), vocabsize(vocab_param), isSelf(attentionType_param), inTraining(inTraining_param), 
    learning(learning_rate_param), lambda_L1(lambda_L1_param), lambda_L2(lambda_L2_param), epochs(EPOCHS), 
    error(0.0f), trainCount(0), epochCount(0), seqChat(nullptr), embeddings(vocab_param, d_param), 
    tokenEmbed(m_param * n_param, d_param), positional(m_param * n_param, d_param), embedPlusPos(m_param * n_param, d_param),
    contextTrain(contextTrain_param)
{
    if(inTraining) {
        std::cout << "TRANSFORMER: About to initialize block vector t (training, size " << m << ")..." << std::endl << std::flush;
        // training
        blocks.reserve(m);
        for (int i = 0; i < m; ++i) {
            std::cout << "-----------------------------------------------------------------------" << std::endl;
            // modelNameBi_
            blocks.emplace_back(clcontext, modelName + "B" + std::to_string(i+1) + "_", x, y, n, d, h, l, vocabsize, isSelf, inTraining, i+1, modelDir_param, learning);
            std::cout << "                   Block " << i+1 << " constructed                     " << std::endl;
        }
        std::cout << "-----------------------------------------------------------------------" << std::endl;
        std::cout << "block vector t initialised." << std::endl;
        if(contextTrain == 0) {
            otok.resize(d, 0.0f);
        }
        else {
            oneHotEncode.resize(vocabsize, 0.0f);
            pred.resize(vocabsize, 0.0f);
            otok.resize(x * d, 0.0f);
            deEmbeddings = mat(vocabsize, d*x); // Initialize deEmbeddings for context training
        }
        std::cout << "otok initialised. Size: " << otok.size() << std::endl;
        currentTokenCount = 0;
        blockCount = 1;
        sequence1Count = 0;
        indexForToken = 0;
        plateau_patience = LR_PATIENCE;
        plateau_count = 0;
        // Corrected params calculation for training
        params = (m * blocks[0].params) + d + (vocabsize * d) + (m * n * d);
        std::cout << "TRANSFORMER constructed. TOTAL PARAMETERS: " << params << std::endl;
    }
    else {
        // m is 1 for inference
        std::cout << "TRANSFORMER: About to initialize block vector t (inference, size " << m << ")..." << std::endl << std::flush; // m is 1
        // for inference
        blocks.reserve(1); // m is 1
        std::cout << "-----------------------------------------------------------------------" << std::endl;
        std::cout << "                     Block " << 1 << " constructed                     " << std::endl;
        blocks.emplace_back(clcontext, modelName, x, y, n, d, h, l, vocabsize, isSelf, inTraining, 1, modelDir_param, learning);
        std::cout << "-----------------------------------------------------------------------" << std::endl;
        std::cout << "block vector t initialised." << std::endl;
        if(contextTrain == 0) {
            otok.resize(d, 0.0f);
        }
        else {
            oneHotEncode.resize(vocabsize, 0.0f);
            pred.resize(vocabsize, 0.0f);
            otok.resize(x * d, 0.0f);
            deEmbeddings = mat(vocabsize, d*x); // Initialize deEmbeddings for context training
        }
        std::cout << "otok initialised. Size: " << otok.size() << std::endl;
        tokForBlock = mat(n_param, d_param);
        embedQKed = mat(CONTEXT_WIN, d_param);
        EVuse.resize(x, std::vector<std::vector<std::vector<float>>>(y, std::vector<std::vector<float>>(n, std::vector<float>(d, 0))));
        currentTokenCount = 0;
        blockCount = 1;
        sequence1Count = 0;
        indexForToken = 0;
        // Corrected params calculation for inference (m is 1)
        params = blocks[0].params + d + (vocabsize * d) + (m * n * d) + (x * y * n * d) + (n * d);
        std::cout << "TRANSFORMER constructed. TOTAL PARAMETERS: " << params << std::endl;
    }
    std::cout << "TRANSFORMER constructed with OpenCL context using device: "
              << clcontext.device.getInfo<CL_DEVICE_NAME>() << std::endl << std::flush;
    std::cout << "-----------------------------------------------------------------------" << std::endl;
}

#endif
