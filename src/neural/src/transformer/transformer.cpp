#include <maths.hpp>
#include "include/block.hpp"
#include "include/transformer.hpp"
#include <cstring>
#include <limits>
#include <cmath>
#include <numeric>
#include <algorithm>

#ifndef USE_OPENCL

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
    int l_param, unsigned long long vocab_param, float learning_rate_param, float lambda_L1_param, float lambda_L2_param, 
    bool attentionType_param, bool& inTraining_param, int epoch, const std::string& modelDir_param) :
    m(inTraining_param ? (m_param > 0 ? m_param : 1) : 1), x(x_param), y(y_param), n(n_param),
    d(d_param), h(h_param), l(l_param), vocabsize(vocab_param), isSelf(attentionType_param),
    inTraining(inTraining_param), learning(learning_rate_param), lambda_L1(lambda_L1_param),
    lambda_L2(lambda_L2_param), epsilon(0.0f), epochs(EPOCHS), error(0.0f), trainCount(0), epochCount(0), 
    promptNresponse(nullptr), embeddings(this->vocabsize, d), deEmbeddings(vocab_param, d_param * y_param),
    tokenEmbed(m_param * n_param, d_param), epochs(epoch)
{
#ifdef USE_CPU
    std::cout << "TRANSFORMER constructed with CPU" << std::endl;
#elif USE_CUDA
    std::cout << "TRANSFORMER constructed with CUDA" << std::endl;
#endif
    if(this->inTraining) {
        t.reserve(this->m); // Reserve space
        for (int i = 0; i < this->m; ++i) {
            std::cout << "-----------------------------------------------------------------------" << std::endl;
            std::cout << "                   Block " << i+1 << " constructed                     " << std::endl;
            t.emplace_back(this->x, this->y, this->n, this->d, this->h, this->l, this->vocabsize, this->isSelf,
                this->inTraining, i+1, modelDir_param, learning, lambda_L1, lambda_L2);
        }
        std::cout << "-----------------------------------------------------------------------" << std::endl;
        // training
        std::cout << "block vector t initialised." << std::endl;
        otok.resize(d * x_param, 0.0f);
        tokForBlock = mat(n_param, d_param);
        pred.resize(vocab_param, 0.0f);
        oneHotEncode.resize(vocab_param, 0.0f);
        currentTokenCount = 0;
        blockCount = 1;
        promptCount = 0;
        indexForToken = 0;
        isTerminate = 0;
        t_step_adam = 0;
        params = (this->m * t[0].params) + d + (this->vocabsize * d) + (this->n*this->m)*d;
        this->learning = 0.01f; // Or your initial learning rate
        this->best_loss_for_lr_schedule = std::numeric_limits<float>::(max)();
        this->lr_patience_counter = 0;
        std::cout << "TRANSFORMER constructed. TOTAL PARAMETERS: " << params << std::endl;
    }
    else {
        // for inference
        t.reserve(1); // this->m is 1 for inference
        std::cout << "-----------------------------------------------------------------------" << std::endl;
        std::cout << "                      Block " << 1 << " constructed                    " << std::endl;
        t.emplace_back(this->x, this->y, this->n, this->d, this->h, this->l, this->vocabsize, this->isSelf, this->inTraining, 1,
            modelDir_param, learning, lambda_L1, lambda_L2);
        std::cout << "-----------------------------------------------------------------------" << std::endl;
        std::cout << "\nblock vector t initialised." << std::endl;
        tokForBlock = mat(n_param, d_param);
        otok.resize(d * x_param, 0.0f);
        std::cout << "otok initialised." << std::endl;
        EVuse.resize(x, std::vector<std::vector<std::vector<float>>>(y, std::vector<std::vector<float>>(n, std::vector<float>(d, 0))));
        tokForBlock = mat(n, d);
        currentTokenCount = 0;
        blockCount = 1;
        promptCount = 0;
        indexForToken = 0;
        isTerminate = 0;
        clip_norm = MAX_GRAD_CLIP;
        params = t[0].params + d + (this->vocabsize * d) + (this->n*this->m)*d + (this->x * this->y * this->n * this->d) + this->n*this->d;
        std::cout << "TRANSFORMER constructed. TOTAL PARAMETERS: " << params << std::endl;
    }
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
transformer::transformer(OpenCLContext& context_param, int m_param, int x_param, int y_param, int n_param, 
    int d_param, int h_param, int l_param, unsigned long long vocab_param, float learning_rate_param, 
    float lambda_L1_param, float lambda_L2_param, bool attentionType_param, bool& inTraining_param, 
    int epoch, const std::string& modelDir_param) :
    clcontext(context_param), m(inTraining_param ? (m_param > 0 ? m_param : 1) : 1), x(x_param),
    y(y_param), n(n_param), d(d_param), h(h_param), l(l_param), vocabsize(vocab_param), epochs(epoch),
    isSelf(attentionType_param),  inTraining(inTraining_param), learning(learning_rate_param),
    lambda_L1(lambda_L1_param), lambda_L2(lambda_L2_param), epsilon(0.0f), error(0.0f), trainCount(0),
    epochCount(0), promptNresponse(nullptr), embeddings(vocab_param, d_param), tokenEmbed(m_param * n_param, d_param),
    deEmbeddings(vocab_param, d_param * x_param)
{
    std::cout << "TRANSFORMER constructed with OpenCL context" << std::endl;
    if(this->inTraining) {
        std::cout << "TRANSFORMER: About to initialize block vector t (training, size " << this->m << ")..." << std::endl << std::flush;
        // training
        t.reserve(this->m);
        for (int i = 0; i < this->m; ++i) {
            std::cout << "-----------------------------------------------------------------------" << std::endl;
            std::cout << "                   Block " << i+1 << " constructed                     " << std::endl;
            t.emplace_back(clcontext, this->x, this->y, this->n, this->d, this->h, this->l, this->vocabsize, this->isSelf, 
                this->inTraining, i+1, modelDir_param, learning, lambda_L1, lambda_L2);
        }
        std::cout << "-----------------------------------------------------------------------" << std::endl;
        std::cout << "block vector t initialised." << std::endl;
        otok.resize(d * x_param, 0.0f);
        pred.resize(vocab_param, 0.0f);
        oneHotEncode.resize(vocab_param, 0.0f);
        currentTokenCount = 0;
        blockCount = 1;
        promptCount = 0;
        indexForToken = 0;
        isTerminate = 0;
        t_step_adam = 0;
        this->best_loss_for_lr_schedule = (std::numeric_limits<float>::max)();
        this->lr_patience_counter = 0;
        clip_norm = MAX_GRAD_CLIP;
        params = (this->m * t[0].params) + this->d + (this->vocabsize * this->d) + (this->m * this->n * this->d);
        std::cout << "TRANSFORMER constructed. TOTAL PARAMETERS: " << params << std::endl;
    }
    else {
        // this->m = 1 for inference
        std::cout << "TRANSFORMER: About to initialize block vector t (inference, size " << this->m << ")..." << std::endl << std::flush; // this->m is 1
        // for inference
        t.reserve(1); // this->m is 1
        std::cout << "-----------------------------------------------------------------------" << std::endl;
        std::cout << "                     Block " << 1 << " constructed                     " << std::endl;
        t.emplace_back(clcontext, this->x, this->y, this->n, this->d, this->h, this->l, this->vocabsize, this->isSelf,
            this->inTraining, 1, modelDir_param, learning, lambda_L1, lambda_L2);
        std::cout << "-----------------------------------------------------------------------" << std::endl;
        std::cout << "block vector t initialised." << std::endl;
        otok.resize(d * x_param, 0.0f);
        tokForBlock = mat(n_param, d_param);        // for inference only
        EVuse.resize(x, std::vector<std::vector<std::vector<float>>>(y, std::vector<std::vector<float>>(n, std::vector<float>(d, 0))));
        currentTokenCount = 0;
        blockCount = 1;
        promptCount = 0;
        indexForToken = 0;
        isTerminate = 0;
        // Corrected params calculation for inference (this->m is 1)
        params = t[0].params + this->d + (this->vocabsize * this->d) + (this->m * this->n * this->d) /*tokenEmbed*/ + (this->x * this->y * this->n * this->d) /*EVuse*/ + (this->n * this->d) /*tokForBlock*/;
        std::cout << "TRANSFORMER constructed. TOTAL PARAMETERS: " << params << std::endl;
    }
    std::cout << "-----------------------------------------------------------------------" << std::endl;
}

#endif
