
// transformer class constructor
#include "include/attention.hpp"
#include "include/block.hpp"
#include "include/transformer.hpp"

#ifndef USE_OPENCL

/**
 * @brief Constructor for single-block transformer for use
 * @param x number of partial attentions in block
 * @param y number of attention in each partial attention
 * @param n number of tokens for each attention head
 * @param d dimension of each token
 * @param h height of MQ, MK and columns of MV, MH
 * @param l layers of mlp
 */
transformer::transformer(int x, int y, int n, int d, int h, int l, int vocab):
    m(1), x(x), y(y), n(n), d(d), h(h), l(l) 
{
    t = std::vector<block>(1, block(x, y, n, d, h, l, vocab));
    tokens.resize(n*m, std::string(0));
    mTokens.resize(vocabsize, std::string(0));
    otok.resize(d, 0);
    embeddings.resize(vocabsize, std::vector<float>(d, 0));
    tokenEmbed.resize(n * m, std::vector<float>(d, 0));
    totalParams = ((2 * h) + (l * d)) * 2 * d * x * y * n;
    isSelf = true;
    currentTokenCount = 0;
    blockCount = 1;
    promptCount = 0;
    indexForToken = 0;
}


/**
 * @brief Constructor for many-block transformer for training
 * @param m number of blocks in transformer
 * @param x number of partial attentions in block
 * @param y number of attention in each partial attention
 * @param n number of tokens for each attention head
 * @param d dimension of each token
 * @param h height of MQ, MK and columns of MV, MH
 * @param l layers of mlp
 */
transformer::transformer(int m, int x, int y, int n, int d, int h, int l, int vocab) :
    m(m), x(x), y(y), n(n), d(d), h(h), l(l) 
{
    t = std::vector<block>(m, block(x, y, n, d, h, l, vocab));
    // total permissible tokens = m * n
    tokens.resize(n*m, std::string(0));
    mTokens.resize(vocabsize, std::string(0));
    otok.resize(d, 0);
    embeddings.resize(vocabsize, std::vector<float>(d, 0));
    tokenEmbed.resize(n * m, std::vector<float>(d, 0));
    isSelf = 1;
    currentTokenCount = 0;
    blockCount = 1;
    promptCount = 0;
    indexForToken = 0;
    isTerminate = 0;
}


/**
 * @brief Constructor for single-block transformer for use
 * @param x number of partial attentions in block
 * @param y number of attention in each partial attention
 * @param n number of tokens for each attention head
 * @param d dimension of each token
 * @param h height of MQ, MK and columns of MV, MH
 * @param l layers of mlp
 */
transformer::transformer(int x, int y, int n, int d, int h, int l, int vocab, bool attentionType):
    m(1), x(x), y(y), n(n), d(d), h(h), l(l), isSelf(attentionType) 
{
    t = std::vector<block>(1, block(x, y, n, d, h, l, vocab, attentionType));
    // total permissible tokens = n
    tokens.resize(n, std::string(0));
    mTokens.resize(vocabsize, std::string(0));
    otok.resize(d, 0);
    embeddings.resize(vocabsize, std::vector<float>(d, 0));
    tokenEmbed.resize(n, std::vector<float>(d, 0));
    isSelf = attentionType;
    currentTokenCount = 0;
    blockCount = 1;
    promptCount = 0;
    indexForToken = 0;
    isTerminate = 0;
}


/**
 * @brief Constructor for many-block transformer for training
 * @param m number of blocks in transformer
 * @param x number of partial attentions in block
 * @param y number of attention in each partial attention
 * @param n number of tokens for each attention head
 * @param d dimension of each token
 * @param h height of MQ, MK and columns of MV, MH
 * @param l layers of mlp
 */
transformer::transformer(int m, int x, int y, int n, int d, int h, int l, int vocab, bool attentionType) :
    m(m), x(x), y(y), n(n), d(d), h(h), l(l), isSelf(attentionType) 
{
    t = std::vector<block>(m, block(x, y, n, d, h, l, vocab, attentionType, inTraining));
    tokens.resize(n*m, std::string(0));
    mTokens.resize(vocabsize, std::string(0));
    otok.resize(d, 0);
    embeddings.resize(vocabsize, std::vector<float>(d, 0));
    tokenEmbed.resize(n * m, std::vector<float>(d, 0));
    isSelf = attentionType;
    currentTokenCount = 0;
    blockCount = 1;
    promptCount = 0;
    indexForToken = 0;
    isTerminate = 0;
}


/**
 * @brief constructor for transformer when case is defined
 * @param m number of blocks in trained transformer
 * @param x number of layers of PA
 * @param y number of attention heads in PA
 * @param n context window for single head
 * @param h height of matrices
 * @param l layer of MLPs
 * @param vocab vocabulary size
 * @param attentionType self (1) or cross (0) attention
 * @param inTraining training (TRUE) or use (FALSE)
 */
transformer::transformer(int m, int x, int y, int n, int d, int h, int l, int vocab, bool attentionType, bool& inTraining) :
    m(m), x(x), y(y), n(n), d(d), h(h), l(l), vocabsize(vocab), isSelf(attentionType) 
{
    if(inTraining == true || inTraining == 1) {
        // training
        t = std::vector<block>(m, block(x, y, n, d, h, l, vocab, attentionType, inTraining));
        tokens.resize(n*m, std::string(0));
        mTokens.resize(vocabsize, std::string(0));
        otok.resize(d, 0);
        embeddings.resize(vocabsize, std::vector<float>(d, 0));
        tokenEmbed.resize(n * m, std::vector<float>(d, 0));
        isSelf = attentionType;
        currentTokenCount = 0;
        blockCount = 1;
        promptCount = 0;
        indexForToken = 0;
        isTerminate = 0;
    }
    else {
        // for inference
        t = std::vector<block>(1, block(x, y, n, d, h, l, vocab, attentionType, inTraining));
        tokens.resize(n*m, std::string(0));
        mTokens.resize(vocabsize, std::string(0));
        otok.resize(d, 0);
        embeddings.resize(vocabsize, std::vector<float>(d, 0));
        tokenEmbed.resize(n * m, std::vector<float>(d, 0));
        EVuse.resize(x, std::vector<std::vector<std::vector<float>>>(y, std::vector<std::vector<float>>(n, std::vector<float>(d, 0))));
        tokForBlock.resize(CONTEXT_WIN, std::vector<float>(d, 0));
        isSelf = attentionType;
        currentTokenCount = 0;
        blockCount = 1;
        promptCount = 0;
        indexForToken = 0;
        isTerminate = 0;
    }
}

#else

/**
 * @brief Constructor for single-block transformer for use
 * @param x number of partial attentions in block
 * @param y number of attention in each partial attention
 * @param n number of tokens for each attention head
 * @param d dimension of each token
 * @param h height of MQ, MK and columns of MV, MH
 * @param l layers of mlp
 */
transformer::transformer(OpenCLContext& context, int x, int y, int n, int d, int h, int l, int vocab):
    clcontext(context), m(1), x(x), y(y), n(n), d(d), h(h), l(l)
{
    t = std::vector<block>(1, block(context, x, y, n, d, h, l, vocab));
    tokens.resize(n*m, std::string(0));
    mTokens.resize(vocabsize, std::string(0));
    otok.resize(d, 0);
    embeddings.resize(vocabsize, std::vector<float>(d, 0));
    tokenEmbed.resize(n * m, std::vector<float>(d, 0));
    totalParams = ((2 * h) + (l * d)) * 2 * d * x * y * n;
    isSelf = true;
    currentTokenCount = 0;
    blockCount = 1;
    promptCount = 0;
    indexForToken = 0;
}


/**
 * @brief Constructor for many-block transformer for training
 * @param m number of blocks in transformer
 * @param x number of partial attentions in block
 * @param y number of attention in each partial attention
 * @param n number of tokens for each attention head
 * @param d dimension of each token
 * @param h height of MQ, MK and columns of MV, MH
 * @param l layers of mlp
 */
transformer::transformer(OpenCLContext& context, int m, int x, int y, int n, int d, int h, int l, int vocab) :
    clcontext(context), m(m), x(x), y(y), n(n), d(d), h(h), l(l) 
{
    t = std::vector<block>(m, block(context, x, y, n, d, h, l, vocab));
    // total permissible tokens = m * n
    tokens.resize(n*m, std::string(0));
    mTokens.resize(vocabsize, std::string(0));
    otok.resize(d, 0);
    embeddings.resize(vocabsize, std::vector<float>(d, 0));
    tokenEmbed.resize(n * m, std::vector<float>(d, 0));
    isSelf = 1;
    currentTokenCount = 0;
    blockCount = 1;
    promptCount = 0;
    indexForToken = 0;
    isTerminate = 0;
}


/**
 * @brief Constructor for single-block transformer for use
 * @param x number of partial attentions in block
 * @param y number of attention in each partial attention
 * @param n number of tokens for each attention head
 * @param d dimension of each token
 * @param h height of MQ, MK and columns of MV, MH
 * @param l layers of mlp
 */
transformer::transformer(OpenCLContext& context, int x, int y, int n, int d, int h, int l, int vocab, bool attentionType):
    clcontext(context), m(1), x(x), y(y), n(n), d(d), h(h), l(l), isSelf(attentionType) 
{
    t = std::vector<block>(1, block(context, x, y, n, d, h, l, vocab, attentionType));
    // total permissible tokens = n
    tokens.resize(n, std::string(0));
    mTokens.resize(vocabsize, std::string(0));
    otok.resize(d, 0);
    embeddings.resize(vocabsize, std::vector<float>(d, 0));
    tokenEmbed.resize(n, std::vector<float>(d, 0));
    isSelf = attentionType;
    currentTokenCount = 0;
    blockCount = 1;
    promptCount = 0;
    indexForToken = 0;
    isTerminate = 0;
}


/**
 * @brief Constructor for many-block transformer for training
 * @param m number of blocks in transformer
 * @param x number of partial attentions in block
 * @param y number of attention in each partial attention
 * @param n number of tokens for each attention head
 * @param d dimension of each token
 * @param h height of MQ, MK and columns of MV, MH
 * @param l layers of mlp
 */
transformer::transformer(OpenCLContext& context, int m, int x, int y, int n, int d, int h, int l, int vocab, bool attentionType) :
    clcontext(context), m(m), x(x), y(y), n(n), d(d), h(h), l(l), isSelf(attentionType) 
{
    t = std::vector<block>(m, block(context, x, y, n, d, h, l, vocab, attentionType, inTraining));
    tokens.resize(n*m, std::string(0));
    mTokens.resize(vocabsize, std::string(0));
    otok.resize(d, 0);
    embeddings.resize(vocabsize, std::vector<float>(d, 0));
    tokenEmbed.resize(n * m, std::vector<float>(d, 0));
    isSelf = attentionType;
    currentTokenCount = 0;
    blockCount = 1;
    promptCount = 0;
    indexForToken = 0;
    isTerminate = 0;
}


/**
 * @brief constructor for transformer when case is defined
 * @param m number of blocks in trained transformer
 * @param x number of layers of PA
 * @param y number of attention heads in PA
 * @param n context window for single head
 * @param h height of matrices
 * @param l layer of MLPs
 * @param vocab vocabulary size
 * @param attentionType self (1) or cross (0) attention
 * @param inTraining training (TRUE) or use (FALSE)
 */
transformer::transformer(OpenCLContext& context, int m, int x, int y, int n, int d, int h, int l, int vocab, bool attentionType, bool& inTraining) :
    clcontext(context), m(m), x(x), y(y), n(n), d(d), h(h), l(l), vocabsize(vocab), isSelf(attentionType) 
{
    if(inTraining == true || inTraining == 1) {
        // training
        t = std::vector<block>(m, block(context, x, y, n, d, h, l, vocab, attentionType, inTraining));
        tokens.resize(n*m, std::string(0));
        mTokens.resize(vocabsize, std::string(0));
        otok.resize(d, 0);
        embeddings.resize(vocabsize, std::vector<float>(d, 0));
        tokenEmbed.resize(n * m, std::vector<float>(d, 0));
        isSelf = attentionType;
        currentTokenCount = 0;
        blockCount = 1;
        promptCount = 0;
        indexForToken = 0;
        isTerminate = 0;
    }
    else {
        // for inference
        t = std::vector<block>(1, block(context, x, y, n, d, h, l, vocab, attentionType, inTraining));
        tokens.resize(n*m, std::string(0));
        mTokens.resize(vocabsize, std::string(0));
        otok.resize(d, 0);
        embeddings.resize(vocabsize, std::vector<float>(d, 0));
        tokenEmbed.resize(n * m, std::vector<float>(d, 0));
        EVuse.resize(x, std::vector<std::vector<std::vector<float>>>(y, std::vector<std::vector<float>>(n, std::vector<float>(d, 0))));
        tokForBlock.resize(CONTEXT_WIN, std::vector<float>(d, 0));
        isSelf = attentionType;
        currentTokenCount = 0;
        blockCount = 1;
        promptCount = 0;
        indexForToken = 0;
        isTerminate = 0;
    }
}

#endif

/**
 * @brief set all the dimension for transformer
 * @param m number of blocks in transformer
 * @param x number of partial attentions in block
 * @param y number of attention in each partial attention
 * @param n number of tokens for each attention head
 * @param d dimension of each token
 * @param h height of MQ, MK and columns of MV, MH
 * @param l layers of mlp
 */
void transformer::setDims(int m, int x, int y, int n, int d, int h, int l) {
    this->m = m;
    this->x = x;
    this->y = y;
    this->n = n;
    this->d = d;
    this->h = h;
    this->l = l;
}

/**
 * @brief set learning rate for MLPs
 * @param learning learning rate
 */
void transformer::setLearning(float learning) {
    this->learning = learning;
}

/**
 * @brief set training cycle for training
 * @param epochs training cycle
 */
void transformer::setEpochs(int epochs) {
    //
}

/**
 * @brief set attention type for transformer
 * @param attentionType type of attention (1 for self and 0 for cross) 
 */
void transformer::setAttention(bool attentionType) {
    this->isSelf = attentionType;
}
