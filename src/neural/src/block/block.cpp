
#include "include/attention.hpp"
#include "include/block.hpp"

/**
 * @brief Constructor for complete attention block (default: self attention, training)
 * @param x number of partial attentions in block
 * @param y number of attention in each partial attention
 * @param n number of tokens for each attention head
 * @param d dimension of each token
 * @param h height of MQ, MK and columns of MV, MH
 * @param l layers of mlp
 * @param vocab vocabulary size
 */
block::block(int x, int y, int n, int d, int h, int l, int vocab) : x(x), y(y)
{
    // initialize attention block (complete attention)
    b.resize(x, std::vector<attention>(y, attention(n, d, h, l)));
    // probabiltiy vector
    probability.resize(vocab, 0.0f);
    // collection of vertical retention vectors from all heads
    EV.resize(x, std::vector<std::vector<std::vector<float>>>(y, std::vector<std::vector<float>>(n, std::vector<float>(d, 0.0f))));
    isSelfAttention = 1;    // default
    inTraining = 1;         // default
    tokForBlock.resize(n, std::vector<float>(d, 0.0f));
}


/**
 * @brief Constructor for complete attention block (default: training)
 * @param x number of partial attentions in block
 * @param y number of attention in each partial attention
 * @param n number of tokens for each attention head
 * @param d dimension of each token
 * @param h height of MQ, MK and columns of MV, MH
 * @param l layers of mlp
 * @param vocab vocabulary size
 * @param attentionType attention type of heads, 1 if self and 0 if cross
 */
block::block(int x, int y, int n, int d, int h, int l, int vocab, bool attentionType) : x(x), y(y)
{
    // initialize attention block (complete attention)
    b.resize(x, std::vector<attention>(y, attention(n, d, h, l, attentionType)));
    // probabiltiy vector
    probability.resize(vocab, 0.0f);
    // collection of vertical retention vectors from all heads
    EV.resize(x, std::vector<std::vector<std::vector<float>>>(y, std::vector<std::vector<float>>(n, std::vector<float>(d, 0.0f))));
    isSelfAttention = attentionType;
    inTraining = 1;     // default
    tokForBlock.resize(n, std::vector<float>(d, 0.0f));
}


/**
 * @brief Constructor for complete attention block
 * @param x number of partial attentions in block
 * @param y number of attention in each partial attention
 * @param n number of tokens for each attention head
 * @param d dimension of each token
 * @param h height of MQ, MK and columns of MV, MH
 * @param l layers of mlp
 * @param vocab vocabulary size
 * @param attentionType attention type of heads, 1 if self and 0 if cross
 */
block::block(int x, int y, int n, int d, int h, int l, int vocab, bool attentionType, bool inTraining) : x(x), y(y) 
{
    // initialize attention block (complete attention)
    b.resize(x, std::vector<attention>(y, attention(n, d, h, l, attentionType, inTraining)));
    // probabiltiy vector
    probability.resize(vocab, 0.0f);
    // collection of vertical retention vectors from all heads
    EV.resize(x, std::vector<std::vector<std::vector<float>>>(y, std::vector<std::vector<float>>(n, std::vector<float>(d, 0.0f))));
    isSelfAttention = attentionType;
    this->inTraining = inTraining;
    tokForBlock.resize(n, std::vector<float>(d, 0.0f));
}


/**
 * @brief set vertical retention vectors of heads to blocks in single vector
 * @param EV shared space for vertical retention vectors of all heads of single block
 */
void block::setVerticalRetention(std::vector<std::vector<std::vector<std::vector<float>>>> &EV)
{
    // complete block
    for(int i = 0; i < x; i++) {
        // layers of partial attention
        for(int j = 0; j < y; j++) {
            // attention heads of each partial attention
            for(int k = 0; k < CONTEXT_WIN; k++) {
                // each vertical retention vector of head
                EV[i][j][k] = b[i][j].EV[k];
            }
        }
    }
}
