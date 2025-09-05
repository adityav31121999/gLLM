
#include "include/transformer.hpp"

/**
 * @brief update learning rate when error increases
 * @param pErr previous iteration's error
 * @param cErr current iteration's error
 */
void transformer::updateLearning(float& pErr, float& cErr) {
    //
}

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
    this->epochs = epochs;
}

/**
 * @brief set attention type for transformer
 * @param attentionType type of attention (1 for self and 0 for cross) 
 */
void transformer::setAttention(bool attentionType) {
    this->isSelf = attentionType;
}

void transformer::clearValues() {
    // Clear string vectors
    tokens.clear();
    mTokens.clear();

    // Clear 1D float vector and set to 0
    std::fill(otok.begin(), otok.end(), 0.0f);

    // Clear mat objects using memset
    if (embeddings.mapped_data && embeddings.mapped_size > 0)
        memset(embeddings.mapped_data, 0, embeddings.mapped_size);
    if (tokenEmbed.mapped_data && tokenEmbed.mapped_size > 0)
        memset(tokenEmbed.mapped_data, 0, tokenEmbed.mapped_size);
    if (tokForBlock.mapped_data && tokForBlock.mapped_size > 0)
        memset(tokForBlock.mapped_data, 0, tokForBlock.mapped_size);

    // Clear 4D float vector (EVuse) and set to 0
    for (auto& dim1 : EVuse) {
        for (auto& dim2 : dim1) {
            for (auto& dim3 : dim2) {
                std::fill(dim3.begin(), dim3.end(), 0.0f);
            }
        }
    }
    std::cout << "<====Transformer values cleared====>" << std::endl;
}


/**
 * @brief function for adaptive learning rate
 * @param[in] prev_Error previous epochs error
 * @param[in] current_Error current epochs error
 * @param[in] learning current learning rate
 * @param[in] epochCount current epoch of prediction
 * @return new learning rate
 */ 
float transformer::adaptiveLearningOptimiser(float prev_Error, float current_Error, float learning, int epochCount)
{
    // adaptive learning with global macros
    // #define LEARNING_MAX 0.1         // maximum learning rate allowable
    // #define LEARNING_MIN 0.00001     // minimum learning rate allowable
    // update learning rate starting from second epoch and specific conditions
    float learningNew = 0.0f;
    if(epochCount > 1) {
        if(current_Error <= prev_Error) {
            if(epochCount <= 6)   
                learningNew = learning * 1.1;
            else if (epochCount % 6 == 0)
                learningNew = learning * (1.05 + (epochCount/6)*0.05);
        }
        else {
            if(epochCount <= 6)   
                learningNew = learning * 0.95;
            else if (epochCount % 6 == 0)
                learningNew = learning * (0.95 - (epochCount/6)*0.01);
        }
        if(learningNew > LEARNING_MAX)
            learningNew = LEARNING_MAX;
        if(learningNew < LEARNING_MIN)
            learningNew = LEARNING_MIN;
    }
    return learningNew;
}