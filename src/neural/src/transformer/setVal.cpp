#include "include/transformer.hpp"
#include <cmath> // For std::cos
#include <limits>
#include <numeric>
#include <algorithm>

#ifndef M_PI
// Define M_PI
    #define M_PI 3.141592653
#endif

#define WARMUP_EPOCHS 5

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
    this->m = m;        // number of blocks
    this->x = x;        // number of layers in block
    this->y = y;        // number of heads in each layer
    this->n = n;        // number of tokens for each head
    this->d = d;        // dimension of each token
    this->h = h;        // height of MQ, MK and columns of MV, MH
    this->l = l;        // layers of mlp
}


/**
 * @brief Clear values in transformer to avoid data leakage between training iterations.
 * This function resets string vectors, float vectors, and mat objects to their initial
 * 0 states.
 */
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
 * @brief Adjusts the learning rate when a metric stops improving (plateau detection).
 * @param current_error Current epoch's error for token prediction.
 * @param previous_error Previous epoch's error for token prediction.
 * @param learning_rate The current learning rate (modified by reference).
 * @param factor Factor by which the learning rate is reduced (new_lr = lr * factor).
 * @param max_lr Upper bound for the learning rate.
 * @param min_lr Lower bound for the learning rate.
 * @param epsilon Small threshold to detect significant error improvement.
 * @param patience Number of epochs to wait before reducing the learning rate.
 * @return The adjusted learning rate, constrained between min_lr and max_lr.
 */
float transformer::adaptiveLearningRateOnPlateau(float current_error, float previous_error, float& learning_rate, 
                                               float factor, float max_lr, float min_lr, float epsilon, int patience) 
{
    // Ensure learning rate is within bounds at the start
    learning_rate = std::min<float>(std::max<float>(learning_rate, min_lr), max_lr);

    // Check if error has improved significantly
    if ((100.0f * (previous_error + epsilon - current_error)/(previous_error + epsilon)) > 0.1f) {
        // Error improved, reset plateau counter
        plateau_count = 0;
        std::cout << "Error improved. Current error: " << current_error << ", Previous error: " << previous_error << std::endl;
    }
    else {
        // Error did not improve, increment plateau counter
        plateau_count++;
        // std::cout << "No significant improvement. Plateau count: " << plateau_count << ", Current error: " << current_error << std::endl;
    }

    // Reduce learning rate if plateau persists and learning rate is above minimum
    if (plateau_count >= patience && learning_rate > min_lr) {
        float new_lr = std::max<float>(learning_rate * factor, min_lr);
        std::cout << "Plateau detected after " << patience << " epochs. Reducing learning rate from " << learning_rate 
                  << " to " << new_lr << std::endl;
        learning_rate = new_lr;
        plateau_count = 0; // Reset counter after adjustment
    }

    // Ensure learning rate stays within bounds
    return std::min<float>(learning_rate, max_lr);
}


/**
 * @brief Sets the learning rate of a parameter group using a cosine annealing schedule.
 * @param current_epoch The current epoch number (0-indexed).
 * @param total_epochs The total number of epochs in the cycle.
 * @param max_lr The maximum learning rate (initial learning rate).
 * @param min_lr The minimum learning rate.
 * @return The calculated learning rate for the current epoch.
 */
float transformer::cosineAnnealingLR(int current_epoch, int total_epochs, float max_lr, float min_lr) {
    if (total_epochs <= 1) {
        return max_lr;
    }
    // The `(total_epochs - 1)` ensures the cosine reaches its minimum at the last epoch.
    // return min_lr + (0.5f * (max_lr - min_lr) * (1.0f + std::cos(0.55f * static_cast<float>(current_epoch) / (total_epochs + 1))));
    return min_lr + (0.5f * (max_lr - min_lr) * (1.0f + std::cos(1.57 * static_cast<float>(current_epoch) / static_cast<float>(total_epochs-1))));
}


/**
 * @brief Softsign-based adaptive learning rate function
 * @param error_del current and previous epochs error difference.
 * @param currentLearning Current epoch's learning rate.
 * @return New adapted learning rate, bounded by LEARNING_MIN and LEARNING_MAX.
 */
float transformer::softsignLearning(float error_del, float currentLearning) {
if(currentLearning <= LEARNING_MIN) return 1.50f * LEARNING_MIN;
    if(currentLearning >= LEARNING_MAX) return 0.90f * LEARNING_MAX;
    if(error_del == 0) return currentLearning;
    // --- Softsign Adjustment ---
    float new_learning = 0.0f;
    new_learning = currentLearning * (1.0f - (softsign(error_del) * ((error_del > 0.001) ? ((error_del > 0.25 ) ? 0.05 : 1) : 50)));
    return std::min<float>(LEARNING_MAX, std::max<float>(new_learning, LEARNING_MIN));
}


/**
 * @brief Calculates an adaptive learning rate based on the prediction error.
 * This function takes the difference between the predicted and expected embeddings,
 * smooths this error signal, and clamps it within a safe range to prevent
 * training instability. A higher error will gently push the learning rate up,
 * while a lower error will bring it down.
 * @param pred A std::vector<float> representing token prediction.
 * @param exp A std::vector<float> representing expected output.
 * @param del The change in error from the previous step.
 * @param currentLearning current epochs learning rate.
 * @return A new, adjusted learning rate as a float.
 */
float transformer::errorGradLearning(const std::vector<float>& pred, const std::vector<float>& exp, const float del, float currentLearning) 
{
    // 1. --- Input Validation ---
    if (pred.size() != exp.size() || pred.empty()) {
        throw std::invalid_argument("Prediction and expectation vectors must be non-empty and of the same size.");
    }

    if(currentLearning <= LEARNING_MIN) return 1.10f * LEARNING_MIN;
    if(currentLearning >= LEARNING_MAX) return 0.95f * LEARNING_MAX;

    // --- Hyperparameter ---
    const float smoothing_factor = 5.0f;

    // 2. --- Calculate rms gradient + epsilon ---
    float squared_err_sum = 0.0f;
    for (size_t i = 0; i < pred.size(); ++i) {
        squared_err_sum += std::pow(pred[i] - exp[i], 2.0f);
    }
    // float d = std::sqrt(squared_err_sum / pred.size()) + 1E-10f;
    float d = std::sqrt(squared_err_sum / pred.size()) + 1E-10f;

    // 3. --- Smooth the Learning Rate Update ---
    float r = ((del < 0) ? 1.10f : -1.0f) / (smoothing_factor * d); // direction of error change
    float new_learning_rate = currentLearning * (1.0f + r);
    std::cout << "Error change (del): " << del << ", Gradient magnitude (d): " << d 
              << ", Adjustment factor (r): " << r << ", New learning rate (pre-clamp): " << new_learning_rate << std::endl;

    // 4. --- Clamp the Result (Safety Rails) ---
    return std::min<float>(LEARNING_MAX, std::max<float>(new_learning_rate, LEARNING_MIN));
}


/**
 * @brief provide positional embeddings
 * @param positin 0-based position of token in sequence
 * @param embeddingDimension dimension of token embeddings
 */
std::vector<float> transformer::positionalEmbeddings(int position, int embeddingDimension)
{
    std::vector<float> positional(embeddingDimension, 0.0f);
    // The loop iterates two steps at a time, calculating sin for even `i` and cos for odd `i+1`.
    for (int i = 0; i < embeddingDimension; i += 2) {
        float div_term = std::pow(100000.0f, static_cast<float>(i) / embeddingDimension);
        float angle = static_cast<float>(position) / div_term;

        positional[i] = std::sin(angle);
        if (i + 1 < embeddingDimension) {
            positional[i + 1] = std::cos(angle);
        }
    }
    return positional;
}


/**
 * @brief get indices of all tokens of lines
 * @param [in] tokensOfLines tokens of line
 * @param [out] indexVec index of all tokens
 */
void transformer::getIndexOfAllTokens(std::vector<std::string> &tokensOfLine, std::vector<int> &indexVec)
{
    for(int i = 0; i < tokensOfLine.size(); i++) {
        for(int j = 0; j < vocabsize; j++) {
            if(tokensOfLine[i] == tokens[j]) {
                indexVec[i] = j;
                break;
            }
        }
        // std::cout << "Found |" << tokensOfLine[i] << "| @ " << indexVec[i] << std::endl;
    }
    int k = 0;
    for(int i = 0; i < indexVec.size(); i++) {
        if(indexVec[i] >= 0) {
            k++;
        }
    }
    if(k == 0)
        throw std::runtime_error("No indinces found for tokens of line in vocabulary!");
    else {
        // std::cout << "Size of indexVec: " << indexVec.size() << std::endl;
        // std::cout << "All tokens found in vocabulary :)" << std::endl;
    }
}