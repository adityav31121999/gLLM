#include "include/transformer.hpp"
#include <cmath> // For std::cos
#include <limits>
#include <numeric>
#include <algorithm>

#ifndef M_PI
    #define M_PI 3.14159
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
    m = m;
    x = x;
    y = y;
    n = n;
    d = d;
    h = h;
    l = l;
}

/**
 * @brief set learning rate for MLPs
 * @param learning learning rate
 */
void transformer::setLearning(float learning) {
    learning = learning;
}

/**
 * @brief set training cycle for training
 * @param epochs training cycle
 */
void transformer::setEpochs(int epochs) {
    epochs = epochs;
}

/**
 * @brief set attention type for transformer
 * @param attentionType type of attention (1 for self and 0 for cross) 
 */
void transformer::setAttention(bool attentionType) {
    isSelf = attentionType;
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
 * @brief experimental: Smoothly adjusts the learning rate based on error changes.
 * @param current_error Current epoch's error for token prediction.
 * @param prev_error Previous epoch's error for token prediction.
 * @param current_lr The current learning rate.
 * @param initial_lr The initial learning rate at the start of training.
 * @param epochs The number of epochs completed so far.
 * @return The adjusted learning rate, smoothed based on error trends.
 */
float transformer::smoothLearningRate(float current_error, float prev_error, float current_lr, float initial_lr, int epochs)
{
    // get difference in error
    float errordif = current_error - prev_error;
    // check percentage change in error
    float del_lr = (current_lr - initial_lr)/initial_lr;
    // 
    return 0.0f;
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


float transformer::adaptiveLearningRate(float current_error, float prev_error, int epochs, float current_lr, float min_lr, float max_lr) {
    float del = current_error - prev_error;
    if (del > 0.1) { // Significant loss increase
        return std::max<float>(current_lr * 0.95f, min_lr);     // Smoother reduction
    } else if (std::fabs(del) < 0.02 && epochs > 5) { // Slow progress
        return std::min<float>(current_lr * 1.025f, max_lr);    // Moderate increase
    } else if (del < -0.02 && epochs > 5) { // Steady decrease
        return std::min<float>(current_lr * 1.05f, max_lr);     // Gradual increase
    } else if (std::fabs(del) < 0.05 && epochs > 10) { // Stagnation
        return std::min<float>(current_lr * 1.075f, max_lr);    // Nudge upward
    }
    return std::max<float>(current_lr, 0.00005f); // Prevent overly small rates
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
    return min_lr + (0.5f * (max_lr - min_lr) * (1.0f + std::cos(static_cast<float>(current_epoch % total_epochs) / (total_epochs - 1) * M_PI)));
}


/**
 * @brief Calculates the value of a periodic function similar to sin(x).
 *      This function models f(x) = -((v - u) / 2) * sin(x) + (v + u) / 2.
 *      It is designed to have a minimum value 'u' at x = PI/2 and a maximum
 *      value 'v' at x = 3*PI/2. The value at x = 0 will be the midline (u+v)/2.
 *      For learning rate over the epochs.
 * @param x The input value (angle in radians).
 * @param u The desired minimum value of the function.
 * @param v The desired maximum value of the function.
 * @param val The desired value of the function at 0 radians.
 * @return The calculated value of the function at x.
 */
float transformer::periodicLearning(float x, float u, float v, float val) {
    // --- Input Validation ---
    // The value 'k' must be between the minimum (u) and maximum (v).
    // We add a small epsilon for floating-point comparison.
    if (val < u - 1e-6f || val > v + 1e-6f) {
        std::cerr << "Error: The value 'k' (" << val
                  << ") must be between u (" << u
                  << ") and v (" << v << ")." << std::endl;
        return std::numeric_limits<float>::quiet_NaN(); // Return Not-a-Number
    }

    // Amplitude of the function
    float amplitude = (v - u) / 2.0f;

    // Vertical shift (midline) of the function
    float midline = (u + v) / 2.0f;

    // Calculate the argument for arcsin to find the phase shift.
    // This is equivalent to (k - midline) / amplitude
    float arcsin_arg = (2.0f * val - u - v) / (v - u);

    // Clamp the argument to [-1, 1] to prevent domain errors from float inaccuracy
    if (arcsin_arg > 1.0f) arcsin_arg = 1.0f;
    if (arcsin_arg < -1.0f) arcsin_arg = -1.0f;

    // Calculate the phase shift 'C' needed to make f(0) = k
    float phase_shift_C = std::asin(arcsin_arg);

    // The final function uses this shift 'C'
    return -amplitude * std::sin(x - phase_shift_C) + midline;
}


/**
 * @brief Softsign-based adaptive learning rate function. (Modified for more responsive behavior)
 * @details Computes an adaptive learning rate based on the error difference (current - previous).
 * Uses a softsign function to smoothly adjust the learning rate. This version is tuned to be
 * more aggressive in its adaptations based on performance.
 * @param errordif Difference in current and previous error (positive if error increased).
 * @param currentLearning Current epoch's learning rate.
 * @return New adapted learning rate, bounded by LEARNING_MIN and LEARNING_MAX.
 */
float transformer::softsignLearning(float errordif, float currentLearning) {
    // for extremes
    if(currentLearning < LEARNING_MIN) return LEARNING_MIN*10.0f;
    if(currentLearning > LEARNING_MAX) return LEARNING_MAX*0.9f;

    // MODIFIED: Increased scale for sensitivity, reduced momentum for responsiveness.
    const float scale = 50.0f;
    const float momentum = 0.70f;
    const float tolerance = 1e-5f;
    // MODIFIED: Increased adjustment factors for more aggressive changes.
    const float reduction_factor = 0.85f; // Penalize error increases more heavily
    const float increase_factor = 1.15f;  // Reward error decreases more strongly

    // Compute softsign scaling factor
    float del = scale * (std::abs(errordif) < tolerance ? 0.0f : errordif);
    float f = del / (1.0f + std::abs(del)); // Maps to (-1, 1)

    // Adjust learning rate: reduce for f > 0 (error increase), increase for f < 0 (error decrease)
    float adjustment = (f > 0.0f) ? reduction_factor * f : increase_factor * f;
    float newLr = currentLearning * (1.0f - adjustment);

    // Apply momentum to smooth updates
    static float prevLr = currentLearning;
    newLr = (momentum * prevLr) + ((1.0f - momentum) * newLr);

    // Clamp to [LEARNING_MIN, LEARNING_MAX]
    newLr = std::min<float>(LEARNING_MAX, std::max<float>(LEARNING_MIN, newLr));
    prevLr = newLr; // Update static previous learning rate for the next call
    
    return newLr;
}

/**
 * @brief Softsign-based adaptive learning rate function. (Modified for stability and recovery)
 * @details Computes an adaptive learning rate based on the error difference. This version includes
 * a hard reset mechanism to handle catastrophic divergence (inf/NaN loss) and uses more
 * conservative parameters to prevent the learning rate from growing too aggressively.
 * @param currentError The current epoch's error/loss value.
 * @param currentLearning Current epoch's learning rate.
 * @param initialLearning The initial learning rate specified at the start of training.
 * @return New adapted learning rate, bounded by LEARNING_MIN and LEARNING_MAX.
 */
float transformer::softsignLearning(float currentError, float currentLearning, float initialLearning) {
    // Static variables to hold state between calls
    static float prevError = -1.0f;
    static float prevLr = currentLearning;

    // --- Emergency Reset for Catastrophic Divergence ---
    // If loss is infinity or NaN, the model has exploded. Reset LR to a safe value.
    if (std::isinf(currentError) || std::isnan(currentError)) {
        prevError = -1.0f; // Reset error history to re-initialize on next valid run
        prevLr = initialLearning;
        return initialLearning; // Return a safe, initial learning rate to recover
    }

    // Initialize prevError on the first valid run after a reset or at the beginning
    if (prevError < 0.0f) {
        prevError = currentError;
        prevLr = currentLearning;
        return currentLearning;
    }
    
    // Boundary checks for safety
    if(currentLearning < LEARNING_MIN) return LEARNING_MIN * 1.1f;
    if(currentLearning > LEARNING_MAX) return LEARNING_MAX * 0.9f;

    // MODIFIED: More conservative parameters to prevent rapid LR growth
    const float scale = 40.0f;          // A balanced scale
    const float momentum = 0.75f;         // Increased momentum for smoother, less jerky changes
    const float tolerance = 1e-6f;
    const float reduction_factor = 0.9f;  // Keep reduction strong to punish failure
    const float increase_factor = 0.5f;   // SIGNIFICANTLY reduce the increase factor to prevent over-excitement

    // Compute error difference
    float errordif = currentError - prevError;

    // Compute softsign scaling factor
    float del = scale * (std::abs(errordif) < tolerance ? 0.0f : errordif);
    float f = del / (1.0f + std::abs(del)); // Maps to (-1, 1)

    // Adjust learning rate: reduce for f > 0 (error increase), increase for f < 0 (error decrease)
    float adjustment = (f > 0.0f) ? reduction_factor * f : increase_factor * f;
    float newLr = currentLearning * (1.0f - adjustment);

    // Apply momentum to smooth updates
    newLr = (momentum * prevLr) + ((1.0f - momentum) * newLr);

    // Clamp to [LEARNING_MIN, LEARNING_MAX]
    newLr = std::min<float>(LEARNING_MAX, std::max<float>(LEARNING_MIN, newLr));
    
    // Update state for the next call
    prevError = currentError;
    prevLr = newLr;
    
    return newLr;
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