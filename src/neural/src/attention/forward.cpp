
#ifdef USE_CPU

#include <vector>
#include <cmath> // For std::abs, std::min
#include <stdexcept> // For error handling if needed
#include "include/attention.hpp"
#include <maths.hpp>

/**
 * @brief forward propagation for first block's attention class (incomplete attention)
 * @param in embedding dimension
 * @param layers layers of hidden weights in mlp
 * @param tokenCount token count for each attention head (how many tokens have been generated or taken as input)
 */
void attention::forprop(int& in, int& layers, int& tokenCount)
{
    if (tokenCount <= 0 || tokenCount > KdotQ.row || tokenCount > KdotQ.col) {
        return;
    }
    if (EH.size() != EMBEDDING || dh.size() != EMBEDDING || dv.size() != EMBEDDING) {
        return;
    }

    // probability distribution
    float k_sum, l_sum;
    mat head = LOTA(KdotQ, tokenCount, isSelfAttention);

    dh.assign(EMBEDDING, 0.0f);
    dv.assign(EMBEDDING, 0.0f);

    for(int i = 0; i < tokenCount; ++i) {
        k_sum = 0.0f;   // row sum
        l_sum = 0.0f;   // column sum
        int limit_j = isSelfAttention ? (i + 1) : tokenCount;
        limit_j = std::min(limit_j, head.col);
        for(int j = 0; j < limit_j; ++j) {
            k_sum += head(i, j);
            if (j < head.row && i < head.col) {
                l_sum += head(j, i);
            }
        }
        // Ensure K and Q have enough rows before accessing
        if (i < K.row && i < Q.row) {
            dh = dh + (k_sum * getRow(K, i)); // Vector addition and scalar multiplication
            dv = dv + (l_sum * getRow(Q, i)); // Vector addition and scalar multiplication
        }
    }
 
    dh = dot(dh, MH);
    dv = dot(dv, MV);
    hor.input = EH + dh;
    for(int i = 0; i < tokenCount; ++i) {
        ver.input += getRow(EV, i);
    }
    ver.input += dv;

    hor.forward(in, layers);
    ver.forward(in, layers);

    // AND gate for the final output
    EH = EH + ReLU(hor.output); // Assumes EH is std::vector<float>
    for(int i = 0; i < CONTEXT_WIN; ++i) {
        EV(i) += ReLU(ver.output);
    }
}


/**
 * @brief forward propagation for a 2nd to last block's attention class (incomplete attention)
 * @param EVp EV matrix from previous block (passed by const reference)
 * @param in input token count
 * @param layers layers of MLPs
 * @param tokenCount number of tokens in full context
 * @param blockCount which block is being processed in full context
 * @param n number of tokens for each attention head (context window)
 */
void attention::forprop(const mat& EVp, int& in, int& layers, int& tokenCount, int& blockCount, int& n) // Changed EVp type to const mat&
{
    if(blockCount == 0) {
        int firstBlockTokenCount = std::min(tokenCount, n);
        forprop(in, layers, firstBlockTokenCount); // Call the first block version
        return;
    }

    int startTokenIndexInFullContext = (blockCount -1) * n; // blockCount is 1-based for subsequent blocks
    int endTokenIndexInFullContext = std::min(tokenCount, (blockCount) * n); // Corrected end index
    int currentBlockTokenCount = endTokenIndexInFullContext - startTokenIndexInFullContext;

    if (currentBlockTokenCount <= 0 || KdotQ.row == 0 || KdotQ.col == 0 || currentBlockTokenCount > KdotQ.row || currentBlockTokenCount > KdotQ.col) {
        return;
    }
    if (EH.size() != EMBEDDING || dh.size() != EMBEDDING || dv.size() != EMBEDDING) {
        return;
    }

    // probability distribution
    float k_sum, l_sum;
    mat head = LOTA(KdotQ, currentBlockTokenCount, isSelfAttention);
    dh.assign(EMBEDDING, 0.0f);
    dv.assign(EMBEDDING, 0.0f);
    // K, Q, KdotQ are for currentBlockTokenCount
    for(int i = 0; i < currentBlockTokenCount; ++i) {
        k_sum = 0.0f;   // row sum
        l_sum = 0.0f;   // column sum
        int limit_j = isSelfAttention ? (i + 1) : currentBlockTokenCount;
        limit_j = std::min(limit_j, head.col);
        for(int j = 0; j < limit_j; ++j) {
            k_sum += head(i, j);
            if (j < head.row && i < head.col) {
                l_sum += head(j, i);
            }
        }
        // K and Q here should be sized/sliced for currentBlockTokenCount
        // If K and Q are full context, an offset is needed.
        // Assuming K and Q members are already correctly populated for this block's 'currentBlockTokenCount'
        if (i < K.row && i < Q.row) { // K.row and Q.row should match currentBlockTokenCount
            dh = dh + (k_sum * getRow(K, i)); // Vector addition and scalar multiplication
            dv = dv + (l_sum * getRow(Q, i)); // Vector addition and scalar multiplication
        }
    }

    dh = dot(dh, MH);
    dv = dot(dv, MV);

    hor.input = EH + dh;

    // Use EVp from the previous block for ver.input
    ver.input.assign(EMBEDDING, 0.0f); // Clear previous ver.input
    // EVp.row should be totalTokenCount (passed as 'tokenCount' parameter to this function)
    // EVp.col should be EMBEDDING
    if (EVp.mapped_data && EVp.row > 0 && EVp.col == EMBEDDING) {
        for(int i = 0; i < EVp.row; ++i) { // Iterate up to EVp.row (which is totalTokenCount)
            ver.input += getRow(EVp, i);
        }
    } else if (tokenCount > 0 && EVp.row != tokenCount) { // If EVp is not valid but we expected tokens
        // Potentially log a warning or handle as an error if EVp is expected to be valid
        // For now, ver.input will just be dv if EVp is not usable.
         throw std::runtime_error("EVp dimension mismatch in attention::forprop. Expected rows: " + std::to_string(tokenCount) + ", got: " + std::to_string(EVp.row));
    }
    ver.input += dv;

    hor.forward(in, layers);
    ver.forward(in, layers);

    // AND gate for the final output
    EH = EH + ReLU(hor.output); // Assumes EH is std::vector<float>
    // Update this head's EV (this->EV) for the current block's tokens
    // this->EV should be sized for CONTEXT_WIN or at least currentBlockTokenCount
    for(int i = 0; i < CONTEXT_WIN; ++i) {
        if (i < EV.row) { // Ensure EV is large enough
            EV(i) += ReLU(ver.output); // Update the i-th row of *this* block's EV
        }
    }
}

#endif
