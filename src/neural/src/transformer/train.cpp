
#include "include/attention.hpp"
#include "include/block.hpp"
#include "include/transformer.hpp"
#include "transformer.hpp"

/**
 * @brief train the transformer (single token training)
 * @param promptCount number of tokens in the prompt
 * @param currentTokenCount number of tokens in the full context
 * @param blockCount current block index
 * @param isSelf attention type
 * @param expected expected token embedding
 */
void transformer::train(int& promptCount, int& currentTokenCount, int& blockCount, bool& isSelf, std::vector<float>& expected) {
    if(blockCount == 0) {
        // compute the kdotQ for each head of the block
        computeKdotQs(promptCount, currentTokenCount, blockCount, isSelf);
        int i = 0;
        forward(blockCount, currentTokenCount, promptCount);
        while (i < epochs) {
            if(errorofv(t[0].EH, expected) < 0.01)
                break;
            backward(expected);
            forward(blockCount, currentTokenCount, promptCount);
            i++;
        }
        trainCount++;
        epochCount += i;
        error += errorofv(t[0].EH, expected);
    }
    else {
        // compute the KdotQ for each head of block using EVs of previous blocks
        computeKdotQs(promptCount, currentTokenCount, blockCount, isSelf);
        forward(blockCount, currentTokenCount, promptCount);
        int i = 0;
        while (i < epochs) {
            if(errorofv(t[blockCount-1].EH, expected) < 0.01)
                break;
            i++;
            backward(expected, blockCount);
            forward(blockCount, currentTokenCount, promptCount);
        }
        trainCount++;
        epochCount += i;
        error += errorofv(t[blockCount-1].EH, expected);
    }
}


/**
 * @brief train the transformer (multi-token training, for chunk of tokens)
 * @param promptCount number of tokens in the prompt
 * @param currentTokenCount number of tokens in the full context
 * @param blockCount current block index
 * @param isSelf attention type
 * @param expected expected token embedding
 */
void transformer::train(int& promptCount, int& currentTokenCount, int& blockCount, bool& isSelf, std::vector<std::vector<float>>& expected) {
    if(blockCount == 0) {
        // compute the kdotQ for each head of the block
        computeKdotQs(promptCount, currentTokenCount, blockCount, isSelf);
        int i = 0;
        float err = 0.0f;
        while (i < epochs) {
            backward(expected);
            forward(blockCount, currentTokenCount, promptCount);
            for(int j = 0; j < x; j++) {
                err += errorofv(t[0].b[j][y-1].EH, expected[i]);
            }
            err /= x;
            if(err < 0.01)
                break;
            i++;
        }
        trainCount++;
        epochCount += i;
        error += err;
    }
    else {
        // compute the KdotQ for each head of block using EVs of previous blocks
        computeKdotQs(promptCount, currentTokenCount, blockCount, isSelf);
        int i = 0;
        float err = 0.0f;
        while (i < epochs) {
            backward(expected, blockCount);
            forward(blockCount, currentTokenCount, promptCount);
            for(int j = 0; j < x; j++) {
                err += errorofv(t[0].b[j][y-1].EH, expected[i]);
            }
            err /= x;
            if(err < 0.01)
                break;
            i++;
        }
        trainCount++;
        epochCount += i;
        error += err;
    }
}


/**
 * @brief compute the prediction for possible token embedding output
 * @param output forward propagation from block: EH
 * @param embeddings token embeddings
 * @param voc size of token vocabulary
 * @param index position of highest probability token embedding
 */
void transformer::computeOutput(std::vector<float>& output, std::vector<std::vector<float>>& embeddings, int & voc, int & index)
{
    std::vector<float> pred(voc, 0.0f);     // predictions
    for(int i = 0; i < voc; i++) {
        pred[i] = std::inner_product(output.begin(), output.end(), embeddings[i].begin(), 0.0f); // dot product
    }
    // find the highest value in the pred vector
    float max = pred[0];
    index = 0;
    for(int i = 1; i < voc; i++) {
        if(pred[i] > max) {
            max = pred[i];
            index = i;
        }
    }
}
