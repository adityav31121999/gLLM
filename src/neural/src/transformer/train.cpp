
#include "include/attention.hpp"
#include "include/block.hpp"
#include "include/transformer.hpp"
#include "transformer.hpp"

/**
 * @brief compute the prediction for possible token embedding output
 * @param output forward propagation from block: EH
 * @param embeddings token embeddings
 * @param voc size of token vocabulary
 * @param index position of highest probability token embedding
 */
void transformer::computeOutput(std::vector<float>& output, std::vector<std::vector<float>>& embeddings, int & voc, 
    int & index)
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

/**
 * @brief train the transformer (single token training)
 * @param promptCount number of tokens in the prompt
 * @param currentTokenCount number of tokens in the full context
 * @param blockCount current block index
 * @param isSelf attention type
 * @param expected expected token embedding
 */
void transformer::train(int& promptCount, int& currentTokenCount, int& blockCount, bool& isSelf, std::vector<float>& expected) 
{
    if(blockCount == 0) {
        // compute the kdotQ for each head of the block
        computeKdotQs(promptCount, currentTokenCount, blockCount, isSelf);
        int i = 0;
        forward(blockCount, currentTokenCount, promptCount);
        while (i <= epochs) {
            if(errorofv(t[0].EH, expected) < 0.01) {
                input[currentTokenCount] = t[0].EH;
                break;
            }
            // if error is not corrected even after epochs, then increase epochs
            if(errorofv(t[0].EH, expected) > 0.01 && i == epochs) {
                // check token: if its is similar to expected then break the loop and set it to input vector
                // else increase epochs by 10
                // If the similarity (dot product) between current output and expected is above threshold,
                // consider it close enough to accept and stop training for this token
                if((std::inner_product(t[0].EH.begin(), t[0].EH.end(), expected.begin(), 0.0f) > 0.01)) 
                {
                    input[currentTokenCount] = t[0].EH;
                    break;
                }
                epochs += 10;
            }
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
            if(errorofv(t[blockCount-1].EH, expected) < 0.01) {
                input[currentTokenCount] = t[blockCount-1].EH;
                break;
            }
            // if error is not corrected even after epochs, then increase epochs
            if(errorofv(t[blockCount-1].EH, expected) > 0.01 && i == epochs) {
                // check token: if its is similar to expected then break the loop and set it to input vector
                // else increase epochs by 10
                // If the similarity (dot product) between current output and expected is above threshold,
                // consider it close enough to accept and stop training for this token
                if((std::inner_product(t[blockCount-1].EH.begin(), t[blockCount-1].EH.end(), expected.begin(), 0.0f) > 0.01)) 
                {
                    input[currentTokenCount] = t[blockCount-1].EH;
                    break;
                }
                epochs += 10;
            }
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
            // if error is not corrected even after epochs, then increase epochs
            if(err > 0.01 && i == epochs) {
                // check token: if its is similar to expected then break the loop and set it to input vector
                // else increase epochs by 10
                // If the similarity (dot product) between current output and expected is above threshold,
                // consider it close enough to accept and stop training for this token
                epochs += 10;
            }
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
            // if error is not corrected even after epochs, then increase epochs
            if(err > 0.01 && i == epochs) {
                // check token: if its is similar to expected then break the loop and set it to input vector
                // else increase epochs by 10
                // If the similarity (dot product) between current output and expected is above threshold,
                // consider it close enough to accept and stop training for this token
                epochs += 10;
            }
            i++;
        }
        trainCount++;
        epochCount += i;
        error += err;
    }
}


/**
 * @brief train the transformer (single sentences/long paragraphs)
 * @param sentence sentence/para to train
 */
void transformer::train(std::vector<std::vector<float>>& sentence) {
    // compute KdotQ for first element
    input[0] = sentence[0];
    promptCount = 1;
    blockCount = 1;
    computeKdotQs(promptCount, currentTokenCount, blockCount, isSelf);
    // keep this in a loop and train for each token in the sentence
    for(int i = 0; i < sentence.size(); i++) {
        train(promptCount, currentTokenCount, blockCount, isSelf, sentence[i]);
    }
}

/**
 * @brief train the transformer (multi-sentences/long paragraphs)
 * @param sentences sentences/paras to train
 */
void transformer::train(std::vector<std::vector<std::vector<float>>>& sentences) {
    // compute KdotQ for first sentence
    for(int i = 0; i < sentences[0].size(); i++) {
        input[i] = sentences[0][i];
    }
    promptCount = sentences[0].size();
    currentTokenCount = 0;
    blockCount = 1;
    computeKdotQs(promptCount, currentTokenCount, blockCount, isSelf);
    // keep this in a loop and train for each token in the sentence
    for(int i = 1; i < sentences.size(); i++) {
        for(int j = 0; j < sentences[i].size(); j++) {
            train(promptCount, currentTokenCount, blockCount, isSelf, sentences[i][j]);
        }
    }
}

/**
 * @brief train the transformer (single prompt and response)
 * @param prompt prompt to model
 * @param response response from model
 */
void transformer::train(std::vector<std::vector<float>>& prompt, std::vector<std::vector<float>>& response) {
    for(int i = 0; i < prompt[0].size(); i++) {
        input[i] = prompt[i];
    }
    promptCount = prompt[0].size();
    computeKdotQs(promptCount, currentTokenCount, blockCount, isSelf);
    // keep this in a loop and train for each token in the sentence
    for(int i = 0; i < response.size(); i++) {
        train(promptCount, currentTokenCount, blockCount, isSelf, response[i]);
    }
}

/**
 * @brief train the transformer (multiple prompts and responses)
 * @param prompts prompts to model
 * @param responses responses from model
 */
void transformer::train(std::vector<std::vector<std::vector<float>>>& prompts, std::vector<std::vector<std::vector<float>>>& responses) {
    if(prompts.size() == responses.size()) {
        for(int i = 0; i < prompts.size(); i++) {
            for(int j = 0; j < prompts[i].size(); j++) {
                input[j] = prompts[i][j];
            }
            promptCount = prompts[i].size();
            computeKdotQs(promptCount, currentTokenCount, blockCount, isSelf);
            // keep this in a loop and train for each token in the sentence
            for(int j = 0; j < responses[i].size(); j++) {
                train(promptCount, currentTokenCount, blockCount, isSelf, responses[i][j]);
            }
        }
    }
    else {
        std::cout << "Error: each prompt should have a response :<" << std::endl;
    }
}
