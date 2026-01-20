#ifdef USE_CPU
// transformer training
#include "include/attention.hpp"
#include "include/block.hpp"
#include "include/transformer.hpp"

/**
 * @brief train the transformer on sentences (single continuous sentece, paragraphs and passages)
 * @param sentence token embedding of sentence
 * @param rString sentence tokens
 */
void transformer::train(std::vector<std::vector<float>>& sentence, std::vector<std::string>& rString)
{
    // constraints for training data
    // --- Basic validation ---
    if(sentence.size() + currentTokenCount > FULL_CONTEXT) {
        throw std::runtime_error("clTrain(sentence): Previous tokens and sentence will exceed the FULL CONTEXT '-'");
    }
    if (sentence.size() > FULL_CONTEXT) {
        throw std::runtime_error("clTrain(sentence): Sentence size (" + std::to_string(sentence.size()) + ") exceeds FULL_CONTEXT (" + std::to_string(FULL_CONTEXT) + ").");
    }
    if (sentence.empty() || sentence.size() != rString.size()) {
        std::cout << "sentence.size(): " << sentence.size() << ", rString.size(): " << rString.size() << std::endl;
        throw std::runtime_error("clTrain(sentence): Sentence embeddings/strings mismatch or empty.");
    }
    if (!sentence.empty() && sentence[0].size() != static_cast<size_t>(d)) {
        throw std::runtime_error("clTrain(sentence): Sentence embedding dimension mismatch. Expected " + std::to_string(d) + ", got " + std::to_string(sentence[0].size()));
    }

    float initial_learning_rate = learning;
    float current_error = 0.0f, prev_error = 0.0f;
    int initial_epochs = epochs;
    int initial_token_count = currentTokenCount;
    int effective_context_size = 0;
    bool blockShifted = 0;

    // --- Device Buffer Allocation & H->D Transfer ---
    size_t totalTokenEmbedFloats = static_cast<size_t>(m) * CONTEXT_WIN * d;
    size_t tokenEmbedBytes = totalTokenEmbedFloats * sizeof(float);

    try {
        otok.clear(); otok.resize(d, 0.0f);
        // --- set all tokens to tokenEmbed ---
        std::fill(tokenEmbed.mapped_data, tokenEmbed.mapped_data + totalTokenEmbedFloats, 0.0f);
        std::fill(positional.mapped_data, positional.mapped_data + totalTokenEmbedFloats, 0.0f);
        std::fill(embedPlusPos.mapped_data, embedPlusPos.mapped_data + totalTokenEmbedFloats, 0.0f);
        for(int i = 0; i < rString.size(); i++) {
            tokenEmbed.addRow(sentence[i], i);
            positional.addRow(positionalEmbeddings(i, d), i);
            // add the last token of local context to first token of next local
            if(i + 1 % CONTEXT_WIN == 0 && i + 1 < rString.size()) {
                tokenEmbed.addRow(sentence[i], i + 1);
                positional.addRow(positionalEmbeddings(i, d), i + 1);
                i++;
            }
        }
        effective_context_size += 1;
        currentTokenCount += 1;
        sequence1Count = 1, blockCount = 1;
        std::cout << "Predicted (Index) | CE Loss | del (cur - pre) | e^Loss | Epochs | Learning Rate" << std::endl;

        for (size_t i = 1; i < sentence.size(); ++i) {
            if (currentTokenCount >= FULL_CONTEXT) {
                std::cerr << "Warning: clTrain(sentence) reached FULL_CONTEXT limit ("
                          << currentTokenCount << "). Stopping training early at sentence index " << i << "." << std::endl;
                break;
            }
            int current_block_idx = blockCount;
            if (current_block_idx <= 0 || current_block_idx > m) {
                throw std::out_of_range("clTrain(sentence): Calculated current_block_idx (" + std::to_string(current_block_idx) + ") is out of range [1, " + std::to_string(m) + "].");
            }

            // Target token for this iteration
            std::vector<float>& expected_vec = sentence[i];
            std::string& expected_str = rString[i];
            size_t currentBytes = static_cast<size_t>(EMBEDDING) * effective_context_size * sizeof(float);

            std::cout << "Training token " << i+1 << "/" << sentence.size() << ": '" << expected_str << "'" << " at " << indexVec[i] << std::endl;
            std::cout << "current block: " << current_block_idx << " | current token count: " << currentTokenCount << " | eff. context size: " << effective_context_size <<std::endl;

            int j = 0; // epoch for each token
            while (j < epochs) {
                // --- K/Q Calculation ---
                embedPlusPos = tokenEmbed + positional;
                if(current_block_idx == 1) {
                    mat partialEP(CONTEXT_WIN, EMBEDDING);
                    const float* host_src_ptr = embedPlusPos.mapped_data;
                    std::copy(host_src_ptr, host_src_ptr + currentBytes/sizeof(float), partialEP.mapped_data);
                    for (int layer_idx = 0; layer_idx < x; ++layer_idx) {
                        for (int parallel_idx = 0; parallel_idx < y; ++parallel_idx) {
                            blocks[0].b[layer_idx][parallel_idx].getKeyQuery(partialEP, blocks[0].b[layer_idx][parallel_idx].MK, 1);
                            blocks[0].b[layer_idx][parallel_idx].getKeyQuery(partialEP, blocks[0].b[layer_idx][parallel_idx].MQ, 0);
                            blocks[0].b[layer_idx][parallel_idx].getKdotQ();
                        }
                    }
                }
                else { // Subsequent blocks
                    mat prevEVs(CONTEXT_WIN, d);
                    mat currentTokens(CONTEXT_WIN, d);
                    size_t fromHereInTokenEmbed = static_cast<size_t>((CONTEXT_WIN) * (blockCount - 1) - 1) * d;
                    const float* host_src_ptr = embedPlusPos.mapped_data + fromHereInTokenEmbed;
                    std::copy(host_src_ptr, host_src_ptr + currentBytes/sizeof(float), currentTokens.mapped_data);
                    for (int layer_idx = 0; layer_idx < x; ++layer_idx) {
                        for (int parallel_idx = 0; parallel_idx < y; ++parallel_idx) {
                            blocks[blockCount - 1].b[layer_idx][parallel_idx].getKeyQuery(blocks[blockCount - 2].b[layer_idx][parallel_idx].EV,
                                                                                          blocks[blockCount - 1].b[layer_idx][parallel_idx].MK, 1);
                            blocks[blockCount - 1].b[layer_idx][parallel_idx].getKeyQuery(currentTokens, blocks[blockCount - 1].b[layer_idx][parallel_idx].MQ, 1);
                            blocks[blockCount - 1].b[layer_idx][parallel_idx].getKdotQ();
                        }
                    }
                }

                // --- Forward Pass ---
                forward(current_block_idx, effective_context_size, sequence1Count);

                // --- Get EH output ---
                std::fill(otok.begin(), otok.end(), 0.0f);
                if (y > 0) {
                    for (int layer_idx = 0; layer_idx < x; ++layer_idx) {
                        for (int k = 0; k < d; ++k) {
                            otok[k] += blocks[blockCount-1].b[layer_idx][y - 1].EH[k];
                        }
                    }
                }
                for(size_t k_dim = 0; k_dim < static_cast<size_t>(d); k_dim++) {
                    if (std::isnan(otok[k_dim])) { otok[k_dim] = 0.0f; }
                    else if (std::isinf(otok[k_dim])) { otok[k_dim] = std::copysign((std::numeric_limits<float>::max)(), otok[k_dim]); }
                }

                // --- Prediction ---
                computePrediction();

                // --- Error Calculation & Logging ---
                current_error = binaryCrossEntropy(expected_vec, otok);
                float del = current_error - prev_error;
                std::string predicted_token_str = (indexForToken >= 0 && indexForToken < static_cast<unsigned long long>(tokens.size()))
                                                  ? tokens[indexForToken] : "INVALID_INDEX";

                std::cout << predicted_token_str << " ( " << indexForToken << " ) \t: "
                          << current_error << " | " << del << " | "
                          << std::exp(current_error) << " | " << j+1 << " | " << learning << std::endl;

                if (predicted_token_str == expected_str && predicted_token_str != "INVALID_INDEX") {
                    std::cout << "Token '" << expected_str << "' predicted correctly after "
                              << j+1 << " epochs. Moving to next token." << std::endl;
                    learning = initial_learning_rate;
                    if(predicted_token_str != "</s>")
                        std::cout << "              -------------- To Next Token --------------              " << std::endl;
                    break;
                }

                // --- Backward Pass ---
                backward(expected_vec, current_block_idx);

                totalLearning += learning;
                prev_error = current_error;
                totalBCELoss += current_error;
                totalBCEPerplexity += std::exp(current_error);
                j++;
            }

            // --- Update Host State ---
            trainCount++;
            epochCount += j;

            currentTokenCount++;
            effective_context_size++;
            if(currentTokenCount > 0 && currentTokenCount % CONTEXT_WIN == 0) {
                blockCount += 1;
                blockShifted = 1;
                tokenEmbed.addRow(sentence[i], currentTokenCount - 1); // repeat last token to new block
                positional.addRow(positionalEmbeddings(currentTokenCount - 1, d), currentTokenCount - 1);
                std::cout << "----> Going to Next block in model -> " << blockCount - 1 << " to " << blockCount << std::endl;
            } else {
                blockShifted = 0;
            }
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Standard Exception in clTrain(sentence): " << e.what() << std::endl;
        epochs = initial_epochs;
        throw;
    }
}



/**
 * @brief train the transformer on sentences (single continuous sentece, paragraphs and passages)
 * @param sentence token embedding of sentence
 * @param rString sentence tokens
 */
void transformer::trainContext(std::vector<std::vector<float>>& sentence, std::vector<std::string>& rString)
{
    // constraints for training data
    // --- Basic validation ---
    if(sentence.size() + currentTokenCount > FULL_CONTEXT) {
        throw std::runtime_error("clTrain(sentence): Previous tokens and sentence will exceed the FULL CONTEXT '-'");
    }
    if (sentence.size() > FULL_CONTEXT) {
        throw std::runtime_error("clTrain(sentence): Sentence size (" + std::to_string(sentence.size()) + ") exceeds FULL_CONTEXT (" + std::to_string(FULL_CONTEXT) + ").");
    }
    if (sentence.empty() || sentence.size() != rString.size()) {
        std::cout << "sentence.size(): " << sentence.size() << ", rString.size(): " << rString.size() << std::endl;
        throw std::runtime_error("clTrain(sentence): Sentence embeddings/strings mismatch or empty.");
    }
    if (!sentence.empty() && sentence[0].size() != static_cast<size_t>(d)) {
        throw std::runtime_error("clTrain(sentence): Sentence embedding dimension mismatch. Expected " + std::to_string(d) + ", got " + std::to_string(sentence[0].size()));
    }

    float initial_learning_rate = learning;
    float current_error = 0.0f, prev_error = 0.0f;
    int initial_epochs = epochs;
    int initial_token_count = currentTokenCount;
    int effective_context_size = 0;
    bool blockShifted = 0;

    // --- Device Buffer Allocation & H->D Transfer ---
    size_t totalTokenEmbedFloats = static_cast<size_t>(m) * CONTEXT_WIN * d;
    size_t tokenEmbedBytes = totalTokenEmbedFloats * sizeof(float);

    try {
        otok.clear(); otok.resize(d * x, 0.0f);
        // --- set all tokens to tokenEmbed ---
        std::fill(tokenEmbed.mapped_data, tokenEmbed.mapped_data + totalTokenEmbedFloats, 0.0f);
        std::fill(positional.mapped_data, positional.mapped_data + totalTokenEmbedFloats, 0.0f);
        std::fill(embedPlusPos.mapped_data, embedPlusPos.mapped_data + totalTokenEmbedFloats, 0.0f);
        for(int i = 0; i < rString.size(); i++) {
            tokenEmbed.addRow(sentence[i], i);
            positional.addRow(positionalEmbeddings(i, d), i);
            // add the last token of local context to first token of next local
            if(i + 1 % CONTEXT_WIN == 0 && i + 1 < rString.size()) {
                tokenEmbed.addRow(sentence[i], i + 1);
                positional.addRow(positionalEmbeddings(i, d), i + 1);
                i++;
            }
        }
        effective_context_size += 1;
        currentTokenCount += 1;
        sequence1Count = 1, blockCount = 1;
        std::cout << "Predicted (Index) | CE Loss | del (cur - pre) | e^Loss | Epochs | Learning Rate" << std::endl;

        for (size_t i = 1; i < sentence.size(); ++i) {
            if (currentTokenCount >= FULL_CONTEXT) {
                std::cerr << "Warning: clTrain(sentence) reached FULL_CONTEXT limit ("
                          << currentTokenCount << "). Stopping training early at sentence index " << i << "." << std::endl;
                break;
            }
            int current_block_idx = blockCount;
            if (current_block_idx <= 0 || current_block_idx > m) {
                throw std::out_of_range("clTrain(sentence): Calculated current_block_idx (" + std::to_string(current_block_idx) + ") is out of range [1, " + std::to_string(m) + "].");
            }

            // Target token for this iteration
            std::vector<float>& expected_vec = sentence[i];
            std::string& expected_str = rString[i];
            size_t currentBytes = static_cast<size_t>(EMBEDDING) * effective_context_size * sizeof(float);

            std::cout << "Training token " << i+1 << "/" << sentence.size() << ": '" << expected_str << "'" << " at " << indexVec[i] << std::endl;
            std::cout << "current block: " << current_block_idx << " | current token count: " << currentTokenCount << " | eff. context size: " << effective_context_size <<std::endl;

            int j = 0; // epoch for each token
            std::fill(oneHotEncode.begin(), oneHotEncode.end(), 0.0f);
            oneHotEncode[indexVec[i]] = 1.0f;

            while (j < epochs) {
                // --- K/Q Calculation ---
                embedPlusPos = tokenEmbed + positional;
                if(current_block_idx == 1) {
                    mat partialEP(CONTEXT_WIN, EMBEDDING);
                    const float* host_src_ptr = embedPlusPos.mapped_data;
                    std::copy(host_src_ptr, host_src_ptr + currentBytes/sizeof(float), partialEP.mapped_data);
                    for (int layer_idx = 0; layer_idx < x; ++layer_idx) {
                        for (int parallel_idx = 0; parallel_idx < y; ++parallel_idx) {
                            blocks[0].b[layer_idx][parallel_idx].getKeyQuery(partialEP, blocks[0].b[layer_idx][parallel_idx].MK, 1);
                            blocks[0].b[layer_idx][parallel_idx].getKeyQuery(partialEP, blocks[0].b[layer_idx][parallel_idx].MQ, 0);
                            blocks[0].b[layer_idx][parallel_idx].getKdotQ();
                        }
                    }
                }
                else { // Subsequent blocks
                    mat prevEVs(CONTEXT_WIN, d);
                    mat currentTokens(CONTEXT_WIN, d);
                    size_t fromHereInTokenEmbed = static_cast<size_t>((CONTEXT_WIN) * (blockCount - 1) - 1) * d;
                    const float* host_src_ptr = embedPlusPos.mapped_data + fromHereInTokenEmbed;
                    std::copy(host_src_ptr, host_src_ptr + currentBytes/sizeof(float), currentTokens.mapped_data);
                    for (int layer_idx = 0; layer_idx < x; ++layer_idx) {
                        for (int parallel_idx = 0; parallel_idx < y; ++parallel_idx) {
                            blocks[blockCount - 1].b[layer_idx][parallel_idx].getKeyQuery(blocks[blockCount - 2].b[layer_idx][parallel_idx].EV,
                                                                                          blocks[blockCount - 1].b[layer_idx][parallel_idx].MK, 1);
                            blocks[blockCount - 1].b[layer_idx][parallel_idx].getKeyQuery(currentTokens, blocks[blockCount - 1].b[layer_idx][parallel_idx].MQ, 1);
                            blocks[blockCount - 1].b[layer_idx][parallel_idx].getKdotQ();
                        }
                    }
                }

                // --- Forward Pass ---
                forward(current_block_idx, effective_context_size, sequence1Count);

                // --- Get EH output from all layers ---
                std::fill(otok.begin(), otok.end(), 0.0f);
                if (y > 0) {
                    for (int layer_idx = 0; layer_idx < x; ++layer_idx) {
                        for (int k = 0; k < d; ++k) {
                            otok[(layer_idx * d) + k] = blocks[blockCount-1].b[layer_idx][y - 1].EH[k];
                        }
                    }
                }
                for(size_t k_dim = 0; k_dim < otok.size(); k_dim++) {
                    if (std::isnan(otok[k_dim])) { otok[k_dim] = 0.0f; }
                    else if (std::isinf(otok[k_dim])) { otok[k_dim] = std::copysign((std::numeric_limits<float>::max)(), otok[k_dim]); }
                }

                // --- Prediction with Scores ---
                computePrediction();
                pred = softmax(pred);

                // --- Error Calculation & Logging ---
                current_error = -std::log(pred[indexVec[i]] + 1e-15f);
                float del = current_error - prev_error;
                std::string predicted_token_str = (indexForToken >= 0 && indexForToken < static_cast<unsigned int>(tokens.size()))
                                                  ? tokens[indexForToken] : "INVALID_INDEX";

                std::cout << predicted_token_str << " ( " << indexForToken << " ) \t: "
                          << current_error << " | " << del << " | "
                          << std::exp(current_error) << " | " << j+1 << " | " << learning << std::endl;

                if (predicted_token_str == expected_str && predicted_token_str != "INVALID_INDEX") {
                    std::cout << "Token '" << expected_str << "' predicted correctly after " << j+1 << " epochs." << std::endl;
                    learning = initial_learning_rate;
                    if(predicted_token_str != "</s>")
                        std::cout << "              -------------- To Next Token --------------              " << std::endl;
                    break;
                }

                // modify the de-embeddings and get gradients for backprop
                std::vector<float> gradEH(d * x, 0.0f);
                updateDeEmbeddings(deEmbeddings, pred, oneHotEncode, learning, lambda_L1, lambda_L2, gradEH);
                // get expected target for backprop
                std::vector<std::vector<float>> targets_for_heads(x, std::vector<float>(EMBEDDING, 0.0f));
                for(int head_idx = 0; head_idx < x; ++head_idx) {
                    for(int eidx = 0; eidx < EMBEDDING; ++eidx) {
                        float gradient = learning * (gradEH[(head_idx * EMBEDDING) + eidx]
                                                  + (lambda_L1 * embeddings(indexForToken, eidx))
                                                  + (2.0f * lambda_L2 * embeddings(indexForToken, eidx)));
                        if (fabs(gradient) >= MAX_GRAD_CLIP) gradient = std::copysign(MAX_GRAD_CLIP, gradient);
                        targets_for_heads[head_idx][eidx] = otok[(head_idx * EMBEDDING) + eidx] - gradient;
                    }
                }
                // backpropagate block
                backwardContext(targets_for_heads, current_block_idx);
                // update embeddings which are in use
                updateEmbeddings(embeddings, blocks[blockCount-1].gradToken, learning, lambda_L1, lambda_L2, vocabsize);
                updateEmbeddings(tokenEmbed, blocks[blockCount-1].gradToken, learning, lambda_L1, lambda_L2, effective_context_size);

                totalLearning += learning;
                prev_error = current_error;
                totalBCELoss += current_error;
                totalBCEPerplexity += std::exp(current_error);
                j++;
            }

            // --- Update Host State ---
            trainCount++;
            epochCount += j;

            currentTokenCount++;
            effective_context_size++;
            if(currentTokenCount > 0 && currentTokenCount % CONTEXT_WIN == 0) {
                blockCount += 1;
                blockShifted = 1;
                tokenEmbed.addRow(sentence[i], currentTokenCount - 1); // repeat last token to new block
                positional.addRow(positionalEmbeddings(currentTokenCount - 1, d), currentTokenCount - 1);
                std::cout << "----> Going to Next block in model -> " << blockCount - 1 << " to " << blockCount << std::endl;
            } else {
                blockShifted = 0;
            }
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Standard Exception in clTrain(sentence): " << e.what() << std::endl;
        epochs = initial_epochs;
        throw;
    }
}

#endif
