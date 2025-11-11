#ifdef USE_CPU
#include <maths.hpp>
#include "include/transformer.hpp"
#include <thread>
#include <mutex>

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
                updateDeEmbeddings(deEmbeddings, otok, pred, oneHotEncode, indexForToken, learning, lambda_L1, lambda_L2, gradEH);
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
                updateEmbeddings(embeddings, blocks[blockCount-1].gradToken, learning, lambda_L1, lambda_L2, 12454);
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


/**
 * @brief train the transformer for sequence1 and sequence2 (single sequence1 and sequence2)
 * @param sequence1 sequence1 token embeddings
 * @param sequence2 sequence2 token embeddings
 * @param rString tokens of sequence2
 */
void transformer::trainContext(std::vector<std::vector<float>>& sequence1, std::vector<std::vector<float>>& sequence2, 
    std::vector<std::string>& rString) 
{
    // --- Basic validation ---
    if (sequence1.empty()) {
        throw std::runtime_error("clTrain(sequence1-sequence2): Initial sequence1 cannot be empty.");
    }
    // Warning for large prompts, but allow up to CONTEXT_WIN
    if (sequence1.size() > CONTEXT_WIN) {
        std::cerr << "Warning: sequence1 size (" << sequence1.size() << ") exceeds context window (" << CONTEXT_WIN << "). Ensure this is intended." << std::endl;
    }
    if (sequence2.empty() || sequence2.size() != rString.size()) {
        throw std::runtime_error("clTrain(sequence1-sequence2): sequence2 embeddings/strings mismatch or empty.");
    }
    if ((!sequence1.empty() && sequence1[0].size() != static_cast<size_t>(d)) || (!sequence2.empty() && sequence2[0].size() != static_cast<size_t>(d))) {
        throw std::runtime_error("clTrain(sequence1-sequence2): Embedding dimension mismatch.");
    }
    if (currentTokenCount + sequence1.size() + sequence2.size() > FULL_CONTEXT) {
        throw std::runtime_error("clTrain(sequence1-sequence2): Adding sequence1 and sequence2 exceeds FULL_CONTEXT limit.");
    }
    if (contextTrain != 0 && otok.size() != static_cast<size_t>(d * x)) {
        otok.clear();
        otok.resize(d * x, 0.0f); // ensure otok is correctly sized
    }

    float initial_learning_rate = learning;                     // Store initial learning rate
    float current_error = 0.0f;
    float prev_error = 0.0f;
    int initial_epochs = epochs;
    int initial_token_count = currentTokenCount;
    bool blockShifted = 0;
    int effective_context_size = 0;
    int resCount = 0;

    // --- Device Buffer Allocation & H->D Transfer ---
    size_t totalTokenEmbedFloats = static_cast<size_t>(m) * CONTEXT_WIN * d;
    size_t tokenEmbedBytes = totalTokenEmbedFloats * sizeof(float);
    size_t embeddingsBytes = static_cast<size_t>(vocabsize) * d * sizeof(float);
    size_t singleTokenBytes = static_cast<size_t>(d) * sizeof(float);
    size_t outputBytes = singleTokenBytes;
    size_t embedding_bytes_loc = static_cast<size_t>(EMBEDDING) * CONTEXT_WIN * sizeof(float);
    size_t projection_matrix_bytes = static_cast<size_t>(CONTEXT_WIN) * EMBEDDING * sizeof(float);
    size_t KQmatbytes = static_cast<size_t>(CONTEXT_WIN) * CONTEXT_WIN * sizeof(float);
    size_t indexBytes = sizeof(int);
    size_t matheights_bytes = static_cast<size_t>(CONTEXT_WIN) * sizeof(float);

    std::cout << "Current Token Count: " << currentTokenCount << " | Sequence 1: " << sequence1.size() << " | Sequence 2: " << sequence2.size() << std::endl;

    try {
        // start training from first
        if(currentTokenCount == 0) {
            // set tokenEmbed
            blockCount = 1;
            for(int i = 0; i < sequence1.size(); i++) {
                tokenEmbed.addRow(sequence1[i] + positionalEmbeddings(i, d), i);
                // prepare EVs
                for(int m1 = 0; m1 < x; m1++) {
                    for(int m2 = 0; m2 < y; m2++) {
                        blocks[0].b[m1][m2].EV.addRow(sequence1[i], i);
                    }
                }
            }
            currentTokenCount = sequence1.size();
            effective_context_size = currentTokenCount;
        }
        // continue training in first block
        else if(currentTokenCount > 0 && currentTokenCount + sequence1.size() <= CONTEXT_WIN) {
            blockCount = 1;
            // add sequence1 tokens from currentTokenCount
            if(currentTokenCount + sequence1.size() < CONTEXT_WIN) {
                for(int i = 0; i < sequence1.size(); i++) {
                    int actual_row_in_ev = (currentTokenCount + i) % CONTEXT_WIN;
                    tokenEmbed.addRow(sequence1[i] + positionalEmbeddings(currentTokenCount + i, d), currentTokenCount + i);
                    // prepare EVs
                    for(int m1 = 0; m1 < x; m1++) {
                        for(int m2 = 0; m2 < y; m2++) {
                            std::vector<float> v(EMBEDDING, 0.0f);
                            blocks[0].b[m1][m2].EV.addRow(sequence1[i], actual_row_in_ev);
                        }
                    }
                }
                currentTokenCount += sequence1.size();
                effective_context_size = currentTokenCount;
            }
            // add tokens to fill context and then perform partial forprop to add 
            else if (currentTokenCount + sequence1.size() == CONTEXT_WIN) {
                int promptCount = 0;
                for(int i = 0; i < sequence1.size(); i++) {
                    int actual_row_in_ev = (currentTokenCount + i) % CONTEXT_WIN;
                    tokenEmbed.addRow(sequence1[i] + positionalEmbeddings(currentTokenCount + i, d), currentTokenCount + i);
                    // prepare EVs
                    for(int m1 = 0; m1 < x; m1++) {
                        for(int m2 = 0; m2 < y; m2++) {
                            std::vector<float> v(EMBEDDING, 0.0f);
                            blocks[0].b[m1][m2].EV.addRow(sequence1[i], actual_row_in_ev);
                        }
                    }
                }
                currentTokenCount += sequence1.size();
                effective_context_size = currentTokenCount;
                forward_ev(blockCount, effective_context_size, promptCount);
                tokenEmbed.addRow(sequence1[sequence1.size() - 1] + positionalEmbeddings(currentTokenCount, d), currentTokenCount);
                // prepare EVs
                for(int m1 = 0; m1 < x; m1++) {
                    for(int m2 = 0; m2 < y; m2++) {
                        std::vector<float> v(EMBEDDING, 0.0f);
                        blocks[1].b[m1][m2].EV.addRow(sequence1[sequence1.size() - 1], 0);
                    }
                }
                // shift to next block
                effective_context_size = 1;
                blockCount = 2;
            }
        }
        // add tokens to fill context and then perform partial forprop to add to EV, then add to next block
        else if (currentTokenCount < CONTEXT_WIN && currentTokenCount + sequence1.size() > CONTEXT_WIN) {
            int promptCount = 0;
            int dif = currentTokenCount + sequence1.size() - CONTEXT_WIN;
            int dif1 = sequence1.size() - dif;
            for(int i = 0; i < dif; i++) {
                int actual_row_in_ev = (currentTokenCount + i) % CONTEXT_WIN;
                tokenEmbed.addRow(sequence1[i] + positionalEmbeddings(currentTokenCount + i, d), currentTokenCount + i);
                for(int m1 = 0; m1 < x; m1++) {
                    for(int m2 = 0; m2 < y; m2++) {
                        std::vector<float> v(EMBEDDING, 0.0f);
                        blocks[0].b[m1][m2].EV.addRow(sequence1[i], actual_row_in_ev);
                    }
                }
            }
            currentTokenCount += dif;
            effective_context_size += dif;
            forward_ev(blockCount, effective_context_size, promptCount);
            // shift to next block
            blockCount = 2;
            for(int i = 0; i < dif1; i++) {
                tokenEmbed.addRow(sequence1[dif + i] + positionalEmbeddings(currentTokenCount + i, d), currentTokenCount + i);
                // prepare EVs
                for(int m1 = 0; m1 < x; m1++) {
                    for(int m2 = 0; m2 < y; m2++) {
                        std::vector<float> v(EMBEDDING, 0.0f);
                        blocks[1].b[m1][m2].EV.addRow(sequence1[i], i);
                    }
                }
            }
            effective_context_size = dif1;
            currentTokenCount += dif1;
            blockCount += 1;
        }
        // training in non-first blocks
        else {
            effective_context_size = currentTokenCount % CONTEXT_WIN;
            // add sequence1 tokens from currentTokenCount: 8192 % 1024 = 0
            if(effective_context_size == 0 && effective_context_size + sequence1.size() < CONTEXT_WIN) {
                for(int i = 0; i < sequence1.size(); i++) {
                    tokenEmbed.addRow(sequence1[i] + positionalEmbeddings(currentTokenCount + i, d), currentTokenCount + i);
                    // prepare EVs
                    for(int m1 = 0; m1 < x; m1++) {
                        for(int m2 = 0; m2 < y; m2++) {
                            blocks[blockCount - 1].b[m1][m2].EV.addRow(sequence1[i], i);
                        }
                    }
                }
                currentTokenCount += sequence1.size();
                effective_context_size = sequence1.size();
            }
            // 8194 % 1024 = 2
            else if(effective_context_size > 0 && effective_context_size + sequence1.size() < CONTEXT_WIN) {
                for(int i = 0; i < sequence1.size(); i++) {
                    tokenEmbed.addRow(sequence1[i] + positionalEmbeddings(currentTokenCount + i, d), currentTokenCount + i);
                    // prepare EVs
                    for(int m1 = 0; m1 < x; m1++) {
                        for(int m2 = 0; m2 < y; m2++) {
                            blocks[blockCount - 1].b[m1][m2].EV.addRow(sequence1[i], i);
                        }
                    }
                }
                currentTokenCount += sequence1.size();
                effective_context_size += sequence1.size();
            }
            // add tokens to fill context and then perform partial forprop to add  to EV
            else if (effective_context_size > 0 && effective_context_size + sequence1.size() == CONTEXT_WIN) {
                int promptCount = 0;
                for(int i = 0; i < sequence1.size(); i++) {
                    tokenEmbed.addRow(sequence1[i] + positionalEmbeddings(currentTokenCount + i, d), currentTokenCount + i);
                    // prepare EVs
                    for(int m1 = 0; m1 < x; m1++) {
                        for(int m2 = 0; m2 < y; m2++) {
                            blocks[blockCount - 1].b[m1][m2].EV.addRow(sequence1[i], i);
                        }
                    }
                }
                currentTokenCount += sequence1.size();
                effective_context_size += sequence1.size();
                forward_ev(blockCount, effective_context_size, promptCount);
                // shift to next block
                tokenEmbed.addRow(sequence1[sequence1.size() - 1] + positionalEmbeddings(currentTokenCount + 1, d), currentTokenCount);
                for(int m1 = 0; m1 < x; m1++) {
                    for(int m2 = 0; m2 < y; m2++) {
                        std::vector<float> v(EMBEDDING, 0.0f);
                        blocks[blockCount].b[m1][m2].EV.addRow(sequence1[sequence1.size() - 1], 0);
                    }
                }
                effective_context_size = 1;
                blockCount += 1;
            }
            // add tokens to fill context and then perform partial forprop to add to EV, then add to next block
            else if (effective_context_size + sequence1.size() > CONTEXT_WIN) {
                int promptCount = 0;
                int dif = currentTokenCount + sequence1.size() - CONTEXT_WIN;
                int dif1 = sequence1.size() - dif;
                for(int i = 0; i < dif; i++) {
                    tokenEmbed.addRow(sequence1[i] + positionalEmbeddings(currentTokenCount + i, d), currentTokenCount + i);
                    // prepare EVs
                    for(int m1 = 0; m1 < x; m1++) {
                        for(int m2 = 0; m2 < y; m2++) {
                            blocks[blockCount - 1].b[m1][m2].EV.addRow(sequence1[i] + positionalEmbeddings(currentTokenCount + i, d), i);
                        }
                    }
                }
                currentTokenCount += dif;
                effective_context_size += dif;
                forward_ev(blockCount, effective_context_size, promptCount);
                // shift to next block
                for(int i = 0; i < dif1; i++) {
                    tokenEmbed.addRow(sequence1[dif + i], currentTokenCount + i);
                    // prepare EVs
                    for(int m1 = 0; m1 < x; m1++) {
                        for(int m2 = 0; m2 < y; m2++) {
                            blocks[blockCount].b[m1][m2].EV.addRow(sequence1[i] + positionalEmbeddings(currentTokenCount + i, d), i);
                        }
                    }
                }
                currentTokenCount += dif1;
                effective_context_size = dif1;
                blockCount += 1;
            }
        }

        // --- Process sequence1 (Add to context on Host and Device) ---
        if ((initial_token_count % CONTEXT_WIN) + sequence1.size() > CONTEXT_WIN) {
            throw std::runtime_error("clTrain(sequence1, sequence2): sequence1 exceeds current block capacity when starting.");
        }

        // Copy sequence1 D->D from d_tokenEmbed into d_EV of each head in block 0
        size_t prompt_bytes = sequence1.size() * d * sizeof(float);
        size_t prompt_start_offset_bytes = initial_token_count * d * sizeof(float);
        for (int i = 0; i < x; ++i) { // Layers
            for (int j = 0; j < y; ++j) { // Parallels
                size_t dest_offset_bytes = (initial_token_count % CONTEXT_WIN) * d * sizeof(float); // Correct offset within the block's context window
            }
        }
        blockCount = (currentTokenCount == 0) ? 1 : ((currentTokenCount - 1) / CONTEXT_WIN) + 1;
        sequence1Count = sequence1.size();

        // --- Train for sequence2 ---
        for (size_t i = 0; i < sequence2.size(); ++i) {
            if (currentTokenCount >= FULL_CONTEXT) {
                std::cerr << "Warning: clTrain(sequence1-sequence2) reached FULL_CONTEXT limit ("
                        << currentTokenCount << ") during sequence2. Stopping training early at sequence2 index " << i << "." << std::endl;
                break;
            }

            std::vector<float> expected_vec(d, 0.0f);
            expected_vec = sequence2[i];
            std::string& expected_str = rString[i];
            int current_block_idx = blockCount; // Block index based on current context size

            if (current_block_idx <= 0 || current_block_idx > m) {
                throw std::out_of_range("clTrain(sequence1-sequence2): Calculated current_block_idx (" 
                                        + std::to_string(current_block_idx) + ") is out of range [1, " 
                                        + std::to_string(m) + "].");
            }

            std::cout << "Training token " << i+1 << "/" << sequence2.size() << ": '" << expected_str << "'" << " at " << indexVec[i] << std::endl;
            std::cout << "current block: " << current_block_idx << " & current token count: " << currentTokenCount << " & eff. context size: " << effective_context_size <<std::endl;
            size_t currentBytes = static_cast<size_t>(EMBEDDING) * effective_context_size * sizeof(float);

            // --- Training Loop for sequence2 token i ---
            int j = 0;
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
                else {
                    // Subsequent blocks
                    mat prevEVs(CONTEXT_WIN, d);
                    mat currentTokens(CONTEXT_WIN, d);
                    size_t fromHereInTokenEmbed = static_cast<size_t>((CONTEXT_WIN) * (blockCount - 1) - 1) * d;
                    const float* host_src_ptr = embedPlusPos.mapped_data + fromHereInTokenEmbed;
                    std::copy(host_src_ptr, host_src_ptr + currentBytes/sizeof(float), currentTokens.mapped_data);

                    for (int layer_idx = 0; layer_idx < x; ++layer_idx) {
                        for (int parallel_idx = 0; parallel_idx < y; ++parallel_idx) {
                            auto& prevEV = blocks[blockCount - 2].b[layer_idx][parallel_idx].EV;
                            std::copy(prevEV.mapped_data, prevEV.mapped_data + currentBytes/sizeof(float), prevEVs.mapped_data);

                            blocks[blockCount - 1].b[layer_idx][parallel_idx].getKeyQuery(prevEVs, blocks[blockCount - 1].b[layer_idx][parallel_idx].MK, 1);
                            blocks[blockCount - 1].b[layer_idx][parallel_idx].getKeyQuery(currentTokens, blocks[blockCount - 1].b[layer_idx][parallel_idx].MQ, 0);
                            blocks[blockCount - 1].b[layer_idx][parallel_idx].getKdotQ();
                        }
                    }
                }

                // --- Forward Pass ---
                int current_prompt_count_in_block = effective_context_size % CONTEXT_WIN == 0 ? CONTEXT_WIN : effective_context_size % CONTEXT_WIN;
                forward(current_block_idx, effective_context_size, current_prompt_count_in_block);

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
                updateDeEmbeddings(deEmbeddings, otok, pred, oneHotEncode, indexForToken, learning, lambda_L1, lambda_L2, gradEH);
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
                updateEmbeddings(embeddings, blocks[blockCount-1].gradToken, learning, lambda_L1, lambda_L2, 12454);
                updateEmbeddings(tokenEmbed, blocks[blockCount-1].gradToken, learning, lambda_L1, lambda_L2, effective_context_size);

                j++;
            }
            // --- Update Host State ---
            trainCount++;
            epochCount += j;
            resCount++;
            tokenEmbed.addRow(sequence2[i], currentTokenCount);
            currentTokenCount++;
            effective_context_size++;

            if(currentTokenCount > 0 && currentTokenCount % CONTEXT_WIN == 0) {
                blockCount += 1;
                blockShifted = 1;
                tokenEmbed.addRow(sequence2[i], currentTokenCount - 1); // repeat last token to new block
                positional.addRow(positionalEmbeddings(currentTokenCount - 1, d), currentTokenCount - 1);
                std::cout << "----> Going to Next block in model -> " << blockCount - 1 << " to " << blockCount << std::endl;
            } else {
                blockShifted = 0;
            }
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Standard Exception in clTrain(sequence1-sequence2): " << e.what() << std::endl;
        epochs = initial_epochs;
        throw;
    }
}

#endif
