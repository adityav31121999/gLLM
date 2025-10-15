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
    if(sentence.size() > FULL_CONTEXT) {
        throw std::runtime_error("Sentence size should not exceed FULL_CONTEXT");
    }
    if(sentence.size() != rString.size()) {
        throw std::runtime_error("Sentence embeddings and sentence strings must be non-empty and have the same size.");
    }
    // compute KdotQ for first element
    setRow(tokenEmbed, 0, sentence[0]);
    sequence1Count = 1;
    blockCount = 1;
    // keep this in a loop and train for each token in the sentence starting from second token
    for(int token_idx_in_sentence = 1; token_idx_in_sentence < sentence.size(); token_idx_in_sentence++) {
        // first block
        if(blockCount == 1 && currentTokenCount < CONTEXT_WIN) {
            // get keys and queries for KdotQ
            for(int j = 0; j < x; j++) {
                for(int k = 0; k < y; k++) {
                    std::vector<float> q_output_vec(h);
                    computeKorQ(sentence[token_idx_in_sentence], blocks[0].b[j][k].MQ, q_output_vec);
                    setRow(blocks[0].b[j][k].Q, token_idx_in_sentence, q_output_vec);

                    std::vector<float> k_output_vec(h);
                    computeKorQ(sentence[token_idx_in_sentence], blocks[0].b[j][k].MK, k_output_vec);
                    setRow(blocks[0].b[j][k].K, token_idx_in_sentence, k_output_vec);

                    setRow(blocks[0].b[j][k].EV, token_idx_in_sentence, sentence[token_idx_in_sentence]);
                }
            }
            // compute KdotQ for all heads of this block
            computeKdotQs(sequence1Count, currentTokenCount, blockCount, isSelf, inTraining);
            int j_epoch = 0; // Renamed to avoid conflict with outer loop variable j
            // TRAIN FOR SENTENCE
            forward(blockCount, currentTokenCount, sequence1Count);
            while (j_epoch <= epochs) {
                // computeOutput(otok, embeddings, vocabsize, indexForToken);
                if((errorofv(otok, sentence[token_idx_in_sentence]) < 0.01) || tokens[indexForToken] == rString[token_idx_in_sentence]) {
                    break;
                }
                // if error is not corrected even after epochs, then increase epochs
                if(errorofv(otok, sentence[token_idx_in_sentence]) > 0.01 && j_epoch == epochs) {
                    epochs += 10;
                }
                // backward(sentence[token_idx_in_sentence]);
                forward(blockCount, currentTokenCount, sequence1Count);
                j_epoch++;
            }
            // update variables
            trainCount++;
            epochCount += j_epoch;
            error += errorofv(otok, sentence[token_idx_in_sentence]);
            currentTokenCount += 1;
            if(currentTokenCount == CONTEXT_WIN) {
                blockCount += 1;
            }
        }
        // if local context of previous block is full, shift to next block directly from here,
        // no need to end the loop and start new one
        else if(blockCount > 1 && currentTokenCount > CONTEXT_WIN) {
            // token_idx_in_block: index of the current token within the current block's window
            int token_idx_in_block = currentTokenCount % CONTEXT_WIN; // Or currentTokenCount - (blockCount-1)*CONTEXT_WIN if currentTokenCount is global
            if (token_idx_in_block == 0 && currentTokenCount >= CONTEXT_WIN) token_idx_in_block = CONTEXT_WIN -1; // If it's the last token of a full window

            // add new token embedding to tokForBlock for this block
            setRow(blocks[blockCount-1].tokForBlock, token_idx_in_block, sentence[currentTokenCount-1]); // sentence[currentTokenCount-1] is the latest input token
            // compute keys and queries
            for(int j = 0; j < x; j++) {
                for(int k = 0; k < y; k++) {
                    // compute Key and Query for this iteration
                    std::vector<float> prev_block_ev_row = getRow(blocks[blockCount-2].b[j][k].EV, token_idx_in_block);
                    std::vector<float> q_output_vec(h);
                    computeKorQ(prev_block_ev_row, blocks[blockCount-1].b[j][k].MQ, q_output_vec);
                    setRow(blocks[blockCount-1].b[j][k].Q, token_idx_in_block, q_output_vec);

                    std::vector<float> current_block_token_row = getRow(blocks[blockCount-1].tokForBlock, token_idx_in_block);
                    std::vector<float> k_output_vec(h);
                    computeKorQ(current_block_token_row, blocks[blockCount-1].b[j][k].MK, k_output_vec);
                    setRow(blocks[blockCount-1].b[j][k].K, token_idx_in_block, k_output_vec);

                    setRow(blocks[blockCount-1].b[j][k].EV, token_idx_in_block, sentence[token_idx_in_sentence]); // Target token for EV? Or input?
                }
            }
            // compute the KdotQ for each head of block using EVs of previous blocks
            computeKdotQs(sequence1Count, currentTokenCount, blockCount, isSelf, inTraining);
            // TRAIN FOE SENTENCE
            forward(blockCount, currentTokenCount, sequence1Count);
            int j_epoch = 0; // Renamed
            while (j_epoch < epochs) {
                // computeOutput(otok, embeddings, vocabsize, indexForToken);
                if(errorofv(otok, sentence[token_idx_in_sentence]) < 0.01 || tokens[indexForToken] == rString[token_idx_in_sentence]) {
                    break;
                }
                // if error is not corrected even after epochs, then increase epochs
                if(errorofv(otok, sentence[token_idx_in_sentence]) > 0.01 && j_epoch == epochs) {
                    epochs += 10;
                }
                j_epoch++;
                backward(sentence[token_idx_in_sentence], blockCount);
                forward(blockCount, currentTokenCount, sequence1Count);
            }
            // update variables
            trainCount++;
            epochCount += j_epoch;
            error += errorofv(otok, sentence[token_idx_in_sentence]);
            currentTokenCount += 1;
            if(currentTokenCount % CONTEXT_WIN == 0) {
                blockCount += 1;
            }
        }
    }
}


/**
 * @brief train the transformer for sequence1 and sequence2 (single sequence1 and sequence2)
 * @param sequence1 sequence1 token embeddings
 * @param sequence2 sequence2 token embeddings
 * @param rString tokens of sequence2
 */
void transformer::train(std::vector<std::vector<float>>& sequence1, std::vector<std::vector<float>>& sequence2, 
    std::vector<std::string>& rString) 
{
    // sequence1 must not be empty
    if (sequence1.empty()) {
        throw std::runtime_error("Initial sequence1 cannot be empty!");
    }
    // sequence1 size should not exceed threshold (one fourth of context window)
    if(sequence1.size() > PROMPT_THRESHOLD) {
        throw std::runtime_error("Pompt size should not exceed CONTEXT_WIN!");
    }
    // Basic validation
    if (sequence2.empty() || sequence2.size() != rString.size()) {
        throw std::runtime_error("Sequence2 embeddings and sequence2 strings must be non-empty and have the same size!");
    }
    
    int initialTokCount = currentTokenCount;
    // Keys and Queries of sequence1s
    if(blockCount == 1) {
        for(int i_pa = 0; i_pa < x; i_pa++) {
            for(int j_head = 0; j_head < y; j_head++) {
                for(int k = 0; k < sequence1.size(); k++) {
                    // make queries using compute KorQ: blocks[0].b[i][j].Q[currentTokenCount%CONTEXT_WIN] = sequence1(i) * blocks[0].b[i][j].MQ
                    blocks[0].b[i_pa][j_head].Q(initialTokCount%CONTEXT_WIN + k) = dot(sequence1[i_pa], blocks[0].b[i_pa][j_head].MQ);
                    // make keys using compute KorQ: blocks[0].b[i][j].K[currentTokenCount%CONTEXT_WIN] = sequence1(i) * blocks[0].b[i][j].MK
                    blocks[0].b[i_pa][j_head].K(initialTokCount%CONTEXT_WIN + k) = dot(sequence1[i_pa], blocks[0].b[i_pa][j_head].MK);
                }
            }
        }
    }
    else {
        for(int i_pa = 0; i_pa < x; i_pa++) {
            for(int j_head = 0; j_head < y; j_head++) {
                for(int k = 0; k < sequence1.size(); k++) {
                    // make queries using compute KorQ: blocks[0].b[i][j].Q[currentTokenCount%CONTEXT_WIN] = sequence1(i) * blocks[0].b[i][j].MQ
                    blocks[blockCount-1].b[i_pa][j_head].Q(initialTokCount%CONTEXT_WIN + k) = dot(sequence1[i_pa], blocks[0].b[i_pa][j_head].MQ);
                }
                for(int k = 0; k < CONTEXT_WIN; k++) {
                    // make queries using compute KorQ: blocks[0].b[i][j].K[i] = sequence1(i) * blocks[blockCount-1].b[i][j].MK
                    blocks[blockCount-1].b[i_pa][j_head].K(k) = dot(blocks[blockCount-1].b[i_pa][j_head].EV(k), blocks[0].b[i_pa][j_head].MQ);
                }
            }
        }
    }

    int resCount = 0;

    for(int i = 0; i < sequence2.size(); i++) {
        // compute the KdotQ for each head of block using EVs of previous blocks
        computeKdotQs(sequence1Count, currentTokenCount, blockCount, isSelf, inTraining);
        forward(blockCount, currentTokenCount, sequence1Count);

        int j_epoch = 0;
        while (j_epoch < epochs) {
            // computeOutput(otok, embeddings, vocabsize, indexForToken);
            float current_error = MSE(otok, sequence2[i]);
            if(tokens[indexForToken] == rString[i])
            {
                std::cout << "indexForToken: " << indexForToken << std::endl;
                std::cout << "Computed token is -> " << tokens[indexForToken] << " <- with error " << current_error << std::endl;
                break;
            }
            else if(j_epoch == epochs - 1) {
                std::cout << "Computed token is -> " << tokens[indexForToken] << " <- with error " << current_error << std::endl;
                std::cout << "Increasing Epoch Count by 10 '-'" << std::endl;
                epochs += 10;
            }
            else {
                std::cout << "Computed token is -> " << tokens[indexForToken] << " <- with error " << current_error << std::endl;
            }

            j_epoch++;
            backward(sequence2[i], blockCount);

            // Keys and Queries of both sequence1s and sequence2
            if(blockCount == 1) {
                for(int i_pa = 0; i_pa < x; i_pa++) {
                    for(int j_head = 0; j_head < y; j_head++) {
                        for(int k = 0; k < sequence1.size(); k++) {
                            // make queries using compute KorQ: blocks[0].b[i][j].Q[currentTokenCount%CONTEXT_WIN] = sequence1(i) * blocks[0].b[i][j].MQ
                            blocks[0].b[i_pa][j_head].Q(initialTokCount%CONTEXT_WIN + k) = dot(sequence1[i_pa], blocks[0].b[i_pa][j_head].MQ);
                            // make keys using compute KorQ: blocks[0].b[i][j].K[currentTokenCount%CONTEXT_WIN] = sequence1(i) * blocks[0].b[i][j].MK
                            blocks[0].b[i_pa][j_head].K(initialTokCount%CONTEXT_WIN + k) = dot(sequence1[i_pa], blocks[0].b[i_pa][j_head].MK);
                        }
                        if(resCount > 0) {
                            for(int k = 0; k < resCount; k++) {
                                // make queries using compute KorQ: blocks[0].b[i][j].Q[currentTokenCount%CONTEXT_WIN + sequence1.size() + k] = sequence2(i) * blocks[0].b[i][j].MQ
                                blocks[0].b[m][n].Q(initialTokCount%CONTEXT_WIN + sequence1.size() + k) = dot(sequence2[k], blocks[0].b[m][n].MQ);
                                // make keys using compute KorQ: blocks[0].b[i][j].K[currentTokenCount%CONTEXT_WIN + sequence1.size() + k] = sequence2(i) * blocks[0].b[i][j].MK
                                blocks[0].b[m][n].K(initialTokCount%CONTEXT_WIN + sequence1.size() + k) = dot(sequence2[k], blocks[0].b[m][n].MK);
                            }
                        }
                    }
                }
            }
            else {
                for(int i_pa = 0; i_pa < x; i_pa++) {
                    for(int j_head = 0; j_head < y; j_head++) {
                        for(int k = 0; k < sequence1.size(); k++) {
                            // make queries using compute KorQ: blocks[0].b[i][j].Q[currentTokenCount%CONTEXT_WIN] = sequence1(i) * blocks[0].b[i][j].MQ
                            blocks[blockCount-1].b[i_pa][j_head].Q(initialTokCount%CONTEXT_WIN + k) = dot(sequence1[i_pa], blocks[0].b[i_pa][j_head].MQ);
                        }
                        for(int k = 0; k < CONTEXT_WIN; k++) {
                            // make queries using compute KorQ: blocks[0].b[i][j].K[i] = sequence1(i) * blocks[blockCount-1].b[i][j].MK
                            blocks[blockCount-1].b[i_pa][j_head].K(k) = dot(blocks[blockCount-1].b[i_pa][j_head].EV(k), blocks[0].b[i_pa][j_head].MQ);
                        }
                        if(resCount > 0) {
                            for(int k = 0; k < resCount; k++) {
                                // make queries using compute KorQ: blocks[blockCount-1].b[i][j].Q[currentTokenCount%CONTEXT_WIN + sequence1.size() + k] = sequence2(i) * blocks[0].b[i][j].MQ
                                blocks[blockCount-1].b[m][n].Q(initialTokCount%CONTEXT_WIN + sequence1.size() + k) = dot(sequence2[k], blocks[blockCount-1].b[m][n].MQ);
                                // make keys using compute KorQ: blocks[blockCount-1].b[i][j].K[currentTokenCount%CONTEXT_WIN + sequence1.size() + k] = sequence2(i) * blocks[0].b[i][j].MK
                                blocks[blockCount-1].b[m][n].K(initialTokCount%CONTEXT_WIN + sequence1.size() + k) = dot(sequence2[k], blocks[blockCount-1].b[m][n].MK);
                            }
                        }
                    }
                }
            }
            // forward(blockCount, currentTokenCount, sequence1Count);
        }
        // update variables
        resCount++;
        trainCount++;
        epochCount += j_epoch;
        error += errorofv(otok, sequence2[i]);
        currentTokenCount += 1;
        if(currentTokenCount % CONTEXT_WIN == 0) {
            blockCount += 1;
            // blocks[blockCount-1].deserialise(blocks[blockCount-1].blockFilePath);
        }
        // Keys and Queries of sequence2
        if(resCount > 0) {
            if(blockCount == 1) {
                for(int m = 0; m < x; m++) {
                    for(int n = 0; n < y; n++) {
                        for(int k = 0; k < resCount; k++) {
                            // make queries using compute KorQ: blocks[0].b[i][j].Q[currentTokenCount%CONTEXT_WIN + sequence1.size() + k] = sequence2(i) * blocks[0].b[i][j].MQ
                            blocks[0].b[m][n].Q(initialTokCount%CONTEXT_WIN + sequence1.size() + k) = dot(sequence2[k], blocks[0].b[m][n].MQ);
                            // make keys using compute KorQ: blocks[0].b[i][j].K[currentTokenCount%CONTEXT_WIN + sequence1.size() + k] = sequence2(i) * blocks[0].b[i][j].MK
                            blocks[0].b[m][n].K(initialTokCount%CONTEXT_WIN + sequence1.size() + k) = dot(sequence2[k], blocks[0].b[m][n].MK);
                        }
                    }
                }
            }
            else {
                for(int m = 0; m < x; m++) {
                    for(int n = 0; n < y; n++) {
                        for(int k = 0; k < resCount; k++) {
                            // make queries using compute KorQ: blocks[blockCount-1].b[i][j].Q[currentTokenCount%CONTEXT_WIN + sequence1.size() + k] = sequence2(i) * blocks[0].b[i][j].MQ
                            blocks[blockCount-1].b[m][n].Q(initialTokCount%CONTEXT_WIN + sequence1.size() + k) = dot(sequence2[k], blocks[blockCount-1].b[m][n].MQ);
                            // make keys using compute KorQ: blocks[blockCount-1].b[i][j].K[currentTokenCount%CONTEXT_WIN + sequence1.size() + k] = sequence2(i) * blocks[0].b[i][j].MK
                            blocks[blockCount-1].b[m][n].K(initialTokCount%CONTEXT_WIN + sequence1.size() + k) = dot(sequence2[k], blocks[blockCount-1].b[m][n].MK);
                        }
                    }
                }
            }
        }
    }
}

#endif
