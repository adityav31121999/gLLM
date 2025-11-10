#ifdef USE_OPENCL
#include "include/transformer.hpp"
#include <maths.hpp>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <string>
#include <cmath>
#include <iomanip>

/**
 * @brief (OpenCL) sequence-to-sequence training of the transformer.
 * @param sequence1 sequence1 token embeddings (on host).
 * @param sequence2 sequence2 token embeddings (on host).
 * @param rString Tokens of the sequence2 (on host).
 */
void transformer::clTrainContext(std::vector<std::vector<float>>& sequence1, std::vector<std::vector<float>>& sequence2, 
        std::vector<std::string>& rString)
{
    // --- Basic validation ---
    if (sequence1.empty()) {
        throw std::runtime_error("clTrainContext(sequence1-sequence2): Initial sequence1 cannot be empty.");
    }
    // Warning for large prompts, but allow up to CONTEXT_WIN
    if (sequence1.size() > CONTEXT_WIN) {
        std::cerr << "Warning: sequence1 size (" << sequence1.size() << ") exceeds context window (" << CONTEXT_WIN << "). Ensure this is intended." << std::endl;
    }
    if (sequence2.empty() || sequence2.size() != rString.size()) {
        throw std::runtime_error("clTrainContext(sequence1-sequence2): sequence2 embeddings/strings mismatch or empty.");
    }
    if ((!sequence1.empty() && sequence1[0].size() != static_cast<size_t>(d)) || (!sequence2.empty() && sequence2[0].size() != static_cast<size_t>(d))) {
        throw std::runtime_error("clTrainContext(sequence1-sequence2): Embedding dimension mismatch.");
    }
    if (currentTokenCount + sequence1.size() + sequence2.size() > FULL_CONTEXT) {
        throw std::runtime_error("clTrainContext(sequence1-sequence2): Adding sequence1 and sequence2 exceeds FULL_CONTEXT limit.");
    }

    cl_int cl_err;
    cl::Kernel kq_kernel, predKernel;
    cl::Buffer d_tokenEmbed, d_deEmbeddings, d_expected_token;
    cl::Buffer d_Q_cl, d_K_cl, d_mQ_cl, d_mK_cl, d_tok_cl;
    float initial_learning_rate = learning;
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
    size_t deEmbeddingsBytes = static_cast<size_t>(vocabsize) * x * d * sizeof(float);
    size_t singleTokenBytes = static_cast<size_t>(d) * sizeof(float);
    size_t outputBytes = singleTokenBytes;
    size_t embedding_bytes_loc = static_cast<size_t>(EMBEDDING) * CONTEXT_WIN * sizeof(float);
    size_t projection_matrix_bytes = static_cast<size_t>(CONTEXT_WIN) * EMBEDDING * sizeof(float);
    size_t KQmatbytes = static_cast<size_t>(CONTEXT_WIN) * CONTEXT_WIN * sizeof(float);
    size_t indexBytes = sizeof(int);
    size_t matheights_bytes = static_cast<size_t>(CONTEXT_WIN) * sizeof(float);
    size_t predBytes = static_cast<size_t>(vocabsize) * sizeof(float);

    // allot buffers
    d_deEmbeddings = cl::Buffer(clcontext.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, deEmbeddingsBytes, deEmbeddings.mapped_data, &cl_err);
    d_tokenEmbed = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, tokenEmbedBytes, nullptr, &cl_err); CL_CHECK(cl_err);
    d_expected_token = cl::Buffer(clcontext.context, CL_MEM_READ_ONLY, singleTokenBytes, nullptr, &cl_err); CL_CHECK(cl_err); // Buffer for the target token
    d_Q_cl = cl::Buffer(clcontext.context, CL_MEM_WRITE_ONLY, KQmatbytes, nullptr, &cl_err); CL_CHECK(cl_err);
    d_K_cl = cl::Buffer(clcontext.context, CL_MEM_WRITE_ONLY, KQmatbytes, nullptr, &cl_err); CL_CHECK(cl_err);
    d_mQ_cl = cl::Buffer(clcontext.context, CL_MEM_READ_ONLY, projection_matrix_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
    d_mK_cl = cl::Buffer(clcontext.context, CL_MEM_READ_ONLY, projection_matrix_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
    d_tok_cl = cl::Buffer(clcontext.context, CL_MEM_READ_ONLY, embedding_bytes_loc, nullptr, &cl_err); CL_CHECK(cl_err);
    // get kernel needed
    kq_kernel = clcontext.kernels.at("kernelComputeKQall");
    predKernel = clcontext.kernels.at("kernelComputePredictionWithScores");

    try {
        // for token prediction
        otok.clear(); otok.resize(d*x, 0.0f);
        pred.clear(); pred.resize(vocabsize, 0.0f);
        oneHotEncode.clear(); oneHotEncode.resize(vocabsize, 0.0f);

        // --- set all tokens to tokenEmbed ---
        std::fill(tokenEmbed.mapped_data, tokenEmbed.mapped_data + totalTokenEmbedFloats, 0.0f);
        std::fill(positional.mapped_data, positional.mapped_data + totalTokenEmbedFloats, 0.0f);
        std::fill(embedPlusPos.mapped_data, embedPlusPos.mapped_data + totalTokenEmbedFloats, 0.0f);
        // start training from first
        if(currentTokenCount == 0) {
            // set tokenEmbed
            blockCount = 1;
            for(int i = 0; i < sequence1.size(); i++) {
                tokenEmbed.addRow(sequence1[i], i);
                positional.addRow(positionalEmbeddings(i, d), i);
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
                    tokenEmbed.addRow(sequence1[i], currentTokenCount + i);
                    positional.addRow(positionalEmbeddings(currentTokenCount + i, d), currentTokenCount + i);
                    // prepare EVs
                    for(int m1 = 0; m1 < x; m1++) {
                        for(int m2 = 0; m2 < y; m2++) {
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
                    tokenEmbed.addRow(sequence1[i], currentTokenCount + i);
                    positional.addRow(positionalEmbeddings(currentTokenCount + i, d), currentTokenCount + i);
                    // prepare EVs
                    for(int m1 = 0; m1 < x; m1++) {
                        for(int m2 = 0; m2 < y; m2++) {
                            blocks[0].b[m1][m2].EV.addRow(sequence1[i], actual_row_in_ev);
                        }
                    }
                }
                currentTokenCount += sequence1.size();
                effective_context_size = currentTokenCount;
                clForward_ev(blockCount, effective_context_size, promptCount);
                tokenEmbed.addRow(sequence1[sequence1.size() - 1], currentTokenCount);
                positional.addRow(positionalEmbeddings(currentTokenCount, d), currentTokenCount);
                // prepare EVs
                for(int m1 = 0; m1 < x; m1++) {
                    for(int m2 = 0; m2 < y; m2++) {
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
                tokenEmbed.addRow(sequence1[i], currentTokenCount + i);
                positional.addRow(positionalEmbeddings(currentTokenCount + i, d), currentTokenCount + i);
                for(int m1 = 0; m1 < x; m1++) {
                    for(int m2 = 0; m2 < y; m2++) {
                        blocks[0].b[m1][m2].EV.addRow(sequence1[i], actual_row_in_ev);
                    }
                }
            }
            currentTokenCount += dif;
            effective_context_size += dif;
            clForward_ev(blockCount, effective_context_size, promptCount);
            // shift to next block
            blockCount = 2;
            for(int i = 0; i < dif1; i++) {
                tokenEmbed.addRow(sequence1[dif + i], currentTokenCount + i);
                positional.addRow(positionalEmbeddings(currentTokenCount + i, d), currentTokenCount + i);
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
                    tokenEmbed.addRow(sequence1[i], currentTokenCount + i);
                    positional.addRow(positionalEmbeddings(currentTokenCount + i, d), currentTokenCount + i);
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
                    tokenEmbed.addRow(sequence1[i], currentTokenCount + i);
                    positional.addRow(positionalEmbeddings(currentTokenCount + i, d), currentTokenCount + i);
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
                    tokenEmbed.addRow(sequence1[i], currentTokenCount + i);
                    positional.addRow(positionalEmbeddings(currentTokenCount + i, d), currentTokenCount + i);
                    // prepare EVs
                    for(int m1 = 0; m1 < x; m1++) {
                        for(int m2 = 0; m2 < y; m2++) {
                            blocks[blockCount - 1].b[m1][m2].EV.addRow(sequence1[i], i);
                        }
                    }
                }
                currentTokenCount += sequence1.size();
                effective_context_size += sequence1.size();
                clForward_ev(blockCount, effective_context_size, promptCount);
                // shift to next block
                tokenEmbed.addRow(sequence1[sequence1.size() - 1], currentTokenCount);
                positional.addRow(positionalEmbeddings(currentTokenCount, d), currentTokenCount);
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
                    tokenEmbed.addRow(sequence1[i], currentTokenCount + i);
                    positional.addRow(positionalEmbeddings(currentTokenCount + i, d), currentTokenCount + i);
                    // prepare EVs
                    for(int m1 = 0; m1 < x; m1++) {
                        for(int m2 = 0; m2 < y; m2++) {
                            blocks[blockCount - 1].b[m1][m2].EV.addRow(sequence1[i], i);
                        }
                    }
                }
                currentTokenCount += dif;
                effective_context_size += dif;
                clForward_ev(blockCount, effective_context_size, promptCount);
                // shift to next block
                for(int i = 0; i < dif1; i++) {
                    tokenEmbed.addRow(sequence1[dif + i], currentTokenCount + i);
                    positional.addRow(positionalEmbeddings(currentTokenCount + i, d), currentTokenCount + i);
                    // prepare EVs
                    for(int m1 = 0; m1 < x; m1++) {
                        for(int m2 = 0; m2 < y; m2++) {
                            blocks[blockCount].b[m1][m2].EV.addRow(sequence1[i], i);
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
            throw std::runtime_error("clTrainContext(sequence1, sequence2): sequence1 exceeds current block capacity when starting.");
        }

        // Copy sequence1 D->D from d_tokenEmbed into d_EV of each head in block 0
        size_t prompt_bytes = sequence1.size() * d * sizeof(float);
        size_t prompt_start_offset_bytes = initial_token_count * d * sizeof(float);
        for (int i = 0; i < x; ++i) {
            for (int j = 0; j < y; ++j) {
                cl::Buffer& d_head_ev = blocks[0].b[i][j].getDeviceEVBuffer(); // Assuming getter exists
                size_t dest_offset_bytes = (initial_token_count % CONTEXT_WIN) * d * sizeof(float); // Correct offset within the block's context window
                CL_CHECK(clcontext.queue.enqueueCopyBuffer(d_tokenEmbed, d_head_ev, prompt_start_offset_bytes, dest_offset_bytes, prompt_bytes));
            }
        }
        blockCount = (currentTokenCount == 0) ? 1 : ((currentTokenCount - 1) / CONTEXT_WIN) + 1;
        sequence1Count = sequence1.size();

        // --- Train for sequence2 ---
        for (size_t i = 0; i < sequence2.size(); ++i) {
            if (currentTokenCount >= FULL_CONTEXT) {
                std::cerr << "Warning: clTrainContext(sequence1-sequence2) reached FULL_CONTEXT limit ("
                        << currentTokenCount << ") during sequence2. Stopping training early at sequence2 index " << i << "." << std::endl;
                break;
            }

            std::vector<float> expected_vec(d, 0.0f);
            expected_vec = sequence2[i];
            std::string& expected_str = rString[i];
            int current_block_idx = blockCount; // Block index based on current context size

            if (current_block_idx <= 0 || current_block_idx > m) {
                throw std::out_of_range("clTrainContext(sequence1-sequence2): Calculated current_block_idx (" 
                                        + std::to_string(current_block_idx) + ") is out of range [1, " 
                                        + std::to_string(m) + "].");
            }

            std::cout << "Training token " << i+1 << "/" << sequence2.size() << ": '" << expected_str << "'" << " at " << indexVec[i] << std::endl;
            std::cout << "current block: " << current_block_idx << " & current token count: " << currentTokenCount << " & eff. context size: " << effective_context_size <<std::endl;
            size_t currentBytes = static_cast<size_t>(EMBEDDING) * effective_context_size * sizeof(float);

            int j = 0;
            std::fill(oneHotEncode.begin(), oneHotEncode.end(), 0.0f);
            oneHotEncode[indexVec[i]] = 1.0f;

            while (j < epochs) {
                // add
                embedPlusPos = tokenEmbed + positional;
                CL_CHECK(clcontext.queue.enqueueWriteBuffer(d_tok_cl, CL_TRUE, 0, currentBytes, embedPlusPos.mapped_data));
                if(current_block_idx == 1) {
                    // keys and queries for each head of first block
                    int tokInContext = i;
                    for (int layer_idx = 0; layer_idx < x; ++layer_idx) {
                        for (int parallel_idx = 0; parallel_idx < y; ++parallel_idx) {
                            auto& qMat = blocks[0].b[layer_idx][parallel_idx].MQ;
                            auto& kMat = blocks[0].b[layer_idx][parallel_idx].MK;

                            // queries <- tokenEmbed
                            CL_CHECK(clcontext.queue.enqueueWriteBuffer(d_mQ_cl, CL_TRUE, 0, projection_matrix_bytes, qMat.mapped_data));
                            kq_kernel.setArg(3, effective_context_size); kq_kernel.setArg(4, EMBEDDING); kq_kernel.setArg(5, CONTEXT_WIN);
                            kq_kernel.setArg(0, d_tok_cl); kq_kernel.setArg(1, d_mQ_cl); kq_kernel.setArg(2, d_Q_cl);
                            CL_CHECK(clcontext.queue.enqueueNDRangeKernel(kq_kernel, cl::NullRange, cl::NDRange(2), cl::NullRange));
                            CL_CHECK(clcontext.queue.enqueueReadBuffer(d_Q_cl, CL_TRUE, 0, KQmatbytes, blocks[0].b[layer_idx][parallel_idx].Q.mapped_data));

                            // keys <- tokenEmbed
                            CL_CHECK(clcontext.queue.enqueueWriteBuffer(d_mK_cl, CL_TRUE, 0, projection_matrix_bytes, kMat.mapped_data));
                            kq_kernel.setArg(0, d_tok_cl); kq_kernel.setArg(1, d_mK_cl); kq_kernel.setArg(2, d_K_cl);
                            CL_CHECK(clcontext.queue.enqueueNDRangeKernel(kq_kernel, cl::NullRange, cl::NDRange(2), cl::NullRange));
                            CL_CHECK(clcontext.queue.enqueueReadBuffer(d_K_cl, CL_TRUE, 0, KQmatbytes, blocks[0].b[layer_idx][parallel_idx].K.mapped_data));
                        }
                    }
                    clcontext.queue.finish();
                }
                else {
                    int tokInContext = currentTokenCount % CONTEXT_WIN;
                    cl::Buffer pEV = cl::Buffer(clcontext.context, CL_MEM_READ_ONLY, embedding_bytes_loc, nullptr, &cl_err); CL_CHECK(cl_err);
                    // start from last token of previous local context
                    size_t fromHereInTokenEmbed = static_cast<size_t>((CONTEXT_WIN) * (blockCount - 1) - 1) * sizeof(float);
                    const float* host_src_ptr = embedPlusPos.mapped_data + (fromHereInTokenEmbed / sizeof(float));
                    CL_CHECK(clcontext.queue.enqueueWriteBuffer(d_tok_cl, CL_TRUE, 0, currentBytes, host_src_ptr));

                    // keys and queries for each head of non-first block
                    for (int layer_idx = 0; layer_idx < x; ++layer_idx) {
                        for (int parallel_idx = 0; parallel_idx < y; ++parallel_idx) {
                            auto& qMat = blocks[blockCount - 1].b[layer_idx][parallel_idx].MQ;
                            auto& kMat = blocks[blockCount - 1].b[layer_idx][parallel_idx].MK;
                            auto& prevEV = blocks[blockCount - 2].b[layer_idx][parallel_idx].EV;
                            CL_CHECK(clcontext.queue.enqueueWriteBuffer(pEV, CL_TRUE, 0, currentBytes, prevEV.mapped_data));

                            // queries <- EVs of previous block
                            CL_CHECK(clcontext.queue.enqueueWriteBuffer(d_mQ_cl, CL_TRUE, 0, projection_matrix_bytes, qMat.mapped_data));
                            kq_kernel.setArg(0, pEV); kq_kernel.setArg(1, d_mQ_cl); kq_kernel.setArg(2, d_Q_cl);
                            kq_kernel.setArg(3, effective_context_size); kq_kernel.setArg(4, EMBEDDING); kq_kernel.setArg(5, CONTEXT_WIN);
                            CL_CHECK(clcontext.queue.enqueueNDRangeKernel(kq_kernel, cl::NullRange, cl::NDRange(2), cl::NullRange));
                            CL_CHECK(clcontext.queue.enqueueReadBuffer(d_Q_cl, CL_TRUE, 0, KQmatbytes, blocks[blockCount - 1].b[layer_idx][parallel_idx].Q.mapped_data));

                            // keys <- TokenEmbed from CONTEXT_WIN*(blockCount-1) - 1
                            CL_CHECK(clcontext.queue.enqueueWriteBuffer(d_mK_cl, CL_TRUE, 0, projection_matrix_bytes, kMat.mapped_data));
                            kq_kernel.setArg(0, d_tok_cl); kq_kernel.setArg(1, d_mK_cl); kq_kernel.setArg(2, d_K_cl);
                            kq_kernel.setArg(3, effective_context_size); kq_kernel.setArg(4, EMBEDDING); kq_kernel.setArg(5, CONTEXT_WIN);
                            CL_CHECK(clcontext.queue.enqueueNDRangeKernel(kq_kernel, cl::NullRange, cl::NDRange(2), cl::NullRange));
                            CL_CHECK(clcontext.queue.enqueueReadBuffer(d_K_cl, CL_TRUE, 0, KQmatbytes, blocks[blockCount - 1].b[layer_idx][parallel_idx].K.mapped_data));
                        }
                    }
                    clcontext.queue.finish();
                }

                // --- Forward Pass ---
                // Pass the promptCount relevant for the *current* block/context state
                int current_prompt_count_in_block = effective_context_size % CONTEXT_WIN;
                if (current_prompt_count_in_block == 0 && effective_context_size > 0) current_prompt_count_in_block = CONTEXT_WIN;
                clForward(current_block_idx, effective_context_size, current_prompt_count_in_block);

                // --- Get EH output ---
                if (y > 0) {
                    std::fill(otok.begin(), otok.end(), 0.0f);
                    for (int j = 0; j < x; ++j) {
                        for (int k = 0; k < d; ++k) {
                            otok[(j * d) +k] = blocks[blockCount-1].b[j][y - 1].EH[k];  // sigmoid(blocks[blockCount-1].b[j][y - 1].EH[k]);
                        }
                    }
                }
                else {
                    std::cerr << "Warning: clForward called with y = 0 columns. Cannot accumulate EH." << std::endl;
                }
                for(size_t k_dim = 0; k_dim < static_cast<size_t>(x*d); k_dim++) {
                    if (std::isnan(otok[k_dim])) { otok[k_dim] = 0.0f; }
                    else if (std::isinf(otok[k_dim])) { otok[k_dim] = std::copysign((std::numeric_limits<float>::max)(), otok[k_dim]); }
                }
                if (contextTrain == 0 && otok.size() != static_cast<size_t>(d)) {
                    throw std::runtime_error("clTrainContext(sentence): otok from clForward has incorrect size: " + std::to_string(otok.size()) + " != " + std::to_string(d) + ".");
                }

                // use kernelComputePredictionWithScores for output prediction
                {
                    unsigned int host_indexForToken = -1;
                    cl::Buffer d_otok_buffer, d_predictions, d_result_index_buffer;
                    try {
                        size_t otok_bytes = static_cast<size_t>(x) * d * sizeof(float);
                        d_otok_buffer = cl::Buffer(clcontext.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, otok_bytes, otok.data(), &cl_err);
                        d_predictions = cl::Buffer(clcontext.context, CL_MEM_WRITE_ONLY, predBytes, nullptr, &cl_err);
                        d_result_index_buffer = cl::Buffer(clcontext.context, CL_MEM_WRITE_ONLY, sizeof(cl_int), nullptr, &cl_err);
                        CL_CHECK(predKernel.setArg(0, d_otok_buffer));
                        CL_CHECK(predKernel.setArg(1, d_deEmbeddings));
                        CL_CHECK(predKernel.setArg(2, d_predictions));
                        CL_CHECK(predKernel.setArg(3, d_result_index_buffer));
                        CL_CHECK(predKernel.setArg(4, static_cast<cl_int>(d * x)));
                        CL_CHECK(predKernel.setArg(5, static_cast<cl_int>(vocabsize)));
                        cl::NDRange global(1);
                        cl::NDRange local(1);
                        CL_CHECK(clcontext.queue.enqueueNDRangeKernel(predKernel, cl::NullRange, global, local));
                        CL_CHECK(clcontext.queue.enqueueReadBuffer(d_result_index_buffer, CL_TRUE, 0, indexBytes, &host_indexForToken));
                        CL_CHECK(clcontext.queue.enqueueReadBuffer(d_predictions, CL_TRUE, 0, predBytes, pred.data()));
                        indexForToken = host_indexForToken;
                        pred = softmax(pred);
                        CL_CHECK(cl_err);
                    }
                    catch (const std::exception& e) {
                        std::cerr << "Error during kernelComputePredictionWithScores in clTrainCONTEXT: " << e.what() << std::endl;
                        throw;
                    }
                }

                current_error = - std::log(pred[indexVec[i]] + 1e-15f);
                float del = current_error - prev_error;
                std::string predicted_token_str = (indexForToken >= 0 && indexForToken < static_cast<unsigned int>(tokens.size()))
                                                  ? tokens[indexForToken] : "INVALID_INDEX";

                std::cout << predicted_token_str << " ( " << indexForToken << " ) \t: "
                          << current_error << " | " << del << " | "
                          << std::exp(current_error) << " | " << j+1 << " | " << learning << std::endl;

                if (predicted_token_str == expected_str && predicted_token_str != "INVALID_INDEX") {
                    std::cout << "Token '" << expected_str << "' predicted correctly after " 
                              << j+1 << " epochs. Moving to next token." << std::endl;
                    if(predicted_token_str != "</s>")
                        std::cout << "              -------------- To Next Token --------------              " << std::endl;
                    break;
                }
                if(j == epochs - 1) {
                    std::cout << "Reached maximum epochs (" << epochs << ") for current token without correct prediction." << std::endl;
                    epochs += EPOCHS/2;
                    std::cout << "Increasing Epochs by " << EPOCHS/2 << "." << std::endl;
                }

                std::vector<float> gradEH(d * x, 0.0f);
                clUpdateDeEmbeddings(deEmbeddings, otok, pred, oneHotEncode, indexForToken, learning, lambda_L1, lambda_L2, gradEH);
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
                // backpropagate within block
                clBackwardContext(targets_for_heads, current_block_idx);
                // update embeddings which are in use
                clUpdateEmbeddings(embeddings, blocks[blockCount-1].gradToken, learning, lambda_L1, lambda_L2, 12454);
                clUpdateEmbeddings(tokenEmbed, blocks[blockCount-1].gradToken, learning, lambda_L1, lambda_L2, effective_context_size);

                totalLearning += learning;
                prev_error = current_error;
                totalBCELoss += current_error;
                totalBCEPerplexity += std::exp(current_error);
                j++;
            }

            // --- Update Host State ---
            trainCount++;
            epochCount += j;
            totalLearning += learning;
            prev_error = current_error;
            totalBCELoss += current_error;
            totalBCEPerplexity += std::exp(current_error);
            resCount += 1;
            currentTokenCount++;
            effective_context_size++;
            // add response embedding for predicted token
            tokenEmbed.addRow(sequence2[i], currentTokenCount);

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
        std::cerr << "Standard Exception in clTrainContext(sequence1-sequence2): " << e.what() << std::endl;
        epochs = initial_epochs;
        throw;
    }
}

#endif // USE_OPENCL