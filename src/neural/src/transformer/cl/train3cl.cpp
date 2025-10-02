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
 * @brief (OpenCL) sequential training of the transformer.
 * @param sentence Token embeddings of the sentence (on host).
 * @param rString Sentence tokens (on host).
 */
void transformer::clTrainContext(std::vector<std::vector<float>>& sentence, std::vector<std::string>& rString)
{
    // --- Basic validation ---
    if(sentence.size() + currentTokenCount > FULL_CONTEXT) {
        throw std::runtime_error("clTrainContext(sentence): Previous tokens and sentence will exceed the FULL CONTEXT '-'");
    }
    if (sentence.size() > FULL_CONTEXT) {
        throw std::runtime_error("clTrainContext(sentence): Sentence size (" + std::to_string(sentence.size()) + ") exceeds FULL_CONTEXT (" + std::to_string(FULL_CONTEXT) + ").");
    }
    if (sentence.empty() || sentence.size() != rString.size()) {
        std::cout << "sentence.size(): " << sentence.size() << ", rString.size(): " << rString.size() << std::endl;
        throw std::runtime_error("clTrainContext(sentence): Sentence embeddings/strings mismatch or empty.");
    }
    if (!sentence.empty() && sentence[0].size() != static_cast<size_t>(d)) {
        throw std::runtime_error("clTrainContext(sentence): Sentence embedding dimension mismatch. Expected " + std::to_string(d) + ", got " + std::to_string(sentence[0].size()));
    }

    cl_int cl_err;          // error
    cl::Kernel kq_kernel, predKernel;
    cl::Buffer d_tokenEmbed, d_deEmbeddings, d_expected_token;    // embeddings
    cl::Buffer d_Q_cl, d_K_cl, d_mQ_cl, d_mK_cl, d_tok_cl;      // matrices and output
    float initial_learning_rate = learning; // Store initial learning rate
    float current_error = 0.0f;
    float prev_error = 0.0f;
    int initial_epochs = epochs;
    int initial_token_count = currentTokenCount; // Store initial count
    bool blockShifted = 0;
    int effective_context_size = 0;
    int start = 0;

    // --- Device Buffer Allocation & H->D Transfer ---
    size_t totalTokenEmbedFloats = static_cast<size_t>(m) * CONTEXT_WIN * d;
    size_t tokenEmbedBytes = totalTokenEmbedFloats * sizeof(float);
    size_t deEmbeddingsBytes = static_cast<size_t>(vocabsize) * x * d * sizeof(float);
    size_t singleTokenBytes = static_cast<size_t>(d) * sizeof(float);
    size_t embedding_bytes_loc = static_cast<size_t>(EMBEDDING) * CONTEXT_WIN * sizeof(float);
    size_t projection_matrix_bytes = static_cast<size_t>(CONTEXT_WIN) * EMBEDDING * sizeof(float);
    size_t KQmatbytes = static_cast<size_t>(CONTEXT_WIN) * CONTEXT_WIN * sizeof(float);
    size_t predBytes = static_cast<size_t>(vocabsize) * sizeof(float);
    size_t outputBytes = singleTokenBytes; // Size of h_otok_buffer
    size_t indexBytes = sizeof(int);       // Size for the result index

    // allot buffers
    std::vector<float> flat_deEmbeddings = ::flatten(deEmbeddings);
    d_deEmbeddings = cl::Buffer(clcontext.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, deEmbeddingsBytes, flat_deEmbeddings.data(), &cl_err); CL_CHECK(cl_err);
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
        otok.clear(); otok.resize(d*x, 0.0f);
        pred.clear(); pred.resize(vocabsize, 0.0f);
        oneHotEncode.clear(); oneHotEncode.resize(vocabsize, 0.0f);
        // check for currentTokenCount : if 0, start from original, else, continue from where left
        std::vector<float> flat_host_tokenEmbed(totalTokenEmbedFloats, 0.0f);
        // start training from first
        if(currentTokenCount == 0) {
            // set tokenEmbed
            blockCount = 1;
            tokenEmbed.addRow(sentence[0], 0);
            effective_context_size = 1;
            currentTokenCount += 1;
            start = 1;
        }
        // continue training in first block
        else if(currentTokenCount > 0 && currentTokenCount < CONTEXT_WIN) {
            // set tokens from currentTokenCount index
            blockCount = 1;
            effective_context_size = currentTokenCount;
            for (int tk = 0; tk < currentTokenCount; ++tk) { // Copy existing context
                float* row_ptr = tokenEmbed.mapped_data + (static_cast<size_t>(tk) * d);
                flat_host_tokenEmbed.insert(flat_host_tokenEmbed.end(), row_ptr, row_ptr + d);
            }
            start = 0;
        }
        // non-first block training
        else {
            effective_context_size = currentTokenCount % CONTEXT_WIN;
            blockCount = (currentTokenCount / CONTEXT_WIN) + 1;
            for (int tk = 0; tk < currentTokenCount; ++tk) {
                // Copy existing context from block specific tokForBlock
                float* row_ptr = tokenEmbed.mapped_data + (static_cast<size_t>(tk) * d);
                flat_host_tokenEmbed.insert(flat_host_tokenEmbed.end(), row_ptr, row_ptr + d);
            }
            start = 0;
        }

        sequence1Count = 1;
        std::cout << "Prediction | Index | Entropy LOSS | del | e^Loss | EPOCHS | Learning Rate" << std::endl;

        // --- Train for each subsequent token in the sentence (i=1 to N-1) ---
        for (size_t i = start; i < sentence.size(); ++i) {
            if (currentTokenCount >= FULL_CONTEXT) {
                std::cerr << "Warning: clTrainContext(sentence) reached FULL_CONTEXT limit ("
                          << currentTokenCount << "). Stopping training early at sentence index " << i << "." << std::endl;
                break;
            }

            // Target token for this iteration
            std::vector<float>& expected_vec = sentence[i];
            std::string& expected_str = rString[i];
            CL_CHECK(clcontext.queue.enqueueWriteBuffer(d_expected_token, CL_TRUE, 0, singleTokenBytes, expected_vec.data()));

            // Context size *before* adding token i
            int current_block_idx = blockCount;
            if (current_block_idx <= 0 || current_block_idx > m) {
                throw std::out_of_range("clTrainContext(sentence): Calculated current_block_idx (" + std::to_string(current_block_idx) + ") is out of range [1, " + std::to_string(m) + "].");
            }

            int j = 0;          // Epoch counter for this token
            std::fill(oneHotEncode.begin(), oneHotEncode.end(), 0.0f);
            oneHotEncode[indexVec[i]] = 1.0f;

            std::cout << "Training token " << i+1 << "/" << sentence.size() << ": '" << expected_str << "'" << " at " << indexVec[i] << std::endl;
            std::cout << "current block: " << current_block_idx << " | current token count: " << currentTokenCount << " | eff. context size: " << effective_context_size <<std::endl;
            size_t currentBytes = static_cast<size_t>(EMBEDDING) * effective_context_size * sizeof(float);
            CL_CHECK(clcontext.queue.enqueueWriteBuffer(d_tok_cl, CL_TRUE, 0, currentBytes, tokenEmbed.mapped_data));

            while (j < epochs) {
                current_error = 0.0f;

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
                    cl::Buffer pEV = cl::Buffer(clcontext.context, CL_MEM_READ_ONLY, embedding_bytes_loc, nullptr, &cl_err); CL_CHECK(cl_err);
                    // start from previous local context's last token
                    size_t fromHereInTokenEmbed = static_cast<size_t>((CONTEXT_WIN) * (blockCount - 1) - 1) * sizeof(float);
                    const float* host_src_ptr = tokenEmbed.mapped_data + (fromHereInTokenEmbed / sizeof(float));
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
                clForward(current_block_idx, effective_context_size, sequence1Count);

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
                    cl::Buffer d_otok_buffer, d_predictions, d_result_index_buffer;
                    try {
                        size_t otok_bytes = static_cast<size_t>(x) * d * sizeof(float);
                        d_otok_buffer = cl::Buffer(clcontext.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, otok_bytes, otok.data(), &cl_err); CL_CHECK(cl_err);
                        d_predictions = cl::Buffer(clcontext.context, CL_MEM_WRITE_ONLY, predBytes, nullptr, &cl_err); CL_CHECK(cl_err);
                        d_result_index_buffer = cl::Buffer(clcontext.context, CL_MEM_WRITE_ONLY, sizeof(cl_int), nullptr, &cl_err); CL_CHECK(cl_err);
                        CL_CHECK(predKernel.setArg(0, d_otok_buffer));
                        CL_CHECK(predKernel.setArg(1, d_deEmbeddings));
                        CL_CHECK(predKernel.setArg(2, d_predictions));
                        CL_CHECK(predKernel.setArg(3, d_result_index_buffer));
                        CL_CHECK(predKernel.setArg(4, static_cast<cl_int>(d * x)));
                        CL_CHECK(predKernel.setArg(5, static_cast<cl_int>(vocabsize)));
                        cl::NDRange global(1);
                        cl::NDRange local(1);
                        int host_indexForToken = -1;
                        CL_CHECK(clcontext.queue.enqueueNDRangeKernel(predKernel, cl::NullRange, global, local));
                        CL_CHECK(clcontext.queue.enqueueReadBuffer(d_result_index_buffer, CL_TRUE, 0, sizeof(cl_int), &host_indexForToken));
                        CL_CHECK(clcontext.queue.enqueueReadBuffer(d_predictions, CL_TRUE, 0, predBytes, pred.data()));
                        indexForToken = host_indexForToken;
                    }
                    catch (const std::exception& e) {
                        std::cerr << "Error during kernelComputePredictionWithScores in clTrainCONTEXT: " << e.what() << std::endl;
                        throw;
                    }
                }

                current_error = crossEntropy(oneHotEncode, pred);
                std::string predicted_token_str = (indexForToken >= 0 && indexForToken < static_cast<unsigned int>(tokens.size()))
                                                  ? tokens[indexForToken] : "INVALID_INDEX";

                std::cout << predicted_token_str << "\t: " << indexForToken << " | "
                          << std::fixed << std::setprecision(8) << current_error << " | "
                          << std::fixed << std::setprecision(8) << current_error - prev_error << " | "
                          << std::fixed << std::setprecision(8) << std::exp(current_error) << " | "
                          << j+1 << " | " << learning << std::endl;

                if (predicted_token_str == expected_str && predicted_token_str != "INVALID_INDEX") {
                    std::cout << "Token '" << expected_str << "' predicted correctly after " 
                              << j+1 << " epochs. Moving to next token." << std::endl;
                    if(predicted_token_str != "</s>")
                        std::cout << "              -------------- To Next Token --------------              " << std::endl;
                    break;
                }
                if(j == epochs - 1) {
                    std::cout << "Reached maximum epochs (" << epochs << ") for current token without correct prediction." << std::endl;
                    std::cout << "Increasing Epochs by 15." << std::endl;
                    epochs += 15;
                }

                // modify the de-embeddings and get gradients for backprop
                learning *= (current_error > prev_error) ? 1.05 : 0.95;
                std::vector<float> gradEH(d * x, 0.0f);
                clUpdateDeEmbeddings(deEmbeddings, otok, pred, oneHotEncode, indexForToken, learning, lambda_L1, lambda_L2, gradEH);
                // get expected target for backprop
                std::vector<std::vector<float>> targets_for_heads(x, std::vector<float>(EMBEDDING, 0.0f));
                for(int head_idx = 0; head_idx < x; ++head_idx) {
                    for(int eidx = 0; eidx < EMBEDDING; ++eidx) {
                        float gradient = learning * (gradEH[(head_idx * EMBEDDING) + eidx]
                                                  + (lambda_L1 * embeddings(indexForToken, eidx))
                                                  + (lambda_L2 * embeddings(indexForToken, eidx)));
                        if (fabs(gradient) >= MAX_GRAD_CLIP) gradient = std::copysign(MAX_GRAD_CLIP, gradient);
                        targets_for_heads[head_idx][eidx] = otok[(head_idx * EMBEDDING) + eidx] - gradient;
                    }
                }

                // backpropagate
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
            CL_CHECK(clcontext.queue.enqueueReadBuffer(d_tokenEmbed, CL_TRUE, currentBytes, singleTokenBytes, expected_vec.data()));
            if (tokenEmbed.mapped_data && static_cast<size_t>(currentTokenCount) < tokenEmbed.row && tokenEmbed.col == static_cast<size_t>(d)) {
                // place the embedding at the current location
                tokenEmbed.addRow(expected_vec, currentTokenCount - 1);
            }

            currentTokenCount++;
            effective_context_size++;
            if(currentTokenCount > 0 && currentTokenCount % CONTEXT_WIN == 0) {
                blockCount += 1;
                blockShifted = 1;
                std::cout << "----> Going to Next block in model -> " << blockCount - 1 << " to " << blockCount << std::endl;
            } else {
                blockShifted = 0;
            }
        }
        // learning = initial_learning_rate; // Reset for next line
    }
    catch (const std::runtime_error& e) { // Catches std::runtime_error from CL_CHECK
        std::cerr << "Standard Exception in clTrainContext(sentence): " << e.what() << std::endl;
        epochs = initial_epochs;
        throw;
    }

    d_tokenEmbed = cl::Buffer();
    d_expected_token = cl::Buffer();
    d_Q_cl = cl::Buffer();
    d_K_cl = cl::Buffer();
    d_mQ_cl = cl::Buffer();
    d_mK_cl = cl::Buffer();
    d_tok_cl = cl::Buffer();
    d_deEmbeddings = cl::Buffer();
    kq_kernel = cl::Kernel();
    predKernel = cl::Kernel();
}

#endif // USE_OPENCL
