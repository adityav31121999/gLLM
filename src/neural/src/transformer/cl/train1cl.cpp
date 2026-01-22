#ifdef USE_CL
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
void transformer::clTrain(std::vector<std::vector<float>>& sentence, std::vector<std::string>& rString)
{
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

    cl_int cl_err;
    cl::Kernel kq_kernel, predKernel;
    cl::Buffer d_tokenEmbed, d_embeddings, d_expected_token;
    cl::Buffer d_Q_cl, d_K_cl, d_mQ_cl, d_mK_cl, d_tok_cl;
    float initial_learning_rate = learning;
    float current_error = 0.0f, prev_error = 0.0f;
    int initial_epochs = epochs;
    int initial_token_count = currentTokenCount;
    int effective_context_size = 0;
    bool blockShifted = 0;

    // --- Device Buffer Allocation & H->D Transfer ---
    size_t totalTokenEmbedFloats = static_cast<size_t>(m) * CONTEXT_WIN * d;
    size_t tokenEmbedBytes = totalTokenEmbedFloats * sizeof(float);
    size_t embeddingsBytes = static_cast<size_t>(vocabsize) * d * sizeof(float);
    size_t singleTokenBytes = static_cast<size_t>(d) * sizeof(float);
    size_t embedding_bytes_loc = static_cast<size_t>(EMBEDDING) * CONTEXT_WIN * sizeof(float);
    size_t projection_matrix_bytes = static_cast<size_t>(CONTEXT_WIN) * EMBEDDING * sizeof(float);
    size_t KQmatbytes = static_cast<size_t>(CONTEXT_WIN) * CONTEXT_WIN * sizeof(float);
    size_t indexBytes = sizeof(int);


    // allot buffers
    std::vector<float> flat_embeddings = ::flatten(embeddings);
    d_embeddings = cl::Buffer(clcontext.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, embeddingsBytes, flat_embeddings.data(), &cl_err); CL_CHECK(cl_err);
    d_tokenEmbed = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, tokenEmbedBytes, nullptr, &cl_err); CL_CHECK(cl_err);
    d_expected_token = cl::Buffer(clcontext.context, CL_MEM_READ_ONLY, singleTokenBytes, nullptr, &cl_err); CL_CHECK(cl_err); // Buffer for the target token
    d_Q_cl = cl::Buffer(clcontext.context, CL_MEM_WRITE_ONLY, KQmatbytes, nullptr, &cl_err); CL_CHECK(cl_err);
    d_K_cl = cl::Buffer(clcontext.context, CL_MEM_WRITE_ONLY, KQmatbytes, nullptr, &cl_err); CL_CHECK(cl_err);
    d_mQ_cl = cl::Buffer(clcontext.context, CL_MEM_READ_ONLY, projection_matrix_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
    d_mK_cl = cl::Buffer(clcontext.context, CL_MEM_READ_ONLY, projection_matrix_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
    d_tok_cl = cl::Buffer(clcontext.context, CL_MEM_READ_ONLY, embedding_bytes_loc, nullptr, &cl_err); CL_CHECK(cl_err);
    kq_kernel = clcontext.kernels.at("kernelComputeKQall");
    predKernel = clcontext.kernels.at("kernelComputePrediction");

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

        // --- Train for each subsequent token in the sentence (i=1 to N-1) ---
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
            CL_CHECK(clcontext.queue.enqueueWriteBuffer(d_expected_token, CL_TRUE, 0, singleTokenBytes, expected_vec.data()));

            std::cout << "Training token " << i+1 << "/" << sentence.size() << ": '" << expected_str << "'" << " at " << indexVec[i] << std::endl;
            std::cout << "current block: " << current_block_idx << " | current token count: " << currentTokenCount << " | eff. context size: " << effective_context_size <<std::endl;
            size_t currentBytes = static_cast<size_t>(EMBEDDING) * effective_context_size * sizeof(float);
            
            int j = 0;
            while (j < epochs) {
                if(current_block_idx == 1) {
                    // keys and queries for each head of first block
                    embedPlusPos = tokenEmbed + positional;
                    CL_CHECK(clcontext.queue.enqueueWriteBuffer(d_tok_cl, CL_TRUE, 0, currentBytes, embedPlusPos.mapped_data));
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
                    // start from previous local context's last token (only for this training)
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
                clForward(current_block_idx, effective_context_size, sequence1Count);
            
                // --- Get EH output ---
                if (y > 0) {
                    std::fill(otok.begin(), otok.end(), 0.0f);
                    for (int j = 0; j < x; ++j) {
                        for (int k = 0; k < d; ++k) {
                            otok[k] += blocks[blockCount-1].b[j][y - 1].EH[k];
                        }
                    }
                }
                else {
                    std::cerr << "Warning: clForward called with y = 0 columns. Cannot accumulate EH." << std::endl;
                }
                for(size_t k_dim = 0; k_dim < static_cast<size_t>(d); k_dim++) {
                    if (std::isnan(otok[k_dim])) { otok[k_dim] = 0.0f; }
                    else if (std::isinf(otok[k_dim])) { otok[k_dim] = std::copysign((std::numeric_limits<float>::max)(), otok[k_dim]); }
                }

                // use kernelComputePrediction for output prediction
                {
                    cl::Buffer d_otok, d_result_index_buffer;
                    try {
                        size_t otok_bytes = otok.size() * sizeof(float);
                        d_otok = cl::Buffer(clcontext.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, otok_bytes, otok.data(), &cl_err); CL_CHECK(cl_err);
                        d_result_index_buffer = cl::Buffer(clcontext.context, CL_MEM_WRITE_ONLY, sizeof(cl_int), nullptr, &cl_err); CL_CHECK(cl_err);
                        cl::Kernel kernel = clcontext.kernels.at("kernelComputePrediction");
                        CL_CHECK(kernel.setArg(0, d_otok));
                        CL_CHECK(kernel.setArg(1, d_embeddings));
                        CL_CHECK(kernel.setArg(2, d_result_index_buffer));
                        CL_CHECK(kernel.setArg(3, static_cast<cl_int>(d)));
                        CL_CHECK(kernel.setArg(4, static_cast<cl_int>(vocabsize)));
                        cl::NDRange global(1);
                        cl::NDRange local(1);
                        CL_CHECK(clcontext.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local));
                        int result_idx = -1;
                        CL_CHECK(clcontext.queue.enqueueReadBuffer(d_result_index_buffer, CL_TRUE, 0, sizeof(cl_int), &result_idx));
                        indexForToken = result_idx; // Also update the class member
                    }
                    catch (const std::exception& e) {
                        std::cerr << "Error during kernelComputePrediction in clTrain: " << e.what() << std::endl;
                        throw;
                    }
                }

                // calculate error
                current_error = binaryCrossEntropy(expected_vec, otok);
                float del = current_error - prev_error;
                std::string predicted_token_str = (indexForToken >= 0 && indexForToken < static_cast<unsigned long long>(tokens.size()))
                                                  ? tokens[indexForToken] : "INVALID_INDEX";

                std::cout << predicted_token_str << " ( " << indexForToken << " ) \t: "
                          << current_error << " | " << del << " | "
                          << std::exp(current_error) << " | " << j+1 << " | " << learning << std::endl;

                if (predicted_token_str == expected_str && predicted_token_str != "INVALID_INDEX") 
                {
                    std::cout << "Token '" << expected_str << "' predicted correctly after " 
                              << j+1 << " epochs. Moving to next token." << std::endl;
                    learning = initial_learning_rate;
                    if(predicted_token_str != "</s>")
                        std::cout << "              -------------- To Next Token --------------              " << std::endl;
                    break;
                }

                clBackward(expected_vec, current_block_idx);
                if (j > 0) learning = softsignLearning(del, learning);
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
    catch (const std::runtime_error& e) { // Catches std::runtime_error from CL_CHECK
        std::cerr << "Standard Exception in clTrain(sentence): " << e.what() << std::endl;
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
    d_embeddings = cl::Buffer();
    kq_kernel = cl::Kernel();
    predKernel = cl::Kernel();
}

#endif // USE_CL