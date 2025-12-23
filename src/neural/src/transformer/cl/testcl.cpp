#ifdef USE_CL
#if defined(_WIN64)
    #include <CL/cl.hpp>
#elif defined(__linux__)
    #include <CL/opencl.hpp>
#endif
#include "include/transformer.hpp"
#include <iomanip>

/**
 * @brief (OpenCL) Host-side test function for prompt-response testing.
 * @param sequence1 prompt embeddings
 * @param rString expected response vector
 */
void transformer::clTest(std::vector<std::vector<float>> &sequence1, std::vector<std::string> &rString)
{
    // --- Basic validation ---
    if (sequence1.empty()) {
        throw std::runtime_error("clTest: Initial sequence1 (prompt) cannot be empty.");
    }
    if (sequence1.size() > CONTEXT_WIN) {
        std::cerr << "Warning: clTest: sequence1 size (" << sequence1.size() << ") exceeds context window (" << CONTEXT_WIN << "). Ensure this is intended." << std::endl;
    }
    if (rString.empty()) {
        throw std::runtime_error("clTest: rString (expected response) cannot be empty for testing.");
    }
    if (currentTokenCount + sequence1.size() + rString.size() > FULL_CONTEXT) {
        std::cerr << "Warning: clTest: Adding prompt and expected response may exceed FULL_CONTEXT limit." << std::endl;
    }
    if (!sequence1.empty() && sequence1[0].size() != static_cast<size_t>(d)) {
        throw std::runtime_error("clTest: Embedding dimension mismatch in sequence1.");
    }

    cl_int cl_err;
    cl::Kernel kq_kernel, predKernel;
    cl::Buffer d_tokenEmbed, d_embeddings, d_deEmbeddings;
    cl::Buffer d_Q_cl, d_K_cl, d_mQ_cl, d_mK_cl, d_tok_cl;

    int initial_token_count = currentTokenCount;
    int effective_context_size = 0;
    std::vector<std::string> generated_response;

    // --- Device Buffer Allocation & H->D Transfer ---
    size_t totalTokenEmbedFloats = static_cast<size_t>(m) * CONTEXT_WIN * d;
    size_t tokenEmbedBytes = totalTokenEmbedFloats * sizeof(float);
    size_t embeddingsBytes = static_cast<size_t>(vocabsize) * d * sizeof(float);
    size_t deEmbeddingsBytes = static_cast<size_t>(vocabsize) * x * d * sizeof(float);
    size_t singleTokenBytes = static_cast<size_t>(d) * sizeof(float);
    size_t embedding_bytes_loc = static_cast<size_t>(EMBEDDING) * CONTEXT_WIN * sizeof(float);
    size_t projection_matrix_bytes = static_cast<size_t>(CONTEXT_WIN) * EMBEDDING * sizeof(float);
    size_t KQmatbytes = static_cast<size_t>(CONTEXT_WIN) * CONTEXT_WIN * sizeof(float);
    size_t indexBytes = sizeof(int);
    size_t predBytes = static_cast<size_t>(vocabsize) * sizeof(float);

    // Allot buffers
    std::vector<float> flat_embeddings = ::flatten(embeddings);
    d_embeddings = cl::Buffer(clcontext.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, embeddingsBytes, flat_embeddings.data(), &cl_err); CL_CHECK(cl_err);
    if (contextTrain) {
        std::vector<float> flat_deEmbeddings = ::flatten(deEmbeddings);
        d_deEmbeddings = cl::Buffer(clcontext.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, deEmbeddingsBytes, flat_deEmbeddings.data(), &cl_err); CL_CHECK(cl_err);
    }
    d_tokenEmbed = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, tokenEmbedBytes, nullptr, &cl_err); CL_CHECK(cl_err);
    d_Q_cl = cl::Buffer(clcontext.context, CL_MEM_WRITE_ONLY, KQmatbytes, nullptr, &cl_err); CL_CHECK(cl_err);
    d_K_cl = cl::Buffer(clcontext.context, CL_MEM_WRITE_ONLY, KQmatbytes, nullptr, &cl_err); CL_CHECK(cl_err);
    d_mQ_cl = cl::Buffer(clcontext.context, CL_MEM_READ_ONLY, projection_matrix_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
    d_mK_cl = cl::Buffer(clcontext.context, CL_MEM_READ_ONLY, projection_matrix_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
    d_tok_cl = cl::Buffer(clcontext.context, CL_MEM_READ_ONLY, embedding_bytes_loc, nullptr, &cl_err); CL_CHECK(cl_err);
    
    // Get kernels
    kq_kernel = clcontext.kernels.at("kernelComputeKQall");
    predKernel = contextTrain ? clcontext.kernels.at("kernelComputePredictionWithScores") : clcontext.kernels.at("kernelComputePrediction");

    std::cout << "--- Starting clTest ---" << std::endl;
    std::cout << "Prompt Size: " << sequence1.size() << " | Expected Response Size: " << rString.size() << std::endl;

    try {
        // --- 1. Process sequence1 (Prompt) ---
        // This logic mirrors the context setup from clTrain
        if (currentTokenCount == 0) {
            blockCount = 1;
            for(size_t i = 0; i < sequence1.size(); i++) {
                tokenEmbed.addRow(sequence1[i] + positionalEmbeddings(i, d), i);
            }
            currentTokenCount = sequence1.size();
            effective_context_size = currentTokenCount;
        }
        else {
            // Simplified logic for testing: assume prompt fits in the current block
            if ( (currentTokenCount % CONTEXT_WIN) + sequence1.size() > CONTEXT_WIN) {
                throw std::runtime_error("clTest: Prompt does not fit in the current block. This scenario is not handled in test function.");
            }
            blockCount = ((currentTokenCount - 1) / CONTEXT_WIN) + 1;
            for(size_t i = 0; i < sequence1.size(); i++) {
                tokenEmbed.addRow(sequence1[i] + positionalEmbeddings(currentTokenCount + i, d), currentTokenCount + i);
            }
            currentTokenCount += sequence1.size();
            effective_context_size = currentTokenCount % CONTEXT_WIN;
            if (effective_context_size == 0) effective_context_size = CONTEXT_WIN;
        }

        // --- 2. Auto-regressive Generation Loop ---
        for (size_t i = 0; i < rString.size(); ++i) {
            if (currentTokenCount >= FULL_CONTEXT) {
                std::cerr << "Warning: clTest reached FULL_CONTEXT limit. Stopping generation." << std::endl;
                break;
            }

            int current_block_idx = blockCount;
            size_t currentBytes = static_cast<size_t>(EMBEDDING) * effective_context_size * sizeof(float);

            // --- Forward Pass ---
            embedPlusPos = tokenEmbed + positional;
            CL_CHECK(clcontext.queue.enqueueWriteBuffer(d_tok_cl, CL_TRUE, 0, currentBytes, embedPlusPos.mapped_data));

            // Calculate K, Q for all heads
            for (int layer_idx = 0; layer_idx < x; ++layer_idx) {
                for (int parallel_idx = 0; parallel_idx < y; ++parallel_idx) {
                    auto& head = blocks[current_block_idx - 1].b[layer_idx][parallel_idx];
                    CL_CHECK(clcontext.queue.enqueueWriteBuffer(d_mQ_cl, CL_TRUE, 0, projection_matrix_bytes, head.MQ.mapped_data));
                    kq_kernel.setArg(0, d_tok_cl); kq_kernel.setArg(1, d_mQ_cl); kq_kernel.setArg(2, d_Q_cl);
                    kq_kernel.setArg(3, effective_context_size); kq_kernel.setArg(4, EMBEDDING); kq_kernel.setArg(5, CONTEXT_WIN);
                    CL_CHECK(clcontext.queue.enqueueNDRangeKernel(kq_kernel, cl::NullRange, cl::NDRange(2), cl::NullRange));
                    CL_CHECK(clcontext.queue.enqueueReadBuffer(d_Q_cl, CL_TRUE, 0, KQmatbytes, head.Q.mapped_data));

                    CL_CHECK(clcontext.queue.enqueueWriteBuffer(d_mK_cl, CL_TRUE, 0, projection_matrix_bytes, head.MK.mapped_data));
                    kq_kernel.setArg(0, d_tok_cl); kq_kernel.setArg(1, d_mK_cl); kq_kernel.setArg(2, d_K_cl);
                    CL_CHECK(clcontext.queue.enqueueNDRangeKernel(kq_kernel, cl::NullRange, cl::NDRange(2), cl::NullRange));
                    CL_CHECK(clcontext.queue.enqueueReadBuffer(d_K_cl, CL_TRUE, 0, KQmatbytes, head.K.mapped_data));
                }
            }
            clcontext.queue.finish();

            // Run forward pass for the block
            int promptCountForForward = 0; // Not relevant for this simplified test
            clForward(current_block_idx, effective_context_size, promptCountForForward);

            // Accumulate output
            if (contextTrain) {
                otok.assign(d * x, 0.0f);
                for (int j = 0; j < x; ++j) {
                    for (int k = 0; k < d; ++k) {
                        otok[(j * d) + k] = blocks[blockCount - 1].b[j][y - 1].EH[k];
                    }
                }
            }
            else {
                otok.assign(d, 0.0f);
                for (int j = 0; j < x; ++j) {
                    for (int k = 0; k < d; ++k) {
                        otok[k] += blocks[blockCount - 1].b[j][y - 1].EH[k];
                    }
                }
            }

            // --- Prediction ---
            int predicted_idx = -1;
            {
                cl::Buffer d_otok_buffer, d_result_index_buffer, d_predictions;
                size_t otok_bytes = otok.size() * sizeof(float);
                d_otok_buffer = cl::Buffer(clcontext.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, otok_bytes, otok.data(), &cl_err); CL_CHECK(cl_err);
                d_result_index_buffer = cl::Buffer(clcontext.context, CL_MEM_WRITE_ONLY, indexBytes, nullptr, &cl_err); CL_CHECK(cl_err);

                CL_CHECK(predKernel.setArg(0, d_otok_buffer));
                if (contextTrain) {
                    d_predictions = cl::Buffer(clcontext.context, CL_MEM_WRITE_ONLY, predBytes, nullptr, &cl_err); CL_CHECK(cl_err);
                    CL_CHECK(predKernel.setArg(1, d_deEmbeddings));
                    CL_CHECK(predKernel.setArg(2, d_predictions));
                    CL_CHECK(predKernel.setArg(3, d_result_index_buffer));
                    CL_CHECK(predKernel.setArg(4, static_cast<cl_int>(d * x)));
                    CL_CHECK(predKernel.setArg(5, static_cast<cl_int>(vocabsize)));
                }
                else {
                    CL_CHECK(predKernel.setArg(1, d_embeddings));
                    CL_CHECK(predKernel.setArg(2, d_result_index_buffer));
                    CL_CHECK(predKernel.setArg(3, static_cast<cl_int>(d)));
                    CL_CHECK(predKernel.setArg(4, static_cast<cl_int>(vocabsize)));
                }
                
                CL_CHECK(clcontext.queue.enqueueNDRangeKernel(predKernel, cl::NullRange, cl::NDRange(1), cl::NullRange));
                CL_CHECK(clcontext.queue.enqueueReadBuffer(d_result_index_buffer, CL_TRUE, 0, indexBytes, &predicted_idx));
            }

            if (predicted_idx < 0 || predicted_idx >= vocabsize) {
                std::cerr << "Error: Invalid token index predicted: " << predicted_idx << std::endl;
                generated_response.push_back("INVALID_INDEX");
                break;
            }

            std::string predicted_token_str = tokens[predicted_idx];
            generated_response.push_back(predicted_token_str);

            // --- Update state for next iteration ---
            std::vector<float> next_token_embed = embeddings(predicted_idx);
            tokenEmbed.addRow(next_token_embed + positionalEmbeddings(currentTokenCount, d), currentTokenCount);

            currentTokenCount++;
            effective_context_size++;

            if (currentTokenCount > 0 && currentTokenCount % CONTEXT_WIN == 0) {
                std::cout << "----> Block transition during testing <----" << std::endl;
                blockCount++;
                effective_context_size = 1; // Reset for the new block
                // In a real scenario, you'd handle EV transfer here. For this test, we assume it's not needed or handled by clForward.
            }

            if (predicted_token_str == "</s>") {
                break; // End of sentence
            }
        }

        // --- 3. Compare and Report Results ---
        std::cout << "\n--- Test Results ---" << std::endl;
        std::cout << "Expected Response: ";
        for(const auto& token : rString) std::cout << token << " ";
        std::cout << std::endl;

        std::cout << "Generated Response: ";
        for(const auto& token : generated_response) std::cout << token << " ";
        std::cout << std::endl;

        int correct_tokens = 0;
        size_t min_len = std::min<float>(rString.size(), generated_response.size());
        for(size_t i = 0; i < min_len; ++i) {
            if (rString[i] == generated_response[i]) {
                correct_tokens++;
            }
        }
        float accuracy = (rString.empty()) ? 0.0f : (static_cast<float>(correct_tokens) / rString.size()) * 100.0f;
        std::cout << "Correct Tokens: " << correct_tokens << "/" << rString.size() << std::endl;
        std::cout << "Accuracy: " << std::fixed << std::setprecision(5) << accuracy << "%" << std::endl;
        std::cout << "----------------------" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Standard Exception in clTest: " << e.what() << std::endl;
        throw;
    }
}

#endif