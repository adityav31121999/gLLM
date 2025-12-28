#ifdef USE_CU

#include "include/transformer.hpp"
#include "include/block.hpp"
#include "include/attention.hpp"
#include "include/mlp.hpp" // For errorofv, MSE
#include <maths.hpp>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <string>
#include <cmath>
#include <iomanip>
#include <algorithm>

// --- CUDA Error Checking Macro ---
#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                     \
do {                                                                         \
    cudaError_t err = call;                                                  \
    if (err != cudaSuccess) {                                                \
        fprintf(stderr, "CUDA Error in %s at line %d: %s (%d)\n",            \
                __FILE__, __LINE__, cudaGetErrorString(err), err);           \
        throw std::runtime_error("CUDA Error: " + std::string(cudaGetErrorString(err)));    \
    }                                                                        \
} while (0)
#endif

/**
 * @brief (CUDA) Test the transformer for sequence1 and sequence2.
 * @param sequence1 Sequence1 token embeddings (on host).
 * @param rString Tokens of the sequence2 (on host).
 */
void transformer::cuTest(std::vector<std::vector<float>>& sequence1, std::vector<std::string>& rString)
{
    // --- Basic validation ---
    if (sequence1.empty()) {
        throw std::runtime_error("cuTest: Initial sequence1 (prompt) cannot be empty.");
    }
    if (rString.empty()) {
        throw std::runtime_error("cuTest: rString (expected response) cannot be empty for testing.");
    }
    if (currentTokenCount + sequence1.size() + rString.size() > FULL_CONTEXT) {
        std::cerr << "Warning: cuTest: Adding prompt and expected response may exceed FULL_CONTEXT limit." << std::endl;
    }
    if (!sequence1.empty() && sequence1[0].size() != static_cast<size_t>(d)) {
        throw std::runtime_error("cuTest: Embedding dimension mismatch in sequence1.");
    }

    // Set for inference
    inTraining = 0;

    // --- Device Memory Allocation ---
    float* d_tokenEmbed = nullptr;      // Holds all token embeddings
    float* d_embeddings = nullptr;      // Holds the vocabulary embeddings
    float* d_EVuse = nullptr;           // Holds the EV state from the previous block
    float* d_tokForBlock = nullptr;     // Holds the token embeddings for the current block's context window
    float* d_otok = nullptr;            // Holds the output token embedding

    // Determine sizes
    size_t full_context_bytes = static_cast<size_t>(FULL_CONTEXT) * d * sizeof(float);
    size_t vocab_bytes = static_cast<size_t>(vocabsize) * d * sizeof(float);
    size_t context_win_bytes = static_cast<size_t>(CONTEXT_WIN) * d * sizeof(float);
    size_t ev_use_bytes = static_cast<size_t>(x) * y * CONTEXT_WIN * d * sizeof(float);
    size_t otok_bytes = static_cast<size_t>(d) * sizeof(float);

    std::vector<std::string> generated_response;
    int initial_token_count = currentTokenCount;

    try {
        CUDA_CHECK(cudaMalloc(&d_tokenEmbed, full_context_bytes));
        CUDA_CHECK(cudaMalloc(&d_embeddings, vocab_bytes));
        CUDA_CHECK(cudaMalloc(&d_EVuse, ev_use_bytes));
        CUDA_CHECK(cudaMalloc(&d_tokForBlock, context_win_bytes));
        CUDA_CHECK(cudaMalloc(&d_otok, otok_bytes));

        // --- Initial Data Transfer ---
        if (!embeddings.mapped_data || embeddings.row != vocabsize || embeddings.col != d) {
            throw std::runtime_error("Vocabulary embeddings mat is not properly initialized.");
        }
        CUDA_CHECK(cudaMemcpy(d_embeddings, embeddings.mapped_data, vocab_bytes, cudaMemcpyHostToDevice));

        std::cout << "--- Starting cuTest ---" << std::endl;
        std::cout << "Prompt Size: " << sequence1.size() << " | Expected Response Size: " << rString.size() << std::endl;

        // --- 1. Process sequence1 (Prompt) ---
        // Update host tokenEmbed first as we might need it for block transitions or debugging
        for(size_t i = 0; i < sequence1.size(); ++i) {
            tokenEmbed.addRow(sequence1[i] + positionalEmbeddings(currentTokenCount + i, d), currentTokenCount + i);
        }

        // Copy sequence1 to device
        std::vector<float> sequence1_embeddings_flat;
        sequence1_embeddings_flat.reserve(sequence1.size() * d);
        for(size_t i = 0; i < sequence1.size(); ++i) {
             // We use the already updated tokenEmbed to get pos-encoded vectors
             std::vector<float> vec = tokenEmbed(currentTokenCount + i);
             sequence1_embeddings_flat.insert(sequence1_embeddings_flat.end(), vec.begin(), vec.end());
        }
        
        CUDA_CHECK(cudaMemcpy(d_tokenEmbed + static_cast<size_t>(currentTokenCount) * d,
                                sequence1_embeddings_flat.data(),
                                sequence1_embeddings_flat.size() * sizeof(float),
                                cudaMemcpyHostToDevice));

        // Block Logic
        int sequence1Count = sequence1.size();
        int c = std::abs(currentTokenCount - (blockCount - 1) * CONTEXT_WIN);
        
        // Logic mirroring run.cu / test.cpp
        if (c + sequence1Count <= CONTEXT_WIN) {
             // Fits in current block
             if (blockCount > 1) {
                int start_idx = currentTokenCount - CONTEXT_WIN;
                if (start_idx < 0) start_idx = 0; // Should not happen if blockCount > 1
                CUDA_CHECK(cudaMemcpy(d_tokForBlock, d_tokenEmbed + static_cast<size_t>(start_idx) * d,
                                      context_win_bytes, cudaMemcpyDeviceToDevice));
             }
             
             for (int col = 0; col < y; ++col) {
                int effSeq1 = sequence1Count;
                cuParallelKdotQs(effSeq1, currentTokenCount, blockCount, col, isSelf, inTraining);
             }
             CUDA_CHECK(cudaDeviceSynchronize());
             currentTokenCount += sequence1Count;
        }
        else {
            // Spans across blocks
            int m2 = CONTEXT_WIN - c; // Space in current
            int m1 = sequence1Count - m2; // Next block

            // Part 1
            if (m2 > 0) {
                for (int col = 0; col < y; ++col) {
                    int effSeq1 = m2;
                    cuParallelKdotQs(effSeq1, currentTokenCount, blockCount, col, isSelf, inTraining);
                }
                CUDA_CHECK(cudaDeviceSynchronize());
                currentTokenCount += m2;
            }

            // Transition
            // Copy EV to EVuse
            size_t head_ev_bytes = static_cast<size_t>(CONTEXT_WIN) * d * sizeof(float);
            for (int i = 0; i < x; ++i) {
                for (int j = 0; j < y; ++j) {
                    float* d_src = blocks[0].b[i][j].d_EV;
                    size_t dest_offset = (static_cast<size_t>(i) * y + j) * CONTEXT_WIN * d;
                    CUDA_CHECK(cudaMemcpy(d_EVuse + dest_offset, d_src, head_ev_bytes, cudaMemcpyDeviceToDevice));
                }
            }
            CUDA_CHECK(cudaDeviceSynchronize());
            
            blockCount += 1;

            // Update d_tokForBlock
            int start_idx = currentTokenCount - CONTEXT_WIN;
            CUDA_CHECK(cudaMemcpy(d_tokForBlock, d_tokenEmbed + static_cast<size_t>(start_idx) * d,
                                  context_win_bytes, cudaMemcpyDeviceToDevice));

            // Part 2
            if (m1 > 0) {
                for (int col = 0; col < y; ++col) {
                    int effSeq1 = m1;
                    cuParallelKdotQs(effSeq1, currentTokenCount, blockCount, col, isSelf, inTraining);
                }
                CUDA_CHECK(cudaDeviceSynchronize());
                currentTokenCount += m1;
            }
        }

        // --- 2. Generation Loop ---
        for (size_t r = 0; r < rString.size(); ++r) {
            if (currentTokenCount >= FULL_CONTEXT) {
                std::cout << "\nTOKEN LIMIT REACHED AT FULL CONTEXT!" << std::endl;
                break;
            }

            int genSeq1Count = 0;
            cuForward(blockCount, currentTokenCount, genSeq1Count);
            CUDA_CHECK(cudaDeviceSynchronize());

            // Aggregate Output (EH) from device to host for prediction
            // Assuming cuForward updates blocks[0].b[i][y-1].d_EH on device
            std::fill(otok.begin(), otok.end(), 0.0f);
            std::vector<float> temp_eh(d);
            for(int i=0; i<x; ++i) {
                CUDA_CHECK(cudaMemcpy(temp_eh.data(), blocks[0].b[i][y-1].d_EH, d * sizeof(float), cudaMemcpyDeviceToHost));
                for(int k=0; k<d; ++k) otok[k] += temp_eh[k];
            }

            computePrediction(); // Host function using `otok` and `embeddings` (host)
            
            std::string gen_token = tokens[indexForToken];
            generated_response.push_back(gen_token);
            std::cout << gen_token << " " << std::flush;

            // Update state
            // Host update
            for(int i=0; i<d; ++i) tokenEmbed(currentTokenCount, i) = embeddings(indexForToken, i);
            if(currentTokenCount < mTokens.size()) mTokens[currentTokenCount] = gen_token;
            else mTokens.push_back(gen_token);

            // Device update
            CUDA_CHECK(cudaMemcpy(d_tokenEmbed + static_cast<size_t>(currentTokenCount) * d,
                                  tokenEmbed(currentTokenCount).data(),
                                  d * sizeof(float), cudaMemcpyHostToDevice));

            currentTokenCount++;

            // Block transition
            if (currentTokenCount % CONTEXT_WIN == 0) {
                // Copy EV to EVuse
                size_t head_ev_bytes = static_cast<size_t>(CONTEXT_WIN) * d * sizeof(float);
                for (int i = 0; i < x; ++i) {
                    for (int j = 0; j < y; ++j) {
                        float* d_src = blocks[0].b[i][j].d_EV;
                        size_t dest_offset = (static_cast<size_t>(i) * y + j) * CONTEXT_WIN * d;
                        CUDA_CHECK(cudaMemcpy(d_EVuse + dest_offset, d_src, head_ev_bytes, cudaMemcpyDeviceToDevice));
                    }
                }
                CUDA_CHECK(cudaDeviceSynchronize());
                
                blockCount += 1;
                
                int start_idx = currentTokenCount - CONTEXT_WIN;
                CUDA_CHECK(cudaMemcpy(d_tokForBlock, d_tokenEmbed + static_cast<size_t>(start_idx) * d,
                                      context_win_bytes, cudaMemcpyDeviceToDevice));
            }

            if (gen_token == "</s>") break;
        }

        std::cout << "\n\n--- Test Results ---" << std::endl;
        std::cout << "Expected: ";
        for(const auto& s : rString) std::cout << s << " ";
        std::cout << "\nGenerated: ";
        for(const auto& s : generated_response) std::cout << s << " ";
        std::cout << std::endl;

        int correct = 0;
        size_t min_len = std::min(rString.size(), generated_response.size());
        for(size_t i=0; i<min_len; ++i) {
            if(rString[i] == generated_response[i]) correct++;
        }
        float acc = rString.empty() ? 0.0f : (float)correct / rString.size() * 100.0f;
        std::cout << "Accuracy: " << std::fixed << std::setprecision(2) << acc << "%" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Standard Exception in cuTest: " << e.what() << std::endl;
        if (d_tokenEmbed) cudaFree(d_tokenEmbed);
        if (d_embeddings) cudaFree(d_embeddings);
        if (d_EVuse) cudaFree(d_EVuse);
        if (d_tokForBlock) cudaFree(d_tokForBlock);
        if (d_otok) cudaFree(d_otok);
        throw;
    }

    // Cleanup
    if (d_tokenEmbed) cudaFree(d_tokenEmbed);
    if (d_embeddings) cudaFree(d_embeddings);
    if (d_EVuse) cudaFree(d_EVuse);
    if (d_tokForBlock) cudaFree(d_tokForBlock);
    if (d_otok) cudaFree(d_otok);
}

#endif
