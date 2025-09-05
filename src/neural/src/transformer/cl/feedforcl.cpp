
#ifdef USE_OPENCL
#include "include/transformer.hpp"
#include <vector>
#include <numeric>
#include <stdexcept>
#include <iostream>
#include <limits>
#include <string>

/**
 * @brief OpenCL forward propagation for transformers.
 *        Mirrors the logic of cuForward using OpenCL kernels and functions provided
 *        by the block, attention, and mlp classes. Uses kernelComputePredictionIndex for prediction.
 * @param blockCount current block index (0-based).
 * @param currentTokenCount current total number of tokens processed. (in full cntext, not just the current block).
 * @param promptCount number of tokens in the current prompt segment being processed (unused in this specific logic but kept for signature consistency).
 */
void transformer::clForward(int &blockCount, int &currentTokenCount, int &promptCount)
{
    // --- Basic Validation ---
    if (blockCount < 0 || blockCount >= m) {
        throw std::out_of_range("clForward: blockCount (" + std::to_string(blockCount) + ") is out of range [0, " + std::to_string(m - 1) + "].");
    }
    if (t.empty() || static_cast<size_t>(blockCount) >= t.size()) {
        throw std::runtime_error("clForward: Transformer blocks not initialized or blockCount exceeds allocated blocks.");
    }
    if (t[blockCount].b.empty() || t[blockCount].b[0].empty()) {
        throw std::runtime_error("clForward: Attention heads not initialized for block " + std::to_string(blockCount) + ".");
    }
    if (embeddings.row <= 0 || embeddings.col <= 0 || embeddings.mapped_data == nullptr || vocabsize <= 0) {
        throw std::runtime_error("transformer::forward: Embeddings not loaded/initialized or vocabsize is zero/negative.");
    }
    if (d <= 0 || x <= 0 || y <= 0 || l <= 0) {
         throw std::runtime_error("clForward: Transformer dimensions (d, x, y, l) are not valid.");
    }

    // Ensure output token vector is sized correctly on the host
    if (otok.size() != static_cast<size_t>(d)) {
        otok.resize(d, 0.0f);
        std::cout << "clForward: Resized host otok vector to size " << d << std::endl;
    }
    else {
        // Reset host accumulator before accumulating new values
        std::fill(otok.begin(), otok.end(), 0.0f);
    }

    cl_int cl_err; // For OpenCL error codes

    try {
        // Step 1: Compute KdotQ matrices
        // Step 2 & 3: Perform forward propagation for the relevant block using its OpenCL method
        if (blockCount == 1) {
            // std::cout << "-> clForward: Executing clForprop for Block 1\n";
            // t[0].deserialise(t[0].blockFilePath);
            for(int i = 0; i < x; ++i) {
                clParallelKdotQs(promptCount, currentTokenCount, blockCount, i, isSelf, inTraining);
            }
            // std::cout << "clForward: Scaled Dot Products Calculated" << std::endl;
            t[0].clForprop(d, currentTokenCount, l);
            // std::cout << "clForward: Block 1 clForprop finished." << std::endl;
        }
        else {
            if (static_cast<size_t>(blockCount - 1) >= t.size()) {
                throw std::logic_error("clForward: Invalid blockCount logic: trying to access EV from non-existent previous block index " + std::to_string(blockCount - 1));
            }
            // std::cout << "-> clForward: Executing clForprop for Block " << blockCount << " using EV from Block " << blockCount - 1 << ":- ";
            // t[blockCount-1].deserialise(t[blockCount-1].blockFilePath);
            for(int i = 0; i < x; ++i) {
                clParallelKdotQs(promptCount, currentTokenCount, blockCount, i, isSelf, inTraining);
            }
            // std::cout << "clForward: Scaled Dot Products Calculated" << std::endl;
            t[blockCount-1].clForprop(t[blockCount - 1].EV, d, currentTokenCount, blockCount, l, n);
            // std::cout << "clForward: Block " << blockCount << " clForprop finished." << std::endl;
        }

        // Step 3.5: Accumulate EH from the last column of the current block (HOST-SIDE)
        // std::cout << "clForward: Accumulating EH from last column (y-1=" << y-1 << ") of Block " << blockCount << " on host..." << std::endl;
        if (y > 0) {
            for (int j = 0; j < x; ++j) {
                const std::vector<float>& eh_vector = t[blockCount-1].b[j][y - 1].EH;
                if (eh_vector.size() != static_cast<size_t>(d)) {
                    throw std::runtime_error("clForward: EH vector size mismatch during host accumulation for head ["
                                             + std::to_string(j) + "][" + std::to_string(y - 1) + "]. Expected "
                                             + std::to_string(d) + ", got " + std::to_string(eh_vector.size()));
                }
                for (int k = 0; k < d; ++k) {
                    otok[k] += eh_vector[k];
                }
            }
            // std::cout << "clForward: Host EH accumulation finished." << std::endl;
        } 
        else {
            std::cerr << "Warning: clForward called with y=0 columns. Cannot accumulate EH." << std::endl;
        }

/*        // ====================================================================
        // Step 4: Compute Output Token Index using kernelComputePredictionIndex
        // ====================================================================
        // std::cout << "clForward: Computing output token index using kernelComputePredictionIndex..." << std::endl;
        cl::Buffer d_otok_buffer;
        cl::Buffer d_embeddings_buffer;
        cl::Buffer d_result_index_buffer; // Buffer for the single integer result

        try {
            std::vector<float> flat_embeddings = ::flatten(embeddings); // Assuming global or from basic.hpp
            size_t embeddings_bytes = flat_embeddings.size() * sizeof(float);
            if (embeddings_bytes != static_cast<size_t>(vocabsize) * d * sizeof(float)) {
                 throw std::runtime_error("clForward: Flattened embeddings size mismatch. Expected " // Ensure ::flatten is in scope
                                          + std::to_string(static_cast<size_t>(vocabsize) * d * sizeof(float)) + " bytes, got "
                                          + std::to_string(embeddings_bytes) + " bytes.");
            }

            // Create device buffers for input EH (otok) and embeddings
            size_t otok_bytes = otok.size() * sizeof(float);
            d_otok_buffer = cl::Buffer(this->clcontext.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, otok_bytes, otok.data(), &cl_err); CL_CHECK(cl_err);
            d_embeddings_buffer = cl::Buffer(this->clcontext.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, embeddings_bytes, flat_embeddings.data(), &cl_err); CL_CHECK(cl_err);

            // Create device buffer for the single integer output index
            d_result_index_buffer = cl::Buffer(this->clcontext.context, CL_MEM_WRITE_ONLY, sizeof(cl_int), nullptr, &cl_err); CL_CHECK(cl_err);

            // Get the prediction kernel
            cl::Kernel predictionKernel;
            try {
                predictionKernel = this->clcontext.kernels.at("kernelComputePrediction");
            } 
            catch (const std::out_of_range& oor) {
                throw std::runtime_error("Kernel 'kernelComputePredictionIndex' not found. Ensure it's defined in a .cl file and loaded during setup.");
            }

            // Set kernel arguments
            CL_CHECK(predictionKernel.setArg(0, d_otok_buffer));
            CL_CHECK(predictionKernel.setArg(1, d_embeddings_buffer));
            CL_CHECK(predictionKernel.setArg(2, d_result_index_buffer));
            CL_CHECK(predictionKernel.setArg(3, static_cast<cl_int>(d)));
            CL_CHECK(predictionKernel.setArg(4, static_cast<cl_int>(vocabsize)));

            // Define NDRange for a SINGLE work-item (as the kernel handles the loop)
            cl::NDRange global_size(1);
            cl::NDRange local_size(1);

            // Enqueue the kernel
            CL_CHECK(this->clcontext.queue.enqueueNDRangeKernel(predictionKernel, cl::NullRange, global_size, local_size));
            CL_CHECK(this->clcontext.queue.enqueueReadBuffer(d_result_index_buffer, CL_TRUE, 0, sizeof(cl_int), &this->indexForToken));

            // std::cout << "clForward: kernelComputePredictionIndex finished. Predicted index: " << this->indexForToken \
            //           << ". Token is: " << tokens[this->indexForToken] << std::endl;
        }
        catch (const std::exception& e) {
            std::cerr << "Standard Exception during prediction kernel execution: " << e.what() << std::endl;
            throw;
        }
        // ================= End of Step 4 =====================================
        // Optional: Update host-side tokenEmbed based on indexForToken if needed
        // ... (No change from previous version) ...
        if (indexForToken >= 0 && indexForToken < vocabsize) {
            // Update logic if needed
        }
        else {
            std::cerr << "Warning: clForward resulted in invalid indexForToken: " << indexForToken << std::endl;
        }
*/
    }
    catch (const std::exception& e) {
        std::cerr << "Standard Exception in transformer::clForward: " << e.what() << std::endl;
        throw;
    }
    // std::cout << "clForward: Finished forward propagation for block " << blockCount << "." << std::endl;
}

#endif // USE_OPENCL
