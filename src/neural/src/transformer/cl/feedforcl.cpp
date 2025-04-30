// transformer_cl_forward.cpp (or add to an existing transformer_cl.cpp)
#ifdef USE_OPENCL

#include "include/transformer.hpp" // Includes block.hpp, attention.hpp, mlp.hpp
#include <vector>
#include <numeric>   // std::accumulate
#include <stdexcept>
#include <iostream>
#include <limits>    // std::numeric_limits
#include <string>    // For error messages

/**
 * @brief OpenCL forward propagation for transformers.
 *        Mirrors the logic of cuForward using OpenCL kernels and functions provided
 *        by the block, attention, and mlp classes. Uses kernelComputePredictionIndex for prediction.
 * @param blockCount current block index (0-based).
 * @param currentTokenCount current total number of tokens processed.
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
    if (embeddings.empty() || vocabsize <= 0) {
         throw std::runtime_error("clForward: Embeddings not loaded or vocabsize is zero.");
    }
    if (d <= 0 || x <= 0 || y <= 0 || l <= 0) {
         throw std::runtime_error("clForward: Transformer dimensions (d, x, y, l) are not valid.");
    }

    // Ensure output token vector is sized correctly on the host
    if (otok.size() != static_cast<size_t>(d)) {
        otok.resize(d, 0.0f);
        std::cout << "clForward: Resized host otok vector to size " << d << std::endl;
    } else {
        // Reset host accumulator before accumulating new values
        std::fill(otok.begin(), otok.end(), 0.0f);
    }

    try {
        // Step 1: Compute KdotQ matrices (Assumed handled within block::clForprop)
        // ... (No change from previous version) ...

        // Step 2 & 3: Perform forward propagation for the relevant block using its OpenCL method
        // ... (No change from previous version) ...
        if (blockCount == 0) {
            std::cout << "clForward: Executing clForprop for Block 0..." << std::endl;
            t[0].clForprop(d, currentTokenCount, l);
            std::cout << "clForward: Block 0 clForprop finished." << std::endl;
        } else {
            if (static_cast<size_t>(blockCount - 1) >= t.size()) {
                 throw std::logic_error("clForward: Invalid blockCount logic: trying to access EV from non-existent previous block index " + std::to_string(blockCount - 1));
            }
            std::cout << "clForward: Executing clForprop for Block " << blockCount << " using EV from Block " << blockCount - 1 << "..." << std::endl;
            t[blockCount].clForprop(t[blockCount - 1].EV, d, currentTokenCount, blockCount, l, n);
             std::cout << "clForward: Block " << blockCount << " clForprop finished." << std::endl;
        }
        // Assumed synchronization happens within block::clForprop

        // Step 3.5: Accumulate EH from the last column of the current block (HOST-SIDE)
        // ... (No change from previous version) ...
        std::cout << "clForward: Accumulating EH from last column (y-1=" << y-1 << ") of Block " << blockCount << " on host..." << std::endl;
        if (y > 0) {
            for (int j = 0; j < x; ++j) {
                const std::vector<float>& eh_vector = t[blockCount].b[j][y - 1].EH;
                if (eh_vector.size() != static_cast<size_t>(d)) {
                    throw std::runtime_error("clForward: EH vector size mismatch during host accumulation for head ["
                                             + std::to_string(j) + "][" + std::to_string(y - 1) + "]. Expected "
                                             + std::to_string(d) + ", got " + std::to_string(eh_vector.size()));
                }
                for (int k = 0; k < d; ++k) {
                    otok[k] += eh_vector[k];
                }
            }
             std::cout << "clForward: Host EH accumulation finished." << std::endl;
        } else {
             std::cerr << "Warning: clForward called with y=0 columns. Cannot accumulate EH." << std::endl;
        }


        // ====================================================================
        // Step 4: Compute Output Token Index using kernelComputePredictionIndex
        // ====================================================================
        std::cout << "clForward: Computing output token index using kernelComputePredictionIndex..." << std::endl;
        cl::Buffer d_otok_buffer;
        cl::Buffer d_embeddings_buffer;
        cl::Buffer d_result_index_buffer; // Buffer for the single integer result

        try {
            // Flatten embeddings (assuming embeddings is vector<vector<float>>)
            std::vector<float> flat_embeddings = flatten(embeddings);
            size_t embeddings_bytes = flat_embeddings.size() * sizeof(float);
            if (embeddings_bytes != static_cast<size_t>(vocabsize) * d * sizeof(float)) {
                 throw std::runtime_error("clForward: Flattened embeddings size mismatch. Expected "
                                          + std::to_string(static_cast<size_t>(vocabsize) * d * sizeof(float)) + " bytes, got "
                                          + std::to_string(embeddings_bytes) + " bytes.");
            }

            // Create device buffers for input EH (otok) and embeddings
            size_t otok_bytes = otok.size() * sizeof(float);
            d_otok_buffer = cl::Buffer(this->clcontext.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, otok_bytes, otok.data());
            d_embeddings_buffer = cl::Buffer(this->clcontext.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, embeddings_bytes, flat_embeddings.data());

            // Create device buffer for the single integer output index
            d_result_index_buffer = cl::Buffer(this->clcontext.context, CL_MEM_WRITE_ONLY, sizeof(cl_int));

            // Get the prediction kernel
            cl::Kernel predictionKernel;
            try {
                // Use the specific kernel name defined previously
                predictionKernel = this->clcontext.kernels.at("kernelComputePredictionIndex");
            } catch (const std::out_of_range& oor) {
                 throw std::runtime_error("Kernel 'kernelComputePredictionIndex' not found. Ensure it's defined in a .cl file and loaded during setup.");
            }

            // Set kernel arguments
            predictionKernel.setArg(0, d_otok_buffer);              // __global const float* EH
            predictionKernel.setArg(1, d_embeddings_buffer);        // __global const float* embeddings
            predictionKernel.setArg(2, d_result_index_buffer);      // __global int* result_index
            predictionKernel.setArg(3, static_cast<cl_int>(d));         // int dim
            predictionKernel.setArg(4, static_cast<cl_int>(vocabsize)); // int voc

            // Define NDRange for a SINGLE work-item (as the kernel handles the loop)
            cl::NDRange global_size(1);
            cl::NDRange local_size(1);

            // Enqueue the kernel
            this->clcontext.queue.enqueueNDRangeKernel(predictionKernel, cl::NullRange, global_size, local_size);

            // Read the single integer result back to the host member variable indexForToken
            // Use blocking read (CL_TRUE) to ensure the kernel finishes and the result is available.
            this->clcontext.queue.enqueueReadBuffer(d_result_index_buffer, CL_TRUE, 0, sizeof(cl_int), &this->indexForToken);

            std::cout << "clForward: kernelComputePredictionIndex finished. Predicted index: " << this->indexForToken << std::endl;

            // Device buffers (d_otok_buffer, d_embeddings_buffer, d_result_index_buffer)
            // will be released automatically by cl::Buffer RAII destructor.

        } catch (const cl::Error& err) {
             std::cerr << "OpenCL Error during prediction kernel execution: " << err.what() << " (" << err.err() << ")" << std::endl;
             // Buffers are released by RAII
             throw; // Re-throw
        } catch (const std::exception& e) {
             std::cerr << "Standard Exception during prediction kernel execution: " << e.what() << std::endl;
             // Buffers are released by RAII
             throw; // Re-throw
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
        // Check for termination token (example)
        // if (indexForToken >= 0 && indexForToken < vocabsize && tokens[indexForToken] == TERMINATE) {
        //     isTerminate = true;
        // }

    } catch (const cl::Error& err) {
        std::cerr << "OpenCL Error in transformer::clForward: " << err.what() << " (" << err.err() << ")" << std::endl;
        if (err.err() == CL_BUILD_PROGRAM_FAILURE) {
             try {
                 std::string log = this->clcontext.program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(this->clcontext.device);
                 std::cerr << "Build Log:\n" << log << std::endl;
             } catch(...) { /* Ignore errors getting build log */ }
        }
        throw; // Re-throw exception
    } catch (const std::exception& e) {
        std::cerr << "Standard Exception in transformer::clForward: " << e.what() << std::endl;
        throw; // Re-throw exception
    }
}


// The standalone clComputeOutput function is no longer needed by clForward.
// You can remove it or keep it if it's used elsewhere.
/*
void clComputeOutput(cl::Context& context, cl::CommandQueue& queue, std::map<std::string, cl::Kernel>& kernels,
                     cl::Buffer& d_output, cl::Buffer& d_embeddings, int voc_size, int& index, int embedding_dim)
{
    // ... (Implementation using computeAllDotsKernel and host-side reduction) ...
    // THIS FUNCTION IS NO LONGER CALLED BY THE UPDATED clForward
}
*/


#endif // USE_OPENCL
