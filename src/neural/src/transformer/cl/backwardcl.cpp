#ifdef USE_CL
#if defined(_WIN64) 
    #include <CL/cl.hpp>
#elif defined(__linux__)
    #include <CL/opencl.hpp>
#endif
#include <maths.hpp>
#include "include/mlp.hpp"
#include "include/attention.hpp"
#include "include/block.hpp"
#include "include/transformer.hpp"
#include <vector>
#include <stdexcept>
#include <iostream>
#include <string>


///////////////////////// static training /////////////////////////

/**
 * @brief OpenCL backward propagation from a specific block 'k' down to the first block (0).
 *        Uses a single common expected horizontal error vector for block 'k-1'.
 * @param expectedH Expected horizontal embedding for block 'k-1's output.
 * @param k The block number (1-based index) to start backpropagation from.
 * @param clip_norm Maximum L2 norm for gradient clipping (new parameter).
 */
void transformer::clBackward(std::vector<float>& expectedH, int& k) {
    if (k <= 0 || k > m) {
        throw std::out_of_range("clBackward(vector<float>, k): Block index k=" + std::to_string(k) + " is out of range [1, " + std::to_string(m) + "].");
    }

    int start_block_index = k - 1; // 0-based index
    // std::cout << "-> clBackward (H, start_block = " << k << ")" << std::endl;

    try {
        std::vector<float> exp = expectedH/8;
        // --- Starting Block (k-1) ---
        // Receives the external horizontal error signal 'expectedH'.
        // If k=1, this is the first block.
        if (start_block_index == 0) { // Starting from the first block (k=1)
            blocks[0].tokenCount = currentTokenCount;
            blocks[0].clbackward1stBlock(exp, d, l, learning, lambda_L1, lambda_L2);
        }
        else { // Handles all k > 1
            blocks[start_block_index].tokenCount = currentTokenCount % CONTEXT_WIN;
            blocks[start_block_index].clbackward(exp, start_block_index, d, l, learning, lambda_L1, lambda_L2);
        }
    } 
    catch (const std::exception& e) {
        throw std::runtime_error("Exception during transformer::clBackward(vector<float>, k=" + std::to_string(k) + "): " + std::string(e.what()));
    }
}


/**
 * @brief OpenCL backward propagation from a specific block 'k' down to the first block (0).
 *        Uses distinct expected horizontal error vectors (one per column/parallel) for block 'k-1'.
 * @param expectedH Vector of expected horizontal embeddings for block 'k-1's output (shape [y][EMBEDDING]).
 * @param k The block number (1-based index) to start backpropagation from.
 * @param clip_norm Maximum L2 norm for gradient clipping (new parameter).
 */
void transformer::clBackward(std::vector<std::vector<float>>& expectedH, int& k) {
    if (k <= 0 || k > m) {
        throw std::out_of_range("clBackward(vector<vector<float>>, k): Block index k=" + std::to_string(k) + " is out of range [1, " + std::to_string(m) + "].");
    }
    if (expectedH.size() != static_cast<size_t>(x)) { // Changed `y` to `x`
        throw std::runtime_error("clBackward(vector<vector<float>>, k): Outer dimension of expectedH (" + std::to_string(expectedH.size()) + ") does not match number of heads x (" + std::to_string(x) + ").");
    }
    if (!expectedH.empty() && expectedH[0].size() != EMBEDDING) {
        throw std::runtime_error("clBackward(vector<vector<float>>, k): Inner dimension of expectedH (" + std::to_string(expectedH[0].size()) + ") does not match EMBEDDING (" + std::to_string(EMBEDDING) + ").");
    }

    int start_block_index = k - 1; // 0-based index
    // std::cout << "-> clBackward (H_2D, start_block = " << k << ")" << std::endl;

    try {
        // --- Starting Block (k-1) ---
        // Receives the external horizontal error signal 'expectedH'.
        // If k=1, this is the first block.
        if (start_block_index == 0) { // Starting from the first block (k=1)
            blocks[0].tokenCount = currentTokenCount;
            blocks[0].clbackward1stBlock(expectedH, d, l, learning, lambda_L1, lambda_L2);
        }
        else {
            blocks[start_block_index].tokenCount = currentTokenCount % CONTEXT_WIN;
            blocks[start_block_index].clbackward(expectedH, start_block_index, d, l, learning, lambda_L1, lambda_L2);
        }
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Exception during transformer::clBackward(vector<vector<float>>, k=" + std::to_string(k) + "): " + std::string(e.what()));
    }
}

//////////////////// contextualised de-embedding from backprop gradient ////////////////////

/**
 * @brief Updates the deEmbeddings matrix and calculates the gradients for the EH vector.
 *        This function performs:
 *        1. LOTA activation on raw logits to get probabilities.
 *        2. Categorical Cross-Entropy backward pass for dL/du.
 *        3. Computes gradients for deEmbeddings (dL/dW_deEmbeddings).
 *        4. Updates deEmbeddings using Elastic Net regularization and clipping.
 *        5. Propagates error back to the input of deEmbeddings (dL/dEH).
 * @param deEmbeddings deEmbeddings ((d*x) * vocabsize)
 * @param otok Final hidden state (EH), concatenated, from the last layer of the block (size = d*x).
 * @param prediction Raw logits for each token in the vocabulary, result of (otok * deEmbeddings^T). Size: vocabsize.
 * @param oneHotEncode Host-side one-hot vector of true label. (Size: vocabsize)
 * @param indexForToken index of token for training
 * @param learning Learning rate.
 * @param lambda_L1 L1 regularization parameter.
 * @param lambda_L2 L2 regularization parameter.
 * @param clip_norm Maximum L2 norm for gradient clipping.
 * @param gradForEh Host-side vector to store gradients for the EH vector (dL/dEH). Size: d*x.
 *        This will be filled with the error to propagate back to the block.
 */
void transformer::clUpdateDeEmbeddings(mat& deEmbeddings, std::vector<float> prediction,
            std::vector<float> oneHotEncode, float learning, float lambda_L1, float lambda_L2,
            std::vector<float> &gradForEh)
{
    try {
        cl_int cl_err;
        cl::CommandQueue& queue = clcontext.queue;
        cl::Context context = clcontext.context;
        int vocab_size = this->vocabsize;
        int p_dim = d * x;

        // --- 0. Validate sizes and prepare host vectors ---
        if (prediction.size() != vocab_size || oneHotEncode.size() != vocab_size) {
            throw std::runtime_error("clUpdateDeEmbeddings: Input vector size mismatch. Expected " +
                                        std::to_string(vocab_size) + ", got " +
                                        std::to_string(prediction.size()) + " for prediction and " +
                                        std::to_string(oneHotEncode.size()) + " for oneHotEncode.");
        }
        if (otok.size() != static_cast<size_t>(p_dim)) {
            throw std::runtime_error("clUpdateDeEmbeddings: 'otok' (final_hidden_state_input) size mismatch. Expected " +
                                    std::to_string(p_dim) + ", got " + std::to_string(otok.size()) + ".");
        }
        if (gradForEh.size() != static_cast<size_t>(p_dim)) {
            gradForEh.resize(p_dim);
        }

        const size_t local_work_size_1d = 64;
        cl::NDRange local_1d(local_work_size_1d);
        auto calculate_global_1d = [&](size_t total_size) {
            return cl::NDRange(((total_size + local_work_size_1d - 1) / local_work_size_1d) * local_work_size_1d);
        };
        
        const size_t local_work_size_2d_arr[2] = { 8, 8 };
        cl::NDRange local_2d(local_work_size_2d_arr[0], local_work_size_2d_arr[1]); // This local_2d is useful for 2D kernels
        auto calculate_global_2d = [&](size_t dim0, size_t dim1) {
            // Access individual elements of the array using [0] and [1]
            size_t g0 = ((dim0 + local_work_size_2d_arr[0] - 1) / local_work_size_2d_arr[0]) * local_work_size_2d_arr[0];
            size_t g1 = ((dim1 + local_work_size_2d_arr[1] - 1) / local_work_size_2d_arr[1]) * local_work_size_2d_arr[1];
            return cl::NDRange(g0, g1);
        };

        // --- All buffers ---
        cl::Buffer d_deEmbed(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, vocab_size * p_dim * sizeof(float), deEmbeddings.mapped_data, &cl_err); CL_CHECK(cl_err);
        cl::Buffer d_gdeEmbed(context, CL_MEM_READ_WRITE, vocab_size * p_dim * sizeof(float), nullptr, &cl_err); CL_CHECK(cl_err);
        cl::Buffer d_prediction_raw_logits(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, vocab_size * sizeof(float), prediction.data(), &cl_err); CL_CHECK(cl_err);
        cl::Buffer d_final_hidden_state_input(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, p_dim * sizeof(float), otok.data(), &cl_err); CL_CHECK(cl_err);
        cl::Buffer d_oneHotEncode(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, vocab_size * sizeof(float), oneHotEncode.data(), &cl_err); CL_CHECK(cl_err);
        cl::Buffer d_predNorm(context, CL_MEM_READ_WRITE, vocab_size * sizeof(float), nullptr, &cl_err); CL_CHECK(cl_err); // Output of LOTA (P_hat)
        cl::Buffer d_delta(context, CL_MEM_READ_WRITE, vocab_size * sizeof(float), nullptr, &cl_err); CL_CHECK(cl_err); // dL/dz
        cl::Buffer d_gradForEh_device(context, CL_MEM_READ_WRITE, p_dim * sizeof(float), nullptr, &cl_err); CL_CHECK(cl_err);

        float zero = 0.0f;
        CL_CHECK(queue.enqueueFillBuffer(d_gdeEmbed, zero, 0, vocab_size * p_dim * sizeof(float)));

        // --- 1. LOTA activation on raw logits (`d_prediction_raw_logits`) to get probabilities (`d_predNorm`) ---
        cl::Kernel lota_kernel = clcontext.kernels.at("clLOTA1d");
        CL_CHECK(lota_kernel.setArg(0, d_prediction_raw_logits));
        CL_CHECK(lota_kernel.setArg(1, d_predNorm));
        CL_CHECK(lota_kernel.setArg(2, vocab_size));
        CL_CHECK(queue.enqueueNDRangeKernel(lota_kernel, cl::NullRange, calculate_global_1d(vocab_size), local_1d));
        CL_CHECK(queue.finish());

        // --- 2. Categorical Cross-Entropy backward pass for dL/dz (`d_delta`) ---
        cl::Kernel calculate_output_delta_lota_kernel = clcontext.kernels.at("kernelComputeGradpred");
        CL_CHECK(calculate_output_delta_lota_kernel.setArg(0, d_predNorm));
        CL_CHECK(calculate_output_delta_lota_kernel.setArg(1, d_oneHotEncode));
        CL_CHECK(calculate_output_delta_lota_kernel.setArg(2, d_delta));
        CL_CHECK(calculate_output_delta_lota_kernel.setArg(3, vocab_size));
        CL_CHECK(queue.enqueueNDRangeKernel(calculate_output_delta_lota_kernel, cl::NullRange, calculate_global_1d(vocab_size), local_1d));
        CL_CHECK(queue.finish());

        // --- 3. Compute gradients for deEmbeddings (`d_gdeEmbed`) ---
        // This is an outer product: grad[i,j] = delta[i] * otok[j].
        // We use a vectorized kernel that processes 4 elements at a time.
        cl::Kernel compute_gradient_de_embeddings_kernel = clcontext.kernels.at("KernelComputeGradDeEmbeddings");
        cl::NDRange global_size_grad_deEmbed_2d = calculate_global_2d(vocab_size, p_dim / 4);
        CL_CHECK(compute_gradient_de_embeddings_kernel.setArg(0, d_delta));
        CL_CHECK(compute_gradient_de_embeddings_kernel.setArg(1, d_final_hidden_state_input));
        CL_CHECK(compute_gradient_de_embeddings_kernel.setArg(2, d_gdeEmbed));
        CL_CHECK(compute_gradient_de_embeddings_kernel.setArg(3, vocab_size));
        CL_CHECK(compute_gradient_de_embeddings_kernel.setArg(4, p_dim / 4)); // Pass width_div_4
        CL_CHECK(queue.enqueueNDRangeKernel(compute_gradient_de_embeddings_kernel, cl::NullRange, global_size_grad_deEmbed_2d, local_2d));
        CL_CHECK(queue.finish());

        // --- 4. Update deEmbeddings weights (`d_deEmbed`) ---
        cl::Kernel update_weights_general_kernel = clcontext.kernels.at("kernelUpdateWeightsGeneral_f4");
        CL_CHECK(update_weights_general_kernel.setArg(0, d_deEmbed));
        CL_CHECK(update_weights_general_kernel.setArg(1, d_gdeEmbed));
        CL_CHECK(update_weights_general_kernel.setArg(2, learning));
        CL_CHECK(update_weights_general_kernel.setArg(3, lambda_L1));
        CL_CHECK(update_weights_general_kernel.setArg(4, lambda_L2));
        int totalElements = vocab_size * p_dim;
        CL_CHECK(update_weights_general_kernel.setArg(5, totalElements));
        CL_CHECK(queue.enqueueNDRangeKernel(update_weights_general_kernel, cl::NullRange, calculate_global_1d(vocab_size * p_dim), local_1d));
        CL_CHECK(queue.finish());

        // --- 5. Propagate error back to the input of deEmbeddings (`d_gradForEh_device`) ---
        cl::Kernel propagate_error_to_hidden_kernel = clcontext.kernels.at("kernelGradForAttentionOutput");
        CL_CHECK(propagate_error_to_hidden_kernel.setArg(0, d_deEmbed));
        CL_CHECK(propagate_error_to_hidden_kernel.setArg(1, d_delta));
        CL_CHECK(propagate_error_to_hidden_kernel.setArg(2, d_gradForEh_device));
        CL_CHECK(propagate_error_to_hidden_kernel.setArg(3, vocab_size));
        CL_CHECK(propagate_error_to_hidden_kernel.setArg(4, p_dim));
        CL_CHECK(queue.enqueueNDRangeKernel(propagate_error_to_hidden_kernel, cl::NullRange, calculate_global_1d(p_dim), local_1d));
        CL_CHECK(queue.finish());

        // --- Transfer updated deEmbeddings weights and propagated gradient back to host `mat` and vector ---
        CL_CHECK(queue.enqueueReadBuffer(d_deEmbed, CL_TRUE, 0, vocab_size * p_dim * sizeof(float), deEmbeddings.mapped_data));
        CL_CHECK(queue.enqueueReadBuffer(d_gradForEh_device, CL_TRUE, 0, p_dim * sizeof(float), gradForEh.data()));

        d_deEmbed = cl::Buffer();
        d_gdeEmbed = cl::Buffer();
        d_prediction_raw_logits = cl::Buffer();
        d_final_hidden_state_input = cl::Buffer();
        d_oneHotEncode = cl::Buffer();
        d_predNorm = cl::Buffer();
        d_delta = cl::Buffer();
        d_gradForEh_device = cl::Buffer();
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Exception during transformer::clUpdateDeEmbeddings: " + std::string(e.what()));
    }
}

///////////////////////// contextualised training /////////////////////////

/**
 * @brief OpenCL backward propagation from a specific block 'k' down to the first block (0).
 *        Uses distinct expected horizontal error vectors (one per column/parallel) for block 'k-1'.
 * @param expectedH Vector of expected horizontal embeddings for block 'k-1's output (shape [y][EMBEDDING]).
 * @param k The block number (1-based index) to start backpropagation from.
 * @param clip_norm Maximum L2 norm for gradient clipping (new parameter).
 */
void transformer::clBackwardContext(std::vector<std::vector<float>>& expectedH, int& k) {
    if (k <= 0 || k > m) {
        throw std::out_of_range("clBackwardContext(vector<vector<float>>, k): Block index k=" + std::to_string(k) + " is out of range [1, " + std::to_string(m) + "].");
    }
    if (expectedH.size() != static_cast<size_t>(x)) { // Changed `y` to `x`
        throw std::runtime_error("clBackwardContext(vector<vector<float>>, k): Outer dimension of expectedH (" + std::to_string(expectedH.size()) + ") does not match number of heads x (" + std::to_string(x) + ").");
    }
    if (!expectedH.empty() && expectedH[0].size() != EMBEDDING) {
        throw std::runtime_error("clBackwardContext(vector<vector<float>>, k): Inner dimension of expectedH (" + std::to_string(expectedH[0].size()) + ") does not match EMBEDDING (" + std::to_string(EMBEDDING) + ").");
    }

    int start_block_index = k - 1; // 0-based index
    // std::cout << "-> clBackwardContext (H_2D, start_block = " << k << ")" << std::endl;

    try {
        // --- Starting Block (k-1) ---
        // Receives the external horizontal error signal 'expectedH'.
        // If k=1, this is the first block.
        if (start_block_index == 0) { // Starting from the first block (k=1)
            blocks[0].tokenCount = currentTokenCount;
            blocks[0].clrbackward1stBlock(expectedH, d, l, learning, lambda_L1, lambda_L2);
        }
        else {
            blocks[start_block_index].tokenCount = currentTokenCount % CONTEXT_WIN;
            blocks[start_block_index].clbackward(expectedH, start_block_index, d, l, learning, lambda_L1, lambda_L2);
        }
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Exception during transformer::clBackwardContext(vector<vector<float>>, k=" + std::to_string(k) + "): " + std::string(e.what()));
    }
}

//////////////////// contextualised embedding from backprop gradients ////////////////////

/**
 * @brief update embedding using propagated gradients from training
 */
void transformer::clUpdateEmbeddings(mat tokenEmbedding, std::vector<float>& gradForEh, float learning,
    float lambda_L1, float lambda_L2, int rows)
{
    try {
        cl_int cl_err;
        cl::CommandQueue& queue = clcontext.queue;
        cl::Context context = clcontext.context;
        int vocab_size = vocabsize;
        int dim = d;
        size_t rowBytes = static_cast<size_t>(rows) * d * sizeof(float);

        cl::Buffer d_dembed(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, rowBytes, tokenEmbedding.mapped_data, &cl_err); CL_CHECK(cl_err);
        cl::Buffer d_gEmbed(context, CL_MEM_READ_WRITE, dim * sizeof(float), nullptr, &cl_err); CL_CHECK(cl_err);

        const size_t local_work_size_1d = 256;
        cl::NDRange local_1d(local_work_size_1d);
        auto calculate_global_1d = [&](size_t total_size) {
            return cl::NDRange(((total_size + local_work_size_1d - 1) / local_work_size_1d) * local_work_size_1d);
        };

        cl::Kernel update_weights_general_kernel = clcontext.kernels.at("updateEmbeddings");
        CL_CHECK(update_weights_general_kernel.setArg(0, d_dembed));
        CL_CHECK(update_weights_general_kernel.setArg(1, d_gEmbed));
        CL_CHECK(update_weights_general_kernel.setArg(2, learning));
        CL_CHECK(update_weights_general_kernel.setArg(3, lambda_L1));
        CL_CHECK(update_weights_general_kernel.setArg(4, lambda_L2));
        CL_CHECK(update_weights_general_kernel.setArg(5, vocab_size));
        CL_CHECK(update_weights_general_kernel.setArg(6, d));
        CL_CHECK(queue.enqueueNDRangeKernel(update_weights_general_kernel, cl::NullRange, calculate_global_1d(dim), local_1d));
        CL_CHECK(queue.finish());

        d_gEmbed = cl::Buffer();
        d_dembed = cl::Buffer();
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Exception during transformer::clUpdateEmbeddings: " + std::string(e.what()));
    }
}

#endif