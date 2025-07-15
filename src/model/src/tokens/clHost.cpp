#ifdef USE_OPENCL
#include <chrono>
#include "include/tokenise.hpp"
#include <CL/cl.hpp>

// Now, the clEmbeddingFormula implementation within the tokeniser class
// Signature updated to include r1 and r2
void tokeniser::clEmbeddingFormula(OpenCLContext& ocl_context, std::vector<std::vector<float>>& embedding, const std::vector<float>& seeds_ignored, int& d_dim, 
    int& vocSize_val, float r1, float r2)
{
    if (!ocl_context.context() || !ocl_context.queue()) { // Use () for cl.hpp accessors
        std::cerr << "OpenCL context or command queue not initialized via singleton." << std::endl;
        return;
    }

    // Resize embedding vector to hold the results
    embedding.resize(vocSize_val, std::vector<float>(d_dim));

    // Calculate total number of elements
    size_t total_elements = (size_t)vocSize_val * d_dim;
    if (total_elements == 0) return;

    cl_int err;

    // Create a flat array for host memory, then copy back to 2D vector
    std::vector<float> flat_embeddings(total_elements);

    // Create device buffer
    cl::Buffer embeddings_buffer(ocl_context.context, CL_MEM_WRITE_ONLY, sizeof(float) * total_elements, NULL, &err);
    CL_CHECK(err);

    // Create kernel object
    cl::Kernel kernel = ocl_context.kernels.at("generate_embeddings");
    CL_CHECK(err);

    // Set Kernel Arguments
    kernel.setArg(0, embeddings_buffer);
    kernel.setArg(1, d_dim);
    kernel.setArg(2, r1); // <--- Using the passed r1
    kernel.setArg(3, r2); // <--- Using the passed r2
    unsigned int initial_seed_offset = static_cast<unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    kernel.setArg(4, initial_seed_offset);


    // Execute Kernel
    cl::NDRange global_work_size(total_elements);
    cl::NDRange local_work_size = cl::NullRange; // Let OpenCL decide optimal local size

    err = ocl_context.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_work_size, local_work_size);
    CL_CHECK(err);

    // Read Results Back
    err = ocl_context.queue.enqueueReadBuffer(embeddings_buffer, CL_TRUE, 0,
                                              sizeof(float) * total_elements, flat_embeddings.data());
    CL_CHECK(err);

    // Copy flat_embeddings to the 2D embedding vector
    for (int i = 0; i < vocSize_val; ++i) {
        for (int j = 0; j < d_dim; ++j) {
            embedding[i][j] = flat_embeddings[i * d_dim + j];
        }
    }
    // No explicit release needed for cl.hpp objects as they manage resources via RAII
}


// --- Host Wrapper for Vector Inverse ---
void tokeniser::clVectorInverse(OpenCLContext& ocl, std::vector<std::vector<float>>& deEmbedding,
    const std::vector<std::vector<float>>& embedding, int& d, int& vocSize) 
{
    if (vocSize == 0 || d == 0) return;
    if (embedding.size() != vocSize || (vocSize > 0 && embedding[0].size() != d)) {
        throw std::runtime_error("Input embedding dimensions do not match vocSize and d.");
    }

    // 1. Flatten input and resize output
    deEmbedding.assign(vocSize, std::vector<float>(d));
    std::vector<float> h_flat_input(vocSize * d);
    for (int i = 0; i < vocSize; ++i) {
        for (int j = 0; j < d; ++j) {
            h_flat_input[i * d + j] = embedding[i][j];
        }
    }

    // 2. Create device buffers
    cl_int err;
    cl::Buffer d_input(ocl.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (size_t)vocSize * d * sizeof(float), h_flat_input.data(), &err);
    CL_CHECK(err);
    cl::Buffer d_output(ocl.context, CL_MEM_WRITE_ONLY, (size_t)vocSize * d * sizeof(float), nullptr, &err);
    CL_CHECK(err);

    // 3. Create kernel and set arguments
    cl::Kernel  kernel = ocl.kernels.at("batchedVectorInverseKernel");
    CL_CHECK(err);
    CL_CHECK(kernel.setArg(0, d_output));
    CL_CHECK(kernel.setArg(1, d_input));
    CL_CHECK(kernel.setArg(2, vocSize));
    CL_CHECK(kernel.setArg(3, d));
    // Set the local memory argument
    const int block_size = 256;
    CL_CHECK(kernel.setArg(4, cl::Local(block_size * sizeof(float))));

    // 4. Configure and launch kernel
    cl::NDRange local_size(block_size, 1);
    cl::NDRange global_size(
        (d + local_size[0] - 1) / local_size[0] * local_size[0],
        vocSize
    );
    CL_CHECK(ocl.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_size, local_size));

    // 5. Read results back and un-flatten
    std::vector<float> h_flat_output(vocSize * d);
    CL_CHECK(ocl.queue.enqueueReadBuffer(d_output, CL_TRUE, 0, (size_t)vocSize * d * sizeof(float), h_flat_output.data()));
    
    for (int i = 0; i < vocSize; ++i) {
        for (int j = 0; j < d; ++j) {
            deEmbedding[i][j] = h_flat_output[i * d + j];
        }
    }
}

#endif