#ifdef USE_OPENCL

#include <CL/cl.hpp>
#include "include/tokenise.hpp"

// --- Host Wrapper for Embedding Formula ---
void tokeniser::clEmbeddingFormula(OpenCLContext& ocl, std::vector<std::vector<float>>& embedding,
    const std::vector<float>& seeds, int& d, int& vocSize)
{
    if (vocSize == 0 || d == 0) return;
    if (seeds.size() != vocSize) {
        throw std::runtime_error("Seed vector size must match vocSize.");
    }

    // 1. Resize output vector
    embedding.assign(vocSize, std::vector<float>(d));
    
    // 2. Create device buffers
    cl_int err;
    cl::Buffer d_seeds(ocl.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, vocSize * sizeof(float), (void*)seeds.data(), &err);
    CHECK_CL(err);
    cl::Buffer d_embeddings(ocl.context, CL_MEM_WRITE_ONLY, (size_t)vocSize * d * sizeof(float), nullptr, &err);
    CHECK_CL(err);

    // 3. Create kernel, set arguments
    cl::Kernel kernel(ocl.embeddingProgram, "embeddingFormulaBatchKernel", &err);
    CHECK_CL(err);
    CHECK_CL(kernel.setArg(0, d_embeddings));
    CHECK_CL(kernel.setArg(1, d_seeds));
    CHECK_CL(kernel.setArg(2, vocSize));
    CHECK_CL(kernel.setArg(3, d));

    // 4. Configure work sizes and launch kernel
    cl::NDRange local_size(16, 16);
    cl::NDRange global_size(
        (d + local_size[0] - 1) / local_size[0] * local_size[0],
        (vocSize + local_size[1] - 1) / local_size[1] * local_size[1]
    );
    CHECK_CL(ocl.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_size, local_size));

    // 5. Read results back to a flat buffer and un-flatten
    std::vector<float> h_flat_output(vocSize * d);
    CHECK_CL(ocl.queue.enqueueReadBuffer(d_embeddings, CL_TRUE, 0, (size_t)vocSize * d * sizeof(float), h_flat_output.data()));
    
    for (int i = 0; i < vocSize; ++i) {
        for (int j = 0; j < d; ++j) {
            embedding[i][j] = h_flat_output[i * d + j];
        }
    }
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
    CHECK_CL(err);
    cl::Buffer d_output(ocl.context, CL_MEM_WRITE_ONLY, (size_t)vocSize * d * sizeof(float), nullptr, &err);
    CHECK_CL(err);

    // 3. Create kernel and set arguments
    cl::Kernel kernel(ocl.inverseProgram, "batchedVectorInverseKernel", &err);
    CHECK_CL(err);
    CHECK_CL(kernel.setArg(0, d_output));
    CHECK_CL(kernel.setArg(1, d_input));
    CHECK_CL(kernel.setArg(2, vocSize));
    CHECK_CL(kernel.setArg(3, d));
    // Set the local memory argument
    const int block_size = 256;
    CHECK_CL(kernel.setArg(4, cl::Local(block_size * sizeof(float))));

    // 4. Configure and launch kernel
    cl::NDRange local_size(block_size, 1);
    cl::NDRange global_size(
        (d + local_size[0] - 1) / local_size[0] * local_size[0],
        vocSize
    );
    CHECK_CL(ocl.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_size, local_size));

    // 5. Read results back and un-flatten
    std::vector<float> h_flat_output(vocSize * d);
    CHECK_CL(ocl.queue.enqueueReadBuffer(d_output, CL_TRUE, 0, (size_t)vocSize * d * sizeof(float), h_flat_output.data()));
    
    for (int i = 0; i < vocSize; ++i) {
        for (int j = 0; j < d; ++j) {
            deEmbedding[i][j] = h_flat_output[i * d + j];
        }
    }
}

#endif