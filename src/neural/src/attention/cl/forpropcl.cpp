#ifdef USE_OPENCL

#include "include/attention.hpp"
#include <vector>
#include <stdexcept>
#include <iostream>
#include <string>
#include <CL/cl.hpp>

// Assume these OpenCL objects are globally accessible or passed appropriately
extern cl::Context cl_context;
extern cl::CommandQueue cl_queue;
extern cl::Program cl_program;

void attention::clForProp(int in, int layers) {
    // Basic validation
    if (in != EMBEDDING) {
        throw std::runtime_error("Embedding dimension mismatch");
    }
    if (tokenCount <= 0) {
        std::cerr << "Warning: clForProp called with tokenCount <= 0. Skipping." << std::endl;
        return;
    }

    try {
        // Kernel preparation
        cl::Kernel kernelComputeK(cl_program, "kernelComputeK");
        cl::Kernel kernelComputeQ(cl_program, "kernelComputeQ");
        cl::Kernel kernelKdotQ(cl_program, "kernelKdotQ");
        cl::Kernel kernelLOTA(cl_program, "kernelLOTA");
        
        // Buffer sizes
        size_t embed_bytes = sizeof(float) * in;
        size_t kq_bytes = sizeof(float) * tokenCount * MATHEIGHTS;
        size_t kdotq_bytes = sizeof(float) * tokenCount * tokenCount;

        // Device buffers
        cl::Buffer d_K(cl_context, CL_MEM_READ_WRITE, kq_bytes);
        cl::Buffer d_Q(cl_context, CL_MEM_READ_WRITE, kq_bytes);
        cl::Buffer d_KdotQ(cl_context, CL_MEM_READ_WRITE, kdotq_bytes);
        cl::Buffer d_head(cl_context, CL_MEM_READ_WRITE, kdotq_bytes);

        // Data transfer
        std::vector<float> flat_K = flatten(K);
        std::vector<float> flat_Q = flatten(Q);
        cl_queue.enqueueWriteBuffer(d_K, CL_TRUE, 0, kq_bytes, flat_K.data());
        cl_queue.enqueueWriteBuffer(d_Q, CL_TRUE, 0, kq_bytes, flat_Q.data());

        // NDRange config
        cl::NDRange global(tokenCount);
        cl::NDRange local = cl::NullRange;

        // Compute K and Q
        kernelComputeK.setArg(0, d_K);
        kernelComputeK.setArg(1, in);
        kernelComputeK.setArg(2, MATHEIGHTS);
        cl_queue.enqueueNDRangeKernel(kernelComputeK, cl::NullRange, global, local);

        kernelComputeQ.setArg(0, d_Q);
        kernelComputeQ.setArg(1, in);
        kernelComputeQ.setArg(2, MATHEIGHTS);
        cl_queue.enqueueNDRangeKernel(kernelComputeQ, cl::NullRange, global, local);

        // Compute KdotQ
        kernelKdotQ.setArg(0, d_K);
        kernelKdotQ.setArg(1, d_Q);
        kernelKdotQ.setArg(2, d_KdotQ);
        kernelKdotQ.setArg(3, tokenCount);
        kernelKdotQ.setArg(4, MATHEIGHTS);
        cl_queue.enqueueNDRangeKernel(kernelKdotQ, cl::NullRange, global, local);

        // Apply LOTA
        kernelLOTA.setArg(0, d_KdotQ);
        kernelLOTA.setArg(1, d_head);
        kernelLOTA.setArg(2, tokenCount);
        cl_queue.enqueueNDRangeKernel(kernelLOTA, cl::NullRange, global, local);

        // Read back results
        std::vector<float> flat_KdotQ(tokenCount * tokenCount);
        cl_queue.enqueueReadBuffer(d_KdotQ, CL_TRUE, 0, kdotq_bytes, flat_KdotQ.data());
        unflatten(flat_KdotQ, KdotQ, tokenCount, tokenCount);

    } catch (const cl::Error& err) {
        std::cerr << "OpenCL Error in attention::clForProp: " 
                  << err.what() << " (" << err.err() << ")" << std::endl;
        throw;
    }
}

#endif // USE_OPENCL