#ifdef USE_OPENCL
#include <maths.hpp>
#include "include/mlp.hpp"

// adam optimiser for mlp
void mlp::clAdamUpdate(OpenCLContext& clContext, unsigned long long t_adam, float beta1, float beta2, float epsilon, float learning_rate) {
    cl_int cl_err;

    // Get the Adam kernel from your context's map
    cl::Kernel adam_kernel;
    try {
        adam_kernel = clContext.kernels.at("adam_optimizer_kernel");
    } catch (const std::out_of_range& e) {
        std::cerr << "Error: OpenCL kernel 'adam_optimizer_kernel' not found in context.kernels map: " << e.what() << std::endl;
        throw; // Re-throw or handle appropriately
    }

    for (size_t i = 0; i < weights.size(); ++i) {
        // Ensure all required mapped data pointers are valid
        /*
        if (!weights[i].mapped_data || !gweights[i].mapped_data || !moments[i].mapped_data || !velocity[i].mapped_data) {
            std::cerr << "Warning: MLP Adam update for layer " << i << " detected unmapped data. Skipping this layer." << std::endl;
            continue;
        }

        size_t num_elements = static_cast<size_t>(weights[i].row) * weights[i].col;
        size_t bytes = num_elements * sizeof(float);

        // --- Device Buffers and Transfer Host Data to Device ---
        cl::Buffer d_weights = cl::Buffer(clContext.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, bytes, weights[i].mapped_data, &cl_err); CL_CHECK(cl_err);
        cl::Buffer d_gradients = cl::Buffer(clContext.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, gweights[i].mapped_data, &cl_err); CL_CHECK(cl_err);
        cl::Buffer d_moments = cl::Buffer(clContext.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, bytes, moments[i].mapped_data, &cl_err); CL_CHECK(cl_err);
        cl::Buffer d_velocity = cl::Buffer(clContext.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, bytes, velocity[i].mapped_data, &cl_err); CL_CHECK(cl_err);

        // --- Set Kernel Arguments ---
        CL_CHECK(adam_kernel.setArg(0, d_weights));
        CL_CHECK(adam_kernel.setArg(1, d_gradients));
        CL_CHECK(adam_kernel.setArg(2, d_moments));
        CL_CHECK(adam_kernel.setArg(3, d_velocity));
        CL_CHECK(adam_kernel.setArg(4, learning_rate));
        CL_CHECK(adam_kernel.setArg(5, beta1));
        CL_CHECK(adam_kernel.setArg(6, beta2));
        CL_CHECK(adam_kernel.setArg(7, epsilon));
        CL_CHECK(adam_kernel.setArg(8, t_adam));
        CL_CHECK(adam_kernel.setArg(9, static_cast<cl_int>(num_elements)));

        // --- Enqueue Kernel ---
        // Global work size should be the number of elements in the matrix.
        // Local work size can be NULL, letting OpenCL determine.
        CL_CHECK(clContext.queue.enqueueNDRangeKernel(adam_kernel, cl::NullRange, cl::NDRange(num_elements), cl::NullRange));

        // --- Read Updated Data back to Host Mapped Memory ---
        // CL_TRUE for blocking read, ensuring data is updated before the next iteration.
        CL_CHECK(clContext.queue.enqueueReadBuffer(d_weights, CL_TRUE, 0, bytes, weights[i].mapped_data));
        // CL_CHECK(clContext.queue.enqueueReadBuffer(d_moments, CL_TRUE, 0, bytes, moments[i].mapped_data));
        // CL_CHECK(clContext.queue.enqueueReadBuffer(d_velocity, CL_TRUE, 0, bytes, velocity[i].mapped_data));
        
        // Ensure all commands in the queue are finished before proceeding to the next matrix/loop iteration
        // (This might be redundant if enqueueReadBuffer is CL_TRUE, but good for explicit synchronization).
        clContext.queue.finish();
        */
    }
}


#endif // USE_OPENCL
