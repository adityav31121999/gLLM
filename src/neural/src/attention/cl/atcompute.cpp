#ifdef USE_OPENCL
#include <maths.hpp>
#include "include/attention.hpp"
#include <iostream> // For std::cerr debugging

void attention::clAdamUpdate(OpenCLContext& clContext, unsigned long long t_adam, float beta1, float beta2,
    float epsilon, float learning_rate)
{
    cl_int cl_err; // Declare cl_err here for use with CL_CHECK macro
    cl::Kernel adam_kernel;
    try {
        adam_kernel = clContext.kernels.at("adam_optimizer_kernel");
    } catch (const std::out_of_range& e) {
        std::cerr << "Error: OpenCL kernel 'adam_optimizer_kernel' not found in context.kernels map: " << e.what() << std::endl;
        throw; // Re-throw the exception after logging
    }

    // Helper lambda to apply Adam to a single matrix (weight, gradient, m, v)
    // This reduces code duplication for MQ, MK, MV, MH.
    auto apply_adam_to_matrix_pair = [&](mat& weight_mat, mat& grad_mat, mat& m_mat, mat& v_mat) {
        bool unmapped_found = false;

/*
        // --- VERY DETAILED DIAGNOSTIC LOGGING ADDED HERE ---
        // Print the state of each matrix passed to the lambda
        std::cerr << "DEBUG (clAdamUpdate): Checking matrix pair:" << std::endl;
        std::cerr << "  weight_mat: row=" << weight_mat.row << ", col=" << weight_mat.col
                  << ", mapped_data=" << static_cast<void*>(weight_mat.mapped_data)
                  << ", mapped_size=" << weight_mat.mapped_size
                  << ", is_shared=" << weight_mat.is_shared_segment << std::endl;
        if (weight_mat.mapped_data == nullptr) unmapped_found = true;

        std::cerr << "  grad_mat:   row=" << grad_mat.row << ", col=" << grad_mat.col
                  << ", mapped_data=" << static_cast<void*>(grad_mat.mapped_data)
                  << ", mapped_size=" << grad_mat.mapped_size
                  << ", is_shared=" << grad_mat.is_shared_segment << std::endl;
        if (grad_mat.mapped_data == nullptr) unmapped_found = true;

        std::cerr << "  m_mat:      row=" << m_mat.row << ", col=" << m_mat.col
                  << ", mapped_data=" << static_cast<void*>(m_mat.mapped_data)
                  << ", mapped_size=" << m_mat.mapped_size
                  << ", is_shared=" << m_mat.is_shared_segment << std::endl;
        if (m_mat.mapped_data == nullptr) unmapped_found = true;

        std::cerr << "  v_mat:      row=" << v_mat.row << ", col=" << v_mat.col
                  << ", mapped_data=" << static_cast<void*>(v_mat.mapped_data)
                  << ", mapped_size=" << v_mat.mapped_size
                  << ", is_shared=" << v_mat.is_shared_segment << std::endl;
        if (v_mat.mapped_data == nullptr) unmapped_found = true;
*/
        if (unmapped_found) {
            std::cerr << "Warning: Attention Adam update detected unmapped data for a matrix. Skipping this pair." << std::endl;
            return;
        }

        size_t num_elements = static_cast<size_t>(weight_mat.row) * weight_mat.col;
        if (num_elements == 0) { // Handle empty matrices
            std::cerr << "Info: Skipping Adam update for a 0-element matrix pair." << std::endl;
            return;
        }
        size_t bytes = num_elements * sizeof(float);

        // Create OpenCL buffers using cl::Buffer (RAII)
        // CL_CHECK is applied to buffer creation calls which return cl_int
        cl::Buffer d_weights = cl::Buffer(clContext.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, bytes, weight_mat.mapped_data, &cl_err); CL_CHECK(cl_err);
        cl::Buffer d_gradients = cl::Buffer(clContext.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, grad_mat.mapped_data, &cl_err); CL_CHECK(cl_err);
        cl::Buffer d_moments = cl::Buffer(clContext.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, bytes, m_mat.mapped_data, &cl_err); CL_CHECK(cl_err);
        cl::Buffer d_velocity = cl::Buffer(clContext.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, bytes, v_mat.mapped_data, &cl_err); CL_CHECK(cl_err);

        // Set kernel arguments
        // The order must match the kernel signature: weights, gradients, moments, velocity, LR, beta1, beta2, epsilon, t_step, num_elements
        CL_CHECK(adam_kernel.setArg(0, d_weights));
        CL_CHECK(adam_kernel.setArg(1, d_gradients));
        CL_CHECK(adam_kernel.setArg(2, d_moments));
        CL_CHECK(adam_kernel.setArg(3, d_velocity));
        CL_CHECK(adam_kernel.setArg(4, learning_rate));
        CL_CHECK(adam_kernel.setArg(5, beta1));
        CL_CHECK(adam_kernel.setArg(6, beta2));
        CL_CHECK(adam_kernel.setArg(7, epsilon));
        CL_CHECK(adam_kernel.setArg(8, t_adam)); // Use the global 't_adam' passed to this function
        CL_CHECK(adam_kernel.setArg(9, static_cast<cl_int>(num_elements))); // Ensure int cast for kernel arg

        // Enqueue kernel
        CL_CHECK(clContext.queue.enqueueNDRangeKernel(adam_kernel, cl::NullRange, cl::NDRange(num_elements), cl::NullRange));

        // Enqueue read operations (these will be blocking because CL_TRUE is used)
        CL_CHECK(clContext.queue.enqueueReadBuffer(d_weights, CL_TRUE, 0, bytes, weight_mat.mapped_data));
        CL_CHECK(clContext.queue.enqueueReadBuffer(d_moments, CL_TRUE, 0, bytes, m_mat.mapped_data));
        CL_CHECK(clContext.queue.enqueueReadBuffer(d_velocity, CL_TRUE, 0, bytes, v_mat.mapped_data));
    };

    // Apply Adam to attention head's core matrices
/*
    apply_adam_to_matrix_pair(MQ, gMQ, m_MQ, v_MQ);
    apply_adam_to_matrix_pair(MK, gMK, m_MK, v_MK);
    apply_adam_to_matrix_pair(MV, gMV, m_MV, v_MV);
    apply_adam_to_matrix_pair(MH, gMH, m_MH, v_MH);
*/
    // Apply Adam to internal MLPs (recursive call)
    // Each MLP's clAdamUpdate will handle its own 'this->t' increment.
    hor.clAdamUpdate(clContext, t_adam, beta1, beta2, epsilon, learning_rate);
    ver.clAdamUpdate(clContext, t_adam, beta1, beta2, epsilon, learning_rate);

    // Ensure all OpenCL operations enqueued by this function and its recursive calls are complete
    CL_CHECK(clContext.queue.finish());
}

#endif