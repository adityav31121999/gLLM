#ifdef USE_OPENCL
#include <CL/cl.hpp>
#include "include/mlp.hpp"
/**
 * @brief Helper function to perform gradient clipping on a single OpenCL buffer.
 * @param queue The OpenCL command queue for kernel execution.
 * @param gradient_buffer The cl::Buffer containing the gradients to be clipped.
 * @param clip_norm_value The L2 norm threshold for clipping.
 * @param num_elements The number of float elements in the gradient_buffer.
 * @param local_work_size_1d The local work group size for 1D kernels.
 * @param clcontext The OpenCL context (needed for buffer creation).
 */
void clipGradientBuffer(cl::CommandQueue& queue, cl::Kernel& clip_kernel,
    cl::Kernel& apply_kernel, cl::Buffer& gradient_buffer, float clip_norm_value,
    size_t num_elements, size_t local_work_size_1d, OpenCLContext& clcontext) 
{
    if (num_elements == 0) return; // Nothing to clip

    cl_int cl_err;
    // Calculate the number of workgroups needed for the clipping kernel
    size_t num_workgroups = (num_elements + local_work_size_1d - 1) / local_work_size_1d;
    if (num_workgroups == 0) num_workgroups = 1; // Ensure at least one workgroup for very small buffers

    // Allocate a buffer on the device to store partial sums from each workgroup
    cl::Buffer d_partial_sums(clcontext.context, CL_MEM_READ_WRITE, num_workgroups * sizeof(float), nullptr, &cl_err);
    CL_CHECK(cl_err);
    
    // Determine global work size for the clipping kernel to be a multiple of local work size
    size_t global_work_size_clip = num_workgroups * local_work_size_1d;

    // 1. Calculate sum of squares of gradients
    CL_CHECK(clip_kernel.setArg(0, gradient_buffer));
    CL_CHECK(clip_kernel.setArg(1, 0.0f)); // max_norm is not used in this particular kernel (dummy value)
    CL_CHECK(clip_kernel.setArg(2, (float)num_elements)); // Pass total_elements as float for kernel
    CL_CHECK(clip_kernel.setArg(3, local_work_size_1d * sizeof(float), nullptr)); // Local memory argument size
    CL_CHECK(clip_kernel.setArg(4, d_partial_sums));
    CL_CHECK(queue.enqueueNDRangeKernel(clip_kernel, cl::NullRange, cl::NDRange(global_work_size_clip), cl::NDRange(local_work_size_1d)));

    // 2. Read partial sums back to host and aggregate them
    std::vector<float> h_partial_sums(num_workgroups);
    // Use CL_TRUE for blocking read to ensure data is available before computing norm
    CL_CHECK(queue.enqueueReadBuffer(d_partial_sums, CL_TRUE, 0, num_workgroups * sizeof(float), h_partial_sums.data())); 

    float total_norm_sq = 0.0f;
    for (float s : h_partial_sums) {
        total_norm_sq += s;
    }

    float norm = std::sqrt(total_norm_sq);
    float scale_factor = 1.0f;
    // Apply clipping if clip_norm_value is positive and the norm exceeds it
    if (clip_norm_value > 0 && norm > clip_norm_value) { 
        scale_factor = clip_norm_value / norm;
    }

    // 3. Apply the scaling factor to the gradients on the device
    // The global work size for applying scaling should cover all elements
    size_t global_work_size_apply = ((num_elements + local_work_size_1d - 1) / local_work_size_1d) * local_work_size_1d;
    if (global_work_size_apply == 0 && num_elements > 0) global_work_size_apply = local_work_size_1d; // Handle case for very small elements

    CL_CHECK(apply_kernel.setArg(0, gradient_buffer));
    CL_CHECK(apply_kernel.setArg(1, scale_factor));
    CL_CHECK(apply_kernel.setArg(2, (float)num_elements)); // Pass total_elements as float for kernel
    CL_CHECK(queue.enqueueNDRangeKernel(apply_kernel, cl::NullRange, cl::NDRange(global_work_size_apply), cl::NDRange(local_work_size_1d)));

    if(num_workgroups > 0) {
        d_partial_sums = cl::Buffer();
    }
}

#endif