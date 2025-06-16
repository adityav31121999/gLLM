#ifdef USE_OPENCL

#include "include/mlp.hpp" // This now includes basic.hpp where OpenCLContext is defined
#include <maths.hpp>       // Includes basic utilities like flatten
#include <vector>
#include <stdexcept>
#include <iostream>
#include <string>

/**
 * @brief Performs forward propagation using OpenCL kernels.
 *        Calculates weighted sums and applies sigmoid activation for each layer.
 * @param in_param (DEPRECATED/IGNORED) Kept for signature compatibility if strictly needed, but logic now uses class members.
 * @param layers_param (DEPRECATED/IGNORED) Kept for signature compatibility if strictly needed, but logic now uses class members.
 */
void mlp::clForward(int in_param, int layers_param) {
    // Parameters in_param and layers_param are largely ignored in favor of class members
    // for a more general MLP implementation.

    if (this->input.empty() || this->weights.empty() || this->num_layers < 2) {
        throw std::runtime_error("MLP not properly initialized for clForward.");
    }
    if (this->weights.size() != (this->num_layers - 1)) {
         throw std::runtime_error("Mismatch between num_layers and weights size in clForward.");
    }
    
    hlayers.resize(layers_param, std::vector<float>(in_param, 0.0f));
    activations.resize(layers_param, std::vector<float>(in_param, 0.0f));
    output.resize(in_param, 0.0f); // Resize host output vector

    try {
        cl_int err; // For OpenCL error codes
        OpenCLContext& context_obj = this->clContext;

        cl::Kernel layerForwardKernel = context_obj.kernels.at("kernelLayerForward"); // From kernel.cl
        cl::Kernel sigmoidKernel = context_obj.kernels.at("clSigmoid1d"); // From activations.cl

        // Ensure activations[0] is set from input
        if (this->activations[0].size() != this->layer_sizes[0]) {
            this->activations[0].resize(this->layer_sizes[0]);
        }
        if (this->input.size() == this->layer_sizes[0]) {
            this->activations[0] = this->input;
        } else {
            throw std::runtime_error("MLP clForward: Input vector size mismatch with input layer size.");
        }

        // Determine max layer size for buffer reuse if desired, or size dynamically
        // For simplicity here, we'll size dynamically, but for performance, pre-allocate larger buffers.
        cl::Buffer d_prev_layer_activations; // Will hold activations from the previous layer

        // Initial input
        size_t input_layer_size_bytes = this->layer_sizes[0] * sizeof(float);
        d_prev_layer_activations = cl::Buffer(context_obj.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                              input_layer_size_bytes, this->activations[0].data(), &err);
        CL_CHECK(err);

        // Iterate through weight matrices (num_layers - 1 of them)
        for (unsigned int l_idx = 0; l_idx < this->num_layers - 1; ++l_idx) {
            const mat& current_weights_mat = this->weights[l_idx];
            unsigned int current_input_neuron_count = this->layer_sizes[l_idx];
            unsigned int current_output_neuron_count = this->layer_sizes[l_idx+1];

            if (current_weights_mat.row != static_cast<int>(current_output_neuron_count) ||
                current_weights_mat.col != static_cast<int>(current_input_neuron_count)) {
                throw std::runtime_error("MLP clForward: Weight mat dimensions mismatch for weights[" + std::to_string(l_idx) + "]. Expected " +
                                          std::to_string(current_output_neuron_count) + "x" + std::to_string(current_input_neuron_count) +
                                          ", got " + std::to_string(current_weights_mat.row) + "x" + std::to_string(current_weights_mat.col));
            }

            size_t weights_bytes = static_cast<size_t>(current_weights_mat.row) * current_weights_mat.col * sizeof(float);
            size_t output_hlayer_bytes = current_output_neuron_count * sizeof(float);
            size_t output_activation_bytes = current_output_neuron_count * sizeof(float);

            cl::Buffer d_weights(context_obj.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, weights_bytes, current_weights_mat.mapped_data, &err);
            CL_CHECK(err);

            cl::Buffer d_current_hlayer_output(context_obj.context, CL_MEM_READ_WRITE, output_hlayer_bytes, nullptr, &err);
            CL_CHECK(err);
            cl::Buffer d_current_activation_output(context_obj.context, CL_MEM_READ_WRITE, output_activation_bytes, nullptr, &err);
            CL_CHECK(err);

            // 1. Compute Weighted Sum (Layer Forward Kernel)
            //    kernelLayerForward(inputs, weights, outputs, input_size, output_size)
            CL_CHECK(layerForwardKernel.setArg(0, d_prev_layer_activations)); // Input activations
            CL_CHECK(layerForwardKernel.setArg(1, d_weights));            // Weights for this layer
            CL_CHECK(layerForwardKernel.setArg(2, d_current_hlayer_output));  // Output buffer for weighted sum
            CL_CHECK(layerForwardKernel.setArg(3, static_cast<cl_int>(current_input_neuron_count)));
            CL_CHECK(layerForwardKernel.setArg(4, static_cast<cl_int>(current_output_neuron_count)));

            cl::NDRange global_fwd(current_output_neuron_count);
            cl::NDRange local_fwd = cl::NullRange;
            CL_CHECK(context_obj.queue.enqueueNDRangeKernel(layerForwardKernel, cl::NullRange, global_fwd, local_fwd));

            // 2. Apply Activation Function (Sigmoid Kernel)
            //    clSigmoid1d(input_buffer, output_buffer, size)
            CL_CHECK(sigmoidKernel.setArg(0, d_current_hlayer_output));   // Input buffer (weighted sum)
            CL_CHECK(sigmoidKernel.setArg(1, d_current_activation_output)); // Output buffer for activations
            CL_CHECK(sigmoidKernel.setArg(2, static_cast<cl_int>(current_output_neuron_count)));

            cl::NDRange global_act(current_output_neuron_count);
            cl::NDRange local_act = cl::NullRange;
            CL_CHECK(context_obj.queue.enqueueNDRangeKernel(sigmoidKernel, cl::NullRange, global_act, local_act));

            // Update d_prev_layer_activations for the next iteration
            d_prev_layer_activations = d_current_activation_output;

            // Read back intermediate results to host hlayers and activations
            // Ensure host vectors are correctly sized by constructor
            if (this->hlayers[l_idx].size() != current_output_neuron_count) {
                this->hlayers[l_idx].resize(current_output_neuron_count);
            }
            if (this->activations[l_idx+1].size() != current_output_neuron_count) {
                this->activations[l_idx+1].resize(current_output_neuron_count);
            }
            CL_CHECK(context_obj.queue.enqueueReadBuffer(d_current_hlayer_output, CL_TRUE, 0, output_hlayer_bytes, this->hlayers[l_idx].data()));
            CL_CHECK(context_obj.queue.enqueueReadBuffer(d_current_activation_output, CL_TRUE, 0, output_activation_bytes, this->activations[l_idx+1].data()));
        }

        // The final output is in activations[num_layers - 1]
        if (this->output.size() != this->layer_sizes.back()) {
            this->output.resize(this->layer_sizes.back());
        }
        this->output = this->activations[this->num_layers - 1];
    }
     catch (const std::out_of_range& oor) {
        // Specific catch for kernel lookup failure from the map
        std::cerr << "Error: Kernel not found in the shared OpenCLContext kernel map. "
                  << "Ensure kernels 'kernelLayerForward' and 'clSigmoid1d' were provided during OpenCLContext initialization. "
                  << "Details: " << oor.what() << std::endl;
        throw; // Re-throw exception
    }
    catch (const std::exception& e) {
        std::cerr << "Standard Exception in mlp::clForward: " << e.what() << std::endl;
        throw; // Re-throw exception
    }
}

#endif // USE_OPENCL
