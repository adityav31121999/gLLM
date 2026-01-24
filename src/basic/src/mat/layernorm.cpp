#include "include/mat.hpp"

void layerNorm(mat &input, mat &gamma, mat &beta, mat &normalised)
{
    if (input.row != gamma.row || input.row != beta.row) {
        throw std::runtime_error("LayerNorm: Input, Gamma, and Beta must have the same number of rows.");
    }
    if (input.col != gamma.col || input.col != beta.col) {
        throw std::runtime_error("LayerNorm: Input, Gamma, and Beta must have the same number of rows.");
    }
    normalised.row = input.row;
    normalised.col = input.col;
    float mean = 0.0f;
    float sum = 0.0f;
    for (int i = 0; i < input.row; ++i) sum += std::accumulate(input.mapped_data + i * input.col, input.mapped_data + (i + 1) * input.col, 0.0f);
    mean = sum / (input.row * input.col);

    float var_sum = 0.0f;
    for (int i = 0; i < input.row; ++i) {
        for (int j = 0; j < input.col; ++j) {
            var_sum += (input.mapped_data[i * input.col + j] - mean) * (input.mapped_data[i * input.col + j] - mean);
        }
    }

    // get variance
    float variance = var_sum / (input.row * input.col);
    float epsilon = 1e-5f;
    float inv_std = 1.0f / std::sqrt(variance + epsilon);
    for (int i = 0; i < input.row; ++i) {
        for (int j = 0; j < input.col; ++j) {
            normalised.mapped_data[i * input.col + j] = ((input.mapped_data[i * input.col + j] - mean) * inv_std) * 
                                                          gamma.mapped_data[i] + beta.mapped_data[i];
        }
    }
}