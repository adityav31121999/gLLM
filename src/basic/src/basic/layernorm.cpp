#include "cppsup.hpp"
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <stdexcept>

void layerNorm(std::vector<float> &input, std::vector<float>& gamma, std::vector<float>& beta, std::vector<float>& normalised)
{
    if (input.size() != gamma.size() || input.size() != beta.size()) {
        throw std::runtime_error("LayerNorm: Input, Gamma, and Beta must have the same size.");
    }
    size_t size = input.size();
    if (size == 0) return;
    normalised.resize(size);

    // mean
    float sum = std::accumulate(input.begin(), input.end(), 0.0f);
    float mean = sum / size;

    // get variance
    float var_sum = 0.0f;
    for (float val : input) {
        var_sum += (val - mean) * (val - mean);
    }
    float variance = var_sum / size;
    float epsilon = 1e-5f;

    // normalise and apply affine transformation
    float inv_std = 1.0f / std::sqrt(variance + epsilon);
    for (size_t i = 0; i < size; ++i) {
        normalised[i] = ((input[i] - mean) * inv_std) * gamma[i] + beta[i];
    }
}

void layerNorm(std::vector<std::vector<float>>& input, std::vector<std::vector<float>>& gamma, std::vector<std::vector<float>>& beta, 
                std::vector<std::vector<float>>& normalised)
{
    if (input.size() != gamma.size() || input.size() != beta.size()) {
        throw std::runtime_error("LayerNorm 2D: Input, Gamma, and Beta must have the same number of rows.");
    }
    if (input[0].size() != gamma[0].size() || input[0].size() != beta[0].size()) {
        throw std::runtime_error("LayerNorm 2D: Input, Gamma, and Beta must have the same number of rows.");
    }

    normalised.resize(input.size(), std::vector<float>(input[0].size(), 0.0f));
    float mean = 0.0f;
    float sum = 0.0f;
    for (int i = 0; i < input.size(); ++i) sum += std::accumulate(input[i].begin(), input[i].end(), 0.0f);
    mean = sum / (input.size() * input[0].size());

    float var_sum = 0.0f;
    for (int i = 0; i < input.size(); ++i) {
        for (int j = 0; j < input[0].size(); ++j) {
            var_sum += (input[i][j] - mean) * (input[i][j] - mean);
        }
    }

    // get variance
    float variance = var_sum / (input.size() * input[0].size());
    float epsilon = 1e-5f;
    float inv_std = 1.0f / std::sqrt(variance + epsilon);

    for (int i = 0; i < input.size(); ++i) {
        for (int j = 0; j < input[0].size(); ++j) {
            normalised[i][j] = ((input[i][j] - mean) * inv_std) * gamma[i][j] + beta[i][j];
        }
    }
}