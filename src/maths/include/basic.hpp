
// basic.hpp: header source of basic library
#ifndef BASIC_HPP
#define BASIC_HPP 1

#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <functional>
#include <atomic>
#include <random>

// vect.cpp

bool operator==(std::vector<float>, std::vector<float>);
bool operator!=(std::vector<float>, std::vector<float>);
std::vector<float> operator+(std::vector<float>, std::vector<float>);
std::vector<float> operator-(std::vector<float>, std::vector<float>);
std::vector<float> operator*(std::vector<float>, float);
std::vector<float> operator*(float, std::vector<float>);
std::vector<float> operator/(std::vector<float>, float);
std::vector<float> operator+=(std::vector<float>, std::vector<float>);
std::vector<float> operator-=(std::vector<float>, std::vector<float>);
std::vector<float> operator*=(std::vector<float>, float);
std::vector<float> operator*=(float, std::vector<float>);
std::vector<float> operator/=(std::vector<float>, float);
std::vector<std::vector<float>> operator+(std::vector<std::vector<float>>, std::vector<std::vector<float>>);
std::vector<std::vector<float>> operator-(std::vector<std::vector<float>>, std::vector<std::vector<float>>);
std::vector<std::vector<float>> operator*(std::vector<std::vector<float>>, float y);
std::vector<std::vector<float>> operator/(std::vector<std::vector<float>>, float y);
std::vector<std::vector<float>> operator+=(std::vector<std::vector<float>>, std::vector<std::vector<float>>);
std::vector<std::vector<float>> operator-=(std::vector<std::vector<float>>, std::vector<std::vector<float>>);
std::vector<std::vector<float>> operator*=(std::vector<std::vector<float>>, float y);
std::vector<std::vector<float>> operator/=(std::vector<std::vector<float>>, float y);

float errorofv(std::vector<float>&, std::vector<float>&);
float gradientdesc1(std::vector<float>, std::vector<float>);
float vdotv2val(std::vector<float>, std::vector<float>);
float vdotv2scal(std::vector<float> , std::vector<float>);
float MSE(std::vector<float>, std::vector<float>);
float sum(std::vector<float>);
float sum(std::vector<std::vector<float>>);
float product(std::vector<float>);
float product(std::vector<std::vector<float>>);

std::vector<float> error(std::vector<float>, std::vector<float>);
std::vector<float> percenterrorofvec(std::vector<float> , std::vector<float>);
std::vector<float> gradient_descent(std::vector<float>, std::vector<float>, float);
std::vector<float> power(std::vector<float>, float);
std::vector<float> sumofrow(std::vector<std::vector<float>>);
std::vector<float> sumofcol(std::vector<std::vector<float>>);
std::vector<float> vxv2v(std::vector<float>, std::vector<float>);
std::vector<float> vdotv2v(std::vector<float>, std::vector<float>);
std::vector<float> vxmat2vec(std::vector<float>, std::vector<std::vector<float>>);
std::vector<float> mat2vec(std::vector<std::vector<float>>);
std::vector<std::vector<float>> vec2mat(std::vector<float>, unsigned int, unsigned int);
std::vector<std::vector<float>> vdotmat2mat(std::vector<float>, std::vector<std::vector<float>>);
std::vector<std::vector<float>> vxv2mat(std::vector<float>, std::vector<float>);
std::vector<std::vector<float>> iproduct(std::vector<std::vector<float>>);
std::vector<std::vector<float>> power(std::vector<std::vector<float>>, float);

// activations.cpp

float sigmoid(const float& x);
float sigmoidder(const float& x);
std::vector<float> sigmoid(const std::vector<float>& x);
std::vector<float> sigmoidder(const std::vector<float>& x);
std::vector<std::vector<float>> sigmoid(const std::vector<std::vector<float>>& x);
std::vector<std::vector<float>> sigmoidder(const std::vector<std::vector<float>>& x);
std::vector<float> softmax(const std::vector<float>& x, float& temp);
std::vector<float> softmaxder(const std::vector<float>& x, float& temp);
std::vector<std::vector<float>> softmax(const std::vector<std::vector<float>>& x, float& temp);
std::vector<std::vector<float>> softmaxder(const std::vector<std::vector<float>>& x, float& temp);
float ReLU(const float& x);
float ReLUder(const float& x);
std::vector<float> ReLU(const std::vector<float>& x);
std::vector<float> ReLUder(const std::vector<float>& x);
std::vector<std::vector<float>> ReLU(const std::vector<std::vector<float>>& x, int& t);
std::vector<std::vector<float>> ReLUder(const std::vector<std::vector<float>>& x, int& t);
std::vector<float> LOTA(const std::vector<float>& y);
std::vector<float> LOTAder(const std::vector<float>& y);
std::vector<std::vector<float>> LOTA(const std::vector<std::vector<float>>& y, int& t, bool& attentionType);
std::vector<std::vector<float>> LOTAder(const std::vector<std::vector<float>>& y, int& t, bool& attentionType);

// weights.cpp

void randomweights(std::vector<std::vector<float>>);
void jumbledwbs(std::vector<std::vector<float>>);
void ijbasedwbs(std::vector<std::vector<float>>);
void Random(std::vector<std::vector<float>>);


#ifdef USE_CUDA

#include <cuda_runtime.h>

__global__ void cuSigmoid(float x, float* result);
__global__ void cuSigmoid(float* x, float* out, int size);
__global__ void cuSigmoid(float* x, float* out, int rows, int cols);
__global__ void cuSoftmax(const float* __restrict__ x, float* __restrict__ out, float temp, int size);
__global__ void cuSoftmax(const float* __restrict__ x, float* __restrict__ out, float temp, int rows, int cols);
__global__ void cuReLU(float x, float* result);
__global__ void cuReLU(float* x, float* out, int size);
__global__ void cuLOTA(float* y, float* out, int size);
__global__ void cuLOTA(float* y, float* out, int rows, int cols);
__global__ void cuLOTA(float* y, float* out, int rows, int cols, bool attentionType);

__global__ void cuSigmoidder(float x, float* result);
__global__ void cuSigmoidder(float* x, float* out, int rows, int cols);
__global__ void cuSoftmaxder(float* x, float* out, float temp, int size);
__global__ void cuSoftmaxder(float* x, float* out, float temp, int rows, int cols);
__global__ void cuReLUder(float x, float* result);
__global__ void cuReLUder(float* x, float* out, int size);
__global__ void cuLOTAder(float* y, float* out, int size);
__global__ void cuLOTAder(float* y, float* out, int rows, int cols);
__global__ void cuLOTAder(float* y, float* out, int rows, int cols, bool attentionType);


__global__ void operator_add(const float* a, const float* b, float* result, int size);
__global__ void operator_sub(const float* a, const float* b, float* result, int size);
__global__ void operator_mul(const float* a, float scalar, float* result, int size);
__global__ void operator_mul_reverse(float scalar, const float* a, float* result, int size);
__global__ void operator_div(const float* a, float scalar, float* result, int size);
__global__ void operator_add_2d(const float* a, const float* b, float* result, int rows, int cols);
__global__ void operator_sub_2d(const float* a, const float* b, float* result, int rows, int cols);
__global__ void operator_mul_2d(const float* a, float scalar, float* result, int rows, int cols);
__global__ void operator_div_2d(const float* a, float scalar, float* result, int rows, int cols);

__global__ void errorofv(const float* a, const float* b, float* result, int size);
__global__ void gradientdesc(const float* a, const float* b, float* result, int size);
__global__ void vdotv2val(const float* a, const float* b, float* result, int size);
__global__ void vdotv2scal(const float* a, const float* b, float* result, int size);
__global__ void MSE(const float* a, const float* b, float* result, int size);
__global__ void sum(const float* a, float* result, int size);
__global__ void sum_2d(const float* a, float* result, int rows, int cols);
__global__ void product(const float* a, float* result, int size);
__global__ void product_2d(const float* a, float* result, int rows, int cols);

__device__ float compute_dot_product(const float* vec1, const float* vec2, int dim);
__global__ void matrixMultiplyKernel(const float* A, const float* B, float* C, int rowsA, int colsA, int colsB);
__global__ void vectorAddKernel(const float* A, const float* B, float* C, int len);

#endif

#endif
