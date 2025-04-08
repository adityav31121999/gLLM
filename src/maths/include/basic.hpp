
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

float sigmoid(float& x);
float sigmoidder(float& x);
std::vector<float> sigmoid(std::vector<float>& x);
std::vector<float> sigmoidder(std::vector<float> x);
std::vector<std::vector<float>> sigmoid(std::vector<std::vector<float>>& x);
std::vector<std::vector<float>> sigmoidder(std::vector<std::vector<float>>& x);
std::vector<float> softmax(std::vector<float>& x, float& temp);
std::vector<float> softmaxder(std::vector<float>& x, float& temp);
std::vector<std::vector<float>> softmax(std::vector<std::vector<float>>& x, float& temp);
std::vector<std::vector<float>> softmaxder(std::vector<std::vector<float>>& x, float& temp);
float ReLU(float& x);
float ReLUder(float& x);
std::vector<float> ReLU(std::vector<float> x);
std::vector<float> ReLUder(std::vector<float> x);
std::vector<std::vector<float>> ReLU(std::vector<std::vector<float>>& x, int& t);
std::vector<std::vector<float>> ReLUder(std::vector<std::vector<float>>& x, int& t);
std::vector<float> LOTA(std::vector<float>& y);
std::vector<float> LOTAder(std::vector<float>& y);
std::vector<std::vector<float>> LOTA(std::vector<std::vector<float>>& y, int& t, bool& attentionType);
std::vector<std::vector<float>> LOTAder(std::vector<std::vector<float>>& y, int& t, bool& attentionType);

// weights.cpp

void randomweights(std::vector<std::vector<float>>);
void jumbledwbs(std::vector<std::vector<float>>);
void ijbasedwbs(std::vector<std::vector<float>>);
void Random(std::vector<std::vector<float>>);


#ifdef USE_OPENCL

#include <CL/cl.hpp>

// Sigmoid kernel sources
extern const char* sigmoidKernelSource;
extern const char* sigmoidderKernelSource;
extern const char* sigmoid1DKernelSource;
extern const char* sigmoid2DKernelSource;
extern const char* sigmoid1DderKernelSource;
extern const char* sigmoidDer2DKernelSource;

// Softmax kernel sources
extern const char* softmax1DKernelSource;
extern const char* softmax2DKernelSource;
extern const char* softmax1DderKernelSource;
extern const char* softmax2DderKernelSource;

// ReLU kernel sources
extern const char* reluKernelSource;
extern const char* relu1DKernelSource;
extern const char* relu2DKernelSource;
extern const char* reluDerKernelSource;
extern const char* relu1DderKernelSource;
extern const char* relu2DDerKernelSource;

// LOTA kernel sources
extern const char* lota1DKernelSource;
extern const char* lota2DKernelSource;
extern const char* lota2DWithLimitKernelSource;
extern const char* lota1DDerKernelSource;
extern const char* lota2DDerKernelSource;
extern const char* lota2DDerWithLimitKernelSource;

__kernel void operator_eq(__global float* a, __global float* b, __global float* result, int size);
__kernel void operator_add(__global float* a, __global float* b, __global float* result, int size);
__kernel void operator_sub(__global float* a, __global float* b, __global float* result, int size);
__kernel void operator_mul_scalar(__global float* a, float scalar, __global float* result, int size);
__kernel void operator_mul_scalar_reverse(float scalar, __global float* a, __global float* result, int size);
__kernel void operator_div_scalar(__global float* a, float scalar, __global float* result, int size);
__kernel void operator_add_2d(__global float* a, __global float* b, __global float* result, int rows, int cols);
__kernel void operator_sub_2d(__global float* a, __global float* b, __global float* result, int rows, int cols);
__kernel void operator_mul_2d_scalar(__global float* a, float scalar, __global float* result, int rows, int cols);
__kernel void operator_div_2d_scalar(__global float* a, float scalar, __global float* result, int rows, int cols);

__kernel void errorofv(__global float* a, __global float* b, __global float* result, int size);
__kernel void gradientdesc(__global float* a, __global float* b, __global float* result, int size);
__kernel void vdotv2val(__global float* a, __global float* b, __global float* result, int size);
__kernel void vdotv2scal(__global float* a, __global float* b, __global float* result, int size);
__kernel void MSE(__global float* a, __global float* b, __global float* result, int size);
__kernel void sum(__global float* a, __global float* result, int size);
__kernel void sum_2d(__global float* a, __global float* result, int rows, int cols);
__kernel void product(__global float* a, __global float* result, int size);
__kernel void product_2d(__global float* a, __global float* result, int rows, int cols);


#elif USE_CUDA

#include <cuda_runtime.h>

__global__ void cuSigmoid(float x, float* result);
__global__ void cuSigmoid(float* x, float* out, int size);
__global__ void cuSigmoid(float* x, float* out, int rows, int cols);
__global__ void cuSoftmax(float* x, float* out, float temp, int size);
__global__ void cuSoftmax(float* x, float* out, float temp, int rows, int cols);
__global__ void cuReLU(float x, float* result);
__global__ void cuReLU(float* x, float* out, int size);
__global__ void cuSeLU(float x, float* result);
__global__ void cuSeLU(float* x, float* out, int size);
__global__ void cuLOTA(float* y, float* out, int size);
__global__ void cuLOTA(float* y, float* out, int rows, int cols);
__global__ void cuLOTA(float* y, float* out, int rows, int cols, int limit);

__global__ void cuSigmoidder(float x, float* result);
__global__ void cuSigmoidder(float* x, float* out, int rows, int cols);
__global__ void cuSoftmaxder(float* x, float* out, float temp, int size);
__global__ void cuSoftmaxder(float* x, float* out, float temp, int rows, int cols);
__global__ void cuReLUder(float x, float* result);
__global__ void cuReLUder(float* x, float* out, int size);
__global__ void cuSeLUder(float x, float* result);
__global__ void cuSeLUder(float* x, float* out, int size);
__global__ void cuLOTAder(float* y, float* out, int size);
__global__ void cuLOTAder(float* y, float* out, int rows, int cols);
__global__ void cuLOTAder(float* y, float* out, int rows, int cols, int limit);


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

#endif

#endif
