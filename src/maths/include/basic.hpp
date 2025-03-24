
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
std::vector<std::vector<float>> operator+(std::vector<std::vector<float>>, std::vector<std::vector<float>>);
std::vector<std::vector<float>> operator-(std::vector<std::vector<float>>, std::vector<std::vector<float>>);
std::vector<std::vector<float>> operator*(std::vector<std::vector<float>>, float y);
std::vector<std::vector<float>> operator/(std::vector<std::vector<float>>, float y);

float errorofv(std::vector<float> , std::vector<float> );
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
std::vector<std::vector<float>> kronecker(std::vector<std::vector<float>>, std::vector<std::vector<float>>);
std::vector<std::vector<float>> kronecker(std::vector<std::vector<float>>, std::vector<float>);
std::vector<std::vector<float>> hadamard(std::vector<std::vector<float>>, std::vector<std::vector<float>>);
std::vector<std::vector<float>> vec2mat(std::vector<float>, unsigned int, unsigned int);
std::vector<std::vector<float>> vdotmat2mat(std::vector<float>, std::vector<std::vector<float>>);
std::vector<std::vector<float>> vxv2mat(std::vector<float>, std::vector<float>);
std::vector<std::vector<float>> iproduct(std::vector<std::vector<float>>);
std::vector<std::vector<float>> power(std::vector<std::vector<float>>, float);

// activations.cpp

float sigmoid(float);
std::vector<float> sigmoidv(std::vector<float>);
std::vector<std::vector<float>> sigmoid(std::vector<std::vector<float>>);
std::vector<float> softmax(std::vector<float>, float);
std::vector<std::vector<float>> softmax(std::vector<std::vector<float>>, float);
float ReLU(float);
std::vector<float> ReLUv(std::vector<float>);
float SeLU(float);
std::vector<float> SeLUv(std::vector<float>);
std::vector<float> LOTA(std::vector<float> y);
std::vector<std::vector<float>> LOTA(std::vector<std::vector<float>>);
std::vector<std::vector<float>> LOTA(std::vector<std::vector<float>>, int);

float sigmoidder(float);
std::vector<float> sigmoidvder(std::vector<float>);
std::vector<std::vector<float>> sigmoidder(std::vector<std::vector<float>>);
std::vector<float> softmaxder(std::vector<float>, float);
std::vector<std::vector<float>> softmaxder(std::vector<std::vector<float>>, float);
float ReLUder(float);
std::vector<float> ReLUvder(std::vector<float>);
float SeLUder(float);
std::vector<float> SeLUvder(std::vector<float>);
std::vector<float> LOTAder(std::vector<float> y);
std::vector<std::vector<float>> LOTAder(std::vector<std::vector<float>>);
std::vector<std::vector<float>> LOTAder(std::vector<std::vector<float>>, int);

// weights.cpp

void randomweights(std::vector<std::vector<float>>);
void jumbledwbs(std::vector<std::vector<float>>);
void ijbasedwbs(std::vector<std::vector<float>>);
void Random(std::vector<std::vector<float>>);


#ifdef USE_OPENCL

#include <CL/cl.hpp>

float cl_sigmoid(float x);
float cl_sigmoidder(float x);

__kernel void operator_eq(__global float* a, __global float* b, __global int* result, int size);
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

__kernel void sigmoidv(__global float* x, __global float* out, int size);
__kernel void sigmoid2D(__global float* x, __global float* out, int rows, int cols);
__kernel void softmax(__global float* x, __global float* out, float temp, int size);
__kernel void softmax2D(__global float* x, __global float* out, float temp, int rows, int cols);
__kernel void ReLUv(__global float* x, __global float* out, int size);
__kernel void SeLUv(__global float* x, __global float* out, int size);
__kernel void LOTA(__global float* y, __global float* out, int size);
__kernel void LOTA2D(__global float* y, __global float* out, int rows, int cols);
__kernel void LOTA2D(__global float* y, __global float* out, int rows, int cols, int limit);
__kernel void LOTA3D(__global float* y, __global float* out, int rows, int cols, int depth);

__kernel void sigmoidvder(__global float* x, __global float* out, int size);
__kernel void sigmoidder2D(__global float* x, __global float* out, int rows, int cols);
__kernel void softmaxder(__global float* x, __global float* out, float temp, int size);
__kernel void softmaxder2D(__global float* x, __global float* out, float temp, int rows, int cols);
__kernel void ReLUvder(__global float* x, __global float* out, int size);
__kernel void SeLUvder(__global float* x, __global float* out, int size);
__kernel void LOTAder(__global float* y, __global float* out, int size);
__kernel void LOTAder2D(__global float* y, __global float* out, int rows, int cols);
__kernel void LOTAder3D(__global float* y, __global float* out, int rows, int cols, int depth);

__kernel void randomweights(__global float* weights, int rows, int cols, unsigned int seed);
__kernel void jumbledwbs(__global float* weights, int rows, int cols, unsigned int seed);
__kernel void ijbasedwbs(__global float* weights, int rows, int cols);
__kernel void Random(__global float* weights, int rows, int cols, unsigned int seed);

#elif USE_CUDA

#include <cuda_runtime.h>

__device__ float cuda_sigmoid(float x);
__device__ float cuda_sigmoidder(float x);
__device__ bool operator_eq(const float* a, const float* b, int size);

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

__global__ void sigmoidv(float* x, float* out, int size);
__global__ void sigmoid2D(float* x, float* out, int rows, int cols);
__global__ void softmax(float* x, float* out, float temp, int size);
__global__ void softmax2D(float* x, float* out, float temp, int rows, int cols);
__global__ void ReLUv(float* x, float* out, int size);
__global__ void SeLUv(float* x, float* out, int size);
__global__ void LOTA(float* y, float* out, int size);
__global__ void LOTA2D(float* y, float* out, int rows, int cols);
__global__ void LOTA2D(float* y, float* out, int rows, int cols, int limit);
__global__ void LOTA3D(float* y, float* out, int rows, int cols, int depth);

__global__ void sigmoidvder(float* x, float* out, int size);
__global__ void sigmoidder2D(float* x, float* out, int rows, int cols);
__global__ void softmaxder(float* x, float* out, float temp, int size);
__global__ void softmaxder2D(float* x, float* out, float temp, int rows, int cols);
__global__ void ReLUvder(float* x, float* out, int size);
__global__ void SeLUvder(float* x, float* out, int size);
__global__ void LOTAder(float* y, float* out, int size);
__global__ void LOTAder2D(float* y, float* out, int rows, int cols);
__global__ void LOTAder3D(float* y, float* out, int rows, int cols, int depth);

__global__ void randomweights(float* weights, int rows, int cols, unsigned int seed);
__global__ void jumbledwbs(float* weights, int rows, int cols, unsigned int seed);
__global__ void ijbasedwbs(float* weights, int rows, int cols);
__global__ void Random(float* weights, int rows, int cols, unsigned int seed);

#endif

#endif
