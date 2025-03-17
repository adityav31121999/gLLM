
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

bool operator==(std::vector<double>, std::vector<double>);
bool operator!=(std::vector<double>, std::vector<double>);
std::vector<double> operator+(std::vector<double>, std::vector<double>);
std::vector<double> operator-(std::vector<double>, std::vector<double>);
std::vector<double> operator*(std::vector<double>, double);
std::vector<double> operator*(double, std::vector<double>);
std::vector<double> operator/(std::vector<double>, double);
std::vector<std::vector<double>> operator+(std::vector<std::vector<double>>, std::vector<std::vector<double>>);
std::vector<std::vector<double>> operator-(std::vector<std::vector<double>>, std::vector<std::vector<double>>);
std::vector<std::vector<double>> operator*(std::vector<std::vector<double>>, double y);
std::vector<std::vector<double>> operator/(std::vector<std::vector<double>>, double y);

double errorofv(std::vector<double> , std::vector<double> );
double gradientdesc1(std::vector<double>, std::vector<double>);
double vdotv2val(std::vector<double>, std::vector<double>);
double vdotv2scal(std::vector<double> , std::vector<double>);
double MSE(std::vector<double>, std::vector<double>);
double sum(std::vector<double>);
double sum(std::vector<std::vector<double>>);
double product(std::vector<double>);
double product(std::vector<std::vector<double>>);

std::vector<double> error(std::vector<double>, std::vector<double>);
std::vector<double> percenterrorofvec(std::vector<double> , std::vector<double>);
std::vector<double> gradient_descent(std::vector<double>, std::vector<double>, double);
std::vector<double> power(std::vector<double>, double);
std::vector<double> sumofrow(std::vector<std::vector<double>>);
std::vector<double> sumofcol(std::vector<std::vector<double>>);
std::vector<double> vxv2v(std::vector<double>, std::vector<double>);
std::vector<double> vdotv2v(std::vector<double>, std::vector<double>);
std::vector<double> vxmat2vec(std::vector<double>, std::vector<std::vector<double>>);
std::vector<double> mat2vec(std::vector<std::vector<double>>);
std::vector<std::vector<double>> kronecker(std::vector<std::vector<double>>, std::vector<std::vector<double>>);
std::vector<std::vector<double>> kronecker(std::vector<std::vector<double>>, std::vector<double>);
std::vector<std::vector<double>> hadamard(std::vector<std::vector<double>>, std::vector<std::vector<double>>);
std::vector<std::vector<double>> vec2mat(std::vector<double>, unsigned int, unsigned int);
std::vector<std::vector<double>> vdotmat2mat(std::vector<double>, std::vector<std::vector<double>>);
std::vector<std::vector<double>> vxv2mat(std::vector<double>, std::vector<double>);
std::vector<std::vector<double>> iproduct(std::vector<std::vector<double>>);
std::vector<std::vector<double>> power(std::vector<std::vector<double>>, double);

// activations.cpp

double sigmoid(double);
std::vector<double> sigmoidv(std::vector<double>);
std::vector<std::vector<double>> sigmoid(std::vector<std::vector<double>>);
std::vector<double> softmax(std::vector<double>, double);
std::vector<std::vector<double>> softmax(std::vector<std::vector<double>>, double);
double ReLU(double);
std::vector<double> ReLUv(std::vector<double>);
double SeLU(double);
std::vector<double> SeLUv(std::vector<double>);
std::vector<double> LOTA(std::vector<double> y);
std::vector<std::vector<double>> LOTA(std::vector<std::vector<double>>);
std::vector<std::vector<double>> LOTA(std::vector<std::vector<double>>, int);

double sigmoidder(double);
std::vector<double> sigmoidvder(std::vector<double>);
std::vector<std::vector<double>> sigmoidder(std::vector<std::vector<double>>);
std::vector<double> softmaxder(std::vector<double>, double);
std::vector<std::vector<double>> softmaxder(std::vector<std::vector<double>>, double);
double ReLUder(double);
std::vector<double> ReLUvder(std::vector<double>);
double SeLUder(double);
std::vector<double> SeLUvder(std::vector<double>);
std::vector<double> LOTAder(std::vector<double> y);
std::vector<std::vector<double>> LOTAder(std::vector<std::vector<double>>);
std::vector<std::vector<double>> LOTAder(std::vector<std::vector<double>>, int);

// weights.cpp

void randomweights(std::vector<std::vector<double>>);
void jumbledwbs(std::vector<std::vector<double>>);
void ijbasedwbs(std::vector<std::vector<double>>);
void Random(std::vector<std::vector<double>>);


#ifdef USE_OPENCL

#include <CL/cl.hpp>

__kernel void operator_eq(__global double* a, __global double* b, __global int* result, int size);
__kernel void operator_add(__global double* a, __global double* b, __global double* result, int size);
__kernel void operator_sub(__global double* a, __global double* b, __global double* result, int size);
__kernel void operator_mul_scalar(__global double* a, double scalar, __global double* result, int size);
__kernel void operator_mul_scalar_reverse(double scalar, __global double* a, __global double* result, int size);
__kernel void operator_div_scalar(__global double* a, double scalar, __global double* result, int size);
__kernel void operator_add_2d(__global double* a, __global double* b, __global double* result, int rows, int cols);
__kernel void operator_sub_2d(__global double* a, __global double* b, __global double* result, int rows, int cols);
__kernel void operator_mul_2d_scalar(__global double* a, double scalar, __global double* result, int rows, int cols);
__kernel void operator_div_2d_scalar(__global double* a, double scalar, __global double* result, int rows, int cols);

__kernel void errorofv(__global double* a, __global double* b, __global double* result, int size);
__kernel void gradientdesc(__global double* a, __global double* b, __global double* result, int size);
__kernel void vdotv2val(__global double* a, __global double* b, __global double* result, int size);
__kernel void vdotv2scal(__global double* a, __global double* b, __global double* result, int size);
__kernel void MSE(__global double* a, __global double* b, __global double* result, int size);
__kernel void sum(__global double* a, __global double* result, int size);
__kernel void sum_2d(__global double* a, __global double* result, int rows, int cols);
__kernel void product(__global double* a, __global double* result, int size);
__kernel void product_2d(__global double* a, __global double* result, int rows, int cols);

__kernel void sigmoidv(__global double* x, __global double* out, int size);
__kernel void sigmoid2D(__global double* x, __global double* out, int rows, int cols);
__kernel void softmax(__global double* x, __global double* out, double temp, int size);
__kernel void softmax2D(__global double* x, __global double* out, double temp, int rows, int cols);
__kernel void ReLUv(__global double* x, __global double* out, int size);
__kernel void SeLUv(__global double* x, __global double* out, int size);
__kernel void LOTA(__global double* y, __global double* out, int size);
__kernel void LOTA2D(__global double* y, __global double* out, int rows, int cols);
__kernel void LOTA2D(__global double* y, __global double* out, int rows, int cols, int limit);
__kernel void LOTA3D(__global double* y, __global double* out, int rows, int cols, int depth);

__kernel void sigmoidvder(__global double* x, __global double* out, int size);
__kernel void sigmoidder2D(__global double* x, __global double* out, int rows, int cols);
__kernel void softmaxder(__global double* x, __global double* out, double temp, int size);
__kernel void softmaxder2D(__global double* x, __global double* out, double temp, int rows, int cols);
__kernel void ReLUvder(__global double* x, __global double* out, int size);
__kernel void SeLUvder(__global double* x, __global double* out, int size);
__kernel void LOTAder(__global double* y, __global double* out, int size);
__kernel void LOTAder2D(__global double* y, __global double* out, int rows, int cols);
__kernel void LOTAder3D(__global double* y, __global double* out, int rows, int cols, int depth);

__kernel void randomweights(__global double* weights, int rows, int cols, unsigned int seed);
__kernel void jumbledwbs(__global double* weights, int rows, int cols, unsigned int seed);
__kernel void ijbasedwbs(__global double* weights, int rows, int cols);
__kernel void Random(__global double* weights, int rows, int cols, unsigned int seed);

#elif USE_CUDA

#include <cuda_runtime.h>

__device__ bool operator_eq(const double* a, const double* b, int size);
__global__ void operator_add(const double* a, const double* b, double* result, int size);
__global__ void operator_sub(const double* a, const double* b, double* result, int size);
__global__ void operator_mul(const double* a, double scalar, double* result, int size);
__global__ void operator_mul_reverse(double scalar, const double* a, double* result, int size);
__global__ void operator_div(const double* a, double scalar, double* result, int size);
__global__ void operator_add_2d(const double* a, const double* b, double* result, int rows, int cols);
__global__ void operator_sub_2d(const double* a, const double* b, double* result, int rows, int cols);
__global__ void operator_mul_2d(const double* a, double scalar, double* result, int rows, int cols);
__global__ void operator_div_2d(const double* a, double scalar, double* result, int rows, int cols);

__global__ void errorofv(const double* a, const double* b, double* result, int size);
__global__ void gradientdesc(const double* a, const double* b, double* result, int size);
__global__ void vdotv2val(const double* a, const double* b, double* result, int size);
__global__ void vdotv2scal(const double* a, const double* b, double* result, int size);
__global__ void MSE(const double* a, const double* b, double* result, int size);
__global__ void sum(const double* a, double* result, int size);
__global__ void sum_2d(const double* a, double* result, int rows, int cols);
__global__ void product(const double* a, double* result, int size);
__global__ void product_2d(const double* a, double* result, int rows, int cols);

__global__ void sigmoidv(double* x, double* out, int size);
__global__ void sigmoid2D(double* x, double* out, int rows, int cols);
__global__ void softmax(double* x, double* out, double temp, int size);
__global__ void softmax2D(double* x, double* out, double temp, int rows, int cols);
__global__ void ReLUv(double* x, double* out, int size);
__global__ void SeLUv(double* x, double* out, int size);
__global__ void LOTA(double* y, double* out, int size);
__global__ void LOTA2D(double* y, double* out, int rows, int cols);
__global__ void LOTA2D(double* y, double* out, int rows, int cols, int limit);
__global__ void LOTA3D(double* y, double* out, int rows, int cols, int depth);

__global__ void sigmoidvder(double* x, double* out, int size);
__global__ void sigmoidder2D(double* x, double* out, int rows, int cols);
__global__ void softmaxder(double* x, double* out, double temp, int size);
__global__ void softmaxder2D(double* x, double* out, double temp, int rows, int cols);
__global__ void ReLUvder(double* x, double* out, int size);
__global__ void SeLUvder(double* x, double* out, int size);
__global__ void LOTAder(double* y, double* out, int size);
__global__ void LOTAder2D(double* y, double* out, int rows, int cols);
__global__ void LOTAder3D(double* y, double* out, int rows, int cols, int depth);

__global__ void randomweights(double* weights, int rows, int cols, unsigned int seed);
__global__ void jumbledwbs(double* weights, int rows, int cols, unsigned int seed);
__global__ void ijbasedwbs(double* weights, int rows, int cols);
__global__ void Random(double* weights, int rows, int cols, unsigned int seed);

#endif

#endif
