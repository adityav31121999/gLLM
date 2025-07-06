
// matrix.hpp : header source of matrix library
#ifndef MAT_HPP
#define MAT_HPP 1

#include <vector>
#include <string>
#include <stdexcept> 
#include <utility>  
#include <cstdio>
#include <limits>
#include <fstream>
#include <cstring>
#include "include/basic.hpp"
#include <memory_map.h>

// Conceptual helper function - implement this elsewhere (e.g., memory_map.c or utils)
// Ensures file exists and has at least required_size bytes. Returns 0 on success.
int create_or_resize_file(const char* filepath, size_t required_size);

/**
 * @brief CLASS: Matrix class
 * @param r number of rows
 * @param c number of columns
 * Data is stored in a memory-mapped file.
 */
class mat {
public:
    // for other classes to access this class
    int row;
    int col;
    float* mapped_data = nullptr;               // Pointer to the memory-mapped data
    MappedFile* mapped_file_handle = nullptr;   // Handle for mapped file resources
    size_t mapped_size = 0;                     // Size of the mapped region in bytes
    std::string backing_filename;               // Path to the backing file
    bool is_temp_file = false;                  // Flag if the backing file is temporary

    // New members for shared segment functionality
    bool is_shared_segment = false;        // True if this mat is a segment of a larger shared map
    float* data_segment_start = nullptr;   // Actual start of this matrix's data within the shared map

    mat() = default;
    mat(int x, int y);
    mat(int x);
    mat(const std::string& filename, int r, int c, bool create_new);
    mat(const std::vector<std::vector<float>>& b);
    mat(mat&& b) noexcept;
    mat(const mat &b);

    void assign_shared_segment(MappedFile* shared_map_handle, float* shared_map_base_ptr,
                size_t segment_byte_offset_in_shared_map,int new_row, int new_col, const std::string& path_to_shared_file);
    void row2Square(bool dia = 1);
    void col2Square(bool dia = 1);

    mat& operator=(const mat& other); // Needs deep copy logic
    mat& operator=(mat&& other) noexcept; // Needs move logic
    std::vector<float> operator=(int i);    // return ith row
    mat& operator=(const std::vector<std::vector<float>>& b); // Needs file creation/mapping/copy

    float& operator()(int i, int j);
    const float& operator()(int i, int j) const;
    std::vector<float> operator()(int i) const;
    void addRow(const std::vector<float>&, int i);   // add row in ith location of matrix
    std::vector<std::vector<float>> make2dVector(const mat& other, int row, int col);

    mat operator+(const mat& other) const;
    mat operator-(const mat& other) const;
    mat operator*(float scalar) const;
    mat operator*(const mat& other) const; // Matrix multiplication
    mat operator/(float scalar) const;
    mat operator/(const mat& other) const; // Matrix division (A * B^-1)
    mat operator+(const std::vector<std::vector<float>>& b) const; // Add vector
    mat operator-(const std::vector<std::vector<float>>& b) const; // Subtract vector

    mat& operator+=(const mat& other);
    mat& operator-=(const mat& other);
    mat& operator*=(float scalar);
    mat& operator*=(const mat& other);
    mat& operator/=(float scalar);
    mat& operator/=(const mat& other); // Element-wise or matrix inverse? (Currently element-wise)
    mat& operator+=(const std::vector<std::vector<float>>& other); // Add-assign vector
    mat& operator-=(const std::vector<std::vector<float>>& other); // Subtract-assign vector

    mat gaussjordan() const;
    void mult_A_Bt(const mat& a, const mat& b);
    mat mult(const mat& a, const mat& b); // Static or friend? Needs update.
    float trace() const;
    void set(int i, int j, float val);  // set val to (i, j)th element
    bool ifsquare() const;              // check if matrix is square

    bool ifsymmetric() const;
    bool ifidentity() const;
    bool ifdiagonal() const;
    bool ifupper() const;
    bool iflower() const;
    bool ifskew() const;
    mat transpose() const; // Return new transposed matrix
    void transpose_inplace(); // Transpose the current matrix (only if square)
    static mat Random(int row, int col); // Static random matrix generator

    // File-based operations
    void serialise(long long int offset, const std::string& locationWithFileName);
    void deserialise(long long int offset, const std::string& locationWithFileName);

    // Destructor
    ~mat() {
        if (is_shared_segment) {
            mapped_file_handle = nullptr;
            mapped_data = nullptr;
            data_segment_start = nullptr;
        } 
        else {
            // This matrix owns its mapping and potentially its file.
            if (mapped_file_handle) {
                close_mapped_file(mapped_file_handle);
            }
            if (is_temp_file && !backing_filename.empty()) {
                remove(backing_filename.c_str());
            }
        }
        // Reset members to a safe, default state
        row = 0;
        col = 0;
        mapped_size = 0;
        mapped_file_handle = nullptr;
        mapped_data = nullptr;
        data_segment_start = nullptr;
        backing_filename.clear();
        is_temp_file = false;
        is_shared_segment = false;
    }
};

// Helper functions for row operations on mat
std::vector<float> getRow(const mat& m, int row_idx);
void setRow(mat& m, int row_idx, const std::vector<float>& data);
std::vector<float> dot(const mat& a, const std::vector<float>& b);
std::vector<float> dot(const std::vector<float>& a, const mat& b);
void swap(mat& first, mat& second) noexcept;
void write2filefrommat(const mat& matrix, const std::string& locationWithFileName);

// Activation function adapted for mat
mat LOTA(const mat& y, int t, bool attentionType);
mat LOTAder(const mat& y, int t, bool attentionType);


#ifdef USE_CUDA
// device
    __device__ void swap_rows_gj(float* matA, float* matB, int row1, int row2, int n);      // row swap in matrix
// global
    __global__ void trace(float*, float*, int);
    __global__ void inva(float*, float*, int, int);                         // additive inverse of matrix
    __global__ void gaussjordan(float*, float*, int, int);                  // inverse of matrix using gauss jordan elimination method
    __global__ void dot(float*, float*, float*, int, int, int);             // product (dot) of vector and matrix
    __global__ void dot(float*, float*, float*, float*, int, int, int, int);    // product (dot) of vector, matrix and vector
    __global__ void strassen2x2(float*, float*, float*);                    // kernel for strassen algorithm for 2x2 matrix
    __global__ void mult(float*, float*, float*, int, int, int, int);       // multiplication of two matrices
// host backed
    std::vector<float> host_additive_inverse(const std::vector<float>& h_matrix, int rows, int cols);
    mat host_additive_inverse(const mat& input_mat);
    mat host_inverse_gauss_jordan(const mat& input_mat);
    mat host_dot(const std::vector<float>& h_vec, const mat& input_mat); // Matrix-Vector
    float host_dot(const std::vector<float>& h_vec1, const mat& input_mat, const std::vector<float>& h_vec2); // Vector-Matrix-Vector
#endif

std::vector<float> flatten(const std::vector<std::vector<float>>& vec2d);
std::vector<float> flatten(const mat&);
void unflatten(const std::vector<float>& flat, std::vector<std::vector<float>>& vec2d, size_t rows, size_t cols);
void flatten2DVector(const std::vector<std::vector<float>>& vec2d, std::vector<float>& output_flat, size_t expected_rows, size_t expected_cols);
void transposeMatToFlatVector(const mat& m, std::vector<float>& output_flat);
void transposeFlattenMatrix(const std::vector<std::vector<float>>& input, std::vector<float>& output_flat, int rows, int cols);

#endif
