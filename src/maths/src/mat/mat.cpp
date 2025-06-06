
#include "include/mat.hpp"
#include <fstream>    // For std::ofstream
#include <string>     // For std::string, std::to_string
#include <stdexcept>  // For std::runtime_error
#include <cstdint>    // For uint64_t
#include <algorithm>  // For std::copy


/**
 * @brief Constructor for matrix of size x*y
 * @param x number of rows
 * @param y number of columns
 */
mat::mat(int x, int y) : row(x), col(y) {
    if (x < 0 || y < 0) { // Dimensions cannot be negative
        throw std::invalid_argument("Matrix dimensions must be positive.");
    }

    if (x == 0 || y == 0) {
        // Handle zero-dimension matrix: no file, no mapping
        mapped_data = nullptr;
        mapped_file_handle = nullptr;
        mapped_size = 0;
        backing_filename.clear(); // No backing file
        is_temp_file = false;     // Not a temporary file
        return; // Done for zero-dimension matrix
    }

    // Create a temporary file for the matrix data
    char temp_name_buffer[L_tmpnam];
    if (!std::tmpnam(temp_name_buffer)) {
            throw std::runtime_error("Failed to generate temporary filename.");
    }
    backing_filename = temp_name_buffer;
    is_temp_file = true;

    size_t required_bytes = static_cast<size_t>(row) * col * sizeof(float);
    if (create_or_resize_file(backing_filename.c_str(), required_bytes) != 0) {
        throw std::runtime_error("Failed to create or resize temporary backing file: " + backing_filename);
    }

    if (open_mapped_file(backing_filename.c_str(), true, &mapped_file_handle, reinterpret_cast<void**>(&mapped_data), &mapped_size) != 0) {
        remove(backing_filename.c_str()); // Clean up temp file on failure
        throw std::runtime_error("Failed to map temporary file: " + backing_filename);
    }

    if (mapped_size < required_bytes) {
            close_mapped_file(mapped_file_handle);
            remove(backing_filename.c_str());
            throw std::runtime_error("Mapped file size is smaller than required.");
    }
}


/**
 * @brief Constructor for square matrix of size x*x
 * @param x number of rows and columns
 */
mat::mat(int x) : mat(x, x) {} // Delegate to the (int, int) constructor

/**
 * @brief Constructor to map an existing file or create a new one.
 * @param filename Path to the file.
 * @param r Rows.
 * @param c Columns.
 * @param create_new If true, create/truncate the file; otherwise, open existing.
 */
mat::mat(const std::string& filename, int r, int c, bool create_new = false)
    : row(r), col(c), backing_filename(filename), is_temp_file(false)
{
    if (r <= 0 || c <= 0) {
        throw std::invalid_argument("Matrix dimensions must be positive.");
    }
    size_t required_bytes = static_cast<size_t>(row) * col * sizeof(float);

    if (create_new) {
        if (create_or_resize_file(backing_filename.c_str(), required_bytes) != 0) {
            throw std::runtime_error("Failed to create or resize backing file: " + backing_filename);
        }
    }

    if (open_mapped_file(backing_filename.c_str(), true, &mapped_file_handle, reinterpret_cast<void**>(&mapped_data), &mapped_size) != 0) {
        throw std::runtime_error("Failed to map file: " + backing_filename);
    }

    if (mapped_size < required_bytes) {
            close_mapped_file(mapped_file_handle);
            throw std::runtime_error("Mapped file size is smaller than required for specified dimensions.");
    }
}

/**
 * @brief Constructor for matrix from a 2D vector
 * @param b 2D vector of floats representing the matrix
 * Creates a temporary memory-mapped file and copies data from the vector.
 */
mat::mat(const std::vector<std::vector<float>>& b) : mat(b.empty() ? 0 : b.size(), b.empty() ? 0 : b[0].size()) {
    if (row == 0 || col == 0) return; // Handle empty input

    // Constructor mat(row, col) already created and mapped a temporary file.
    // Now copy data from the vector to the mapped memory.
    for (int i = 0; i < row; ++i) {
        if (b[i].size() != static_cast<size_t>(col)) {
            // Cleanup before throwing
            close_mapped_file(mapped_file_handle);
            if (is_temp_file) remove(backing_filename.c_str());
            throw std::invalid_argument("Input vector has inconsistent column sizes.");
        }
        // Check potential overflow before calculating offset
        size_t offset = static_cast<size_t>(i) * col;
        if (offset + col > mapped_size / sizeof(float)) {
                close_mapped_file(mapped_file_handle);
                if (is_temp_file) remove(backing_filename.c_str());
                throw std::out_of_range("Vector data exceeds mapped file capacity during copy.");
        }
        std::copy(b[i].begin(), b[i].end(), mapped_data + offset);
    }
}

/**
 * @brief Move constructor for matrix
 * @param b matrix to be moved from
 * @note This constructor is marked as noexcept as it is guaranteed to not throw any exceptions
 */
mat::mat(mat&& b) noexcept {
    // Move the number of rows and columns from the input matrix
    row = b.row;
    col = b.col;
    mapped_data = b.mapped_data;
    mapped_file_handle = b.mapped_file_handle;
    mapped_size = b.mapped_size;
    backing_filename = std::move(b.backing_filename);
    is_temp_file = b.is_temp_file;

    // Leave the moved-from object in a valid state
    b.row = 0;
    b.col = 0;
    b.mapped_data = nullptr;
    b.mapped_file_handle = nullptr;
    b.mapped_size = 0;
    b.is_temp_file = false;
}

/**
 * @brief Copy constructor for matrix.
 * @param b The matrix to be copied from.
 * Creates a deep copy (new temporary file and data copy).
 */
mat::mat(const mat &b) {
    row = b.row;
    col = b.col;
    if (row == 0 || col == 0) return; // Handle empty source

    // Create a new temporary file for the copy
    char temp_name_buffer[L_tmpnam];
    if (!std::tmpnam(temp_name_buffer)) {
            throw std::runtime_error("Failed to generate temporary filename for copy.");
    }
    backing_filename = temp_name_buffer;
    is_temp_file = true;

    size_t required_bytes = static_cast<size_t>(row) * col * sizeof(float);
    if (create_or_resize_file(backing_filename.c_str(), required_bytes) != 0) {
        throw std::runtime_error("Failed to create or resize temporary backing file for copy: " + backing_filename);
    }

    if (open_mapped_file(backing_filename.c_str(), true, &mapped_file_handle, reinterpret_cast<void**>(&mapped_data), &mapped_size) != 0) {
        remove(backing_filename.c_str()); // Clean up temp file on failure
        throw std::runtime_error("Failed to map temporary file for copy: " + backing_filename);
    }

    if (mapped_size < required_bytes) {
            close_mapped_file(mapped_file_handle);
            remove(backing_filename.c_str());
            throw std::runtime_error("Copied mapped file size is smaller than required.");
    }

    // Copy data from source mapped region to destination mapped region
    if (b.mapped_data) {
        memcpy(mapped_data, b.mapped_data, required_bytes);
    } else {
        // Source might be uninitialized or invalid, maybe zero-fill?
        std::fill_n(mapped_data, row * col, 0.0f);
    }
}

// New method to initialize from a shared map segment
void mat::assign_shared_segment(MappedFile* shared_map_handle, float* shared_map_base_ptr,
                            size_t segment_byte_offset_in_shared_map,
                            int new_row, int new_col, const std::string& path_to_shared_file) {
    // Release any previously owned resources if this mat was initialized differently
    if (this->mapped_file_handle && !this->is_shared_segment) {
        close_mapped_file(this->mapped_file_handle);
    }
    if (this->is_temp_file && !this->backing_filename.empty() && !this->is_shared_segment) {
        remove(this->backing_filename.c_str());
    }

    this->row = new_row;
    this->col = new_col;
    this->mapped_file_handle = shared_map_handle;
    this->mapped_data = shared_map_base_ptr;
    this->data_segment_start = (float*)((char*)shared_map_base_ptr + segment_byte_offset_in_shared_map);
    this->mapped_size = static_cast<size_t>(new_row) * new_col * sizeof(float);
    this->backing_filename = path_to_shared_file;
    this->is_temp_file = false;
    this->is_shared_segment = true;
}


// Access operator (non-const)
float& mat::operator()(int i, int j) {
    float* current_data_ptr = is_shared_segment ? data_segment_start : mapped_data;
    if (!current_data_ptr || i < 0 || i >= row || j < 0 || j >= col) {
        throw std::out_of_range("Matrix index out of bounds or matrix not mapped.");
    }
    // Check potential overflow before calculating index
    size_t index = static_cast<size_t>(i) * col + j;
    if (index >= (static_cast<size_t>(row) * col) ) { // Check against logical elements
        throw std::out_of_range("Calculated index exceeds mapped data bounds.");
    }
    return current_data_ptr[index];
}

// Access operator (const)
const float& mat::operator()(int i, int j) const {
    const float* current_data_ptr = is_shared_segment ? data_segment_start : mapped_data;
    if (!current_data_ptr || i < 0 || i >= row || j < 0 || j >= col) {
        throw std::out_of_range("Matrix index out of bounds or matrix not mapped.");
    }
    size_t index = static_cast<size_t>(i) * col + j;
        if (index >= (static_cast<size_t>(row) * col) ) { // Check against logical elements
            throw std::out_of_range("Calculated index exceeds mapped data bounds.");
        }
    return current_data_ptr[index];
}

// return row of matrix from mapped file using 0-based row index
std::vector<float> mat::operator()(int i) const {
    const float* current_data_ptr = is_shared_segment ? data_segment_start : mapped_data;
    if (!current_data_ptr || i < 0 || i >= row) {
        throw std::out_of_range("Row index out of bounds or matrix not mapped.");
    }
    std::vector<float> row_data(col);
    // Check potential overflow before calculating offset
    size_t offset = static_cast<size_t>(i) * col;
    if (offset + col > mapped_size / sizeof(float)) {
            throw std::out_of_range("Row data exceeds mapped file capacity.");
    }
    std::copy(current_data_ptr + offset, current_data_ptr + offset + col, row_data.begin());
    return row_data;
}


/**
 * @brief Retrieves a specific row from the matrix.
 * @param m The matrix.
 * @param row_idx The 0-based index of the row to retrieve.
 * @return A std::vector<float> containing the elements of the specified row.
 * @throws std::out_of_range if row_idx is invalid or if access is out of mapped bounds.
 * @throws std::runtime_error if matrix data is not mapped when it's expected.
 */
std::vector<float> getRow(const mat& m, int row_idx) {
    if (row_idx < 0 || row_idx >= m.row) {
        throw std::out_of_range("getRow: Row index " + std::to_string(row_idx) + " is out of bounds for matrix with " + std::to_string(m.row) + " rows.");
    }

    if (m.col == 0) { // Matrix has columns of zero width
        return {}; // Return an empty vector
    }

    if (!m.mapped_data) {
        // This should ideally only be true if row or col is 0.
        // If row > 0 and col > 0, mapped_data should be valid.
        throw std::runtime_error("getRow: Matrix data is not mapped for a non-empty row.");
    }

    std::vector<float> row_vector(m.col);
    size_t offset = static_cast<size_t>(row_idx) * m.col;

    // Check if the read will go out of bounds of the mapped data
    if (m.mapped_size > 0 && (offset + static_cast<size_t>(m.col)) * sizeof(float) > m.mapped_size) {
        throw std::out_of_range("getRow: Calculated row data exceeds mapped memory bounds.");
    }
    
    const float* start_ptr = m.mapped_data + offset;
    std::copy(start_ptr, start_ptr + m.col, row_vector.begin());
    
    return row_vector;
}

/**
 * @brief Assigns new values to a specific row in the matrix.
 * @param m The matrix to modify.
 * @param row_idx The 0-based index of the row to set.
 * @param data A std::vector<float> containing the new values for the row.
 * @throws std::out_of_range if row_idx is invalid or if access is out of mapped bounds.
 * @throws std::invalid_argument if data size does not match matrix column count.
 * @throws std::runtime_error if matrix data is not mapped when it's expected.
 */
void setRow(mat& m, int row_idx, const std::vector<float>& data) {
    if (row_idx < 0 || row_idx >= m.row) {
        throw std::out_of_range("setRow: Row index " + std::to_string(row_idx) + " is out of bounds for matrix with " + std::to_string(m.row) + " rows.");
    }
    if (static_cast<int>(data.size()) != m.col) {
        throw std::invalid_argument("setRow: Input data size (" + std::to_string(data.size()) + ") does not match matrix column count (" + std::to_string(m.col) + ").");
    }

    if (m.col == 0) { // Matrix has columns of zero width; data must be empty.
        return; // Nothing to set
    }

    if (!m.mapped_data) {
        // This should ideally only be true if row or col is 0.
        // If row > 0 and col > 0, mapped_data should be valid.
        throw std::runtime_error("setRow: Matrix data is not mapped for a non-empty row.");
    }

    size_t offset = static_cast<size_t>(row_idx) * m.col;
    // Check if the write will go out of bounds of the mapped data
    if (m.mapped_size > 0 && (offset + static_cast<size_t>(m.col)) * sizeof(float) > m.mapped_size) {
        throw std::out_of_range("setRow: Calculated row data exceeds mapped memory bounds for writing.");
    }

    float* start_ptr = m.mapped_data + offset;
    std::copy(data.begin(), data.end(), start_ptr);
}

/**
 * @brief write mapped data of mat and appened it to binary file
 * @param matrix matrix to be written
 * @param locationWithFileName location of file with its file name (Location/filename.bin)
 */
void write2filefrommat(const mat& matrix, const std::string& locationWithFileName) {
    //
    // Open the file in binary mode for output and append to it.
    // If the file doesn't exist, it will be created.
    std::ofstream outFile(locationWithFileName, std::ios::binary | std::ios::out | std::ios::app);

    if (!outFile.is_open()) {
        throw std::runtime_error("Failed to open file for appending: " + locationWithFileName);
    }

    // Serialize matrix dimensions (rows, cols)
    // We use uint64_t for dimensions for robustness and consistent sizing.
    uint64_t num_rows = static_cast<uint64_t>(matrix.row);
    uint64_t num_cols = static_cast<uint64_t>(matrix.col);

    // optional when dimension are kept same for all heads and blocks
    // outFile.write(reinterpret_cast<const char*>(&num_rows), sizeof(num_rows));
    // outFile.write(reinterpret_cast<const char*>(&num_cols), sizeof(num_cols));

    if (!outFile) { // Check for errors after writing dimensions
        outFile.close();
        throw std::runtime_error("Error writing matrix dimensions to file: " + locationWithFileName);
    }

    // Serialize matrix data
    size_t num_elements = static_cast<size_t>(num_rows) * static_cast<size_t>(num_cols);
    if (num_elements > 0) {
        const float* data_ptr = matrix.mapped_data;
        if (data_ptr == nullptr && num_elements > 0) { // Should not happen for a valid matrix
            outFile.close();
            throw std::runtime_error("Matrix has non-zero size but data pointer is null. File: " + locationWithFileName);
        }
        size_t data_size_bytes = num_elements * sizeof(float);
        outFile.write(reinterpret_cast<const char*>(data_ptr), data_size_bytes);
        
        if (!outFile) { // Check for errors after writing data
            outFile.close();
            throw std::runtime_error("Error writing matrix data to file: " + locationWithFileName);
        }
    }
    // If num_elements is 0, only dimensions are written, which is fine.

    outFile.close();
    if (!outFile) { // Check for errors that might occur during close (e.g., buffer flush failure)
        throw std::runtime_error("Error occurred while closing file (e.g., flush error): " + locationWithFileName);
    }
}
