#include "include/mat.hpp"
#include <fstream>
#include <string>
#include <stdexcept>
#include <cstdint>
#include <algorithm>
#include <cstring>
#include <chrono>

#ifdef _WIN64
#include <windows.h> 
#else // POSIX
#include <unistd.h>   // For mkstemp, ftruncate, close
#include <cerrno>     // For errno
#endif

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

    size_t required_bytes = static_cast<size_t>(row) * col * sizeof(float);

#ifdef _WIN32
    char temp_path[MAX_PATH];
    if (GetTempPathA(MAX_PATH, temp_path) == 0) {
        throw std::runtime_error("Failed to get temporary path: Error " + std::to_string(GetLastError()));
    }

    char temp_filename_win[MAX_PATH];
    if (GetTempFileNameA(temp_path, "gLLM_", 0, temp_filename_win) == 0) {
        throw std::runtime_error("Failed to create temporary file name: Error " + std::to_string(GetLastError()));
    }
    backing_filename = temp_filename_win;
    is_temp_file = true;

    HANDLE hFile = CreateFileA(
        backing_filename.c_str(),
        GENERIC_READ | GENERIC_WRITE,
        0, NULL, CREATE_ALWAYS,
        FILE_ATTRIBUTE_TEMPORARY, // Removed FILE_FLAG_DELETE_ON_CLOSE
        NULL
    );

    if (hFile == INVALID_HANDLE_VALUE) {
        DeleteFileA(backing_filename.c_str()); // Clean up file created by GetTempFileNameA
        throw std::runtime_error("Failed to create temporary file '" + backing_filename + "': Error " + std::to_string(GetLastError()));
    }

    LARGE_INTEGER li;
    li.QuadPart = required_bytes;
    if (!SetFilePointerEx(hFile, li, NULL, FILE_BEGIN)) {
        CloseHandle(hFile); DeleteFileA(backing_filename.c_str());
        throw std::runtime_error("Failed to set file pointer for resizing '" + backing_filename + "': Error " + std::to_string(GetLastError()));
    }
    if (!SetEndOfFile(hFile)) {
        CloseHandle(hFile); DeleteFileA(backing_filename.c_str());
        throw std::runtime_error("Failed to resize temporary file '" + backing_filename + "' with SetEndOfFile: Error " + std::to_string(GetLastError()));
    }
    CloseHandle(hFile);
#else // POSIX
    char temp_name_tpl[] = "/tmp/gLLM_mat_XXXXXX"; // Template for mkstemp
    int fd = mkstemp(temp_name_tpl);
    if (fd == -1) {
        throw std::runtime_error("Failed to generate temporary filename with mkstemp: " + std::string(strerror(errno)));
    }
    backing_filename = temp_name_tpl;
    is_temp_file = true;

    // ftruncate to 0 bytes is well-defined and necessary.
    // The check for required_bytes > 0 is more for the Windows SetEndOfFile issue.
    // Here, we just perform the truncate.
    if (ftruncate(fd, required_bytes) == -1) { 
        close(fd);
        remove(backing_filename.c_str());
        throw std::runtime_error("Failed to resize temporary file '" + backing_filename + "' with ftruncate: " + std::string(strerror(errno)));
    }
    close(fd);
#endif

    if (open_mapped_file(backing_filename.c_str(), true, &mapped_file_handle, reinterpret_cast<void**>(&mapped_data), &mapped_size) != 0) {
#ifndef _WIN32 // On Windows, FILE_FLAG_DELETE_ON_CLOSE should handle this if CreateFileA was successful
        remove(backing_filename.c_str()); // Clean up temp file on failure
#endif
        throw std::runtime_error("Failed to map temporary file: " + backing_filename);
    }

    if (mapped_size < required_bytes) {
        close_mapped_file(mapped_file_handle);
#ifndef _WIN32
        remove(backing_filename.c_str()); // For POSIX, explicitly remove
#endif
        throw std::runtime_error("Mapped file size is smaller than required.");
    }
}


/**
 * @brief Constructor for square matrix of size x*x
 * @param x number of rows and columns
 */
mat::mat(int x) : mat(x, x) {} // Delegate to the (int, int) constructor


/**
 * @brief create the matrix with user defined file name
 * @param filename user defined file name
 * @param r rows of matrix to be mapped
 * @param c columns of matrix to be mapped
 */
mat::mat(const std::string &filename, int r, int c)
    : row(r), col(c), backing_filename(filename)
{
    if (r <= 0 || c <= 0) {
        throw std::invalid_argument("Matrix dimensions must be positive.");
    }
    size_t required_bytes = static_cast<size_t>(row) * col * sizeof(float);

    // Per the comments, create or resize the file to match the specified dimensions.
    if (create_or_resize_file(backing_filename.c_str(), required_bytes) != 0) {
        throw std::runtime_error("Failed to create or resize backing file: " + backing_filename);
    }

    // Map the newly created/resized file into memory.
    if (open_mapped_file(backing_filename.c_str(), true, &mapped_file_handle, reinterpret_cast<void**>(&mapped_data), &mapped_size) != 0) {
        throw std::runtime_error("Failed to map file: " + backing_filename);
    }

    if (mapped_size < required_bytes) {
        close_mapped_file(mapped_file_handle);
        throw std::runtime_error("Mapped file size is smaller than required for specified dimensions.");
    }
    // std::cout << "Matrix with file " << filename << " created." << std::endl;
}


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
            if (is_temp_file) {
#ifndef _WIN32 // On Windows, FILE_FLAG_DELETE_ON_CLOSE handles deletion
                remove(backing_filename.c_str());
#endif
            }
            throw std::invalid_argument("Input vector has inconsistent column sizes.");
        }
        // Check potential overflow before calculating offset
        size_t offset = static_cast<size_t>(i) * col;
        if (offset + col > mapped_size / sizeof(float)) {
                close_mapped_file(mapped_file_handle);
                if (is_temp_file) remove(backing_filename.c_str());
                if (is_temp_file) {
#ifndef _WIN32 // On Windows, FILE_FLAG_DELETE_ON_CLOSE handles deletion
                    remove(backing_filename.c_str());
#endif
                }
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
    size_t required_bytes = static_cast<size_t>(row) * col * sizeof(float);

#ifdef _WIN32
    char temp_path_copy[MAX_PATH];
    if (GetTempPathA(MAX_PATH, temp_path_copy) == 0) {
        throw std::runtime_error("Failed to get temporary path for copy: Error " + std::to_string(GetLastError()));
    }
    char temp_filename_copy_win[MAX_PATH];
    if (GetTempFileNameA(temp_path_copy, "gLLMc", 0, temp_filename_copy_win) == 0) {
        throw std::runtime_error("Failed to create temporary file name for copy: Error " + std::to_string(GetLastError()));
    }
    backing_filename = temp_filename_copy_win;
    is_temp_file = true;

    HANDLE hFileCopy = CreateFileA(
        backing_filename.c_str(),
        GENERIC_READ | GENERIC_WRITE,
        0, NULL, CREATE_ALWAYS,
        FILE_ATTRIBUTE_TEMPORARY, // Removed FILE_FLAG_DELETE_ON_CLOSE
        NULL
    );

    if (hFileCopy == INVALID_HANDLE_VALUE) {
        DeleteFileA(backing_filename.c_str());
        throw std::runtime_error("Failed to create temporary file for copy '" + backing_filename + "': Error " + std::to_string(GetLastError()));
    }

    // Only resize the file if it's non-zero. Creating a zero-byte file and then trying to
    // set its size to zero can cause "Error 87: Invalid Parameter" on some Windows systems.
    if (required_bytes > 0) {
        LARGE_INTEGER li_copy;
        li_copy.QuadPart = required_bytes;
        if (!SetFilePointerEx(hFileCopy, li_copy, NULL, FILE_BEGIN)) {
            CloseHandle(hFileCopy); DeleteFileA(backing_filename.c_str());
            throw std::runtime_error("Failed to set file pointer for resizing copy '" + backing_filename + "': Error " + std::to_string(GetLastError()));
        }
        if (!SetEndOfFile(hFileCopy)) {
            CloseHandle(hFileCopy); DeleteFileA(backing_filename.c_str());
            throw std::runtime_error("Failed to resize temporary file for copy '" + backing_filename + "' with SetEndOfFile: Error " + std::to_string(GetLastError()));
        }
    }
    CloseHandle(hFileCopy); // Always close the handle outside the conditional block.

#else // POSIX
    char temp_name_copy_tpl[] = "/tmp/gLLM_mat_copy_XXXXXX";
    int fd_copy = mkstemp(temp_name_copy_tpl);
    if (fd_copy == -1) {
        throw std::runtime_error("Failed to generate temporary filename for copy with mkstemp: " + std::string(strerror(errno)));
    }
    backing_filename = temp_name_copy_tpl;
    is_temp_file = true;

    if (ftruncate(fd_copy, required_bytes) == -1) {
        close(fd_copy);
        remove(backing_filename.c_str());
        throw std::runtime_error("Failed to resize temporary file for copy '" + backing_filename + "' with ftruncate: " + std::string(strerror(errno)));
    }
    close(fd_copy);
#endif

    if (open_mapped_file(backing_filename.c_str(), true, &mapped_file_handle, reinterpret_cast<void**>(&mapped_data), &mapped_size) != 0) {
#ifndef _WIN32
        remove(backing_filename.c_str()); // Clean up temp file on failure
#endif
        throw std::runtime_error("Failed to map temporary file for copy: " + backing_filename);
    }

    if (mapped_size < required_bytes) {
        close_mapped_file(mapped_file_handle);
#ifndef _WIN32
        remove(backing_filename.c_str());
#endif
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
    if (mapped_file_handle && !is_shared_segment) {
        close_mapped_file(mapped_file_handle);
    }
    if (is_temp_file && !backing_filename.empty() && !is_shared_segment) {
        remove(backing_filename.c_str());
    }

    row = new_row;
    col = new_col;
    mapped_file_handle = shared_map_handle;
    mapped_data = shared_map_base_ptr;
    data_segment_start = (float*)((char*)shared_map_base_ptr + segment_byte_offset_in_shared_map);
    mapped_size = static_cast<size_t>(new_row) * new_col * sizeof(float);
    backing_filename = path_to_shared_file;
    is_temp_file = false;
    is_shared_segment = true;
}
