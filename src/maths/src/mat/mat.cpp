#include "include/mat.hpp"
#include <fstream>    // For std::ofstream
#include <string>     // For std::string, std::to_string
#include <stdexcept>  // For std::runtime_error
#include <cstdint>    // For uint64_t
#include <algorithm>  // For std::copy, std::fill_n
#include <cstring>    // For memcpy, strerror (strerror for POSIX)
#include <chrono>     // For generating unique temporary filenames

// OS-specific includes for temporary file creation
#ifdef _WIN64
#include <windows.h> // For Windows API functions (GetTempPathA, GetTempFileNameA, CreateFileA, SetFilePointerEx, SetEndOfFile, CloseHandle, DeleteFileA, MAX_PATH, GetLastError, FormatMessageA)
#else // POSIX
#include <unistd.h>   // For mkstemp, ftruncate, close
#include <cerrno>     // For errno
#endif

// Helper function to generate a unique temporary filename (moved here for encapsulation)
namespace { // Anonymous namespace for internal linkage
std::string generate_temp_filename() {
    static int counter = 0; // Ensures uniqueness across calls in the same process
#ifdef _WIN64
    char temp_path[MAX_PATH];
    if (GetTempPathA(MAX_PATH, temp_path) == 0) {
        throw std::runtime_error("Failed to get temporary path for mat: " + std::to_string(GetLastError()));
    }
    char temp_filename_win[MAX_PATH];
    // Use an actual suffix like ".bin" and attempt to create a unique file
    if (GetTempFileNameA(temp_path, "gllm", 0, temp_filename_win) == 0) {
         throw std::runtime_error("Failed to create temporary file name for mat: " + std::to_string(GetLastError()));
    }
    std::string filename_str = temp_filename_win;
    // The previous GetTempFileNameA creates the file. We need to close and reopen if CreateFileA with DELETE_ON_CLOSE.
    // However, for mapping, it's often better to just get the name and then explicitly create/resize.
    // This is safer; GetTempFileNameA creates it, we get the name, then close and we can delete if we want to.
    // The current code already handles the deletion if needed in the constructor.
    return filename_str;
#else // POSIX
    std::string temp_name_tpl = "/tmp/gllm_mat_";
    temp_name_tpl += std::to_string(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    temp_name_tpl += "_";
    temp_name_tpl += std::to_string(counter++);
    temp_name_tpl += "XXXXXX"; // Template for mkstemp
    // mkstemp modifies the template string to be the actual unique filename
    // We explicitly call mkstemp where it's used to avoid double handling file descriptor.
    return temp_name_tpl; // This returns the template string to be modified by mkstemp
#endif
}
} // end anonymous namespace

// Assumed implementations for open_mapped_file, close_mapped_file, get_mapped_data, create_or_resize_file
// These are external to mat.cpp and should be defined in memory_map.c or utils.

/**
 * @brief Constructor for matrix of size x*y
 * @param x number of rows
 * @param y number of columns
 */
mat::mat(int x, int y) : row(x), col(y) {
    // Initialize pointers/handles to nullptr and flags to default
    mapped_data = nullptr;
    mapped_file_handle = nullptr;
    mapped_size = 0;
    backing_filename.clear();
    is_temp_file = false;
    is_shared_segment = false;
    data_segment_start = nullptr;

    if (x < 0 || y < 0) {
        throw std::invalid_argument("Matrix dimensions must be non-negative.");
    }

    if (x == 0 || y == 0) {
        // For zero-dimension matrices, all members are already correctly nulled/cleared.
        return;
    }

    size_t required_bytes = static_cast<size_t>(row) * col * sizeof(float);

#ifdef _WIN64
    backing_filename = generate_temp_filename(); // Get a unique name
    is_temp_file = true;

    HANDLE hFile = CreateFileA(
        backing_filename.c_str(),
        GENERIC_READ | GENERIC_WRITE,
        0, NULL, CREATE_ALWAYS,
        FILE_ATTRIBUTE_TEMPORARY, // Removed FILE_FLAG_DELETE_ON_CLOSE to prevent premature deletion
        NULL
    );

    if (hFile == INVALID_HANDLE_VALUE) {
        // No need to DeleteFileA here, as CreateFileA CREATE_ALWAYS would truncate existing
        // but if it fails to create, the file might not exist or be accessible.
        throw std::runtime_error("Failed to create temporary file '" + backing_filename + "': Error " + std::to_string(GetLastError()));
    }

    LARGE_INTEGER li;
    li.QuadPart = required_bytes;
    if (!SetFilePointerEx(hFile, li, NULL, FILE_BEGIN)) {
        CloseHandle(hFile); remove(backing_filename.c_str()); // Clean up temp file
        throw std::runtime_error("Failed to set file pointer for resizing '" + backing_filename + "': Error " + std::to_string(GetLastError()));
    }
    if (!SetEndOfFile(hFile)) {
        CloseHandle(hFile); remove(backing_filename.c_str()); // Clean up temp file
        throw std::runtime_error("Failed to resize temporary file '" + backing_filename + "' with SetEndOfFile: Error " + std::to_string(GetLastError()));
    }
    CloseHandle(hFile); // Close the file handle before mapping.
#else // POSIX
    // Generate unique name and create file atomically using mkstemp
    char temp_name_buffer[256]; // Use a buffer for mkstemp
    std::string temp_name_base = generate_temp_filename(); // Get base template
    strncpy(temp_name_buffer, temp_name_base.c_str(), sizeof(temp_name_buffer) - 1);
    temp_name_buffer[sizeof(temp_name_buffer) - 1] = '\0'; // Ensure null termination

    int fd = mkstemp(temp_name_buffer);
    if (fd == -1) {
        throw std::runtime_error("Failed to generate temporary filename with mkstemp: " + std::string(strerror(errno)));
    }
    backing_filename = temp_name_buffer; // mkstemp modified the buffer to the actual unique filename
    is_temp_file = true;

    if (ftruncate(fd, required_bytes) == -1) {
        close(fd);
        remove(backing_filename.c_str());
        throw std::runtime_error("Failed to resize temporary file '" + backing_filename + "' with ftruncate: " + std::string(strerror(errno)));
    }
    close(fd); // Close the file descriptor before mapping.
#endif

    // Open and map the file
    int map_result = open_mapped_file(backing_filename.c_str(), true, &mapped_file_handle, reinterpret_cast<void**>(&mapped_data), &mapped_size);

    if (map_result != 0) {
        remove(backing_filename.c_str()); // Clean up temp file on mapping failure
        throw std::runtime_error("Failed to map temporary file: " + backing_filename + " (Error code from open_mapped_file: " + std::to_string(map_result) + ")");
    }

    // --- CRITICAL ADDED CHECK HERE: Ensure mapped_data is valid ---
    if (mapped_data == nullptr) {
        // This indicates a severe issue in open_mapped_file logic
        if (mapped_file_handle) {
            close_mapped_file(mapped_file_handle);
            mapped_file_handle = nullptr; // Explicitly clear handle
        }
        remove(backing_filename.c_str()); // Clean up temp file
        throw std::runtime_error("FATAL ERROR: open_mapped_file returned success but mapped_data is nullptr for file: " + backing_filename);
    }
    // --- END CRITICAL ADDED CHECK ---

    // Verify mapped size
    if (mapped_size < required_bytes) {
        close_mapped_file(mapped_file_handle);
        mapped_file_handle = nullptr; // Explicitly clear handle
        remove(backing_filename.c_str()); // For POSIX, explicitly remove
        throw std::runtime_error("Mapped file size is smaller than required.");
    }
    // For non-shared mats, data_segment_start points to the beginning of its own mapping
    data_segment_start = mapped_data;
    is_shared_segment = false; // This matrix owns its mapping
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
    : row(r), col(c), backing_filename(filename), is_temp_file(false),
      mapped_data(nullptr), mapped_file_handle(nullptr), mapped_size(0),
      is_shared_segment(false), data_segment_start(nullptr) // Initialize all members
{
    if (r <= 0 || c <= 0) {
        throw std::invalid_argument("Matrix dimensions must be positive for file-backed matrix.");
    }
    size_t required_bytes = static_cast<size_t>(row) * col * sizeof(float);

    if (create_new) {
        if (create_or_resize_file(backing_filename.c_str(), required_bytes) != 0) {
            throw std::runtime_error("Failed to create or resize backing file: " + backing_filename);
        }
    }

    int map_result = open_mapped_file(backing_filename.c_str(), true, &mapped_file_handle, reinterpret_cast<void**>(&mapped_data), &mapped_size);

    if (map_result != 0) {
        throw std::runtime_error("Failed to map file: " + backing_filename + " (Error code from open_mapped_file: " + std::to_string(map_result) + ")");
    }

    // --- CRITICAL ADDED CHECK HERE: Ensure mapped_data is valid ---
    if (mapped_data == nullptr) {
        if (mapped_file_handle) {
            close_mapped_file(mapped_file_handle);
            mapped_file_handle = nullptr;
        }
        throw std::runtime_error("FATAL ERROR: open_mapped_file returned success but mapped_data is nullptr for file: " + backing_filename);
    }
    // --- END CRITICAL ADDED CHECK ---

    if (mapped_size < required_bytes) {
            close_mapped_file(mapped_file_handle);
            mapped_file_handle = nullptr;
            throw std::runtime_error("Mapped file size is smaller than required for specified dimensions.");
    }
    // For non-shared matrices, data_segment_start points to the beginning of its own mapping
    data_segment_start = mapped_data;
}

/**
 * @brief Constructor for matrix from a 2D vector
 * @param b 2D vector of floats representing the matrix
 * Creates a temporary memory-mapped file and copies data from the vector.
 */
mat::mat(const std::vector<std::vector<float>>& b)
    : mat(b.empty() ? 0 : b.size(), b.empty() ? 0 : b[0].size()) // Delegates to mat(int, int)
{
    // The delegated constructor mat(row, col) has already handled mapping and error checks.
    // If row or col is 0, it initialized to an empty state and returned.
    if (row == 0 || col == 0) {
        return;
    }

    // Now, copy data from the vector to the mapped memory.
    // 'mapped_data' should be valid here due to prior checks in delegated constructor.
    for (int i = 0; i < row; ++i) {
        if (b[i].size() != static_cast<size_t>(col)) {
            // Cleanup the temporary file and mapping before throwing if it's owned
            if (!is_shared_segment && mapped_file_handle) {
                close_mapped_file(mapped_file_handle);
                mapped_file_handle = nullptr;
            }
            if (is_temp_file && !backing_filename.empty()) {
                remove(backing_filename.c_str());
            }
            throw std::invalid_argument("Input vector has inconsistent column sizes.");
        }
        size_t offset = static_cast<size_t>(i) * col;
        // The previous constructor's size check and mapped_data check largely cover this.
        // This check guards against logic errors or unexpected changes to mapped_size.
        if (!mapped_data || (offset + col) * sizeof(float) > mapped_size) {
            // This indicates a severe issue. Cleanup and re-throw.
            if (!is_shared_segment && mapped_file_handle) {
                close_mapped_file(mapped_file_handle);
                mapped_file_handle = nullptr;
            }
            if (is_temp_file && !backing_filename.empty()) {
                remove(backing_filename.c_str());
            }
            throw std::out_of_range("Vector data copy would exceed mapped file capacity or mapped_data is null unexpectedly.");
        }
        std::memcpy(mapped_data + offset, b[i].data(), col * sizeof(float)); // Use memcpy for efficiency
    }
}

/**
 * @brief Move constructor for matrix.
 * Transfers ownership of resources from 'b' to this object.
 * @param b matrix to be moved from.
 */
mat::mat(mat&& b) noexcept
    : row(b.row),
      col(b.col),
      mapped_data(b.mapped_data),
      mapped_file_handle(b.mapped_file_handle),
      mapped_size(b.mapped_size),
      backing_filename(std::move(b.backing_filename)), // Move string
      is_temp_file(b.is_temp_file),
      is_shared_segment(b.is_shared_segment),
      data_segment_start(b.data_segment_start)
{
    // Leave the moved-from object in a valid, empty state
    b.row = 0;
    b.col = 0;
    b.mapped_data = nullptr;
    b.mapped_file_handle = nullptr;
    b.mapped_size = 0;
    b.backing_filename.clear();
    b.is_temp_file = false;
    b.is_shared_segment = false;
    b.data_segment_start = nullptr;
}

/**
 * @brief Copy constructor for matrix.
 * Creates a deep copy (new temporary file and data copy) if the source is not a shared segment.
 * If the source is a shared segment, this copy also becomes a shared segment (shallow pointer copy).
 * @param b The matrix to be copied from.
 */
mat::mat(const mat &b)
    : row(b.row),
      col(b.col),
      mapped_size(0), // Will be set based on ownership type
      mapped_data(nullptr),
      mapped_file_handle(nullptr),
      backing_filename(), // Will be set based on ownership type
      is_temp_file(false), // Will be set based on ownership type
      is_shared_segment(false), // Will be set based on ownership type
      data_segment_start(nullptr) // Will be set based on ownership type
{
    if (b.is_shared_segment) {
        // Case 1: 'b' is a shared segment. This copy also becomes a shared segment.
        is_shared_segment = true;
        mapped_file_handle = b.mapped_file_handle;   // Share the handle to the base file
        mapped_data = b.mapped_data;                 // Share the base mapped data pointer
        data_segment_start = b.data_segment_start;   // Point to the same segment start
        backing_filename = b.backing_filename;       // Informational: path to the shared file
        mapped_size = b.mapped_size;                 // Size of *this specific segment*
        is_temp_file = false;                        // Shared segments are not temporary files owned by 'this'
    } else {
        // Case 2: 'b' owns its file. Perform a deep copy: create a new file and copy data.
        if (b.row == 0 || b.col == 0) {
            // Handle empty source matrix: new object is also empty.
            row = 0;
            col = 0;
            return;
        }

        size_t required_bytes = static_cast<size_t>(b.row) * b.col * sizeof(float);

        // Generate a new temporary filename and create/resize file for the copy
#ifdef _WIN64
        backing_filename = generate_temp_filename(); // Get a unique name
        is_temp_file = true;

        HANDLE hFileCopy = CreateFileA(
            backing_filename.c_str(),
            GENERIC_READ | GENERIC_WRITE,
            0, NULL, CREATE_ALWAYS,
            FILE_ATTRIBUTE_TEMPORARY,
            NULL
        );

        if (hFileCopy == INVALID_HANDLE_VALUE) {
            throw std::runtime_error("Failed to create temporary file for copy '" + backing_filename + "': Error " + std::to_string(GetLastError()));
        }

        LARGE_INTEGER li_copy;
        li_copy.QuadPart = required_bytes;
        if (!SetFilePointerEx(hFileCopy, li_copy, NULL, FILE_BEGIN)) {
            CloseHandle(hFileCopy); remove(backing_filename.c_str());
            throw std::runtime_error("Failed to set file pointer for resizing copy '" + backing_filename + "': Error " + std::to_string(GetLastError()));
        }
        if (!SetEndOfFile(hFileCopy)) {
            CloseHandle(hFileCopy); remove(backing_filename.c_str());
            throw std::runtime_error("Failed to resize temporary file for copy '" + backing_filename + "' with SetEndOfFile: Error " + std::to_string(GetLastError()));
        }
        CloseHandle(hFileCopy);
#else // POSIX
        char temp_name_buffer[256];
        std::string temp_name_base = generate_temp_filename();
        strncpy(temp_name_buffer, temp_name_base.c_str(), sizeof(temp_name_buffer) - 1);
        temp_name_buffer[sizeof(temp_name_buffer) - 1] = '\0';

        int fd_copy = mkstemp(temp_name_buffer);
        if (fd_copy == -1) {
            throw std::runtime_error("Failed to generate temporary filename for copy with mkstemp: " + std::string(strerror(errno)));
        }
        backing_filename = temp_name_buffer;
        is_temp_file = true;

        if (ftruncate(fd_copy, required_bytes) == -1) {
            close(fd_copy);
            remove(backing_filename.c_str());
            throw std::runtime_error("Failed to resize temporary file for copy '" + backing_filename + "' with ftruncate: " + std::string(strerror(errno)));
        }
        close(fd_copy);
#endif
        // Open and map the new temporary file
        int map_result = open_mapped_file(backing_filename.c_str(), true, &mapped_file_handle, reinterpret_cast<void**>(&mapped_data), &mapped_size);

        if (map_result != 0) {
            remove(backing_filename.c_str());
            throw std::runtime_error("Failed to map temporary file for copy: " + backing_filename + " (Error code from open_mapped_file: " + std::to_string(map_result) + ")");
        }

        // --- CRITICAL ADDED CHECK HERE: Ensure mapped_data is valid ---
        if (mapped_data == nullptr) {
            if (mapped_file_handle) {
                close_mapped_file(mapped_file_handle);
                mapped_file_handle = nullptr;
            }
            remove(backing_filename.c_str());
            throw std::runtime_error("FATAL ERROR: open_mapped_file returned success but mapped_data is nullptr during copy for file: " + backing_filename);
        }
        // --- END CRITICAL ADDED CHECK ---

        if (mapped_size < required_bytes) {
            close_mapped_file(mapped_file_handle);
            mapped_file_handle = nullptr;
            remove(backing_filename.c_str());
            throw std::runtime_error("Copied mapped file size is smaller than required.");
        }
        // For non-shared mats, data_segment_start is the same as mapped_data
        data_segment_start = mapped_data;

        // Copy data from source mapped region to destination mapped region
        const float* source_data_ptr = b.is_shared_segment ? b.data_segment_start : b.mapped_data;
        if (source_data_ptr && mapped_data && required_bytes > 0) {
            std::memcpy(mapped_data, source_data_ptr, required_bytes);
        } else if (required_bytes > 0) {
            // If source data pointer is null for a non-empty matrix,
            // fill with zeros to avoid garbage.
            std::fill_n(mapped_data, row * col, 0.0f);
        }
    }
}

/**
 * @brief Move assignment operator for matrix.
 * Releases current resources, transfers ownership from 'other', and leaves 'other' empty.
 * @param other The matrix to be moved from.
 * @return Reference to this matrix.

mat& mat::operator=(mat&& other) noexcept {
    if (this == &other) {
        return *this; // Handle self-assignment
    }

    // --- 1. Release this object's current resources if it owns them ---
    if (!is_shared_segment && mapped_file_handle) {
        close_mapped_file(mapped_file_handle);
        mapped_file_handle = nullptr;
    }
    if (is_temp_file && !backing_filename.empty()) {
        remove(backing_filename.c_str());
    }

    // --- 2. Transfer ownership from 'other' to 'this' ---
    row = other.row;
    col = other.col;
    mapped_data = other.mapped_data;
    mapped_file_handle = other.mapped_file_handle;
    mapped_size = other.mapped_size;
    backing_filename = std::move(other.backing_filename); // Use std::move for string
    is_temp_file = other.is_temp_file;
    is_shared_segment = other.is_shared_segment;
    data_segment_start = other.data_segment_start;

    // --- 3. Nullify 'other' to prevent its destructor from freeing resources ---
    other.row = 0;
    other.col = 0;
    other.mapped_data = nullptr;
    other.mapped_file_handle = nullptr;
    other.mapped_size = 0;
    other.backing_filename.clear();
    other.is_temp_file = false;
    other.is_shared_segment = false;
    other.data_segment_start = nullptr;

    return *this;
}
 */
/**
 * @brief Copy assignment operator for matrix.
 * Releases current resources and then performs a copy (deep or shallow) from 'other'.
 * Uses the copy-and-swap idiom for strong exception safety.
 * @param other The matrix to be copied from.
 * @return Reference to this matrix.
mat& mat::operator=(const mat& other) {
    // Create a temporary copy using the copy constructor
    // This handles deep/shallow copy logic based on 'other.is_shared_segment'
    // and correctly allocates/maps a new file if needed.
    mat temp(other);

    // Swap all internal state with the temporary object.
    // This object now owns the resources of 'temp', and 'temp' now owns the
    // old resources of 'this'. When 'temp' goes out of scope, it will
    // correctly clean up the old resources of 'this'.
    std::swap(row, temp.row);
    std::swap(col, temp.col);
    std::swap(mapped_data, temp.mapped_data);
    std::swap(mapped_file_handle, temp.mapped_file_handle);
    std::swap(mapped_size, temp.mapped_size);
    std::swap(backing_filename, temp.backing_filename);
    std::swap(is_temp_file, temp.is_temp_file);
    std::swap(is_shared_segment, temp.is_shared_segment);
    std::swap(data_segment_start, temp.data_segment_start);

    return *this;
}
 */

// New method to initialize from a shared map segment
void mat::assign_shared_segment(MappedFile* shared_map_handle, float* shared_map_base_ptr,
                            size_t segment_byte_offset_in_shared_map,
                            int new_row, int new_col, const std::string& path_to_shared_file) {
    // Release any previously owned resources if this mat was initialized differently
    if (this->mapped_file_handle && !this->is_shared_segment) {
        close_mapped_file(this->mapped_file_handle);
        this->mapped_file_handle = nullptr;
        this->mapped_data = nullptr;
    }
    if (this->is_temp_file && !this->backing_filename.empty() && !this->is_shared_segment) {
        remove(this->backing_filename.c_str());
    }

    this->row = new_row;
    this->col = new_col;
    this->mapped_file_handle = shared_map_handle; // Assign the shared handle
    this->mapped_data = shared_map_base_ptr;     // Assign the base pointer of the full mapping
    // Calculate the start of this matrix's segment within the full mapping
    this->data_segment_start = (float*)((char*)shared_map_base_ptr + segment_byte_offset_in_shared_map);
    // The mapped_size for a shared segment refers to the size of *this specific segment*
    this->mapped_size = static_cast<size_t>(new_row) * new_col * sizeof(float);
    this->backing_filename = path_to_shared_file; // Keep track of the shared file path
    this->is_temp_file = false;                  // A shared segment is not a temporary file owned by 'this'
    this->is_shared_segment = true;              // Mark as a shared segment
}


// Access operator (non-const)
float& mat::operator()(int i, int j) {
    // Determine the effective base pointer for this matrix's data
    float* current_data_ptr = is_shared_segment ? data_segment_start : mapped_data;

    // Check for null pointer AND bounds
    if (!current_data_ptr || i < 0 || i >= row || j < 0 || j >= col) {
        throw std::out_of_range("Matrix index (" + std::to_string(i) + "," + std::to_string(j) +
                                ") out of bounds for matrix " + std::to_string(row) + "x" + std::to_string(col) +
                                (current_data_ptr ? "" : " (matrix data not mapped)."));
    }
    size_t index = static_cast<size_t>(i) * col + j;
    // This check against mapped_size/sizeof(float) ensures we don't go out of the physically mapped region.
    // For shared segments, mapped_size is the size of the *segment*, so this is correct.
    if (index * sizeof(float) >= mapped_size) { // Check if byte offset of element is within mapped_size
        throw std::out_of_range("Calculated element offset (byte " + std::to_string(index * sizeof(float)) +
                                ") exceeds mapped data size (" + std::to_string(mapped_size) + ") bytes.");
    }
    return current_data_ptr[index];
}

// Access operator (const)
const float& mat::operator()(int i, int j) const {
    // Determine the effective base pointer for this matrix's data
    const float* current_data_ptr = is_shared_segment ? data_segment_start : mapped_data;

    // Check for null pointer AND bounds
    if (!current_data_ptr || i < 0 || i >= row || j < 0 || j >= col) {
        throw std::out_of_range("Matrix index (" + std::to_string(i) + "," + std::to_string(j) +
                                ") out of bounds for matrix " + std::to_string(row) + "x" + std::to_string(col) +
                                (current_data_ptr ? "" : " (matrix data not mapped)."));
    }
    size_t index = static_cast<size_t>(i) * col + j;
    if (index * sizeof(float) >= mapped_size) { // Check if byte offset of element is within mapped_size
        throw std::out_of_range("Calculated element offset (byte " + std::to_string(index * sizeof(float)) +
                                ") exceeds mapped data size (" + std::to_string(mapped_size) + ") bytes.");
    }
    return current_data_ptr[index];
}

// return row of matrix from mapped file using 0-based row index
std::vector<float> mat::operator()(int i) const {
    const float* current_data_ptr = is_shared_segment ? data_segment_start : mapped_data;
    if (!current_data_ptr || i < 0 || i >= row) {
        throw std::out_of_range("Row index " + std::to_string(i) + " out of bounds for matrix with " + std::to_string(row) + " rows or matrix not mapped.");
    }
    std::vector<float> row_data(col);
    size_t offset = static_cast<size_t>(i) * col;
    size_t row_bytes = static_cast<size_t>(col) * sizeof(float);

    // Check if the read will go out of bounds of the mapped data for this matrix/segment
    if ((offset * sizeof(float)) + row_bytes > mapped_size) {
        throw std::out_of_range("Row data (offset " + std::to_string(offset * sizeof(float)) + ", length " + std::to_string(row_bytes) +
                                ") would exceed mapped data size (" + std::to_string(mapped_size) + ") bytes.");
    }
    std::memcpy(row_data.data(), current_data_ptr + offset, row_bytes);
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

    if (m.col == 0) { // Matrix has zero columns
        return {}; // Return an empty vector
    }

    // Use the correct data pointer for shared vs. non-shared
    const float* current_data_ptr = m.is_shared_segment ? m.data_segment_start : m.mapped_data;

    if (!current_data_ptr) {
        throw std::runtime_error("getRow: Matrix data is not mapped.");
    }

    std::vector<float> row_vector(m.col);
    size_t offset = static_cast<size_t>(row_idx) * m.col;
    size_t row_bytes = static_cast<size_t>(m.col) * sizeof(float);

    // Check if the read will go out of bounds of the mapped data for *this specific mat's view/mapping*
    // This assumes mapped_size correctly reflects the size of the portion 'm' is responsible for.
    if ((offset * sizeof(float)) + row_bytes > m.mapped_size) {
        throw std::out_of_range("getRow: Calculated row data exceeds mapped memory bounds for this matrix instance.");
    }

    std::memcpy(row_vector.data(), current_data_ptr + offset, row_bytes);

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

    if (m.col == 0) { // Matrix has zero columns; data must be empty.
        return; // Nothing to set
    }

    // Use the correct data pointer for shared vs. non-shared
    float* current_data_ptr = m.is_shared_segment ? m.data_segment_start : m.mapped_data;

    if (!current_data_ptr) {
        throw std::runtime_error("setRow: Matrix data is not mapped.");
    }

    size_t offset = static_cast<size_t>(row_idx) * m.col;
    size_t row_bytes = static_cast<size_t>(m.col) * sizeof(float);

    // Check if the write will go out of bounds of the mapped data for *this specific mat's view/mapping*
    if ((offset * sizeof(float)) + row_bytes > m.mapped_size) {
        throw std::out_of_range("setRow: Calculated row data exceeds mapped memory bounds for writing to this matrix instance.");
    }

    std::memcpy(current_data_ptr + offset, data.data(), row_bytes);
}

/**
 * @brief write mapped data of mat and appened it to binary file
 * @param matrix matrix to be written
 * @param locationWithFileName location of file with its file name (Location/filename.bin)
 */
void write2filefrommat(const mat& matrix, const std::string& locationWithFileName) {
    // Open the file in binary mode for output and append to it.
    // If the file doesn't exist, it will be created.
    std::ofstream outFile(locationWithFileName, std::ios::binary | std::ios::out | std::ios::app);

    if (!outFile.is_open()) {
        throw std::runtime_error("Failed to open file for appending: " + locationWithFileName);
    }

    // Serialize matrix dimensions (rows, cols)
    uint64_t num_rows = static_cast<uint64_t>(matrix.row);
    uint64_t num_cols = static_cast<uint64_t>(matrix.col);

    // Optional: write dimensions. Uncomment if needed, but the comment says it's optional
    // outFile.write(reinterpret_cast<const char*>(&num_rows), sizeof(num_rows));
    // outFile.write(reinterpret_cast<const char*>(&num_cols), sizeof(num_cols));

    if (!outFile) { // Check for errors after potential dimension write
        outFile.close();
        throw std::runtime_error("Error writing matrix dimensions (or checking stream) to file: " + locationWithFileName);
    }

    // Serialize matrix data
    size_t num_elements = static_cast<size_t>(num_rows) * static_cast<size_t>(num_cols);
    if (num_elements > 0) {
        // Use the correct data pointer for shared vs. non-shared
        const float* data_ptr = matrix.is_shared_segment ? matrix.data_segment_start : matrix.mapped_data;

        if (data_ptr == nullptr) { // Critical check for unmapped data
            outFile.close();
            throw std::runtime_error("Matrix has non-zero size but data pointer is null (unmapped). Cannot write to file: " + locationWithFileName);
        }
        size_t data_size_bytes = num_elements * sizeof(float);
        outFile.write(reinterpret_cast<const char*>(data_ptr), data_size_bytes);

        if (!outFile) { // Check for errors after writing data
            outFile.close();
            throw std::runtime_error("Error writing matrix data to file: " + locationWithFileName);
        }
    }
    // If num_elements is 0, nothing else is written, which is fine.

    outFile.close();
    if (!outFile) { // Check for errors that might occur during close (e.g., buffer flush failure)
        throw std::runtime_error("Error occurred while closing file (e.g., flush error): " + locationWithFileName);
    }
}