#ifndef MEMORY_MAP_H
#define MEMORY_MAP_H

#include <stddef.h> // For size_t
#include <stdbool.h> // For bool

#ifdef __cplusplus
extern "C" {
#endif

// Opaque structure to hold platform-specific mapping details
typedef struct MappedFile MappedFile;

/**
 * @brief Maps a file into memory.
 *
 * Opens the specified file and maps its contents into the process's
 * address space.
 *
 * @param filepath Absolute path to the file to map.
 * @param read_write If true, map for reading and writing; otherwise, read-only.
 * @param mapped_file_out Pointer to a MappedFile pointer. On success, this will
 *                          be allocated and populated with mapping details.
 *                          The caller is responsible for calling close_mapped_file
 *                          on this structure when done.
 * @param mapped_ptr_out Pointer to a void pointer. On success, this will point
 *                       to the beginning of the mapped memory region.
 * @param size_out Pointer to a size_t. On success, this will contain the size
 *                 of the mapped region (file size).
 * @return 0 on success, non-zero on failure.
 */
int open_mapped_file(const char* filepath, bool read_write, MappedFile** mapped_file_out, void** mapped_ptr_out, size_t* size_out);

/**
 * @brief Ensures a file exists and is at least the specified size.
 *
 * Creates the file if it doesn't exist. If it exists but is smaller
 * than required_size, it extends the file to required_size.
 * @param filepath Path to the file.
 * @param required_size The minimum desired size of the file in bytes.
 * @return 0 on success, non-zero on failure.
 */
int create_or_resize_file(const char* filepath, size_t required_size);

/**
 * @brief Unmaps the memory region and closes associated resources.
 *
 * @param mapped_file The MappedFile structure obtained from open_mapped_file.
 */
void close_mapped_file(MappedFile* mapped_file);

#ifdef __cplusplus
}
#endif

#endif // MEMORY_MAP_H
