#include "include/memory_map.h" // Adjust path as necessary, e.g., "../utils/memory_map.h"
#include <cstdlib>  // For malloc, free
#include <cstdio>   // For fprintf (for optional error reporting)

// Platform-specific includes and MappedFile struct definition
#if defined(_WIN64)
    #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
    #endif
#include <windows.h>

    // Define the MappedFile structure for Windows
    struct MappedFile {
        HANDLE hFile;
        HANDLE hMapping;
        void* pData;
        size_t fileSize;
    };

#elif defined(__linux__)
    #include <sys/types.h>
    #include <sys/stat.h>
    #include <fcntl.h>
    #include <unistd.h>
    #include <sys/mman.h>
    #include <errno.h>   // For errno
    #include <string.h>  // For strerror

    // Define the MappedFile structure for Linux
    struct MappedFile {
        int fd;
        void* pData;
        size_t fileSize;
    };
#else
    #error "Unsupported platform. Please define _WIN32 or __linux__."
    struct MappedFile { char _dummy; }; // Dummy for compilers that might proceed past #error
#endif

extern "C" {

#if defined(_WIN64)
int create_or_resize_file(const char* filepath, size_t required_size) {
    HANDLE hFile = CreateFileA(
        filepath,
        GENERIC_READ | GENERIC_WRITE,
        FILE_SHARE_READ, // Allow other processes to read the file
        NULL,
        OPEN_ALWAYS,     // Open if exists, create if not
        FILE_ATTRIBUTE_NORMAL,
        NULL
    );

    if (hFile == INVALID_HANDLE_VALUE) {
        // Optional: fprintf(stderr, "create_or_resize_file (Win): Failed to open/create file '%s'. Error: %lu\n", filepath, GetLastError());
        return -1;
    }

    LARGE_INTEGER fileSize;
    if (!GetFileSizeEx(hFile, &fileSize)) {
        // Optional: fprintf(stderr, "create_or_resize_file (Win): Failed to get file size for '%s'. Error: %lu\n", filepath, GetLastError());
        CloseHandle(hFile);
        return -2;
    }

    if (static_cast<size_t>(fileSize.QuadPart) < required_size) {
        LARGE_INTEGER newSize;
        newSize.QuadPart = required_size;
        if (SetFilePointerEx(hFile, newSize, NULL, FILE_BEGIN) == 0 && GetLastError() != NO_ERROR) {
            // Optional: fprintf(stderr, "create_or_resize_file (Win): Failed to set file pointer for '%s'. Error: %lu\n", filepath, GetLastError());
            CloseHandle(hFile);
            return -3;
        }
        if (!SetEndOfFile(hFile)) {
            // Optional: fprintf(stderr, "create_or_resize_file (Win): Failed to set end of file for '%s'. Error: %lu\n", filepath, GetLastError());
            CloseHandle(hFile);
            return -4;
        }
    }

    CloseHandle(hFile);
    return 0;
}
#elif defined(__linux__)
int create_or_resize_file(const char* filepath, size_t required_size) {
    int fd = open(filepath, O_RDWR | O_CREAT, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH); // 0644 permissions
    if (fd == -1) {
        // Optional: fprintf(stderr, "create_or_resize_file (Linux): Failed to open/create file '%s'. Error: %s\n", filepath, strerror(errno));
        return -1;
    }

    struct stat st;
    if (fstat(fd, &st) == -1) {
        // Optional: fprintf(stderr, "create_or_resize_file (Linux): Failed to get file size for '%s'. Error: %s\n", filepath, strerror(errno));
        close(fd);
        return -2;
    }

    if (static_cast<size_t>(st.st_size) < required_size) {
        if (ftruncate(fd, required_size) == -1) {
            // Optional: fprintf(stderr, "create_or_resize_file (Linux): Failed to resize file '%s' to %zu bytes. Error: %s\n", filepath, required_size, strerror(errno));
            close(fd);
            return -3;
        }
    }

    close(fd);
    return 0;
}
#else // Unsupported platform
int create_or_resize_file(const char* filepath, size_t required_size) {
    (void)filepath; (void)required_size;
    // Optional: fprintf(stderr, "create_or_resize_file: Unsupported platform.\n");
    return -99; // Indicate unsupported platform
}
#endif

#if defined(_WIN64)
int open_mapped_file(const char* filepath, bool read_write, MappedFile** handle, void** data, size_t* size) {
    if (!handle || !data || !size) {
        return -1; // Invalid arguments
    }

    *handle = NULL;
    *data = NULL;
    *size = 0;

    MappedFile* mf = (MappedFile*)malloc(sizeof(MappedFile));
    if (!mf) {
        return -2; // Out of memory
    }
    mf->hFile = INVALID_HANDLE_VALUE;
    mf->hMapping = NULL;
    mf->pData = NULL;
    mf->fileSize = 0;

    DWORD dwDesiredAccess = read_write ? (GENERIC_READ | GENERIC_WRITE) : GENERIC_READ;
    // Allow sharing for read, but exclusive for write to prevent conflicts during mapping
    DWORD dwShareMode = FILE_SHARE_READ;

    mf->hFile = CreateFileA(
        filepath,
        dwDesiredAccess,
        dwShareMode,
        NULL,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        NULL
    );

    if (mf->hFile == INVALID_HANDLE_VALUE) {
        // Optional: fprintf(stderr, "open_mapped_file (Win): Failed to open file '%s'. Error: %lu\n", filepath, GetLastError());
        free(mf);
        return -3;
    }

    LARGE_INTEGER fileSizeLI;
    if (!GetFileSizeEx(mf->hFile, &fileSizeLI) || fileSizeLI.QuadPart == 0) {
        CloseHandle(mf->hFile);
        // Optional: fprintf(stderr, "open_mapped_file (Win): Failed to get file size or file is empty for '%s'. Error: %lu\n", filepath, GetLastError());
        free(mf);
        return -4; // Cannot map an empty file or failed to get size
    }
    mf->fileSize = static_cast<size_t>(fileSizeLI.QuadPart);

    DWORD flProtect = read_write ? PAGE_READWRITE : PAGE_READONLY;
    mf->hMapping = CreateFileMappingA(
        mf->hFile,
        NULL,
        flProtect,
        0, 0, // Map entire file
        NULL
    );

    if (mf->hMapping == NULL) {
        // Optional: fprintf(stderr, "open_mapped_file (Win): CreateFileMappingA failed for '%s'. Error: %lu\n", filepath, GetLastError());
        CloseHandle(mf->hFile);
        free(mf);
        return -5;
    }

    DWORD dwMapViewAccess = read_write ? FILE_MAP_ALL_ACCESS : FILE_MAP_READ;
    mf->pData = MapViewOfFile(mf->hMapping, dwMapViewAccess, 0, 0, 0);

    if (mf->pData == NULL) {
        // Optional: fprintf(stderr, "open_mapped_file (Win): MapViewOfFile failed for '%s'. Error: %lu\n", filepath, GetLastError());
        CloseHandle(mf->hMapping);
        CloseHandle(mf->hFile);
        free(mf);
        return -6;
    }

    *handle = mf;
    *data = mf->pData;
    *size = mf->fileSize;
    return 0; // Success
}
#elif defined(__linux__)
int open_mapped_file(const char* filepath, bool read_write, MappedFile** handle, void** data, size_t* size) {
    if (!handle || !data || !size) {
        return -1; // Invalid arguments
    }

    *handle = NULL;
    *data = NULL;
    *size = 0;

    MappedFile* mf = (MappedFile*)malloc(sizeof(MappedFile));
    if (!mf) {
        return -2; // Out of memory
    }
    mf->fd = -1;
    mf->pData = NULL;
    mf->fileSize = 0;

    int open_flags = read_write ? O_RDWR : O_RDONLY;
    mf->fd = open(filepath, open_flags);
    if (mf->fd == -1) {
        // Optional: fprintf(stderr, "open_mapped_file (Linux): Failed to open file '%s'. Error: %s\n", filepath, strerror(errno));
        free(mf);
        return -3;
    }

    struct stat st;
    if (fstat(mf->fd, &st) == -1) {
        // Optional: fprintf(stderr, "open_mapped_file (Linux): Failed to fstat file '%s'. Error: %s\n", filepath, strerror(errno));
        close(mf->fd);
        free(mf);
        return -4; // Failed to get size
    }
    if (st.st_size == 0) {
        // Optional: fprintf(stderr, "open_mapped_file (Linux): File '%s' is empty, cannot map.\n", filepath);
        close(mf->fd);
        free(mf);
        return -4; // Cannot map an empty file
    }
    mf->fileSize = static_cast<size_t>(st.st_size);

    int prot = read_write ? (PROT_READ | PROT_WRITE) : PROT_READ;
    mf->pData = mmap(NULL, mf->fileSize, prot, MAP_SHARED, mf->fd, 0);

    if (mf->pData == MAP_FAILED) {
        // Optional: fprintf(stderr, "open_mapped_file (Linux): mmap failed for '%s'. Error: %s\n", filepath, strerror(errno));
        mf->pData = NULL; // Ensure pData is NULL on failure path
        close(mf->fd);
        free(mf);
        return -5;
    }

    *handle = mf;
    *data = mf->pData;
    *size = mf->fileSize;
    return 0; // Success
}
#else // Unsupported platform
int open_mapped_file(const char* filepath, bool read_write, MappedFile** handle, void** data, size_t* size) {
    (void)filepath; (void)read_write;
    if (handle) *handle = NULL;
    if (data) *data = NULL;
    if (size) *size = 0;
    // Optional: fprintf(stderr, "open_mapped_file: Unsupported platform.\n");
    return -99;
}
#endif

#if defined(_WIN64)
void close_mapped_file(MappedFile* handle) {
    if (!handle) return;
    if (handle->pData) UnmapViewOfFile(handle->pData);
    if (handle->hMapping) CloseHandle(handle->hMapping);
    if (handle->hFile != INVALID_HANDLE_VALUE) CloseHandle(handle->hFile);
    free(handle);
}
#elif defined(__linux__)
void close_mapped_file(MappedFile* handle) {
    if (!handle) return;
    if (handle->pData) { // pData will be NULL if mmap failed
        munmap(handle->pData, handle->fileSize);
    }
    if (handle->fd != -1) {
        close(handle->fd);
    }
    free(handle);
}
#else // Unsupported platform
void close_mapped_file(MappedFile* handle) {
    (void)handle;
    // Optional: fprintf(stderr, "close_mapped_file: Unsupported platform.\n");
    // If a MappedFile struct were allocated by a stub open_mapped_file, free(handle) would be needed.
    // However, the stub open_mapped_file above does not allocate.
}
#endif
} // extern "C"