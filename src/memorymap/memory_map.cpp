#include "memory_map.h" // Adjust path as necessary, e.g., "../utils/memory_map.h"
#include <windows.h>
#include <cstdio>   // For remove, though mat.hpp handles its own temp file removal
#include <iostream> // For potential debugging, can be removed

// Define the MappedFile structure for Windows
struct MappedFile {
    HANDLE hFile;
    HANDLE hMapping;
    void* pData;
    size_t fileSize;
};

extern "C" {

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
        // Optional: fprintf(stderr, "create_or_resize_file: Failed to open/create file '%s'. Error: %lu\n", filepath, GetLastError());
        return -1;
    }

    LARGE_INTEGER fileSize;
    if (!GetFileSizeEx(hFile, &fileSize)) {
        // Optional: fprintf(stderr, "create_or_resize_file: Failed to get file size for '%s'. Error: %lu\n", filepath, GetLastError());
        CloseHandle(hFile);
        return -2;
    }

    if (static_cast<size_t>(fileSize.QuadPart) < required_size) {
        LARGE_INTEGER newSize;
        newSize.QuadPart = required_size;
        if (SetFilePointerEx(hFile, newSize, NULL, FILE_BEGIN) == 0 && GetLastError() != NO_ERROR) {
            // Optional: fprintf(stderr, "create_or_resize_file: Failed to set file pointer for '%s'. Error: %lu\n", filepath, GetLastError());
            CloseHandle(hFile);
            return -3;
        }
        if (!SetEndOfFile(hFile)) {
            // Optional: fprintf(stderr, "create_or_resize_file: Failed to set end of file for '%s'. Error: %lu\n", filepath, GetLastError());
            CloseHandle(hFile);
            return -4;
        }
    }

    CloseHandle(hFile);
    return 0;
}

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
        free(mf);
        return -3;
    }

    LARGE_INTEGER fileSizeLI;
    if (!GetFileSizeEx(mf->hFile, &fileSizeLI) || fileSizeLI.QuadPart == 0) {
        CloseHandle(mf->hFile);
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
        CloseHandle(mf->hFile);
        free(mf);
        return -5;
    }

    DWORD dwMapViewAccess = read_write ? FILE_MAP_ALL_ACCESS : FILE_MAP_READ;
    mf->pData = MapViewOfFile(mf->hMapping, dwMapViewAccess, 0, 0, 0);

    if (mf->pData == NULL) {
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

void close_mapped_file(MappedFile* handle) {
    if (!handle) return;
    if (handle->pData) UnmapViewOfFile(handle->pData);
    if (handle->hMapping) CloseHandle(handle->hMapping);
    if (handle->hFile != INVALID_HANDLE_VALUE) CloseHandle(handle->hFile);
    free(handle);
}

} // extern "C"