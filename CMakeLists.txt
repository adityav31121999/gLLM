
# root CMakeLists.txt
cmake_minimum_required(VERSION 3.30.0 FATAL_ERROR)

# project and language
project(gLLM C CXX)
set(VERSION 0.0.0.1)

# language settings
enable_language(C CXX)
set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# binaries for debug and release
set(OSbit 32)
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(OSbit 64)
endif()
set(FullOutputDir "${CMAKE_SOURCE_DIR}/bin/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}-${OSbit}/${CMAKE_BUILD_TYPE}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${FullOutputDir}/static_libs")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${FullOutputDir}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${FullOutputDir}")

# OpenCL setup
set(OpenCL_CL_VERSION "300")
set(CL_TARGET_OPENCL_VERSION "300")
find_package(OpenCL REQUIRED)
if(OpenCL_FOUND)
    message(STATUS "Found OpenCL version ${OpenCL_CL_VERSION}")
    include_directories(${OpenCL_INCLUDE_DIRS})
else()
    message(WARNING "OpenCL not found, using CPU instead.")
endif()

# CUDA setup
find_package(CUDAToolkit REQUIRED)
if(CUDAToolkit_FOUND)
    message(STATUS "Found CUDA version ${CUDAToolkit_VERSION}")
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_ARCHITECTURES 86)  # Ampere (RTX 30 series)
else()
    message(WARNING "CUDA Toolkit not found, using OpenCL instead.")
endif()

# Set all subdirectories as subProjects
set(subProjects
    src/maths
    src/neural
    src/model
    src/script
)

# Include each subdirectory
foreach(projects IN LISTS subProjects)
    include_directories(${projects})
    if(NOT TARGET ${projects})
        add_subdirectory(${projects})
    endif()
endforeach()
