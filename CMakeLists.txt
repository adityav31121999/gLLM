
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

# CUDA setup
find_package(CUDAToolkit REQUIRED)

if(CUDAToolkit_FOUND AND OpenCL_FOUND)
    set(USE_CUDA ON)
    set(USE_OPENCL OFF)
    set(USE_CPU OFF)
    message(STATUS "Using CUDA for compilation.")
elseif(OpenCL_FOUND AND NOT CUDAToolkit_FOUND)
    set(USE_OPENCL ON)
    set(USE_CUDA OFF)
    set(USE_CPU OFF)
    message(STATUS "Using OpenCL for compilation.")
else()
    set(USE_CUDA OFF)
    set(USE_OPENCL OFF)
    set(USE_CPU ON)
    message(FATAL_ERROR "Neither CUDA nor OpenCL found, provide CUDA/OPENCL.")
endif()

if(USE_CUDA)
    message(STATUS "Found CUDA version ${CUDAToolkit_VERSION}")
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_ARCHITECTURES 86)  # Ampere (RTX 30 series)
    add_definitions(-DUSE_CUDA)
    include_directories(${CUDAToolkit_INCLUDE_DIRS}) # Ensure CUDA headers are available
elseif(USE_OPENCL)
    message(STATUS "Found OpenCL version ${OpenCL_CL_VERSION}")
    include_directories(${OpenCL_INCLUDE_DIRS})
    add_definitions(-DUSE_OPENCL)
else(USE_CPU)
  message(STATUS "Using CPU Compilation")
  add_definitions(-DUSE_CPU)
endif()

# Set all subdirectories as subProjects
set(subProjects
    src/maths
    src/neural
    src/model
    # src/script
)

# Include each subdirectory
foreach(projects IN LISTS subProjects)
    include_directories(${projects})
    if(NOT TARGET ${projects})
        add_subdirectory(${projects})
    endif()
endforeach()

# Add a global compile definitions
if(USE_CUDA)
  add_compile_definitions(USE_CUDA)
elseif(USE_OPENCL)
  add_compile_definitions(USE_OPENCL)
else()
    add_compile_definitions(USE_CPU)
endif()
