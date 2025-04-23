
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

# Check for which dependency to use
if(CUDAToolkit_FOUND AND OpenCL_FOUND)
    set(USE_CUDA ON)
    set(USE_OPENCL OFF) # not for nvidia
    set(USE_CPU OFF)    # gpu available
    message(STATUS "Using CUDA for compilation.")
elseif(OpenCL_FOUND AND NOT CUDAToolkit_FOUND)
    set(USE_OPENCL ON)
    set(USE_CUDA OFF)   # for non-nvidia
    set(USE_CPU OFF)
    message(STATUS "Using OpenCL for compilation.")
else()
    set(USE_CUDA OFF)
    set(USE_OPENCL OFF)
    set(USE_CPU ON)     # cpu available
    message(FATAL_ERROR "Neither CUDA nor OpenCL found, provide CUDA/OPENCL, 
                        or use core C++ code with cpu.")
endif()

# set compile definitions
if(USE_CUDA)
    message(STATUS "Found CUDA version ${CUDAToolkit_VERSION}")
    message(STATUS "Using CUDA for better efficiency on CUDA-based GPUs :)")
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    # Ampere (RTX 30 series)
    set(MY_CUDA_ARCH "86" CACHE STRING "Target CUDA Architecture(s) e.g., 75, 86, 90")
    set(CMAKE_CUDA_ARCHITECTURES ${MY_CUDA_ARCH})
    add_compile_definitions(USE_CUDA)
    include_directories(${CUDAToolkit_INCLUDE_DIRS}) # Ensure CUDA headers are available
elseif(USE_OPENCL)
    message(STATUS "Found OpenCL version ${OpenCL_CL_VERSION}")
    message(STATUS "Using OpenCL for better efficiency on non-CUDA CL-conformant GPUs :)")
    include_directories(${OpenCL_INCLUDE_DIRS})
    add_compile_definitions(USE_OPENCL)
else(USE_CPU)
  message(STATUS "Using CPU Compilation")
  message(STATUS "Using C++ for CPU based operations :)")
  add_compile_definitions(USE_CPU)
endif()

# Set all subdirectories as subProjects
set(subProjects
    src/maths       # math functions
    src/neural      # neural network and mechanism
    src/model       # model files
    # src/script      # script for CLI
)

# Include each subdirectory
foreach(projects IN LISTS subProjects)
    include_directories(${projects})
    if(NOT TARGET ${projects})
        add_subdirectory(${projects})
    endif()
endforeach()
