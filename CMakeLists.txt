# root CMakeLists.txt
cmake_minimum_required(VERSION 3.30.0 FATAL_ERROR)
project(gLLM)
set(VERSION 0.1.5.0)
message("> Project Name: ${PROJECT_NAME} | Version: ${VERSION}")
message("> DCA Experimental Project for structure and logic verification")

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
    message("> Running on ${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}-${OSbit}-bit System")
endif()
set(FullOutputDir "${CMAKE_SOURCE_DIR}/bin/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}-${OSbit}/${CMAKE_BUILD_TYPE}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${FullOutputDir}/static_libs")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${FullOutputDir}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${FullOutputDir}")

# turn off both opencl and cuda macro to turn on USE_CPU macro
set(USE_CL OFF)
set(USE_CU ON)

# language settings
set(LANGUAGES C CXX
    if(USE_CU)
        CUDA
    endif()
)
enable_language(C CXX)
set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# set compile definitions
if(USE_CU)
    find_package(CUDAToolkit REQUIRED)
    message(STATUS "Found CUDA version ${CUDAToolkit_VERSION}")
    message(STATUS "Using CUDA for compilation.")
    message(STATUS "Using CUDA for better operations on Nvidia GPUs:)")
    if(LINUX)
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static-libstdc++")
    endif()
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(MY_CUDA_ARCH "86" CACHE STRING "Target CUDA Architecture(s) e.g., 75, 86, 90")
    set(CMAKE_CUDA_ARCHITECTURES ${MY_CUDA_ARCH})
    add_compile_definitions(USE_CU)
    include_directories(${CUDAToolkit_INCLUDE_DIRS})

    # Find nvidia-smi and print GPU names
    find_program(NVIDIA_SMI_EXECUTABLE nvidia-smi)
    if(NVIDIA_SMI_EXECUTABLE)
        execute_process(
            COMMAND ${NVIDIA_SMI_EXECUTABLE} --query-gpu=name --format=csv,noheader
            OUTPUT_VARIABLE GPU_NAMES
            OUTPUT_STRIP_TRAILING_WHITESPACE)
        message(STATUS "Detected GPUs: ${GPU_NAMES}")
    endif()

    if(MSVC)
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
    endif()
elseif(USE_CL)
    set(OpenCL_CL_VERSION "300")
    set(CL_TARGET_OPENCL_VERSION "300")
    find_package(OpenCL REQUIRED)
    message(STATUS "Found OpenCL version ${OpenCL_VERSION_STRING}")
    message(STATUS "Using OpenCL for compilation.")
    message(STATUS "Using OpenCL for GPUs with OpenCL support:)")
    include_directories(${OpenCL_INCLUDE_DIRS})
    add_compile_definitions(USE_CL)
else()
    message(STATUS "OpenCL and CUDA not found or not using:(")
    message(STATUS "Using C++ on CPU for operations:)")
    message(STATUS "CPU: ${CMAKE_SYSTEM_PROCESSOR}")
    message(STATUS "C Standard: ${CMAKE_C_STANDARD}")
    message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
    message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
    add_compile_definitions(USE_CPU)
endif()

# Set all subdirectories as subProjects
set(subProjects
    src/memorymap   # memory mapping
    src/basic       # math functions
    src/neural      # neural network and mechanism
    src/model       # model
)

# Include each subdirectory
foreach(projects IN LISTS subProjects)
    include_directories(${projects})
    if(NOT TARGET ${projects})
        add_subdirectory(${projects})
    endif()
endforeach()

# FOR EXECUTABLE
add_executable(${PROJECT_NAME} main.cpp gllm.h kernels.cpp)
target_link_libraries(gLLM PRIVATE memorymap maths neural model)
if(USE_CU)
    target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart)
elseif(USE_CL)
    target_link_libraries(${PROJECT_NAME} PRIVATE OpenCL::OpenCL)
endif()