# src/maths/CMakeLists.txt
cmake_minimum_required(VERSION 3.30.0 FATAL_ERROR)
project(maths C CXX)
# all .hpp and .h files
include_directories(include)
# all source files
add_library(maths STATIC
    # basic files
    src/basic/activations.cpp
    src/basic/vect.cpp
    src/basic/weights.cpp
    src/basic/loss.cpp
    # cuda
    src/basic/cu/activations.cu
    src/basic/cu/vect.cu
    #cl
    src/basic/cl/activations.cl
    src/basic/cl/vect.cl

    # mat files
    src/mat/mat.cpp
    src/mat/inverse.cpp
    src/mat/operators.cpp
    src/mat/matops.cpp
    src/mat/mult.cpp
    src/mat/prop.cpp
    src/mat/write.cpp
    src/mat/flat.cpp
    # cl
    src/mat/cl/operators.cl
    src/mat/cl/inverse.cl
    # cuda
    src/mat/cu/operators.cu
    src/mat/cu/inverse.cu
)

target_link_libraries(maths
    PUBLIC # Important: Make OpenCL and CUDA linking public
        ${OpenCL_LIBRARIES}
        ${CUDAToolkit_LIBRARIES}
    memorymap
)
