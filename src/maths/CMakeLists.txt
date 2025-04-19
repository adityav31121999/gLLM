
# src/maths/CMakeLists.txt
cmake_minimum_required(VERSION 3.30.0 FATAL_ERROR)
project(maths C CXX)

include_directories(include)

add_library(maths STATIC
    # basic files
    basic/activations.cpp
    basic/vect.cpp
    basic/weights.cpp

    basic/cu/activations.cu
    basic/cu/vect.cu

    basic/cl/clactivations.cl
    basic/cl/clvect.cl

    # mat files
    mat/det.cpp
    mat/inverse.cpp
    mat/operators.cpp
    mat/prop.cpp
    mat/submat.cpp

    # stats files
    stats/basic.cpp
)

target_link_libraries(maths
    PUBLIC # Important: Make OpenCL and CUDA linking public
        ${OpenCL_LIBRARIES}
        ${CUDAToolkit_LIBRARIES}
)
