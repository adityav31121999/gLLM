
# src/maths/CMakeLists.txt
cmake_minimum_required(VERSION 3.25.0 FATAL_ERROR)
project(maths C CXX)

include_directories(include)

add_library(maths STATIC
    # basic files
    src/basic/activations.cpp
    src/basic/vect.cpp
    src/basic/weights.cpp

    src/basic/cu/activations.cu
    src/basic/cu/vect.cu

    src/basic/cl/activations.cl
    src/basic/cl/vect.cl

    # mat files
    src/mat/mat.cpp
    src/mat/inverse.cpp
    src/mat/operators.cpp
    src/mat/prop.cpp
    src/mat/write.cpp
    src/mat/flat.cpp

    src/mat/cl/operators.cl
    src/mat/cl/inverse.cl
    
    src/mat/cu/operators.cu
    src/mat/cu/inverse.cu

    # stats files
    src/stats/basic.cpp

    src/stats/cl/basic.cl

    src/stats/cu/basic.cu
)

target_link_libraries(maths
    PUBLIC # Important: Make OpenCL and CUDA linking public
        ${OpenCL_LIBRARIES}
        ${CUDAToolkit_LIBRARIES}
    memorymap
)
