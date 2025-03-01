
# src/neural/CMakeLists.txt
cmake_minimum_required(VERSION 3.30.0 FATAL_ERROR)
project(neural C CXX)

include_directories(include)

add_library(neural STATIC
    # mlp files
    mlp/mlp.cpp
    mlp/forprop.cpp
    mlp/backprop.cpp
    mlp/loss.cpp
    mlp/weights.cpp
    mlp/train.cpp
    
    # attention
    attention/attention.cpp
    attention/forward.cpp
    attention/backward.cpp
    attention/train.cpp

    # block
    attention/block.cpp

    # transformer
    transformer/transformer.cpp
)

target_link_libraries(neural
    PUBLIC # Important: Make OpenCL and CUDA linking public
        ${OpenCL_LIBRARIES}
        ${CUDAToolkit_LIBRARIES}
    maths
)
