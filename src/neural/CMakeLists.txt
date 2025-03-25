
# src/neural/CMakeLists.txt
cmake_minimum_required(VERSION 3.30.0 FATAL_ERROR)
project(neural C CXX)

include_directories(include)

add_library(neural STATIC
    # mlp files
    src/mlp/mlp.cpp
    src/mlp/forprop.cpp
    src/mlp/backprop.cpp
    src/mlp/loss.cpp
    src/mlp/weights.cpp
    src/mlp/train.cpp

    src/mlp/cl/forprop.cl

    # src/mlp/cu/forprop.cu

    # attention
    src/attention/attention.cpp
    src/attention/forward.cpp
    src/attention/backward.cpp
    src/attention/train.cpp

    # block
    src/block/block.cpp
    src/block/forward.cpp
    src/block/backward.cpp
    src/block/train.cpp

    # transformer
    src/transformer/tokens.cpp
    src/transformer/transformer.cpp
    src/transformer/feedfor.cpp
    src/transformer/backward.cpp
    src/transformer/train.cpp
    src/transformer/finetune.cpp
)

target_link_libraries(neural
    PUBLIC # Important: Make OpenCL and CUDA linking public
        ${OpenCL_LIBRARIES}
        # ${CUDAToolkit_LIBRARIES}
    maths
)
