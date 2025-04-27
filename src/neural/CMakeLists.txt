
# src/neural/CMakeLists.txt
cmake_minimum_required(VERSION 3.30.0 FATAL_ERROR)
project(neural C CXX)

include_directories(include)

add_library(neural STATIC
# mlp files
    # C++
    src/mlp/mlp.cpp
    src/mlp/forprop.cpp
    src/mlp/backprop.cpp
    src/mlp/cu/backward.cu
    src/mlp/cu/backwithL1.cu
    src/mlp/cu/backwithL2.cu
    src/mlp/loss.cpp
    src/mlp/weights.cpp
    src/mlp/train.cpp
    # CUDA
    src/mlp/cu/kernel.cu
    src/mlp/cu/loss.cu
    src/mlp/cu/forprop.cu
    src/mlp/cu/backprop.cu
    src/mlp/cu/backward.cu
    src/mlp/cu/backwithL1.cu
    src/mlp/cu/backwithL2.cu
    src/mlp/cu/rprop.cu
    src/mlp/cu/train.cu
    # OpenCL
    src/mlp/cl/kernel.cl
    src/mlp/cl/losscl.cpp
    src/mlp/cl/forpropcl.cpp
    src/mlp/cl/backpropcl.cpp
    src/mlp/cl/backwardcl.cpp
    src/mlp/cl/backwithL1cl.cpp
    src/mlp/cl/backwithL2cl.cpp
    src/mlp/cl/rpropcl.cpp
    src/mlp/cl/traincl.cpp

# attention files
    # C++
    src/attention/attention.cpp
    src/attention/forward.cpp
    src/attention/backward.cpp
    src/attention/backward1sthead.cpp
    # CUDA
    src/attention/cu/forward.cu
    src/attention/cu/backward.cu
    src/attention/cu/backward1sthead.cu
    src/attention/cu/kernels.cu
    # OpenCL
    src/attention/cl/kernels.cl
    src/attention/cl/forwardcl.cpp
    src/attention/cl/backward1stheadcl.cpp
    src/attention/cl/backwardcl.cpp

# block
    # C++
    src/block/block.cpp
    src/block/forward.cpp
    src/block/partialback.cpp
    src/block/backward.cpp
    # CUDA
    src/block/cu/kdotq.cu
    src/block/cu/forward.cu
    src/block/cu/partialback.cu
    src/block/cu/backward.cu
    # OpenCL
    src/block/cl/kdotqcl.cpp
    src/block/cl/forwardcl.cpp
    src/block/cl/partialbackcl.cpp
    src/block/cl/backwardcl.cpp

# transformer
    # C++
    src/transformer/compute.cpp
    src/transformer/tokens.cpp
    src/transformer/transformer.cpp
    src/transformer/feedfor.cpp
    src/transformer/backward.cpp
    src/transformer/train.cpp
    src/transformer/run.cpp
    src/transformer/getVal.cpp
    # CUDA
    src/transformer/cu/kernels.cu
    src/transformer/cu/cucompute.cu
    src/transformer/cu/feedfor.cu
    src/transformer/cu/backward.cu
    src/transformer/cu/train.cu
    src/transformer/cu/run.cu
    # OpenCL
    src/transformer/cl/kernels.cl
    src/transformer/cl/clcompute.cpp
    src/transformer/cl/feedforcl.cpp
    src/transformer/cl/backwardcl.cpp
    src/transformer/cl/traincl.cpp
    src/transformer/cl/runcl.cpp
)

# link libraries to this library
target_link_libraries(neural
    PUBLIC # Important: Make OpenCL and CUDA linking public
        ${OpenCL_LIBRARIES}
        ${CUDAToolkit_LIBRARIES}
    maths
)
