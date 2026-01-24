# src/model/CMakeLists.txt
cmake_minimum_required(VERSION 3.30.0 FATAL_ERROR)
project(neural C CXX)
# all .hpp and .h files
include_directories(include)
# all source files
add_library(neural STATIC
# mlp
    src/mlp/mlp.cpp
    src/mlp/weights.cpp
    src/mlp/writemlp.cpp
    # cpp
        src/mlp/cpp/forprop.cpp
        src/mlp/cpp/loss.cpp
        src/mlp/cpp/backprop.cpp
        src/mlp/cpp/train.cpp
    # cuda
        src/mlp/cu/kernel.cu
        src/mlp/cu/forprop.cu
        src/mlp/cu/loss.cu
        src/mlp/cu/backprop.cu
        src/mlp/cu/backward.cu
        src/mlp/cu/backwithL1.cu
        src/mlp/cu/backwithL2.cu
        src/mlp/cu/backwithElasticNet.cu
        src/mlp/cu/rprop.cu
        src/mlp/cu/train.cu
    # opencl
        src/mlp/cl/mlp.cl
        src/mlp/cl/forpropcl.cpp
        src/mlp/cl/gradClipcl.cpp
        src/mlp/cl/backpropcl.cpp
        src/mlp/cl/backwardcl.cpp
        src/mlp/cl/backwithL1cl.cpp
        src/mlp/cl/backwithL2cl.cpp
        src/mlp/cl/backwithElasticNetcl.cpp
        src/mlp/cl/rpropcl.cpp
        src/mlp/cl/traincl.cpp

# attention
    src/attention/attention.cpp
    src/attention/writeatt.cpp
    # cpp
        src/attention/cpp/kdotq.cpp
        src/attention/cpp/forward.cpp
        src/attention/cpp/backward1H.cpp
        src/attention/cpp/backward.cpp
    # cuda
        src/attention/cu/attention.cu
        src/attention/cu/kdotq.cu
        src/attention/cu/weights.cu
    # opencl
        src/attention/cl/attention.cl
        src/attention/cl/kdotq.cl
        src/attention/cl/weights.cl

# block
    src/block/block.cpp
    src/block/writeblock.cpp
    # cpp
        src/block/cpp/forwardblock.cpp
        src/block/cpp/partialback.cpp
        src/block/cpp/backward.cpp
    # cuda
        src/block/cu/forward.cu
        src/block/cu/partialbackH.cu
        src/block/cu/partialbackiH.cu
        src/block/cu/partialbackiHr.cu
        src/block/cu/backward.cu
    # opencl
        src/block/cl/forwardcl.cpp
        src/block/cl/forwardicl.cpp
        src/block/cl/partialbackHcl.cpp
        src/block/cl/partialbackiHcl.cpp
        src/block/cl/partialbackiHrcl.cpp
        src/block/cl/backwardcl.cpp

# transformer
    src/transformer/transformer.cpp
    src/transformer/getVal.cpp
    src/transformer/setVal.cpp
    src/transformer/tokens.cpp
    # cpp
        src/transformer/cpp/compute.cpp
        src/transformer/cpp/feedfor.cpp
        src/transformer/cpp/backward.cpp
        src/transformer/cpp/trainf.cpp
        src/transformer/cpp/train1.cpp
        src/transformer/cpp/train2.cpp
        src/transformer/cpp/train3.cpp
        src/transformer/cpp/test.cpp
        src/transformer/cpp/run.cpp
    # cuda
        src/transformer/cu/compute.cu
        src/transformer/cu/feedfor.cu
        src/transformer/cu/backward.cu
        src/transformer/cu/trainf.cu
        src/transformer/cu/train1.cu
        src/transformer/cu/train2.cu
        src/transformer/cu/train3.cu
        src/transformer/cu/train4.cu
        src/transformer/cu/test.cu
        src/transformer/cu/run.cu
    # opencl
        src/transformer/cl/clcompute.cpp
        src/transformer/cl/feedforcl.cpp
        src/transformer/cl/backwardcl.cpp
        src/transformer/cl/trainfcl.cpp
        src/transformer/cl/train1cl.cpp
        src/transformer/cl/train2cl.cpp
        src/transformer/cl/train3cl.cpp
        src/transformer/cl/train4cl.cpp
        src/transformer/cl/testcl.cpp
        src/transformer/cl/runcl.cpp
)

target_link_libraries(neural
    PUBLIC # Important: Make OpenCL and CUDA linking public
        ${OpenCL_LIBRARIES}
        ${CUDAToolkit_LIBRARIES}
    memorymap
    maths
)