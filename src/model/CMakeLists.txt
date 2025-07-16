
# src/model/CMakeLists.txt
cmake_minimum_required(VERSION 3.30.0 FATAL_ERROR)
project(model C CXX)

# include_directories(include)

add_library(model STATIC
    src/tokens/tokeniser.cpp
    src/tokens/set.cpp
    src/tokens/clHost.cpp
    src/tokens/embedding.cpp
    src/tokens/split.cpp
    src/tokens/corpus.cpp
    src/tokens/token.cu
    src/tokens/token.cl
    src/tokens/group.cpp
    src/tokens/pairstats.cpp
    src/tokens/merge.cpp
    src/tokens/readFiles.cpp

    src/model.cpp
    src/create.cpp
    src/fetch.cpp
    src/set.cpp
    src/csvsup.cpp
    src/ser.cpp
    src/token.cpp

    src/training/trainpr.cpp
    src/training/trainsent.cpp
    src/training/testpr.cpp
    src/training/testsent.cpp
    src/inference/chat.cpp
    src/inference/run.cpp
)

# Make the include directory available to this target and any
# other target that links to it (like the main executable).
target_include_directories(model PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(model
    PUBLIC # Important: Make OpenCL and CUDA linking public
        ${OpenCL_LIBRARIES}
        ${CUDAToolkit_LIBRARIES}
    memorymap
    maths
    neural
)
