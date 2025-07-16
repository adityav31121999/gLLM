
# src/memorymap/CMakeLists.txt
cmake_minimum_required(VERSION 3.30.0 FATAL_ERROR)
project(memorymap C)

# This command only affects the current target's compilation
# include_directories(include)

add_library(memorymap STATIC
    # basic files
    memory_map.cpp
)

# Make the include directory available to this target and any
# other target that links to it (like the main executable).
target_include_directories(memorymap PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

# Make the 'include' directory public so targets linking against memorymap can find its headers
target_include_directories(memorymap PUBLIC include)
