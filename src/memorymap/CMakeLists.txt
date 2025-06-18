
# src/memorymap/CMakeLists.txt
cmake_minimum_required(VERSION 3.25.0 FATAL_ERROR)
project(memorymap C)

# This command only affects the current target's compilation
# include_directories(include)

add_library(memorymap STATIC
    # basic files
    memory_map.cpp
)

# Make the 'include' directory public so targets linking against memorymap can find its headers
target_include_directories(memorymap PUBLIC include)
