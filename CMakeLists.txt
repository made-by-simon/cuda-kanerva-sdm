cmake_minimum_required(VERSION 3.18)
project(KanervaSDM_CUDA LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# Set CUDA architectures (adjust for your GPU).
# RTX 20 series: 75
# RTX 30 series: 86
# RTX 40 series: 89
# RTX 50 series: 120
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90 120)

# Add executable.
add_executable(kanerva_sdm_cuda main.cu)

# Include directories.
target_include_directories(kanerva_sdm_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Optional: Add optimization flags.
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(kanerva_sdm_cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
        $<$<COMPILE_LANGUAGE:CXX>:-O3>
    )
endif()
