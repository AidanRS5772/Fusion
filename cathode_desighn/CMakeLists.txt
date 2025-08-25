cmake_minimum_required(VERSION 3.26)
project(Fusion VERSION 1.0)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Detect Apple Silicon
if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    execute_process(COMMAND uname -m OUTPUT_VARIABLE ARCH OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(ARCH STREQUAL "arm64")
        message(STATUS "Building for Apple Silicon (M1/M2)")
        set(CMAKE_OSX_ARCHITECTURES "arm64")
    endif()
endif()

# Enhanced debug/release flags
set(CMAKE_CXX_FLAGS_DEBUG "-g3 -O0 -DDEBUG -fno-omit-frame-pointer -fno-inline \
    -Wall -Wextra -Wpedantic \
    -fstack-protector-strong")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -mtune=native -ffast-math -funroll-loops -fvectorize -fslp-vectorize -flto=thin -w")

# Enable dSYM generation for Debug builds on macOS
if(APPLE AND CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_XCODE_ATTRIBUTE_DEBUG_INFORMATION_FORMAT "dwarf-with-dsym")
    set(CMAKE_XCODE_ATTRIBUTE_DWARF_DSYM_FILE_SHOULD_ACCOMPANY_PRODUCT YES)
endif()

# Find packages
find_package(Eigen3 3.3 REQUIRED)
find_package(nlohmann_json 3.10.5 REQUIRED)
find_package(dlib REQUIRED)

set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/candi/dealii" ${CMAKE_PREFIX_PATH})
find_package(deal.II 9.7.0 REQUIRED)
DEAL_II_INITIALIZE_CACHED_VARIABLES()

# GMSH configuration
set(GMSH_INCLUDE_DIR "/opt/homebrew/include")
set(GMSH_LIBRARY "/opt/homebrew/lib/libgmsh.dylib")

find_library(LAPACKE_LIB lapacke PATHS
    /opt/homebrew/opt/lapack/lib
    /opt/homebrew/opt/openblas/lib
)

find_library(PROFILER_LIB profiler
    PATHS /opt/homebrew/lib
    NO_DEFAULT_PATH
)

# Single executable target
add_executable(Fusion src/main.cpp)

# Set output directory based on build type
set_target_properties(Fusion PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}"
)

# Set properties based on build type
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(Fusion PRIVATE DEBUG PROJECT_ROOT="${CMAKE_SOURCE_DIR}")
else()
    target_compile_definitions(Fusion PRIVATE
        EIGEN_NO_DEBUG
        EIGEN_FAST_MATH=1
        EIGEN_USE_BLAS
        PROJECT_ROOT="${CMAKE_SOURCE_DIR}"
    )
    set_property(TARGET Fusion PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)

    # M1-specific optimizations for release
    if(ARCH STREQUAL "arm64")
        target_compile_options(Fusion PRIVATE
            -mcpu=apple-m1
            -falign-functions=16
            -falign-loops=16
        )
    endif()
endif()

# Deal.II setup
deal_ii_setup_target(Fusion)

# Additional include directories
target_include_directories(Fusion PRIVATE
    ${GMSH_INCLUDE_DIR}
)

# Link libraries
target_link_libraries(Fusion
    ${GMSH_LIBRARY}
    ${PROFILER_LIB}
    ${LAPACKE_LIB}
    Eigen3::Eigen
    nlohmann_json::nlohmann_json
    dlib::dlib
)

# Link with Accelerate framework on macOS
if(APPLE)
    find_library(ACCELERATE_FRAMEWORK Accelerate)
    if(ACCELERATE_FRAMEWORK)
        target_link_libraries(Fusion ${ACCELERATE_FRAMEWORK})
    endif()
endif()

# Generate dSYM for debug builds
if(APPLE AND CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_custom_command(TARGET Fusion POST_BUILD
        COMMAND dsymutil $<TARGET_FILE:Fusion> -o $<TARGET_FILE_DIR:Fusion>/Fusion.dSYM
        COMMENT "Generating dSYM bundle for Fusion"
    )
endif()

message(STATUS "")
message(STATUS "Build configuration:")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Output: ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}/")
message(STATUS "")
message(STATUS "To build different configurations:")
message(STATUS "  Debug:   cmake -B build-debug -DCMAKE_BUILD_TYPE=Debug . && cmake --build build-debug")
message(STATUS "  Release: cmake -B build-release -DCMAKE_BUILD_TYPE=Release . && cmake --build build-release")
