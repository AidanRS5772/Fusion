cmake_minimum_required(VERSION 3.10)
project(YourProjectName VERSION 1.0)
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

# Enhanced debug flags for better debugging with dSYM support
set(CMAKE_CXX_FLAGS_DEBUG "-g3 -O0 -DDEBUG \
    -fno-omit-frame-pointer -fno-inline -fno-optimize-sibling-calls \
    -fno-builtin -ggdb -fstandalone-debug \
    -Wall -Wextra -Wpedantic \
    -fsanitize=undefined -fstack-protector-strong")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -mtune=native -ffast-math -funroll-loops -fvectorize -fslp-vectorize -flto=thin -w")

# Split flags into lists for proper handling
separate_arguments(CMAKE_CXX_FLAGS_DEBUG_LIST UNIX_COMMAND "${CMAKE_CXX_FLAGS_DEBUG}")
separate_arguments(CMAKE_CXX_FLAGS_RELEASE_LIST UNIX_COMMAND "${CMAKE_CXX_FLAGS_RELEASE}")

# Enable dSYM generation for Debug builds on macOS
if(APPLE)
    set(CMAKE_XCODE_ATTRIBUTE_DEBUG_INFORMATION_FORMAT "dwarf-with-dsym")
    set(CMAKE_XCODE_ATTRIBUTE_DWARF_DSYM_FILE_SHOULD_ACCOMPANY_PRODUCT YES)
endif()

# Find packages
find_package(Eigen3 3.3 REQUIRED)
find_package(nlohmann_json 3.10.5 REQUIRED)
find_package(MPI REQUIRED)

set(DEAL_II_DIR "/Users/aidansgarlato/dealii-candi/deal.II-v9.6.1")
find_package(deal.II 9.5.0 REQUIRED HINTS ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR})
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

# Function to create executable with specific configuration
function(create_fusion_target TARGET_NAME BUILD_TYPE OUTPUT_DIR)
    add_executable(${TARGET_NAME} src/main.cpp)

    # Set build type specific properties
    if(BUILD_TYPE STREQUAL "Debug")
        target_compile_options(${TARGET_NAME} PRIVATE ${CMAKE_CXX_FLAGS_DEBUG_LIST})
        target_compile_definitions(${TARGET_NAME} PRIVATE DEBUG PROJECT_ROOT="${CMAKE_SOURCE_DIR}")

        # macOS-specific dSYM generation for debug builds
        if(APPLE)
            set_target_properties(${TARGET_NAME} PROPERTIES
                XCODE_ATTRIBUTE_DEBUG_INFORMATION_FORMAT "dwarf-with-dsym"
                XCODE_ATTRIBUTE_DWARF_DSYM_FILE_SHOULD_ACCOMPANY_PRODUCT YES
                XCODE_ATTRIBUTE_DWARF_DSYM_FOLDER_PATH "${CMAKE_BINARY_DIR}/${OUTPUT_DIR}"
            )

            # Add custom command to generate dSYM for non-Xcode generators
            if(NOT CMAKE_GENERATOR STREQUAL "Xcode")
                add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
                    COMMAND dsymutil $<TARGET_FILE:${TARGET_NAME}> -o ${CMAKE_BINARY_DIR}/${OUTPUT_DIR}/${TARGET_NAME}.dSYM
                    COMMENT "Generating dSYM file for ${TARGET_NAME}"
                )
            endif()
        endif()
    else()
        target_compile_options(${TARGET_NAME} PRIVATE ${CMAKE_CXX_FLAGS_RELEASE_LIST})
        target_compile_definitions(${TARGET_NAME} PRIVATE
            EIGEN_NO_DEBUG
            EIGEN_FAST_MATH=1
            EIGEN_USE_BLAS
            EIGEN_USE_LAPACKE
            PROJECT_ROOT="${CMAKE_SOURCE_DIR}"
        )
        set_property(TARGET ${TARGET_NAME} PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    endif()

    # M1-specific optimizations for release builds only
    if(ARCH STREQUAL "arm64" AND BUILD_TYPE STREQUAL "Release")
        target_compile_options(${TARGET_NAME} PRIVATE
            -mcpu=apple-m1
            -falign-functions=16
            -falign-loops=16
        )
    endif()

    # Set output directory
    set_target_properties(${TARGET_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${OUTPUT_DIR}"
    )

    # Deal.II setup
    deal_ii_setup_target(${TARGET_NAME})

    # Include directories
    target_include_directories(${TARGET_NAME} PRIVATE
        ${GMSH_INCLUDE_DIR}
        "/Users/aidansgarlato/dealii-candi/deal.II-v9.6.1/include"
        ${MPI_CXX_INCLUDE_DIRS}
    )

    # Libraries
    file(GLOB DEAL_II_LIBS "/Users/aidansgarlato/dealii-candi/deal.II-v9.6.1/lib/*.dylib")
    target_link_libraries(${TARGET_NAME}
        ${DEAL_II_LIBS}
        ${GMSH_LIBRARY}
        ${PROFILER_LIB}
        ${LAPACKE_LIB}
        Eigen3::Eigen
        nlohmann_json::nlohmann_json
        MPI::MPI_CXX
    )

    # Link with Accelerate framework on macOS
    if(APPLE)
        find_library(ACCELERATE_FRAMEWORK Accelerate)
        if(ACCELERATE_FRAMEWORK)
            target_link_libraries(${TARGET_NAME} ${ACCELERATE_FRAMEWORK})
        endif()
    endif()
endfunction()

# Create both debug and release targets
create_fusion_target(Fusion_Debug "Debug" "debug")
create_fusion_target(Fusion_Release "Release" "release")

# Custom targets for convenience
add_custom_target(all_builds)
add_dependencies(all_builds Fusion_Debug Fusion_Release)

add_custom_target(debug)
add_dependencies(debug Fusion_Debug)

add_custom_target(release)
add_dependencies(release Fusion_Release)

# Default target
add_custom_target(fusion)
add_dependencies(fusion Fusion_Release)

# Default build to create both versions
add_custom_target(default ALL)
add_dependencies(default all_builds)

message(STATUS "Targets created:")
message(STATUS "  make            - Build both debug and release (default)")
message(STATUS "  make all_builds - Build both debug and release")
message(STATUS "  make debug      - Build debug version only")
message(STATUS "  make release    - Build release version only")
message(STATUS "  make fusion     - Build release version (default)")
message(STATUS "")
message(STATUS "Output directories:")
message(STATUS "  Debug:   ${CMAKE_BINARY_DIR}/debug/")
message(STATUS "  Release: ${CMAKE_BINARY_DIR}/release/")
if(APPLE)
    message(STATUS "")
    message(STATUS "dSYM files will be generated for debug builds")
endif()
