cmake_minimum_required(VERSION 3.10)
project(YourProjectName VERSION 1.0)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type" FORCE)
endif()
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -Wall -Wextra -Wpedantic")

find_package(Eigen3 3.3 REQUIRED)
find_package(nlohmann_json 3.10.5 REQUIRED)

set(DEAL_II_DIR "/Users/aidansgarlato/dealii-candi/deal.II-v9.6.1")
find_package(deal.II 9.5.0 REQUIRED HINTS ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR})

DEAL_II_INITIALIZE_CACHED_VARIABLES()

set(GMSH_INCLUDE_DIR "/opt/homebrew/include")
set(GMSH_LIBRARY "/opt/homebrew/lib/libgmsh.dylib")

add_executable(Fusion main.cpp)
deal_ii_setup_target(Fusion)

target_include_directories(Fusion PRIVATE
    ${GMSH_INCLUDE_DIR}
    "/Users/aidansgarlato/dealii-candi/deal.II-v9.6.1/include"
)

file(GLOB DEAL_II_LIBS "/Users/aidansgarlato/dealii-candi/deal.II-v9.6.1/lib/*.dylib")

target_link_libraries(Fusion
    ${DEAL_II_LIBS}
    ${GMSH_LIBRARY}
    Eigen3::Eigen
    nlohmann_json::nlohmann_json
)
