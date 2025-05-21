cmake_minimum_required(VERSION 3.10)
project(YourProjectName VERSION 1.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Eigen3 3.3 REQUIRED)
find_package(nlohmann_json 3.10.5 REQUIRED)

set(GMSH_INCLUDE_DIR "/opt/homebrew/include")
set(GMSH_LIBRARY "/opt/homebrew/lib/libgmsh.dylib")

add_executable(Fusion main.cpp)

target_include_directories(Fusion PRIVATE ${GMSH_INCLUDE_DIR})
target_link_libraries(Fusion PRIVATE
    Eigen3::Eigen
    nlohmann_json::nlohmann_json
    ${GMSH_LIBRARY}
)
