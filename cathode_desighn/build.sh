#!/bin/bash
set -e

echo "Building Debug version..."
rm -rf build-debug
cmake -B build-debug -DCMAKE_BUILD_TYPE=Debug .
cmake --build build-debug --parallel 4

echo "Building Release version..."
rm -rf build-release
cmake -B build-release -DCMAKE_BUILD_TYPE=Release .
cmake --build build-release --parallel 4

echo "Build complete!"
echo "Debug:   ./build-debug/Debug/Fusion"
echo "Release: ./build-release/Release/Fusion"
