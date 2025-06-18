#!/bin/bash
set -e

echo "Building Debug version..."
cmake -B build-debug -DCMAKE_BUILD_TYPE=Debug .
cmake --build build-debug --parallel 8

echo "Building Release version..."
cmake -B build-release -DCMAKE_BUILD_TYPE=Release .
cmake --build build-release --parallel 8

echo "Build complete!"
echo "Debug:   ./build-debug/Debug/Fusion"
echo "Release: ./build-release/Release/Fusion"
