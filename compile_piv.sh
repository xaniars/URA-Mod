#!/bin/bash
set -e

CXX="${CXX:-clang++}"
"$CXX" -std=c++11 piv_movie_v11.cpp $(pkg-config --cflags --libs opencv4 tbb) -o piv_movie_v11
