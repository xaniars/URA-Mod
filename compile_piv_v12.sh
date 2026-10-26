#!/bin/bash
# Needs OpenCV 4 and oneTBB development packages (Ubuntu/Debian: libopencv-dev libtbb-dev pkg-config)
g++ -std=c++17 -O2 piv_movie_v12.cpp $(pkg-config --cflags --libs opencv4) -ltbb -o piv_v12
