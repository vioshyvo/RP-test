#!/bin/bash
g++ -O3 -std=c++11 -Wall -I../../eigen_3.3.3 -I../../mrpt mrpt_comparison.cpp -o ../bin/mrpt_comparison -lgomp
