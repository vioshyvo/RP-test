# Tests for MRPT

This repo contains simple tests for [MRPT algorithm](https://github.com/teemupitkanen/mrpt) to aid the development. Basically, we test that when the random seed given for the function which grows the trees is fixed, the query results are always the same. So if anything related to the random vectors of trees is changed, the tests break with a high probability.

To start testing, first clone [googletest](https://github.com/google/googletest.git) project if you do not already have it:
```
git clone https://github.com/google/googletest.git
```
Then edit the directory containing MRPT code, your preferred C++ compiler and a googletest directory (if you cloned googletest into the root of this directory, then the default directory works) into `test/Makefile`, for example:
```
MRPT_DIR = ../../mrpt
CXX = g++-8
GTEST_DIR = ../googletest/googletest
```

Then make and run the tests:
```
cd test
make
./test_rp1
```

If you want to run only a subset of the tests, you can use the flag `--gtest_filter`, for instance:
```
./test_rp1 --gtest_filter=*.Query
```
