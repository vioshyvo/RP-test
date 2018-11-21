# Tests for MRPT
<center>
![Fifty shades of green](voting-candidates2.png)

</center>


This repo contains tests and [timing code](timing/README.md) for [MRPT algorithm](https://github.com/teemupitkanen/mrpt).

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
./test
./test_implementation
```

If you want to run only a subset of the tests, you can use the flag `--gtest_filter`, for instance:
```
./test --gtest_filter=UtilityTest.*
```
File `test.cpp` contains the most useful tests. File `test_implementation.cpp` contains tests that depend on the implementation of the trees, especially on the generation of random vectors. If the tree structure is changed, the tests in this file will break.
