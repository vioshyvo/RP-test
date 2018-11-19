CXX=g++-8
EIGEN_PATH=../../../mrpt/cpp/lib
MRPT_PATH=../timing_tester
INCLUDE_PATH=../../include
HAYAI_PATH=hayai/src

CXXFLAGS=-O3 -march=native -fno-rtti -fno-stack-protector -ffast-math -DNDEBUG -fopenmp

all: test

test.o : test.cpp $(INCLUDE_PATH)/common.h $(MRPT_PATH)/Mrpt.h
	$(CXX) -I$(EIGEN_PATH) -I$(MRPT_PATH) -I$(INCLUDE_PATH) -I$(HAYAI_PATH) $(CXXFLAGS) -c test.cpp

test: test.o
	$(CXX) $(CXXFLAGS) $^ -o $@

.PHONY: clean
clean:
	$(RM) test *.o
