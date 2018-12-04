CXX=g++
EIGEN_PATH=../../../mrpt/cpp/lib
MRPT_PATH=../../../mrpt/cpp
INCLUDE_PATH=../../include

CXXFLAGS=-O3 -march=native -fno-rtti -fno-stack-protector -ffast-math -DNDEBUG -fopenmp

all: tester

tester.o : tester.cpp $(INCLUDE_PATH)/common.h $(MRPT_PATH)/Mrpt.h
	$(CXX) -I$(EIGEN_PATH) -I$(MRPT_PATH) -I$(INCLUDE_PATH) $(CXXFLAGS) -c tester.cpp

tester: tester.o
	$(CXX) $(CXXFLAGS) $^ -o $@

.PHONY: clean
clean:
	$(RM) tester *.o
