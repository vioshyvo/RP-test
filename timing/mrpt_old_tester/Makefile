CXX=g++-8
EIGEN_PATH=../../mrpt_old/cpp/lib
MRPT_PATH=../../mrpt_old/cpp
INCLUDE_PATH=../../include

CXXFLAGS=-O3 -march=native -fno-rtti -fno-stack-protector -fno-exceptions -ffast-math -DNDEBUG -fopenmp

all: tester

tester.o : tester.cpp $(INCLUDE_PATH)/common.h $(MRPT_PATH)/Mrpt_old.h
	$(CXX) -I$(EIGEN_PATH) -I$(MRPT_PATH) -I$(INCLUDE_PATH) $(CXXFLAGS) -c tester.cpp

tester: tester.o
	$(CXX) $(CXXFLAGS) $^ -o $@

.PHONY: clean
clean:
	$(RM) tester *.o
