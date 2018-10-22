CXX=g++-8
EIGEN_PATH=../../../mrpt/cpp/lib
MRPT_PATH=../../../mrpt/cpp

CXXFLAGS=-O3 -march=native -fno-rtti -fno-stack-protector -fno-exceptions -ffast-math -DNDEBUG -DEIGEN_DONT_PARALLELIZE -fopenmp

all: tester

tester.o : tester.cpp ../common.h $(MRPT_PATH)/Mrpt.h
	$(CXX) -I$(EIGEN_PATH) -I$(MRPT_PATH) -I.. $(CXXFLAGS) -c tester.cpp

tester: tester.o
	$(CXX) $(CXXFLAGS) $^ -o $@

.PHONY: clean
clean:
	$(RM) tester *.o
