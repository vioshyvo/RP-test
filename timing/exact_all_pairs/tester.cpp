#include <vector>
#include <cstdio>
#include <stdint.h>
#include <omp.h>
#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Eigen/Core>
#include <typeinfo>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>


#include "Mrpt2.h"
#include "common.h"


using namespace Eigen;

int main(int argc, char **argv) {
  if (argc != 7) {
      std::cerr << "Usage: " << argv[0] << " n n_test k dim mmap data_path" << std::endl;
      return 1;
  }

  int n = atoi(argv[1]);
  int n_test = atoi(argv[2]);
  int k = atoi(argv[3]);
  int dim = atoi(argv[4]);
  int mmap = atoi(argv[5]);
  std::string data_path(argv[6]);
  if (!data_path.empty() && data_path.back() != '/')
    data_path += '/';

  int n_train = n - n_test;

  ////////////////////////////////////////////////////////////////////////////////////////////////
  // read the train and test data

  float *train, *test;

  test = read_memory((data_path + "test.bin").c_str(), n_test, dim);
  if(!test) {
      std::cerr << "in tester : test data " << data_path + "test.bin" << " could not be read\n";
      return -1;
  }

  if(mmap) {
      train = read_mmap((data_path + "train.bin").c_str(), n_train, dim);
  } else {
      train = read_memory((data_path + "train.bin").c_str(), n_train, dim);
  }

  if(!test) {
      std::cerr << "in tester : training data " << data_path + "train.bin" << " could not be read\n";
      return -1;
  }

  //////////////////////////////////////////////////////////////////////////////////////////////
  // compute the exact k-nn

  double start = omp_get_wtime();
  Eigen::MatrixXi true_knn = Mrpt::exact_all_pairs(train, n_train, dim, k);
  double end = omp_get_wtime();
  std::cout << "All pairs " << k  << "-nn for " << n_train << " points took "
            << (end - start) / 60 << " min." << std::endl << std::endl;

  write_memory(true_knn.data(), data_path + "exact_all_pairs_" + std::to_string(k) + ".bin", k, n_train);

  Eigen::MatrixXi test_knn(k, n_test);
  start = omp_get_wtime();
  for (int i = 0; i < n_test; ++i)
      Mrpt::exact_knn(test + (i * dim), train, dim, n_train, k, test_knn.data() + (i * k));
  end = omp_get_wtime();
  std::cout << "Exact " << k << "-nn for " << n_test << " test queries took "
            << (end - start) << " s." << std::endl << std::endl;

  write_memory(test_knn.data(), data_path + "exact_test_" + std::to_string(k) + ".bin", k, n_test);

  return 0;
}
