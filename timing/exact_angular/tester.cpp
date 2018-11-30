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


#include "Mrpt.h"
#include "common.h"


using namespace Eigen;

int main(int argc, char **argv) {
  if (argc != 7) {
      std::cerr << "Usage: " << argv[0] << " n ntest k dim mmap data_path" << std::endl;
      return 1;
  }

  int n = atoi(argv[1]);
  int ntest = atoi(argv[2]);
  int k = atoi(argv[3]);
  int dim = atoi(argv[4]);
  int mmap = atoi(argv[5]);
  std::string data_path(argv[6]);
  if (!data_path.empty() && data_path.back() != '/')
    data_path += '/';

  int n_points = n - ntest;

  ////////////////////////////////////////////////////////////////////////////////////////////////
  // read the train and test data

  float *train, *test;

  test = read_memory((data_path + "test.bin").c_str(), ntest, dim);
  if(!test) {
      std::cerr << "in tester : test data " << data_path + "test.bin" << " could not be read\n";
      return -1;
  }

  if(mmap) {
      train = read_mmap((data_path + "train.bin").c_str(), n_points, dim);
  } else {
      train = read_memory((data_path + "train.bin").c_str(), n_points, dim);
  }

  if(!test) {
      std::cerr << "in tester : training data " << data_path + "train.bin" << " could not be read\n";
      return -1;
  }


    //////////////////////////////////////////////////////////////////////////////////////////////
    // compute the exact k-nn

    // build dummy index
    const Map<const MatrixXf> M(train, dim, n_points);
    MatrixXf M_norm(dim, n_points);
    for(int i = 0; i < n_points; ++i)
      M_norm.col(i) = M.col(i).normalized();

    Mrpt index(M_norm);

    VectorXi idx(n_points);
    std::iota(idx.data(), idx.data() + n_points, 0);

    omp_set_num_threads(1);
    for (int i = 0; i < ntest; ++i) {
        std::vector<int> result(k);
        double start = omp_get_wtime();
        VectorXf q_norm = Map<VectorXf>(&test[i * dim], dim).normalized();
        index.exact_knn(q_norm, k, &result[0]);
        double end = omp_get_wtime();
        printf("%g\n", end - start);
        for (int i = 0; i < k; ++i) printf("%d ", result[i]);
        printf("\n");
    }

    return 0;
}
