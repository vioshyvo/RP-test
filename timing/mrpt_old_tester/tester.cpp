#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <typeinfo>

#include <vector>
#include <cstdio>
#include <stdint.h>
#include <omp.h>

#include <algorithm>
#include <functional>
#include <numeric>
#include <queue>
#include <string>
#include <memory>

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include "Mrpt_old.h"
#include "common.h"




using namespace Eigen;

int main(int argc, char **argv) {
    size_t n = atoi(argv[1]);
    size_t ntest = atoi(argv[2]);
    int k = atoi(argv[3]);
    int n_trees = atoi(argv[4]);
    int depth = atoi(argv[5]);
    size_t dim = atoi(argv[6]);
    int mmap = atoi(argv[7]);
    std::string result_path(argv[8]);
    if (!result_path.empty() && result_path.back() != '/')
      result_path += '/';

    std::string infile_path(argv[9]);
    if (!infile_path.empty() && infile_path.back() != '/')
      infile_path += '/';

    float sparsity = atof(argv[10]);
    bool parallel = atoi(argv[11]);

    int last_arg = 11;
    size_t n_points = n - ntest;
    bool verbose = false;


    /////////////////////////////////////////////////////////////////////////////////////////
    // test mrpt
    float *train, *test;

    test = read_memory((infile_path + "test.bin").c_str(), ntest, dim);
    if(!test) {
        std::cerr << "in mrpt_comparison: test data " << infile_path + "test.bin" << " could not be read\n";
        return -1;
    }

    if(mmap) {
        train = read_mmap((infile_path + "train.bin").c_str(), n_points, dim);
    } else {
        train = read_memory((infile_path + "train.bin").c_str(), n_points, dim);
    }

    if(!train) {
        std::cerr << "in mrpt_comparison: training data " << infile_path + "train.bin" << " could not be read\n";
        return -1;
    }


    const Map<const MatrixXf> *M = new Map<const MatrixXf>(train, dim, n_points);

    if(!parallel) omp_set_num_threads(1); 
    double build_start = omp_get_wtime();
    Mrpt_old index_dense(M, n_trees, depth, sparsity);
    index_dense.grow();
    double build_time = omp_get_wtime() - build_start;

    std::vector<int> ks{1, 10, 100};
    for (int j = 0; j < ks.size(); ++j) {
      int k = ks[j];
      for (int arg = last_arg + 1; arg < argc; ++arg) {
          int votes = atoi(argv[arg]);
          if (votes > n_trees) continue;

          std::vector<double> times;
          std::vector<std::set<int>> idx;

          for (int i = 0; i < ntest; ++i) {
                  std::vector<int> result(k);
                  double start = omp_get_wtime();
                  index_dense.query(Map<VectorXf>(&test[i * dim], dim), k, votes, &result[0]);

                  double end = omp_get_wtime();
                  times.push_back(end - start);
                  idx.push_back(std::set<int>(result.begin(), result.begin() + k)); // k_found (<= k) is the number of k-nn canditates returned
              }

          if(verbose)
              std::cout << "k: " << k << ", # of trees: " << n_trees << ", depth: " << depth << ", sparsity: " << sparsity << ", votes: " << votes << "\n";
          else
              std::cout << k << " " << n_trees << " " << depth << " " << sparsity << " " << votes << " ";

          results(k, times, idx, (result_path + "truth_" + std::to_string(k)).c_str(), verbose);
          std::cout << build_time << endl;
      }

    }
    delete[] test;
    if(!mmap) delete[] train;


    return 0;
}
