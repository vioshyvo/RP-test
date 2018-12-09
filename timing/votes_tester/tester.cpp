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

#include "Mrpt.h"
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

    if(!parallel) omp_set_num_threads(1);

    double build_start = omp_get_wtime();
    Mrpt index_dense(train, dim, n_points);
    index_dense.grow(n_trees, depth, sparsity);
    double build_time = omp_get_wtime() - build_start;
    std::vector<int> ks{1, 10, 100};

    // std::vector<double> target_recalls {0.5, 0.6, 0.7, 0.8, 0.85, 0.9,
    //                                     0.925, 0.95, 0.97, 0.98, 0.99, 0.995};
    std::vector<int> cs_sizes {20, 30, 40, 50, 60, 70, 80, 90, 100, 125, 150, 175, 200,
                               250, 300, 350, 400, 500};
    std::vector<double> times, projection_times, voting_times, exact_times, sorting_times;

    for (int j = 0; j < ks.size(); ++j) {
      int k = ks[j];
      for (const auto &cs_size : cs_sizes) {
        double projection_time = 0.0, voting_time = 0.0, exact_time = 0.0, sorting_time = 0.0;

        std::vector<double> times;
        std::vector<std::set<int>> idx;

        for (int i = 0; i < ntest; ++i) {
          std::vector<int> result(k, -1);
          const Map<const VectorXf> q(&test[i * dim], dim);

          double start = omp_get_wtime();
          index_dense.query_size(q, k, cs_size, &result[0],
                                 projection_time, voting_time, exact_time, sorting_time);
          double end = omp_get_wtime();

          times.push_back(end - start - sorting_time);
          idx.push_back(std::set<int>(result.begin(), result.begin() + k));
          projection_times.push_back(projection_time);
          voting_times.push_back(voting_time);
          exact_times.push_back(exact_time);
          sorting_times.push_back(sorting_time);
        }

        if(verbose)
            std::cout << "k: " << k << ", # of trees: " << n_trees << ", depth: " << depth << ", sparsity: " << sparsity << ", votes: " << 0 << "\n";
        else
            std::cout << k << " " << n_trees << " " << depth << " " << sparsity << " " << cs_size << " ";

        results(k, times, idx, (result_path + "truth_" + std::to_string(k)).c_str(), verbose);
        std::cout << median(projection_times) << " ";
        std::cout << median(voting_times) << " ";
        std::cout << median(exact_times) << " ";
        std::cout << build_time << " ";
        std::cout << median(sorting_times) << " ";
        std::cout << std::endl;
      }
    }


    delete[] test;
    if(!mmap) delete[] train;

    return 0;
}
