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
    std::string results_file(argv[12]);

    int last_arg = 12;
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

    std::ofstream outf(results_file, std::ios::app);
    if(!outf) {
      std::cerr << results_file << " could not be opened for writing." << std::endl;
      return -1;
    }

    if(!parallel) omp_set_num_threads(1);

    double build_start = omp_get_wtime();
    Mrpt index_dense(train, dim, n_points);
    index_dense.grow(n_trees, depth, sparsity);
    double build_time = omp_get_wtime() - build_start;
    std::vector<int> ks{1, 10, 100};

    std::vector<double> times, projection_times, voting_times, exact_times,
                        sorting_times, choosing_times;

    for (int j = 0; j < ks.size(); ++j) {
      int k = ks[j];
      for (int arg = last_arg + 1; arg < argc; ++arg) {
        int cs_size = atoi(argv[arg]);
        int max_leaf_size = n_points / (1 << depth) + 1;
        if (cs_size > n_points || cs_size > n_trees * max_leaf_size) continue;

        double projection_time = 0.0, voting_time = 0.0, exact_time = 0.0,
               sorting_time = 0.0, choosing_time = 0.0;

        std::vector<double> times;
        std::vector<std::set<int>> idx;

        for (int i = 0; i < ntest; ++i) {
          std::vector<int> result(k, -1);
          const Map<const VectorXf> q(&test[i * dim], dim);

          double start = omp_get_wtime();
          index_dense.query_size2(q, k, cs_size, &result[0],
                                  projection_time, voting_time, exact_time,
                                  sorting_time, choosing_time);
          double end = omp_get_wtime();

          times.push_back(end - start);
          idx.push_back(std::set<int>(result.begin(), result.begin() + k));
          projection_times.push_back(projection_time);
          voting_times.push_back(voting_time);
          exact_times.push_back(exact_time);
          sorting_times.push_back(sorting_time);
          choosing_times.push_back(choosing_time);
        }

        if(verbose)
            outf << "k: " << k << ", # of trees: " << n_trees << ", depth: " << depth << ", sparsity: " << sparsity << ", votes: " << 0 << "\n";
        else
            outf << k << " " << n_trees << " " << depth << " " << sparsity << " " << cs_size << " ";

        results(k, times, idx, (result_path + "truth_" + std::to_string(k)).c_str(), verbose, outf);
        outf << median(projection_times) << " ";
        outf << median(voting_times) << " ";
        outf << median(exact_times) << " ";
        outf << build_time << " ";
        outf << median(sorting_times) << " ";
        outf << median(choosing_times) << " ";
        outf << std::endl;
      }
    }


    delete[] test;
    if(!mmap) delete[] train;

    return 0;
}
