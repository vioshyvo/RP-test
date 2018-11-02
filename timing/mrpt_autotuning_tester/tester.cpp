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
    // for(auto &s : std::vector<char*>(argv, argv + argc))
    //   std::cout << s << '\n';

    size_t n = atoi(argv[1]);
    size_t ntest = atoi(argv[2]);
    int k = atoi(argv[3]);
    int trees_max = atoi(argv[4]);
    int depth_min = atoi(argv[5]);
    int depth_max = atoi(argv[6]);
    int votes_max = atoi(argv[7]);

    size_t dim = atoi(argv[8]);
    int mmap = atoi(argv[9]);
    std::string result_path(argv[10]);
    if (!result_path.empty() && result_path.back() != '/')
      result_path += '/';

    std::string infile_path(argv[11]);
    if (!infile_path.empty() && infile_path.back() != '/')
      infile_path += '/';

    float density = atof(argv[12]);
    bool parallel = atoi(argv[13]);

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
    Map<MatrixXf> *test_queries = new Map<MatrixXf>(test, dim, ntest);

    if(!parallel) omp_set_num_threads(1);
    int seed_mrpt = 12345;

    std::vector<int> ks{1, 10, 100};
    std::vector<double> build_times;

    Mrpt index(M);
    index.grow(trees_max, depth_max, density, seed_mrpt);

    for (int j = 0; j < ks.size(); ++j) {
      int k = ks[j];

      double build_start = omp_get_wtime();
      Autotuning at(M, test_queries);
      at.tune(trees_max, depth_min, depth_max, votes_max, density, k, seed_mrpt);
      double build_time = omp_get_wtime() - build_start;
      build_times.push_back(build_time);

      bool add = j ? true : false;
      at.write_results((result_path + "mrpt_auto_write").c_str(), add);

      std::vector<int> target_recalls(99);
      std::iota(target_recalls.begin(), target_recalls.end(), 0);

      for(const auto &tr : target_recalls) {
        Parameters par = at.get_optimal_parameters(tr);
        if(!par.n_trees) {
          continue;
        }

        std::vector<double> times;
        std::vector<std::set<int>> idx;

        for(int i = 0; i < ntest; ++i) {
          std::vector<int> result(k);

          double start = omp_get_wtime();
          at.query(Map<VectorXf>(&test[i * dim], dim), tr, &result[0], index);
          double end = omp_get_wtime();

          times.push_back(end - start);
          idx.push_back(std::set<int>(result.begin(), result.begin() + k));
        }

        if(verbose)
            std::cout << "k: " << k << ", # of trees: " << par.n_trees << ", depth: " << par.depth << ", density: " << density << ", votes: " << par.votes << "\n";
        else
            std::cout << k << " " << par.n_trees << " " << par.depth << " " << density << " " << par.votes << " ";

        results(k, times, idx, (result_path + "truth_" + std::to_string(k)).c_str(), verbose);
        std::cout << build_time << endl;

      }
    }

    delete[] test;
    if(!mmap) delete[] train;

    return 0;
}
