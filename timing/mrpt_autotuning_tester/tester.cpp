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
    size_t n_test = atoi(argv[2]);
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

    size_t n_points = n - n_test;
    bool verbose = false;

    /////////////////////////////////////////////////////////////////////////////////////////
    // test mrpt
    float *train, *test;

    test = read_memory((infile_path + "test.bin").c_str(), n_test, dim);
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


    const Map<const MatrixXf> M(train, dim, n_points);
    Map<MatrixXf> *test_queries = new Map<MatrixXf>(test, dim, n_test);

    if(!parallel) omp_set_num_threads(1);
    int seed_mrpt = 12345;

    std::vector<int> ks{1, 10, 100};
    double build_time;

    for (int j = 0; j < ks.size(); ++j) {
      int k = ks[j];

      double build_start = omp_get_wtime();
      Mrpt mrpt(M);
      // mrpt.grow(test_queries, k, trees_max, depth_max, depth_min, votes_max, density, seed_mrpt);
      mrpt.grow(test_queries, k);
      double build_end = omp_get_wtime();

      std::vector<Mrpt_Parameters> pars = mrpt.optimal_pars();
      for(const auto &par : pars) {

        Mrpt mrpt_new(mrpt.subset(par.estimated_recall));

        if(mrpt_new.empty()) {
          continue;
        }

        std::vector<double> times;
        std::vector<std::set<int>> idx;

        for (int i = 0; i < n_test; ++i) {
          std::vector<int> result(k);
          Map<VectorXf> q(&test[i * dim], dim);

          double start = omp_get_wtime();
          mrpt_new.query(q, &result[0]);
          double end = omp_get_wtime();

          times.push_back(end - start);
          idx.push_back(std::set<int>(result.begin(), result.begin() + k));
        }

        if(verbose)
          std::cout << "k: " << k << ", # of trees: " << par.n_trees << ", depth: " << par.depth << ", density: " << density << ", votes: " << par.votes << "\n";
        else
          std::cout << k << " " << par.n_trees << " " << par.depth << " " << density << " " << par.votes << " ";

        results(k, times, idx, (result_path + "truth_" + std::to_string(k)).c_str(), verbose);
        std::cout << build_end - build_start << std::endl;

      }

    }



    delete[] test;
    if(!mmap) delete[] train;
    delete test_queries;

    return 0;
}
