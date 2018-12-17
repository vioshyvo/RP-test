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
    std::string parameters_filename(argv[13]);

    int last_arg = 13;
    size_t n_points = n - ntest;
    bool verbose = false;
    std::string results_file2(results_file + "_size");
    std::string results_file3(results_file + "_size2");
    std::string results_file4(results_file + "_size3");


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

    std::ofstream outf2(results_file2, std::ios::app);
    if(!outf) {
      std::cerr << results_file2 << " could not be opened for writing." << std::endl;
      return -1;
    }

    std::ifstream inf(parameters_filename);

    if(!inf) {
      std::cerr << parameters_filename << " could not be opened for reading";
      return -1;
    }

    // assumes that all vote thresholds are on one line!
    std::vector<int> votes = read_parameters("MRPT_VOTES", inf);

    if(!parallel) omp_set_num_threads(1);

    double build_start = omp_get_wtime();
    Mrpt index_dense(train, dim, n_points);
    index_dense.grow(n_trees, depth, sparsity);
    double build_time = omp_get_wtime() - build_start;
    std::vector<int> ks{1, 10, 100};

    for (int j = 0; j < ks.size(); ++j) {
      int k = ks[j];

      for(const auto &v : votes) {
        if(v > n_trees) {
          continue;
        }

        std::vector<double> times, projection_times, voting_times, exact_times,
                            cs_sizes;
        std::vector<std::set<int>> idx;

        for (int i = 0; i < ntest; ++i) {
          double projection_time = 0.0, voting_time = 0.0, exact_time = 0.0;
          int n_elected = 0;

          std::vector<int> result(k, -1);
          std::vector<float> distances(k, -1.0);
          const Map<const VectorXf> q(&test[i * dim], dim);

          double start = omp_get_wtime();
          Eigen::VectorXi votes_vec = Eigen::VectorXi::Zero(n_points);
          index_dense.query(q, k, v, &result[0],
                            projection_time, voting_time, exact_time,
                            votes_vec, &distances[0], &n_elected);
          double end = omp_get_wtime();

          times.push_back(end - start);
          idx.push_back(std::set<int>(result.begin(), result.begin() + k));
          projection_times.push_back(projection_time);
          voting_times.push_back(voting_time);
          exact_times.push_back(exact_time);
          cs_sizes.push_back(n_elected);
        }

        outf << k << " " << n_trees << " " << depth << " " << sparsity << " " << v << " ";
        results(k, times, idx, (result_path + "truth_" + std::to_string(k)).c_str(), verbose, outf);

        outf << median(projection_times) << " ";
        outf << median(voting_times) << " ";
        outf << median(exact_times) << " ";
        outf << build_time << " ";
        outf << median(cs_sizes) << " ";
        outf << std::endl;
      }

      for (int arg = last_arg + 1; arg < argc; ++arg) {
        int cs_size = atoi(argv[arg]);
        int max_leaf_size = n_points / (1 << depth) + 1;
        if (cs_size > n_points || cs_size > n_trees * max_leaf_size) continue;

        std::vector<double> times2, projection_times2, voting_times2, exact_times2,
                            sorting_times2, choosing_times2;

        std::vector<double> cs_sizes2;

        std::vector<std::set<int>> idx2;

        for (int i = 0; i < ntest; ++i) {
          double projection_time = 0.0, voting_time = 0.0, exact_time = 0.0,
                 sorting_time = 0.0, choosing_time;

          int n_elected = 0;

          std::vector<int> result(k, -1);
          std::vector<float> distances(k, -1.0);
          const Map<const VectorXf> q(&test[i * dim], dim);

          double start = omp_get_wtime();
          index_dense.query_size3(q, k, cs_size, &result[0],
                                  projection_time, voting_time, exact_time,
                                  sorting_time, choosing_time, &distances[0],
                                  &n_elected);
          double end = omp_get_wtime();

          times2.push_back(end - start);
          idx2.push_back(std::set<int>(result.begin(), result.begin() + k));

          projection_times2.push_back(projection_time);
          voting_times2.push_back(voting_time);
          exact_times2.push_back(exact_time);
          cs_sizes2.push_back(n_elected);
          sorting_times2.push_back(sorting_time);
          choosing_times2.push_back(choosing_time);
        }

        outf2 << k << " " << n_trees << " " << depth << " " << sparsity << " " << cs_size << " ";
        results(k, times2, idx2, (result_path + "truth_" + std::to_string(k)).c_str(), verbose, outf2);

        outf2 << median(projection_times2) << " ";
        outf2 << median(voting_times2) << " ";
        outf2 << median(exact_times2) << " ";
        outf2 << build_time << " ";
        outf2 << median(cs_sizes2) << " ";
        outf2 << median(sorting_times2) << " ";
        outf2 << median(choosing_times2) << " ";
        outf2 << std::endl;
      }
    }


    delete[] test;
    if(!mmap) delete[] train;

    return 0;
}
