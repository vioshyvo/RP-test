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

    std::ofstream outf3(results_file3, std::ios::app);
    if(!outf) {
      std::cerr << results_file3 << " could not be opened for writing." << std::endl;
      return -1;
    }

    std::ofstream outf4(results_file4, std::ios::app);
    if(!outf) {
      std::cerr << results_file4 << " could not be opened for writing." << std::endl;
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
                            sorting_times2;

        std::vector<double> times3, projection_times3, voting_times3, exact_times3,
                            sorting_times3, choosing_times3;

        std::vector<double> times4, projection_times4, voting_times4, exact_times4,
                            sorting_times4, choosing_times4;

        std::vector<double> cs_sizes2, cs_sizes3, cs_sizes4;

        std::vector<std::set<int>> idx2, idx3, idx4;

        for (int i = 0; i < ntest; ++i) {
          double projection_time2 = 0.0, voting_time2 = 0.0, exact_time2 = 0.0,
                 sorting_time2 = 0.0;

          double projection_time3 = 0.0, voting_time3 = 0.0, exact_time3 = 0.0,
                 sorting_time3 = 0.0, choosing_time3 = 0.0;

          double projection_time4 = 0.0, voting_time4 = 0.0, exact_time4 = 0.0,
                 sorting_time4 = 0.0, choosing_time4 = 0.0;

          int n_elected2 = 0, n_elected3 = 0, n_elected4 = 0;

          std::vector<int> result2(k, -1), result3(k, -1), result4(k, -1);
          std::vector<float> distances(k, -1.0);
          const Map<const VectorXf> q2(&test[i * dim], dim);

          double start = omp_get_wtime();
          index_dense.query_size(q2, k, cs_size, &result2[0],
                                  projection_time2, voting_time2, exact_time2,
                                  sorting_time2, &distances[0], &n_elected2);
          double end = omp_get_wtime();

          times2.push_back(end - start);
          idx2.push_back(std::set<int>(result2.begin(), result2.begin() + k));
          projection_times2.push_back(projection_time2);
          voting_times2.push_back(voting_time2);
          exact_times2.push_back(exact_time2);
          cs_sizes2.push_back(n_elected2);
          sorting_times2.push_back(sorting_time2);

          start = omp_get_wtime();
          index_dense.query_size2(q2, k, cs_size, &result3[0],
                                  projection_time3, voting_time3, exact_time3,
                                  sorting_time3, choosing_time3,
                                  &distances[0], &n_elected3);
          end = omp_get_wtime();

          times3.push_back(end - start);
          idx3.push_back(std::set<int>(result3.begin(), result3.begin() + k));
          projection_times3.push_back(projection_time3);
          voting_times3.push_back(voting_time3);
          exact_times3.push_back(exact_time3);
          cs_sizes3.push_back(n_elected3);
          sorting_times3.push_back(sorting_time3);
          choosing_times3.push_back(choosing_time3);

          start = omp_get_wtime();
          index_dense.query_size3(q2, k, cs_size, &result4[0],
                                  projection_time4, voting_time4, exact_time4,
                                  sorting_time4, choosing_time4,
                                  &distances[0], &n_elected4);
          end = omp_get_wtime();

          times4.push_back(end - start);
          idx4.push_back(std::set<int>(result4.begin(), result4.begin() + k));
          projection_times4.push_back(projection_time4);
          voting_times4.push_back(voting_time4);
          exact_times4.push_back(exact_time4);
          cs_sizes4.push_back(n_elected4);
          sorting_times4.push_back(sorting_time4);
          choosing_times4.push_back(choosing_time4);
        }

        outf2 << k << " " << n_trees << " " << depth << " " << sparsity << " " << cs_size << " ";
        outf3 << k << " " << n_trees << " " << depth << " " << sparsity << " " << cs_size << " ";
        outf4 << k << " " << n_trees << " " << depth << " " << sparsity << " " << cs_size << " ";

        results(k, times2, idx2, (result_path + "truth_" + std::to_string(k)).c_str(), verbose, outf2);
        results(k, times3, idx3, (result_path + "truth_" + std::to_string(k)).c_str(), verbose, outf3);
        results(k, times4, idx4, (result_path + "truth_" + std::to_string(k)).c_str(), verbose, outf4);

        outf2 << median(projection_times2) << " ";
        outf2 << median(voting_times2) << " ";
        outf2 << median(exact_times2) << " ";
        outf2 << build_time << " ";
        outf2 << median(cs_sizes2) << " ";
        outf2 << median(sorting_times2) << " ";
        outf2 << std::endl;

        outf3 << median(projection_times3) << " ";
        outf3 << median(voting_times3) << " ";
        outf3 << median(exact_times3) << " ";
        outf3 << build_time << " ";
        outf3 << median(cs_sizes3) << " ";
        outf3 << median(sorting_times3) << " ";
        outf3 << median(choosing_times3) << " ";
        outf3 << std::endl;

        outf4 << median(projection_times4) << " ";
        outf4 << median(voting_times4) << " ";
        outf4 << median(exact_times4) << " ";
        outf4 << build_time << " ";
        outf4 << median(cs_sizes4) << " ";
        outf4 << median(sorting_times4) << " ";
        outf4 << median(choosing_times4) << " ";
        outf4 << std::endl;
      }
    }


    delete[] test;
    if(!mmap) delete[] train;

    return 0;
}
