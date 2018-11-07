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

    const Map<const MatrixXf> *M = new Map<const MatrixXf>(train, dim, n_points);
    Map<MatrixXf> *test_queries = new Map<MatrixXf>(test, dim, n_test);

    if(!parallel) omp_set_num_threads(1);
    std::cerr << "parallel: " << parallel << "\n\n\n";
    int seed_mrpt = 12345;

    std::vector<int> ks{1, 10, 100};
    double build_time;

    for (int j = 0; j < ks.size(); ++j) {
      int k = ks[j];
      double build_start = omp_get_wtime();
      Mrpt at(M);
      at.grow(test_queries, k, trees_max, depth_min, depth_max, votes_max, density, seed_mrpt);
      double build_end = omp_get_wtime();

      std::vector<Parameters> pars = at.optimal_parameter_list();
      for(const auto &par : pars) {

        Mrpt index2(M);
        at.subset_trees(par.estimated_recall, index2);


        if(index2.is_empty()) {
          continue;
        }

        std::vector<double> times, projection_times, voting_times, exact_times;
        std::vector<std::set<int>> idx;
        int elected = 0;

        for (int i = 0; i < n_test; ++i) {
          double projection_time = 0.0, voting_time = 0.0, exact_time = 0.0;
          std::vector<int> result(k);
          std::vector<float> distances(k);
          int n_elected = 0;
          Map<VectorXf> q(&test[i * dim], dim);

          double start = omp_get_wtime();
          index2.query(q, &result[0], projection_time, voting_time, exact_time, &distances[0], &n_elected);
          double end = omp_get_wtime();

          times.push_back(end - start);
          idx.push_back(std::set<int>(result.begin(), result.begin() + k));
          projection_times.push_back(projection_time);
          voting_times.push_back(voting_time);
          exact_times.push_back(exact_time);
          elected += n_elected;
        }

        double mean_projection_time = mean(projection_times);
        double mean_voting_time = mean(voting_times);
        double mean_exact_time = mean(exact_times);
        double est_projection_time = at.get_projection_time(par.n_trees, par.depth, par.votes);
        double est_voting_time = at.get_voting_time(par.n_trees, par.depth, par.votes);
        double est_exact_time = at.get_exact_time(par.n_trees, par.depth, par.votes);
        double mean_n_elected = elected / static_cast<double>(n_test);

        if(verbose)
          std::cout << "k: " << k << ", # of trees: " << index2.get_n_trees() << ", depth: " << index2.get_depth() << ", density: " << density << ", votes: " << index2.get_votes() << "\n";
        else
          std::cout << k << " " << index2.get_n_trees() << " " << index2.get_depth() << " " << density << " " << index2.get_votes() << " ";

        results(k, times, idx, ("results/mnist/truth_" + std::to_string(k)).c_str(), verbose);
        std::cout << build_end - build_start <<  " ";
        std::cout << par.estimated_recall << " ";
        std::cout << par.estimated_qtime * n_test << " ";
        std::cout << est_projection_time * n_test << " ";
        std::cout << est_voting_time * n_test << " ";
        std::cout << est_exact_time * n_test << " ";
        std::cout << (mean_projection_time + mean_voting_time + mean_exact_time) * n_test << " ";
        std::cout << mean_projection_time * n_test << " ";
        std::cout << mean_voting_time * n_test << " ";
        std::cout << mean_exact_time * n_test << " ";
        std::cout << mean_n_elected << " ";
        std::cout << std::endl;

      }

    }



    delete[] test;
    if(!mmap) delete[] train;

    return 0;
}