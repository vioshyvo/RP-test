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

class MrptTest {
public:
  static double get_projection_time(const Mrpt &mrpt, int n_trees, int depth, int v) {
    return mrpt.get_projection_time(n_trees, depth, v);
  }

  static double get_voting_time(const Mrpt &mrpt, int n_trees, int depth, int v) {
    return mrpt.get_voting_time(n_trees, depth, v);
  }

  static  double get_exact_time(const Mrpt &mrpt, int n_trees, int depth, int v) {
    return mrpt.get_exact_time(n_trees, depth, v);
  }


};

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
    const Map<const MatrixXf> test_queries(test, dim, n_test);

    if(!parallel) omp_set_num_threads(1);

    std::cerr << std::endl;
    std::cerr << "parallel: " << parallel << std::endl;
    std::cerr << "trees_max: " << trees_max << std::endl;
    std::cerr << "depth_min: " << depth_min << std::endl;
    std::cerr << "depth_max: " << depth_max << std::endl;
    std::cerr << "votes_max: " << votes_max << std::endl;
    std::cerr << "density: " << density << std::endl;
    std::cerr << std::endl;

    int seed_mrpt = 12345;

    std::vector<int> ks{1, 10, 100};
    double build_time;

    for (int j = 0; j < ks.size(); ++j) {
      int k = ks[j];
      double build_start = omp_get_wtime();
      Mrpt mrpt(M);
      mrpt.grow_autotune(k, trees_max, depth_max, depth_min, votes_max, density, seed_mrpt);

      double build_end = omp_get_wtime();

      std::vector<Mrpt_Parameters> pars = mrpt.optimal_parameters();
      for(const auto &par : pars) {
        Mrpt mrpt_new = mrpt.subset(par.estimated_recall);

        if(mrpt_new.empty()) {
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
          const Map<const VectorXf> q(&test[i * dim], dim);

          double start = omp_get_wtime();
          mrpt_new.query(q, &result[0], projection_time, voting_time, exact_time, &distances[0], &n_elected);
          double end = omp_get_wtime();

          times.push_back(end - start);
          idx.push_back(std::set<int>(result.begin(), result.begin() + k));
          projection_times.push_back(projection_time);
          voting_times.push_back(voting_time);
          exact_times.push_back(exact_time);
          elected += n_elected;
        }

        double median_projection_time = median(projection_times);
        double median_voting_time = median(voting_times);
        double median_exact_time = median(exact_times);
        double est_projection_time = MrptTest::get_projection_time(mrpt, par.n_trees, par.depth, par.votes);
        double est_voting_time = MrptTest::get_voting_time(mrpt, par.n_trees, par.depth, par.votes);
        double est_exact_time = MrptTest::get_exact_time(mrpt, par.n_trees, par.depth, par.votes);
        double mean_n_elected = elected / static_cast<double>(n_test);

        if(verbose)
          std::cout << "k: " << k << ", # of trees: " << par.n_trees << ", depth: " << par.depth << ", density: " << density << ", votes: " << par.votes << "\n";
        else
          std::cout << k << " " << par.n_trees << " " << par.depth << " " << density << " " << par.votes << " ";

        results(k, times, idx, ("results/mnist/truth_" + std::to_string(k)).c_str(), verbose);
        std::cout << build_end - build_start <<  " ";
        std::cout << par.estimated_recall << " ";
        std::cout << par.estimated_qtime * n_test << " ";
        std::cout << est_projection_time * n_test << " ";
        std::cout << est_voting_time * n_test << " ";
        std::cout << est_exact_time * n_test << " ";
        std::cout << (median_projection_time + median_voting_time + median_exact_time) * n_test << " ";
        std::cout << median_projection_time * n_test << " ";
        std::cout << median_voting_time * n_test << " ";
        std::cout << median_exact_time * n_test << " ";
        std::cout << mean_n_elected << " ";
        std::cout << std::endl;

      }

    }



    delete[] test;
    if(!mmap) delete[] train;

    return 0;
}
