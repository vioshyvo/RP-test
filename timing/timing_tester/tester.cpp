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

std::vector<std::vector<int>> read_results(std::string truth, int k) {
  std::ifstream fs(truth);
  if (!fs) {
     std::cerr << "File " << truth << " could not be opened for reading!" << std::endl;
     exit(1);
  }

  double time;
  std::vector<std::vector<int>> correct;
  while(fs >> time) {
      std::vector<int> res;
      for (int i = 0; i < k; ++i) {
          int r;
          fs >> r;
          res.push_back(r);
      }
      correct.push_back(res);
  }
  return correct;
}


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
    int n_auto = atoi(argv[14]);

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

    int seed_mrpt = 12345;

    std::vector<int> ks{1, 10, 100};
    std::vector<double> target_recalls {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9,
                                        0.925, 0.95, 0.97, 0.98, 0.99, 0.995};

    double build_time;

    int big_k = *ks.rbegin();
    std::string votes_file(result_path + "votes_" + std::to_string(big_k));
    std::string top_votes_file(result_path + "top_votes_" + std::to_string(big_k));
    std::ofstream ofvotes(votes_file), oftop(top_votes_file);
    if (!ofvotes) {
       std::cerr << "File " << votes_file << " could not be opened for reading!" << std::endl;
       exit(1);
    }
    if (!oftop) {
       std::cerr << "File " << top_votes_file << " could not be opened for reading!" << std::endl;
       exit(1);
    }

    std::string result_file(result_path + "truth_" + std::to_string(big_k));
    std::vector<std::vector<int>> correct = read_results(result_file, big_k);

    for (int j = 0; j < ks.size(); ++j) {
      int k = ks[j];
      double build_start = omp_get_wtime();
      Mrpt mrpt(M);
      mrpt.grow_autotune(k, trees_max, depth_max, depth_min, votes_max, density, seed_mrpt, n_auto);

      double build_end = omp_get_wtime();

      for(const auto &tr : target_recalls) {
        Mrpt mrpt_new = mrpt.subset(tr);
        Mrpt_Parameters par(mrpt_new.parameters());

        if(mrpt_new.empty()) {
          continue;
        }

        std::vector<double> times, projection_times, voting_times, exact_times;
        std::vector<std::set<int>> idx;
        int elected = 0;

        if(k == big_k) {
          oftop << par.k << " " << par.n_trees << " " << par.depth << " " << par.votes << std::endl;
          ofvotes << par.k << " " << par.n_trees << " " << par.depth << " " << par.votes << std::endl;
        }

        for (int i = 0; i < n_test; ++i) {
          double projection_time = 0.0, voting_time = 0.0, exact_time = 0.0;
          std::vector<int> result(k);
          std::vector<float> distances(k);
          int n_elected = 0;
          const Map<const VectorXf> q(&test[i * dim], dim);

          double start = omp_get_wtime();
          Eigen::VectorXi votes = Eigen::VectorXi::Zero(n_points);
          mrpt_new.query(q, &result[0], projection_time, voting_time, exact_time,
                         votes, &distances[0], &n_elected);
          double end = omp_get_wtime();

          times.push_back(end - start);
          idx.push_back(std::set<int>(result.begin(), result.begin() + k));
          projection_times.push_back(projection_time);
          voting_times.push_back(voting_time);
          exact_times.push_back(exact_time);
          elected += n_elected;

          if(k == big_k) {
            const std::vector<int> &exact = correct[i];
            for(const auto &e : exact)
              ofvotes << votes[e] << " ";
            ofvotes << std::endl;

            std::partial_sort(votes.data(), votes.data() + k, votes.data() + n_points, std::greater<int>());
            for(int l = 0; l < k; ++l)
              oftop << votes(l) << " ";
            oftop << std::endl;
          }
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

        results(k, times, idx, (result_path + "truth_" + std::to_string(k)).c_str(), verbose);
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
