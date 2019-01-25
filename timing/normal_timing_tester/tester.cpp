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
// #include <random>
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
    std::string results_file2(results_file + "_adaptive");

    int last_arg = 12;
    size_t n_points = n - ntest;
    bool verbose = false;

    std::vector<double> target_recalls {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.87,
                                        0.90, 0.92, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0};

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
    if(!outf2) {
      std::cerr << results_file2 << " could not be opened for writing." << std::endl;
      return -1;
    }

    if(!parallel) omp_set_num_threads(1);

    double build_start = omp_get_wtime();
    Mrpt index_dense(train, dim, n_points);
    index_dense.grow(n_trees, depth, sparsity);
    double build_time = omp_get_wtime() - build_start;
    std::vector<int> ks{1, 10, 100};

    for (const auto &k : ks) {
      std::string votes_file(result_path + "votes_" + std::to_string(k));
      std::string top_votes_file(result_path + "top_votes_" + std::to_string(k));
      std::string cs_sizes_file(result_path + "cs_sizes_" + std::to_string(k));
      std::string vote_thresholds_file(result_path + "vote_thresholds_" + std::to_string(k));
      std::ofstream ofvotes(votes_file), oftop(top_votes_file);
      std::ofstream ofsizes(cs_sizes_file), ofthresholds(vote_thresholds_file);
      if (!ofvotes) {
         std::cerr << "File " << votes_file << " could not be opened for reading!" << std::endl;
         exit(1);
      }
      if (!oftop) {
         std::cerr << "File " << top_votes_file << " could not be opened for reading!" << std::endl;
         exit(1);
      }
      if(!ofsizes) {
        std::cerr << "File " << cs_sizes_file << " could not be opened for reading!" << std::endl;
        exit(1);
      }
      if(!ofthresholds) {
        std::cerr << "File " << vote_thresholds_file << " could not be opened for reading!" << std::endl;
        exit(1);
      }

      std::string result_file(result_path + "truth_" + std::to_string(k));
      std::vector<std::vector<int>> correct = read_results(result_file, k);

      std::vector<std::vector<int>> all_vote_counts;
      std::vector<std::vector<int>> all_nn_found;
      std::vector<std::vector<int>> all_top_votes;

      oftop << k << " " << n_trees << " " << depth << " " << std::endl;
      ofvotes << k << " " << n_trees << " " << depth << " " << std::endl;

      for (int arg = last_arg + 1; arg < argc; ++arg) {
        int vote_threshold = atoi(argv[arg]);
        if (vote_threshold > n_trees) continue;

        std::vector<double> times;
        std::vector<std::set<int>> idx;

        for (int i = 0; i < ntest; ++i) {
          double projection_time = 0.0, voting_time = 0.0, exact_time = 0.0;
          std::vector<int> result(k);
          std::vector<float> distances(k);
          int n_elected = 0;
          const Map<const VectorXf> q(&test[i * dim], dim);

          double start = omp_get_wtime();
          Eigen::VectorXi votes = Eigen::VectorXi::Zero(n_points);
          index_dense.query(q, k, vote_threshold, &result[0], projection_time, voting_time,
                            exact_time, votes, &distances[0], &n_elected);

          double end = omp_get_wtime();
          times.push_back(end - start);
          idx.push_back(std::set<int>(result.begin(), result.begin() + k));

          std::map<int,int,std::greater<int>> vote_counts;

          if(arg == last_arg + 1) {
            const std::vector<int> &exact = correct[i];
            for(const auto &e : exact)
              ofvotes << votes[e] << " ";
            ofvotes << n_elected << std::endl;

            for(int l = 0; l < votes.size(); ++l) {
              int v = votes(l);
              if(v)
                if(std::find(exact.begin(), exact.end(), l) != exact.end())
                  ++vote_counts[v];
            }

            auto vec_pair = map2vec(vote_counts);
            all_vote_counts.push_back(vec_pair.first);
            all_nn_found.push_back(vec_pair.second);
            for(const auto &v : vec_pair.first)
              ofthresholds << v << " ";
            ofthresholds << std::endl;

            for(const auto &v : vec_pair.second)
              ofsizes << v << " ";
            ofsizes << std::endl;

            std::vector<int> tvotes;
            std::partial_sort(votes.data(), votes.data() + k, votes.data() + n_points, std::greater<int>());
            for(int l = 0; l < k; ++l) {
              oftop << votes(l) << " ";
              tvotes.push_back(votes(l));
            }
            oftop << n_elected << std::endl;
            all_top_votes.push_back(tvotes);
          }
        }

        outf << k << " " << n_trees << " " << depth << " " << sparsity << " " << vote_threshold << " ";
        results(k, times, idx, (result_path + "truth_" + std::to_string(k)).c_str(), verbose, outf);
        outf << build_time << std::endl;
      }

      for(int j = 0; j < target_recalls.size(); ++j) {
        double tr = target_recalls[j];
        std::vector<double> vote_thresholds;
        std::vector<double> times;
        std::vector<std::set<int>> idx;
        int elected = 0;

        for(int i = 0; i < ntest; ++i) {
          double projection_time = 0.0, voting_time = 0.0, exact_time = 0.0;
          std::vector<int> result(k);
          std::vector<float> distances(k);
          int n_elected = 0;
          const Map<const VectorXf> q(&test[i * dim], dim);

          const std::vector<int> &vote_counts = all_vote_counts[i];
          const std::vector<int> &nn_found = all_nn_found[i];
          const std::vector<int> &tvotes = all_top_votes[i];
          int vote_threshold = get_vote_threshold_probability(tr, k, vote_counts, nn_found, tvotes);
          vote_thresholds.push_back(vote_threshold);
        
          double start = omp_get_wtime();
          Eigen::VectorXi votes = Eigen::VectorXi::Zero(n_points);
          index_dense.query(q, k, vote_threshold, &result[0], projection_time, voting_time,
                            exact_time, votes, &distances[0], &n_elected);
          double end = omp_get_wtime();

          times.push_back(end - start);
          idx.push_back(std::set<int>(result.begin(), result.begin() + k));
        }

        outf2 << k << " " << n_trees << " " << depth << " " << sparsity << " " << median(vote_thresholds) << " ";
        results(k, times, idx, (result_path + "truth_" + std::to_string(k)).c_str(), verbose, outf2);
        outf2 << build_time << std::endl;
      }
    }


    delete[] test;
    if(!mmap) delete[] train;

    return 0;
}
