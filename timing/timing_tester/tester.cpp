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
#include <stdexcept>

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

int get_vote_threshold(int target_nn, const std::vector<int> &vote_thresholds,
                       const std::vector<int> &nn_found) {
  if(vote_thresholds.size() != nn_found.size()) {
    throw std::logic_error("vote_thresholds.size and nn_found.size are different.");
  }

  int v = 1;
  for(int i = 0; i < nn_found.size(); ++i) {
    if(nn_found[i] >= target_nn) {
      v = vote_thresholds[i];
      break;
    }
  }
  return v;
}

int get_vote_threshold_probability(double target_nn, int k, const std::vector<int> &vote_thresholds,
                       const std::vector<int> &inn_found) {
  if(vote_thresholds.size() != inn_found.size()) {
    throw std::logic_error("vote_thresholds.size and nn_found.size are different.");
  }

  std::vector<double> nn_found;
  for(int i = 0; i < vote_thresholds.size(); ++i)
    nn_found.push_back(inn_found[i] / static_cast<double>(k));

  std::random_device rd;
  std::mt19937 gen(rd());

  int v = 1;
  for(int i = 0; i < nn_found.size(); ++i) {
    if(nn_found[i] >= target_nn) {
      v = vote_thresholds[i];
      if(nn_found[i] > target_nn && i != 0) {
        double interval = nn_found[i] - nn_found[i-1];
        double top = nn_found[i] - target_nn;
        double prob = top / interval;
        std::bernoulli_distribution dist(prob);
        if(dist(gen)) {
          v = vote_thresholds[i-1];
        }
      }
      break;
    }
  }
  return v;
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
    std::string results_file(argv[15]);
    std::string results_file2(results_file + "_adaptive");

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

    const Map<const MatrixXf> M(train, dim, n_points);
    const Map<const MatrixXf> test_queries(test, dim, n_test);

    if(!parallel) omp_set_num_threads(1);

    std::cerr << std::endl;
    std::cerr << "parallel: " << parallel << std::endl;

    int seed_mrpt = 12345;

    std::vector<int> ks{1, 10, 100};

    std::vector<double> target_recalls {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.87,
                                        0.90, 0.92, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0};

    double build_time;

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

      double build_start = omp_get_wtime();
      Mrpt mrpt(M);
      mrpt.grow_autotune(k, trees_max, depth_max, depth_min, votes_max, density, seed_mrpt, n_auto);

      double build_end = omp_get_wtime();

      std::vector<std::vector<std::vector<int>>> vec_vote_counts;
      std::vector<std::vector<std::vector<int>>> vec_nn_found;
      for(const auto &tr : target_recalls) {
        Mrpt mrpt_new = mrpt.subset(tr);
        Mrpt_Parameters par(mrpt_new.parameters());

        if(mrpt_new.empty()) {
          continue;
        }

        std::vector<double> times, projection_times, voting_times, exact_times;
        std::vector<std::set<int>> idx;
        int elected = 0;

        if(k == 10 || k == 100) {
          oftop << par.k << " " << par.n_trees << " " << par.depth << " " << par.votes << std::endl;
          ofvotes << par.k << " " << par.n_trees << " " << par.depth << " " << par.votes << std::endl;
        }

        std::vector<std::vector<int>> all_vote_counts;
        std::vector<std::vector<int>> all_nn_found;

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

          std::map<int,int,std::greater<int>> vote_counts;

          if(k == 10 || k == 100) {
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

            std::partial_sort(votes.data(), votes.data() + k, votes.data() + n_points, std::greater<int>());
            for(int l = 0; l < k; ++l)
              oftop << votes(l) << " ";
            oftop << n_elected << std::endl;
          }
        }

        vec_vote_counts.push_back(all_vote_counts);
        vec_nn_found.push_back(all_nn_found);

        double median_projection_time = median(projection_times);
        double median_voting_time = median(voting_times);
        double median_exact_time = median(exact_times);
        double est_projection_time = MrptTest::get_projection_time(mrpt, par.n_trees, par.depth, par.votes);
        double est_voting_time = MrptTest::get_voting_time(mrpt, par.n_trees, par.depth, par.votes);
        double est_exact_time = MrptTest::get_exact_time(mrpt, par.n_trees, par.depth, par.votes);
        double mean_n_elected = elected / static_cast<double>(n_test);

        outf << k << " " << par.n_trees << " " << par.depth << " " << density << " " << par.votes << " ";

        results(k, times, idx, (result_path + "truth_" + std::to_string(k)).c_str(), verbose, outf);
        outf << build_end - build_start <<  " ";
        outf << par.estimated_recall << " ";
        outf << par.estimated_qtime * n_test << " ";
        outf << est_projection_time * n_test << " ";
        outf << est_voting_time * n_test << " ";
        outf << est_exact_time * n_test << " ";
        outf << (median_projection_time + median_voting_time + median_exact_time) * n_test << " ";
        outf << median_projection_time * n_test << " ";
        outf << median_voting_time * n_test << " ";
        outf << median_exact_time * n_test << " ";
        outf << mean_n_elected << " ";
        outf << std::endl;
      }

      if(k == 10 || k == 100) {
        for(int j = 0; j < target_recalls.size(); ++j) {
          double tr = target_recalls[j];

          Mrpt mrpt_new = mrpt.subset(tr);
          Mrpt_Parameters par(mrpt_new.parameters());

          if(mrpt_new.empty()) {
            continue;
          }

          std::vector<double> times, projection_times, voting_times, exact_times;
          std::vector<std::set<int>> idx;
          int elected = 0;

          const std::vector<std::vector<int>> &all_vote_counts = vec_vote_counts[j];
          const std::vector<std::vector<int>> &all_nn_found = vec_nn_found[j];

          for(int i = 0; i < n_test; ++i) {
            double projection_time = 0.0, voting_time = 0.0, exact_time = 0.0;
            std::vector<int> result(k);
            std::vector<float> distances(k);
            int n_elected = 0;
            const Map<const VectorXf> q(&test[i * dim], dim);

            const std::vector<int> &vote_counts = all_vote_counts[i];
            const std::vector<int> &nn_found = all_nn_found[i];
            int vote_threshold = get_vote_threshold_probability(tr, k, vote_counts, nn_found);

            double start = omp_get_wtime();
            Eigen::VectorXi votes = Eigen::VectorXi::Zero(n_points);
            mrpt_new.query(q, k, vote_threshold, &result[0], projection_time, voting_time, exact_time,
                           votes, &distances[0], &n_elected);
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

          outf2 << k << " " << par.n_trees << " " << par.depth << " " << density << " " << par.votes << " ";

          results(k, times, idx, (result_path + "truth_" + std::to_string(k)).c_str(), verbose, outf2);
          outf2 << build_end - build_start <<  " ";
          outf2 << par.estimated_recall << " ";
          outf2 << par.estimated_qtime * n_test << " ";
          outf2 << est_projection_time * n_test << " ";
          outf2 << est_voting_time * n_test << " ";
          outf2 << est_exact_time * n_test << " ";
          outf2 << (median_projection_time + median_voting_time + median_exact_time) * n_test << " ";
          outf2 << median_projection_time * n_test << " ";
          outf2 << median_voting_time * n_test << " ";
          outf2 << median_exact_time * n_test << " ";
          outf2 << mean_n_elected << " ";
          outf2 << std::endl;
        }
      }
    }


    delete[] test;
    if(!mmap) delete[] train;

    return 0;
}
