#include <stdexcept>
#include <hayai.hpp>
#include "Mrpt.h"
#include "common.h"

using namespace std;
using namespace Eigen;

void printParameters(const Mrpt_Parameters &op) {
    std::cout << "n_trees:                      " << op.n_trees << "\n";
    std::cout << "depth:                        " << op.depth << "\n";
    std::cout << "votes:                        " << op.votes << "\n";
    std::cout << "k:                            " << op.k << "\n";
    std::cout << "estimated query time:         " << op.estimated_qtime * 1000.0 << " ms.\n";
    std::cout << "estimated recall:             " << op.estimated_recall << "\n";
}

class MrptTest : public ::hayai::Fixture {
public:
    MrptTest() : M(nullptr, 0, 0), test_queries(nullptr, 0, 0), q(nullptr, 0) {
      std::string infile_path("/Users/DJKesoil/git/rp_test/timing/data/mnist/");

      test = read_memory((infile_path + "test.bin").c_str(), n_test, dim);
      if(!test) {
        throw std::invalid_argument("in file " + std::string(__FILE__) + ", line " + std::to_string(__LINE__) + ": file " + infile_path + "test.bin could not be read");
      }

      train = read_memory((infile_path + "train.bin").c_str(), n_points, dim);
      if(!train) {
        throw std::invalid_argument("in file " + std::string(__FILE__) + ", line " + std::to_string(__LINE__) + ": file " + infile_path + "train.bin could not be read");
      }

      new (&M) Map<const MatrixXf>(train, dim, n_points);
      new (&test_queries) Map<const MatrixXf>(test, dim, n_test);
      new (&q) Map<const VectorXf>(test, dim);

      omp_set_num_threads(1);
      double target_recall = 0.8;
      mrpt.data(M);

      //////////////////////// Autotuning ///////////////////////
      // mrpt.grow(target_recall, test_queries, k);
      // Mrpt_Parameters par = mrpt.parameters();
      // printParameters(par);
      // std::cout << std::endl;

      //////////////////////// Normal index //////////////////////
      int n_trees = 59, depth = 9; // + votes = 4 for 80% recall
      mrpt.grow(n_trees, depth);

    }

    void test_runner() {
      std::vector<int> res(k);
      for(int i = 0; i < n_test; ++i) {
        Mrpt::exact_knn(Map<const VectorXf>(test + i * dim, dim), M, k, &res[0]);
      }
    }

    void approximate_runner() {
      double pt = 0.0, vt = 0.0, et = 0.0;
      std::vector<int> res(k);
      for(int i = 0; i < n_test; ++i) {
        mrpt.query(Map<const VectorXf>(test + i * dim, dim), &res[0], pt, vt, et);
      }
    }

    void normal_runner() {
      double pt = 0.0, vt = 0.0, et = 0.0;
      std::vector<int> res(k);
      for(int i = 0; i < n_test; ++i) {
        mrpt.query(Map<const VectorXf>(test + i * dim, dim), k, votes, &res[0], pt, vt, et);
      }
    }

    Map<const MatrixXf> M;
    Map<const MatrixXf> test_queries;
    Map<const VectorXf> q;
    float *train = nullptr, *test = nullptr;
    Mrpt mrpt;

    int n_test = 100, n_points = 59900, dim = 784;
    int k = 5, votes = 4;
};

// BENCHMARK_F(MrptTest, ExactKnn, 5, 10) {
//   test_runner();
// }

// BENCHMARK_F(MrptTest, ApproximateKnn, 5, 10) {
//   approximate_runner();
// }

BENCHMARK_F(MrptTest, NormalQuery, 5, 10) {
  normal_runner();
}

int main(int argc, char **argv) {
    hayai::ConsoleOutputter consoleOutputter;

    hayai::Benchmarker::AddOutputter(consoleOutputter);
    hayai::Benchmarker::RunAllTests();
}
