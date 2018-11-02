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
    size_t n_test = atoi(argv[2]);
    size_t dim = atoi(argv[3]);
    int trees_max = atoi(argv[4]);
    int depth_min = atoi(argv[5]);
    int depth_max = atoi(argv[6]);
    float density = atof(argv[7]);
    bool parallel = atoi(argv[8]);

    std::string infile_path(argv[9]);
    if (!infile_path.empty() && infile_path.back() != '/')
      infile_path += '/';

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

    train = read_memory((infile_path + "train.bin").c_str(), n_points, dim);

    if(!train) {
        std::cerr << "in mrpt_comparison: training data " << infile_path + "train.bin" << " could not be read\n";
        return -1;
    }

    const Map<const MatrixXf> *M = new Map<const MatrixXf>(train, dim, n_points);
    Map<MatrixXf> *test_queries = new Map<MatrixXf>(test, dim, n_test);

    if(!parallel) omp_set_num_threads(1);
    int seed = 12345;
    int n_pool = trees_max * depth_max;

    SparseMatrix<float, RowMajor> spmat;
    Mrpt::build_sparse_random_matrix(spmat, n_pool, dim, density, seed);
    double nsum = 0;

    std::vector<double> times;
    for(int i = 0; i < n_test; ++i) {
      Map<VectorXf> q(&test[i * dim], dim);
      double start = omp_get_wtime();
      VectorXf projected_query(n_pool);
      projected_query.noalias() = spmat * q;
      double end = omp_get_wtime();
      times.push_back(end - start);
      nsum += projected_query.norm();
    }
    std::cout << "sum of norms: " << nsum << "\n";

    std::sort(times.begin(), times.end());
    std::cout << "\n\n\n";
    std::cout << "mean projection time: " << Autotuning::mean(times) * 1000.0 << " ms. " << "\n";
    std::cout << "median projection time: " << times[n_test / 2] * 1000.0 << " ms. " << "\n";
    std::cout << "standard deviation: " << std::sqrt(Autotuning::var(times)) << "\n";

    ////////////////////////////////////////////////////////////////////
    // Projection by loop

    double nsum2 = 0;

    std::vector<double> times2;
    for(int i = 0; i < n_test; ++i) {
      Map<VectorXf> q(&test[i * dim], dim);
      double qtime = 0.0;

      #pragma omp parallel for
      for (int n_tree = 0; n_tree < trees_max; ++n_tree) {

        double start2 = omp_get_wtime();
        VectorXf projected_query2(depth_max);
        projected_query2.noalias() = spmat.middleRows(n_tree * depth_max, depth_max) * q;
        double end2 = omp_get_wtime();

        nsum2 += projected_query2.norm();
        qtime += end2 - start2;
      }

      times2.push_back(qtime);
    }
    std::cout << "sum of norms: " << nsum2 << "\n";

    std::sort(times2.begin(), times2.end());
    std::cout << "\n\n\n";
    std::cout << "mean projection time: " << Autotuning::mean(times2) * 1000.0 << " ms. " << "\n";
    std::cout << "median projection time: " << times2[n_test / 2] * 1000.0 << " ms. " << "\n";
    std::cout << "standard deviation: " << std::sqrt(Autotuning::var(times2)) << "\n";





    delete[] test;
    if(!mmap) delete[] train;

    return 0;
}
