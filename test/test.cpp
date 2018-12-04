#include <random>
#include <vector>
#include <algorithm>
#include <set>
#include <omp.h>
#include <numeric>
#include <utility>
#include <stdexcept>

#include "gtest/gtest.h"
#include "Mrpt.h"
#include "Eigen/Dense"

using namespace Eigen;

static double mean(const std::vector<double> &x) {
  int n = x.size();
  double xsum = 0;
  for(int i = 0; i < n; ++i)
    xsum += x[i];
  return xsum / n;
}

static double var(const std::vector<double> &x) {
  int n = x.size();
  double xmean = mean(x);
  double ssr = 0;
  for(int i = 0; i < n; ++i)
    ssr += (x[i] - xmean) * (x[i] - xmean);
  return ssr / (n - 1);
}


class MrptTest : public testing::Test {
  protected:

  MrptTest() : d(100), n(1024), n2(155), n_test(100), seed_data(56789), seed_mrpt(12345),
    M(nullptr, 0, 0), M2(nullptr, 0, 0), test_queries(nullptr, 0, 0) {
          std::mt19937 mt(seed_data);
          std::normal_distribution<double> dist(5.0,2.0);

          X = MatrixXf(d,n);
          for(int i = 0; i < d; ++i)
            for(int j = 0; j < n; ++j)
              X(i,j) = dist(mt);

          q = VectorXf(d);
          for(int i = 0; i < d; ++i) q(i) = dist(mt);

          X2 = MatrixXf(d, n2);
          for(int i = 0; i < d; ++i)
            for(int j = 0; j < n2; ++j)
              X2(i,j) = dist(mt);

          Q = MatrixXf(d, n_test);
          for(int i = 0; i < d; ++i)
            for(int j = 0; j < n_test; ++j)
              Q(i,j) = dist(mt);

          true_knn = getTrueKnn(q, X2, true_distances);

          new (&M) Map<const MatrixXf>(X.data(), d, n);
          new (&M2) Map<const MatrixXf>(X2.data(), d, n2);
          new (&test_queries) Map<const MatrixXf>(Q.data(), d, n_test);
  }

  /**
  * Accessor for split points of trees (for testing purposes)
  * @param tree - index of tree in (0, ... , T-1)
  * @param index - the index of branch in (0, ... , (2^depth) - 1):
  * 0 = root
  * 1 = first branch of first level
  * 2 = second branch of first level
  * 3 = first branch of second level etc.
  * @return split point of index:th branch of tree:th tree
  */
  float getSplitPoint(const Mrpt &mrpt, int tree, int index) {
    return mrpt.split_points(index, tree);
  }


  /**
  * Accessor for point stored in leaves of trees (for testing purposes)
  * @param tree - index of tree in (0, ... T-1)
  * @param leaf - index of leaf in (0, ... , 2^depth)
  * @param index - index of a data point in a leaf
  * @return index of index:th data point in leaf:th leaf of tree:th tree
  */
  int getLeafPoint(const Mrpt &mrpt, int tree, int leaf, int index) {
    const std::vector<int> &leaf_first_indices = mrpt.leaf_first_indices_all[mrpt.depth];
    int leaf_begin = leaf_first_indices[leaf];
    return mrpt.tree_leaves[tree][leaf_begin + index];
  }

  /**
  * Accessor for the number of points in a leaf of a tree (for test purposes)
  * @param tree - index of tree in (0, ... T-1)
  * @param leaf - index of leaf in (0, ... , 2^depth)
  * @return - number of data points in leaf:th leaf of tree:th tree
  */
  int getLeafSize(const Mrpt &mrpt, int tree, int leaf) const {
    const std::vector<int> &leaf_first_indices = mrpt.leaf_first_indices_all[mrpt.depth];
    return leaf_first_indices[leaf + 1] - leaf_first_indices[leaf];
  }


  void autotuningGrowTester(float density, int trees_max, int depth_max,
        int depth_min, int votes_max, int k) {

    omp_set_num_threads(1);

    Mrpt mrpt(M);
    mrpt.grow(test_queries, k, trees_max, depth_max, depth_min, votes_max, density, seed_mrpt);

    Mrpt index_normal(M);
    index_normal.grow(trees_max, depth_max, density, seed_mrpt);

    splitPointsEqual(index_normal, mrpt);
    leavesEqual(index_normal, mrpt);
  }

  void floatPointerAutotuningGrowTester(float density, int trees_max, int depth_max,
        int depth_min, int votes_max, int k) {

    omp_set_num_threads(1);

    Mrpt mrpt(M);
    mrpt.grow(Q.data(), n_test, k, trees_max, depth_max, depth_min, votes_max, density, seed_mrpt);

    Mrpt index_normal(M);
    index_normal.grow(trees_max, depth_max, density, seed_mrpt);

    splitPointsEqual(index_normal, mrpt);
    leavesEqual(index_normal, mrpt);
  }

  void trainingSetAutotuningGrowTester(float density, int trees_max, int depth_max,
        int depth_min, int votes_max, int k, int n_test = 100) {

    omp_set_num_threads(1);

    Mrpt mrpt(M);
    mrpt.grow_autotune(k, trees_max, depth_max, depth_min, votes_max, density, seed_mrpt, n_test);

    Mrpt index_normal(M);
    index_normal.grow(trees_max, depth_max, density, seed_mrpt);

    splitPointsEqual(index_normal, mrpt);
    leavesEqual(index_normal, mrpt);
  }


  void autotuningTester(double target_recall, float density, int trees_max) {
    omp_set_num_threads(1);
    int depth_max = 7, depth_min = 5, votes_max = trees_max - 1, k = 5;
    MatrixXi exact = computeExactNeighbors(Q, X);

    Mrpt mrpt(X);
    mrpt.grow(test_queries, k, trees_max, depth_max, depth_min, votes_max, density, seed_mrpt);

    Mrpt mrpt1(mrpt.subset(target_recall));
    Mrpt mrpt2(mrpt.subset(target_recall));
    mrpt.prune(target_recall);

    std::vector<std::vector<int>> res1 = autotuningQuery(mrpt1);
    std::vector<std::vector<int>> res2 = autotuningQuery(mrpt2);
    std::vector<std::vector<int>> res3 = autotuningQuery(mrpt);

    EXPECT_EQ(res1, res2);
    EXPECT_EQ(res1, res3);
    EXPECT_FLOAT_EQ(mrpt1.parameters().estimated_recall, getRecall(res1, exact));
  }


  void trainingSetAutotuningTester(double target_recall, float density, int trees_max) {
    omp_set_num_threads(1);
    int depth_max = 6, depth_min = 4, votes_max = trees_max - 1, k = 5;
    int n_test = 100;

    Mrpt mrpt(X2);
    MatrixXf Q_self(mrpt.subset(mrpt.sample_indices(n_test, seed_mrpt)));
    MatrixXi exact = computeExactNeighbors(Q_self, X2);

    mrpt.grow_autotune(k, trees_max, depth_max, depth_min, votes_max, density, seed_mrpt);

    Mrpt mrpt1(mrpt.subset(target_recall));
    mrpt1.k = k + 1;
    mrpt1.par.k = k + 1;
    Mrpt mrpt2(mrpt.subset(target_recall));
    mrpt2.k = k + 1;
    mrpt2.par.k = k + 1;
    mrpt.prune(target_recall);
    mrpt.k = k + 1;
    mrpt.par.k = k + 1;

    std::vector<std::vector<int>> res1 = autotuningQuery(mrpt1, Q_self);
    std::vector<std::vector<int>> res2 = autotuningQuery(mrpt2, Q_self);
    std::vector<std::vector<int>> res3 = autotuningQuery(mrpt, Q_self);

    EXPECT_EQ(res1, res2);
    EXPECT_EQ(res1, res3);
    EXPECT_FLOAT_EQ(((mrpt1.parameters().estimated_recall * k) + 1) / (k + 1), getRecall(res1, exact));
  }


  void defaultArgumentTester(int k) {
    omp_set_num_threads(1);

    int trees_max = std::sqrt(n);
    int depth_max = std::log2(n) - 4;
    float density = 1.0 / std::sqrt(d);
    int depth_min = std::log2(n) - 11;

    Mrpt mrpt(M);
    mrpt.grow(test_queries, k);

    EXPECT_EQ(mrpt.n_trees, trees_max);
    EXPECT_EQ(mrpt.depth, depth_max);
    EXPECT_EQ(mrpt.depth_min, std::max(depth_min, 5));
    EXPECT_FLOAT_EQ(mrpt.density, density);

    std::mt19937 mt(seed_data);
    std::normal_distribution<double> dist(5.0,2.0);

    int nn = 10000, dd = 10;

    MatrixXf XX(dd,nn);
    for(int i = 0; i < dd; ++i)
      for(int j = 0; j < nn; ++j)
        XX(i,j) = dist(mt);

    trees_max = std::sqrt(nn);
    depth_max = std::log2(nn) - 4;
    density = 1.0 / std::sqrt(dd);
    depth_min = std::log2(nn) - 11;

    Mrpt mrpt2(XX);
    mrpt2.grow_autotune(k);

    EXPECT_EQ(mrpt2.n_trees, trees_max);
    EXPECT_EQ(mrpt2.depth, depth_max);
    EXPECT_EQ(mrpt2.depth_min, std::max(depth_min, 5));
    EXPECT_FLOAT_EQ(mrpt2.density, density);
  }

  void expect_equal(const Mrpt_Parameters &par, const Mrpt_Parameters &par2) {
    EXPECT_EQ(par.n_trees, par2.n_trees);
    EXPECT_EQ(par.depth, par2.depth);
    EXPECT_EQ(par.votes, par2.votes);
    EXPECT_EQ(par.k, par2.k);
    EXPECT_FLOAT_EQ(par.estimated_qtime, par2.estimated_qtime);
    EXPECT_FLOAT_EQ(par.estimated_recall, par2.estimated_recall);
  }

  void testOptimalParameters(const std::vector<Mrpt_Parameters> &pars, const std::vector<Mrpt_Parameters> &pars2) {
    ASSERT_EQ(pars.size(), pars2.size());
    for(auto it = pars.begin(), it2 = pars2.begin(); it != pars.end(); ++it, ++it2)
      expect_equal(*it, *it2);
  }

  void saveTester(int n_trees, int depth, float density, int seed_mrpt) {
    Mrpt mrpt(M2);
    mrpt.grow(n_trees, depth, density, seed_mrpt);
    mrpt.save("save/mrpt_saved");

    Mrpt mrpt_reloaded(M2);
    mrpt_reloaded.load("save/mrpt_saved");

    splitPointsEqual(mrpt, mrpt_reloaded);
    leavesEqual(mrpt, mrpt_reloaded);
    normalQueryEquals(mrpt, mrpt_reloaded, 5, 1);
  }

  void saveTesterAutotuning(int k, int trees_max, int depth_max, int depth_min,
      int votes_max, float density, int seed_mrpt) {
    Mrpt mrpt(M2);
    mrpt.grow(test_queries, k, trees_max, depth_max, depth_min, votes_max, density, seed_mrpt);
    mrpt.save("save/mrpt_saved");

    Mrpt mrpt_reloaded(M2);
    mrpt_reloaded.load("save/mrpt_saved");

    splitPointsEqual(mrpt, mrpt_reloaded);
    leavesEqual(mrpt, mrpt_reloaded);
    autotuningQueryEquals(mrpt, mrpt_reloaded, 0.4);
    expect_equal(mrpt.parameters(), mrpt_reloaded.parameters());
    testOptimalParameters(mrpt.optimal_parameters(), mrpt_reloaded.optimal_parameters());
  }

  void saveTesterAutotuningTargetRecall(double target_recall, int k, int trees_max,
      int depth_max, int depth_min, int votes_max, float density, int seed_mrpt) {
    Mrpt mrpt(M2);
    mrpt.grow(target_recall, test_queries, k, trees_max, depth_max, depth_min, votes_max, density, seed_mrpt);
    mrpt.save("save/mrpt_saved");

    Mrpt mrpt_reloaded(M2);
    mrpt_reloaded.load("save/mrpt_saved");

    splitPointsEqual(mrpt, mrpt_reloaded);
    leavesEqual(mrpt, mrpt_reloaded);
    normalQueryEquals(mrpt, mrpt_reloaded, 5, 1);
    autotuningQueryEquals(mrpt, mrpt_reloaded);
    expect_equal(mrpt.parameters(), mrpt_reloaded.parameters());
  }



  void queryTester(int n_trees, int depth, float density, int votes, int k) {

    Mrpt mrpt(M);
    mrpt.grow(n_trees, depth, density, seed_mrpt);

    std::vector<int> result(k);
    std::vector<float> distances(k);
    for(int i = 0; i < k; ++i)
      distances[i] = 0;

    int n_el = 0;
    mrpt.query(q, k, votes, &result[0], &distances[0], &n_el);

    for(int i = 0; i < k; ++i)  {
      if(i > 0) {
        EXPECT_LE(distances[i-1], distances[i]);
      }
      if(result[i] >= 0) {
        EXPECT_FLOAT_EQ(distances[i], (X.col(result[i]) - q).norm());
      }
    }
  }


  void splitPointsEqual(Mrpt &mrpt1, Mrpt &mrpt2) {
    int n_trees = mrpt1.n_trees;
    int n_trees_old = mrpt2.n_trees;
    ASSERT_EQ(n_trees, n_trees_old);

    int depth = mrpt1.depth, depth_old = mrpt2.depth;
    ASSERT_EQ(depth, depth_old);

    for(int tree = 0; tree < n_trees; ++tree) {
      int per_level = 1, idx = 0;

      for(int level = 0; level < depth; ++level) {
        for(int j = 0; j < per_level; ++j) {
          float split = getSplitPoint(mrpt1, tree, idx);
          float split_old = getSplitPoint(mrpt2, tree, idx);
          ++idx;
          ASSERT_FLOAT_EQ(split, split_old);
        }
      }
      per_level *= 2;
    }
  }


  void leavesEqual(Mrpt &mrpt1, Mrpt &mrpt2) {
    int n_trees = mrpt1.n_trees;
    int n_trees_old = mrpt2.n_trees;
    ASSERT_EQ(n_trees, n_trees_old);

    int depth = mrpt1.depth, depth_old = mrpt2.depth;
    ASSERT_EQ(depth, depth_old);

    int n_points = mrpt1.n_samples;
    int n_points_old = mrpt2.n_samples;
    ASSERT_EQ(n_points, n_points_old);

    for(int tree = 0; tree < n_trees; ++tree) {
      int n_leaf = std::pow(2, depth);
      VectorXi leaves = VectorXi::Zero(n_points);

      for(int j = 0; j < n_leaf; ++j) {
        int leaf_size = getLeafSize(mrpt1, tree, j);
        int leaf_size_old = getLeafSize(mrpt2, tree, j);
        ASSERT_EQ(leaf_size, leaf_size_old);

        std::vector<int> leaf(leaf_size), leaf_old(leaf_size);
        for(int i = 0; i < leaf_size; ++i) {
          leaf[i] = getLeafPoint(mrpt1, tree, j, i);
          leaf_old[i] = getLeafPoint(mrpt2, tree, j, i);
        }
        std::sort(leaf.begin(), leaf.end());
        std::sort(leaf_old.begin(), leaf_old.end());

        for(int i = 0; i < leaf_size; ++i) {
          ASSERT_EQ(leaf_old[i], leaf[i]);
          leaves(leaf[i]) = 1;
        }
      }

      // Test that all data points are found at a tree
      EXPECT_EQ(leaves.sum(), n_points);
    }
  }

  MatrixXi computeExactNeighbors(const MatrixXf &query, const MatrixXf &data) {
    int n_points = data.cols();
    int n_test = query.cols();
    MatrixXi res(n_points, n_test);
    Mrpt mrpt(data);

    for(int i = 0; i < n_test; ++i) {
      VectorXi idx(n_points);
      std::iota(idx.data(), idx.data() + n_points, 0);

      mrpt.exact_knn(Map<const VectorXf>(query.data() + i * d, d), n_points, idx, n_points, res.data() + i * n_points);
    }
    return res;
  }


  void printParameters(const Mrpt_Parameters &op) {
    std::cout << "n_trees:                      " << op.n_trees << "\n";
    std::cout << "depth:                        " << op.depth << "\n";
    std::cout << "votes:                        " << op.votes << "\n";
    std::cout << "k:                            " << op.k << "\n";
    std::cout << "estimated query time:         " << op.estimated_qtime * 1000.0 << " ms.\n";
    std::cout << "estimated recall:             " << op.estimated_recall << "\n";
  }

  double getRecall(std::vector<std::vector<int>> results, MatrixXi exact) {
    int n_test = results.size();
    int k = results[0].size();
    double recall = 0;
    for(int i = 0; i < n_test; ++i) {
      std::vector<int> &result = results[i];
      VectorXi knn = exact.col(i);

      std::sort(result.begin(), result.end());
      std::sort(knn.data(), knn.data() + k);
      std::set<int> intersect;
      std::set_intersection(knn.data(), knn.data() + k, result.begin(), result.end(),
                       std::inserter(intersect, intersect.begin()));

      recall += intersect.size();
    }
    return recall / (k * n_test);
  }

  std::vector<std::vector<int>> normalQuery(const Mrpt &mrpt, int k, int v) {
    std::vector<std::vector<int>> res;
    for(int i = 0; i < n_test; ++i) {
      std::vector<int> result(k);
      mrpt.query(Map<const VectorXf>(Q.data() + i * d, d), k, v, &result[0]);
      res.push_back(result);
    }
    return res;
  }

  std::vector<std::vector<int>> normalQueryVector(const Mrpt &mrpt, int k, int v) {
    std::vector<std::vector<int>> res;
    for(int i = 0; i < n_test; ++i) {
      std::vector<int> result(k);
      mrpt.query(Q.col(i), k, v, &result[0]);
      res.push_back(result);
    }
    return res;
  }

  std::vector<std::vector<int>> normalQueryFloatPointer(const Mrpt &mrpt, int k, int v) {
    std::vector<std::vector<int>> res;
    for(int i = 0; i < n_test; ++i) {
      std::vector<int> result(k);
      mrpt.query(Q.data() + i * d, k, v, &result[0]);
      res.push_back(result);
    }
    return res;
  }

  std::vector<std::vector<int>> autotuningQuery(const Mrpt &mrpt) {
    std::vector<std::vector<int>> res;
    for(int i = 0; i < n_test; ++i) {
      std::vector<int> result(mrpt.parameters().k);
      mrpt.query(Map<const VectorXf>(Q.data() + i * d, d), &result[0]);
      res.push_back(result);
    }
    return res;
  }

  std::vector<std::vector<int>> autotuningQuery(const Mrpt &mrpt, const MatrixXf &Q) {
    std::vector<std::vector<int>> res;
    for(int i = 0; i < n_test; ++i) {
      std::vector<int> result(mrpt.parameters().k);
      mrpt.query(Map<const VectorXf>(Q.data() + i * d, d), &result[0]);
      res.push_back(result);
    }
    return res;
  }

  std::vector<std::vector<int>> autotuningQueryVector(const Mrpt &mrpt) {
    std::vector<std::vector<int>> res;
    for(int i = 0; i < n_test; ++i) {
      std::vector<int> result(mrpt.parameters().k);
      mrpt.query(Q.col(i), &result[0]);
      res.push_back(result);
    }
    return res;
  }

  std::vector<std::vector<int>> autotuningQueryFloatPointer(const Mrpt &mrpt) {
    std::vector<std::vector<int>> res;
    for(int i = 0; i < n_test; ++i) {
      std::vector<int> result(mrpt.parameters().k);
      mrpt.query(Q.data() + i * d, &result[0]);
      res.push_back(result);
    }
    return res;
  }

  std::vector<std::vector<int>> autotuningQuery(const Mrpt &mrpt, double target_recall) {
    Mrpt mrpt2 = mrpt.subset(target_recall);
    std::vector<std::vector<int>> res;
    for(int i = 0; i < n_test; ++i) {
      std::vector<int> result(mrpt.parameters().k);
      mrpt2.query(Map<const VectorXf>(Q.data() + i * d, d), &result[0]);
      res.push_back(result);
    }
    return res;
  }

  std::vector<std::vector<int>> autotuningQueryVector(const Mrpt &mrpt, double target_recall) {
    Mrpt mrpt2 = mrpt.subset(target_recall);
    std::vector<std::vector<int>> res;
    for(int i = 0; i < n_test; ++i) {
      std::vector<int> result(mrpt.parameters().k);
      mrpt2.query(Q.col(i), &result[0]);
      res.push_back(result);
    }
    return res;
  }

  std::vector<std::vector<int>> autotuningQueryFloatPointer(const Mrpt &mrpt, double target_recall) {
    Mrpt mrpt2 = mrpt.subset(target_recall);
    std::vector<std::vector<int>> res;
    for(int i = 0; i < n_test; ++i) {
      std::vector<int> result(mrpt.parameters().k);
      mrpt2.query(Q.data() + i * d, &result[0]);
      res.push_back(result);
    }
    return res;
  }

  void normalQueryEquals(const Mrpt &mrpt1, const Mrpt &mrpt2, int k, int v) {
    std::vector<std::vector<int>> res1 = normalQuery(mrpt1, k, v);
    std::vector<std::vector<int>> res2 = normalQuery(mrpt2, k, v);

    EXPECT_EQ(res1, res2);
  }

  void autotuningQueryEquals(const Mrpt &mrpt1, const Mrpt &mrpt2) {
    std::vector<std::vector<int>> res1 = autotuningQuery(mrpt1);
    std::vector<std::vector<int>> res2 = autotuningQuery(mrpt2);

    EXPECT_EQ(res1, res2);
  }

  void autotuningQueryEquals(const Mrpt &mrpt1, const Mrpt &mrpt2, double target_recall) {
    std::vector<std::vector<int>> res1 = autotuningQuery(mrpt1, target_recall);
    std::vector<std::vector<int>> res2 = autotuningQuery(mrpt2, target_recall);

    EXPECT_EQ(res1, res2);
  }

  void exactKnnEquals(const std::vector<int> &result, const std::vector<float> &distances) {
    int k = result.size();
    ASSERT_EQ(k, distances.size());

    for(int i = 0; i < k; ++i) {
      ASSERT_EQ(result[i], true_knn[i]);
      ASSERT_FLOAT_EQ(distances[i], true_distances[i]);
    }
  }

  void privateExactKnnTester(int k) {
    Mrpt mrpt(X2);
    std::vector<float> distances(k);
    std::vector<int> result(k);
    VectorXi idx(n2);
    std::iota(idx.data(), idx.data() + n2, 0);

    mrpt.exact_knn(Map<const VectorXf>(q.data(), d), k, idx, n2, &result[0], &distances[0]);
    exactKnnEquals(result, distances);
  }

  void exactKnnTester(int k) {
    Mrpt mrpt(X2);
    std::vector<float> distances(k > 0 ? k : 0);
    std::vector<int> result(k > 0 ? k : 0);

    mrpt.exact_knn(Map<const VectorXf>(q.data(), d), k, &result[0], &distances[0]);
    exactKnnEquals(result, distances);
  }

  void vectorExactKnnTester(int k) {
    Mrpt mrpt(X2);
    std::vector<float> distances(k > 0 ? k : 0);
    std::vector<int> result(k > 0 ? k : 0);

    mrpt.exact_knn(q, k, &result[0], &distances[0]);
    exactKnnEquals(result, distances);
  }

  void floatPointerExactKnnTester(int k) {
    Mrpt mrpt(X2);
    std::vector<float> distances(k > 0 ? k : 0);
    std::vector<int> result(k > 0 ? k : 0);

    mrpt.exact_knn(q.data(), k, &result[0], &distances[0]);
    exactKnnEquals(result, distances);
  }

  void staticExactKnnTester(int k) {
    std::vector<float> distances(k > 0 ? k : 0);
    std::vector<int> result(k > 0 ? k : 0);

    Mrpt::exact_knn(Map<const VectorXf>(q.data(), d), M2, k, &result[0], &distances[0]);
    exactKnnEquals(result, distances);
  }

  void staticVectorExactKnnTester(int k) {
    std::vector<float> distances(k > 0 ? k : 0);
    std::vector<int> result(k > 0 ? k : 0);

    Mrpt::exact_knn(q, X2, k, &result[0], &distances[0]);
    exactKnnEquals(result, distances);
  }

  void staticFloatPointerExactKnnTester(int k) {
    std::vector<float> distances(k > 0 ? k : 0);
    std::vector<int> result(k > 0 ? k : 0);

    Mrpt::exact_knn(q.data(), M2.data(), M2.rows(), M2.cols(), k, &result[0], &distances[0]);
    exactKnnEquals(result, distances);
  }

  std::vector<int> getTrueKnn(const VectorXf &query, const MatrixXf &data,
      std::vector<float> &true_distances) {
    int n_points = data.cols();
    std::vector<int> true_knn(n_points);
    std::iota(true_knn.begin(), true_knn.end(), 0);

    true_distances = std::vector<float>(n_points);
    for(int i = 0; i < n_points; ++i)
      true_distances[i] = (data.col(i) - query).norm();

    std::sort(true_knn.begin(), true_knn.end(),
      [&true_distances](int i, int j) { return true_distances[i] < true_distances[j]; });

    std::sort(true_distances.begin(), true_distances.end());
    return true_knn;
  }

  void generate_x(std::vector<int> &x, int max_generated, int n_tested, int max_val) {
    Mrpt::generate_x(x, max_generated, n_tested, max_val);
  }

  void prune(Mrpt &mrpt, double target_recall) {
    mrpt.prune(target_recall);
  }


  void samplingTester(const Mrpt &mrpt, int n_test, int seed = 0) {
    std::vector<int> v(mrpt.sample_indices(n_test, seed));
    std::sort(v.begin(), v.end());
    int unique_count = std::unique(v.begin(), v.end()) - v.begin();

    EXPECT_EQ(unique_count, n_test);
  }

  void subsetTester(int n_test, int seed = 0) {
    Mrpt mrpt(X2);
    std::vector<int> indices(mrpt.sample_indices(n_test, seed));
    Eigen::MatrixXf Q(mrpt.subset(indices));
    for(int i = 0; i < n_test; ++i)
      for(int j = 0; j < d; ++j)
        ASSERT_FLOAT_EQ(X2(j, indices[i]), Q(j, i));
  }

  int n_samples(const Mrpt &mrpt) {
    return mrpt.n_samples;
  }

  int dim(const Mrpt &mrpt) {
    return mrpt.dim;
  }

  void expect_equal(const Mrpt &mrpt1, const Mrpt &mrpt2) {
    EXPECT_EQ(n_samples(mrpt1), n_samples(mrpt2));
    EXPECT_EQ(dim(mrpt1), dim(mrpt2));
    EXPECT_EQ(mrpt1.X.data(), mrpt2.X.data());

    ASSERT_EQ(mrpt1.split_points.size(), mrpt2.split_points.size());
    for(int i = 0; i < mrpt1.split_points.size(); ++i)
      ASSERT_FLOAT_EQ(mrpt1.split_points(i), mrpt2.split_points(i));

    ASSERT_EQ(mrpt1.tree_leaves.size(), mrpt2.tree_leaves.size());
    for(int i = 0; i < mrpt1.tree_leaves.size(); ++i)
      ASSERT_EQ(mrpt1.tree_leaves[i], mrpt2.tree_leaves[i]);

    ASSERT_FLOAT_EQ(mrpt1.density, mrpt2.density);
    if(mrpt1.density < 1.0) {
      ASSERT_FLOAT_EQ(mrpt1.sparse_random_matrix.cols(), mrpt2.sparse_random_matrix.cols());
      ASSERT_FLOAT_EQ(mrpt1.sparse_random_matrix.rows(), mrpt2.sparse_random_matrix.rows());
      for(int i = 0; i < mrpt1.sparse_random_matrix.outerSize(); ++i)
        for(Eigen::SparseMatrix<float, Eigen::RowMajor>::InnerIterator it(mrpt1.sparse_random_matrix, i), it2(mrpt2.sparse_random_matrix, i); it && it2; ++it, ++it2)
          ASSERT_FLOAT_EQ(it.value(), it2.value());
    } else {
      ASSERT_FLOAT_EQ(mrpt1.dense_random_matrix.cols(), mrpt2.dense_random_matrix.cols());
      ASSERT_FLOAT_EQ(mrpt1.dense_random_matrix.rows(), mrpt2.dense_random_matrix.rows());
      for(int i = 0; i < mrpt1.dense_random_matrix.rows(); ++i)
        for(int j = 0; j < mrpt1.dense_random_matrix.rows(); ++j)
          ASSERT_FLOAT_EQ(mrpt1.dense_random_matrix(i, j), mrpt2.dense_random_matrix(i, j));
    }

    ASSERT_EQ(mrpt1.leaf_first_indices_all.size(), mrpt2.leaf_first_indices_all.size());
    for(int i = 0; i < mrpt1.leaf_first_indices_all.size(); ++i)
      ASSERT_EQ(mrpt1.leaf_first_indices_all[i], mrpt2.leaf_first_indices_all[i]);

    ASSERT_EQ(mrpt1.leaf_first_indices, mrpt2.leaf_first_indices);
    expect_equal(mrpt1.par, mrpt2.par);

    EXPECT_EQ(mrpt1.n_trees, mrpt2.n_trees);
    EXPECT_EQ(mrpt1.depth, mrpt2.depth);
    EXPECT_EQ(mrpt1.n_pool, mrpt2.n_pool);
    EXPECT_EQ(mrpt1.n_array, mrpt2.n_array);
    EXPECT_EQ(mrpt1.votes, mrpt2.votes);
    EXPECT_EQ(mrpt1.k, mrpt2.k);
    EXPECT_EQ(mrpt1.index_type, mrpt2.index_type);

    EXPECT_EQ(mrpt1.depth_min, mrpt2.depth_min);
    EXPECT_EQ(mrpt1.votes_max, mrpt2.votes_max);
  }

  int d, n, n2, n_test, seed_data, seed_mrpt;
  double epsilon = 0.001; // error bound for floating point comparisons of recall
  MatrixXf X, X2, Q;
  VectorXf q;
  Map<const MatrixXf> M, M2;
  Map<const MatrixXf> test_queries;
  std::vector<float> true_distances;
  std::vector<int> true_knn;
};


// Test that:
// a) approximate k-nn are returned in correct order
// b) distances to the approximate k-nn are computed correctly
TEST_F(MrptTest, Query) {
  int n_trees = 10, depth = 6, votes = 1, k = 5;
  float density = 1;

  queryTester(1, depth, density, votes, k);
  queryTester(5, depth, density, votes, k);
  queryTester(100, depth, density, votes, k);

  queryTester(n_trees, 1, density, votes, k);
  queryTester(n_trees, 3, density, votes, k);
  queryTester(n_trees, 8, density, votes, k);
  queryTester(n_trees, 10, density, votes, k);

  queryTester(n_trees, depth, 0.05, votes, k);
  queryTester(n_trees, depth, 1.0 / std::sqrt(d), votes, k);
  queryTester(n_trees, depth, 0.5, votes, k);

  queryTester(n_trees, depth, density, 1, k);
  queryTester(n_trees, depth, density, 3, k);
  queryTester(30, depth, density, 5, k);

  queryTester(n_trees, depth, density, votes, 1);
  queryTester(n_trees, depth, density, votes, 2);
  queryTester(n_trees, depth, density, votes, 10);
}


// Test that the nearest neighbors returned by the index are different
// when rng is initialized with a random seed (no seed is given to
// grow() - method). Obs. this test may fail with a very small probability
// if the nearest neighbors returned by two different indices happen
// to be exactly same by change.
TEST_F(MrptTest, RandomSeed) {
  int n_trees = 10, depth = 6;
  float density = 1.0 / std::sqrt(d);

  Mrpt mrpt(M);
  mrpt.grow(n_trees, depth, density); // initialize rng with random seed
  Mrpt mrpt2(M);
  mrpt2.grow(n_trees, depth, density);

  int k = 10, votes = 1;
  std::vector<int> r(k), r2(k);

  mrpt.query(q, k, votes, &r[0]);
  mrpt2.query(q, k, votes, &r2[0]);

  bool same_neighbors = true;
  for(int i = 0; i < k; ++i) {
    if(r[i] != r2[i]) {
      same_neighbors = false;
      break;
    }
  }

  EXPECT_FALSE(same_neighbors);
}


// Test that the exact k-nn search (private version which is used by the
// approximate knn search) returns the true nearest neighbors and the
// true distances.
TEST_F(MrptTest, PrivateExactKnn) {
  privateExactKnnTester(1);
  privateExactKnnTester(5);
  privateExactKnnTester(n2);
}

// Test that the loaded index is identical to the original one that was saved.
TEST_F(MrptTest, Saving) {
  int n_trees = 3, depth = 6, seed_mrpt = 12345;
  float density = 1.0 / std::sqrt(d);

  saveTester(n_trees, depth, density, seed_mrpt);
  saveTester(n_trees, depth, 1.0, seed_mrpt);
  saveTester(1, depth, density, seed_mrpt);
}

// Test that the loaded autotuned index is identical to the original one that
// was saved.
TEST_F(MrptTest, AutotuningSaving) {
  int k = 5, trees_max = 5, depth_max = 6, depth_min = 4, votes_max = trees_max;
  int seed_mrpt = 12345;

  saveTesterAutotuning(k, trees_max, depth_max, depth_min, votes_max, 1.0 / std::sqrt(d), seed_mrpt);
  saveTesterAutotuning(k, trees_max, depth_max, depth_min, votes_max, 1.0, seed_mrpt);
}

// Test that the loaded index which was autotuned to the target recall level
// is identical to the original one that was saved.
TEST_F(MrptTest, AutotuningTargetRecallSaving) {
  int k = 5, trees_max = 5, depth_max = 6, depth_min = 4, votes_max = trees_max;
  int seed_mrpt = 12345;

  saveTesterAutotuningTargetRecall(0.1, k, trees_max, depth_max, depth_min, votes_max, 1.0 / std::sqrt(d), seed_mrpt);
  saveTesterAutotuningTargetRecall(0.9, k, trees_max, depth_max, depth_min, votes_max, 1.0, seed_mrpt);
  saveTesterAutotuningTargetRecall(0.1, k, trees_max, depth_max, depth_min, votes_max, 1.0 / std::sqrt(d), seed_mrpt);
  saveTesterAutotuningTargetRecall(0.9, k, trees_max, depth_max, depth_min, votes_max, 1.0, seed_mrpt);
}


// Test that an index grown with autotuning gives the same trees as the index grown
// with an old school grow-function
TEST_F(MrptTest, AutotuningGrowing) {
  int trees_max = 10, depth_max = 6, depth_min = 4, votes_max = std::max(1, trees_max - 1), k = 5;
  float density = 1.0 / std::sqrt(d);

  autotuningGrowTester(1.0 / std::sqrt(d), trees_max, depth_max, depth_min, votes_max, k);
  autotuningGrowTester(1.0, trees_max, depth_max, depth_min, votes_max, k);

  autotuningGrowTester(density, 2, depth_max, depth_min, 2, k);
  autotuningGrowTester(density, 5, depth_max, depth_min, 5, k);
  autotuningGrowTester(density, 100, depth_max, depth_min, 100, k);

  autotuningGrowTester(density, trees_max, 7, 5, votes_max, k);
  autotuningGrowTester(density, trees_max, 3, 2, votes_max, k);
  autotuningGrowTester(density, trees_max, 3, 1, votes_max, k);
  autotuningGrowTester(density, trees_max, 1, 1, votes_max, k);

  autotuningGrowTester(density, trees_max, depth_max, depth_min, 1, k);
  autotuningGrowTester(density, trees_max, depth_max, depth_min, 5, k);
  autotuningGrowTester(density, trees_max, depth_max, depth_min, 10, k);

  autotuningGrowTester(density, trees_max, depth_max, depth_min, votes_max, 1);
  autotuningGrowTester(density, trees_max, depth_max, depth_min, votes_max, 100);
  autotuningGrowTester(density, trees_max, depth_max, depth_min, votes_max, n2);

}

// Test that an index grown with autotuning gives the same trees as the index grown
// with an old school grow-function, when test set is given as a float pointer
TEST_F(MrptTest, FloatPointerAutotuningGrowing) {
  int trees_max = 10, depth_max = 6, depth_min = 4, votes_max = std::max(1, trees_max - 1), k = 5;
  float density = 1.0 / std::sqrt(d);

  floatPointerAutotuningGrowTester(1.0 / std::sqrt(d), trees_max, depth_max, depth_min, votes_max, k);
  floatPointerAutotuningGrowTester(1.0, trees_max, depth_max, depth_min, votes_max, k);

  floatPointerAutotuningGrowTester(density, 2, depth_max, depth_min, 2, k);
  floatPointerAutotuningGrowTester(density, 5, depth_max, depth_min, 5, k);
  floatPointerAutotuningGrowTester(density, 100, depth_max, depth_min, 100, k);

  floatPointerAutotuningGrowTester(density, trees_max, 7, 5, votes_max, k);
  floatPointerAutotuningGrowTester(density, trees_max, 3, 2, votes_max, k);
  floatPointerAutotuningGrowTester(density, trees_max, 3, 1, votes_max, k);
  floatPointerAutotuningGrowTester(density, trees_max, 1, 1, votes_max, k);

  floatPointerAutotuningGrowTester(density, trees_max, depth_max, depth_min, 1, k);
  floatPointerAutotuningGrowTester(density, trees_max, depth_max, depth_min, 5, k);
  floatPointerAutotuningGrowTester(density, trees_max, depth_max, depth_min, 10, k);

  floatPointerAutotuningGrowTester(density, trees_max, depth_max, depth_min, votes_max, 1);
  floatPointerAutotuningGrowTester(density, trees_max, depth_max, depth_min, votes_max, 100);
  floatPointerAutotuningGrowTester(density, trees_max, depth_max, depth_min, votes_max, n2);

}

// Test that an index grown with autotuning gives the same trees as the index grown
// with an old school grow-function, when using a version of autotuning which
// requires no test queries to build the index.
TEST_F(MrptTest, TrainingSetAutotuningGrowing) {
  int trees_max = 10, depth_max = 6, depth_min = 4, votes_max = std::max(1, trees_max - 1), k = 5;
  float density = 1.0 / std::sqrt(d);

  trainingSetAutotuningGrowTester(1.0 / std::sqrt(d), trees_max, depth_max, depth_min, votes_max, k);
  trainingSetAutotuningGrowTester(1.0, trees_max, depth_max, depth_min, votes_max, k);

  trainingSetAutotuningGrowTester(density, 2, depth_max, depth_min, 2, k);
  trainingSetAutotuningGrowTester(density, 5, depth_max, depth_min, 5, k);
  trainingSetAutotuningGrowTester(density, 100, depth_max, depth_min, 100, k);

  trainingSetAutotuningGrowTester(density, trees_max, 7, 5, votes_max, k);
  trainingSetAutotuningGrowTester(density, trees_max, 3, 2, votes_max, k);
  trainingSetAutotuningGrowTester(density, trees_max, 3, 1, votes_max, k);
  trainingSetAutotuningGrowTester(density, trees_max, 1, 1, votes_max, k);

  trainingSetAutotuningGrowTester(density, trees_max, depth_max, depth_min, 1, k);
  trainingSetAutotuningGrowTester(density, trees_max, depth_max, depth_min, 5, k);
  trainingSetAutotuningGrowTester(density, trees_max, depth_max, depth_min, 10, k);

  trainingSetAutotuningGrowTester(density, trees_max, depth_max, depth_min, votes_max, 1);
  trainingSetAutotuningGrowTester(density, trees_max, depth_max, depth_min, votes_max, 100);
  trainingSetAutotuningGrowTester(density, trees_max, depth_max, depth_min, votes_max, n2);

  trainingSetAutotuningGrowTester(density, trees_max, depth_max, depth_min, votes_max, 1, 10);
  trainingSetAutotuningGrowTester(density, trees_max, depth_max, depth_min, votes_max, 100, 100);
  trainingSetAutotuningGrowTester(density, trees_max, depth_max, depth_min, votes_max, n2, 1000);
}


// Test that the indices sampled into the test set are unique.
TEST_F(MrptTest, SamplingTestSet) {
  Mrpt mrpt(X2);
  samplingTester(mrpt, 1);
  samplingTester(mrpt, 100);
  samplingTester(mrpt, n2);
}

// Test that subsetting points from the data matrix works.
TEST_F(MrptTest, SubettingData) {
  subsetTester(1);
  subsetTester(100);
  subsetTester(n2);
}

// Test that:
// a) When subsetting the index from the original autotuning index and
// and using the validation set of autotuning as a test set, the
// recall level is exactly the estimated recall level.
// b) When subsetting a second index from the same autotuning index with
// the same target recall, the recall level stays the same.
// c) When deleting the trees of the original index with the same recall level
// as the subsetted index, the recall level stays the same
TEST_F(MrptTest, Autotuning) {
  int trees_max = 10;
  autotuningTester(0.2, 1.0 / std::sqrt(d), trees_max);
  autotuningTester(0.4, 1.0, trees_max);

  autotuningTester(0.2, 1.0, trees_max);
  autotuningTester(0.4, 1.0 / std::sqrt(d), trees_max);

  int tmax = 7;
  autotuningTester(0.2, 1.0, tmax);
  autotuningTester(0.2, 1.0 / std::sqrt(d), 7);
}

// Test that:
// a) When subsetting the index from the original autotuning index and
// and using the validation set of autotuning as a test set, the
// recall level is exactly the estimated recall level.
// b) When subsetting a second index from the same autotuning index with
// the same target recall, the recall level stays the same.
// c) When deleting the trees of the original index with the same recall level
// as the subsetted index, the recall level stays the same
TEST_F(MrptTest, TrainingSetAutotuning) {
  int trees_max = 10;
  trainingSetAutotuningTester(0.2, 1.0 / std::sqrt(d), trees_max);
  trainingSetAutotuningTester(0.4, 1.0, trees_max);

  trainingSetAutotuningTester(0.2, 1.0, trees_max);
  trainingSetAutotuningTester(0.4, 1.0 / std::sqrt(d), trees_max);

  int tmax = 7;
  trainingSetAutotuningTester(0.2, 1.0, tmax);
  trainingSetAutotuningTester(0.2, 1.0 / std::sqrt(d), 7);
}


// Test that the calling autotuning with default values for the parameters
// gives the index with the expected parameters
TEST_F(MrptTest, DefaultArguments) {
  defaultArgumentTester(1);
  defaultArgumentTester(5);
  defaultArgumentTester(20);
}

// Test that doing queries into an empty index returns correctly and sets
// output buffers to -1
TEST_F(MrptTest, EmptyIndexThrows) {
  int k = 5, v = 1;
  Mrpt mrpt(M);
  std::vector<int> res(k);
  std::vector<float> distances(k);
  int n_elected;

  EXPECT_THROW(mrpt.query(q, k, v, &res[0]), std::logic_error);
  EXPECT_THROW(mrpt.query(q, k, v, &res[0], &distances[0]), std::logic_error);
  EXPECT_THROW(mrpt.query(q, k, v, &res[0], &distances[0], &n_elected), std::logic_error);

  mrpt.grow(1, 5);
  EXPECT_NO_THROW(mrpt.query(q, k, v, &res[0]));
}

// Test that the normal query function works also on the index which is
// grown by autotuning without specifying the recall level.
TEST_F(MrptTest, NormalQuery) {
  int trees_max = 10, depth_max = 7, depth_min = 5, votes_max = trees_max - 1, k = 5;
  float density = 1.0 / std::sqrt(d);

  Mrpt mrpt(M2);
  mrpt.grow(trees_max, depth_max, density, seed_mrpt);

  Mrpt mrpt_at(M2);
  mrpt_at.grow(test_queries, k, trees_max, depth_max, depth_min, votes_max, density, seed_mrpt);

  Mrpt mrpt_at2(M2);
  mrpt_at2.grow(test_queries, 10, trees_max, depth_max, depth_min, votes_max, density, seed_mrpt);

  int v = 2;
  normalQueryEquals(mrpt, mrpt_at, k, v);
  normalQueryEquals(mrpt, mrpt_at2, k, v);
}

// Test that the normal query works, and returns the same results when used on
// the index subsetted to the target recall level from the original index, and
// when used on the original index from which trees are deleted to the
// same target recall level.
TEST_F(MrptTest, QuerySubsetting) {
  int trees_max = 10, depth_max = 7, depth_min = 5, votes_max = trees_max - 1, k = 5;
  float density = 1.0 / std::sqrt(d), target_recall = 0.2;

  Mrpt mrpt_at(M2);
  mrpt_at.grow(test_queries, k, trees_max, depth_max, depth_min, votes_max, density, seed_mrpt);

  Mrpt mrpt_at2(mrpt_at.subset(target_recall));
  prune(mrpt_at, target_recall);

  int v = 2;
  normalQueryEquals(mrpt_at, mrpt_at2, k, v);
}

// Test that the normal query throws an out-of-range exception when called with
// bad values for k or vote threshold.
TEST_F(MrptTest, QueryThrows) {
  Mrpt mrpt(M2);
  int n_trees = 10, v = 1, k =5;
  std::vector<int> res(n2);
  mrpt.grow(n_trees, 7, 1.0 / std::sqrt(d), seed_mrpt);

  EXPECT_THROW(mrpt.query(q, -1, v, &res[0]), std::out_of_range);
  EXPECT_THROW(mrpt.query(q, 0, v, &res[0]), std::out_of_range);
  EXPECT_THROW(mrpt.query(q, n2 + 1, v, &res[0]), std::out_of_range);
  EXPECT_NO_THROW(mrpt.query(q, 1, v, &res[0]));
  EXPECT_NO_THROW(mrpt.query(q, n2, v, &res[0]));

  EXPECT_THROW(mrpt.query(q, k, -1, &res[0]), std::out_of_range);
  EXPECT_THROW(mrpt.query(q, k, 0, &res[0]), std::out_of_range);
  EXPECT_THROW(mrpt.query(q, k, n_trees + 1, &res[0]), std::out_of_range);
  EXPECT_NO_THROW(mrpt.query(q, k, 1, &res[0]));
  EXPECT_NO_THROW(mrpt.query(q, k, n_trees, &res[0]));
}

// Test that the query meant for autotuned index throws an exception when
// called on the non-autotuned index.
TEST_F(MrptTest, RecallQueryThrows) {
  Mrpt mrpt(M2);
  int k = 5;
  mrpt.grow(10, 7, 1.0 / std::sqrt(d), seed_mrpt);

  std::vector<int> res(k);
  std::vector<float> dist(k);

  EXPECT_THROW(mrpt.query(q, &res[0]), std::logic_error);
  EXPECT_THROW(mrpt.query(q, &res[0], &dist[0]), std::logic_error);
}

// Test that normal tree growing throws an expection when parameters are out of bounds
TEST_F(MrptTest, GrowThrows) {
  int n_trees = 10, depth = 7;
  float density = 1.0 / std::sqrt(d);

  Mrpt mrpt(M2);

  EXPECT_THROW(mrpt.grow(-1, depth, density), std::out_of_range);
  EXPECT_TRUE(mrpt.empty());
  EXPECT_THROW(mrpt.grow(0, depth, density), std::out_of_range);
  EXPECT_TRUE(mrpt.empty());

  EXPECT_THROW(mrpt.grow(n_trees, -1, density), std::out_of_range);
  EXPECT_TRUE(mrpt.empty());
  EXPECT_THROW(mrpt.grow(n_trees, 0, density), std::out_of_range);
  EXPECT_TRUE(mrpt.empty());
  EXPECT_THROW(mrpt.grow(n_trees, 8, density), std::out_of_range);
  EXPECT_TRUE(mrpt.empty());

  EXPECT_NO_THROW(mrpt.grow(n_trees, 7, density));
  EXPECT_FALSE(mrpt.empty());

  Mrpt mrpt2(M2);
  EXPECT_THROW(mrpt2.grow(n_trees, depth, -0.001), std::out_of_range);
  EXPECT_TRUE(mrpt2.empty());
  EXPECT_THROW(mrpt2.grow(n_trees, depth, 1.1), std::out_of_range);
  EXPECT_TRUE(mrpt2.empty());

  EXPECT_NO_THROW(mrpt2.grow(n_trees, depth, 0.001));
  EXPECT_FALSE(mrpt2.empty());

  Mrpt mrpt3(M2);
  EXPECT_NO_THROW(mrpt3.grow(n_trees, depth, 1.0));
  EXPECT_FALSE(mrpt3.empty());

  Mrpt mrpt4(M2);
  EXPECT_NO_THROW(mrpt4.grow(n_trees, depth));
  EXPECT_FALSE(mrpt4.empty());
}

// Test that the autotuning throws an out-of-range exception, when called
// with bad value for k.
TEST_F(MrptTest, AutotuningKThrows) {
  Mrpt mrpt(M2);

  EXPECT_THROW(mrpt.grow(test_queries, -1), std::out_of_range);
  EXPECT_TRUE(mrpt.empty());
  EXPECT_THROW(mrpt.grow(test_queries, 0), std::out_of_range);
  EXPECT_TRUE(mrpt.empty());
  EXPECT_THROW(mrpt.grow(test_queries, n2 + 1), std::out_of_range);
  EXPECT_TRUE(mrpt.empty());

  EXPECT_THROW(mrpt.grow_autotune(-1), std::out_of_range);
  EXPECT_TRUE(mrpt.empty());
  EXPECT_THROW(mrpt.grow_autotune(0), std::out_of_range);
  EXPECT_TRUE(mrpt.empty());
  EXPECT_THROW(mrpt.grow_autotune(n2 + 1), std::out_of_range);
  EXPECT_TRUE(mrpt.empty());

  EXPECT_NO_THROW(mrpt.grow(test_queries, 1));
  EXPECT_FALSE(mrpt.empty());

  Mrpt mrpt3(M2);
  EXPECT_NO_THROW(mrpt3.grow_autotune(1));
  EXPECT_FALSE(mrpt3.empty());

  Mrpt mrpt2(M2);
  EXPECT_NO_THROW(mrpt2.grow(test_queries, n2));
  EXPECT_FALSE(mrpt2.empty());

  Mrpt mrpt4(M2);
  EXPECT_NO_THROW(mrpt4.grow_autotune(n2));
  EXPECT_FALSE(mrpt4.empty());
}

// Test that the autotuning throws an out-of-range expection when depth_max is
// non-positive or larger than log2(n)
TEST_F(MrptTest, AutotuningDepthmaxThrows) {
  Mrpt mrpt(M2);
  int k = 5, trees_max = 10;

  EXPECT_THROW(mrpt.grow(test_queries, k, trees_max, -2), std::out_of_range);
  EXPECT_TRUE(mrpt.empty());
  EXPECT_THROW(mrpt.grow(test_queries, k, trees_max, 0), std::out_of_range);
  EXPECT_TRUE(mrpt.empty());
  EXPECT_THROW(mrpt.grow(test_queries, k, trees_max, 8), std::out_of_range);
  EXPECT_TRUE(mrpt.empty());

  EXPECT_THROW(mrpt.grow_autotune(k, trees_max, -2), std::out_of_range);
  EXPECT_TRUE(mrpt.empty());
  EXPECT_THROW(mrpt.grow_autotune(k, trees_max, 0), std::out_of_range);
  EXPECT_TRUE(mrpt.empty());
  EXPECT_THROW(mrpt.grow_autotune(k, trees_max, 8), std::out_of_range);
  EXPECT_TRUE(mrpt.empty());

  EXPECT_NO_THROW(mrpt.grow(test_queries, k, trees_max, -1));
  EXPECT_FALSE(mrpt.empty());

  Mrpt mrpt3(M2);
  EXPECT_NO_THROW(mrpt3.grow_autotune(k, trees_max, -1));
  EXPECT_FALSE(mrpt3.empty());

  Mrpt mrpt2(M2);
  EXPECT_NO_THROW(mrpt2.grow(test_queries, k, trees_max, std::log2(n2)));
  EXPECT_FALSE(mrpt2.empty());

  Mrpt mrpt4(M2);
  EXPECT_NO_THROW(mrpt4.grow_autotune(k, trees_max, std::log2(n2)));
  EXPECT_FALSE(mrpt4.empty());
}

// Test that the autotuning throws an out-of-range exception when depth_min is
// not positive or is larger than depth_max.
TEST_F(MrptTest, AutotuningDepthminThrows) {
  Mrpt mrpt(M2);
  int k = 5, trees_max = 10, depth_max = 6;

  EXPECT_THROW(mrpt.grow(test_queries, k, trees_max, depth_max, -2), std::out_of_range);
  EXPECT_TRUE(mrpt.empty());
  EXPECT_THROW(mrpt.grow(test_queries, k, trees_max, depth_max, 0), std::out_of_range);
  EXPECT_TRUE(mrpt.empty());
  EXPECT_THROW(mrpt.grow(test_queries, k, trees_max, depth_max, 7), std::out_of_range);
  EXPECT_TRUE(mrpt.empty());

  EXPECT_THROW(mrpt.grow_autotune(k, trees_max, depth_max, -2), std::out_of_range);
  EXPECT_TRUE(mrpt.empty());
  EXPECT_THROW(mrpt.grow_autotune(k, trees_max, depth_max, 0), std::out_of_range);
  EXPECT_TRUE(mrpt.empty());
  EXPECT_THROW(mrpt.grow_autotune(k, trees_max, depth_max, 7), std::out_of_range);
  EXPECT_TRUE(mrpt.empty());

  EXPECT_NO_THROW(mrpt.grow(test_queries, k, trees_max, depth_max, -1));
  EXPECT_FALSE(mrpt.empty());

  Mrpt mrpt3(M2);
  EXPECT_NO_THROW(mrpt3.grow_autotune(k, trees_max, depth_max, -1));
  EXPECT_FALSE(mrpt3.empty());

  Mrpt mrpt2(M2);
  EXPECT_NO_THROW(mrpt2.grow(test_queries, k, trees_max, depth_max, depth_max));
  EXPECT_FALSE(mrpt2.empty());
}

// Test that the autotuning throws an out-of-range exception when votes_max is
// not positive or is larger than trees_max.
TEST_F(MrptTest, AutotuningVotesmaxThrows) {
  Mrpt mrpt(M2);
  int k = 5, trees_max = 10, depth_max = 7, depth_min = 5;

  EXPECT_THROW(mrpt.grow(test_queries, k, trees_max, depth_max, depth_min, -2), std::out_of_range);
  EXPECT_TRUE(mrpt.empty());
  EXPECT_THROW(mrpt.grow(test_queries, k, trees_max, depth_max, depth_min, 0), std::out_of_range);
  EXPECT_TRUE(mrpt.empty());
  EXPECT_THROW(mrpt.grow(test_queries, k, trees_max, depth_max, depth_min, trees_max + 1), std::out_of_range);
  EXPECT_TRUE(mrpt.empty());

  EXPECT_THROW(mrpt.grow_autotune(k, trees_max, depth_max, depth_min, -2), std::out_of_range);
  EXPECT_TRUE(mrpt.empty());
  EXPECT_THROW(mrpt.grow_autotune(k, trees_max, depth_max, depth_min, 0), std::out_of_range);
  EXPECT_TRUE(mrpt.empty());
  EXPECT_THROW(mrpt.grow_autotune(k, trees_max, depth_max, depth_min, trees_max + 1), std::out_of_range);
  EXPECT_TRUE(mrpt.empty());

  EXPECT_NO_THROW(mrpt.grow(test_queries, k, trees_max, depth_max, depth_min, -1));
  EXPECT_FALSE(mrpt.empty());

  Mrpt mrpt3(M2);
  EXPECT_NO_THROW(mrpt3.grow_autotune(k, trees_max, depth_max, depth_min, -1));
  EXPECT_FALSE(mrpt3.empty());

  Mrpt mrpt2(M2);
  EXPECT_NO_THROW(mrpt2.grow(test_queries, k, trees_max, depth_max, depth_min, trees_max));
  EXPECT_FALSE(mrpt2.empty());

  Mrpt mrpt4(M2);
  EXPECT_NO_THROW(mrpt4.grow_autotune(k, trees_max, depth_max, depth_min, trees_max));
  EXPECT_FALSE(mrpt4.empty());
}

// Test that the autotuning throws an out-of-range exception if density is
// non-positive or greater than one.
TEST_F(MrptTest, AutotuningDensityThrows) {
  Mrpt mrpt(M2);
  int k = 5, trees_max = 10, depth_max = 7, depth_min = 5, votes_max = trees_max - 1;

  EXPECT_THROW(mrpt.grow(test_queries, k, trees_max, depth_max, depth_min, votes_max, -0.001), std::out_of_range);
  EXPECT_TRUE(mrpt.empty());
  EXPECT_THROW(mrpt.grow(test_queries, k, trees_max, depth_max, depth_min, votes_max, -1.001), std::out_of_range);
  EXPECT_TRUE(mrpt.empty());
  EXPECT_THROW(mrpt.grow(test_queries, k, trees_max, depth_max, depth_min, votes_max, 1.1), std::out_of_range);
  EXPECT_TRUE(mrpt.empty());

  EXPECT_THROW(mrpt.grow_autotune(k, trees_max, depth_max, depth_min, votes_max, -0.001), std::out_of_range);
  EXPECT_TRUE(mrpt.empty());
  EXPECT_THROW(mrpt.grow_autotune(k, trees_max, depth_max, depth_min, votes_max, -1.001), std::out_of_range);
  EXPECT_TRUE(mrpt.empty());
  EXPECT_THROW(mrpt.grow_autotune(k, trees_max, depth_max, depth_min, votes_max, 1.1), std::out_of_range);
  EXPECT_TRUE(mrpt.empty());

  EXPECT_NO_THROW(mrpt.grow(test_queries, k, trees_max, depth_max, depth_min, votes_max, -1.0));
  EXPECT_FALSE(mrpt.empty());

  Mrpt mrpt4(M2);
  EXPECT_NO_THROW(mrpt4.grow_autotune(k, trees_max, depth_max, depth_min, votes_max, -1.0));
  EXPECT_FALSE(mrpt4.empty());

  Mrpt mrpt2(M2);
  EXPECT_NO_THROW(mrpt2.grow(test_queries, k, trees_max, depth_max, depth_min, votes_max, 1.0));
  EXPECT_FALSE(mrpt2.empty());

  Mrpt mrpt5(M2);
  EXPECT_NO_THROW(mrpt5.grow_autotune(k, trees_max, depth_max, depth_min, votes_max, 1.0));
  EXPECT_FALSE(mrpt5.empty());

  Mrpt mrpt3(M2);
  EXPECT_NO_THROW(mrpt3.grow(test_queries, k, trees_max, depth_max, depth_min, votes_max, 0.001));
  EXPECT_FALSE(mrpt3.empty());

  Mrpt mrpt6(M2);
  EXPECT_NO_THROW(mrpt6.grow_autotune(k, trees_max, depth_max, depth_min, votes_max, 0.001));
  EXPECT_FALSE(mrpt6.empty());
}


// Test that the autotuning grows invalid argument error if the dimensions
// of the data set and the validation set do not match.
TEST_F(MrptTest, AutotuningDimThrows) {
  int n_test2 = 100, d2 = 50, k = 5;
  MatrixXf q2 = MatrixXf::Random(d2, n_test2);
  Map<const MatrixXf> test_queries2(q2.data(), d2, n_test2);
  Mrpt mrpt(M2);

  EXPECT_THROW(mrpt.grow(test_queries2, k), std::invalid_argument);
  EXPECT_TRUE(mrpt.empty());

  Map<const MatrixXf> test_queries3(q.data(), d, 1);
  EXPECT_NO_THROW(mrpt.grow(test_queries3, k));
  EXPECT_FALSE(mrpt.empty());
}

TEST_F(MrptTest, SampleSizeThrows) {
  int k = 5;
  MatrixXf XS = MatrixXf::Random(d, 100);
  Mrpt mrpt(XS);
  EXPECT_THROW(mrpt.grow_autotune(k), std::out_of_range);

  MatrixXf S = MatrixXf::Random(d, 101);
  Mrpt mrpt2(S);
  EXPECT_NO_THROW(mrpt2.grow_autotune(k));
  EXPECT_FALSE(mrpt2.empty());
}

// Test that the autotuning function throws an out-of-range exception if
// the target recall level is not on the interval [0,1].
TEST_F(MrptTest, AutotuningTargetRecallThrows) {
  int k = 5;
  Mrpt mrpt(M2);

  EXPECT_THROW(mrpt.grow(-0.01, test_queries, k), std::out_of_range);
  EXPECT_TRUE(mrpt.empty());
  EXPECT_THROW(mrpt.grow(1.01, test_queries, k), std::out_of_range);
  EXPECT_TRUE(mrpt.empty());

  EXPECT_THROW(mrpt.grow_autotune(-0.01, k), std::out_of_range);
  EXPECT_TRUE(mrpt.empty());
  EXPECT_THROW(mrpt.grow_autotune(1.01, k), std::out_of_range);
  EXPECT_TRUE(mrpt.empty());

  EXPECT_NO_THROW(mrpt.grow(0, test_queries, k));
  EXPECT_FALSE(mrpt.empty());

  Mrpt mrpt3(M2);
  EXPECT_NO_THROW(mrpt3.grow_autotune(0.0, k));
  EXPECT_FALSE(mrpt3.empty());

  Mrpt mrpt2(M2);
  EXPECT_NO_THROW(mrpt2.grow(1.0, test_queries, k));
  EXPECT_FALSE(mrpt2.empty());

  Mrpt mrpt4(M2);
  EXPECT_NO_THROW(mrpt4.grow_autotune(1, k));
  EXPECT_FALSE(mrpt4.empty());
}

// Test that the function that prunes the autotuned index throws an
// out-of-range exception if the target recall level is not on the interval [0,1].
TEST_F(MrptTest, PruningTargetRecallThrows) {
  int k = 5;
  Mrpt mrpt(M2);
  mrpt.grow(test_queries, k);

  EXPECT_THROW(prune(mrpt, -0.01), std::out_of_range);
  EXPECT_THROW(prune(mrpt, 1.01), std::out_of_range);

  EXPECT_NO_THROW(prune(mrpt, 0));

  Mrpt mrpt2(M2);
  mrpt2.grow(test_queries, k);
  EXPECT_NO_THROW(prune(mrpt2, 1));
}

// Test that the function that prunes the autotuned (built without test queries)
// index throws an exception if the target recall level is not on the interval [0,1].
TEST_F(MrptTest, PruningTargetRecallTrainingSetThrows) {
  int k = 5;
  Mrpt mrpt(M2);
  mrpt.grow_autotune(k);

  EXPECT_THROW(prune(mrpt, -0.01), std::out_of_range);
  EXPECT_THROW(prune(mrpt, 1.01), std::out_of_range);

  EXPECT_NO_THROW(prune(mrpt, 0));

  Mrpt mrpt2(M2);
  mrpt2.grow_autotune(k);
  EXPECT_NO_THROW(prune(mrpt2, 1));
}



// Test that the function that subsets a new index fromt the autotuned index
// throws an out-of-range exception if the target recall level is not
// on the interval [0,1].
TEST_F(MrptTest, SubsettingTargetRecallThrows) {
  int k = 5;
  Mrpt mrpt(M2);
  mrpt.grow(test_queries, k);

  EXPECT_THROW(mrpt.subset(-0.01), std::out_of_range);
  EXPECT_THROW(mrpt.subset(1.01), std::out_of_range);

  EXPECT_NO_THROW(Mrpt mrpt_new(mrpt.subset(0)));

  Mrpt mrpt_new2(M2);
  EXPECT_NO_THROW(Mrpt mrpt_new2(mrpt.subset(1)));
}

// Test that the function that subsets a new index fromt the autotuned index
// throws an out-of-range exception if the target recall level is not
// on the interval [0,1].
TEST_F(MrptTest, SubsettingTargetRecallTrainingSetThrows) {
  int k = 5;
  Mrpt mrpt(M2);
  mrpt.grow_autotune(k);

  EXPECT_THROW(mrpt.subset(-0.01), std::out_of_range);
  EXPECT_THROW(mrpt.subset(1.01), std::out_of_range);

  EXPECT_NO_THROW(Mrpt mrpt_new(mrpt.subset(0)));

  Mrpt mrpt_new2(M2);
  EXPECT_NO_THROW(Mrpt mrpt_new2(mrpt.subset(1)));
}

// Test that the autotuning grow function throws an exception if a test set
// size is non-positive (but larger than data set size is OK; in this case
// test set size is set to data set size).
TEST_F(MrptTest, AutotuningTrainingSetTestSetSizeThrows) {
  int trees_max = 8, depth_max = 6, depth_min = 4, votes_max = trees_max - 1, k = 5, seed = 12345;
  float density = 1.0 / std::sqrt(d);

  Mrpt mrpt(M2);
  EXPECT_THROW(mrpt.grow_autotune(k, trees_max, depth_max, depth_min, votes_max, density, seed, -1), std::out_of_range);
  EXPECT_TRUE(mrpt.empty());
  EXPECT_THROW(mrpt.grow_autotune(k, trees_max, depth_max, depth_min, votes_max, density, seed, 0), std::out_of_range);
  EXPECT_TRUE(mrpt.empty());

  EXPECT_NO_THROW(mrpt.grow_autotune(k, trees_max, depth_max, depth_min, votes_max, density, seed, n2 + 1));
  EXPECT_FALSE(mrpt.empty());
}

// Test that when the index is not yet built, the parameter getter returns
// default values for the parameters and the estimated query time and the
// estimated recall.
TEST_F(MrptTest, ParameterGetterEmptyIndex) {
  Mrpt mrpt(M2);
  expect_equal(mrpt.parameters(), Mrpt_Parameters());
}

// Test that when the index is not autotuned, the getter for parameters returns
// the number and depth of trees, but not vote threshold and estimated query
// time and recall level.
TEST_F(MrptTest, ParameterGetter) {
  Mrpt mrpt(M2);
  int n_trees = 10, depth = 6;
  mrpt.grow(n_trees, depth);
  expect_equal(mrpt.parameters(), {n_trees, depth, 0, 0, 0.0, 0.0});
}

// Test that when the index is autotuned, but the target recall level
// is not yet set, the getter for parameters returns the maximum number and
// maximum depth of the trees, but not vote threshold and estimated query time
// and recall level.
TEST_F(MrptTest, ParameterGetterAutotuning) {
  Mrpt mrpt(M2);
  int k = 5, trees_max = 8, depth_max = 7;
  mrpt.grow(test_queries, k, trees_max, depth_max);
  expect_equal(mrpt.parameters(), {trees_max, depth_max, k, 0, 0.0, 0.0});
}

// Test that the getter for the list of optimal parameters throws a
// logic expection when called on either an empty index or an index which
// has not been autotuned.
TEST_F(MrptTest, NotAutotunedOptimalParameterGetterThrows) {
  Mrpt mrpt(M2);

  EXPECT_THROW(mrpt.optimal_parameters(), std::logic_error);

  mrpt.grow(10, 6);
  EXPECT_THROW(mrpt.optimal_parameters(), std::logic_error);
}

// Test that the getter for the list of optimal parameters throws a
// logic expection when called on an index which is autotuned to the
// target recall level.
TEST_F(MrptTest, AutotunedOptimalParameterGetterThrows) {
  Mrpt mrpt(M2);
  mrpt.grow(0.2, test_queries, 5);
  EXPECT_THROW(mrpt.optimal_parameters(), std::logic_error);
}

// Test that the getter for the list of optimal parameters throws a
// logic expection when called on an index which is autotuned to the
// target recall level.
TEST_F(MrptTest, AutotunedOptimalParameterTrainingSetGetterThrows) {
  Mrpt mrpt(M2);
  mrpt.grow_autotune(0.2, 5);
  EXPECT_THROW(mrpt.optimal_parameters(), std::logic_error);
}


// Test that the getter for the list of optimal parameters throws a
// logic expection when called on an index which is autotuned , but also
// already pruned to the target recall level.
TEST_F(MrptTest, PrunedOptimalParameterGetterThrows) {
  Mrpt mrpt(M2);
  mrpt.grow(test_queries, 5);
  prune(mrpt, 0.2);
  EXPECT_THROW(mrpt.optimal_parameters(), std::logic_error);
}

// Test that the getter for the list of optimal parameters throws a
// logic expection when called on an index which is autotuned , but also
// already pruned to the target recall level.
TEST_F(MrptTest, PrunedOptimalParameterGetterTrainingSetThrows) {
  Mrpt mrpt(M2);
  mrpt.grow_autotune(5);
  prune(mrpt, 0.2);
  EXPECT_THROW(mrpt.optimal_parameters(), std::logic_error);
}


// Test that the getter for the list of optimal parameters throws a
// logic expection when called on an index which is subsetted from
// the autotuned index, but does not throw when called on the original
// autotuned index.
TEST_F(MrptTest, SubsettedOptimalParameterGetterThrows) {
  Mrpt mrpt(M2);
  mrpt.grow(test_queries, 5);

  Mrpt mrpt_subsetted(mrpt.subset(0.2));
  EXPECT_THROW(mrpt_subsetted.optimal_parameters(), std::logic_error);
  EXPECT_NO_THROW(mrpt.optimal_parameters());
}

// Test that the when an autotuned index is subsetted to the target recall
// level, the estimated recall level is at least as high as the target recall
// level unless the target recall level is higher than highest estimated recall
// level. In this case, test that the estimated recall level is equal to the
// highest estimated recall level. Additionally, verify that the true recall
// level is equal to the estimated recall level when the validation set
// is used as a test set.
TEST_F(MrptTest, ParameterGetterSubsettedIndex) {
  int k = 5;
  MatrixXi exact2 = computeExactNeighbors(Q, X2);

  Mrpt mrpt(M2);
  mrpt.grow(test_queries, k, 20, 7, 3, 10, 1.0 / std::sqrt(d), seed_mrpt);

  std::vector<Mrpt_Parameters> pars = mrpt.optimal_parameters();
  double highest_estimated_recall = pars.rbegin()->estimated_recall;
  std::vector<double> target_recalls {0.1, 0.5, 0.9, 0.99};

  for(const auto &tr : target_recalls) {
    Mrpt mrpt_new(mrpt.subset(tr));
    Mrpt_Parameters par = mrpt_new.parameters();
    if(tr < highest_estimated_recall) {
      EXPECT_TRUE(par.estimated_recall - tr > -epsilon);
    } else {
      EXPECT_FLOAT_EQ(par.estimated_recall, highest_estimated_recall);
    }
    EXPECT_FLOAT_EQ(getRecall(autotuningQuery(mrpt_new), exact2), par.estimated_recall);
  }
}


// Test that the when an autotuned index is pruned to the target recall
// level, the estimated recall level is at least as high as the target recall
// level unless the target recall level is higher than highest estimated recall
// level. In this case, test that the estimated recall level is equal to the
// highest estimated recall level. Additionally, verify that the true recall
// level is equal to the estimated recall level when the validation set
// is used as a test set.
TEST_F(MrptTest, ParameterGetterPrunedIndex) {
  int k = 5;
  MatrixXi exact2 = computeExactNeighbors(Q, X2);

  std::vector<double> target_recalls {0.1, 0.5, 0.9, 0.99};
  for(const auto &tr : target_recalls) {
    Mrpt mrpt(M2);
    mrpt.grow(test_queries, k, 20, 7, 3, 10, 1.0 / std::sqrt(d), seed_mrpt);

    std::vector<Mrpt_Parameters> pars = mrpt.optimal_parameters();
    double highest_estimated_recall = pars.rbegin()->estimated_recall;

    prune(mrpt, tr);
    Mrpt_Parameters par = mrpt.parameters();

    if(tr < highest_estimated_recall) {
      EXPECT_TRUE(par.estimated_recall - tr > -epsilon);
    } else {
      EXPECT_FLOAT_EQ(par.estimated_recall, highest_estimated_recall);
    }
    EXPECT_FLOAT_EQ(getRecall(autotuningQuery(mrpt), exact2), par.estimated_recall);
  }
}

// Test that the when an index is autotuned to the target recall
// level, the estimated recall level is at least as high as the target recall
// level. Additionally, verify that the true recall
// level is equal to the estimated recall level when the validation set
// is used as a test set.
TEST_F(MrptTest, ParameterGetterTargetRecall) {
  int k = 5;
  MatrixXi exact2 = computeExactNeighbors(Q, X2);

  // with the parameters used in this test, the highest recall is 0.98
  // sot that the target recall level 0.95 is met
  std::vector<double> target_recalls {0.1, 0.5, 0.9, 0.95};
  for(const auto &tr : target_recalls) {
    Mrpt mrpt(M2);
    mrpt.grow(tr, test_queries, k, 20, 7, 3, 10, 1.0 / std::sqrt(d), seed_mrpt);
    Mrpt_Parameters par = mrpt.parameters();

    EXPECT_TRUE(par.estimated_recall - tr > -epsilon);
    EXPECT_FLOAT_EQ(getRecall(autotuningQuery(mrpt), exact2), par.estimated_recall);
  }
}

// Test that the constructor which takes the data as an Eigen::MatrixXf works and
// the index gives the same query results as an index built with a constructor
// taking the data as an Eigen::Map.
TEST_F(MrptTest, MatrixXfConstructor) {
  int n_trees = 10, depth = 6;
  float density = 1.0 / std::sqrt(d);

  Mrpt mrpt(M2);
  mrpt.grow(n_trees, depth, density, seed_mrpt);

  Mrpt mrpt_matrix(X2);
  mrpt_matrix.grow(n_trees, depth, density, seed_mrpt);

  normalQueryEquals(mrpt, mrpt_matrix, 5, 1);
}

// Test that the constructor which takes the data as an Eigen::MatrixXf works
// and that the true recall is at least the target recall.
TEST_F(MrptTest, AutotuningMatrixXfConstructor) {
  int trees_max = 10, depth_max = 6, depth_min = 4, votes_max = 10, k = 5;
  float density = 1.0 / std::sqrt(1.0);
  double target_recall = 0.6;
  MatrixXi exact2 = computeExactNeighbors(Q, X2);

  Mrpt mrpt(M2);
  mrpt.grow(target_recall, test_queries, k, trees_max, depth_max, depth_min, votes_max, density, seed_mrpt);

  EXPECT_GE(getRecall(autotuningQuery(mrpt), exact2), target_recall - epsilon);
}

// Test that the constructor which takes the data as a float pointer works and
// the index gives the same query results as the on built with a constructor
// taking the data as an Eigen::Map.
TEST_F(MrptTest, FloatPointerConstructor) {
  int n_trees = 10, depth = 6;
  float density = 1.0 / std::sqrt(1.0);

  Mrpt mrpt(M2);
  mrpt.grow(n_trees, depth, density, seed_mrpt);

  Mrpt mrpt_pointer(X2.data(), X2.rows(), X2.cols());
  mrpt_pointer.grow(n_trees, depth, density, seed_mrpt);

  normalQueryEquals(mrpt, mrpt_pointer, 5, 1);
}

// Test that the constructor which takes the data as a float pointer works
// and that the true recall is at least the target recall.
TEST_F(MrptTest, AutotuningFloatPointerConstructor) {
  int trees_max = 10, depth_max = 6, depth_min = 4, votes_max = 10, k = 5;
  float density = 1.0 / std::sqrt(1.0);
  double target_recall = 0.6;
  MatrixXi exact2 = computeExactNeighbors(Q, X2);

  Mrpt mrpt(M2);
  mrpt.grow(target_recall, test_queries, k, trees_max, depth_max, depth_min, votes_max, density, seed_mrpt);

  EXPECT_GE(getRecall(autotuningQuery(mrpt), exact2), target_recall - epsilon);
}

// Test that the autotuning function which takes the validation set as an
// Eigen::MatrixXf works and that the true recall is at least the target recall.
TEST_F(MrptTest, AutotuningMatrixXfGrowing) {
  int trees_max = 10, depth_max = 6, depth_min = 4, votes_max = 10, k = 5;
  float density = 1.0 / std::sqrt(1.0);
  double target_recall = 0.6;
  MatrixXi exact2 = computeExactNeighbors(Q, X2);

  Mrpt mrpt(M2);
  mrpt.grow(test_queries, k, trees_max, depth_max, depth_min, votes_max, density, seed_mrpt);

  EXPECT_GE(getRecall(autotuningQuery(mrpt, target_recall), exact2), target_recall - epsilon);
}

// Test that the autotuning function which takes the validation set as a
// float pointer works and that the true recall is at least the target recall.
TEST_F(MrptTest, AutotuningFloatPointerGrowing) {
  int trees_max = 10, depth_max = 6, depth_min = 4, votes_max = 10, k = 5;
  float density = 1.0 / std::sqrt(1.0);
  double target_recall = 0.6;
  MatrixXi exact2 = computeExactNeighbors(Q, X2);

  Mrpt mrpt(M2);
  mrpt.grow(test_queries, k, trees_max, depth_max, depth_min, votes_max, density, seed_mrpt);

  EXPECT_GE(getRecall(autotuningQuery(mrpt, target_recall), exact2), target_recall - epsilon);
}

// Test that the autotuning function with preset target recall which takes the
// validation set as an Eigen::MatrixXf works and that the true recall is
// at least the target recall.
TEST_F(MrptTest, AutotuningTargetRecallMatrixXfGrowing) {
  int trees_max = 10, depth_max = 6, depth_min = 4, votes_max = 10, k = 5;
  float density = 1.0 / std::sqrt(1.0);
  double target_recall = 0.6;
  MatrixXi exact2 = computeExactNeighbors(Q, X2);

  Mrpt mrpt(M2);
  mrpt.grow(target_recall, test_queries, k, trees_max, depth_max, depth_min, votes_max, density, seed_mrpt);

  EXPECT_GE(getRecall(autotuningQuery(mrpt), exact2), target_recall - epsilon);
}

// Test that the autotuning function with preset target recall which takes the
// validation set as a float pointer works and that the true recall is
// at least the target recall.
TEST_F(MrptTest, AutotuningTargetRecallFloatPointerGrowing) {
  int trees_max = 10, depth_max = 6, depth_min = 4, votes_max = 10, k = 5;
  float density = 1.0 / std::sqrt(1.0);
  double target_recall = 0.6;
  MatrixXi exact2 = computeExactNeighbors(Q, X2);

  Mrpt mrpt(M2);
  mrpt.grow(target_recall, test_queries, k, trees_max, depth_max, depth_min, votes_max, density, seed_mrpt);

  EXPECT_GE(getRecall(autotuningQuery(mrpt, target_recall), exact2), target_recall - epsilon);
}


// Test that the version of query function which takes the query point as
// Eigen::VectorXf gives the same results as the query function which takes
// the query point as Eigen::Map<VectorXf>
TEST_F(MrptTest, VectorXfQuery) {
  int n_trees = 10, depth = 6, k = 5, v = 1;
  float density = 1.0 / std::sqrt(d);

  Mrpt mrpt(M2);
  mrpt.grow(n_trees, depth, density, seed_mrpt);

  std::vector<std::vector<int>> res1(normalQuery(mrpt, k, v));
  std::vector<std::vector<int>> res2(normalQueryVector(mrpt, k, v));

  EXPECT_EQ(res1, res2);
}

// Test that the version of the autotuning query function which takes the query
// point as Eigen::VectorXf gives the same results as the query function which
// takes the query point as Eigen::Map<VectorXf>
TEST_F(MrptTest, AutotuningVectorXfQuery) {
  int trees_max = 10, depth_max = 6, depth_min = 4, votes_max = 10, k = 5;
  float density = 1.0 / std::sqrt(d);
  double target_recall = 0.6;

  Mrpt mrpt(M2);
  mrpt.grow(test_queries, k, trees_max, depth_max, depth_min, votes_max, density, seed_mrpt);

  std::vector<std::vector<int>> res1(autotuningQuery(mrpt, target_recall));
  std::vector<std::vector<int>> res2(autotuningQueryVector(mrpt, target_recall));

  EXPECT_EQ(res1, res2);
}

// Test that the version of the autotuning (with target recall prespecified)
// query function which takes the query point as Eigen::VectorXf gives the same
// results as the query function which takes the query point as
// Eigen::Map<VectorXf>
TEST_F(MrptTest, AutotuningTargetRecallVectorXfQuery) {
  int trees_max = 10, depth_max = 6, depth_min = 4, votes_max = 10, k = 5;
  float density = 1.0 / std::sqrt(d);
  double target_recall = 0.6;

  Mrpt mrpt(M2);
  mrpt.grow(target_recall, test_queries, k, trees_max, depth_max, depth_min, votes_max, density, seed_mrpt);

  std::vector<std::vector<int>> res1(autotuningQuery(mrpt));
  std::vector<std::vector<int>> res2(autotuningQueryVector(mrpt));

  EXPECT_EQ(res1, res2);
}

// Test that the version of query function which takes the query point as
// float pointer gives the same results as the query function which takes
// the query point as Eigen Map
TEST_F(MrptTest, FloatPointerQuery) {
  int n_trees = 10, depth = 6, k = 5, v = 1;
  float density = 1.0 / std::sqrt(d);

  Mrpt mrpt(M2);
  mrpt.grow(n_trees, depth, density, seed_mrpt);

  std::vector<std::vector<int>> res1(normalQuery(mrpt, k, v));
  std::vector<std::vector<int>> res2(normalQueryFloatPointer(mrpt, k, v));

  EXPECT_EQ(res1, res2);
}

// Test that the version of the autotuning query function which takes the query
// point as a float pointer gives the same results as the query function which
// takes the query point as Eigen Map.
TEST_F(MrptTest, AutotuningFloatPointerQuery) {
  int trees_max = 10, depth_max = 6, depth_min = 4, votes_max = 10, k = 5;
  float density = 1.0 / std::sqrt(d);
  double target_recall = 0.6;

  Mrpt mrpt(M2);
  mrpt.grow(test_queries, k, trees_max, depth_max, depth_min, votes_max, density, seed_mrpt);

  std::vector<std::vector<int>> res1(autotuningQuery(mrpt, target_recall));
  std::vector<std::vector<int>> res2(autotuningQueryFloatPointer(mrpt, target_recall));

  EXPECT_EQ(res1, res2);
}

// Test that the version of the autotuning (with target recall prespecified)
// query function which takes the query point as a float pointer gives the same
// results as the query function which takes the query point as
// Eigen Map.
TEST_F(MrptTest, AutotuningTargetRecallFloatPointerQuery) {
  int trees_max = 10, depth_max = 6, depth_min = 4, votes_max = 10, k = 5;
  float density = 1.0 / std::sqrt(d);
  double target_recall = 0.6;

  Mrpt mrpt(M2);
  mrpt.grow(target_recall, test_queries, k, trees_max, depth_max, depth_min, votes_max, density, seed_mrpt);

  std::vector<std::vector<int>> res1(autotuningQuery(mrpt));
  std::vector<std::vector<int>> res2(autotuningQueryFloatPointer(mrpt));

  EXPECT_EQ(res1, res2);
}

// Test that the exact k-nn search from the whole data set returns the true
// nearest neighbors and the true distances.
TEST_F(MrptTest, ExactKnn) {
  exactKnnTester(1);
  exactKnnTester(5);
  exactKnnTester(n2);
}

// Test that the version exact k-nn search which takes query point as VectorXf
// returns the true nearest neighbors and the true distances
TEST_F(MrptTest, VectorExactKnn) {
  vectorExactKnnTester(1);
  vectorExactKnnTester(5);
  vectorExactKnnTester(n2);
}

// Test that the version exact k-nn search which takes query point as a float
// pointer returns the true nearest neighbors and the true distances
TEST_F(MrptTest, FloatPointerExactKnn) {
  floatPointerExactKnnTester(1);
  floatPointerExactKnnTester(5);
  floatPointerExactKnnTester(n2);
}

// Test that the static version of exact k-nn search from the whole data set
// returns the true nearest neighbors and the true distances.
TEST_F(MrptTest, StaticExactKnn) {
  staticExactKnnTester(1);
  staticExactKnnTester(5);
  staticExactKnnTester(n2);
}

// Test that the static version exact k-nn search which takes query point as
// VectorXf returns the true nearest neighbors and the true distances.
TEST_F(MrptTest, StaticVectorExactKnn) {
  staticVectorExactKnnTester(1);
  staticVectorExactKnnTester(5);
  staticVectorExactKnnTester(n2);
}

// Test that the static version exact k-nn search which takes query point as
// a float pointer returns the true nearest neighbors and the true distances.
TEST_F(MrptTest, StaticFloatPointerExactKnn) {
  staticFloatPointerExactKnnTester(1);
  staticFloatPointerExactKnnTester(5);
  staticFloatPointerExactKnnTester(n2);
}

// Test that exact knn functions throw an out of range expection if k is
// greater than sample size of data or non-positive.
TEST_F(MrptTest, ExactKnnThrows) {
  EXPECT_THROW(exactKnnTester(-1), std::out_of_range);
  EXPECT_THROW(exactKnnTester(0), std::out_of_range);
  EXPECT_THROW(exactKnnTester(n2 + 1), std::out_of_range);

  EXPECT_THROW(vectorExactKnnTester(-1), std::out_of_range);
  EXPECT_THROW(vectorExactKnnTester(0), std::out_of_range);
  EXPECT_THROW(vectorExactKnnTester(n2 + 1), std::out_of_range);

  EXPECT_THROW(floatPointerExactKnnTester(-1), std::out_of_range);
  EXPECT_THROW(floatPointerExactKnnTester(0), std::out_of_range);
  EXPECT_THROW(floatPointerExactKnnTester(n2 + 1), std::out_of_range);

  EXPECT_THROW(staticExactKnnTester(-1), std::out_of_range);
  EXPECT_THROW(staticExactKnnTester(0), std::out_of_range);
  EXPECT_THROW(staticExactKnnTester(n2 + 1), std::out_of_range);

  EXPECT_THROW(staticVectorExactKnnTester(-1), std::out_of_range);
  EXPECT_THROW(staticVectorExactKnnTester(0), std::out_of_range);
  EXPECT_THROW(staticVectorExactKnnTester(n2 + 1), std::out_of_range);

  EXPECT_THROW(staticFloatPointerExactKnnTester(-1), std::out_of_range);
  EXPECT_THROW(staticFloatPointerExactKnnTester(0), std::out_of_range);
  EXPECT_THROW(staticFloatPointerExactKnnTester(n2 + 1), std::out_of_range);
}

// Test that the function that generates search set sizes gives the same
// results as the reference implementation.
TEST_F(MrptTest, GenerateXCoordinatesExact) {
  int n_samples = 60000;
  std::vector<int> s_tested2 {1,2,5,10,20,35,50,75,100,150,200,300,400,500};
  generate_x(s_tested2, n_samples / 20, 20, n_samples);

  std::vector<int> s_tested {1,2,5,10,20,35,50,75,100,150,200,300,400,500};
  int s_max = n_samples / 20;
  int n_s_tested = 20;
  int increment = s_max / n_s_tested;
  for(int i = 1; i <= n_s_tested; ++i)
    if(std::find(s_tested.begin(), s_tested.end(), i * increment) == s_tested.end()) {
      s_tested.push_back(i * increment);
    }

  // remove candidate set sizes that are larger than the size of the data set
  std::sort(s_tested.begin(), s_tested.end());
  auto s = s_tested.begin();
  for(; s != s_tested.end() && *s <= n_samples; ++s);
  s_tested.erase(s, s_tested.end());

  std::sort(s_tested.begin(), s_tested.end());
  std::sort(s_tested2.begin(), s_tested2.end());

  ASSERT_EQ(s_tested, s_tested2);
}

// Test that the function that generates tree numbers for measuring
// the projection times gives the same results as the reference implementation.
TEST_F(MrptTest, GenerateXCoordinatesProjection) {
  int n_trees = 1000;
  std::vector<int> tested_trees2 {1,2,3,4,5,7,10,15,20,25,30,40,50};
  generate_x(tested_trees2, n_trees, 10, n_trees);

  std::vector<int> tested_trees {1,2,3,4,5,7,10,15,20,25,30,40,50};

  int n_tested_trees = 10;
  n_tested_trees = n_trees > n_tested_trees ? n_tested_trees : n_trees;
  int incr = n_trees / n_tested_trees;
  for(int i = 1; i <= n_tested_trees; ++i)
    if(std::find(tested_trees.begin(), tested_trees.end(), i * incr) == tested_trees.end() && i * incr <= n_trees) {
      tested_trees.push_back(i * incr);
    }

  int nt = n_trees;
  auto end = std::remove_if(tested_trees.begin(), tested_trees.end(), [nt](int t) { return t > nt; });
  tested_trees.erase(end, tested_trees.end());

  std::sort(tested_trees.begin(), tested_trees.end());
  std::sort(tested_trees2.begin(), tested_trees2.end());
  ASSERT_EQ(tested_trees, tested_trees2);
}

// Test that the function that generates tree numbers for measuring
// the voting times gives the same results as the reference implementation.
TEST_F(MrptTest, GenerateXCoordinatesTreesVoting) {
  int n_trees = 523;
  std::vector<int> tested_trees2 {1,2,3,4,5,7,10,15,20,25,30,40,50};
  generate_x(tested_trees2, n_trees, 10, n_trees);

  std::vector<int> tested_trees {1,2,3,4,5,7,10,15,20,25,30,40,50};
  int n_tested_trees = 10;
  n_tested_trees = n_trees > n_tested_trees ? n_tested_trees : n_trees;
  int incr = n_trees / n_tested_trees;
  for(int i = 1; i <= n_tested_trees; ++i)
  if(std::find(tested_trees.begin(), tested_trees.end(), i * incr) == tested_trees.end()) {
    tested_trees.push_back(i * incr);
  }

  int nt = n_trees;
  auto end = std::remove_if(tested_trees.begin(), tested_trees.end(), [nt](int t) { return t > nt; });
  tested_trees.erase(end, tested_trees.end());

  std::sort(tested_trees.begin(), tested_trees.end());
  std::sort(tested_trees2.begin(), tested_trees2.end());
  ASSERT_EQ(tested_trees, tested_trees2);
}

// Test that the function that generates vote thresholds for measuring
// the voting times gives the same results as the reference implementation.
TEST_F(MrptTest, GenerateXCoordinatesVoteThresholdsVoting) {
  int votes_max = 523;
  std::vector<int> vote_thresholds_x2 {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
  generate_x(vote_thresholds_x2, votes_max, 10, votes_max);

  std::vector<int> vote_thresholds_x {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
  int n_votes = 10; // for how many different vote thresholds voting is tested
  n_votes = votes_max > n_votes ? n_votes : votes_max;
  int inc = votes_max / n_votes;
  for(int i = 1; i <= n_votes; ++i)
    if(std::find(vote_thresholds_x.begin(), vote_thresholds_x.end(), i * inc) == vote_thresholds_x.end()) {
      vote_thresholds_x.push_back(i * inc);
    }

  // remove tested vote thresholds that are larger than the preset maximum vote threshold
  std::sort(vote_thresholds_x.begin(), vote_thresholds_x.end());
  auto vt = vote_thresholds_x.begin();
  for(; vt != vote_thresholds_x.end() && *vt <= votes_max; ++vt);
  vote_thresholds_x.erase(vt, vote_thresholds_x.end());

  std::sort(vote_thresholds_x.begin(), vote_thresholds_x.end());
  std::sort(vote_thresholds_x2.begin(), vote_thresholds_x2.end());
  ASSERT_EQ(vote_thresholds_x, vote_thresholds_x2);
}

// Test that index cannot be grown second time on the same Mrpt object when
// the first version has been grown with autotuning.
TEST_F(MrptTest, AutotunedGrowingSecondTimeThrows) {
  int k = 10, n_trees = 8, depth = 6;
  double target_recall = 0.8;
  Mrpt mrpt(M2);
  mrpt.grow(target_recall, Q, k);

  EXPECT_THROW(mrpt.grow(Q, k), std::logic_error);
  EXPECT_THROW(mrpt.grow(Q.data(), n_test, k), std::logic_error);
  EXPECT_THROW(mrpt.grow(target_recall, Q, k), std::logic_error);
  EXPECT_THROW(mrpt.grow(target_recall, Q.data(), n_test, k), std::logic_error);
  EXPECT_THROW(mrpt.grow(Q, n_trees, depth), std::logic_error);
  EXPECT_THROW(mrpt.grow(Q.data(), n_trees, depth), std::logic_error);
}


// Test that index cannot be grown second time on the same Mrpt object when
// the first version has been grown with autotuning (without pruning).
TEST_F(MrptTest, AutotunedUnprunedGrowingSecondTimeThrows) {
  int k = 10, n_trees = 8, depth = 6;
  double target_recall = 0.8;
  Mrpt mrpt(M2);
  mrpt.grow(Q, k);

  EXPECT_THROW(mrpt.grow(Q, k), std::logic_error);
  EXPECT_THROW(mrpt.grow(Q.data(), n_test, k), std::logic_error);
  EXPECT_THROW(mrpt.grow(target_recall, Q, k), std::logic_error);
  EXPECT_THROW(mrpt.grow(target_recall, Q.data(), n_test, k), std::logic_error);
  EXPECT_THROW(mrpt.grow(Q, n_trees, depth), std::logic_error);
  EXPECT_THROW(mrpt.grow(Q.data(), n_trees, depth), std::logic_error);
}

// Test that index cannot be grown second time on the same Mrpt object
// when the first version has been grown with normal index building.
TEST_F(MrptTest, NormalGrowingSecondTimeThrows) {
  int k = 10, n_trees = 8, depth = 6;
  double target_recall = 0.8;
  Mrpt mrpt(M2);
  mrpt.grow(n_trees, depth);

  EXPECT_THROW(mrpt.grow(Q, k), std::logic_error);
  EXPECT_THROW(mrpt.grow(Q.data(), n_test, k), std::logic_error);
  EXPECT_THROW(mrpt.grow(target_recall, Q, k), std::logic_error);
  EXPECT_THROW(mrpt.grow(target_recall, Q.data(), n_test, k), std::logic_error);
  EXPECT_THROW(mrpt.grow(Q, n_trees, depth), std::logic_error);
  EXPECT_THROW(mrpt.grow(Q.data(), n_trees, depth), std::logic_error);
}


class UtilityTest : public testing::Test {
  protected:

  UtilityTest() {}

  void leafTester(int n, int depth, const std::vector<int> &indices_reference) {
    std::vector<int> indices;
    Mrpt::count_first_leaf_indices(indices, n, depth);
    EXPECT_EQ(indices, indices_reference);
  }

  void allLeavesTester(int n, const std::vector<std::vector<int>> &indices_reference) {
    std::vector<std::vector<int>> indices;
    Mrpt::count_first_leaf_indices_all(indices, n, indices_reference.size() - 1);
    EXPECT_EQ(indices, indices_reference);
  }

  void testTheilSen(std::vector<double> x, std::vector<double> y, double intercept, double slope) {
    std::pair<double,double> theil_sen = Mrpt::fit_theil_sen(x, y);

    EXPECT_FLOAT_EQ(theil_sen.first, intercept);
    EXPECT_FLOAT_EQ(theil_sen.second, slope);
  }

};


// Test that the implementation of Theil-Sen estimator gives a correct solution
// for the toy data.
TEST_F(UtilityTest, TheilSen) {
  std::vector<double> x {1,2,3,4,5,6,7,8,9,10};
  std::vector<double> y {1,2,2,3,5,4,7,7,8,9};
  testTheilSen(x, y, -1.0, 1.0);
}

// Test that when we split the data into half at each node, and if the number
// of data points at that node is odd, the extra point goes always to the
// left branch, and the tree is represented by a vector of length n so that
// the points of each branch are always stored contiguously, the beginning and
// the end points of each branch at each level are computed correctly.
//
// For instance, when there are 19 data points:
//  - at the root : the index of first data point is 0, and 18 is the index of
//    the last data point
//  - when depth = 1 : the index of the first data point of the left branch is
//    0, the index of the first data point of the right branch is 10, etc.
TEST_F(UtilityTest, LeafSizes) {
  std::vector<std::vector<int>> indices_reference;
  indices_reference.push_back({0,19});
  indices_reference.push_back({0,10,19});
  indices_reference.push_back({0,5,10,15,19});
  indices_reference.push_back({0,3,5,8,10,13,15,17,19});
  indices_reference.push_back({0,2,3,4,5,7,8,9,10,12,13,14,15,16,17,18,19});

  for(int depth = 0; depth < indices_reference.size(); ++depth)
    leafTester(19, depth, indices_reference[depth]);

  allLeavesTester(19, indices_reference);
}

// Test that the mean and variance are computed correctly for the toy data.
TEST_F(UtilityTest, Statistics) {
  std::vector<double> x {8.0, 4.0, 10.0, -8.0, 100.0, 13.0, 7.0};
  EXPECT_FLOAT_EQ(mean(x), 19.14286);
  EXPECT_FLOAT_EQ(var(x), 1316.143);
}
