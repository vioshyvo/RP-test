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


  void autotuningTester(double target_recall, float density, int trees_max) {
    omp_set_num_threads(1);
    int depth_max = 7, depth_min = 5, votes_max = trees_max - 1, k = 5;

    Mrpt mrpt(M);
    mrpt.grow(test_queries, k, trees_max, depth_max, depth_min, votes_max, density, seed_mrpt);

    MatrixXi exact(k, n_test);
    computeExactNeighbors(mrpt, exact, n);

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


  void defaultArgumentTester(int k) {
    omp_set_num_threads(1);

    int trees_max = std::sqrt(n);
    int depth_max = std::log2(n) - 4;
    float density = 1.0 / std::sqrt(d);

    Mrpt mrpt(M);
    mrpt.grow(test_queries, k);

    EXPECT_EQ(mrpt.n_trees, trees_max);
    EXPECT_EQ(mrpt.depth, depth_max);
    EXPECT_FLOAT_EQ(mrpt.density, density);
  }

  void testParameters(const Mrpt_Parameters &par, const Mrpt_Parameters &par2) {
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
      testParameters(*it, *it2);
  }

  void saveTester(int n_trees, int depth, float density, int seed_mrpt) {
    Mrpt mrpt(M2);
    mrpt.grow(n_trees, depth, density, seed_mrpt);
    mrpt.save("save/mrpt_saved");

    Mrpt mrpt_reloaded(M2);
    mrpt_reloaded.load("save/mrpt_saved");

    splitPointsEqual(mrpt, mrpt_reloaded);
    leavesEqual(mrpt, mrpt_reloaded);
    normalQueryTester(mrpt, mrpt_reloaded, 5, 1);
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
    normalQueryTester(mrpt, mrpt_reloaded, 5, 1);
    testParameters(mrpt.parameters(), mrpt_reloaded.parameters());
    testOptimalParameters(mrpt.optimal_pars(), mrpt_reloaded.optimal_pars());
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
    normalQueryTester(mrpt, mrpt_reloaded, 5, 1);
    testParameters(mrpt.parameters(), mrpt_reloaded.parameters());
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

  void computeExactNeighbors(Mrpt &mrpt, MatrixXi &out_exact, int n) {
    int k = out_exact.rows();
    int nt = out_exact.cols();

    for(int i = 0; i < nt; ++i) {
      VectorXi idx(n);
      std::iota(idx.data(), idx.data() + n, 0);

      mrpt.exact_knn(Map<VectorXf>(Q.data() + i * d, d), k, idx, n, out_exact.data() + i * k);
      std::sort(out_exact.data() + i * k, out_exact.data() + i * k + k);
    }
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

      std::sort(result.begin(), result.end());
      std::sort(exact.data() + i * k, exact.data() + i * k + k);
      std::set<int> intersect;
      std::set_intersection(exact.data() + i * k, exact.data() + i * k + k, result.begin(), result.end(),
                       std::inserter(intersect, intersect.begin()));

      recall += intersect.size();
    }
    return recall / (k * n_test);
  }

  std::vector<std::vector<int>> normalQuery(const Mrpt &mrpt, int k, int v) {
    std::vector<std::vector<int>> res;
    for(int i = 0; i < n_test; ++i) {
      std::vector<int> result(k);
      mrpt.query(Q.col(i), k, v, &result[0]);
      res.push_back(result);
    }
    return res;
  }

  std::vector<std::vector<int>> autotuningQuery(const Mrpt &mrpt) {
    std::vector<std::vector<int>> res;
    for(int i = 0; i < n_test; ++i) {
      std::vector<int> result(mrpt.parameters().k);
      mrpt.query(Q.col(i), &result[0]);
      res.push_back(result);
    }
    return res;
  }

  void normalQueryTester(const Mrpt &mrpt1, const Mrpt &mrpt2, int k, int v) {
    std::vector<std::vector<int>> res1 = normalQuery(mrpt1, k, v);
    std::vector<std::vector<int>> res2 = normalQuery(mrpt2, k, v);

    EXPECT_EQ(res1, res2);
  }

  int d, n, n2, n_test, seed_data, seed_mrpt;
  double epsilon = 0.001; // error bound for floating point comparisons of recall
  MatrixXf X, X2, Q;
  VectorXf q;
  Map<const MatrixXf> M, M2;
  Map<const MatrixXf> test_queries;
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


// Test that the exact k-nn search returns true nearest neighbors
TEST_F(MrptTest, ExactKnn) {

  Mrpt mrpt(M);

  int k = 5;
  std::vector<int> result(k);
  std::vector<float> distances(k);

  VectorXi idx(n);
  std::iota(idx.data(), idx.data() + n, 0);

  mrpt.exact_knn(q, k, idx, n, &result[0], &distances[0]);

  EXPECT_EQ(result[0], 501);
  EXPECT_EQ(result[1], 682);
  EXPECT_EQ(result[2], 566);
  EXPECT_EQ(result[3], 541);
  EXPECT_EQ(result[4], 747);

  for(int i = 0; i < k; ++i) {
    float distance_true = (X.col(result[i]) - q).norm();
    EXPECT_FLOAT_EQ(distances[i], distance_true);
  }

  VectorXf dd(n);
  for(int i = 0; i < n; ++i)
    dd(i) = (X.col(i) - q).norm();

  std::partial_sort(idx.data(), idx.data() + k, idx.data() + n,
    [&dd](int i, int j) { return dd(i) < dd(j); });

  // test that the k nearest neighbors returned by exact-knn are true
  // k nearest neighbors
  for(int i = 0; i < k; ++i) {
    EXPECT_EQ(result[i], idx[i]);
    EXPECT_FLOAT_EQ(distances[i], dd(idx(i)));
  }

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
  normalQueryTester(mrpt, mrpt_at, k, v);
  normalQueryTester(mrpt, mrpt_at2, k, v);
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
  mrpt_at.prune(target_recall);

  int v = 2;
  normalQueryTester(mrpt_at, mrpt_at2, k, v);
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

// Test that normal tree growing returns throws a correct expection when
// the parameters are out of bounds
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
  EXPECT_NO_THROW(mrpt2.grow(n_trees, depth, 1.0));
  EXPECT_NO_THROW(mrpt2.grow(n_trees, depth));
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

  EXPECT_NO_THROW(mrpt.grow(test_queries, 1));
  EXPECT_FALSE(mrpt.empty());

  Mrpt mrpt2(M2);
  EXPECT_NO_THROW(mrpt2.grow(test_queries, n2));
  EXPECT_FALSE(mrpt2.empty());
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

  EXPECT_NO_THROW(mrpt.grow(test_queries, k, trees_max, -1));
  EXPECT_FALSE(mrpt.empty());

  Mrpt mrpt2(M2);
  EXPECT_NO_THROW(mrpt2.grow(test_queries, k, trees_max, std::log2(n2)));
  EXPECT_FALSE(mrpt2.empty());
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

  EXPECT_NO_THROW(mrpt.grow(test_queries, k, trees_max, depth_max, -1));
  EXPECT_FALSE(mrpt.empty());

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

  EXPECT_NO_THROW(mrpt.grow(test_queries, k, trees_max, depth_max, depth_min, -1));
  EXPECT_FALSE(mrpt.empty());

  Mrpt mrpt2(M2);
  EXPECT_NO_THROW(mrpt2.grow(test_queries, k, trees_max, depth_max, depth_min, trees_max));
  EXPECT_FALSE(mrpt2.empty());
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

  EXPECT_NO_THROW(mrpt.grow(test_queries, k, trees_max, depth_max, depth_min, votes_max, -1.0));
  EXPECT_FALSE(mrpt.empty());

  Mrpt mrpt2(M2);
  EXPECT_NO_THROW(mrpt2.grow(test_queries, k, trees_max, depth_max, depth_min, votes_max, 1.0));
  EXPECT_FALSE(mrpt2.empty());

  Mrpt mrpt3(M2);
  EXPECT_NO_THROW(mrpt3.grow(test_queries, k, trees_max, depth_max, depth_min, votes_max, 0.001));
  EXPECT_FALSE(mrpt3.empty());
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

// Test that the autotuning function throws an out-of-range exception if
// the target recall level is not on the interval [0,1].
TEST_F(MrptTest, AutotuningTargetRecallThrows) {
  int k = 5;
  Mrpt mrpt(M2);

  EXPECT_THROW(mrpt.grow(-0.01, test_queries, k), std::out_of_range);
  EXPECT_THROW(mrpt.grow(1.01, test_queries, k), std::out_of_range);

  EXPECT_NO_THROW(mrpt.grow(0, test_queries, k));

  Mrpt mrpt2(M2);
  EXPECT_NO_THROW(mrpt2.grow(1, test_queries, k));
}

// Test that the function that prunes the autotuned index throws an
// out-of-range exception if the target recall level is not on the interval [0,1].
TEST_F(MrptTest, PruningTargetRecallThrows) {
  int k = 5;
  Mrpt mrpt(M2);
  mrpt.grow(test_queries, k);

  EXPECT_THROW(mrpt.prune(-0.01), std::out_of_range);
  EXPECT_THROW(mrpt.prune(1.01), std::out_of_range);

  EXPECT_NO_THROW(mrpt.prune(0));

  Mrpt mrpt2(M2);
  EXPECT_NO_THROW(mrpt2.prune(1));
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
  EXPECT_NO_THROW(Mrpt mrpt_new(mrpt.subset(1)));
}



// Test that when the index is not yet built, the parameter getter returns
// default values for the parameters and the estimated query time and the
// estimated recall.
TEST_F(MrptTest, ParameterGetterEmptyIndex) {
  Mrpt mrpt(M2);
  testParameters(mrpt.parameters(), Mrpt_Parameters());
}

// Test that when the index is not autotuned, the getter for parameters returns
// the number and depth of trees, but not vote threshold and estimated query
// time and recall level.
TEST_F(MrptTest, ParameterGetter) {
  Mrpt mrpt(M2);
  int n_trees = 10, depth = 6;
  mrpt.grow(n_trees, depth);
  testParameters(mrpt.parameters(), {n_trees, depth, 0, 0, 0.0, 0.0});
}

// Test that when the index is autotuned, but the target recall level
// is not yet set, the getter for parameters returns the maximum number and
// maximum depth of the trees, but not vote threshold and estimated query time
// and recall level.
TEST_F(MrptTest, ParameterGetterAutotuning) {
  Mrpt mrpt(M2);
  int k = 5, trees_max = 8, depth_max = 7;
  mrpt.grow(test_queries, k, trees_max, depth_max);
  testParameters(mrpt.parameters(), {trees_max, depth_max, 0, k, 0.0, 0.0});
}

// Test that the getter for the list of optimal parameters throws a
// logic expection when called on either an empty index or an index which
// has not been autotuned.
TEST_F(MrptTest, NotAutotunedOptimalParameterGetterThrows) {
  Mrpt mrpt(M2);

  EXPECT_THROW(mrpt.optimal_pars(), std::logic_error);

  mrpt.grow(10, 6);
  EXPECT_THROW(mrpt.optimal_pars(), std::logic_error);
}

// Test that the getter for the list of optimal parameters throws a
// logic expection when called on an index which is autotuned to the
// target recall level.
TEST_F(MrptTest, AutotunedOptimalParameterGetterThrows) {
  Mrpt mrpt(M2);
  mrpt.grow(0.2, test_queries, 5);
  EXPECT_THROW(mrpt.optimal_pars(), std::logic_error);
}

// Test that the getter for the list of optimal parameters throws a
// logic expection when called on an index which is autotuned , but also
// already pruned to the target recall level.
TEST_F(MrptTest, PrunedOptimalParameterGetterThrows) {
  Mrpt mrpt(M2);
  mrpt.grow(test_queries, 5);
  mrpt.prune(0.2);
  EXPECT_THROW(mrpt.optimal_pars(), std::logic_error);
}

// Test that the getter for the list of optimal parameters throws a
// logic expection when called on an index which is subsetted from
// the autotuned index, but does not throw when called on the original
// autotuned index.
TEST_F(MrptTest, SubsettedOptimalParameterGetterThrows) {
  Mrpt mrpt(M2);
  mrpt.grow(test_queries, 5);

  Mrpt mrpt_subsetted(mrpt.subset(0.2));
  EXPECT_THROW(mrpt_subsetted.optimal_pars(), std::logic_error);
  EXPECT_NO_THROW(mrpt.optimal_pars());
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
  Mrpt mrpt(M2);
  MatrixXi exact2(k, n_test);
  computeExactNeighbors(mrpt, exact2, n2);

  mrpt.grow(test_queries, k, 20, 7, 3, 10, 1.0 / std::sqrt(d), seed_mrpt);

  std::vector<Mrpt_Parameters> pars = mrpt.optimal_pars();
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
  Mrpt mrpt_exact(M2);

  MatrixXi exact2(k, n_test);
  computeExactNeighbors(mrpt_exact, exact2, n2);

  std::vector<double> target_recalls {0.1, 0.5, 0.9, 0.99};
  for(const auto &tr : target_recalls) {
    Mrpt mrpt(M2);
    mrpt.grow(test_queries, k, 20, 7, 3, 10, 1.0 / std::sqrt(d), seed_mrpt);

    std::vector<Mrpt_Parameters> pars = mrpt.optimal_pars();
    double highest_estimated_recall = pars.rbegin()->estimated_recall;

    mrpt.prune(tr);
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
  Mrpt mrpt_exact(M2);

  MatrixXi exact2(k, n_test);
  computeExactNeighbors(mrpt_exact, exact2, n2);

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