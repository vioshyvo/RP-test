#include <random>
#include <vector>
#include <algorithm>
#include <set>
#include <omp.h>
#include <numeric>
#include <utility>

#include "gtest/gtest.h"
#include "Mrpt.h"
#include "Mrpt_old.h"
#include "Eigen/Dense"

// Do not wrap the tests into a namespace, because otherwise the friend
// declarations of Mrpt.h would not work.
// namespace {

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

double median(std::vector<double> x) {
  int n = x.size();
  std::nth_element(x.begin(), x.begin() + n/2, x.end());

  if(n % 2) {
    return x[n/2];
  }

  double smaller = *std::max_element(x.begin(), x.begin() + n/2);
  return (smaller + x[n/2]) / 2.0;
}


class MrptTest : public testing::Test {
  protected:

  MrptTest() : d(100), n(1024), n2(155), n_test(100), seed_data(56789), seed_mrpt(12345) {
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

          M = new Map<const MatrixXf>(X.data(), d, n);
          M2 = new Map<const MatrixXf>(X2.data(), d, n2);
          test_queries = new Map<MatrixXf>(Q.data(), d, n_test);
          new (&test_query) Map<VectorXf>(q.data(), d);
  }

  ~MrptTest() {
    delete M;
    delete M2;
    delete test_queries;
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
  float get_split_point(const Mrpt &mrpt, int tree, int index) {
    return mrpt.split_points(index, tree);
  }


  /**
  * Accessor for point stored in leaves of trees (for testing purposes)
  * @param tree - index of tree in (0, ... T-1)
  * @param leaf - index of leaf in (0, ... , 2^depth)
  * @param index - index of a data point in a leaf
  * @return index of index:th data point in leaf:th leaf of tree:th tree
  */
  int get_leaf_point(const Mrpt &mrpt, int tree, int leaf, int index) {
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
  int get_leaf_size(const Mrpt &mrpt, int tree, int leaf) const {
    const std::vector<int> &leaf_first_indices = mrpt.leaf_first_indices_all[mrpt.depth];
    return leaf_first_indices[leaf + 1] - leaf_first_indices[leaf];
  }


  void autotuningGrowTester(float density, int trees_max, int depth_max,
        int depth_min, int votes_max, int k) {

    omp_set_num_threads(1);

    Mrpt index_at(M);
    index_at.grow(test_queries, k, trees_max, depth_max, depth_min, votes_max, density, seed_mrpt);

    Mrpt index_normal(M);
    index_normal.grow(trees_max, depth_max, density, seed_mrpt);

    TestSplitPoints(index_normal, index_at);
    TestLeaves(index_normal, index_at);
  }


  void autotuningTester(double target_recall, float density, int trees_max) {
    omp_set_num_threads(1);
    int depth_max = 7, depth_min = 5, votes_max = trees_max - 1, k = 5;

    Mrpt index_at(M);
    index_at.grow(test_queries, k, trees_max, depth_max, depth_min, votes_max, density, seed_mrpt);

    MatrixXi exact(k, n_test);
    compute_exact_neighbors(index_at, exact);

    Parameters par = index_at.parameters(target_recall);
    std::cout << std::endl;
    print_parameters(par);
    std::cout << std::endl;

    std::vector<std::vector<int>> res, res2, res3;
    double recall = 0;

    Mrpt index_new(M);
    index_at.subset_trees(target_recall, index_new);

    for(int i = 0; i < n_test; ++i) {
      std::vector<int> result(k, -1);
      const Map<VectorXf> q(Q.data() + i * d, d);

      index_new.query(q, &result[0]);
      res.push_back(result);

      std::sort(result.begin(), result.end());
      std::set<int> intersect;
      std::set_intersection(exact.data() + i * k, exact.data() + i * k + k, result.begin(), result.end(),
                       std::inserter(intersect, intersect.begin()));
      recall += intersect.size();
    }

    recall /= (k * n_test);

    double rec1 = get_recall(res, exact);
    EXPECT_FLOAT_EQ(recall, rec1);
    EXPECT_FLOAT_EQ(par.estimated_recall, rec1);

    Mrpt index2(M);
    index_at.subset_trees(target_recall, index2);

    for(int i = 0; i < n_test; ++i) {
      const Map<VectorXf> q(Q.data() + i * d, d);
      std::vector<int> result(k, -1);
      index2.query(q, &result[0]);
      res2.push_back(result);
    }

    EXPECT_EQ(res, res2); // Test that 2 subsetted indices with same target recall give the same results
    EXPECT_FLOAT_EQ(rec1, get_recall(res2, exact));

    index_at.delete_extra_trees(target_recall);

    for(int i = 0; i < n_test; ++i) {
      const Map<VectorXf> q(Q.data() + i * d, d);
      std::vector<int> result(k, -1);
      index_at.query(q, &result[0]);
      res3.push_back(result);
    }

    EXPECT_EQ(res, res3); // Test that the original index with extra trees deleted gives the same results
    EXPECT_FLOAT_EQ(rec1, get_recall(res3, exact));
  }


  void defaultArgumentTester(int k) {
    omp_set_num_threads(1);

    int trees_max = std::sqrt(n);
    int depth_max = std::log2(n) - 4;
    float density = 1.0 / std::sqrt(d);

    Mrpt index(M);
    index.grow(test_queries, k);

    EXPECT_EQ(index.n_trees, trees_max);
    EXPECT_EQ(index.depth, depth_max);
    EXPECT_FLOAT_EQ(index.density, density);
  }


  void saveTester(int n_trees, int depth, float density, int seed_mrpt) {

    Mrpt index(M2);
    index.grow(n_trees, depth, density, seed_mrpt);
    index.save("save/mrpt_saved");

    Mrpt index_reloaded(M2);
    index_reloaded.load("save/mrpt_saved");

    ASSERT_EQ(n_trees, index_reloaded.n_trees);
    ASSERT_EQ(depth, index_reloaded.depth);
    ASSERT_EQ(n2, index_reloaded.n_samples);

    for(int tree = 0; tree < n_trees; ++tree) {
      int n_leaf = std::pow(2, depth);
      VectorXi leaves = VectorXi::Zero(n2);

      for(int j = 0; j < n_leaf; ++j) {
        int leaf_size = get_leaf_size(index, tree, j);
        int leaf_size_old = get_leaf_size(index_reloaded,tree, j);
        ASSERT_EQ(leaf_size, leaf_size_old);

        std::vector<int> leaf(leaf_size), leaf_old(leaf_size);
        for(int i = 0; i < leaf_size; ++i) {
          leaf[i] = get_leaf_point(index, tree, j, i);
          leaf_old[i] = get_leaf_point(index_reloaded, tree, j, i);
        }
        std::sort(leaf.begin(), leaf.end());
        std::sort(leaf_old.begin(), leaf_old.end());

        for(int i = 0; i < leaf_size; ++i) {
          ASSERT_EQ(leaf_old[i], leaf[i]);
          leaves(leaf[i]) = 1;
        }
      }

      int per_level = 1, idx = 0;

      for(int level = 0; level < depth; ++level) {
        for(int j = 0; j < per_level; ++j) {
          float split = get_split_point(index, tree, idx);
          float split_old = get_split_point(index_reloaded, tree, idx);
          ++idx;
          ASSERT_FLOAT_EQ(split, split_old);
        }
      }
      per_level *= 2;

      // Test that all data points are found at a tree
      EXPECT_EQ(leaves.sum(), n2);
    }

  }


  void QueryTester(int n_trees, int depth, float density, int votes, int k,
      std::vector<int> approximate_knn) {
    ASSERT_EQ(approximate_knn.size(), k);

    Mrpt index_dense(M);
    index_dense.grow(n_trees, depth, density, seed_mrpt);

    std::vector<int> result(k);
    std::vector<float> distances(k);
    for(int i = 0; i < k; ++i) distances[i] = 0;

    int n_el = 0;
    index_dense.query(test_query, k, votes, &result[0], &distances[0], &n_el);

    EXPECT_EQ(result, approximate_knn);
    for(int i = 0; i < k; ++i)  {
      if(i > 0) {
        EXPECT_LE(distances[i-1], distances[i]);
      }
      if(result[i] >= 0) {
        EXPECT_FLOAT_EQ(distances[i], (X.col(result[i]) - q).norm());
      }
    }
  }

  void TestSplitPoints(Mrpt &index, Mrpt_old &index_old) {
    int n_trees = index.n_trees;
    int n_trees_old = index_old.get_n_trees();
    ASSERT_EQ(n_trees, n_trees_old);

    int depth = index.depth, depth_old = index_old.get_depth();
    ASSERT_EQ(depth, depth_old);

    for(int tree = 0; tree < n_trees; ++tree) {
      int per_level = 1, idx = 0;

      for(int level = 0; level < depth; ++level) {
        for(int j = 0; j < per_level; ++j) {
          float split = get_split_point(index, tree, idx);
          float split_old = index_old.get_split_point(tree, idx);
          ++idx;
          ASSERT_FLOAT_EQ(split, split_old);
        }
      }
      per_level *= 2;
    }
  }

  void TestSplitPoints(Mrpt &index, Mrpt &index_old) {
    int n_trees = index.n_trees;
    int n_trees_old = index_old.n_trees;
    ASSERT_EQ(n_trees, n_trees_old);

    int depth = index.depth, depth_old = index_old.depth;
    ASSERT_EQ(depth, depth_old);

    for(int tree = 0; tree < n_trees; ++tree) {
      int per_level = 1, idx = 0;

      for(int level = 0; level < depth; ++level) {
        for(int j = 0; j < per_level; ++j) {
          float split = get_split_point(index, tree, idx);
          float split_old = get_split_point(index_old, tree, idx);
          ++idx;
          ASSERT_FLOAT_EQ(split, split_old);
        }
      }
      per_level *= 2;
    }
  }


  void SplitPointTester(int n_trees, int depth, float density,
        const Map<const MatrixXf> *M) {
    Mrpt index(M);
    index.grow(n_trees, depth, density, seed_mrpt);
    Mrpt_old index_old(M, n_trees, depth, density);
    index_old.grow(seed_mrpt);

    TestSplitPoints(index, index_old);
  }

  void TestLeaves(Mrpt &index, Mrpt_old &index_old) {
    int n_trees = index.n_trees;
    int n_trees_old = index_old.get_n_trees();
    ASSERT_EQ(n_trees, n_trees_old);

    int depth = index.depth, depth_old = index_old.get_depth();
    ASSERT_EQ(depth, depth_old);

    int n_points = index.n_samples;
    int n_points_old = index_old.get_n_points();
    ASSERT_EQ(n_points, n_points_old);

    for(int tree = 0; tree < n_trees; ++tree) {
      int n_leaf = std::pow(2, depth);
      VectorXi leaves = VectorXi::Zero(n_points);

      for(int j = 0; j < n_leaf; ++j) {
        int leaf_size = get_leaf_size(index, tree, j);
        int leaf_size_old = index_old.get_leaf_size(tree, j);
        ASSERT_EQ(leaf_size, leaf_size_old);

        std::vector<int> leaf(leaf_size), leaf_old(leaf_size);
        for(int i = 0; i < leaf_size; ++i) {
          leaf[i] = get_leaf_point(index, tree, j, i);
          leaf_old[i] = index_old.get_leaf_point(tree, j, i);
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

  void TestLeaves(Mrpt &index, Mrpt &index_old) {
    int n_trees = index.n_trees;
    int n_trees_old = index_old.n_trees;
    ASSERT_EQ(n_trees, n_trees_old);

    int depth = index.depth, depth_old = index_old.depth;
    ASSERT_EQ(depth, depth_old);

    int n_points = index.n_samples;
    int n_points_old = index_old.n_samples;
    ASSERT_EQ(n_points, n_points_old);

    for(int tree = 0; tree < n_trees; ++tree) {
      int n_leaf = std::pow(2, depth);
      VectorXi leaves = VectorXi::Zero(n_points);

      for(int j = 0; j < n_leaf; ++j) {
        int leaf_size = get_leaf_size(index, tree, j);
        int leaf_size_old = get_leaf_size(index_old, tree, j);
        ASSERT_EQ(leaf_size, leaf_size_old);

        std::vector<int> leaf(leaf_size), leaf_old(leaf_size);
        for(int i = 0; i < leaf_size; ++i) {
          leaf[i] = get_leaf_point(index, tree, j, i);
          leaf_old[i] = get_leaf_point(index_old, tree, j, i);
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

  void LeafTester(int n_trees, int depth, float density,
      const Map<const MatrixXf> *M) {
    Mrpt index(M);
    index.grow(n_trees, depth, density, seed_mrpt);
    Mrpt_old index_old(M, n_trees, depth, density);
    index_old.grow(seed_mrpt);

    TestLeaves(index, index_old);
  }


  void compute_exact_neighbors(Mrpt &index, MatrixXi &out_exact) {
    int k = out_exact.rows();
    int nt = out_exact.cols();
    for(int i = 0; i < nt; ++i) {
      VectorXi idx(n);
      std::iota(idx.data(), idx.data() + n, 0);

      index.exact_knn(Map<VectorXf>(Q.data() + i * d, d), k, idx, n, out_exact.data() + i * k);
      std::sort(out_exact.data() + i * k, out_exact.data() + i * k + k);
    }
  }


  void print_parameters(const Parameters &op) {
    std::cout << "n_trees:                      " << op.n_trees << "\n";
    std::cout << "depth:                        " << op.depth << "\n";
    std::cout << "votes:                        " << op.votes << "\n";
    std::cout << "estimated query time:         " << op.estimated_qtime * 1000.0 << " ms.\n";
    std::cout << "estimated recall:             " << op.estimated_recall << "\n";
  }

  double get_recall(std::vector<std::vector<int>> results, MatrixXi exact) {
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

  int d, n, n2, n_test, seed_data, seed_mrpt;
  MatrixXf X, X2, Q;
  VectorXf q;
  const Map<const MatrixXf> *M, *M2;
  Map<MatrixXf> *test_queries;
  Map<VectorXf> test_query = Map<VectorXf>(nullptr, 0);
};


// Test that:
// a) the indices of the returned approximate k-nn are same as before
// b) approximate k-nn are returned in correct order
// c) distances to the approximate k-nn are computed correctly
TEST_F(MrptTest, Query) {
  int n_trees = 10, depth = 6, votes = 1, k = 5;
  float density = 1;

  QueryTester(1, depth, density, votes, k, std::vector<int> {949, 84, 136, 133, 942});
  QueryTester(5, depth, density, votes, k, std::vector<int> {949, 720, 84, 959, 447});
  QueryTester(100, depth, density, votes, k, std::vector<int> {501, 682, 566, 541, 747});

  QueryTester(1, 0, density, votes, k, std::vector<int> {501, 682, 566, 541, 747});
  QueryTester(n_trees, 1, density, votes, k, std::vector<int> {501, 682, 566, 541, 747});
  QueryTester(n_trees, 3, density, votes, k, std::vector<int> {501, 682, 541, 747, 882});
  QueryTester(n_trees, 8, density, votes, k, std::vector<int> {949, 629, 860, 954, 121});
  QueryTester(n_trees, 10, density, votes, k, std::vector<int> {949, 713, 574, 88, 900});

  QueryTester(n_trees, depth, 0.05, votes, k, std::vector<int> {566, 353, 199, 115, 84});
  QueryTester(n_trees, depth, 1.0 / std::sqrt(d), votes, k, std::vector<int> {566, 882, 949, 802, 110});
  QueryTester(n_trees, depth, 0.5, votes, k, std::vector<int> {682, 882, 802, 115, 720});

  QueryTester(n_trees, depth, density, 1, k, std::vector<int> {541, 949, 720, 629, 84});
  QueryTester(n_trees, depth, density, 3, k, std::vector<int> {-1, -1, -1, -1, -1});
  QueryTester(30, depth, density, 5, k, std::vector<int> {-1, -1, -1, -1, -1});

  QueryTester(n_trees, depth, density, votes, 1, std::vector<int> {541});
  QueryTester(n_trees, depth, density, votes, 2, std::vector<int> {541, 949});
  QueryTester(n_trees, depth, density, votes, 10,
      std::vector<int> {541, 949, 720, 629, 84, 928, 959, 438, 372, 447});

}


// Test that the nearest neighbors returned by the index are different
// when rng is initialized with a random seed (no seed is given to
// grow() - method). Obs. this test may fail with a very small probability
// if the nearest neighbors returned by two different indices happen
// to be exactly same by change.
TEST_F(MrptTest, RandomSeed) {
  int n_trees = 10, depth = 6;
  float density = 1.0 / std::sqrt(d);

  Mrpt index_dense(M);
  index_dense.grow(n_trees, depth, density); // initialize rng with random seed
  Mrpt index_dense2(M);
  index_dense2.grow(n_trees, depth, density);

  int k = 10, votes = 1;
  std::vector<int> r(k), r2(k);

  index_dense.query(test_query, k, votes, &r[0]);
  index_dense2.query(test_query, k, votes, &r2[0]);

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

  Mrpt index_dense(M);
  index_dense.grow(0, 0, 1.0);

  int k = 5;
  std::vector<int> result(k);
  std::vector<float> distances(k);

  VectorXi idx(n);
  std::iota(idx.data(), idx.data() + n, 0);

  index_dense.exact_knn(test_query, k, idx, n, &result[0], &distances[0]);

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

// Test that the split points of trees are identical to the split n_points
// generated by the reference implementation.
TEST_F(MrptTest, SplitPoints) {
  int n_trees = 10, depth = 6;
  float density = 1.0 / std::sqrt(d);

  SplitPointTester(1, depth, density, M);
  SplitPointTester(5, depth, density, M);
  SplitPointTester(100, depth, density, M);
  SplitPointTester(1, depth, density, M2);
  SplitPointTester(5, depth, density, M2);
  SplitPointTester(100, depth, density, M2);

  SplitPointTester(n_trees, 1, density, M);
  SplitPointTester(n_trees, 3, density, M);
  SplitPointTester(n_trees, 8, density, M);
  SplitPointTester(n_trees, 10, density, M);
  SplitPointTester(n_trees, 1, density, M2);
  SplitPointTester(n_trees, 3, density, M2);
  SplitPointTester(n_trees, 6, density, M2);
  SplitPointTester(n_trees, 7, density, M2);

  SplitPointTester(n_trees, depth, 0.05, M);
  SplitPointTester(n_trees, depth, 0.5, M);
  SplitPointTester(n_trees, depth, 1, M);
  SplitPointTester(n_trees, depth, 0.05, M2);
  SplitPointTester(n_trees, depth, 0.5, M2);
  SplitPointTester(n_trees, depth, 1, M2);
}


// Test that the leaves of the trees are identical to the leaves generated
// by the reference implementation.
TEST_F(MrptTest, Leaves) {
  int n_trees = 10, depth = 6;
  float density = 1.0 / std::sqrt(d);

  LeafTester(1, depth, density, M);
  LeafTester(5, depth, density, M);
  LeafTester(100, depth, density, M);
  LeafTester(1, depth, density, M2);
  LeafTester(5, depth, density, M2);
  LeafTester(100, depth, density, M2);

  LeafTester(n_trees, 1, density, M);
  LeafTester(n_trees, 3, density, M);
  LeafTester(n_trees, 8, density, M);
  LeafTester(n_trees, 10, density, M);
  LeafTester(n_trees, 1, density, M2);
  LeafTester(n_trees, 3, density, M2);
  LeafTester(n_trees, 6, density, M2);
  LeafTester(n_trees, 7, density, M2);

  LeafTester(n_trees, depth, 0.05, M);
  LeafTester(n_trees, depth, 0.5, M);
  LeafTester(n_trees, depth, 1, M);
  LeafTester(n_trees, depth, 0.05, M2);
  LeafTester(n_trees, depth, 0.5, M2);
  LeafTester(n_trees, depth, 1, M2);
}


// Test that the loaded index is identical to the original one that was saved.
TEST_F(MrptTest, Saving) {
  int n_trees = 3, depth = 6, seed_mrpt = 12345;
  float density = 1.0 / std::sqrt(d);

  saveTester(n_trees, depth, density, seed_mrpt);
  saveTester(n_trees, depth, 1.0, seed_mrpt);
  saveTester(1, depth, density, seed_mrpt);
}

// Test that an index grown with autotuning gives the same trees as the index grown
// with an old school grow-function
TEST_F(MrptTest, AutotuningGrowing) {
  int trees_max = 10, depth_max = 6, depth_min = 4, votes_max = std::max(1, trees_max - 1), k = 5;
  float density = 1.0 / std::sqrt(d);

  autotuningGrowTester(1.0 / std::sqrt(d), trees_max, depth_max, depth_min, votes_max, k);
  autotuningGrowTester(1.0, trees_max, depth_max, depth_min, votes_max, k);

  autotuningGrowTester(density, 2, depth_max, depth_min, votes_max, k);
  autotuningGrowTester(density, 5, depth_max, depth_min, votes_max, k);
  autotuningGrowTester(density, 100, depth_max, depth_min, votes_max, k);

  autotuningGrowTester(density, trees_max, 7, 5, votes_max, k);
  autotuningGrowTester(density, trees_max, 3, 2, votes_max, k);
  autotuningGrowTester(density, trees_max, 3, 1, votes_max, k);
  autotuningGrowTester(density, trees_max, 1, 1, votes_max, k);
  autotuningGrowTester(density, trees_max, 0, 0, votes_max, k);

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
TEST_F(MrptTest, EmptyIndex) {
  int k = 5, v = 1;
  Mrpt mrpt(M);
  std::vector<int> res(k), res_empty(k, -1);
  std::vector<float> distances(k), distances_empty(k, -1);
  int n_elected, n_elected_empty = 0;

  mrpt.query(test_query, k, v, &res[0]);
  EXPECT_EQ(res, res_empty);

  mrpt.query(test_query, k, v, &res[0], &distances[0]);
  EXPECT_EQ(res, res_empty);
  for(int i = 0; i < k; ++i)
    EXPECT_FLOAT_EQ(distances[i], distances_empty[i]);

  mrpt.query(test_query, k, v, &res[0], &distances[0], &n_elected);
  EXPECT_EQ(res, res_empty);
  for(int i = 0; i < k; ++i)
    EXPECT_FLOAT_EQ(distances[i], distances_empty[i]);
  EXPECT_EQ(n_elected, n_elected_empty);
}

// Test that the normal query function works also on the autotuned index
TEST_F(MrptTest, NormalQuery) {
  int trees_max = 10, depth_max = 7, depth_min = 5, votes_max = trees_max - 1, k = 5;
  float density = 1.0 / std::sqrt(d);

  Mrpt mrpt_at(M2);
  mrpt_at.grow(test_queries, k, trees_max, depth_max, depth_min, votes_max, density, seed_mrpt);

  Mrpt mrpt_at2(M2);
  mrpt_at2.grow(test_queries, 10, trees_max, depth_max, depth_min, votes_max, density, seed_mrpt);

  Mrpt mrpt(M2);
  mrpt.grow(trees_max, depth_max, density, seed_mrpt);

  int v = 2;
  std::vector<std::vector<int>> res, res_at, res_at2;

  for(int i = 0; i < n_test; ++i) {
    const Map<VectorXf> q(Q.data() + i * d, d);
    std::vector<int> result(k), result_at(k), result_at2(k);
    mrpt_at.query(q, k, v, &result_at[0]);
    mrpt_at.query(q, k, v, &result_at2[0]);
    mrpt.query(q, k, v, &result[0]);
    res_at.push_back(result_at);
    res_at2.push_back(result_at2);
    res.push_back(result);
  }

  EXPECT_EQ(res, res_at);
  EXPECT_EQ(res, res_at2);
}



class UtilityTest : public testing::Test {
  protected:

  UtilityTest() {}

  void LeafTester(int n, int depth, const std::vector<int> &indices_reference) {
    std::vector<int> indices;
    Mrpt::count_first_leaf_indices(indices, n, depth);
    EXPECT_EQ(indices, indices_reference);
  }

  void AllLeavesTester(int n, const std::vector<std::vector<int>> &indices_reference) {
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
    LeafTester(19, depth, indices_reference[depth]);

  AllLeavesTester(19, indices_reference);
}

// Test that the mean and variance are computed correctly for the toy data.
TEST_F(UtilityTest, Statistics) {
  std::vector<double> x {8.0, 4.0, 10.0, -8.0, 100.0, 13.0, 7.0};
  EXPECT_FLOAT_EQ(mean(x), 19.14286);
  EXPECT_FLOAT_EQ(var(x), 1316.143);
}

// }
