#include <random>
#include <vector>
#include <algorithm>
#include <set>
#include <omp.h>
#include <numeric>

#include "gtest/gtest.h"
#include "Mrpt.h"
#include "Mrpt_old.h"
#include "Eigen/Dense"

using namespace Eigen;

class MrptTest : public testing::Test {
  protected:

  MrptTest() : d(100), n(1024), n2(155), n_test(100), seed_data(56789), seed_mrpt(12345),
    M(nullptr, 0, 0), M2(nullptr, 0, 0) {
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

          new (&M) Map<const MatrixXf>(X.data(), d, n);
          new (&M2) Map<const MatrixXf>(X2.data(), d, n2);
          M2_pointer = new Map<const MatrixXf>(X2.data(), d, n2);
  }

  ~MrptTest() {
    delete M2_pointer;
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


  void queryTester(int n_trees, int depth, float density, int votes, int k,
      std::vector<int> approximate_knn) {
    ASSERT_EQ(approximate_knn.size(), k);

    Mrpt index_dense(M);
    index_dense.grow(n_trees, depth, density, seed_mrpt);

    std::vector<int> result(k);
    std::vector<float> distances(k);
    for(int i = 0; i < k; ++i) distances[i] = 0;

    int n_el = 0;
    index_dense.query(q, k, votes, &result[0], &distances[0], &n_el);

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

  void testSplitPoints(Mrpt &index, Mrpt_old &index_old) {
    int n_trees = index.n_trees;
    int n_trees_old = index_old.get_n_trees();
    ASSERT_EQ(n_trees, n_trees_old);

    int depth = index.depth, depth_old = index_old.get_depth();
    ASSERT_EQ(depth, depth_old);

    for(int tree = 0; tree < n_trees; ++tree) {
      int per_level = 1, idx = 0;

      for(int level = 0; level < depth; ++level) {
        for(int j = 0; j < per_level; ++j) {
          float split = getSplitPoint(index, tree, idx);
          float split_old = index_old.get_split_point(tree, idx);
          ++idx;
          ASSERT_FLOAT_EQ(split, split_old);
        }
      }
      per_level *= 2;
    }
  }


  void splitPointTester(int n_trees, int depth, float density) {
    Mrpt index(M2);
    index.grow(n_trees, depth, density, seed_mrpt);
    Mrpt_old index_old(M2_pointer, n_trees, depth, density);
    index_old.grow(seed_mrpt);

    testSplitPoints(index, index_old);
  }

  void testLeaves(Mrpt &index, Mrpt_old &index_old) {
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
        int leaf_size = getLeafSize(index, tree, j);
        int leaf_size_old = index_old.get_leaf_size(tree, j);
        ASSERT_EQ(leaf_size, leaf_size_old);

        std::vector<int> leaf(leaf_size), leaf_old(leaf_size);
        for(int i = 0; i < leaf_size; ++i) {
          leaf[i] = getLeafPoint(index, tree, j, i);
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


  void leafTester(int n_trees, int depth, float density) {
    Mrpt index(M2);
    index.grow(n_trees, depth, density, seed_mrpt);
    Mrpt_old index_old(M2_pointer, n_trees, depth, density);
    index_old.grow(seed_mrpt);

    testLeaves(index, index_old);
  }

  int d, n, n2, n_test, seed_data, seed_mrpt;
  MatrixXf X, X2;
  VectorXf q;
  Map<const MatrixXf> M, M2;
  const Map<const MatrixXf> *M2_pointer;
};


// Test that:
// a) the indices of the returned approximate k-nn are same as before
// b) approximate k-nn are returned in correct order
// c) distances to the approximate k-nn are computed correctly
TEST_F(MrptTest, Query) {
  int n_trees = 10, depth = 6, votes = 1, k = 5;
  float density = 1;

  queryTester(1, depth, density, votes, k, std::vector<int> {949, 84, 136, 133, 942});
  queryTester(5, depth, density, votes, k, std::vector<int> {949, 720, 84, 959, 447});
  queryTester(100, depth, density, votes, k, std::vector<int> {501, 682, 566, 541, 747});

  queryTester(n_trees, 1, density, votes, k, std::vector<int> {501, 682, 566, 541, 747});
  queryTester(n_trees, 3, density, votes, k, std::vector<int> {501, 682, 541, 747, 882});
  queryTester(n_trees, 8, density, votes, k, std::vector<int> {949, 629, 860, 954, 121});
  queryTester(n_trees, 10, density, votes, k, std::vector<int> {949, 713, 574, 88, 900});

  queryTester(n_trees, depth, 0.05, votes, k, std::vector<int> {566, 353, 199, 115, 84});
  queryTester(n_trees, depth, 1.0 / std::sqrt(d), votes, k, std::vector<int> {566, 882, 949, 802, 110});
  queryTester(n_trees, depth, 0.5, votes, k, std::vector<int> {682, 882, 802, 115, 720});

  queryTester(n_trees, depth, density, 1, k, std::vector<int> {541, 949, 720, 629, 84});
  queryTester(n_trees, depth, density, 3, k, std::vector<int> {-1, -1, -1, -1, -1});
  queryTester(30, depth, density, 5, k, std::vector<int> {-1, -1, -1, -1, -1});

  queryTester(n_trees, depth, density, votes, 1, std::vector<int> {541});
  queryTester(n_trees, depth, density, votes, 2, std::vector<int> {541, 949});
  queryTester(n_trees, depth, density, votes, 10,
      std::vector<int> {541, 949, 720, 629, 84, 928, 959, 438, 372, 447});

}


// Test that the split points of trees are identical to the split n_points
// generated by the reference implementation.
TEST_F(MrptTest, SplitPoints) {
  int n_trees = 10, depth = 6;
  float density = 1.0 / std::sqrt(d);

  splitPointTester(1, depth, density);
  splitPointTester(5, depth, density);
  splitPointTester(100, depth, density);

  splitPointTester(n_trees, 1, density);
  splitPointTester(n_trees, 3, density);
  splitPointTester(n_trees, 6, density);
  splitPointTester(n_trees, 7, density);

  splitPointTester(n_trees, depth, 0.05);
  splitPointTester(n_trees, depth, 0.5);
  splitPointTester(n_trees, depth, 1);
}


// Test that the leaves of the trees are identical to the leaves generated
// by the reference implementation.
TEST_F(MrptTest, Leaves) {
  int n_trees = 10, depth = 6;
  float density = 1.0 / std::sqrt(d);

  leafTester(1, depth, density);
  leafTester(5, depth, density);
  leafTester(100, depth, density);

  leafTester(n_trees, 1, density);
  leafTester(n_trees, 3, density);
  leafTester(n_trees, 6, density);
  leafTester(n_trees, 7, density);

  leafTester(n_trees, depth, 0.05);
  leafTester(n_trees, depth, 0.5);
  leafTester(n_trees, depth, 1);
}
