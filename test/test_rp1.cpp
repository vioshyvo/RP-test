#include <random>

#include "gtest/gtest.h"
#include "Mrpt.h"
#include "Mrpt_old.h"
#include "Eigen/Dense"


namespace {

class MrptTest : public testing::Test {
  protected:

  MrptTest() : d(100), n(1024), seed_data(56789), seed_mrpt(12345) {
          std::mt19937 mt(seed_data);
          std::normal_distribution<double> dist(5.0,2.0);

          X = MatrixXf(d,n);
          for(int i = 0; i < d; ++i)
            for(int j = 0; j < n; ++j)
              X(i,j) = dist(mt);

          q = VectorXf(d);
          for(int i = 0; i < d; ++i) q(i) = dist(mt);
  }

  // Test that:
  // a) the indices of the returned approximate k-nn are same as before
  // b) approximate k-nn are returned in correct order
  // c) distances to the approximate k-nn are computed correctly
  void QueryTester(int n_trees, int depth, float density, int votes, int k,
      std::vector<int> approximate_knn) {
    ASSERT_EQ(approximate_knn.size(), k);
    const Map<const MatrixXf> *M = new Map<const MatrixXf>(X.data(), d, n);
    Mrpt index_dense(M, n_trees, depth, density);
    index_dense.grow(seed_mrpt);

    std::vector<int> result(k);
    std::vector<float> distances(k);
    for(int i = 0; i < k; ++i) distances[i] = 0;

    const Map<VectorXf> V(q.data(), d);
    index_dense.query(V, k, votes, &result[0], &distances[0]);

    for(int i = 0; i < k; ++i)  {
      EXPECT_EQ(result[i], approximate_knn[i]);
      if(i > 0) {
        EXPECT_LE(distances[i-1], distances[i]);
      }
      EXPECT_FLOAT_EQ(distances[i], (X.col(result[i]) - q).norm());
    }
  }

  void SplitPointTester(int n_trees, int depth, float density) {
    const Map<const MatrixXf> *M = new Map<const MatrixXf>(X.data(), d, n);
    Mrpt index(M, n_trees, depth, density);
    index.grow(seed_mrpt);
    Mrpt_old index_old(M, n_trees, depth, density);
    index_old.grow(seed_mrpt);


    for(int tree = 0; tree < n_trees; ++tree) {
      int per_level = 1, idx = 0;

      for(int level = 0; level < depth; ++level) {
        for(int j = 0; j < per_level; ++j) {
          float split = index.get_split_point(tree, idx);
          float split_old = index_old.get_split_point(tree, idx);
          ++idx;
          EXPECT_FLOAT_EQ(split, split_old);
        }
      }
      per_level *= 2;
    }
  }

  void LeafTester(int n_trees, int depth, float density) {
    const Map<const MatrixXf> *M = new Map<const MatrixXf>(X.data(), d, n);
    Mrpt index(M, n_trees, depth, density);
    index.grow(seed_mrpt);
    Mrpt_old index_old(M, n_trees, depth, density);
    index_old.grow(seed_mrpt);


    for(int tree = 0; tree < n_trees; ++tree) {
      int n_leaf = std::pow(2, depth);
      VectorXi leaves = VectorXi::Zero(n);

      for(int j = 0; j < n_leaf; ++j) {
        VectorXi leaf = index.get_leaf(tree, j);
        VectorXi leaf_old = index_old.get_leaf(tree, j);
        std::sort(leaf.data(), leaf.data() + leaf.size());
        std::sort(leaf_old.data(), leaf_old.data() + leaf_old.size());

        for(int i = 0; i < leaf.size(); ++i) {
          EXPECT_EQ(leaf_old(i), leaf(i));
          leaves(leaf(i)) = 1;
        }
      }

      // Test that all data points are found at a tree
      EXPECT_EQ(leaves.sum(), n);
    }
  }


  int d, n, seed_data, seed_mrpt;
  MatrixXf X;
  VectorXf q;
};


// Test that the nearest neighbors returned by the index
// are same as before when a seed for rng is fixed
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

  QueryTester(n_trees, depth, 0.01, votes, k, std::vector<int> {501, 566, 802, 84, 928});
  QueryTester(n_trees, depth, 1.0 / std::sqrt(d), votes, k, std::vector<int> {566, 882, 949, 802, 110});
  QueryTester(n_trees, depth, 0.5, votes, k, std::vector<int> {682, 882, 802, 115, 720});

  QueryTester(n_trees, depth, density, 1, k, std::vector<int> {541, 949, 720, 629, 84});
  QueryTester(n_trees, depth, density, 3, k, std::vector<int> {949, 629, 359, 109, 942});
  QueryTester(30, depth, density, 5, k, std::vector<int> {629, 84, 779, 838, 713});

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

  const Map<const MatrixXf> *M = new Map<const MatrixXf>(X.data(), d, n);

  Mrpt index_dense(M, n_trees, depth, density);
  index_dense.grow(); // initialize rng with random seed
  Mrpt index_dense2(M, n_trees, depth, density);
  index_dense2.grow();

  int k = 10, votes = 3;
  std::vector<int> r(k), r2(k);

  const Map<VectorXf> V(q.data(), d);

  index_dense.query(V, k, votes, &r[0]);
  index_dense2.query(V, k, votes, &r2[0]);

  bool same_neighbors = true;
  for(int i = 0; i < k; ++i) {
    if(r[i] != r2[i]) {
      same_neighbors = false;
      break;
    }
  }

  EXPECT_FALSE(same_neighbors);
}


// Test that the exact k-nn search works correctly
TEST_F(MrptTest, ExactKnn) {

  const Map<const MatrixXf> *M = new Map<const MatrixXf>(X.data(), d, n);
  Mrpt index_dense(M, 0, 0, 1);
  index_dense.grow();

  int k = 5;
  std::vector<int> result(k);
  std::vector<float> distances(k);

  const Map<VectorXf> V(q.data(), d);
  VectorXi idx(n);
  std::iota(idx.data(), idx.data() + n, 0);

  index_dense.exact_knn(V, k, idx, n, &result[0], &distances[0]);

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

TEST_F(MrptTest, SplitPoints) {
  int n_trees = 10, depth = 6;
  float density = 1.0 / std::sqrt(d);

  SplitPointTester(1, depth, density);
  SplitPointTester(5, depth, density);
  SplitPointTester(100, depth, density);

  SplitPointTester(n_trees, 1, density);
  SplitPointTester(n_trees, 3, density);
  SplitPointTester(n_trees, 8, density);
  SplitPointTester(n_trees, 10, density);

  SplitPointTester(n_trees, depth, 0.01);
  SplitPointTester(n_trees, depth, 0.5);
  SplitPointTester(n_trees, depth, 1);
}

TEST_F(MrptTest, Leaves) {
  int n_trees = 10, depth = 6;
  float density = 1.0 / std::sqrt(d);

  LeafTester(1, depth, density);
  LeafTester(5, depth, density);
  LeafTester(100, depth, density);

  LeafTester(n_trees, 1, density);
  LeafTester(n_trees, 3, density);
  LeafTester(n_trees, 8, density);
  LeafTester(n_trees, 10, density);

  LeafTester(n_trees, depth, 0.01);
  LeafTester(n_trees, depth, 0.5);
  LeafTester(n_trees, depth, 1);
}


}
