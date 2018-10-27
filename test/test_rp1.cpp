#include <random>

#include "gtest/gtest.h"
#include "Mrpt.h"
#include "Mrpt_old.h"
#include "Eigen/Dense"


namespace {

class MrptTest : public testing::Test {
  protected:

  MrptTest() : d(100), n(1024), n2(1255), seed_data(56789), seed_mrpt(12345) {
          std::mt19937 mt(seed_data);
          std::normal_distribution<double> dist(5.0,2.0);

          X = MatrixXf(d,n);
          for(int i = 0; i < d; ++i)
            for(int j = 0; j < n; ++j)
              X(i,j) = dist(mt);

          q = VectorXf(d);
          for(int i = 0; i < d; ++i) q(i) = dist(mt);

          X2 = MatrixXf(d,n2);
          for(int i = 0; i < d; ++i)
            for(int j = 0; j < n2; ++j)
              X2(i,j) = dist(mt);

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

  void TestSplitPoints(Mrpt &index, Mrpt_old &index_old) {
    int n_trees = index.get_n_trees();
    int n_trees_old = index_old.get_n_trees();
    ASSERT_EQ(n_trees, n_trees_old);

    int depth = index.get_depth(), depth_old = index_old.get_depth();
    ASSERT_EQ(depth, depth_old);

    for(int tree = 0; tree < n_trees; ++tree) {
      int per_level = 1, idx = 0;

      for(int level = 0; level < depth; ++level) {
        for(int j = 0; j < per_level; ++j) {
          float split = index.get_split_point(tree, idx);
          float split_old = index_old.get_split_point(tree, idx);
          ++idx;
          ASSERT_FLOAT_EQ(split, split_old);
        }
      }
      per_level *= 2;
    }
  }

  void SplitPointTester(int n_trees, int depth, float density,
        const Map<const MatrixXf> *M) {
    Mrpt index(M, n_trees, depth, density);
    index.grow(seed_mrpt);
    Mrpt_old index_old(M, n_trees, depth, density);
    index_old.grow(seed_mrpt);

    TestSplitPoints(index, index_old);
  }

  void TestLeaves(Mrpt &index, Mrpt_old &index_old) {
    int n_trees = index.get_n_trees();
    int n_trees_old = index_old.get_n_trees();
    ASSERT_EQ(n_trees, n_trees_old);

    int depth = index.get_depth(), depth_old = index_old.get_depth();
    ASSERT_EQ(depth, depth_old);

    int n_points = index.get_n_points();
    int n_points_old = index_old.get_n_points();
    ASSERT_EQ(n_points, n_points_old);

    for(int tree = 0; tree < n_trees; ++tree) {
      int n_leaf = std::pow(2, depth);
      VectorXi leaves = VectorXi::Zero(n_points);

      for(int j = 0; j < n_leaf; ++j) {
        int leaf_size = index.get_leaf_size(tree, j);
        int leaf_size_old = index_old.get_leaf_size(tree, j);
        ASSERT_EQ(leaf_size, leaf_size_old);

        std::vector<int> leaf(leaf_size), leaf_old(leaf_size);
        for(int i = 0; i < leaf_size; ++i) {
          leaf[i] = index.get_leaf_point(tree, j, i);
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

  void LeafTester(int n_trees, int depth, float density,
      const Map<const MatrixXf> *M) {
    Mrpt index(M, n_trees, depth, density);
    index.grow(seed_mrpt);
    Mrpt_old index_old(M, n_trees, depth, density);
    index_old.grow(seed_mrpt);

    TestLeaves(index, index_old);
  }

  void SaveIndex(int n_trees, int depth, float density,
      const Map<const MatrixXf> *M) {
    Mrpt_old index_old(M, n_trees, depth, density);
    index_old.grow(seed_mrpt);
    index_old.save("save/index_old.bin");
    Mrpt index_saved(M, n_trees, depth, density);
    index_saved.grow(seed_mrpt);
    index_saved.save("save/index_saved.bin");
  }

  // void generate_data(MatrixXf &XXX, int seed) {
  //   std::mt19937 mt(seed);
  //   std::normal_distribution<double> dist(5.0,2.0);
  //   int nnn = XXX.cols(), ddd = XXX.rows();
  //
  //   for(int i = 0; i < ddd; ++i)
  //     for(int j = 0; j < nnn; ++j)
  //       XXX(i,j) = dist(mt);
  // }


  int d, n, n2, seed_data, seed_mrpt;
  MatrixXf X, X2;
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

  QueryTester(n_trees, depth, 0.05, votes, k, std::vector<int> {566, 353, 199, 115, 84});
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

  const Map<const MatrixXf> *M = new Map<const MatrixXf>(X.data(), d, n);
  const Map<const MatrixXf> *M2 = new Map<const MatrixXf>(X2.data(), d, n2);

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

TEST_F(MrptTest, Leaves) {
  int n_trees = 10, depth = 6;
  float density = 1.0 / std::sqrt(d);

  const Map<const MatrixXf> *M = new Map<const MatrixXf>(X.data(), d, n);
  const Map<const MatrixXf> *M2 = new Map<const MatrixXf>(X2.data(), d, n2);

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

// TEST_F(MrptTest, Loading) {
//   // int n_trees = 10, depth = 6;
//   // float density = 1.0;
//   // const Map<const MatrixXf> *M = new Map<const MatrixXf>(X.data(), d, n);
//   // SaveIndex(n_trees, depth, density, M);
//
//   int nnn = 3750, ddd = 100, n_trees = 3, depth = 6, seed = 12345;
//   float density = 1.0 / std::sqrt(d);
//
//   MatrixXf XXX(ddd,nnn);
//   generate_data(XXX, seed);
//   const Map<const MatrixXf> *MMM = new Map<const MatrixXf>(XXX.data(), ddd, nnn);
//
//
//   Mrpt index_new(MMM, n_trees, depth, density);
//   index_new.load("mrpt_saved");
//   // TestLeaves(index_new, index_old);
//   // TestSplitPoints(index_new, index_old);
//   //
//   // Mrpt_old index_loaded(M, n_trees, depth, density);
//   // index_loaded.load("save/index_saved.bin");
//   // TestLeaves(index_saved, index_loaded);
//   // TestSplitPoints(index_saved, index_loaded);
// }

TEST(SaveTest, Loading) {
  int n = 3749, d = 100, n_trees = 3, depth = 6, seed = 12345, seed_mrpt = 56789;
  float density = 1.0 / std::sqrt(d);

  MatrixXf X(d,n);
  std::mt19937 mtt(seed);
  std::normal_distribution<double> distr(5.0,2.0);
  for(int i = 0; i < d; ++i)
    for(int j = 0; j < n; ++j)
      X(i,j) = distr(mtt);

  const Map<const MatrixXf> *M = new Map<const MatrixXf>(X.data(), d, n);
  Mrpt index(M, n_trees, depth, density);
  index.grow(12345);
  index.save("save/mrpt_saved");

  Mrpt index_old(M, n_trees, depth, density);
  index_old.load("save/mrpt_saved");

  ASSERT_EQ(n_trees, index_old.get_n_trees());
  ASSERT_EQ(depth, index_old.get_depth());
  ASSERT_EQ(n, index_old.get_n_points());

  for(int tree = 0; tree < n_trees; ++tree) {
    int n_leaf = std::pow(2, depth);
    VectorXi leaves = VectorXi::Zero(n);

    for(int j = 0; j < n_leaf; ++j) {
      int leaf_size = index.get_leaf_size(tree, j);
      int leaf_size_old = index_old.get_leaf_size(tree, j);
      ASSERT_EQ(leaf_size, leaf_size_old);

      std::vector<int> leaf(leaf_size), leaf_old(leaf_size);
      for(int i = 0; i < leaf_size; ++i) {
        leaf[i] = index.get_leaf_point(tree, j, i);
        leaf_old[i] = index_old.get_leaf_point(tree, j, i);
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
        float split = index.get_split_point(tree, idx);
        float split_old = index_old.get_split_point(tree, idx);
        ++idx;
        ASSERT_FLOAT_EQ(split, split_old);
      }
    }
    per_level *= 2;

    // Test that all data points are found at a tree
    EXPECT_EQ(leaves.sum(), n);
  }

}

}
