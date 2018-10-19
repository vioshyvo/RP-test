// Step 1. Include necessary header files such that the stuff your
// test logic needs is declared.
//
// Don't forget gtest.h, which declares the testing framework.

#include <random>

#include "rp1.h"
#include "gtest/gtest.h"
#include "Mrpt.h"
#include "Eigen/Dense"

// Step 2. Use the TEST macro to define your tests.
//
// TEST has two parameters: the test case name and the test name.
// After using the macro, you should define your test logic between a
// pair of braces.  You can use a bunch of macros to indicate the
// success or failure of a test.  EXPECT_TRUE and EXPECT_EQ are
// examples of such macros.  For a complete list, see gtest.h.
//
// <TechnicalDetails>
//
// In Google Test, tests are grouped into test cases.  This is how we
// keep test code organized.  You should put logically related tests
// into the same test case.
//
// The test case name and the test name should both be valid C++
// identifiers.  And you should not use underscore (_) in the names.
//
// Google Test guarantees that each test you define is run exactly
// once, but it makes no guarantee on the order the tests are
// executed.  Therefore, you should write your tests in such a way
// that their results don't depend on their order.
//
// </TechnicalDetails>


namespace {

TEST(TrivialTest, Addition) {
  EXPECT_EQ(add(1,1), 1+1);
  EXPECT_TRUE(add(1,1) == 1+1);
  EXPECT_GT(add(1,1), 1);
  EXPECT_LT(add(1,1), 3);
}

TEST(TrivialTest, Substraction) {
  EXPECT_EQ(substract(1,1), 1-1);
  EXPECT_TRUE(substract(1,1) == 1-1);
}

TEST(AnotherTrivialTest, Multiplication) {
  EXPECT_EQ(multiply(2,2), 2*2);
  EXPECT_EQ(multiply(0,2), 0);
}


class QueryTest : public testing::Test {
  protected:

  QueryTest() : d(100), n(1024), seed_data(56789), seed_mrpt(12345) {
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

  int d, n, seed_data, seed_mrpt;
  MatrixXf X;
  VectorXf q;
};


// Test that the nearest neighbors returned by the index
// are same as before when a seed for rng is fixed
TEST_F(QueryTest, DenseTrees) {
  int n_trees = 10, depth = 6, votes = 1, k = 5;
  float density = 1;

  // test different numbers of trees
  QueryTester(1, depth, density, votes, k, std::vector<int> {949, 84, 136, 133, 942});
  QueryTester(5, depth, density, votes, k, std::vector<int> {949, 720, 84, 959, 447});
  QueryTester(100, depth, density, votes, k, std::vector<int> {501, 682, 566, 541, 747});

  // test different depths
  QueryTester(n_trees, 1, density, votes, k, std::vector<int> {501, 682, 566, 541, 747});
  QueryTester(n_trees, 3, density, votes, k, std::vector<int> {501, 682, 541, 747, 882});
  QueryTester(n_trees, 8, density, votes, k, std::vector<int> {949, 629, 860, 954, 121});
  QueryTester(n_trees, 10, density, votes, k, std::vector<int> {949, 713, 574, 88, 900});

  // test different densities
  QueryTester(n_trees, depth, 1.0 / std::sqrt(d), votes, k, std::vector<int> {566, 882, 949, 802, 110});

  // test different vote thresholds
  QueryTester(n_trees, depth, density, 1, k, std::vector<int> {541, 949, 720, 629, 84});
  QueryTester(n_trees, depth, density, 3, k, std::vector<int> {949, 629, 359, 109, 942});
  QueryTester(30, depth, density, 5, k, std::vector<int> {629, 84, 779, 838, 713});

}

// Test that the nearest neighbors returned by the index stay
// same when an index with sparse random vectors is used
TEST_F(QueryTest, SparseTrees) {
  int n_trees = 10, depth = 6;
  float density = 1.0 / std::sqrt(d);

  const Map<const MatrixXf> *M = new Map<const MatrixXf>(X.data(), d, n);
  Mrpt index_dense(M, n_trees, depth, density);
  index_dense.grow(seed_mrpt);

  int k = 5, votes = 3;
  std::vector<int> result(k);
  std::vector<float> distances(k);

  const Map<VectorXf> V(q.data(), d);
  index_dense.query(V, k, votes, &result[0], &distances[0]);

  EXPECT_EQ(result[0], 949);
  EXPECT_EQ(result[1], 692);
  EXPECT_EQ(result[2], 258);
  EXPECT_EQ(result[3], 39);
  EXPECT_EQ(result[4], 192);

  for(int i = 0; i < k; ++i) {
    EXPECT_LE(distances[i-1], distances[i]);
    float distance_true = (X.col(result[i]) - q).norm();
    EXPECT_FLOAT_EQ(distances[i], distance_true);
  }

}

// Test that the nearest neighbors returned by the index are different
// when rng is initialized with a random seed (no seed is given to
// grow() - method). Obs. this test may fail with a very small probability
// if the nearest neighbors returned by two different indices happen
// to be exactly same by change.
TEST_F(QueryTest, RandomSeed) {
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

// Test that index with only one tree works correctly
TEST_F(QueryTest, OneTree) {
  int n_trees = 1, depth = 6;
  float density = 1.0 / std::sqrt(d);

  const Map<const MatrixXf> *M = new Map<const MatrixXf>(X.data(), d, n);
  Mrpt index_dense(M, n_trees, depth, density);
  index_dense.grow(seed_mrpt);

  int k = 5, votes = 1;
  std::vector<int> result(k);
  std::vector<float> distances(k);

  const Map<VectorXf> V(q.data(), d);
  index_dense.query(V, k, votes, &result[0], &distances[0]);

  EXPECT_EQ(result[0], 20);
  EXPECT_EQ(result[1], 833);
  EXPECT_EQ(result[2], 638);
  EXPECT_EQ(result[3], 654);
  EXPECT_EQ(result[4], 972);

  for(int i = 0; i < k; ++i) {
    EXPECT_LE(distances[i-1], distances[i]);
    float distance_true = (X.col(result[i]) - q).norm();
    EXPECT_FLOAT_EQ(distances[i], distance_true);
  }
}

// Test that the exact k-nn search works correctly
TEST_F(QueryTest, ExactKnn) {

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


}

// Step 3. Call RUN_ALL_TESTS() in main().
//
// We do this by linking in src/gtest_main.cc file, which consists of
// a main() function which calls RUN_ALL_TESTS() for us.
//
// This runs all the tests you've defined, prints the result, and
// returns 0 if successful, or 1 otherwise.
//
// Did you notice that we didn't register the tests?  The
// RUN_ALL_TESTS() macro magically knows about all the tests we
// defined.  Isn't this convenient?
