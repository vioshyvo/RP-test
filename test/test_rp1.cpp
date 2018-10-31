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


namespace {



class MrptTest : public testing::Test {
  protected:

  MrptTest() : d(100), n(1024), n2(1255), n_test(100), seed_data(56789), seed_mrpt(12345) {
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

          Q = MatrixXf(d, n_test);
          for(int i = 0; i < d; ++i)
            for(int j = 0; j < n_test; ++j)
              Q(i,j) = dist(mt);

  }




  // Test that:
  // a) the indices of the returned approximate k-nn are same as before
  // b) approximate k-nn are returned in correct order
  // c) distances to the approximate k-nn are computed correctly
  void QueryTester(int n_trees, int depth, float density, int votes, int k,
      std::vector<int> approximate_knn) {
    ASSERT_EQ(approximate_knn.size(), k);
    const Map<const MatrixXf> *M = new Map<const MatrixXf>(X.data(), d, n);
    Mrpt index_dense(M);
    index_dense.grow(n_trees, depth, density, seed_mrpt);

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
    Mrpt index(M);
    index.grow(n_trees, depth, density, seed_mrpt);
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
    Mrpt index(M);
    index.grow(n_trees, depth, density, seed_mrpt);
    Mrpt_old index_old(M, n_trees, depth, density);
    index_old.grow(seed_mrpt);

    TestLeaves(index, index_old);
  }

  void compute_exact(Mrpt &index, MatrixXi &out_exact) {
    int k = out_exact.rows();
    int nt = out_exact.cols();
    for(int i = 0; i < nt; ++i) {
      VectorXi idx(n);
      std::iota(idx.data(), idx.data() + n, 0);

      index.exact_knn(Map<VectorXf>(Q.data() + i * d, d), k, idx, n, out_exact.data() + i * k);
      std::sort(out_exact.data() + i * k, out_exact.data() + i * k + k);
    }
  }

  int d, n, n2, n_test, seed_data, seed_mrpt;
  MatrixXf X, X2, Q;
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

  Mrpt index_dense(M);
  index_dense.grow(n_trees, depth, density); // initialize rng with random seed
  Mrpt index_dense2(M);
  index_dense2.grow(n_trees, depth, density);

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
  Mrpt index_dense(M);
  index_dense.grow(0, 0, 1.0);

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

TEST(SaveTest, Loading) {
  int n = 3749, d = 100, n_trees = 3, depth = 6, seed_mrpt = 12345, seed_data = 56789;
  float density = 1.0 / std::sqrt(d);

  MatrixXf X(d,n);
  std::mt19937 mtt(seed_data);
  std::normal_distribution<double> distr(5.0,2.0);
  for(int i = 0; i < d; ++i)
    for(int j = 0; j < n; ++j)
      X(i,j) = distr(mtt);

  const Map<const MatrixXf> *M = new Map<const MatrixXf>(X.data(), d, n);
  Mrpt index(M);
  index.grow(n_trees, depth, density, seed_mrpt);
  index.save("save/mrpt_saved");

  Mrpt index_reloaded(M);
  index_reloaded.load("save/mrpt_saved");

  ASSERT_EQ(n_trees, index_reloaded.get_n_trees());
  ASSERT_EQ(depth, index_reloaded.get_depth());
  ASSERT_EQ(n, index_reloaded.get_n_points());

  for(int tree = 0; tree < n_trees; ++tree) {
    int n_leaf = std::pow(2, depth);
    VectorXi leaves = VectorXi::Zero(n);

    for(int j = 0; j < n_leaf; ++j) {
      int leaf_size = index.get_leaf_size(tree, j);
      int leaf_size_old = index_reloaded.get_leaf_size(tree, j);
      ASSERT_EQ(leaf_size, leaf_size_old);

      std::vector<int> leaf(leaf_size), leaf_old(leaf_size);
      for(int i = 0; i < leaf_size; ++i) {
        leaf[i] = index.get_leaf_point(tree, j, i);
        leaf_old[i] = index_reloaded.get_leaf_point(tree, j, i);
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
        float split_old = index_reloaded.get_split_point(tree, idx);
        ++idx;
        ASSERT_FLOAT_EQ(split, split_old);
      }
    }
    per_level *= 2;

    // Test that all data points are found at a tree
    EXPECT_EQ(leaves.sum(), n);
  }
}

TEST_F(MrptTest, RecallMatrix) {
  omp_set_num_threads(1);

  int trees_max = 10, depth_min = 5, depth_max = 7, votes_max = trees_max - 1, k = 5;
  float density = 1.0 / std::sqrt(d);

  const Map<const MatrixXf> *M = new Map<const MatrixXf>(X.data(), d, n);
  Map<MatrixXf> *test_queries = new Map<MatrixXf>(Q.data(), d, n_test);

  Autotuning at(M, test_queries);
  at.tune(trees_max, depth_min, depth_max, votes_max, density, k, seed_mrpt);

  MatrixXi exact(k, n_test);
  Mrpt index_exact(M);
  compute_exact(index_exact, exact);

  std::vector<MatrixXd> recalls(depth_max - depth_min + 1);
  std::vector<MatrixXd> cs_sizes(depth_max - depth_min + 1);
  std::vector<MatrixXd> query_times(depth_max - depth_min + 1);
  std::vector<MatrixXd> projection_times(depth_max - depth_min + 1);
  std::vector<MatrixXd> voting_times(depth_max - depth_min + 1);
  std::vector<MatrixXd> exact_times(depth_max - depth_min + 1);

  std::vector<MatrixXd> query_times_at(depth_max - depth_min + 1);
  std::vector<MatrixXd> projection_times_at(depth_max - depth_min + 1);
  std::vector<MatrixXd> voting_times_at(depth_max - depth_min + 1);
  std::vector<MatrixXd> exact_times_at(depth_max - depth_min + 1);

  for(int depth = depth_min; depth <= depth_max; ++depth) {
    MatrixXd recall_matrix = MatrixXd::Zero(votes_max, trees_max);
    MatrixXd candidate_set_size = MatrixXd::Zero(votes_max, trees_max);
    MatrixXd query_time = MatrixXd::Zero(votes_max, trees_max);
    MatrixXd projection_time = MatrixXd::Zero(votes_max, trees_max);
    MatrixXd voting_time = MatrixXd::Zero(votes_max, trees_max);
    MatrixXd exact_time = MatrixXd::Zero(votes_max, trees_max);

    MatrixXd query_time_at = MatrixXd::Zero(votes_max, trees_max);
    MatrixXd projection_time_at = MatrixXd::Zero(votes_max, trees_max);
    MatrixXd voting_time_at = MatrixXd::Zero(votes_max, trees_max);
    MatrixXd exact_time_at = MatrixXd::Zero(votes_max, trees_max);

    float proj_sum = 0, idx_sum = 0, exact_sum = 0;

    for(int t = 1; t <= trees_max; ++t) {
      Mrpt index(M);
      index.grow(t, depth, density, seed_mrpt);

      int votes_index = votes_max < t ? votes_max : t;
      for(int v = 1; v <= votes_index; ++v) {
        int n_elected = 0;
        for(int i = 0; i < n_test; ++i) {
          std::vector<int> result(k);
          std::vector<float> distances(k);
          int n_elected_tmp = 0;

          double start = omp_get_wtime();
          index.query(Map<VectorXf>(Q.data() + i * d, d), k, v, &result[0],
            &distances[0], &n_elected_tmp);
          double end = omp_get_wtime();

          double start_proj = omp_get_wtime();
          VectorXf projected_query(t * depth);
          index.project_query(Map<VectorXf>(Q.data() + i * d, d), projected_query);
          double end_proj = omp_get_wtime();
          proj_sum += projected_query.norm();

          double start_voting = omp_get_wtime();
          int n_el = 0;
          VectorXi elected;
          index.vote(projected_query, v, elected, n_el, t);
          double end_voting = omp_get_wtime();
          for(int i = 0; i < n_el; ++i)
            idx_sum += elected(i);

          double start_exact = omp_get_wtime();
          std::vector<int> res(k);
          index.exact_knn(Map<VectorXf>(Q.data() + i * d, d), k, elected, n_el, &res[0]);
          double end_exact = omp_get_wtime();
          for(int i = 0; i < k; ++i)
            exact_sum += res[i];

          n_elected += n_elected_tmp;
          std::sort(result.begin(), result.end());

          std::set<int> intersect;
          std::set_intersection(exact.data() + i * k, exact.data() + i * k + k, result.begin(), result.end(),
                           std::inserter(intersect, intersect.begin()));

          recall_matrix(v - 1, t - 1) += intersect.size();
          query_time(v - 1, t - 1) += end - start;
          projection_time(v - 1, t - 1) += end_proj - start_proj;
          voting_time(v - 1, t - 1) += end_voting - start_voting;
          exact_time(v - 1, t - 1) += end_exact - start_exact;
        }

      candidate_set_size(v - 1, t - 1) = n_elected;

      query_time_at(v - 1, t - 1) = at.get_query_time(t, depth, v);
      projection_time_at(v - 1, t - 1) = at.get_projection_time(t, depth, v);
      voting_time_at(v - 1, t - 1) = at.get_voting_time(t, depth, v);
      exact_time_at(v - 1, t - 1) = at.get_exact_time(t, depth, v);
      }
    }

    recall_matrix /= (k * n_test);
    candidate_set_size /= n_test;
    query_time /= n_test;
    projection_time /= n_test;
    voting_time /= n_test;
    exact_time /= n_test;

    recalls[depth - depth_min] = recall_matrix;
    cs_sizes[depth - depth_min] = candidate_set_size;
    query_times[depth - depth_min] = query_time;
    projection_times[depth - depth_min] = projection_time;
    voting_times[depth - depth_min] = voting_time;
    exact_times[depth - depth_min] = exact_time;

    query_times_at[depth - depth_min] = query_time_at;
    projection_times_at[depth - depth_min] = projection_time_at;
    voting_times_at[depth - depth_min] = voting_time_at;
    exact_times_at[depth - depth_min] = exact_time_at;

    std::cout << "recall, depth: " << depth << "\n";
    std::cout << recall_matrix << "\n\n";

    // std::cout << "cs size, depth: " << depth << "\n";
    // std::cout << candidate_set_size << "\n\n";

    std::cout << "proj_sum: " << proj_sum << " idx_sum: " << idx_sum << " exact_sum: " << exact_sum << "\n";

    std::cout << "query time, depth: " << depth << "\n";
    std::cout << query_time * 1000 << "\n\n";

    // std::cout << "projection time, depth: " << depth << "\n";
    // std::cout << projection_time * 1000 << "\n\n";
    //
    // std::cout << "projection time (at), depth: " << depth << "\n";
    // std::cout << projection_time_at * 1000 << "\n\n";

    // std::cout << "voting time, depth: " << depth << "\n";
    // std::cout << voting_time * 1000 << "\n\n";
    //
    // std::cout << "voting time (at), depth: " << depth << "\n";
    // std::cout << voting_time_at * 1000 << "\n\n";

    // std::cout << "exact time, depth: " << depth << "\n";
    // std::cout << exact_time * 1000 << "\n\n";
    //
    // std::cout << "exact time (at), depth: " << depth << "\n";
    // std::cout << exact_time_at * 1000 << "\n\n";

  }

  // int depth = depth_max;
  // for(int t = 1; t <= trees_max; ++t)
  //   for(int v = 1; v <= votes_max; ++v) {
  //     ASSERT_FLOAT_EQ(recalls[depth - depth_min](v - 1, t - 1), at.get_recall(t, depth, v));
  //     ASSERT_FLOAT_EQ(cs_sizes[depth - depth_min](v - 1, t - 1), at.get_candidate_set_size(t, depth, v));
  //   }

  std::cout << "\n\n\n\n";
  for(int depth = depth_min; depth <= depth_max; ++depth) {
    // std::cout << "query time, depth: " << depth << "\n";
    // std::cout << query_times[depth - depth_min] * 1000 << "\n\n";

    // std::cout << "composite query time, depth: " << depth << "\n";
    // std::cout << (projection_times[depth - depth_min] + voting_times[depth - depth_min]
    //   + exact_times[depth - depth_min]) * 1000 << "\n\n";

    // std::cout << "composite query time (at), depth: " << depth << "\n";
    // std::cout << query_times_at[depth - depth_min] * 1000 << "\n\n";
  }
}


TEST_F(MrptTest, Autotuning) {
  omp_set_num_threads(1);

  int trees_max = 10, depth_min = 5, depth_max = 7, votes_max = trees_max - 1, k = 5;
  float density = 1.0 / std::sqrt(d);

  const Map<const MatrixXf> *M = new Map<const MatrixXf>(X.data(), d, n);
  Map<MatrixXf> *test_queries = new Map<MatrixXf>(Q.data(), d, n_test);

  MatrixXi exact(k, n_test);
  Mrpt index_exact(M);
  compute_exact(index_exact, exact);

  int optimal_votes;
  Mrpt index(M);
  Autotuning at(M, test_queries);
  at.tune(trees_max, depth_min, depth_max, votes_max, density, k, seed_mrpt);
  at.grow(0.2, optimal_votes, index);

  double query_time = 0, recall = 0;

  for(int i = 0; i < n_test; ++i) {
    std::vector<int> result(k);

    double start = omp_get_wtime();
    index.query(Map<VectorXf>(Q.data() + i * d, d), k, optimal_votes, &result[0]);
    double end = omp_get_wtime();

    std::sort(result.begin(), result.end());
    std::set<int> intersect;
    std::set_intersection(exact.data() + i * k, exact.data() + i * k + k, result.begin(), result.end(),
                     std::inserter(intersect, intersect.begin()));

    query_time += end - start;
    recall += intersect.size();
  }

  std::cout << "Mean recall: " << recall / (k * n_test) << "\n";
  std::cout << "Mean query time: " << query_time / n_test * 1000 << " ms. \n\n";
}


TEST(UtilityTest, TheilSen) {
  int n = 10;
  std::vector<double> x(n);
  std::iota(x.begin(), x.end(), 1);
  std::vector<double> y {1,2,2,3,5,4,7,7,8,9};

  std::pair<double,double> theil_sen = Autotuning::fit_theil_sen(x, y);

  double intercept = -1.0, slope = 1.0;
  EXPECT_FLOAT_EQ(theil_sen.first, intercept);
  EXPECT_FLOAT_EQ(theil_sen.second, slope);
}




}
