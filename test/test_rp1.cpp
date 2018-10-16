// Step 1. Include necessary header files such that the stuff your
// test logic needs is declared.
//
// Don't forget gtest.h, which declares the testing framework.

#include "rp1.h"
#include "gtest/gtest.h"

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
