//
// Created by Dylan Verrezen on 2016/07/01.
//

#include "gtest/gtest.h"
#include "util.h"
#include <math.h>
#include <vector>

TEST(DigammaTest, IntegerValues) {
    ASSERT_NEAR(1.2561176684318004, digamma(4), 1e-10);
    ASSERT_NEAR(-0.577215664901532, digamma(1), 1e-10);
    ASSERT_NEAR(2.2517525890667211, digamma(10), 1e-10);
}

TEST(DigammaTest, FractionValues) {
    ASSERT_NEAR(-10.4237549404110767, digamma(0.1), 1e-9);
    ASSERT_NEAR(-1.96351002602142347, digamma(0.5), 1e-10);
    ASSERT_NEAR(-50.5447893104561797, digamma(0.02), 1e-9);
}

TEST(TrigammaTest, IntegerValues) {
    ASSERT_NEAR(0.2838229557371153, trigamma(4), 1e-5);
    ASSERT_NEAR(1.6449340668482264, trigamma(1), 1e-5);
}

TEST(TrigammaTest, FractionValues) {
    ASSERT_NEAR(101.43329915079275881721, trigamma(0.1), 1e-5);
    ASSERT_NEAR(2501.5981181918680666, trigamma(0.02), 1e-5);
}

TEST(LogSumTest, IntegerValues) {
    ASSERT_EQ(log(7), log_sum(log(3), log(4)));
    ASSERT_EQ(log(30), log_sum(log(10), log(20)));
    ASSERT_EQ(log(2), log_sum(log(1), log(1)));
}

TEST(LogSumTest, FractionValues) {
    ASSERT_NEAR(log(5.0), log_sum(log(2.5),log(2.5)), 1e-10);
}

TEST(SumTest, SumVector){
    int length = 3;
    std::vector<double> vec;
    for(int i=0; i<length; i++){
        vec.push_back(1);
    }
    double result = sum(vec);
    ASSERT_EQ(length, result);
}

TEST(NormTest, NormaliseVector) {
    int length = 3;
    std::vector<double> vec;
    for(int i=0; i<length; i++){
        vec.push_back(1);
    }
    norm(vec);
    for(int i=0; i<length; i++){
        ASSERT_EQ(1.0/length, vec[i]);
    }
}

TEST(DirichletExpectationTest, DirihletExpectationOfVectorRV) {
    std::vector<double> rv;
    rv.push_back(0.4);
    rv.push_back(0.5);
    rv.push_back(0.1);
    std::vector<double> result = dirichlet_expectation(rv);
    ASSERT_NEAR(-1.984168879683583, result[0], 1e-10);
    ASSERT_NEAR(-1.386294361119890, result[1], 1e-10);
    ASSERT_NEAR(-9.846539275509543, result[2], 1e-10);
}