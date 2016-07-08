//
// Created by Dylan Verrezen on 2016/07/01.
//

#include "gtest/gtest.h"
#include "util.h"

TEST(DigammaTest, IntegerValues) {
    ASSERT_NEAR(1.2561176684318004, digamma(4), 1e-10);
    ASSERT_NEAR(-0.577215664901532, digamma(1), 1e-10);
    ASSERT_NEAR(2.2517525890667211, digamma(10), 1e-10);
}

TEST(DigammaTest, FractionValues) {
    ASSERT_NEAR(-10.4237549404110767, digamma(0.1), 1e-10);
    ASSERT_NEAR(-1.96351002602142347, digamma(0.5), 1e-10);
    ASSERT_NEAR(-50.5447893104561797, digamma(0.02), 1e-9);
}