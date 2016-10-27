#include "gtest/gtest.h"
#include "dirichlet.h"
#include "util.h"
#include <iostream>

TEST(DirichletTest, ConstructorsCreateCorrectPropertiesSymmetric){
    std::vector<double> true_mean = std::vector<double>(4, 0.25);
    double true_s = 4.0;
    bool true_symmetric = true;
    std::vector<double> true_alpha = std::vector<double>(4, 1.0);

    dirichlet d1 = dirichlet(4, 1.0);
    ASSERT_EQ(true_alpha, d1.alpha);
    ASSERT_EQ(true_mean, d1.mean);
    ASSERT_EQ(true_s, d1.s);
    ASSERT_EQ(true_symmetric, d1.symmetric);
}

TEST(DirichletTest, ConstructorsCreateCorrectPropertiesAsymmetric){
    std::vector<double> true_mean = std::vector<double>(4);
    double true_s = 10.0;
    bool true_symmetric = false;
    std::vector<double> true_alpha = std::vector<double>(4, 1.0);

    for(int k=0; k<4; k++){
        true_alpha[k] += k;
        true_mean[k] = true_alpha[k]/10.0;
    }

    dirichlet d1 = dirichlet(4, true_alpha);
    ASSERT_EQ(true_alpha, d1.alpha);
    ASSERT_EQ(true_mean, d1.mean);
    ASSERT_EQ(true_s, d1.s);
    ASSERT_EQ(true_symmetric, d1.symmetric);
}

TEST(DirichletTest, SampleUniform){
    dirichlet d1 = dirichlet(4, std::vector<double>(4, 1.0));
    ASSERT_TRUE(is_on_simplex(d1.sample()));
}

TEST(DirichletTest, MultipleSampleAsymmetric){
    std::vector<double> alpha = std::vector<double>(4, 1.0);

    for(int k=0; k<4; k++){
        alpha[k] += k;
        alpha[k] *= 10.0;
    }

    dirichlet d1 = dirichlet(4, alpha);

    int num_samples = 300;
    std::vector<std::vector<double>> samples = d1.sample(num_samples);
    std::vector<double> avg = std::vector<double>(4, 0.0);
    for(int n=0; n<num_samples; ++n){
        for(int k=0; k<4.0; ++k){
            avg[k] += samples[n][k];
        }
    }

    for(int k=0; k<4.0; ++k){
        avg[k] /= num_samples*1.0;
        ASSERT_NEAR(avg[k], d1.mean[k], 1e-2);
    }

}