//
// Created by Dylan on 13/04/2016.
//
/*!
 \file alpha.h
 */
#ifndef LDA_ALPHA_H
#define LDA_ALPHA_H

#include <vector>

#define NEWTON_THRESH 1e-5
#define MAX_ALPHA_ITER 1000

//! the alpha likelihood
double alhood(double a, double ss, int D, int K);
//! the first derivative of the alpha likelihood
double d_alhood(double a, double ss, int D, int K);
//! the second derivative of the alpha likelihood
double d2_alhood(double a, int D, int K);
//! newton-raphson update of alpha
double opt_alpha(double ss, int D, int K);
//! newton-raphson update of vector alpha
void update_alpha(std::vector<double> & alpha, std::vector<double> ss, int D, int K);

#endif //LDA_ALPHA_H
