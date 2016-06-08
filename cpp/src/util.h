//
// Created by Dylan on 13/04/2016.
//

/*!
 \file util.h
 */

#ifndef LDA_UTIL_H
#define LDA_UTIL_H
#include <vector>

//! the trigamma function
double trigamma(double x);
//! the digamma function
double digamma(double x);
//! the log sum of a and b
double log_sum(double log_a, double log_b);

//! sum a vector
template <typename T> inline
T sum(const std::vector<T>& vec) {
    T sum = 0;
    for(const T& val : vec){
        sum += val;
    }
    return sum;
}

template <typename T> inline
void norm(std::vector<T>& vec) {
    T vec_sum = sum(vec);
    typename std::vector<T>::iterator itr;
    for(itr = vec.begin(); itr != vec.end(); ++itr){
        *itr /= vec_sum;
    }
}

std::vector<double> dirichlet_expectation(const std::vector<double>& prob);

#endif //LDA_UTIL_H
