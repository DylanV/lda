/*!
 \file util.h
 */

#ifndef LDA_UTIL_H
#define LDA_UTIL_H
#include <vector>

//! Trigamma of x
double trigamma(double x);
//! Digamma of x
double digamma(double x);
//! Log sum of log_a and log_b, i.e. log(a+b)
double log_sum(double log_a, double log_b);

//! Sum a vector
template <typename T> inline
T sum(const std::vector<T>& vec) {
    T sum = 0;
    for(const T& val : vec){
        sum += val;
    }
    return sum;
}

//! Normalise a vector
template <typename T> inline
void norm(std::vector<T>& vec) {
    T vec_sum = sum(vec);
    typename std::vector<T>::iterator itr;
    for(itr = vec.begin(); itr != vec.end(); ++itr){
        *itr /= vec_sum;
    }
}

//! Dirichlet expectation of vector prob
std::vector<double> dirichlet_expectation(const std::vector<double>& prob);

#endif //LDA_UTIL_H
