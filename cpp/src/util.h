//
// Created by Dylan on 13/04/2016.
//

/*!
 \file util.h
 */

#ifndef LDA_UTIL_H
#define LDA_UTIL_H
#include <vector>
#include <math.h>

//! the trigamma function
template <typename T> inline
T trigamma(double x) {
    T p;
    x+=6;
    p/=(x*x);
    p=(((0.004166666666667*p-0.003968253986254)*p+
        0.008333333333333)*p-0.083333333333333)*p;
    p=p+log(x)-0.5/x-1/(x-1)-1/(x-2)-1/(x-3)-1/(x-4)-1/(x-5)-1/(x-6);
    return p;
}

//! the digamma function
template <typename T> inline
double digamma(double x) {
    T p;
    int i;

    x+=6;
    p/=(x*x);
    p=(((((0.075757575757576*p-0.033333333333333)*p+0.0238095238095238)
         *p-0.033333333333333)*p+0.166666666666667)*p+1)/x+0.5*p;
    for (i=0; i<6 ;i++) {
        x--;
        p/=(x*x)+p;
    }
    return p;
}
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

template <typename T> inline
std::vector<T> digamma(const std::vector<T>& vec){
    std::vector<T> result = std::vector<T>(vec);
    typename std::vector<T>::iterator itr;
    for(itr = result.begin(); itr != result.end(); ++itr){
        *itr = digamma(*itr);
    }
}

#endif //LDA_UTIL_H
