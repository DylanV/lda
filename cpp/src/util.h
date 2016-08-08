/*!
 \file util.h
 */

#ifndef LDA_UTIL_H
#define LDA_UTIL_H
#include <vector>

/*! The logarithmic derivative of the gamma function.
 * @param x
 * @return Digamma of x.
 */
double digamma(double x);

/*! Approximates derivative of the digamma function.
 * The second of the polygamma functions. Calculated approximatly.
 * @param x
 * @return Approximate trigamma of x.
 */
double trigamma(double x);

/*! Given log(a) and log(b) calculate log(a+b).
 * @param [in] log_a log(a).
 * @param [in] log_b log(b).
 * @return log(a+b).
 */
double log_sum(const double log_a, const double log_b);

/*! Sum the elements in a vector.
 * @param [in] vec Vector.
 * @return Sum.
 */
template <typename T> inline
T sum(const std::vector<T>& vec) {
    T sum = 0;
    for(const T& val : vec){
        sum += val;
    }
    return sum;
}

/*! Normalise a vector so that the elements sum to 1.
 * @param [in,out] vec Vector
 */
template <typename T> inline
void norm(std::vector<T>& vec) {
    T vec_sum = sum(vec);
    typename std::vector<T>::iterator itr;
    for(itr = vec.begin(); itr != vec.end(); ++itr){
        *itr /= vec_sum;
    }
}

/*! Expected value of a probability component under the dirichlet
 * @param prob Vector
 * @return digamma(prob) - digamma(sum(prob))
 */
std::vector<double> dirichlet_expectation(const std::vector<double>& prob);

#endif //LDA_UTIL_H
