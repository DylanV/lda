/*!
 \file util.cpp
 */

#include "util.h"
#include <math.h>

double digamma(double x)
{
    double p = 1;
    x+=6;
    p/=(x*x);
    p=(((0.004166666666667*p-0.003968253986254)*p+
        0.008333333333333)*p-0.083333333333333)*p;
    p=p+log(x)-0.5/x-1/(x-1)-1/(x-2)-1/(x-3)-1/(x-4)-1/(x-5)-1/(x-6);
    return p;
}

double trigamma(double x)
{
    double p = 0;

    int iter = 20000000; // this can be adjusted for more accuracy at the cost of performance
    for(int i=0; i<iter; i++){
        p += 1.0/((x+i)*(x+i));
    }

    return p;
}

double log_sum(double log_a, double log_b)
{
    double v;

    if (log_a < log_b)
    {
        v = log_b+log(1 + exp(log_a-log_b));
    }
    else
    {
        v = log_a+log(1 + exp(log_b-log_a));
    }
    return(v);
}

std::vector<double> dirichlet_expectation(const std::vector<double>& prob) {
    std::vector<double> result = std::vector<double>(prob.size());
    double sum = 0;
    for(int i=0; i<prob.size(); ++i){
        sum += prob[i];
    }

    sum = digamma(sum);

    for(int i=0; i<result.size(); ++i){
        result[i] = digamma(prob[i]) - sum;
    }

    return result;
}