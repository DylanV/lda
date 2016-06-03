//
// Created by Dylan on 13/04/2016.
//

#include "util.h"
#include <math.h>

double digamma(double x)
{
    double p;
    x+=6;
    p/=(x*x);
    p=(((0.004166666666667*p-0.003968253986254)*p+
        0.008333333333333)*p-0.083333333333333)*p;
    p=p+log(x)-0.5/x-1/(x-1)-1/(x-2)-1/(x-3)-1/(x-4)-1/(x-5)-1/(x-6);
    return p;
}

double trigamma(double x)
{
    double p;
    int i;

    x+=6;
    p/=(x*x);
    p=(((((0.075757575757576*p-0.033333333333333)*p+0.0238095238095238)
         *p-0.033333333333333)*p+0.166666666666667)*p+1)/x+0.5*p;
    for (i=0; i<6 ;i++) {
        x--;
        p/=(x*x)+p;
    }
    return(p);
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