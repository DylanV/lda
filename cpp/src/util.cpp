//
// Created by Dylan on 13/04/2016.
//

#include "util.h"
#include <math.h>


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