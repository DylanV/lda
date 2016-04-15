//
// Created by Dylan on 13/04/2016.
//

#include "alpha.h"
#include <math.h>
#include "util.h"

double alhood(double a, double ss, int D, int K)
{
    return(D * (lgamma(K * a) - K * lgamma(a)) + (a - 1) * ss);
}

double d_alhood(double a, double ss, int D, int K)
{
    return(D * (K * digamma(K * a) - K * digamma(a)) + ss);
}

double d2_alhood(double a, int D, int K)
{
    return(D * (K * K * trigamma(K * a) - K * trigamma(a)));
}

double opt_alpha(double ss, int D, int K)
{
    double a, log_a, init_a = 100;
    double f, df, d2f;
    int iter = 0;

    log_a = log(init_a);
    do{
        iter++;
        a = exp(log_a);
        if (isnan(a)){
            init_a = init_a * 10;
            a = init_a;
            log_a = log(a);
        }
        f = alhood(a, ss, D, K);
        df = d_alhood(a, ss, D, K);
        d2f = d2_alhood(a, D, K);
        log_a = log_a - df/(d2f * a + df);
    }
    while ((fabs(df) > NEWTON_THRESH) && (iter < MAX_ALPHA_ITER));
    return(exp(log_a));
}