//
// Created by Dylan on 18/05/2016.
//

#include "dirichlet.h"
#include "util.h"
#include <math.h>

dirichlet::dirichlet()
{
    s = 0;
    K = 0;
}

dirichlet::dirichlet(std::vector<double> init_mean, alpha_settings settings)
{
    s = settings.init_prec;
    mean = init_mean;
    K = int(mean.size());
    alpha = std::vector<double>(K);
    for(int i=0; i<K; i++){
        alpha[i] = mean[i]*s;
    }

    INIT_S = settings.init_s;
    NEWTON_THRESH = settings.newton_threshold;
    MAX_ALPHA_ITER = settings.max_iterations;
}

dirichlet::dirichlet(int K, alpha_settings settings)
{
    s = settings.init_prec;
    dirichlet::K = K;
    mean = std::vector<double>(K, 1.0/K);
    alpha = std::vector<double>(K);
    for(int i=0; i<K; i++){
        alpha[i] = mean[i]*s;
    }

    INIT_S = settings.init_s;
    NEWTON_THRESH = settings.newton_threshold;
    MAX_ALPHA_ITER = settings.max_iterations;
}

void dirichlet::estimate_precision(double ss, int D)
{
    double s, log_s, init_s = INIT_S;
    double f, df, d2f;
    int iter = 0;

    log_s = log(init_s);
    do{
        iter++;
        s = exp(log_s);
        if (isnan(s)){
            init_s = init_s * 10;
            s = init_s;
            log_s = log(s);
        }
        f = D * (lgamma(K * s) - K * lgamma(s) + (s - 1) * ss);
        df = D * (K * digamma(K * s) - K * digamma(s)) + ss;
        d2f = D * (K * K * trigamma(K * s) - K * trigamma(s));
        log_s = log_s - df/(d2f * s + df);
    } while ((fabs(df) > NEWTON_THRESH) && (iter < MAX_ALPHA_ITER));
    dirichlet::s = exp(log_s);
}
