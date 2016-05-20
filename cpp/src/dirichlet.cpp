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
    if(s == 0){
        s = 1.0/K;
    }
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
    if(s == 0){
        s = 1.0/K;
    }
    mean = std::vector<double>(K, 1.0/K);
    alpha = std::vector<double>(K, 1.0/K);
//    for(int i=0; i<K; i++){
//        alpha[i] = mean[i]*s;
//    }

    INIT_S = settings.init_s;
    NEWTON_THRESH = settings.newton_threshold;
    MAX_ALPHA_ITER = settings.max_iterations;
}

void dirichlet::estimate_precision(double ss, int D)
{
/*!
Update the precision of the dirichlet given some sufficient statistic. The sufficient statistic is the observed samples
of the dirichlet in question put in the form sum(digamma(samples) - K * digamma(sum(samples))))
Estimates and updates the precision (s) of the dirichlet only, however the alpha will change.
The mean remains unchanged.
\param ss the sufficient statistics
\param D the number of observed samples for the sufficient statistics
*/
    double s, log_s, init_s = INIT_S;
    double f, df, d2f;
    int iter = 0;

    log_s = log(init_s);
    double prev_s = dirichlet::s;
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

    if(!isnan(exp(log_s))){
        dirichlet::s = exp(log_s);
        for(int i=0; i<K; i++){
            alpha[i] = mean[i]*s;
        }
    }
}

void dirichlet::update(std::vector<double> ss, int D)
{
/*!
Performs Newton-Raphson for dirichlet with special hessian. Linear time
Fully updates the dirichlet from the given sufficient statistics. Mean and precision are estimated.
Sufficient statistics are the observed samples of the dirichlet where for one sample and dimension k:
ss[k] += digamma(sample[k]) - digamma(sum(samples))
/param ss the sufficient statistics in a K dimension vector
/param D the number of samples in the sufficient statistics
*/
    int iteration = 0;
    double thresh = 0;
    double alpha_sum = 0;
    std::vector<double> gradient(K);
    std::vector<double> hessian(K);
    double sum_g_h, sum_1_h;
    double z, c;

    do {
        iteration++;
        for(int k=0; k<K; k++){
            alpha_sum += alpha[k];
        }

        z = D*trigamma(alpha_sum);
        sum_g_h = 0, sum_1_h = 0;

        for(int k=0; k<K; k++){
            gradient[k] = D * (digamma(alpha_sum) - digamma(alpha[k])) + ss[k];
            hessian[k] = -1.0*(D * trigamma(alpha[k]));
            sum_g_h += gradient[k] / hessian[k];
            sum_1_h += 1.0/hessian[k];
        }

        c = sum_g_h / ((1.0 / z) + sum_1_h);

        std::vector<double> alpha_new(alpha);
        bool broke = false;

        std::vector<double> step_size(K);
        for(int k=0; k<K;  k++) {
            step_size[k] = 0.001*(gradient[k] - c) / hessian[k];
            alpha_new[k] -= step_size[k];
        }

        for(int k=0; k<K; k++) {
            thresh += fabs(alpha_new[k] - alpha[k]);
            if(alpha_new[k] < 0){
                broke = true;
            }
        }
        thresh /= K;

        if(broke){
            iteration = MAX_ALPHA_ITER;
        }else{
            alpha = alpha_new;
        }

    }while((iteration < MAX_ALPHA_ITER && thresh > NEWTON_THRESH) || iteration < 50);

    alpha_sum = 0;
    for(int k=0; k<K; k++){
        alpha_sum += alpha[k];
    }
    s = alpha_sum;
    for(int k=0; k<K; k++){
        mean[k] = alpha[k] / alpha_sum;
    }
}