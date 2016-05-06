//
// Created by Dylan on 13/04/2016.
//

#include "alpha.h"
#include <math.h>
#include <cmath>
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
    } while ((fabs(df) > NEWTON_THRESH) && (iter < MAX_ALPHA_ITER));
    return(exp(log_a));
}

void update_alpha(std::vector<double> & alpha, std::vector<double> ss, int D, int K){
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

        z = 1.0*trigamma(alpha_sum);
        sum_g_h = 0, sum_1_h = 0;

        for(int k=0; k<K; k++){
            gradient[k] = D * (digamma(alpha_sum) - digamma(alpha[k])) + ss[k];
            hessian[k] = -1.0*(D * trigamma(alpha[k]));
            sum_g_h += gradient[k] / hessian[k];
            sum_1_h += 1.0/hessian[k];
        }

        c = sum_g_h / ((1.0 / z) + sum_1_h);

        int decay_factor = 0;
        std::vector<double> alpha_new(alpha);
        bool singular_hessian;

        do{
            singular_hessian = false;
            std::vector<double> step_size(K);
            for(int k=0; k<K;  k++){
                step_size[k] = (gradient[k] - c)/hessian[k];
                for(int d=0; d<decay_factor; d++){
                    step_size[k] *= 0.9;
                }

                if(alpha[k] < step_size[k]){
                    singular_hessian = true;
                }
            }

            if(singular_hessian){
                decay_factor++;
            } else{
                for(int k=0; k<K; k++){
                    alpha_new[k] = alpha[k] - step_size[k];
                }
            }

        }while(singular_hessian && decay_factor < 10);

        for(int k=0; k<K; k++) {
            thresh += std::abs(alpha_new[k] - alpha[k]);
        }
        thresh /= K;

        alpha = alpha_new;

    }while((iteration < MAX_ALPHA_ITER && thresh > NEWTON_THRESH) || iteration < 50);
}