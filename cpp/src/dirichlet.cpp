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
    }
}

void dirichlet::update(std::vector<double> ss, int D){

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
            hessian[k] = -1.0*D*(D * trigamma(alpha[k]));
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
            thresh += fabs(alpha_new[k] - alpha[k]);
        }
        thresh /= K;

        alpha = alpha_new;

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