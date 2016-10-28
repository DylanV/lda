/*!
 \file util.cpp
 */

#include "util.h"
#include <tgmath.h>

#ifndef M_PIl
/** The constant Pi in high precision */
#define M_PIl 3.1415926535897932384626433832795029
#endif
#ifndef M_GAMMAl
/** Euler's constant in high precision */
#define M_GAMMAl 0.5772156649015328606065120900824024
#endif
#ifndef M_LN2l
/** the natural logarithm of 2 in high precision */
#define M_LN2l 0.6931471805599453094172321214581766
#endif

double digamma(double x)
    //****************************************************************************80
//
//  Purpose:
//
//    DIGAMMA calculates DIGAMMA ( X ) = d ( LOG ( GAMMA ( X ) ) ) / dX
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    20 March 2016
//
//  Author:
//
//    Original FORTRAN77 version by Jose Bernardo.
//    C++ version by John Burkardt.
//
//  Reference:
//
//    Jose Bernardo,
//    Algorithm AS 103:
//    Psi ( Digamma ) Function,
//    Applied Statistics,
//    Volume 25, Number 3, 1976, pages 315-317.
//
//  Parameters:
//
//    Input, double X, the argument of the digamma function.
//    0 < X.
//
//    Output, int *IFAULT, error flag.
//    0, no error.
//    1, X <= 0.
//
//    Output, double DIGAMMA, the value of the digamma function at X.
//
    {
        static double c = 8.5;
        static double euler_mascheroni = 0.57721566490153286060;
        double r;
        double value;
        double x2;
//
//  Check the input.
//
        if ( x <= 0.0 )
        {
            value = 0.0;
            return value;
        }
//
//  Initialize.
//

//
//  Use approximation for small argument.
//
        if ( x <= 0.000001 )
        {
            value = - euler_mascheroni - 1.0 / x + 1.6449340668482264365 * x;
            return value;
        }
//
//  Reduce to DIGAMA(X + N).
//
        value = 0.0;
        x2 = x;
        while ( x2 < c )
        {
            value = value - 1.0 / x2;
            x2 = x2 + 1.0;
        }
//
//  Use Stirling's (actually de Moivre's) expansion.
//
        r = 1.0 / x2;
        value = value + log ( x2 ) - 0.5 * r;

        r = r * r;

        value = value
                - r * ( 1.0 / 12.0
                        - r * ( 1.0 / 120.0
                                - r * ( 1.0 / 252.0
                                        - r * ( 1.0 / 240.0
                                                - r * ( 1.0 / 132.0 ) ) ) ) );

        return value;
}

double trigamma ( double x )

//****************************************************************************80
//
//  Purpose:
//
//    TRIGAMMA calculates trigamma(x) = d**2 log(gamma(x)) / dx**2
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    19 January 2008
//
//  Author:
//
//    Original FORTRAN77 version by BE Schneider.
//    C++ version by John Burkardt.
//
//  Reference:
//
//    BE Schneider,
//    Algorithm AS 121:
//    Trigamma Function,
//    Applied Statistics,
//    Volume 27, Number 1, pages 97-99, 1978.
//
//  Parameters:
//
//    Input, double X, the argument of the trigamma function.
//    0 < X.
//
//    Output, int *IFAULT, error flag.
//    0, no error.
//    1, X <= 0.
//
//    Output, double TRIGAMMA, the value of the trigamma function at X.
//
{
    double a = 0.0001;
    double b = 5.0;
    double b2 =  0.1666666667;
    double b4 = -0.03333333333;
    double b6 =  0.02380952381;
    double b8 = -0.03333333333;
    double value;
    double y;
    double z;
//
//  Check the input.
//
    if ( x <= 0.0 )
    {
        value = 0.0;
        return value;
    }

    z = x;
//
//  Use small value approximation if X <= A.
//
    if ( x <= a )
    {
        value = 1.0 / x / x;
        return value;
    }
//
//  Increase argument to ( X + I ) >= B.
//
    value = 0.0;

    while ( z < b )
    {
        value = value + 1.0 / z / z;
        z = z + 1.0;
    }
//
//  Apply asymptotic formula if argument is B or greater.
//
    y = 1.0 / z / z;

    value = value + 0.5 *
                    y + ( 1.0
                          + y * ( b2
                                  + y * ( b4
                                          + y * ( b6
                                                  + y *   b8 )))) / z;

    return value;
}

double log_sum(const double log_a, const double log_b) {

    double v=0;

    if (log_a < log_b) {
        v = log_b+log(1 + exp(log_a-log_b));
    } else {
        v = log_a+log(1 + exp(log_b-log_a));
    }
    return v;
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

bool is_on_simplex(std::vector<double> point) {
    return (sum(point) == 1.0);
}

double inv_digamma(double y){

    double x;
    if(y >= -2.22){
        x = exp(y) + 0.5;
    } else {
        x = -1.0 / (y - digamma(1));
    }

    for(int i=0; i<5; ++i){
        x -= (digamma(x) - y) / trigamma(x);
    }

    return x;
}
