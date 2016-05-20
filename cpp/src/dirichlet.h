//
// Created by Dylan on 18/05/2016.
//

#ifndef LDA_DIRICHLET_H
#define LDA_DIRICHLET_H


#include <vector>
#include "data.h"

class dirichlet {
public:
    //! Default constructor
    dirichlet();
    //! Constructor to start with centered dirichlet
    dirichlet(int K, alpha_settings settings);
    //! Constructor with mean specified
    dirichlet(std::vector<double> init_mean, alpha_settings settings);

    double s;                   /*!< The concentration coefficient or precision of the dirichlet.*/
    int K;                      /*!< The number of dimensions*/
    std::vector<double> alpha;  /*!< The alpha psuedo count parameter*/
    std::vector<double> mean;   /*!< The mean of the dirichlet which is simply alpha / precision*/

    void estimate_precision(double ss, int D);
    void update(std::vector<double> ss, int D);

private:
    int INIT_S = 100;
    double NEWTON_THRESH = 1e-5;
    int MAX_ALPHA_ITER = 1000;
};



#endif //LDA_DIRICHLET_H
