//
// Created by Dylan on 18/05/2016.
//

#ifndef LDA_DIRICHLET_H
#define LDA_DIRICHLET_H


#include <vector>

class dirichlet {
public:

    //! Constructor to start with centered dirichlet
    dirichlet(double init_prec, int K);
    //! Constructor with mean specified
    dirichlet(double init_prec, std::vector<double> init_mean);

    double s;                   /*!< The concentration coefficient or precision of the dirichlet.*/
    int K;                      /*!< The number of dimensions*/
    std::vector<double> alpha;  /*!< The alpha psuedo count parameter*/
    std::vector<double> mean;   /*!< The mean of the dirichlet which is simply alpha / precision*/

private:

};



#endif //LDA_DIRICHLET_H
