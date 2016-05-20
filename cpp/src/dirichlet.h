//
// Created by Dylan on 18/05/2016.
//

#ifndef LDA_DIRICHLET_H
#define LDA_DIRICHLET_H


#include <vector>
#include "data.h"

//! Represents a K dimension dirichlet distribution
/*!
    Keeps track of the three dirichlet parameters.
    Alpha, mean and s (precision, concentration coeeficient)
    Can update the dirichlet from observed samples.
    Class is limited in the assumption that is works with LDA. As such the sufficient statistics
    are calculated externally then passed to the dirichlet. It would perhaps be better to have
    a less specific implementation that simply gets passed the observed samples for the update.
    It is however probably slightly more efficient to calculate the ss in the lda document inference
    as it saves a D sized for loop to calculate the ss.
 */
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
    //! Estimate the precision / concentration coeeficient of the dirichlet
    void estimate_precision(double ss, int D);
    //! Estimate the alpha and precision of the dirichlet
    void update(std::vector<double> ss, int D);

private:
    // Update settings
    int INIT_S = 100;               /*! The initial precision for the precision update */
    double NEWTON_THRESH = 1e-5;    /*! The threshold for netwon-raphson update convergance */
    int MAX_ALPHA_ITER = 1000;      /*! Max number of iterations for newton-raphson */
};



#endif //LDA_DIRICHLET_H
