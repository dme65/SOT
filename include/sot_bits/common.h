//
//  common.h
//  Surrogate Optimization
//
//  Created by David Eriksson on 2/2/16.
//  Copyright © 2016 David Eriksson. All rights reserved.
//

#ifndef __Surrogate_Optimization__Common__
#define __Surrogate_Optimization__Common__

namespace sot {
    
    #define ARMA_USE_BLAS

    #define __pi__ 3.14159265358979323846

    typedef arma::mat mat;
    typedef arma::vec vec;
    typedef arma::rowvec rowvec;
    typedef arma::fmat fmat;
    typedef arma::fvec fvec;
    typedef arma::Col<int> ivec;
    typedef arma::uvec uvec;
}

#endif /* common_h */
