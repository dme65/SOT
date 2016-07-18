//
//  ExperimentalDesign.h
//  Surrogate Optimization
//
//  Created by David Eriksson on 7/30/15.
//  Copyright (c) 2015 David Eriksson. All rights reserved.
//

#ifndef __Surrogate_Optimization__ExperimentalDesign__
#define __Surrogate_Optimization__ExperimentalDesign__

#include <stdio.h>
#include <cassert>
#include "utils.h"
#include "common.h"
#include <iostream>

namespace sot {
    
    class ExpDesign {
    public:
        int dim;
        int num_points;
        virtual mat generate_points() const = 0;
    };
    
    class FixedDesign : public ExpDesign {
    protected:
        mat points;
    public:
        FixedDesign(mat& points) { this->points = points; dim = points.n_rows; num_points = points.n_cols; }
        virtual mat generate_points() const { return points; }
    };
    
    class SymmetricLatinHypercube : public ExpDesign {
    protected:
        mat create_design() const;
    public:
        SymmetricLatinHypercube(int, int);
        virtual mat generate_points() const;
    };
    
    class LatinHypercube : public ExpDesign {
    public:
        LatinHypercube(int, int);
        virtual mat generate_points() const;
    };
    
    class MultiSymmetricLatinHypercube : public ExpDesign {
    private:
        int num_designs;
        double min_dist(mat) const;
        double corr(mat) const;
    public:
        MultiSymmetricLatinHypercube(int, int, int);
        virtual mat generate_points() const;
    };
    
    class TwoFactorial : public ExpDesign {
    public:
        TwoFactorial(int);
        virtual mat generate_points() const;
    };
    
    class CornersMid : public ExpDesign {
    public:
        CornersMid(int);
        virtual mat generate_points() const;
    };
}
#endif /* defined(__Surrogate_Optimization__ExperimentalDesign__) */
