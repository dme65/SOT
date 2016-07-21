/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   approximant.h
 * Author: davideriksson
 *
 * Created on May 2, 2016, 12:13 PM
 */

#ifndef SURROGATE_H
#define SURROGATE_H

#include "common.h"

namespace sot {
    
    // Template
    class Surrogate {
    protected:
        int max_points;
        int num_points;
        mat Xmat;
        vec fXvec;
        int d;
    public:
        int npts() { return num_points; }
        int dim() const { return d; }
        virtual void reset() = 0;
        virtual vec get_X(int i) const = 0;
        virtual mat get_X() const = 0;
        virtual double get_fX(int) const = 0;
        virtual vec get_fX() const = 0;
        virtual void add_point(const vec&, double) = 0;
        virtual void add_points(const mat&, const vec&) = 0;
        virtual double eval(const vec&) const = 0;
        virtual vec evals(const mat&) const = 0;
        virtual vec deriv(const vec&) const = 0;
        virtual void fit() = 0;
    };
}


#endif /* APPROXIMANT_H */

