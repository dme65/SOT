/* 
 * File:   surrogate.h
 * Author: David Eriksson
 *
 * Created on May 2, 2016, 12:13 PM
 */

#ifndef Surrogate_Optimization_surrogate_h
#define Surrogate_Optimization_surrogate_h

#include "common.h"

namespace sot {
    
    // Template
    class Surrogate {
    public:
        virtual int numPoints() const = 0;
        virtual int dim() const = 0;
        virtual void reset() = 0;
        virtual vec X(int i) const = 0;
        virtual mat X() const = 0;
        virtual double fX(int) const = 0;
        virtual vec fX() const = 0;
        virtual void addPoint(const vec&, double) = 0;
        virtual void addPoints(const mat&, const vec&) = 0;
        virtual double eval(const vec&) const = 0;
        virtual vec evals(const mat&) const = 0;
        virtual vec deriv(const vec&) const = 0;
        virtual void fit() = 0;
    };
}


#endif

