//
//  shepard.h
//  Surrogate Optimization
//
//  Created by David Eriksson on 2/4/16.
//  Copyright © 2016 David Eriksson. All rights reserved.
//

#ifndef Surrogate_Optimization_shepard_h
#define Surrogate_Optimization_shepard_h

#include "common.h"
#include "utils.h"
#include "surrogate.h"

namespace sot {
    
    class Shepard : public Surrogate {
    protected:
        double mp;
        double mDistTol = 1e-10;
        int mMaxPoints;
        int mNumPoints;
        int mDim;
        mat mX;
        mat mfX;
    public:
        Shepard(int maxPoints, int dim, double p) {
            mNumPoints = 0;
            mMaxPoints = maxPoints;
            mp = p;
            mDim = dim;
            mX.resize(dim, maxPoints);
            mfX.resize(maxPoints);
        }
        int dim() const {
            return mDim;
        }
        int numPoints() const {
            return mNumPoints;
        }
        vec X(int i) const {
            return mX.col(i);
        }
        mat X() const {
            return mX.cols(0, mNumPoints-1);
        }
        double fX(int i) const {
            return mfX(i);
        }
        vec fX() const {
            return mfX.rows(0, mNumPoints-1);
        }
        void addPoint(const vec &point, double funVal) {
            mX.col(mNumPoints) = point;
            mfX(mNumPoints) = funVal;
            mNumPoints++;
        }
        void addPoints(const mat &points, const vec &funVals) {
            int n = points.n_cols;
            mX.cols(mNumPoints, mNumPoints + n - 1) = points;
            mfX.rows(mNumPoints, mNumPoints + n - 1) = funVals;
            mNumPoints += n;
        }
        double eval(const vec &point) const {
            vec dists = squaredPointSetDistance<mat,vec>(point, X());
            if (arma::min(dists) < mDistTol) { // Just return the closest point
                arma::uword closest;
                double scores = dists.min(closest);
                return mfX(closest);
            }
            else {
                vec weights = arma::pow(dists, -mp/2.0);
                return arma::dot(weights, fX())/arma::sum(weights);
            }
        }
        vec evals(const mat &points) const {
            vec vals = arma::zeros<vec>(points.n_cols);
            for(int i=0; i < points.n_cols; i++) {
                vals(i) = eval(points.col(i));
            }
            return vals;
        }
        vec deriv(const vec &point) const {
            throw std::logic_error("No derivatives for Shepard");
        }
        
        void reset() {
            mNumPoints = 0;
        }
        
        void fit() {
            return;
        }
    };
}

#endif
