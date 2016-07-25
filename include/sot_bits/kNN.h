//
//  kNN.h
//  Surrogate Optimization
//
//  Created by David Eriksson on 2/2/16.
//  Copyright © 2016 David Eriksson. All rights reserved.
//

#ifndef Surrogate_Optimization_kNN_h
#define Surrogate_Optimization_kNN_h

#include "common.h"
#include "utils.h"
#include "surrogate.h"

namespace sot {
    
    class kNN : public Surrogate {
    protected:
        int mDim;
        int mMaxPoints;
        int mNumPoints;
        int mk;
        mat mX;
        vec mfX;
    public:
        kNN(int maxPoints, int dim, int k) {
            mDim = dim;
            mNumPoints = 0;
            mMaxPoints = maxPoints;
            mk = k;
            mX.resize(dim, maxPoints);
            mfX.resize(maxPoints);
        }
        int dim() const {
            return mDim;
        }
        
        int numPoints() const {
            return mNumPoints;
        }
        
        mat X() const {
            return mX.cols(0, mNumPoints-1);
        }
        
        vec X(int i) const {
            return mX.col(i);
        }
        
        vec fX() const {
            return mfX.rows(0, mNumPoints-1);
        }
        
        double fX(int i) const {
            return mfX(i);
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
            vec dists = squaredPointSetDistance(point, X());
            uvec indices = sort_index(dists);
            return arma::mean(mfX(indices.rows(0, mk - 1)));
        }
        
        vec evals(const mat &points) const {
            vec vals = arma::zeros<vec>(points.n_cols);
            for(int i=0; i < points.n_cols; i++) {
                vals(i) = eval(points.col(i));
            }
            return vals;
        }
        
        vec deriv(const vec& point) const {
            throw std::logic_error("No derivatives for kNN");
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
