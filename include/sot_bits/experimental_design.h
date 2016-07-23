//
//  experimental_design.h
//  Surrogate Optimization
//
//  Created by David Eriksson on 7/30/15.
//  Copyright (c) 2015 David Eriksson. All rights reserved.
//

#ifndef Surrogate_Optimization_experimental_design_h
#define Surrogate_Optimization_experimental_design_h

#include <iostream>
#include "common.h"
#include "utils.h"

namespace sot {
    
    class ExpDesign {
    public:
        virtual int dim() const = 0;
        virtual int numPoints() const = 0;
        virtual mat generatePoints() const = 0;
    };
    
    class FixedDesign : public ExpDesign {
    protected:
        int mDim;
        int mNumPoints;
        mat mPoints;
    public:
        FixedDesign(mat& points) { 
            mPoints = points; 
            mDim = points.n_rows; 
            mNumPoints = points.n_cols; 
        }
        int dim() const { return mDim; }
        int numPoints() const { return mNumPoints; }
        mat generatePoints() const { return mPoints; }
    };
    
    class SymmetricLatinHypercube : public ExpDesign {
    protected:
        int mDim;
        int mNumPoints;
        mat createDesign() const {
            mat points = arma::zeros<mat>(mDim, mNumPoints);
            points.row(0) = arma::linspace<vec>(1, mNumPoints, mNumPoints).t();

            int middleInd = mNumPoints/2;

            if (mNumPoints % 2 == 1) {
                points.row(middleInd).fill(middleInd + 1);
            }

            // Fill upper
            for(int j=1; j < mDim; j++) {
                for(int i=0; i < middleInd; i++) {
                    if (rand() < 0.5) {
                        points(j, i) = mNumPoints -i;
                    }
                    else {
                        points(j, i) = i + 1;
                    }
                }
                // Shuffle
                points(j, arma::span(0, middleInd - 1)) = arma::shuffle(points(j, arma::span(0, middleInd - 1)));
            }

            // Fill bottom
            for(int i=middleInd; i < mNumPoints; i++) {
                points.col(i) = mNumPoints + 1 - points.col(mNumPoints - 1 - i);
            }

            return points/double(mNumPoints);
        }
    public:
        SymmetricLatinHypercube(int numPoints, int dim) {
            mNumPoints = numPoints;
            mDim = dim;
        }
        int dim() const { return mDim; }
        int numPoints() const { return mNumPoints; }
        mat generatePoints() const {
            arma::uword rankPmat = 0;
            mat pMat = arma::ones<mat>(mDim + 1, mNumPoints);
            mat xSample;
            do {
                xSample = createDesign();
                pMat.rows(1, mDim) = xSample;
                rankPmat = arma::rank(pMat);
            } while(rankPmat != mDim + 1);
            return xSample;
        }
    };
    
    class LatinHypercube : public ExpDesign {
    protected:
        int mDim;
        int mNumPoints;
    public:
        LatinHypercube(int numPoints, int dim) {
            mNumPoints = numPoints;
            mDim = dim;
        }
        int dim() const { return mDim; }        
        int numPoints() const { return mNumPoints; }
        mat generatePoints() const {
            mat XBest;
            mat X;
            double bestScore = 0;

            // Generate 100 LHD and pick the best one
            for(int iter=0; iter < 100; iter++) {
                X = arma::zeros(mDim, mNumPoints);
                vec xvec = (arma::linspace<vec>(1, mNumPoints, mNumPoints) - 0.5) / mNumPoints;

                for(int j=0; j < mDim; j++) {
                    X.row(j) = xvec(arma::shuffle(arma::linspace<uvec>(0, mNumPoints - 1, mNumPoints))).t();
                }

                mat dists = sqrt(mDim)*arma::eye(mNumPoints, mNumPoints) + arma::sqrt(squaredPairwiseDistance(X, X));
                double score = arma::min((vec)arma::min(dists).t());

                if (score > bestScore) {
                    XBest = X;
                    bestScore = score;
                }
            }

            return XBest;
        }   
    };
    
    class TwoFactorial : public ExpDesign {
    protected:
        int mNumPoints;
        int mDim;
    public:
        TwoFactorial(int dim) {
            mNumPoints = pow(2, dim);
            mDim = dim;
            if(dim >= 15) {throw std::logic_error("Using 2-Factorial for dim >= 15 is a bad idea"); }
        }
        int dim() const { return mDim; }
        int numPoints() const { return mNumPoints; }
        mat generatePoints() const {
            mat xSample = arma::zeros<mat>(mDim, mNumPoints);
            for(int i=0; i < mDim; i++) {
                int elem = 0;
                int flip = pow(2, i);
                for(int j=0; j < mNumPoints; j++) {
                    xSample(i, j) = elem;
                    if((j+1) % flip == 0) { elem = (elem + 1) % 2; }
                }
            }
            return xSample;
        }
    };
    
    class CornersMid : public ExpDesign {
    protected:
        int mNumPoints;
        int mDim;
    public:
        CornersMid(int dim) {
            mNumPoints = 1 + pow(2, dim);
            mDim = dim;
            if(dim >= 15) {throw std::logic_error("Using Corners + Mid for dim >= 15 is a bad idea"); }
        }
        int dim() const { return mDim; }
        int numPoints() const { return mNumPoints; }
        mat generatePoints() const {
            mat xSample = arma::zeros<mat>(mDim, mNumPoints);

            for(int i=0; i < mDim; i++) {
                int elem = 0;
                int flip = pow(2, i);
                for(int j = 0; j < mNumPoints; j++) {
                    xSample(i, j) = elem;
                    if((j + 1) % flip == 0) { elem = (elem + 1) % 2; }
                }
            }
            xSample.col(mNumPoints - 1).fill(0.5);

            return xSample;
        }
    };
}
#endif
