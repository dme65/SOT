//
//  CandidatePoints.h
//  Surrogate Optimization
//
//  Created by David Eriksson on 7/30/15.
//  Copyright (c) 2015 David Eriksson. All rights reserved.
//

#ifndef __Surrogate_Optimization__CandidatePoints__
#define __Surrogate_Optimization__CandidatePoints__

#include "common.h"
#include "utils.h"
#include "problem.h"

namespace sot {
    
    class MeritWeightedDistance {
    protected:
        vec mWeights = {0.3, 0.5, 0.8, 0.95};
        int mNextWeight = 0;
    public:
        inline mat pickPoints(const mat &cand, const std::shared_ptr<Surrogate>& surf, const mat &points, int newPoints, double distTol) {
            int dim = cand.n_rows; 

            // Evaluate the RBF at the candidate points
            const mat dists = arma::sqrt(squaredPairwiseDistance<mat>(points, cand));
            vec surfVals = surf->evals(cand);
            vec valScores = unitRescale(surfVals);
            vec minDists = arma::min(dists).t();
            vec distScores = 1.0 - unitRescale(minDists);

            mat newx = arma::zeros<mat>(dim, newPoints);

            arma::uword winner;
            for(int i=0; i < newPoints; i++) {
                double weight = mWeights[mNextWeight % mWeights.n_elem];
                mNextWeight++;

                // Update distances if necessary
                if (i > 0) {
                    vec newDists = arma::sqrt(squaredPointSetDistance<mat,vec>((vec)newx.col(i-1), cand));
                    minDists = arma::min(minDists, newDists);
                    valScores(winner) = std::numeric_limits<double>::max();
                    distScores = 1.0 - unitRescale(minDists);
                }

                // Pick a winner
                vec merit = weight * valScores + (1.0 -  weight) * distScores;
                merit.elem(arma::find(minDists < distTol)).fill(std::numeric_limits<double>::max());
                double scores = merit.min(winner);
                newx.col(i) = cand.col(winner);
            }

            return newx;
        }
    };
    
    // Template for selecting the next evaluation
    class Sampling {
    public:
        virtual void reset(int) = 0;
        virtual mat makePoints(vec &xBest, const mat &points, double sigma, int newPoints) = 0;
    };
    
    // DYCORS
    template<class MeritFunction = MeritWeightedDistance>
    class DYCORS : public Sampling {
    protected:
        std::shared_ptr<Problem> mData;
        std::shared_ptr<Surrogate> mSurf;
        int mNumCand;
        int mDim;
        vec mxLow;
        vec mxUp;
        double mDistTol;
        int mNumEvals = 0;
        int mBudget;
        MeritFunction mMerit;
    public:
        DYCORS(const std::shared_ptr<Problem>& data, const std::shared_ptr<Surrogate>& surf, int numCand, int budget) {
            mData = std::shared_ptr<Problem>(data);
            mSurf = std::shared_ptr<Surrogate>(surf);
            mBudget = budget;
            mNumCand = numCand;
            mDim = data->dim();
            mxLow = data->lBounds();
            mxUp = data->uBounds();
            mDistTol = 1e-3*sqrt(arma::sum(arma::square(mxUp - mxLow)));
        }
        void reset(int budget) {
            mBudget = budget;
            mNumEvals = 0;
        }
        mat makePoints(vec &xBest, const mat &points, double sigma, int newPoints) {                
            double dds_prob = fmin(20.0/mDim, 1.0) * (1.0 - (log(mNumEvals + 1.0) / log(mBudget)));
            mat cand = arma::repmat(xBest, 1, mNumCand);
            for(int i=0; i < mNumCand; i++) {

                int count = 0;
                for(int j=0; j < mDim; j++) {
                    if(rand() < dds_prob) {
                        count++;
                        cand(j, i) += sigma*randn();
                    }
                }
                // If no index was perturbed we force one
                if(count == 0) {
                    int ind = randi(mDim);
                    cand(ind, i) += sigma*randn();
                }

                // Make sure we are still in the domain
                for(int j=0; j < mDim; j++) {
                    if(cand(j, i) > mxUp(j)) { 
                        cand(j, i) = fmax(2*mxUp(j) - cand(j, i), mxLow(j)); 
                    }
                    else if(cand(j, i) < mxLow(j)) { 
                        cand(j, i) = fmin(2*mxLow(j) - cand(j, i), mxUp(j)); 
                    }
                }
            }
            
            // Update counter
            mNumEvals += newPoints;
            
            return mMerit.pickPoints(cand, mSurf, points, newPoints, mDistTol);
        }
    };

    // SRBF
    template<class MeritFunction = MeritWeightedDistance>
    class SRBF : public Sampling {
    protected:
        std::shared_ptr<Problem> mData;
        std::shared_ptr<Surrogate> mSurf;
        int mNumCand;
        int mDim;
        vec mxLow;
        vec mxUp;
        double mDistTol;
        int mNumEvals = 0;
        int mBudget;
        MeritFunction mMerit;
    public:
        SRBF(const std::shared_ptr<Problem>& data, const std::shared_ptr<Surrogate>& surf, int numCand, int budget) {
            mData = std::shared_ptr<Problem>(data);
            mSurf = std::shared_ptr<Surrogate>(surf);
            mBudget = budget;
            mNumCand = numCand;
            mDim = data->dim();
            mxLow = data->lBounds();
            mxUp = data->uBounds();
            mDistTol = 1e-3*sqrt(arma::sum(arma::square(mxUp - mxLow)));
        }
        void reset(int budget) {
            mBudget = budget;
            mNumEvals = 0;
        }
        mat makePoints(vec &xBest, const mat &points, double sigma, int newPoints) {

            mat cand = arma::repmat(xBest, 1, mNumCand);

            // Perturbs one randomly chosen coordinate
            for(int i=0; i < mNumCand; i++) {
                for(int j=0; j < mDim; j++) {
                    cand(j, i) += sigma * randn();
                    if(cand(j, i) > mxUp(j)) { 
                        cand(j, i) = fmax(2*mxUp(j) - cand(j, i), mxLow(j)); 
                    }
                    else if(cand(j, i) < mxLow(j)) { 
                        cand(j, i) = fmin(2*mxLow(j) - cand(j, i), mxUp(j)); 
                    }
                }
            }
            
            // Update counter
            mNumEvals += newPoints;
            
            return mMerit.pickPoints(cand, mSurf, points, newPoints, mDistTol);
        }
    };
    
    // Uniform
    template<class MeritFunction = MeritWeightedDistance>
    class Uniform : public Sampling {
    protected:
        std::shared_ptr<Problem> mData;
        std::shared_ptr<Surrogate> mSurf;
        int mNumCand;
        int mDim;
        vec mxLow;
        vec mxUp;
        double mDistTol;
        int mNumEvals = 0;
        int mBudget;
        MeritFunction mMerit;
    public:
        Uniform(const std::shared_ptr<Problem>& data, const std::shared_ptr<Surrogate>& surf, int numCand, int budget) {
            mData = std::shared_ptr<Problem>(data);
            mSurf = std::shared_ptr<Surrogate>(surf);
            mBudget = budget;
            mNumCand = numCand;
            mDim = data->dim();
            mxLow = data->lBounds();
            mxUp = data->uBounds();
            mDistTol = 1e-3*sqrt(arma::sum(arma::square(mxUp - mxLow)));
        }
        void reset(int budget) {
            mBudget = budget;
            mNumEvals = 0;
        }
        mat makePoints(vec &xbest, const mat &points, double sigma, int newPoints) {
         
            mat cand = arma::randu<mat>(mDim, mNumCand);
            for(int j=0; j < mDim; j++) {
                cand.row(j) = mxLow(j) + (mxUp(j) - mxLow(j)) * cand.row(j);
            }
        
            // Update counter
            mNumEvals += newPoints;
            
            return mMerit.pick_points(cand, mSurf, points, newPoints, mDistTol);
        }
    };
}

#endif /* defined(__Surrogate_Optimization__CandidatePoints__) */
