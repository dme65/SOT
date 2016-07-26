/*!
 * File:   dds.h
 * Author: David Eriksson, dme65@cornell.edu
 *
 * Created on 7/18/16.
 */

#ifndef SOT_DDS_H
#define SOT_DDS_H

#include <iostream>
#include "common.h"
#include "utils.h"

//!SOT namespace
namespace sot {

    //!  The Dynamically Dimensioned Search optimization algorithm
    /*!
     * This is an implementation of the DDS algorithm. DDS generates a new point
     * to evaluate by perturbing the best solution found so far. Each variable
     * is perturbed with a probability that decreases with the number of 
     * evaluations carried out, which means that fewer and fewer variables are 
     * perturbed throughout the optimization process.
     * 
     * \author David Eriksson, dme65@cornell.edu
     */
    
    class DDS {
    protected:
        std::shared_ptr<Problem> mData; /*!< A shared pointer to the optimization problem */
        std::shared_ptr<ExpDesign> mExpDes; /*!< A shared pointer to the experimental design */
        int mMaxEvals; /*!< Evaluation budget */
        int mNumEvals; /*!< Evaluation counter */
        int mInitPoints; /*!< Number of points in the experimental design (extracted from mExpDes) */
        int mDim; /*!< Number of dimensions (extracted from mData) */
        vec mxLow; /*!< Lower variable bounds (extracted from mData) */
        vec mxUp; /*!< Upper variable bounds (extracted from mData) */
        std::string mName = "DDS"; /*!< Strategy name */
    public:
        //! Constructor
        /*!
         * \param data A shared pointer to the optimization problem
         * \param expDes A shared pointer to the experimental design
         * \param maxEvals Evaluation budget
         */
        DDS(std::shared_ptr<Problem>& data, std::shared_ptr<ExpDesign>& expDes, int maxEvals) {
            mData = std::shared_ptr<Problem>(data);
            mExpDes = std::shared_ptr<ExpDesign>(expDes);
            mMaxEvals = maxEvals;
            mNumEvals = 0;
            mInitPoints = expDes->numPoints();
            mDim = data->dim();
            mxLow = data->lBounds();
            mxUp = data->uBounds();
            mName = "DDS";
            if(mMaxEvals < mInitPoints) { throw std::logic_error("Experimental design larger than evaluation budget"); }
        }
        
        //! Runs the optimization algorithm
        /*!
         * \return A Result object with the results from the run
         */
        Result run() {
            Result res(mMaxEvals, mDim);
            mNumEvals = 0;
            
            double sigma = 0.2*(mxUp(0) - mxLow(0));
            mat init_des = fromUnitBox(mExpDes->generatePoints(), mxLow, mxUp);
            
            for(int i=0; i < mInitPoints ; i++) {
                vec x = init_des.col(i);
                res.addEval(x, mData->eval(x));
                mNumEvals++;
            }
            
            while (mNumEvals < mMaxEvals) {
                
                ////////////////////////// Select a new evaluations //////////////////////////
                double ddsProb = 1 - log(mNumEvals - mInitPoints)/log(mMaxEvals - mInitPoints);
                ddsProb = fmax(ddsProb, 1.0/mDim);
                
                vec cand = res.xBest();
                int count = 0;
                for(int j=0; j < mDim; j++) {
                    if(rand() < ddsProb) {
                        count++;
                        cand(j) += sigma * randn();
                        if(cand(j) > mxUp(j)) { 
                            cand(j) = fmax(2*mxUp(j) - cand(j), mxLow(j)); 
                        }
                        else if(cand(j) < mxLow(j)) { 
                            cand(j) = fmin(2*mxLow(j) - cand(j), mxUp(j)); 
                        }
                    }
                }
                // If no index was perturbed we force one
                if(count == 0) {
                    int ind = randi(mDim);
                    cand(ind) += sigma * randn();
                    if(cand(ind) > mxUp(ind)) { 
                        cand(ind) = fmax(2*mxUp(ind) - cand(ind), mxLow(ind)); 
                    }
                    else if(cand(ind) < mxLow(ind)) { 
                        cand(ind) = fmin(2*mxLow(ind) - cand(ind), mxUp(ind)); 
                    }
                }
                
                /////////////////////// Evaluate ///////////////////////
                res.addEval(cand, mData->eval(cand));
                
                mNumEvals++;
            }
                                
            return res;
        }
    };
}

#endif
