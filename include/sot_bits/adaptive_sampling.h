/*!
 * File:   adaptive_sampling.h
 * Author: David Eriksson, dme65@cornell.edu
 *
 * Created on 7/18/16.
 */

#ifndef SOT_ADAPTIVE_SAMPLING_H
#define SOT_ADAPTIVE_SAMPLING_H

#include "common.h"
#include "utils.h"
#include "problem.h"
#include "genetic_algorithm.h"

//!SOT namespace
namespace sot {
    
    //!  Abstract class for a SOT adaptive sampling class
    /*!
     * This is the abstract class that should be used as a Base class for all
     * sampling objects in SOT. The sampling object is used to propose new 
     * evaluations after the initial experimental design has been evaluated.
     *
     * \class Sampling
     * \author David Eriksson, dme65@cornell.edu
     */
    
    class Sampling {
    public:
        //! Virtual method for reseting the object
        /*!
         * \param budget The remaining evaluation budget
         */
        virtual void reset(int budget) = 0;
        
        //! Virtual method for proposing new evaluations
        /*!
         * \param xBest The best solution found so far
         * \param points Previously evaluated points
         * \param sigma The sampling radius for each dimension
         * \param newPoints Number of new evaluations to be generated
         * \return The proposed points
         */
        virtual mat makePoints(const vec &xBest, const mat &points, const vec &sigma, int newPoints) = 0;
    };
    
    //! Stochastic RBF
    /*!
     * This is an implementation of the SRBF method that generates the candidate
     * points by perturbing each variable by a normally distrubuted realization. 
     *
     * \class SRBF
     *
     * \tparam MeritFunction The merit function is used to pick the most promising out of 
     * the generated candidate points.
     *
     * \author David Eriksson, dme65@cornell.edu
     */
    template<class MeritFunction = MeritWeightedDistance>
    class SRBF : public Sampling {
    protected:
        std::shared_ptr<Problem> mData; /*!< A shared pointer to the optimization problem */
        std::shared_ptr<Surrogate> mSurf; /*!< A shared pointer to the surrogate model */
        int mNumCand; /*!< Number of candidate points that are generated in makePoints */
        int mDim; /*!< Number of dimensions (extracted from mData) */
        vec mxLow; /*!< Lower variable bounds (extracted from mData) */
        vec mxUp; /*!< Upper variable bounds (extracted from mData) */
        double mDistTol; /*!< Distance tolerance */
        int mNumEvals = 0; /*!< Current evaluation count */
        int mBudget; /*!< Evaluation budget for the adaptive sampling phase */
        MeritFunction mMerit; /*!< Merit function that is used for picking candidate points */
    public:
        //! Constructor
        /*!
         * \param data A shared pointer to the optimization problem
         * \param surf A shared pointer to the surrogate model
         * \param numCand Number of candidate points that are generated in makePoints
         * \param budget Evaluation budget for the adaptive sampling phase
         */
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
        
        //! Resets the object for a new budget (useful if a strategy restarts)
        /*!
         * \param budget New evaluation budget
         */
        void reset(int budget) {
            mBudget = budget;
            mNumEvals = 0;
        }
        
        //! Proposes new evaluations
        /*!
         * \param xBest The best solution found so far
         * \param points Previously evaluated points
         * \param sigma The sampling radius for each dimension
         * \param newPoints Number of new evaluations to be generated
         * \return The proposed points
         */
        mat makePoints(const vec &xBest, const mat &points, const vec &sigma, int newPoints) {

            // Generate perturbations
            mat cand = arma::repmat(xBest, 1, mNumCand);
            mat pert = arma::randn(mDim, mNumCand);
            pert.each_col() %= sigma;
            cand += pert;
            
            for(int i=0; i < mNumCand; i++) {
                for(int j=0; j < mDim; j++) {
                    if(cand(j, i) > mxUp(j)) { // 
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
    
    //! DYnamic COordinate search using Response Surface models
    /*!
     * This is an implementation of the DYCORS method that perturbs fewer and
     * fewer variables as the optimization proceeds. The candidate points
     * are generated by perturbing each variable using the probability proposed
     * by DYCORS. 
     *
     * \class DYCORS
     *
     * \tparam MeritFunction The merit function is used to pick the most promising out of 
     * the generated candidate points.
     * 
     * \todo Should use SRBF as a Base class
     * 
     * \author David Eriksson, dme65@cornell.edu
     */
    
    template<class MeritFunction = MeritWeightedDistance>
    class DYCORS : public Sampling {
    protected:
        std::shared_ptr<Problem> mData; /*!< A shared pointer to the optimization problem */
        std::shared_ptr<Surrogate> mSurf; /*!< A shared pointer to the surrogate model */
        int mNumCand; /*!< Number of candidate points that are generated in makePoints */
        int mDim; /*!< Number of dimensions (extracted from mData) */
        vec mxLow; /*!< Lower variable bounds (extracted from mData) */
        vec mxUp;  /*!< Upper variable bounds (extracted from mData) */
        double mDistTol; /*!< Distance tolerance */
        int mNumEvals = 0; /*!< Current evaluation count */
        int mBudget; /*!< Evaluation budget for the adaptive sampling phase */
        MeritFunction mMerit; /*!< Merit function that is used for picking candidate points */
    public:
        //! Constructor
        /*!
         * \param data A shared pointer to the optimization problem
         * \param surf A shared pointer to the surrogate model
         * \param numCand Number of candidate points that are generated in makePoints
         * \param budget Evaluation budget for the adaptive sampling phase
         */
        DYCORS(const std::shared_ptr<Problem>& data, const std::shared_ptr<Surrogate>& surf, 
                int numCand, int budget) {
            mData = std::shared_ptr<Problem>(data);
            mSurf = std::shared_ptr<Surrogate>(surf);
            mBudget = budget;
            mNumCand = numCand;
            mDim = data->dim();
            mxLow = data->lBounds();
            mxUp = data->uBounds();
            mDistTol = 1e-3*sqrt(arma::sum(arma::square(mxUp - mxLow)));
        }
        
        //! Resets the object for a new budget (useful if a strategy restarts)
        /*!
         * \param budget New evaluation budget
         */
        void reset(int budget) {
            mBudget = budget;
            mNumEvals = 0;
        }
        
        //! Proposes new evaluations
        /*!
         * \param xBest The best solution found so far
         * \param points Previously evaluated points
         * \param sigma The sampling radius
         * \param newPoints Number of new evaluations to be generated
         * \return The proposed points
         */
        mat makePoints(const vec &xBest, const mat &points, const vec &sigma, int newPoints) {                
            double dds_prob = std::min(20.0/mDim, 1.0) * 
                (1.0 - (std::log(mNumEvals + 1.0) / std::log(double(mBudget))));
            mat cand = arma::repmat(xBest, 1, mNumCand);
            
            for(int i=0; i < mNumCand; i++) {

                int count = 0;
                for(int j=0; j < mDim; j++) {
                    if(rand() < dds_prob) {
                        count++;
                        cand(j, i) += sigma(j)*randn();
                    }
                }
                // If no index was perturbed we force one
                if(count == 0) {
                    int ind = randi(mDim);
                    cand(ind, i) += sigma(ind)*randn();
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
    
    //! Uniformly chosen candidate points
    /*!
     * This method generates each candidate points as a uniformly chosen point from
     * the domain.
     *
     * \class Uniform
     *
     * \tparam MeritFunction The merit function is used to pick the most promising out of 
     * the generated candidate points.
     * 
     * \todo Should use SRBF as a Base class
     * 
     * \author David Eriksson, dme65@cornell.edu
     */    
    template<class MeritFunction = MeritWeightedDistance>
    class Uniform : public Sampling {
    protected:
        std::shared_ptr<Problem> mData; /*!< A shared pointer to the optimization problem */
        std::shared_ptr<Surrogate> mSurf; /*!< A shared pointer to the surrogate model */
        int mNumCand; /*!< Number of candidate points that are generated in makePoints */
        int mDim; /*!< Number of dimensions (extracted from mData) */
        vec mxLow; /*!< Lower variable bounds (extracted from mData) */
        vec mxUp;  /*!< Upper variable bounds (extracted from mData) */
        double mDistTol; /*!< Distance tolerance */
        int mNumEvals = 0; /*!< Current evaluation count */
        int mBudget; /*!< Evaluation budget for the adaptive sampling phase */
        MeritFunction mMerit; /*!< Merit function that is used for picking candidate points */
    public:
        //! Constructor
        /*!
         * \param data A shared pointer to the optimization problem
         * \param surf A shared pointer to the surrogate model
         * \param numCand Number of candidate points that are generated in makePoints
         * \param budget Evaluation budget for the adaptive sampling phase
         */
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
        
        //! Resets the object for a new budget (useful if a strategy restarts)
        /*!
         * \param budget New evaluation budget
         */
        void reset(int budget) {
            mBudget = budget;
            mNumEvals = 0;
        }
        
        //! Proposes new evaluations
        /*!
         * \param xBest The best solution found so far
         * \param points Previously evaluated points
         * \param sigma The sampling radius
         * \param newPoints Number of new evaluations to be generated
         * \return The proposed points
         */
        mat makePoints(const vec &xBest, const mat &points, const vec &sigma, int newPoints) {
         
            mat cand = arma::randu<mat>(mDim, mNumCand);
            for(int j=0; j < mDim; j++) {
                cand.row(j) = mxLow(j) + (mxUp(j) - mxLow(j)) * cand.row(j);
            }
        
            // Update counter
            mNumEvals += newPoints;
            
            return mMerit.pickPoints(cand, mSurf, points, newPoints, mDistTol);
        }
    };

    //! Wrapper to turn a surrogate model into an optimization problem
    /*!
     * This method generates each candidate points as a uniformly chosen point from
     * the domain.
     *
     * \class GAWrapper
     *
     * \author David Eriksson, dme65@cornell.edu
     */
    class GAWrapper : public Problem {
    protected:
        std::shared_ptr<Surrogate> mSurf; /*!< Surrogate model */
        int mDim; /*!< Number of dimensions */  
        mat mPoints; /*!< Previous evaluations */
        vec mxLow; /*!< Lower variable bounds */     
        vec mxUp; /*!< Upper variable bounds */     
        vec mOptimum; /*!< Global minimizer */     
        double mMinimum = 0; /*!< Global minimum value */   
        std::string mName = "GA surrogate wrapper"; /*!< Optimization problem name */  
        double mDistTol; /*!< Distance tolerance */
    public:
        //! Constructor
        /*!
         * \param data A shared pointer to the optimization problem
         * \param surf A shared pointer to the surrogate model
         * \param points Previous evaluations
         * \param distTol Distance tolerance
         */
        GAWrapper(const std::shared_ptr<Problem> &data, const std::shared_ptr<Surrogate> &surf,
                const mat &points, double distTol) {
            mDim = data->dim();
            mxLow = data->lBounds();
            mxUp = data->uBounds();
            mSurf = std::shared_ptr<Surrogate>(surf);
            mPoints = points;
            mDistTol = distTol;
        }
        vec lBounds() const { return mxLow; }
        vec uBounds() const { return mxUp; }
        int dim() const { return mDim; }
        double min() const { return mMinimum; }
        vec optimum() const { return mOptimum; }
        std::string name() const { return mName; }
        double eval(const vec &x) const { return 0; }
        //! Method for evaluating the objective function at multiple points
        /*!
         * \param X Is the next points for which to evaluate the objective function
         * \return The values of the objective function at the input
         */
        vec evals(const mat &X) const {
            mat dists = arma::sqrt(squaredPairwiseDistance<mat>(mPoints, X));
            // Evaluate the Surrogate at the points
            vec surfVals = mSurf->evals(X, dists);
            vec minDists = arma::min(dists).t();
            // Set the points that are too close to something large
            surfVals.elem(arma::find(minDists < mDistTol)).fill(arma::datum::inf);
            return surfVals;
        }
    };

    //! Use a GA to minimize the surrogate
    /*!
     * This method generates each candidate points as a uniformly chosen point from
     * the domain.
     *
     * \class GASampling
     *
     * \author David Eriksson, dme65@cornell.edu
     */
    class GASampling : public Sampling {
    protected:
        std::shared_ptr<Problem> mData; /*!< A shared pointer to the optimization problem */
        std::shared_ptr<Surrogate> mSurf; /*!< A shared pointer to the surrogate model */
        int mDim; /*!< Number of dimensions (extracted from mData) */
        vec mxLow; /*!< Lower variable bounds (extracted from mData) */
        vec mxUp;  /*!< Upper variable bounds (extracted from mData) */
        int mNumIndividuals; /*!< Population size */
        int mNumGenerations; /*!< Number of generations */
        double mDistTol; /*!< Distance tolerance */
    public:
        //! Constructor
        /*!
         * \param data A shared pointer to the optimization problem
         * \param surf A shared pointer to the surrogate model
         * \param numIndividuals Population size
         * \param numGenerations Number of generations
         */
        GASampling(const std::shared_ptr<Problem>& data, const std::shared_ptr<Surrogate>& surf,
                   int numIndividuals, int numGenerations) {
            mData = std::shared_ptr<Problem>(data);
            mSurf = std::shared_ptr<Surrogate>(surf);
            mDim = data->dim();
            mxLow = data->lBounds();
            mxUp = data->uBounds();
            mNumIndividuals = numIndividuals;
            mNumGenerations = numGenerations;
            mDistTol = 1e-3*sqrt(arma::sum(arma::square(mxUp - mxLow)));
        }
        
        void reset(int budget) {}

        mat makePoints(const vec &xBest, const mat &points, const vec &sigma, int newPoints) {
            if (newPoints > 1) {
                mat points2 = points;
                mat xNew = arma::zeros(mDim, newPoints);
                for(int i=0; i < newPoints; i++) {
                    std::shared_ptr<Problem> gaWrapper(
                            std::make_shared<GAWrapper>(mData, mSurf, points2, arma::norm(sigma)));
                    GeneticAlgorithm ga(gaWrapper, mNumIndividuals, mNumGenerations);
                    Result res = ga.run();
                    xNew.col(i) = res.xBest();
                    points2 = arma::join_horiz(points2, res.xBest());
                }
                return xNew;
            }
            else {
                std::shared_ptr<Problem> gaWrapper(
                        std::make_shared<GAWrapper>(mData, mSurf, points, arma::norm(sigma)));
                GeneticAlgorithm ga(gaWrapper, mNumIndividuals, mNumGenerations);
                Result res = ga.run();
                return res.xBest();
            }
        }
    };
}

#endif
