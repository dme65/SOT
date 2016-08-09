/*!
 * File:   shepard.h
 * Author: David Eriksson, dme65@cornell.edu
 *
 * Created on 7/18/16.
 */


#ifndef SOT_SHEPARD_H
#define SOT_SHEPARD_H

#include "common.h"
#include "utils.h"
#include "surrogate.h"

//!SOT namespace
namespace sot {
    
    //!  %Shepard's method
    /*!
     * Shepard's method, also known as Inverse Distance Weighting (IMW), assigns 
     * function values to unknown points as a weighted average of the values 
     * available at the known points. The weights are given by 
     * \f$ w_i(x) = \|x-x_i\|^{-p/2}\f$ which makes it clear that points close 
     * to \f$x\f$ are weighted higher.
     *
     * \class Shepard
     *
     * \author David Eriksson, dme65@cornell.edu
     */
    class Shepard : public Surrogate {
    protected:
        double mp; /*!< Value of the exponent p */
        double mDistTol = 1e-10; /*!< Distance tolerance for distinguishing points */
        int mMaxPoints; /*!< Capacity */
        int mNumPoints; /*!< Current number of points */
        int mDim; /*!< Number of dimensions */
        mat mX; /*!< Current points */
        mat mfX; /*!< Current point values */
    public:
        //! Constructor
        /*!
         * \param maxPoints Capacity
         * \param dim Number of dimensions
         * \param p Value of the exponent, 2 is a common choice
         */
        Shepard(int maxPoints, int dim, double p) {
            mNumPoints = 0;
            mMaxPoints = maxPoints;
            mp = p;
            mDim = dim;
            mX.resize(dim, maxPoints);
            mfX.resize(maxPoints);
        }
        int dim() const { return mDim; }
        int numPoints() const { return mNumPoints;}
        vec X(int i) const { return mX.col(i); }
        mat X() const { return mX.cols(0, mNumPoints-1); }
        double fX(int i) const { return mfX(i); }
        vec fX() const { return mfX.rows(0, mNumPoints-1); }
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
        double eval(const vec &point, const vec &dists) const {
            return eval(point);
        }
        vec evals(const mat &points) const {
            vec vals = arma::zeros<vec>(points.n_cols);
            for(int i=0; i < points.n_cols; i++) {
                vals(i) = eval(points.col(i));
            }
            return vals;
        }
        vec evals(const mat &points, const mat &dists) const {
            return evals(points);
        }
        
        //! Method for evaluating the kNN derivative at one point (not implemented)
        /*!
         * \throws std::logic_error Not available for Shepard
         */
        vec deriv(const vec &point) const {
            throw std::logic_error("No derivatives for Shepard");
        }
        void reset() { mNumPoints = 0; }
        //! Fits the interpolant (does nothing)
        void fit() { return; }
    };
}

#endif
