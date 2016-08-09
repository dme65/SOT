/*!
 * File:   experimental_design.h
 * Author: David Eriksson, dme65@cornell.edu
 *
 * Created on 7/18/16.
 */

#ifndef SOT_EXPERIMENTAL_DESIGN_H
#define SOT_EXPERIMENTAL_DESIGN_H

#include "common.h"
#include "utils.h"
#include "merit_functions.h"

//!SOT namespace
namespace sot {
    
    //!  Abstract class for a SOT experimental design class
    /*!
     * This is the abstract class that should be used as a Base class for all
     * experimental design objects in SOT. The experimental design generates an
     * initial set of points that are first evaluated and then used to fit the
     * initial surrogate model. All experimental design should be generated for
     * the hypercube [0, 1]^dim.
     *
     * \class ExpDesign
     *
     * \author David Eriksson, dme65@cornell.edu
     */
    
    class ExpDesign {
    public:
        //! Virtual method for getting the number of dimensions
        virtual int dim() const = 0; /*!< \returns The number of dimensions */
        //! Virtual method for getting the number of points in the experimental design
        virtual int numPoints() const = 0; /*!< \returns The number of points */
        //! Virtual method for generating an experimental design
        virtual mat generatePoints() const = 0; /*!< \returns An experimental design */
    };
    
    //!  Fixed experimental design
    /*!
     * This is a simple class that always returns the experimental design points
     * that were supplied by the user when the object was constructed. This
     * object is handy in order to compare different algorithms for the same
     * experimental design points.
     *
     * \class FixedDesign
     *
     * \author David Eriksson, dme65@cornell.edu
     */
    
    class FixedDesign : public ExpDesign {
    protected:
        int mDim; /*!< Number of dimensions */
        int mNumPoints; /*!< Number of points in the experimental design */
        mat mPoints; /*!< The experimental design points supplied by the user */
    public:
        //! Constructor
        /*!
          * \param points Experimental design points
          */
        FixedDesign(mat& points) { 
            mPoints = points; 
            mDim = points.n_rows; 
            mNumPoints = points.n_cols; 
        }
        //! Method for getting the number of dimensions
        int dim() const { return mDim; } /*!< \returns The number of dimensions */
        //! Method for getting the number of points in the experimental design
        int numPoints() const { return mNumPoints; } /*!< \returns The number of points */
        //! Method that returns the user supplied experimental design
        mat generatePoints() const { return mPoints; } /*!< \returns The fixed experimental design */
    };
        
    //!  Symmetric Latin hypercube design
    /*!
     * Symmetric Latin hypercube sampling are popular for generating near-random
     * samples of parameter values from a multidimensional distribution. The
     * Symmetric Latin hypercube does better than the original Latin hypercube
     * when it comes to entropy and maximin and is the experimental design of
     * choice for surrogate optimization. Due to rank-deficiencies it's 
     * recommended to use 2*dim points to ensure that the sample has rank dim.
     *
     * \class SymmetricLatinHypercube
     *
     * \author David Eriksson, dme65@cornell.edu
     */
    class SymmetricLatinHypercube : public ExpDesign {
    protected:
        int mDim; /*!< Number of dimensions */
        int mNumPoints; /*!< Number of points in the experimental design */

    public:
        //! Constructor
        /*!
         * \param numPoints Number of points in the experimental design
         * \param dim Number of dimensions 
         */
        SymmetricLatinHypercube(int numPoints, int dim) {
            mNumPoints = numPoints;
            mDim = dim;
        }
        //! Method for getting the number of dimensions
        int dim() const { return mDim; } /*!< \returns The number of dimensions */
        //! Method for getting the number of points in the experimental design
        int numPoints() const { return mNumPoints; } /*!< \returns The number of points */
        //! Method that generates a symmetric Latin hypercube design
        /*!
         *  \returns A symmetric Latin hypercube design
         */ 
        mat generatePoints() const { 
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
    };
    
    //!  Latin hypercube design
    /*!
     * This is a simple class that always returns the experimental design points
     * that were supplied by the user when the object was constructed. This
     * object is handy in order to compare different algorithms for the same
     * experimental design points.
     *
     * \class LatinHypercube
     *
     * \author David Eriksson, dme65@cornell.edu
     */
    class LatinHypercube : public ExpDesign {
    protected:
        int mDim; /*!< Number of dimensions */
        int mNumPoints; /*!< Number of points in the experimental design */
    public:
        //! Constructor
        /*!
         * \param numPoints Number of points in the experimental design
         * \param dim Number of dimensions 
         */
        LatinHypercube(int numPoints, int dim) {
            mNumPoints = numPoints;
            mDim = dim;
        }
        //! Method for getting the number of dimensions
        int dim() const { return mDim; } /*!< \returns The number of dimensions */
        //! Method for getting the number of points in the experimental design
        int numPoints() const { return mNumPoints; } /*!< \returns The number of points */
        //! Method that generates a symmetric Latin hypercube design
        /*!
         *  \returns A symmetric Latin hypercube design
         */ 
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
    
    //!  2-Factorial design
    /*!
     * The 2-Factorial design is the corners of the hypercube [0,1]^dim and
     * therefore consists of exactly 2^dim. It's a popular experimental design
     * for low-dimensional problems.
     *
     * \class TwoFactorial
     *
     * \author David Eriksson, dme65@cornell.edu
     */
    class TwoFactorial : public ExpDesign {
    protected:
        int mDim; /*!< Number of dimensions */
        int mNumPoints; /*!< Number of points in the experimental design */
    public:
        //! Constructor
        /*!
         * \param dim Number of dimensions 
         */
        TwoFactorial(int dim) {
            mNumPoints = pow(2, dim);
            mDim = dim;
            if(dim >= 15) {throw std::logic_error("Using 2-Factorial for dim >= 15 is a bad idea"); }
        }
        //! Method for getting the number of dimensions
        int dim() const { return mDim; } /*!< \returns The number of dimensions */
        //! Method for getting the number of points in the experimental design
        int numPoints() const { return mNumPoints; } /*!< \returns The number of points */
        //! Method that generates a symmetric Latin hypercube design
        /*!
         *  \returns A 2-Factorial design
         */ 
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
    
    //!  Corners + Midpoint
    /*!
     * This is an experimental design that consists of the 2-Factorial design 
     * plus the midpoint of the [0,1]^dim hypercube.
     *
     * \class CornersMid
     *
     * \author David Eriksson, dme65@cornell.edu
     */
    class CornersMid : public ExpDesign {
    protected:
        int mDim; /*!< Number of dimensions */
        int mNumPoints; /*!< Number of points in the experimental design */
    public:
       //! Constructor
        /*!
         * \param dim Number of dimensions 
         */
        CornersMid(int dim) {
            mNumPoints = 1 + pow(2, dim);
            mDim = dim;
            if(dim >= 15) {throw std::logic_error("Using Corners + Mid for dim >= 15 is a bad idea"); }
        }
        //! Method for getting the number of dimensions
        int dim() const { return mDim; } /*!< \returns The number of dimensions */
        //! Method for getting the number of points in the experimental design
        int numPoints() const { return mNumPoints; } /*!< \returns The number of points */
        //! Method that generates a symmetric Latin hypercube design
        /*!
         *  \returns A 2-Factorial design
         */ 
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
