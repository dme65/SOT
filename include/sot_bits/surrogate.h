/*!
 * File:   surrogate.h
 * Author: David Eriksson, dme65@cornell.edu
 *
 * Created on 7/18/16.
 */

#ifndef SOT_SURROGATE_H
#define SOT_SURROGATE_H

#include "common.h"

//!SOT namespace
namespace sot {
    
    //! Abstract class for a SOT surrogate model
    /*!
     * This is the abstract class that should be used as a Base class for all
     * surrogate models in SOT.
     * 
     * \author David Eriksson, dme65@cornell.edu
     */
    class Surrogate {
    public:
        //! Method for getting the current number of points
        virtual int numPoints() const = 0; /*!< \returns Current number of points */
        //! Method for getting the number of dimensions
        virtual int dim() const = 0; /*!< \returns Number of dimensions */
        //! Method for resetting the surrogate model
        virtual void reset() = 0;
        //! Method for getting the current points
        virtual mat X() const = 0; /*!< \returns Current points */
        //! Method for getting current point number i (0 is the first)
        virtual vec X(int i) const = 0; /*!< \returns Point number i */
        //! Method for getting the values of the current points
        virtual vec fX() const = 0; /*!< \returns Values of current points */
        //! Method for getting the value of current point number i (0 is the first)
        virtual double fX(int i) const = 0; /*!< \returns Value of point number i */
        //! Method for adding a point with a known value
        /*!
         * \param point Point to be added
         * \param funVal Function value at point
         */
        virtual void addPoint(const vec &point, double funVal) = 0;
        //! Method for adding multiple points with known values
        /*!
         * \param points Points to be added
         * \param funVals Function values at the points
         */
        virtual void addPoints(const mat &points, const vec &funVals) = 0;
        //! Method for evaluating the surrogate model at a point
        /*!
         * \param point Point for which to evaluate the surrogate
         * \returns Value of the surrogate model at the point
         */
        virtual double eval(const vec &point) const = 0;
        //! Method for evaluating the surrogate at multiple points
        /*!
         * \param points Points for which to evaluate the surrogate model
         * \returns Values of the surrogate model at the points
         */
        virtual vec evals(const mat &points) const = 0;
        //! Method for evaluating the derivative of the surrogate model at a point
        /*!
         * \param point Point for which to evaluate the surrogate model
         * \returns Value of the derivative of the surrogate model at the points
         */
        virtual vec deriv(const vec &point) const = 0;
        //! Method for fitting the surrogate model
        virtual void fit() = 0;
    };
}


#endif

