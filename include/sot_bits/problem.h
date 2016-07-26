/*!
 * File:   problem.h
 * Author: David Eriksson, dme65@cornell.edu
 *
 * Created on 7/18/16.
 */


#ifndef SOT_PROBLEM_H
#define SOT_PROBLEM_H

//!SOT namespace
namespace sot {
    
    //! Abstract class for a SOT optimization problem
    /*!
     * This is the abstract class that should be used as a Base class for all
     * optimization problems in SOT.
     * 
     * \author David Eriksson, dme65@cornell.edu
     */
    class Problem {
    public:
        //! Method for getting the lower variable bounds.
        virtual vec lBounds() const = 0; /*!< \returns Upper variable bounds */      
        //! Method for getting the upper variable bounds.
        virtual vec uBounds() const = 0; /*!< \returns Upper variable bounds */
        //! Method for getting the number of dimensions      
        virtual int dim() const = 0; /*!< \returns The number of dimensions */
        //! Method for getting global minimum value
        virtual double min() const = 0; /*!< \returns Value at the global minumum */
        //! Method for getting the global minimizer
        virtual vec optimum() const = 0; /*!< \returns The global minimizer */
        //! Method for getting the name of the optimization problem
        virtual std::string name() const = 0; /*!< \returns The optimization problem name */
        //! Method for evaluating the objective function
        /*!
         * \param point Is the next point for which to evaluate the objective function
         * \return The value of the objective function at the input
         */
        virtual double eval(const vec &point) const = 0;
        //! Method for evaluating the objective functions at multiple points
        /*!
         * \param points The points for which to evaluate the objective function
         * \return The values of the objective function at the inputs
         */
        vec evals(const mat &points) const {
            vec fvals = arma::zeros<vec>(points.n_cols);
            for(int i=0; i < points.n_cols; i++) {
                vec x = points.col(i);
                fvals(i) = eval(x);
            }
            return fvals;
        }
    };
}

#endif

