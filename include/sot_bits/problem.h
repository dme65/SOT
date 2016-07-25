/*!
 * File:   problem.h
 * Author: David Eriksson
 *
 * Created on July 21, 2016, 1:36 AM
 */

#ifndef Surrogate_Optimization_problem_h
#define Surrogate_Optimization_problem_h

namespace sot {
    
    //!  Shepard interpolation
    /*!
        A more elaborate class description.
    */
    
    class Problem {
    public:
        
        //! Virtual method for getting the lower variable bounds.
        virtual vec lBounds() const = 0;        
        //! Virtual method for getting the upper variable bounds.
        virtual vec uBounds() const = 0;
        //! Virtual method for getting the dimensionality
        virtual int dim() const = 0;
        //! Virtual method for getting global minimum
        virtual double min() const = 0;
        //! Virtual for getting the input that is the global minimizer
        virtual vec optimum() const = 0;
        //! Virtual method for getting the name of the optimization problem
        virtual std::string name() const = 0;
        // Virtual method for evaluating the objective function
        /*!
            \param The point for which to evaluate the objective function
            \return The value of the objective function at the input
        */
        virtual double eval(const vec&) const = 0;
        // Method for evaluating the objective functions at multiple points
        /*!
            \param The points for which to evaluate the objective function
            \return The values of the objective function at the inputs
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

