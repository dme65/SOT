/* 
 * File:   problem.h
 * Author: David Eriksson
 *
 * Created on July 21, 2016, 1:36 AM
 */

#ifndef Surrogate_Optimization_problem_h
#define Surrogate_Optimization_problem_h

namespace sot {
    
    class Problem {
    public:
        virtual vec lBounds() const = 0;
        virtual vec uBounds() const = 0;
        virtual int dim() const = 0;
        virtual double min() const = 0;
        virtual vec optimum() const = 0;
        virtual std::string name() const = 0;
        virtual double eval(vec&) const = 0;
        vec evals(mat &points) const {
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

