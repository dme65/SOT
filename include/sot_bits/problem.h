/* 
 * File:   problem.h
 * Author: David Eriksson
 *
 * Created on July 21, 2016, 1:36 AM
 */

#ifndef PROBLEM_H
#define PROBLEM_H

namespace sot {
    
    class Problem {
    protected:
        int d;
        vec xlow;
        vec xup;
        double global_min;
        vec optimum;
        std::string my_name;
    public:
        vec lbound() const { return this->xlow; }
        vec rbound() const { return this->xup; }
        int dim() const { return this->d; }
        double min() const { return this->global_min; }
        vec minimum() const { return this->optimum; }
        std::string name() const { return this->my_name; }
        virtual double eval(vec) const = 0;
        virtual vec evals(mat &x) const {
            vec fvals = arma::zeros<vec>(x.n_cols);
            for(int i=0; i < x.n_cols; i++) {
                fvals(i) = eval(x.col(i));
            }
            return fvals;
        }
    };
}

#endif /* PROBLEM_H */

