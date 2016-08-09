/* 
 * File:   test_shepard.cpp
 * Author: David Eriksson
 *
 * Created on July 19, 2016, 12:44 PM
 */

#include <sot.h>
using namespace sot;

int test_shepard() {    
    int dim = 2;
    int n = 500;
    int m = 10;
    mat x = arma::randu<mat>(dim, n);
    vec fx = arma::zeros<vec>(n);
    for(int i=0; i<n; i++) {
        fx(i) = x(1, i) * sin(x(0, i)) + x(0, i) * cos(x(1, i));
    }
    mat y = arma::randu<mat>(dim, m);;

    Shepard surf(n, dim, 2);
    surf.addPoints(x, fx);
    surf.fit();

    // Evaluate at the center to see that we are interpolating
    vec vals = surf.evals(x);
    for(int i=0; i < x.n_cols; i++) {
        if (fabs(vals(i) - fx(i)) >= 1e-1) { // LCOV_EXCL_LINE
            return (EXIT_FAILURE); // LCOV_EXCL_LINE
        }
    }
    
    // Evaluate at some other points
    vals = surf.evals(y);
    for(int i=0; i < y.n_cols; i++) {
        double fval = y(1, i) * sin(y(0, i)) + y(0, i) * cos(y(1, i));
        if (fabs(vals(i) - fval) >= 2e-1) { // LCOV_EXCL_LINE
            return (EXIT_FAILURE); // LCOV_EXCL_LINE
        }
    }
        
    return (EXIT_SUCCESS);
}

int main(int argc, char** argv) {
    return test_shepard();
}