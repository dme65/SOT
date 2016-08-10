/* 
 * File:   test_shepard.cpp
 * Author: David Eriksson
 *
 * Created on July 19, 2016
 */

#include <sot.h>
using namespace sot;

int test_shepard() {    
    int dim = 2;
    int n = 500;
    int m = 10;
    mat x = arma::randu<mat>(dim, n);
    vec fx = (x.row(1) % arma::sin(x.row(0)) + x.row(0) % arma::cos(x.row(1))).t();
    mat y = arma::randu<mat>(dim, m);;
    vec fy = (y.row(1) % arma::sin(y.row(0)) + y.row(0) % arma::cos(y.row(1))).t();

    Shepard surf(n, dim, 2);
    surf.addPoints(x, fx);
    surf.fit();

    // Evaluate at the center to see that we are interpolating
    vec vals = surf.evals(x);
    if (arma::max(arma::abs(vals - fx)) >= 1e-1) { // LCOV_EXCL_LINE
        return (EXIT_FAILURE); // LCOV_EXCL_LINE
    }
    
    // Evaluate at some other points
    vals = surf.evals(y);
    if (arma::max(arma::abs(vals - fy)) >= 2e-1) { // LCOV_EXCL_LINE
        return (EXIT_FAILURE); // LCOV_EXCL_LINE
    }
        
    return (EXIT_SUCCESS);
}

int main(int argc, char** argv) {
    return test_shepard();
}