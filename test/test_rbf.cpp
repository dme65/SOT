/* 
 * File:   test_rbf.cpp
 * Author: David Eriksson
 *
 * Created on July 19, 2016
 */

#include <sot.h>
using namespace sot;

// Throws an exception if the Kernel and Tail don't match
template<class Kernel, class Tail>
bool check_exception(int maxPoints, int dim) {
    bool exceptionThrown = false;
    try {
        RBFInterpolant<Kernel, Tail> rbf(maxPoints, dim); // Should throw an exception
    }
    catch (const std::logic_error& e) {
        exceptionThrown = true;
    }
    return exceptionThrown;
}

int test_kernels_tails() {
    int dim = 10;
    int maxPoints = 20;

    // Check kernel - tail compatibility

    // Cubic + Linear should work
    if(check_exception<CubicKernel, LinearTail>(maxPoints, dim)) { return EXIT_FAILURE; } // LCOV_EXCL_LINE
    // Cubic + Constant should NOT work
    if(not check_exception<CubicKernel, ConstantTail>(maxPoints, dim)) { return EXIT_FAILURE; } // LCOV_EXCL_LINE
    // TPS + Linear should work
    if(check_exception<ThinPlateKernel, LinearTail>(maxPoints, dim)) { return EXIT_FAILURE; } // LCOV_EXCL_LINE
    // TPS + Constant should NOT work
    if(not check_exception<ThinPlateKernel, ConstantTail>(maxPoints, dim)) { return EXIT_FAILURE; } // LCOV_EXCL_LINE
    // Linear + Linear should work
    if(check_exception<LinearKernel, LinearTail>(maxPoints, dim)) { return EXIT_FAILURE; } // LCOV_EXCL_LINE
    // Linear + Constant should work
    if(check_exception<LinearKernel, ConstantTail>(maxPoints, dim)) { return EXIT_FAILURE; } // LCOV_EXCL_LINE

    mat dists = arma::randu(10, 10);
    mat X = arma::randu(dim, maxPoints);
    vec x = X.col(0);

    // Check Cubic kernel
    CubicKernel cubicKernel;
    if(cubicKernel.order() != 2) { return EXIT_FAILURE; } // LCOV_EXCL_LINE
    if(std::abs(cubicKernel.phiZero()) > 1e-10) { return EXIT_FAILURE; } // LCOV_EXCL_LINE
    if(std::abs(cubicKernel.eval(2) - 8) > 1e-10) { return EXIT_FAILURE; } // LCOV_EXCL_LINE
    if(std::abs(cubicKernel.deriv(2) - 12) > 1e-10) { return EXIT_FAILURE; } // LCOV_EXCL_LINE
    if(arma::norm(cubicKernel.eval(dists) - dists % dists % dists) > 1e-10) { return EXIT_FAILURE; } // LCOV_EXCL_LINE
    if(arma::norm(cubicKernel.deriv(dists) - 3 * dists % dists) > 1e-10) { return EXIT_FAILURE; } // LCOV_EXCL_LINE

    // Check TPS kernel
    ThinPlateKernel tpsKernel;
    if(tpsKernel.order() != 2) { return EXIT_FAILURE; } // LCOV_EXCL_LINE
    if(std::abs(tpsKernel.phiZero()) > 1e-10) { return EXIT_FAILURE; } // LCOV_EXCL_LINE
    if(std::abs(tpsKernel.eval(2) - 4*std::log(2.0)) > 1e-10) { return EXIT_FAILURE; } // LCOV_EXCL_LINE
    if(std::abs(tpsKernel.deriv(2) - 2*(1 + 2*std::log(2.0))) > 1e-10) { return EXIT_FAILURE; } // LCOV_EXCL_LINE
    if(arma::norm(tpsKernel.eval(dists) - dists % dists % arma::log(dists)) > 1e-10) { return EXIT_FAILURE; } // LCOV_EXCL_LINE
    if(arma::norm(tpsKernel.deriv(dists) - dists % (1 + 2 * arma::log(dists))) > 1e-10) { return EXIT_FAILURE; } // LCOV_EXCL_LINE

    // Check Linear kernel
    LinearKernel linearKernel;
    if(linearKernel.order() != 1) { return EXIT_FAILURE; } // LCOV_EXCL_LINE
    if(std::abs(linearKernel.phiZero()) > 1e-10) { return EXIT_FAILURE; } // LCOV_EXCL_LINE
    if(std::abs(linearKernel.eval(2) - 2) > 1e-10) { return EXIT_FAILURE; } // LCOV_EXCL_LINE
    if(std::abs(linearKernel.deriv(2) - 1) > 1e-10) { return EXIT_FAILURE; } // LCOV_EXCL_LINE
    if(arma::norm(linearKernel.eval(dists) - dists) > 1e-10) { return EXIT_FAILURE; } // LCOV_EXCL_LINE
    if(arma::norm(linearKernel.deriv(dists) - arma::ones(dists.n_rows, dists.n_cols)) > 1e-10) { return EXIT_FAILURE; } // LCOV_EXCL_LINE

    // Check linear tail
    LinearTail linearTail;
    if(linearTail.degree() != 1) { return EXIT_FAILURE; } // LCOV_EXCL_LINE
    if(linearTail.dimTail(dim) != dim + 1) { return EXIT_FAILURE; } // LCOV_EXCL_LINE
    mat Y = linearTail.eval(X);
    mat Z = arma::ones(linearTail.dimTail(dim), maxPoints);
    Z.rows(1, Z.n_rows - 1) = X;
    if(arma::norm(Y - Z) > 1e-10) { return EXIT_FAILURE; } // LCOV_EXCL_LINE
    vec y = linearTail.eval(x);
    vec z = arma::ones(linearTail.dimTail(dim));
    z.rows(1, Z.n_rows - 1) = x;
    if(arma::norm(y - z) > 1e-10) { return EXIT_FAILURE; } // LCOV_EXCL_LINE
    Y = linearTail.deriv(x);
    Z = arma::zeros(linearTail.dimTail(dim), dim);
    Z.rows(1, Z.n_rows - 1) = arma::eye(dim, dim);
    if(arma::norm(Y - Z) > 1e-10) { return EXIT_FAILURE; } // LCOV_EXCL_LINE

    // Check constant tail
    ConstantTail constantTail;
    if(constantTail.degree() != 0) { return EXIT_FAILURE; } // LCOV_EXCL_LINE
    if(constantTail.dimTail(dim) != 1) { return EXIT_FAILURE; } // LCOV_EXCL_LINE
    if(arma::norm(constantTail.eval(X) - arma::ones<mat>(1, maxPoints)) > 1e-10) { return EXIT_FAILURE; } // LCOV_EXCL_LINE
    if(arma::norm(constantTail.eval(x) - arma::ones<mat>(1, 1)) > 1e-10) { return EXIT_FAILURE; } // LCOV_EXCL_LINE
    if(arma::norm(constantTail.deriv(x) - arma::zeros<mat>(1, dim)) > 1e-10) { return EXIT_FAILURE; } // LCOV_EXCL_LINE

    return EXIT_SUCCESS;
}

int test_rbf() {
    int dim = 2;

    int n = 1000;
    mat X = arma::randu<mat>(dim, n);
    vec fX = (X.row(1) % arma::sin(X.row(0)) + X.row(0) % arma::cos(X.row(1))).t();

    int m = 10;
    mat Y = arma::randu<mat>(dim, m);
    vec fY = (Y.row(1) % arma::sin(Y.row(0)) + Y.row(0) % arma::cos(Y.row(1))).t();

    CubicRBF rbf(n, dim, arma::zeros(dim), arma::ones(dim), 0.0);
    rbf.addPoints(X, fX);
    rbf.fit();

    // Evaluate at the center to see that we are interpolating
    vec vals = rbf.evals(X);
    if (arma::max(arma::abs(vals - fX)) >= 1e-10) { // LCOV_EXCL_LINE
        return (EXIT_FAILURE); // LCOV_EXCL_LINE
    }
    
    // Evaluate at some other points
    vals = rbf.evals(Y);
    if (arma::max(arma::abs(vals - fY)) >= 1e-3) { // LCOV_EXCL_LINE
        return (EXIT_FAILURE); // LCOV_EXCL_LINE
    }
    mat dists = arma::sqrt(squaredPairwiseDistance(X, Y));
    if(arma::norm(vals - rbf.evals(Y, dists)) > 1e-10) { // LCOV_EXCL_LINE
        return (EXIT_FAILURE); // LCOV_EXCL_LINE
    }
    
    // Look at derivatives
    vec pred = arma::zeros<vec>(2);
    for(int i=0; i < Y.n_cols; i++) {
        vec deriv = rbf.deriv(Y.col(i));
        pred(0) = Y(1, i) * cos(Y(0, i)) + cos(Y(1,i));
        pred(1) = sin(Y(0, i)) - Y(0, i) * sin(Y(1, i));
        if (arma::norm(deriv - pred) >= 1e-2) { // LCOV_EXCL_LINE
            return (EXIT_FAILURE); // LCOV_EXCL_LINE
        }
    }

    // Check that all of the getters are correct
    if(dim != rbf.dim() || n != rbf.numPoints()) {
        return (EXIT_FAILURE); // LCOV_EXCL_LINE
    }
    if (arma::norm(fX - rbf.fX()) > 1e-10) { // LCOV_EXCL_LINE
        return (EXIT_FAILURE); // LCOV_EXCL_LINE
    }
    if (arma::norm(X - rbf.X()) > 1e-10) { // LCOV_EXCL_LINE
        return (EXIT_FAILURE); // LCOV_EXCL_LINE
    }
    for(int i=0; i < n; i++) {
        if (std::abs(fX(i) - rbf.fX(i)) > 1e-10) { // LCOV_EXCL_LINE
            return (EXIT_FAILURE); // LCOV_EXCL_LINE
        }
        if (arma::norm(X.col(i) - rbf.X(i)) > 1e-10) { // LCOV_EXCL_LINE
            return (EXIT_FAILURE); // LCOV_EXCL_LINE
        }
    }

    // Reset the surrogate
    rbf.reset();

    // Add all points again
    mat X1 = X.cols(0, n-11);
    vec fX1 = fX.rows(0, n-11);
    mat X2 = X.cols(n-10, n-1);
    vec fX2 = fX.rows(n-10, n-1);

    rbf.addPoints(X1, fX1);
    for(int i=0; i < 10; i++) {
        vec x = X2.col(i);
        rbf.addPoint(x, fX2(i));
    }
    rbf.fit();

    // Try to add one more point (exceeds the capacity)
    // Finally check the derivative, which isn't supported
    bool exceptionThrown = false;
    try {
        vec y = Y.col(0);
        rbf.addPoint(y, fY(0));
    }
    catch (const std::logic_error& e) {
        exceptionThrown = true;
    }
    if(not exceptionThrown) { return EXIT_FAILURE; } // LCOV_EXCL_LINE

    // Check all of the evaluation methods
    vals = arma::zeros(n);
    for(int i=0; i < n; i++) {
        vals(i) = rbf.eval(X.col(i));
    }
    if (arma::max(arma::abs(vals - fX)) >= 1e-3) { // LCOV_EXCL_LINE
        return (EXIT_FAILURE); // LCOV_EXCL_LINE
    }

    // Evaluate at some other points
    vals = arma::zeros(m);
    for(int i=0; i < m; i++) {
        vec y = Y.col(i);
        vals(i) = rbf.eval(y);
        vec dists = arma::sqrt(squaredPointSetDistance(y, X));
        if (std::abs(vals(i) - rbf.eval(y, dists)) > 1e-3) { // LCOV_EXCL_LINE
            return (EXIT_FAILURE); // LCOV_EXCL_LINE
        }
    }
    if (arma::max(arma::abs(vals - fY)) >= 1e-3) { // LCOV_EXCL_LINE
        return (EXIT_FAILURE); // LCOV_EXCL_LINE
    }

    return (EXIT_SUCCESS);
}

int test_capped_rbf() {
    int dim = 2;

    int n = 1000;
    mat X = arma::randu<mat>(dim, n);
    vec fX = (X.row(1) % arma::sin(X.row(0)) + X.row(0) % arma::cos(X.row(1))).t();

    // Set half the points to crazy values
    fX.rows(0, (n/2)-2).fill(std::numeric_limits<double>::max());

    int m = 10;
    mat Y = arma::randu<mat>(dim, m);
    vec fY = (Y.row(1) % arma::sin(Y.row(0)) + Y.row(0) % arma::cos(Y.row(1))).t();

    RBFInterpolantCap<CubicKernel,LinearTail> rbf(n, dim, arma::zeros(dim), arma::ones(dim), 0.0);
    rbf.addPoints(X, fX);
    rbf.fit();

    // Evaluate at some other points
    vec vals = rbf.evals(Y);
    if (arma::max(arma::abs(vals - fY)) >= 2) { // LCOV_EXCL_LINE
        return (EXIT_FAILURE); // LCOV_EXCL_LINE
    }

    return EXIT_SUCCESS;
}

int main(int argc, char** argv) {
    int exit = EXIT_SUCCESS;

    int result = test_kernels_tails();
    if(result != EXIT_SUCCESS) { exit = EXIT_FAILURE; } // LCOV_EXCL_LINE

    result = test_rbf();
    if(result != EXIT_SUCCESS) { exit = EXIT_FAILURE; } // LCOV_EXCL_LINE

    result = test_capped_rbf();
    if(result != EXIT_SUCCESS) { exit = EXIT_FAILURE; } // LCOV_EXCL_LINE

    return exit;
}