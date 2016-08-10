/*
 * File:   test_utils.cpp
 * Author: David Eriksson
 *
 * Created on Aug 10, 2016
 */

#include <sot.h>
using namespace sot;

int test_dists() {

    int dim = 10;
    int n = 2000;
    int m = 1000;
    mat X = arma::randu(dim, n);
    mat Y = arma::randu(dim, m);

    // Use fast distance computation
    StopWatch watch;
    watch.start();
    mat Z1 = squaredPairwiseDistance(X, Y);
    double time1 = watch.stop();

    // Compare to naive distance computation
    watch.start();
    mat Z2 = arma::zeros(n, m);
    for(int i=0; i < n; i++) {
        for(int j=0; j < m; j++) {
            Z2(i, j) = arma::sum(arma::square(X.col(i) - Y.col(j)));
        }
    }
    double time2 = watch.stop();

    std::cout << "Time for fast distances: " << time1
              << "\nTime for naive distances: " << time2 << std::endl;
    std::cout << "Error: " << arma::norm(Z1 - Z2) << std::endl;

    if (arma::norm(Z1 - Z2) > 1e-10) { // LCOV_EXCL_LINE
        return (EXIT_FAILURE); // LCOV_EXCL_LINE
    }

    return EXIT_SUCCESS;
}

int test_scaling() {
    int dim = 10;
    int n = 100;
    mat X = 10*arma::randu(dim, n);
    mat Y = toUnitBox(X, arma::zeros(dim), 10*arma::ones(dim));

    // Check that points are mapped to the unit box
    if(arma::max(arma::vectorise(Y)) > 1.0 || arma::min(arma::vectorise(Y)) < 0.0) { // LCOV_EXCL_LINE
        return (EXIT_FAILURE); // LCOV_EXCL_LINE
    }

    // Check that points are mapped back to the original points
    mat Z = fromUnitBox(Y, arma::zeros(dim), 10*arma::ones(dim));
    if( arma::norm(X - Z) > 1e-10) { // LCOV_EXCL_LINE
        return (EXIT_FAILURE); // LCOV_EXCL_LINE
    }

    // Test scaling a vector
    vec x = arma::randu(n);
    vec y = unitRescale(x);
    if( std::abs(1 - arma::max(y)) > 1e-10 || std::abs(arma::min(y)) > 1e-10) {
        return (EXIT_FAILURE); // LCOV_EXCL_LINE
    }

    x = arma::ones(n);
    y = unitRescale(x);
    if(arma::min(y) != arma::max(y)) { // LCOV_EXCL_LINE
        return (EXIT_FAILURE); // LCOV_EXCL_LINE
    }

    return EXIT_SUCCESS;
}

int test_result() {
    int maxEvals = 100;
    int dim = 10;
    Result res(maxEvals, dim);

    mat X = arma::randu(dim, maxEvals);
    vec fX = arma::randu(maxEvals);
    res.addEvals(X, fX);

    // Test res
    if(res.fBest() != arma::min(fX)) { return (EXIT_FAILURE); } // LCOV_EXCL_LINE
    if(res.dim() != dim) { return (EXIT_FAILURE); } // LCOV_EXCL_LINE
    if(arma::norm(res.fX() - fX) > 1e-10) { return (EXIT_FAILURE); } // LCOV_EXCL_LINE
    if(arma::norm(res.X() - X) > 1e-10) { return (EXIT_FAILURE); } // LCOV_EXCL_LINE
    if(res.numEvals() != maxEvals) { return (EXIT_FAILURE); } // LCOV_EXCL_LINE

    // Check that the optimum is correct
    arma::uword indBest;
    double val = fX.min(indBest);
    if(arma::norm(res.xBest() - X.col(indBest)) > 1e-10) { return (EXIT_FAILURE); } // LCOV_EXCL_LINE

    // Make sure an exception is thrown is we add another point
    bool exceptionThrown = false;
    try {
        vec y = arma::randu(dim);
        res.addEval(y, 0.5);
    }
    catch (const std::logic_error& e) {
        exceptionThrown = true;
    }

    if(not exceptionThrown) { return EXIT_FAILURE; } // LCOV_EXCL_LINE

    res.reset();

    return EXIT_SUCCESS;
}

int test_pareto() {
    int n = 100;
    vec x = arma::randu(n);
    vec y = arma::randu(n);
    uvec frontInd = paretoFront(x, y);

    // Check that we found the correct pareto front
    uvec bruteInd = arma::ones<uvec>(n);
    arma::uword bruteCounter = 0;
    for(int i=0; i < n; i++) {
        bool dominated = false;
        for(int j=0; j < n;j ++) {
            // Check if i is dominated by j
            if(i != j && x(j) <= x(i) && y(j) <= y(i)) {
                dominated = true;
                break;
            }
        }
        if(not dominated) {
            bruteInd(bruteCounter) = (arma::uword) i;
            bruteCounter++;
        }
    }

    bruteInd = bruteInd.rows(0, bruteCounter-1);
    if(not arma::all(arma::sort(bruteInd) == arma::sort(frontInd))) { return (EXIT_FAILURE); } // LCOV_EXCL_LINE

    // Make sure that an exception is thrown when x and y have different length
    bool exceptionThrown = false;
    try {
        y = arma::randu(n+1);
        frontInd = paretoFront(x, y);
    }
    catch (const std::logic_error& e) {
        exceptionThrown = true;
    }

    if(not exceptionThrown) { return EXIT_FAILURE; } // LCOV_EXCL_LINE

    return EXIT_SUCCESS;
}

int test_cumMin() {
    int n = 100;
    vec x = arma::randu(n);
    vec xCumMin = cumMin(x);
    vec testCumMin = arma::zeros(n);
    testCumMin(0) = x(0);
    for(int i=1; i < n; i++) {
        testCumMin(i) = std::min(x(i), testCumMin(i-1));
    }

    if(not arma::all(xCumMin == testCumMin)) { return (EXIT_FAILURE); } // LCOV_EXCL_LINE

    return EXIT_SUCCESS;
}

int test_seed() {
    setSeed(0); // Seed with an unsigned int

    int n = 10000;
    vec x = arma::zeros(n);

    // Test the U[0,1] distribution
    for(int i=0; i < n; i++) {
        x(i) = sot::rand();
    }
    if(arma::max(x) > 1 || arma::min(x) < 0) { return (EXIT_FAILURE); } // LCOV_EXCL_LINE
    if(std::abs(arma::mean(x) - 0.5) > 0.02) { return (EXIT_FAILURE); } // LCOV_EXCL_LINE
    if(std::abs(arma::var(x) - 1.0/12) > 0.02) { return (EXIT_FAILURE); } // LCOV_EXCL_LINE

    // Test the N(0,1) distribution
    for(int i=0; i < n; i++) {
        x(i) = sot::randn();
    }
    if(std::abs(arma::mean(x)) > 0.02) { return (EXIT_FAILURE); } // LCOV_EXCL_LINE
    if(std::abs(arma::var(x) - 1) > 0.02) { return (EXIT_FAILURE); } // LCOV_EXCL_LINE

    // Test the Random integer
    for(int i=0; i < n; i++) {
        x(i) = sot::randi(10);
    }
    if(arma::max(x) != 10 || arma::min(x) != 0) { return (EXIT_FAILURE); } // LCOV_EXCL_LINE
    if(std::abs(arma::mean(x) - 5) > 0.02) { return (EXIT_FAILURE); } // LCOV_EXCL_LINE
    if(std::abs(arma::var(x) - 10) > 0.02) { return (EXIT_FAILURE); } // LCOV_EXCL_LINE

    // Set the seed to something random
    setSeedRandom();

    // Set the seed back to 0
    setSeed(0);

    return EXIT_SUCCESS;
}

int main(int argc, char** argv) {
    int exit = EXIT_SUCCESS;

    int result = test_dists();
    if(result != EXIT_SUCCESS) { exit = EXIT_FAILURE; }

    result = test_scaling();
    if(result != EXIT_SUCCESS) { exit = EXIT_FAILURE; }

    result = test_result();
    if(result != EXIT_SUCCESS) { exit = EXIT_FAILURE; }

    result = test_pareto();
    if(result != EXIT_SUCCESS) { exit = EXIT_FAILURE; }

    result = test_cumMin();
    if(result != EXIT_SUCCESS) { exit = EXIT_FAILURE; }

    result = test_seed();
    if(result != EXIT_SUCCESS) { exit = EXIT_FAILURE; }

    return exit;
}