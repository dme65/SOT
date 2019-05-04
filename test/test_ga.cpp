/* 
 * File:   test_ga.cpp
 * Author: David Eriksson
 *
 * Created on July 25, 2016
 */

#include <sot.h>
using namespace sot;

int test_dycors() {

    int dim = 10;
    int numIndivuduals = 50;
    int numGenerations = 10;

    for(int i=1; i <= 2; i++) {
        std::cout << "GA with " << i << " threads" << std::endl;

        std::shared_ptr<Problem> data(std::make_shared<Ackley>(dim));
        std::shared_ptr<ExpDesign> slhd(std::make_shared<SLHD>(numIndivuduals, dim));

        setSeedRandom();
        GeneticAlgorithm opt(data, slhd, numIndivuduals, numGenerations, i);
        Result res = opt.run();

        // Best value found
        std::cout << res.fBest() << std::endl;

        // Scale back the best solution to the original domain
        std::shared_ptr<Problem> origData(std::make_shared<Ackley>(dim)); // Non-scaled version of the problem

        // Best solution in the original domain
        std::cout << fromUnitBox(res.xBest(), origData->lBounds(), origData->uBounds()).t() << std::endl;

        // Check that we made enough progress and that we are feasible
        if (res.fBest() > -15.0) { // LCOV_EXCL_LINE
            return (EXIT_FAILURE); // LCOV_EXCL_LINE
        }
        if (!arma::all(res.xBest() >= data->lBounds())) { // LCOV_EXCL_LINE
            return (EXIT_FAILURE); // LCOV_EXCL_LINE
        }
        if (!arma::all(res.xBest() <= data->uBounds())) { // LCOV_EXCL_LINE
            return (EXIT_FAILURE); // LCOV_EXCL_LINE
        }
    }

    return EXIT_SUCCESS;
}

int main(int argc, char** argv) {
    return test_dycors();
}

