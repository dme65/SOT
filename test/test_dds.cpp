/* 
 * File:   test_dds.cpp
 * Author: David Eriksson
 *
 * Created on July 19, 2016, 12:44 PM
 */

#include <sot.h>
using namespace sot;

int test_dycors() {
    int dim = 10;
    int maxEvals = 500;
    
    std::shared_ptr<Problem> data(std::make_shared<Ackley>(dim));
    std::shared_ptr<ExpDesign> slhd(std::make_shared<SLHD>(2*(dim+1), dim));

    setSeedRandom();
    DDS opt(data, slhd, maxEvals);
    Result res = opt.run();    
    
    std::cout << res.fBest() << std::endl;
    // Check that we made enough progress and that we are feasible
    if (res.fBest() > -18.0) {
        return (EXIT_FAILURE);
    }
    if (not arma::all(res.xBest() >= data->lBounds())) {
        return (EXIT_FAILURE);
    }    
    if (not arma::all(res.xBest() <= data->uBounds())) {
        return (EXIT_FAILURE);
    }
    
    return EXIT_SUCCESS;
}

int main(int argc, char** argv) {
    return test_dycors();
}

