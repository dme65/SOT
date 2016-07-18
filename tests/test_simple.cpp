/*
 * Simple test function. CMAKE to be added soon.
 */

#include <sot.h>
#include <armadillo>

#include "include/sot_bits/utils.h"

int main() {
    int dim = 30;
    int maxeval = 500;
    
    sot::Problem *data = new sot::Ackley(dim);
    sot::ExpDesign *slhd = new sot::SymmetricLatinHypercube(2*(dim+1), dim);
    sot::OptimizerDYCORS opt(data, slhd, maxeval);
    
    sot::StopWatch watch;
    watch.start();
    sot::Result res = opt.run();
    double time = watch.stop();
    
    std::cout << "Best value found: " << res.fbest << std::endl;
    std::cout << "Best solution found[: [" << res.xbest.t() << " ]";
    std::cout << "Time Elapsed: " << time << " seconds\n";
}