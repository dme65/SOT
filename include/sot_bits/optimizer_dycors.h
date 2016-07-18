
//
//  optimizer_dycors.h
//  Surrogate Optimization
//
//  Created by David Eriksson on 8/12/15.
//  Copyright (c) 2015 David Eriksson. All rights reserved.
//

#ifndef Surrogate_Optimization_optimizer_dycors_h
#define Surrogate_Optimization_optimizer_dycors_h

#include "rbf.h"
#include <iostream>
#include "utils.h"
#include "candidate_points.h"
#include <cassert>
#include "common.h"
#include "test_problems.h"

namespace sot {
    
    typedef RBFInterpolant<CubicKernel,LinearTail> Surface;
    
    class OptimizerDYCORS : public Optimizer {
    private:
        double sigma_max = 0.2, sigma_min = 0.005;
        int failtol, succtol;
        bool do_restart;
    public:
        OptimizerDYCORS(Problem *data, ExpDesign *exp_des, int maxeval) {
            this->data = data;
            this->exp_des = exp_des;
            this->maxeval = maxeval;
            this->numeval = 0;
            this->initp = exp_des->num_points;
            this->dim = data->dim();
            this->failtol = data->dim();
            this->succtol = 3;
            this->do_restart = true;
            my_name = "DYCORS";
            
            assert(maxeval > initp);
        }
        
        OptimizerDYCORS(Problem *data, ExpDesign *exp_des, int maxeval, bool do_restart) : OptimizerDYCORS(data, exp_des, maxeval) {
            this->do_restart = do_restart;
        }
        
        Result run() {
            Result res(maxeval, exp_des->num_points, data->dim());
            numeval = 0;
            
            double fbest_tot = std::numeric_limits<double>::max();
            vec xbest_tot;
            vec weights = {0.3, 0.5, 0.8, 0.95};
            vec weight = arma::zeros<vec>(1, 1);
            
            double dtol = 1e-3*sqrt(arma::sum(arma::square(data->rbound() - data->lbound())));
            
        start:
            double sigma = sigma_max;
            int fail = 0;
            int succ = 0;
            
            mat init_des = FromUnitBox(exp_des->generate_points(), data->lbound(), data->rbound());
            
            ////////////////////////////// Build an RBF //////////////////////////////
            CubicKernel kernel;
            LinearTail tail;
            Surface rbf(kernel, tail, maxeval, dim, data->lbound(), data->rbound());
            
            ////////////////////////////// Evaluate the initial design //////////////////////////////
            int istart = numeval;
            int iend = fmin(numeval + initp - 1, maxeval - 1);
            for(int i=numeval; i <= iend ; i++) {
                res.x.col(i) = init_des.col(i-istart);
                res.fx(i) = data->eval(res.x.col(i));
                if(res.fx(i) < res.fbest) {
                    res.fbest = res.fx(i);
                    res.xbest = res.x.col(i);
                }
                numeval++;
            }
            
            ////////////////////////////// Add points to the rbf //////////////////////////////
            if(istart < iend) {
                rbf.set_points(res.x.cols(istart, iend), res.fx.rows(istart, iend));
            }
            ////////////////////////////// The fun starts now! //////////////////////////////////////
            while (numeval < maxeval) {
                
                // Fit the RBF
                rbf.fit();
                
                // Find new points to evaluate
                weight[0] = weights[numeval % weights.n_elem];
                vec newx = DYCORS<Surface>(rbf, rbf.get_centers(), data->lbound(), data->rbound(), 
                                                                           res.xbest, sigma*(data->rbound()(0) - data->lbound()(0)),
                                                                           maxeval, numeval, initp, dim, dtol, weight, fmin(100*dim, 5000), 1);
                // Evaluate
                res.x.col(numeval) = newx;
                res.fx(numeval) = data->eval(newx);
                
                // Process evaluation
                if(res.fx(numeval) < res.fbest) {
                    if(res.fx(numeval) < res.fbest - 1e-3 * fabs(res.fbest)) {
                        fail = 0;
                        succ++;
                    }
                    else {
                        fail++;
                        succ = 0;
                    }
                    res.fbest = res.fx(numeval);
                    res.xbest = res.x.col(numeval);
                }
                else {
                    fail++;
                    succ = 0;
                }
                
                // Update sigma if necessary
                if(fail == failtol) {
                    fail = 0;
                    succ = 0;
                    sigma /= 2.0;
                    if (sigma < sigma_min) {
                        if (res.fbest < fbest_tot) {
                            xbest_tot = res.xbest;
                            fbest_tot = res.fbest;
                        }
                        if (!do_restart) { // Stop if we aren't restarting
                            res.x = res.x.cols(0, numeval);
                            res.fx = res.fx.rows(0, numeval);
                            return res;
                        }
                        res.fbest = std::numeric_limits<double>::max();
                        goto start;
                    }
                }
                if(succ == succtol) {
                    fail = 0;
                    succ = 0;
                    sigma = fmin(sigma * 2.0, sigma_max);
                }
                                
                // Add to surface
                rbf.add_point(newx, res.fx(numeval));
                
                numeval++;
            }
            
            if (fbest_tot < res.fbest) {
                res.xbest = xbest_tot;
                res.fbest = fbest_tot;
            }
            
            return res;
        }
    };
}

#endif
