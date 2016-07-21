
//
//  optimizer_dycors.h
//  Surrogate Optimization
//
//  Created by David Eriksson on 8/12/15.
//  Copyright (c) 2015 David Eriksson. All rights reserved.
//

#ifndef Surrogate_Optimization_optimizer_h
#define Surrogate_Optimization_optimizer_h

#include <cassert>
#include <iostream>
#include "common.h"
#include "utils.h"

namespace sot {  
    class Optimizer {
    protected:
        std::shared_ptr<Problem> data;
        std::shared_ptr<ExpDesign> exp_des;
        std::shared_ptr<Surrogate> surf;
        std::shared_ptr<Sampling> sampling;
        double sigma_max = 0.2, sigma_min = 0.005;
        int failtol, succtol;
        int maxeval;
        int numeval;
        int initp;
        int dim;
        vec xlow;
        vec xup;
        int numthreads = 1;
        std::string my_name;
    public:
        Optimizer(std::shared_ptr<Problem>& data, std::shared_ptr<ExpDesign>& exp_des, 
                std::shared_ptr<Surrogate>& surf, std::shared_ptr<Sampling>& sampling, 
                int maxeval) {
            this->data = std::shared_ptr<Problem>(data);
            this->exp_des = std::shared_ptr<ExpDesign>(exp_des);
            this->surf = std::shared_ptr<Surrogate>(surf);
            this->sampling = std::shared_ptr<Sampling>(sampling);
            this->maxeval = maxeval;
            numeval = 0;
            initp = exp_des->npts();
            dim = data->dim();
            xlow = data->lbound();
            xup = data->rbound();
            failtol = data->dim();
            succtol = 3;
            my_name = "DYCORS";
            
            assert(maxeval > initp);
        }
        
        Optimizer(std::shared_ptr<Problem>& data, std::shared_ptr<ExpDesign>& exp_des, 
                std::shared_ptr<Surrogate>& surf, std::shared_ptr<Sampling>& sampling, 
                int maxeval, int numthreads) {
            this->numthreads = numthreads;
            Optimizer(data, exp_des, surf, sampling, maxeval);
        }
        
        Result run() {   
            arma::arma_rng::set_seed_random();
            Result res(maxeval, initp, dim);
            numeval = 0;
            
            double fbest_tot = std::numeric_limits<double>::max();
            vec xbest_tot;
            vec weights = {0.3, 0.5, 0.8, 0.95};
            vec weight = arma::zeros<vec>(1, 1);
            
            double dtol = 1e-3*sqrt(arma::sum(arma::square(xup - xlow)));
            
        start:
            double sigma = sigma_max;
            int fail = 0;
            int succ = 0;
            
            mat init_des = FromUnitBox(exp_des->generate_points(), xlow, xup);
            
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
                surf->add_points(res.x.cols(istart, iend), res.fx.rows(istart, iend));
            }
            ////////////////////////////// The fun starts now! //////////////////////////////////////
            while (numeval < maxeval) {
                
                // Fit the RBF
                surf->fit();
                
                // Find new points to evaluate
                weight[0] = weights[numeval % weights.n_elem];
                mat X = res.x.cols(istart, numeval - 1);
                vec newx = sampling->make_points(res.xbest, X, sigma*(xup(0) - xlow(0)), 1);
                                
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
                    int budget = maxeval - numeval - 1;
                    // Restart if sigma is too small and the budget 
                    // is larger than the initial design
                    if (sigma < sigma_min and budget > initp) {
                        if (res.fbest < fbest_tot) {
                            xbest_tot = res.xbest;
                            fbest_tot = res.fbest;
                        }
                        res.fbest = std::numeric_limits<double>::max();
                        surf->reset();
                        sampling->reset(maxeval - numeval - initp);
                        goto start;
                    }
                }
                if(succ == succtol) {
                    fail = 0;
                    succ = 0;
                    sigma = fmin(sigma * 2.0, sigma_max);
                }
                                
                // Add to surface
                surf->add_point(newx, res.fx(numeval));
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
