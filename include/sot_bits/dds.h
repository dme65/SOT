//
//  dds.h
//  Surrogate Optimization
//
//  Created by David Eriksson on 8/22/15.
//  Copyright (c) 2015 David Eriksson. All rights reserved.
//

#ifndef Surrogate_Optimization_optimizer_dds_h
#define Surrogate_Optimization_optimizer_dds_h

#include <cassert>
#include <iostream>
#include "common.h"
#include "utils.h"

namespace sot {

    class DDS {
    protected:
        std::shared_ptr<Problem> data;
        std::shared_ptr<ExpDesign> exp_des;
        int maxeval;
        int numeval;
        int initp;
        int dim;
        vec xlow;
        vec xup;
        std::string my_name;
    public:
        DDS(std::shared_ptr<Problem>& data, std::shared_ptr<ExpDesign>& exp_des, int maxeval) {
            this->data = std::shared_ptr<Problem>(data);
            this->exp_des = std::shared_ptr<ExpDesign>(exp_des);
            this->exp_des = exp_des;
            this->maxeval = maxeval;
            this->numeval = 0;
            this->initp = exp_des->npts();
            this->dim = data->dim();
            my_name = "DDS";
            assert(maxeval > initp);
        }
        
        Result run() {
            arma::arma_rng::set_seed_random();
            Result res(maxeval, initp, dim);
            numeval = 0;
            
            double sigma = 0.2*(xup(0) - xlow(0));
            mat init_des = FromUnitBox(exp_des->generate_points(), xlow, xup);
            
            for(int i=0; i < initp ; i++) {
                res.x.col(i) = init_des.col(i);
                res.fx(i) = data->eval(res.x.col(i));
                if(res.fx(i) < res.fbest) {
                    res.fbest = res.fx(i);
                    res.xbest = res.x.col(i);
                }
                numeval++;
            }
            
            while (numeval < maxeval) {
                
                ////////////////////////// Select a new evaluations //////////////////////////
                double dds_prob = 1 - log(numeval - initp)/log(maxeval - initp);
                dds_prob = fmax(dds_prob, 1.0/dim);
                
                vec cand = res.xbest;
                int count = 0;
                for(int j=0; j < dim; j++) {
                    if(rand() < dds_prob) {
                        count++;
                        cand(j) += sigma * randn();
                        if(cand(j) > xup(j)) { 
                            cand(j) = fmax(2*xup(j) - cand(j), xlow(j)); 
                        }
                        else if(cand(j) < xlow(j)) { 
                            cand(j) = fmin(2*xlow(j) - cand(j), xup(j)); 
                        }
                    }
                }
                // If no index was perturbed we force one
                if(count == 0) {
                    int ind = randi(dim);
                    cand(ind) += sigma * randn();
                    if(cand(ind) > xup(ind)) { 
                        cand(ind) = fmax(2*xup(ind) - cand(ind), xlow(ind)); 
                    }
                    else if(cand(ind) < xlow(ind)) { 
                        cand(ind) = fmin(2*xlow(ind) - cand(ind), xup(ind)); 
                    }
                }
                
                /////////////////////// Evaluate ///////////////////////
                double fval = data->eval(cand);
                
                res.x.col(numeval) = cand;
                res.fx(numeval) = fval;
                
                if (fval < res.fbest) {
                    res.xbest = cand;
                    res.fbest = fval;
                }
                
                numeval++;
            }
                                
            return res;
        }
    };
}

#endif
