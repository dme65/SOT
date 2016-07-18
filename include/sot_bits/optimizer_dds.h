//
//  dds.h
//  Surrogate Optimization
//
//  Created by David Eriksson on 8/22/15.
//  Copyright (c) 2015 David Eriksson. All rights reserved.
//

#ifndef Surrogate_Optimization_optimizer_dds_h
#define Surrogate_Optimization_optimizer_dds_h

#include "rbf.h"
#include <iostream>
#include <assert.h>
#include "utils.h"
#include "candidate_points.h"
#include <cassert>
#include <thread>
#include <atomic>
#include "genetic_algorithm.h"
#include "common.h"
#include "optimizer.h"

namespace sot {

    class OptimizerDDS : public Optimizer {
    public:
        OptimizerDDS(Problem *data, ExpDesign *exp_des, int maxeval) {
            this->data = data;
            this->exp_des = exp_des;
            this->maxeval = maxeval;
            this->numeval = 0;
            this->initp = exp_des->num_points;
            this->dim = data->dim();
            my_name = "DDS";
            assert(maxeval > initp);
        }
        
        Result run() {
            Result res(maxeval, exp_des->num_points, data->dim());
            numeval = 0;
            
            double sigma = 0.2;
            
            std::random_device rd;
            std::default_random_engine e1(rd());
            std::uniform_real_distribution<double> rand(0, 1);
            std::uniform_int_distribution<int> randi(0, dim-1);
            std::normal_distribution<double> randn(0.0, sigma*(data->lbound()(0) - data->rbound()(0)));
            
            mat init_des = FromUnitBox(exp_des->generate_points(), data->lbound(), data->rbound());
            
            for(int i=0; i < exp_des->num_points ; i++) {
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
                double dds_prob = 1 - log(numeval - exp_des->num_points)/log(maxeval - exp_des->num_points);
                dds_prob = fmax(dds_prob, 1.0/dim);
                
                vec cand = res.xbest;
                int count = 0;
                for(int j=0; j < dim; j++) {
                    if(rand(e1) < dds_prob) {
                        count++;
                        cand(j) += randn(e1);
                        if(cand(j) > data->rbound()(j)) { cand(j) = fmax(2*data->rbound()(j) - cand(j), data->lbound()(j)); }
                        else if(cand(j) < data->lbound()(j)) { cand(j) = fmin(2*data->lbound()(j) - cand(j), data->rbound()(j)); }
                    }
                }
                // If no index was perturbed we force one
                if(count == 0) {
                    int ind = randi(e1);
                    cand(ind) += randn(e1);
                    if(cand(ind) > data->rbound()(ind)) { cand(ind) = fmax(2*data->rbound()(ind) - cand(ind), data->lbound()(ind)); }
                    else if(cand(ind) < data->lbound()(ind)) { cand(ind) = fmin(2*data->lbound()(ind) - cand(ind), data->rbound()(ind)); }
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
