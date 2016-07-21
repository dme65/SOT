//
//  CandidatePoints.h
//  Surrogate Optimization
//
//  Created by David Eriksson on 7/30/15.
//  Copyright (c) 2015 David Eriksson. All rights reserved.
//

#ifndef __Surrogate_Optimization__CandidatePoints__
#define __Surrogate_Optimization__CandidatePoints__

#include "common.h"
#include "utils.h"
#include "problem.h"

namespace sot {
    
    class MeritWeightedDistance {
    private:
        vec weights = {0.3, 0.5, 0.8, 0.95};
        int next_weight = 0;
    public:
        inline mat pick_points(const mat &cand, const std::shared_ptr<Surrogate>& surf, const mat &points, int newpts, double dtol) {
            int dim = cand.n_rows; 

            // Evaluate the RBF at the candidate points
            const mat dists = arma::sqrt(SquaredPairwiseDistance<mat>(points, cand));
            vec surf_val = surf->evals(cand);
            vec val_score = UnitRescale(surf_val);
            vec min_dist = arma::min(dists).t();
            vec dist_score = 1.0 - UnitRescale(min_dist);

            mat newx = arma::zeros<mat>(dim, newpts);

            arma::uword winner;
            for(int i=0; i < newpts; i++) {
                double weight = weights[next_weight % weights.n_elem];
                next_weight++;

                // Update distances if necessary
                if (i > 0) {
                    vec new_dist = arma::sqrt(SquaredPointSetDistance<mat,vec>((vec)newx.col(i-1), cand));
                    min_dist = arma::min(min_dist, new_dist);
                    val_score(winner) = std::numeric_limits<double>::max();
                    dist_score = 1.0 - UnitRescale(min_dist);
                }

                // Pick a winner
                vec merit = weight * val_score + (1.0 -  weight) * dist_score;
                merit.elem(arma::find(min_dist < dtol)).fill(std::numeric_limits<double>::max());
                double score = merit.min(winner);
                newx.col(i) = cand.col(winner);
            }

            return newx;
        }
    };
    
    // Template for selecting the next evaluation
    class Sampling {
    protected:
        double dtol;
        int numeval;
        int budget;
    public:
        virtual void reset(int) = 0;
        virtual mat make_points(vec &xbest, const mat &points, double sigma, int newpts) = 0;
    };
    
    // DYCORS
    template<class MeritFunction = MeritWeightedDistance>
    class DYCORS : public Sampling {
    protected:
        std::shared_ptr<Problem> data;
        std::shared_ptr<Surrogate> surf;
        int ncand;
        MeritFunction merit;
    public:
        DYCORS(const std::shared_ptr<Problem>& data, const std::shared_ptr<Surrogate>& surf, int ncand, int budget) {
            this->data = std::shared_ptr<Problem>(data);
            this->surf = std::shared_ptr<Surrogate>(surf);
            this->budget = budget;
            this->ncand = ncand;
            this->dtol = 1e-3*sqrt(arma::sum(arma::square(data->rbound() - data->lbound())));
        }
        void reset(int budget) {
            this->budget = budget;
            int numeval = 0;
        }
        mat make_points(vec &xbest, const mat &points, double sigma, int newpts) {                
            double dds_prob = fmin(20.0/data->dim(), 1.0) * (1.0 - (log(this->numeval + 1.0) / log(this->budget)));
            mat cand = arma::repmat(xbest, 1, ncand);
            for(int i=0; i < ncand; i++) {

                int count = 0;
                for(int j=0; j < data->dim(); j++) {
                    if(rand() < dds_prob) {
                        count++;
                        cand(j, i) += sigma*randn();
                    }
                }
                // If no index was perturbed we force one
                if(count == 0) {
                    int ind = randi(data->dim());
                    cand(ind, i) += sigma*randn();
                }

                // Make sure we are still in the domain
                for(int j=0; j < data->dim(); j++) {
                    if(cand(j, i) > data->rbound()(j)) { cand(j, i) = fmax(2*data->rbound()(j) - cand(j, i), data->lbound()(j)); }
                    else if(cand(j, i) < data->lbound()(j)) { cand(j, i) = fmin(2*data->lbound()(j) - cand(j, i), data->rbound()(j)); }
                }
            }
            
            // Update counter
            numeval = numeval + newpts;
            
            return merit.pick_points(cand, surf, points, newpts, dtol);
        }
    };

    // SRBF
    template<class MeritFunction = MeritWeightedDistance>
    class SRBF : public Sampling {
    protected:
        std::shared_ptr<Problem> data;
        std::shared_ptr<Surrogate> surf;
        int ncand;
        MeritFunction merit;
    public:
        SRBF(const std::shared_ptr<Problem>& data, const std::shared_ptr<Surrogate>& surf, int ncand, int budget) {
            this->data = std::shared_ptr<Problem>(data);
            this->surf = std::shared_ptr<Surrogate>(surf);
            this->budget = budget;
            this->ncand = ncand;
            this->dtol = 1e-3*sqrt(arma::sum(arma::square(data->rbound() - data->lbound())));
        }
        void reset(int budget) {
            this->budget = budget;
            int numeval = 0;
        }
        mat make_points(vec &xbest, const mat &points, double sigma, int newpts) {

            mat cand = arma::repmat(xbest, 1, ncand);

            // Perturbs one randomly chosen coordinate
            for(int i=0; i < ncand; i++) {
                for(int j=0; j < data->dim(); j++) {
                    cand(j, i) += sigma * randn();
                    if(cand(j, i) > data->rbound()(j)) { cand(j, i) = fmax(2*data->rbound()(j) - cand(j, i), data->lbound()(j)); }
                    else if(cand(j, i) < data->lbound()(j)) { cand(j, i) = fmin(2*data->lbound()(j) - cand(j, i), data->rbound()(j)); }
                }
            }
            
            // Update counter
            numeval = numeval + newpts;
            
            return merit.pick_points(cand, surf, points, newpts, dtol);
        }
    };
    
    // Uniform
    template<class MeritFunction = MeritWeightedDistance>
    class Uniform : public Sampling {
    protected:
        std::shared_ptr<Problem> data;
        std::shared_ptr<Surrogate> surf;
        int ncand;
        MeritFunction merit;
    public:
        Uniform(const std::shared_ptr<Problem>& data, const std::shared_ptr<Surrogate>& surf, int ncand, int budget) {
            this->data = std::shared_ptr<Problem>(data);
            this->surf = std::shared_ptr<Surrogate>(surf);
            this->budget = budget;
            this->ncand = ncand;
            this->dtol = 1e-3*sqrt(arma::sum(arma::square(data->rbound() - data->lbound())));
        }
        void reset(int budget) {
            this->budget = budget;
            int numeval = 0;
        }
        mat make_points(vec &xbest, const mat &points, double sigma, int newpts) {
         
            mat cand = arma::randu<mat>(data->dim(), ncand);
            for(int j=0; j < data->dim(); j++) {
                cand.row(j) = data->lbound()(j) + (data->rbound()(j) - data->lbound()(j)) * cand.row(j);
            }
        
            // Update counter
            numeval = numeval + newpts;
            
            return merit.pick_points(cand, surf, points, newpts, dtol);
        }
    };
}

#endif /* defined(__Surrogate_Optimization__CandidatePoints__) */
