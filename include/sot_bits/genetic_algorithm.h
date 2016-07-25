//
//  genetic_algorithm.h
//  Surrogate Optimization
//
//  Created by David Eriksson on 8/3/15.
//  Copyright (c) 2015 David Eriksson. All rights reserved.
//

#ifndef Surrogate_Optimization_genetic_algorithm_h
#define Surrogate_Optimization_genetic_algorithm_h

#include "common.h"
#include "utils.h"

namespace sot {
    
    class GeneticAlgorithm  {
    protected:
        std::shared_ptr<Problem> mData;
        std::shared_ptr<ExpDesign> mExpDes;
        
        double mSigma = 0.2;
        int mTournamentSize = 5;
        double mpCross = 0.9;
        double mpMutation;
        int mDim;
        vec mxLow;
        vec mxUp;
        int mNumVariables;
        int mNumIndividuals;
        int mNumGenerations;
        std::string mName = "Genetic Algorithm";
        bool mRandomInit;
    
    public:   
        GeneticAlgorithm(Problem *data, int numIndividuals, int numGenerations) {
            mData = std::shared_ptr<Problem>(data);
            mDim = data->dim();
            mNumVariables = mDim;
            mpMutation = 1.0/mDim;
            mNumIndividuals = numIndividuals;
            mNumGenerations = numGenerations;
            mRandomInit = true;
            mxLow = data->lBounds();
            mxUp= data->uBounds();
        }
        GeneticAlgorithm(std::shared_ptr<Problem>& data, int numIndividuals, int numGenerations) {
            mData = std::shared_ptr<Problem>(data);
            mDim = data->dim();
            mNumVariables = mDim;
            mpMutation = 1.0/mDim;
            mNumIndividuals = numIndividuals;
            mNumGenerations = numGenerations;
            mRandomInit = true;
            mxLow = data->lBounds();
            mxUp= data->uBounds();
        }
        GeneticAlgorithm(std::shared_ptr<Problem>& data, std::shared_ptr<ExpDesign>& expDes, 
        int numIndividuals, int numGenerations) : GeneticAlgorithm(data, numIndividuals, numGenerations) {
            mExpDes = std::shared_ptr<ExpDesign>(expDes);
            mRandomInit = false;
        }
        GeneticAlgorithm(Problem *data, ExpDesign *expDes,  int numIndividuals, int numGenerations) : 
        GeneticAlgorithm(data, numIndividuals, numGenerations) {
            mExpDes = std::shared_ptr<ExpDesign>(expDes);
            mRandomInit = false;
        }

        Result run() {
            int maxEvals = mNumIndividuals * mNumGenerations;
            Result res(mNumIndividuals * mNumGenerations, mDim);
            
            mat population;
            mat newPopulation = arma::zeros<mat>(mNumVariables, mNumIndividuals);
            if (mRandomInit) {
                population.randu(mNumVariables, mNumIndividuals);
            }
            else{
                population = mExpDes->generatePoints();
            }
            population = fromUnitBox(population, mxLow, mxUp);

            //  Evaluate all individuals
            vec functionValues = mData->evals(population);
            
            //Save the best individual
            arma::uword ind;
            functionValues.min(ind);
            vec bestIndividual = population.col(ind);
            double bestValue = functionValues(ind);

            for(int gen = 0; gen < mNumGenerations - 1; gen++) {
                
                ////////////////// Tournament selection and crossover ////////////////////
                arma::imat tournament = arma::randi<arma::imat>(mTournamentSize, mNumIndividuals, arma::distr_param(0, mNumIndividuals - 1));
                for(int i = 0; i < mNumIndividuals/2; i++) {
                    double minval1 = std::numeric_limits<double>::max();
                    double minval2 = std::numeric_limits<double>::max();
                    int ind1, ind2;
                    for(int j=0; j < mTournamentSize; j++) {
                        if (functionValues(tournament(j, 2*i)) < minval1) {
                            minval1 = functionValues(tournament(j, 2*i));
                            ind1 = tournament(j, 2*i);
                        }
                        if (functionValues(tournament(j, 2*i + 1)) < minval2) {
                            minval2 = functionValues(tournament(j, 2*i + 1));
                            ind2 = tournament(j, 2*i + 1);
                        }
                    }
                    
                    double alpha = rand();
                    if( rand() < mpCross) {
                        newPopulation.col(2*i) = alpha * population.col(ind1) + (1 - alpha) * population.col(ind2);
                        newPopulation.col(2*i + 1) = alpha * population.col(ind2) + (1 - alpha) * population.col(ind1);
                    }
                    else {
                        newPopulation.col(2*i) = population.col(ind1);
                        newPopulation.col(2*i + 1) = population.col(ind2);
                    }
                }

                // Mutation
                for(int i=0; i < mNumIndividuals; i++) {
                    for(int j=0; j<mNumVariables; j++) {
                        if(rand() < mpMutation) {
                            newPopulation(j, i) += (mxUp(j) - mxLow(j)) * mSigma * randn();
                            if(newPopulation(j,i) > mxUp(j)) {
                                newPopulation(j,i) = fmax(2*mxUp(j) - newPopulation(j,i), mxLow(j));
                            }
                            else if(newPopulation(j,i) < mxLow(j)) {
                                newPopulation(j,i) = fmin(2*mxLow(j) - newPopulation(j,i), mxUp(j));
                            }
                        }
                    }
                }
                
                // Elitism
                newPopulation.col(mNumIndividuals - 1) = bestIndividual;
                
                //  Evaluate all individuals
                functionValues = mData->evals(newPopulation);
                
                // Save the results
                for(int i=0; i < mNumIndividuals; i++) {
                    vec x = newPopulation.col(i);
                    res.addEval(x, functionValues(i));
                }

                //Save the best individual
                arma::uword ind;
                functionValues.min(ind);
                bestIndividual = newPopulation.col(ind);
                bestValue = functionValues(ind);
                
                // Kill the old population
                population = newPopulation;
            }
            
            return res;
        }
    };
}

#endif