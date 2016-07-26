/*!
 * File:   genetic_algorithm.h
 * Author: David Eriksson, dme65@cornell.edu
 *
 * Created on 7/18/16.
 */

#ifndef SOT_GENETIC_ALGORITHM_H
#define SOT_GENETIC_ALGORITHM_H

#include "common.h"
#include "utils.h"

//!SOT namespace
namespace sot {
    
    //!  Genetic Algorithm
    /*!
     * This is an implementation of the popular Genetic Algorithm. The 
     * implementation is of a real-valued GA and the mutation operator used
     * is a normally distributed perturbation. The number of evaluations carried
     * out by the method is mNumIndividuals * mNumGenerations.
     * 
     * \author David Eriksson, dme65@cornell.edu
     */
    
    class GeneticAlgorithm  {
    protected:
        std::shared_ptr<Problem> mData; /*!< A shared pointer to the optimization problem */
        std::shared_ptr<ExpDesign> mExpDes; /*!< A shared pointer to the experimental design (if used) */
        double mSigma = 0.2; /*!< Standard deviation for the mutatio w.r.t. the unit box */
        int mTournamentSize = 5; /*!< Tournament size */
        double mpCross = 0.9; /*!< Crossover probability */
        double mpMutation; /*!< Mutation probability */
        int mDim; /*!< Number of dimensions (extracted from mData) */
        vec mxLow; /*!< Lower variable bounds (extracted from mData) */
        vec mxUp; /*!< Upper variable bounds (extracted from mData) */
        int mNumVariables; /*!< Number of variables (extracted from mData) */
        int mNumIndividuals; /*!< Number of individuals in the population */
        int mNumGenerations; /*!< Number of generations */
        std::string mName = "Genetic Algorithm"; /*!< Strategy name */
        bool mRandomInit; /*!< True if the initial population is uniformly random */
    public: 
        //! Constructor
        /*!
         * \param data A shared pointer to the optimization problem
         * \param numIndividuals Number of individuals in the population
         * \param numGenerations Number of generations
         */
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
        //! Constructor
        /*!
         * \param data A shared pointer to the optimization problem
         * \param expDes Experimental design used to generate the initial population
         * \param numIndividuals Number of individuals in the population
         * \param numGenerations Number of generations
         * 
         * \throws std::logic_error if the size of the experimental design doesn't match
         */
        GeneticAlgorithm(std::shared_ptr<Problem>& data, std::shared_ptr<ExpDesign>& expDes, 
        int numIndividuals, int numGenerations) : GeneticAlgorithm(data, numIndividuals, numGenerations) {
            mExpDes = std::shared_ptr<ExpDesign>(expDes);
            if(expDes->numPoints() != mNumIndividuals) {
                throw std::logic_error("Experimental design doesn't match the population size");
            }
            if(expDes->dim() != mNumVariables) {
                throw std::logic_error("Experimental design has incorrect dimensionality");
            } 
            mRandomInit = false;
        }
        //! Runs the optimization algorithm
        /*!
         * \return A Result object with the results from the run
         */
        Result run() {
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
                arma::imat tournament = arma::randi<arma::imat>(mTournamentSize, mNumIndividuals, 
                        arma::distr_param(0, mNumIndividuals - 1));
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