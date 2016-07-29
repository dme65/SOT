/*!
 * File:   merit_functions.h
 * Author: David Eriksson, dme65@cornell.edu
 *
 * Created on 7/25/16.
 */


#ifndef SOT_MERIT_FUNCTIONS_H
#define SOT_MERIT_FUNCTIONS_H

#include "common.h"
#include "utils.h"

//!SOT namespace
namespace sot {

    //! Abstract class for a SOT merit function
    /*!
     * This is the abstract class that should be used as a Base class for all
     * merit functions in SOT. A merit function is used to select the most
     * promising points to evaluate from a set of candidate points.
     *
     * \author David Eriksson, dme65@cornell.edu
     */
    class MeritFunction {
    public:
        //! Method for picking the next point
        /*!
         * \param cand The candidate points to choose from
         * \param surf The surrogate model that predicts the value of the objective function
         * \param points Previous evaluation that we compute the minimum distance to
         * \param newPoints Number of candidate points that we are picking
         * \param distTol Candidate points that are closer than distTol to a
         * previously evaluated point are discarded
         * \return The proposed evaluations
         */
        virtual inline mat pickPoints(const mat &cand, const std::shared_ptr<Surrogate>& surf,
                const mat &points, int newPoints, double distTol) = 0;
    };

    //!  Merit function for choosing candidate points
    /*!
     * This is the weighted distance merit function that chooses the next
     * evaluation from a set of candidate point as the one that minimizes a
     * criterion based on the Surrogate model prediction and the minimum
     * distance to previously evaluated points.
     *
     * \author David Eriksson, dme65@cornell.edu
     */

    class MeritWeightedDistance : public MeritFunction {
    protected:
        vec mWeights = {0.3, 0.5, 0.8, 0.95}; /*!< Weights that are cycled */
        int mNextWeight = 0; /*!< Next weight (modulo length of mWeights) to be used */
    public:
        inline mat pickPoints(const mat &cand, const std::shared_ptr<Surrogate>& surf, const mat &points, int newPoints, double distTol) {
            int dim = cand.n_rows;

            // Compute the distances in single precision
            //mat dists = arma::sqrt(arma::conv_to<mat>::from(squaredPairwiseDistance<fmat>(
            //        arma::conv_to<fmat>::from(points), arma::conv_to<fmat>::from(cand))));
            mat dists = arma::sqrt(squaredPairwiseDistance<mat>(points, cand));
            // Evaluate the RBF at the candidate points
            vec surfVals = surf->evals(cand);
            vec valScores = unitRescale(surfVals);
            vec minDists = arma::min(dists).t();
            vec distScores = 1.0 - unitRescale(minDists);

            mat newx = arma::zeros<mat>(dim, newPoints);

            arma::uword winner;
            for(int i=0; i < newPoints; i++) {
                double weight = mWeights[mNextWeight % mWeights.n_elem];
                mNextWeight++;

                // Update distances if necessary
                if (i > 0) {
                    vec newDists = arma::sqrt(squaredPointSetDistance<mat,vec>((vec)newx.col(i-1), cand));
                    minDists = arma::min(minDists, newDists);
                    valScores(winner) = std::numeric_limits<double>::max();
                    distScores = 1.0 - unitRescale(minDists);
                }

                // Pick a winner
                vec merit = weight * valScores + (1.0 -  weight) * distScores;
                merit.elem(arma::find(minDists < distTol)).fill(std::numeric_limits<double>::max());
                double scores = merit.min(winner);
                newx.col(i) = cand.col(winner);
            }

            return newx;
        }
    };
}

#endif
