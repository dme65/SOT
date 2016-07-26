/*!
 * File:   sot.h
 * Author: David Eriksson, dme65@cornell.edu
 *
 * Created on 7/18/16.
 */


#ifndef SOT_SOT_H
#define SOT_SOT_H

#include "sot_bits/surrogate.h"
#include "sot_bits/merit_functions.h"
#include "sot_bits/adaptive_sampling.h"
#include "sot_bits/common.h"
#include "sot_bits/experimental_design.h"
#include "sot_bits/genetic_algorithm.h"
#include "sot_bits/kNN.h"
#include "sot_bits/dds.h"
#include "sot_bits/optimizer.h"
#include "sot_bits/rbf.h"
#include "sot_bits/shepard.h"
#include "sot_bits/test_problems.h"
#include "sot_bits/utils.h"

// Useful typedefs
namespace sot {    
    typedef RBFInterpolant<CubicKernel,LinearTail> CubicRBF; /**< Cubic RBF */
    typedef RBFInterpolant<ThinPlateKernel,LinearTail> TpsRBF; /**< Thin-plate spline RBF */
    typedef SymmetricLatinHypercube SLHD; /**< Symmetric Latin hypercube design */
    typedef LatinHypercube LHD; /**< Latin hypercube design */
}

#endif

