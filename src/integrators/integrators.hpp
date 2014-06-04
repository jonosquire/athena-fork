#ifndef INTEGRATORS_HPP
#define INTEGRATORS_HPP
//======================================================================================
/* Athena++ astrophysical MHD code
 * Copyright (C) 2014 James M. Stone  <jmstone@princeton.edu>
 * See LICENSE file for full public license information.
 *====================================================================================*/
/*! \file integrators.hpp
 *  \brief defines FluidIntegrator implements data and functions to integrate fluid
 *====================================================================================*/

// Athena headers
#include "../athena.hpp"         // Real
#include "../athena_arrays.hpp"  // AthenaArray

class Block;
class Fluid;

//! \class FluidIntegrator
//  \brief member functions implement various integration algorithms for the fluid

class FluidIntegrator {
public:
  FluidIntegrator(Fluid *pf);
  ~FluidIntegrator();

  Fluid *pparent_fluid;          // ptr to parent Fluid

  void Predict(Block *pb);
  void Correct(Block *pb);
  void RiemannSolver(
    const int il, const int iu, const int ivx, const int ivy, const int ivz,
    AthenaArray<Real> &wl, AthenaArray<Real> &wr, AthenaArray<Real> &flx);
  void ReconstructionFunc(const int k, const int j, const int il, const int iu, 
    const int dir, AthenaArray<Real> &w, AthenaArray<Real> &wl, AthenaArray<Real> &wr);

private:
  AthenaArray<Real> wl_,wr_,flx_; // 1D scratch vectors (L/R states, flux)
};
#endif