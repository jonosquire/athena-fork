//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file strat.cpp
//! \brief Problem generator for stratified 3D shearing sheet.
//!
//! PURPOSE:  Problem generator for stratified 3D shearing sheet.  Based on the
//!   initial conditions described in "Three-dimensional Magnetohydrodynamic
//!   Simulations of Vertically Stratified Accretion Disks" by Stone, Hawley,
//!   Gammie & Balbus.
//!
//! Several different field configurations and perturbations are possible:
//! - ifield = 1 - Bz=B0 sin(x1) field with zero-net-flux [default]
//! - ifield = 2 - uniform Bz
//! - ifield = 3 - uniform Bz plus sinusoidal perturbation Bz(1+0.5*sin(kx*x1))
//! - ifield = 4 - B=(0,B0cos(kx*x1),B0sin(kx*x1))= zero-net flux w helicity
//! - ifield = 5 - uniform By, but only for |z|<2
//! - ifield = 6 - By with constant beta versus z
//! - ifield = 7 - zero field everywhere
//!
//! - ipert = 1 - random perturbations to P and V [default, used by HGB]
//!
//! Code must be configured using -shear
//!
//! REFERENCE:
//! - Stone, J., Hawley, J., Gammie, C.F. & Balbus, S. A., ApJ 463, 656-673 (1996)
//! - Hawley, J. F. & Balbus, S. A., ApJ 400, 595-609 (1992)
//============================================================================

// C headers

// C++ headers
#include <algorithm>
#include <cmath>      // sqrt()
#include <iostream>
#include <limits>
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../fft/athena_fft.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../orbital_advection/orbital_advection.hpp"
#include "../parameter_input.hpp"
#include "../utils/utils.hpp"     // ran2()

#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

// TODO(felker): many unused arguments in these functions: time, iout, ...
void VertGrav(MeshBlock *pmb, const Real time, const Real dt,
              const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
              const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
              AthenaArray<Real> &cons_scalar);
void StratOutflowInnerX3(MeshBlock *pmb, Coordinates *pco,
                         AthenaArray<Real> &a,
                         FaceField &b, Real time, Real dt,
                         int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void StratOutflowOuterX3(MeshBlock *pmb, Coordinates *pco,
                         AthenaArray<Real> &a,
                         FaceField &b, Real time, Real dt,
                         int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void StratSimon13InnerX3(MeshBlock *pmb, Coordinates *pco,
                       AthenaArray<Real> &prim, FaceField &b,
                       Real time, Real dt,
                       int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void StratSimon13OuterX3(MeshBlock *pmb, Coordinates *pco,
                       AthenaArray<Real> &prim, FaceField &b,
                       Real time, Real dt,
                       int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void StratLesurInnerX3(MeshBlock *pmb, Coordinates *pco,
                       AthenaArray<Real> &prim, FaceField &b,
                       Real time, Real dt,
                       int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void StratLesurOuterX3(MeshBlock *pmb, Coordinates *pco,
                       AthenaArray<Real> &prim, FaceField &b,
                       Real time, Real dt,
                       int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void StratPowerLawInnerX3(MeshBlock *pmb, Coordinates *pco,
                       AthenaArray<Real> &prim, FaceField &b,
                       Real time, Real dt,
                       int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void StratPowerLawOuterX3(MeshBlock *pmb, Coordinates *pco,
                       AthenaArray<Real> &prim, FaceField &b,
                       Real time, Real dt,
                       int il, int iu, int jl, int ju, int kl, int ku, int ngh);

namespace {
Real HistoryBxBy(MeshBlock *pmb, int iout);
Real HistorydVxVy(MeshBlock *pmb, int iout);

// Apply a density floor - useful for large |z| regions
Real dfloor, pfloor, sumrho;
Real Omega_0, qshear, H02, beta, central_den, betaz, betax, bc_pl_index;
bool replensish_density;
} // namespace

//====================================================================================
void Mesh::InitUserMeshData(ParameterInput *pin) {
  
  // Some of these would normally be in ProblemGenerator but need for BCs
  central_den = 1.0;
  // shearing sheet parameter
  qshear = pin->GetReal("orbital_advection","qshear");
  Omega_0 = pin->GetReal("orbital_advection","Omega0");
  if (Omega_0<0.01) Omega_0=1.; // For runs without rotation use Omega_0=0.001 and q=1
  
  Real iso_sound = pin->GetReal("hydro", "iso_sound_speed");
  if (NON_BAROTROPIC_EOS)
    iso_sound = std::sqrt( pin->GetOrAddReal("problem","pres",1.0) / central_den);
  
  H02 = 2.*SQR(iso_sound)/SQR(Omega_0);
  int ifield = pin->GetOrAddInteger("problem","ifield", 1);
  if (MAGNETIC_FIELDS_ENABLED) {
    beta = pin->GetReal("problem","beta");
    H02 *= (1. + 1/beta);
  }

  Real float_min = std::numeric_limits<float>::min();
  dfloor=pin->GetOrAddReal("hydro","dfloor",(1024*(float_min)));
  pfloor=pin->GetOrAddReal("hydro","pfloor",(1024*(float_min)));
  
  replensish_density = pin->GetOrAddBoolean("problem", "replensish_density", false);
  
  // power law index for density boundary. Magnetic field one is (bc_pl_index-2)/2 for a low-beta equilibrium
  bc_pl_index = pin->GetOrAddReal("problem", "dens_boundary_powerlaw", 5.);

  if (MAGNETIC_FIELDS_ENABLED) {
    AllocateUserHistoryOutput(2);
    EnrollUserHistoryOutput(0, HistoryBxBy, "-BxBy");
    EnrollUserHistoryOutput(1, HistorydVxVy, "dVxVy");
  }
    
  // Enroll user-defined physical source terms
  //   vertical external gravitational potential
  EnrollUserExplicitSourceFunction(VertGrav);

  // enroll user-defined boundary conditions
  if (mesh_bcs[BoundaryFace::inner_x3] == GetBoundaryFlag("user")) {
//    EnrollUserBoundaryFunction(BoundaryFace::inner_x3, StratOutflowInnerX3);
//    EnrollUserBoundaryFunction(BoundaryFace::inner_x3, StratLesurInnerX3);
//    if (Globals::my_rank==0) std::cout << "Warning: using Lesur boundary conditions\n";
//    EnrollUserBoundaryFunction(BoundaryFace::inner_x3, StratSimon13InnerX3);
//        if (Globals::my_rank==0) std::cout << "Warning: using Simon13 boundary conditions\n";
    EnrollUserBoundaryFunction(BoundaryFace::inner_x3, StratPowerLawInnerX3);
            if (Globals::my_rank==0) std::cout << "Warning: using PowerLaw boundary conditions with index -" << bc_pl_index <<"\n";
  }
  if (mesh_bcs[BoundaryFace::outer_x3] == GetBoundaryFlag("user")) {
//    EnrollUserBoundaryFunction(BoundaryFace::outer_x3, StratOutflowOuterX3);
//    EnrollUserBoundaryFunction(BoundaryFace::outer_x3, StratLesurOuterX3);
//    EnrollUserBoundaryFunction(BoundaryFace::outer_x3, StratSimon13OuterX3);
    EnrollUserBoundaryFunction(BoundaryFace::outer_x3, StratPowerLawOuterX3);
  }

//  if (!shear_periodic) {
//    std::stringstream msg;
//    msg << "### FATAL ERROR in hb3.cpp ProblemGenerator" << std::endl
//        << "This problem generator requires shearing box." << std::endl;
//    ATHENA_ERROR(msg);
//  }
  
  // turb_flag is initialzed in the Mesh constructor to 0 by default;
  // turb_flag = 1 for decaying turbulence
  // turb_flag = 2 for impulsively driven turbulence
  // turb_flag = 3 for continuously driven turbulence
  turb_flag = pin->GetOrAddInteger("problem","turb_flag",0);
  if (turb_flag != 0) {
    if (Globals::my_rank==0) std::cout << "Including turbulent forcing with parameters from the <turbulence> block\n";
#ifndef FFT
    std::stringstream msg;
    msg << "### FATAL ERROR in TurbulenceDriver::TurbulenceDriver" << std::endl
    << "non zero Turbulence flag is set without FFT!" << std::endl;
    ATHENA_ERROR(msg);
    return;
#endif
  }
  
  
  // Value of sumrho to use. Only needed in a restart because it doesn't run ProblemGenerator
  // have to take from previous output file (this is super hacky...)
  Real replensish_density_val = pin->GetOrAddReal("problem", "replensish_density_val", 0.0);
  if (replensish_density>0.) {
    sumrho = replensish_density_val;
    if (Globals::my_rank==0)
      std::cout << "Replenishing total mass at each time step. Initial sumrho = " << sumrho <<"\n";
  }
  return;
}



//======================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief stratified disk problem generator for 3D problems.
//======================================================================================
void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  int ifield, ipert;
  Real amp, pres;
  Real iso_cs=1.0;
  Real B0 = 0.0;
  Real B0z = 0.0; // just for ifield =6
  Real awdth, npow; // just for ifield = 8. This is width of Lorentzian function 1/(a^2+z^2)

  Real SumRd=0.0, SumRvx=0.0, SumRvy=0.0, SumRvz=0.0;
  // TODO(felker): tons of unused variables in this file: xmin, xmax, rbx, rby, Ly, ky,...
  Real x1, x3;
  //Real xmin, xmax;
  //Real x1f, x2f, x3f;
  Real rd(0.0), rp(0.0);
  Real rvx, rvy, rvz;
  //Real rbx, rby, rbz;
  Real rval;

  // Initialize boxsize
  Real Lx = pmy_mesh->mesh_size.x1max - pmy_mesh->mesh_size.x1min;
  //Real Ly = pmy_mesh->mesh_size.x2max - pmy_mesh->mesh_size.x2min;
  //Real Lz = pmy_mesh->mesh_size.x3max - pmy_mesh->mesh_size.x3min;

  // initialize wavenumbers
  int nwx = pin->GetOrAddInteger("problem","nwx",1);
  //int nwy = pin->GetOrAddInteger("problem","nwy",1);
  //int nwz = pin->GetOrAddInteger("problem","nwz",1);
  Real kx = (2.0*PI/Lx)*(static_cast<Real>(nwx));// nxw=-ve for leading wave
  //Real ky = (2.0*PI/Ly)*(static_cast<Real>(nwy));
  //Real kz = (2.0*PI/Lz)*(static_cast<Real>(nwz));

  // Ensure a different initial random seed for each meshblock.
  std::int64_t iseed = -1 - gid;

  // adiabatic gamma
  Real gam = peos->GetGamma();

  if (pmy_mesh->mesh_size.nx3 == 1) {
    std::stringstream msg;
    msg << "### FATAL ERROR in strat.cpp ProblemGenerator" << std::endl
        << "Stratified shearing sheet only works on a 3D grid" << std::endl;
    ATHENA_ERROR(msg);
  }

  // Read problem parameters for initial conditions
  amp = pin->GetReal("problem","amp");
  ipert = pin->GetOrAddInteger("problem","ipert", 1);


  
  ifield = pin->GetOrAddInteger("problem","ifield", 1);
  awdth = pin->GetOrAddReal("problem", "awidth", 1.0); // Just for ifield=8.
  npow = pin->GetOrAddReal("problem", "npow", 2.0); // Just for ifield=8. powerlaw index of density
  
  

  // Compute pressure based on the EOS.
  if (NON_BAROTROPIC_EOS) {
    pres  = pin->GetOrAddReal("problem","pres",1.0);
  } else {
    iso_cs = peos->GetIsoSoundSpeed();
    pres = central_den*SQR(iso_cs);
  }

  // Compute field strength based on beta.
  if (MAGNETIC_FIELDS_ENABLED) {
    B0 = std::sqrt(static_cast<Real>(2.0*pres/beta));
    if (Globals::my_rank==0) std::cout << "B0=" << B0 << std::endl;
    if (Globals::my_rank==0) std::cout << "H0^2=" << H02 << std::endl;
  }
  
  // vertical field for ifield==6
  betaz = pin->GetOrAddInteger("problem","betaz", 0.0);
  if (betaz > 0.0) B0z = std::sqrt(static_cast<Real>(2.0*pres/fabs(betaz)));
  if (betaz < 0.0) B0z = -1.0*std::sqrt(static_cast<Real>(2.0*pres/fabs(betaz)));
  if (Globals::my_rank==0) std::cout << "B0z=" << B0z << std::endl;
  
  
  
  // For ifield==8 and 9 set beta to give H02=1 for given sound speed (H02=1,
  // cs2 = 1/(1+1/beta), but otherwise beta is not used
  // radial field for ifield ==9
  betax = pin->GetOrAddInteger("problem","betax", 0.0);
  Real signbx = (betax>=0.0)? 1 : -1; // +1 to have the sign to amplitufy By
  betax = fabs(betax);
  if (Globals::my_rank==0 && ifield==9) std::cout << "Starting from radial field: betax=" <<betax << " with sign " << signbx  << std::endl;

  // With viscosity and/or resistivity, read eta_Ohm and nu_V
  // (to be filled in) ???
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        x1 = pcoord->x1v(i);
        //x2 = pcoord->x2v(j);
        x3 = pcoord->x3v(k);
        // x1f = pcoord->x1f(i);
        // x2f = pcoord->x2f(j);
        // x3f = pcoord->x3f(k);
        
        // Initialize perturbations
        // ipert = 1 - random perturbations to P/d and V
        // [default, used by HGB]
        if (ipert == 1 && ifield != 8) {
          rval = amp*(ran2(&iseed) - 0.5);
          rd = central_den*std::exp(-x3*x3 / H02)*(1.0+2.0*rval);
          if (rd < dfloor) rd = dfloor;
          SumRd += rd;
          if (NON_BAROTROPIC_EOS) {
            rp = pres/central_den*rd;
            if (rp < pfloor) rp = pfloor;
          }
          rval = amp*(ran2(&iseed) - 0.5);
          rvx = (0.4/std::sqrt(3.0)) *rval*1e-3;
          SumRvx += rd*rvx;
          
          rval = amp*(ran2(&iseed) - 0.5);
          rvy = (0.4/std::sqrt(3.0)) *rval*1e-3;
          SumRvy += rd*rvy;
          
          rval = amp*(ran2(&iseed) - 0.5);
          rvz = (0.4/std::sqrt(3.0)) *rval*1e-3;
          SumRvz += rd*rvz;
          // no perturbations
        } else if (ifield == 8) {
          rval = amp*(ran2(&iseed) - 0.5);
          rd = 2.*std::pow(awdth,npow-1)/PI/std::pow(SQR(awdth) + SQR(x3),npow/2.)*(1.0+2.0*rval); // Could generalize to other power laws
          if (rd < dfloor) rd = dfloor;
          SumRd += rd;
          
          rval = amp*(ran2(&iseed) - 0.5);
          rvx = (0.4/std::sqrt(3.0)) *rval*1e-3;
          SumRvx += rd*rvx;
          
          rval = amp*(ran2(&iseed) - 0.5);
          rvy = (0.4/std::sqrt(3.0)) *rval*1e-3;
          SumRvy += rd*rvy;
          
          rval = amp*(ran2(&iseed) - 0.5);
          rvz = (0.4/std::sqrt(3.0)) *rval*1e-3;
          SumRvz += rd*rvz;
        } else {
          rd = central_den*std::exp(-x3*x3 / H02);
          rvx = 0;
          rvy = 0;
          rvz = 0;
        }

        // Initialize d, M, and P.
        // for_the_future: if FARGO do not initialize the bg shear
        phydro->u(IDN,k,j,i) = rd;
        phydro->u(IM1,k,j,i) = rd*rvx;
        phydro->u(IM2,k,j,i) = rd*rvy;
        if(!porb->orbital_advection_defined)
          phydro->u(IM2,k,j,i) -= rd*qshear*Omega_0*x1;
        phydro->u(IM3,k,j,i) = rd*rvz;
        if (NON_BAROTROPIC_EOS) {
          phydro->u(IEN,k,j,i) = rp/(gam-1.0)
                                 + 0.5*(SQR(phydro->u(IM1,k,j,i))
                                        + SQR(phydro->u(IM2,k,j,i))
                                        + SQR(phydro->u(IM3,k,j,i)))/rd;
        } // Hydro

        // Initialize magnetic field.  For 3D shearing box B1=Bx, B2=By, B3=Bz
        //  ifield = 1 - Bz=B0 std::sin(x1) field with zero-net-flux [default]
        //  ifield = 2 - uniform Bz
        //  ifield = 3 - Bz(1+0.5*sin(kx*x1))
        //  ifield = 4 - B=(0,B0cos(kx*x1),B0sin(kx*x1)) =
        //               zero-net flux w/ helicity
        //  ifield = 5 - uniform By, but only for |z|<2
        //  ifield = 6 - By with constant beta versus z
        //  ifield = 7 - zero field everywhere
        //  ifield = 8 - low beta generalized Lorentzian
        //  ifield = 9 - Gaussian in Bx
        if (MAGNETIC_FIELDS_ENABLED) {
          if (ifield == 1) {
            pfield->b.x1f(k,j,i) = 0.0;
            pfield->b.x2f(k,j,i) = 0.0;
            pfield->b.x3f(k,j,i) = B0*(std::sin(static_cast<Real>(kx)*x1));
            if (i==ie) pfield->b.x1f(k,j,ie+1) = 0.0;
            if (j==je) pfield->b.x2f(k,je+1,i) = 0.0;
            if (k==ke) pfield->b.x3f(ke+1,j,i) = B0*(std::sin(static_cast<Real>(kx)*x1));
          }
          if (ifield == 2) {
            pfield->b.x1f(k,j,i) = 0.0;
            pfield->b.x2f(k,j,i) = 0.0;
            pfield->b.x3f(k,j,i) = B0;
            if (i==ie) pfield->b.x1f(k,j,ie+1) = 0.0;
            if (j==je) pfield->b.x2f(k,je+1,i) = 0.0;
            if (k==ke) pfield->b.x3f(ke+1,j,i) = B0;
          }
          if (ifield == 3) {
            pfield->b.x1f(k,j,i) = 0.0;
            pfield->b.x2f(k,j,i) = 0.0;
            pfield->b.x3f(k,j,i) = B0*(1.0+0.5*std::sin(static_cast<Real>(kx)*x1));
            if (i==ie) pfield->b.x1f(k,j,ie+1) = 0.0;
            if (j==je) pfield->b.x2f(k,je+1,i) = 0.0;
            if (k==ke) pfield->b.x3f(ke+1,j,i) = B0*(1.0 + 0.5*
                                                     std::sin(static_cast<Real>(kx)*x1));
          }
          if (ifield == 4) {
            pfield->b.x1f(k,j,i) = 0.0;
            pfield->b.x2f(k,j,i) = B0*(std::cos(static_cast<Real>(kx)*x1));
            pfield->b.x3f(k,j,i) = B0*(std::sin(static_cast<Real>(kx)*x1));
            if (i==ie) pfield->b.x1f(k,j,ie+1) = 0.0;
            if (j==je) pfield->b.x2f(k,je+1,i) = B0*(std::cos(static_cast<Real>(kx)*x1));
            if (k==ke) pfield->b.x3f(ke+1,j,i) = B0*(std::sin(static_cast<Real>(kx)*x1));
          }
          if (ifield == 5 ) {
            pfield->b.x1f(k,j,i) = 0.0;
            pfield->b.x2f(k,j,i) = B0*std::sqrt(std::exp(-x3*x3 / 25));
            pfield->b.x3f(k,j,i) = 0.0;
            if (i==ie) pfield->b.x1f(k,j,ie+1) = 0.0;
            if (j==je) pfield->b.x2f(k,je+1,i) = B0*std::sqrt(std::exp(-x3*x3 / 25));
            if (k==ke) pfield->b.x3f(ke+1,j,i) = 0.0;
          }
          if (ifield == 6) {
            // net toroidal field with constant \beta with height
            pfield->b.x1f(k,j,i) = 0.0;
            pfield->b.x2f(k,j,i) = std::sqrt(2.*central_den*std::exp(-x3*x3 / H02)*SQR(iso_cs)/beta);
            pfield->b.x3f(k,j,i) = B0z;
            if (i==ie) pfield->b.x1f(k,j,ie+1) = 0.0;
            if (j==je) pfield->b.x2f(k,je+1,i) = std::sqrt(2.*central_den*std::exp(-x3*x3 / H02)*
                                                           SQR(iso_cs)/beta);
            if (k==ke) pfield->b.x3f(ke+1,j,i) = B0z;
          }
          if (ifield == 7) {
            // zero field everywhere
            pfield->b.x1f(k,j,i) = 0.0;
            pfield->b.x2f(k,j,i) = 0.0;
            pfield->b.x3f(k,j,i) = 0.0;
            if (i==ie) pfield->b.x1f(k,j,ie+1) = 0.0;
            if (j==je) pfield->b.x2f(k,je+1,i) = 0.0;
            if (k==ke) pfield->b.x3f(ke+1,j,i) = 0.0;
          }
          if (ifield == 8) {
            // Lorenzian power-law like solution
            pfield->b.x1f(k,j,i) = 0.0;
            pfield->b.x2f(k,j,i) = std::sqrt(4.*std::pow(awdth,npow-1.)/(PI*(npow-2.)) * (-(npow-2.)*SQR(iso_cs) + SQR(Omega_0)*(SQR(awdth)+SQR(x3))) /
                                             std::pow(SQR(awdth)+SQR(x3), npow/2.) ); // See ParkerModeStability.nb
            pfield->b.x3f(k,j,i) = B0z;
            if (i==ie) pfield->b.x1f(k,j,ie+1) = 0.0;
            if (j==je) pfield->b.x2f(k,je+1,i) = std::sqrt(4.*std::pow(awdth,npow-1.)/(PI*(npow-2.)) * (-(npow-2.)*SQR(iso_cs) + SQR(Omega_0)*(SQR(awdth)+SQR(x3))) /
                                                           std::pow(SQR(awdth)+SQR(x3), npow/2.) );
            if (k==ke) pfield->b.x3f(ke+1,j,i) = B0z;
          }
          if (ifield == 9) {
            // net radial field with constant \beta with height
            pfield->b.x1f(k,j,i) = -signbx * std::sqrt(2.*central_den*std::exp(-x3*x3 / H02)*SQR(iso_cs)/betax);
            pfield->b.x2f(k,j,i) = std::sqrt(2.*central_den*std::exp(-x3*x3 / H02)*SQR(iso_cs)/beta);
            pfield->b.x3f(k,j,i) = B0z;
            if (i==ie) pfield->b.x1f(k,j,ie+1) = -signbx * std::sqrt(2.*central_den*std::exp(-x3*x3 / H02)*SQR(iso_cs)/betax);
            if (j==je) pfield->b.x2f(k,je+1,i) = std::sqrt(2.*central_den*std::exp(-x3*x3 / H02)*SQR(iso_cs)/beta);
            if (k==ke) pfield->b.x3f(ke+1,j,i) = B0z;
          }
        } // MHD
      }
    }
  }

  // For random perturbations as in HGB, ensure net momentum is zero by
  // subtracting off mean of perturbations

  if (ipert == 1) {
    if (lid == pmy_mesh->nblocal - 1) {
#ifdef MPI_PARALLEL
      MPI_Allreduce(MPI_IN_PLACE, &SumRd,  1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &SumRvx, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &SumRvy, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &SumRvz, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
      std::int64_t cell_num = pmy_mesh->GetTotalCells();
      SumRvx /= SumRd*cell_num;
      SumRvy /= SumRd*cell_num;
      SumRvz /= SumRd*cell_num;
      for (int b = 0; b < pmy_mesh->nblocal; ++b) {
        Hydro *ph = pmy_mesh->my_blocks(b)->phydro;
        for (int k=ks; k<=ke; k++) {
          for (int j=js; j<=je; j++) {
            for (int i=is; i<=ie; i++) {
              ph->u(IM1,k,j,i) -= ph->u(IDN,k,j,i)*SumRvx;
              ph->u(IM2,k,j,i) -= ph->u(IDN,k,j,i)*SumRvy;
              ph->u(IM3,k,j,i) -= ph->u(IDN,k,j,i)*SumRvz;
            }
          }
        }
      }
    }
  }
  
  // Compute total density to use in loop
  // (Seems like this shoudld have a sum over meshblocks... May not work with many blocks per proc)
  sumrho = 0.0;
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        sumrho += phydro->u(IDN,k,j,i);
  }}}
  if (lid == pmy_mesh->nblocal - 1) {
#ifdef MPI_PARALLEL
      MPI_Allreduce(MPI_IN_PLACE, &sumrho,  1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
  }
  std::int64_t cell_num = pmy_mesh->GetTotalCells();
  if (replensish_density && Globals::my_rank==0)
    std::cout << "Replenishing total mass at each time step. Initial average density = " << sumrho/cell_num <<"\n";

  
  return;
}


void MeshBlock::UserWorkBeforeOutput(ParameterInput *pin) {
  return;
}

void MeshBlock::UserWorkInLoop() {
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        Real& u_d  = phydro->u(IDN,k,j,i);
        u_d = (u_d > dfloor) ?  u_d : dfloor;
        if (NON_BAROTROPIC_EOS) {
          Real gam = peos->GetGamma();
          Real& w_p  = phydro->w(IPR,k,j,i);
          Real& u_e  = phydro->u(IEN,k,j,i);
          Real& u_m1 = phydro->u(IM1,k,j,i);
          Real& u_m2 = phydro->u(IM2,k,j,i);
          Real& u_m3 = phydro->u(IM3,k,j,i);
          w_p = (w_p > pfloor) ?  w_p : pfloor;
          Real di = 1.0/u_d;
          Real ke = 0.5*di*(SQR(u_m1) + SQR(u_m2) + SQR(u_m3));
          u_e = w_p/(gam-1.0)+ke;
        }
      }
    }
  }
  
  if (replensish_density) {
    Real curr_rho = 0.0;
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
        for (int i=is; i<=ie; i++) {
          curr_rho += phydro->u(IDN,k,j,i);
    }}}
    if (lid == pmy_mesh->nblocal - 1) {
#ifdef MPI_PARALLEL
      MPI_Allreduce(MPI_IN_PLACE, &curr_rho,  1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
    }
    for (int b = 0; b < pmy_mesh->nblocal; ++b) {
      Hydro *ph = pmy_mesh->my_blocks(b)->phydro;
      for (int k=ks; k<=ke; k++) {
        for (int j=js; j<=je; j++) {
          for (int i=is; i<=ie; i++) {
            ph->u(IDN,k,j,i) *= sumrho / curr_rho;
    }}}}
    
    if (Globals::my_rank==0)
      std::cout << "Mass was " << curr_rho <<", replenished to " << sumrho << "\n";
    
  }
  return;
}

void Mesh::UserWorkAfterLoop(ParameterInput *pin) {
  return;
}


void VertGrav(MeshBlock *pmb, const Real time, const Real dt,
              const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
              const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
              AthenaArray<Real> &cons_scalar) {
  Real fsmooth, xi, sign;
  Real Lz = pmb->pmy_mesh->mesh_size.x3max - pmb->pmy_mesh->mesh_size.x3min;
  Real z0 = Lz/2.0;
  Real lambda = 0.1 / z0;
  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    for (int j=pmb->js; j<=pmb->je; ++j) {
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        Real den = prim(IDN,k,j,i);
        Real x3 = pmb->pcoord->x3v(k);
        // smoothing function
        if (x3 >= 0) {
          sign = -1.0;
        } else {
          sign = 1.0;
        }
        xi = z0/x3;
        fsmooth = SQR( std::sqrt( SQR(xi+sign) + SQR(xi*lambda) ) + xi*sign );
        // multiply gravitational potential by smoothing function
        cons(IM3,k,j,i) -= dt*den*SQR(Omega_0)*x3*fsmooth;
        if (NON_BAROTROPIC_EOS) {
          cons(IEN,k,j,i) -= dt*den*SQR(Omega_0)*prim(IVZ,k,j,i)*x3*fsmooth;
        }
      }
    }
  }
  return;
}

//  Here is the lower z outflow boundary.
//  The basic idea is that the pressure and density
//  are exponentially extrapolated in the ghost zones
//  assuming a constant temperature there (i.e., an
//  isothermal atmosphere). The z velocity (NOT the
//  momentum) are set to zero in the ghost zones in the
//  case of the last lower physical zone having an inward
//  flow.  All other variables are extrapolated into the
//  ghost zones with zero slope.

void StratOutflowInnerX3(MeshBlock *pmb, Coordinates *pco,
                         AthenaArray<Real> &prim, FaceField &b,
                         Real time, Real dt,
                         int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  // Copy field components from last physical zone
  // zero slope boundary for B field
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=1; k<=ngh; k++) {
      for (int j=jl; j<=ju; j++) {
        for (int i=il; i<=iu+1; i++) {
          b.x1f(kl-k,j,i) = b.x1f(kl,j,i);
        }
      }
    }
    for (int k=1; k<=ngh; k++) {
      for (int j=jl; j<=ju+1; j++) {
        for (int i=il; i<=iu; i++) {
          b.x2f(kl-k,j,i) = b.x2f(kl,j,i);
        }
      }
    }
    for (int k=1; k<=ngh; k++) {
      for (int j=jl; j<=ju; j++) {
        for (int i=il; i<=iu; i++) {
          b.x3f(kl-k,j,i) = b.x3f(kl,j,i);
        }
      }
    }
  } // MHD

  for (int k=1; k<=ngh; k++) {
    for (int j=jl; j<=ju; j++) {
      for (int i=il; i<=iu; i++) {
        Real x3 = pco->x3v(kl-k);
        Real x3b = pco->x3v(kl);
        Real den = prim(IDN,kl,j,i);
        // First calculate the effective gas temperature (Tkl=cs^2)
        // in the last physical zone. If isothermal, use H=1
        Real Tkl = 0.5*SQR(Omega_0);
        if (NON_BAROTROPIC_EOS) {
          Real presskl = prim(IPR,kl,j,i);
          presskl = std::max(presskl,pfloor);
          Tkl = presskl/den;
        }
        // Now extrapolate the density to balance gravity
        // assuming a constant temperature in the ghost zones
        prim(IDN,kl-k,j,i) = den*std::exp(-(SQR(x3)-SQR(x3b))/H02/
                                          (2.0*Tkl/SQR(Omega_0)));
        // Copy the velocities, but not the momenta ---
        // important because of the density extrapolation above
        prim(IVX,kl-k,j,i) = prim(IVX,kl,j,i);
        prim(IVY,kl-k,j,i) = prim(IVY,kl,j,i);
        // If there's inflow into the grid, set the normal velocity to zero
        if (prim(IVZ,kl,j,i) >= 0.0) {
          prim(IVZ,kl-k,j,i) = 0.0;
        } else {
          prim(IVZ,kl-k,j,i) = prim(IVZ,kl,j,i);
        }
        if (NON_BAROTROPIC_EOS)
          prim(IPR,kl-k,j,i) = prim(IDN,kl-k,j,i)*Tkl;
      }
    }
  }
  return;
}

// Here is the upper z outflow boundary.
// The basic idea is that the pressure and density
// are exponentially extrapolated in the ghost zones
// assuming a constant temperature there (i.e., an
// isothermal atmosphere). The z velocity (NOT the
// momentum) are set to zero in the ghost zones in the
// case of the last upper physical zone having an inward
// flow.  All other variables are extrapolated into the
// ghost zones with zero slope.
void StratOutflowOuterX3(MeshBlock *pmb, Coordinates *pco,
                         AthenaArray<Real> &prim,
                         FaceField &b, Real time, Real dt,
                         int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  // Copy field components from last physical zone
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=1; k<=ngh; k++) {
      for (int j=jl; j<=ju; j++) {
        for (int i=il; i<=iu+1; i++) {
          b.x1f(ku+k,j,i) = b.x1f(ku,j,i);
        }
      }
    }
    for (int k=1; k<=ngh; k++) {
      for (int j=jl; j<=ju+1; j++) {
        for (int i=il; i<=iu; i++) {
          b.x2f(ku+k,j,i) = b.x2f(ku,j,i);
        }
      }
    }
    for (int k=1; k<=ngh; k++) {
      for (int j=jl; j<=ju; j++) {
        for (int i=il; i<=iu; i++) {
          b.x3f(ku+1+k,j,i) = b.x3f(ku+1,j,i);
        }
      }
    }
  } // MHD

  for (int k=1; k<=ngh; k++) {
    for (int j=jl; j<=ju; j++) {
      for (int i=il; i<=iu; i++) {
        Real x3 = pco->x3v(ku+k);
        Real x3b = pco->x3v(ku);
        Real den = prim(IDN,ku,j,i);
        // First calculate the effective gas temperature (Tku=cs^2)
        // in the last physical zone. If isothermal, use H=1
        Real Tku = 0.5*SQR(Omega_0);
        if (NON_BAROTROPIC_EOS) {
          Real pressku = prim(IPR,ku,j,i);
          pressku = std::max(pressku,pfloor);
          Tku = pressku/den;
        }
        // Now extrapolate the density to balance gravity
        // assuming a constant temperature in the ghost zones
        prim(IDN,ku+k,j,i) = den*std::exp(-(SQR(x3)-SQR(x3b))/H02/
                                          (2.0*Tku/SQR(Omega_0)));
        // Copy the velocities, but not the momenta ---
        // important because of the density extrapolation above
        prim(IVX,ku+k,j,i) = prim(IVX,ku,j,i);
        prim(IVY,ku+k,j,i) = prim(IVY,ku,j,i);
        // If there's inflow into the grid, set the normal velocity to zero
        if (prim(IVZ,ku,j,i) <= 0.0) {
          prim(IVZ,ku+k,j,i) = 0.0;
        } else {
          prim(IVZ,ku+k,j,i) = prim(IVZ,ku,j,i);
        }
        if (NON_BAROTROPIC_EOS)
          prim(IPR,ku+k,j,i) = prim(IDN,ku+k,j,i)*Tku;
      }
    }
  }
  return;
}


//  Here is the lower z outflow boundary.
//  These are the modified ones from Simon13, where Bx and By
//  are extrapolated like rho
// ONLY WORKS FOR ISOTHERMAL

void StratSimon13InnerX3(MeshBlock *pmb, Coordinates *pco,
                         AthenaArray<Real> &prim, FaceField &b,
                         Real time, Real dt,
                         int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  // Extrapolate field from last physical zones
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=1; k<=ngh; k++) {
      for (int j=jl; j<=ju; j++) {
        for (int i=il; i<=iu+1; i++) {
          Real x3 = pco->x3v(kl-k);
          Real x3b = pco->x3v(kl);
          b.x1f(kl-k,j,i) = b.x1f(kl,j,i)*std::exp(-(SQR(x3)-SQR(x3b))/H02);
        }
      }
    }
    for (int k=1; k<=ngh; k++) {
      for (int j=jl; j<=ju+1; j++) {
        for (int i=il; i<=iu; i++) {
          Real x3 = pco->x3v(kl-k);
          Real x3b = pco->x3v(kl);
          b.x2f(kl-k,j,i) = b.x2f(kl,j,i)*std::exp(-(SQR(x3)-SQR(x3b))/H02);
        }
      }
    }
    for (int k=1; k<=ngh; k++) {
      for (int j=jl; j<=ju; j++) {
        for (int i=il; i<=iu; i++) {
          b.x3f(kl-k,j,i) = b.x3f(kl,j,i);
        }
      }
    }
  } // MHD

  for (int k=1; k<=ngh; k++) {
    for (int j=jl; j<=ju; j++) {
      for (int i=il; i<=iu; i++) {
        Real x3 = pco->x3v(kl-k);
        Real x3b = pco->x3v(kl);
        Real den = prim(IDN,kl,j,i);
        // First calculate the effective gas temperature (Tkl=cs^2)
        // in the last physical zone. If isothermal, use H=1
        Real Tkl = 0.5*SQR(Omega_0);
        if (NON_BAROTROPIC_EOS) {
          Real presskl = prim(IPR,kl,j,i);
          presskl = std::max(presskl,pfloor);
          Tkl = presskl/den;
        }
        // Now extrapolate the density to balance gravity
        // assuming a constant temperature in the ghost zones
        prim(IDN,kl-k,j,i) = den*std::exp(-(SQR(x3)-SQR(x3b))/H02/
                                          (2.0*Tkl/SQR(Omega_0)));
        // Copy the velocities, but not the momenta ---
        // important because of the density extrapolation above
        prim(IVX,kl-k,j,i) = prim(IVX,kl,j,i);
        prim(IVY,kl-k,j,i) = prim(IVY,kl,j,i);
        // If there's inflow into the grid, set the normal velocity to zero
        if (prim(IVZ,kl,j,i) >= 0.0) {
          prim(IVZ,kl-k,j,i) = 0.0;
        } else {
          prim(IVZ,kl-k,j,i) = prim(IVZ,kl,j,i);
        }
        if (NON_BAROTROPIC_EOS)
          prim(IPR,kl-k,j,i) = prim(IDN,kl-k,j,i)*Tkl;
      }
    }
  }
  return;
}

//  Here is the lower z outflow boundary.
//  These are the modified ones from Simon13, where Bx and By
//  are extrapolated like rho
// ONLY WORKS FOR ISOTHERMAL
void StratSimon13OuterX3(MeshBlock *pmb, Coordinates *pco,
                         AthenaArray<Real> &prim,
                         FaceField &b, Real time, Real dt,
                         int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  // Extrapolate field from last physical zone
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=1; k<=ngh; k++) {
      for (int j=jl; j<=ju; j++) {
        for (int i=il; i<=iu+1; i++) {
          Real x3 = pco->x3v(ku+k);
          Real x3b = pco->x3v(ku);
          b.x1f(ku+k,j,i) = b.x1f(ku,j,i)*std::exp(-(SQR(x3)-SQR(x3b))/H02);
        }
      }
    }
    for (int k=1; k<=ngh; k++) {
      for (int j=jl; j<=ju+1; j++) {
        for (int i=il; i<=iu; i++) {
          Real x3 = pco->x3v(ku+k);
          Real x3b = pco->x3v(ku);
          b.x2f(ku+k,j,i) = b.x2f(ku,j,i)*std::exp(-(SQR(x3)-SQR(x3b))/H02);
        }
      }
    }
    for (int k=1; k<=ngh; k++) {
      for (int j=jl; j<=ju; j++) {
        for (int i=il; i<=iu; i++) {
          b.x3f(ku+1+k,j,i) = b.x3f(ku+1,j,i);
        }
      }
    }
  } // MHD

  for (int k=1; k<=ngh; k++) {
    for (int j=jl; j<=ju; j++) {
      for (int i=il; i<=iu; i++) {
        Real x3 = pco->x3v(ku+k);
        Real x3b = pco->x3v(ku);
        Real den = prim(IDN,ku,j,i);
        // First calculate the effective gas temperature (Tku=cs^2)
        // in the last physical zone. If isothermal, use H=1
        Real Tku = 0.5*SQR(Omega_0);
        if (NON_BAROTROPIC_EOS) {
          Real pressku = prim(IPR,ku,j,i);
          pressku = std::max(pressku,pfloor);
          Tku = pressku/den;
        }
        // Now extrapolate the density to balance gravity
        // assuming a constant temperature in the ghost zones
        prim(IDN,ku+k,j,i) = den*std::exp(-(SQR(x3)-SQR(x3b))/H02/
                                          (2.0*Tku/SQR(Omega_0)));
        // Copy the velocities, but not the momenta ---
        // important because of the density extrapolation above
        prim(IVX,ku+k,j,i) = prim(IVX,ku,j,i);
        prim(IVY,ku+k,j,i) = prim(IVY,ku,j,i);
        // If there's inflow into the grid, set the normal velocity to zero
        if (prim(IVZ,ku,j,i) <= 0.0) {
          prim(IVZ,ku+k,j,i) = 0.0;
        } else {
          prim(IVZ,ku+k,j,i) = prim(IVZ,ku,j,i);
        }
        if (NON_BAROTROPIC_EOS)
          prim(IPR,ku+k,j,i) = prim(IDN,ku+k,j,i)*Tku;
      }
    }
  }
  return;
}

// BCs from Lesur et al. 2013
void StratLesurInnerX3(MeshBlock *pmb, Coordinates *pco,
                         AthenaArray<Real> &prim, FaceField &b,
                         Real time, Real dt,
                         int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  // Bx is set to zero to allow in plane current
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=1; k<=ngh; k++) {
      for (int j=jl; j<=ju; j++) {
        for (int i=il; i<=iu+1; i++) {
          b.x1f(kl-k,j,i) = 0.;
        }
      }
    }
    for (int k=1; k<=ngh; k++) {
      for (int j=jl; j<=ju+1; j++) {
        for (int i=il; i<=iu; i++) {
          b.x2f(kl-k,j,i) = b.x2f(kl,j,i);
        }
      }
    }
    for (int k=1; k<=ngh; k++) {
      for (int j=jl; j<=ju; j++) {
        for (int i=il; i<=iu; i++) {
          b.x3f(kl-k,j,i) = b.x3f(kl,j,i);
        }
      }
    }
  } // MHD


  for (int k=1; k<=ngh; k++) {
    for (int j=jl; j<=ju; j++) {
      for (int i=il; i<=iu; i++) {
        Real x3 = pco->x3v(kl-k);
        Real x3b = pco->x3v(kl);
        Real den = prim(IDN,kl,j,i);
        
        // Outflow on rho
        prim(IDN,kl-k,j,i) = prim(IDN,kl,j,i);
        // outflow on v
        prim(IVX,kl-k,j,i) = prim(IVX,kl,j,i);
        prim(IVY,kl-k,j,i) = prim(IVY,kl,j,i);
        // If there's inflow into the grid, set the normal velocity to zero
        if (prim(IVZ,kl,j,i) >= 0.0) {
          prim(IVZ,kl-k,j,i) = 0.0;
        } else {
          prim(IVZ,kl-k,j,i) = prim(IVZ,kl,j,i);
        }
        if (NON_BAROTROPIC_EOS)
          prim(IPR,kl-k,j,i) = prim(IPR,kl,j,i);
      }
    }
  }
  return;
}

// BCs from Lesur et al. 2013
void StratLesurOuterX3(MeshBlock *pmb, Coordinates *pco,
                         AthenaArray<Real> &prim,
                         FaceField &b, Real time, Real dt,
                         int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  // Copy field components from last physical zone
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=1; k<=ngh; k++) {
      for (int j=jl; j<=ju; j++) {
        for (int i=il; i<=iu+1; i++) {
          b.x1f(ku+k,j,i) = 0.;
        }
      }
    }
    for (int k=1; k<=ngh; k++) {
      for (int j=jl; j<=ju+1; j++) {
        for (int i=il; i<=iu; i++) {
          b.x2f(ku+k,j,i) = b.x2f(ku,j,i);
        }
      }
    }
    for (int k=1; k<=ngh; k++) {
      for (int j=jl; j<=ju; j++) {
        for (int i=il; i<=iu; i++) {
          b.x3f(ku+1+k,j,i) = b.x3f(ku+1,j,i);
        }
      }
    }
  } // MHD

  for (int k=1; k<=ngh; k++) {
    for (int j=jl; j<=ju; j++) {
      for (int i=il; i<=iu; i++) {
        Real x3 = pco->x3v(ku+k);
        Real x3b = pco->x3v(ku);
        Real den = prim(IDN,ku,j,i);
        
        // Outflow on rho
        prim(IDN,ku+k,j,i) = prim(IDN,ku,j,i);
        // Copy the velocities, but not the momenta
        prim(IVX,ku+k,j,i) = prim(IVX,ku,j,i);
        prim(IVY,ku+k,j,i) = prim(IVY,ku,j,i);
        // If there's inflow into the grid, set the normal velocity to zero
        if (prim(IVZ,ku,j,i) <= 0.0) {
          prim(IVZ,ku+k,j,i) = 0.0;
        } else {
          prim(IVZ,ku+k,j,i) = prim(IVZ,ku,j,i);
        }
        if (NON_BAROTROPIC_EOS)
          prim(IPR,ku+k,j,i) = prim(IPR,ku+k,j,i);
      }
    }
  }
  return;
}


//  Here is the lower z outflow boundary.
//  These are the modified ones from Simon13, where Bx and By
//  are extrapolated like rho
// ONLY WORKS FOR ISOTHERMAL

void StratPowerLawInnerX3(MeshBlock *pmb, Coordinates *pco,
                         AthenaArray<Real> &prim, FaceField &b,
                         Real time, Real dt,
                         int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  // At low beta, magnetic pressure ~z^-b if density ~z^(-b-2)
  Real bplind = 0.5*(bc_pl_index-2.);
  // Extrapolate field from last physical zones
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=1; k<=ngh; k++) {
      for (int j=jl; j<=ju; j++) {
        for (int i=il; i<=iu+1; i++) {
          Real x3 = pco->x3v(kl-k);
          Real x3b = pco->x3v(kl);
          b.x1f(kl-k,j,i) = b.x1f(kl,j,i) * pow(x3b/x3, bplind);
        }
      }
    }
    for (int k=1; k<=ngh; k++) {
      for (int j=jl; j<=ju+1; j++) {
        for (int i=il; i<=iu; i++) {
          Real x3 = pco->x3v(kl-k);
          Real x3b = pco->x3v(kl);
          b.x2f(kl-k,j,i) = b.x2f(kl,j,i) * pow(x3b/x3, bplind);
        }
      }
    }
    for (int k=1; k<=ngh; k++) {
      for (int j=jl; j<=ju; j++) {
        for (int i=il; i<=iu; i++) {
          b.x3f(kl-k,j,i) = b.x3f(kl,j,i);
        }
      }
    }
  } // MHD

  for (int k=1; k<=ngh; k++) {
    for (int j=jl; j<=ju; j++) {
      for (int i=il; i<=iu; i++) {
        Real x3 = pco->x3v(kl-k);
        Real x3b = pco->x3v(kl);
        Real den = prim(IDN,kl,j,i);
        // First calculate the effective gas temperature (Tkl=cs^2)
        // in the last physical zone. If isothermal, use H=1
        // Now extrapolate the density to balance gravity
        // assuming a constant temperature in the ghost zones
        prim(IDN,kl-k,j,i) = den * pow(x3b/x3, bc_pl_index);
        // Copy the velocities, but not the momenta ---
        // important because of the density extrapolation above
        prim(IVX,kl-k,j,i) = prim(IVX,kl,j,i);
        prim(IVY,kl-k,j,i) = prim(IVY,kl,j,i);
        // If there's inflow into the grid, set the normal velocity to zero
        if (prim(IVZ,kl,j,i) >= 0.0) {
          prim(IVZ,kl-k,j,i) = 0.0;
        } else {
          prim(IVZ,kl-k,j,i) = prim(IVZ,kl,j,i);
        }
      }
    }
  }
  return;
}


//  Here is the lower z outflow boundary.
//  These are the modified ones from Simon13, where Bx and By
//  are extrapolated like rho
// ONLY WORKS FOR ISOTHERMAL
void StratPowerLawOuterX3(MeshBlock *pmb, Coordinates *pco,
                         AthenaArray<Real> &prim,
                         FaceField &b, Real time, Real dt,
                         int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  // At low beta, magnetic pressure ~z^-b if density ~z^(-b-2)
  Real bplind = 0.5*(bc_pl_index-2.);
  // Extrapolate field from last physical zones
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=1; k<=ngh; k++) {
      for (int j=jl; j<=ju; j++) {
        for (int i=il; i<=iu+1; i++) {
          Real x3 = pco->x3v(ku+k);
          Real x3b = pco->x3v(ku);
          b.x1f(ku+k,j,i) = b.x1f(ku,j,i) * pow(x3b/x3, bplind);
        }
      }
    }
    for (int k=1; k<=ngh; k++) {
      for (int j=jl; j<=ju+1; j++) {
        for (int i=il; i<=iu; i++) {
          Real x3 = pco->x3v(ku+k);
          Real x3b = pco->x3v(ku);
          b.x2f(ku+k,j,i) = b.x2f(ku,j,i) * pow(x3b/x3, bplind);
        }
      }
    }
    for (int k=1; k<=ngh; k++) {
      for (int j=jl; j<=ju; j++) {
        for (int i=il; i<=iu; i++) {
          b.x3f(ku+1+k,j,i) = b.x3f(ku+1,j,i);
        }
      }
    }
  } // MHD

  for (int k=1; k<=ngh; k++) {
    for (int j=jl; j<=ju; j++) {
      for (int i=il; i<=iu; i++) {
        Real x3 = pco->x3v(ku+k);
        Real x3b = pco->x3v(ku);
        Real den = prim(IDN,ku,j,i);
        // First calculate the effective gas temperature (Tku=cs^2)
        // in the last physical zone. If isothermal, use H=1
        // Now extrapolate the density to balance gravity
        // assuming a constant temperature in the ghost zones
        prim(IDN,ku+k,j,i) = den * pow(x3b/x3, bc_pl_index);
        // Copy the velocities, but not the momenta ---
        // important because of the density extrapolation above
        prim(IVX,ku+k,j,i) = prim(IVX,ku,j,i);
        prim(IVY,ku+k,j,i) = prim(IVY,ku,j,i);
        // If there's inflow into the grid, set the normal velocity to zero
        if (prim(IVZ,ku,j,i) <= 0.0) {
          prim(IVZ,ku+k,j,i) = 0.0;
        } else {
          prim(IVZ,ku+k,j,i) = prim(IVZ,ku,j,i);
        }
      }
    }
  }
  return;
}




namespace {

Real HistoryBxBy(MeshBlock *pmb, int iout) {
  Real bxby = 0;
  int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
  AthenaArray<Real> &b = pmb->pfield->bcc;
  AthenaArray<Real> volume; // 1D array of volumes
  // allocate 1D array for cell volume used in usr def history
  volume.NewAthenaArray(pmb->ncells1);

  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      pmb->pcoord->CellVolume(k,j,pmb->is,pmb->ie,volume);
      for (int i=is; i<=ie; i++) {
        bxby -= volume(i)*b(IB1,k,j,i)*b(IB2,k,j,i);
      }
    }
  }
  return bxby;
}


Real HistorydVxVy(MeshBlock *pmb, int iout) {
  Real dvxvy = 0.0;
  int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
  AthenaArray<Real> &w = pmb->phydro->w;
  Real vshear = 0.0;
  AthenaArray<Real> volume; // 1D array of volumes
  // allocate 1D array for cell volume used in usr def history
  volume.NewAthenaArray(pmb->ncells1);

  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      pmb->pcoord->CellVolume(k,j,pmb->is,pmb->ie,volume);
      for (int i=is; i<=ie; i++) {
        if(!pmb->porb->orbital_advection_defined) {
          vshear = -qshear*Omega_0*pmb->pcoord->x1v(i);
        } else {
          vshear = 0.0;
        }
        dvxvy += volume(i)*w(IDN,k,j,i)*w(IVX,k,j,i)*(w(IVY,k,j,i) + vshear);
      }
    }
  }
  return dvxvy;
}
} // namespace
