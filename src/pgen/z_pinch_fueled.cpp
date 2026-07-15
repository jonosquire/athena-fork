//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//========================================================================================
//! \file blast.cpp
//! \brief Problem generator for cylindrical z pinch problem
//!
//! This file sets up the cylindrical z pinch problem for Athena++ by defining initial
//! conditions, boundary conditions, and source/forcing functions.
//
//========================================================================================

// C++ headers
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
#include <iostream>
#include <vector>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"

// Forward declarations
void InnerX1HardWall(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                     FaceField &b, Real time, Real dt,
                     int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void OuterX1HardWall(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                     FaceField &b, Real time, Real dt,
                     int il, int iu, int jl, int ju, int kl, int ku, int ngh);

void AddDensity(MeshBlock *pmb, const Real time, const Real dt,
                const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
                const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
                AthenaArray<Real> &cons_scalar);

// Global variables
// NOTE: input_location_dens is the centre of the density Gaussian (set near rout in input file).
//       sigma_dens controls the width of the density Gaussian independently of sigma (pressure).
Real bin, rin, presmin, densmin, beta, dens_init, pres_init, total_volume,
     sigma, sigma_dens,
     input_location, input_location_dens,
     igm1, dens_added, thermal_energy_added, dens_removed, energy_removed,
     pressure_current, sink_rate, sink_width,
     injection_rate_dens, injection_rate_pres,
     sink_rate_dens, sink_rate_energy,
     d0, rpeak, expdens, exppres, r_const, rcSmooth;

// History output functions
Real HistoryAddedD(MeshBlock *pmb, int iout){
  Real total_dens_added = dens_added;
  dens_added = 0.;
  return total_dens_added;
}
Real HistoryAddedP(MeshBlock *pmb, int iout){
  Real total_energy_added = thermal_energy_added;
  thermal_energy_added = 0.;
  return total_energy_added;
}
Real HistoryAddedp(MeshBlock *pmb, int iout){
  Real total_pressure = pressure_current;
  pressure_current = 0.;
  return total_pressure;
}
Real HistoryAddedDR(MeshBlock *pmb, int iout){
  Real total_dens_removed = dens_removed;
  dens_removed = 0.;
  return total_dens_removed;
}
Real HistoryAddedER(MeshBlock *pmb, int iout){
  Real total_energy_removed = energy_removed;
  energy_removed = 0.;
  return total_energy_removed;
}

//----------------------------------------------------------------------------------------
// Function: Mesh::InitUserMeshData
void Mesh::InitUserMeshData(ParameterInput *pin) {
  bin      = pin->GetReal("problem", "bin");
  rin      = pin->GetReal("mesh", "x1min");
  presmin  = pin->GetOrAddReal("problem", "presmin", 0.1);
  densmin  = pin->GetOrAddReal("problem", "densmin", 0.1);
  beta     = pin->GetOrAddReal("problem", "beta", 0.5);

  // Pressure source parameters (centre of domain, unchanged)
  sigma          = pin->GetOrAddReal("problem", "input_gaussian_sigma", 0.05);
  input_location = pin->GetOrAddReal("problem", "input_location", 1.5);

  // Density source parameters (outer edge).
  // input_location_dens defaults to mesh x1max; set explicitly in input file for control.
  // NOTE: the outer BC (OuterX1HardWall) extrapolates density as r^{-2}. If the density
  //       source is active near rout, the BC ghost cells will drain mass back out each step.
  //       Consider switching the outer density BC to a fixed or zero-gradient condition
  //       when this source is active. See OuterX1HardWall below.
  Real rout_global       = pin->GetReal("mesh", "x1max");
  input_location_dens    = pin->GetOrAddReal("problem", "input_location_dens", rout_global);
  sigma_dens             = pin->GetOrAddReal("problem", "sigma_dens", sigma);

  sink_rate          = pin->GetOrAddReal("problem", "sink_rate", 0.2);
  sink_width         = pin->GetOrAddReal("problem", "sink_width", 0.3);
  dens_init          = pin->GetOrAddReal("problem", "dens_init", 8.5163406660979639);
  pres_init          = pin->GetOrAddReal("problem", "pres_init", 4.4954022196421644);
  injection_rate_dens  = pin->GetOrAddReal("problem", "injection_rate_dens", 0.1);
  injection_rate_pres  = pin->GetOrAddReal("problem", "injection_rate_pres", 0.1);
  sink_rate_dens     = pin->GetOrAddReal("problem", "sink_rate_dens", 0.1);
  sink_rate_energy   = pin->GetOrAddReal("problem", " sink_rate_energy", 0.1);

  d0       = pin->GetOrAddReal("problem", "d0",      1.0);
  rpeak    = pin->GetReal          ("problem", "rpeak");
  expdens  = pin->GetReal          ("problem", "expdens");
  exppres  = pin->GetOrAddReal     ("problem", "exppres", 1.0);
  r_const  = pin->GetOrAddReal     ("problem", "r_const", 2.0);
  rcSmooth = pin->GetOrAddReal     ("problem", "rcSmooth", 5.0);

  if (mesh_bcs[BoundaryFace::inner_x1] == GetBoundaryFlag("user"))
    EnrollUserBoundaryFunction(BoundaryFace::inner_x1, InnerX1HardWall);
  if (mesh_bcs[BoundaryFace::outer_x1] == GetBoundaryFlag("user"))
    EnrollUserBoundaryFunction(BoundaryFace::outer_x1, OuterX1HardWall);

  Real forcing_flag = pin->GetOrAddInteger("problem", "forcing_flag", 0);
  if (forcing_flag == 1){
    EnrollUserExplicitSourceFunction(AddDensity);
    AllocateUserHistoryOutput(5);
    EnrollUserHistoryOutput(0, HistoryAddedD,  "dens_added");
    EnrollUserHistoryOutput(1, HistoryAddedP,  "thermal_energy_added");
    EnrollUserHistoryOutput(2, HistoryAddedp,  "pressure_current");
    EnrollUserHistoryOutput(3, HistoryAddedDR, "dens_removed");
    EnrollUserHistoryOutput(4, HistoryAddedER, "energy_removed");
  }

  return;
}

//----------------------------------------------------------------------------------------
// Function: trapezoidalIntegration1
double trapezoidalIntegration1(double r, int N, double p0, double presmin,
                                double exppres, double rpeak, double rin){
  double h = (r - rin) / N;
  double sum = 0;
  for (int m = 1; m < N; ++m) {
      double x = rin + m * h;
      sum += SQR(x) * (p0-presmin)/2 * (PI/(rpeak-rin)) * std::sin(PI * (x - rin) / (rpeak - rin));
  }
  return h * sum;
}

//----------------------------------------------------------------------------------------
// Function: trapezoidalIntegration2
double trapezoidalIntegration2(double rpeak, double end, int N, double beta,
                                double exppres, double rcSmooth, double r_const){
  double h = (end - rpeak) / N;

  double exp_rc_end    = exp(-rcSmooth*(r_const-end));
  double first_exp_part_end   = rcSmooth*exp_rc_end/SQR(exp_rc_end + 1.);
  double first_decay_part_end = beta*pow(1.+SQR(end-rpeak), -exppres/2.);
  double first_part_end       = first_exp_part_end*first_decay_part_end;

  double second_exp_part_end   = 1./(1. + exp_rc_end);
  double second_power_part_end = pow(1+SQR(end-rpeak), -exppres/2. - 1.);
  double second_decay_part_end = beta*exppres*(end-rpeak)*second_power_part_end;
  double second_part_end       = second_exp_part_end*second_decay_part_end;

  double exp_end_rc          = exp(-rcSmooth*(end-r_const));
  double third_exp_part_end  = rcSmooth*exp_end_rc/SQR(exp_end_rc + 1.);
  double third_decay_part_end = beta*pow(1+SQR(r_const-rpeak), -exppres/2.);
  double third_part_end       = third_exp_part_end*third_decay_part_end;

  double sum = 1/2 * SQR(end)*(-first_part_end - second_part_end + third_part_end);

  for (int m = 1; m < N; ++m){
    double x = rpeak + m*h;

    double exp_rc_x    = exp(-rcSmooth*(r_const-x));
    double first_exp_part_x   = rcSmooth*exp_rc_x/SQR(exp_rc_x+1);
    double first_decay_part_x = beta*pow(1.+SQR(x-rpeak), -exppres/2.);
    double first_part_x       = first_exp_part_x*first_decay_part_x;

    double second_exp_part_x   = 1./(1. + exp_rc_x);
    double second_power_part_x = pow(1+SQR(x-rpeak), -exppres/2. - 1.);
    double second_decay_part_x = beta*exppres*(x-rpeak)*second_power_part_x;
    double second_part_x       = second_exp_part_x*second_decay_part_x;

    double exp_x_rc          = exp(-rcSmooth*(x-r_const));
    double third_exp_part_x  = rcSmooth*exp_x_rc/SQR(exp_x_rc + 1.);
    double third_decay_part_x = beta*pow(1+SQR(r_const-rpeak), -exppres/2.);
    double third_part_x       = third_exp_part_x*third_decay_part_x;

    sum += SQR(x) * (-first_part_x - second_part_x + third_part_x);
  }

  return h * sum;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  Real d0 = pin->GetOrAddReal("problem", "d0", 1.0);
  beta    = pin->GetOrAddReal("problem", "beta", 0.5);
  Real rpeak    = pin->GetReal("problem", "rpeak");
  Real r_const  = pin->GetOrAddReal("problem", "r_const", 2);
  presmin = pin->GetOrAddReal("problem", "presmin", 0.1);
  densmin = pin->GetOrAddReal("problem", "densmin", 0.1);
  Real exppres  = pin->GetOrAddReal("problem", "exppres", 1);
  Real expdens  = pin->GetReal("problem", "expdens");
  Real sharpness = pin->GetOrAddReal("problem", "sharpness", 1);
  Real gamma = peos->GetGamma();
  Real gm1   = gamma - 1.0;
  Real N     = pin->GetOrAddInteger("problem", "N", 1000);
  rin  = pmy_mesh->mesh_size.x1min;
  Real rout  = pmy_mesh->mesh_size.x1max;
  Real Lz    = pmy_mesh->mesh_size.x3max - pmy_mesh->mesh_size.x3min;
  Real Lx    = rout - rin;
  Real Amp   = pin->GetOrAddReal("problem", "amp", 0.01);
  Real emode_width = pin->GetOrAddReal("problem", "emode_width", 0.2);
  Real rcSmooth    = pin->GetOrAddReal("problem", "rcSmooth", 5.);
  Real beta_correct = 1/(1+exp(-rcSmooth*(r_const-rpeak))) * beta +
                      1/(1+exp(-rcSmooth*(rpeak-r_const))) * beta * (pow(1+SQR(r_const-rpeak),-exppres/2.));
  Real centre = (rout+rin)/2;

  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    // cylindrical: no action needed
  } else {
    std::stringstream msg;
    msg << "### FATAL ERROR in z_pinch.cpp ProblemGenerator" << std::endl
        << "Unrecognized COORDINATE_SYSTEM=" << COORDINATE_SYSTEM << std::endl;
    ATHENA_ERROR(msg);
  }
  if (not MAGNETIC_FIELDS_ENABLED) {
    std::stringstream msg;
    msg << "### FATAL ERROR in z_pinch.cpp ProblemGenerator" << std::endl
        << "Magnetic fields not enabled" << std::endl;
    ATHENA_ERROR(msg);
  }

  Real pfloor = 1.e-2;
  Real dfloor = 1.e-2;

  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        Real r = pcoord->x1v(i);
        if (r < rpeak) {
          phydro->u(IDN,k,j,i) = (d0+densmin)/2 - (d0-densmin)/2*std::cos(PI * (r - rin) / (rpeak - rin));
          if (NON_BAROTROPIC_EOS)
            phydro->u(IEN,k,j,i) = ((beta_correct+presmin)/2 - (beta_correct-presmin)/2*std::cos(PI * (r - rin) / (rpeak - rin)))/gm1;
        } else {
          phydro->u(IDN,k,j,i) = d0 * pow(1 + SQR(r-rpeak), -expdens/2);
          if (NON_BAROTROPIC_EOS)
             phydro->u(IEN,k,j,i) = 1/gm1 * ( 1/(1+exp(-rcSmooth*(r_const-r))) * beta * (pow(1+SQR(r-rpeak),-exppres/2.)) +
                                              1/(1+exp(-rcSmooth*(r-r_const))) * beta * (pow(1+SQR(r_const-rpeak),-exppres/2.)) );
        }
        phydro->u(IM1,k,j,i) = phydro->u(IDN,k,j,i) * Amp * std::sin(2*PI/Lz * pcoord->x3v(k)) *
                               std::exp(-SQR(r-centre)/(2*SQR(emode_width))) * std::sin(2*PI/(rout-rin)*(r-rin));
        phydro->u(IM2,k,j,i) = 0.0;
        phydro->u(IM3,k,j,i) = 0.0;
      }
    }
  }

  dens_init    = 0.;
  pres_init    = 0.;
  total_volume = 0.0;
  AthenaArray<Real> vol(ncells1);
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      pcoord->CellVolume(k, j, is, ie, vol);
      for (int i=is; i<=ie; i++) {
        Real den  = phydro->u(IDN,k,j,i);
        Real pres = phydro->u(IEN,k,j,i)*gm1;
        dens_init    += den*vol(i);
        pres_init    += pres*vol(i);
        total_volume += vol(i);
      }
    }
  }

  #ifdef MPI_PARALLEL
    MPI_Allreduce(MPI_IN_PLACE, &dens_init, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &pres_init, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  #endif

  std::cout << "dens_init = " << dens_init << "; press_init = " << pres_init << "\n";

  Real result;
  for (int i=is; i<=ie; ++i) {
    Real r = pcoord->x1v(i);
    if (r < rpeak){
      result = trapezoidalIntegration1(r, N, beta_correct, presmin, exppres, rpeak, rin);
    } else{
      Real result_part1 = trapezoidalIntegration1(rpeak, N, beta_correct, presmin, exppres, rpeak, rin);
      Real result_part2 = trapezoidalIntegration2(rpeak, r, N, beta, exppres, rcSmooth, r_const);
      result = result_part1 + result_part2;
    }
    for (int j=js; j<=je+1; ++j) {
      for (int k=ks; k<=ke; ++k) {
        pfield->b.x2f(k,j,i) = 1/r*sqrt(SQR(bin*rin) - 2. * result);
      }
    }
  }
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie+1; ++i) {
        pfield->b.x1f(k,j,i) = 0.;
      }
    }
  }
  for (int k=ks; k<=ke+1; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        pfield->b.x3f(k,j,i) = 0.;
      }
    }
  }

  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        Real lw, rw;
        const Real& x2f_j  = pcoord->x2f(j);
        const Real& x2f_jp = pcoord->x2f(j+1);
        const Real& x2v_j  = pcoord->x2v(j);
        const Real& dx2_j  = pcoord->dx2f(j);
        lw = (x2f_jp - x2v_j)/dx2_j;
        rw = (x2v_j  - x2f_j)/dx2_j;
        phydro->u(IEN,k,j,i) += 0.5*SQR(lw*pfield->b.x2f(k,j,i) + rw*pfield->b.x2f(k,j+1,i)) +
                                0.5*SQR(phydro->u(IM1,k,j,i))/phydro->u(IDN,k,j,i);
      }
    }
  }
}


//----------------------------------------------------------------------------------------
// Function: InnerX1HardWall
void InnerX1HardWall(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                     FaceField &b, Real time, Real dt,
                     int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  for (int n=0; n<(NHYDRO); ++n) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
        if (n==(IDN)) {
#pragma omp simd
          for (int i=1; i<=ngh; ++i) {
            prim(n,k,j,il-i) = densmin;
          }
        } else if (n==(IPR)) {
#pragma omp simd
          for (int i=1; i<=ngh; ++i) {
            prim(n,k,j,il-i) = presmin;
          }
        } else {
#pragma omp simd
          for (int i=1; i<=ngh; ++i) {
            prim(n,k,j,il-i) = 0.;
          }
        }
      }
    }
  }
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
#pragma omp simd
      for (int i=1; i<=ngh; ++i) {
        b.x1f(k,j,(il-i)) = 0.;
      }
    }
  }
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju+1; ++j) {
#pragma omp simd
      for (int i=1; i<=ngh; ++i) {
        Real r = pco->x1v(il-i);
        b.x2f(k,j,(il-i)) = 1/r * bin * rin;
      }
    }
  }
  for (int k=kl; k<=ku+1; ++k) {
    for (int j=jl; j<=ju; ++j) {
#pragma omp simd
      for (int i=1; i<=ngh; ++i) {
        b.x3f(k,j,(il-i)) = 0.;
      }
    }
  }
  return;
}


//----------------------------------------------------------------------------------------
// Function: OuterX1HardWall
//
// NOTE: The density BC here uses a power-law extrapolation (r^{-2}).
// With the density source now located near rout, this BC will drain mass
// injected near the outer boundary back out through the ghost zone each step.
// If this causes unphysical behaviour (e.g. density source immediately cancelled
// by the BC), switch the density ghost zone to zero-gradient (outflow):
//   prim(IDN,k,j,iu+i) = prim(IDN,k,j,iu);
// or a fixed floor value, depending on the intended physical setup.
void OuterX1HardWall(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                     FaceField &b, Real time, Real dt,
                     int il, int iu, int jl, int ju, int kl, int ku, int ngh){
  for (int n=0; n<(NHYDRO); ++n) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
        if (n==(IDN)) {
#pragma omp simd
          for (int i=1; i<=ngh; ++i) {
            Real r     = pco->x1v(i+iu);
            Real r_out = pco->x1v(iu);
            // CURRENT: power-law extrapolation. See note above about interaction
            // with the outer-edge density source.
            prim(n,k,j,iu+i) = prim(n,k,j,(iu)) * pow(r/r_out,-2.);
            // ALTERNATIVE (zero-gradient / outflow):
            // prim(n,k,j,iu+i) = prim(n,k,j,iu);
          }
        } else if (n==(IPR)) {
#pragma omp simd
          for (int i=1; i<=ngh; ++i) {
            Real r     = pco->x1v(i+iu);
            Real r_out = pco->x1v(iu);
            prim(n,k,j,iu+i) = prim(n,k,j,(iu)) * pow(r/r_out,-10./3.);
          }
        } else if (n==(IVX)) {
#pragma omp simd
          for (int i=1; i<=ngh; ++i) {
            prim(n,k,j,iu+i) = 0.;
          }
        } else {
#pragma omp simd
          for (int i=1; i<=ngh; ++i) {
            prim(n,k,j,iu+i) = 0.;
          }
        }
      }
    }
  }

  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
#pragma omp simd
      for (int i=1; i<=ngh; ++i) {
        b.x1f(k,j,(iu+i+1)) = 0.;
      }
    }
  }
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju+1; ++j) {
#pragma omp simd
      for (int i=1; i<=ngh; ++i) {
        Real r     = pco->x1v(iu+i);
        Real routv = pco->x1v(iu);
        b.x2f(k,j,(iu+i)) = 1/r * b.x2f(k,j,iu) * routv;
      }
    }
  }
  for (int k=kl; k<=ku+1; ++k) {
    for (int j=jl; j<=ju; ++j) {
#pragma omp simd
      for (int i=1; i<=ngh; ++i) {
        b.x3f(k,j,(iu+i)) = 0.;
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
// Function: AddDensity
//
// Density source: Gaussian centred at input_location_dens (near rout), width sigma_dens.
// Pressure source: Gaussian centred at input_location (centre), width sigma. 
// Sink: exponential sink at outer edge, normalised over the full domain.
void AddDensity(MeshBlock *pmb, const Real time, const Real dt,
              const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
              const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
              AthenaArray<Real> &cons_scalar) {

  if (time == 0.0) return;

  Real gamma       = pmb->peos->GetGamma();
  Real gm1         = gamma - 1.0;
  Real local_igm1  = 1. / gm1;

  Real dens_curr   = 0.0;
  Real pres_curr   = 0.0;
  Real d_added     = 0.0;
  Real energy_added = 0.0;

  Real Lz     = pmb->pmy_mesh->mesh_size.x3max - pmb->pmy_mesh->mesh_size.x3min;
  Real Ltheta = pmb->pmy_mesh->mesh_size.x2max - pmb->pmy_mesh->mesh_size.x2min;

  // Separate normalisation factors for the density and pressure Gaussians.
  // norm_dens uses sigma_dens and input_location_dens (outer edge).
  // norm_pres uses sigma and input_location (centre). Both normalise over the 2D
  // theta-z domain; the radial Gaussian normalisation is handled implicitly by the
  // injection rate multiplied by dens_init / pres_init.
  Real norm_dens = 1. / (std::sqrt(2. * M_PI) * sigma_dens * Lz * Ltheta);
  Real norm_pres = 1. / (std::sqrt(2. * M_PI) * sigma       * Lz * Ltheta);

  AthenaArray<Real> vol(pmb->ncells1);

  //------------------------------------------------------------------------------------
  // Source stage: inject density at outer edge, pressure at centre.
  for (int k = pmb->ks; k <= pmb->ke; k++) {
    for (int j = pmb->js; j <= pmb->je; j++) {
      pmb->pcoord->CellVolume(k, j, pmb->is, pmb->ie, vol);
#pragma omp simd
      for (int i = pmb->is; i <= pmb->ie; i++) {
        Real x    = pmb->pcoord->x1v(i);
        Real den  = cons(IDN, k, j, i);
        Real iden = 1. / den;

        Real tot_energy      = cons(IEN, k, j, i);
        Real kinetic_energy  = 0.5 * iden * (SQR(cons(IM1, k, j, i)) +
                                              SQR(cons(IM2, k, j, i)) +
                                              SQR(cons(IM3, k, j, i)));
        Real magnetic_energy = 0.5 * (SQR(bcc(IB1, k, j, i)) +
                                      SQR(bcc(IB2, k, j, i)) +
                                      SQR(bcc(IB3, k, j, i)));
        Real plasma_energy   = tot_energy - kinetic_energy - magnetic_energy;

        // Density Gaussian: centred at input_location_dens (outer edge) with sigma_dens.
        Real gaussian_dens = norm_dens * exp(-0.5 * SQR(x - input_location_dens) / SQR(sigma_dens));

        // Pressure Gaussian: centred at input_location (centre) with sigma. 
        Real gaussian_pres = norm_pres * exp(-0.5 * SQR(x - input_location) / SQR(sigma));

        Real dens_to_add = dt * injection_rate_dens * dens_init * gaussian_dens;
        Real prs_to_add  = dt * injection_rate_pres * pres_init * gaussian_pres;

        // Add mass
        cons(IDN, k, j, i) += dens_to_add;
        d_added            += dens_to_add * vol(i);

        // Rescale kinetic energy for momentum conservation (injected material is at rest)
        Real kinetic_energy_new = kinetic_energy * den / cons(IDN, k, j, i);

        // Add thermal energy from pressure source
        plasma_energy  += prs_to_add * local_igm1;
        energy_added   += prs_to_add * local_igm1 * vol(i);

        cons(IEN, k, j, i) = kinetic_energy_new + magnetic_energy + plasma_energy;
      }
    }
  }

#ifdef MPI_PARALLEL
  MPI_Allreduce(MPI_IN_PLACE, &d_added,      1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &energy_added, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif

  dens_added           = d_added;
  thermal_energy_added = energy_added;

  if (Globals::my_rank == 0) {
    std::cout << "Δmass      = " << d_added
              << ";  Δth-energy = " << energy_added << "\n";
  }

  //------------------------------------------------------------------------------------
  // Compute integrated density and pressure after injection.
  for (int k = pmb->ks; k <= pmb->ke; k++) {
    for (int j = pmb->js; j <= pmb->je; j++) {
      pmb->pcoord->CellVolume(k, j, pmb->is, pmb->ie, vol);
#pragma omp simd
      for (int i = pmb->is; i <= pmb->ie; i++) {
        Real density_value  = cons(IDN, k, j, i);
        Real pressure_value = gm1 * (cons(IEN, k, j, i)
           - 0.5/density_value*(SQR(cons(IM1,k,j,i))+SQR(cons(IM2,k,j,i))+SQR(cons(IM3,k,j,i)))
           - 0.5*(SQR(bcc(IB1,k,j,i))+SQR(bcc(IB2,k,j,i))+SQR(bcc(IB3,k,j,i))));
        dens_curr += density_value  * vol(i);
        pres_curr += pressure_value * vol(i);
      }
    }
  }

#ifdef MPI_PARALLEL
  MPI_Allreduce(MPI_IN_PLACE, &dens_curr, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &pres_curr, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif

  pressure_current = pres_curr;

  if (Globals::my_rank == 0) {
    std::cout << "dens_init = " << dens_init << "; pres_init = " << pres_init << "\n";
    std::cout << "dens_curr = " << dens_curr << "; pres_curr = " << pres_curr << "\n";
  }

  //------------------------------------------------------------------------------------
  // Sink stage: normalised exponential sink at outer edge.

  Real sink_total = 0.0;
  Real rout       = pmb->pmy_mesh->mesh_size.x1max;

  // First pass: accumulate un-normalised sink profile over this rank's cells.
  for (int k = pmb->ks; k <= pmb->ke; k++) {
    for (int j = pmb->js; j <= pmb->je; j++) {
      pmb->pcoord->CellVolume(k, j, pmb->is, pmb->ie, vol);
#pragma omp simd
      for (int i = pmb->is; i <= pmb->ie; i++) {
        Real x            = pmb->pcoord->x1v(i);
        Real sink_profile = exp((x - rout) / sink_width);
        sink_total       += sink_profile * vol(i);
      }
    }
  }

  // Reduce sink_total across all MPI ranks BEFORE computing sink_norm.
#ifdef MPI_PARALLEL
  MPI_Allreduce(MPI_IN_PLACE, &sink_total, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif

  Real sink_norm = 1.0 / sink_total;
  Real d_removed = 0.0, e_removed = 0.0;

  // Second pass: apply normalised sink.
  for (int k = pmb->ks; k <= pmb->ke; k++) {
    for (int j = pmb->js; j <= pmb->je; j++) {
      pmb->pcoord->CellVolume(k, j, pmb->is, pmb->ie, vol);
#pragma omp simd
      for (int i = pmb->is; i <= pmb->ie; i++) {
        Real x            = pmb->pcoord->x1v(i);
        Real sink_profile = exp((x - rout) / sink_width);
        Real sink         = sink_profile * sink_norm;

        Real den              = cons(IDN, k, j, i);
        Real den_old          = den;
        Real iden             = 1.0 / den;
        Real kinetic_energy   = 0.5 * iden * (SQR(cons(IM1, k, j, i)) +
                                               SQR(cons(IM2, k, j, i)) +
                                               SQR(cons(IM3, k, j, i)));
        Real magnetic_energy  = 0.5 * (SQR(bcc(IB1, k, j, i)) +
                                       SQR(bcc(IB2, k, j, i)) +
                                       SQR(bcc(IB3, k, j, i)));
        Real plasma_energy    = cons(IEN, k, j, i) - kinetic_energy - magnetic_energy;
        Real plasma_energy_old = plasma_energy;

        Real red_fact_dens = 1.0 - dt * sink * sink_rate_dens;
        Real red_fact_Eth  = 1.0 - dt * sink * sink_rate_energy;

        if (red_fact_dens < 0.0) {
          std::cout << "### WARNING in SinkStep: red_fact_dens < 0 at (k,j,i)=("
                    << k << "," << j << "," << i << ") = " << red_fact_dens
                    << "  [dt * sink * sink_rate_dens = "
                    << (dt * sink * sink_rate_dens) << "]" << std::endl;
        }
        if (red_fact_Eth < 0.0) {
          std::cout << "### WARNING in SinkStep: red_fact_Eth < 0 at (k,j,i)=("
                    << k << "," << j << "," << i << ") = " << red_fact_Eth
                    << "  [dt * sink * sink_rate_energy = "
                    << (dt * sink * sink_rate_energy) << "]" << std::endl;
        }

        cons(IDN, k, j, i) = den_old * red_fact_dens;
        Real den_removed_val = den_old - cons(IDN, k, j, i);
        d_removed           += den_removed_val * vol(i);

        Real kinetic_energy_new = kinetic_energy * den / cons(IDN, k, j, i);

        plasma_energy    = plasma_energy_old * red_fact_Eth;
        Real eth_removed  = plasma_energy_old - plasma_energy;
        e_removed        += eth_removed * vol(i);

        cons(IEN, k, j, i) = kinetic_energy_new + magnetic_energy + plasma_energy;
      }
    }
  }

#ifdef MPI_PARALLEL
  MPI_Allreduce(MPI_IN_PLACE, &d_removed, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &e_removed, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif

  dens_removed   = d_removed;
  energy_removed = e_removed;

  if (Globals::my_rank == 0) {
    std::cout << "Δmass_removed     = " << d_removed
              << ";  Δth-energy_removed = " << e_removed << "\n";
  }

  //------------------------------------------------------------------------------------
  // Diagnostic: final integrated density and pressure.
  Real dens_fin = 0.0, pres_fin = 0.0;
  for (int k = pmb->ks; k <= pmb->ke; k++) {
    for (int j = pmb->js; j <= pmb->je; j++) {
      pmb->pcoord->CellVolume(k, j, pmb->is, pmb->ie, vol);
#pragma omp simd
      for (int i = pmb->is; i <= pmb->ie; i++) {
        Real density_value  = cons(IDN, k, j, i);
        Real pressure_value = gm1 * (cons(IEN, k, j, i)
                             - 0.5 / density_value * (SQR(cons(IM1, k, j, i)) +
                                                      SQR(cons(IM2, k, j, i)) +
                                                      SQR(cons(IM3, k, j, i)))
                             - 0.5 * (SQR(bcc(IB1, k, j, i)) +
                                      SQR(bcc(IB2, k, j, i)) +
                                      SQR(bcc(IB3, k, j, i))));
        dens_fin += density_value  * vol(i);
        pres_fin += pressure_value * vol(i);
      }
    }
  }
#ifdef MPI_PARALLEL
  MPI_Allreduce(MPI_IN_PLACE, &dens_fin, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &pres_fin, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif

  return;
}