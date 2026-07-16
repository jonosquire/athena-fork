//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//========================================================================================
//! \file z_pinch_fueled.cpp
//! \brief Driven cylindrical z-pinch with separated particle/heat sources and tracers
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
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
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
#include "../scalars/scalars.hpp"

// Forward declarations
void InnerX1HardWall(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                     FaceField &b, Real time, Real dt,
                     int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void OuterX1HardWall(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                     FaceField &b, Real time, Real dt,
                     int il, int iu, int jl, int ju, int kl, int ku, int ngh);

void ApplyFueling(MeshBlock *pmb, const Real time, const Real dt,
                  const AthenaArray<Real> &prim,
                  const AthenaArray<Real> &prim_scalar,
                  const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
                  AthenaArray<Real> &cons_scalar);

namespace {

// Per-MeshBlock values written by the source routine.  Storing these on each block makes
// Athena++'s standard history reduction correct for multiple blocks, MPI, and OpenMP.
enum BlockDiagnostic {
  kMassSourceRate = 0,
  kThermalSourceRate,
  kScalar1SourceRate,
  kScalar2SourceRate,
  kMassSinkRate,
  kThermalSinkRate,
  kScalar1SinkRate,
  kScalar2SinkRate,
  kTotalEnergySourceRate,
  kTotalEnergySinkRate,
  kNumBlockDiagnostics
};

enum HistoryDiagnostic {
  kThermalEnergy = 0,
  kHistMassSourceRate,
  kHistThermalSourceRate,
  kHistScalar1SourceRate,
  kHistScalar2SourceRate,
  kHistMassSinkRate,
  kHistThermalSinkRate,
  kHistScalar1SinkRate,
  kHistScalar2SinkRate,
  kHistTotalEnergySourceRate,
  kHistTotalEnergySinkRate,
  kPressureSourceNormalization,
  kDensitySourceNormalization,
  kSinkNormalization,
  kMassFluxProbe1,
  kMassFluxProbe2,
  kMassFluxProbe3,
  kBoundaryMassFlux,
  kBoundaryEnergyFlux,
  kBoundaryScalar1Flux,
  kBoundaryScalar2Flux,
  kNumHistoryDiagnostics
};

Real bin, rin, presmin, densmin, beta, dens_init, pres_init, total_volume;
Real sigma_p, sigma_rho, r_p, r_rho;
Real source_integral_p, source_integral_rho, sink_integral;
Real sink_width, injection_rate_dens, injection_rate_pres;
Real injection_rate_scalar1, injection_rate_scalar2;
Real sink_rate_dens, sink_rate_energy;
Real d0, rpeak, expdens, exppres, r_const, rcSmooth;
Real probe_radius[3], probe_volume[3];
int forcing_flag;

Real GetNewOrLegacyReal(ParameterInput *pin, const char *new_name,
                        const char *legacy_name, Real default_value) {
  if (pin->DoesParameterExist("problem", new_name)) {
    return pin->GetReal("problem", new_name);
  }
  if (legacy_name != nullptr && pin->DoesParameterExist("problem", legacy_name)) {
    Real value = pin->GetReal("problem", legacy_name);
    pin->GetOrAddReal("problem", new_name, value);
    if (Globals::my_rank == 0) {
      std::cout << "### NOTE: <problem>/" << legacy_name
                << " is deprecated; use " << new_name << " = " << value << "\n";
    }
    return value;
  }
  return pin->GetOrAddReal("problem", new_name, default_value);
}

inline Real UnnormalizedGaussian(Real r, Real centre, Real width) {
  return std::exp(-0.5*SQR((r-centre)/width));
}

inline Real UnnormalizedSink(Real r, Real rout) {
  return std::exp((r-rout)/sink_width);
}

Real HistoryDiagnostics(MeshBlock *pmb, int iout) {
  AthenaArray<Real> vol(pmb->ncells1);

  if (iout == kThermalEnergy) {
    Real result = 0.0;
    for (int k=pmb->ks; k<=pmb->ke; ++k) {
      for (int j=pmb->js; j<=pmb->je; ++j) {
        pmb->pcoord->CellVolume(k, j, pmb->is, pmb->ie, vol);
        for (int i=pmb->is; i<=pmb->ie; ++i) {
          Real den = pmb->phydro->u(IDN,k,j,i);
          Real kinetic = 0.5/den*(SQR(pmb->phydro->u(IM1,k,j,i))
                                  + SQR(pmb->phydro->u(IM2,k,j,i))
                                  + SQR(pmb->phydro->u(IM3,k,j,i)));
          Real magnetic = 0.5*(SQR(pmb->pfield->bcc(IB1,k,j,i))
                               + SQR(pmb->pfield->bcc(IB2,k,j,i))
                               + SQR(pmb->pfield->bcc(IB3,k,j,i)));
          result += vol(i)*(pmb->phydro->u(IEN,k,j,i)-kinetic-magnetic);
        }
      }
    }
    return result;
  }

  if (iout >= kHistMassSourceRate && iout <= kHistTotalEnergySinkRate) {
    int block_index = iout-kHistMassSourceRate;
    return pmb->ruser_meshblock_data[0](block_index);
  }

  if (iout >= kPressureSourceNormalization && iout <= kSinkNormalization) {
    Real result = 0.0;
    Real rout = pmb->pmy_mesh->mesh_size.x1max;
    for (int k=pmb->ks; k<=pmb->ke; ++k) {
      for (int j=pmb->js; j<=pmb->je; ++j) {
        pmb->pcoord->CellVolume(k, j, pmb->is, pmb->ie, vol);
        for (int i=pmb->is; i<=pmb->ie; ++i) {
          Real r = pmb->pcoord->x1v(i);
          if (iout == kPressureSourceNormalization) {
            result += vol(i)*UnnormalizedGaussian(r, r_p, sigma_p)/source_integral_p;
          } else if (iout == kDensitySourceNormalization) {
            result += vol(i)*UnnormalizedGaussian(r, r_rho, sigma_rho)
                      /source_integral_rho;
          } else {
            result += vol(i)*UnnormalizedSink(r, rout)/sink_integral;
          }
        }
      }
    }
    return result;
  }

  if (iout >= kMassFluxProbe1 && iout <= kMassFluxProbe3) {
    int nprobe = iout-kMassFluxProbe1;
    Real result = 0.0;
    for (int k=pmb->ks; k<=pmb->ke; ++k) {
      for (int j=pmb->js; j<=pmb->je; ++j) {
        pmb->pcoord->CellVolume(k, j, pmb->is, pmb->ie, vol);
        for (int i=pmb->is; i<=pmb->ie; ++i) {
          bool in_shell = (probe_radius[nprobe] >= pmb->pcoord->x1f(i)
                           && probe_radius[nprobe] < pmb->pcoord->x1f(i+1));
          if (in_shell) {
            result += vol(i)*pmb->phydro->u(IM1,k,j,i)/probe_volume[nprobe];
          }
        }
      }
    }
    return result;
  }

  if (iout >= kBoundaryMassFlux && iout <= kBoundaryScalar2Flux) {
    if (pmb->pmy_mesh->time <= pmb->pmy_mesh->start_time) return 0.0;
    Real result = 0.0;
    Real xmin = pmb->pmy_mesh->mesh_size.x1min;
    Real xmax = pmb->pmy_mesh->mesh_size.x1max;
    Real scale = std::max(std::abs(xmin), std::abs(xmax));
    Real tol = 32.0*std::numeric_limits<Real>::epsilon()*std::max(scale, 1.0);
    bool at_inner = std::abs(pmb->pcoord->x1f(pmb->is)-xmin) < tol;
    bool at_outer = std::abs(pmb->pcoord->x1f(pmb->ie+1)-xmax) < tol;
    for (int k=pmb->ks; k<=pmb->ke; ++k) {
      for (int j=pmb->js; j<=pmb->je; ++j) {
        if (at_inner) {
          Real area = pmb->pcoord->GetFace1Area(k,j,pmb->is);
          if (iout == kBoundaryMassFlux) {
            result -= area*pmb->phydro->flux[X1DIR](IDN,k,j,pmb->is);
          } else if (iout == kBoundaryEnergyFlux) {
            result -= area*pmb->phydro->flux[X1DIR](IEN,k,j,pmb->is);
          } else {
            int scalar = iout-kBoundaryScalar1Flux;
            result -= area*pmb->pscalars->s_flux[X1DIR](scalar,k,j,pmb->is);
          }
        }
        if (at_outer) {
          Real area = pmb->pcoord->GetFace1Area(k,j,pmb->ie+1);
          if (iout == kBoundaryMassFlux) {
            result += area*pmb->phydro->flux[X1DIR](IDN,k,j,pmb->ie+1);
          } else if (iout == kBoundaryEnergyFlux) {
            result += area*pmb->phydro->flux[X1DIR](IEN,k,j,pmb->ie+1);
          } else {
            int scalar = iout-kBoundaryScalar1Flux;
            result += area*pmb->pscalars->s_flux[X1DIR](scalar,k,j,pmb->ie+1);
          }
        }
      }
    }
    return result;
  }

  return 0.0;
}

} // namespace

//----------------------------------------------------------------------------------------
// Function: Mesh::InitUserMeshData
void Mesh::InitUserMeshData(ParameterInput *pin) {
  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") != 0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in z_pinch_fueled: cylindrical coordinates are required\n";
    ATHENA_ERROR(msg);
  }
  if (!MAGNETIC_FIELDS_ENABLED || !NON_BAROTROPIC_EOS) {
    std::stringstream msg;
    msg << "### FATAL ERROR in z_pinch_fueled: adiabatic MHD is required\n";
    ATHENA_ERROR(msg);
  }
  if (NSCALARS != 2) {
    std::stringstream msg;
    msg << "### FATAL ERROR in z_pinch_fueled: configure with --nscalars=2; got "
        << NSCALARS << "\n";
    ATHENA_ERROR(msg);
  }
  if (multilevel) {
    std::stringstream msg;
    msg << "### FATAL ERROR in z_pinch_fueled: SMR/AMR is outside this pgen's "
        << "discrete-normalization contract\n";
    ATHENA_ERROR(msg);
  }

  bin      = pin->GetReal("problem", "bin");
  rin      = pin->GetReal("mesh", "x1min");
  presmin  = pin->GetOrAddReal("problem", "presmin", 0.1);
  densmin  = pin->GetOrAddReal("problem", "densmin", 0.1);
  beta     = pin->GetOrAddReal("problem", "beta", 0.5);

  r_p       = GetNewOrLegacyReal(pin, "r_p", "input_location", 1.0);
  sigma_p   = GetNewOrLegacyReal(pin, "sigma_p", "input_gaussian_sigma", 0.2);
  r_rho     = GetNewOrLegacyReal(pin, "r_rho", "input_location_dens", 2.3);
  sigma_rho = GetNewOrLegacyReal(pin, "sigma_rho", "sigma_dens", 0.15);
  sink_width = pin->GetOrAddReal("problem", "sink_width", 0.3);

  injection_rate_dens  = pin->GetOrAddReal("problem", "injection_rate_dens", 0.1);
  injection_rate_pres  = pin->GetOrAddReal("problem", "injection_rate_pres", 0.1);
  injection_rate_scalar1 = pin->GetOrAddReal("problem", "injection_rate_scalar1", 1.e-4);
  injection_rate_scalar2 = pin->GetOrAddReal("problem", "injection_rate_scalar2", 1.e-4);
  sink_rate_dens     = pin->GetOrAddReal("problem", "sink_rate_dens", 0.1);
  sink_rate_energy   = pin->GetOrAddReal("problem", "sink_rate_energy", 0.1);

  d0       = pin->GetOrAddReal("problem", "d0",      1.0);
  rpeak    = pin->GetReal          ("problem", "rpeak");
  expdens  = pin->GetReal          ("problem", "expdens");
  exppres  = pin->GetOrAddReal     ("problem", "exppres", 1.0);
  r_const  = pin->GetOrAddReal     ("problem", "r_const", 2.0);
  rcSmooth = pin->GetOrAddReal     ("problem", "rcSmooth", 5.0);

  probe_radius[0] = pin->GetOrAddReal("problem", "mass_flux_probe1", 1.0);
  probe_radius[1] = pin->GetOrAddReal("problem", "mass_flux_probe2", 1.8);
  probe_radius[2] = pin->GetOrAddReal("problem", "mass_flux_probe3", 2.4);

  Real rout = mesh_size.x1max;
  if (sigma_p <= 0.0 || sigma_rho <= 0.0 || sink_width <= 0.0
      || r_p < rin || r_p > rout || r_rho < rin || r_rho > rout) {
    std::stringstream msg;
    msg << "### FATAL ERROR in z_pinch_fueled: invalid source/sink location or width\n";
    ATHENA_ERROR(msg);
  }
  for (int n=0; n<3; ++n) {
    if (probe_radius[n] < rin || probe_radius[n] >= rout) {
      std::stringstream msg;
      msg << "### FATAL ERROR in z_pinch_fueled: mass_flux_probe" << n+1
          << " must lie in [x1min,x1max)\n";
      ATHENA_ERROR(msg);
    }
    probe_volume[n] = 0.0;
  }

  // Compute exact root-grid cylindrical shell volumes and volume-centred radii.  This
  // duplicates Cylindrical's geometry formulas without needing MeshBlocks (which do not
  // exist yet here), and is safe before the OpenMP task list begins.
  source_integral_p = source_integral_rho = sink_integral = 0.0;
  dens_init = pres_init = total_volume = 0.0;
  Real dphi_dz = (mesh_size.x2max-mesh_size.x2min)
                 *(mesh_size.x3max-mesh_size.x3min);
  Real beta_correct = beta/(1.0+std::exp(-rcSmooth*(r_const-rpeak)))
                      + beta*std::pow(1.0+SQR(r_const-rpeak), -exppres/2.0)
                        /(1.0+std::exp(-rcSmooth*(rpeak-r_const)));
  for (int i=0; i<mesh_size.nx1; ++i) {
    Real xl = ComputeMeshGeneratorX(i, mesh_size.nx1,
                                    use_uniform_meshgen_fn_[X1DIR]);
    Real xr = ComputeMeshGeneratorX(i+1, mesh_size.nx1,
                                    use_uniform_meshgen_fn_[X1DIR]);
    Real rm = MeshGenerator_[X1DIR](xl, mesh_size);
    Real rp = MeshGenerator_[X1DIR](xr, mesh_size);
    Real r = TWO_3RD*(rp*rp*rp-rm*rm*rm)/(rp*rp-rm*rm);
    Real shell_volume = 0.5*(rp*rp-rm*rm)*dphi_dz;

    Real den, pres;
    if (r < rpeak) {
      den = 0.5*(d0+densmin)
            - 0.5*(d0-densmin)*std::cos(PI*(r-rin)/(rpeak-rin));
      pres = 0.5*(beta_correct+presmin)
             - 0.5*(beta_correct-presmin)*std::cos(PI*(r-rin)/(rpeak-rin));
    } else {
      den = d0*std::pow(1.0+SQR(r-rpeak), -expdens/2.0);
      pres = beta*std::pow(1.0+SQR(r-rpeak), -exppres/2.0)
             /(1.0+std::exp(-rcSmooth*(r_const-r)))
             + beta*std::pow(1.0+SQR(r_const-rpeak), -exppres/2.0)
               /(1.0+std::exp(-rcSmooth*(r-r_const)));
    }
    dens_init += den*shell_volume;
    pres_init += pres*shell_volume;
    total_volume += shell_volume;
    source_integral_p += UnnormalizedGaussian(r, r_p, sigma_p)*shell_volume;
    source_integral_rho += UnnormalizedGaussian(r, r_rho, sigma_rho)*shell_volume;
    sink_integral += UnnormalizedSink(r, rout)*shell_volume;
    for (int n=0; n<3; ++n) {
      if (probe_radius[n] >= rm && probe_radius[n] < rp) {
        probe_volume[n] = shell_volume;
      }
    }
  }

  if (source_integral_p <= 0.0 || source_integral_rho <= 0.0 || sink_integral <= 0.0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in z_pinch_fueled: zero discrete source/sink integral\n";
    ATHENA_ERROR(msg);
  }

  if (mesh_bcs[BoundaryFace::inner_x1] == GetBoundaryFlag("user"))
    EnrollUserBoundaryFunction(BoundaryFace::inner_x1, InnerX1HardWall);
  if (mesh_bcs[BoundaryFace::outer_x1] == GetBoundaryFlag("user"))
    EnrollUserBoundaryFunction(BoundaryFace::outer_x1, OuterX1HardWall);

  forcing_flag = pin->GetOrAddInteger("problem", "forcing_flag", 0);
  if (forcing_flag == 1) {
    EnrollUserExplicitSourceFunction(ApplyFueling);
  }

  AllocateUserHistoryOutput(kNumHistoryDiagnostics);
  EnrollUserHistoryOutput(kThermalEnergy, HistoryDiagnostics, "thermal-E");
  EnrollUserHistoryOutput(kHistMassSourceRate, HistoryDiagnostics, "mass_src");
  EnrollUserHistoryOutput(kHistThermalSourceRate, HistoryDiagnostics, "Eth_src");
  EnrollUserHistoryOutput(kHistScalar1SourceRate, HistoryDiagnostics, "ash_src");
  EnrollUserHistoryOutput(kHistScalar2SourceRate, HistoryDiagnostics, "imp_src");
  EnrollUserHistoryOutput(kHistMassSinkRate, HistoryDiagnostics, "mass_sink");
  EnrollUserHistoryOutput(kHistThermalSinkRate, HistoryDiagnostics, "Eth_sink");
  EnrollUserHistoryOutput(kHistScalar1SinkRate, HistoryDiagnostics, "ash_sink");
  EnrollUserHistoryOutput(kHistScalar2SinkRate, HistoryDiagnostics, "imp_sink");
  EnrollUserHistoryOutput(kHistTotalEnergySourceRate, HistoryDiagnostics, "Etot_src");
  EnrollUserHistoryOutput(kHistTotalEnergySinkRate, HistoryDiagnostics, "Etot_sink");
  EnrollUserHistoryOutput(kPressureSourceNormalization, HistoryDiagnostics, "Gp_int");
  EnrollUserHistoryOutput(kDensitySourceNormalization, HistoryDiagnostics, "Grho_int");
  EnrollUserHistoryOutput(kSinkNormalization, HistoryDiagnostics, "W_int");
  EnrollUserHistoryOutput(kMassFluxProbe1, HistoryDiagnostics, "rho_ur_p1");
  EnrollUserHistoryOutput(kMassFluxProbe2, HistoryDiagnostics, "rho_ur_p2");
  EnrollUserHistoryOutput(kMassFluxProbe3, HistoryDiagnostics, "rho_ur_p3");
  EnrollUserHistoryOutput(kBoundaryMassFlux, HistoryDiagnostics, "mass_bnd_out");
  EnrollUserHistoryOutput(kBoundaryEnergyFlux, HistoryDiagnostics, "Etot_bnd_out");
  EnrollUserHistoryOutput(kBoundaryScalar1Flux, HistoryDiagnostics, "ash_bnd_out");
  EnrollUserHistoryOutput(kBoundaryScalar2Flux, HistoryDiagnostics, "imp_bnd_out");

  if (Globals::my_rank == 0) {
    std::cout << "z_pinch_fueled discrete setup:\n"
              << "  M0=" << dens_init << "  integral(p0)=" << pres_init
              << "  volume=" << total_volume << "\n"
              << "  r_p=" << r_p << " sigma_p=" << sigma_p
              << "  integral(g_p)=" << source_integral_p << "\n"
              << "  r_rho=" << r_rho << " sigma_rho=" << sigma_rho
              << "  integral(g_rho)=" << source_integral_rho << "\n"
              << "  sink_width=" << sink_width
              << "  integral(w)=" << sink_integral << "\n"
              << "  analytic rates: mass=" << injection_rate_dens*dens_init
              << " Eth=" << injection_rate_pres*pres_init
                                 /(pin->GetReal("hydro", "gamma")-1.0)
              << " ash=" << injection_rate_scalar1*dens_init
              << " impurity=" << injection_rate_scalar2*dens_init << "\n";
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshBlock::InitUserMeshBlockData(ParameterInput *pin)
void MeshBlock::InitUserMeshBlockData(ParameterInput *pin) {
  AllocateRealUserMeshBlockDataField(1);
  ruser_meshblock_data[0].NewAthenaArray(kNumBlockDiagnostics);
  for (int n=0; n<kNumBlockDiagnostics; ++n) ruser_meshblock_data[0](n) = 0.0;
}

//----------------------------------------------------------------------------------------
// Function: trapezoidalIntegration1
double trapezoidalIntegration1(double r, int N, double p0, double presmin,
                                double exppres, double rpeak, double rin) {
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
                                double exppres, double rcSmooth, double r_const) {
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

  for (int m = 1; m < N; ++m) {
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
  Real gamma = peos->GetGamma();
  Real gm1   = gamma - 1.0;
  int N      = pin->GetOrAddInteger("problem", "N", 1000);
  rin  = pmy_mesh->mesh_size.x1min;
  Real rout  = pmy_mesh->mesh_size.x1max;
  Real Lz    = pmy_mesh->mesh_size.x3max - pmy_mesh->mesh_size.x3min;
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
  if (!MAGNETIC_FIELDS_ENABLED) {
    std::stringstream msg;
    msg << "### FATAL ERROR in z_pinch.cpp ProblemGenerator" << std::endl
        << "Magnetic fields not enabled" << std::endl;
    ATHENA_ERROR(msg);
  }

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
        pscalars->s(0,k,j,i) = 0.0;
        pscalars->s(1,k,j,i) = 0.0;
      }
    }
  }

  Real result;
  for (int i=is; i<=ie; ++i) {
    Real r = pcoord->x1v(i);
    if (r < rpeak) {
      result = trapezoidalIntegration1(r, N, beta_correct, presmin, exppres, rpeak, rin);
    } else {
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
  // User hydro boundaries do not automatically fill passive-scalar ghosts in this
  // Athena++ version.  Continue concentration with zero gradient; Athena++ converts it
  // back to conserved scalar mass using the boundary density after this callback.
  for (int n=0; n<NSCALARS; ++n) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
        for (int i=1; i<=ngh; ++i) {
          pmb->pscalars->r(n,k,j,il-i) = pmb->pscalars->r(n,k,j,il);
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
// Continue the marginal density and pressure power laws and B_theta proportional to 1/r.
void OuterX1HardWall(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                     FaceField &b, Real time, Real dt,
                     int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  for (int n=0; n<(NHYDRO); ++n) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
        if (n==(IDN)) {
#pragma omp simd
          for (int i=1; i<=ngh; ++i) {
            Real r     = pco->x1v(i+iu);
            Real r_out = pco->x1v(iu);
            prim(n,k,j,iu+i) = prim(n,k,j,(iu)) * pow(r/r_out,-2.);
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

  for (int n=0; n<NSCALARS; ++n) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
        for (int i=1; i<=ngh; ++i) {
          pmb->pscalars->r(n,k,j,iu+i) = pmb->pscalars->r(n,k,j,iu);
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
//! \fn void ApplyFueling(...)
//! \brief Apply fixed-rate particle/heat/tracer sources, then the post-source edge sink.
void ApplyFueling(MeshBlock *pmb, const Real time, const Real dt,
                  const AthenaArray<Real> &prim,
                  const AthenaArray<Real> &prim_scalar,
                  const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
                  AthenaArray<Real> &cons_scalar) {
  (void)time;
  (void)prim;
  (void)prim_scalar;
  if (dt <= 0.0) return;

  const Real gm1 = pmb->peos->GetGamma()-1.0;
  const Real rout = pmb->pmy_mesh->mesh_size.x1max;
  AthenaArray<Real> vol(pmb->ncells1);

  Real mass_added = 0.0;
  Real thermal_added = 0.0;
  Real scalar1_added = 0.0;
  Real scalar2_added = 0.0;
  Real total_energy_added = 0.0;

  // Source step.  Momentum and magnetic field are unchanged, corresponding to material
  // injected at rest.  The two tracer sources are independent conserved scalar masses.
  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    for (int j=pmb->js; j<=pmb->je; ++j) {
      pmb->pcoord->CellVolume(k, j, pmb->is, pmb->ie, vol);
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        Real r = pmb->pcoord->x1v(i);
        Real g_p = UnnormalizedGaussian(r, r_p, sigma_p)/source_integral_p;
        Real g_rho = UnnormalizedGaussian(r, r_rho, sigma_rho)/source_integral_rho;
        Real dmass = dt*injection_rate_dens*dens_init*g_rho;
        Real dpressure = dt*injection_rate_pres*pres_init*g_p;
        Real dscalar1 = dt*injection_rate_scalar1*dens_init*g_p;
        Real dscalar2 = dt*injection_rate_scalar2*dens_init*g_rho;

        Real den_old = cons(IDN,k,j,i);
        Real energy_old = cons(IEN,k,j,i);
        Real momentum2 = SQR(cons(IM1,k,j,i))+SQR(cons(IM2,k,j,i))
                         + SQR(cons(IM3,k,j,i));
        Real kinetic_old = 0.5*momentum2/den_old;
        Real magnetic = 0.5*(SQR(bcc(IB1,k,j,i))+SQR(bcc(IB2,k,j,i))
                             + SQR(bcc(IB3,k,j,i)));
        Real thermal_old = energy_old-kinetic_old-magnetic;

        cons(IDN,k,j,i) = den_old+dmass;
        cons_scalar(0,k,j,i) += dscalar1;
        cons_scalar(1,k,j,i) += dscalar2;
        Real kinetic_new = 0.5*momentum2/cons(IDN,k,j,i);
        Real thermal_new = thermal_old+dpressure/gm1;
        cons(IEN,k,j,i) = kinetic_new+magnetic+thermal_new;

        mass_added += dmass*vol(i);
        thermal_added += dpressure/gm1*vol(i);
        scalar1_added += dscalar1*vol(i);
        scalar2_added += dscalar2*vol(i);
        total_energy_added += (cons(IEN,k,j,i)-energy_old)*vol(i);
      }
    }
  }

  AthenaArray<Real> &diag = pmb->ruser_meshblock_data[0];
  diag(kMassSourceRate) = mass_added/dt;
  diag(kThermalSourceRate) = thermal_added/dt;
  diag(kScalar1SourceRate) = scalar1_added/dt;
  diag(kScalar2SourceRate) = scalar2_added/dt;
  diag(kTotalEnergySourceRate) = total_energy_added/dt;

  Real mass_removed = 0.0;
  Real thermal_removed = 0.0;
  Real scalar1_removed = 0.0;
  Real scalar2_removed = 0.0;
  Real total_energy_removed = 0.0;

  // Sink step.  Scalars receive exactly the density reduction factor, so concentration
  // is unchanged by local removal and tracer mass cannot accumulate artificially.
  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    for (int j=pmb->js; j<=pmb->je; ++j) {
      pmb->pcoord->CellVolume(k, j, pmb->is, pmb->ie, vol);
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        Real r = pmb->pcoord->x1v(i);
        Real w = UnnormalizedSink(r, rout)/sink_integral;
        Real density_factor = 1.0-dt*sink_rate_dens*w;
        Real thermal_factor = 1.0-dt*sink_rate_energy*w;
        if (density_factor <= 0.0 || thermal_factor <= 0.0) {
          std::stringstream msg;
          msg << "### FATAL ERROR in z_pinch_fueled sink: negative reduction factor at r="
              << r << "; reduce sink rate or timestep\n";
          ATHENA_ERROR(msg);
        }

        Real den_old = cons(IDN,k,j,i);
        Real scalar1_old = cons_scalar(0,k,j,i);
        Real scalar2_old = cons_scalar(1,k,j,i);
        Real energy_old = cons(IEN,k,j,i);
        Real momentum2 = SQR(cons(IM1,k,j,i))+SQR(cons(IM2,k,j,i))
                         + SQR(cons(IM3,k,j,i));
        Real kinetic_old = 0.5*momentum2/den_old;
        Real magnetic = 0.5*(SQR(bcc(IB1,k,j,i))+SQR(bcc(IB2,k,j,i))
                             + SQR(bcc(IB3,k,j,i)));
        Real thermal_old = energy_old-kinetic_old-magnetic;

        cons(IDN,k,j,i) = den_old*density_factor;
        cons_scalar(0,k,j,i) = scalar1_old*density_factor;
        cons_scalar(1,k,j,i) = scalar2_old*density_factor;
        Real kinetic_new = 0.5*momentum2/cons(IDN,k,j,i);
        Real thermal_new = thermal_old*thermal_factor;
        cons(IEN,k,j,i) = kinetic_new+magnetic+thermal_new;

        mass_removed += (den_old-cons(IDN,k,j,i))*vol(i);
        scalar1_removed += (scalar1_old-cons_scalar(0,k,j,i))*vol(i);
        scalar2_removed += (scalar2_old-cons_scalar(1,k,j,i))*vol(i);
        thermal_removed += (thermal_old-thermal_new)*vol(i);
        total_energy_removed += (energy_old-cons(IEN,k,j,i))*vol(i);
      }
    }
  }

  diag(kMassSinkRate) = mass_removed/dt;
  diag(kThermalSinkRate) = thermal_removed/dt;
  diag(kScalar1SinkRate) = scalar1_removed/dt;
  diag(kScalar2SinkRate) = scalar2_removed/dt;
  diag(kTotalEnergySinkRate) = total_energy_removed/dt;
}
