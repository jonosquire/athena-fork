//======================================================================================
// Athena++ astrophysical MHD code
// Copyright (C) 2014 James M. Stone  <jmstone@princeton.edu>
//
// This program is free software: you can redistribute and/or modify it under the terms
// of the GNU General Public License (GPL) as published by the Free Software Foundation,
// either version 3 of the License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A 
// PARTICULAR PURPOSE.  See the GNU General Public License for more details.
//
// You should have received a copy of GNU GPL in the file LICENSE included in the code
// distribution.  If not see <http://www.gnu.org/licenses/>.
//======================================================================================
//! \file uniform_chem.cpp
//  \brief problem generator, uniform mesh with chemistry
//======================================================================================

// C++ headers
#include <string>     // c_str()
#include <iostream>   // endl
#include <vector>     // vector container
#include <sstream>    // stringstream
#include <stdio.h>    // c style file
#include <string.h>   // strcmp()
#include <algorithm>  // std::find()
#include <stdexcept>  // std::runtime_error()

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../parameter_input.hpp"
#include "../mesh/mesh.hpp"
#include "../hydro/hydro.hpp"
#include "../globals.hpp"
#include "../scalars/scalars.hpp"
#include "../chemistry/utils/thermo.hpp"
#include "../radiation/radiation.hpp"
#include "../radiation/integrators/rad_integrators.hpp"
#include "../field/field.hpp"
#include "../eos/eos.hpp"
#include "../coordinates/coordinates.hpp"

Real threshold;
int RefinementCondition(MeshBlock *pmb);

//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) {
  if (adaptive) {
    EnrollUserRefinementCondition(RefinementCondition);
    threshold = pin->GetReal("problem", "thr");
  }
  return;
}

//======================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief initialize problem by reading in vtk file.
//======================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin)
{
  //dimensions of meshblock
  const int Nx = ie - is + 1;
  const int Ny = je - js + 1;
  const int Nz = ke - ks + 1;
	//read input parameters
	const Real nH = pin->GetReal("problem", "nH"); //density
	const Real vx = pin->GetOrAddReal("problem", "vx", 0); //velocity x
  //mean and std of the initial gaussian profile
	const Real gaussian_mean = pin->GetOrAddReal("problem", "gaussian_mean", 0.5);
	const Real gaussian_std = pin->GetOrAddReal("problem", "gaussian_std", 0.1);
  const Real iso_cs = pin->GetReal("hydro", "iso_sound_speed");
  const Real pres = nH*SQR(iso_cs);
  const Real gm1  = peos->GetGamma() - 1.0;

	for (int k=ks; k<=ke; ++k) {
		for (int j=js; j<=je; ++j) {
			for (int i=is; i<=ie; ++i) {
        //density
				phydro->u(IDN, k, j, i) = nH;
        //velocity, x direction
				phydro->u(IM1, k, j, i) = nH*vx;
        //energy
        if (NON_BAROTROPIC_EOS) {
          phydro->u(IEN, k, j, i) = pres/gm1 + 0.5*nH*SQR(vx);
        }
			}
		}
	}

	//intialize chemical species
  if (NSCALARS > 0) {
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie; ++i) {
          for (int ispec=0; ispec < NSCALARS; ++ispec) {
#ifdef INCLUDE_CHEMISTRY
            Real x1 = pcoord->x1v(i);
            //gaussian initial H abundance in [0, 1), and no H in [1, 2]
            if (x1 <= 1) {
              pscalars->s(0, k, j, i) = exp( -SQR(x1-gaussian_mean)/(2.*SQR(gaussian_std)) )*nH; //H
              pscalars->s(1, k, j, i) = 0.5*(nH - pscalars->s(0, k, j, i)); //H2
            } else {
              pscalars->s(0, k, j, i) = 0; //H
              pscalars->s(1, k, j, i) = 0.5*nH; //H2
            }
#endif
          }
        }
      }
    }
  }

  return;
}

// refinement condition: maximum gradient of each passive scalar profile

int RefinementCondition(MeshBlock *pmb) {
  int f2 = pmb->pmy_mesh->f2, f3 = pmb->pmy_mesh->f3;
  AthenaArray<Real> &r = pmb->pscalars->r;
  Real maxeps = 0.0;
  if (f3) {
    for (int n=0; n<NSCALARS; ++n) {
      for (int k=pmb->ks-1; k<=pmb->ke+1; k++) {
        for (int j=pmb->js-1; j<=pmb->je+1; j++) {
          for (int i=pmb->is-1; i<=pmb->ie+1; i++) {
            Real eps = std::sqrt(SQR(0.5*(r(n,k,j,i+1) - r(n,k,j,i-1)))
                                 + SQR(0.5*(r(n,k,j+1,i) - r(n,k,j-1,i)))
                                 + SQR(0.5*(r(n,k+1,j,i) - r(n,k-1,j,i))));
            // /r(n,k,j,i); Do not normalize by scalar, since (unlike IDN and IPR) there
            // are are no physical floors / r=0 might be allowed. Compare w/ blast.cpp.
            maxeps = std::max(maxeps, eps);
          }
        }
      }
    }
  } else if (f2) {
    int k = pmb->ks;
    for (int n=0; n<NSCALARS; ++n) {
      for (int j=pmb->js-1; j<=pmb->je+1; j++) {
        for (int i=pmb->is-1; i<=pmb->ie+1; i++) {
          Real eps = std::sqrt(SQR(0.5*(r(n,k,j,i+1) - r(n,k,j,i-1)))
                               + SQR(0.5*(r(n,k,j+1,i) - r(n,k,j-1,i)))); // /r(n,k,j,i);
          maxeps = std::max(maxeps, eps);
        }
      }
    }
  } else {
    return 0;
  }

  if (maxeps > threshold) return 1;
  if (maxeps < 0.25*threshold) return -1;
  return 0;
}
//========================================================================================
//! \fn void Mesh::UserWorkAfterLoop(ParameterInput *pin)
//========================================================================================

void Mesh::UserWorkAfterLoop(ParameterInput *pin) {
  if (!pin->GetOrAddBoolean("problem", "compute_error", false)) return;

	//read input parameters
	const Real nH = pin->GetReal("problem", "nH"); //density
	const Real vx = pin->GetOrAddReal("problem", "vx", 0); //velocity x
	const Real gaussian_mean = pin->GetOrAddReal("problem", "gaussian_mean", 0.5);
	const Real gaussian_std = pin->GetOrAddReal("problem", "gaussian_std", 0.1);
	//chemistry parameters
	const Real unit_density_in_nH = pin->GetReal("chemistry", "unit_density_in_nH");
	const Real unit_length_in_cm = pin->GetReal("chemistry", "unit_length_in_cm");
	const Real unit_vel_in_cms = pin->GetReal("chemistry", "unit_vel_in_cms");
  const Real unit_time_in_s = unit_length_in_cm/unit_vel_in_cms;
	const Real xi_cr = pin->GetOrAddReal("chemistry", "xi_cr", 2e-16);
  const Real kcr = xi_cr * 3.;
  const Real kgr = 3e-17;
  const Real a1 = kcr + 2.*nH*kgr*unit_density_in_nH;
  const Real a2 = kcr;
  
  //end of the simulation time
  const Real tchem = time*unit_time_in_s;
  const Real mu = gaussian_mean + vx*time;
  const Real xg_min = vx*time;
  const Real xg_max = xg_min + 1.;
  //only compute error if the Gaussian profile did not travel outside of the
  //simulation domain at the end of the simulation
  if (xg_max > mesh_size.x1max) {
    std::stringstream msg;
    msg << "### FATAL ERROR in function Mesh::UserWorkAfterLoop"
      << std::endl << "Gaussian profile outside of the simulation domain" <<std::endl;
    ATHENA_ERROR(msg);
  }

  // Initialize errors to zero
  Real l1_err[NSCALARS]{}, max_err[NSCALARS]{}, cons_err[1]{};

  MeshBlock *pmb = pblock;
  while (pmb != nullptr) {
    int il = pmb->is, iu = pmb->ie, jl = pmb->js, ju = pmb->je,
        kl = pmb->ks, ku = pmb->ke;
    //  Compute errors at cell centers
    for (int k=kl; k<=ku; k++) {
      for (int j=jl; j<=ju; j++) {
        for (int i=il; i<=iu; i++) {
          Real x = pmb->pcoord->x1v(i);
          Real fH0 = 0;
          if ( (x < xg_min) || (x > xg_max) ) {
            fH0 = 0;
          } else {
            fH0 = exp( -SQR(x-mu)/(2.*SQR(gaussian_std)) );
          }
          Real fH = (fH0 - a2/a1)*exp(-a1*tchem) + a2/a1;
          Real fH2 = 0.5*(1. - fH);
          // Weight l1 error by cell volume
          Real vol = pmb->pcoord->GetCellVolume(k, j, i);
          l1_err[0] += std::abs(fH - pmb->pscalars->r(0,k,j,i))*vol;
          max_err[0] = std::max(
              static_cast<Real>(std::abs(fH - pmb->pscalars->r(0,k,j,i))),
              max_err[0]);
          l1_err[1] += std::abs(fH2 - pmb->pscalars->r(1,k,j,i))*vol;
          max_err[1] = std::max(
              static_cast<Real>(std::abs(fH2 - pmb->pscalars->r(1,k,j,i))),
              max_err[1]);
          cons_err[0] += std::abs(pmb->pscalars->r(0,k,j,i) +
                                  2*pmb->pscalars->r(1,k,j,i) - 1.)*vol;
        }
      }
    }
    pmb = pmb->next;
  }

#ifdef MPI_PARALLEL
  if (Globals::my_rank == 0) {
    MPI_Reduce(MPI_IN_PLACE, &l1_err, NSCALARS, MPI_ATHENA_REAL, MPI_SUM, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, &max_err, NSCALARS, MPI_ATHENA_REAL, MPI_MAX, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, &cons_err, 1, MPI_ATHENA_REAL, MPI_SUM, 0,
               MPI_COMM_WORLD);
  } else {
    MPI_Reduce(&l1_err, &l1_err, NSCALARS, MPI_ATHENA_REAL, MPI_SUM, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(&max_err, &max_err, NSCALARS, MPI_ATHENA_REAL, MPI_MAX, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(&cons_err, &cons_err, 1, MPI_ATHENA_REAL, MPI_SUM, 0,
               MPI_COMM_WORLD);
  }
#endif

  // only the root process outputs the data
  if (Globals::my_rank == 0) {
    // normalize errors by number of cells
    Real vol= (mesh_size.x1max - mesh_size.x1min)*(mesh_size.x2max - mesh_size.x2min)
              *(mesh_size.x3max - mesh_size.x3min);
    for (int i=0; i<NSCALARS; ++i) {
      l1_err[i] = l1_err[i]/vol;
      cons_err[i] = cons_err[i]/vol;
    }

    // open output file and write out errors
    std::stringstream msg;
    std::string fname;
    fname.assign("chem_H2-errors.dat");
    FILE *pfile;

    // The file exists -- reopen the file in append mode
    if ((pfile = std::fopen(fname.c_str(), "r")) != nullptr) {
      if ((pfile = std::freopen(fname.c_str(), "a", pfile)) == nullptr) {
        msg << "### FATAL ERROR in function Mesh::UserWorkAfterLoop"
            << std::endl << "Error output file could not be opened" <<std::endl;
        ATHENA_ERROR(msg);
      }

      // The file does not exist -- open the file in write mode and add headers
    } else {
      if ((pfile = std::fopen(fname.c_str(), "w")) == nullptr) {
        msg << "### FATAL ERROR in function Mesh::UserWorkAfterLoop"
            << std::endl << "Error output file could not be opened" <<std::endl;
        ATHENA_ERROR(msg);
      }
      std::fprintf(pfile, "# Nx1  Nx2  Nx3  Ncycle  ");
      for (int n=0; n<NSCALARS; ++n)
        std::fprintf(pfile, "r%d_L1  ", n);
      for (int n=0; n<NSCALARS; ++n)
        std::fprintf(pfile, "r%d_max  ", n);
      std::fprintf(pfile, "cons_L1  \n");
    }

    // write errors
    std::fprintf(pfile, "%d  %d", mesh_size.nx1, mesh_size.nx2);
    std::fprintf(pfile, "  %d  %d", mesh_size.nx3, ncycle);
    for (int n=0; n<NSCALARS; ++n)
      std::fprintf(pfile, "  %e", l1_err[n]);
    for (int n=0; n<NSCALARS; ++n)
      std::fprintf(pfile, "  %e", max_err[n]);
    std::fprintf(pfile, "  %e", cons_err[0]);
    std::fprintf(pfile, "\n");
    std::fclose(pfile);
  }
  return;
}
