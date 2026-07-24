#ifndef PGEN_Z_PINCH_FROM_ARRAY_READER_HPP_
#define PGEN_Z_PINCH_FROM_ARRAY_READER_HPP_

// Optional HDF5 initial-condition reader for z_pinch_heated.cpp.
//
// Include this header from the production pgen and call
// InitializeZPinchFromArrayIfRequested(this, pin) at the start of
// MeshBlock::ProblemGenerator.  If <problem>/input_filename is absent, the
// function returns false and the original analytic initialization is unchanged.

#include <algorithm>
#include <iostream>
#include <string>

#include "../globals.hpp"
#include "../inputs/hdf5_reader.hpp"

namespace zpinch_from_array {

inline void ReadBlock(MeshBlock *pmb, const std::string &input_filename,
                      const std::string &dataset_cons,
                      const std::string &dataset_scalars,
                      const std::string &dataset_b1,
                      const std::string &dataset_b2,
                      const std::string &dataset_b3,
                      int file_block, bool noop) {
  int start_cons_file[5] = {0, file_block, 0, 0, 0};
  int count_cons_file[5] = {
      1, 1, pmb->block_size.nx3, pmb->block_size.nx2, pmb->block_size.nx1};
  int start_cons_mem[4] = {0, pmb->ks, pmb->js, pmb->is};
  int count_cons_mem[4] = {
      1, pmb->block_size.nx3, pmb->block_size.nx2, pmb->block_size.nx1};
  for (int n=0; n<NHYDRO; ++n) {
    start_cons_file[0] = n;
    start_cons_mem[0] = n;
    HDF5ReadRealArray(input_filename.c_str(), dataset_cons.c_str(), 5,
                      start_cons_file, count_cons_file, 4,
                      start_cons_mem, count_cons_mem, pmb->phydro->u,
                      true, noop);
  }

  if (NSCALARS > 0) {
    int start_scalar_file[5] = {0, file_block, 0, 0, 0};
    int count_scalar_file[5] = {
        1, 1, pmb->block_size.nx3, pmb->block_size.nx2, pmb->block_size.nx1};
    int start_scalar_mem[4] = {0, pmb->ks, pmb->js, pmb->is};
    int count_scalar_mem[4] = {
        1, pmb->block_size.nx3, pmb->block_size.nx2, pmb->block_size.nx1};
    for (int n=0; n<NSCALARS; ++n) {
      start_scalar_file[0] = n;
      start_scalar_mem[0] = n;
      HDF5ReadRealArray(input_filename.c_str(), dataset_scalars.c_str(), 5,
                        start_scalar_file, count_scalar_file, 4,
                        start_scalar_mem, count_scalar_mem, pmb->pscalars->s,
                        true, noop);
    }
  }

  int start_field_file[4] = {file_block, 0, 0, 0};
  int count_field_file[4] = {
      1, pmb->block_size.nx3, pmb->block_size.nx2, pmb->block_size.nx1 + 1};
  int start_field_mem[3] = {pmb->ks, pmb->js, pmb->is};
  int count_field_mem[3] = {
      pmb->block_size.nx3, pmb->block_size.nx2, pmb->block_size.nx1 + 1};
  HDF5ReadRealArray(input_filename.c_str(), dataset_b1.c_str(), 4,
                    start_field_file, count_field_file, 3,
                    start_field_mem, count_field_mem, pmb->pfield->b.x1f,
                    true, noop);

  count_field_file[1] = pmb->block_size.nx3;
  count_field_file[2] = pmb->block_size.nx2 + 1;
  count_field_file[3] = pmb->block_size.nx1;
  count_field_mem[0] = pmb->block_size.nx3;
  count_field_mem[1] = pmb->block_size.nx2 + 1;
  count_field_mem[2] = pmb->block_size.nx1;
  HDF5ReadRealArray(input_filename.c_str(), dataset_b2.c_str(), 4,
                    start_field_file, count_field_file, 3,
                    start_field_mem, count_field_mem, pmb->pfield->b.x2f,
                    true, noop);

  count_field_file[1] = pmb->block_size.nx3 + 1;
  count_field_file[2] = pmb->block_size.nx2;
  count_field_file[3] = pmb->block_size.nx1;
  count_field_mem[0] = pmb->block_size.nx3 + 1;
  count_field_mem[1] = pmb->block_size.nx2;
  count_field_mem[2] = pmb->block_size.nx1;
  HDF5ReadRealArray(input_filename.c_str(), dataset_b3.c_str(), 4,
                    start_field_file, count_field_file, 3,
                    start_field_mem, count_field_mem, pmb->pfield->b.x3f,
                    true, noop);
}

}  // namespace zpinch_from_array

inline void InitializeZPinchFromArray(MeshBlock *pmb, ParameterInput *pin,
                                      int file_block, int no_op_reads) {
  std::string input_filename = pin->GetString("problem", "input_filename");
  std::string dataset_cons =
      pin->GetOrAddString("problem", "dataset_cons", "cons");
  std::string dataset_scalars =
      pin->GetOrAddString("problem", "dataset_scalars", "scalars");
  std::string dataset_b1 =
      pin->GetOrAddString("problem", "dataset_b1", "b1");
  std::string dataset_b2 =
      pin->GetOrAddString("problem", "dataset_b2", "b2");
  std::string dataset_b3 =
      pin->GetOrAddString("problem", "dataset_b3", "b3");

  zpinch_from_array::ReadBlock(
      pmb, input_filename, dataset_cons, dataset_scalars,
      dataset_b1, dataset_b2, dataset_b3, file_block, false);

  if (pmb->gid == 0 && Globals::my_rank == 0) {
    std::cout << "z_pinch_heated: initialized from " << input_filename << "\n";
  }

  // The calling MeshBlock member computes the shortage because Mesh's load
  // distribution is private.  Empty collective reads keep all ranks in lockstep.
  for (int block=0; block<no_op_reads; ++block) {
    zpinch_from_array::ReadBlock(
        pmb, input_filename, dataset_cons, dataset_scalars,
        dataset_b1, dataset_b2, dataset_b3, file_block, true);
  }
}

#endif  // PGEN_Z_PINCH_FROM_ARRAY_READER_HPP_
