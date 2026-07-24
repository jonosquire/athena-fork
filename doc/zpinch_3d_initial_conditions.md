# Z-pinch 2D-to-3D initial conditions

The `zpinch` branch can initialize the production `z_pinch_heated` problem
from a conservatively restricted and theta-extruded 2D Athena HDF5 dump. Only
`MeshBlock::ProblemGenerator` initialization changes; the normal heating,
target-mass controller and edge density source, sink, tracers, boundaries, and
history diagnostics remain active.

## Build

Configure Athena as for the driven Task2a runs, including HDF5 and two scalars:

```bash
python3 configure.py \
  --prob=z_pinch_heated \
  --coord=cylindrical \
  --eos=adiabatic \
  --flux=hlld \
  --nscalars=2 \
  --nghost=2 \
  -b -mpi -hdf5 -h5double
make clean
make -j
```

Use the same compiler and HDF5 options as the production cluster build.

## Generate the IC and athinput

The converter requires NumPy and h5py:

```bash
python3 vis/python/extrude_athdf.py \
  /path/to/relaxed.out1.athdf \
  --source-athinput /path/to/the/2d/athinput \
  --output /path/to/case.ic.h5 \
  --coarsen-r 2 \
  --coarsen-z 2 \
  --ntheta 64 \
  --block-r 32 \
  --block-theta 8 \
  --block-z 60
```

The original athinput is authoritative: its forcing, controller, sink, tracer,
boundary, output, and time parameters are copied to the generated 3D athinput.
The default added-noise amplitude is exactly zero.

The requested block sizes must divide the target grid. The IC is tied to this
MeshBlock geometry, and the generated athinput records matching dimensions.
It is not tied to an MPI rank count: Athena distributes those MeshBlocks among
the ranks at runtime, including uneven distributions.

The generated athinput sets `<problem>/input_filename` and the IC dataset
names. Without `input_filename`, `z_pinch_heated` retains its original analytic
initialization.

Run normally:

```bash
mpirun -n N ./bin/athena -i /path/to/case.athinput -d /path/to/output
```
