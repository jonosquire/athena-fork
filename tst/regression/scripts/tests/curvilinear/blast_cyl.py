# Regression test to check whether blast wave remains spherical in cylindrical coords

# Modules
import numpy as np
import scripts.utils.athena as athena


# Prepare Athena++
def prepare(**kwargs):
    athena.configure(
        prob='blast',
        coord='cylindrical', **kwargs)
    athena.make()


# Run Athena++
def run(**kwargs):
    arguments = ['time/ncycle_out=0', 'problem/compute_error=true']
    athena.run('hydro/athinput.blast_cyl', arguments)


# Analyze output
def analyze():
    # read data from error file
    filename = 'bin/blastwave-shape.dat'
    data = np.loadtxt(filename, ndmin=2)

    # check blast is spherical
    if data[0][3] > 1.0:
        print("Distortion of blast wave in cylindrical coords too large", data[0][3])
        return False

    return True
