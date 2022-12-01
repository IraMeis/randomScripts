# mpiexec -n 4 python block_t.py

from mpi4py import MPI
import numpy as np


def get_send_params(dim: int, n_proc: int):
    count_1 = int(dim / n_proc) + 1
    count_2 = int(dim / n_proc)
    last_with_count_1 = dim % n_proc - 1
    return count_1, last_with_count_1, count_2


def get_test_mm_matrix(m: int):
    return np.reshape(np.arange(m*m, dtype=np.float_), (m, m))


def mpi_func():
    # m = m
    # n = n

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    dim = 7
    am1, last1, am2 = get_send_params(dim, size)

    if rank == 0:
        matrixOrig = get_mm_matrix(dim)
        print(matrixOrig)
        print('rank: ', rank)
        print('data: ', np.reshape(np.copy(matrixOrig[0: am1]), (dim * am1)))

        for i in range(1, last1 + 1):
            comm.Send(np.reshape(np.copy(matrixOrig[i * am1: am1 + i * am1]), (dim * am1)), dest=i)
        
        for i in range(last1 + 1, size):
            comm.Send(np.reshape(np.copy(matrixOrig[(last1 + 1) * am1 + (i - last1 - 1) * am2: 
                                                    (last1 + 1) * am1 + am2 + (i - last1 - 1) * am2]), 
                                 (dim * am2)), dest=i)

    else:
        data = np.empty(am1 * dim if rank <= last1 else am2 * dim, dtype=np.float_)
        comm.Recv(data, source=0)
        print('rank: ', rank)
        print('data: ', data)
        
        
if __name__ in "__main__":
    mpi_func()
