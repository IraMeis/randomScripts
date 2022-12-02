# mpiexec -n 3 python block_t.py -dim 7
import argparse
from mpi4py import MPI
import numpy as np


def get_send_params(dim: int, n_proc: int):
    if dim % n_proc == 0:
        count_1 = count_2 = int(dim / n_proc)
        total_with_count_1 = n_proc
    else:
        count_1 = int(dim / n_proc) + 1
        count_2 = int(dim / n_proc)
        total_with_count_1 = dim % n_proc
    return count_1, total_with_count_1, count_2


def get_test_mm_matrix(m: int):
    return np.reshape(np.arange(m * m, dtype=np.float_), (m, m))


def mpi_func(dim: int = None):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    am1, total1, am2 = get_send_params(dim, size)
    str_num = am1 if rank < total1 else am2
    data = np.empty(str_num * dim, dtype=np.float_)

    if rank == 0:
        matrixOrig = get_test_mm_matrix(dim)
        data = np.copy(matrixOrig[0: am1])
        print('--- original ---\n', matrixOrig, '\n---')

        for i in range(1, total1):
            comm.Send(np.copy(matrixOrig[i * am1: am1 + i * am1]), dest=i)

        for i in range(total1, size):
            comm.Send(np.copy(matrixOrig[total1 * am1 + (i - total1) * am2:
                                         total1 * am1 + am2 + (i - total1) * am2]), dest=i)

    else:
        comm.Recv(data, source=0)

    data = np.reshape(data, (str_num, dim))
    # print('rank: ', rank, '\ndata: ', data)

    obj_np_arr = np.empty(size, dtype=object)
    added = 0
    for i in range(0, size):
        col_num = am1 if i < total1 else am2
        obj_np_arr[i] = np.copy(data[:, added: added + col_num])
        added += col_num

    for i in range(0, size):
        elem = np.transpose(obj_np_arr[i])
        if rank == i:
            obj_np_arr[i] = elem
        else:
            col_num = am1 if i < total1 else am2
            comm.Send(np.reshape(elem, (str_num * col_num)), dest=i)
        del elem

    for i in range(0, size):
        if i == rank:
            pass
        else:
            comm.Recv(obj_np_arr[i], source=i)
    
    added = 0
    for i in range(0, size):
        col_num = am1 if i < total1 else am2
        data[:, added: added + col_num] = obj_np_arr[i]
        added += col_num
        
    # print('rank: ', rank, '\ndata: ', data)
    
    if rank == 0:
        matrixOrig[0: am1] = data

        for i in range(1, total1):
            comm.Recv(matrixOrig[i * am1: am1 + i * am1], source=i)

        for i in range(total1, size):
            comm.Recv(matrixOrig[total1 * am1 + (i - total1) * am2:
                                 total1 * am1 + am2 + (i - total1) * am2], source=i)
        print('--- transposed ---\n', matrixOrig, '\n---')
    else:
        comm.Send(data, dest=0)


if __name__ in "__main__":
    parser = argparse.ArgumentParser(description="mpi line-block matrix transpose")
    parser.add_argument("-dim", default=None, type=int, help="dimension for test matrix")
    # parser.add_argument("-file", default=None, type=str, help="json file with actual matrix")
    opt = parser.parse_args()
    mpi_func(opt.dim)
