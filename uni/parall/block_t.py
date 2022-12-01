# mpiexec -n 3 python block_t.py -dim 7
import argparse
from mpi4py import MPI
import numpy as np


def get_send_params(dim: int, n_proc: int):
    if dim % n_proc == 0:
        count_1 = count_2 = int(dim / n_proc)
        last_with_count_1 = n_proc - 1
    else:
        count_1 = int(dim / n_proc) + 1
        count_2 = int(dim / n_proc)
        last_with_count_1 = dim % n_proc - 1
    return count_1, last_with_count_1, count_2


def get_test_mm_matrix(m: int):
    return np.reshape(np.arange(m * m, dtype=np.float_), (m, m))


def mpi_func(dim: int = None):
    # m = m
    # n = n

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    am1, last1, am2 = get_send_params(dim, size)
    str_num = am1 if rank <= last1 else am2
    data = np.empty(str_num * dim, dtype=np.float_)

    if rank == 0:
        matrixOrig = get_test_mm_matrix(dim)
        data = np.copy(matrixOrig[0: am1])
        print(matrixOrig)

        for i in range(1, last1 + 1):
            comm.Send(np.copy(matrixOrig[i * am1: am1 + i * am1]), dest=i)

        for i in range(last1 + 1, size):
            comm.Send(np.copy(matrixOrig[(last1 + 1) * am1 + (i - last1 - 1) * am2:
                                         (last1 + 1) * am1 + am2 + (i - last1 - 1) * am2]), dest=i)

    else:
        comm.Recv(data, source=0)

    data = np.reshape(data, (str_num, dim))
    print('rank: ', rank, '\ndata: ', data)

    obj_np_arr = np.empty(size, dtype=object)
    added = 0
    for i in range(0, size):
        col_num = am1 if i <= last1 else am2
        obj_np_arr[i] = np.copy(data[:, added: added + col_num])
        added += col_num

    del data

    for i in range(0, size):
        elem = np.transpose(obj_np_arr[i])
        if rank == i:
            obj_np_arr[i] = elem
        else:
            col_num = am1 if i <= last1 else am2
            comm.Send(np.reshape(elem, (str_num * col_num)), dest=i)
        del elem

    for i in range(0, size):
        if i == rank:
            pass
        else:
            comm.Recv(obj_np_arr[i], source=i)

    print('rank: ', rank, '\narr: ', obj_np_arr)

    # for k in range(0, size):
    #     for i in range(str_num):
    #         col_num = am1 if i <= last1 else am2
    #         for j in range(col_num):
    #             data[i, j] = list_of_blocs[k][i][j]
    # if rank == 0:
    #     matrixOrig.reshape()
    #     data = np.copy(matrixOrig[0: am1])
    #     print(matrixOrig)
    # 
    #     for i in range(1, last1 + 1):
    #         comm.Isend(np.copy(matrixOrig[i * am1: am1 + i * am1]), dest=i)
    # 
    #     for i in range(last1 + 1, size):
    #         comm.Isend(np.copy(matrixOrig[(last1 + 1) * am1 + (i - last1 - 1) * am2:
    #                                       (last1 + 1) * am1 + am2 + (i - last1 - 1) * am2]), dest=i)
    # 
    # else:
    #     req = comm.Isend(obj_np_arr, source=0)


if __name__ in "__main__":
    parser = argparse.ArgumentParser(description="mpi line-block matrix transpose")
    parser.add_argument("-dim", default=None, type=int, help="dimension for test matrix")
    # parser.add_argument("-file", default=None, type=str, help="json file with actual matrix")
    opt = parser.parse_args()
    mpi_func(opt.dim)
