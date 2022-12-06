# mpiexec -n 3 python block_t.py -cols 4 -rows 3
# optional args: -file matrix_r3c4.json
#                -random

import argparse
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


def get_test_mm_matrix(rows, cols, random):
    dim = max(cols, rows)
    matrix = np.zeros((dim, dim), dtype=np.float_)
    if random:
        matrix[0:rows, 0:cols] = np.random.uniform(1, 100, [rows, cols])
    else:
        matrix[0:rows, 0:cols] = np.reshape(np.arange(rows * cols, dtype=np.float_), (rows, cols))
    return matrix   


def read_file(rows, cols, path):
    import json
    with open(path) as f:
        data = json.load(f)
    dcols = data['cols']
    drows = data['rows']
    if cols != dcols or rows != drows:
        raise ValueError
    dim = max(cols, rows)
    matrix = np.zeros((dim, dim), dtype=np.float_)
    matrix[0:rows, 0:cols] = np.array(data['matrix'], dtype=np.float_)
    return matrix


def mpi_func(rows: int, cols: int, random: bool = None, file: str = None):
    import time
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    dim = max(rows, cols)
    am1, total1, am2 = get_send_params(dim, size)
    str_num = am1 if rank < total1 else am2
    data = np.empty(str_num * dim, dtype=np.float_)

    if rank == 0:
        start = time.perf_counter()
        if not file:
            matrixOrig = get_test_mm_matrix(rows, cols, random)
        else:
            matrixOrig = read_file(rows, cols, path=file)

        # print('--- original ---\n', matrixOrig, '\n---')

        data = matrixOrig[0: am1]

        for i in range(1, total1):
            comm.Send(matrixOrig[i * am1: am1 + i * am1], dest=i)

        for i in range(total1, size):
            comm.Send(matrixOrig[total1 * am1 + (i - total1) * am2:
                                 total1 * am1 + am2 + (i - total1) * am2], dest=i)
    else:
        comm.Recv(data, source=0)

    data = np.reshape(data, (str_num, dim))

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
            continue
        comm.Recv(obj_np_arr[i], source=i)
    added = 0
    for i in range(0, size):
        col_num = am1 if i < total1 else am2
        data[:, added: added + col_num] = obj_np_arr[i]
        added += col_num
    
    if rank == 0:
        matrixOrig[0: am1] = data

        for i in range(1, total1):
            comm.Recv(matrixOrig[i * am1: am1 + i * am1], source=i)

        for i in range(total1, size):
            comm.Recv(matrixOrig[total1 * am1 + (i - total1) * am2:
                                 total1 * am1 + am2 + (i - total1) * am2], source=i)
        # print('--- transposed ---\n', matrixOrig, '\n---')
        stop = time.perf_counter()
        t = stop - start
        print(f"\ncalculated in {t:0.5f} seconds\n")

        start1 = time.perf_counter()
        matrixOrig.transpose()
        stop1 = time.perf_counter()
        t1 = stop1 - start1
        print(f"np-transpose calculated in {t1:0.5f} seconds")

        start2 = time.perf_counter()
        for i in range(dim):
            for j in range(i + 1, dim):
                val = matrixOrig[i, j]
                matrixOrig[i, j] = matrixOrig[j, i]
                matrixOrig[j, i] = val
        stop2 = time.perf_counter()
        t2 = stop2 - start2
        print(f"naive transpose calculated in {t2:0.5f} seconds")

        print(f"acc mpi vs np-transpose is {t1 / t:0.5f}")
        print(f"acc mpi vs naive transpose is {t2 / t:0.5f}")
    else:
        comm.Send(data, dest=0)


if __name__ in "__main__":
    parser = argparse.ArgumentParser(description="mpi line-block matrix transpose")
    parser.add_argument("-cols", default=None, type=int, help="cols number")
    parser.add_argument("-rows", default=None, type=int, help="rows number")
    parser.add_argument("-file", default=None, type=str, help="json file with actual matrix")
    parser.add_argument("-random", action='store_true', help="fill matrix with random numbers if set")
    opt = parser.parse_args()
    if opt.rows and opt.cols:
        mpi_func(opt.rows, opt.cols, opt.random, opt.file)
