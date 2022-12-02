# python matrix_file_creator.py -cols 7 -rows 8

import argparse
import json
import numpy as np


def create_matr_file(rows, cols):
    matrix = np.random.uniform(1, 100, [rows, cols])
    data = {'matrix': matrix.tolist(), 'rows': rows, 'cols': cols}
    with open('matrix_r' + str(rows) + 'c' + str(cols) + '.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="mpi line-block matrix transpose")
    parser.add_argument("-cols", default=None, type=int, help="cols number")
    parser.add_argument("-rows", default=None, type=int, help="rows number")
    opt = parser.parse_args()
    if opt.rows and opt.cols:
        create_matr_file(rows=opt.rows, cols=opt.cols)
