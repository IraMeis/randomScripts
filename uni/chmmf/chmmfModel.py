import matplotlib.pyplot as plt
import math
import numpy as np
import time


class ChmmfModel:
    def __init__(self, R, T, k, c, amount_I, amount_K, psi=None):
        self.R = R
        self.T = T
        self.k = k  # теплопроводность
        self.c = c  # теплоемкость
        self.psi = psi or (lambda r: 200 * math.exp(-1 * ((10 * r / self.R) ** 2)))
        self.amount_I = amount_I
        self.amount_K = amount_K

    # проверка матрицы
    @staticmethod
    def is_matrix_correct(a):
        n = len(a)
        # диагональное преобладание - достаточное условие устойчивости прогонки
        for row in range(1, n - 1):
            if abs(a[row][row]) < abs(a[row][row - 1]) + abs(a[row][row + 1]):
                return False
        return not ((abs(a[0][0]) < abs(a[0][1])) or (abs(a[n - 1][n - 1]) < abs(a[n - 1][n - 2])))

    # прогонка
    @staticmethod
    def internal_solution(a, b):
        n = len(a)
        x = [0 for _ in range(0, n)]

        # прямой ход
        alph = [0 for _ in range(0, n)]
        bett = [0 for _ in range(0, n)]
        alph[0] = -a[0][1] / a[0][0]
        bett[0] = b[0] / a[0][0]
        for i in range(1, n - 1):
            znam = a[i][i] + a[i][i - 1] * alph[i - 1]
            alph[i] = -a[i][i + 1] / znam
            bett[i] = (-a[i][i - 1] * bett[i - 1] + b[i]) / znam
        alph[n - 1] = 0
        bett[n - 1] = (-a[n - 1][n - 2] * bett[n - 2] + b[n - 1]) / (a[n - 1][n - 1] + a[n - 1][n - 2] * alph[n - 2])

        # обратный ход
        x[n - 1] = bett[n - 1]
        for i in range(n - 1, 0, -1):
            x[i - 1] = alph[i - 1] * x[i] + bett[i - 1]

        return x

    # решение задачи
    def schema_solution(self):
        start_time = time.time()

        h_t = self.T / self.amount_K
        h_r = self.R / self.amount_I
        BETTA = self.k * h_t / (self.c * h_r ** 2)
        GAMMAxR = self.k * h_t / (self.c * h_r)

        # прогоночная матрица а и массив b - инициализация
        a = []
        b = [0. for _ in range(0, self.amount_I)]
        for i in range(self.amount_I):
            a.append([0 for _ in range(0, self.amount_I)])
            b[i] = self.psi(i * h_r)
        a[0][0] = 1 + 6 * BETTA
        a[0][1] = -6 * BETTA
        for i in range(1, self.amount_I - 1):
            GAMMA = GAMMAxR / (i * h_r)
            a[i][i - 1] = GAMMA - BETTA
            a[i][i] = 1 + 2 * BETTA
            a[i][i + 1] = - GAMMA - BETTA
        a[self.amount_I - 1][self.amount_I - 2] = GAMMAxR / ((self.amount_I - 1) * h_r) - BETTA
        a[self.amount_I - 1][self.amount_I - 1] = 1 + BETTA - GAMMAxR / ((self.amount_I - 1) * h_r)

        if not ChmmfModel.is_matrix_correct(a):
            raise ValueError('Matrix check failed')

        # решение по слоям
        matrix = [b]
        for t in range(1, self.amount_K + 1):  # остальные
            b = ChmmfModel.internal_solution(a, b)
            matrix.append(b)

        # I слой, k = 0
        # u [0,i] = psi(r_i)
        matrix[0].append(self.psi(self.amount_I * h_r))

        # I слой, k = [1, K]
        # u [k,I] - u [k,I-1]
        # ------------------- = 0
        #        h_t
        for t in range(1, self.amount_K + 1):
            matrix[t].append(matrix[t][self.amount_I - 1])

        matrix = np.array(matrix)
        end_time = time.time()
        print("Matrix with I = {0}, K = {1} calculated in {2} seconds"
              .format(self.amount_I, self.amount_K, end_time - start_time))

        return matrix

    def showPlotsByCoordinate(self, matrix, ind: list, coordinate='t', ylim=None, xlim=None, grid=False):
        fig = plt.figure()
        if coordinate == 't':
            r = np.linspace(0, self.R, self.amount_I + 1)
            for i in ind:
                plt.plot(r, matrix[i], label="t = " + str(i * self.T / self.amount_K))
                plt.legend()
            plt.ylabel('u(r, t)')
            plt.xlabel('r')
        elif coordinate == 'r':
            res = matrix.transpose()
            t = np.linspace(0, self.T, self.amount_K + 1)
            for i in ind:
                plt.plot(t, res[i], label="r = " + str(i * self.R / self.amount_I))
                plt.legend()
            plt.ylabel('u(r, t)')
            plt.xlabel('t')
        if ylim:
            plt.ylim(ylim)
        if xlim:
            plt.xlim(xlim)
        if grid:
            plt.grid()
        return fig

