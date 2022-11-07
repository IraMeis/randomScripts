import numpy as np
from PIL import Image


class Model:
    def __init__(self):
        # user input as constants
        self.N = 0
        self.As = None
        self.Ds = None
        self.matrix = None
        self.T = 0
        self.K = 0

        # user input as images + autodetect lx and ly while setting f_currents0
        self.f_currents = None
        self.lx = 0
        self.ly = 0

        # next step and render adaptation of func
        self.f_nexts = None
        self.f_renders = None
        self.N2irgb = None

        # todo user input, will be implemented later
        self.axs = None
        self.bxs = None
        self.ays = None
        self.bys = None

    def set_N(self, N):
        self.N = N

    def set_f_currents0_make_f_renders0(self, paths: list, resize: tuple = None):
        if self.N < 1 or self.N > 3 * len(paths):
            raise ValueError("Incorrect number of images for given N = {0}".format(self.N))
        self.f_currents = []
        img0 = Image.open(paths[0])
        for i in range(len(paths)):
            img = Image.open(paths[i])
            if not img.mode == 'RGB':
                print("Warning! Image format is {0}, autoconvertation may produce something unexpected".format(
                    img.format))
                img.convert('RGB')
            if len(paths) > 1:
                if resize:
                    img = img.resize(resize)
                else:
                    img = img.resize(img0.size)
            self.f_currents.append(np.array(img))
            print("Image with (w, h) = {0} processed".format(img.size))
            if self.N >= (i + 1) * 3:
                break

        self.lx = self.f_currents[0].shape[0]
        self.ly = self.f_currents[0].shape[1]

        self.f_renders = self.f_currents.copy()
        self.f_nexts = []
        for k in range(len(self.f_currents)):
            self.f_nexts.append(np.zeros((self.lx, self.ly, 3), dtype=np.int_))
        for k in range(len(self.f_currents)):
            for i in range(self.lx):
                for j in range(self.ly):
                    for rgb in range(3):
                        self.f_nexts[k][i, j, rgb] = np.int_(self.f_currents[k][i, j, rgb])

    def make_dict(self):
        self.N2irgb = {}
        for i in range(self.N):
            self.N2irgb[i] = (np.uint8(i / 3), np.uint8(i % 3))

    def make_sum_for_Ni(self, func_number, i, j):
        summ = 0
        func_value = self.f_currents[self.N2irgb[func_number][0]][i, j, self.N2irgb[func_number][1]]
        for func in range(self.N):
            # indexes of image and color (species id)
            img = self.N2irgb[func][0]
            color = self.N2irgb[func][1]
            summ += self.f_currents[img][i, j, color] * func_value * self.matrix[func_number][func]
        return summ

    def calculate_next_upd_f_renders(self):
        step = self.T / self.K

        self.f_currents = self.f_nexts
        self.f_nexts = []
        for i in range(len(self.f_currents)):
            self.f_nexts.append(np.zeros((self.lx, self.ly, 3), dtype=np.int_))

        # Ix, Iy == 1
        for func in range(self.N):
            img = self.N2irgb[func][0]
            color = self.N2irgb[func][1]
            for i in range(1, self.lx - 1):
                for j in range(1, self.ly - 1):
                    val = self.f_currents[img][i, j, color] \
                          + step * self.As[func] * self.f_currents[img][i, j, color] \
                          + step * self.make_sum_for_Ni(func, i, j) \
                          + step * self.Ds[func] \
                          * (self.f_currents[img][i + 1, j, color] + self.f_currents[img][i - 1, j, color]
                             + self.f_currents[img][i, j + 1, color] + self.f_currents[img][i, j - 1, color]
                             - 4 * self.f_currents[img][i, j, color])
                    self.f_nexts[img][i, j, color] = int(val > 0) * val
                    if val > 255:
                        self.f_renders[img][i, j, color] = np.uint8(255)
                    else:
                        self.f_renders[img][i, j, color] = np.uint8(int(val > 0) * val)
