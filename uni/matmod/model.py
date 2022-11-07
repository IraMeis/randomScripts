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
        self.It = 0
        self.Ix = 0
        self.Iy = 0

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

    def set_f_nexts0_set_f_renders0(self, paths: list, resize: tuple = None):
        if self.N < 1 or self.N > 3 * len(paths):
            raise ValueError("Incorrect number of images for given N = {0}".format(self.N))
        self.f_renders = []

        img0 = Image.open(paths[0])
        img0_size = img0.size
        img0.close()

        for i in range(len(paths)):
            with Image.open(paths[i]) as img:
                if not img.mode == 'RGB':
                    raise ValueError("Incorrect image format {0}".format(img.format))
                if len(paths) > 1:
                    if resize:
                        img = img.resize(resize)
                    else:
                        img = img.resize(img0_size)
                self.f_renders.append(np.array(img))
                print("Image with (w, h) = {0} processed".format(img.size))
                if self.N >= (i + 1) * 3:
                    break
        self.f_renders = np.array(self.f_renders)
        self.f_nexts = np.empty_like(self.f_renders, dtype=np.float_)
        self.f_nexts[:] = self.f_renders.astype(np.float_)
        self.lx = self.f_renders[0].shape[0]
        self.ly = self.f_renders[0].shape[1]
        self.__make_dict()

    def __make_dict(self):
        self.N2irgb = {}
        for i in range(self.N):
            self.N2irgb[i] = (np.uint8(i / 3), np.uint8(i % 3))

    def __make_sum_for_Ni(self, func_number, i, j):
        summ = 0
        func_value = self.f_currents[self.N2irgb[func_number][0], i, j, self.N2irgb[func_number][1]]
        for func in range(self.N):
            # indexes of image and color (species id)
            img = self.N2irgb[func][0]
            color = self.N2irgb[func][1]
            summ += self.f_currents[img, i, j, color] * func_value * self.matrix[func_number][func]
        return summ

    def calculate_f_nexts_update_f_renders(self):
        self.f_currents = self.f_nexts
        self.f_nexts = np.empty_like(self.f_currents)

        for func in range(self.N):
            img = self.N2irgb[func][0]
            color = self.N2irgb[func][1]
            for i in range(1, self.lx - 1):
                for j in range(1, self.ly - 1):
                    step_func = self.f_currents[img, i, j, color]
                    val = step_func + self.It * self.As[func] * step_func \
                        + self.It * self.__make_sum_for_Ni(func, i, j) \
                        + self.It * self.Ds[func] \
                        * ((self.f_currents[img, i + 1, j, color] + self.f_currents[img, i - 1, j, color]
                           - 2 * step_func) / self.Ix ** 2
                           + (self.f_currents[img, i, j + 1, color] + self.f_currents[img, i, j - 1, color]
                           - 2 * step_func) / self.Iy ** 2)
                    self.f_nexts[img, i, j, color] = int(val > 0) * val
                    self.f_renders[img, i, j, color] = \
                        np.uint8(int(val > 0) * val * int(val <= 255) + 255 * int(val > 255))
