import time
import numpy as np


class Model:
    def __init__(self, N=0, As=None, Ds=None, matrix=None, T=0, It=0, Ix=0, Iy=0):
        # user input as constants | arrays of constants
        self.N = N
        self.As = As
        self.Ds = Ds
        self.matrix = matrix
        self.T = T
        self.It = It
        self.Ix = Ix
        self.Iy = Iy
        # user input as images
        self.f_currents = None
        # autodetect lx and ly while init 0 step
        self.lx = 0
        self.ly = 0
        # next step and render adaptation of func
        self.f_nexts = None
        self.f_renders = None
        # utility dict
        self.N2irgb = None
        self.max_dim = 200
        # todo user input, will be implemented later
        self.axs = None
        self.bxs = None
        self.ays = None
        self.bys = None
        # run control param
        self.is_running = False

    def set_f_nexts0_set_f_renders0(self, imgs: list, resize: tuple = None):
        if self.N < 1 or self.N > 3 * len(imgs):
            raise ValueError("Incorrect number of images for given N = {0}".format(self.N))
        self.f_renders = []

        rs = None
        if not resize:
            rs = self.__make_resize(imgs)

        for i in range(len(imgs)):
            if not imgs[i].mode == 'RGB':
                raise ValueError("Incorrect image format {0}".format(imgs[i].format))
            if resize:
                imgs[i] = imgs[i].resize(resize)
            else:
                imgs[i] = imgs[i].resize(rs)
            self.f_renders.append(np.array(imgs[i]))
            print("Image with (w, h) = {0} processed".format(imgs[i].size))
            if self.N >= (i + 1) * 3:
                break
        self.f_renders = np.array(self.f_renders)
        self.f_nexts = np.empty_like(self.f_renders, dtype=np.float_)
        self.f_nexts[:] = self.f_renders.astype(np.float_)
        self.lx = self.f_renders[0].shape[0]
        self.ly = self.f_renders[0].shape[1]
        self.__make_dict()
        self.__make_null_borders()

    def __make_resize(self, imgs: list):
        for i in range(len(imgs)):
            w, h = imgs[i].size
            if w < self.max_dim and h < self.max_dim:
                continue
            if w > h:
                return tuple([int(self.max_dim), int(h / w * self.max_dim)])
            else:
                return tuple([int(w / h * self.max_dim), int(self.max_dim)])
        return imgs[0].size

    def __make_dict(self):
        self.N2irgb = {}
        for i in range(self.N):
            self.N2irgb[i] = (np.uint8(i / 3), np.uint8(i % 3))

    def __make_sum_for_Ni(self, func_number, i, j):
        summ = np.float_(0)
        func_value = self.f_currents[self.N2irgb[func_number][0], i, j, self.N2irgb[func_number][1]]
        for func in range(self.N):
            # indexes of image and color (species id)
            img = self.N2irgb[func][0]
            color = self.N2irgb[func][1]
            summ += self.f_currents[img, i, j, color] * func_value * self.matrix[func_number, func]
        return summ

    def __make_null_borders(self):
        for func in range(self.N):
            img_pos = self.N2irgb[func][0]
            img_n = self.f_nexts[img_pos]
            img_r = self.f_renders[img_pos]
            for i in [0, self.lx - 1]:
                    img_n[i] = np.zeros((self.ly, 3), dtype=np.float_)
                    img_r[i] = np.zeros((self.ly, 3), dtype=np.uint8)
            for j in [0, self.ly - 1]:
                for i in range(1, self.lx - 1):
                    img_n[i, j] = np.zeros(3, dtype=np.float_)
                    img_r[i, j] = np.zeros(3, dtype=np.uint8)

    def calculate_f_nexts_update_f_renders(self):
        self.f_currents = self.f_nexts
        self.f_nexts[:] = self.f_currents
        start = time.perf_counter()
        for func in range(self.N):
            img_pos = self.N2irgb[func][0]
            color = self.N2irgb[func][1]
            img_c = self.f_currents[img_pos]
            img_n = self.f_nexts[img_pos]
            img_r = self.f_renders[img_pos]
            for i in range(1, self.lx - 1):
                for j in range(1, self.ly - 1):
                    step_func = img_c[i, j, color]
                    val = step_func + self.It * self.As[func] * step_func \
                        + self.It * self.__make_sum_for_Ni(func, i, j) \
                        + self.It * self.Ds[func] \
                        * ((img_c[i + 1, j, color] + img_c[i - 1, j, color] - 2 * step_func) / self.Ix ** 2
                           + (img_c[i, j + 1, color] + img_c[i, j - 1, color] - 2 * step_func) / self.Iy ** 2)
                    img_n[i, j, color] = int(val > 1) * val
                    img_r[i, j, color] = np.uint8(int(val > 0) * val * int(val <= 255) + 255 * int(val > 255))
        stop = time.perf_counter()
        print(f"Calculated in {stop - start:0.5f} seconds")
