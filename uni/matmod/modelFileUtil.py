import json
import numpy as np
from uni.matmod.model import Model as mm


def create_border_file(N, lax: list, lbx: list, lay: list, lby: list, sizes=None, xsz=mm.max_dim, ysz=mm.max_dim):
    pics = mm.get_pic_amount(N)
    p2val = mm.make_dict(N)
    if sizes:
        ly, lx = mm.make_resize(sizes, mm.max_dim)
    else:
        ly, lx = xsz, ysz
    ax = np.zeros((pics, ly, 3), dtype=np.uint8)
    bx = np.zeros((pics, ly, 3), dtype=np.uint8)
    ay = np.zeros((pics, lx, 3), dtype=np.uint8)
    by = np.zeros((pics, lx, 3), dtype=np.uint8)
    for fi in range(N):
        img_pos = p2val[fi][0]
        color = p2val[fi][1]
        for i in range(ly):
            ax[img_pos, i, color] = mm.to_rgb_value(lax[fi]())
            bx[img_pos, i, color] = mm.to_rgb_value(lbx[fi]())
        for j in range(lx):
            ay[img_pos, j, color] = mm.to_rgb_value(lay[fi]())
            by[img_pos, j, color] = mm.to_rgb_value(lby[fi]())

    data = {'ax': ax.tolist(), 'ay': ay.tolist(), 'by': by.tolist(), 'bx': bx.tolist()}
    with open('borders.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    # paste actual functions here

    # y
    # ^        _____________________
    # |        |        ^          |
    # | ay ->  |        |          | <- by
    # |        |       bx          |
    # |        _____________________ 
    # |                ^
    # |                |
    # |               ax
    #  -----------------------------------> x

    lax = [lambda: 200, lambda: 200]
    lay = [lambda: 15, lambda: 190]
    lbx = [lambda: 250, lambda: 200]
    lby = [lambda: 250, lambda: 3]

    # paste target images sizes (w, h) here
    sizes = [(96, 72)]

    create_border_file(N=2, lax=lax, lay=lay, lby=lby, lbx=lbx, sizes=sizes)
