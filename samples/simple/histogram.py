#!/usr/bin/env python
from __future__ import print_function

import argparse
import dace
import math
import numpy as np

W = dace.symbol('W')
H = dace.symbol('H')
BINS = 256  # dace.symbol('BINS')

# CR version (for a declarative version, see histogram_declarative.py)


@dace.program(dace.float32[H, W], dace.uint32[BINS])
def histogram(A, hist):
    @dace.map(_[0:H, 0:W])
    def compute(i, j):
        a << A[i, j]
        out >> hist(1, lambda x, y: x + y)[:]

        out[min(int(a * BINS), BINS - 1)] = 1


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("W", type=int, nargs="?", default=32)
    parser.add_argument("H", type=int, nargs="?", default=32)
    args = vars(parser.parse_args())

    A = dace.ndarray([H, W], dtype=dace.float32)
    hist = dace.ndarray([BINS], dtype=dace.uint32)

    W.set(args["W"])
    H.set(args["H"])

    print('Histogram %dx%d' % (W.get(), H.get()))

    A[:] = np.random.rand(H.get(),
                          W.get()).astype(dace.float32.type)  #randint(0, 256,
    #        (H.get(), W.get())).astype(dace.uint8.type)
    hist[:] = dace.uint32(0)

    histogram(A, hist)

    if dace.Config.get_bool('profiling'):
        dace.timethis('histogram', 'numpy', dace.eval(H * W), np.histogram, A,
                      BINS)

    diff = np.linalg.norm(
        np.histogram(A, bins=BINS, range=(0.0, 1.0))[0][1:-1] - hist[1:-1])
    print("Difference:", diff)
    print("==== Program end ====")
    exit(0 if diff <= 1e-5 else 1)