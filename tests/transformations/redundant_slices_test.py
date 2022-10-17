# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np
import warnings

import dace
from dace import data, nodes
from dace.transformation.dataflow import RedundantReadSlice, RedundantWriteSlice, RemoveSliceView


def _count_views(sdfg: dace.SDFG) -> int:
    num = 0
    for n, _ in sdfg.all_nodes_recursive():
        if (isinstance(n, nodes.AccessNode) and isinstance(sdfg.arrays[n.data], data.View)):
            num += 1
    return num


@dace.program
def jacobi1d_half(TMAX: dace.int32, A: dace.float32[12], B: dace.float32[12]):
    for _ in range(TMAX):
        B[1:-1] = 0.3333 * (A[:-2] + A[1:-1] + A[2:])


def test_read_slice():
    sdfg = jacobi1d_half.to_sdfg(simplify=False)
    num_views_before = _count_views(sdfg)
    if num_views_before != 3:
        warnings.Warn("Incorrect number of Views detected. Please ensure that "
                      "the test is compatible with this DaCe version.")
    sdfg.apply_transformations_repeated(RedundantReadSlice)
    num_views_after = _count_views(sdfg)
    assert (num_views_after == 0)


@dace.program
def jacobi1d_half2(TMAX: dace.int32, A: dace.float32[12, 12, 12], B: dace.float32[12]):
    for _ in range(TMAX):
        B[1:-1] = 0.3333 * (A[:-2, 3, 4] + A[5, 1:-1, 6] + A[7, 8, 2:])


def test_read_slice2():
    sdfg = jacobi1d_half2.to_sdfg(simplify=False)
    num_views_before = _count_views(sdfg)
    if num_views_before != 3:
        warnings.Warn("Incorrect number of Views detected. Please ensure that "
                      "the test is compatible with this DaCe version.")
    sdfg.apply_transformations_repeated(RedundantReadSlice)
    num_views_after = _count_views(sdfg)
    assert (num_views_after == 0)


@dace.program
def write_slice(A: dace.float32[10]):
    B = A[2:8]
    B[:] = np.pi


def test_write_slice():
    sdfg = write_slice.to_sdfg(simplify=False)
    num_views_before = _count_views(sdfg)
    if num_views_before == 0:
        warnings.Warn("Incorrect number of Views detected. Please ensure that "
                      "the test is compatible with this DaCe version.")
    sdfg.apply_transformations_repeated(RedundantWriteSlice)
    num_views_after = _count_views(sdfg)
    assert (num_views_after == 0)


@dace.program
def write_slice2(A: dace.float32[10, 10, 10]):
    B1 = A[2:8, 3, 4]
    B2 = A[5, 2:8, 6]
    B3 = A[7, 8, 2:8]
    B1[:] = np.pi
    B2[:] = np.pi
    B3[:] = np.pi


def test_write_slice2():
    sdfg = write_slice2.to_sdfg(simplify=False)
    num_views_before = _count_views(sdfg)
    if num_views_before == 0:
        warnings.Warn("Incorrect number of Views detected. Please ensure that "
                      "the test is compatible with this DaCe version.")
    sdfg.apply_transformations_repeated(RedundantWriteSlice)
    num_views_after = _count_views(sdfg)
    assert (num_views_after == 0)


def test_view_slice_detect_simple():
    adesc = dace.float64[1, 1]
    vdesc = dace.data.View(dace.float64, [1])
    mapping, unsqueezed, squeezed = RemoveSliceView.get_matching_dimensions(vdesc, adesc)
    assert mapping == {0: 0}
    assert len(unsqueezed) == 0
    assert tuple(squeezed) == (1, )


def test_view_slice_detect_complex():
    M = dace.symbol('M')
    N = dace.symbol('N')
    K = dace.symbol('K')

    adesc = dace.float64[2, 2, 1, 1, N]
    adesc.strides = [5 * M * N * K, M * N * K, M * N, 1, N]
    vdesc = dace.data.View(dace.float64, [2, 1, 2, 1, N, 1], strides=[5 * M * N * K, M * N * K, M * N * K, M * N, N, N])
    mapping, unsqueezed, squeezed = RemoveSliceView.get_matching_dimensions(vdesc, adesc)
    assert mapping == {0: 0, 2: 1, 3: 2, 4: 4}
    assert tuple(unsqueezed) == (1, 5)
    assert tuple(squeezed) == (3, )


if __name__ == '__main__':
    test_read_slice()
    test_read_slice2()
    test_write_slice()
    test_write_slice2()
    test_view_slice_detect_simple()
    test_view_slice_detect_complex()
