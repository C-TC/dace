# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
from __future__ import annotations
from typing import List, Dict, Any
from dace.libraries.sparse.nodes.taco import TacoProxy, TensorFormat, CSR, CSC, CSF, DCSR, DCSC, COO, SPARSE_VECTOR, TensorModeType
import os

def test_format_comb(name: str, expr: str, format_dict_list: List[Dict[str, TensorFormat]], transformations: List[List[Any]], output_dir: str):
    i = 0
    for format_dict in format_dict_list:
        for trans_list in transformations:
            output_folder = os.path.join(output_dir, name)
            taco = TacoProxy(expr, output_dir=output_folder, name=name + str(i), debug=True)
            taco.set_formats(format_dict)
            taco.set_transformations(trans_list)
            taco.generate()
            i += 1

output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".dacecache")

def test_spmv():
    test_format_comb('spmv', 'A(i) = B(i,j) * C(j)', [{'B': CSR}, {'B': CSC}, {'B': DCSR}, {'B': DCSC}, {'B': COO}], [[],], output_dir)

def test_spmm():
    test_format_comb('spmm', 'A(i,j) = B(i,k) * C(k,j)', [{'B': CSR}, {'B': CSC}, {'B': DCSR}, {'B': DCSC}, {'B': COO}], [[],], output_dir)

def test_sddmm():
    test_format_comb('sddmm', 'A(i,j) = B(i,j) * C(i,k) * D(k,j)', [{'A': CSR, 'B': CSR}, {'A': CSC, 'B': CSC}, {'A': DCSR, 'B': DCSR}, {'A': DCSC, 'B': DCSC}, {'A': COO,'B': COO}], [[['sampled_replace','A','B'],],], output_dir)

def test_sampled():
    test_format_comb('sampled', 'A(i,j) = B(i,j) * C(i,k) * D(k,l) * E(l,j)', [{'A': CSR, 'B': CSR}, {'A': CSC, 'B': CSC}, {'A': DCSR, 'B': DCSR}, {'A': DCSC, 'B': DCSC}, {'A': COO, 'B': COO}], [[['sampled_replace','A','B'],],], output_dir)

def test_plus3():
    test_format_comb('plus3', 'A(i,j) = B(i,j) + C(i,j) + D(i,j)', [{'A': CSR, 'B': CSR, 'C': CSR, 'D': CSR}, {'A': CSC, 'B': CSC, 'C': CSC, 'D': CSC}, {'A': DCSR, 'B': DCSR, 'C': DCSR, 'D': DCSR}, {'A': DCSC, 'B': DCSC, 'C': DCSC, 'D': DCSC}], [[],], output_dir)

def test_mattransmul():
    test_format_comb('mattransmul', 'y(j) = 2 * A(i,j) * x(i) + 5 * z(j)', [{'A': CSR}, {'A': CSC}, {'A': DCSR}, {'A': DCSC}, {'A': COO}], [[],], output_dir)

def test_residual():
    test_format_comb('residual', 'r(i) = b(i) - A(i,j) * x(j)', [{'A': CSR}, {'A': CSC}, {'A': DCSR}, {'A': DCSC}, {'A': COO}], [[],], output_dir)

def test_ttv():
    test_format_comb('ttv', 'A(i,j) = B(i,j,k) * C(k)', [{'B': CSF}, ], [[],], output_dir)

def test_ttm():
    test_format_comb('ttm', 'A(i,j,k) = B(i,j,l) * C(k,l)', [{'B': CSF}, ], [[],], output_dir)

def test_mttkrp():
    test_format_comb('mttkrp', 'A(i,j) = B(i,k,l) * C(k,j) * D(l,j)', [{'B': CSF}, ], [[],], output_dir)

def test_tensor_plus():
    test_format_comb('tensor_plus', 'A(i,j,k) = B(i,j,k) + C(i,j,k)', [{'B': CSF}, ], [[],], output_dir)

def test_tensor_inner():
    test_format_comb('tensor_inner', 'a = B(i,j,k) * C(i,j,k)', [{'B': CSF}, ], [[],], output_dir)

def test_sched_spmv():
    test_format_comb('sched_spmv', 'A(i) = B(i,j) * C(j)', [{'B': CSR},], [[],[['split','i','i0','i1',16], ['reorder','i0','i1','j']]], output_dir)

def test_sched_spmm():
    test_format_comb('sched_spmm', 'A(i,j) = B(i,k) * C(k,j)', [{'B': CSR},], [[],[['split','i','i0','i1',16], ['pos','k','kpos','B'], ['split','kpos','kpos0','kpos1',8], ['reorder','i0','i1','kpos0','j','kpos1']]], output_dir)

def test_sched_spmspv():
    test_format_comb('sched_spmspv', 'y(i) = A(i,j) * x(j)', [{'A': CSC, 'x': SPARSE_VECTOR},], [[],[['reorder','j', 'i'], ['pos', 'j', 'jpos', 'x'], ['split','jpos','jpos0','jpos1',16]]], output_dir)

def test_sched_sddmm():
    test_format_comb('sched_sddmm', 'A(i,k) += B(i,k) * C(i,j) * D(j,k)', [{'A': CSR, 'B': CSR},], [[['sampled_replace','A','B'],],[['sampled_replace','A','B'],['split','i','i0','i1',16]],[['sampled_replace','A','B'],['reorder','i','k','j']]], output_dir)

def test_sched_ttv():
    format_A = TensorFormat([TensorModeType.Dense, TensorModeType.Sparse])
    format_B = TensorFormat([TensorModeType.Dense, TensorModeType.Sparse, TensorModeType.Sparse])
    test_format_comb('sched_ttv', 'A(i,j) = B(i,j,k) * c(k)', [{'A': format_A, 'B': format_B},], [[], [['fuse','i','j','f'], ['pos','f','fpos','B'], ['split','fpos','chunk','fpos2',16], ['reorder','chunk','fpos2','k'],]], output_dir)

def test_sched_ttm():
    # segfault
    format_B = TensorFormat([TensorModeType.Dense, TensorModeType.Sparse, TensorModeType.Sparse])
    test_format_comb('sched_ttm', 'A(i,j) = B(i,k,l) * C(k,j) * D(l,j)', [{'B': format_B},], [[], [['fuse','i','j','f'], ['pos','f','fpos','B'], ['split','fpos','chunk','fpos2',16], ['pos','k','kpos','B'], ['split','kpos','kpos1','kpos2',16], ['reorder','chunk','fpos2','kpos1','l','kpos2']]], output_dir)


# def test_sched_mattransmul():
#     test_format_comb('sched_mattransmul', 'y(j) = 2 * A(i,j) * x(i) + 5 * z(j)', [{'A': CSR},], [[], [['precompute', '2 * A(i,j) * x(i) + 5 * z(j)', 'j', 'j'],]], output_dir)

# def test_sched_sampled():
#     test_format_comb('sched_sampled', 'A(i,j) = B(i,j) * C(i,k) * D(k,l) * E(l,j)', [{'A': CSR, 'B': CSR},], [[['precompute', 'B(i,j) * E(l,j)', 'j', 'j'],],], output_dir)


if __name__ == "__main__":
    test_spmv()
    test_spmm()
    test_sddmm()
    test_plus3()
    test_mattransmul()
    test_residual()
    test_ttv()
    test_ttm()
    test_mttkrp()
    test_tensor_plus()
    test_tensor_inner()
    test_sched_spmv()
    test_sched_spmm()
    test_sched_spmspv()
    test_sched_sddmm()
    test_sched_ttv()
    test_sched_ttm()
    test_sampled()
