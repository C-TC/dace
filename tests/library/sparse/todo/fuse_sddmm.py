import dace
import numpy as np
from math import sqrt


'''
A = B * (C @ D)
[M, N] = [M, N] * ([M, K] @ [K, N])
A[i,j] = B[i,j] * (C[i,k] @ D[k,j])
'''

C1_dimension = dace.symbol('C1_dimension')
C2_dimension = dace.symbol('C2_dimension')
D1_dimension = dace.symbol('D1_dimension')
D2_dimension = dace.symbol('D2_dimension')
size_A_vals = dace.symbol('size_A_vals')
size_B2_crd = dace.symbol('size_B2_crd')
size_B2_pos = dace.symbol('size_B2_pos')
size_B_vals = dace.symbol('size_B_vals')
size_C_vals = dace.symbol('size_C_vals')
size_D_vals = dace.symbol('size_D_vals')
M = dace.symbol('M')
N = dace.symbol('N')
K = dace.symbol('K')
nnz_A = dace.symbol('nnz_A')
nnz_B = dace.symbol('nnz_B')

@dace.program
def sddmm_unfused(A_vals: dace.float64[nnz_A], B2_crd: dace.int32[nnz_B], B2_pos: dace.int32[M+1], B_vals: dace.float64[nnz_B], C: dace.float64[M,K], D: dace.float64[K,N]):
    E = np.zeros([M, N], dtype=np.float64)
    for i in dace.map[0:M]:
        for j in dace.map[0:N]:
            for k in dace.map[0:K]:
                E[i,j] += C[i,k] * D[k,j]

    for i in dace.map[0:M]:
        for j in dace.map[B2_pos[i]:B2_pos[i+1]]:
            A_vals[j] = B_vals[j] * E[i, B2_crd[j]]

sdfg0 = sddmm_unfused.to_sdfg().view()


@dace.program
def sddmm_fused(A_vals: dace.float64[nnz_A], B2_crd: dace.int32[nnz_B], B2_pos: dace.int32[M+1], B_vals: dace.float64[nnz_B], C: dace.float64[M,K], D: dace.float64[K,N]):

    for i in dace.map[0:M]:
        for j in dace.map[B2_pos[i]:B2_pos[i+1]]:
            for k in dace.map[0:K]:
                A_vals[j] += B_vals[j] * C[i,k] * D[k, B2_crd[j]]


sdfg = sddmm_fused.to_sdfg().view()