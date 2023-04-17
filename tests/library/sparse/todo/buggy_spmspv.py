import dace
import numpy as np

'''
C = A @ B
[M, K] = [M, N] @ [N, K]
C[i, k] = A[i, j] * B[j, k]
'''

M = dace.symbol('M')
N = dace.symbol('N')
K = dace.symbol('K')
nnz_A = dace.symbol('nnz_A')
nnz_B = dace.symbol('nnz_B')
nnz_ax_A = dace.symbol('nnz_ax_A')
nnz_ax_B = dace.symbol('nnz_ax_B')

# buggy
@dace.program
def spmspm_csr_csr(A2_pos: dace.int32[M + 1], A2_crd: dace.int32[nnz_A], A_val: dace.float32[nnz_A], B2_pos: dace.int32[N + 1], B2_crd: dace.int32[nnz_B], B_val: dace.float32[nnz_B]):
    C = np.zeros([M, K], dtype=np.float32)

    for i in dace.map[0:M]:
        for pj in dace.map[A2_pos[i]:A2_pos[i + 1]]:
            for pk in dace.map[B2_pos[A2_crd[pj]]:B2_pos[A2_crd[pj] + 1]]:
                C[i, B2_crd[pk]] += A_val[pj] * B_val[pk]

    return C
sdfg = spmspm_csr_csr.to_sdfg()
