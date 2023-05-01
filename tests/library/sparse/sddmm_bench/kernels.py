import cupy as cp
import cupyx as cpx
import numpy as np
from scipy import sparse

@cpx.jit.rawkernel()
def sddmm_cp(D_data, A_indices, A_indptr, A_data, B, C):
    bid = cpx.jit.blockIdx.x
    num_blocks = cpx.jit.gridDim.x
    tid = cpx.jit.threadIdx.x
    num_threads = cpx.jit.blockDim.x

    for i in range(bid, A_indptr.shape[0] - 1, num_blocks):
        for j in range(A_indptr[i] + tid, A_indptr[i + 1], num_threads):
            rowNo = i
            colNo = A_indices[j]
            tmp = 0.0
            for k in range(B.shape[1]):
                tmp += A_data[j] * B[rowNo, k] * C[colNo, k]
            D_data[j] = tmp


FULL_MASK = 0xFFFFFFFF

@cpx.jit.rawkernel()
def sddmm_cp_shfl(D_data, A_indices, A_indptr, A_data, B, C):

    bid = cpx.jit.blockIdx.x  # Block ID
    num_blocks = cpx.jit.gridDim.x
    tid = cpx.jit.threadIdx.x  # Thread ID
    num_threads = cpx.jit.blockDim.x
    wid = tid // cpx.jit.warpsize  # Warp ID
    num_warps = cp.int32(cp.ceil(num_threads / cpx.jit.warpsize))
    twid = tid % cpx.jit.warpsize  # Thread ID within warp

    for i in range(bid, (len(A_indptr) - 1), num_blocks):
        for j in range(A_indptr[i] + wid, A_indptr[i + 1], num_warps):
            rowNo = i
            colNo = A_indices[j]
            a = cp.float32(0)
            for k in range(twid, B.shape[1], cpx.jit.warpsize):
                a += B[rowNo, k] * C[colNo, k]
            a += cpx.jit.shfl_down_sync(FULL_MASK, a, 16) 
            a += cpx.jit.shfl_down_sync(FULL_MASK, a, 8)
            a += cpx.jit.shfl_down_sync(FULL_MASK, a, 4)
            a += cpx.jit.shfl_down_sync(FULL_MASK, a, 2)
            a += cpx.jit.shfl_down_sync(FULL_MASK, a, 1)
            if twid == 0:
                D_data[j] = A_data[j] * a


@cpx.jit.rawkernel()
def sddvm_cp(D_data, A_indices, A_indptr, A_data, B, C):
    bid = cpx.jit.blockIdx.x
    num_blocks = cpx.jit.gridDim.x
    tid = cpx.jit.threadIdx.x
    num_threads = cpx.jit.blockDim.x

    for i in range(bid, A_indptr.shape[0] - 1, num_blocks):
        for j in range(A_indptr[i] + tid, A_indptr[i + 1], num_threads):
            rowNo = i
            colNo = A_indices[j]
            D_data[j] += A_data[j] * B[rowNo] * C[colNo]