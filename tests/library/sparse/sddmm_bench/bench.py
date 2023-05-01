from numpy.random import default_rng
from typing import List, Tuple
from copy import deepcopy
import numpy as np
import cupy as cp
import dace
from timeit import repeat
from scipy import sparse
import kernels
from dace.transformation.dataflow import MapInterchange, StripMining, MapReduceFusion, MapExpansion, MapToForLoop, TrivialTaskletElimination, WarpTiling
from dace.transformation.interstate import GPUTransformSDFG


def copy_to_gpu(sdfg):
    for k, v in sdfg.arrays.items():
        if not v.transient and isinstance(v, dace.data.Array):
            v.storage = dace.dtypes.StorageType.GPU_Global


def find_map_entry(sdfg: dace.SDFG, map_name_list: List[str]) -> Tuple[dace.sdfg.nodes.MapEntry]:
    if isinstance(map_name_list, str):
        map_name_list = [
            map_name_list,
        ]
    ret_list = [None] * len(map_name_list)
    for state in sdfg.states():
        for node in state.nodes():
            if isinstance(node, dace.sdfg.nodes.MapEntry):
                for i, map_name in enumerate(map_name_list):
                    if map_name == node.map.params[0]:
                        ret_list[i] = node
    # check if all map entries are found
    assert all([x is not None for x in ret_list])

    # unpack if only one map entry is found
    if len(ret_list) == 1:
        return ret_list[0]
    else:
        return tuple(ret_list)


M = dace.symbol('M')
N = dace.symbol('N')
K = dace.symbol('K')
nnz_A = dace.symbol('nnz_A')
nnz_D = dace.symbol('nnz_D')


@dace.program
def sddmm(D_vals: dace.float32[nnz_D], A2_crd: dace.int32[nnz_A], A2_pos: dace.int32[M + 1],
          A_vals: dace.float32[nnz_A], B: dace.float32[M, K], C: dace.float32[K, N]):
    for i in dace.map[0:M]:
        for j in dace.map[A2_pos[i]:A2_pos[i + 1]]:
            for k in dace.map[0:K]:
                D_vals[j] += A_vals[j] * B[i, k] * C[k, A2_crd[j]]


@dace.program
def sddvm(D_vals: dace.float32[nnz_D], A2_crd: dace.int32[nnz_A], A2_pos: dace.int32[M + 1],
          A_vals: dace.float32[nnz_A], B: dace.float32[M], C: dace.float32[N]):
    for i in dace.map[0:M]:
        for j in dace.map[A2_pos[i]:A2_pos[i + 1]]:
            D_vals[j] += A_vals[j] * B[i] * C[A2_crd[j]]


@dace.program
def sddva(D_vals: dace.float32[nnz_D], A2_crd: dace.int32[nnz_A], A2_pos: dace.int32[M + 1],
          A_vals: dace.float32[nnz_A], B: dace.float32[M], C: dace.float32[N]):
    for i in dace.map[0:M]:
        for j in dace.map[A2_pos[i]:A2_pos[i + 1]]:
            D_vals[j] += A_vals[j] * (B[i] + C[A2_crd[j]])


def sddmm_baseline(sdfg: dace.SDFG) -> dace.SDFG:
    sdfg.simplify()

    ime, jme, kme = find_map_entry(sdfg, ["i", "j", "k"])

    # MapToForLoop.apply_to(sdfg, map_entry=kme)
    sdfg.apply_transformations_repeated(TrivialTaskletElimination)

    copy_to_gpu(sdfg)
    ime.map.schedule = dace.ScheduleType.GPU_Device
    jme.map.schedule = dace.ScheduleType.GPU_ThreadBlock_Dynamic  # TODO GPU_ThreadBlock fallback

    for e, _ in sdfg.all_edges_recursive():
        if isinstance(e.data, dace.Memlet) and e.data.wcr:
            e.data.wcr_nonatomic = True

    sdfg.validate()


def sddmm_strided(sdfg: dace.SDFG) -> dace.SDFG:
    sdfg.simplify()

    ime, jme, kme = find_map_entry(sdfg, ["i", "j", "k"])

    sdfg.apply_transformations_repeated(TrivialTaskletElimination)

    StripMining.apply_to(sdfg, map_entry=ime, strided=True)
    StripMining.apply_to(sdfg, map_entry=jme, strided=True)

    tile_ime = find_map_entry(sdfg, "tile_i")
    tile_jme = find_map_entry(sdfg, "tile_j")
    sdfg.view()

    MapInterchange.apply_to(sdfg, outer_map_entry=ime, inner_map_entry=tile_jme) # fails

    copy_to_gpu(sdfg)

    tile_ime.map.schedule = dace.ScheduleType.GPU_Device
    tile_jme.map.schedule = dace.ScheduleType.GPU_ThreadBlock_Dynamic

    sdfg.validate()


def sddmm_k_j(sdfg: dace.SDFG) -> dace.SDFG:
    sdfg.simplify()

    ime, jme, kme = find_map_entry(sdfg, ["i", "j", "k"])

    MapInterchange.apply_to(sdfg, outer_map_entry=jme, inner_map_entry=kme)

    # MapToForLoop.apply_to(sdfg, map_entry=jme)
    sdfg.apply_transformations_repeated(TrivialTaskletElimination)

    copy_to_gpu(sdfg)
    ime.map.schedule = dace.ScheduleType.GPU_Device
    kme.map.schedule = dace.ScheduleType.GPU_ThreadBlock_Dynamic  # TODO: GPU_ThreadBlock fall back to seq

    sdfg.validate()


def sddv_baseline(sdfg: dace.SDFG) -> dace.SDFG:
    sdfg.simplify()

    ime, jme = find_map_entry(sdfg, ["i", "j"])

    sdfg.apply_transformations_repeated(TrivialTaskletElimination)

    copy_to_gpu(sdfg)
    ime.map.schedule = dace.ScheduleType.GPU_Device
    jme.map.schedule = dace.ScheduleType.GPU_ThreadBlock_Dynamic

    sdfg.validate()


def sddv_tile_i(sdfg: dace.SDFG) -> dace.SDFG:
    sdfg.simplify()

    ime, jme = find_map_entry(sdfg, ["i", "j"])

    sdfg.apply_transformations_repeated(TrivialTaskletElimination)

    StripMining.apply_to(sdfg, map_entry=jme)

    tile_jme = find_map_entry(sdfg, "tile_j")

    # MapInterchange.apply_to(sdfg, outer_map_entry=ime, inner_map_entry=tile_jme)

    copy_to_gpu(sdfg)
    ime.map.schedule = dace.ScheduleType.GPU_Device
    tile_jme.map.schedule = dace.ScheduleType.GPU_ThreadBlock_Dynamic  # TODO: GPU_ThreadBlock fall back to seq, GPU_ThreadBlock_Dynamic NotImplementedError

    sdfg.validate()


def bench_dace_gpu(sdfg: dace.SDFG,
                   name: str,
                   problem_size: int,
                   nnz: int,
                   density: int,
                   A_vals: cp.ndarray,
                   A_crd: cp.ndarray,
                   A_pos: cp.ndarray,
                   B: cp.ndarray,
                   C: cp.ndarray,
                   D: cp.ndarray,
                   num_warmup=1,
                   num_repeats=3):
    M = N = K = problem_size
    nnz_A = nnz_D = nnz

    #validation
    # D_ref = D.copy()
    # kernels.sddmm_cp[min(65535, len(A_pos) - 1), 128](D_ref, A_crd, A_pos, A_vals, B, C)
    # sdfg(D_vals=D, A2_crd=A_crd, A2_pos=A_pos, A_vals=A_vals, B=B, C=C, M=M, N=N, K=K, nnz_A=nnz_A, nnz_D=nnz_D)
    # assert cp.allclose(D_ref, D)

    sdfg_compiled = sdfg.compile()

    gpu_setup = """D[:] = 0"""
    gpu_stmt = """sdfg_compiled(D_vals=D, A2_crd=A_crd, A2_pos=A_pos, A_vals=A_vals, B=B, C=C, M=M, N=N, K=K, nnz_A=nnz_A, nnz_D=nnz_D)"""

    gpu_runtimes = repeat(gpu_stmt, setup=gpu_setup, repeat=num_warmup + num_repeats, number=1, globals=locals())
    print(f"{name} GPU: {np.median(gpu_runtimes[num_warmup:])} +- {np.std(gpu_runtimes[num_warmup:])}")


def bench_cupy(kernel,
               problem_size: int,
               density: int,
               A_vals: cp.ndarray,
               A_crd: cp.ndarray,
               A_pos: cp.ndarray,
               B: cp.ndarray,
               C: cp.ndarray,
               D: cp.ndarray,
               num_warmup=1,
               num_repeats=20):
    gpu_setup = """D[:] = 0;cp.cuda.get_current_stream().synchronize()"""

    gpu_stmt = """kernel[min(65535, len(A_pos) - 1), 128](D, A_crd, A_pos, A_vals, B, C);cp.cuda.get_current_stream().synchronize()"""

    gpu_runtimes = repeat(gpu_stmt,
                          setup=gpu_setup,
                          repeat=num_warmup + num_repeats,
                          number=1,
                          globals={
                              **locals(),
                              **globals()
                          })
    print(f"{kernel.__name__} GPU: {np.median(gpu_runtimes[num_warmup:])} +- {np.std(gpu_runtimes[num_warmup:])}")


def benchmark_sddmm(problem_size: int, density: int, sdfg: dace.SDFG, schedules, cp_kernels):
    dtype = np.float32
    rng = default_rng(42)
    B_mat = rng.random((problem_size, problem_size), dtype=dtype)
    C_mat = rng.random((problem_size, problem_size), dtype=dtype)
    A = sparse.random(problem_size, problem_size, density=density, format='csr', dtype=dtype, random_state=rng)
    D_vals = np.zeros_like(A.data)

    for sched in schedules:
        sddmm_sdfg = deepcopy(sdfg)
        sched(sddmm_sdfg)
        # sddmm_sdfg.view()
        bench_dace_gpu(sddmm_sdfg, sched.__name__, problem_size, A.nnz, density, cp.asarray(A.data),
                       cp.asarray(A.indices), cp.asarray(A.indptr), cp.asarray(B_mat), cp.asarray(C_mat),
                       cp.asarray(D_vals))
    for kernel in cp_kernels:
        bench_cupy(kernel, problem_size, density, cp.asarray(A.data), cp.asarray(A.indices), cp.asarray(A.indptr),
                cp.asarray(B_mat), cp.asarray(C_mat), cp.asarray(D_vals))


def benchmark_sddv(problem_size: int, density: int, sdfg: dace.SDFG, schedules):
    dtype = np.float32
    rng = default_rng(42)
    B_vec = rng.random(problem_size, dtype=dtype)
    C_vec = rng.random(problem_size, dtype=dtype)
    A = sparse.random(problem_size, problem_size, density=density, format='csr', dtype=dtype, random_state=rng)
    D_vals = np.zeros_like(A.data)

    for sched in schedules:
        sddmm_sdfg = deepcopy(sdfg)
        sched(sddmm_sdfg)
        # sddmm_sdfg.view()
        bench_dace_gpu(sddmm_sdfg, sched.__name__, problem_size, A.nnz, density, cp.asarray(A.data),
                       cp.asarray(A.indices), cp.asarray(A.indptr), cp.asarray(B_vec), cp.asarray(C_vec),
                       cp.asarray(D_vals))
    bench_cupy(kernels.sddvm_cp, problem_size, density, cp.asarray(A.data), cp.asarray(A.indices), cp.asarray(A.indptr),
               cp.asarray(B_vec), cp.asarray(C_vec), cp.asarray(D_vals))


SDDMM_SCHEDULES = [
    # sddmm_baseline,
    # sddmm_k_j,
    sddmm_strided,
]
SDDMM_CP_KERNELS = [kernels.sddmm_cp, kernels.sddmm_cp_shfl,]

SDDV_SCHEDULES = [
    sddv_baseline,
    sddv_tile_i,
]


def profile_sddmm(problem_size: int, density: int, sdfg: dace.SDFG):

    dtype = np.float32
    rng = default_rng(42)
    B_mat = cp.asarray(rng.random((problem_size, problem_size), dtype=dtype))
    C_mat = cp.asarray(rng.random((problem_size, problem_size), dtype=dtype))
    A = sparse.random(problem_size, problem_size, density=density, format='csr', dtype=dtype, random_state=rng)
    A_vals = cp.asarray(A.data)
    A_crd = cp.asarray(A.indices)
    A_pos = cp.asarray(A.indptr)
    D_vals = cp.zeros_like(A.data)

    kernels.sddmm_cp[min(65535, len(A_pos) - 1), 128](D_vals, A_crd, A_pos, A_vals, B_mat, C_mat)
    D_vals[:] = 0
    cp.cuda.get_current_stream().synchronize()

    sddmm_baseline(sdfg)
    sdfg_compiled = sdfg.compile()

    M = N = K = problem_size
    nnz_A = nnz_D = len(A_vals)
    sdfg_compiled(D_vals=D_vals,
                  A2_crd=A_crd,
                  A2_pos=A_pos,
                  A_vals=A_vals,
                  B=B_mat,
                  C=C_mat,
                  M=M,
                  N=N,
                  K=K,
                  nnz_A=nnz_A,
                  nnz_D=nnz_D)
    D_vals[:] = 0

    with dace.profile(repetitions=10, warmup=1) as profiler:
        sdfg_compiled(D_vals=D_vals,
                      A2_crd=A_crd,
                      A2_pos=A_pos,
                      A_vals=A_vals,
                      B=B_mat,
                      C=C_mat,
                      M=M,
                      N=N,
                      K=K,
                      nnz_A=nnz_A,
                      nnz_D=nnz_D)

    print(profiler.times)


if __name__ == '__main__':

    problem_size = 10000
    density = 0.01

    benchmark_sddmm(problem_size, density, sddmm.to_sdfg(simplify=True), SDDMM_SCHEDULES, SDDMM_CP_KERNELS)
    # benchmark_sddv(problem_size, density, sddvm.to_sdfg(simplify=True), SDDV_SCHEDULES)

    # profile_sddmm(problem_size, density, sddmm.to_sdfg(simplify=True))

    #TODO: strided access? how to define block size?