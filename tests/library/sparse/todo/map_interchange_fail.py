import dace
import numpy as np
from math import sqrt
from dace.transformation.dataflow.map_interchange import MapInterchange
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
@dace.program
def sched_sddmm0compute(A_vals: dace.float64[size_A_vals], B2_crd: dace.int32[size_B2_crd], B2_pos: dace.int32[size_B2_pos], B_vals: dace.float64[size_B_vals], C_vals: dace.float64[size_C_vals], D_vals: dace.float64[size_D_vals]):

    for i in dace.map[0: C1_dimension: 1]:
        for j in dace.map[0: D1_dimension: 1]:
            jC = i * C2_dimension + j
            for kB in dace.map[B2_pos[i]: B2_pos[(i + 1)]: 1]:
                k = B2_crd[kB]
                kD = j * D2_dimension + k
                A_vals[kB] = A_vals[kB] + (B_vals[kB] * C_vals[jC]) * D_vals[kD]
    
    # for i in dace.map[0: C1_dimension: 1]:
    #     for j in dace.map[0: D1_dimension: 1]:
    #         for kB in dace.map[B2_pos[i]: B2_pos[(i + 1)]: 1]:
    #             A_vals[kB] = A_vals[kB] + (B_vals[kB] * C_vals[i * C2_dimension + j]) * D_vals[j * D2_dimension + B2_crd[kB]]
            
        
    

sdfg = sched_sddmm0compute.to_sdfg()
# sdfg.apply_transformations_repeated(MapInterchange)
sdfg.view()
ome, ime = None, None
for state in sdfg.states():
    for node in state.nodes():
        if isinstance(node, dace.sdfg.nodes.MapEntry):
            if node.map.params[0] == 'j':
                ome = node
            elif node.map.params[0] == 'kB':
                ime = node
assert ome is not None and ime is not None
MapInterchange.apply_to(sdfg, outer_map_entry=ome, inner_map_entry=ime)