# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
from __future__ import annotations
from dataclasses import dataclass
import warnings
from typing import List, Dict, Any
from enum import Enum
import dace
from dace.transformation.interstate.loop_to_map import LoopToMap
import os
import subprocess


class TensorModeType(str, Enum):
    Dense = "d"
    Sparse = "s"
    Sparse_Not_Unique = "u"
    Singleton = "q"
    Singleton_Not_Unique = "c"
    Singleton_Padded = "p"

    def __str__(self) -> str:
        return self.value


@dataclass
class TensorFormat:
    mode_types: List[TensorModeType] = None
    mode_ordering: List[int] = None

    def defined(self) -> bool:
        return self.mode_types is not None and self.mode_ordering is not None

    def __str__(self) -> str:
        ret = ""
        if self.mode_types is not None:
            ret += ':' + ''.join([str(m) for m in self.mode_types])

        if self.mode_ordering is not None:
            ret += ':' + ','.join([str(m) for m in self.mode_ordering])

        return ret


CSR = TensorFormat([TensorModeType.Dense, TensorModeType.Sparse])
CSC = TensorFormat([TensorModeType.Dense, TensorModeType.Sparse], [1, 0])
CSF = TensorFormat([TensorModeType.Sparse, TensorModeType.Sparse, TensorModeType.Sparse])
DCSR = TensorFormat([TensorModeType.Sparse, TensorModeType.Sparse])
DCSC = TensorFormat([TensorModeType.Sparse, TensorModeType.Sparse], [1, 0])
COO = TensorFormat([TensorModeType.Sparse_Not_Unique, TensorModeType.Singleton])
SPARSE_VECTOR = TensorFormat([TensorModeType.Sparse])

class TacoProxy:
    def __init__(self,
                 expr: str,
                 output_dir: str = "/tmp",
                 name: str = 'Taco',
                 debug: bool = False,
                 assemble_while_compute: bool = False,
                 simplify_sdfg: bool = True,
                 force_dace_map: bool = True) -> None:
        self.expr = expr
        self.output_dir = output_dir
        self.prefix = name
        self.debug = debug
        self.assemble_while_compute = assemble_while_compute
        self.simplify_sdfg = simplify_sdfg
        self.force_dace_map = force_dace_map
        # TODO: replace this
        self.taco_exec = "/home/tiachen/testspace/taco/build/bin/dace_taco"
        self.taco_args = []

        # make a directory for the intermediate files
        os.makedirs(self.output_dir, exist_ok=True)

        # intermediate files
        self.compute_filename = self.prefix + "compute.py"
        self.assemble_filename = self.prefix + "assemble.py"
        self.iteration_graph_filename = self.prefix + "iterationGraph.dot"
        self.concrete_notation_filename = self.prefix + "concrete.txt"

        self.compute_sdfg_filename = self.prefix + "compute.sdfg"

        self.formats = {}
        self.transformations = []

        self.sampled_replace_dict = {}

        self.gen_sdfg = None

    def set_formats(self, formats: Dict[str, TensorFormat]) -> TacoProxy:
        for k, v in formats.items():
            if not isinstance(k, str) or not isinstance(v, TensorFormat):
                warnings.warn("Invalid format: " + str(k) + " " + str(v))
            self.formats[k] = v

        return self
    
    def sampled_replace(self, src_tensor: str, target_tensor: str) ->TacoProxy:
        # special transformation for sddmm, see help info of dace_taco
        self.sampled_replace_dict[src_tensor] = target_tensor

    def pos(self, i: str, ipos: str, tensor: str) -> TacoProxy:
        self.transformations.append("\"pos(" + i + "," + ipos + "," + tensor + ")\"")
        return self

    def fuse(self, i: str, j: str, f: str) -> TacoProxy:
        self.transformations.append("\"fuse(" + i + "," + j + "," + f + ")\"")
        return self

    def split(self, i: str, i0: str, i1: str, factor: int) -> TacoProxy:
        self.transformations.append("\"split(" + i + "," + i0 + "," + i1 + "," + str(factor) + ")\"")
        return self

    def precompute(self, expr: str, i: str, iw: str) -> TacoProxy:
        self.transformations.append("\"precompute(" + expr + "," + i + "," + iw + ")\"")
        return self

    def reorder(self, modes: List[str]) -> TacoProxy:
        self.transformations.append("\"reorder(" + ','.join(modes) + ")\"")
        return self

    # TODO: bound, unroll, parallelize

    def set_transformations(self, transformations: List[Any]) -> TacoProxy:
        for trans_args in transformations:
            if trans_args[0] == 'pos':
                self.pos(*trans_args[1:])
            elif trans_args[0] == 'fuse':
                self.fuse(*trans_args[1:])
            elif trans_args[0] == 'split':
                self.split(*trans_args[1:])
            elif trans_args[0] == 'precompute':
                self.precompute(*trans_args[1:])
            elif trans_args[0] == 'reorder':
                self.reorder(trans_args[1:])
            elif trans_args[0] == 'sampled_replace':
                self.sampled_replace(*trans_args[1:])
            else:
                warnings.warn(f'Unidentified transformation {trans_args[0]}, ignored.')

        return self

    def generate(self) -> None:
        self.taco_args = []
        self.taco_args.append(self.taco_exec)
        self.taco_args.append(' \"' + self.expr + '\" ')

        for k, v in self.formats.items():
            self.taco_args.append("-f=" + k + str(v))

        for trans in self.transformations:
            self.taco_args.append("-s=" + trans)

        if self.assemble_while_compute:
            self.taco_args.append("-c")

        self.taco_args.append("-O=" + self.output_dir)

        self.taco_args.append("-prefix=" + self.prefix)

        if self.force_dace_map:
            self.taco_args.append("-force-map")

        if len(self.sampled_replace_dict) == 1:
            # only support one pair for now
            for src, target in self.sampled_replace_dict.items():
                self.taco_args.append(f'-sampled-replace={src}:{target}')

        self.taco_args.append("-write-compute")
        self.taco_args.append("-write-assemble")

        if self.debug:
            self.taco_args.append("-write-concrete")
            self.taco_args.append("-write-iteration-graph")

        if self.debug:
            # print the taco command
            print(" ".join(self.taco_args))

        # run taco
        try:
            subprocess.run(" ".join(self.taco_args), check=True, shell=True)
        except subprocess.CalledProcessError as e:
            print(e)
            warnings.warn("Taco failed to generate code.")
            return

        # run DaCe frontend on the generated .py files
        try:
            subprocess.run(["python3", os.path.join(self.output_dir, self.compute_filename)], check=True)
        except subprocess.CalledProcessError as e:
            print(e)
            warnings.warn("Failed to generate compute SDFG.")
            return

        # load the generated SDFG
        self.gen_sdfg = dace.SDFG.from_file(os.path.join(self.output_dir, self.compute_sdfg_filename))

        if self.simplify_sdfg:
            self.gen_sdfg.apply_transformations_repeated(LoopToMap)
            self.gen_sdfg.save(os.path.join(self.output_dir, self.compute_sdfg_filename))

    def get_sdfg(self) -> dace.SDFG:
        return self.gen_sdfg


if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".dacecache")
    taco = TacoProxy("A(i,j)=B(i,k)*C(k,j)", output_dir=output_dir, debug=True)
    taco.set_formats({"B": CSR})
    taco.generate()
    sdfg = taco.get_sdfg()
    sdfg.view()

# @dace.library.node
# class TACO(dace.sdfg.nodes.LibraryNode):

#     # Global properties
#     implementations = {"orignial": ExpandTacoOrignial}
#     default_implementation = "orignial"

#     # Object fields
