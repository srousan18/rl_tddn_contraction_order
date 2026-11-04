"""TDD table of a tensor network and a contraction cost estimate."""

from __future__ import annotations

import math


def tdd_table(tn) -> dict:
    """Label -> ``[tdd, tensor_count, gate_names]`` for every tensor of ``tn``.

    Labels follow the contraction graph (``tdd_<i>``). The TDDs are built in
    the active TDD session.
    """

    return {f"tdd_{i}": [tn.tensors[i].tdd(), 1, [tn.tensors[i].name]]
            for i in range(len(tn.tensors))}


def _dim(tdd, key, default=2):
    return tdd.key_width.get(key, default)


def estimate_tdd_contraction_flops(tdd1, tdd2, log10=True):
    """Dense real-valued FLOP estimate for contracting two TDDs.

    Counts one complex multiply (6 real FLOPs) and one complex add (2 real
    FLOPs) per summed term of every output element. Returns log10 of the
    count when ``log10`` is true.
    """

    keys1 = {idx.key for idx in tdd1.index_set}
    keys2 = {idx.key for idx in tdd2.index_set}
    contracted = keys1 & keys2
    out1 = keys1 - contracted
    out2 = keys2 - contracted

    contraction_size = math.prod(_dim(tdd1, k) for k in contracted)
    output_size = math.prod(_dim(tdd1, k) for k in out1)
    output_size *= math.prod(_dim(tdd2, k) for k in out2)

    complex_mults = contraction_size * output_size
    complex_adds = (contraction_size - 1) * output_size
    real_flops = 6 * complex_mults + 2 * complex_adds
    return math.log10(real_flops) if log10 else real_flops
