# structure_analysis.py

from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list

from .utils import resolve_mask_for_atoms, MaskType
from .config import GLOBAL_CUTOFFS


def compute_elementwise_coordination_map(
    atoms: Atoms,
    mask: MaskType = None
) -> Dict[str, Dict[str, float]]:
    """
    元素種ごとの平均配位数マップを返す。

    返り値:
        { center_element: { neighbor_element: avg_coordination, ... }, ... }

    例:
        {
            "P": {"O": 3.5, "Fe": 0.1},
            "O": {"P": 1.8, "Zn": 0.1},
        }

    cutoff 判定は GLOBAL_CUTOFFS.pair_cutoffs を使用。
    """

    n_atoms = len(atoms)
    if n_atoms == 0:
        return {}

    symbols = atoms.get_chemical_symbols()
    unique_elements = sorted(set(symbols))

    # resolve mask into numpy bool array or None
    mask_arr = resolve_mask_for_atoms(mask, atoms)

    # neighbor_list 用 cutoffs（原子ごとの最大 cutoff）
    cutoffs = np.zeros(n_atoms, dtype=float)

    for i, si in enumerate(symbols):
        max_cut = 0.0
        for sj in unique_elements:
            pair = tuple(sorted((si, sj)))
            if pair in GLOBAL_CUTOFFS.pair_cutoffs:
                max_cut = max(max_cut, GLOBAL_CUTOFFS.pair_cutoffs[pair])
        cutoffs[i] = max_cut

    # 全近傍候補を検索
    i_list, j_list, d_list = neighbor_list("ijd", atoms, cutoffs)

    # 集計用:
    #   counts[(si, sj)] = si（中心元素）1個あたりの "sj との結合個数" の総和
    #   totals[si] = 元素 si の原子数
    counts = {(si, sj): 0 for si in unique_elements for sj in unique_elements}
    totals = {si: 0 for si in unique_elements}

    # 元素ごとの原子数をカウント
    # totals: mask がある場合は mask True の原子のみ、ない場合は全原子
    for idx, si in enumerate(symbols):
        if mask_arr is None or bool(mask_arr[idx]):
            totals[si] += 1


    # 結合集計
    for i, j, dist in zip(i_list, j_list, d_list):
        si = symbols[i]
        sj = symbols[j]
        pair = tuple(sorted((si, sj)))
        cutoff = GLOBAL_CUTOFFS.pair_cutoffs.get(pair)

        if cutoff is None:
            continue

        if dist > cutoff:
            continue

        # i を中心としたとき（i が mask に含まれるならカウント）
        if mask_arr is None or bool(mask_arr[i]):
            counts[(si, sj)] += 1

        # j を中心としたとき（j が mask に含まれるならカウント）
        # if mask_arr is None or bool(mask_arr[j]):
        #     counts[(sj, si)] += 1


    # 平均配位数マップを作成（selected centers の数で割る）
    result: Dict[str, Dict[str, float]] = {}
    for si in unique_elements:
        result[si] = {}
        n_si = totals.get(si, 0)
        for sj in unique_elements:
            if n_si == 0:
                avg = 0.0
            else:
                avg = counts[(si, sj)] / n_si
            result[si][sj] = float(avg)
    return result
