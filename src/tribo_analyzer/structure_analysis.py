# structure_analysis.py

from __future__ import annotations
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list
from ase.data import chemical_symbols

from .utils import resolve_mask_for_atoms, MaskType
from .config import GLOBAL_CUTOFFS
from .config import get_pair_cutoff


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

# --- 公開用 dataclass ---
@dataclass
class BondDensityResult:
    n_bonds: int
    volume_A3: float
    bond_density_per_A3: float


# --- ヘルパー関数 ---
def _validate_element(symbol: str) -> None:
    if symbol not in set(chemical_symbols[1:]):
        raise ValueError(f"Invalid element symbol: {symbol!r}")


def _approx_volume_for_nonperiodic(atoms: Atoms, pad: float = 2.0) -> float:
    """
    非周期系の簡易ボリューム見積り（Å^3）。
    bounding box の各辺に pad を足した直方体体積を返す。
    pad には通常 cutoff を渡すと安全。
    """
    pos = atoms.get_positions()
    mins = pos.min(axis=0)
    maxs = pos.max(axis=0)
    lengths = np.maximum(maxs - mins, 1e-8)
    lengths = lengths + 2.0 * pad
    return float(np.prod(lengths))


# --- メイン関数 ---
def bond_density(
    atoms: Atoms,
    element_a: str,
    element_b: str,
    cutoff: Optional[float] = None,
    use_config_cutoff: bool = True,
    assume_volume_padding: Optional[float] = None,
) -> BondDensityResult:
    """
    指定元素ペアの "結合数 / 体積(Å^3)" を返す。

    Parameters
    ----------
    atoms : ase.Atoms
        対象構造（PBC対応）。
    element_a, element_b : str
        元素記号（例: "P", "O"）。ASEのchemical_symbolsに準拠。
    cutoff : float | None
        結合判定距離(Å)。Noneかつ use_config_cutoff=True の場合は config.get_pair_cutoff() を使う。
    use_config_cutoff : bool
        True の場合、cutoff が None なら config から取得を試みる。
    assume_volume_padding : float | None
        非周期系の体積推定で bounding-box に足すパッド。None なら cutoff を使う（または 2.0Å のデフォルト）。

    Returns
    -------
    BondDensityResult
        n_bonds, volume_A3, bond_density_per_A3
    """
    # validate element symbols
    _validate_element(element_a)
    _validate_element(element_b)

    # determine cutoff
    if cutoff is None and use_config_cutoff:
        cutoff = get_pair_cutoff(element_a, element_b)
    if cutoff is None:
        raise ValueError(f"No cutoff specified for pair ({element_a}, {element_b}). Provide cutoff or set in config.")
    cutoff = float(cutoff)

    # build per-atom cutoffs (neighbor_list API)
    cutoffs = np.full(len(atoms), cutoff, dtype=float)

    # neighbor_list で i,j,d を取得（PBC対応）
    i_list, j_list, d_list = neighbor_list("ijd", atoms, cutoffs)

    # bond count (unique pairs i<j)
    n_bonds = 0
    for i, j, d in zip(i_list, j_list, d_list):
        if i >= j:
            continue
        si = atoms[i].symbol
        sj = atoms[j].symbol
        if (si == element_a and sj == element_b) or (si == element_b and sj == element_a):
            n_bonds += 1

    # volume: try atoms.get_volume(); if zero or negative -> estimate
    vol = float(atoms.get_volume())
    if vol <= 0.0:
        pad = cutoff if assume_volume_padding is None else float(assume_volume_padding)
        vol = _approx_volume_for_nonperiodic(atoms, pad=pad)

    density = float(n_bonds) / vol if vol > 0.0 else float("nan")

    return BondDensityResult(n_bonds=int(n_bonds), volume_A3=float(vol), bond_density_per_A3=float(density))
