# structure_analysis.py

from __future__ import annotations
from typing import Dict, Tuple, FrozenSet, Optional
from dataclasses import dataclass
from collections import Counter

import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list
from ase.data import chemical_symbols
import ase.units as units

from .utils import resolve_mask_for_atoms, MaskType
from .config import GLOBAL_CUTOFFS, _normalize_pair
from .config import get_pair_cutoff
from .phosphate import classify_phosphorus_qi


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

    # 最大cutoff
    max_cutoff = 0.0

    for i, si in enumerate(symbols):
        for sj in unique_elements:
            pair = _normalize_pair(si, sj)
            if pair in GLOBAL_CUTOFFS.pair_cutoffs:
                max_cutoff = max(max_cutoff, GLOBAL_CUTOFFS.pair_cutoffs[pair])


    # 全近傍候補を検索
    i_list, j_list, d_list = neighbor_list("ijd", atoms, max_cutoff)

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
        pair = _normalize_pair(si, sj)
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


def density(atoms):
    # total mass in atomic mass unit (amu)
    total_mass_amu = atoms.get_masses().sum()

    # cell volume in Å^3
    volume_ang3 = atoms.get_volume()

    if volume_ang3 <= 0:
        raise ValueError("Cell volume must be positive to compute density.")

    # unit conversions
    amu_to_g = 1.66053906660e-24   # g
    ang3_to_cm3 = 1.0e-24         # cm^3

    total_mass_g = total_mass_amu * amu_to_g
    volume_cm3 = volume_ang3 * ang3_to_cm3

    density = total_mass_g / volume_cm3
    return density

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




CoordinationKey = FrozenSet[Tuple[str, int]]


def coordination_environment_distribution(
    atoms: Atoms,
    center_element: str,
    mask: MaskType = None,
) -> Dict[CoordinationKey, int]:
    """
    指定元素の配位環境分布を返す。

    Returns
    -------
    dict:
        {
            frozenset({("Zn", 2)}): 26,
            frozenset({("P", 2)}): 30,
            frozenset({("Zn", 1), ("P", 1)}): 19,
        }
    """

    _validate_element(center_element)

    symbols = atoms.get_chemical_symbols()
    mask_arr = resolve_mask_for_atoms(mask, atoms)

    # --- 最大カットオフを取得 ---
    max_cutoff = 0.0
    for s in set(symbols):
        pair = _normalize_pair(center_element, s)
        c = GLOBAL_CUTOFFS.pair_cutoffs.get(pair)
        if c is not None:
            max_cutoff = max(max_cutoff, c)

    if max_cutoff <= 0.0:
        return {}

    # --- neighbor list ---
    i_list, j_list, d_list = neighbor_list("ijd", atoms, max_cutoff)

    # --- 各 center 原子の配位カウント ---
    env_map: Dict[int, Dict[str, int]] = {}

    for i, j, d in zip(i_list, j_list, d_list):
        si = symbols[i]
        sj = symbols[j]

        if si != center_element:
            continue

        if mask_arr is not None and not bool(mask_arr[i]):
            continue

        cutoff = GLOBAL_CUTOFFS.pair_cutoffs.get(_normalize_pair(si, sj))
        if cutoff is None or d > cutoff:
            continue

        env_map.setdefault(i, {})
        env_map[i][sj] = env_map[i].get(sj, 0) + 1

    # --- 配位環境を Counter にまとめる ---
    dist: Counter[CoordinationKey] = Counter()

    for env in env_map.values():
        key = frozenset((elem, int(cnt)) for elem, cnt in env.items())
        dist[key] += 1

    return dict(dist)

def znp_coordinated_oxygen_bo_distribution(
    atoms: Atoms,
    *,
    zn_symbol: str = "Zn",
    o_symbol: str = "O",
    p_symbol: str = "P",
    cutoff_zn_o: float | None = None,
    cutoff_p_o: float | None = None,
    mask: MaskType = None,
    normalize: bool = True,   # ← 追加
) -> Dict[str, float]:
    """
    ZnとPに配位しているOについて、そのOがBO/NBOかを分類した分布を返す。

    Parameters
    ----------
    normalize : bool
        True の場合は割合、False の場合は個数

    Returns
    -------
    dict:
        {
            "BO": float,
            "NBO": float,
            "Other": float,
        }
    """

    symbols = atoms.get_chemical_symbols()
    mask_arr = resolve_mask_for_atoms(mask, atoms)

    cutoff_zn_o = cutoff_zn_o or get_pair_cutoff(zn_symbol, o_symbol)
    cutoff_p_o = cutoff_p_o or get_pair_cutoff(p_symbol, o_symbol)

    max_cutoff = max(cutoff_zn_o, cutoff_p_o)

    i_list, j_list, d_list = neighbor_list("ijd", atoms, max_cutoff)

    # --- OごとのZn配位判定 & P配位数カウント ---
    o_indices = [i for i, s in enumerate(symbols) if s == o_symbol]

    zn_neighbors = {i: 0 for i in o_indices}
    p_neighbors = {i: 0 for i in o_indices}

    for i, j, d in zip(i_list, j_list, d_list):
        si = symbols[i]
        sj = symbols[j]

        if si == o_symbol:
            if mask_arr is not None and not mask_arr[i]:
                continue

            if sj == zn_symbol and d <= cutoff_zn_o:
                zn_neighbors[i] += 1

            if sj == p_symbol and d <= cutoff_p_o:
                p_neighbors[i] += 1

    # --- 分類 ---
    counts = {"BO": 0, "NBO": 0, "Other": 0}

    for i in o_indices:
        if zn_neighbors[i] == 0:
            continue

        n_p = p_neighbors[i]
        if n_p == 0:
            continue

        if n_p == 2:
            counts["BO"] += 1
        elif n_p == 1:
            counts["NBO"] += 1
        else:
            counts["Other"] += 1

    # --- 正規化 ---
    if normalize:
        total = sum(counts.values())
        if total > 0:
            return {k: v / total for k, v in counts.items()}
        else:
            return {k: 0.0 for k in counts}

    # 個数そのまま
    return {k: float(v) for k, v in counts.items()}




def zn_o_p_qi_distribution(
    atoms: Atoms,
    *,
    zn_symbol: str = "Zn",
    p_symbol: str = "P",
    o_symbol: str = "O",
    cutoff_zn_o: float | None = None,
    cutoff_p_o: float | None = None,
    mask: MaskType = None,
    normalize: bool = True,
) -> Dict[str, float]:
    """
    Zn–O–P 状態にある P 原子の Qi 分布を返す。

    Returns
    -------
    dict:
        {
            "Q0": ...,
            "Q1": ...,
            ...
        }
    """

    symbols = atoms.get_chemical_symbols()
    mask_arr = resolve_mask_for_atoms(mask, atoms)

    cutoff_zn_o = cutoff_zn_o or get_pair_cutoff(zn_symbol, o_symbol)
    cutoff_p_o = cutoff_p_o or get_pair_cutoff(p_symbol, o_symbol)

    max_cutoff = max(cutoff_zn_o, cutoff_p_o)

    i_list, j_list, d_list = neighbor_list("ijd", atoms, max_cutoff)

    # --- Oごとの接続情報 ---
    o_to_p = {}
    o_to_zn = {}

    for i, j, d in zip(i_list, j_list, d_list):
        si = symbols[i]
        sj = symbols[j]

        if si == o_symbol:
            if mask_arr is not None and not mask_arr[i]:
                continue

            # O-P
            if sj == p_symbol and d <= cutoff_p_o:
                o_to_p.setdefault(i, []).append(j)

            # O-Zn
            if sj == zn_symbol and d <= cutoff_zn_o:
                o_to_zn.setdefault(i, []).append(j)

    # --- Zn-O-P を満たす P を抽出 ---
    target_p_indices = set()

    for o_idx in o_to_p:
        if o_idx not in o_to_zn:
            continue  # Znと繋がってないOは除外

        for p_idx in o_to_p[o_idx]:
            target_p_indices.add(p_idx)

    if len(target_p_indices) == 0:
        return {f"Q{i}": 0.0 for i in range(5)}

    # --- Qi を計算（全体に対して） ---
    classify_phosphorus_qi(
        atoms,
        p_symbol=p_symbol,
        o_symbol=o_symbol,
        cutoff_po=cutoff_p_o,
        ensure_bo=True,
        set_array=True,
    )

    qi_array = atoms.arrays["P_qi"]

    # --- 対象Pのみ抽出 ---
    labels = [
        str(qi_array[i])
        for i in target_p_indices
        if qi_array[i] not in (None, "")
    ]

    counter = Counter(labels)

    # --- Q0〜Q4を固定で作成 ---
    result: Dict[str, float] = {}
    total = sum(counter.values())

    for q in range(5):
        key = f"Q{q}"
        cnt = counter.get(key, 0)

        if normalize:
            result[key] = cnt / total if total > 0 else 0.0
        else:
            result[key] = float(cnt)

    return result

from typing import Dict
from ase import Atoms
from ase.neighborlist import neighbor_list

from .utils import resolve_mask_for_atoms, MaskType
from .config import get_pair_cutoff


def zn_bridging_p_o_zn_o_p(
    atoms: Atoms,
    *,
    zn_symbol: str = "Zn",
    o_symbol: str = "O",
    p_symbol: str = "P",
    cutoff_zn_o: float | None = None,
    cutoff_p_o: float | None = None,
    mask: MaskType = None,
    normalize: bool = True,
) -> Dict[str, float]:
    """
    P-O-Zn-O-P 構造を形成している Zn の数または割合を返す。

    Returns
    -------
    dict:
        {
            "bridging": ...,
            "non_bridging": ...,
        }
    """

    symbols = atoms.get_chemical_symbols()
    mask_arr = resolve_mask_for_atoms(mask, atoms)

    cutoff_zn_o = cutoff_zn_o or get_pair_cutoff(zn_symbol, o_symbol)
    cutoff_p_o = cutoff_p_o or get_pair_cutoff(p_symbol, o_symbol)

    max_cutoff = max(cutoff_zn_o, cutoff_p_o)

    i_list, j_list, d_list = neighbor_list("ijd", atoms, max_cutoff)

    # --- ZnごとのO配位 ---
    zn_to_o = {}
    # --- OごとのP配位 ---
    o_to_p = {}

    for i, j, d in zip(i_list, j_list, d_list):
        si = symbols[i]
        sj = symbols[j]

        # Zn-O
        if si == zn_symbol and sj == o_symbol and d <= cutoff_zn_o:
            if mask_arr is not None and not mask_arr[i]:
                continue
            zn_to_o.setdefault(i, []).append(j)

        # O-P
        if si == o_symbol and sj == p_symbol and d <= cutoff_p_o:
            o_to_p.setdefault(i, []).append(j)

    # --- Znごとに判定 ---
    bridging = 0
    non_bridging = 0

    for zn_idx, o_list in zn_to_o.items():

        # Znに配位しているOのうち、Pと結合しているものを数える
        p_connected_o = 0

        for o_idx in o_list:
            if o_idx in o_to_p and len(o_to_p[o_idx]) > 0:
                p_connected_o += 1

        if p_connected_o >= 2:
            bridging += 1
        else:
            non_bridging += 1

    total = bridging + non_bridging

    if normalize:
        if total > 0:
            return {
                "bridging": bridging / total,
                "non_bridging": non_bridging / total,
            }
        else:
            return {
                "bridging": 0.0,
                "non_bridging": 0.0,
            }

    return {
        "bridging": float(bridging),
        "non_bridging": float(non_bridging),
    }
