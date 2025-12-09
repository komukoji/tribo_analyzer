from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, List, Dict, Optional, Sequence
from collections import Counter

import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list

from .utils import resolve_mask_for_atoms, MaskType
from .config import get_pair_cutoff

OType = Literal["BO", "NBO", "other"]


@dataclass
class OxygenClassification:
    """単一O原子のBO/NBO分類結果"""
    index: int          # Atoms 内インデックス
    n_p_neighbors: int  # cutoff以内のP原子数
    kind: OType         # "BO" / "NBO" / "other"


def classify_oxygen_bo_nbo(
    atoms: Atoms,
    p_symbol: str = "P",
    o_symbol: str = "O",
    cutoff_po: float | None = None,
    set_array: bool = True,
) -> List[OxygenClassification]:
    """P-O近接数に基づきO原子を BO / NBO / other に分類する。

    Parameters
    ----------
    atoms : Atoms
        対象構造。
    p_symbol : str
        リン元素記号（デフォルト "P"）。
    o_symbol : str
        酸素元素記号（デフォルト "O"）。
    cutoff_po : float | None
        P-O結合とみなす距離(Å)。
        None の場合は config.get_pair_cutoff(p_symbol, o_symbol) を使用。
    set_array : bool
        True の場合、atoms.arrays["O_role"] に "BO"/"NBO"/"other" を書き込む。

    Returns
    -------
    List[OxygenClassification]
        各O原子の分類結果。
    """
    # カットオフ決定（未指定なら config から）
    if cutoff_po is None:
        cutoff_po = get_pair_cutoff(p_symbol, o_symbol)
        if cutoff_po is None:
            raise ValueError(
                f"P-Oカットオフが未設定です: get_pair_cutoff({p_symbol!r}, {o_symbol!r}) -> None"
            )

    # 対象O原子のインデックス
    o_indices = [i for i, a in enumerate(atoms) if a.symbol == o_symbol]
    if not o_indices:
        return []

    o_index_set = set(o_indices)

    # 全原子に同じ cutoff_po を設定して neighbor_list
    i_list, j_list = neighbor_list("ij", atoms, cutoff_po)

    # Oごとの P 近傍数カウント
    p_neighbor_counts = {oi: 0 for oi in o_indices}

    for i, j in zip(i_list, j_list):
        sj = atoms[j].symbol

        # O(i) - P(j)
        if i in o_index_set and sj == p_symbol:
            # if i==0:
            #     print(j)
            p_neighbor_counts[i] += 1

    # 分類 & atoms.arrays への書き込み
    roles = np.empty(len(atoms), dtype=object)
    roles[:] = ""

    results: List[OxygenClassification] = []

    for oi in o_indices:
        n_p = p_neighbor_counts.get(oi, 0)
        if n_p == 2:
            kind: OType = "BO"
        elif n_p == 1:
            kind = "NBO"
        else:
            kind = "other"

        roles[oi] = kind
        results.append(
            OxygenClassification(
                index=oi,
                n_p_neighbors=n_p,
                kind=kind,
            )
        )

    if set_array:
        atoms.set_array("O_role", roles)

    return results


@dataclass
class BONBOSummary:
    """系全体における BO/NBO 統計"""
    n_bo: int
    n_nbo: int
    n_other: int
    n_oxygen: int
    frac_bo: float
    frac_nbo: float
    frac_other: float
    bo_nbo_ratio: float | None  # ← BO/NBO比を追加


def summarize_oxygen_bo_nbo(
    atoms: Atoms,
    p_symbol: str = "P",
    o_symbol: str = "O",
    cutoff_po: float | None = None,
    ensure_classify: bool = True,
    mask: Optional[Sequence[bool]] = None,
) -> BONBOSummary:
    """系全体の BO / NBO / other の個数と割合、BO/NBO比を計算する。"""

    # 必要なら BO/NBO 判定を実行
    if ensure_classify or "O_role" not in atoms.arrays:
        classify_oxygen_bo_nbo(
            atoms,
            p_symbol=p_symbol,
            o_symbol=o_symbol,
            cutoff_po=cutoff_po,
            set_array=True,
        )

    if "O_role" not in atoms.arrays:
        raise ValueError('atoms.arrays["O_role"] が存在しません。')

    # mask処理
    mask_arr = resolve_mask_for_atoms(mask, atoms)

    roles = atoms.arrays["O_role"]

    # 全原子について O であるかどうかのマスク
    o_is = np.array([a.symbol == o_symbol for a in atoms], dtype=bool)

    # 最終的な対象 O のマスク = (Oである) AND (mask が与えられていれば mask が True)
    if mask_arr is None:
        final_o_mask = o_is
    else:
        final_o_mask = o_is & mask_arr

    o_roles = roles[final_o_mask]
    n_oxygen = int(o_roles.size)

    if n_oxygen == 0:
        return BONBOSummary(
            n_bo=0,
            n_nbo=0,
            n_other=0,
            n_oxygen=0,
            frac_bo=0.0,
            frac_nbo=0.0,
            frac_other=0.0,
            bo_nbo_ratio=None,
        )

    c = Counter(o_roles)
    n_bo = int(c.get("BO", 0))
    n_nbo = int(c.get("NBO", 0))
    n_other = int(c.get("other", 0))

    frac_bo = n_bo / n_oxygen
    frac_nbo = n_nbo / n_oxygen
    frac_other = n_other / n_oxygen

    # BO/NBO比（NBO=0 の場合は None）
    bo_nbo_ratio = (n_bo / n_nbo) if n_nbo > 0 else None

    return BONBOSummary(
        n_bo=n_bo,
        n_nbo=n_nbo,
        n_other=n_other,
        n_oxygen=n_oxygen,
        frac_bo=frac_bo,
        frac_nbo=frac_nbo,
        frac_other=frac_other,
        bo_nbo_ratio=bo_nbo_ratio,
    )


@dataclass
class QiClassification:
    """単一P原子の Qi 分類結果"""
    index: int        # Atoms 内インデックス
    n_bo: int         # 結合している BO の数
    label: str        # "Q0", "Q1", "Q2", ...


def classify_phosphorus_qi(
    atoms: Atoms,
    p_symbol: str = "P",
    o_symbol: str = "O",
    cutoff_po: float | None = None,
    ensure_bo: bool = True,
    set_array: bool = True,
) -> List[QiClassification]:
    """P原子を Qi (Q0, Q1, Q2, ...) で分類する。

    定義:
      - まず O の BO/NBO を決定（ensure_bo=True の場合）
      - 各Pについて、cutoff_po 以内にいる "BO" な O の個数を n_bo とし Q{n_bo} と分類

    Parameters
    ----------
    atoms : Atoms
        対象構造。
    p_symbol : str
        リン元素記号（デフォルト "P"）。
    o_symbol : str
        酸素元素記号（デフォルト "O"）。
    cutoff_po : float | None
        P-O結合とみなす距離(Å)。Noneなら config の P-O カットオフを使用。
    ensure_bo : bool
        True の場合、先に classify_oxygen_bo_nbo を呼んで O_role を更新する。
        False の場合、既存 atoms.arrays["O_role"] を利用する前提。
    set_array : bool
        True の場合、atoms.arrays["P_qi"] に "Q0","Q1",... を書き込む。

    Returns
    -------
    List[QiClassification]
        各P原子の Qi 分類。
    """
    # cutoff の決定
    if cutoff_po is None:
        cutoff_po = get_pair_cutoff(p_symbol, o_symbol)
        if cutoff_po is None:
            raise ValueError(
                f"P-Oカットオフが未設定です: get_pair_cutoff({p_symbol!r}, {o_symbol!r}) -> None"
            )

    # 必要なら BO/NBO を更新
    if ensure_bo:
        classify_oxygen_bo_nbo(
            atoms,
            p_symbol=p_symbol,
            o_symbol=o_symbol,
            cutoff_po=cutoff_po,
            set_array=True,
        )

    # O_role が無ければエラー
    if "O_role" not in atoms.arrays:
        raise ValueError(
            'atoms.arrays["O_role"] が存在しません。先に classify_oxygen_bo_nbo を実行してください。'
        )

    o_roles = atoms.arrays["O_role"]

    # 対象P原子
    p_indices = [i for i, a in enumerate(atoms) if a.symbol == p_symbol]
    if not p_indices:
        return []

    p_index_set = set(p_indices)

    # neighbor_list で P-O 近傍を取得
    i_list, j_list = neighbor_list("ij", atoms, cutoff_po)

    # Pごとの BO接続数
    bo_counts = {pi: 0 for pi in p_indices}

    for i, j in zip(i_list, j_list):
        si = atoms[i].symbol
        sj = atoms[j].symbol

        # P(i) - O(j)
        if i in p_index_set and sj == o_symbol and o_roles[j] == "BO":
            bo_counts[i] += 1

    # 分類＋ atoms.arrays への書き込み
    qi_array = np.empty(len(atoms), dtype=object)
    qi_array[:] = ""

    results: List[QiClassification] = []
    for pi in p_indices:
        n_bo = bo_counts.get(pi, 0)
        label = f"Q{n_bo}"
        qi_array[pi] = label
        results.append(
            QiClassification(
                index=pi,
                n_bo=n_bo,
                label=label,
            )
        )

    if set_array:
        atoms.set_array("P_qi", qi_array)

    return results

@dataclass
class QiSummary:
    """系全体における Qi 分布"""
    n_p: int                         # 対象P原子数
    counts: dict[str, int]           # 各Qiの個数 {"Q0": ..., "Q1": ..., ...}
    fractions: dict[str, float]      # 各Qiの割合（P全体に対する）


def summarize_phosphorus_qi(
    atoms: Atoms,
    p_symbol: str = "P",
    o_symbol: str = "O",
    cutoff_po: float | None = None,
    ensure_qi: bool = True,
    mask: bool = None
) -> QiSummary:
    """系全体の P 原子について Qi (Q0, Q1, Q2, ...) の割合を求める。

    Parameters
    ----------
    atoms : Atoms
        対象構造。
    p_symbol : str
        リンの元素記号（デフォルト "P"）。
    o_symbol : str
        酸素の元素記号（デフォルト "O"）。
    cutoff_po : float | None
        P-O結合カットオフ(Å)。Noneなら config の P-O を使用。
    ensure_qi : bool
        True の場合、内部で classify_phosphorus_qi を実行して P_qi を更新する。
        False の場合、既存 atoms.arrays["P_qi"] を利用する前提。

    Returns
    -------
    QiSummary
        各Qi種の個数と、P全体に対する割合。
    """
    # 必要なら Qi を再計算（内部で BO/NBO も処理される）
    if ensure_qi or "P_qi" not in atoms.arrays:
        classify_phosphorus_qi(
            atoms,
            p_symbol=p_symbol,
            o_symbol=o_symbol,
            cutoff_po=cutoff_po,
            ensure_bo=True,
            set_array=True,
        )

    if "P_qi" not in atoms.arrays:
        raise ValueError('atoms.arrays["P_qi"] が存在しません。')

    qi_array = atoms.arrays["P_qi"]




    p_indices = [i for i, a in enumerate(atoms) if a.symbol == p_symbol]

    # P原子のみ対象に Qi をカウント
    mask_arr = resolve_mask_for_atoms(mask, atoms)
    # mask_arr is None or numpy bool array of length len(atoms)
    if mask_arr is not None:
        # use mask_arr to filter indices
        p_indices = [i for i in p_indices if mask_arr[i]]

    # if mask is not None:
    #     if len(mask) != len(atoms):
    #         raise ValueError(f"mask の長さが atoms の長さ ({len(atoms)}) と一致しません (got {len(mask)})")
    #     # mask が True の P 原子だけ残す
    #     p_indices = [i for i in p_indices if bool(mask[i])]


    n_p = len(p_indices)

    if n_p == 0:
        return QiSummary(n_p=0, counts={}, fractions={})

    labels = [str(qi_array[i]) for i in p_indices if qi_array[i] not in (None, "")]
    c = Counter(labels)

    # 固定レンジ Q0..Q4 を常に用意（存在しないものは0に）
    counts_ordered: Dict[str, int] = {}
    fractions_ordered: Dict[str, float] = {}
    for q in range(0, 5):
        key = f"Q{q}"
        cnt = int(c.get(key, 0))
        counts_ordered[key] = cnt
        fractions_ordered[key] = cnt / n_p

    # （もし非標準ラベルが混じっていれば、それらも末尾に追加しておく）
    nonstandard_keys = [k for k in c.keys() if not (isinstance(k, str) and k.startswith("Q") and k[1:].isdigit())]
    for k in sorted(nonstandard_keys):
        counts_ordered[k] = int(c[k])
        fractions_ordered[k] = counts_ordered[k] / n_p

    return QiSummary(n_p=n_p, counts=counts_ordered, fractions=fractions_ordered)
