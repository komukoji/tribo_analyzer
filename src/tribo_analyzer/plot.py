# src/tribo_analyzer/plot.py
from __future__ import annotations
import re
from pathlib import Path
from typing import List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np

from .io import read_cfg
from .phosphate import summarize_phosphorus_qi, QiSummary
from .structure_analysis import compute_elementwise_coordination_map
from .utils import resolve_mask_for_atoms, MaskType
from .plot_config import apply_plot_style

_FILENAME_NUMBER_RE = re.compile(r"(\d+)\.cfg$")  # "1234.cfg" -> 1234

def _files_sorted_by_number(dirpath: str | Path) -> List[Path]:
    p = Path(dirpath)
    files = []
    for f in p.iterdir():
        if not f.is_file():
            continue
        m = _FILENAME_NUMBER_RE.search(f.name)
        if m:
            files.append((int(m.group(1)), f))
    files.sort(key=lambda x: x[0])
    return [f for _, f in files]


def _extract_step_from_filename(fname: Path) -> int | None:
    m = _FILENAME_NUMBER_RE.search(fname.name)
    return int(m.group(1)) if m else None

def compute_qi_time_series(
    dirpath: str | Path,
    *,
    time_unit_scale: float = 1.0,
    cutoff_po: float | None = None,
    max_frames: int | None = None,
    mask: MaskType = None,
) -> Tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    指定ディレクトリ内の "N.cfg" (N: integer) を時系列として読み、Q0..Q4 の割合を返す。

    Returns
    -------
    times : 1D np.ndarray
        各フレームの時刻（整数 or index）を格納。
    fractions_dict : dict[str, np.ndarray]
        {"Q0": array([...]), "Q1": ..., "Q4": ...}
    """
    files = _files_sorted_by_number(dirpath)
    #print(files)
    if max_frames is not None:
        files = files[:max_frames]

    times: List[float] = []
    q_keys = [f"Q{q}" for q in range(0, 5)]
    frac_storage = {k: [] for k in q_keys}

    for idx, fpath in enumerate(files):
        # 時刻（ファイル名の数値 or 単純 index）
        tnum = _extract_step_from_filename(fpath)
        time_val = (tnum if tnum is not None else idx) * time_unit_scale
        times.append(time_val)

        # read_cfg は複数構造を返すが各 .cfg が単一フレームなら最初の構造を使う
        atoms = read_cfg(str(fpath))

        # resolve mask for this frame
        mask_for_frame = resolve_mask_for_atoms(mask, atoms)
        if mask_for_frame is not None:
            if len(mask_for_frame) != len(atoms):
                raise ValueError(f"Mask length ({len(mask_for_frame)}) != number of atoms ({len(atoms)}) in file {fpath}")

        # Qi サマリ（Q0..Q4 固定）
        qi_summary: QiSummary = summarize_phosphorus_qi(
            atoms,
            cutoff_po=cutoff_po,
            ensure_qi=True,
            mask=mask_for_frame
        )

        for k in q_keys:
            frac_storage[k].append(qi_summary.fractions.get(k, 0.0))

    times_arr = np.array(times)
    frac_arrays = {k: np.array(v) for k, v in frac_storage.items()}
    return times_arr, frac_arrays


def plot_qi_time_series(
    dirpath: str | Path,
    *,
    time_unit_scale: float = 0.001, # 1step あたりのps
    cutoff_po: float | None = None,
    figsize: Tuple[int, int] = (8, 4),
    title: str | None = None,
    max_frames: int | None = None,
    mask: MaskType = None,
) -> plt.Figure:
    """
    指定ディレクトリ中の .cfg ファイル（番号付き）を時系列にプロットする。

    - Q0..Q4 を色分けしてプロット
    - savefig を渡すと図を保存
    - 戻り値は matplotlib Figure
    """
    apply_plot_style()

    times, fracs = compute_qi_time_series(
        dirpath,
        time_unit_scale=time_unit_scale,
        cutoff_po=cutoff_po,
        max_frames=max_frames,
        mask=mask
    )

    fig, ax = plt.subplots(figsize=figsize)
    q_keys = [f"Q{q}" for q in range(0, 5)]

    for k in q_keys:
        qi = k[1]
        ax.plot(times, fracs[k], label=f"Q$_{{{qi}}}$", )

    ax.set_xlabel("time [ps]")
    ax.set_ylabel("fraction of P atoms")
    if title:
        ax.set_title(title)
    ax.legend(title=r"$\mathrm{Q_i}$",loc='upper left', bbox_to_anchor=(1,1))
    ax.set_xlim(times[0], times[-1])
    ax.set_ylim(0.0, 1.0)

    return fig


def compute_pair_coord_time_series(
    dirpath: str | Path,
    elem_a: str,
    elem_b: str,
    *,
    use_filename_as_time: bool = True,
    time_unit_scale: float = 0.001,
    max_frames: Optional[int] = None,
    mask: MaskType = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    指定ディレクトリ内の番号付き .cfg を順に読み、元素 elem_a の "elem_b に対する平均配位数" を
    各フレームごとに計算して返す。

    Returns
    -------
    times : 1D np.ndarray
        各フレームの時刻（ファイル名中の数値 or index）。
    values : 1D np.ndarray
        各フレームにおける平均配位数 (elem_a -> elem_b)
    """
    files = _files_sorted_by_number(dirpath)
    if max_frames is not None:
        files = files[:max_frames]

    times: List[float] = []
    values: List[float] = []

    for idx, fpath in enumerate(files):
        tnum = _extract_step_from_filename(fpath)
        time_val = (tnum if use_filename_as_time and tnum is not None else idx) * time_unit_scale
        times.append(time_val)

        # read_cfg の戻り値を安全に取り出す
        structures = read_cfg(str(fpath))
        atoms = None
        # if structures is None:
        #     atoms = None
        # elif isinstance(structures, Atoms):
        #     atoms = structures
        # elif isinstance(structures, (list, tuple)):
        #     atoms = structures[0] if len(structures) > 0 else None
        # else:
        #     raise TypeError(f"read_cfg returned unsupported type: {type(structures)!r} for file {fpath}")

        # if atoms is None:
        #     values.append(0.0)
        #     continue
        atoms = read_cfg(str(fpath))

        # resolve mask (callable or fixed sequence) for this frame
        mask_for_frame = resolve_mask_for_atoms(mask, atoms) if mask is not None else None

        # compute elementwise coordination map for this frame
        cn_map = compute_elementwise_coordination_map(atoms, mask=mask_for_frame)

        # if element missing in map, treat as 0.0
        val = 0.0
        if elem_a in cn_map and elem_b in cn_map[elem_a]:
            val = float(cn_map[elem_a][elem_b])
        values.append(val)

    return np.array(times), np.array(values)


def plot_pair_coord_time_series(
    dirpath: str | Path,
    elem_a: str,
    elem_b: str,
    *,
    use_filename_as_time: bool = True,
    time_unit_scale: float = 1.0,
    max_frames: Optional[int] = None,
    mask: MaskType = None,
    figsize: Tuple[int, int] = (8, 4),
    title: Optional[str] = None,
) -> plt.Figure:
    """
    指定ディレクトリの cfg シリーズから (elem_a -> elem_b) の平均結合数の時刻歴を描画して Figure を返す。
    mask は None / sequence / callable を受け取る（Qi と同様）。
    """
    # optional: apply_plot_style() if you want module-wide style applied
    # from .plot_config import apply_plot_style
    # apply_plot_style()

    times, vals = compute_pair_coord_time_series(
        dirpath,
        elem_a,
        elem_b,
        use_filename_as_time=use_filename_as_time,
        time_unit_scale=time_unit_scale,
        max_frames=max_frames,
        mask=mask,
    )

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(times, vals)
    ax.set_xlabel("time [ps]")
    ax.set_ylabel(f"avg coordination: {elem_a}-{elem_b}")
    if title:
        ax.set_title(title)
    # legend not necessary for single line; add if needed

    return fig
