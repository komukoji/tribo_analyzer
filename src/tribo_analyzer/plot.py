# src/tribo_analyzer/plot.py
from __future__ import annotations
import re
from pathlib import Path
from typing import List, Tuple, Optional
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from ase.data import atomic_numbers

from .io import read_cfg
from .phosphate import summarize_phosphorus_qi, QiSummary
from .structure_analysis import compute_elementwise_coordination_map, coordination_environment_distribution
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

    fig.tight_layout()

    return fig, ax


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
    apply_plot_style()

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

    return fig, ax

def compute_master_coord_time_series(
    dirpath: str | Path,
    master_elem: str,
    *,
    use_filename_as_time: bool = True,
    time_unit_scale: float = 0.001,
    max_frames: Optional[int] = None,
    mask: MaskType = None,
    include_self: bool = False,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    master 元素に対して、存在する全相手元素との平均配位数時系列を計算する。

    Returns
    -------
    times : np.ndarray
    values_dict : dict[str, np.ndarray]
        key = 相手元素, value = 配位数時系列
    """
    files = _files_sorted_by_number(dirpath)
    if max_frames is not None:
        files = files[:max_frames]

    times: list[float] = []
    values_dict: dict[str, list[float]] = {}

    for idx, fpath in enumerate(files):
        tnum = _extract_step_from_filename(fpath)
        time_val = (tnum if use_filename_as_time and tnum is not None else idx) * time_unit_scale
        times.append(time_val)

        atoms = read_cfg(str(fpath))

        mask_for_frame = resolve_mask_for_atoms(mask, atoms) if mask is not None else None
        cn_map = compute_elementwise_coordination_map(atoms, mask=mask_for_frame)

        master_map = cn_map.get(master_elem, {})

        for partner, val in master_map.items():
            if not include_self and partner == master_elem:
                continue
            values_dict.setdefault(partner, []).append(float(val))

        # 存在しない相手元素の 0 埋め（時系列長を揃える）
        for partner in values_dict:
            if len(values_dict[partner]) < len(times):
                values_dict[partner].append(0.0)

    values_dict_np = {k: np.array(v) for k, v in values_dict.items()}
    return np.array(times), values_dict_np

def plot_master_coord_time_series(
    dirpath: str | Path,
    master_elem: str,
    *,
    use_filename_as_time: bool = True,
    time_unit_scale: float = 0.001,
    max_frames: Optional[int] = None,
    mask: MaskType = None,
    include_self: bool = False,
    figsize: tuple[int, int] = (8, 5),
    title: Optional[str] = None,
) -> tuple[plt.Figure, plt.Axes]:

    apply_plot_style()

    times, values_dict = compute_master_coord_time_series(
        dirpath,
        master_elem,
        use_filename_as_time=use_filename_as_time,
        time_unit_scale=time_unit_scale,
        max_frames=max_frames,
        mask=mask,
        include_self=include_self,
    )

    fig, ax = plt.subplots(figsize=figsize)

    for partner, vals in sorted(values_dict.items()):
        ax.plot(times, vals, label=f"{master_elem}-{partner}")

    ax.set_xlabel("time [ps]")
    ax.set_ylabel(f"avg coordination of {master_elem}")
    ax.set_title(title or f"{master_elem}-X coordination time series")
    ax.legend(frameon=False)

    return fig, ax

def plot_z_distribution(
    cfg_paths: list[str | Path],
    elements: list[str],
    labels: list[str],
    *,
    z_bin_width: float = 1.0,
    mask: MaskType = None,
    z_interface: float | None = 0.0,
    density: bool = False,
    figsize: tuple[int, int] = (5, 6),
    title: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    複数 cfg に対して原子の z 座標分布を plot + scatter で描画する。
    縦軸: z, 横軸: 分布数（または密度）
    """

    if not (len(cfg_paths) == len(elements) == len(labels)):
        raise ValueError("cfg_paths, elements, labels must have the same length")

    fig, ax = plt.subplots(figsize=figsize)

    for cfg, elem, label in zip(cfg_paths, elements, labels):
        atoms = read_cfg(str(cfg))

        mask_for_atoms = (
            resolve_mask_for_atoms(mask, atoms, is_wrap=False)
            if mask is not None
            else None
        )

        z_vals = []
        for i, a in enumerate(atoms):
            if a.symbol != elem:
                continue
            if mask_for_atoms is not None and not mask_for_atoms[i]:
                continue
            z_vals.append(a.position[2])

        if len(z_vals) == 0:
            continue

        z_vals = np.asarray(z_vals)

        z_min, z_max = z_vals.min(), z_vals.max()
        bins = np.arange(z_min, z_max + z_bin_width, z_bin_width)

        counts, edges = np.histogram(z_vals, bins=bins, density=density)
        z_centers = 0.5 * (edges[:-1] + edges[1:])

        # plot
        ax.plot(counts, z_centers, label=label)
        ax.scatter(counts, z_centers, s=20)

    if z_interface is not None:
        ax.axhline(
            z_interface,
            color="k",
            linestyle="--",
            linewidth=1.0,
            label="interface" if z_interface == 0.0 else None,
        )

    ax.set_xlabel("count" if not density else "probability density")
    ax.set_ylabel("z [Å]")

    if title:
        ax.set_title(title)

    ax.legend(frameon=False)

    return fig, ax



def coordination_key_to_label(key: CoordinationKey) -> str:
    """
    frozenset({("P", 2), ("Zn", 1)}) -> "PPZn"
    frozenset({("Zn", 2), ("P", 1)}) -> "PZnZn"
    ※ 元素番号順でソート
    """
    elems = []
    for elem, cnt in sorted(
        key,
        key=lambda x: atomic_numbers[x[0]]  # ← 元素番号順
    ):
        elems.extend([elem] * cnt)
    return "".join(elems)


def compute_coordination_env_time_series(
    dirpath: str | Path,
    center_element: str,
    *,
    use_filename_as_time: bool = True,
    time_unit_scale: float = 0.001,
    max_frames: Optional[int] = None,
    mask: MaskType = None,
    min_fraction: float = 0.05
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    指定元素の配位環境分布の時系列を計算する。

    Returns
    -------
    times : np.ndarray
    series_dict : dict[str, np.ndarray]
        key = 配位環境ラベル (例: "PPZn")
        value = 個数の時系列
    """
    files = _files_sorted_by_number(dirpath)
    if max_frames is not None:
        files = files[:max_frames]

    times: list[float] = []
    series_dict: dict[str, list[float]] = {}

    for idx, fpath in enumerate(files):
        # --- time ---
        tnum = _extract_step_from_filename(fpath)
        t = (tnum if use_filename_as_time and tnum is not None else idx) * time_unit_scale
        times.append(t)

        atoms = read_cfg(str(fpath))

        symbols = atoms.get_chemical_symbols()
        mask_arr = resolve_mask_for_atoms(mask, atoms)

        n_center = sum(
            1 for i, s in enumerate(symbols)
            if s == center_element and (mask_arr is None or mask_arr[i])
)


        # --- 配位環境分布 ---
        dist = coordination_environment_distribution(
            atoms,
            center_element=center_element,
            mask=mask,
        )

        # このフレームで観測されたラベル
        current_labels: set[str] = set()

        for key, count in dist.items():
            label = coordination_key_to_label(key)
            current_labels.add(label)

            # 新しい配位環境が出現したら 0 埋めで初期化
            if label not in series_dict:
                series_dict[label] = [0] * (len(times) - 1)

            #series_dict[label].append(count)
            frac = count / n_center if n_center > 0 else 0.0
            series_dict[label].append(frac)

        # このフレームで出なかった既存ラベルは 0 を足す
        for label in series_dict:
            if label not in current_labels:
                series_dict[label].append(0)

    series_dict_np = {k: np.array(v) for k, v in series_dict.items()}

    # --- 出現頻度が小さいラベルを Others にまとめる ---
    others = np.zeros(len(times))
    kept: dict[str, np.ndarray] = {}

    for label, series in series_dict_np.items():
        if series.max() < min_fraction:
            others += series
        else:
            kept[label] = series

    # --- 並び順を制御 ---
    ordered: dict[str, np.ndarray] = {}

    if np.any(others > 0):
        ordered["Others"] = others   # ← 先に入れる（一番下）

    # 残りは出現率の大きい順などにしても良い
    for label, series in kept.items():
        ordered[label] = series

    return np.array(times), ordered

def plot_coordination_env_time_series(
    dirpath: str | Path,
    center_element: str,
    *,
    use_filename_as_time: bool = True,
    time_unit_scale: float = 0.001,
    max_frames: Optional[int] = None,
    mask: MaskType = None,
    figsize: tuple[int, int] = (8, 4),
    title: Optional[str] = None,
    legend_outside: bool = True,
    min_fraction: float = 0.05
) -> tuple[plt.Figure, plt.Axes]:

    apply_plot_style()

    times, series_dict = compute_coordination_env_time_series(
        dirpath,
        center_element,
        use_filename_as_time=use_filename_as_time,
        time_unit_scale=time_unit_scale,
        max_frames=max_frames,
        mask=mask,
        min_fraction=min_fraction
    )

    fig, ax = plt.subplots(figsize=figsize)

    for label, values in sorted(series_dict.items()):
        ax.plot(times, values, label=label)

    ax.set_xlabel("time [ps]")
    ax.set_ylabel(f"fraction of {center_element} coordination env.")
    #ax.set_title(title or f"{center_element} coordination environment time series")

    if legend_outside:
        ax.legend(
            frameon=False,
            bbox_to_anchor=(1., 1),
            loc="upper left",
            borderaxespad=0,
        )

    else:
        ax.legend(frameon=False)

    ax.set_xlim(times[0], times[-1])
    ax.set_ylim(0.0, 1.0)
    fig.tight_layout()

    return fig, ax
