from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple
from ase.data import chemical_symbols, atomic_numbers

Pair = Tuple[str, str]

# ASEの周期表に基づく有効元素リスト
_VALID_ELEMENTS = set(chemical_symbols[1:])  # 'H'～'Og'

def _normalize_pair(a: str, b: str) -> Pair:
    """元素ペアを原子番号順（小さい方→大きい方）に正規化。

    - 両方が有効な元素記号でない場合は ValueError。
    """
    if a not in _VALID_ELEMENTS or b not in _VALID_ELEMENTS:
        raise ValueError(f"無効な元素記号です: ({a}, {b})")

    za, zb = atomic_numbers[a], atomic_numbers[b]
    return (a, b) if za <= zb else (b, a)


@dataclass
class CutoffConfig:
    pair_cutoffs: Dict[Pair, float] = field(default_factory=lambda: {
        # 金属同士は組成によってRDFばらばらだったから結合に関してあまり気にする必要ないかも
        ("O", "O"): 1.40,
        ("O", "P"): 2.22,
        ("O", "S"): 2.10,
        ("O", "Fe"): 2.62,
        ("O", "Zn"): 2.35,
        ("P", "P"): 2.62,
        ("P", "S"): 2.74,
        ("P", "Fe"): 2.78,
        ("P", "Zn"): 2.78,
        ("S", "S"): 2.38,
        ("S", "Fe"): 2.55,
        ("S", "Zn"): 2.74,
        ("Fe", "Fe"): 3.42,
        ("Fe", "Zn"): 3.46,
        ("Zn", "Zn"): 3.66,
    })


GLOBAL_CUTOFFS = CutoffConfig()


def get_pair_cutoff(a: str, b: str) -> float | None:
    key = _normalize_pair(a, b)
    return GLOBAL_CUTOFFS.pair_cutoffs.get(key)


def set_pair_cutoff(a: str, b: str, value: float) -> None:
    key = _normalize_pair(a, b)
    GLOBAL_CUTOFFS.pair_cutoffs[key] = value
