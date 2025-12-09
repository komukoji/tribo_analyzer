# src/tribo_analyzer/utils.py
from __future__ import annotations
from typing import Optional, Sequence, Union, Callable, Any
import numpy as np
from ase import Atoms

# mask type: None | sequence of bool | callable returning sequence of bool
MaskType = Optional[Union[Sequence[bool], Callable[[Atoms], Sequence[bool]]]]

def resolve_mask_for_atoms(mask: MaskType, atoms: Atoms) -> Optional[np.ndarray]:
    """
    Normalize mask argument into a numpy boolean array or None.

    Accepts:
      - None -> returns None
      - sequence-like (list/np.array/tuple) -> converted to numpy bool array
      - callable(atoms) -> called and its return converted to numpy bool array

    Raises:
      - ValueError if the resolved mask length != len(atoms)
      - TypeError if callable returns None or something non-iterable
    """
    if mask is None:
        return None

    if callable(mask):
        res = mask(atoms)
    else:
        res = mask

    if res is None:
        raise TypeError("mask callable returned None; expected sequence of booleans")

    arr = np.asarray(res, dtype=bool)

    if arr.shape[0] != len(atoms):
        raise ValueError(f"mask length ({arr.shape[0]}) != number of atoms ({len(atoms)})")

    return arr
