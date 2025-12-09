from ase.io.cfg import read_cfg as _ase_read_cfg
from ase.io.cif import read_cif as _ase_read_cif

read_cfg = _ase_read_cfg
read_cif = _ase_read_cif

__all__ = ["read_cfg", "read_cif"]
