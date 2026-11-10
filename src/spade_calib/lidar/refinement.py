"""Legacy reference XYZ loader for *post-optimization regression only*.

This module does not reconstruct or infer pose corrections from saved output.
Its outputs must never be fed to either selected refinement solver.
"""
from __future__ import annotations
from io import BytesIO
from pathlib import Path
from typing import Literal
from zipfile import ZipFile
import numpy as np
from numpy.typing import NDArray

RefinementMethod = Literal['Joint', 'PGO']
SUPPORTED_METHODS: tuple[RefinementMethod, ...] = ('Joint', 'PGO')
MAX_REFERENCE_BYTES = 128 * 1024 * 1024


def load_archived_refined_xyz(archive_path: str | Path, method: RefinementMethod) -> NDArray[np.float64]:
    if method not in SUPPORTED_METHODS:
        raise ValueError(f'Unsupported refinement method: {method!r}; expected Joint or PGO')
    with ZipFile(archive_path) as archive:
        suffix = f'/inputs/{method}_xyz.npy'
        entries = [info for info in archive.infolist() if info.filename.endswith(suffix)]
        if len(entries) != 1:
            raise ValueError(f'Expected exactly one {suffix} in reference archive')
        info = entries[0]
        if info.file_size > MAX_REFERENCE_BYTES:
            raise ValueError('Refined cloud exceeds configured size limit')
        values = np.load(BytesIO(archive.read(info)), allow_pickle=False)
    if values.ndim != 2 or values.shape[1] != 3 or values.dtype.kind not in 'f':
        raise ValueError(f'{method}: expected floating-point Nx3 reference PCD')
    if not np.isfinite(values).all():
        raise ValueError(f'{method}: non-finite refined XYZ')
    return np.asarray(values, dtype=np.float64)