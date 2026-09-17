"""Import point for any builtin FITS handlers.

Handlers imported here are accessible to the PreviewFITS stage.
"""

from ._fits_handler import FITSHandler, SingleHDUInfo, UUIDMapping
from .default_fits_handler import DefaultFITSHandler
from .reversed_fits_handler import ReversedFITSHandler
from .spectral_cube_fits_handler import (
    SpectralCubeMaxFITSHandler,
    SpectralCubeSumFITSHandler,
)

__all__ = [
    "DefaultFITSHandler",
    "FITSHandler",
    "ReversedFITSHandler",
    "SingleHDUInfo",
    "SpectralCubeMaxFITSHandler",
    "SpectralCubeSumFITSHandler",
    "UUIDMapping",
]
