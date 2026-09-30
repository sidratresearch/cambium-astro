"""Import point for any builtin FITS handlers.

Handlers imported here are accessible to the PreviewFITS stage.
"""

from ._fits_handler import FITSHandler, SingleHDUInfo, UUIDMapping
from .cube_fits_handler import (
    CubeMaxFITSHandler,
    CubeSumFITSHandler,
)
from .default_fits_handler import DefaultFITSHandler
from .reversed_fits_handler import ReversedFITSHandler

__all__ = [
    "CubeMaxFITSHandler",
    "CubeSumFITSHandler",
    "DefaultFITSHandler",
    "FITSHandler",
    "ReversedFITSHandler",
    "SingleHDUInfo",
    "UUIDMapping",
]
