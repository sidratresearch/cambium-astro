"""Make stages and macros from this package importable."""

from cambium.cli.log import init_logging

from .macros import ListFITSFiles
from .preview_fits.preview_fits import PreviewFITS

init_logging(__package__.split(".")[0])

__all__ = ["ListFITSFiles", "PreviewFITS"]
