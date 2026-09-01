"""Make stages from this package importable."""

from cambium.cli.log import init_logging

from .preview_fits import preview_fits

init_logging(__package__.split(".")[0])

__all__ = ["preview_fits"]
