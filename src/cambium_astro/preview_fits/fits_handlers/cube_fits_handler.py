"""Handler for creating previews of cubes."""

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
from astropy.io import fits

from . import default_fits_handler as dfh
from ._fits_handler import SingleHDUInfo

logger = logging.getLogger(__name__)


class CubeSumFITSHandler(dfh.DefaultFITSHandler):
    """FITS preview handler to make preview images of data cubes.

    This particular handler uses the sum of the third axis.
    """

    @classmethod
    def matches_file(cls, hdus: list[SingleHDUInfo]) -> bool:
        """Match files which have 3D PrimaryHDU data."""
        return (
            len(hdus) == 1
            and hdus[0].header.get("NAXIS") == 3
            and hdus[0].header.get("GROUPS", "F") == "F"
        )

    @classmethod
    def choose_preview_type(cls, hdu_info: SingleHDUInfo) -> dfh.PreviewType:
        return "image", 1

    def flatten_function(self) -> tuple[Callable[[np.ndarray], np.ndarray], str]:
        """Set the function used to flatten a 3D cube into a 2D image."""
        flatten = lambda datacube: np.nansum(datacube, axis=0)
        label = "NaN sum"
        return flatten, label

    def make_hdu_images(
        self, paths: list[Path], hdu: fits.ImageHDU | fits.TableHDU, initial_path: Path
    ) -> None:
        header = hdu.header.copy()  # make changes to a copy only

        flatten, description = self.flatten_function()
        flat_data = flatten(hdu.data)

        cbar_label = f"{description} over spectral axis"
        if header.get("CTYPE3") is not None:
            cbar_label = f"{description} over {header.get('CTYPE3')}"
        if header.get("CUNIT3") is not None:
            cbar_label += f" ({header.get('CUNIT3')})"
        header["BUNIT"] = cbar_label

        dfh._make_basic_image(paths[0], flat_data, header)


class CubeMaxFITSHandler(CubeSumFITSHandler):
    """FITS preview handler to make images of data cubes with maximum values."""

    def flatten_function(self) -> tuple[Callable[[np.ndarray], np.ndarray], str]:
        """Set the function used to flatten a 3D cube into a 2D image."""
        flatten = lambda datacube: np.nanmax(datacube, axis=0)
        label = "NaN max"
        return flatten, label
