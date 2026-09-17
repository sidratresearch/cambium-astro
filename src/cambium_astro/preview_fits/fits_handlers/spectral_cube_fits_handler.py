"""Handler for creating previews of spectral cubes."""

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from . import default_fits_handler as dfh
from ._fits_handler import SingleHDUInfo, UUIDMapping

logger = logging.getLogger(__name__)


class SpectralCubeSumFITSHandler(dfh.DefaultFITSHandler):
    """FITS preview handler to make preview images of spectral cubes.

    This particular handler uses the sum of the spectral axis.
    """

    @classmethod
    def matches_file(cls, hdus: list[SingleHDUInfo]) -> bool:
        """Match files which have PrimaryHDU with the SPECSYS keyword."""
        wcs = WCS(hdus[0].header)
        return (
            len(hdus) == 1
            and wcs.has_spectral
            and wcs.has_celestial
            and hdus[0].header.get("GROUPS", "F") == "F"
        )

    def make_uuid_mapping(
        self,
        preview_page_uuid: str,
        hdu_info: list[SingleHDUInfo],
        fits_path: Path,
        add_leaf: Callable[[Path], str],
        image_extension: str,
    ) -> UUIDMapping:
        """Function that gets called if `self.matches_file` passes."""
        fits_file_uuid = add_leaf(fits_path / fits_path.name)

        # create the object that will be attached to the FITS file UUID in the output
        file_info = dfh.DFH_FITSFileInfo(
            initial_fits_path=fits_path,
            fits_file_uuid=fits_file_uuid,
            preview_page_uuid=preview_page_uuid,
            n_hdus=len(hdu_info),
            preview_handler_name=self.__class__.__name__,
        )

        index = 0
        preview_type = "image"

        # and create a new leaf for a preview image
        # this handler only matches single-hdu files
        image_filename = dfh.make_image_filename(
            fits_path, hdu_info[0], image_extension
        )
        image_uuid = add_leaf(fits_path / image_filename)
        file_info.add_hdu(index, preview_type, image_uuid, image_filename)

        # return all of the UUIDs associated with this fits file
        uuid_mapping = dict.fromkeys(file_info.image_uuids_to_filenames)
        uuid_mapping.update({preview_page_uuid: None, fits_file_uuid: file_info})

        return uuid_mapping

    def flatten_function(self) -> tuple[Callable[[np.ndarray], np.ndarray], str]:
        """Set the function used to flatten a 3D cube into a 2D image."""
        flatten = lambda datacube: np.nansum(datacube, axis=0)
        label = "NaN sum"
        return flatten, label

    def make_image(
        self, path: Path, hdu: fits.ImageHDU | fits.TableHDU, initial_path: Path
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

        if "CRPIX1" in header:
            wcs = WCS(header).celestial
            header.update(**wcs.to_header())
            header["NAXIS"] = 2
            for key in ("UNIT", "TYPE", "RVAL", "RPIX", "DELT"):
                key = f"C{key}3"
                if key in header:
                    header.remove(key)
            dfh._make_wcs_image(path, flat_data, header)
        else:
            dfh._make_basic_image(path, flat_data, header)


class SpectralCubeMaxFITSHandler(SpectralCubeSumFITSHandler):
    """FITS preview handler to make images of spectral cubes with maximum values."""

    def flatten_function(self) -> tuple[Callable[[np.ndarray], np.ndarray], str]:
        """Set the function used to flatten a 3D cube into a 2D image."""
        flatten = lambda datacube: np.nanmax(datacube, axis=0)
        label = "NaN max"
        return flatten, label
