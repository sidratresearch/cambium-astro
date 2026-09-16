"""Base types for FITSHandler classes."""

import shutil
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path

from astropy.io import fits
from cambium.tree import TreeSpan
from cambium.utils.path_utils import abs_leaf_path
from jinja2 import Environment


class SingleHDUInfo:
    """Information about a single HDU used during the TreeHook.

    List of these (one for each HDU in a file) gets passed to a FITSHandler to
    determine what should be done with the file.
    """

    def __init__(
        self,
        index: int,
        hdu_class: type[fits.ImageHDU | fits.TableHDU],
        header: fits.Header,
    ) -> None:
        self.index = index
        self.hdu_class = hdu_class
        self.header = header


class BaseFITSFileInfo:
    """Info on an entire FITS file.

    Used to pass data from the tree hook to the pre-hook.
    Should be serializable for debugging and potential futur parallelization.
    """

    def __init__(
        self,
        initial_fits_path: Path,
        fits_file_uuid: str,
        preview_page_uuid: str,
        n_hdus: int,
        preview_handler_name: str,
    ) -> None:
        self.initial_fits_path = initial_fits_path
        self.fits_file_uuid = fits_file_uuid
        self.preview_page_uuid = preview_page_uuid
        self.n_hdus = n_hdus
        self.preview_handler_name = preview_handler_name
        self.image_uuids = []


UUIDMapping = dict[str, None | BaseFITSFileInfo]
"""Mapping of UUIDs to be passed back from the FITSHandler to the PreviewFITS Stage."""


class FITSHandler(ABC):
    """Abstract base for FITSHandler classes. See DefaultFITSHandler for an example."""

    @classmethod
    @abstractmethod
    def matches_file(cls, hdu_info: list[SingleHDUInfo]) -> bool:
        """Check whether this FITSHandler should be applied to a given FITS file.

        `hdu_info` contains the header, HDU number, and HDU type for each HDU in a file
        """
        ...

    def post_init(self, jinja_environment: Environment, tree: TreeSpan) -> None:
        """Any initialization requiring args.

        In particular, fetching the Jinja template used by this handler should be
        done here.
        """
        return

    @classmethod
    @abstractmethod
    def make_uuid_mapping(
        cls,
        preview_page_uuid: str,
        hdu_info: list[SingleHDUInfo],
        fits_path: Path,
        add_leaf: Callable[[Path], str],
        image_extension: str,
    ) -> UUIDMapping:
        """Function that gets called if `self.matches_file` passes.

        This function's primary job is list off any leaves (image files) that need
        to get made; as well as pack information about which UUIDs correspond to
        which final files into an object that can be read later on.

        While this function recieves the parameter `fits_path` it should *not* open the
        file. All decisions should be made based on the provided `hdu_info`.

        Return the serializable dictionary:
            {
                "fits_file_uuid": DumpableFITSFileInfo,
                "preview_page_uuid": None,
                "image_#_uuid": None,
                ...
            }
        """
        ...

    @abstractmethod
    def write_files(self, fits_info: BaseFITSFileInfo, tree: TreeSpan) -> None:
        """Write to all the files affected by PreviewFITS.

        Workhorse function called during the pre-hook step which writes data to all
        of the relevant files:
        - Copies the FITS file from the initial path to the temporary directory
        - Render preview images of HDUs to files on disk
        - Write the HTML-formatted preview page (written to a .md file to take
          advantage of TemplateMarkdown)
        """
        ...

    def _copy_fits_to_tmp(self, fits_info: BaseFITSFileInfo, tree: TreeSpan) -> None:
        """Copy the FITS file into the temporary directory at it's new location.

        Should be called by `self.write_files`.
        """
        tmpdir_fits_path = abs_leaf_path(tree, fits_info.fits_file_uuid)
        shutil.copyfile(fits_info.initial_fits_path, tmpdir_fits_path)
