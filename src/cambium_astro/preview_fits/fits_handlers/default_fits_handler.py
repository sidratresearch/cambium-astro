"""Default FITSHandler which can work with any FITS file."""

import logging
import re
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, TypedDict

import numpy as np
from astropy.io import fits
from cambium.tree import TreeSpan
from cambium.utils.md_html_utils import wrap_with_div
from cambium.utils.path_utils import abs_leaf_path
from jinja2 import Environment
from matplotlib import pyplot as plt

from ._fits_handler import BaseFITSFileInfo, FITSHandler, SingleHDUInfo, UUIDMapping
from ._make_healpix_image import HEALPIX_AVAILABLE, _make_healpix_images

logger = logging.getLogger(__name__)

PreviewType = tuple[Literal["image", "table", "unavailable"], int]


class DFH_FITSFileInfo(BaseFITSFileInfo):
    """Info on an entire FITS file used to pass data from TreeHook to PreHook."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.image_uuid_to_filename: dict[str, str] = {}
        self.hdu_index_to_image_uuids: dict[int, list[str]] = defaultdict(list)
        self.hdu_index_to_preview_type: dict[int, PreviewType] = {}

    def add_hdu(
        self,
        index: int,
        preview_type: PreviewType,
        image_uuids: str | list[str] | None = None,
        image_filenames: str | list[str] | None = None,
    ) -> None:
        """Add information about an HDU to this file."""
        self.hdu_index_to_preview_type[index] = preview_type

        if preview_type == "image":
            if isinstance(image_uuids, str):
                image_uuids = [image_uuids]
            if isinstance(image_filenames, str):
                image_filenames = [image_filenames]

            for uuid, filename in zip(image_uuids, image_filenames):
                self.image_uuid_to_filename[uuid] = filename
                self.hdu_index_to_image_uuids[index].append(uuid)
                self.image_uuids.append(uuid)


class _JinjaHDUInfo(TypedDict):
    """Data passed to the Jinja template."""

    index: int
    class_name: str
    header: fits.Header
    preview_template: str
    """Name of jinja template to use, renders table or image or other or nothing"""
    preview_data: dict[str, Any]  # data to pass to preview_template


class DefaultFITSHandler(FITSHandler):
    """FITS preview handler for any type of FITS file."""

    def post_init(self, jinja_environment: Environment, _: TreeSpan) -> None:
        self.jinja_template = jinja_environment.get_template(
            "DefaultFITSHandler/PreviewFITS-DefaultFITSHandler.html.jinja"
        )

    @classmethod
    def matches_file(cls, _: list[SingleHDUInfo]) -> bool:
        """Class matches all files."""
        return True

    def make_uuid_mapping(
        self,
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
        fits_file_uuid = add_leaf(fits_path / fits_path.name)

        # create the object that will be attached to the FITS file UUID in the output
        file_info = DFH_FITSFileInfo(
            initial_fits_path=fits_path,
            fits_file_uuid=fits_file_uuid,
            preview_page_uuid=preview_page_uuid,
            n_hdus=len(hdu_info),
            preview_handler_name=self.__class__.__name__,
        )

        # look through each HDU, store the method by which we want to preview it,
        for single_hdu in hdu_info:
            index = single_hdu.index
            preview_type, num_new_leaves = self.choose_preview_type(single_hdu)

            # and create a new leaf for a preview image if necessary
            image_uuids, image_filenames = [], []
            if preview_type == "image":
                for i in range(num_new_leaves):
                    image_filenames.append(
                        make_image_filename(fits_path, single_hdu, image_extension, i)
                    )
                    image_uuids.append(add_leaf(fits_path / image_filenames[-1]))

            file_info.add_hdu(index, preview_type, image_uuids, image_filenames)

        # return all of the UUIDs associated with this fits file
        uuid_mapping = dict.fromkeys(file_info.image_uuid_to_filename)
        uuid_mapping.update({preview_page_uuid: None, fits_file_uuid: file_info})

        return uuid_mapping

    def write_files(
        self,
        fits_info: DFH_FITSFileInfo,
        max_preview_rows: int,
        tree: TreeSpan,
    ) -> None:
        """Write to all the files affected by PreviewFITS.

        Workhorse function called during the pre-hook step which writes data to all
        of the relevant files:
        - Copies the FITS file from the initial path to the temporary directory
        - Render preview images of HDUs to files on disk
        - Write the HTML-formatted preview page (written to a .md file to take
          advantage of TemplateMarkdown)
        """
        # copy fits file into tmpdir
        self._copy_fits_to_tmp(fits_info, tree)

        fits_path = fits_info.initial_fits_path

        # collect all data to put into the preview page, generating any images as we go
        jinja_data = []
        with fits.open(fits_path) as hdu_list:
            for i in range(fits_info.n_hdus):
                hdu = hdu_list[i]
                jinja_hdu_info = self.get_hdu_preview_info(i, hdu, fits_info, tree)
                jinja_data.append(jinja_hdu_info)

        # write the preview page
        preview_page_content = self.jinja_template.render(
            download_info={"link": fits_path.name, "size": fits_path.stat().st_size},
            hdu_data=jinja_data,
            cambium_wrap=wrap_with_div,
            max_preview_rows=max_preview_rows,
        )
        # strip all indentation and newlines (safe since we have no <pre> tags)
        # prevents marko from thinking there's an indented code block
        replaced = re.sub(r"^\s*", "", preview_page_content, flags=re.MULTILINE)

        abs_leaf_path(tree, fits_info.preview_page_uuid).write_text(replaced)

    def get_hdu_preview_info(
        self,
        index: int,
        hdu: fits.ImageHDU | fits.TableHDU,
        fits_info: DFH_FITSFileInfo,
        tree: TreeSpan,
    ) -> _JinjaHDUInfo:
        """Extract Jinja preview info for an HDU, and create an image if necessary."""
        header, data = hdu.header, hdu.data

        match fits_info.hdu_index_to_preview_type[index]:
            case "unavailable":
                jinja_preview_data = {}
                jinja_preview_template = (
                    "PreviewFITS-DefaultFITSHandler-unavailable.html.jinja"
                )

            case "image":
                image_uuids = fits_info.hdu_index_to_image_uuids[index]
                image_paths = [abs_leaf_path(tree, uuid) for uuid in image_uuids]
                image_statistics = self.make_hdu_images(
                    image_paths,
                    hdu,
                    initial_path=fits_info.initial_fits_path,
                )

                jinja_preview_data = {
                    "image_paths": [
                        fits_info.image_uuid_to_filename[uuid] for uuid in image_uuids
                    ],
                    "image_statistics": image_statistics,
                }

                jinja_preview_template = (
                    "PreviewFITS-DefaultFITSHandler-image.html.jinja"
                )

            case "table":
                jinja_preview_data = {"table_data": data}
                jinja_preview_template = (
                    "PreviewFITS-DefaultFITSHandler-table.html.jinja"
                )

        return _JinjaHDUInfo(
            index=index,
            class_name=type(hdu).__name__,
            header=header,
            preview_template=f"DefaultFITSHandler/{jinja_preview_template}",
            preview_data=jinja_preview_data,
        )

    def make_hdu_images(
        self, paths: list[Path], hdu: fits.ImageHDU | fits.TableHDU, initial_path: Path
    ) -> list[dict[str, Any]]:
        """Make preview image(s) with matplotlib.

        Accounts for HEALPix where relevant+available.
        """
        header, data = hdu.header, hdu.data
        if header.get("XTENSION") == "BINTABLE" and header.get("PIXTYPE") == "HEALPIX":
            # if HEALPix is not available, the HDU has a table preview type, not image
            _make_healpix_images(paths, hdu, initial_path)
            return [
                self.get_image_statistics(data.columns[i].array)
                for i in range(header.get("TFIELDS"))
            ]

        if header.get("NAXIS") == 1:
            _make_line_plot(paths[0], data, header)
            return [self.get_image_statistics(data)]

        if header.get("NAXIS") == 2:
            _make_basic_image(paths[0], data, header)
            return [self.get_image_statistics(data)]

        raise RuntimeError(f"Unclear how to make preview image {paths}")

    @classmethod
    def get_image_statistics(cls, data: np.ndarray) -> dict[str, Any]:
        """Make a dictionary of statistics to display alongside an image."""
        return {
            "nanmin": np.nanmin(data),
            "nanmax": np.nanmax(data),
            "size": data.size,
            "nan_count": np.sum(np.isnan(data)),
        }

    @staticmethod
    def choose_preview_type(
        hdu_info: SingleHDUInfo,
    ) -> PreviewType:
        """Inspect an HDU to determine in what way the preview should be displayed."""
        header = hdu_info.header
        hdu_classref = hdu_info.hdu_class

        if (  # HEALPix images
            issubclass(hdu_classref, fits.BinTableHDU)
            and header.get("PIXTYPE") == "HEALPIX"
        ):
            if HEALPIX_AVAILABLE:
                # TODO: should also check anything else that might make generating an image impossible - npix
                return ("image", header.get("TFIELDS"))
            return ("table", 0)

        if (  # regular images (also gets CompImageHDU)
            issubclass(hdu_classref, (fits.PrimaryHDU, fits.ImageHDU))
            and header.get("NAXIS") == 2
            and header.get("NAXIS1", default=0) > 0
            and header.get("NAXIS2", default=0) > 0
        ):
            return ("image", 1)

        if (  # tables
            issubclass(hdu_classref, (fits.BinTableHDU, fits.TableHDU))
            and header.get("NAXIS2", default=0) > 0
        ):
            return ("table", 0)

        return ("unavailable", 0)


def make_image_filename(
    fits_path: Path, hdu_info: SingleHDUInfo, image_extension: str, index: int
) -> str:
    """Make up a filename in which to put an image made from one HDU of a FITS file."""
    image_name_parts = [
        fits_path.stem,
        f"HDU{hdu_info.index}",
        hdu_info.hdu_class.__name__,
        f"{index}",
    ]

    hdu_name = hdu_info.header.get("EXTNAME")
    if hdu_name is not None:
        image_name_parts.append(hdu_name)

    image_name = "-".join(image_name_parts)

    return f"{image_name}.{image_extension}"


def _make_basic_image(path: Path, image_data: np.ndarray, header: fits.Header) -> None:
    """Display a colourmapped array with pixel coordinates."""
    # fig, ax = plt.subplots()
    fig = plt.figure()
    ax = fig.add_axes((0.1, 0.1, 0.8, 0.8))
    im_min, im_max = np.nanpercentile(image_data, [1, 99])
    im = ax.imshow(image_data, vmin=im_min, vmax=im_max)
    cbar = fig.colorbar(
        im,
        label=header.get("BUNIT"),
        extend="both",
        location="bottom",
        shrink=0.8,
        aspect=30,
    )
    cbar.minorticks_off()  # override generic yaxis settings
    fig.savefig(path)
    plt.close(fig)


def _make_line_plot(path: Path, data: np.ndarray, _: fits.Header) -> None:
    path.touch()
