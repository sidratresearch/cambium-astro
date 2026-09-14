"""Default FITSHandler which can work with any FITS file."""

import logging
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, TypedDict

import matplotlib as mpl
import numpy as np
from astropy.io import fits
from astropy.visualization import wcsaxes
from astropy.wcs import WCS
from cambium.tree import TreeSpan
from cambium.utils.md_html_utils import wrap_with_div
from cambium.utils.path_utils import abs_leaf_path
from jinja2 import Environment
from matplotlib import pyplot as plt
from reproject import reproject_from_healpix

from .abs_fits_handler import BaseFITSFileInfo, FITSHandler, SingleHDUInfo, UUIDMapping

logger = logging.getLogger(__name__)

PreviewType = Literal["image", "table", "unavailable"]


class DFH_FITSFileInfo(BaseFITSFileInfo):
    """Info on an entire FITS file used to pass data from TreeHook to PreHook."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.image_uuids_to_filenames: dict[str, str] = {}
        self.hdu_index_to_image_uuid: dict[int, str] = {}
        self.hdu_index_to_preview_type: dict[int, PreviewType] = {}

    def add_hdu(
        self,
        index: int,
        preview_type: PreviewType,
        image_uuid: str | None = None,
        image_filename: str | None = None,
    ) -> None:
        """Add information about an HDU to this file."""
        self.hdu_index_to_preview_type[index] = preview_type

        if preview_type == "image":
            self.image_uuids_to_filenames[image_uuid] = image_filename
            self.hdu_index_to_image_uuid[index] = image_uuid


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
            "PreviewFITS-DefaultFITSHandler.html.jinja"
        )

    def matches_file(self, _: list[SingleHDUInfo]) -> bool:
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
            preview_type = choose_preview_type(single_hdu)

            # and create a new leaf for a preview image if necessary
            image_uuid, image_filename = None, None
            if preview_type == "image":
                image_filename = make_image_filename(
                    fits_path, single_hdu, image_extension
                )
                image_uuid = add_leaf(fits_path / image_filename)

            file_info.add_hdu(index, preview_type, image_uuid, image_filename)

        # return all of the UUIDs associated with this fits file
        uuid_mapping = dict.fromkeys(file_info.image_uuids_to_filenames)
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
                jinja_hdu_info = get_hdu_preview_info(i, hdu, fits_info, tree)
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
    index: int,
    hdu: fits.ImageHDU | fits.TableHDU,
    fits_info: DFH_FITSFileInfo,
    tree: TreeSpan,
) -> _JinjaHDUInfo:
    """Extract the Jinja preview info for an HDU, and create an image if necessary."""
    header, data = hdu.header, hdu.data

    match fits_info.hdu_index_to_preview_type[index]:
        case "unavailable":
            jinja_preview_data = {}
            jinja_preview_template = (
                "PreviewFITS-DefaultFITSHandler-unavailable.html.jinja"
            )

        case "image":
            image_uuid = fits_info.hdu_index_to_image_uuid[index]
            make_image(
                abs_leaf_path(tree, image_uuid),
                hdu,
                initial_path=fits_info.initial_fits_path,
            )

            jinja_preview_data = {
                "image_path": fits_info.image_uuids_to_filenames[image_uuid]
            }
            jinja_preview_template = "PreviewFITS-DefaultFITSHandler-image.html.jinja"

        case "table":
            jinja_preview_data = {"table_data": data}
            jinja_preview_template = "PreviewFITS-DefaultFITSHandler-table.html.jinja"

    return _JinjaHDUInfo(
        index=index,
        class_name=type(hdu).__name__,
        header=header,
        preview_template=jinja_preview_template,
        preview_data=jinja_preview_data,
    )


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
        # TODO: should also check anything else that might make generating an image impossible - npix
        return "image"

    if (  # regular images (also gets CompImageHDU)
        issubclass(hdu_classref, (fits.PrimaryHDU, fits.ImageHDU))
        and header.get("NAXIS") == 2
        and header.get("NAXIS1", default=0) > 0
        and header.get("NAXIS2", default=0) > 0
    ):
        return "image"

    if (  # tables
        issubclass(hdu_classref, (fits.BinTableHDU, fits.TableHDU))
        and header.get("NAXIS2", default=0) > 0
    ):
        return "table"

    return "unavailable"


def make_image_filename(
    fits_path: Path, hdu_info: SingleHDUInfo, image_extension: str
) -> str:
    """Make up a filename in which to put an image made from one HDU of a FITS file."""
    image_name_parts = [
        fits_path.stem,
        f"HDU{hdu_info.index}",
        hdu_info.hdu_class.__name__,
    ]

    hdu_name = hdu_info.header.get("EXTNAME")
    if hdu_name is not None:
        image_name_parts.append(hdu_name)

    image_name = "-".join(image_name_parts)

    return f"{image_name}.{image_extension}"


def make_image(
    path: Path, hdu: fits.ImageHDU | fits.TableHDU, initial_path: Path
) -> None:
    """Make a preview image with matplotlib.

    Accounts for WCS and HEALPix where relevant.
    """
    header, data = hdu.header, hdu.data
    if header.get("XTENSION") == "BINTABLE" and header.get("PIXTYPE") == "HEALPIX":
        _make_healpix_image(path, hdu, initial_path)
        return

    if header.get("WCSAXES") == 2 or "CRPIX1" in header:
        _make_wcs_image(path, data, header)
        return

    if header.get("NAXIS") == 1:
        _make_line_plot(path, data, header)
        return

    if header.get("NAXIS") == 2:
        _make_basic_image(path, data, header)
        return

    raise RuntimeError(f"Unclear how to make preview image {path}")


def _make_basic_image(path: Path, image_data: np.ndarray, header: fits.Header) -> None:
    """Display a colourmapped array with pixel coordinates."""
    fig, ax = plt.subplots()
    im = ax.imshow(image_data)
    cbar = fig.colorbar(im, label=header.get("BUNIT"))
    cbar.minorticks_off()  # override generic yaxis settings
    fig.savefig(path)
    # print(f"{path} saved as basic image")
    plt.close(fig)


def _make_line_plot(path: Path, data: np.ndarray, _: fits.Header) -> None:
    path.touch()
    # print(f"{path} saved as line image")


def _make_wcs_image(path: Path, image_data: np.ndarray, header: fits.Header) -> None:
    wcs = WCS(header)
    fig, ax = plt.subplots(subplot_kw={"projection": wcs})
    im = ax.imshow(image_data)
    cbar = fig.colorbar(im, label=header.get("BUNIT"))
    cbar.minorticks_off()  # override generic yaxis settings
    apply_tick_styles(ax.coords[0], "x", mpl.rcParams)
    apply_tick_styles(ax.coords[1], "y", mpl.rcParams)

    fig.savefig(path)
    # print(f"{path} saved as wcs image")
    plt.close(fig)


def _make_healpix_image(path: Path, hdu: fits.BinTableHDU, initial_path: Path) -> None:
    target_header = fits.Header(
        {
            "NAXIS": 2,
            "NAXIS1": 480,
            "NAXIS2": 240,
            "CTYPE1": "RA---MOL",
            "CRPIX1": 240.5,
            "CRVAL1": 180.0,
            "CDELT1": -0.675,
            "CUNIT1": "deg",
            "CTYPE2": "DEC--MOL",
            "CRPIX2": 120.5,
            "CRVAL2": 0.0,
            "CDELT2": 0.675,
            "CUNIT2": "deg",
            "COORDSYS": "icrs",
        }
    )
    # HACK - assuming G (galactic) coords if the file doesn't already have anything
    if "COORDSYS" not in hdu.header:
        hdu.header["COORDSYS"] = "G"
        logger.warning(
            f"Assuming galactic coordinates for HEALPix file {initial_path} without COORDSYS keyword"
        )

    images = []
    try:
        for i in range(hdu.header["TFIELDS"]):
            image_data, _ = reproject_from_healpix(hdu, target_header, field=i)
            images.append(image_data)
    except Exception as e:
        logger.warning(f"Error when creating preview HEALPix file {initial_path}: {e}")
        path.touch()
        return

    hdu.header.extend(target_header.cards, update=True)

    wcs = WCS(hdu.header)
    fig, axs = plt.subplots(
        nrows=len(images),
        subplot_kw={"projection": wcs, "frame_class": wcsaxes.frame.EllipticalFrame},
        # for some reason figsize is required to not push the axes to the far right
        figsize=(6, len(images) * 2.5),
    )
    if len(images) == 1:
        axs = [axs]
    for i, image_data in enumerate(images):
        ax = axs[i]
        im = ax.imshow(image_data)
        label = hdu.header.get(f"TTYPE{i+1}")
        unit = hdu.header.get(f"TUNIT{i+1}")
        if unit is not None:
            label = f"{label}\n({unit})"
        cbar = fig.colorbar(im, ax=ax, label=label, pad=0.1)
        cbar.minorticks_off()  # override generic yaxis settings
        apply_tick_styles(ax.coords[0], "x", mpl.rcParams)
        apply_tick_styles(ax.coords[1], "y", mpl.rcParams)

    fig.savefig(path, bbox_inches="tight")  # avoid extra space from the preset figsize
    plt.close(fig)


def apply_tick_styles(
    axis: wcsaxes.CoordinateHelper, which: Literal["x", "y"], rc_params: mpl.RcParams
) -> None:
    """Override WCS tick styling with the loaded styles."""
    generic = {
        key.removeprefix(f"{which}tick."): value
        for key, value in rc_params.find_all(f"{which}tick\\.(?!major|minor)").items()
    }
    tick_major = {
        key.removeprefix(f"{which}tick.major."): value
        for key, value in rc_params.find_all(f"{which}tick.major").items()
    }
    options = {**generic, **tick_major}

    # don't try to edit cardinal axes on elliptical frame plots
    if axis.get_axislabel_position()[0] in ["c", "h"]:
        for key in ["bottom", "top", "left", "right"]:
            options.pop(key, None)
            options.pop(f"label{key}", None)

    axis.tick_params(which="major", **options)
    axis.tick_params(
        which="minor",
        length=rc_params.find_all(f"{which}tick.minor.size")[f"{which}tick.minor.size"],
    )
