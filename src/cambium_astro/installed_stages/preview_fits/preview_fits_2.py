"""Cambium stage to create preview pages for FITS files."""

import logging
import re
from pathlib import Path
from typing import Any, Literal, TypedDict

import numpy as np
from astropy.io import fits
from cambium.builtin_stages.utils import WrappedBlocksMixin, get_relative_path_modifier
from cambium.stage import Stage, StageConfig
from cambium.tree import TreeSpan
from cambium.utils import path_matches_patterns, sort_user_paths
from jinja2 import Environment, FileSystemLoader
from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)


class PreviewFITSConfig(StageConfig):
    enable_paths: list[str] = ["*.fits", "*.fit"]
    disable_paths: list[str] = []
    image_filetype: str = "png"


class UUIDMapping(TypedDict):
    preview_index: int
    """Index of `PreviewFITS2.preview_objs` that holds info relevant to this UUID."""
    leaf_type: Literal["md", "fits", "image"]


class HDUInfo(TypedDict):
    image_uuid: str | None
    """UUID of leaf in which to store an image, if any."""
    display_as_table: bool
    """Whether the data attribute should be read and displayed as a table. Mutually
    exclusive with `image_uuid`"""


def _fits_path_updater(fits_path: Path) -> Path:
    return fits_path / "index.md"


def data_needs_image_leaf(hdu_data: np.ndarray | None) -> bool:
    """Check if the given HDU data can be previewed as an image.

    Should only recieve data from PrimaryHDU and ImageHDU objects, not TableHDUs.
    """
    if hdu_data is None:
        return False
    if hdu_data.size == 0:
        return False
    return hdu_data.ndim == 2


def get_image_path(
    fits_path: Path,
    hdu: fits.PrimaryHDU | fits.ImageHDU,
    hdu_number: int,
    image_suffix: str,
) -> Path:
    """Make up a filepath in which to put an image made from one HDU of a FITS file."""
    base_image_name = fits_path.stem

    hdu_name = f"HDU{hdu_number}"
    if isinstance(hdu, fits.PrimaryHDU):
        hdu_name += "-PrimaryHDU"
    else:
        hdu_name += "-ImageHDU"
        if hdu.name is not None:
            hdu_name += f"-{hdu.name}"

    image_name = f"{base_image_name}-{hdu_name}.{image_suffix}"

    return fits_path / image_name


class _Preview:
    def __init__(
        self,
        fits_initial_path: Path,
        md_uuid: str,
        caller: "PreviewFITS2",
        tree: TreeSpan,
    ) -> None:
        # store the markdown leaf
        self.md_uuid = md_uuid

        # create a companion leaf for the new data location
        self.fits_uuid = caller.add_leaf(
            # setting initial_path as fits_initial_path errors
            # during the copy stage, the source path is a directory
            fits_initial_path / fits_initial_path.name,
            tree,
            final_path=fits_initial_path / fits_initial_path.name,
        )

        # create image leaves
        self.image_uuids = []
        self.hdu_info: list[HDUInfo] = []
        with fits.open(fits_initial_path) as hdu_list:
            if isinstance(hdu_list[0], (fits.GroupsHDU, fits.StreamingHDU)):
                raise RuntimeError(
                    f"Can't handle FITS file {fits_initial_path} - incompatible HDU type."
                )

            # handle the PrimaryHDU
            primary_needs_leaf = data_needs_image_leaf(hdu_list[0].data)
            if primary_needs_leaf:
                image_path = get_image_path(
                    fits_initial_path, hdu_list[0], 0, caller.config.image_filetype
                )
                image_uuid = caller.add_leaf(image_path, tree)
                self.image_uuids.append(image_uuid)
                self.hdu_info.append(
                    HDUInfo(image_uuid=image_uuid, display_as_table=False)
                )
            else:
                self.hdu_info.append(HDUInfo(image_uuid=None, display_as_table=False))

            # handle the remaining HDUs
            for i, hdu in enumerate(hdu_list[1:]):
                if isinstance(hdu, (fits.TableHDU, fits.BinTableHDU)):
                    self.hdu_info.append(
                        HDUInfo(image_uuid=None, display_as_table=True)
                    )

                elif isinstance(hdu, fits.ImageHDU):
                    if not data_needs_image_leaf(hdu.data):
                        logger.warning(
                            f"Not creating preview image for {fits_initial_path} HDU {i}"
                        )
                        self.hdu_info.append(
                            HDUInfo(image_uuid=None, display_as_table=False)
                        )
                        continue
                    image_path = get_image_path(
                        fits_initial_path, hdu, i + 1, caller.config.image_filetype
                    )
                    image_uuid = caller.add_leaf(image_path, tree)

                    self.image_uuids.append(image_uuid)
                    self.hdu_info.append(
                        HDUInfo(image_uuid=image_uuid, display_as_table=False)
                    )

                else:
                    raise RuntimeError(f"Unknown HDU type {type(hdu)}")

        # move the fits file
        tree.update_leaf_path(self.md_uuid, "final", _fits_path_updater)
        tree.update_leaf_path(self.md_uuid, "latest", _fits_path_updater)

        # register hooks
        caller._register_hook(self.md_uuid, tree, "pre_hooks")
        caller._register_hook(self.fits_uuid, tree, "pre_hooks")
        for image_uuid in self.image_uuids:
            caller._register_hook(image_uuid, tree, "pre_hooks")

    def update_uuid_mapping(self, index: int, stage: "PreviewFITS2") -> None:
        """Update the stage's mapping to include the newly created leaves."""
        stage.uuid_mapping[self.md_uuid] = UUIDMapping(
            preview_index=index, leaf_type="md"
        )
        stage.uuid_mapping[self.fits_uuid] = UUIDMapping(
            preview_index=index, leaf_type="fits"
        )
        for image_uuid in self.image_uuids:
            stage.uuid_mapping[image_uuid] = UUIDMapping(
                preview_index=index, leaf_type="image"
            )


class PreviewFITS2(Stage):
    def __init__(self, config_dict: dict[str, Any]) -> None:
        # Cambium stage management
        self.requires = []
        self.runs_before = ["TransformMarkdown", "IdentifyMetadata"]
        self.runs_after = []

        # config handling
        self.config = PreviewFITSConfig.model_validate(config_dict)
        self.enable_patterns = sort_user_paths(self.config.enable_paths)
        self.disable_patterns = sort_user_paths(self.config.disable_paths)

        # other long-term storage
        self.css_file = "css/preview_fits.css"
        # path from includes/static to the CSS file we want to import on preview pages
        style_path = Path(__file__).parent / "root.mplstyle/root.mplstyle"
        plt.style.use(style_path)

        # store the files we'll operate on
        self.preview_objs: list[_Preview] = []
        self.uuid_mapping: dict[str, UUIDMapping] = {}

    # --------------------------------------------------------------------#
    #                        Tree hook + helpers                          #
    # --------------------------------------------------------------------#

    def tree_hook(self, tree: TreeSpan) -> None:
        # get what the actual path of the CSS file will be in the build directory
        static_dir = tree.abs_static_stage_path(self.__class__.__name__).relative_to(
            tree.build_directory
        )
        self.css_link = static_dir / self.css_file

        self._get_jinja_template(tree)

        # cast the deque to a list so that we can add new leaves to the end
        # we don't want to re-visit the added leaves anyway
        for leaf_uuid in list(tree.leaves["uuids"]):
            initial_path = tree.leaves["initial_path"][leaf_uuid]

            if path_matches_patterns(
                initial_path, self.enable_patterns
            ) and not path_matches_patterns(initial_path, self.disable_patterns):
                self._tree_hook_for_fits(leaf_uuid, initial_path, tree)

    def _tree_hook_for_fits(
        self, md_uuid: str, fits_initial_path: Path, tree: TreeSpan
    ) -> None:
        preview = _Preview(fits_initial_path, md_uuid, self, tree)
        preview.update_uuid_mapping(len(self.preview_objs), self)
        self.preview_objs.append(preview)

    def _get_jinja_template(self, tree: TreeSpan) -> None:
        jinja_environment = Environment(
            loader=FileSystemLoader(tree.config.template_directories),
            lstrip_blocks=True,
            trim_blocks=True,  # stops Jinja lines from being replaced with newlines
            # if not enabled, Marko doesn't recognize the table as being a single HTMLBlock
        )

        self.md_template = jinja_environment.get_template("preview-fits.html.jinja")

    # --------------------------------------------------------------------#
    #                         Pre hook + helpers                          #
    # --------------------------------------------------------------------#

    def pre_hook(self, leaf_uuid: str, tree: TreeSpan) -> None:
        leaf_type = self.uuid_mapping[leaf_uuid]["leaf_type"]
        preview_index = self.uuid_mapping[leaf_uuid]["preview_index"]
        preview = self.preview_objs[preview_index]
        match leaf_type:
            case "md":
                self._pre_hook_md(leaf_uuid, preview, tree)
            case "fits":
                self._pre_hook_fits(leaf_uuid, preview, tree)
            case "image":
                self._pre_hook_image(leaf_uuid, preview, tree)

    def _pre_hook_md(self, md_uuid: str, preview: _Preview, tree: TreeSpan) -> None:
        # download link: ./fits name
        # download size
        # primary headers
        # maybe primary image
        # some quantity of additional HDUs (header, table, image)

        fits_path = tree.leaves["initial_path"][preview.md_uuid]
        download_info = {
            "link": fits_path.name,
            "size": fits_path.stat().st_size,
        }

        hdu_previews = []

        with fits.open(fits_path) as hdu_list:
            for i, hdu in enumerate(hdu_list):
                hdu_preview = {"hdu_type": type(hdu).__name__, "header": hdu.header}
                hdu_info = preview.hdu_info[i]
                if hdu_info["image_uuid"] is not None:
                    hdu_preview["image_link"] = tree.leaves["final_path"][
                        hdu_info["image_uuid"]
                    ].name
                    hdu_preview["preview_type"] = "image"
                if hdu_info["display_as_table"]:
                    hdu_preview["table_data"] = hdu.data
                    hdu_preview["preview_type"] = "table"

                hdu_previews.append(hdu_preview)

        md_content = self.md_template.render(
            download_info=download_info,
            hdu_entries=hdu_previews,
            # shared across all leaves
            cambium_wrap=WrappedBlocksMixin.wrap_anything,
            css_link=self.css_link,
            relative_path_modifier=get_relative_path_modifier(
                tree.leaves["final_path"][md_uuid]
            ),
        )
        # strip all indentation and newlines (safe since we have no <pre> tags)
        # prevents marko from thinking there's an indented code block
        replaced = re.sub(r"^\s*", "", md_content, flags=re.MULTILINE)
        tree.abs_leaf_path(md_uuid).write_text(replaced)

    def _pre_hook_fits(self, fits_uuid: str, preview: _Preview, tree: TreeSpan) -> None:
        """Copy the FITS data from its original location to the new FITS leaf."""
        md_uuid = preview.md_uuid
        fits_content = tree.leaves["initial_path"][md_uuid].read_bytes()
        tree.abs_leaf_path(fits_uuid).write_bytes(fits_content)

    def _pre_hook_image(
        self, image_uuid: str, preview: _Preview, tree: TreeSpan
    ) -> None:
        for i, hdu_info in enumerate(preview.hdu_info):
            if hdu_info["image_uuid"] == image_uuid:
                hdu_index = i
                break

        fits_filepath = tree.leaves["initial_path"][preview.md_uuid]

        with fits.open(fits_filepath) as hdu_list:
            image_hdu = hdu_list[hdu_index]
            image_header, image_data = image_hdu.header, image_hdu.data

        plt.imshow(image_data)
        plt.savefig(tree.abs_leaf_path(image_uuid))
