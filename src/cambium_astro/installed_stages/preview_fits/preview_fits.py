"""Cambium stage to create preview pages for FITS files."""

import logging
from pathlib import Path
from typing import Any, Literal

from astropy.io import fits
from cambium.builtin_stages.utils import (
    WrappedBlocksMixin,
    get_relative_path_modifier,
    path_matches_patterns,
)
from cambium.config import sort_user_paths
from cambium.stage import Stage, StageConfig
from cambium.tree import TreeSpan
from jinja2 import Environment, FileSystemLoader
from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)


class PreviewFITSConfig(StageConfig):
    enable_paths: list[str] = ["*.fits", "*.fit"]
    disable_paths: list[str] = []
    image_filetype: str = "jpg"


class PreviewFITS(Stage):
    def _fits_path_updater(self, fits_path: Path) -> Path:
        return fits_path / "index.md"

    def __init__(self, config_dict: dict[str, Any]) -> None:
        self.config = PreviewFITSConfig.model_validate(config_dict)
        self.enable_patterns = sort_user_paths(self.config.enable_paths)
        self.disable_patterns = sort_user_paths(self.config.disable_paths)
        self.image_suffix = "." + self.config.image_filetype

        self.requires = []
        self.runs_before = ["TransformMarkdown", "IdentifyMetadata"]
        self.runs_after = []

        # store mappings between the preview leaves and the data leaves
        self.fits_to_md = {}
        self.md_to_fits = {}
        self.image_to_fits = {}
        self.fits_to_image = {}
        self.uuid_to_type: dict[str, Literal["md", "fits", "image"]] = {}

        self.css_file = "css/preview_fits.css"
        # path from includes/static to the CSS file we want to import on preview pages

    def tree_hook(self, tree: TreeSpan) -> None:
        # get what the actual path of the CSS file will be in the build directory
        static_dir = tree.abs_static_stage_path(self.__class__.__name__).relative_to(
            tree.build_directory
        )
        self.css_link = static_dir / self.css_file

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
        # change the final path for this leaf to be subfoldered and end in md
        tree.update_leaf_path(md_uuid, "final", self._fits_path_updater)
        tree.update_leaf_path(md_uuid, "latest", self._fits_path_updater)
        self._register_hook(md_uuid, tree, "pre_hooks")

        # create a companion leaf for the new data location
        fits_uuid = self.add_leaf(
            fits_initial_path,
            tree,
            final_path=fits_initial_path / fits_initial_path.name,
        )
        self._register_hook(fits_uuid, tree, "pre_hooks")

        # create a companion leaf for the image
        image_uuid = self.add_leaf(
            fits_initial_path.with_suffix(self.image_suffix),
            tree,
            final_path=fits_initial_path
            / fits_initial_path.with_suffix(self.image_suffix).name,
        )
        self._register_hook(image_uuid, tree, "pre_hooks")

        self.md_to_fits[md_uuid] = fits_uuid
        self.fits_to_md[fits_uuid] = md_uuid
        self.image_to_fits[image_uuid] = fits_uuid
        self.fits_to_image[fits_uuid] = image_uuid
        self.uuid_to_type[md_uuid] = "md"
        self.uuid_to_type[fits_uuid] = "fits"
        self.uuid_to_type[image_uuid] = "image"

        logger.info(f"Creating preview page for {fits_initial_path}")

    def pre_hook(self, leaf_uuid: str, tree: TreeSpan) -> None:
        # figure out if this is the index.html
        match self.uuid_to_type[leaf_uuid]:
            case "fits":
                self._pre_hook_fits(leaf_uuid, tree)
            case "md":
                self._pre_hook_md(leaf_uuid, tree)
            case "image":
                self._pre_hook_image(leaf_uuid, tree)

    def _pre_hook_md(self, md_uuid: str, tree: TreeSpan) -> None:
        fits_uuid = self.md_to_fits[md_uuid]
        preview_content = get_md_content(
            md_uuid,
            fits_uuid,
            tree,
            self.image_suffix,
            {
                "css_link": self.css_link,
                "relative_path_modifier": get_relative_path_modifier(
                    tree.leaves["final_path"][md_uuid]
                ),
            },
        )
        tree.abs_leaf_path(md_uuid).write_text(preview_content)

    def _pre_hook_fits(self, fits_uuid: str, tree: TreeSpan) -> None:
        md_uuid = self.fits_to_md[fits_uuid]
        fits_content = tree.leaves["initial_path"][md_uuid].read_bytes()
        tree.abs_leaf_path(fits_uuid).write_bytes(fits_content)

    def _pre_hook_image(self, image_uuid: str, tree: TreeSpan) -> None:
        fits_path = tree.leaves["initial_path"][
            self.fits_to_md[self.image_to_fits[image_uuid]]
        ]
        logger.debug(f"Reading FITS file {fits_path}")
        fits_data = fits.getdata(fits_path)

        image_path = tree.abs_leaf_path(image_uuid)

        style_path = Path(__file__).parent / "root.mplstyle/root.mplstyle"

        plt.style.use(style_path)
        plt.imshow(fits_data)
        plt.savefig(image_path)


def get_md_content(
    md_uuid: str,
    fits_uuid: str,
    tree: TreeSpan,
    image_suffix: str,
    jinja_variables: dict[str, Any],
) -> str:
    """Get the content for the Markdown preview page."""
    fits_path = tree.leaves["initial_path"][md_uuid]
    download_filename = tree.leaves["initial_path"][fits_uuid].name
    fits_header = fits.getheader(fits_path)

    jinja_environment = Environment(
        loader=FileSystemLoader(tree.config.template_directories),
        lstrip_blocks=True,
        trim_blocks=True,  # stops Jinja lines from being replaced with newlines
        # if not enabled, Marko doesn't recognize the table as being a single HTMLBlock
    )

    template = jinja_environment.get_template("preview-fits.md.jinja")

    return template.render(
        download_filename=download_filename,
        img_src="./" + fits_path.with_suffix(image_suffix).name,
        fits_filesize=fits_path.stat().st_size,
        fits_header=fits_header,
        cambium_wrap=WrappedBlocksMixin.wrap_anything,
        **jinja_variables,
    )
