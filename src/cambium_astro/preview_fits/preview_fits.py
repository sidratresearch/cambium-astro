"""Cambium stage to create preview pages for FITS files."""

import logging
from pathlib import Path
from typing import Any

import matplotlib as mpl
from astropy.io import fits
from cambium.stage import Stage, StageConfig
from cambium.tree import TreeSpan
from cambium.utils.other_utils import make_jinja_environment
from cambium.utils.path_utils import (
    abs_static_stage_path,
    path_matches_patterns,
    sort_user_paths,
)
from pydantic import PositiveInt

from .fits_handler import (
    DefaultFITSHandler,
    SingleHDUInfo,
    UUIDMapping,
)

logger = logging.getLogger(__name__)
FITS_FILE_EXTENSIONS = ["fits", "fit"]
ARCHIVE_FILE_EXTENSIONS = ["gz", "zip", "bz2", "xz"]

DEFAULT_ENABLE_PATHS = [f"*.{f}" for f in FITS_FILE_EXTENSIONS] + [
    f"*.{f}.{a}" for f in FITS_FILE_EXTENSIONS for a in ARCHIVE_FILE_EXTENSIONS
]


class PreviewFITSConfig(StageConfig):
    enable_paths: list[str] = DEFAULT_ENABLE_PATHS
    disable_paths: list[str] = []
    image_filetype: str = "png"
    max_preview_rows: PositiveInt | None = 10
    mplstyle_path: Path | None = None


def _fits_path_updater(fits_path: Path) -> Path:
    return fits_path / "index.md"


class PreviewFITS(Stage):
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
        self.style_directory = Path(__file__).parent / "mplstyle"
        self.mpl_style_paths = [self.style_directory / "maple.mplstyle"]

        self.fits_handlers = {"DefaultFITSHandler": DefaultFITSHandler()}

        self.full_uuid_mapping: UUIDMapping = {}

    # --------------------------------------------------------------------#
    #                        Tree hook + helpers                          #
    # --------------------------------------------------------------------#

    def tree_hook(self, tree: TreeSpan) -> None:
        # get what the actual path of the CSS file will be in the build directory
        static_dir = abs_static_stage_path(tree, self.__class__.__name__).relative_to(
            tree.build_directory
        )
        self.css_link = static_dir / self.css_file

        self._setup_matplotlib(tree)

        # get jinja templates
        jinja_environment = make_jinja_environment(tree)
        for handler_instance in self.fits_handlers.values():
            handler_instance.post_init(jinja_environment, tree)

        # cast the deque to a list so that we can add new leaves to the end
        # we don't want to re-visit the added leaves anyway
        for leaf_uuid in list(tree.leaves["uuids"]):
            initial_path = tree.leaves["initial_path"][leaf_uuid]

            if path_matches_patterns(
                initial_path, self.enable_patterns
            ) and not path_matches_patterns(initial_path, self.disable_patterns):
                self._tree_hook_for_fits(leaf_uuid, initial_path, tree)

    def _tree_hook_for_fits(
        self, preview_page_uuid: str, fits_path: Path, tree: TreeSpan
    ) -> None:
        hdu_info = []
        with fits.open(fits_path) as hdu_list:
            for i, hdu in enumerate(hdu_list):
                hdu_info.append(
                    SingleHDUInfo(index=i, hdu_class=type(hdu), header=hdu.header)
                )

        add_leaf = lambda path: self.add_leaf(path, tree)

        for handler_instance in self.fits_handlers.values():
            if handler_instance.matches_file(hdu_info):
                uuid_mapping = handler_instance.make_uuid_mapping(
                    preview_page_uuid,
                    hdu_info,
                    fits_path,
                    add_leaf,
                    self.config.image_filetype,
                )
                break

        tree.update_leaf_path(preview_page_uuid, "final", _fits_path_updater)
        tree.update_leaf_path(preview_page_uuid, "latest", _fits_path_updater)
        for uuid in uuid_mapping:
            self._register_hook(uuid, tree, "pre_hooks")

        self.full_uuid_mapping.update(uuid_mapping)

    def _setup_matplotlib(self, tree: TreeSpan) -> None:
        # load fonts into matplotlib
        font_directories = [self.style_directory] + [
            d for d, _ in tree.config.static_directories["theme"]
        ]
        font_files = mpl.font_manager.findSystemFonts(fontpaths=font_directories)
        for font_file in font_files:
            mpl.font_manager.fontManager.addfont(font_file)

        # load style files
        if self.config.mplstyle_path is not None:
            style_path = tree.root_directory / self.config.mplstyle_path
            if not style_path.exists():
                raise FileNotFoundError(
                    f"Matplotlib style file `{self.config.mplstyle_path}` not found in {tree.root_directory.absolute()}"
                )
            self.mpl_style_paths.append(style_path)
        mpl.pyplot.style.use(self.mpl_style_paths)

    # --------------------------------------------------------------------#
    #                              Pre hook                               #
    # --------------------------------------------------------------------#

    def pre_hook(self, leaf_uuid: str, tree: TreeSpan) -> None:
        file_info = self.full_uuid_mapping[leaf_uuid]

        # don't run anything for (what will become) FITS or image files
        if file_info is None:
            return

        handler_instance = self.fits_handlers[file_info.preview_handler_name]
        handler_instance.write_files(file_info, self.config.max_preview_rows, tree)

        # TODO: make css file come from the handler? then different handlers need
        # to have different CSS files, ideally without copying all of them into _build
        self._set_css_include(self.css_file, file_info.preview_page_uuid, tree)
