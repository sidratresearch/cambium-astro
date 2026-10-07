"""Handler to use when a FITS file cannot even be opened."""

import re
from collections.abc import Callable
from pathlib import Path

from cambium.tree import TreeSpan
from cambium.utils.path_utils import abs_leaf_path
from jinja2 import Environment

from ._fits_handler import FITSHandler, SingleHDUInfo, UUIDMapping
from .default_fits_handler import DFH_FITSFileInfo


class CorruptFITSHandler(FITSHandler):

    @classmethod
    def matches_file(cls) -> None:
        """Not used for matching files."""
        return False

    def post_init(self, jinja_environment: Environment, _: TreeSpan) -> None:
        self.jinja_template = jinja_environment.get_template(
            "CorruptFITSHandler.html.jinja"
        )

    def make_uuid_mapping(
        self,
        preview_page_uuid: str,
        hdu_info: list[SingleHDUInfo],
        fits_path: Path,
        add_leaf: Callable[[Path], str],
        image_extension: str,
    ) -> UUIDMapping:
        fits_file_uuid = add_leaf(fits_path / fits_path.name)

        file_info = DFH_FITSFileInfo(
            initial_fits_path=fits_path,
            fits_file_uuid=fits_file_uuid,
            preview_page_uuid=preview_page_uuid,
            n_hdus=0,
            preview_handler_name=self.__class__.__name__,
        )

        uuid_mapping = {}
        uuid_mapping[preview_page_uuid] = None
        uuid_mapping[fits_file_uuid] = file_info
        return uuid_mapping

    def write_files(
        self, fits_info: DFH_FITSFileInfo, max_preview_rows: int, tree: TreeSpan
    ) -> None:
        self._copy_fits_to_tmp(fits_info, tree)
        fits_path = fits_info.initial_fits_path

        preview_page_content = self.jinja_template.render(
            download_info={"link": fits_path.name, "size": fits_path.stat().st_size}
        )

        # strip all indentation and newlines (safe since we have no <pre> tags)
        # prevents marko from thinking there's an indented code block
        replaced = re.sub(r"^\s*", "", preview_page_content, flags=re.MULTILINE)

        abs_leaf_path(tree, fits_info.preview_page_uuid).write_text(replaced)
