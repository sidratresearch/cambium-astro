"""Default FITSHandler which can work with any FITS file."""

import logging

from cambium.tree import TreeSpan
from jinja2 import Environment

from . import default_fits_handler as dfh
from ._fits_handler import SingleHDUInfo

logger = logging.getLogger(__name__)


class ReversedFITSHandler(dfh.DefaultFITSHandler):
    """FITS preview handler which puts the PrimaryHDU at the bottom."""

    def post_init(self, jinja_environment: Environment, _: TreeSpan) -> None:
        self.jinja_template = jinja_environment.get_template(
            "ReversedFITSHandler/PreviewFITS-ReversedFITSHandler.html.jinja"
        )

    @classmethod
    def matches_file(cls, hdus: list[SingleHDUInfo]) -> bool:
        """Matches if the PrimaryHDU has no preview-able data."""
        primary_hdu = hdus[0]
        return dfh.choose_preview_type(primary_hdu) == "unavailable"
