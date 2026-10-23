import logging
from pathlib import Path

import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)
try:
    from mpl_toolkits.basemap import Basemap
    from reproject import reproject_from_healpix

    HEALPIX_AVAILABLE = True

except ImportError:
    HEALPIX_AVAILABLE = False
    logger.warning(
        "Previews for HEALPix images are not available. "
        "Install the HEALPix dependency group with `pip install cambium-astro[healpix]`"
    )


def _make_healpix_images(
    paths: list[Path], hdu: fits.BinTableHDU, initial_path: Path
) -> None:
    n_images = len(paths)

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
    # assuming G (galactic) coords if the file doesn't already have anything
    if "COORDSYS" not in hdu.header:
        hdu.header["COORDSYS"] = "G"
        logger.warning(
            f"Assuming galactic coordinates for HEALPix file {initial_path} without COORDSYS keyword"
        )

    for i in range(n_images):
        fig = plt.figure(figsize=(6, 4))

        im_ax_pad, im_ax_height = 0.01, 0.7
        im_ax = fig.add_axes(
            (im_ax_pad, 1 - im_ax_height - im_ax_pad, 1 - 2 * im_ax_pad, im_ax_height)
        )
        image_data, _ = reproject_from_healpix(hdu, target_header, field=i)
        m = Basemap(projection="moll", lon_0=0, celestial=True, ax=im_ax)
        im_min, im_max = np.nanpercentile(image_data, [1, 99])
        im = m.imshow(image_data, vmin=im_min, vmax=im_max)

        # fix the clipping of the outline
        # https://github.com/matplotlib/basemap/issues/344
        for p in im_ax.patches:
            p.set_clip_on(False)

        im_bbox = im_ax.get_position()
        cbar_ax_xpad, cbar_ax_ypad, cbar_ax_height = im_ax_pad + 0.01, 0.05, 0.04
        cbar_ax_xpad = 0.01
        cbar_ax = fig.add_axes(
            (
                im_bbox.x0 + cbar_ax_xpad,
                im_bbox.y0 - cbar_ax_ypad - cbar_ax_height,
                im_bbox.width - 2 * cbar_ax_xpad,
                cbar_ax_height,
            )
        )
        label = hdu.header.get(f"TTYPE{i+1}")
        unit = hdu.header.get(f"TUNIT{i+1}")
        if unit is not None:
            label = f"{label}\n({unit})"
        cbar = fig.colorbar(
            im,
            cax=cbar_ax,
            label=label,
            pad=0.1,
            extend="both",
            location="bottom",
        )
        cbar.minorticks_off()  # override generic yaxis settings
        cbar_fontsize = plt.rcParams["font.size"] - 2
        cbar_ax.tick_params(axis="x", which="major", length=2, labelsize=cbar_fontsize)

        fig.savefig(paths[i])
        plt.close(fig)
