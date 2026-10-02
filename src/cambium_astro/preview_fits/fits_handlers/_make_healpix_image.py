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


def _make_healpix_image(path: Path, hdu: fits.BinTableHDU, initial_path: Path) -> None:
    n_images = hdu.header["TFIELDS"]

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

    fig, axs = plt.subplots(
        nrows=n_images,
    )
    if n_images == 1:
        axs = [axs]

    for i in range(n_images):

        image_data, _ = reproject_from_healpix(hdu, target_header, field=i)
        m = Basemap(projection="moll", lon_0=0, celestial=True, ax=axs[i])

        im_min, im_max = np.nanpercentile(image_data, [1, 99])
        m.imshow(image_data, vmin=im_min, vmax=im_max)

        label = hdu.header.get(f"TTYPE{i+1}")
        unit = hdu.header.get(f"TUNIT{i+1}")
        if unit is not None:
            label = f"{label}\n({unit})"
        cbar = m.colorbar(
            label=label,
            pad=0.1,
            extend="both",
            location="bottom",
            shrink=0.8,
            aspect=30,
        )
        cbar.minorticks_off()  # override generic yaxis settings

    fig.savefig(path)
    plt.close(fig)
