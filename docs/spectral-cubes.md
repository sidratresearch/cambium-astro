# Spectral Cube Previews

`PreviewFITS` has two builtin methods for previewing spectral cubes. Both are activated only on FITS files where the cube is stored in the Primary HDU. Active by default is the "sum" handler. This uses numpy to sum the values along the spectral axis (ignoring NaNs) to create a 2D image. Alternatively, there is also the built-in "max" handler (uses `numpy.nanmax`), which can be activated by setting the following configuration in `.cambium/config.yaml`:

```yaml

stage_config:
  PreviewFITS:
    FITS_handlers:
      - "SpectralCubeMaxFITSHandler"
```

This will override the default configuration in which `SpectralCubeSumFITSHandler` is used.
