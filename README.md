# Astronomy Tools for Cambium

This package provides astronomy-specific tooling for the static site generator [Cambium](https://github.com/sidratresearch/cambium).

> [!WARNING]
> This package is under active development. Provided functionality may be incomplete and is subject to major changes.

Currently `cambium-astro` includes the `PreviewFITS` stage. Much like the built-in `PreviewCSV` stage, `PreviewFITS` replaces markdown links that point at FITS files with links to new webpages which show a preview of the content (as well as a download link to the original file!).

Try it out with one of the following methods:

- Install this package, and run `cambium-astro` in place of the regular `cambium` command.
  - This method is only recommended for trying out Cambium Astro, as it cannot be combined with any other Cambium packages
- Edit your Cambium configuration file to include `cambium_astro` in the `extensions` and `cambium_astro.PreviewFITS` in the stages.
  - If you do not have a configuration file, run `cambium --dump-default-config` to display the full default configuration, and make the above changes.
  - Note that you do not need to keep all of the configuration keys - only the ones changed from their default values.

```yaml
# Sample configuration to run Cambium with Cambium Astro
# Do not copy this configuration!
# Cambium is fast-changing and the default stages listed here may be out of date
# Create your own configuration file with `cambium --dump-default-config`

stages:
  - PreviewCSV
  - cambium_astro.PreviewFITS
  - IdentifyMetadata
  - WriteReports
  - TransformMarkdown
  - TemplateMarkdown
  - EnsureIndexPages
  - PagefindSearch
  - CheckLinks

extensions:
  - cambium_astro
```
