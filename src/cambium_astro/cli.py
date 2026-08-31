"""`cambium-astro` command."""

from cambium import config
from cambium.cli.cli import app
from pydantic import create_model


def main() -> None:
    """Replacement for the `cambium` command which enables astro-specific stages
    by default."""
    default_stages = config.FileConfiguration().stages
    new_default_stages = ["cambium_astro.PreviewFITS2", *default_stages]

    NewFileConfiguration = create_model(
        "newFileConfiguration",
        stages=(list[str] | None, new_default_stages),
        __base__=config.FileConfiguration,
    )
    config.FileConfiguration = NewFileConfiguration

    app()
