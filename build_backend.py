"""Setuptools builds with dependencies selected from the kernel manifest."""

import os
import tomllib
from pathlib import Path

from setuptools.build_meta import build_editable as build_editable
from setuptools.build_meta import build_sdist as build_sdist
from setuptools.build_meta import build_wheel as build_wheel
from setuptools.build_meta import prepare_metadata_for_build_editable as prepare_metadata_for_build_editable
from setuptools.build_meta import prepare_metadata_for_build_wheel as prepare_metadata_for_build_wheel


def get_requires_for_build_wheel(config_settings=None):
    manifest = tomllib.loads((Path(__file__).parent / "prime_kernels/kernels.toml").read_text())
    selected = {name for name in os.environ.get("PRIME_KERNELS", "").split(",") if name}
    unknown = selected - manifest.keys()
    if unknown:
        raise ValueError(f"PRIME_KERNELS contains unknown kernel(s): {', '.join(sorted(unknown))}")
    return sorted(
        {
            requirement
            for name, kernel in manifest.items()
            if not selected or name in selected
            for requirement in kernel.get("build-requires", [])
        }
    )


get_requires_for_build_editable = get_requires_for_build_wheel
get_requires_for_build_sdist = get_requires_for_build_wheel
