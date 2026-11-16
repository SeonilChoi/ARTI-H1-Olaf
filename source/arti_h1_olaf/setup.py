"""Installation script for the 'arti_h1_olaf' python package."""

import os

import toml
from setuptools import find_packages, setup

EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
EXTENSION_TOML_DATA = toml.load(os.path.join(EXTENSION_PATH, "config", "extension.toml"))

setup(
    name="arti_h1_olaf",
    packages=find_packages(),
    author=EXTENSION_TOML_DATA["package"]["author"],
    maintainer=EXTENSION_TOML_DATA["package"]["maintainer"],
    url=EXTENSION_TOML_DATA["package"]["repository"],
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    install_requires=["psutil"],
    license="Apache-2.0",
    include_package_data=True,
    package_data={
        "arti_h1_olaf": [
            "assets/meshes/arti_h1/*.STL",
            "assets/models/arti_h1/*.urdf",
            "assets/models/arti_h1/arti_h1/*.usd",
            "assets/models/arti_h1/arti_h1/configuration/*.usd",
            "assets/reference/*.npz",
            "assets/thermal/*.json",
        ]
    },
    # `isaaclab train|play|list_envs` import every package registered under this group
    entry_points={"isaaclab.tasks": ["arti_h1_olaf = arti_h1_olaf.tasks"]},
    python_requires=">=3.12",
    zip_safe=False,
)
