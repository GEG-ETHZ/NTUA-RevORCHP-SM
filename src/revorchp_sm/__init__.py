"""RevORCHP-SM: surrogate models for reversible ORC / heat-pump modules."""
from pathlib import Path

__version__ = "1.0.2"


def bundled_model_path(filename: str) -> Path:
    """Path of a model file shipped with the code.

    After ``pip install`` the models are inside the package (revorchp_sm/models).
    In a copy of the repository they are in the top-level models/ folder.
    """
    package_dir = Path(__file__).resolve().parent
    for folder in (package_dir / "models", package_dir.parents[1] / "models"):
        if (folder / filename).is_file():
            return folder / filename
    return package_dir / "models" / filename
