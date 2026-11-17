"""RevORCHP-SM: surrogate models for reversible ORC / heat-pump modules.

Modules
-------
design_ann_vopt6_predictor
    Nominal design of one 10 kg/s reversible module (``DesignANNModule``).
reversible_design_ann_vopt6_wrapper
    HP-priority and ORC-priority design of a plant made of identical modules
    (``ReversibleDesignANN``).
hp_offdesign_surrogate_1mw_filtered_variants_vopt6
    Off-design heat-pump performance of the fixed 1 MW nominal hardware
    (``predict_1mw_hp_performance``).
orc_offdesign_surrogate_filtered_variants_vopt6
    Off-design ORC performance of the fixed one- or two-stage nominal hardware
    (``predict_orc_performance_for_design``, ``predict_1mw_orc_performance``).

Units in the Python API (see README.md): temperatures in °C, pressures in
absolute bar, heat and power in W, mass flow in kg/s, temperature differences
in K.
"""
from pathlib import Path

__version__ = "1.0.2"


def bundled_model_path(filename: str) -> Path:
    """Path of a model file shipped with the code.

    After ``pip install`` the models are inside the package (revorchp_sm/models).
    In a copy of the repository they are in the top-level models/ folder.

    Parameters
    ----------
    filename : str
        File name of the bundle, e.g. ``"design_ann_vopt6_module.joblib"``.

    Returns
    -------
    pathlib.Path
        The first of the two locations above where the file exists. If it
        exists in neither, the path inside the package is returned anyway
        (loading it then raises ``FileNotFoundError``).
    """
    package_dir = Path(__file__).resolve().parent
    for folder in (package_dir / "models", package_dir.parents[1] / "models"):
        if (folder / filename).is_file():
            return folder / filename
    return package_dir / "models" / filename
