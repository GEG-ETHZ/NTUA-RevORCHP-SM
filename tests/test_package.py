"""Check that the package finds the model files it ships with.

After `pip install`, the models are copied into the installed package; this
checks that they are byte-for-byte the files in models/.
"""
from __future__ import annotations

from cases import REPO
from revorchp_sm import bundled_model_path


def test_bundled_models_are_the_repository_models():
    models = sorted((REPO / "models").glob("*.joblib"))
    assert len(models) == 4
    for path in models:
        assert bundled_model_path(path.name).read_bytes() == path.read_bytes(), path.name
