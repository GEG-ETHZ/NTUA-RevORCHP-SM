"""Small, stat-aware cache for repeated local prediction calls.

The off-design surrogates (HP and ORC) load their ``.joblib`` model bundles
through :func:`load_bundle`, so that calling a prediction function many times
does not read the file from disk every time.
"""
from functools import lru_cache
from pathlib import Path
import joblib
from threadpoolctl import threadpool_limits


@lru_cache(maxsize=4)
def _load(path: str, modified_ns: int, size: int):
    """Load a bundle with joblib; cached on path, modification time and size.

    ``modified_ns`` and ``size`` are not used in the body. They are part of the
    cache key, so a bundle file that is replaced on disk is loaded again.
    At most four bundles are kept in the cache.
    """
    return joblib.load(path)


def load_bundle(path: str | Path):
    """Load a model bundle (``.joblib`` file), reusing it if already loaded.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the ``.joblib`` bundle file.

    Returns
    -------
    object
        The object stored in the file. For the bundles shipped in ``models/``
        this is a ``dict`` holding the fitted scikit-learn models and their
        metadata.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.

    Notes
    -----
    The same object is returned to every caller while it is cached, so callers
    must not modify it.
    """
    resolved = Path(path).resolve()
    info = resolved.stat()
    return _load(str(resolved), info.st_mtime_ns, info.st_size)


def clear_bundle_cache() -> None:
    """Forget all bundles cached by :func:`load_bundle`."""
    _load.cache_clear()


def predict_model(model, inputs):
    """Call ``model.predict(inputs)`` using a single native thread.

    Parameters
    ----------
    model : object
        A fitted scikit-learn estimator (anything with a ``predict`` method).
    inputs : numpy.ndarray
        Feature array of shape ``(n_samples, n_features)``.

    Returns
    -------
    numpy.ndarray
        Whatever ``model.predict`` returns. Limiting the number of threads
        only affects speed, not the predicted values.
    """
    # Tiny per-module calls benefit from one native worker; avoid OpenMP startup
    # overhead for each target of a multioutput boosted model.
    with threadpool_limits(limits=1):
        return model.predict(inputs)
