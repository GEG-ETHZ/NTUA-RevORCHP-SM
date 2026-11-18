# Tasks

Each item is a separate pull request, so it can be reviewed on its own. Owners are agreed in `design.md`.

## 0. Agreement

- [ ] 0.1 Go through this proposal and the open questions in `design.md` with the authors; record the answers and who does each item.

## 1. Regression tests and CI

- [ ] 1.1 Tests against stored reference outputs, CI on Linux and Windows (drafted: branch `test/regression-baseline`). Verify the tests pass on unchanged code and fail when a stored value is changed.

## 2. Installable package

- [ ] 2.1 Package `revorchp_sm` with `pyproject.toml`, model files included, top-level scripts kept, dependency ranges (drafted: branch `build/installable-package`). Verify the stored outputs are unchanged when installed, with the oldest and the newest allowed dependency versions.

## 3. Docstrings

- [ ] 3.1 Docstrings and type hints (drafted: branch `docs/docstrings`). Verify the code is otherwise unchanged and the tests pass.
- [ ] 3.2 Authors check the physics descriptions and answer the open questions about the models.

## 4. Program-friendly interface

- [ ] 4.1 Batch functions for the off-design models. Verify they equal single-point calls.
- [ ] 4.2 Status values and model bounds readable by code. Verify against the out-of-range reference cases.
- [ ] 4.3 Find the bundled model files without passing a path.

## 5. Portable model files

- [ ] 5.1 Export the neural networks as arrays and the boosted model as ONNX, with a loader. Verify equal results at every reference point.

## 6. Detailed model

- [ ] 6.1 Agree where the detailed model lives and its licence.
- [ ] 6.2 Apply items 1 to 3 to it (reference-output tests, an installable package with agreed pins including CoolProp, docstrings), with CI on Linux and Windows.
- [ ] 6.3 A batch interface with the same inputs, outputs and status values as the surrogate's (item 4), so TANGO can use either.

## 7. Release

- [ ] 7.1 Tag a release (for example 1.1.0) once the agreed items are merged, so TANGO can depend on it.
