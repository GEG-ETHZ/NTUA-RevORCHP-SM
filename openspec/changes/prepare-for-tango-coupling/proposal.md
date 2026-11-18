# Proposal: prepare RevORCHP-SM for coupling with TANGO

## Why

In FlexGEO, the reversible ORC / heat-pump system of task T2.2 is to become the surface plant of TANGO, the simulator that couples the subsurface, the plant and the economics (task T4.5). The first coupled simulation is due at the end of January 2027 (milestone MS4.2). TANGO will call these surrogate models for every hour of one representative day per month, many times per simulated year, inside optimizations.

The README states the inputs, units and validity ranges of each model, and its example commands run as documented with the pinned versions. A few things make the models hard to call from another program:

- The code is a set of scripts, not a Python package that can be installed as a dependency and imported from anywhere.
- `requirements.txt` pins exact versions. TANGO needs to install the models in its own environment, as one of its own dependencies.
- There are no automated tests, so neither of us can show that a change to the code leaves the model outputs unchanged.
- Each call evaluates one operating point and returns warnings as sentences, which a program cannot act on reliably.
- The `.joblib` files only load reliably with scikit-learn 1.8.0, which could cause trouble because a `.joblib` file stores the trained model as scikit-learn's internal Python objects, and scikit-learn only guarantees that such a file loads with the version that wrote it. Every program that uses the models must then stay on 1.8.0 for everything else in its environment too, and cannot take scikit-learn's bug fixes or support for newer Python versions. When 1.8.0 no longer installs on a current Python, the models stop working, even though the trained numbers in them are still valid. Inside, the models are numbers: weight matrices, scaling factors, tree thresholds and small property tables; storing those in a format that does not depend on one library version keeps them usable for as long as the physics is.

This document proposes what to change, so that we agree on it before any code is merged. For each task it leaves open who does it: you, or the TANGO team at ETH Zurich. The TANGO team has already drafted the first three as pull requests, to make the proposal concrete, but nothing is final until you agree.

## What changes

1. **Regression tests and CI.** Tests that run every README example and direct calls to each model, and compare the results with stored reference outputs. GitHub runs them on Linux and Windows for every pull request. No existing file changes.
2. **An installable package.** The code moves into a package, `revorchp_sm`, installable with `pip install`, with the model files inside it. Every README command keeps working exactly as written. scikit-learn stays pinned to 1.8.0 (the model files need it); the other dependencies get version ranges instead of exact pins. The tests show identical outputs with the oldest and newest allowed versions.
3. **Docstrings.** Every function documents its inputs with units, what it returns, the errors it raises and the validity ranges from the README. No code changes.
4. **A program-friendly interface.** Evaluate many operating points in one call; return status as values (for example `supported: False`) next to the warning sentences; read the validity bounds of each model from code; find the bundled model files without passing a path.
5. **Model files that do not depend on one scikit-learn version.** Export the three neural networks as plain arrays and the boosted model in a portable format (ONNX), next to the `.joblib` files, with a test that both give the same results.

Items 1 to 3 change no model output. Items 4 and 5 add new functions and files; the existing functions keep working.

### Not part of this proposal

- Any change to the physics, the training data or the trained models.
- The detailed thermodynamic model. Whether and how it can be shared is a question for you (see `design.md`).
- How TANGO uses the models. That is described in TANGO's own proposal, `add-revorc-coupling`, in the TANGO repository.

## Capabilities

### New Capabilities

- `tango-interface`: what the package provides so that a program like TANGO can install it, call it and check its results.

## Impact

- New: `tests/`, `.github/workflows/tests.yml`, `pyproject.toml`, `src/revorchp_sm/`, `openspec/`.
- Moved: the five Python modules into `src/revorchp_sm/`; thin scripts with the original names stay at the top level, so your commands do not change.
- Unchanged: `models/`, `requirements.txt` (kept as the exact environment you tested), `README.md` except for an install section, all model outputs.
