# Spec Delta

## Purpose

What RevORCHP-SM provides so that a program such as TANGO can install it, call it at many operating points, and check its results, without changing any model output.

## ADDED Requirements

### Requirement: Outputs are protected by tests

The repository SHALL contain tests that compare the outputs of every README example and of direct calls to each model with stored reference outputs, and CI SHALL run them on Linux and Windows for every pull request.

#### Scenario: A change alters an output

- **WHEN** a change makes any model return a different value at a stored reference point
- **THEN** the tests fail, naming the case

### Requirement: Installable package

The code SHALL be installable with `pip install` as the package `revorchp_sm`, including the model files, and importable from any directory. Every README command SHALL keep working as written.

#### Scenario: Installed from GitHub

- **WHEN** a user runs `pip install git+https://github.com/trx09009/RevORCHP-SM-1.0.2` in a fresh environment and calls the ORC off-design model from another directory
- **THEN** the call returns the same result as the README example

### Requirement: Dependency ranges

Only scikit-learn SHALL be pinned to an exact version while the model files need it. Other dependencies SHALL have version ranges, tested at their oldest allowed and newest versions.

#### Scenario: Installed next to another package

- **WHEN** the package is installed in an environment with pandas 2.2 or pandas 3
- **THEN** it installs, and its tests pass

### Requirement: Documented functions

Every public function SHALL document its inputs with units, its return values, the errors it raises and its validity ranges.

#### Scenario: Calling a model from code

- **WHEN** a developer reads a function's documentation
- **THEN** they can call it correctly without reading its code

### Requirement: Batch evaluation

Each off-design model SHALL offer a function that evaluates many operating points in one call and returns one result per point, equal to calling the existing function once per point.

#### Scenario: One representative day

- **WHEN** 24 hourly operating points are evaluated in one call
- **THEN** the results equal 24 single-point calls

### Requirement: Status and bounds as values

Each result SHALL carry its status as values (for example whether the point is in the supported regime, and whether it was extrapolated), next to the existing warning texts. Each model SHALL expose its validity bounds to code.

#### Scenario: Heat pump outside its supported regime

- **WHEN** the heat-pump model is called outside its supported regime
- **THEN** the result says so as a value, not only as a warning sentence

### Requirement: Portable model files

The models SHALL also be provided in files that do not depend on one scikit-learn version, giving the same results as the `.joblib` files at the reference points.

#### Scenario: A newer scikit-learn

- **WHEN** the portable files are loaded with a newer scikit-learn, or without scikit-learn
- **THEN** the results equal the `.joblib` results at the reference points
