"""Inputs used by the regression tests, and helpers shared with generate_reference.py.

Each case is a call to one of the public prediction functions. The outputs of
these calls are stored in tests/data/*.json and the tests check that the code
still returns the same numbers.
"""
from __future__ import annotations

import json
import math
import re
import shlex
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = Path(__file__).resolve().parent / "data"
try:
    import revorchp_sm  # noqa: F401  (installed with pip: test the installed copy)
except ImportError:
    sys.path.insert(0, str(REPO / "src"))  # not installed: test the code in src/

from revorchp_sm.design_ann_vopt6_predictor import DesignANNModule, json_default  # noqa: E402
from revorchp_sm.hp_offdesign_surrogate_1mw_filtered_variants_vopt6 import predict_1mw_hp_performance  # noqa: E402
from revorchp_sm.orc_offdesign_surrogate_filtered_variants_vopt6 import (  # noqa: E402
    predict_1mw_orc_performance,
    predict_orc_performance_for_design,
)
from revorchp_sm.reversible_design_ann_vopt6_wrapper import ReversibleDesignANN  # noqa: E402

# Tolerances used to compare numbers with the stored reference values.
RTOL = 1e-9
ATOL = 1e-12
# Balance residuals (keys containing "residual") are floating-point round-off
# around zero, so they only need to stay below one microwatt.
RESIDUAL_ATOL = 1e-6

DESIGN_MODEL = REPO / "models" / "design_ann_vopt6_module.joblib"
HP_MODEL = REPO / "models" / "hp_synthetic_year_surrogate_1mw_filtered_variants_vopt6_hp55_75_out35_45_sourcebalanced.joblib"
ORC_ONE_STAGE_MODEL = REPO / "models" / "orc_synthetic_year_surrogate_filtered_variants_vopt6.joblib"
ORC_TWO_STAGE_MODEL = REPO / "models" / "orc_synthetic_year_surrogate_filtered_variants_vopt6_two_stage.joblib"


# ---------------------------------------------------------------------------
# README command examples
# ---------------------------------------------------------------------------

def readme_commands() -> list[str]:
    """Every `python ...` line inside the code blocks of README.md."""
    text = (REPO / "README.md").read_text(encoding="utf-8")
    commands = []
    for block in re.findall(r"```[a-z]*\n(.*?)```", text, flags=re.S):
        for line in block.splitlines():
            if line.startswith("python "):
                commands.append(line.strip())
    return commands


def run_readme_command(command: str):
    """Run a README command as written, with this Python instead of `python`."""
    args = shlex.split(command)
    args[0] = sys.executable
    done = subprocess.run(args, cwd=REPO, capture_output=True, text=True, check=True)
    return normalise(json.loads(done.stdout))


# ---------------------------------------------------------------------------
# Direct calls to the Python functions
# ---------------------------------------------------------------------------

def _hp(**kwargs):
    return predict_1mw_hp_performance(model_path=HP_MODEL, **kwargs)


def _orc_one(**kwargs):
    return predict_1mw_orc_performance(model_path=ORC_ONE_STAGE_MODEL, **kwargs)


def _orc_two(**kwargs):
    return predict_1mw_orc_performance(model_path=ORC_TWO_STAGE_MODEL, **kwargs)


def _orc_design(**kwargs):
    return predict_orc_performance_for_design(
        one_stage_model_path=ORC_ONE_STAGE_MODEL, two_stage_model_path=ORC_TWO_STAGE_MODEL, **kwargs)


def _design_module(**kwargs):
    return DesignANNModule(DESIGN_MODEL).predict_one(**kwargs)


def _error(function):
    """Run a case that is expected to stop with an error and record the message."""
    def run(**kwargs):
        try:
            function(**kwargs)
        except Exception as error:  # the type and message are part of the reference
            return {"error": type(error).__name__, "message": str(error)}
        raise AssertionError("expected an error, but the call succeeded")
    return run


def _design_hp_priority(**kwargs):
    return ReversibleDesignANN(DESIGN_MODEL).hp_priority(**kwargs)


def _design_orc_priority(**kwargs):
    return ReversibleDesignANN(DESIGN_MODEL).orc_priority(**kwargs)


def _hp_case(T_in, p, T_out_min, m, load_W, lift, sink_dT, **extra):
    return dict(T_brine_in_C=T_in, p_brine_bar=p, T_brine_out_min_C=T_out_min, m_brine_max_kg_s=m,
                heating_load_W=load_W, temperature_lift_K=lift, sink_deltaT_K=sink_dT, **extra)


def _orc_case(T_amb, T_in, p, T_out_min, m, **extra):
    return dict(T_amb_C=T_amb, T_brine_in_C=T_in, p_brine_bar=p, T_brine_out_min_C=T_out_min,
                m_brine_max_kg_s=m, **extra)


def _design_case(load_W, T_amb, T_in, p, T_out_min):
    return dict(target_load_W=load_W, avg_T_amb_C=T_amb, T_brine_in_C=T_in, p_brine_bar=p,
                T_brine_out_min_C=T_out_min)


# Groups of cases: {group: {case name: (function, keyword arguments)}}.
# Each group is stored in tests/data/<group>.json.
FUNCTION_CASES = {
    "hp_offdesign": {
        # In-domain points spread over the declared HP bounds.
        "in_domain_1": (_hp, _hp_case(55, 12, 35, 5, 300e3, 25, 10)),
        "in_domain_2": (_hp, _hp_case(60, 15, 40, 8, 500e3, 30, 15)),
        "in_domain_3": (_hp, _hp_case(65, 18, 45, 10, 800e3, 25, 15, return_all_variants=False)),
        "in_domain_4": (_hp, _hp_case(70, 21, 42, 11, 1e6, 20, 20)),
        "in_domain_5": (_hp, _hp_case(75, 24, 38, 12, 600e3, 25, 25, variant="direct_w")),
        # Out of domain: load below the 20 % fitting regime.
        "low_load_extrapolation": (_hp, _hp_case(65, 18, 45, 10, 100e3, 25, 15)),
        # Out of domain: brine inlet above 75 C and sink supply above 100 C;
        # the pressure state is reported as unsupported.
        "outside_bounds": (_hp, _hp_case(85, 18, 45, 10, 500e3, 20, 15)),
        "pressure_state_unsupported": (_hp, _hp_case(65, 18, 45, 12, 500e3, 45, 15)),
    },
    "orc_offdesign": {
        # One-stage hardware, in domain.
        "one_stage_1": (_orc_one, _orc_case(-11.9, 95, 12, 60, 3)),
        "one_stage_2": (_orc_one, _orc_case(0, 100, 15, 65, 5)),
        "one_stage_3": (_orc_one, _orc_case(9.7, 105, 18, 70, 8)),
        "one_stage_4": (_orc_one, _orc_case(20, 110, 21, 72, 6)),
        "one_stage_5": (_orc_one, _orc_case(34.8, 115, 24, 75, 4)),
        # Two-stage hardware, in domain.
        "two_stage_1": (_orc_two, _orc_case(-15.2, 95, 12, 60, 3)),
        "two_stage_2": (_orc_two, _orc_case(-4, 105, 18, 70, 10)),
        "two_stage_3": (_orc_two, _orc_case(5, 100, 15, 65, 6)),
        "two_stage_4": (_orc_two, _orc_case(20, 110, 21, 72, 8)),
        "two_stage_5": (_orc_two, _orc_case(34.8, 115, 24, 75, 5)),
        # Stage selection through predict_orc_performance_for_design.
        "design_explicit_one_stage": (_orc_design, _orc_case(9.7, 105, 18, 70, 8, design_expander_stage_count=1)),
        "design_explicit_two_stage": (_orc_design, _orc_case(-4, 105, 18, 70, 10, design_expander_stage_count=2)),
        "design_pressure_ratio_one_stage": (_orc_design, _orc_case(9.7, 105, 18, 70, 8, design_p_evap_bar=18.0, design_p_cond_bar=3.0)),
        "design_pressure_ratio_two_stage": (_orc_design, _orc_case(-4, 105, 18, 70, 10, design_p_evap_bar=20.0, design_p_cond_bar=2.0)),
        "design_legacy_default": (_orc_design, _orc_case(9.7, 105, 18, 70, 8)),
        # Out of domain: one-stage hardware at nominal 10 kg/s (outside its 3-8 kg/s data).
        "one_stage_flow_extrapolation": (_orc_one, _orc_case(9.7, 105, 18, 70, 10)),
        # Out of domain: hot ambient and cool brine.
        "two_stage_outside_bounds": (_orc_two, _orc_case(40, 90, 18, 60, 12)),
        # No usable temperature difference: ORC is reported inactive.
        "one_stage_inactive": (_orc_one, _orc_case(9.7, 75, 18, 75, 5)),
        "two_stage_inactive": (_orc_two, _orc_case(-4, 75, 18, 75, 5)),
    },
    "design": {
        # Design ANN for one module, spread over the declared bounds.
        "module_1": (_design_module, _design_case(500e3, -10, 95, 12, 60)),
        "module_2": (_design_module, _design_case(650e3, -4, 115, 18, 65)),
        "module_3": (_design_module, _design_case(800e3, 9.7, 105, 18, 70)),
        "module_4": (_design_module, _design_case(900e3, 15, 110, 21, 72)),
        "module_5": (_design_module, _design_case(1e6, 20, 100, 24, 75)),
        # Out of domain: load, ambient and brine inlet above/below the declared bounds.
        "module_outside_bounds": (_design_module, _design_case(300e3, 25, 117, 18, 70)),
        # Brine inlet too hot for the saturation-pressure fit: the call stops with an error.
        "module_saturation_range_error": (_error(_design_module), _design_case(800e3, 9.7, 120, 18, 70)),
        # HP-priority wrapper.
        "hp_priority_1": (_design_hp_priority, dict(total_heating_load_W=1.8e6, total_brine_flow_kg_s=20,
                                                    T_brine_in_HP_C=65, avg_T_amb_C=9.7, p_brine_bar=18,
                                                    T_brine_out_min_ORC_C=70)),
        "hp_priority_2": (_design_hp_priority, dict(total_heating_load_W=0.7e6, total_brine_flow_kg_s=8,
                                                    T_brine_in_HP_C=60, avg_T_amb_C=-5, p_brine_bar=15,
                                                    T_brine_out_min_ORC_C=65)),
        # Below the per-module minimum: module is sized at 0.5 MW with a warning.
        "hp_priority_low_load": (_design_hp_priority, dict(total_heating_load_W=0.6e6, total_brine_flow_kg_s=30,
                                                           T_brine_in_HP_C=70, avg_T_amb_C=15, p_brine_bar=21,
                                                           T_brine_out_min_ORC_C=72)),
        # ORC-priority inverse search (small grids to keep the tests fast;
        # the 201-point grid of the README is covered by readme_commands.json).
        "orc_priority_1": (_design_orc_priority, dict(total_brine_flow_kg_s=10, T_brine_well_C=110, avg_T_amb_C=-4,
                                                      p_brine_bar=18, T_brine_out_min_ORC_C=65, grid_points=21)),
        "orc_priority_2": (_design_orc_priority, dict(total_brine_flow_kg_s=25, T_brine_well_C=100, avg_T_amb_C=15,
                                                      p_brine_bar=15, T_brine_out_min_ORC_C=70, grid_points=11,
                                                      objective="orc_P_gross_W")),
    },
}


def run_function_case(group: str, name: str):
    function, kwargs = FUNCTION_CASES[group][name]
    return normalise(function(**kwargs))


# ---------------------------------------------------------------------------
# Normalising and comparing results
# ---------------------------------------------------------------------------

def normalise(result):
    """Plain JSON data with model paths written relative to the repository.

    Absolute paths differ between computers, so they are stored as e.g.
    "models/design_ann_vopt6_module.joblib".
    """
    data = json.loads(json.dumps(result, default=json_default))
    return _relative_paths(data)


def _relative_paths(value):
    if isinstance(value, dict):
        return {k: _relative_paths(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_relative_paths(v) for v in value]
    if isinstance(value, str) and value.endswith(".joblib"):
        path = Path(value)
        if path.is_absolute():
            try:
                return path.relative_to(REPO).as_posix()
            except ValueError:
                return value
    return value


def differences(expected, actual, where="result"):
    """List of human-readable differences between two JSON-like values."""
    if isinstance(expected, bool) or isinstance(actual, bool):
        return [] if expected is actual else [f"{where}: expected {expected!r}, got {actual!r}"]
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if math.isnan(expected) and math.isnan(actual):
            return []
        atol = RESIDUAL_ATOL if "residual" in where else ATOL
        if math.isclose(expected, actual, rel_tol=RTOL, abs_tol=atol):
            return []
        return [f"{where}: expected {expected!r}, got {actual!r}"]
    if isinstance(expected, dict) and isinstance(actual, dict):
        problems = []
        missing = [k for k in expected if k not in actual]
        extra = [k for k in actual if k not in expected]
        if missing or extra:
            problems.append(f"{where}: missing keys {missing}, unexpected keys {extra}")
        for key in expected:
            if key in actual:
                problems += differences(expected[key], actual[key], f"{where}[{key!r}]")
        return problems
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return [f"{where}: expected {len(expected)} items, got {len(actual)}"]
        problems = []
        for i, (e, a) in enumerate(zip(expected, actual)):
            problems += differences(e, a, f"{where}[{i}]")
        return problems
    if type(expected) is not type(actual) or expected != actual:
        return [f"{where}: expected {expected!r}, got {actual!r}"]
    return []
