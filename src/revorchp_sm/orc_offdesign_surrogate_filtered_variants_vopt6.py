#!/usr/bin/env python3
"""Off-design surrogate of the fixed one- and two-stage nominal ORC hardware.

Main entry points:

- :func:`predict_orc_performance_for_design` chooses the one-stage or the
  two-stage bundle from the **design** expander stage count (given directly or
  from the design pressure ratio; ratio > 8 selects two stages) and calls
- :func:`predict_1mw_orc_performance`, which evaluates one bundle.

Fixed geometry is preserved in off-design operation. Net electrical power is
signed and includes generator efficiency, pump, air-condenser fan and other
auxiliaries. Heat rejection includes pump shaft work. Brine source and
electrical balances are reconciled explicitly (README.md, "ORC off-design
command").

Training domain (README.md):

- one stage: ``T_amb_C`` -11.9–34.8 (the one-stage bundle's
  ``input_bounds_filtered`` stores -15.2–34.8), ``T_brine_in_C`` 95–115,
  ``p_brine_bar`` 12–24, ``T_brine_out_min_C`` 60–75,
  ``m_brine_max_kg_s`` 3–8 (operation at the nominal 10 kg/s is outside the
  validated flow domain);
- two stages: ``T_amb_C`` -15.2–34.8, same brine ranges, ``m_brine_max_kg_s``
  3–10.

Inputs outside the bundle's ``input_bounds_filtered`` are reported in
``warnings``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from .bundle_io import load_bundle, predict_model


SMALL = 1.0e-12
CORE_OUTPUTS = [
    "orc_Q_in_W", "orc_Q_rej_W", "orc_P_gross_W", "orc_P_exp_el_W",
    "orc_P_pump_el_W", "orc_P_acc_fan_W", "orc_P_other_aux_W", "orc_P_net_W",
    "orc_m_brine_kg_s", "orc_T_brine_out_C", "orc_Q_geo_avail_W",
    "orc_Q_geo_used_W", "orc_eta_gross", "orc_eta_net",
]


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the derived input features used by the ORC models.

    Parameters
    ----------
    df : pandas.DataFrame
        Columns ``T_amb_C``, ``T_brine_in_C``, ``p_brine_bar``,
        ``T_brine_out_min_C`` and ``m_brine_max_kg_s``.

    Returns
    -------
    pandas.DataFrame
        Copy of ``df`` with ``source_deltaT_available_K`` (K),
        ``brine_minus_ambient_K`` (K), ``brine_out_min_minus_ambient_K`` (K),
        ``geo_capacity_proxy_W`` (= ``m_brine_max_kg_s *
        source_deltaT_available_K``, in kg K/s; multiplied by a heat capacity it
        gives W, as in ``Q_geo_avail_est_W``),
        ``Q_geo_avail_est_W`` (= 4180 * ``geo_capacity_proxy_W``, W) and
        ``ambient_abs_C`` (absolute value of ``T_amb_C``).
    """
    out = df.copy()

    out["source_deltaT_available_K"] = (
        out["T_brine_in_C"].astype(float) - out["T_brine_out_min_C"].astype(float)
    )
    out["brine_minus_ambient_K"] = (
        out["T_brine_in_C"].astype(float) - out["T_amb_C"].astype(float)
    )
    out["brine_out_min_minus_ambient_K"] = (
        out["T_brine_out_min_C"].astype(float) - out["T_amb_C"].astype(float)
    )
    out["geo_capacity_proxy_W"] = (
        out["m_brine_max_kg_s"].astype(float) * out["source_deltaT_available_K"].astype(float)
    )
    out["Q_geo_avail_est_W"] = 4180.0 * out["geo_capacity_proxy_W"]
    out["ambient_abs_C"] = out["T_amb_C"].astype(float).abs()

    return out


def inverse_y(pred: np.ndarray, spec: dict[str, Any]) -> pd.DataFrame:
    """Undo the ``log1p`` target transform of a legacy variant model.

    Parameters
    ----------
    pred : numpy.ndarray
        Model output, one column per ``spec["target_columns"]``.
    spec : dict
        ``"target_columns"`` and optional ``"log_columns"``; ``exp(y) - 1`` is
        applied to the log columns.

    Returns
    -------
    pandas.DataFrame
        Predictions with the target column names.
    """
    cols = spec["target_columns"]
    arr = np.asarray(pred, dtype=float).copy()
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    log_set = set(spec.get("log_columns", []))

    for j, c in enumerate(cols):
        if c in log_set:
            arr[:, j] = np.expm1(arr[:, j])

    return pd.DataFrame(arr, columns=cols)


def outputs_from_targets(targets: pd.DataFrame, variant: str) -> pd.DataFrame:
    """Turn legacy variant targets into the standard ORC output columns.

    Used only for bundles without ``schema_version ==
    "corrected_orc_surrogate_v1"``; the bundles shipped in ``models/`` do not
    use it.

    Parameters
    ----------
    targets : pandas.DataFrame
        Predicted targets (output of :func:`inverse_y`).
    variant : str
        ``"direct_power"``, ``"log_power"``, ``"efficiency_direct"`` or
        ``"specific_power"``.

    Returns
    -------
    pandas.DataFrame
        The ``CORE_OUTPUTS`` columns. Heat, power and flow columns are clipped
        at zero; ``orc_eta_gross`` and ``orc_eta_net`` are recomputed from
        ``orc_Q_in_W``.

    Raises
    ------
    ValueError
        If ``variant`` is not one of the names above.
    """
    out = pd.DataFrame(index=targets.index)

    def col(name: str, default: float = np.nan) -> np.ndarray:
        if name in targets.columns:
            return targets[name].astype(float).to_numpy()
        return np.full(len(targets), default, dtype=float)

    if variant in {"direct_power", "log_power"}:
        for c in [
            "orc_Q_in_W",
            "orc_Q_rej_W",
            "orc_P_gross_W",
            "orc_P_exp_el_W",
            "orc_P_pump_el_W",
            "orc_P_acc_fan_W",
            "orc_P_other_aux_W",
            "orc_P_net_W",
            "orc_m_brine_kg_s",
            "orc_T_brine_out_C",
            "orc_Q_geo_avail_W",
            "orc_Q_geo_used_W",
        ]:
            out[c] = col(c)

    elif variant == "efficiency_direct":
        out["orc_Q_in_W"] = col("orc_Q_in_W")
        out["orc_Q_rej_W"] = col("orc_Q_rej_W")
        eta_gross = np.clip(col("orc_eta_gross"), 0.0, 0.5)
        eta_net = np.clip(col("orc_eta_net"), -0.1, 0.5)
        out["orc_P_gross_W"] = out["orc_Q_in_W"] * eta_gross
        out["orc_P_net_W"] = out["orc_Q_in_W"] * eta_net
        out["orc_P_exp_el_W"] = col("orc_P_exp_el_W")
        out["orc_P_pump_el_W"] = col("orc_P_pump_el_W")
        out["orc_P_acc_fan_W"] = col("orc_P_acc_fan_W")
        out["orc_P_other_aux_W"] = col("orc_P_other_aux_W")
        out["orc_m_brine_kg_s"] = col("orc_m_brine_kg_s")
        out["orc_T_brine_out_C"] = col("orc_T_brine_out_C")
        out["orc_Q_geo_avail_W"] = col("orc_Q_geo_avail_W")
        out["orc_Q_geo_used_W"] = col("orc_Q_geo_used_W")

    elif variant == "specific_power":
        q = np.clip(col("orc_Q_in_W"), SMALL, None)
        out["orc_Q_in_W"] = q
        out["orc_Q_rej_W"] = q * col("orc_rej_per_Q")
        out["orc_P_gross_W"] = q * np.clip(col("orc_gross_per_Q"), 0.0, 0.5)
        out["orc_P_net_W"] = q * np.clip(col("orc_net_per_Q"), -0.1, 0.5)
        out["orc_P_exp_el_W"] = out["orc_P_gross_W"]
        out["orc_P_pump_el_W"] = q * col("orc_pump_per_Q")
        out["orc_P_acc_fan_W"] = q * col("orc_fan_per_Q")
        out["orc_P_other_aux_W"] = q * col("orc_aux_per_Q")
        out["orc_m_brine_kg_s"] = col("orc_m_brine_kg_s")
        out["orc_T_brine_out_C"] = col("orc_T_brine_out_C")
        out["orc_Q_geo_avail_W"] = col("orc_Q_geo_avail_W")
        out["orc_Q_geo_used_W"] = col("orc_Q_geo_used_W")

    else:
        raise ValueError(f"Unknown variant: {variant}")

    for c in [
        "orc_Q_in_W",
        "orc_Q_rej_W",
        "orc_P_gross_W",
        "orc_P_exp_el_W",
        "orc_P_pump_el_W",
        "orc_P_acc_fan_W",
        "orc_P_other_aux_W",
        "orc_P_net_W",
        "orc_m_brine_kg_s",
        "orc_Q_geo_avail_W",
        "orc_Q_geo_used_W",
    ]:
        out[c] = np.clip(out[c].astype(float), 0.0, None)

    q = np.clip(out["orc_Q_in_W"].astype(float).to_numpy(), SMALL, None)
    out["orc_eta_gross"] = out["orc_P_gross_W"].astype(float).to_numpy() / q
    out["orc_eta_net"] = out["orc_P_net_W"].astype(float).to_numpy() / q

    return out


def domain_warnings(row: pd.Series, bounds: dict[str, Any]) -> list[str]:
    """List inputs outside the filtered training range.

    Parameters
    ----------
    row : pandas.Series
        Model inputs, indexed by column name.
    bounds : dict
        ``{column: {"min": value, "max": value}}`` (bundle
        ``input_bounds_filtered``). Columns not in ``row`` are skipped.

    Returns
    -------
    list of str
        One message per value below its ``min`` or above its ``max``; empty
        if all are inside.
    """
    warnings: list[str] = []

    for col, lim in (bounds or {}).items():
        if col not in row.index:
            continue

        val = float(row[col])
        lo = lim.get("min", None)
        hi = lim.get("max", None)

        if lo is not None and val < float(lo):
            warnings.append(f"{col}={val:.6g} below filtered training range min {float(lo):.6g}")
        if hi is not None and val > float(hi):
            warnings.append(f"{col}={val:.6g} above filtered training range max {float(hi):.6g}")

    return warnings


def enforce_operating_constraints(outputs: pd.DataFrame, inputs: pd.DataFrame,
                                  cp_J_kgK: float = 4200.0,
                                  generator_efficiency: float = 0.97,
                                  pump_motor_efficiency: float = 0.95,
                                  active: np.ndarray | None = None) -> pd.DataFrame:
    """Reconcile compact predictions with the source and electrical balances.

    The reference pump relation is shaft=electrical*motor_efficiency.
    Thus Qrej=Qin+pump_shaft-Pgross. Net power retains its physical sign.

    Steps, as coded:

    - ``orc_Q_in_W`` is clipped to the source capacity ``cp_J_kgK *
      m_brine_max_kg_s * (T_brine_in_C - T_brine_out_min_C)``; rejected heat,
      gross power, pump/fan/auxiliary power and ``orc_m_dot_orc_kg_s`` are
      scaled by the same factor.
    - ``orc_P_gross_W`` is limited to ``orc_Q_in_W + pump shaft power``.
    - ``orc_P_exp_el_W = generator_efficiency * orc_P_gross_W`` and
      ``orc_P_net_W = orc_P_exp_el_W - orc_P_pump_el_W - orc_P_acc_fan_W -
      orc_P_other_aux_W``.
    - The brine flow is clipped between the flow needed for ``orc_Q_in_W``
      and ``m_brine_max_kg_s``; the brine outlet temperature follows from a
      constant-cp balance.
    - Rows where ``active`` is false get zero heat, power, flow, efficiencies
      and speeds (``orc_Q_geo_avail_W`` is kept) and a brine outlet
      temperature equal to the inlet.

    Parameters
    ----------
    outputs : pandas.DataFrame
        Raw model predictions (columns as in the bundle's ``target_columns``).
    inputs : pandas.DataFrame
        Model inputs with ``T_brine_in_C`` (°C), ``T_brine_out_min_C`` (°C)
        and ``m_brine_max_kg_s`` (kg/s), one row per row of ``outputs``.
    cp_J_kgK : float, optional
        Constant brine specific heat, J/(kg K). Default 4200.
    generator_efficiency : float, optional
        Default 0.97.
    pump_motor_efficiency : float, optional
        Pump shaft power / pump electrical power. Default 0.95.
    active : numpy.ndarray of bool, optional
        Rows predicted to operate. If None, all rows are treated as active.

    Returns
    -------
    pandas.DataFrame
        Copy of ``outputs`` with the reconciled values, plus
        ``orc_P_pump_shaft_W``, ``orc_Q_geo_avail_W``, ``orc_Q_geo_used_W``
        and the residuals ``orc_source_balance_residual_W``,
        ``orc_electrical_balance_residual_W`` and
        ``orc_thermal_balance_residual_W`` (W, round-off around zero).
    """
    out = outputs.copy()
    tin = inputs['T_brine_in_C'].to_numpy(dtype=float)
    tmin = inputs['T_brine_out_min_C'].to_numpy(dtype=float)
    mmax = inputs['m_brine_max_kg_s'].to_numpy(dtype=float)
    delta = np.maximum(tin-tmin, 0.0)
    avail = cp_J_kgK*mmax*delta
    qraw = out['orc_Q_in_W'].to_numpy(dtype=float)
    q = np.clip(qraw, 0.0, avail)
    factor = np.divide(q,qraw,out=np.ones_like(q),where=qraw>0)
    for key in ['orc_Q_rej_W','orc_P_gross_W','orc_P_pump_el_W',
                'orc_P_acc_fan_W','orc_P_other_aux_W']:
        out[key] = np.maximum(out[key].to_numpy(dtype=float),0.0)*factor
    if 'orc_m_dot_orc_kg_s' in out:
        out['orc_m_dot_orc_kg_s'] = np.maximum(out['orc_m_dot_orc_kg_s'],0.0)*factor
    out['orc_Q_in_W'] = q
    out['orc_Q_geo_used_W'] = q
    out['orc_Q_geo_avail_W'] = avail
    pump_shaft = pump_motor_efficiency*out['orc_P_pump_el_W'].to_numpy(dtype=float)
    out['orc_P_gross_W'] = np.minimum(out['orc_P_gross_W'], q+pump_shaft)
    out['orc_P_exp_el_W'] = generator_efficiency*out['orc_P_gross_W']
    out['orc_P_net_W'] = (out['orc_P_exp_el_W']-out['orc_P_pump_el_W']
                         -out['orc_P_acc_fan_W']-out['orc_P_other_aux_W'])
    out['orc_Q_rej_W'] = q+pump_shaft-out['orc_P_gross_W']
    out['orc_P_pump_shaft_W'] = pump_shaft
    minimum_mass = np.divide(q,cp_J_kgK*delta,out=np.zeros_like(q),where=delta>0)
    mass = np.clip(out['orc_m_brine_kg_s'].to_numpy(dtype=float),minimum_mass,mmax)
    out['orc_m_brine_kg_s'] = mass
    out['orc_T_brine_out_C'] = tin-np.divide(q,cp_J_kgK*mass,out=np.zeros_like(q),where=mass>0)
    out['orc_eta_gross'] = np.divide(out['orc_P_gross_W'],q,out=np.zeros_like(q),where=q>0)
    out['orc_eta_net'] = np.divide(out['orc_P_net_W'],q,out=np.zeros_like(q),where=q>0)
    if active is not None:
        inactive = ~np.asarray(active,dtype=bool)
        extensive = [c for c in out if c.endswith('_W') or c.endswith('_kg_s')]
        for c in extensive:
            if c != 'orc_Q_geo_avail_W':
                out.loc[inactive,c] = 0.
        out.loc[inactive,'orc_T_brine_out_C'] = tin[inactive]
        out.loc[inactive,['orc_eta_gross','orc_eta_net']] = 0.
        for c in out:
            if c.endswith('_n_rpm'):
                out.loc[inactive,c] = 0.
            elif c.endswith('_eta_is') or c.endswith('_eta_vol'):
                out[c] = np.clip(out[c],1e-8,1.)
                out.loc[inactive,c] = 0.
    out['orc_source_balance_residual_W'] = (out['orc_Q_in_W']-cp_J_kgK*out['orc_m_brine_kg_s']
                                           *(tin-out['orc_T_brine_out_C']))
    out['orc_electrical_balance_residual_W'] = (out['orc_P_net_W']-(out['orc_P_exp_el_W']
        -out['orc_P_pump_el_W']-out['orc_P_acc_fan_W']-out['orc_P_other_aux_W']))
    out['orc_thermal_balance_residual_W'] = (out['orc_Q_in_W']+out['orc_P_pump_shaft_W']
                                            -out['orc_P_gross_W']-out['orc_Q_rej_W'])
    return out


def predict_frames(bundle: dict[str, Any], inputs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The same batch inference used by the public single-module wrapper.

    For bundles with ``schema_version == "corrected_orc_surrogate_v1"``
    (all bundles shipped in ``models/``). No input checks are made here; see
    :func:`predict_1mw_orc_performance` for those.

    A row is "active" if the bundle's ``active_classifier`` predicts so, the
    learned machine diagnostics are valid (speeds ``*_n_rpm`` > 0,
    efficiencies ``*_eta_is``/``*_eta_vol`` in (0, 1]),
    ``T_brine_in_C > T_brine_out_min_C`` and the raw ``orc_Q_in_W`` > 0.

    Parameters
    ----------
    bundle : dict
        Corrected ORC bundle.
    inputs : pandas.DataFrame
        One row per case with ``T_amb_C``, ``T_brine_in_C``, ``p_brine_bar``,
        ``T_brine_out_min_C`` and ``m_brine_max_kg_s``.

    Returns
    -------
    raw : pandas.DataFrame
        Model predictions before reconciliation (bundle ``target_columns``).
    deployed : pandas.DataFrame
        Output of :func:`enforce_operating_constraints`, plus
        ``orc_prediction_active`` (bool), ``orc_machine_diagnostics_valid``
        (bool) and ``orc_active_probability``.
    """
    frame = add_derived_features(inputs).reset_index(drop=True)
    X = frame[bundle['input_columns']].to_numpy(float)
    raw = pd.DataFrame(predict_model(bundle['models']['direct_power'],X),
                       columns=bundle['target_columns'])
    classifier = bundle['active_classifier']
    active = np.asarray(classifier.predict(X),dtype=bool)
    diagnostic_valid = np.ones(len(frame),dtype=bool)
    for c in raw:
        if c.endswith('_n_rpm'):
            diagnostic_valid &= np.isfinite(raw[c].to_numpy()) & (raw[c].to_numpy()>0)
        elif c.endswith('_eta_is') or c.endswith('_eta_vol'):
            diagnostic_valid &= np.isfinite(raw[c].to_numpy()) & (raw[c].to_numpy()>0) & (raw[c].to_numpy()<=1)
    active &= diagnostic_valid
    active &= (frame.T_brine_in_C.to_numpy()>frame.T_brine_out_min_C.to_numpy()) & (raw.orc_Q_in_W.to_numpy()>0)
    classes = list(classifier.classes_)
    probability = (classifier.predict_proba(X)[:,classes.index(1)] if 1 in classes
                   else np.zeros(len(frame)))
    deployed = enforce_operating_constraints(raw,frame,
        cp_J_kgK=float(bundle.get('source_cp_J_kgK',4200.)),
        generator_efficiency=float(bundle.get('generator_efficiency',.97)),
        pump_motor_efficiency=float(bundle.get('pump_motor_efficiency',.95)),active=active)
    deployed['orc_prediction_active'] = active
    deployed['orc_machine_diagnostics_valid'] = diagnostic_valid
    deployed['orc_active_probability'] = probability
    return raw,deployed


def resolve_design_expander_stage_count(
    *,
    design_expander_stage_count: int | None = None,
    design_p_evap_bar: float | None = None,
    design_p_cond_bar: float | None = None,
    dual_stage_pressure_ratio_threshold: float = 8.0,
) -> dict[str, Any]:
    """Decide whether the design has a one-stage or a two-stage expander.

    Topology is set by the **design** pressure ratio, not the current
    operating ratio (README.md). Priority: an explicit
    ``design_expander_stage_count``; else the design pressure ratio
    ``design_p_evap_bar / design_p_cond_bar`` (> threshold selects two
    stages); else one stage (legacy default). If both are given they must
    agree.

    Parameters
    ----------
    design_expander_stage_count : int, optional
        1 or 2.
    design_p_evap_bar, design_p_cond_bar : float, optional
        Design evaporation and condensation pressures, absolute bar. Give
        both or neither; both must be positive.
    dual_stage_pressure_ratio_threshold : float, optional
        Pressure ratio above which two stages are selected. Must be finite and
        > 1. Default 8.

    Returns
    -------
    dict
        ``"stage_count"`` (1 or 2), ``"selection_basis"``
        (``"explicit_design_stage_count"``, ``"design_pressure_ratio"`` or
        ``"legacy_single_stage_default"``), ``"design_pressure_ratio"``
        (float or None) and ``"dual_stage_pressure_ratio_threshold"``.

    Raises
    ------
    ValueError
        If the threshold is not finite or not > 1, only one design pressure
        is given, a design pressure is not finite or not positive, the stage
        count is not 1 or 2, or the explicit stage count conflicts with the
        pressure-ratio rule.
    """
    if not np.isfinite(float(dual_stage_pressure_ratio_threshold)):
        raise ValueError("dual_stage_pressure_ratio_threshold must be finite")
    threshold = float(dual_stage_pressure_ratio_threshold)
    if threshold <= 1.0:
        raise ValueError("dual_stage_pressure_ratio_threshold must be greater than 1")

    pressure_values_provided = (
        design_p_evap_bar is not None or design_p_cond_bar is not None
    )
    design_pressure_ratio: float | None = None
    stage_from_pressure: int | None = None

    if pressure_values_provided:
        if design_p_evap_bar is None or design_p_cond_bar is None:
            raise ValueError(
                "Provide both design_p_evap_bar and design_p_cond_bar, or neither."
            )
        p_evap = float(design_p_evap_bar)
        p_cond = float(design_p_cond_bar)
        if not np.isfinite(p_evap) or not np.isfinite(p_cond) or p_evap <= 0.0 or p_cond <= 0.0:
            raise ValueError("Design evaporation and condensation pressures must be positive")
        design_pressure_ratio = p_evap / p_cond
        stage_from_pressure = 2 if design_pressure_ratio > threshold else 1

    if design_expander_stage_count is not None:
        stage_count = int(design_expander_stage_count)
        if stage_count not in (1, 2) or stage_count != design_expander_stage_count:
            raise ValueError("design_expander_stage_count must be 1 or 2")
        if stage_from_pressure is not None and stage_count != stage_from_pressure:
            raise ValueError(
                "Explicit design expander stage count conflicts with the pressure-ratio "
                f"rule: stage_count={stage_count}, design_pressure_ratio="
                f"{design_pressure_ratio:.6g}, threshold={threshold:.6g}."
            )
        basis = "explicit_design_stage_count"
    elif stage_from_pressure is not None:
        stage_count = stage_from_pressure
        basis = "design_pressure_ratio"
    else:
        stage_count = 1
        basis = "legacy_single_stage_default"

    return {
        "stage_count": stage_count,
        "selection_basis": basis,
        "design_pressure_ratio": design_pressure_ratio,
        "dual_stage_pressure_ratio_threshold": threshold,
    }


def predict_1mw_orc_performance(
    *,
    model_path: str | Path,
    T_amb_C: float,
    T_brine_in_C: float,
    p_brine_bar: float,
    T_brine_out_min_C: float,
    m_brine_max_kg_s: float,
    variant: str | None = None,
) -> dict[str, Any]:
    """Predict the off-design performance of one ORC bundle.

    Most callers should use :func:`predict_orc_performance_for_design`,
    which picks the right bundle for the design expander stage count.

    Parameters
    ----------
    model_path : str or pathlib.Path
        Path to an ORC bundle, e.g.
        ``models/orc_synthetic_year_surrogate_filtered_variants_vopt6.joblib``
        (one stage) or ``..._two_stage.joblib`` (two stages).
    T_amb_C : float
        Ambient temperature, °C.
    T_brine_in_C : float
        Brine inlet temperature, °C (training domain 95–115).
    p_brine_bar : float
        Brine pressure, absolute bar (training domain 12–24). Must be > 0.
    T_brine_out_min_C : float
        Minimum brine outlet temperature, °C (training domain 60–75). Must
        not exceed ``T_brine_in_C``.
    m_brine_max_kg_s : float
        Available brine flow, kg/s (training domain 3–8 one stage, 3–10 two
        stages). Must be > 0.
    variant : str, optional
        For the shipped (corrected) bundles only ``None`` or
        ``"direct_power"`` is accepted.

    Returns
    -------
    dict
        - ``"selected_variant"``: ``"direct_power"`` for the shipped bundles.
        - ``"orc"`` (and the same dict under ``"selected"``): main keys
          ``orc_Q_in_W``, ``orc_Q_rej_W``, ``orc_P_gross_W``,
          ``orc_P_exp_el_W``, ``orc_P_pump_el_W``, ``orc_P_pump_shaft_W``,
          ``orc_P_acc_fan_W``, ``orc_P_other_aux_W``, ``orc_P_net_W``
          (signed), ``orc_m_brine_kg_s``, ``orc_T_brine_out_C``,
          ``orc_Q_geo_avail_W``, ``orc_Q_geo_used_W``, ``orc_eta_gross``,
          ``orc_eta_net``, ``orc_m_dot_orc_kg_s``, per-stage machine
          diagnostics ``orc_stage<N>_n_rpm``/``_eta_is``/``_eta_vol``,
          ``orc_prediction_active`` (bool) and ``*_residual_W`` balance
          residuals. Units as in the key suffix.
        - ``"all_variants"``: ``{variant name: dict like "orc"}``.
        - ``"derived_inputs"``: inputs including derived features.
        - ``"model_metadata"``: model path, ``expander_stage_count`` and
          (for corrected bundles) release version, reference hashes and
          ``nominal_design``.
        - ``"warnings"``: list of str.

    Raises
    ------
    ValueError
        If an input is not finite, ``p_brine_bar <= 0``,
        ``m_brine_max_kg_s <= 0`` or ``T_brine_in_C < T_brine_out_min_C``.
    KeyError
        If ``variant`` is not available in the bundle.
    FileNotFoundError
        If ``model_path`` does not exist.

    Notes
    -----
    The second half of this function handles older bundles without
    ``schema_version == "corrected_orc_surrogate_v1"``; the shipped bundles
    return from the first branch.
    """
    resolved_model_path = Path(model_path).resolve()
    values = [T_amb_C, T_brine_in_C, p_brine_bar, T_brine_out_min_C, m_brine_max_kg_s]
    if not np.isfinite(values).all():
        raise ValueError("ORC inputs must be finite")
    if p_brine_bar <= 0 or m_brine_max_kg_s <= 0 or T_brine_in_C < T_brine_out_min_C:
        raise ValueError("Invalid ORC pressure, flow or temperature boundary")
    bundle = load_bundle(resolved_model_path)

    if bundle.get('schema_version') == 'corrected_orc_surrogate_v1':
        if variant not in (None,'direct_power'):
            raise KeyError("Only direct_power is available in this corrected release")
        inputs = pd.DataFrame([dict(T_amb_C=float(T_amb_C),T_brine_in_C=float(T_brine_in_C),
            p_brine_bar=float(p_brine_bar),T_brine_out_min_C=float(T_brine_out_min_C),
            m_brine_max_kg_s=float(m_brine_max_kg_s))])
        raw,deployed = predict_frames(bundle,inputs)
        selected = {k:(bool(v) if isinstance(v,(bool,np.bool_)) else float(v))
                    for k,v in deployed.iloc[0].to_dict().items()}
        messages = domain_warnings(inputs.iloc[0],bundle.get('input_bounds_filtered',{}))
        if not selected['orc_prediction_active']:
            messages.append('Surrogate feasibility or physical checks indicate inactive ORC operation.')
        for c in raw:
            if (c.endswith('_eta_is') or c.endswith('_eta_vol')) and not (0<float(raw.iloc[0][c])<=1):
                messages.append(f'{c} raw estimate is outside its physical interval; the reported diagnostic is bounded.')
        if abs(float(raw.iloc[0]['orc_Q_rej_W'])-selected['orc_Q_rej_W'])>1.:
            messages.append('Rejected heat reconstructed from heat input, pump shaft work and gross expander work.')
        return dict(selected_variant='direct_power',orc=selected,selected=selected,
            all_variants={'direct_power':selected},
            derived_inputs={k:float(v)for k,v in add_derived_features(inputs).iloc[0].items()},
            model_metadata=dict(model_path=str(resolved_model_path),release_version=bundle['release_version'],
                expander_stage_count=int(bundle['expander_stage_count']),
                reference_source_sha256=bundle['reference_source_sha256'],
                reference_hardware_sha256=bundle['reference_hardware_sha256'],
                nominal_design=bundle['nominal_design'],
                physical_projection='source_electric_and_cycle_thermal_balance_v2'),warnings=messages)

    selected_variant = variant or bundle.get("selected_variant", "direct_power")
    if selected_variant not in bundle["models"]:
        raise KeyError(
            f"Variant {selected_variant!r} is not available. "
            f"Available variants: {list(bundle['models'].keys())}"
        )

    row = pd.DataFrame(
        [
            {
                "T_amb_C": float(T_amb_C),
                "T_brine_in_C": float(T_brine_in_C),
                "p_brine_bar": float(p_brine_bar),
                "T_brine_out_min_C": float(T_brine_out_min_C),
                "m_brine_max_kg_s": float(m_brine_max_kg_s),
            }
        ]
    )

    row = add_derived_features(row)

    input_columns = bundle["input_columns"]
    X = row[input_columns].astype(float).to_numpy()

    def predict_variant(name: str) -> dict[str, float]:
        """Predict one legacy variant and return its outputs as floats."""
        model = bundle["models"][name]
        spec = bundle["variant_specs"][name]
        pred_targets = inverse_y(predict_model(model,X), spec)
        out = outputs_from_targets(pred_targets, name)
        if bundle.get('physical_projection') == 'source_and_electric_v1':
            out = enforce_operating_constraints(
                out, row, cp_J_kgK=float(bundle.get('source_cp_J_kgK',4200.0)),
                generator_efficiency=float(bundle.get('generator_efficiency',0.97)))
        return {k: float(v) for k, v in out.iloc[0].to_dict().items()}

    selected = predict_variant(selected_variant)
    all_variants = {name: (selected if name == selected_variant else predict_variant(name))
                    for name in bundle["models"].keys()}

    return {
        "selected_variant": selected_variant,
        "model_metadata": {
            "model_path": str(resolved_model_path),
            "expander_stage_count": (
                None
                if bundle.get("expander_stage_count") is None
                else int(bundle["expander_stage_count"])
            ),
            "physical_projection": bundle.get('physical_projection'),
        },
        "orc": selected,
        "selected": selected,
        "all_variants": all_variants,
        "derived_inputs": {
            k: float(row.iloc[0][k])
            for k in input_columns
            if k in row.columns
        },
        "warnings": domain_warnings(row.iloc[0], bundle.get("input_bounds_filtered", {})),
    }


def predict_orc_performance_for_design(
    *,
    one_stage_model_path: str | Path,
    two_stage_model_path: str | Path | None,
    T_amb_C: float,
    T_brine_in_C: float,
    p_brine_bar: float,
    T_brine_out_min_C: float,
    m_brine_max_kg_s: float,
    design_expander_stage_count: int | None = None,
    design_p_evap_bar: float | None = None,
    design_p_cond_bar: float | None = None,
    dual_stage_pressure_ratio_threshold: float = 8.0,
    variant: str | None = None,
) -> dict[str, Any]:
    """Predict ORC off-design performance with the bundle matching the design.

    The design expander stage count is resolved with
    :func:`resolve_design_expander_stage_count`; the one-stage or two-stage
    bundle is then evaluated with :func:`predict_1mw_orc_performance`.

    Parameters
    ----------
    one_stage_model_path : str or pathlib.Path
        Path to the one-stage ORC bundle.
    two_stage_model_path : str or pathlib.Path or None
        Path to the two-stage ORC bundle. Required if the design has two
        stages.
    T_amb_C, T_brine_in_C, p_brine_bar, T_brine_out_min_C, m_brine_max_kg_s : float
        Operating conditions, as in :func:`predict_1mw_orc_performance`
        (°C, °C, absolute bar, °C, kg/s).
    design_expander_stage_count : int, optional
        1 or 2.
    design_p_evap_bar, design_p_cond_bar : float, optional
        Design evaporation and condensation pressures, absolute bar.
    dual_stage_pressure_ratio_threshold : float, optional
        Default 8.
    variant : str, optional
        Passed to :func:`predict_1mw_orc_performance`.

    Returns
    -------
    dict
        The result of :func:`predict_1mw_orc_performance`, plus
        ``"expander_selection"``: the result of
        :func:`resolve_design_expander_stage_count` with
        ``"selected_model_path"`` and ``"model_metadata_stage_count"``.

    Raises
    ------
    ValueError
        As raised by :func:`resolve_design_expander_stage_count` and
        :func:`predict_1mw_orc_performance`; also if two stages are required
        but ``two_stage_model_path`` is None, or if the selected bundle
        declares a different stage count.
    KeyError
        If ``variant`` is not available in the bundle.
    """
    selection = resolve_design_expander_stage_count(
        design_expander_stage_count=design_expander_stage_count,
        design_p_evap_bar=design_p_evap_bar,
        design_p_cond_bar=design_p_cond_bar,
        dual_stage_pressure_ratio_threshold=dual_stage_pressure_ratio_threshold,
    )

    if selection["stage_count"] == 2:
        if two_stage_model_path is None:
            raise ValueError(
                "This design requires the two-stage ORC surrogate, but "
                "two_stage_model_path was not provided."
            )
        selected_model_path = Path(two_stage_model_path)
    else:
        selected_model_path = Path(one_stage_model_path)

    result = predict_1mw_orc_performance(
        model_path=selected_model_path,
        T_amb_C=T_amb_C,
        T_brine_in_C=T_brine_in_C,
        p_brine_bar=p_brine_bar,
        T_brine_out_min_C=T_brine_out_min_C,
        m_brine_max_kg_s=m_brine_max_kg_s,
        variant=variant,
    )
    bundle_stage_count = result["model_metadata"]["expander_stage_count"]
    if bundle_stage_count is not None and int(bundle_stage_count) != int(selection["stage_count"]):
        raise ValueError(
            f"Selected model declares {bundle_stage_count} expander stage(s), but "
            f"the design requires {selection['stage_count']}."
        )
    result["expander_selection"] = {
        **selection,
        "selected_model_path": result["model_metadata"]["model_path"],
        "model_metadata_stage_count": (
            None if bundle_stage_count is None else int(bundle_stage_count)
        ),
    }
    if bundle_stage_count is None and selection["stage_count"] == 1:
        result["warnings"].append(
            "The selected legacy one-stage model has no stage metadata; it is "
            "accepted as the existing single-stage surrogate."
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    """Command-line options of this module (see ``--help``)."""
    p = argparse.ArgumentParser(description="Predict with filtered ORC off-design surrogate.")

    p.add_argument("--model", required=True)
    p.add_argument("--two-stage-model", default=None)
    p.add_argument("--T-amb-c", type=float, required=True)
    p.add_argument("--T-brine-in-c", type=float, required=True)
    p.add_argument("--p-brine-bar", type=float, required=True)
    p.add_argument("--T-brine-out-min-c", type=float, required=True)
    p.add_argument("--m-brine-max-kg-s", type=float, required=True)
    p.add_argument("--variant", default=None)
    p.add_argument("--design-expander-stage-count", type=int, choices=[1, 2], default=None)
    p.add_argument("--design-p-evap-bar", type=float, default=None)
    p.add_argument("--design-p-cond-bar", type=float, default=None)
    p.add_argument("--dual-stage-pressure-ratio-threshold", type=float, default=8.0)

    return p


def main() -> None:
    """Run one ORC off-design prediction from the command line; print JSON."""
    args = build_parser().parse_args()

    result = predict_orc_performance_for_design(
        one_stage_model_path=args.model,
        two_stage_model_path=args.two_stage_model,
        T_amb_C=args.T_amb_c,
        T_brine_in_C=args.T_brine_in_c,
        p_brine_bar=args.p_brine_bar,
        T_brine_out_min_C=args.T_brine_out_min_c,
        m_brine_max_kg_s=args.m_brine_max_kg_s,
        design_expander_stage_count=args.design_expander_stage_count,
        design_p_evap_bar=args.design_p_evap_bar,
        design_p_cond_bar=args.design_p_cond_bar,
        dual_stage_pressure_ratio_threshold=args.dual_stage_pressure_ratio_threshold,
        variant=args.variant,
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
