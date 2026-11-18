#!/usr/bin/env python3
"""Off-design surrogate of the fixed 1 MW nominal heat-pump hardware.

Main entry point: :func:`predict_1mw_hp_performance`. The bundle describes its
own fixed nominal hardware; a geometry from the design surrogate cannot be
passed in (README.md).

Summary of the calculation (README.md, "HP off-design command"):

- Sink supply temperature = HP brine inlet + ``temperature_lift_K``; sink
  return = supply - ``sink_deltaT_K``; condensation temperature = supply +
  5 K (``sink_approach_K`` in the bundle).
- Learned models predict delivered heat and one of three "variants" for the
  compressor power: ``inverse_cop`` (the selected variant), ``direct_w`` or
  ``pressure_hybrid``.
- Delivered heat is then capped by the requested load, the source heat and
  the fixed condenser capacity, with ``Q_delivered = Q_source + W_comp``. The
  full available brine flow is used. The minimum brine outlet temperature is
  a lower operating constraint.

Training domain (README.md): ``T_brine_in_C`` 55–75, ``p_brine_bar`` 12–24,
``T_brine_out_min_C`` 35–45, ``m_brine_max_kg_s`` 5–12, ``heating_load_W``
50000–1e6, ``T_sink_supply_C`` 75–100, ``sink_deltaT_K`` 10–25. Loads below
0.2 MW are extrapolation. Inputs outside the bundle's
``input_bounds_filtered`` are reported in ``warnings``.
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


def json_default(obj: Any) -> Any:
    """Convert NumPy scalars/arrays and paths so ``json.dumps`` can write them.

    NumPy integers and floats become ``int``/``float``, arrays become lists,
    and anything else becomes ``str(obj)``.
    """
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def undo_log(y: np.ndarray) -> np.ndarray:
    """Inverse of the ``log1p`` target transform: ``exp(y) - 1``."""
    return np.expm1(np.asarray(y, dtype=float))


def eval_eta_poly(poly: dict[str, Any], pr: np.ndarray) -> np.ndarray:
    """Evaluate a polynomial efficiency fit of pressure ratio, then clip it.

    Only used by the ``pressure_hybrid`` variant for bundles that do not
    predict ``hp_eta_is`` directly (see :func:`compute_variant`).

    Parameters
    ----------
    poly : dict
        ``"coeffs_desc"`` (polynomial coefficients, highest power first),
        ``"eta_clip_min"`` and ``"eta_clip_max"``.
    pr : numpy.ndarray
        Pressure ratio ``p_cond / p_evap`` (dimensionless).

    Returns
    -------
    numpy.ndarray
        Efficiency (dimensionless), clipped to the given limits.
    """
    eta = np.polyval(np.asarray(poly["coeffs_desc"], dtype=float), np.asarray(pr, dtype=float))
    return np.clip(eta, float(poly["eta_clip_min"]), float(poly["eta_clip_max"]))


def build_feature_frame(
    *,
    T_brine_in_C: float,
    p_brine_bar: float,
    T_brine_out_min_C: float,
    m_brine_max_kg_s: float,
    heating_load_W: float,
    temperature_lift_K: float,
    sink_deltaT_K: float,
    sink_approach_K: float,
    nominal_load_W: float,
    cp_proxy_J_kgK: float,
    input_columns: list[str],
) -> pd.DataFrame:
    """Build the one-row input frame for the HP models, with derived features.

    Parameters
    ----------
    T_brine_in_C : float
        HP brine inlet temperature, °C.
    p_brine_bar : float
        Brine pressure, absolute bar.
    T_brine_out_min_C : float
        Minimum brine outlet temperature, °C.
    m_brine_max_kg_s : float
        Available brine flow, kg/s.
    heating_load_W : float
        Requested heating load, W.
    temperature_lift_K : float
        Sink supply temperature minus HP brine inlet temperature, K.
    sink_deltaT_K : float
        Sink supply minus sink return temperature, K.
    sink_approach_K : float
        Condensation temperature minus sink supply temperature, K.
    nominal_load_W : float
        Nominal heating load used for ``load_fraction_nominal``, W.
    cp_proxy_J_kgK : float
        Constant specific heat used only for the ``geo_capacity_proxy_W``
        feature, J/(kg K).
    input_columns : list of str
        Columns the models expect, in order (bundle ``input_columns``).

    Returns
    -------
    pandas.DataFrame
        One row with exactly ``input_columns``. Derived features include
        ``T_sink_supply_C``, ``T_sink_return_C``, ``T_cond_HP_target_C``,
        ``load_fraction_nominal``, ``source_deltaT_available_K`` and
        ``geo_capacity_proxy_W``.

    Raises
    ------
    KeyError
        If ``input_columns`` asks for a column that is not built here.
    """
    row: dict[str, float] = {
        "T_brine_in_C": float(T_brine_in_C),
        "p_brine_bar": float(p_brine_bar),
        "T_brine_out_min_C": float(T_brine_out_min_C),
        "m_brine_max_kg_s": float(m_brine_max_kg_s),
        "heating_load_W": float(heating_load_W),
        "temperature_lift_K": float(temperature_lift_K),
        "sink_deltaT_K": float(sink_deltaT_K),
    }

    row["T_sink_supply_C"] = row["T_brine_in_C"] + row["temperature_lift_K"]
    row["T_sink_return_C"] = row["T_sink_supply_C"] - row["sink_deltaT_K"]
    row["T_cond_HP_target_C"] = row["T_sink_supply_C"] + float(sink_approach_K)
    row["load_fraction_nominal"] = row["heating_load_W"] / max(float(nominal_load_W), 1.0)
    row["source_deltaT_available_K"] = row["T_brine_in_C"] - row["T_brine_out_min_C"]
    row["geo_capacity_proxy_W"] = (
        row["m_brine_max_kg_s"] * float(cp_proxy_J_kgK) * row["source_deltaT_available_K"]
    )
    row["sink_supply_minus_brine_out_min_K"] = row["T_sink_supply_C"] - row["T_brine_out_min_C"]
    row["cond_target_minus_brine_in_K"] = row["T_cond_HP_target_C"] - row["T_brine_in_C"]
    row["cond_target_minus_brine_out_min_K"] = row["T_cond_HP_target_C"] - row["T_brine_out_min_C"]

    if abs(row["source_deltaT_available_K"]) > SMALL:
        row["sink_deltaT_over_source_deltaT"] = row["sink_deltaT_K"] / row["source_deltaT_available_K"]
    else:
        row["sink_deltaT_over_source_deltaT"] = np.nan

    missing = [c for c in input_columns if c not in row]
    if missing:
        raise KeyError(f"Missing HP input columns: {missing}")

    return pd.DataFrame([{c: row[c] for c in input_columns}])


def domain_warnings(row: pd.Series, bounds: dict[str, Any]) -> list[str]:
    """List inputs outside the training range.

    Parameters
    ----------
    row : pandas.Series
        Model inputs (including derived features), indexed by column name.
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
        lo = lim.get("min")
        hi = lim.get("max")
        if lo is not None and val < float(lo):
            warnings.append(f"{col}={val:.6g} below training range min {float(lo):.6g}")
        if hi is not None and val > float(hi):
            warnings.append(f"{col}={val:.6g} above training range max {float(hi):.6g}")
    return warnings


def compute_variant(bundle: dict[str, Any], X: np.ndarray, q_pred: np.ndarray, variant: str) -> dict[str, float]:
    """Compressor power and COP from one variant model, before constraints.

    Variants:

    - ``direct_w``: the model predicts ``hp_W_comp_W``.
    - ``inverse_cop``: the model predicts ``hp_inverse_COP`` (W / Q);
      ``hp_W_comp_W = Q * hp_inverse_COP``.
    - ``pressure_hybrid``: the model predicts condensing and evaporating
      pressures, ``hp_W_is_proxy_W`` and (in the corrected bundles)
      ``hp_eta_is``; ``hp_W_comp_W = hp_W_is_proxy_W / eta``.

    Afterwards ``hp_W_comp_W`` is clipped to ``[0, Q]`` and
    ``hp_Q_source_W`` is set to ``Q - W`` so that the energy balance closes.

    Parameters
    ----------
    bundle : dict
        HP bundle.
    X : numpy.ndarray
        Model inputs, shape ``(1, n_features)``.
    q_pred : numpy.ndarray
        Output of the bundle's ``q_model`` for ``X``, shape
        ``(1, len(bundle["q_output_columns"]))``.
    variant : str
        Name of the variant (a key of ``bundle["variant_models"]``).

    Returns
    -------
    dict
        Includes ``hp_Q_delivered_W`` (W, >= 0), ``hp_W_comp_W`` (W),
        ``hp_COP``, ``hp_inverse_COP``, ``hp_Q_source_W`` (W),
        ``hp_Q_source_surrogate_W`` (the learned source heat, kept as a
        diagnostic, W) and ``hp_energy_balance_residual_W``, plus the raw
        variant outputs. ``pressure_hybrid`` also adds ``hp_pressure_ratio``
        and ``hp_eta_is_fit``.

    Raises
    ------
    KeyError
        If the variant is not in the bundle or has no target columns.
    ValueError
        If the variant name is not one of the three above.
    """
    if variant not in bundle["variant_models"]:
        raise KeyError(
            f"Variant {variant!r} is not available. "
            f"Available variants: {list(bundle['variant_models'].keys())}"
        )

    q_cols = list(bundle["q_output_columns"])
    cfg = bundle["variant_configs"][variant]
    cols = cfg.get("cols") or cfg.get("target_columns")
    if not cols:
        raise KeyError(f"No target columns found for HP variant {variant!r}")

    log_cols = set(cfg.get("log_cols", [])) | set(cfg.get("log_columns", []))
    raw = np.asarray(predict_model(bundle["variant_models"][variant],X), dtype=float)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)

    out: dict[str, float] = {}
    for i, col in enumerate(q_cols):
        out[col] = float(q_pred[0, i])

    for j, col in enumerate(cols):
        val = raw[0, j]
        if col in log_cols:
            val = undo_log(np.array([val]))[0]
        out[col] = float(val)

    Q = max(float(out["hp_Q_delivered_W"]), 0.0)
    source_diagnostic = max(float(out.get("hp_Q_source_W", 0.0)), 0.0)

    if variant == "direct_w":
        W = max(float(out["hp_W_comp_W"]), SMALL)
        out["hp_inverse_COP"] = W / max(Q, SMALL)
        out["hp_COP"] = Q / W

    elif variant == "inverse_cop":
        inv = max(float(out["hp_inverse_COP"]), SMALL)
        W = Q * inv
        out["hp_W_comp_W"] = W
        out["hp_COP"] = 1.0 / inv

    elif variant == "pressure_hybrid":
        pcond = max(float(out["hp_p_cond_bar"]), SMALL)
        pevap = max(float(out["hp_p_evap_bar"]), SMALL)
        pr = pcond / pevap
        # Corrected models learn the actual map efficiency. Historical bundles
        # used a pressure-only polynomial, which cannot express flow effects.
        eta = (float(np.clip(out["hp_eta_is"], 0.05, 1.0))
               if "hp_eta_is" in out else float(eval_eta_poly(bundle["eta_poly"], np.array([pr]))[0]))
        Wis = max(float(out["hp_W_is_proxy_W"]), SMALL)
        W = Wis / max(eta, SMALL)
        out["hp_pressure_ratio"] = pr
        out["hp_eta_is_fit"] = eta
        out["hp_W_comp_W"] = W
        out["hp_inverse_COP"] = W / max(Q, SMALL)
        out["hp_COP"] = Q / max(W, SMALL)

    else:
        raise ValueError(f"Unknown HP variant: {variant}")

    W = float(np.clip(out["hp_W_comp_W"], 0.0, Q))
    out["hp_Q_delivered_W"] = Q
    out["hp_W_comp_W"] = W
    out["hp_inverse_COP"] = W / Q if Q > 0 else 0.0
    out["hp_COP"] = Q / max(W, SMALL) if Q > 0 else 0.0
    out["hp_Q_source_W"] = Q - W
    out["hp_Q_source_surrogate_W"] = source_diagnostic
    out["hp_energy_balance_residual_W"] = Q - out["hp_Q_source_W"] - W
    return out


def enforce_operating_constraints(out: dict[str, Any], *, heating_load_W: float, T_brine_in_C: float,
                                  T_brine_out_min_C: float, m_brine_max_kg_s: float,
                                  cp_J_kgK: float = 4200.0, full_available_flow: bool = False,
                                  UA_common_W_K: float | None = None, sink_deltaT_K: float | None = None,
                                  sink_approach_K: float = 5.0) -> dict[str, Any]:
    """Project onto the existing constant-cp source envelope at predicted COP.

    This protects feasibility outside stable fitting data; it is not a detailed
    thermodynamic re-solve. The unprojected source estimate remains diagnostic.

    Delivered heat, compressor power and source heat are multiplied by one
    common factor (at most 1, so the COP is kept), chosen so that none of
    these limits is exceeded:

    - delivered heat <= ``heating_load_W``;
    - source heat <= ``cp_J_kgK * m_brine_max_kg_s * (T_brine_in_C -
      T_brine_out_min_C)``;
    - delivered heat <= condenser limit ``UA_common_W_K * sink_deltaT_K /
      ln(1 + sink_deltaT_K / sink_approach_K)`` (only if both
      ``UA_common_W_K`` and ``sink_deltaT_K`` are given).

    The brine outlet temperature then follows from a constant-cp balance, and
    ``hp_T_evap_C`` is set to the brine outlet temperature minus 5 K.

    Parameters
    ----------
    out : dict
        Output of :func:`compute_variant` (not modified; a copy is returned).
    heating_load_W : float
        Requested heating load, W.
    T_brine_in_C : float
        HP brine inlet temperature, °C.
    T_brine_out_min_C : float
        Minimum brine outlet temperature, °C.
    m_brine_max_kg_s : float
        Available brine flow, kg/s.
    cp_J_kgK : float, optional
        Constant brine specific heat, J/(kg K). Default 4200.
    full_available_flow : bool, optional
        If true, the brine flow is always ``m_brine_max_kg_s``; otherwise it
        is the learned ``hp_m_brine_kg_s`` clipped between the flow needed
        for the source heat and ``m_brine_max_kg_s``.
    UA_common_W_K : float, optional
        UA of the fixed condenser (common heat exchanger), W/K.
    sink_deltaT_K : float, optional
        Sink supply minus return temperature, K.
    sink_approach_K : float, optional
        Condensation minus sink supply temperature, K. Default 5.

    Returns
    -------
    dict
        Copy of ``out`` with updated ``hp_Q_delivered_W``, ``hp_W_comp_W``,
        ``hp_Q_source_W``, ``hp_COP``, ``hp_inverse_COP``, and added
        ``hp_constraint_projection_applied`` (bool), ``hp_m_brine_kg_s``,
        ``hp_T_brine_out_C``, ``hp_T_evap_C``, ``hp_unmet_load_W``,
        ``hp_load_coverage_frac``, ``hp_source_capacity_limit_W``,
        ``hp_condenser_capacity_limit_W`` (only if a condenser limit was
        computed) and ``hp_energy_balance_residual_W``.
    """
    out = dict(out)
    q = max(float(out["hp_Q_delivered_W"]), 0.0)
    w = float(np.clip(out["hp_W_comp_W"], 0.0, q))
    source = q - w
    delta = max(T_brine_in_C - T_brine_out_min_C, 0.0)
    capacity = cp_J_kgK * m_brine_max_kg_s * delta
    hx_limit = (float(UA_common_W_K) * sink_deltaT_K /
                np.log1p(sink_deltaT_K / sink_approach_K)
                if UA_common_W_K is not None and sink_deltaT_K is not None else float("inf"))
    factor = min(1.0, heating_load_W / q if q > 0 else 1.0,
                 capacity / source if source > 0 else 1.0,
                 hx_limit / q if q > 0 else 1.0)
    out["hp_constraint_projection_applied"] = bool(factor < 1.0 - 1e-12)
    out["hp_Q_delivered_W"] = q * factor
    out["hp_W_comp_W"] = w * factor
    out["hp_Q_source_W"] = source * factor
    q, w, source = (out[k] for k in ("hp_Q_delivered_W", "hp_W_comp_W", "hp_Q_source_W"))
    out["hp_inverse_COP"] = w / q if q > 0 else 0.0
    out["hp_COP"] = q / max(w, SMALL) if q > 0 else 0.0
    required_mass = source / (cp_J_kgK * delta) if delta > 0 else 0.0
    mass = float(np.clip(out.get("hp_m_brine_kg_s", m_brine_max_kg_s),
                         required_mass, m_brine_max_kg_s))
    if full_available_flow:
        mass = float(m_brine_max_kg_s)
    out["hp_m_brine_kg_s"] = mass
    out["hp_T_brine_out_C"] = T_brine_in_C - source / (cp_J_kgK * mass) if mass > 0 else T_brine_in_C
    if "hp_T_evap_C" in out:
        out["hp_T_evap_C"] = out["hp_T_brine_out_C"] - 5.0
    out["hp_unmet_load_W"] = max(heating_load_W - q, 0.0)
    out["hp_load_coverage_frac"] = q / heating_load_W if heating_load_W > 0 else 0.0
    out["hp_source_capacity_limit_W"] = capacity
    if np.isfinite(hx_limit): out["hp_condenser_capacity_limit_W"] = hx_limit
    out["hp_T_evap_C"] = out["hp_T_brine_out_C"] - 5.0
    out["hp_energy_balance_residual_W"] = q - source - w
    return out


def reconcile_pressures(out: dict[str, Any], bundle: dict[str, Any], *, T_brine_in_C: float,
                        temperature_lift_K: float) -> dict[str, Any]:
    """Reconstruct saturation pressure at final temperatures without CoolProp.

    Learned pressure-head values remain diagnostics. Outside the embedded
    saturation range actual pressure fields are omitted, never endpoint-clipped.

    Parameters
    ----------
    out : dict
        Output of :func:`enforce_operating_constraints` (must contain
        ``hp_T_evap_C``). Not modified; a copy is returned.
    bundle : dict
        HP bundle. If it has no ``saturation_pressure_table``, ``out`` is
        returned unchanged.
    T_brine_in_C : float
        HP brine inlet temperature, °C.
    temperature_lift_K : float
        Sink supply temperature minus HP brine inlet temperature, K.

    Returns
    -------
    dict
        Copy of ``out`` where learned ``hp_p_cond_bar``/``hp_p_evap_bar`` are
        renamed to ``hp_p_cond_surrogate_bar``/``hp_p_evap_surrogate_bar``,
        and with ``hp_T_cond_C`` (= ``T_brine_in_C + temperature_lift_K +
        sink_approach_K``, °C), ``hp_p_cond_bar`` and ``hp_p_evap_bar``
        (bar, interpolated in the table; each omitted if its temperature is
        outside the table), ``hp_pressure_state_supported`` (bool) and
        ``hp_pressure_ratio`` (only if both pressures are supported).
    """
    out = dict(out)
    table = bundle.get("saturation_pressure_table")
    if table is None:
        return out
    for field in ["hp_p_cond_bar", "hp_p_evap_bar"]:
        if field in out:
            out[field.replace("_bar", "_surrogate_bar")] = out.pop(field)
    out["hp_T_cond_C"] = T_brine_in_C + temperature_lift_K + float(bundle.get("sink_approach_K", 5.))
    temps = np.asarray(table["temperature_C"], dtype=float)
    pressures = np.asarray(table["pressure_bar"], dtype=float)
    supported = True
    for target, temperature in [("hp_p_cond_bar", out["hp_T_cond_C"]),
                                ("hp_p_evap_bar", out["hp_T_evap_C"])]:
        if temps[0] <= temperature <= temps[-1]:
            out[target] = float(np.interp(temperature, temps, pressures))
        else:
            supported = False
    out["hp_pressure_state_supported"] = supported
    out.pop("hp_pressure_ratio", None)
    if supported:
        out["hp_pressure_ratio"] = out["hp_p_cond_bar"] / out["hp_p_evap_bar"]
    return out


def predict_1mw_hp_performance(
    *,
    model_path: str | Path,
    T_brine_in_C: float,
    p_brine_bar: float,
    T_brine_out_min_C: float,
    m_brine_max_kg_s: float,
    heating_load_W: float,
    temperature_lift_K: float,
    sink_deltaT_K: float,
    variant: str | None = None,
    return_all_variants: bool = True,
) -> dict[str, Any]:
    """Predict the off-design performance of the fixed 1 MW nominal heat pump.

    Parameters
    ----------
    model_path : str or pathlib.Path
        Path to the HP bundle, normally
        ``models/hp_synthetic_year_surrogate_1mw_filtered_variants_vopt6_hp55_75_out35_45_sourcebalanced.joblib``.
    T_brine_in_C : float
        HP brine inlet temperature, °C (training domain 55–75).
    p_brine_bar : float
        Brine pressure, absolute bar (training domain 12–24). Must be > 0.
    T_brine_out_min_C : float
        Minimum brine outlet temperature, °C (training domain 35–45). A lower
        operating constraint; must be below ``T_brine_in_C``.
    m_brine_max_kg_s : float
        Available brine flow, kg/s (training domain 5–12). Must be > 0. With
        the shipped bundle the full available flow is always used.
    heating_load_W : float
        Requested heating load, W (training domain 50000–1e6; below 200000 W
        is extrapolation). Must be >= 0.
    temperature_lift_K : float
        Sink supply temperature minus ``T_brine_in_C``, K. The resulting sink
        supply temperature has training domain 75–100 °C.
    sink_deltaT_K : float
        Sink supply minus sink return temperature, K (training domain 10–25).
        Must be > 0.
    variant : str, optional
        ``"inverse_cop"``, ``"direct_w"`` or ``"pressure_hybrid"``. Default:
        the bundle's ``selected_variant`` (``"inverse_cop"``).
    return_all_variants : bool, optional
        If true (default), also evaluate every variant in the bundle and
        return them under ``"variants"``.

    Returns
    -------
    dict
        - ``"surrogate"``: ``"hp_offdesign"``.
        - ``"selected_variant"``: name of the variant reported in ``"hp"``.
        - ``"inputs"``: model inputs including derived features (e.g.
          ``T_sink_supply_C``, ``T_sink_return_C``, ``T_cond_HP_target_C``).
        - ``"hp"``: results of the selected variant. Main keys:
          ``hp_Q_delivered_W``, ``hp_W_comp_W``, ``hp_Q_source_W`` (W),
          ``hp_COP``, ``hp_m_brine_kg_s``, ``hp_T_brine_out_C``,
          ``hp_unmet_load_W``, ``hp_load_coverage_frac``, ``hp_T_cond_C``,
          ``hp_T_evap_C``, ``hp_p_cond_bar``, ``hp_p_evap_bar`` (omitted if
          outside the saturation table), ``hp_constraint_projection_applied``,
          ``hp_physical_status_supported``, ``hp_pressure_state_supported``;
          raw diagnostics ``hp_Q_delivered_raw_W``, ``hp_W_comp_raw_W`` and
          ``hp_Q_source_surrogate_W``; and ``hp_energy_balance_residual_W``.
        - ``"warnings"``: list of str (inputs outside the training range,
          projection applied, load below the fitting regime, unsupported
          regime or pressure state).
        - ``"model_metadata"``: ``bundle_version`` and reference hashes.
        - ``"variants"``: ``{name: dict like "hp"}`` (only if
          ``return_all_variants``).

    Raises
    ------
    ValueError
        If an input is not finite, ``T_brine_in_C <= T_brine_out_min_C``,
        ``p_brine_bar <= 0``, ``m_brine_max_kg_s <= 0``,
        ``heating_load_W < 0`` or ``sink_deltaT_K <= 0``.
    KeyError
        If ``variant`` is not available in the bundle.
    FileNotFoundError
        If ``model_path`` does not exist.
    """
    values = [T_brine_in_C, p_brine_bar, T_brine_out_min_C,
              m_brine_max_kg_s, heating_load_W, temperature_lift_K, sink_deltaT_K]
    if not np.isfinite(values).all():
        raise ValueError("HP inputs must be finite")
    if (T_brine_in_C <= T_brine_out_min_C or p_brine_bar <= 0
            or m_brine_max_kg_s <= 0 or heating_load_W < 0 or sink_deltaT_K <= 0):
        raise ValueError("Invalid HP operating boundary, flow, pressure, load or sink difference")
    bundle = load_bundle(model_path)
    input_columns = list(bundle["input_columns"])

    Xdf = build_feature_frame(
        T_brine_in_C=T_brine_in_C,
        p_brine_bar=p_brine_bar,
        T_brine_out_min_C=T_brine_out_min_C,
        m_brine_max_kg_s=m_brine_max_kg_s,
        heating_load_W=heating_load_W,
        temperature_lift_K=temperature_lift_K,
        sink_deltaT_K=sink_deltaT_K,
        sink_approach_K=float(bundle.get("sink_approach_K", 5.0)),
        nominal_load_W=float(bundle.get("nominal_load_W", 1.0e6)),
        cp_proxy_J_kgK=float(bundle.get("cp_proxy_J_kgK", 4180.0)),
        input_columns=input_columns,
    )

    X = Xdf[input_columns].to_numpy(dtype=float)
    q_pred = np.asarray(predict_model(bundle["q_model"],X), dtype=float)
    if q_pred.ndim == 1:
        q_pred = q_pred.reshape(1, -1)

    selected = variant or bundle.get("selected_variant", "inverse_cop")
    classifier = bundle.get("physical_status_classifier")
    status_supported = True if classifier is None else bool(predict_model(classifier, X)[0])
    def evaluate(name: str) -> dict[str, Any]:
        """Variant prediction -> operating constraints -> pressures.

        If the bundle's physical-status classifier rejects the input, heat and
        power are set to zero before the constraints are applied.
        """
        out = compute_variant(bundle, X, q_pred, name)
        out["hp_Q_delivered_raw_W"] = out["hp_Q_delivered_W"]
        out["hp_W_comp_raw_W"] = out["hp_W_comp_W"]
        if not status_supported:
            out["hp_Q_delivered_W"] = out["hp_W_comp_W"] = 0.0
        out = enforce_operating_constraints(
            out, heating_load_W=heating_load_W,
            T_brine_in_C=T_brine_in_C, T_brine_out_min_C=T_brine_out_min_C,
            m_brine_max_kg_s=m_brine_max_kg_s,
            cp_J_kgK=float(bundle.get("source_cp_J_kgK", 4200.0)),
            full_available_flow=bundle.get("source_flow_policy") == "full_available",
            UA_common_W_K=bundle.get("reference_hardware", {}).get("UA_common_W_K"),
            sink_deltaT_K=sink_deltaT_K, sink_approach_K=float(bundle.get("sink_approach_K", 5.)))
        out["hp_physical_status_supported"] = status_supported
        return reconcile_pressures(out, bundle, T_brine_in_C=T_brine_in_C,
                                   temperature_lift_K=temperature_lift_K)

    result: dict[str, Any] = {
        "surrogate": "hp_offdesign",
        "selected_variant": selected,
        "inputs": {k: float(v) for k, v in Xdf.iloc[0].to_dict().items()},
        "hp": evaluate(selected),
        "warnings": domain_warnings(Xdf.iloc[0], bundle.get("input_bounds_filtered", {})),
        "model_metadata": {k: bundle.get(k) for k in ["bundle_version", "reference_source_sha256", "reference_hardware_sha256"]},
    }

    if return_all_variants:
        result["variants"] = {
            name: (result["hp"] if name == selected else evaluate(name))
            for name in bundle["variant_models"].keys()
        }

    if result["hp"]["hp_constraint_projection_applied"]:
        result["warnings"].append("HP output projected to requested-load, condenser and source-capacity limits at predictedCOP.")
    if heating_load_W < float(bundle.get("minimum_fit_load_W", 200000.)):
        result["warnings"].append("Requested load is below the20% fitting regime; this is extrapolation.")
    if not status_supported:
        result["warnings"].append("Input is classified outside the supported active physical regime; heat and power are returned as zero.")
    if result["hp"].get("hp_pressure_state_supported") is False:
        result["warnings"].append("Final temperature is outside the supported saturation-pressure range; unsupported actual pressure fields are omitted. No endpoint clipping is applied.")
    return result


def build_parser() -> argparse.ArgumentParser:
    """Command-line options of this module (see ``--help``)."""
    p = argparse.ArgumentParser(description="Predict 1 MW HP off-design performance.")
    p.add_argument("--model", required=True)
    p.add_argument("--T-brine-in-c", type=float, required=True)
    p.add_argument("--p-brine-bar", type=float, required=True)
    p.add_argument("--T-brine-out-min-c", type=float, required=True)
    p.add_argument("--m-brine-max-kg-s", type=float, required=True)

    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--heating-load-W", type=float)
    group.add_argument("--heating-load-mw", type=float)

    p.add_argument("--temperature-lift-K", type=float, required=True)
    p.add_argument("--sink-deltaT-K", type=float, required=True)
    p.add_argument("--variant", choices=["direct_w", "inverse_cop", "pressure_hybrid"], default=None)
    p.add_argument("--selected-only", action="store_true")
    return p


def main() -> None:
    """Run one HP off-design prediction from the command line; print JSON."""
    args = build_parser().parse_args()
    heating_load_W = args.heating_load_W
    if heating_load_W is None:
        heating_load_W = float(args.heating_load_mw) * 1.0e6

    result = predict_1mw_hp_performance(
        model_path=args.model,
        T_brine_in_C=args.T_brine_in_c,
        p_brine_bar=args.p_brine_bar,
        T_brine_out_min_C=args.T_brine_out_min_c,
        m_brine_max_kg_s=args.m_brine_max_kg_s,
        heating_load_W=heating_load_W,
        temperature_lift_K=args.temperature_lift_K,
        sink_deltaT_K=args.sink_deltaT_K,
        variant=args.variant,
        return_all_variants=not bool(args.selected_only),
    )
    print(json.dumps(result, indent=2, default=json_default))


if __name__ == "__main__":
    main()
