# Design

## Context

- RevORCHP-SM 1.0.2: four scikit-learn models in `models/` (design; heat-pump off-design; one-stage and two-stage ORC off-design) and five Python scripts at the top level, tested by the authors on Windows with Python 3.12.10.
- TANGO is a Python package (Python 3.11 to 3.13) that already depends on scikit-learn, numpy, scipy and pandas. It couples external models through a small wrapper module per model, and depends only on releases their authors publish; it never copies or modifies another team's code.
- For each hour TANGO would send brine inlet temperature, pressure and mass flow per module, ambient temperature, the minimum brine outlet temperature and, in heat-pump mode, the heating load and sink temperatures; it would read back net electric power, delivered heat, brine outlet temperature and the status of the result.

## Decisions

### D1: Tests first

The regression tests come before any other change, so that every later change, by anyone, can be shown not to change the model outputs. The stored reference outputs are produced by the current, unchanged code. A comparison tolerance of 1e-9 (relative) allows only floating-point noise.

### D2: Your commands keep working

The package layout keeps a script with each original name at the top level, and the model files stay in `models/`, so every README command works unchanged. The installed package also contains the model files.

### D3: Only scikit-learn stays pinned

The `.joblib` files store the scikit-learn version that wrote them, so scikit-learn stays at exactly 1.8.0 until D5 is done. numpy, scipy, pandas, joblib and threadpoolctl get lower bounds. The tests pass with numpy 2.0 and pandas 2.2 (the oldest allowed) and with numpy 2.5 and pandas 3.1 (the newest at the time of writing). `requirements.txt` stays as the exact environment you tested.

### D4: New functions, not changed ones

The batch interface, status values and bounds (item 4) are new functions next to the existing ones, which keep their names, arguments and results.

### D5: Portable model files alongside the originals

The exported files sit next to the `.joblib` files, and a test checks that both give the same results at the reference points. The `.joblib` files stay the reference.

## Who does what

To agree with the authors. The TANGO team has drafted items 1 to 3 as pull requests.

| Item | Proposed | Agreed |
|---|---|---|
| 1. Regression tests and CI | TANGO team (drafted) | |
| 2. Installable package | TANGO team (drafted) | |
| 3. Docstrings | TANGO team (drafted); authors check the physics descriptions | |
| 4. Program-friendly interface | open | |
| 5. Portable model files | open | |

## Open Questions

About the models:

1. The README gives the one-stage ORC ambient temperature range as -11.9 to 34.8 °C, but the one-stage model file stores -15.2 to 34.8 °C, the same as the two-stage one. Which is correct?
2. The design model sets the ORC evaporation temperature to brine inlet minus 10 K in the code, while the condensation offset (10.03 K) comes from the model file. Is the fixed 10 K intended?
3. The heat-pump off-design model sets the evaporation temperature to brine outlet minus 5 K in the code, while the condenser approach (5 K) comes from the model file. Should the evaporator approach also come from the model file?
4. What does the design output `acc_margin` mean, and in which units?
5. What is `orc_expander_fallback_probability` for? It is always 0 in the design output.
6. In the ORC model files, the "active" classifier is a constant (always active), so whether a point is active depends only on the checks in the code. Is that intended?
7. Are the older code paths that no shipped model file uses (the non-"corrected" ORC branch, `outputs_from_targets`, `inverse_y`, the heat-pump `eval_eta_poly` fallback) still needed?
8. A few warning texts contain typos ("predictedCOP", "the20%", "efficiency0.95"). Fixing them changes the output text; shall we fix them in a separate pull request?

About the coupling:

9. Who should decide between ORC and heat-pump operation at each hour: the plant model, TANGO, or a dispatch model?
10. TANGO's wells can deliver brine outside the trained ranges (95 to 115 °C for the ORC, 55 to 75 °C for the heat pump). Answered 2026-11-18: flag such points, never clip them, and evaluate them with the detailed model where possible. Still open: could the ranges be widened in a later version?
11. For plants of several MW, is it right to sum identical 1 MW modules?
12. Are there cost data per module (investment, operation and maintenance) consistent with the model?
13. Partly answered 2026-11-18: the detailed model is Python with CoolProp, accepts batches of hourly inputs, can size the hardware within the machine limits, and takes about 50 s per design, 0.04 to 0.6 s per hourly heat-pump point and 1.2 to 4.5 s per hourly ORC point. An updated version is to be shared by the meeting of 2026-11-20. Still open: where it will live (this repository or its own), its licence, Linux support, and pinned versions, including CoolProp (TANGO uses 6.8.0).
14. Would you like to keep the `openspec/` folder (these documents) in the repository, or would you rather we keep this discussion in a pull request or an issue?
