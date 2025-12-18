"""This code can be used to fit and save a linear calibration for Pozyx UWB distances.
It estimates coefficients a, b such that corrected = a * measured + b, evaluates errors
before/after calibration, and (optionally) plots the error vs. ground truth."""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
from typing import Tuple, Dict, Iterable, Optional


def fit_uwb_calibration(
    measured: Iterable[float],
    actual: Iterable[float],
    output_json: str = "uwb_calibration.json",
    make_plot: bool = True,
    title: str = "UWB Measurement Error Before and After Calibration",
) -> Dict[str, float]:
    """RELEVANT FUNCTION INPUTS:
    - measured: iterable of raw UWB distances (meters) recorded by the device (x-values)
    - actual: iterable of corresponding ground-truth distances (meters) (y-values)
    - output_json: filename for saving calibration coefficients in {"a": ..., "b": ...} format
    - make_plot: if True, show a plot of errors before and after calibration
    - title: plot title string

    RETURNS:
    - {"a": a, "b": b}: the linear calibration parameters for corrected = a * measured + b
    """
    measured = np.asarray(measured, dtype=float)
    actual = np.asarray(actual, dtype=float)
    if measured.shape != actual.shape:
        raise ValueError("measured and actual must have the same length.")

    # --- Linear fit: actual ≈ a*measured + b ---
    a, b = np.polyfit(measured, actual, 1)
    print(f"Calibration equation: corrected = {a:.6f} * measured + {b:.6f}")

    # --- Save to JSON (compatible with your logger/postprocess scripts) ---
    calib = {"a": float(a), "b": float(b)}
    with open(output_json, "w") as f:
        json.dump(calib, f, indent=4)
    print(f"✅ Calibration saved to: {os.path.abspath(output_json)}")

    # --- Apply calibration and compute errors ---
    corrected = a * measured + b
    error_before = measured - actual
    error_after = corrected - actual

    avg_abs_before = float(np.mean(np.abs(error_before)))
    avg_abs_after = float(np.mean(np.abs(error_after)))

    print(f"Average |error| before calibration: {avg_abs_before:.4f} m")
    print(f"Average |error| after  calibration: {avg_abs_after:.4f} m")

    # --- Plot (optional) ---
    if make_plot:
        plt.figure(figsize=(6, 5))
        plt.plot(actual, error_before, "o-", label="RAW")
        plt.plot(actual, error_after, "--",  label="Calibrated")
        plt.axhline(0, linestyle=":", linewidth=0.8)
        plt.xlabel("Actual distance (m)")
        plt.ylabel("$\epsilon_{UWB}$ (m)")
        plt.title(title)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return calib


if __name__ == "__main__":
    # Example data: three static samples per distance (std ≤ ~0.03 m as measured)
    measured = np.array([4.95, 6.91, 8.93, 10.91, 12.90, 14.97])
    actual   = np.array([5.00, 7.00, 9.00, 11.00, 13.00, 15.00])

    fit_uwb_calibration(
        measured=measured,
        actual=actual,
        output_json="uwb_calibration.json",
        make_plot=True,
    )
