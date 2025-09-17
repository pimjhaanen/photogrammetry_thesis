"""This code can be used to estimate depth error for stereo vision as a function of baseline.
It plots (or returns) the expected depth error ε_z (cm) at given scene depths z (m), for one or
more focal lengths (in pixels) and assumed disparity errors (in pixels)."""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union, Optional


def depth_error_vs_baseline(
    depths: List[float] = [7.0],
    focal_lengths: Dict[str, float] = {"Wide": 2337.803},  # focal length in pixels
    disparity_errors: List[float] = [0.25, 0.5, 0.75, 1.0],  # ε_d in pixels
    baseline_range: Tuple[float, float, float] = (0.01, 1.0, 0.01),  # start, stop, step in meters
    accuracy_max_cm: float = 5.0,
    ylim_max_cm: float = 6.0,
    plot: bool = True
) -> Dict[str, Union[np.ndarray, Dict[str, Dict[float, np.ndarray]]]]:
    """RELEVANT FUNCTION INPUTS:
    - depths: list of scene depths z in meters, taken as estimated maximum distance of camera to kite
    - focal_lengths: mapping {lens_label: focal_length_px} where focal length is in pixels
    - disparity_errors: list of assumed disparity errors ε_d in pixels
    - baseline_range: (start, stop, step) of baseline B (meters) to evaluate
    - accuracy_max_cm: requirement line (cm) drawn on the plot
    - ylim_max_cm: upper limit (cm) for y-axis
    - plot: if True, show matplotlib figures; if False, return computed arrays without plotting
    """

    B = np.arange(*baseline_range)  # Baseline array in meters
    results = {
        "baseline": B,
        "curves": {},
        "threshold_points": {}
    }

    for z in depths:
        if plot:
            plt.figure()
            plt.axhline(y=accuracy_max_cm, color='grey', linestyle='--', label='REQ-EXP-01.2')

        # For minimal changes: keep one black dot (first time we meet the requirement) for Wide, ε_d=1
        plotted_point = False

        for lens_label, f_px in focal_lengths.items():
            results["curves"].setdefault(lens_label, {})
            results["threshold_points"].setdefault(lens_label, {})

            for eps_d in disparity_errors:
                # ε_z (cm) = (z^2 * ε_d) / (f * B) * 100
                eps_z_cm = (z**2 * eps_d) / (f_px * B) * 100.0
                results["curves"][lens_label][eps_d] = eps_z_cm

                if plot:
                    label = f"{lens_label}, $\\epsilon_d$ = {eps_d}"
                    plt.plot(B, eps_z_cm, label=label)

                # Find first baseline meeting accuracy requirement
                B_min = None
                eps_at = None
                for b_val, ez in zip(B, eps_z_cm):
                    if ez <= accuracy_max_cm:
                        # NOTE: for backward compatibility with your original print, keep *2 here.
                        # If you prefer the true baseline, change to just 'b_val'.
                        if (lens_label == "Wide") and (eps_d == 1.0) and (not plotted_point) and plot:
                            plt.plot(b_val, ez + 0.02, 'ko', label='Minimum baseline')
                            print(f"baseline minimum: {round(b_val*2, 2)} m")
                            plotted_point = True
                        B_min, eps_at = b_val, ez
                        break

                results["threshold_points"][lens_label][eps_d] = (B_min, eps_at) if B_min is not None else None

        if plot:
            plt.ylim(0, ylim_max_cm)
            plt.title(f"Estimated depth error ($\\epsilon_z$) at $z$ = {z} m")
            plt.xlabel("Baseline $B$ (m)")
            plt.ylabel("Depth error $\\epsilon_z$ (cm)")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()

    return results

if __name__ == "__main__":
    # Example run with plotting enabled
    depth_error_vs_baseline(plot=True)
