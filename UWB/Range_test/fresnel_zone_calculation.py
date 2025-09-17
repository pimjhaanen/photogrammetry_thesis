"""This code can be used to calculate the Fresnel zone radius for UWB ground clearance.
It assumes a worst-case scenario where both devices are at equal height and separated
by a given distance. The Fresnel radius gives the minimum clearance required to avoid
ground obstruction of the signal."""

import numpy as np

def fresnel_zone(frequency_hz: float, distance_m: float) -> float:
    """RELEVANT FUNCTION INPUTS:
    - frequency_hz: operating frequency in Hz (e.g. 6.5e9 for UWB at 6.5 GHz)
    - distance_m: separation distance between transmitter and receiver in meters

    RETURNS:
    - r1: radius of the first Fresnel zone at the midpoint (in meters)
    """
    c = 3.0e8  # speed of light in m/s
    wavelength = c / frequency_hz
    d1 = d2 = distance_m / 2.0
    r1 = np.sqrt((wavelength * d1 * d2) / distance_m)
    return r1

if __name__ == "__main__":
    f_uwb = 6.5e9     # UWB frequency (Hz)
    D_est = 60.0      # max estimated distance (m)

    clearance = fresnel_zone(f_uwb, D_est)
    print(f"The minimum ground clearance (worst-case, equal sensor heights) "
          f"should be {clearance:.2f} m")
