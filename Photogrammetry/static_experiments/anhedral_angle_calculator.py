import numpy as np
from plot_static_results import _prep_dataset_from_csv

def compute_anhedral(csv_path, b,
                     *,
                     endpoints_k=10,
                     jump_factor=2.5,
                     tip_policy="lowestZ",
                     global_up=(0,0,1),
                     force_flip_x=False,
                     transform_to_local=True):
    """
    Compute anhedral angle from a static reconstruction file.

    Steps:
        1. Transform to wing reference frame (same as strut/twist).
        2. Origin in yz-plane = (0,0).
        3. Detect LE-tips → take tip with lowest idx and highest idx.
        4. Compute midpoint of their (y,z) coordinates.
        5. h = distance from origin to midpoint (only yz-plane).
        6. angle = arctan( h / (0.5*b) )
    """

    pts_local, le_path = _prep_dataset_from_csv(
        csv_path,
        endpoints_k=endpoints_k,
        jump_factor=jump_factor,
        tip_policy=tip_policy,
        global_up=global_up,
        force_flip_x=force_flip_x,
        transform_to_local=transform_to_local,
    )

    if le_path is None or le_path.shape[0] < 2:
        raise ValueError(f"LE path missing in {csv_path}")

    # -------- 1. extract LE tips (lowest and highest idx in LE group) --------
    # they are already ordered by idx in the CSV
    LE = pts_local.get("LE")
    if LE is None or LE.shape[0] < 2:
        raise ValueError("LE points missing or too few")

    # lowest idx = first rows in CSV, highest idx = last rows
    tip_left  = LE[0]       # (x,y,z)
    tip_right = LE[-1]

    # -------- 2. midpoint of the tips in yz-plane --------
    y_mid = 0.5 * (tip_left[1] + tip_right[1])
    z_mid = 0.5 * (tip_left[2] + tip_right[2])

    # -------- 3. origin = (0,0) in yz-plane --------
    h = np.sqrt(y_mid**2 + z_mid**2)

    # -------- 4. compute anhedral angle --------
    half_span = 0.5 * b
    angle_rad = np.arctan2(h, half_span)
    angle_deg = np.degrees(angle_rad)

    return angle_deg, h, (y_mid, z_mid)


if __name__ == "__main__":
    b_powered = 7.32  # example span
    b_depowered = 7.12  # example span

    angle_p, h_p, mid_p = compute_anhedral(
        "static_test_output/powered_state_reelin_frame_17372.csv",
        b_powered
    )

    angle_d, h_d, mid_d = compute_anhedral(
        "static_test_output/depowered_state_reelin_frame_17611.csv",
        b_depowered
    )

    print("Powered anhedral:", angle_p, "deg   h =", h_p, "   midpoint =", mid_p)
    print("Depowered anhedral:", angle_d, "deg   h =", h_d, "   midpoint =", mid_d)
