import numpy as np
from plot_static_results import _prep_dataset_from_csv

def _tip_points(le_path):
    """Return first and last LE points as tipL, tipR."""
    if le_path is None or len(le_path) < 2:
        raise ValueError("LE path has too few points.")
    return le_path[0], le_path[-1]   # already ordered by your preprocessing

def _angle_in_plane(p1, p2, idx_a, idx_b):
    """
    Compute angle of vector p2-p1 projected onto plane (idx_a, idx_b).
    Example: (1,0) gives angle in y–x plane.
    """
    v = np.array(p2) - np.array(p1)
    a = v[idx_a]
    b = v[idx_b]
    return np.degrees(np.arctan2(b, a))

def _wrap_deg(angle):
    """Wrap angle to (-180, 180]."""
    return (angle + 180) % 360 - 180

def _smallest_angle_diff(a, b):
    """Return signed minimal difference b - a in (-180, 180]."""
    da = _wrap_deg(a)
    db = _wrap_deg(b)
    return _wrap_deg(db - da)


def compute_zeta_xi(csv_path_straight, csv_path_turn):
    """
    zeta = change of tip-line angle in the y–x plane
    xi   = change of tip-line angle in the z–y plane
    Both in the WING reference frame.
    """

    # --- load + transform both datasets ---
    (pts_s, le_s) = _prep_dataset_from_csv(
        csv_path_straight,
        endpoints_k=10,
        jump_factor=2.5,
        tip_policy="lowestZ",
        global_up=(0,0,1),
        force_flip_x=False,
        transform_to_local=True
    )

    (pts_t, le_t) = _prep_dataset_from_csv(
        csv_path_turn,
        endpoints_k=10,
        jump_factor=2.5,
        tip_policy="lowestZ",
        global_up=(0,0,1),
        force_flip_x=False,
        transform_to_local=True
    )

    # --- extract tips ---
    tipL_s, tipR_s = _tip_points(le_s)
    tipL_t, tipR_t = _tip_points(le_t)

    # --- angles straight flight ---
    ang_xy_s = _angle_in_plane(tipL_s, tipR_s, idx_a=1, idx_b=0)   # y–x plane
    ang_zy_s = _angle_in_plane(tipL_s, tipR_s, idx_a=1, idx_b=2)   # z–y plane

    # --- angles turning flight ---
    ang_xy_t = _angle_in_plane(tipL_t, tipR_t, idx_a=1, idx_b=0)
    ang_zy_t = _angle_in_plane(tipL_t, tipR_t, idx_a=1, idx_b=2)

    # --- compute wrapped angle differences ---
    zeta = _smallest_angle_diff(ang_xy_s, ang_xy_t)
    xi = _smallest_angle_diff(ang_zy_s, ang_zy_t)

    print("\n--- TIP-LINE ANGLE CHANGES (local wing frame) ---")
    print(f"Angle (y–x plane) straight: {ang_xy_s:.3f} deg")
    print(f"Angle (y–x plane) turn    : {ang_xy_t:.3f} deg")
    print(f"Δ zeta (wrapped)          : {zeta:.3f} deg\n")

    print(f"Angle (z–y plane) straight: {ang_zy_s:.3f} deg")
    print(f"Angle (z–y plane) turn    : {ang_zy_t:.3f} deg")
    print(f"Δ xi   (wrapped)          : {xi:.3f} deg\n")

    return zeta, xi

# -------------------------
# Example usage:
# -------------------------

compute_zeta_xi(
    "static_test_output/straight_flight_reelout_frame_7182.csv",
    "static_test_output/left_turn_reelout_frame_7362.csv"
)
