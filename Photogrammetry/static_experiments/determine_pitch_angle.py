import numpy as np
from plot_static_results import _prep_dataset_from_csv


def _fit_pitch_angle_yz(points):
    """
    Fit z = a*y + b in camera-frame y–z and return angle = arctan(a) in degrees.
    """
    if points is None or len(points) < 2:
        return None

    P = np.asarray(points)
    y = P[:, 1]
    z = P[:, 2]

    # Linear regression for the strut direction in y–z
    A = np.vstack([y, np.ones_like(y)]).T
    a, b = np.linalg.lstsq(A, z, rcond=None)[0]

    # Pitch angle relative to horizontal axis in y–z
    angle_deg = np.degrees(np.arctan(a))
    return angle_deg


def compute_wing_pitch(csv_path, *,
                       expected_struts=8,
                       endpoints_k=10,
                       jump_factor=2.5,
                       tip_policy="lowestZ",
                       global_up=(0, 0, 1),
                       force_flip_x=False,
                       transform_to_local=False):
    """
    Computes wing pitch angle from the y–z projection of all struts in camera frame.

    RETURNS:
        angles_per_strut : dict(strut_index → pitch angle [deg])
        avg_pitch_deg    : average pitch over all valid struts
    """

    # Load raw data (no transformation!)
    pts_local, _ = _prep_dataset_from_csv(
        csv_path,
        endpoints_k=endpoints_k,
        jump_factor=jump_factor,
        tip_policy=tip_policy,
        global_up=global_up,
        force_flip_x=force_flip_x,
        transform_to_local=transform_to_local,  # <-- remains False for camera frame!
    )

    angles = {}
    for i in range(expected_struts):
        key = f"strut{i}"
        P = pts_local.get(key)
        angle = _fit_pitch_angle_yz(P)
        if angle is not None:
            angles[i] = angle

    if len(angles) == 0:
        raise ValueError("No valid strut pitch angles found.")

    avg_pitch = np.mean(list(angles.values()))
    return angles, avg_pitch

if __name__ == "__main__":
    angles, pitch_avg = compute_wing_pitch("static_test_output/depowered_state_reelin_frame_17611.csv")
    print(angles)
    print(f"The avg pitch is: {pitch_avg}")