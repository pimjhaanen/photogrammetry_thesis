import numpy as np
import matplotlib.pyplot as plt

from plot_static_results import _prep_dataset_from_csv

# ============================================================
# 1. BASIC UTILITIES
# ============================================================

def _pca_dir(points):
    if points is None or len(points) < 2:
        return None
    P = np.asarray(points)
    c = P.mean(axis=0)
    _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    d = vt[0]
    n = np.linalg.norm(d)
    return d / n if n > 0 else None

def _span_mapping_from_le(le_path):
    ys = le_path[:, 1]
    ymin, ymax = float(np.min(ys)), float(np.max(ys))
    y_center = 0.5 * (ymin + ymax)
    span_y = ymax - ymin
    return y_center, span_y

def _compute_s_from_y(y, y_center, span_y):
    return 2.0 * (y - y_center) / span_y

# ============================================================
# 2. LE-BASED TWIST HELPERS
# ============================================================

def _le_tangent_yz_at(le_path, y_query):
    if le_path is None or le_path.shape[0] < 2:
        return None

    le = np.asarray(le_path, float)
    ys = le[:, 1]
    zs = le[:, 2]
    order = np.argsort(ys)
    ys = ys[order]
    zs = zs[order]

    if y_query <= ys[0]:
        y0, y1 = ys[0], ys[1]
        z0, z1 = zs[0], zs[1]
    elif y_query >= ys[-1]:
        y0, y1 = ys[-2], ys[-1]
        z0, z1 = zs[-2], zs[-1]
    else:
        j = int(np.searchsorted(ys, y_query))
        y0, y1 = ys[j - 1], ys[j]
        z0, z1 = zs[j - 1], zs[j]

    v = np.array([0.0, y1 - y0, z1 - z0], float)
    n = np.linalg.norm(v)
    return None if n == 0 else v / n

# ============================================================
# 2B. NEW: UNWRAP LARGE TWIST JUMPS (>50°)
# ============================================================

def _unwrap_twist_along_span(twist_deg, thresh=50.0):
    """
    Remove ±360° jumps when twist jumps more than 'thresh' degrees.
    """
    t = np.asarray(twist_deg, float).copy()
    for i in range(1, len(t)):
        delta = t[i] - t[i-1]
        if delta > thresh:
            t[i:] -= 360.0
        elif delta < -thresh:
            t[i:] += 360.0
    return t

# ============================================================
# 3. GEOMETRIC TWIST (LE-relative)
# ============================================================

def compute_geometric_twist(
        csv_path,
        *,
        expected_struts=8,
        endpoints_k=10,
        jump_factor=2.5,
        tip_policy="lowestZ",
        global_up=(0,0,1),
        force_flip_x=False,
        transform_to_local=True
):
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
        raise ValueError(f"[Twist] LE path missing in {csv_path}")

    y_center, span_y = _span_mapping_from_le(le_path)

    s_list = []
    twist_list = []

    q = np.array([1.0, 0.0, 0.0])

    for i in range(expected_struts):
        P = pts_local.get(f"strut{i}")
        if P is None or len(P) < 2:
            continue
        P = np.asarray(P, float)
        y_mean = float(P[:,1].mean())
        s_val  = _compute_s_from_y(y_mean, y_center, span_y)

        d = _pca_dir(P)
        if d is None:
            continue
        d = d / np.linalg.norm(d)

        n = _le_tangent_yz_at(le_path, y_mean)
        if n is None:
            continue

        d_proj = d - np.dot(d, n)*n
        if np.linalg.norm(d_proj) < 1e-9:
            continue
        d_proj /= np.linalg.norm(d_proj)

        q_proj = q - np.dot(q, n)*n
        if np.linalg.norm(q_proj) < 1e-9:
            continue
        q_proj /= np.linalg.norm(q_proj)

        cross = np.cross(q_proj, n)
        num   = np.dot(cross, d_proj)
        den   = np.dot(q_proj, d_proj)
        twist = np.degrees(np.arctan2(num, den))

        s_list.append(s_val)
        twist_list.append(twist)

    if not s_list:
        raise ValueError(f"[Twist] No valid struts found in {csv_path}")

    s_arr = np.asarray(s_list)
    twist_arr = np.asarray(twist_list)

    order = np.argsort(s_arr)
    s_sorted = s_arr[order]
    twist_sorted = twist_arr[order]

    # 🔥 APPLY YOUR 50° UNWRAP RULE HERE
    twist_sorted = _unwrap_twist_along_span(twist_sorted, thresh=50.0)

    # center twist at s=0
    idx0 = np.where(np.isclose(s_sorted, 0))[0]
    if len(idx0):
        twist0 = twist_sorted[idx0[0]]
    else:
        r = np.searchsorted(s_sorted, 0)
        if r == 0:
            twist0 = twist_sorted[0]
        elif r == len(s_sorted):
            twist0 = twist_sorted[-1]
        else:
            l = r - 1
            tL, tR = twist_sorted[l], twist_sorted[r]
            sL, sR = s_sorted[l], s_sorted[r]
            twist0 = tL + (0 - sL)*(tR - tL)/(sR - sL)

    return s_sorted, twist_sorted - twist0

# ============================================================
# 4. BILLOWING & PLOTTING (unchanged)
# ============================================================

# [UNCHANGED: all your billowing + plotting code continues here...]


# ============================================================
# 4. BILLOWING & PLOTTING (unchanged)
# ============================================================

def _te_point_world(P):
    if P is None or len(P) == 0:
        return None
    idx = np.argmin(P[:, 1])
    return P[idx]

def _tips_world_from_LE(LE):
    LE = np.asarray(LE)
    x = LE[:, 0]
    y = LE[:, 1]

    tipL = None
    L = LE[x < 0]
    if L.size:
        tipL = L[np.argmin(L[:, 1])]
    else:
        L2 = LE[y < 0]
        if L2.size:
            tipL = L2[np.argmin(L2[:, 1])]

    tipR = None
    R = LE[x > 0]
    if R.size:
        tipR = R[np.argmin(R[:, 1])]
    else:
        R2 = LE[y > 0]
        if R2.size:
            tipR = R2[np.argmin(R2[:, 1])]

    return tipL, tipR

def compute_billowing_segments(csv_path, *,
        exclude_struts=None,
        expected_struts=8,
        endpoints_k=10,
        jump_factor=2.5,
        tip_policy="lowestZ",
        global_up=(0,0,1),
        force_flip_x=False,
        transform_to_local=False):

    if exclude_struts is None:
        exclude_struts = []

    pts_world, le_path_world = _prep_dataset_from_csv(
        csv_path,
        endpoints_k=endpoints_k,
        jump_factor=jump_factor,
        tip_policy=tip_policy,
        global_up=global_up,
        force_flip_x=force_flip_x,
        transform_to_local=transform_to_local,
    )

    LE = pts_world.get("LE")
    if LE is None:
        raise ValueError("LE missing")

    tipL, tipR = _tips_world_from_LE(LE)

    strut_te = {}
    for i in range(expected_struts):
        P = pts_world.get(f"strut{i}")
        strut_te[i] = _te_point_world(P) if P is not None else None

    sort = []
    for i in range(expected_struts):
        Pte = strut_te[i]
        x = Pte[0] if Pte is not None else np.nan
        sort.append((np.isnan(x), x, i))
    sort = sorted(sort)
    sorted_ids = [i for _,_,i in sort]

    chain = ["tipL"] + [f"strut{i}" for i in sorted_ids] + ["tipR"]
    te_map = {"tipL": tipL, "tipR": tipR}
    for i in range(expected_struts):
        te_map[f"strut{i}"] = strut_te[i]

    seg_idx = []
    dist = []

    for j,(a,b) in enumerate(zip(chain[:-1], chain[1:])):
        bad = False
        if a.startswith("strut") and int(a[5:]) in exclude_struts:
            bad = True
        if b.startswith("strut") and int(b[5:]) in exclude_struts:
            bad = True

        A = te_map[a]
        B = te_map[b]
        if bad or A is None or B is None:
            seg_idx.append(j)
            dist.append(np.nan)
        else:
            seg_idx.append(j)
            dist.append(np.linalg.norm(B-A))

    return np.asarray(seg_idx), np.asarray(dist)


def percent_change_with_uncertainty(billow_A, billow_B, sigma_L):
    """
    Compute percentage change in mean billowing from case A -> B
    and its uncertainty, assuming:
      - each segment length has standard uncertainty sigma_L (1σ, in meters)
      - segments are independent
      - mean is taken over segments where both A and B are valid (non-NaN)

    Returns:
        pct_change  : 100 * (mean_B / mean_A - 1)   [in %]
        sigma_pct   : 1σ uncertainty on pct_change  [in %]
        mean_A, mean_B, N_used : for reference
    """
    billow_A = np.asarray(billow_A, float)
    billow_B = np.asarray(billow_B, float)

    # Only use segments where both cases have valid values
    valid = ~np.isnan(billow_A) & ~np.isnan(billow_B)
    if not np.any(valid):
        raise ValueError("No overlapping valid segments between A and B.")

    A = billow_A[valid]
    B = billow_B[valid]
    N = len(A)

    mean_A = A.mean()
    mean_B = B.mean()

    # uncertainty of mean billowing in each case
    sigma_mean_A = sigma_L / np.sqrt(N)
    sigma_mean_B = sigma_L / np.sqrt(N)

    # percentage change A -> B
    pct_change = 100.0 * (mean_B / mean_A - 1.0)

    # propagate uncertainty
    dP_dA = -100.0 * (mean_B / (mean_A**2))
    dP_dB =  100.0 / mean_A

    sigma_pct = np.sqrt(
        (dP_dA**2) * (sigma_mean_A**2) +
        (dP_dB**2) * (sigma_mean_B**2)
    )

    return pct_change, sigma_pct, mean_A, mean_B, N

# ============================================================
# PLOTTING: TWIST, BILLOWING, BARS
# ============================================================

def plot_geometric_aoa_relative(csvs, labels, colors,
                                external_curves, external_labels):
    plt.figure(figsize=(10, 5))
    for csv, lab, col in zip(csvs, labels, colors):
        s, twist = compute_geometric_twist(csv)

        # --- FIX: Flip twist for FIRST FILE ONLY ---
        if csv == csvs[0]:  # powered file
            twist = -twist  # invert signs
        # -------------------------------------------

        plt.plot(s, twist, marker="o", label=lab, color=col)

    for (s_ext, twist_ext), lab in zip(external_curves, external_labels):
        plt.plot(s_ext, twist_ext, "r--", marker="o", label=lab)

    plt.xlabel(r"$2y/b$")
    plt.ylabel("Twist angle (deg)")
    plt.title("Spanwise Geometric Twist (LE-relative)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_billowing(csvs, labels, colors,
                   exclude_struts_per_file,
                   external_curves, external_labels):

    x_labels = ["$L_a$", "$L_b$", "$L_c$", "$L_d$", "$L_e$",
                "$L_f$", "$L_g$", "$L_h$", "$L_i$"]
    x_labels_flip = x_labels[::-1]

    plt.figure(figsize=(10,5))
    for csv, lab, col in zip(csvs, labels, colors):
        excl = exclude_struts_per_file.get(csv,[])
        idx, billow = compute_billowing_segments(csv, exclude_struts=excl)
        y = billow[::-1]
        x = np.array(x_labels_flip[:len(y)], dtype=object)
        print(lab)
        for i in reversed(range(len(y))):
            print(np.round(y[i], 3))

        plt.plot(x, y, marker="o", color=col, label=lab)

    for (_, cad), lab in zip(external_curves, external_labels):
        y = cad[::-1]
        x = np.array(x_labels_flip[:len(y)], dtype=object)
        plt.plot(x, y, "r--", marker="o", label=lab)

    plt.xlabel("Segments")
    plt.ylabel("Distance (m)")
    plt.title("Billowing")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_billowing_deviation_bar(csvs, labels,
                                 exclude_struts_per_file,
                                 cad_segments):
    results = []
    for csv in csvs:
        excl = exclude_struts_per_file.get(csv,[])
        idx,billow = compute_billowing_segments(csv, exclude_struts=excl)
        n = min(len(billow), len(cad_segments))
        billow = billow[:n]
        cad = cad_segments[:n]
        valid = ~np.isnan(billow) & ~np.isnan(cad)
        if np.any(valid):
            results.append(np.mean(billow[valid] / cad[valid]))
        else:
            results.append(np.nan)

    labels_plot = ["CAD"] + labels
    vals = np.array([1.0] + results)

    plt.figure(figsize=(10,5))
    colors = ["red"] + ["C0","C1","C2"][:len(labels)]
    plt.bar(labels_plot, (vals-1)*100, color=colors, width=0.4)
    plt.axhline(0,color="black")
    plt.ylabel("% difference")
    plt.title("Billowing deviation vs CAD")
    plt.grid()
    plt.tight_layout()
    plt.show()


def compute_centerstrut_tip_distances(csv_path):
    """
    Computes L_CS,T,R and L_CS,T,L for a given CSV file.

    Steps:
      1. Load pts_world and LE path
      2. For strut3 and strut4: find the TE point (max idx)
      3. Extract their x-coordinates
      4. Find LE point with closest x-coordinate
      5. Compute Euclidean distances:
           - LE[0]     <-> nearest-to-strut3  (Right)
           - LE[-1]    <-> nearest-to-strut4  (Left)
    """

    pts_world, le_path_world = _prep_dataset_from_csv(
        csv_path,
        endpoints_k=10,
        jump_factor=2.5,
        tip_policy="lowestZ",
        global_up=(0, 0, 1),
        force_flip_x=False,
        transform_to_local=False
    )

    LE = pts_world.get("LE")
    if LE is None:
        raise ValueError(f"LE missing in {csv_path}")

    LE = np.asarray(LE)

    # ---------------------------
    # Helper: max-index TE point
    # ---------------------------
    def max_index_point(P):
        """Return the point with highest index (row) in P."""
        if P is None or len(P) == 0:
            return None
        return P[-1]  # highest idx = last in list

    # Get strut3 TE (right side)
    P3 = pts_world.get("strut3")
    S3 = max_index_point(P3)
    if S3 is None:
        raise ValueError(f"strut3 missing in {csv_path}")

    # Get strut4 TE (left side)
    P4 = pts_world.get("strut4")
    S4 = max_index_point(P4)
    if S4 is None:
        raise ValueError(f"strut4 missing in {csv_path}")

    # Extract x-coordinates
    x3 = S3[0]  # right side
    x4 = S4[0]  # left side

    # ---------------------------
    # Find LE point closest in x
    # ---------------------------
    LE_x = LE[:, 0]

    idx_closest3 = np.argmin(np.abs(LE_x - x3))
    idx_closest4 = np.argmin(np.abs(LE_x - x4))

    LE_closest3 = LE[idx_closest3]
    LE_closest4 = LE[idx_closest4]

    # ---------------------------
    # Distances:
    #   Right side:  LE[0]   <-> LE_closest3
    #   Left side :  LE[-1]  <-> LE_closest4
    # ---------------------------
    L_CS_T_R = np.linalg.norm(LE[0] - LE_closest3)
    L_CS_T_L = np.linalg.norm(LE[-1] - LE_closest4)

    return L_CS_T_R, L_CS_T_L, idx_closest3, idx_closest4

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    csvs = [
        "static_test_output/P1_S.csv",
        "static_test_output/P1_PL1.csv",
        "static_test_output/P1_PL2.csv",
        "static_test_output/P1_TL1.csv",
        "static_test_output/P1_TL2.csv",
        "static_test_output/P2_S.csv",
        "static_test_output/P2_PL1.csv",
        "static_test_output/P2_PL2.csv",
        "static_test_output/P2_TL1.csv",
        "static_test_output/P2_TL2.csv",
    ]

    labels = [
        "P1+S",
        "P1+PL1",
        "P1+PL2",
        "P1+TL1",
        "P1+TL2",
        "P2+S",
        "P2+PL1",
        "P2+PL2",
        "P2+TL1",
        "P2+TL2",
    ]

    colors = ["C0","C1","C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]

    cad_s = np.array([-1,-0.875,-0.625,-0.325,0,0.325,0.625,0.875,1])
    cad_twist = np.array([2.3,1.9,3.0,3.5,3.7,3.5,3.0,1.9,2.3])
    cad_twist -= cad_twist[cad_s==0]

    cad_billow = np.array([
        0.889277, 1.328906, 1.313777, 1.311283,
        1.321442, 1.311283, 1.313777, 1.328906, 0.889277
    ])

    cad_idx = np.arange(9)

    exclude_struts_per_file = {}

    plot_geometric_aoa_relative(
        csvs, labels, colors,
        external_curves=[(cad_s,cad_twist)],
        external_labels=["CAD twist"]
    )

    plot_billowing(
        csvs, labels, colors,
        exclude_struts_per_file,
        external_curves=[(cad_idx,cad_billow)],
        external_labels=["CAD"]
    )

    plot_billowing_deviation_bar(
        csvs, labels,
        exclude_struts_per_file,
        cad_segments=cad_billow
    )

    # Example: compute billowing arrays for three cases
    idx_p, billow_p = compute_billowing_segments(
        "static_test_output/powered_state_reelin_frame_17372.csv"
    )
    idx_d, billow_d = compute_billowing_segments(
        "static_test_output/depowered_state_reelin_frame_17701.csv"
    )
    idx_t, billow_t = compute_billowing_segments(
        "static_test_output/left_turn_reelout_frame_7362.csv"
    )

    print("\n===== Center-Strut–Tip Distances =====")
    for csv, lab in zip(csvs, labels):
        try:
            R, L, idx3, idx4 = compute_centerstrut_tip_distances(csv)
            print(f"{lab}:  L_CS,T,R = {R:.4f} m   (LE idx={idx3}),   "
                  f"L_CS,T,L = {L:.4f} m   (LE idx={idx4})")
        except Exception as e:
            print(f"{lab}: ERROR -> {e}")

    # Choose an estimate for segment-length uncertainty (in meters)
    # e.g. if you think each segment length is known within ±0.02 m (1σ):
    sigma_L = 0.02

    # 1) powered straight -> depowered straight
    pct_pd, sig_pd, m_p, m_d, N_pd = percent_change_with_uncertainty(
        billow_p, billow_d, sigma_L
    )
    print(f"Powered → Depowered: Δ = {pct_pd:.2f}% ± {sig_pd:.2f}% "
          f"(means: {m_p:.3f} m → {m_d:.3f} m, N={N_pd})")

    # 2) powered straight -> powered left turn
    pct_pt, sig_pt, m_p2, m_t, N_pt = percent_change_with_uncertainty(
        billow_p, billow_t, sigma_L
    )
    print(f"Powered → Powered turn: Δ = {pct_pt:.2f}% ± {sig_pt:.2f}% "
          f"(means: {m_p2:.3f} m → {m_t:.3f} m, N={N_pt})")
