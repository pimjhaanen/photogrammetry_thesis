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

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    csvs = [
        "static_test_output/powered_state_reelin_frame_17372.csv",
        "static_test_output/depowered_state_reelin_frame_17701.csv",
        "static_test_output/left_turn_reelout_frame_7362.csv",
    ]

    labels = [
        "Powered, straight",
        "Depowered, straight",
        "Powered, left turn",
    ]

    colors = ["C0","C1","C2"]

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
