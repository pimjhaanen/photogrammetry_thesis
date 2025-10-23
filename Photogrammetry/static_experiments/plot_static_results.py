# plot_static_shape.py
"""
Static 3D plotting for LE + struts with a simple legend:
- Dot in legend = 'Inflatable structure' (LE + strut points)
- Cross in legend = 'Canopy' (CAN points)

Reference frame (built in world, then EVERYTHING is transformed & plotted in this frame):
- ẑ: robust normal of the plane spanned by struts 3 & 4 (SVD on all strut3/4 points)
- ŷ: LE tip-to-tip vector projected into that plane (follows span)
- x̂: ŷ × ẑ  (right-handed)
- Origin O: midpoint between strut3 & strut4 centroids

All points (and the LE path) are transformed into this frame and plotted.
Optionally write transformed coords to CSV via save_transformed_csv.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Try to use your project util for line fit; otherwise fallback
_fit_line_impl = None
try:
    import Photogrammetry.kite_shape_reconstruction_utils as ks
    _fit_line_impl = ks.fit_line_3d
except Exception:
    pass


# --------------------------- distances & helpers ---------------------------

def _pairwise_dist(P: np.ndarray) -> np.ndarray:
    diffs = P[:, None, :] - P[None, :, :]
    return np.linalg.norm(diffs, axis=2)

def _path_len(P: np.ndarray) -> float:
    if P is None or P.shape[0] < 2:
        return 0.0
    diffs = P[1:, :] - P[:-1, :]
    return float(np.linalg.norm(diffs, axis=1).sum())

def _two_opt_locked(order: List[int], D: np.ndarray, s: int, t: int) -> List[int]:
    n = len(order)
    if n < 4:
        return order
    improved = True
    while improved:
        improved = False
        for i in range(0, n - 3):
            for j in range(i + 2, n - 1):
                a, b = order[i], order[i + 1]
                c, d = order[j], order[j + 1]
                # keep endpoints at ends
                if (i == 0 and a == s) or (j + 1 == n - 1 and d == t):
                    pass
                old = D[a, b] + D[c, d]
                new = D[a, c] + D[b, d]
                if new + 1e-12 < old:
                    order[i + 1 : j + 1] = reversed(order[i + 1 : j + 1])
                    improved = True
    return order


# --------------------------- LE anchored solvers ---------------------------

def _le_path_fixed_endpoints_heur(points_3d: np.ndarray, s: int, t: int,
                                  jump_factor: float = 2.5) -> np.ndarray:
    """Fast anchored path: NN grow (avoid t until last) + locked 2-opt, with jump limiter."""
    P = np.asarray(points_3d, float); n = P.shape[0]
    D = _pairwise_dist(P)

    # jump limiter based on median NN scale
    nn = np.partition(D + np.eye(n) * 1e9, 1, axis=1)[:, 1]
    thresh = float(np.median(nn) * jump_factor)
    big = 1e6
    Dlim = D.copy()
    Dlim[Dlim > thresh] = big

    used = np.zeros(n, dtype=bool); used[s] = True; used[t] = False
    order = [s]

    while len(order) < n - 1:
        last = order[-1]
        cand = np.where(~used)[0]
        cand = cand[cand != t]             # don't take t until the end
        nxt = int(cand[np.argmin(Dlim[last, cand])])
        order.append(nxt); used[nxt] = True

    order.append(t)
    order = _two_opt_locked(order, Dlim, s, t)
    return P[np.array(order)]


def _le_path_fixed_endpoints_exact(points_3d: np.ndarray, s: int, t: int,
                                   jump_factor: float = 2.5) -> np.ndarray:
    """Exact DP anchored to endpoints (reindexes s->0, t->n-1). With jump limiter."""
    P = np.asarray(points_3d, float); n = P.shape[0]
    D = _pairwise_dist(P).astype(np.float32)

    # jump limiter
    nn = np.partition(D + np.eye(n) * 1e9, 1, axis=1)[:, 1]
    thresh = float(np.median(nn) * jump_factor)
    big = 1e6
    D[D > thresh] = big

    # reindex so s->0, t->n-1
    idx = [s] + [k for k in range(n) if k not in (s, t)] + [t]
    D = D[np.ix_(idx, idx)]
    m = n - 2                          # middle nodes
    ALL = 1 << m

    dp = np.full((ALL, m + 1), np.inf, dtype=np.float32)   # columns: 0=start, 1..m=middle
    parent = np.full((ALL, m + 1), -1, dtype=np.int16)
    dp[0, 0] = 0.0

    for mask in range(ALL):
        for j in range(m + 1):
            val = dp[mask, j]
            if not np.isfinite(val):
                continue
            for k in range(1, m + 1):
                bit = 1 << (k - 1)
                if mask & bit:
                    continue
                newm = mask | bit
                cost = val + D[j, k]
                if cost < dp[newm, k]:
                    dp[newm, k] = cost
                    parent[newm, k] = j

    full = ALL - 1
    best_last = -1
    best_cost = np.inf
    for j in range(m + 1):
        c = dp[full, j] + D[j, m + 1]
        if c < best_cost:
            best_cost = c; best_last = j

    # backtrack
    order_re = [m + 1, best_last]
    mask = full; j = best_last
    while j != 0:
        pj = int(parent[mask, j]); order_re.append(pj)
        mask ^= (1 << (j - 1)); j = pj
    order_re.append(0)
    order_re = order_re[::-1]                     # 0...m+1
    order_orig = [idx[r] for r in order_re]
    return P[np.array(order_orig)]


def _best_le_path_among_lowZ(points_3d: np.ndarray, *,
                             endpoints_k: int = 10,
                             use_exact_if_small: bool = False,
                             exact_fixed_limit: int = 24,
                             jump_factor: float = 2.5) -> np.ndarray:
    """
    Compute LE path by anchoring to the best pair among the K lowest-Z points.
    """
    P = np.asarray(points_3d, float); n = P.shape[0]
    if n <= 1:
        return P.copy()

    k = max(2, min(endpoints_k, n))
    cand = np.argsort(P[:, 2])[:k]   # lowest Z
    best_path = None
    best_cost = np.inf
    for a in range(len(cand)):
        for b in range(a + 1, len(cand)):
            s, t = int(cand[a]), int(cand[b])
            path = _le_path_fixed_endpoints_heur(P, s, t, jump_factor=jump_factor)
            cost = _path_len(path)
            if cost < best_cost:
                best_cost = cost
                best_path = path

    if use_exact_if_small and n <= exact_fixed_limit and best_path is not None:
        first_pt, last_pt = best_path[0], best_path[-1]
        s = int(np.argmin(np.linalg.norm(P - first_pt, axis=1)))
        t = int(np.argmin(np.linalg.norm(P - last_pt, axis=1)))
        best_path = _le_path_fixed_endpoints_exact(P, s, t, jump_factor=jump_factor)

    return best_path if best_path is not None else P.copy()


# --------------------------- line fit (struts) ---------------------------

def fit_line_3d(points: np.ndarray) -> Optional[np.ndarray]:
    """Least-squares line through points; returns a polyline along the segment."""
    if points is None or points.shape[0] < 2:
        return None
    if _fit_line_impl is not None:
        return _fit_line_impl(points)
    # PCA fallback
    p0 = points.mean(axis=0)
    _, _, Vt = np.linalg.svd(points - p0, full_matrices=False)
    d = Vt[0]
    t = (points - p0) @ d
    tmin, tmax = float(t.min()), float(t.max())
    if np.isclose(tmax - tmin, 0.0):
        tmin, tmax = -0.5, 0.5
    ts = np.linspace(tmin, tmax, 100)
    return p0 + np.outer(ts, d)

def _line_center_and_dir(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (center, unit direction) via PCA; robust to how fit_line_3d renders."""
    P = np.asarray(points, float)
    c = P.mean(axis=0)
    _, _, Vt = np.linalg.svd(P - c, full_matrices=False)
    d = Vt[0]
    d = d / (np.linalg.norm(d) + 1e-12)
    return c, d


# --------------------------- reference frame ---------------------------

def _best_fit_plane_normal(P: np.ndarray) -> np.ndarray:
    """Unit normal of best-fit plane through points (SVD)."""
    Q = np.asarray(P, float)
    c = Q.mean(axis=0)
    _, _, Vt = np.linalg.svd(Q - c, full_matrices=False)
    n = Vt[-1]                      # least-variance direction = plane normal
    return n / (np.linalg.norm(n) + 1e-12)

def _compute_ref_frame(LE_path_world: Optional[np.ndarray],
                       P3_world: Optional[np.ndarray],
                       P4_world: Optional[np.ndarray]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Build frame from strut3/4 and LE path (WORLD coords):
        ẑ = best-fit plane normal of all strut3/4 points
        ŷ = LE tip-to-tip vector projected to that plane (fallback: c4-c3 projected)
        x̂ = ŷ × ẑ (right-handed)
        O  = 0.5*(centroid(strut3) + centroid(strut4))
    """
    if P3_world is None or P4_world is None or P3_world.shape[0] < 2 or P4_world.shape[0] < 2:
        return None, None

    # robust plane normal from all points on the two struts
    all_pts = np.vstack([P3_world, P4_world])
    z_hat = _best_fit_plane_normal(all_pts)

    # centroids + mid-origin
    c3 = P3_world.mean(axis=0); c4 = P4_world.mean(axis=0)
    origin = 0.5 * (c3 + c4)

    # y from LE tip-to-tip, projected to the plane
    y_raw = None
    if LE_path_world is not None and LE_path_world.shape[0] >= 2:
        span_vec = LE_path_world[-1] - LE_path_world[0]
        y_raw = span_vec - np.dot(span_vec, z_hat) * z_hat

    # fallback if LE projection degenerate
    if y_raw is None or np.linalg.norm(y_raw) < 1e-9:
        span_mid = c4 - c3
        y_raw = span_mid - np.dot(span_mid, z_hat) * z_hat

    y_hat = y_raw / (np.linalg.norm(y_raw) + 1e-12)

    # x completes right-handed (x × y = z) => x = y × z
    x_hat = np.cross(y_hat, z_hat)
    x_hat = x_hat / (np.linalg.norm(x_hat) + 1e-12)

    # re-orthonormalize (numerical hygiene)
    z_hat = np.cross(x_hat, y_hat); z_hat /= (np.linalg.norm(z_hat) + 1e-12)
    y_hat = np.cross(z_hat, x_hat); y_hat /= (np.linalg.norm(y_hat) + 1e-12)

    R_axes = np.column_stack([x_hat, y_hat, z_hat])  # columns are unit axes
    return R_axes, origin

def _transform_points(points_by_group_3d: Dict[str, np.ndarray],
                      origin: np.ndarray, R_axes: np.ndarray) -> Dict[str, np.ndarray]:
    """World → local: (P - O) @ R where R columns are [x̂ ŷ ẑ]."""
    out = {}
    for k, P in points_by_group_3d.items():
        if P is None or not np.size(P):
            continue
        out[k] = (P - origin) @ R_axes
    return out

def _transformed_to_df(points_local: Dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for g, P in points_local.items():
        for i, p in enumerate(P):
            rows.append({"group": g, "idx_in_group": i,
                         "x_local": float(p[0]), "y_local": float(p[1]), "z_local": float(p[2])})
    return pd.DataFrame(rows, columns=["group", "idx_in_group", "x_local", "y_local", "z_local"])


# --------------------------- plotting utils ---------------------------

def _set_equal_aspect(ax):
    xs = ax.get_xlim3d(); ys = ax.get_ylim3d(); zs = ax.get_zlim3d()
    cx = (xs[0]+xs[1])/2; cy = (ys[0]+ys[1])/2; cz = (zs[0]+zs[1])/2
    r = max(xs[1]-xs[0], ys[1]-ys[0], zs[1]-zs[0]) * 0.5
    ax.set_xlim(cx-r, cx+r); ax.set_ylim(cy-r, cy+r); ax.set_zlim(cz-r, cz+r)

def _to_groups_dict(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for g, sub in df.groupby("group"):
        out[str(g)] = sub[["x", "y", "z"]].to_numpy(float)
    return out


# ------------------------------- API -------------------------------

def plot_static_shape_from_csv(
    csv_path: str,
    *,
    span_m: Optional[float] = None,            # UWB span (optional)
    save_path: Optional[str] = None,
    show: bool = True,
    sciview: bool = True,
    endpoints_k: int = 10,
    use_exact_if_small: bool = False,
    exact_fixed_limit: int = 24,
    jump_factor: float = 2.5,
    save_transformed_csv: Optional[str] = None,
    figsize: Tuple[int, int] = (7, 6),
    elev: float = 20,
    azim: float = -60,
):
    df = pd.read_csv(csv_path)
    req = {"group", "idx_in_group", "x", "y", "z"}
    miss = req - set(df.columns)
    if miss:
        raise ValueError(f"CSV missing columns: {miss}")
    groups = _to_groups_dict(df)
    return plot_static_shape(
        groups,
        span_m=span_m,
        save_path=save_path,
        show=show,
        sciview=sciview,
        endpoints_k=endpoints_k,
        use_exact_if_small=use_exact_if_small,
        exact_fixed_limit=exact_fixed_limit,
        jump_factor=jump_factor,
        save_transformed_csv=save_transformed_csv,
        figsize=figsize,
        elev=elev,
        azim=azim,
        title=f"Static Reconstruction — {csv_path}",
    )


def plot_static_shape(
    points_by_group_3d_world: Dict[str, np.ndarray],
    *,
    span_m: Optional[float] = None,            # UWB span (optional)
    save_path: Optional[str] = None,
    show: bool = True,
    sciview: bool = True,
    endpoints_k: int = 10,
    use_exact_if_small: bool = False,
    exact_fixed_limit: int = 24,
    jump_factor: float = 2.5,
    save_transformed_csv: Optional[str] = None,
    figsize: Tuple[int, int] = (7, 6),
    elev: float = 20,
    azim: float = -60,
    title: Optional[str] = "Static Reconstruction",
):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    # ---------- 1) Build LE path in WORLD (for robust endpoint selection) ----------
    LE_world = points_by_group_3d_world.get("LE")
    le_path_world = None
    if LE_world is not None and LE_world.size:
        le_path_world = _best_le_path_among_lowZ(
            LE_world,
            endpoints_k=endpoints_k,
            use_exact_if_small=use_exact_if_small,
            exact_fixed_limit=exact_fixed_limit,
            jump_factor=jump_factor,
        )

    # ---------- 2) Compute reference frame in WORLD, then transform EVERYTHING ----------
    P3_world = points_by_group_3d_world.get("strut3")
    P4_world = points_by_group_3d_world.get("strut4")
    R_axes, origin = _compute_ref_frame(le_path_world, P3_world, P4_world)
    if R_axes is None or origin is None:
        raise RuntimeError("Not enough points on strut3/strut4 to define reference frame.")

    # Transform groups
    points_by_group_3d = _transform_points(points_by_group_3d_world, origin, R_axes)

    # Transform LE path
    le_path = None
    if le_path_world is not None:
        le_path = (le_path_world - origin) @ R_axes

    # Photogrammetry span (local frame; invariant to rigid transform)
    phot_span_m = None
    if le_path is not None and le_path.shape[0] >= 2:
        phot_span_m = float(np.linalg.norm(le_path[-1] - le_path[0]))

    # Optional: save transformed coordinates
    if save_transformed_csv:
        df_local = _transformed_to_df(points_by_group_3d)
        pd.DataFrame(df_local).to_csv(save_transformed_csv, index=False)

    # ---------- 3) Plot in LOCAL FRAME ----------
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    # LE in local
    LE = points_by_group_3d.get("LE")
    if LE is not None and LE.size:
        ax.scatter(LE[:, 0], LE[:, 1], LE[:, 2], s=18)  # not in legend
        if le_path is not None:
            ax.plot(le_path[:, 0], le_path[:, 1], le_path[:, 2], linewidth=1.6)  # not in legend

    # Struts (local)
    for i in range(8):
        key = f"strut{i}"
        P = points_by_group_3d.get(key)
        if P is not None and P.shape[0] >= 1:
            ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=12)  # not in legend
            if P.shape[0] >= 2:
                line = fit_line_3d(P)
                if line is not None:
                    ax.plot(line[:, 0], line[:, 1], line[:, 2], linewidth=1.25)  # not in legend

    # CAN (local)
    CAN = points_by_group_3d.get("CAN")
    if CAN is not None and CAN.size:
        ax.scatter(CAN[:, 0], CAN[:, 1], CAN[:, 2], s=42, marker="x")  # not in legend

    # Draw canonical axes at origin (local frame)
    # choose axis length
    if phot_span_m is not None:
        L = 0.15 * phot_span_m
    elif span_m is not None:
        L = 0.15 * span_m
    else:
        allP = np.vstack([v for v in points_by_group_3d.values() if v is not None and v.size])
        rng = np.linalg.norm(allP.max(axis=0) - allP.min(axis=0))
        L = 0.1 * rng
    ax.quiver(0, 0, 0, L, 0, 0, color='r', arrow_length_ratio=0.15)
    ax.quiver(0, 0, 0, 0, L, 0, color='g', arrow_length_ratio=0.15)
    ax.quiver(0, 0, 0, 0, 0, L, color='b', arrow_length_ratio=0.15)
    ax.text(L*1.1, 0, 0, "X", color='r', fontsize=10)
    ax.text(0, L*1.1, 0, "Y", color='g', fontsize=10)
    ax.text(0, 0, L*1.1, "Z", color='b', fontsize=10)

    # Labels / title (local frame)
    ax.set_xlabel("x_local [m]"); ax.set_ylabel("y_local [m]"); ax.set_zlabel("z_local [m]")
    if title:
        ax.set_title(title + " (local frame)")

    # Spans text (top-right)
    ypos = 0.98
    if span_m is not None:
        ax.text2D(0.98, ypos, f"UWB span: {span_m:.3f} m",
                  transform=ax.transAxes, ha="right", va="top", fontsize=11,
                  bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.7, linewidth=0.0))
        ypos -= 0.05
    if phot_span_m is not None:
        ax.text2D(0.98, ypos, f"Photogrammetry span: {phot_span_m:.3f} m",
                  transform=ax.transAxes, ha="right", va="top", fontsize=11,
                  bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.7, linewidth=0.0))

    # Minimal legend (points vs crosses)
    legend_handles = [
        Line2D([0], [0], marker='o', linestyle='None', markersize=7, label='Inflatable structure'),
        Line2D([0], [0], marker='x', linestyle='None', markersize=8, label='Canopy'),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=10)

    # View & aspect
    ax.view_init(elev=elev, azim=azim)
    try:
        _set_equal_aspect(ax)
    except Exception:
        pass

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=180)
    if show:
        if sciview:
            plt.show(block=False); plt.pause(0.001)
        else:
            plt.show()
    else:
        plt.close(fig)

    # Return extras, including transform + transformed points
    extras = {
        "phot_span_m": phot_span_m,
        "uwb_span_m": span_m,
        "ref_R": R_axes,
        "ref_origin": origin,
        "points_local": points_by_group_3d,
    }
    return fig, ax, extras


# ------------------------------- CLI example -------------------------------
if __name__ == "__main__":
    plot_static_shape_from_csv(
        "static_test_output/P1_S.csv",
        span_m=5.042,
        endpoints_k=10,
        use_exact_if_small=False,
        exact_fixed_limit=24,
        save_transformed_csv= None,
        sciview=False,
    )
