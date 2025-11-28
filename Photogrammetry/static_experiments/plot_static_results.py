# static_two_csv_x_flip_option_nodes_axes.py
from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Sequence
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_fit_line_impl = None
try:
    import Photogrammetry.kite_shape_reconstruction_utils as ks
    _fit_line_impl = ks.fit_line_3d
except Exception:
    pass


# --------------------------- utilities ---------------------------

def _to_groups_dict(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for g, sub in df.groupby("group"):
        out[str(g)] = sub[["x", "y", "z"]].to_numpy(float)
    return out

def _set_equal_aspect(ax):
    xs = ax.get_xlim3d(); ys = ax.get_ylim3d(); zs = ax.get_zlim3d()
    cx = (xs[0]+xs[1])/2; cy = (ys[0]+ys[1])/2; cz = (zs[0]+zs[1])/2
    r = max(xs[1]-xs[0], ys[1]-ys[0], zs[1]-zs[0]) * 0.5
    ax.set_xlim(cx-r, cx+r); ax.set_ylim(cy-r, cy+r); ax.set_zlim(cz-r, cz+r)

def fit_line_3d(points: np.ndarray) -> Optional[np.ndarray]:
    if points is None or points.shape[0] < 2:
        return None
    if _fit_line_impl is not None:
        return _fit_line_impl(points)
    p0 = points.mean(axis=0)
    _, _, Vt = np.linalg.svd(points - p0, full_matrices=False)
    d = Vt[0]
    t = (points - p0) @ d
    tmin, tmax = float(t.min()), float(t.max())
    if np.isclose(tmax - tmin, 0.0):
        tmin, tmax = -0.5, 0.5
    ts = np.linspace(tmin, tmax, 100)
    return p0 + np.outer(ts, d)

def _pca_dir(points: np.ndarray) -> Optional[np.ndarray]:
    if points is None or points.shape[0] < 2:
        return None
    P = np.asarray(points, float)
    c = P.mean(axis=0)
    _, _, Vt = np.linalg.svd(P - c, full_matrices=False)
    d = Vt[0]
    n = np.linalg.norm(d)
    return d / n if n > 0 else None

def _pca_span_length(points: np.ndarray) -> float:
    if points is None or points.shape[0] < 2:
        return 0.0
    P = np.asarray(points, float)
    c = P.mean(axis=0)
    _, _, Vt = np.linalg.svd(P - c, full_matrices=False)
    d = Vt[0]
    t = (P - c) @ d
    return float(t.max() - t.min())

def _best_fit_plane_normal(P: np.ndarray) -> np.ndarray:
    Q = np.asarray(P, float)
    c = Q.mean(axis=0)
    _, _, Vt = np.linalg.svd(Q - c, full_matrices=False)
    n = Vt[-1]
    return n / (np.linalg.norm(n) + 1e-12)

def _transform_points(points_by_group_3d: Dict[str, np.ndarray],
                      origin: np.ndarray, R_axes: np.ndarray) -> Dict[str, np.ndarray]:
    out = {}
    for k, P in points_by_group_3d.items():
        if P is None or not np.size(P):
            continue
        out[k] = (P - origin) @ R_axes
    return out


# --------------------------- LE path ---------------------------

def _pairwise_dist(P: np.ndarray) -> np.ndarray:
    diffs = P[:, None, :] - P[None, :, :]
    return np.linalg.norm(diffs, axis=2)

def _path_len(P: np.ndarray) -> float:
    if P is None or P.shape[0] < 2:
        return 0.0
    diffs = P[1:, :] - P[:-1, :]
    return float(np.linalg.norm(diffs, axis=1).sum())

def _two_opt_locked(order, D, s, t):
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
                old = D[a, b] + D[c, d]
                new = D[a, c] + D[b, d]
                if new + 1e-12 < old:
                    order[i + 1:j + 1] = reversed(order[i + 1:j + 1])
                    improved = True
    return order

def _le_path_fixed_endpoints_heur(points_3d: np.ndarray, s: int, t: int,
                                  jump_factor: float = 2.5) -> np.ndarray:
    P = np.asarray(points_3d, float); n = P.shape[0]
    D = _pairwise_dist(P)
    nn = np.partition(D + np.eye(n) * 1e9, 1, axis=1)[:, 1]
    thresh = float(np.median(nn) * jump_factor)
    big = 1e6
    Dlim = D.copy()
    Dlim[Dlim > thresh] = big
    used = np.zeros(n, dtype=bool); used[s] = True
    order = [s]
    while len(order) < n - 1:
        last = order[-1]
        cand = np.where(~used)[0]
        cand = cand[cand != t]
        nxt = int(cand[np.argmin(Dlim[last, cand])])
        order.append(nxt); used[nxt] = True
    order.append(t)
    order = _two_opt_locked(order, Dlim, s, t)
    return P[np.array(order)]

def _best_le_path_among_lowZ(points_3d: np.ndarray, *args, endpoints_k: int = 10, jump_factor: float = 2.5) -> np.ndarray:
    P = np.asarray(points_3d, float); n = P.shape[0]
    if n <= 1:
        return P.copy()
    k = max(2, min(endpoints_k, n))
    cand = np.argsort(P[:, 2])[:k]
    best_path, best_cost = None, np.inf
    for a in range(len(cand)):
        for b in range(a + 1, len(cand)):
            s, t = int(cand[a]), int(cand[b])
            path = _le_path_fixed_endpoints_heur(P, s, t, jump_factor=jump_factor)
            cost = _path_len(path)
            if cost < best_cost:
                best_cost, best_path = cost, path
    return best_path if best_path is not None else P.copy()

def _orient_le_by_world_z(le_path_world: np.ndarray, tip_policy: str) -> np.ndarray:
    if le_path_world is None or le_path_world.shape[0] < 2:
        return le_path_world
    if tip_policy == "lowestZ":
        if le_path_world[0, 2] > le_path_world[-1, 2]:
            le_path_world = le_path_world[::-1]
    elif tip_policy == "highestZ":
        if le_path_world[0, 2] < le_path_world[-1, 2]:
            le_path_world = le_path_world[::-1]
    return le_path_world


# --------------------------- Frame computation ---------------------------

def _compute_ref_frame_x_from_center_struts(
    LE_path_world: Optional[np.ndarray],
    P3_world: Optional[np.ndarray],
    P4_world: Optional[np.ndarray],
    *,
    global_up: np.ndarray = np.array([0.0, 0.0, 1.0], dtype=float),
    force_flip_x: bool = False,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if P3_world is None or P4_world is None or P3_world.shape[0] < 2 or P4_world.shape[0] < 2:
        return None, None

    all_pts = np.vstack([P3_world, P4_world])
    z_hat = _best_fit_plane_normal(all_pts)
    global_up = np.asarray(global_up, float) / (np.linalg.norm(global_up) + 1e-12)
    if np.dot(z_hat, global_up) < 0.0:
        z_hat = -z_hat

    d3 = _pca_dir(P3_world); d4 = _pca_dir(P4_world)
    if d3 is None or d4 is None:
        return None, None
    if np.dot(d3, d4) < 0.0:
        d4 = -d4
    x_raw = d3 + d4
    x_plane = x_raw - np.dot(x_raw, z_hat) * z_hat
    if np.linalg.norm(x_plane) < 1e-12:
        tmp = np.array([1.0, 0.0, 0.0], float)
        if abs(np.dot(tmp, z_hat)) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0], float)
        x_plane = tmp - np.dot(tmp, z_hat) * z_hat
    x_hat = x_plane / (np.linalg.norm(x_plane) + 1e-12)

    if LE_path_world is not None and LE_path_world.shape[0] >= 2:
        le_vec = LE_path_world[-1] - LE_path_world[0]
        le_proj = le_vec - np.dot(le_vec, z_hat) * z_hat
        if np.linalg.norm(le_proj) > 1e-12 and np.dot(x_hat, le_proj) < 0.0:
            x_hat = -x_hat

    if force_flip_x:
        x_hat = -x_hat

    y_hat = np.cross(z_hat, x_hat); y_hat /= (np.linalg.norm(y_hat) + 1e-12)
    z_hat = np.cross(x_hat, y_hat); z_hat /= (np.linalg.norm(z_hat) + 1e-12)
    y_hat = np.cross(z_hat, x_hat); y_hat /= (np.linalg.norm(y_hat) + 1e-12)

    origin = 0.5 * (P3_world.mean(axis=0) + P4_world.mean(axis=0))
    R_axes = np.column_stack([x_hat, y_hat, z_hat])
    return R_axes, origin


# --------------------------- Single kite plotting ---------------------------

def plot_static_shape(
    points_by_group_3d_world: Dict[str, np.ndarray],
    *,
    span_m: Optional[float] = None,
    strut_target_lengths: Optional[List[float]] = None,
    expected_nodes_per_strut: Optional[List[int]] = None,
    save_path: Optional[str] = None,
    show: bool = False,
    sciview: bool = True,
    endpoints_k: int = 10,
    jump_factor: float = 2.5,
    tip_policy: str = "lowestZ",
    global_up: Tuple[float, float, float] = (0.0, 0.0, 1.0),
    force_flip_x: bool = False,
    transform_to_local: bool = True,
    figsize: Tuple[int, int] = (7, 6),
    elev: float = 20,
    azim: float = -60,
    title: Optional[str] = "Static Reconstruction",
    ax: Optional[plt.Axes] = None,
    color: str = "C0",
    label: Optional[str] = None,
):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    LE_world = points_by_group_3d_world.get("LE")
    le_path_world = None
    if LE_world is not None and LE_world.size:
        le_path_world = _best_le_path_among_lowZ(LE_world, endpoints_k=endpoints_k, jump_factor=jump_factor)
        le_path_world = _orient_le_by_world_z(le_path_world, tip_policy=tip_policy)

    P3_world = points_by_group_3d_world.get("strut3")
    P4_world = points_by_group_3d_world.get("strut4")
    R_axes, origin = _compute_ref_frame_x_from_center_struts(
        le_path_world, P3_world, P4_world,
        global_up=np.asarray(global_up, float),
        force_flip_x=force_flip_x
    )

    # Either transform or keep world coordinates
    if transform_to_local and R_axes is not None and origin is not None:
        pts_local = _transform_points(points_by_group_3d_world, origin, R_axes)
        le_path = (le_path_world - origin) @ R_axes if le_path_world is not None else None
        coord_label = "local"
    else:
        pts_local = points_by_group_3d_world
        le_path = le_path_world
        coord_label = "world"

    # Metrics (printed for convenience)
    phot_span_m = float(np.linalg.norm(le_path[-1] - le_path[0])) if le_path is not None and le_path.shape[0] >= 2 else None
    strut_lengths_meas = {}
    node_counts = {}
    for i in range(8):
        key = f"strut{i}"
        P = pts_local.get(key)
        k = int(P.shape[0]) if (P is not None and P.ndim == 2) else 0
        node_counts[i] = k
        strut_lengths_meas[i] = _pca_span_length(P) if P is not None and P.shape[0] >= 2 else np.nan

    print("\n================  STATIC SHAPE METRICS  ================")
    if phot_span_m is not None:
        print(f"Span ({coord_label} frame): {phot_span_m:.4f} m")
    else:
        print("Span: not available.")

    rows = []
    for i in range(8):
        m = strut_lengths_meas.get(i, np.nan)
        tgt = (strut_target_lengths[i] if (strut_target_lengths is not None and i < len(strut_target_lengths)) else np.nan)
        k_obs = node_counts.get(i, 0)
        nodes_ok = True
        if expected_nodes_per_strut is not None and i < len(expected_nodes_per_strut):
            nodes_ok = (k_obs == int(expected_nodes_per_strut[i]))
        err = m - tgt if np.isfinite(m) and np.isfinite(tgt) and nodes_ok else np.nan
        rows.append({"strut": i, "nodes_obs": k_obs, "measured_m": m, "target_m": tgt, "error_m": err})
    tbl = pd.DataFrame(rows)
    print(tbl.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Plot
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
        created_fig = fig
    else:
        created_fig = None

    LE = pts_local.get("LE")
    handle = None
    if LE is not None and LE.size:
        ax.scatter(LE[:, 0], LE[:, 1], LE[:, 2], s=18, c=color)
        if le_path is not None:
            handle, = ax.plot(le_path[:, 0], le_path[:, 1], le_path[:, 2], linewidth=1.8, c=color, label=label)

    for i in range(8):
        key = f"strut{i}"
        P = pts_local.get(key)
        if P is not None and P.shape[0] >= 1:
            ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=12, c=color)
            if P.shape[0] >= 2:
                line = fit_line_3d(P)
                if line is not None:
                    ax.plot(line[:, 0], line[:, 1], line[:, 2], linewidth=1.2, c=color)

    ax.set_xlabel(f"x_{coord_label} [m]")
    ax.set_ylabel(f"y_{coord_label} [m]")
    ax.set_zlabel(f"z_{coord_label} [m]")
    if title:
        ax.set_title(title + ("" if transform_to_local else " (world frame)"))
    ax.view_init(elev=elev, azim=azim)
    try:
        _set_equal_aspect(ax)
    except Exception:
        pass

    if save_path and created_fig is not None:
        created_fig.savefig(save_path, dpi=180)
    if show and created_fig is not None:
        plt.tight_layout()
        if sciview:
            plt.show(block=False); plt.pause(0.001)
        else:
            plt.show()

    extras = {
        "phot_span_m": phot_span_m,
        "ref_R": R_axes,
        "ref_origin": origin,
        "points_used": pts_local,
        "le_path_used": le_path,
        "metrics_table": tbl,
        "legend_handle": handle,
    }
    return ax.get_figure(), ax, extras


# ---------- helper: version-safe figure ----------
def _new_figure(figsize):
    try:
        return plt.figure(figsize=figsize, layout="constrained")   # mpl ≥ 3.6
    except TypeError:
        try:
            return plt.figure(figsize=figsize, constrained_layout=True)  # mpl ≥ 3.1
        except TypeError:
            fig = plt.figure(figsize=figsize)
            try:
                fig.set_tight_layout(True)
            except Exception:
                pass
            return fig


# ---------- helper: 2D projection ----------
def _plot_projection_2d(ax, pts_local: Dict[str, np.ndarray], le_path: Optional[np.ndarray],
                        color: str, label: Optional[str], proj: Tuple[int, int],
                        draw_lines: bool = True,
                        ms_pts: float = 10.0,
                        lw_path: float = 1.8,
                        lw_strut: float = 1.2):
    """Project 3D points to 2D using indices in proj (e.g., (0,1) -> x,y)."""
    handle = None
    LE = pts_local.get("LE")
    if LE is not None and LE.size:
        ax.scatter(LE[:, proj[0]], LE[:, proj[1]], s=ms_pts, c=color)
        if le_path is not None:
            handle = ax.plot(le_path[:, proj[0]], le_path[:, proj[1]],
                             linewidth=lw_path, color=color, label=label)[0]

    for i in range(8):
        key = f"strut{i}"
        P = pts_local.get(key)
        if P is None or P.size == 0:
            continue
        ax.scatter(P[:, proj[0]], P[:, proj[1]], s=ms_pts*0.8, c=color)
        if draw_lines and P.shape[0] >= 2:
            line = fit_line_3d(P)
            if line is not None:
                ax.plot(line[:, proj[0]], line[:, proj[1]], linewidth=lw_strut, color=color)

    ax.set_aspect("equal", adjustable="datalim")
    return handle


# ---------- helper: prepare dataset from CSV ----------
def _prep_dataset_from_csv(
    csv_path: str,
    *,
    endpoints_k: int,
    jump_factor: float,
    tip_policy: str,
    global_up: Tuple[float, float, float],
    force_flip_x: bool,
    transform_to_local: bool,
) -> Tuple[Dict[str, np.ndarray], Optional[np.ndarray]]:
    df = pd.read_csv(csv_path)
    need = {"group", "idx_in_group", "x", "y", "z"}
    if need - set(df.columns):
        raise ValueError(f"CSV '{csv_path}' missing required columns {sorted(need)}.")
    groups_world = _to_groups_dict(df)

    LE_world = groups_world.get("LE")
    le_path_world = None
    if LE_world is not None and LE_world.size:
        le_path_world = _best_le_path_among_lowZ(LE_world, endpoints_k=endpoints_k, jump_factor=jump_factor)
        le_path_world = _orient_le_by_world_z(le_path_world, tip_policy=tip_policy)

    P3_world = groups_world.get("strut3")
    P4_world = groups_world.get("strut4")
    R_axes, origin = _compute_ref_frame_x_from_center_struts(
        le_path_world, P3_world, P4_world,
        global_up=np.asarray(global_up, float),
        force_flip_x=force_flip_x
    )

    if transform_to_local and R_axes is not None and origin is not None:
        pts_local = _transform_points(groups_world, origin, R_axes)
        le_path = (le_path_world - origin) @ R_axes if le_path_world is not None else None
    else:
        pts_local = groups_world
        le_path = le_path_world
    return pts_local, le_path


# --------------------------- Multi-CSV 4-view figure ---------------------------
from matplotlib.lines import Line2D

def plot_multi_static_shapes_from_csv(
    csv_paths: Sequence[str],
    *,
    labels: Optional[Sequence[str]] = None,
    colors: Optional[Sequence[str]] = None,
    tip_policy: str = "lowestZ",
    global_up: Tuple[float, float, float] = (0.0, 0.0, 1.0),
    flip_x: Optional[Sequence[bool]] = None,
    transform_to_local: bool = True,
    endpoints_k: int = 10,
    jump_factor: float = 2.5,
    elev_3d: float = 20,
    azim_3d: float = -60,
    figsize: Tuple[int, int] = (12, 10),
    save_path: Optional[str] = None,
    dpi: int = 300,
    show: bool = True,
    add_grid: bool = True,
    grid_linestyle: str = "-",
    pad_frac: float = 0.08,
    fixed_limits: Optional[dict] = None,
    figure_title: Optional[str] = "Static Reconstruction — Four Views",
    # legend controls
    legend_title: Optional[str] = None,
    legend_ncol: int = 1,
    legend_position: str = "between_right",  # "between_right" | "outside" | "none"
    legend_outside_loc: str = "center left",
    legend_outside_anchor: Tuple[float, float] = (1.02, 0.5),
    # NEW: always flip Top view axes in local frame (y,x). Set False to keep (x,y).
    flip_top_xy_in_local: bool = True,
    # Optional: direction flips for the Top view after swapping
    invert_top_x: bool = False,
    invert_top_y: bool = False,
    x_shifts: Optional[Sequence[float]] = None,
):
    """
    Panels:
      Top row   : [ 3D | Top ]
      Bottom row: [ Side | Front ]

    Axis mapping:
      transform_to_local = False (world):
          Side  = (y, z)
          Top   = (x, y)
          Front = (x, z)
      transform_to_local = True (local):
          Side  = (x, z)
          Top   = (y, x) if flip_top_xy_in_local else (x, y)
          Front = (y, z)
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    n = len(csv_paths)
    if n == 0:
        raise ValueError("No CSV files provided.")

    if labels is None:
        labels = [f"Kite {i+1}" for i in range(n)]
    if colors is None:
        default_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', [f"C{i}" for i in range(10)])
        colors = [default_cycle[i % len(default_cycle)] for i in range(n)]
    if flip_x is None:
        flip_x = [False] * n
    if not (len(labels) == len(colors) == len(flip_x) == n):
        raise ValueError("labels, colors, and flip_x must match the number of CSVs.")

    # Prepare datasets (already handles transform_to_local)
    # --------------------------------------------------
    # Load all datasets FIRST
    # --------------------------------------------------
    datasets = []
    for i, p in enumerate(csv_paths):
        pts_local, le_path = _prep_dataset_from_csv(
            p,
            endpoints_k=endpoints_k,
            jump_factor=jump_factor,
            tip_policy=tip_policy,
            global_up=global_up,
            force_flip_x=bool(flip_x[i]),
            transform_to_local=transform_to_local,
        )
        datasets.append((pts_local, le_path))

    # --------------------------------------------------
    # Apply manual X-translation AFTER loading all sets
    # --------------------------------------------------
    if x_shifts is None:
        x_shifts = [0.0] * len(datasets)

    if len(x_shifts) != len(datasets):
        raise ValueError("x_shifts must have same length as csv_paths.")

    shifted_datasets = []
    for (pts_local, le_path), dx in zip(datasets, x_shifts):

        if abs(dx) < 1e-12:
            shifted_datasets.append((pts_local, le_path))
            continue

        pts_shift = {g: (P + np.array([dx, 0, 0])) for g, P in pts_local.items()}
        le_shift = None if le_path is None else (le_path + np.array([dx, 0, 0]))

        shifted_datasets.append((pts_shift, le_shift))

    datasets = shifted_datasets

    # Compute cube limits (or use fixed)
    def _collect_axis_values(dsets, axis_index: int) -> np.ndarray:
        vals = []
        for pts_local, le_path in dsets:
            for P in pts_local.values():
                if P is not None and P.size:
                    vals.append(P[:, axis_index])
            if le_path is not None and le_path.size:
                vals.append(le_path[:, axis_index])
        return np.concatenate(vals) if vals else np.array([0.0])

    if fixed_limits is not None:
        xlim = tuple(fixed_limits.get('x', (-1.0, 1.0)))
        ylim = tuple(fixed_limits.get('y', (-1.0, 1.0)))
        zlim = tuple(fixed_limits.get('z', (-1.0, 1.0)))
    else:
        xs = _collect_axis_values(datasets, 0)
        ys = _collect_axis_values(datasets, 1)
        zs = _collect_axis_values(datasets, 2)
        xmin, xmax = float(np.nanmin(xs)), float(np.nanmax(xs))
        ymin, ymax = float(np.nanmin(ys)), float(np.nanmax(ys))
        zmin, zmax = float(np.nanmin(zs)), float(np.nanmax(zs))
        centers = np.array([(xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2], float)
        spans   = np.array([xmax-xmin, ymax-ymin, zmax-zmin], float)
        base_span = float(np.max(spans)) or 1.0
        span_padded = base_span * (1.0 + pad_frac*2)
        half = span_padded / 2.0
        xlim = (centers[0] - half, centers[0] + half)
        ylim = (centers[1] - half, centers[1] + half)
        zlim = (centers[2] - half, centers[2] + half)

    # Projections and labels by frame
    if transform_to_local:
        side_proj = (0, 2); side_labels = ("x [m]", "z [m]"); side_xlim, side_ylim = xlim, zlim
        if flip_top_xy_in_local:
            top_proj  = (1, 0)               # (y, x)
            top_labels = ("y [m]", "x [m]")
            top_xlim, top_ylim = ylim, xlim
            top_title = "Bottom view (y, x)"
        else:
            top_proj  = (0, 1)               # (x, y)
            top_labels = ("x [m]", "y [m]")
            top_xlim, top_ylim = xlim, ylim
            top_title = "Top view (x, y)"
        front_proj= (1, 2); front_labels= ("y [m]", "z [m]"); front_xlim, front_ylim = ylim, zlim
        side_title, front_title = "Side view (x, z)", "Front view (y, z)"
    else:
        side_proj = (1, 2); side_labels = ("y [m]", "z [m]"); side_xlim, side_ylim = ylim, zlim
        top_proj  = (0, 1); top_labels  = ("x [m]", "y [m]"); top_xlim,  top_ylim  = xlim, ylim
        front_proj= (0, 2); front_labels= ("x [m]", "z [m]"); front_xlim, front_ylim = xlim, zlim
        side_title, top_title, front_title = "Side view (y, z)", "Top view (x, y)", "Front view (x, z)"

    # Figure & axes
    was_interactive = plt.isinteractive()
    if was_interactive:
        plt.ioff()

    right_margin = 0.82 if legend_position == "outside" else 0.96
    top_margin = 0.90 if figure_title else 0.96

    fig = _new_figure(figsize)
    gs = fig.add_gridspec(2, 2, left=0.06, right=right_margin, bottom=0.06, top=top_margin, wspace=0.18, hspace=0.18)

    # Layout: [0,0]=3D, [0,1]=Top, [1,0]=Side, [1,1]=Front
    ax3d  = fig.add_subplot(gs[0, 0], projection="3d")
    ax_tp = fig.add_subplot(gs[0, 1])
    ax_sd = fig.add_subplot(gs[1, 0])
    ax_fr = fig.add_subplot(gs[1, 1])

    # 3D
    for (pts_local, le_path), color, _ in zip(datasets, colors, labels):
        plot_static_shape(
            points_by_group_3d_world=pts_local,
            strut_target_lengths=TARGETS,
            show=False, sciview=False,
            endpoints_k=endpoints_k, jump_factor=jump_factor,
            tip_policy=tip_policy, global_up=global_up,
            force_flip_x=False, transform_to_local=False,
            title=None, ax=ax3d, color=color, label=None,
        )

    ax3d.set_xlim(*xlim); ax3d.set_ylim(*ylim); ax3d.set_zlim(*zlim)
    ax3d.set_xlabel("x [m]"); ax3d.set_ylabel("y [m]"); ax3d.set_zlabel("z [m]")
    ax3d.view_init(elev=elev_3d, azim=azim_3d)
    try:
        _set_equal_aspect(ax3d)
    except Exception:
        pass
    if add_grid:
        ax3d.grid(True, which="both")

    # Top view
    for (pts_local, le_path), color, _ in zip(datasets, colors, labels):
        _plot_projection_2d(ax_tp, pts_local, le_path, color, None, proj=top_proj)
    ax_tp.set_title(top_title)
    ax_tp.set_xlabel(top_labels[0]); ax_tp.set_ylabel(top_labels[1])
    ax_tp.set_xlim(*top_xlim); ax_tp.set_ylim(*top_ylim)
    ax_tp.set_aspect("equal", adjustable="box")
    if invert_top_x: ax_tp.invert_xaxis()
    if invert_top_y: ax_tp.invert_yaxis()
    if add_grid:
        ax_tp.grid(True, which="both", linestyle=grid_linestyle, linewidth=0.8, alpha=0.6)

    # Side view
    for (pts_local, le_path), color, _ in zip(datasets, colors, labels):
        _plot_projection_2d(ax_sd, pts_local, le_path, color, None, proj=side_proj)
    ax_sd.set_title(side_title)
    ax_sd.set_xlabel(side_labels[0]); ax_sd.set_ylabel(side_labels[1])
    ax_sd.set_xlim(*side_xlim); ax_sd.set_ylim(*side_ylim)
    ax_sd.set_aspect("equal", adjustable="box")
    if add_grid:
        ax_sd.grid(True, which="both", linestyle=grid_linestyle, linewidth=0.8, alpha=0.6)

    # Front view
    for (pts_local, le_path), color, _ in zip(datasets, colors, labels):
        _plot_projection_2d(ax_fr, pts_local, le_path, color, None, proj=front_proj)
    ax_fr.set_title(front_title)
    ax_fr.set_xlabel(front_labels[0]); ax_fr.set_ylabel(front_labels[1])
    ax_fr.set_xlim(*front_xlim); ax_fr.set_ylim(*front_ylim)
    ax_fr.set_aspect("equal", adjustable="box")
    if add_grid:
        ax_fr.grid(True, which="both", linestyle=grid_linestyle, linewidth=0.8, alpha=0.6)

    # Figure title
    if figure_title:
        fig.suptitle(figure_title, fontsize=14)

    # Single legend
    proxies = [Line2D([0], [0], color=colors[i], lw=2, label=labels[i]) for i in range(n)]
    if legend_position == "outside":
        fig.legend(handles=proxies, loc=legend_outside_loc, bbox_to_anchor=legend_outside_anchor,
                   frameon=True, title=legend_title, ncol=legend_ncol, borderaxespad=0.0)
    elif legend_position == "between_right":
        try:
            fig.canvas.draw()
        except Exception:
            pass
        b_tp = ax_tp.get_position(fig).bounds
        b_fr = ax_fr.get_position(fig).bounds
        x_center = b_tp[0] + b_tp[2] / 2.0
        gap_bottom = b_fr[1] + b_fr[3]
        gap_top = b_tp[1]
        y_center = (gap_top + gap_bottom) / 2.0
        fig.legend(handles=proxies, loc="center",
                   bbox_to_anchor=(x_center, y_center), bbox_transform=fig.transFigure,
                   frameon=True, title=legend_title, ncol=legend_ncol, borderaxespad=0.0)

    try:
        fig.tight_layout()
    except Exception:
        pass
    fig.subplots_adjust(right=right_margin, top=top_margin, wspace=0.18, hspace=0.18)

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        if was_interactive:
            plt.ion()
        plt.show(block=False); plt.pause(0.001)
    else:
        if was_interactive:
            plt.ion()

    return fig, (ax3d, ax_tp, ax_sd, ax_fr)



# --------------------------- CLI Example ---------------------------

if __name__ == "__main__":
    TARGETS = [1.14, 1.53, 1.92, 2.02, 2.02, 1.92, 1.53, 1.14]

    fig, axes = plot_multi_static_shapes_from_csv(
        csv_paths=[
            "static_test_output/P2_S.csv",
            "static_test_output/P2_PL1.csv",
            "static_test_output/P2_PL2.csv",
            "static_test_output/P2_TL1.csv",
            "static_test_output/P2_TL2.csv",
        ],
        labels=["P2 + S", "P2 + PL1", "P2 + PL2", "P2 + TL1", "P2 + TL2", ],
        colors=["C0", "C1", "C2", "C3", "C4"],
        transform_to_local=True,
        fixed_limits={'x': (-2, 2), 'y': (-5, 5), 'z': (-4, 1)},
        flip_x=[False, False, False, False, False],
        x_shifts=[0.0, 0.0, 0.0, 0.0, 0.0],
        flip_top_xy_in_local=True,
        invert_top_x=False,
        invert_top_y=False,
        figure_title="Kite shape for different load cases",
        legend_title=None,
        legend_position="between_right",
        save_path="static_kite_four_views_with_title_and_legend.png",
        dpi=300,
    )


