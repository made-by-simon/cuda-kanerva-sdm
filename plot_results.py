"""
3D surface plot of KanervaSDM grid-search results.

Reads grid_search_results_cuda.csv and plots two side-by-side interactive 3D surfaces:
  - Left:  single-op mode timing
  - Right: batched mode timing

Axes use log10 scale because both dimension and num_locations span
multiple orders of magnitude, but tick labels show the original values.
"""

import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

CSV_FILE = "grid_search_results_cuda.csv"
PNG_FILE = "grid_search_results_cuda.png"


def make_grid(df_ok, dims_unique, locs_unique, col):
    """
    Build a (n_locs × n_dims) numpy array from a dataframe column.
    Missing / failed combinations are left as NaN.
    """
    Z = np.full((len(locs_unique), len(dims_unique)), np.nan)
    for i, loc in enumerate(locs_unique):
        for j, dim in enumerate(dims_unique):
            row = df_ok[
                (df_ok["dimension"] == dim) & (df_ok["num_locations"] == loc)
            ]
            if not row.empty:
                val = row[col].values[0]
                if pd.notna(val):
                    Z[i, j] = val
    return Z


def make_surface_trace(X, Y, Z, dims_unique, locs_unique, colorscale, name):
    """Return a plotly Surface trace with log10 axes and human-readable hover."""
    hover = np.empty(Z.shape, dtype=object)
    for i, loc in enumerate(locs_unique):
        for j, dim in enumerate(dims_unique):
            z = Z[i, j]
            if np.isnan(z):
                hover[i, j] = f"dim={dim}<br>locs={loc:,}<br>Time: N/A"
            else:
                hover[i, j] = f"dim={dim}<br>locs={loc:,}<br>Time: {z:.3f} s"

    return go.Surface(
        x=X,
        y=Y,
        z=Z,
        colorscale=colorscale,
        opacity=0.85,
        name=name,
        showscale=False,
        hovertemplate="%{text}<extra></extra>",
        text=hover,
        connectgaps=True,
    )


def make_scatter_trace(X, Y, Z, dims_unique, locs_unique):
    """
    Return a markers-only Scatter3d trace and a list of scene annotation dicts.
    Annotations carry white-background label boxes (Scatter3d text has no bgcolor).
    """
    xs, ys, zs, texts = [], [], [], []
    for i, loc in enumerate(locs_unique):
        for j, dim in enumerate(dims_unique):
            z = Z[i, j]
            if not np.isnan(z):
                xs.append(X[i, j])
                ys.append(Y[i, j])
                zs.append(z)
                texts.append(f"{z:.3f}s")

    trace = go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode="markers",
        marker=dict(size=5, color="black", symbol="circle"),
        hovertemplate="Time: %{z:.3f} s<extra></extra>",
        showlegend=False,
    )

    annotations = [
        dict(
            x=x, y=y, z=z,
            text=text,
            showarrow=False,
            font=dict(size=10, color="black"),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="rgba(180,180,180,0.8)",
            borderwidth=1,
            borderpad=4,
            xanchor="center",
            yanchor="bottom",
        )
        for x, y, z, text in zip(xs, ys, zs, texts)
    ]

    return trace, annotations


def axis_config(tick_vals, tick_labels, title):
    """Return a plotly 3D axis dict (no gridlines; used for x and y axes)."""
    return dict(
        title=title,
        tickvals=tick_vals,
        ticktext=tick_labels,
        showgrid=False,
        showbackground=False,
    )


def main():
    try:
        df = pd.read_csv(CSV_FILE)
    except FileNotFoundError:
        print(f"[ERROR] {CSV_FILE} not found. Run grid_search.py first.")
        sys.exit(1)

    df_ok = df.dropna(subset=["single_op_time_s", "batched_time_s"]).copy()

    if df_ok.empty:
        print("[ERROR] No rows with valid timing data found in CSV.")
        sys.exit(1)

    print(f"Loaded {len(df_ok)} result(s) from {CSV_FILE}:")
    print(df_ok[["dimension", "num_locations",
                  "single_op_time_s", "batched_time_s"]].to_string(index=False))

    dims_unique = sorted(df["dimension"].unique())
    locs_unique = sorted(df["num_locations"].unique())

    log_dims = np.log10(dims_unique)
    log_locs = np.log10(locs_unique)
    X, Y = np.meshgrid(log_dims, log_locs)  # shape: (n_locs, n_dims)

    Z_single  = make_grid(df_ok, dims_unique, locs_unique, "single_op_time_s")
    Z_batched = make_grid(df_ok, dims_unique, locs_unique, "batched_time_s")

    tick_lbl_dim  = [str(d) for d in dims_unique]
    tick_lbl_locs = [f"{l:,}" for l in locs_unique]

    xaxis_cfg = axis_config(log_dims, tick_lbl_dim,  "Dimension")
    yaxis_cfg = axis_config(log_locs, tick_lbl_locs, "Num Locations")

    z_max = np.nanmax([np.nanmax(Z_single), np.nanmax(Z_batched)])
    zaxis_cfg = dict(title="Time (s)", showgrid=True, gridcolor="rgba(180,180,180,1)",
                     gridwidth=1, showbackground=False, range=[0, z_max * 1.1])

    memory_count = int(df["memory_count"].iloc[0])
    camera = dict(eye=dict(x=1.6, y=-1.6, z=1.2))

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("Single-Op Mode", "Batched Mode"),
        horizontal_spacing=0.02,
    )

    scatter_single,  annots_single  = make_scatter_trace(X, Y, Z_single,  dims_unique, locs_unique)
    scatter_batched, annots_batched = make_scatter_trace(X, Y, Z_batched, dims_unique, locs_unique)

    fig.add_trace(make_surface_trace(X, Y, Z_single,  dims_unique, locs_unique, "Viridis", "Single-op"), row=1, col=1)
    fig.add_trace(scatter_single,  row=1, col=1)
    fig.add_trace(make_surface_trace(X, Y, Z_batched, dims_unique, locs_unique, "Viridis", "Batched"),   row=1, col=2)
    fig.add_trace(scatter_batched, row=1, col=2)

    fig.update_layout(
        title=dict(
            text=f"KanervaSDM CUDA Performance  (memory_count={memory_count})",
            x=0.5, font=dict(size=16),
        ),
        scene=dict(
            xaxis=xaxis_cfg, yaxis=yaxis_cfg, zaxis=zaxis_cfg,
            camera=camera,
            annotations=annots_single,
        ),
        scene2=dict(
            xaxis=xaxis_cfg, yaxis=yaxis_cfg, zaxis=zaxis_cfg,
            camera=camera,
            annotations=annots_batched,
        ),
        width=1400,
        height=700,
        margin=dict(l=0, r=0, t=80, b=0),
    )

    fig.write_image(PNG_FILE, scale=2)
    print(f"\nPlot saved to {PNG_FILE}")


if __name__ == "__main__":
    main()
