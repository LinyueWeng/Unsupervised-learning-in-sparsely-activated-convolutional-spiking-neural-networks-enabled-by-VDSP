import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from config import *

"""
C2C Variation:
Fig.4.9 & Fig.4.10
change Synapse_Model in config.py
"""

if SYNAPSE_MODEL == 'Ferroelectric_Tanh':

    # ── Raw data ────────────────────────────────────────────────────────────────
    data = {
        # ── Baseline & axis-aligned points ──────────────────────────────────────
        (0.0, 0.0):    {"accs": [88.23, 88.62, 87.91, 88.29, 88.09], "samples": [173, 237, 246, 142,  92]},
        (0.0, 0.02):   {"accs": [87.7,  87.78, 88.13, 86.9,  88.28], "samples": [115, 107, 123, 101, 176]},
        (0.0, 0.1):    {"accs": [86.86, 86.94, 86.96, 87.03, 87.11], "samples": [ 80, 107,  69,  61,  74]},
        (0.0, 0.2):    {"accs": [85.94, 85.57, 85.67, 85.3,  86.31], "samples": [301, 301, 301, 301, 301]},

        (0.05, 0.0):   {"accs": [87.44, 87.52, 88.09, 88.31, 88.18], "samples": [199, 164, 161, 113, 108]},
        (0.3,  0.0):   {"accs": [86.28, 86.05, 85.11, 86.45, 86.39], "samples": [301, 301, 300, 301, 246]},
        (0.5,  0.0):   {"accs": [85.27, 85.51, 84.89, 85.25, 84.71], "samples": [301, 301, 301, 301, 301]},

        # ── Mixed-noise transition points ────────────────────────────────────────
        (0.005, 0.0002): {"accs": [88.39, 87.78, 87.26, 88.12, 86.68], "samples": [135, 141, 132, 151, 173]},
        (0.005, 0.002):  {"accs": [87.75, 87.62, 87.64, 88.36, 87.2],  "samples": [126, 184, 155, 141, 208]},
        (0.005, 0.02):   {"accs": [87.46, 88.53, 87.77, 87.47, 88.18], "samples": [108, 130, 101, 156,  89]},
        (0.05,  0.0002): {"accs": [87.8,  85.92, 88.19, 88.27, 87.91], "samples": [128, 108, 199, 139, 104]},
        (0.05,  0.002):  {"accs": [87.71, 88.02, 85.42, 88.28, 87.39], "samples": [183, 239, 119, 129, 122]},

        # ── High-noise corner points ─────────────────────────────────────────────
        (0.3,  0.1):   {"accs": [86.14, 85.44, 85.62, 85.89, 86.88], "samples": [301, 301, 301, 301, 301]},
        (0.3,  0.2):   {"accs": [85.07, 85.38, 84.87, 85.51, 85.62], "samples": [301, 301, 301, 301, 301]},
        (0.5,  0.1):   {"accs": [84.13, 85.25, 84.65, 85.36, 85.61], "samples": [301, 301, 301, 301, 301]},
        (0.5,  0.2):   {"accs": [85.16, 85.59, 84.56, 85.51, 84.76], "samples": [301, 301, 301, 301, 301]},
    }

    # ── Output path ─────────────────────────────────────────────────────────────
    save_path = "figures"
    os.makedirs(save_path, exist_ok=True)

    # ── Style ───────────────────────────────────────────────────────────────────
    COLOR_ADD  = '#E8694C'
    COLOR_MUL  = '#4C9BE8'
    COLOR_BASE = '#2E8B57'

    # ── Discrete categorical axes ───────────────────────────────────────────────
    x_vals   = [0.0, 0.005, 0.05, 0.3, 0.5]
    y_vals   = [0.0, 0.0002, 0.002, 0.02, 0.1, 0.2]
    x_labels = ['0', '0.005', '0.05', '0.3', '0.5']
    y_labels = ['0', '0.0002', '0.002', '0.02', '0.1', '0.2']

    def build_grid(metric='acc'):
        z = []
        for sa in y_vals:
            row = []
            for sm in x_vals:
                if (sm, sa) in data:
                    vals = np.array(data[(sm, sa)]['accs' if metric == 'acc' else 'samples'], dtype=float)
                    row.append(float(np.mean(vals)))
                else:
                    row.append(np.nan)
            z.append(row)
        return np.array(z, dtype=float)

    def build_text(z, decimals):
        txt = []
        for row in z:
            row_txt = []
            for v in row:
                row_txt.append("" if np.isnan(v) else f"{v:.{decimals}f}")
            txt.append(row_txt)
        return txt

    acc_mean  = build_grid('acc')
    samp_mean = build_grid('samples')
    baseline_acc = float(np.mean(data[(0.0, 0.0)]['accs']))

    # ── Figure 1: Accuracy heatmap ──────────────────────────────────────────────
    fig = go.Figure()

    fig.add_trace(go.Heatmap(
        x=x_labels,
        y=y_labels,
        z=acc_mean,
        text=build_text(acc_mean, 2),
        texttemplate="%{text}",
        textfont=dict(size=14),
        colorscale="Blues",
        zmin=84.5,
        zmax=88.5,
        xgap=3,
        ygap=3,
        colorbar=dict(title="Acc. (%)"),
        hovertemplate="σm=%{x}<br>σa=%{y}<br>Mean Acc=%{z:.2f}%<extra></extra>",
    ))

    fig.update_xaxes(
        title_text='Multiplicative C2C Coefficient (σm)',
        type='category', categoryorder='array', categoryarray=x_labels, tickangle=0
    )
    fig.update_yaxes(
        title_text='Additive C2C Coefficient (σa)',
        type='category', categoryorder='array', categoryarray=y_labels
    )
    fig.update_layout(title={"text": "C2C Variation: Accuracy Landscape in Tanh Model"})

    fig.write_image(f"{save_path}/Tanh_C2C_acc_heatmap.pdf", format="pdf")
    fig.write_image(f"{save_path}/Tanh_C2C_acc_heatmap.png")

    # ── Figure 2: Training samples heatmap ─────────────────────────────────────
    fig = go.Figure()

    fig.add_trace(go.Heatmap(
        x=x_labels,
        y=y_labels,
        z=samp_mean,
        text=build_text(samp_mean, 0),
        texttemplate="%{text}",
        textfont=dict(size=14),
        colorscale="Purples",
        zmin=60,
        zmax=301,
        xgap=3,
        ygap=3,
        colorbar=dict(title="Samples"),
        hovertemplate="σm=%{x}<br>σa=%{y}<br>Mean Samples=%{z:.0f}<extra></extra>",
    ))

    fig.update_xaxes(
        title_text='Multiplicative C2C Coefficient (σm)',
        type='category', categoryorder='array', categoryarray=x_labels, tickangle=0
    )
    fig.update_yaxes(
        title_text='Additive C2C Coefficient (σa)',
        type='category', categoryorder='array', categoryarray=y_labels
    )
    fig.update_layout(title={"text": "C2C Variation: Training-Sample Dynamics in Tanh Model"})

    fig.write_image(f"{save_path}/Tanh_C2C_samples_heatmap.pdf", format="pdf")
    fig.write_image(f"{save_path}/Tanh_C2C_samples_heatmap.png")

    # ── Figure 3: One-dimensional slices ────────────────────────────────────────
    add_only_xs    = [0.0, 0.02, 0.1, 0.2]
    add_only_means = [float(np.mean(data[(0.0, sa)]["accs"])) for sa in add_only_xs]
    add_only_stds  = [float(np.std(data[(0.0, sa)]["accs"]))  for sa in add_only_xs]

    mul_only_xs    = [0.0, 0.05, 0.3, 0.5]
    mul_only_means = [float(np.mean(data[(sm, 0.0)]["accs"])) for sm in mul_only_xs]
    mul_only_stds  = [float(np.std(data[(sm, 0.0)]["accs"]))  for sm in mul_only_xs]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "Additive-only slice (σm = 0)",
            "Multiplicative-only slice (σa = 0)"
        ),
        horizontal_spacing=0.14
    )

    fig.add_trace(go.Scatter(
        x=add_only_xs, y=add_only_means,
        error_y=dict(type='data', array=add_only_stds, visible=True, thickness=1.6, width=5),
        mode='lines+markers',
        marker=dict(size=8, color=COLOR_ADD, symbol='circle'),
        line=dict(color=COLOR_ADD, width=2.5),
        name='Additive-only',
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=mul_only_xs, y=mul_only_means,
        error_y=dict(type='data', array=mul_only_stds, visible=True, thickness=1.6, width=5),
        mode='lines+markers',
        marker=dict(size=8, color=COLOR_MUL, symbol='diamond'),
        line=dict(color=COLOR_MUL, width=2.5),
        name='Multiplicative-only',
    ), row=1, col=2)

    fig.add_hline(y=baseline_acc, line_dash='dot', line_color=COLOR_BASE, line_width=1.4, row=1, col=1)
    fig.add_hline(y=baseline_acc, line_dash='dot', line_color=COLOR_BASE, line_width=1.4, row=1, col=2)

    fig.add_annotation(
        x=0.03, y=baseline_acc + 0.08, text='Baseline',
        showarrow=False, font=dict(size=11, color=COLOR_BASE), row=1, col=1
    )
    fig.add_annotation(
        x=0.32, y=baseline_acc + 0.08, text='Baseline',
        showarrow=False, font=dict(size=11, color=COLOR_BASE), row=1, col=2
    )

    fig.update_xaxes(title_text='σa', tickvals=[0.0, 0.02, 0.1, 0.2], row=1, col=1)
    fig.update_xaxes(title_text='σm', tickvals=[0.0, 0.05, 0.3, 0.5],  row=1, col=2)
    fig.update_yaxes(title_text='Accuracy (%)', range=[84.4, 88.8], row=1, col=1)
    fig.update_yaxes(title_text='Accuracy (%)', range=[84.4, 88.8], row=1, col=2)

    fig.update_layout(
        title={"text": "C2C Variation: One-Dimensional Slices in Tanh Model"},
        legend=dict(
            orientation='h', x=0.5, y=-0.12,
            xanchor='center', yanchor='top',
            bgcolor='rgba(255,255,255,0.85)',
            bordercolor='lightgray', borderwidth=1,
        ),
        margin=dict(t=90, b=90),
    )

    fig.update_traces(selector=dict(type='scatter'), cliponaxis=False)

    fig.write_image(f"{save_path}/Tanh_C2C_slices.pdf", format="pdf")
    fig.write_image(f"{save_path}/Tanh_C2C_slices.png")

if SYNAPSE_MODEL == 'Ferroelectric':

    # ── Raw data ────────────────────────────────────────────────────────────────
    data = {
        # ── Baseline & axis-aligned points ──────────────────────────────────────
        (0.0, 0.0):    {"accs": [88.51, 88.35, 88.51, 88.3,  88.44], "samples": [272, 259, 272, 261, 268]},
        (0.0, 0.02):   {"accs": [87.81, 87.92, 87.65, 87.97, 87.83], "samples": [253, 249, 252, 238, 248]},
        (0.0, 0.1):    {"accs": [87.4,  86.47, 86.82, 87.62, 87.44], "samples": [95,  95,  97,  97,  97]},
        (0.0, 0.2):    {"accs": [87.0,  86.85, 87.26, 87.24, 87.01], "samples": [39,  38,  38,  38,  36]},

        (0.05, 0.0):   {"accs": [88.45, 88.33, 88.71, 88.45, 88.25], "samples": [265, 258, 266, 266, 265]},
        (0.3,  0.0):   {"accs": [88.06, 88.4,  88.5,  88.37, 88.39], "samples": [271, 268, 260, 265, 270]},
        (0.5,  0.0):   {"accs": [88.51, 88.31, 88.45, 88.59, 88.51], "samples": [267, 267, 265, 262, 272]},

        # ── Mixed-noise transition points ───────────────────────────────────────
        (0.005, 0.0002): {"accs": [88.49, 88.27, 88.2,  88.89, 88.2],  "samples": [270, 274, 263, 267, 267]},
        (0.005, 0.002):  {"accs": [88.19, 88.57, 87.98, 88.49, 88.64], "samples": [274, 266, 266, 276, 273]},
        (0.005, 0.02):   {"accs": [88.08, 87.86, 88.08, 87.55, 87.71], "samples": [245, 253, 256, 255, 253]},
        (0.05,  0.0002): {"accs": [88.37, 88.6,  88.24, 88.34, 88.42], "samples": [269, 270, 278, 272, 270]},
        (0.05,  0.002):  {"accs": [88.42, 88.08, 88.44, 88.21, 88.78], "samples": [270, 266, 264, 267, 270]},

        # ── High-noise corner points ────────────────────────────────────────────
        (0.3,  0.1):   {"accs": [87.29, 86.88, 88.06, 87.04, 87.03], "samples": [96, 103, 93, 94, 93]},
        (0.3,  0.2):   {"accs": [87.38, 87.15, 87.18, 87.19, 86.8],  "samples": [38, 38, 36, 38, 36]},
        (0.5,  0.1):   {"accs": [86.75, 87.0,  86.97, 87.55, 86.96], "samples": [96, 97, 97, 92, 95]},
        (0.5,  0.2):   {"accs": [87.21, 87.15, 86.81, 87.17, 86.82], "samples": [39, 39, 39, 35, 37]},
    }

    # ── Output path ─────────────────────────────────────────────────────────────
    save_path = "figures"
    os.makedirs(save_path, exist_ok=True)

    # ── Style ───────────────────────────────────────────────────────────────────
    COLOR_ADD  = '#E8694C'
    COLOR_MUL  = '#4C9BE8'
    COLOR_BASE = '#2E8B57'

    # ── Discrete categorical axes ───────────────────────────────────────────────
    x_vals   = [0.0, 0.005, 0.05, 0.3, 0.5]
    y_vals   = [0.0, 0.0002, 0.002, 0.02, 0.1, 0.2]
    x_labels = ['0', '0.005', '0.05', '0.3', '0.5']
    y_labels = ['0', '0.0002', '0.002', '0.02', '0.1', '0.2']

    def build_grid(metric='acc'):
        z = []
        for sa in y_vals:
            row = []
            for sm in x_vals:
                if (sm, sa) in data:
                    vals = np.array(data[(sm, sa)]['accs' if metric == 'acc' else 'samples'], dtype=float)
                    row.append(float(np.mean(vals)))
                else:
                    row.append(np.nan)
            z.append(row)
        return np.array(z, dtype=float)

    def build_text(z, decimals):
        txt = []
        for row in z:
            row_txt = []
            for v in row:
                row_txt.append("" if np.isnan(v) else f"{v:.{decimals}f}")
            txt.append(row_txt)
        return txt

    acc_mean = build_grid('acc')
    samp_mean = build_grid('samples')
    baseline_acc = float(np.mean(data[(0.0, 0.0)]['accs']))

    # ── Figure 1: Accuracy heatmap ──────────────────────────────────────────────
    fig = go.Figure()

    fig.add_trace(go.Heatmap(
        x=x_labels,
        y=y_labels,
        z=acc_mean,
        text=build_text(acc_mean, 2),
        texttemplate="%{text}",
        textfont=dict(size=14),
        colorscale="Blues",
        zmin=86.5,
        zmax=88.9,
        xgap=3,
        ygap=3,
        colorbar=dict(title="Acc. (%)"),
        hovertemplate="σm=%{x}<br>σa=%{y}<br>Mean Acc=%{z:.2f}%<extra></extra>",
    ))

    fig.update_xaxes(
        title_text='Multiplicative C2C Coefficient (σm)',
        type='category', categoryorder='array', categoryarray=x_labels, tickangle=0
    )
    fig.update_yaxes(
        title_text='Additive C2C Coefficient (σa)',
        type='category', categoryorder='array', categoryarray=y_labels
    )
    fig.update_layout(
        title={"text": "C2C Variation: Accuracy Landscape in Exponential Model"}
    )

    fig.write_image(f"{save_path}/Exp_C2C_acc_heatmap.pdf", format="pdf")
    fig.write_image(f"{save_path}/Exp_C2C_acc_heatmap.png")

    # ── Figure 2: Training samples heatmap ─────────────────────────────────────
    fig = go.Figure()

    fig.add_trace(go.Heatmap(
        x=x_labels,
        y=y_labels,
        z=samp_mean,
        text=build_text(samp_mean, 0),
        texttemplate="%{text}",
        textfont=dict(size=14),
        colorscale="Purples",
        zmin=35,
        zmax=280,
        xgap=3,
        ygap=3,
        colorbar=dict(title="Samples"),
        hovertemplate="σm=%{x}<br>σa=%{y}<br>Mean Samples=%{z:.0f}<extra></extra>",
    ))

    fig.update_xaxes(
        title_text='Multiplicative C2C Coefficient (σm)',
        type='category', categoryorder='array', categoryarray=x_labels, tickangle=0
    )
    fig.update_yaxes(
        title_text='Additive C2C Coefficient (σa)',
        type='category', categoryorder='array', categoryarray=y_labels
    )
    fig.update_layout(
        title={"text": "C2C Variation: Training-Sample Dynamics in Exponential Model"}
    )

    fig.write_image(f"{save_path}/Exp_C2C_samples_heatmap.pdf", format="pdf")
    fig.write_image(f"{save_path}/Exp_C2C_samples_heatmap.png")

    # ── Figure 3: One-dimensional slices ────────────────────────────────────────
    add_only_xs = [0.0, 0.02, 0.1, 0.2]
    add_only_means = [float(np.mean(data[(0.0, sa)]["accs"])) for sa in add_only_xs]
    add_only_stds = [float(np.std(data[(0.0, sa)]["accs"])) for sa in add_only_xs]

    mul_only_xs = [0.0, 0.05, 0.3, 0.5]
    mul_only_means = [float(np.mean(data[(sm, 0.0)]["accs"])) for sm in mul_only_xs]
    mul_only_stds = [float(np.std(data[(sm, 0.0)]["accs"])) for sm in mul_only_xs]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "Additive-only slice (σm = 0)",
            "Multiplicative-only slice (σa = 0)"
        ),
        horizontal_spacing=0.14
    )

    fig.add_trace(go.Scatter(
        x=add_only_xs,
        y=add_only_means,
        error_y=dict(type='data', array=add_only_stds, visible=True, thickness=1.6, width=5),
        mode='lines+markers',
        marker=dict(size=8, color=COLOR_ADD, symbol='circle'),
        line=dict(color=COLOR_ADD, width=2.5),
        name='Additive-only',
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=mul_only_xs,
        y=mul_only_means,
        error_y=dict(type='data', array=mul_only_stds, visible=True, thickness=1.6, width=5),
        mode='lines+markers',
        marker=dict(size=8, color=COLOR_MUL, symbol='diamond'),
        line=dict(color=COLOR_MUL, width=2.5),
        name='Multiplicative-only',
    ), row=1, col=2)

    fig.add_hline(
        y=baseline_acc, line_dash='dot', line_color=COLOR_BASE, line_width=1.4, row=1, col=1
    )
    fig.add_hline(
        y=baseline_acc, line_dash='dot', line_color=COLOR_BASE, line_width=1.4, row=1, col=2
    )

    fig.add_annotation(
        x=0.03, y=baseline_acc + 0.05, text='Baseline',
        showarrow=False, font=dict(size=11, color=COLOR_BASE), row=1, col=1
    )
    fig.add_annotation(
        x=0.32, y=baseline_acc + 0.05, text='Baseline',
        showarrow=False, font=dict(size=11, color=COLOR_BASE), row=1, col=2
    )

    fig.update_xaxes(
        title_text='σa', tickvals=[0.0, 0.02, 0.1, 0.2], row=1, col=1
    )
    fig.update_xaxes(
        title_text='σm', tickvals=[0.0, 0.05, 0.3, 0.5], row=1, col=2
    )
    fig.update_yaxes(
        title_text='Accuracy (%)', range=[86.3, 88.9], row=1, col=1
    )
    fig.update_yaxes(
        title_text='Accuracy (%)', range=[86.3, 88.9], row=1, col=2
    )

    fig.update_layout(
        title={"text": "C2C Variation: One-Dimensional Slices in Exponential Model"},
        legend=dict(
            orientation='h', x=0.5, y=-0.12,
            xanchor='center', yanchor='top',
            bgcolor='rgba(255,255,255,0.85)',
            bordercolor='lightgray', borderwidth=1,
        ),
        margin=dict(t=90, b=90),
    )

    fig.update_traces(selector=dict(type='scatter'), cliponaxis=False)

    fig.write_image(f"{save_path}/Exp_C2C_slices.pdf", format="pdf")
    fig.write_image(f"{save_path}/Exp_C2C_slices.png")