import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from config import *

"""
D2D Variation:
Fig.4.7 & Fig.4.8
change Synapse_Model in config.py
"""

if SYNAPSE_MODEL=='Ferroelectric_Tanh':
    data = {
        0.0:  {"accs": [88.23, 88.62, 87.91, 88.29, 88.09], "defect": 0.0},
        0.01: {"accs": [88.43, 87.26, 87.58, 88.05, 87.77], "defect": 0.0},
        0.03: {"accs": [87.85, 84.87, 87.32, 87.07, 88.28], "defect": 0.0},
        0.05: {"accs": [88.21, 87.23, 88.0,  88.45, 86.86], "defect": 0.0},
        0.08: {"accs": [88.32, 88.36, 87.55, 87.78, 87.4],  "defect": 0.0},
        0.10: {"accs": [87.53, 86.76, 87.9,  86.66, 87.89], "defect": 0.0},
        0.15: {"accs": [87.85, 87.55, 87.63, 87.73, 87.88], "defect": 0.0},
        0.20: {"accs": [87.76, 87.64, 87.26, 87.35, 87.28], "defect": 0.0},
        0.25: {"accs": [87.52, 87.72, 87.64, 87.15, 87.64, 87.2, 88.23, 86.47, 87.55, 87.75,
                        87.72, 87.4, 86.7, 87.68, 87.79, 87.02, 87.45, 87.46, 87.49, 87.63],
               "defect": 0.02647},
        0.30: {"accs": [56.36, 55.41, 87.39, 55.43, 87.32, 87.71, 87.47, 86.39, 55.98, 54.45,
                        55.44, 54.24, 55.06, 53.83, 52.98, 56.61, 50.2, 87.26, 86.74, 86.76],
               "defect": 0.05230},
        0.35: {"accs": [47.83, 51.53, 49.94, 48.12, 49.9, 49.78, 54.35, 46.14, 49.82, 51.88,
                        47.01, 47.69, 50.62, 49.69, 10.0, 50.1, 49.73, 49.24, 48.86, 51.16],
               "defect": 0.08705},
        0.40: {"accs": [49.91, 51.02, 50.32, 49.48, 10.0, 49.2, 46.59, 50.33, 48.86, 48.5,
                        50.05, 47.61, 49.51, 49.45, 53.42, 48.25, 49.3, 50.65, 49.46, 49.36],
               "defect": 0.12755},
        0.45: {"accs": [14.63, 48.94, 10.0, 50.08, 49.14, 52.01, 47.78, 48.4, 49.31, 48.82,
                        51.91, 49.13, 49.31, 50.21, 23.86, 51.91, 48.89, 50.42, 48.71, 48.16],
               "defect": 0.18080},
        0.50: {"accs": [51.71, 50.09, 48.83, 52.12, 13.43, 21.03, 51.13, 52.06, 48.15, 21.51,
                        47.36, 47.84, 48.53, 22.2, 50.94, 24.98, 48.0, 48.74, 48.54, 50.37],
               "defect": 0.24075},
    }

    THR_FP   = 70.0
    THR_DEEP = 30.0

    strengths_all = sorted(data.keys())
    strengths = [s for s in strengths_all if data[s]["accs"] is not None]

    # ── Branch separation ─────────────────────────────────────────────────────────
    s1_xs,   s1_means,   s1_stds   = [], [], []
    fp_xs,   fp_means,   fp_stds   = [], [], []
    mid_xs,  mid_means,  mid_stds  = [], [], []
    deep_xs, deep_means, deep_stds = [], [], []

    for s in strengths:
        accs = np.array(data[s]["accs"])
        fp   = accs[accs > THR_FP]
        mid  = accs[(accs > THR_DEEP) & (accs <= THR_FP)]
        deep = accs[accs <= THR_DEEP]

        if s <= 0.25:
            s1_xs.append(s)
            s1_means.append(float(np.mean(accs)))
            s1_stds.append(float(np.std(accs)))

        if len(fp) > 0:
            fp_xs.append(s)
            fp_means.append(float(np.mean(fp)))
            fp_stds.append(float(np.std(fp)) if len(fp) > 1 else 0.0)

        if len(mid) > 0:
            mid_xs.append(s)
            mid_means.append(float(np.mean(mid)))
            mid_stds.append(float(np.std(mid)) if len(mid) > 1 else 0.0)

        if len(deep) > 0:
            deep_xs.append(s)
            deep_means.append(float(np.mean(deep)))
            deep_stds.append(float(np.std(deep)) if len(deep) > 1 else 0.0)

    defect_xs = strengths_all
    defect_ys = [data[s]["defect"] * 100 for s in defect_xs]

    # ── Colors ────────────────────────────────────────────────────────────────────
    COLOR_FP   = '#4C9BE8'
    COLOR_MID  = '#E8694C'
    COLOR_DEEP = '#8B1A1A'
    COLOR_DEF  = '#7B61FF'
    COLOR_FILL = 'rgba(76,155,232,0.18)'

    # ── Figure ────────────────────────────────────────────────────────────────────
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.06,
    )

    # Stage I std band
    s1_upper = [m + s for m, s in zip(s1_means, s1_stds)]
    s1_lower = [m - s for m, s in zip(s1_means, s1_stds)]
    fig.add_trace(go.Scatter(
        x=s1_xs + s1_xs[::-1], y=s1_upper + s1_lower[::-1],
        fill='toself', fillcolor=COLOR_FILL,
        line=dict(color='rgba(0,0,0,0)'),
        hoverinfo='skip', showlegend=False,
    ), row=1, col=1)

    # FP branch
    fig.add_trace(go.Scatter(
        x=fp_xs, y=fp_means,
        error_y=dict(type='data', array=fp_stds, visible=True, thickness=1.5, width=5),
        mode='lines+markers',
        marker=dict(size=8, symbol='circle', color=COLOR_FP),
        line=dict(color=COLOR_FP, width=2.5),
        name='Function-Preserving Branch',
    ), row=1, col=1)

    # Mid collapse branch (~50%)
    fig.add_trace(go.Scatter(
        x=mid_xs, y=mid_means,
        error_y=dict(type='data', array=mid_stds, visible=True, thickness=1.5, width=5),
        mode='lines+markers',
        marker=dict(size=8, symbol='x-thin', color=COLOR_MID),
        line=dict(color=COLOR_MID, width=2.5, dash='dash'),
        name='Collapse Branch (~50%)',
    ), row=1, col=1)

    # Deep collapse branch (~10%)
    fig.add_trace(go.Scatter(
        x=deep_xs, y=deep_means,
        error_y=dict(type='data', array=deep_stds, visible=True, thickness=1.5, width=5),
        mode='lines+markers',
        marker=dict(size=9, symbol='triangle-down', color=COLOR_DEEP),
        line=dict(color=COLOR_DEEP, width=2.0, dash='dot'),
        name='Deep Collapse Branch (~10%)',
    ), row=1, col=1)

    # Random guess reference
    fig.add_hline(y=10.0, line_dash='dash', line_color=COLOR_DEEP, line_width=1,
                  opacity=0.4, row=1, col=1,
                  annotation_text='Random guess (~10%)',
                  annotation_position='bottom right',
                  annotation_font_size=11, annotation_font_color=COLOR_DEEP)

    # Bifurcation vertical line
    for r in [1, 2]:
        fig.add_vline(x=0.30, line_dash='dot', line_color='gray', line_width=1.5, row=r, col=1)
    fig.add_annotation(x=0.305, y=108, text='Bifurcation (c=0.30)',
                       showarrow=False, font=dict(size=11, color='gray'),
                       xanchor='left', row=1, col=1)

    # Defect ratio bar (bottom panel)
    fig.add_trace(go.Bar(
        x=defect_xs, y=defect_ys,
        marker_color=[COLOR_DEF if d > 0 else 'lightgray' for d in defect_ys],
        marker_line_width=0,
        name='Defect Ratio (%)',
        width=0.018,
    ), row=2, col=1)

    # ── Axes & layout ─────────────────────────────────────────────────────────────
    fig.update_xaxes(
        tickvals=[0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
        title_text='Variation Strength (c)', row=2, col=1,
    )
    fig.update_yaxes(title_text='Accuracy (%)', range=[-2, 112], row=1, col=1)
    fig.update_yaxes(title_text='Defect (%)',   range=[0, 28],   row=2, col=1)

    fig.update_layout(
        title={"text": "D2D Variation: Three-Branch Bifurcation in Tanh Model (Fashion-MNIST)"},
        legend=dict(
            orientation='v',
            x=0.02, y=0.60,
            xanchor='left', yanchor='top',
            bgcolor='rgba(255,255,255,0.85)',
            bordercolor='lightgray', borderwidth=1,
            font=dict(size=12),
        ),
        bargap=0.3,
    )
    fig.update_traces(cliponaxis=False)

    save_path="figures"
    fig.write_image(f"{save_path}/Tanh_D2D.pdf", format="pdf")
    fig.write_image(f"{save_path}/Tanh_D2D.png",)


    data_s1 = {
        0.0:  [88.23, 88.62, 87.91, 88.29, 88.09],
        0.01: [88.43, 87.26, 87.58, 88.05, 87.77],
        0.03: [87.85, 84.87, 87.32, 87.07, 88.28],
        0.05: [88.21, 87.23, 88.0,  88.45, 86.86],
        0.08: [88.32, 88.36, 87.55, 87.78, 87.4 ],
        0.10: [87.53, 86.76, 87.9,  86.66, 87.89],
        0.15: [87.85, 87.55, 87.63, 87.73, 87.88],
        0.20: [87.76, 87.64, 87.26, 87.35, 87.28],
        0.25: [87.52, 87.72, 87.64, 87.15, 87.64, 87.2, 88.23, 86.47, 87.55, 87.75,
               87.72, 87.4, 86.7, 87.68, 87.79, 87.02, 87.45, 87.46, 87.49, 87.63],
    }

    strengths = sorted(data_s1.keys())
    means = [float(np.mean(data_s1[s])) for s in strengths]
    stds  = [float(np.std(data_s1[s]))  for s in strengths]
    mins_ = [float(np.min(data_s1[s]))  for s in strengths]
    maxs_ = [float(np.max(data_s1[s]))  for s in strengths]

    COLOR_MAIN = '#4C9BE8'
    COLOR_ENV  = '#A8C8F0'

    fig = go.Figure()

    # Max envelope line
    fig.add_trace(go.Scatter(
        x=strengths, y=maxs_,
        mode='lines', line=dict(color=COLOR_ENV, width=1, dash='dot'),
        name='Max', showlegend=True,
    ))

    # Min envelope + fill between max and min
    fig.add_trace(go.Scatter(
        x=strengths, y=mins_,
        mode='lines', line=dict(color=COLOR_ENV, width=1, dash='dot'),
        fill='tonexty', fillcolor='rgba(168,200,240,0.18)',
        name='Min (Min–Max Range)', showlegend=True,
    ))

    # Mean line with ±std error bars
    fig.add_trace(go.Scatter(
        x=strengths, y=means,
        mode='lines+markers',
        marker=dict(size=8, color=COLOR_MAIN),
        line=dict(color=COLOR_MAIN, width=2.5),
        error_y=dict(type='data', array=stds, visible=True,
                     thickness=1.8, width=6, color=COLOR_MAIN),
        name='Mean ± Std',
    ))

    fig.update_xaxes(
        title_text='Variation Strength (c)',
        tickvals=[0.0, 0.05, 0.10, 0.15, 0.20, 0.25],
        range=[-0.01, 0.27],
    )
    fig.update_yaxes(
        title_text='Accuracy (%)',
        range=[83.5, 89.8],
        tickvals=[84, 85, 86, 87, 88, 89],
    )

    fig.update_layout(
        title={"text": "Stage I Zoom-in: Gradual Degradation (c \u2264 0.25, Tanh Model)"},
        legend=dict(
            orientation='v',
            x=0.75, y=0.98,
            xanchor='left', yanchor='top',
            bgcolor='rgba(255,255,255,0.85)',
            bordercolor='lightgray', borderwidth=1,
        ),
    )
    fig.update_traces(cliponaxis=False)

    fig.write_image(f"{save_path}/Tanh_D2D_zoom.pdf", format="pdf")
    fig.write_image(f"{save_path}/Tanh_D2D_zoom.png",)

elif SYNAPSE_MODEL == 'Ferroelectric':
    data = {
        0.0:  {"accs": [88.51, 88.35, 88.51, 88.3, 88.44],  "defect": 0.0},
        0.01: {"accs": [88.66, 88.29, 88.87, 88.09, 88.33], "defect": 0.0},
        0.03: {"accs": [88.13, 88.52, 88.33, 88.66, 88.48], "defect": 0.0},
        0.05: {"accs": [88.42, 88.5,  88.38, 88.82, 88.23], "defect": 0.0},
        0.08: {"accs": [88.43, 88.24, 88.3,  88.32, 88.25], "defect": 0.0},
        0.10: {"accs": [88.18, 88.42, 88.59, 87.88, 88.14], "defect": 0.0},
        0.15: {"accs": [88.0,  88.07, 88.43, 88.48, 88.01], "defect": 0.0066964286379516125},
        0.20: {"accs": [88.26, 88.38, 87.85, 88.08, 88.18], "defect": 0.040816325694322586},
        0.25: {"accs": [87.92, 87.66, 88.21, 87.81, 88.08], "defect": 0.09614157676696777},
        0.30: {"accs": [87.33, 87.65, 87.52, 87.59, 87.86, 87.72, 88.01, 87.88, 87.43, 87.88,
                        88.02, 87.82, 87.83, 88.0, 88.0, 87.86, 87.81, 87.53, 87.92, 87.75],
               "defect": 0.16374361515045166},
        0.35: {"accs": [87.74, 87.6, 87.09, 87.46, 88.06, 87.36, 87.69, 87.44, 87.53, 87.44,
                        87.54, 87.53, 87.54, 87.67, 87.57, 87.68, 87.56, 87.24, 87.5, 87.47],
               "defect": 0.23357780277729034},
        0.40: {"accs": [87.44, 87.91, 87.41, 87.65, 87.35, 87.59, 87.79, 87.26, 88.05, 87.04,
                        87.96, 87.58, 87.73, 87.75, 87.4, 87.48, 87.76, 87.55, 87.49, 87.25],
               "defect": 0.30165815353393555},
        0.45: {"accs": [87.35, 87.26, 87.55, 87.38, 87.33, 87.85, 87.31, 87.26, 87.53, 87.48,
                        87.52, 87.36, 87.42, 87.34, 87.21, 87.47, 87.21, 87.79, 87.26, 87.89],
               "defect": 0.3667091727256775},
        0.50: {"accs": [86.62, 87.34, 87.65, 86.86, 87.27, 87.31, 87.67, 87.41, 87.19, 87.41,
                        87.49, 87.05, 87.2, 87.4, 87.45, 87.56, 87.1, 87.38, 87.32, 87.23],
               "defect": 0.42984694242477417},
        0.55: {"accs": [87.48, 87.32, 87.58, 87.16, 86.99, 87.15, 87.12, 87.27, 86.82, 87.24,
                        87.21, 87.22, 87.12, 87.29, 87.43, 87.23, 87.26, 87.21, 87.46, 87.65],
               "defect": 0.48628824949264526},
    }

    # ── sfp values (sfd = 1.90, fixed) ──────────────────────────────────────────
    sfp_data = {
        0.0:  1.0258,
        0.01: 1.0306,
        0.03: 1.0315,
        0.05: 1.0316,
        0.08: 1.0321,
        0.10: 1.0335,
        0.15: 1.0338,
        0.20: 1.0350,
        0.25: 1.0358,
        0.30: 1.0379,
        0.35: 1.0367,
        0.40: 1.0364,
        0.45: 1.0380,
        0.50: 1.0402,
        0.55: 1.0403,
    }

    strengths_all = sorted(data.keys())
    means    = [float(np.mean(data[s]["accs"])) for s in strengths_all]
    stds     = [float(np.std(data[s]["accs"]))  for s in strengths_all]
    defect_ys = [data[s]["defect"] * 100        for s in strengths_all]
    sfp_ys   = [sfp_data[s]                     for s in strengths_all]

    # Stage I for zoom
    stage1_keys = [s for s in strengths_all if s <= 0.25]
    s1_means = [float(np.mean(data[s]["accs"])) for s in stage1_keys]
    s1_stds  = [float(np.std(data[s]["accs"]))  for s in stage1_keys]
    s1_mins  = [float(np.min(data[s]["accs"]))  for s in stage1_keys]
    s1_maxs  = [float(np.max(data[s]["accs"]))  for s in stage1_keys]

    # ── Colors ───────────────────────────────────────────────────────────────────
    COLOR_MAIN = '#4C9BE8'
    COLOR_DEF  = '#7B61FF'
    COLOR_SFP  = '#E8A14C'
    COLOR_FILL = 'rgba(76,155,232,0.18)'
    COLOR_ENV  = '#A8C8F0'

    save_path = "figures"

    # ── Figure 1: full D2D result + sfp ─────────────────────────────────────────
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.06,
        specs=[[{"secondary_y": True}],
               [{"secondary_y": False}]],
    )

    # Stage I std band
    s1_upper = [m + s for m, s in zip(s1_means, s1_stds)]
    s1_lower = [m - s for m, s in zip(s1_means, s1_stds)]
    fig.add_trace(go.Scatter(
        x=stage1_keys + stage1_keys[::-1],
        y=s1_upper + s1_lower[::-1],
        fill='toself', fillcolor=COLOR_FILL,
        line=dict(color='rgba(0,0,0,0)'),
        hoverinfo='skip', showlegend=False,
    ), row=1, col=1, secondary_y=False)

    # Mean accuracy (left y-axis)
    fig.add_trace(go.Scatter(
        x=strengths_all, y=means,
        error_y=dict(type='data', array=stds, visible=True, thickness=1.5, width=5),
        mode='lines+markers',
        marker=dict(size=8, symbol='circle', color=COLOR_MAIN),
        line=dict(color=COLOR_MAIN, width=2.5),
        name='Mean Accuracy',
    ), row=1, col=1, secondary_y=False)

    # sfp curve (right y-axis)
    fig.add_trace(go.Scatter(
        x=strengths_all, y=sfp_ys,
        mode='lines+markers',
        marker=dict(size=7, symbol='square', color=COLOR_SFP),
        line=dict(color=COLOR_SFP, width=2.0, dash='dash'),
        name='s<sub>fp</sub> (sfd = 1.90)',
    ), row=1, col=1, secondary_y=True)

    # Defect ratio bar (bottom panel)
    fig.add_trace(go.Bar(
        x=strengths_all, y=defect_ys,
        marker_color=[COLOR_DEF if d > 0 else 'lightgray' for d in defect_ys],
        marker_line_width=0,
        name='Defect Ratio (%)',
        width=0.018,
    ), row=2, col=1)

    # First defect appearance vline
    for r in [1, 2]:
        fig.add_vline(x=0.15, line_dash='dot', line_color='gray', line_width=1.5, row=r, col=1)
    fig.add_annotation(
        x=0.155, y=88.75,
        text='Defects appear (c=0.15)',
        showarrow=False, font=dict(size=11, color='gray'),
        xanchor='left', row=1, col=1,
    )

    # sfd annotation (top-right corner)
    fig.add_annotation(
        x=0.53, y=1.0430,
        xref='x', yref='y2',
        text='<i>s</i><sub>fd</sub> = 1.90 (fixed)',
        showarrow=False,
        font=dict(size=11, color=COLOR_SFP),
        xanchor='left', yanchor='bottom',
        bgcolor='rgba(255,255,255,0.80)',
        bordercolor=COLOR_SFP,
        borderwidth=1,
    )

    # ── Axes ─────────────────────────────────────────────────────────────────────
    fig.update_xaxes(
        tickvals=[0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55],
        title_text='Variation Strength (c)', row=2, col=1,
    )
    fig.update_yaxes(
        title_text='Accuracy (%)', range=[86.3, 88.95],
        row=1, col=1, secondary_y=False,
    )
    fig.update_yaxes(
        title_text='s<sub>fp</sub>',
        range=[1.020, 1.045],
        tickformat='.4f',
        showgrid=False,
        row=1, col=1, secondary_y=True,
    )
    fig.update_yaxes(title_text='Defect (%)', range=[0, 52], row=2, col=1)

    fig.update_layout(
        title={"text": "D2D Variation: Stable Degradation in Exponential Model (Fashion-MNIST)"},
        legend=dict(
            orientation='v',
            x=0.02, y=0.60,
            xanchor='left', yanchor='top',
            bgcolor='rgba(255,255,255,0.85)',
            bordercolor='lightgray', borderwidth=1,
            font=dict(size=12),
        ),
        bargap=0.3,
    )
    fig.update_traces(selector=dict(type='scatter'), cliponaxis=False)

    fig.write_image(f"{save_path}/Exp_D2D.pdf", format="pdf")
    fig.write_image(f"{save_path}/Exp_D2D.png")

    # ── Figure 2: zoomed Stage I ─────────────────────────────────────────────────
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=stage1_keys, y=s1_maxs,
        mode='lines', line=dict(color=COLOR_ENV, width=1, dash='dot'),
        name='Max', showlegend=True,
    ))
    fig.add_trace(go.Scatter(
        x=stage1_keys, y=s1_mins,
        mode='lines', line=dict(color=COLOR_ENV, width=1, dash='dot'),
        fill='tonexty', fillcolor='rgba(168,200,240,0.18)',
        name='Min (Min–Max Range)', showlegend=True,
    ))
    fig.add_trace(go.Scatter(
        x=stage1_keys, y=s1_means,
        mode='lines+markers',
        marker=dict(size=8, color=COLOR_MAIN),
        line=dict(color=COLOR_MAIN, width=2.5),
        error_y=dict(type='data', array=s1_stds, visible=True,
                     thickness=1.8, width=6, color=COLOR_MAIN),
        name='Mean ± Std',
    ))

    fig.update_xaxes(
        title_text='Variation Strength (c)',
        tickvals=[0.0, 0.05, 0.10, 0.15, 0.20, 0.25],
        range=[-0.01, 0.27],
    )
    fig.update_yaxes(
        title_text='Accuracy (%)',
        range=[87.4, 88.9],
        tickvals=[87.5, 88.0, 88.5],
    )
    fig.update_layout(
        title={"text": "Zoom-in: Gradual Degradation (c \u2264 0.25, Exponential Model)"},
        legend=dict(
            orientation='v',
            x=0.73, y=0.98,
            xanchor='left', yanchor='top',
            bgcolor='rgba(255,255,255,0.85)',
            bordercolor='lightgray', borderwidth=1,
        ),
    )
    fig.update_traces(selector=dict(type='scatter'), cliponaxis=False)

    fig.write_image(f"{save_path}/Exp_D2D_zoom.pdf", format="pdf")
    fig.write_image(f"{save_path}/Exp_D2D_zoom.png")