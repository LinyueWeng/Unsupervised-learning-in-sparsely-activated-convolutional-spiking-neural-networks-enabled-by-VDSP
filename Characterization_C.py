import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import binned_statistic

warnings.filterwarnings("ignore", category=RuntimeWarning)


def tanh_model(x, s_amp, v0, voff, soff):
    return s_amp * np.tanh((x - voff) / v0) + soff


def safe_read_csv(path: Path) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # 兼容你这个文件第一行是 "# pulseAmplitude,deltaCpos...,CposInitial"
            if line.startswith("#"):
                line = line[1:].strip()
                if "pulseAmplitude" in line:
                    continue

            parts = [x.strip() for x in line.split(",")]
            if len(parts) < 3:
                continue

            rows.append(parts[:3])

    df = pd.DataFrame(rows, columns=["pulseAmplitude", "deltaState", "stateInitial"])
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna().reset_index(drop=True)

    if len(df) == 0:
        raise ValueError("CSV 读入后为空，请检查文件格式。")

    return df


def detect_polarity(voltage: pd.Series) -> str:
    has_pos = (voltage > 0).any()
    has_neg = (voltage < 0).any()
    if has_pos and has_neg:
        return "bipolar"
    if has_pos:
        return "positive_only"
    return "negative_only"


def normalize_state(df: pd.DataFrame):
    out = df.copy()

    out["stateFinal"] = out["stateInitial"] + out["deltaState"]
    out["V"] = out["pulseAmplitude"]

    smin = min(out["stateInitial"].min(), out["stateFinal"].min())
    smax = max(out["stateInitial"].max(), out["stateFinal"].max())

    state_range = smax - smin
    print(f"state min = {smin:.6e}, state max = {smax:.6e}, range = {state_range:.6e}")

    if state_range <= 0:
        raise ValueError(
            "stateInitial/stateFinal 完全没有范围变化，无法归一化。"
            "请检查 CSV 是否正确读入。"
        )

    out["w_init"] = (out["stateInitial"] - smin) / state_range
    out["w_final"] = (out["stateFinal"] - smin) / state_range
    out["dw"] = out["w_final"] - out["w_init"]

    meta = {
        "state_min_raw": float(smin),
        "state_max_raw": float(smax),
        "polarity_mode": detect_polarity(out["V"]),
    }
    return out, meta


def build_envelope(df: pd.DataFrame, nbins: int = 40) -> pd.DataFrame:
    V = df["V"].to_numpy()
    S = df["stateFinal"].to_numpy()

    V_mean, _, _ = binned_statistic(V, V, statistic="mean", bins=nbins)
    S_low, _, _ = binned_statistic(V, S, statistic="min", bins=nbins)
    S_high, _, _ = binned_statistic(V, S, statistic="max", bins=nbins)
    counts, _, _ = binned_statistic(V, S, statistic="count", bins=nbins)

    env = pd.DataFrame({
        "V_bin": V_mean,
        "state_low": S_low,
        "state_high": S_high,
        "count": counts,
    })

    env = env.dropna().reset_index(drop=True)

    if len(env) < 6:
        raise ValueError("有效 envelope 点太少，无法稳健拟合。请减少 nbins 或检查数据。")

    return env


def fit_envelopes(env: pd.DataFrame) -> dict:
    V = env["V_bin"].to_numpy()
    low = env["state_low"].to_numpy()
    high = env["state_high"].to_numpy()

    s_amp0 = max((np.max(low) - np.min(low)) / 2, 1e-18)
    soff0 = np.mean(low)
    vspan = max(np.ptp(V), 1e-3)

    p0_low = [s_amp0, 0.2 * vspan, np.median(V), soff0]
    low_params, _ = curve_fit(tanh_model, V, low, p0=p0_low, maxfev=20000)

    # 高包络独立拟合
    s_amp1 = max((np.max(high) - np.min(high)) / 2, 1e-18)
    soff1 = np.mean(high)
    p0_high = [s_amp1, 0.2 * vspan, np.median(V), soff1]
    high_params, _ = curve_fit(tanh_model, V, high, p0=p0_high, maxfev=20000)

    low_s_amp, low_v0, low_voff, low_soff = low_params
    high_s_amp, high_v0, high_voff, high_soff = high_params

    state_min_fit = min(low_soff - abs(low_s_amp), high_soff - abs(high_s_amp))
    state_max_fit = max(low_soff + abs(low_s_amp), high_soff + abs(high_s_amp))

    return {
        "state_min_fit": float(state_min_fit),
        "state_max_fit": float(state_max_fit),
        "low_s_amp": float(low_s_amp),
        "low_v0": float(low_v0),
        "low_voff": float(low_voff),
        "low_soff": float(low_soff),
        "high_s_amp": float(high_s_amp),
        "high_v0": float(high_v0),
        "high_voff": float(high_voff),
        "high_soff": float(high_soff),
    }


def make_plots(df: pd.DataFrame, env: pd.DataFrame, params: dict, outdir: Path, stem: str):
    outdir.mkdir(parents=True, exist_ok=True)

    v_grid = np.linspace(df["V"].min(), df["V"].max(), 500)
    low_fit = tanh_model(
        v_grid,
        params["low_s_amp"], params["low_v0"], params["low_voff"], params["low_soff"]
    )
    high_fit = tanh_model(
        v_grid,
        params["high_s_amp"], params["high_v0"], params["high_voff"], params["high_soff"]
    )

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(
        df["V"], df["stateFinal"],
        c=df["deltaState"], cmap="coolwarm", s=10, alpha=0.35
    )
    plt.colorbar(sc, label="delta state")
    plt.scatter(env["V_bin"], env["state_low"], c="green", s=28, label="lower envelope")
    plt.scatter(env["V_bin"], env["state_high"], c="purple", s=28, label="upper envelope")
    plt.plot(v_grid, low_fit, c="red", lw=2, label="low tanh fit")
    plt.plot(v_grid, high_fit, c="black", lw=2, label="high tanh fit")
    plt.xlabel("Pulse amplitude (V)")
    plt.ylabel("Final state")
    plt.title("Capacitance-state envelope characterization")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(outdir / f"{stem}_state_envelope.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.scatter(df["V"], df["dw"], s=10, alpha=0.35, color="tab:blue")
    plt.axhline(0, color="k", lw=1)
    plt.xlabel("Pulse amplitude (V)")
    plt.ylabel("Normalized state increment dw")
    plt.title("Normalized update map")
    plt.tight_layout()
    plt.savefig(outdir / f"{stem}_dw_map.png", dpi=300, bbox_inches="tight")
    plt.close()


def summarize(df: pd.DataFrame, meta: dict, params: dict) -> pd.DataFrame:
    pos = df[df["V"] > 0]
    neg = df[df["V"] < 0]

    return pd.DataFrame([{
        "n_rows": int(len(df)),
        "n_unique_amplitudes": int(df["V"].nunique()),
        "polarity_mode": meta["polarity_mode"],
        "state_min_raw": meta["state_min_raw"],
        "state_max_raw": meta["state_max_raw"],
        "mean_dw_all": float(df["dw"].mean()),
        "mean_dw_pos": float(pos["dw"].mean()) if len(pos) else np.nan,
        "mean_dw_neg": float(neg["dw"].mean()) if len(neg) else np.nan,
        **params
    }])

def main():
    input_path = Path(r"C:\Users\28218\PycharmProjects\CSNN\data\FeCAPs.csv")
    outdir = Path(r"C:\Users\28218\PycharmProjects\CSNN\figures\FeCAPs_charac")
    nbins_user = 40

    stem = input_path.stem

    raw = safe_read_csv(input_path)
    print(raw.head())
    print(raw.describe())
    norm, meta = normalize_state(raw)

    nbins = min(nbins_user, max(12, norm["V"].nunique()))
    env = build_envelope(norm, nbins=nbins)
    params = fit_envelopes(env)
    summary = summarize(norm, meta, params)

    outdir.mkdir(parents=True, exist_ok=True)
    norm.to_csv(outdir / f"{stem}_normalized.csv", index=False)
    env.to_csv(outdir / f"{stem}_envelope.csv", index=False)
    summary.to_csv(outdir / f"{stem}_characterization_summary.csv", index=False)

    make_plots(norm, env, params, outdir, stem)

    print("Done.")
    print("Normalized data  ->", outdir / f"{stem}_normalized.csv")
    print("Envelope data    ->", outdir / f"{stem}_envelope.csv")
    print("Summary params   ->", outdir / f"{stem}_characterization_summary.csv")
    print("Figure 1         ->", outdir / f"{stem}_state_envelope.png")
    print("Figure 2         ->", outdir / f"{stem}_dw_map.png")


if __name__ == "__main__":
    main()