"""Plot saved real/generated states without bundling observations in this repository.

Inputs stay local. Only figures are exported; optional --audit records source hashes,
full-sample metrics and deterministic illustration selections in an ignored directory.
No training, resampling from a model, or alteration of the saved paths is performed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from common import BINS, TARGETS, atomic_json, metrics, sha256
from fidelity import kl_on_reference, state_features

SEED_COLORS = ("#0072B2", "#D55E00", "#009E73")
LEVEL_COLORS = ("#0072B2", "#56B4E9", "#009E73", "#E69F00", "#D55E00")
FEATURE_LABELS = {
    "DayR": ("Daily returns (DayR)", "Daily log return"),
    "RetAC": ("Return autocorrelation (RetAC)", "Return ACF, lag 1 day"),
    "VolC": ("Volatility clustering (VolC)", "Squared-return correlation, lag 1 day"),
    "LogV": ("Log-volume (LogV)", "log(1 + daily volume), source units"),
}
KDE_LIMIT = 40000


def load_states(path):
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path) as archive:
            states = archive["states"]
    else:
        states = np.load(path, mmap_mode="r")
    if states.ndim != 3 or states.shape[1:] != (2, 60):
        raise ValueError(f"Expected (N,2,60) states: {path.name}")
    if len(states) < 2 or not np.isfinite(states).all():
        raise ValueError(f"Need at least two finite paths: {path.name}")
    return states


def kde(ax, values, grid, color, label=None, ls="-", lw=1.8):
    values = np.asarray(values, dtype=np.float64)
    if len(values) > KDE_LIMIT:
        indices = np.random.default_rng(421).choice(len(values), KDE_LIMIT, replace=False)
        values = values[indices]
    if values.std() < 1e-12:
        ax.axvline(values.mean(), color=color, ls=ls, lw=lw, label=label)
    else:
        ax.plot(grid, gaussian_kde(values)(grid), color=color, ls=ls, lw=lw, label=label)


def grid_for(arrays):
    lo = min(float(np.quantile(a, .001)) for a in arrays)
    hi = max(float(np.quantile(a, .999)) for a in arrays)
    if hi <= lo:
        lo, hi = lo - 1e-5, hi + 1e-5
    return np.linspace(lo, hi, 400)


def save(fig, output, name):
    fig.savefig(output / f"{name}.png", dpi=300, bbox_inches="tight")
    # Vector copies are for local export; *.pdf is excluded from version control.
    fig.savefig(output / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def density_figure(real_features, generated_features, output, bounds, counts):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    scores = {}
    for ax, (key, (title, xlabel)) in zip(axes.flat, FEATURE_LABELS.items()):
        arrays = [real_features[key]] + [g[key] for g in generated_features]
        grid = grid_for(arrays)
        kde(ax, arrays[0], grid, "#202020", "Real test", lw=2.3)
        scores[key] = []
        for seed, (g, color) in enumerate(zip(generated_features, SEED_COLORS)):
            kde(ax, g[key], grid, color, f"Generated / seed {seed}", ls=("-", "--", "-.")[seed])
            scores[key].append(kl_on_reference(real_features[key], g[key]))
        values = np.array([s["kl_real_to_generated"] for s in scores[key]])
        ax.set(title=title, xlabel=xlabel, ylabel="Density", xlim=(grid[0], grid[-1]), ylim=(0, None))
        ax.text(.97, .96, f"KL = {values.mean():.4f} +/- {values.std(ddof=1):.4f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": .85})
        ax.grid(alpha=.14)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(.5, .025))
    fig.suptitle("Generated vs. real market distributions", fontsize=16, weight="bold", y=.99)
    fig.text(.5, .94, f"Unconditional | {bounds} | {counts[0]:,} real test windows | "
             f"3 seeds x {counts[1]:,} generated paths", ha="center", fontsize=10, color="#475569")
    fig.text(.5, .005, "KDE for display; KL on full samples and real-range 50-bin histograms. "
             "KL labels: mean +/- sample SD across seeds.", ha="center", fontsize=8.5, color="#475569")
    fig.tight_layout(rect=(0, .08, 1, .92))
    save(fig, output, "real_vs_generated_density")
    return scores


def path_figure(real, generated, output, bounds):
    # Fixed positions, not selected for return, volatility, visual appeal or similarity.
    if len(real) < 6:
        raise ValueError("Need at least six real paths for the fixed illustration layout")
    real_indices = np.random.default_rng(20261103).choice(len(real), 6, replace=False)
    real_examples = real[real_indices]
    generated_examples = np.concatenate([g[:2] for g in generated])
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.4), sharex=True, sharey="row")
    palette = ("#0072B2", "#56B4E9", "#D55E00", "#E69F00", "#009E73", "#CC79A7")
    for col, examples in enumerate([real_examples, generated_examples]):
        for i, (state, color) in enumerate(zip(examples, palette)):
            price = 100 * np.exp(np.r_[0., state[0].astype(float).cumsum()])
            volume = np.expm1(state[1].astype(float))
            if np.any(volume <= 0):
                raise ValueError("Relative-volume illustration requires positive volumes")
            axes[0, col].plot(np.arange(61), price, color=color, lw=1.1, alpha=.85,
                              label=f"Seed {i // 2}, path {i % 2}" if col else f"Example {i + 1}")
            axes[1, col].plot(np.arange(1, 61), volume / np.median(volume), color=color, lw=1.1, alpha=.8)
        axes[0, col].set_title("Real test examples" if col == 0 else "Generated examples", weight="bold")
        axes[0, col].set_yscale("log")
        axes[1, col].set_yscale("log")
        axes[0, col].axhline(100, color="#94a3b8", lw=.7, ls=":")
        axes[1, col].axhline(1, color="#94a3b8", lw=.7, ls=":")
        axes[1, col].set_xlabel("Trading day")
        axes[0, col].legend(fontsize=7.5, ncol=2, frameon=False)
        for ax in axes[:, col]:
            ax.grid(alpha=.14)
            ax.set_xlim(0, 60)
    axes[0, 0].set_ylabel("Price index (start = 100; log scale)")
    axes[1, 0].set_ylabel("Volume / path median (log scale)")
    fig.suptitle("60-day price and volume paths", fontsize=16, weight="bold", y=.99)
    fig.text(.5, .94, f"Unconditional | {bounds} | Same-colored price and volume curves belong to the same path",
             ha="center", fontsize=9.5, color="#475569")
    fig.text(.5, .008, "Real: 6 fixed-random test windows. Generated: first 2 paths per seed. "
             "Paths are illustrative, not paired predictions. Volume rescaled only for this view.",
             ha="center", fontsize=8.3, color="#475569")
    fig.tight_layout(rect=(0, .04, 1, .92))
    save(fig, output, "market_path_examples")
    return {"real_window_indices": real_indices.tolist(), "generated_indices_per_seed": [0, 1],
            "price_transform": "100*exp(cumsum(log_return)); initial point 100",
            "volume_display_only": "raw volume / median(raw volume) of the same path"}


def clipping_density(real_features, before_features, after_features, output, before_bounds, bounds):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    for ax, key in zip(axes, ("DayR", "LogV")):
        before = np.concatenate([g[key] for g in before_features])
        after = np.concatenate([g[key] for g in after_features])
        grid = grid_for([real_features[key], before, after])
        kde(ax, real_features[key], grid, "#202020", "Real test", lw=2.4)
        kde(ax, before, grid, "#999999", f"Before ({before_bounds})", ls="--")
        kde(ax, after, grid, "#0072B2", f"Selected ({bounds})")
        ax.set(title=FEATURE_LABELS[key][0], xlabel=FEATURE_LABELS[key][1], ylabel="Density",
               xlim=(grid[0], grid[-1]), ylim=(0, None))
        ax.grid(alpha=.14)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(.5, .02))
    fig.suptitle("Sampling bounds: closer return and volume distributions", fontsize=15, weight="bold", y=1)
    n = len(after_features[0]["RetAC"])
    fig.text(.5, .005, f"Pooled KDE: all 3 seeds, {n:,} paths each. Same weights and paired noise. "
             "KL is evaluated separately per seed.", ha="center", fontsize=8.5, color="#475569")
    fig.tight_layout(rect=(0, .11, 1, .95))
    save(fig, output, "clipping_density_comparison")


def control_figure(real, metadata, conditional_root, output, bounds):
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 7.2))
    records, sources = [], []
    real_metrics = metrics(real[:, 0, :])
    for j, target in enumerate(TARGETS):
        all_values = []
        for k, level in enumerate(BINS):
            values = []
            for seed in range(3):
                source = conditional_root / f"seed_{seed}/evaluation/samples_{target}_continuous_{level}.npz"
                states = load_states(source)
                sources.append({"file": source.name, "seed": seed, "sha256": sha256(source)})
                value = metrics(states[:, 0, :])[:, j]
                wanted = metadata["targets"][target]["medians"][k]
                records.append({"seed": seed, "target": target, "bin": level, "n": len(states),
                                "target_value": wanted, "mean": float(value.mean()),
                                "mse": float(np.mean((value - wanted) ** 2))})
                values.append(value)
                if k in (0, 2, 4):
                    for state in states[:2]:
                        index = 100 * np.exp(np.r_[0., state[0].astype(float).cumsum()])
                        axes[0, j].plot(np.arange(61), index, color=LEVEL_COLORS[k], alpha=.45, lw=.85)
            all_values.append(np.concatenate(values))
            if k in (0, 2, 4):
                axes[0, j].plot([], [], color=LEVEL_COLORS[k], label=level)
        grid = grid_for([real_metrics[:, j]] + all_values)
        kde(axes[1, j], real_metrics[:, j], grid, "#202020", "Real test (all regimes)", lw=2.2, ls="--")
        for k, (level, values) in enumerate(zip(BINS, all_values)):
            kde(axes[1, j], values, grid, LEVEL_COLORS[k], level)
        xlabel = ("60-day cumulative log return", "Log-price range over 60 days", "Daily log-return standard deviation")[j]
        axes[0, j].set(title=target.capitalize() + " control", xlabel="Trading day",
                       ylabel="Price index (log scale)", yscale="log", xlim=(0, 60))
        axes[1, j].set(xlabel=xlabel, ylabel="Density", xlim=(grid[0], grid[-1]), ylim=(0, None))
        axes[0, j].legend(frameon=False, fontsize=8, ncol=3)
        axes[1, j].legend(frameon=False, fontsize=7.5, ncol=2)
        for ax in axes[:, j]:
            ax.grid(alpha=.14)
    fig.suptitle("Five-level continuous control: return, amplitude and volatility", fontsize=16, weight="bold", y=.995)
    counts = {r["n"] for r in records}
    if len(counts) != 1:
        raise ValueError("Expected equal path counts for all control levels and seeds")
    n = counts.pop()
    fig.text(.5, .946, f"Historical control evaluation | {bounds} | All 3 generator seeds | "
             f"Density: {n:,} paths per level and seed", ha="center", fontsize=10, color="#475569")
    fig.text(.5, .006, "Top: first 2 paths per seed at Lower / Medium / Higher. "
             "Bottom: all 5 levels. Black curve: unconditional real test distribution, not a per-level match target.",
             ha="center", fontsize=8.3, color="#475569")
    fig.tight_layout(rect=(0, .04, 1, .93))
    save(fig, output, "continuous_control_scenarios")
    return {"mse_records": records, "sources": sources}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", type=Path, required=True)
    parser.add_argument("--generated", type=Path, nargs=3, required=True, metavar="SEED_NPZ")
    parser.add_argument("--before", type=Path, nargs=3, metavar="SEED_NPZ")
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--conditional-root", type=Path)
    parser.add_argument("--bounds-label", default="r6 / v3")
    parser.add_argument("--before-bounds-label", default="r6 / v6")
    parser.add_argument("--control-bounds-label", default="r6 / v6")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args()
    if bool(args.conditional_root) != bool(args.metadata):
        parser.error("--conditional-root and --metadata must be specified together")
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titleweight": "bold",
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": "#334155", "text.color": "#0f172a", "figure.facecolor": "white",
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    real = load_states(args.real)
    generated = [load_states(p) for p in args.generated]
    rf = state_features(real)
    gf = [state_features(g) for g in generated]
    if len({len(g) for g in generated}) != 1:
        raise ValueError("Expected the same generated path count for all three seeds")
    print("Rendering real/generated density and path illustrations", flush=True)
    audit = {"real_sha256": sha256(args.real), "real_paths": len(real),
             "generated_sha256": [sha256(p) for p in args.generated],
             "paths_per_seed": [len(g) for g in generated], "bounds_label": args.bounds_label,
             "kde_max_observations": KDE_LIMIT, "kde_rng_seed": 421,
             "kl_uses_full_samples": True}
    audit["kl_by_seed"] = density_figure(rf, gf, args.output, args.bounds_label, (len(real), len(generated[0])))
    audit["path_illustrations"] = path_figure(real, generated, args.output, args.bounds_label)
    if args.before:
        print("Rendering before/after clipping densities", flush=True)
        before = [load_states(p) for p in args.before]
        if [len(g) for g in before] != [len(g) for g in generated]:
            raise ValueError("Before/after path counts must match")
        bf = [state_features(g) for g in before]
        audit["before_sha256"] = [sha256(p) for p in args.before]
        audit["before_kl_by_seed"] = {
            k: [kl_on_reference(rf[k], b[k]) for b in bf] for k in FEATURE_LABELS}
        clipping_density(rf, bf, gf, args.output, args.before_bounds_label, args.bounds_label)
    if args.conditional_root:
        print("Rendering historical five-level control scenarios", flush=True)
        metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
        audit["metadata_sha256"] = sha256(args.metadata)
        audit["historical_control"] = control_figure(real, metadata, args.conditional_root,
                                                      args.output, args.control_bounds_label)
    if args.audit:
        atomic_json(args.audit, audit)
    print("REPORT_DISTRIBUTIONS_RENDERED", flush=True)


if __name__ == "__main__":
    main()
