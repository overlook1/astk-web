"""ASTK web app - figures.

Every function returns a matplotlib Figure built from plain data frames, so the
plots can be unit-checked without ASTK or a browser. Figure text falls back to
ASCII when no CJK font is installed (typical on a bare cloud container).
"""
from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Ellipse

INK = "#12303C"
MUTED = "#5B7280"
ACCENT = "#1B6E8C"
WARM = "#A8531E"
RULE = "#D9E4E9"
CTRL_C = "#1B6E8C"
CASE_C = "#A8531E"

_CJK_CANDIDATES = [
    "Noto Sans CJK SC", "Noto Sans CJK JP", "Noto Sans SC", "Source Han Sans SC",
    "WenQuanYi Zen Hei", "WenQuanYi Micro Hei", "Microsoft YaHei", "SimHei",
    "PingFang SC", "Hiragino Sans GB", "AR PL UMing CN",
]


def _setup_fonts() -> bool:
    available = {f.name for f in font_manager.fontManager.ttflist}
    for cand in _CJK_CANDIDATES:
        if cand in available:
            plt.rcParams["font.sans-serif"] = [cand, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return True
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return False


HAS_CJK = _setup_fonts()


def L(zh: str, en: str) -> str:
    """Pick a Chinese or ASCII label depending on the available fonts."""
    return zh if HAS_CJK else en


def close(fig) -> None:
    """Release a figure once the caller has rendered or saved it."""
    if fig is not None:
        plt.close(fig)


ATTRIBUTION = "ASTK (Huang et al., 2024)  doi:10.1002/aisy.202300594"


def _finish(fig):
    """Shared layout, plus a small citation stamp on every exported figure."""
    fig.tight_layout()
    fig.text(0.995, 0.002, ATTRIBUTION, ha="right", va="bottom",
             fontsize=6, color="#9AAAB2")
    return fig


def _style(ax, grid_axis: str = "y"):
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(RULE)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, axis=grid_axis, color=RULE, linewidth=0.7, alpha=0.9)
    ax.set_axisbelow(True)


def _group_colors(groups: Mapping[str, str]) -> Dict[str, str]:
    return {"ctrl": CTRL_C, "case": CASE_C}


# ------------------------------------------------------------ event counts --

def fig_event_counts(counts: pd.DataFrame, labels: Optional[Mapping[str, str]] = None):
    fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=150)
    labels = labels or {}
    names = list(counts["event_type"])
    text = [labels.get(n, n) for n in names]
    values = counts["n_events"].to_numpy()
    y = np.arange(len(names))
    ax.barh(y, values, color=ACCENT, height=0.62)
    ax.set_yticks(y, text, fontsize=10, color=INK)
    ax.invert_yaxis()
    for yi, v in zip(y, values):
        ax.text(v, yi, f" {v:,}", va="center", fontsize=9, color=MUTED)
    ax.set_xlabel(L("事件数", "events"), fontsize=10, color=MUTED)
    ax.set_title(L("七类可变剪接事件数量", "AS events by type"), fontsize=11, color=INK, loc="left")
    _style(ax, "x")
    return _finish(fig)


# --------------------------------------------------------------------- PCA --

def fig_pca(psi: pd.DataFrame, groups: Mapping[str, str], title: Optional[str] = None,
            annotate: bool = True):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    data = psi.dropna()
    if data.shape[0] < 3 or data.shape[1] < 2:
        return None
    samples = list(data.columns)
    X = StandardScaler().fit_transform(data.T.to_numpy(dtype=float))
    X = np.nan_to_num(X)
    n_comp = min(2, X.shape[0], X.shape[1])
    model = PCA(n_components=n_comp)
    Z = model.fit_transform(X)
    ratio = model.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(5.6, 4.4), dpi=150)
    colors = _group_colors(groups)
    for cond in ("ctrl", "case"):
        idx = [i for i, s in enumerate(samples) if groups.get(s, "ctrl") == cond]
        if not idx:
            continue
        ax.scatter(Z[idx, 0], Z[idx, 1] if n_comp > 1 else np.zeros(len(idx)),
                   s=90, color=colors.get(cond, ACCENT), edgecolor="white", linewidth=1.2,
                   label=cond, zorder=3)
        if len(idx) > 2:
            _draw_ellipse(ax, Z[idx, :2], colors.get(cond, ACCENT))
    if annotate:
        for i, s in enumerate(samples):
            ax.annotate(s, (Z[i, 0], Z[i, 1]), textcoords="offset points", xytext=(6, 5),
                        fontsize=8, color=MUTED)
    ax.set_xlabel(f"PC1 ({ratio[0]:.1f}%)", fontsize=10, color=MUTED)
    ax.set_ylabel(f"PC2 ({ratio[1]:.1f}%)" if n_comp > 1 else "PC2",
                  fontsize=10, color=MUTED)
    ax.set_title(title or L("PSI 的样本主成分分析", "PCA of sample PSI"),
                 fontsize=11, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=MUTED)
    _style(ax)
    return _finish(fig)


def _draw_ellipse(ax, points: np.ndarray, color: str):
    try:
        cov = np.cov(points.T)
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
        width, height = 2 * 1.5 * np.sqrt(np.maximum(vals, 1e-9))
        ax.add_patch(Ellipse(points.mean(axis=0), width, height, angle=angle,
                             facecolor=color, alpha=0.10, edgecolor=color, linewidth=1))
    except Exception:
        pass


# ------------------------------------------------------------ correlations --

def fig_corr_heatmap(psi: pd.DataFrame, groups: Mapping[str, str]):
    data = psi.dropna(how="any")
    if data.shape[1] < 2 or data.shape[0] < 3:
        return None
    corr = data.corr(method="spearman")
    lo = max(0.0, float(np.nanmin(corr.to_numpy())))
    fig, ax = plt.subplots(figsize=(1.0 + 0.55 * corr.shape[0], 1.6 + 0.55 * corr.shape[0]), dpi=150)
    im = ax.imshow(corr.to_numpy(), cmap="RdYlBu_r", vmin=lo, vmax=1.0)
    ax.set_xticks(range(corr.shape[0]), corr.columns, rotation=45, ha="right", fontsize=8, color=MUTED)
    ax.set_yticks(range(corr.shape[0]), corr.index, fontsize=8, color=MUTED)
    for i, s in enumerate(corr.index):
        if groups.get(s) == "case":
            ax.get_yticklabels()[i].set_color(CASE_C)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            v = float(corr.iat[i, j])
            # flip to white text once the cell is dark, otherwise the value is unreadable
            shade = "white" if (v - lo) / max(1.0 - lo, 1e-9) > 0.45 else INK
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color=shade)
    ax.set_title(L("样本间 PSI 相关性（Spearman）", "PSI correlation (Spearman)"),
                 fontsize=11, color=INK, loc="left")
    fig.colorbar(im, ax=ax, shrink=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return _finish(fig)


# ---------------------------------------------------------------- heatmaps --

def fig_psi_heatmap(psi: pd.DataFrame, groups: Mapping[str, str], top_n: int = 40):
    data = psi.dropna(how="any")
    if data.shape[0] < 3 or data.shape[1] < 2:
        return None
    var = data.var(axis=1).sort_values(ascending=False)
    top = data.loc[var.index[: min(top_n, data.shape[0])]]
    z = top.sub(top.mean(axis=1), axis=0).div(top.std(axis=1).replace(0, np.nan), axis=0).fillna(0)

    fig, ax = plt.subplots(figsize=(1.0 + 0.5 * z.shape[1], 1.8 + 0.16 * z.shape[0]), dpi=150)
    im = ax.imshow(z.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
    ax.set_xticks(range(z.shape[1]), z.columns, rotation=45, ha="right", fontsize=8, color=MUTED)
    for i, s in enumerate(z.columns):
        if groups.get(s) == "case":
            ax.get_xticklabels()[i].set_color(CASE_C)
    ax.set_yticks([])
    ax.set_ylabel(L(f"波动最大的 {z.shape[0]} 个事件", f"top {z.shape[0]} variable events"),
                  fontsize=9, color=MUTED, labelpad=8)
    ax.set_title(L("事件 PSI 热图（按行 z-score）", "PSI heatmap (row z-score)"),
                 fontsize=11, color=INK, loc="left")
    fig.colorbar(im, ax=ax, shrink=0.7, label="z-score")
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    return _finish(fig)


# ---------------------------------------------------------------- volcano --

def fig_volcano(dpsi: pd.DataFrame, abs_dpsi: float = 0.1, qval: float = 0.05,
                event_type: Optional[str] = None, top_label: int = 0):
    data = dpsi.dropna(subset=["pval"]).copy()
    if data.empty:
        return None
    data["neglog"] = -np.log10(data["pval"].clip(lower=1e-300))
    sig = (data["qval"] <= qval) & (data["dpsi"].abs() >= abs_dpsi)
    fig, ax = plt.subplots(figsize=(5.8, 4.4), dpi=150)
    ax.scatter(data.loc[~sig, "dpsi"], data.loc[~sig, "neglog"], s=12, color="#B9C7CE",
               label=L("不显著", "not significant"), zorder=2)
    ax.scatter(data.loc[sig, "dpsi"], data.loc[sig, "neglog"], s=16, color=WARM,
               label=L("显著", "significant"), zorder=3)
    for x in (-abs_dpsi, abs_dpsi):
        ax.axvline(x, color=RULE, linewidth=1, linestyle="--")
    if np.isfinite(data["neglog"]).any():
        cut = -np.log10(max(qval, 1e-300))
        if cut < np.nanmax(data["neglog"]):
            ax.axhline(cut, color=RULE, linewidth=1, linestyle="--")
    if top_label:
        top = data.loc[sig].reindex(data.loc[sig, "dpsi"].abs().sort_values(ascending=False).index)[:top_label]
        for eid, row in top.iterrows():
            ax.annotate(str(eid).split(":")[2] if ":" in str(eid) else str(eid)[:12],
                        (row["dpsi"], row["neglog"]), fontsize=7, color=MUTED,
                        textcoords="offset points", xytext=(4, 3))
    ax.set_xlabel("dPSI (case - ctrl)", fontsize=10, color=MUTED)
    ax.set_ylabel("-log10 p-value", fontsize=10, color=MUTED)
    title = L("火山图", "Volcano") + (f" - {event_type}" if event_type else "")
    ax.set_title(title, fontsize=11, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=MUTED, markerscale=2.5)
    _style(ax, "both")
    return _finish(fig)


# ---------------------------------------------------------------- scatter --

def fig_scatter(dpsi: pd.DataFrame, event_type: Optional[str] = None,
                abs_dpsi: float = 0.1, qval: float = 0.05, max_points: int = 4000):
    data = dpsi.dropna(subset=["mean_ctrl", "mean_case"])
    if data.empty:
        return None
    if data.shape[0] > max_points:
        data = data.sample(max_points, random_state=0)
    sig = (data["qval"] <= qval) & (data["dpsi"].abs() >= abs_dpsi)
    fig, ax = plt.subplots(figsize=(4.9, 4.6), dpi=150)
    ax.plot([0, 1], [0, 1], color=RULE, linewidth=1, linestyle="--", zorder=1)
    ax.scatter(data.loc[~sig, "mean_ctrl"], data.loc[~sig, "mean_case"], s=11,
               color="#B9C7CE", alpha=0.85, zorder=2, label=L("不显著", "not significant"))
    ax.scatter(data.loc[sig, "mean_ctrl"], data.loc[sig, "mean_case"], s=14,
               color=WARM, alpha=0.9, zorder=3, label=L("显著", "significant"))
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("PSI (ctrl)", fontsize=10, color=MUTED)
    ax.set_ylabel("PSI (case)", fontsize=10, color=MUTED)
    title = L("两组 PSI 散点", "PSI scatter") + (f" - {event_type}" if event_type else "")
    ax.set_title(title, fontsize=11, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=MUTED, loc="upper left", markerscale=2.5)
    _style(ax, "both")
    return _finish(fig)


# ----------------------------------------------------------------- violins --

def fig_psi_distribution(psi: Mapping[str, pd.DataFrame], groups: Mapping[str, str],
                         max_points: int = 8000):
    rows = []
    for et, df in psi.items():
        long = df.melt(value_name="psi", ignore_index=False).dropna()
        long = long.rename(columns={"variable": "sample"})
        long["event_type"] = et
        if long.shape[0] > max_points:
            long = long.sample(max_points, random_state=0)
        rows.append(long)
    if not rows:
        return None
    data = pd.concat(rows)
    data["condition"] = data["sample"].map(lambda s: groups.get(s, "ctrl"))
    order = [et for et in ["SE", "A5", "A3", "MX", "RI", "AF", "AL"] if et in set(data["event_type"])]

    fig, ax = plt.subplots(figsize=(8.4, 3.6), dpi=150)
    positions, labels = [], []
    width = 0.34
    for i, et in enumerate(order):
        for k, cond in enumerate(("ctrl", "case")):
            sub = data[(data["event_type"] == et) & (data["condition"] == cond)]["psi"]
            if sub.empty:
                continue
            pos = i + (k - 0.5) * width
            parts = ax.violinplot(sub.to_numpy(), positions=[pos], widths=width * 0.92,
                                  showextrema=False, showmedians=False)
            for body in parts["bodies"]:
                body.set_facecolor(CTRL_C if cond == "ctrl" else CASE_C)
                body.set_alpha(0.45)
                body.set_edgecolor("white")
            ax.scatter([pos], [sub.median()], s=14, color=INK, zorder=4)
        positions.append(i)
        labels.append(et)
    ax.set_xticks(positions, labels, fontsize=10, color=INK)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("PSI", fontsize=10, color=MUTED)
    ax.set_title(L("各事件类型的 PSI 分布（按组）", "PSI distribution by event type"),
                 fontsize=11, color=INK, loc="left")
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=CTRL_C, label="ctrl"),
               plt.Line2D([], [], marker="o", linestyle="", color=CASE_C, label="case")]
    ax.legend(handles=handles, frameon=False, fontsize=9, labelcolor=MUTED)
    _style(ax)
    return _finish(fig)
