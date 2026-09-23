"""ASTK web app - figures.

Every function returns a matplotlib Figure built from plain data frames, so the
plots can be unit-checked without ASTK or a browser. Figure text falls back to
ASCII when no CJK font is installed (typical on a bare cloud container).
"""
from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

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
# 分组配色：颜色顺序固定，同一次分析里同一组在所有图中颜色一致。
PALETTE = ["#1B6E8C", "#A8531E", "#4E7A3A", "#8A4C93", "#B5893B", "#3D6BB3",
           "#A03B5A", "#5B7280"]
CTRL_C = PALETTE[0]   # 兼容：默认第一组 = 第一个颜色
CASE_C = PALETTE[1]

# 事件类型的展示顺序
AS_ORDER = ["SE", "A5", "A3", "MX", "RI", "AF", "AL"]

# 事件类型标签。有中文字体就用中文，没有就用 ASCII。
# 语言必须由绘图函数自己决定：上游把中文标签直接传进来，会在没有中文字体的
# 容器里渲染成方框（Glyph missing），而同一张图其它文字都是英文。
_EVENT_LABELS = {
    "SE": ("外显子跳跃", "skipped exon"),
    "A5": ("可变 5' 剪接位点", "alt 5' splice site"),
    "A3": ("可变 3' 剪接位点", "alt 3' splice site"),
    "MX": ("互斥外显子", "mutually exclusive exons"),
    "RI": ("内含子保留", "intron retention"),
    "AF": ("可变首个外显子", "alternative first exon"),
    "AL": ("可变末端外显子", "alternative last exon"),
}

_CJK_CANDIDATES = [
    "Noto Sans CJK SC", "Noto Sans CJK JP", "Noto Sans SC", "Source Han Sans SC",
    "WenQuanYi Zen Hei", "WenQuanYi Micro Hei", "Microsoft YaHei", "SimHei",
    "PingFang SC", "Hiragino Sans GB", "AR PL UMing CN",
]


def _renders_cjk(family: str) -> bool:
    """这个字体到底能不能画出汉字。

    只看字体名不够：名字匹配上、字形却缺失时，matplotlib 会静默输出方框
    （tofu），这正是云端部署图里标题变方框的原因。这里用 FreeType 直接查字形
    索引，查不到（0 = .notdef）就判定不能用。
    """
    try:
        from matplotlib.font_manager import FontProperties
        from matplotlib.ft2font import FT2Font
        path = font_manager.findfont(FontProperties(family=family), fallback_to_default=False)
        face = FT2Font(path)
        return all(face.get_char_index(ord(ch)) != 0 for ch in "中文事件外显子修饰组")
    except Exception:
        return False


def _setup_fonts() -> bool:
    """True when a CJK font was found and it really has the glyphs.

    Never raises: a font or font-cache problem on the host must not take the
    whole app down - the labels just fall back to ASCII.
    """
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    try:
        available = {f.name for f in font_manager.fontManager.ttflist}
    except Exception:
        return False
    for cand in _CJK_CANDIDATES:
        if cand in available and _renders_cjk(cand):
            plt.rcParams["font.sans-serif"] = [cand, "DejaVu Sans"]
            return True
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


def event_label(event_type) -> str:
    """事件类型标签（自带中英文切换，不要把中文标签从外面传进来）。"""
    code = str(event_type)
    pair = _EVENT_LABELS.get(code)
    if pair is None:
        return code
    return f"{code} {pair[0]}" if HAS_CJK else f"{code} ({pair[1]})"


def group_order(groups: Mapping[str, str]) -> List[str]:
    """分组名，按名字排序 —— 保证同一次分析里各图配色一致。"""
    return sorted({str(v).strip() for v in groups.values() if str(v).strip()})


def _cond_of(sample: str, groups: Mapping[str, str]) -> str:
    return str(groups.get(sample, "ungrouped")).strip() or "ungrouped"


def _group_colors(groups: Mapping[str, str]) -> Dict[str, str]:
    return {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(group_order(groups))}


def _dpsi_axes(dpsi: pd.DataFrame) -> Tuple[str, str, str]:
    """从 dPSI 表里取 (参考组, 比较组, 比较对名)，取不到就给通用名字。"""
    def first(col: str, default: str) -> str:
        if col in dpsi.columns:
            vals = dpsi[col].dropna()
            if not vals.empty:
                return str(vals.iloc[0])
        return default

    ref, test = first("reference", "ref"), first("test", "test")
    comp = first("comparison", f"{ref}_vs_{test}")
    return ref, test, comp

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
    # 遍历样本里真实出现过的分组，而不是写死 ctrl/case：
    # 之前只画两组，第三组及以后的样本会被静默漏掉（图还照常出，很容易漏看）。
    conds: List[str] = []
    for s in samples:
        c = _cond_of(s, groups)
        if c not in conds:
            conds.append(c)
    for i, c in enumerate(conds):
        colors.setdefault(c, PALETTE[i % len(PALETTE)])
    for cond in conds:
        idx = [i for i, s in enumerate(samples) if _cond_of(s, groups) == cond]
        ax.scatter(Z[idx, 0], Z[idx, 1] if n_comp > 1 else np.zeros(len(idx)),
                   s=90, color=colors[cond], edgecolor="white", linewidth=1.2,
                   label=f"{cond} (n={len(idx)})", zorder=3)
        if len(idx) > 2:
            _draw_ellipse(ax, Z[idx, :2], colors[cond])
    if annotate:
        for i, s in enumerate(samples):
            ax.annotate(s, (Z[i, 0], Z[i, 1]), textcoords="offset points", xytext=(6, 5),
                        fontsize=8, color=MUTED)
    ax.set_xlabel(f"PC1 ({ratio[0]:.1f}%)", fontsize=10, color=MUTED)
    ax.set_ylabel(f"PC2 ({ratio[1]:.1f}%)" if n_comp > 1 else "PC2",
                  fontsize=10, color=MUTED)
    ax.set_title(title or L("PSI 的样本主成分分析", "PCA of sample PSI"),
                 fontsize=11, color=INK, loc="left")
    ax.legend(frameon=True, facecolor="white", framealpha=0.78,
              edgecolor="none", fontsize=9, labelcolor=MUTED)
    _style(ax)
    return _finish(fig)

def _draw_ellipse(ax, points: np.ndarray, color: str):
    try:
        cov = np.cov(points.T)
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
        width, height = 2 * 1.2 * np.sqrt(np.maximum(vals, 1e-9))
        ax.add_patch(Ellipse(points.mean(axis=0), width, height, angle=angle,
                             facecolor=color, alpha=0.08, edgecolor=color, linewidth=1.2))
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
    colors = _group_colors(groups)
    for i, s in enumerate(corr.index):
        ax.get_yticklabels()[i].set_color(colors.get(_cond_of(s, groups), MUTED))
    for i, s in enumerate(corr.columns):
        ax.get_xticklabels()[i].set_color(colors.get(_cond_of(s, groups), MUTED))
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
    colors = _group_colors(groups)
    for i, s in enumerate(z.columns):
        ax.get_xticklabels()[i].set_color(colors.get(_cond_of(s, groups), MUTED))
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

def _volcano_stat(data: pd.DataFrame, qval: float) -> Tuple[pd.Series, str, Optional[float]]:
    """火山图的纵轴统计量、轴标签、阈值线位置 —— 三者必须出自同一个统计量。

    有重复、能算 t 检验时用 BH 校正后的 q 值；组内重复不足、q 值全空时退回 p 值，
    并同时改轴标签、不画阈值线（画一条 -log10(0.05) 的线在 p 值轴上是没有意义的）。
    """
    use_q = bool(data["qval"].notna().any())
    stat = data["qval"] if use_q else data["pval"]
    ylabel = (L("-log10 q 值（BH FDR）", "-log10 q-value (BH FDR)") if use_q
              else L("-log10 p 值（未校正，组内重复不足）", "-log10 p-value (uncorrected)"))
    return stat, ylabel, (qval if use_q else None)


def _floor_zero(stat: pd.Series) -> Tuple[pd.Series, int]:
    """把 0 值换成「最小非零值的一半」，返回 (新序列, 被替换的个数)。

    ASTK / SUPPA2 画火山图时就是这么干的（``convert_to_log10pval`` 里那句
    "If p-value = 0 then the -log10_pvalue is calculated using the half of the
    lowest p-value"）：经验分布法偶尔给出 p 恰好等于 0（观测 |dPSI| 正好是本地背景的
    最大值，ECDF = 1），直接取 -log10 会变成无穷，把整张图压扁。
    """
    s = pd.to_numeric(stat, errors="coerce").astype(float)
    zeros = s <= 0
    n_zero = int(zeros.sum())
    positive = s[s > 0]
    if n_zero and not positive.empty:
        s = s.mask(zeros, float(positive.min()) * 0.5)
    return s, n_zero


def fig_volcano(dpsi: pd.DataFrame, abs_dpsi: float = 0.1, qval: float = 0.05,
                event_type: Optional[str] = None, top_label: int = 0,
                max_neglog: float = 20.0):
    """火山图。

    纵轴、显著性判定、水平虚线用同一个统计量（默认是经验法的 BH 校正 q 值）。
    极值处理分两步，都写进图里：p 恰好为 0 的点按 SUPPA2 的做法换成「最小非零值的
    一半」；万一还有超过 20 的点，再截断到 20 并在右下角注明数量，
    不让个别事件决定整根坐标轴。
    """
    need = [c for c in ("pval", "dpsi") if c in dpsi.columns]
    if len(need) < 2:
        return None
    data = dpsi.dropna(subset=["pval", "dpsi"]).copy()
    if data.empty:
        return None
    stat, ylabel, cut_off = _volcano_stat(data, qval)
    stat, n_zero = _floor_zero(stat)
    raw = -np.log10(stat.clip(lower=1e-300))
    data["yval"] = raw.clip(upper=max_neglog)
    if cut_off is None:
        sig = pd.Series(False, index=data.index)
    else:
        sig = (stat <= cut_off) & (data["dpsi"].abs() >= abs_dpsi)
    n_capped = int((raw > max_neglog).sum())

    fig, ax = plt.subplots(figsize=(5.8, 4.4), dpi=150)
    ax.scatter(data.loc[~sig, "dpsi"], data.loc[~sig, "yval"], s=12, color="#B9C7CE",
               label=L(f"不显著（{int((~sig).sum()):,}）", f"not significant ({int((~sig).sum()):,})"),
               zorder=2)
    ax.scatter(data.loc[sig, "dpsi"], data.loc[sig, "yval"], s=16, color=WARM,
               label=L(f"显著（{int(sig.sum()):,}）", f"significant ({int(sig.sum()):,})"),
               zorder=3)
    for x in (-abs_dpsi, abs_dpsi):
        ax.axvline(x, color=RULE, linewidth=1, linestyle="--")
    if cut_off is not None:
        cut = -np.log10(max(cut_off, 1e-300))
        if cut < max_neglog:
            ax.axhline(cut, color=WARM, linewidth=1, linestyle="--", alpha=0.55)
    if top_label:
        # 用位置排序而不是 reindex：多个比较对堆叠时事件名会重复，
        # 按标签取行会抛 "cannot reindex on an axis with duplicate labels"。
        sig_rows = data.loc[sig]
        if len(sig_rows):
            order = np.argsort(-np.abs(sig_rows["dpsi"].to_numpy(dtype=float)))[:int(top_label)]
            top = sig_rows.iloc[order]
        else:
            top = sig_rows
        for eid, row in top.iterrows():
            ax.annotate(str(eid).split(":")[2] if ":" in str(eid) else str(eid)[:12],
                        (row["dpsi"], row["yval"]), fontsize=7, color=MUTED,
                        textcoords="offset points", xytext=(4, 3))
    ref, test, comp = _dpsi_axes(dpsi)
    ax.set_xlabel(f"dPSI ({test} - {ref})", fontsize=10, color=MUTED)
    # 上限只有在真的截到点时才写进轴标签，否则「（上限 20）」会让人以为纵轴到 20
    cap_note = (L(f"（上限 {max_neglog:g}）", f" (capped at {max_neglog:g})")
                if n_capped else "")
    ax.set_ylabel(ylabel + cap_note, fontsize=10, color=MUTED)
    title = L("火山图", "Volcano")
    if event_type:
        title += " - " + event_label(event_type)
    if comp:
        title += f"  |  {comp}"
    ax.set_title(title, fontsize=11, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=MUTED, markerscale=2.5)
    notes = []
    if n_zero:
        notes.append(L(f"{n_zero:,} 个点 p = 0，按最小非零值的一半作图（同 SUPPA2）",
                       f"{n_zero:,} points with p = 0 floored to half the smallest "
                       f"non-zero value (same as SUPPA2)"))
    if n_capped:
        notes.append(L(f"{n_capped:,} 个点超出纵轴上限 {max_neglog:g}",
                       f"{n_capped:,} points above the y-axis cap {max_neglog:g}"))
    if notes:
        ax.text(0.995, 0.02, "\n".join(notes), transform=ax.transAxes,
                ha="right", va="bottom", fontsize=7, color=MUTED)
    _style(ax, "both")
    return _finish(fig)

# ---------------------------------------------------------------- scatter --

def fig_scatter(dpsi: pd.DataFrame, event_type: Optional[str] = None,
                abs_dpsi: float = 0.1, qval: float = 0.05, max_points: int = 4000):
    data = dpsi.dropna(subset=["mean_ref", "mean_test"])
    if data.empty:
        return None
    if data.shape[0] > max_points:
        data = data.sample(max_points, random_state=0)
    sig = (data["qval"] <= qval) & (data["dpsi"].abs() >= abs_dpsi)
    ref, test, comp = _dpsi_axes(dpsi)
    fig, ax = plt.subplots(figsize=(4.9, 4.6), dpi=150)
    ax.plot([0, 1], [0, 1], color=RULE, linewidth=1, linestyle="--", zorder=1)
    ax.scatter(data.loc[~sig, "mean_ref"], data.loc[~sig, "mean_test"], s=11,
               color="#B9C7CE", alpha=0.85, zorder=2, label=L("不显著", "not significant"))
    ax.scatter(data.loc[sig, "mean_ref"], data.loc[sig, "mean_test"], s=14,
               color=WARM, alpha=0.9, zorder=3, label=L("显著", "significant"))
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(f"PSI ({ref})", fontsize=10, color=MUTED)
    ax.set_ylabel(f"PSI ({test})", fontsize=10, color=MUTED)
    title = L("两组 PSI 散点", "PSI scatter")
    if event_type:
        title += " - " + event_label(event_type)
    if comp:
        title += f"  |  {comp}"
    ax.set_title(title, fontsize=11, color=INK, loc="left")
    # 图例挪到右下角：左上角正好是 PSI=0 / PSI=1 两条边上最密的地方
    ax.legend(frameon=False, fontsize=9, labelcolor=MUTED, loc="lower right", markerscale=2.5)
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
    data["condition"] = data["sample"].map(lambda s: _cond_of(s, groups))
    order = [et for et in AS_ORDER if et in set(data["event_type"])]
    order += [et for et in sorted(set(data["event_type"])) if et not in order]
    conds = [c for c in group_order(groups) if c in set(data["condition"])]
    conds += [c for c in sorted(set(data["condition"])) if c not in conds]
    if not conds:
        return None
    colors = _group_colors(groups)
    for i, c in enumerate(conds):
        colors.setdefault(c, PALETTE[i % len(PALETTE)])

    width = 0.8 / max(len(conds), 1)
    fig, ax = plt.subplots(figsize=(max(8.4, 1.35 * len(order)), 3.8), dpi=150)
    positions, labels = [], []
    for i, et in enumerate(order):
        for k, cond in enumerate(conds):
            sub = data[(data["event_type"] == et) & (data["condition"] == cond)]["psi"]
            if sub.empty:
                continue
            pos = i + (k - (len(conds) - 1) / 2) * width
            parts = ax.violinplot(sub.to_numpy(), positions=[pos], widths=width * 0.9,
                                  showextrema=False, showmedians=False)
            for body in parts["bodies"]:
                body.set_facecolor(colors[cond])
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
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=colors[c], label=c)
               for c in conds]
    ax.legend(handles=handles, frameon=False, fontsize=9, labelcolor=MUTED, ncol=min(len(conds), 4))
    _style(ax)
    return _finish(fig)
