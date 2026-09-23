"""Self-check for astk_web - ASTK is NOT required.

    python selftest.py

Runs the whole pipeline on synthetic data (PSI, dPSI, every figure) and then the
salmon-zip + ioe upload path, so you can tell whether an install is healthy
before deploying. Figures are written to a temp folder for eyeballing.
"""
from __future__ import annotations

import io
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core
import plots

OUT = Path(tempfile.mkdtemp(prefix="astk_web_selftest_"))
FAILURES: list = []


def check(name: str, condition: bool, detail: str = "") -> None:
    mark = "PASS" if condition else "FAIL"
    print(f"[{mark}] {name}" + (f"  {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


def render(name: str, fig) -> None:
    if fig is None:
        check(f"figure {name}", False, "returned None")
        return
    fig.savefig(OUT / f"{name}.png", dpi=110, bbox_inches="tight")
    plots.close(fig)
    check(f"figure {name}", True)


def main() -> int:
    print(f"figures -> {OUT}")
    print(f"CJK font available: {plots.HAS_CJK} (labels fall back to ASCII if False)\n")

    ok, msg = core.astk_status()
    print(f"astk: {msg}  (the web path does not need it)\n")

    # --- synthetic dataset -------------------------------------------------
    ioe_tables, tpm_tables, groups = core.demo_dataset()
    check("demo ioe has 7 event types", len(ioe_tables) == 7, str(sorted(ioe_tables)))

    psi = core.psi_tables(ioe_tables, tpm_tables, tpm_threshold=1.0, engine="native")
    shapes = {et: df.shape for et, df in psi.items()}
    check("PSI tables built", all(df.shape[0] > 0 and df.shape[1] == 9 for df in psi.values()),
          str(shapes))
    finite = np.isfinite(pd.concat(psi.values()).to_numpy()).mean()
    check("PSI values are finite", finite > 0.9, f"finite fraction {finite:.3f}")
    rng_ok = all(float(np.nanmin(df.to_numpy())) >= 0.0 and float(np.nanmax(df.to_numpy())) <= 1.0
                 for df in psi.values())
    check("PSI within [0, 1]", rng_ok)

    counts = core.event_counts(ioe_tables)
    check("event_counts", counts.shape[0] == 7, f"{counts['n_events'].sum()} events")
    overview = core.psi_overview(psi, groups)
    check("psi_overview",
          overview.shape[0] == 7
          and {"psi_Control", "psi_cKO", "psi_Rescue"} <= set(overview.columns),
          str([c for c in overview.columns if c.startswith("psi_")]))

    dpsi = core.dpsi_tables(psi, groups, ioe=ioe_tables, tpm=tpm_tables)
    check("dpsi tables", len(dpsi) == 7, str(sorted(dpsi)))
    sig = core.significant_summary(dpsi)
    n_sig = int(sig["n_sig"].sum())
    check("dPSI finds signal", n_sig > 0, f"{n_sig} significant events")
    check("dPSI has both directions", int(sig["n_up"].sum()) > 0 and int(sig["n_down"].sum()) > 0,
          f"up={int(sig['n_up'].sum())} down={int(sig['n_down'].sum())}")
    check("BH q-values in [0, 1]",
          bool(((dpsi["SE"]["qval"].dropna() >= 0) & (dpsi["SE"]["qval"].dropna() <= 1)).all()))
    check("dpsi == mean_test - mean_ref",
          bool(np.allclose(dpsi["SE"]["dpsi"],
                           dpsi["SE"]["mean_test"] - dpsi["SE"]["mean_ref"], equal_nan=True)))

    # --- ASTK empirical（网页默认口径，对齐 astk dsflow）--------------------
    af = dpsi["AF"]
    check("经验法默认开启：列里有 background 诊断列", "background" in af.columns,
          str(list(af.columns)))
    check("经验法：背景取自表达量窗口",
          set(af["background"]) <= {"expression-window", "discarded"},
          str(af["background"].value_counts().to_dict()))
    widths = af.loc[af["background"] == "expression-window", "n_background"]
    check("经验法：窗口长度 = SUPPA2 的 area 窗口（<= area+1 且非空）",
          bool(len(widths) and (widths <= core.EMPIRICAL_AREA + 1).all() and (widths > 0).all()),
          f"median {float(widths.median()):.0f}")
    check("经验法：有表达量坐标的样本才有值", bool(af["avg_logtpm"].notna().any()))
    discarded = af["background"] == "discarded"
    check("经验法：缺失比例超过 nan_th 的事件给 p = 1.0（SUPPA2 行为）",
          bool((af.loc[discarded, "pval"] == 1.0).all()) if bool(discarded.any()) else True,
          f"{int(discarded.sum())} 行")
    kept = af.loc[~discarded, "pval"].dropna()
    check("经验法：其余 p 落在 [0, 0.5]（单尾 ECDF）",
          bool(((kept >= 0) & (kept <= 0.5)).all()), f"max {float(kept.max()):.3f}")
    check("经验法：q 不小于 p（BH 只会上调）",
          bool((af.loc[af["qval"].notna(), "qval"] + 1e-12
                >= af.loc[af["qval"].notna(), "pval"]).all()))
    check("经验法：没有表达量时退回全局背景",
          set(core.dpsi_tables(psi, groups, ioe=None, tpm=None)["AF"]["background"]) == {"global"})
    zero_psi = {et: df.copy() for et, df in psi.items()}
    zero_psi["AF"].iloc[0] = 0.0
    zero_psi["AF"].iloc[1] = 1.0
    zero_af = core.dpsi_tables(zero_psi, groups, ioe=ioe_tables, tpm=tpm_tables)["AF"]
    zero_first = zero_af[zero_af.index.isin(zero_psi["AF"].index[:2])].groupby(level=0).first()
    check("经验法：PSI 全 0 / 全 1 的事件不会被算成强显著",
          bool((zero_first["pval"] > 0.05).all()),
          "p = " + ", ".join(f"{v:.3f}" for v in zero_first["pval"]))
    zero_w = core.dpsi_tables(zero_psi, groups, method="welch",
                              ioe=ioe_tables, tpm=tpm_tables)["AF"]
    zero_wf = zero_w[zero_w.index.isin(zero_psi["AF"].index[:2])].groupby(level=0).first()
    check("Welch 口径仍可切换（火山图的纵轴不再会被 p=0 打崩）",
          bool(zero_wf["pval"].isna().all() or (zero_wf["pval"] > 0).all()))
    welch_af = core.dpsi_tables(psi, groups, method="welch")["AF"]
    check("Welch 口径没有 background 列、q 值正常",
          "background" not in welch_af.columns and bool(welch_af["qval"].notna().any()))

    # --- 多分组：参考组 / 比较对 / 重复数不足 -------------------------------
    check("默认参考组取名字像对照的那组", core.pick_reference(groups) == "Control",
          str(core.pick_reference(groups)))
    default_comps = core.build_comparisons(groups)
    check("默认比较对 = 参考组 vs 其余各组",
          default_comps == [("cKO", "Control"), ("Rescue", "Control")], str(default_comps))
    pair_comps = core.build_comparisons(groups, all_pairs=True)
    check("两两比较 = k(k-1)/2 个", len(pair_comps) == 3, str(pair_comps))
    plan = core.comparison_plan(groups)
    check("比较对清单标出能否算 p 值", list(plan["可算 p 值"]) == ["是", "是"],
          str(list(plan["可算 p 值"])))
    check("三组时每类事件各有 2 个比较对",
          all(sorted(d["comparison"].unique()) == ["Control_vs_Rescue", "Control_vs_cKO"]
              for d in dpsi.values()))
    check("significant_summary 按比较对拆开",
          sig.shape[0] == 14 and "comparison" in sig.columns, str(sig.shape))
    check("dPSI 方向 = 比较组 - 参考组",
          bool(np.allclose(dpsi["SE"]["dpsi"],
                           dpsi["SE"]["mean_test"] - dpsi["SE"]["mean_ref"], equal_nan=True)))

    one_groups = {"A": "Control", "B": "cKO"}
    one_plan = core.comparison_plan(one_groups)
    check("组内重复 < 2 时明确提示算不出 p 值",
          list(one_plan["可算 p 值"]) == ["否（组内重复 < 2）"], str(list(one_plan["可算 p 值"])))
    one_psi = core.psi_tables(ioe_tables,
                              {"A": tpm_tables["Control_1"], "B": tpm_tables["cKO_1"]},
                              engine="native")
    one_dpsi = core.dpsi_tables(one_psi, one_groups)
    check("组内重复 < 2 时 p/q 留空但 dPSI 保留",
          bool(one_dpsi["SE"]["pval"].isna().all() and one_dpsi["SE"]["dpsi"].notna().all()))



    # --- 向后兼容：只有两组时行为要和以前一致 -----------------------------
    two = {"Control_1": "ctrl", "Control_2": "ctrl", "Control_3": "ctrl",
           "cKO_1": "case", "cKO_2": "case", "cKO_3": "case"}
    two_psi = {et: psi[et][list(two)] for et in psi}
    two_dpsi = core.dpsi_tables(two_psi, two)
    check("两组时只生成 1 个比较对、命名为 ctrl_vs_case",
          list(two_dpsi["SE"]["comparison"].unique()) == ["ctrl_vs_case"],
          str(list(two_dpsi["SE"]["comparison"].unique())))
    check("两组时参考组自动取到 ctrl", core.pick_reference(two) == "ctrl")
    two_merged = pd.concat([two_psi[et] for et in core.AS_TYPES], axis=0)
    render("pca_2groups", plots.fig_pca(two_merged, two))
    render("volcano_2groups", plots.fig_volcano(two_dpsi["SE"], 0.1, 0.05, "SE", top_label=3))
    render("volcano_empirical", plots.fig_volcano(dpsi["AF"], 0.1, 0.05, "AF", top_label=3))
    render("scatter_2groups", plots.fig_scatter(two_dpsi["SE"], "SE", 0.1, 0.05))
    render("dist_2groups", plots.fig_psi_distribution(two_psi, two))

    # --- figures -----------------------------------------------------------
    merged = pd.concat([psi[et] for et in core.AS_TYPES], axis=0)
    render("counts", plots.fig_event_counts(counts))
    render("distribution", plots.fig_psi_distribution(psi, groups))
    merged = pd.concat([psi[et] for et in core.AS_TYPES], axis=0)
    pca_fig = plots.fig_pca(merged, groups, title="PCA - ALL")
    legend = []
    if pca_fig is not None:
        leg = pca_fig.axes[0].get_legend()
        legend = [t.get_text() for t in leg.get_texts()] if leg is not None else []
    check("PCA 把三个分组都画出来了", len(legend) == 3, str(legend))
    render("pca", pca_fig)
    render("corr_heatmap", plots.fig_corr_heatmap(merged, groups))
    render("psi_heatmap", plots.fig_psi_heatmap(psi["SE"], groups, top_n=30))
    vd = dpsi["SE"].copy()
    vfig = plots.fig_volcano(vd, 0.1, 0.05, "SE", top_label=5)
    got = want = np.array([])
    n_zero = 0
    if vfig is not None:
        got = np.sort(np.concatenate([c.get_offsets()[:, 1] for c in vfig.axes[0].collections]))
        floored, n_zero = plots._floor_zero(vd["qval"])
        want = np.sort(-np.log10(floored.clip(lower=1e-300)).clip(upper=20.0).dropna().to_numpy())
    check("火山图纵轴就是 -log10 q 值（p=0 的点按 SUPPA2 规则换成最小非零值的一半）",
          got.shape == want.shape and got.size > 0 and bool(np.allclose(got, want, atol=1e-9)),
          f"n={got.size}, p=0 的点 {n_zero} 个")
    check("火山图轴标签写明是 q 值",
          vfig is not None and "q" in vfig.axes[0].get_ylabel(), 
          vfig.axes[0].get_ylabel() if vfig is not None else "")
    check("字体探测能认出没有中文字形的字体", not plots._renders_cjk("DejaVu Sans"))
    check("没有中文字体时事件标签退回 ASCII",
          bool(plots.HAS_CJK) or plots.event_label("AF").isascii(), plots.event_label("AF"))
    render("volcano", vfig)
    render("scatter", plots.fig_scatter(dpsi["SE"], "SE", 0.1, 0.05))

    # --- upload path: <sample>/quant.sf in a zip + an ioe file --------------
    sample_names = ["Ctrl_1", "Ctrl_2", "cKO_1", "cKO_2"]
    rng = np.random.default_rng(7)
    txs = [f"ENSMUST{i:09d}" for i in range(1, 61)]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for s in sample_names:
            tpm = rng.lognormal(2.0, 1.0, size=len(txs))
            zf.writestr(f"{s}/quant.sf", pd.DataFrame({
                "Name": txs, "Length": 2000, "EffectiveLength": 1800.0,
                "TPM": tpm, "NumReads": tpm * 3}).to_csv(sep="\t", index=False).encode())

    zipped = core.read_quant_zip(buf.getvalue())
    check("read_quant_zip keeps folder names as samples",
          sorted(zipped) == sorted(sample_names), str(sorted(zipped)))

    ioe = pd.DataFrame({
        "seqname": "chr1",
        "gene_id": [f"ENSMUSG{i:09d}" for i in range(1, 11)],
        "event_id": [f"SE:ENSMUSG{i:09d}:{i}-{i + 50}" for i in range(1, 11)],
        "alternative_transcripts": [",".join(txs[i * 6:i * 6 + 2]) for i in range(10)],
        "total_transcripts": [",".join(txs[i * 6:i * 6 + 6]) for i in range(10)],
    })
    ioe_df = core.read_ioe(io.BytesIO(ioe.to_csv(sep="\t", index=False).encode()))
    check("read_ioe adds event_type", list(ioe_df["event_type"].unique()) == ["SE"])

    real_psi = core.psi_table(ioe_df, zipped, tpm_threshold=1.0)
    check("PSI from uploaded files", real_psi.shape == (10, 4), str(real_psi.shape))

    real_groups = {"Ctrl_1": "ctrl", "Ctrl_2": "ctrl", "cKO_1": "case", "cKO_2": "case"}
    real_dpsi = core.dpsi_table(real_psi, ["Ctrl_1", "Ctrl_2"], ["cKO_1", "cKO_2"],
                                reference="Ctrl", test="cKO")
    check("dpsi_table columns", {"dpsi", "pval", "qval"} <= set(real_dpsi.columns))

    # --- round trips -------------------------------------------------------
    psi_f = OUT / "SE.psi"
    psi["SE"].to_csv(psi_f, sep="\t")
    check("read_psi_file round trip", core.read_psi_file(psi_f).shape == psi["SE"].shape)

    mat = pd.concat(tpm_tables.values(), axis=1)
    tpm_f = OUT / "tpm.tsv"
    mat.to_csv(tpm_f, sep="\t")
    check("read_tpm_matrix round trip", core.read_tpm_matrix(tpm_f).shape == mat.shape)

    merged_psi = core.merge_psi_files([
        pd.DataFrame({"A": [0.1, 0.2]}, index=["SE:g1:1-2", "SE:g2:3-4"]),
        pd.DataFrame({"B": [0.3, 0.4]}, index=["SE:g2:3-4", "SE:g3:5-6"]),
    ])
    check("merge_psi_files inner-joins samples",
          list(merged_psi) == ["SE"] and merged_psi["SE"].shape == (1, 2))

    check("discover_ioe on a missing folder is empty", core.discover_ioe(OUT / "nope") == {})

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) FAILED: {', '.join(FAILURES)}")
        return 1
    print("所有自检通过。可以 `streamlit run app.py` 了。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())