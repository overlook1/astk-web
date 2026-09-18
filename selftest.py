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
    check("PSI tables built", all(df.shape[0] > 0 and df.shape[1] == 6 for df in psi.values()),
          str(shapes))
    finite = np.isfinite(pd.concat(psi.values()).to_numpy()).mean()
    check("PSI values are finite", finite > 0.9, f"finite fraction {finite:.3f}")
    rng_ok = all(float(np.nanmin(df.to_numpy())) >= 0.0 and float(np.nanmax(df.to_numpy())) <= 1.0
                 for df in psi.values())
    check("PSI within [0, 1]", rng_ok)

    counts = core.event_counts(ioe_tables)
    check("event_counts", counts.shape[0] == 7, f"{counts['n_events'].sum()} events")
    overview = core.psi_overview(psi, groups)
    check("psi_overview", overview.shape[0] == 7 and "psi_ctrl" in overview.columns)

    dpsi = core.dpsi_tables(psi, groups)
    check("dpsi tables", len(dpsi) == 7, str(sorted(dpsi)))
    sig = core.significant_summary(dpsi)
    n_sig = int(sig["n_sig"].sum())
    check("dPSI finds signal", n_sig > 0, f"{n_sig} significant events")
    check("dPSI has both directions", int(sig["n_up"].sum()) > 0 and int(sig["n_down"].sum()) > 0,
          f"up={int(sig['n_up'].sum())} down={int(sig['n_down'].sum())}")
    check("BH q-values in [0, 1]",
          bool(((dpsi["SE"]["qval"].dropna() >= 0) & (dpsi["SE"]["qval"].dropna() <= 1)).all()))
    check("dpsi == mean_case - mean_ctrl",
          bool(np.allclose(dpsi["SE"]["dpsi"],
                           dpsi["SE"]["mean_case"] - dpsi["SE"]["mean_ctrl"], equal_nan=True)))

    # --- figures -----------------------------------------------------------
    merged = pd.concat([psi[et] for et in core.AS_TYPES], axis=0)
    render("counts", plots.fig_event_counts(counts))
    render("distribution", plots.fig_psi_distribution(psi, groups))
    render("pca", plots.fig_pca(merged, groups, title="PCA - ALL"))
    render("corr_heatmap", plots.fig_corr_heatmap(merged, groups))
    render("psi_heatmap", plots.fig_psi_heatmap(psi["SE"], groups, top_n=30))
    render("volcano", plots.fig_volcano(dpsi["SE"], 0.1, 0.05, "SE", top_label=5))
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
    real_dpsi = core.dpsi_table(real_psi, ["Ctrl_1", "Ctrl_2"], ["cKO_1", "cKO_2"])
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