"""ASTK web app - core pipeline helpers.

Only this module touches the ``astk`` package. Everything it returns is plain
pandas / numpy data, so the plotting layer and the UI can be exercised without a
working ASTK installation (see ``demo_dataset``).

Input contract
-------------
* Transcript quantification: salmon ``quant.sf`` files (one per sample, whose
  parent directory name is used as the sample name - the same convention as
  ``astk dsflow``), or one transcript x sample TPM matrix.
* AS events: an ``.ioe`` reference (columns ``seqname, gene_id, event_id,
  alternative_transcripts, total_transcripts``) or a GTF annotation that is
  parsed with ``astk.suppa`` to build one.
"""
from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- constants --

AS_TYPES: List[str] = ["SE", "A5", "A3", "MX", "RI", "AF", "AL"]

AS_LABELS: Dict[str, str] = {
    "SE": "SE 外显子跳跃",
    "A5": "A5 可变 5' 剪接位点",
    "A3": "A3 可变 3' 剪接位点",
    "MX": "MX 互斥外显子",
    "RI": "RI 内含子保留",
    "AF": "AF 可变首个外显子",
    "AL": "AL 可变末端外显子",
}

IOE_COLUMNS = ["seqname", "gene_id", "event_id", "alternative_transcripts", "total_transcripts"]

# salmon quant.sf layout: Name, Length, EffectiveLength, TPM, NumReads
SALMON_TX_COL = 1
SALMON_TPM_COL = 4

# ---------------------------------------------------------------- citation --

ASTK_CITATION = (
    "Huang, S., He, J., Yu, L., Guo, J., Jiang, S., Sun, Z., Cheng, L., Chen, X., "
    "Ji, X. and Zhang, Y. (2024), ASTK: A Machine Learning-Based Integrative "
    "Software for Alternative Splicing Analysis. Adv. Intell. Syst. 2300594."
)
ASTK_DOI = "https://doi.org/10.1002/aisy.202300594"
ASTK_REPO = "https://github.com/huang-sh/astk"
ASTK_DOCS = "https://huang-sh.github.io/astk-doc/"
ASTK_CONTACT = "hsh-me@outlook.com"
ASTK_BIBTEX = """@article{huang2024astk,
  title   = {ASTK: A Machine Learning-Based Integrative Software for Alternative Splicing Analysis},
  author  = {Huang, S. and He, J. and Yu, L. and Guo, J. and Jiang, S. and Sun, Z. and Cheng, L. and Chen, X. and Ji, X. and Zhang, Y.},
  journal = {Advanced Intelligent Systems},
  pages   = {2300594},
  year    = {2024},
  doi     = {10.1002/aisy.202300594}
}"""

# Written into every downloaded result bundle and shown on the "about" tab.
CITATION_NOTE = "\n".join([
    "本网站是 ASTK 的网页前端：事件定义与 PSI 计算口径和 ASTK / SUPPA2 一致。",
    "",
    "如果你使用本网站产生的结果，请引用 ASTK：",
    "",
    "    " + ASTK_CITATION,
    "    " + ASTK_DOI,
    "",
    "ASTK 源码：" + ASTK_REPO,
    "ASTK 文档：" + ASTK_DOCS,
])


# --------------------------------------------------------- 参考集 / 物种 ----

# a reference set sitting directly in ref/ (no species sub-folder)
DEFAULT_REF_LABEL = "default"

SPECIES_LABELS: Dict[str, str] = {
    "human": "人 GRCh38",
    "mouse": "小鼠 GRCm38/M25",
    "rat": "大鼠 mRatBN7.2",
    "zebrafish": "斑马鱼 GRCz11",
    "fly": "果蝇 BDGP6",
    "default": "默认参考集",
}


def species_label(name: str) -> str:
    """Friendly name for a reference set folder (e.g. ``ref/mouse``)."""
    return SPECIES_LABELS.get(name, name)


# ------------------------------------------------------------------- checks --

def astk_status() -> Tuple[bool, str]:
    """Return ``(available, message)`` for the ASTK installation."""
    try:
        import astk  # noqa: F401
    except Exception as exc:  # pragma: no cover - depends on the environment
        return False, f"未检测到 ASTK（{exc.__class__.__name__}: {exc}）。可先跑演示数据。"
    return True, "ASTK 已就绪。"


# --------------------------------------------------------------- IOE / GTF --

def read_ioe(source) -> pd.DataFrame:
    """Read one ``.ioe`` file from a path or a file-like object."""
    df = pd.read_csv(source, sep="\t")
    missing = [c for c in IOE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"ioe 文件缺少列: {', '.join(missing)}")
    for col in ("alternative_transcripts", "total_transcripts"):
        df[col] = df[col].astype(str)
    df["event_type"] = df["event_id"].astype(str).str.extract(r"^([A-Z]{2}):", expand=False)
    return df


def generate_ioe(gtf_path, outdir, events: Sequence[str] = None,
                 idtype: str = "SUPPA2", event_pos: Optional[str] = None) -> Dict[str, Path]:
    """Build one ioe file per event type from a GTF annotation.

    Mirrors ``astk generateEvents``: files are written as
    ``<outdir>/annotation[_<event_pos>]_<ET>_strict.ioe``.
    """
    from astk.suppa.AS_event import make_events
    from astk.suppa.gtf_parse import construct_genome

    events = list(events or AS_TYPES)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    genome = construct_genome(str(gtf_path))
    make_events(str(outdir / "annotation"), genome, events, idtype, event_pos)
    suffix = "" if event_pos is None else f"_{event_pos}"
    return {et: outdir / f"annotation{suffix}_{et}_strict.ioe" for et in events}


def _ioe_in(folder: Path) -> Dict[str, pd.DataFrame]:
    """Every ``*_<ET>_strict.ioe`` in one folder, keyed by event type."""
    tables: Dict[str, pd.DataFrame] = {}
    for path in sorted(folder.glob("*_strict.ioe")):
        parts = path.name.split("_")
        et = parts[-2] if len(parts) >= 2 else ""
        if et not in AS_TYPES:
            continue
        try:
            tables[et] = read_ioe(path)
        except Exception:
            continue
    return tables


def discover_ioe(refdir) -> Dict[str, pd.DataFrame]:
    """Load every ``*_<ET>_strict.ioe`` of ONE folder, keyed by event type."""
    refdir = Path(refdir)
    return _ioe_in(refdir) if refdir.is_dir() else {}


def discover_species(refdir) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Find bundled reference sets laid out as ``ref/<species>/*_strict.ioe``.

    A public deployment should ship the common genomes so a stranger can just
    upload ``quant.sf`` and get results - that is what turns this site into
    something other people actually use. ioe files sitting directly in ``ref/``
    are also accepted, under the label ``default``.
    """
    refdir = Path(refdir)
    if not refdir.is_dir():
        return {}
    found: Dict[str, Dict[str, pd.DataFrame]] = {}
    direct = _ioe_in(refdir)
    if direct:
        found[DEFAULT_REF_LABEL] = direct
    for sub in sorted(p for p in refdir.iterdir() if p.is_dir()):
        tables = _ioe_in(sub)
        if tables:
            found[sub.name] = tables
    return found


# --------------------------------------------------- transcript abundances --

def read_quant_sf(source, sample: Optional[str] = None, tpm_col: int = SALMON_TPM_COL,
                  tx_col: int = SALMON_TX_COL) -> pd.DataFrame:
    """Read a salmon ``quant.sf`` (or ``abundance.tsv``) into a one column frame."""
    df = pd.read_csv(source, sep="\t")
    if df.shape[1] < max(tpm_col, tx_col):
        raise ValueError("量化文件列数不足，请确认是 salmon quant.sf")
    out = pd.DataFrame({sample or "sample": pd.to_numeric(df.iloc[:, tpm_col - 1], errors="coerce")})
    out.index = df.iloc[:, tx_col - 1].astype(str)
    out.index.name = "transcript_id"
    return out


def read_tpm_matrix(source, sep: Optional[str] = None) -> pd.DataFrame:
    """Read a transcript x sample TPM matrix (first column = transcript id)."""
    df = pd.read_csv(source, sep=sep, engine="python", index_col=0)
    df.index = df.index.astype(str)
    df = df.apply(pd.to_numeric, errors="coerce")
    df.index.name = "transcript_id"
    return df


def read_quant_zip(blob) -> Dict[str, pd.DataFrame]:
    """Read a zip of ``<sample>/quant.sf`` folders (sample name = folder name)."""
    tables: Dict[str, pd.DataFrame] = {}
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = Path(info.filename).name.lower()
            if name not in ("quant.sf", "abundance.tsv", "quant.sf.gz"):
                continue
            p = Path(info.filename)
            sample = p.parent.name if p.parent.name else p.stem
            with zf.open(info) as fh:
                tables[sample] = read_quant_sf(fh, sample=sample)
    if not tables:
        raise ValueError("zip 里没有找到 quant.sf，请按 <样本名>/quant.sf 的目录结构打包")
    return tables


# --------------------------------------------------------------------- PSI --

def psi_native(ioe_df: pd.DataFrame, tpm_df: pd.DataFrame,
               tpm_threshold: float = 1.0) -> pd.Series:
    """Vectorised version of ASTK/SUPPA2 PSI (``cal_psi``), used as a fallback.

    Same rules as ``astk.suppa.event_psi.cal_psi``: transcripts missing from the
    quantification are dropped, PSI = sum(alt TPM) / sum(total TPM), values are
    set to NaN when the mean TPM of the total transcripts is below the
    threshold, and to 0 when the alternative TPM is negligible.
    """
    tpm = pd.to_numeric(tpm_df.iloc[:, 0], errors="coerce")
    events = pd.Index(ioe_df["event_id"].astype(str))

    def _expand(column: str) -> pd.DataFrame:
        long = pd.DataFrame({"event": events, "tx": ioe_df[column].astype(str).to_numpy()})
        long = long.assign(tx=long["tx"].str.split(",")).explode("tx")
        long["tpm"] = long["tx"].map(tpm)
        return long.groupby("event")["tpm"].agg(["sum", "count"]).reindex(events)

    alt, tot = _expand("alternative_transcripts"), _expand("total_transcripts")
    with np.errstate(invalid="ignore", divide="ignore"):
        psi = alt["sum"] / tot["sum"]
    mean_tot = tot["sum"] / tot["count"].replace(0, np.nan)
    psi = psi.where(mean_tot >= tpm_threshold)
    psi = psi.where(~(alt["sum"] <= 0.0001), 0.0)
    psi[(alt["count"] == 0) | (tot["count"] == 0)] = np.nan
    return psi


def psi_table(ioe_df: pd.DataFrame, tpm_tables: Mapping[str, pd.DataFrame],
              tpm_threshold: float = 1.0, engine: str = "auto") -> pd.DataFrame:
    """PSI matrix (events x samples) for one event type.

    ``engine`` is ``auto`` (ASTK when importable, otherwise the built-in
    implementation), ``astk`` or ``native``.
    """
    if engine == "auto":
        engine = "astk" if astk_status()[0] else "native"
    events = pd.Index(ioe_df["event_id"].astype(str), name="event_id")
    columns = {}
    for sample, tpm_df in tpm_tables.items():
        if tpm_df.shape[1] != 1:
            tpm_df = tpm_df.iloc[:, [0]]
        if engine == "astk":
            from astk.suppa.event_psi import get_ioe_psi
            values = pd.to_numeric(pd.Series(get_ioe_psi(ioe_df, tpm_df, tpm_th=tpm_threshold)),
                                   errors="coerce")
            values.index = events
        else:
            values = psi_native(ioe_df, tpm_df, tpm_threshold)
        columns[sample] = values.to_numpy()
    return pd.DataFrame(columns, index=events)


def psi_tables(ioe_tables: Mapping[str, pd.DataFrame], tpm_tables: Mapping[str, pd.DataFrame],
               tpm_threshold: float = 1.0, engine: str = "auto") -> Dict[str, pd.DataFrame]:
    return {et: psi_table(ioe, tpm_tables, tpm_threshold, engine) for et, ioe in ioe_tables.items()}


def read_psi_file(source) -> pd.DataFrame:
    """Read a PSI file (events x samples) produced by ASTK/SUPPA2."""
    df = pd.read_csv(source, sep=None, engine="python", index_col=0)
    df.index.name = "event_id"
    return df.apply(pd.to_numeric, errors="coerce")


def merge_psi_files(tables: Sequence[pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Group uploaded PSI tables by event type and inner-join them on event id.

    A PSI file holds one event type, identified by the ``ET:`` prefix of its
    event ids. Tables of the same type are joined with ``how="inner"`` so only
    events quantified in every uploaded file survive.
    """
    buckets: Dict[str, List[pd.DataFrame]] = {}
    for table in tables:
        if table.empty:
            continue
        prefixes = pd.Series([str(i).split(":")[0] for i in table.index])
        et = prefixes.mode().iat[0] if not prefixes.empty else "ALL"
        if et not in AS_TYPES:
            et = "ALL"
        buckets.setdefault(et, []).append(table)

    merged: Dict[str, pd.DataFrame] = {}
    for et, parts in buckets.items():
        acc = parts[0]
        for nxt in parts[1:]:
            acc = acc.join(nxt, how="inner", lsuffix="", rsuffix="_dup")
        merged[et] = acc
    return merged


# ---------------------------------------------------------- event overview --

def event_counts(ioe_tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for et, df in ioe_tables.items():
        genes = df["gene_id"].astype(str).str.split(".").str[0]
        rows.append({"event_type": et, "label": AS_LABELS.get(et, et),
                     "n_events": int(df.shape[0]), "n_genes": int(genes.nunique())})
    order = {et: i for i, et in enumerate(AS_TYPES)}
    out = pd.DataFrame(rows).sort_values("event_type", key=lambda s: s.map(order))
    return out.reset_index(drop=True)


def psi_overview(psi: Mapping[str, pd.DataFrame], groups: Mapping[str, str]) -> pd.DataFrame:
    """Per event type: usable events, mean PSI per condition, high/low counts."""
    rows = []
    for et, df in psi.items():
        valid = df.dropna(how="all")
        row = {"event_type": et, "label": AS_LABELS.get(et, et),
               "n_events": int(df.shape[0]), "n_usable": int(valid.shape[0]),
               "psi_mean": float(np.nanmean(df.to_numpy())) if df.size else np.nan}
        for cond in sorted(set(groups.values())):
            cols = [s for s, c in groups.items() if c == cond and s in df.columns]
            if cols:
                vals = df[cols].mean(axis=1)
                row[f"psi_{cond}"] = float(np.nanmean(vals.to_numpy()))
                row[f"high_{cond}"] = int((vals > 0.75).sum())
                row[f"low_{cond}"] = int((vals < 0.25).sum())
        rows.append(row)
    order = {et: i for i, et in enumerate(AS_TYPES)}
    return pd.DataFrame(rows).sort_values("event_type", key=lambda s: s.map(order)).reset_index(drop=True)


# -------------------------------------------------------------- difference --

def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    ok = ~np.isnan(p)
    q = np.full(p.shape, np.nan)
    if ok.sum() == 0:
        return q
    ordered = np.argsort(p[ok])
    ranked = p[ok][ordered]
    n = ranked.size
    adj = ranked * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(n, dtype=float)
    out[ordered] = np.clip(adj, 0, 1)
    q[ok] = out
    return q


def dpsi_table(psi_df: pd.DataFrame, ctrl_samples: Sequence[str],
               case_samples: Sequence[str], min_reps: int = 2) -> pd.DataFrame:
    """dPSI between two conditions: Welch t-test followed by Benjamini-Hochberg.

    This is a fast screening statistic for the web app; the CLI (``astk ds`` /
    ``astk dsflow``) remains the reference implementation for reporting.
    """
    ctrl = [c for c in ctrl_samples if c in psi_df.columns]
    case = [c for c in case_samples if c in psi_df.columns]
    if not ctrl or not case:
        raise ValueError("两组样本都必须至少有一个，且名字要出现在 PSI 表里")
    a, b = psi_df[ctrl], psi_df[case]
    out = pd.DataFrame(index=psi_df.index)
    out["mean_ctrl"] = a.mean(axis=1)
    out["mean_case"] = b.mean(axis=1)
    out["dpsi"] = out["mean_case"] - out["mean_ctrl"]
    out["ctrl_psi"] = out["mean_ctrl"]
    out["case_psi"] = out["mean_case"]

    pvals = np.full(out.shape[0], np.nan)
    if len(ctrl) >= min_reps and len(case) >= min_reps:
        from scipy import stats
        res = stats.ttest_ind(b.to_numpy(), a.to_numpy(), axis=1, nan_policy="omit",
                              equal_var=False)
        pvals = np.asarray(res.pvalue, dtype=float)
    out["pval"] = pvals
    out["qval"] = _bh_fdr(pvals)
    out["n_ctrl"] = len(ctrl)
    out["n_case"] = len(case)
    return out


def dpsi_tables(psi: Mapping[str, pd.DataFrame], groups: Mapping[str, str]) -> Dict[str, pd.DataFrame]:
    ctrl = [s for s, c in groups.items() if c == "ctrl"]
    case = [s for s, c in groups.items() if c == "case"]
    out = {}
    for et, df in psi.items():
        cols = [c for c in df.columns if c in set(ctrl) | set(case)]
        if len(cols) < 2:
            continue
        out[et] = dpsi_table(df[cols], ctrl, case)
    return out


def significant_summary(dpsi: Mapping[str, pd.DataFrame], abs_dpsi: float = 0.1,
                        qval: float = 0.05) -> pd.DataFrame:
    rows = []
    for et, df in dpsi.items():
        sig = df[(df["qval"] <= qval) & (df["dpsi"].abs() >= abs_dpsi)]
        tested = df["qval"].notna()
        rows.append({
            "event_type": et, "label": AS_LABELS.get(et, et),
            "n_events": int(df.shape[0]), "n_tested": int(tested.sum()),
            "n_sig": int(sig.shape[0]),
            "pct_sig": round(100 * sig.shape[0] / max(int(tested.sum()), 1), 3),
            "n_up": int((sig["dpsi"] > 0).sum()), "n_down": int((sig["dpsi"] < 0).sum()),
        })
    order = {et: i for i, et in enumerate(AS_TYPES)}
    return pd.DataFrame(rows).sort_values("event_type", key=lambda s: s.map(order)).reset_index(drop=True)


# ------------------------------------------------------------------- demo --

def demo_dataset(n_events: int = 150, seed: int = 0,
                 groups: Optional[Mapping[str, str]] = None,
                 effect_frac: float = 0.25, effect_size: float = 2.4):
    """Synthetic ioe + TPM tables so the UI and the plots can run without ASTK.

    Each event type gets its own pool of ``n_events * TX_PER_EVENT`` transcripts,
    grouped in blocks of six (the first ``n_alt`` of a block are the alternative
    transcripts, the rest belong to the other isoform). ``effect_frac`` of the
    events receive a real PSI shift in the ``case`` samples, so the difference
    tab has genuine signal instead of pure noise.
    """
    groups = dict(groups or {"Ctrl_1": "ctrl", "Ctrl_2": "ctrl", "Ctrl_3": "ctrl",
                             "Case_1": "case", "Case_2": "case", "Case_3": "case"})
    rng = np.random.default_rng(seed)

    tx_per_event = 6
    ioe_tables: Dict[str, pd.DataFrame] = {}
    # transcript id -> (event type, event index, is alternative)
    lookup: Dict[str, Tuple[str, int, bool]] = {}

    for et in AS_TYPES:
        prefix = AS_TYPES.index(et)
        alt, tot = [], []
        for i in range(n_events):
            members = [f"ENSMUST{prefix:02d}{(i * tx_per_event + j + 1):09d}"
                       for j in range(tx_per_event)]
            n_alt = int(rng.integers(1, 4))
            alt.append(",".join(members[:n_alt]))
            tot.append(",".join(members))
            for j, tx in enumerate(members):
                lookup[tx] = (et, i, j < n_alt)
        gene = f"ENSMUSG{prefix:02d}"
        ioe_tables[et] = pd.DataFrame({
            "seqname": "chr1",
            "gene_id": [f"{gene}{(i + 1):09d}" for i in range(n_events)],
            "event_id": [f"{et}:{gene}{(i + 1):09d}:{i + 1}-{i + 100}" for i in range(n_events)],
            "alternative_transcripts": alt,
            "total_transcripts": tot,
        })

    txs = list(lookup)
    is_alt = np.array([lookup[t][2] for t in txs])
    et_of = np.array([lookup[t][0] for t in txs])
    ev_of = np.array([lookup[t][1] for t in txs])

    base_level = rng.lognormal(2.2, 1.0, size=len(txs))
    # which events move, and whether the alternative isoform goes up or down
    shifted = {et: rng.random(n_events) < effect_frac for et in AS_TYPES}
    up = {et: rng.random(n_events) < 0.5 for et in AS_TYPES}
    in_shifted = np.array([shifted[et][i] for et, i in zip(et_of, ev_of)])
    goes_up = np.array([up[et][i] for et, i in zip(et_of, ev_of)])
    factor = np.where(is_alt, effect_size, 1.0 / effect_size)
    factor = np.where(goes_up, factor, 1.0 / factor)

    tpm_tables: Dict[str, pd.DataFrame] = {}
    for sample, cond in groups.items():
        tpm = base_level * rng.lognormal(0.0, 0.25, size=len(txs))
        if cond == "case":
            tpm = np.where(in_shifted, tpm * factor, tpm)
        tpm_tables[sample] = pd.Series(tpm, index=txs).to_frame(sample)
    return ioe_tables, tpm_tables, groups
