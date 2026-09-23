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
import math
import re
import zipfile
from bisect import bisect_left
from itertools import combinations
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


# ------------------------------------------------------- 分组与比较对设计 ----

MIN_REPS = 2

# 组名里带这些词就当成对照（参考组）——这是默认参考组的挑选依据
_CONTROL_HINTS = ("control", "ctrl", "wt", "mock", "vehicle", "shctrl", "scramble",
                  "scr", "nc", "igg", "input", "对照", "野生")


def condition_names(groups: Mapping[str, str]) -> List[str]:
    """所有出现过的分组名，按样本出现顺序去重（空名忽略）。"""
    out: List[str] = []
    for c in groups.values():
        c = str(c).strip()
        if c and c not in out:
            out.append(c)
    return out


def samples_of(groups: Mapping[str, str], cond: str) -> List[str]:
    """某个分组下的样本名（保持 groups 的顺序）。"""
    return [s for s, c in groups.items() if str(c).strip() == cond]


def is_control_like(name: str) -> bool:
    low = str(name).strip().lower()
    return any(h in low for h in _CONTROL_HINTS)


def pick_reference(groups: Mapping[str, str],
                   conditions: Optional[Sequence[str]] = None) -> Optional[str]:
    """默认参考组：名字像对照的优先；都不像就取样本数最多的那组（并列取名字序）。"""
    conds = list(conditions or condition_names(groups))
    if not conds:
        return None
    if len(conds) == 1:
        return conds[0]
    pool = [c for c in conds if is_control_like(c)] or conds
    return sorted(pool, key=lambda c: (-len(samples_of(groups, c)), c))[0]


def comparison_label(reference: str, test: str) -> str:
    """和命令行一致：参考组在前，写作 ``参考组_vs_比较组``。"""
    return f"{reference}_vs_{test}"


def build_comparisons(groups: Mapping[str, str], reference: Optional[str] = None,
                      all_pairs: bool = False) -> List[Tuple[str, str]]:
    """要做的比较对，元素是 ``(比较组, 参考组)``。

    默认只做「参考组 vs 其余各组」：k 个组 = k-1 个比较对。
    ``all_pairs=True`` 做全部两两比较：k 个组 = k(k-1)/2 个比较对。

    为什么不默认做两两比较：BH 校正要跨的检验数随比较对增长，4-5 个组时
    比较对从 3-4 个涨到 6-10 个，校正后几乎检不出事件（信号被多重检验吃掉）。
    """
    conds = condition_names(groups)
    if len(conds) < 2:
        return []
    if all_pairs:
        return [(b, a) for i, a in enumerate(conds) for b in conds[i + 1:]]
    ref = reference if reference in conds else pick_reference(groups, conds)
    return [(c, ref) for c in conds if c != ref]


def comparison_plan(groups: Mapping[str, str], reference: Optional[str] = None,
                    all_pairs: bool = False, min_reps: int = MIN_REPS) -> pd.DataFrame:
    """比较对清单（界面预览用），会标出哪一对算不出 p 值。"""
    rows = []
    for test, ref in build_comparisons(groups, reference, all_pairs):
        n_ref, n_test = len(samples_of(groups, ref)), len(samples_of(groups, test))
        rows.append({
            "比较对": comparison_label(ref, test), "参考组": ref, "比较组": test,
            "n_参考": n_ref, "n_比较": n_test,
            "可算 p 值": "是" if min(n_ref, n_test) >= min_reps else f"否（组内重复 < {min_reps}）",
        })
    return pd.DataFrame(rows, columns=["比较对", "参考组", "比较组", "n_参考", "n_比较", "可算 p 值"])


COMPARISON_NOTE = (
    "dPSI = 比较组均值 − 参考组均值。BH 校正在「单个比较对 x 单个事件类型」内进行："
    "默认只做「参考组 vs 其余各组」（k 个组 = k-1 个比较对）；勾选「所有两两比较」会变成 "
    "k(k-1)/2 个比较对，多重检验的检验数随之增加，4-5 个组时校正后往往一个都不显著，"
    "所以默认关闭。展示多个比较对时，各比较对的显著事件数不能直接相加。"
)

EMPIRICAL_AREA = 1000      # SUPPA2 的 -a/--area，astk 固定传 1000
EMPIRICAL_NAN_TH = 0.0     # SUPPA2 的 --nan-threshold，astk 固定传 0
EMPIRICAL_ALPHA = 0.05     # BH 的 alpha
EMPIRICAL_CUTOFF = 0.0     # astk dsflow 传 0；只有 astk diffSplice -adpsi 才非 0

EMPIRICAL_NOTE = (
    "显著性口径 = ASTK 默认的 empirical（SUPPA2 diffSplice）：背景取自条件内两两重复的 "
    "|ΔPSI|，按表达量取事件附近的 1000 个噪声值算经验 p 值，再按基因内做 BH 校正。"
    "所以本页 q 值 = astk 输出文件里的 p 值列；显著判定仍是 q < 阈值 且 |dPSI| > 阈值。"
)


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


def dpsi_table(psi_df: pd.DataFrame, reference_samples: Sequence[str],
               test_samples: Sequence[str], min_reps: int = MIN_REPS,
               reference: str = "reference", test: str = "test") -> pd.DataFrame:
    """dPSI between two conditions: Welch t-test followed by Benjamini-Hochberg.

    ``dpsi = mean(比较组) - mean(参考组)``，方向和 astk 的 ``case - ctrl`` 一致，
    只是把写死的 ctrl/case 换成了可以任意命名的参考组/比较组。

    组内重复少于 ``min_reps`` 时不外推、也不报错：p 值 / q 值留 NaN，只给 dPSI
    （界面上会明确提示，不静默跳过）。
    """
    ref = [c for c in reference_samples if c in psi_df.columns]
    tst = [c for c in test_samples if c in psi_df.columns]
    if not ref or not tst:
        raise ValueError("参考组和比较组都必须至少有一个样本，且名字要出现在 PSI 表里")
    a, b = psi_df[ref], psi_df[tst]
    out = pd.DataFrame(index=psi_df.index)
    out["comparison"] = comparison_label(reference, test)
    out["reference"] = reference
    out["test"] = test
    out["mean_ref"] = a.mean(axis=1)
    out["mean_test"] = b.mean(axis=1)
    out["dpsi"] = out["mean_test"] - out["mean_ref"]

    pvals = np.full(out.shape[0], np.nan)
    if len(ref) >= min_reps and len(tst) >= min_reps:
        from scipy import stats
        res = stats.ttest_ind(b.to_numpy(), a.to_numpy(), axis=1, nan_policy="omit",
                              equal_var=False)
        pvals = np.asarray(res.pvalue, dtype=float)
    out["pval"] = pvals
    out["qval"] = _bh_fdr(pvals)
    out["n_ref"] = len(ref)
    out["n_test"] = len(tst)
    return out

def _gene_map(ioe_df: Optional[pd.DataFrame]) -> Dict[str, str]:
    """event_id → gene_id，给 BH 的「按基因内校正」用。"""
    if ioe_df is None or "gene_id" not in ioe_df.columns:
        return {}
    return {str(e): str(g) for e, g in zip(ioe_df["event_id"], ioe_df["gene_id"])}


def dpsi_tables(psi: Mapping[str, pd.DataFrame], groups: Mapping[str, str],
                reference: Optional[str] = None, all_pairs: bool = False,
                min_reps: int = MIN_REPS,
                comparisons: Optional[Sequence[Tuple[str, str]]] = None,
                method: str = "empirical",
                ioe: Optional[Mapping[str, pd.DataFrame]] = None,
                tpm: Optional[Mapping[str, pd.DataFrame]] = None,
                area: int = EMPIRICAL_AREA,
                gene_correction: bool = True,
                ) -> Dict[str, pd.DataFrame]:
    """每类事件一张表，若干比较对的 dPSI 纵向堆叠（用 ``comparison`` 列区分）。

    返回 ``{事件类型: DataFrame}``；列有 ``comparison / reference / test /
    mean_ref / mean_test / dpsi / pval / qval / n_ref / n_test``。
    不满足 ``min_reps`` 的比较对照样给 dPSI，只是 p/q 为 NaN。

    ``method`` 默认 ``"empirical"``，即 ASTK / SUPPA2 ``diffSplice`` 的经验分布法
    （此时需要 ``ioe`` 与 ``tpm`` 提供表达量坐标与基因注释；缺了就退回全局背景）。
    传 ``"welch"`` 则回到原来的 Welch t 检验 + BH 口径。
    """
    comps = list(comparisons) if comparisons is not None else build_comparisons(
        groups, reference, all_pairs)
    if not comps:
        return {}
    ioe = ioe or {}
    out: Dict[str, pd.DataFrame] = {}
    for et, df in psi.items():
        cols = [c for c in df.columns if c in groups]
        if len(cols) < 2:
            continue
        parts = []
        for test, ref in comps:
            tst = [c for c in samples_of(groups, test) if c in df.columns]
            rfs = [c for c in samples_of(groups, ref) if c in df.columns]
            if not tst or not rfs:
                continue
            if method == "empirical":
                rep_lt = _event_replicate_logtpm(ioe.get(et), tpm, rfs, tst)
                parts.append(empirical_dpsi_table(
                    df[cols], rfs, tst, rep_lt, genes=_gene_map(ioe.get(et)),
                    area=area, gene_correction=gene_correction,
                    min_reps=min_reps, reference=ref, test=test))
            else:
                parts.append(dpsi_table(df[cols], rfs, tst, min_reps=min_reps,
                                        reference=ref, test=test))
        if parts:
            out[et] = pd.concat(parts)
    return out



# ------------------------------------------------- ASTK 经验分布法（默认） ----
#
# 这一节是 astk/suppa/lib/diff_tools.py 里 SUPPA2 ``diffSplice`` 的 Python 移植，
# 目的是让网页算出来的 p/q 和命令行 ``astk dsflow`` 对得上：
#
#   * 背景分布不是假设正态，而是「同一个条件内部两两重复的 |ΔPSI|」——
#     重复之间本来就不该有真差异，它们的差异幅度就是这个事件的噪声尺度。
#   * 背景按表达量坐标（该事件全部转录本的 log10 TPM 之和，条件内先平均重复、
#     再平均两个条件）排序，取事件自己附近的 ``area`` 个噪声值当「本地背景」。
#     这样表达量相近的事件互相比较，避免了高低表达事件混在一起。
#   * p = (1 - ECDF(本地背景, |ΔPSI|)) × 0.5，单尾，取值范围天然落在 [0, 1]。
#   * astk dsflow 传的 cutoff = 0（不提前给 p=1），最后用 p < 0.05 且 |dPSI| > 0.1
#     判定显著；``astk diffSplice -adpsi`` 会被当成这里的 cutoff。
#   * 多重检验走 BH，而且是**按基因内**校正（astk 的 gene-correction=True）。
#     所以 ASTK 输出文件里的第二列 p 值 = 本模块的 qval 列。
#
# 与命令行的唯一差异：本网页允许只上传 PSI、不传表达量。这种情况下没有表达量坐标，
# 就只能把「全部事件的噪声」当背景（``background="global"``），界面上会写明。

def _event_replicate_logtpm(ioe_df: Optional[pd.DataFrame],
                            tpm_tables: Optional[Mapping[str, pd.DataFrame]],
                            ref_samples: Sequence[str],
                            test_samples: Sequence[str],
                            ) -> Dict[str, Dict[str, List[float]]]:
    """每个事件在各重复上的 log10 TPM：（事件 → 条件 → 各重复的值）。

    照 SUPPA2 ``calculate_transcript_abundance`` / ``create_replicates_distribution``：
    一个事件在某个重复里的表达量 = 该事件**全部转录本**（ioe 的 total_transcripts 列）
    在该重复的 TPM 之和，取 log10。TPM 缺失的转录本按 SUPPA2 的做法直接跳过，
    整个事件的转录本一个都对不上时该重复记为 None。
    """
    if ioe_df is None or not tpm_tables:
        return {}
    samples = [s for s in list(ref_samples) + list(test_samples) if s in tpm_tables]
    if not samples:
        return {}
    tpm_of = {s: pd.to_numeric(tpm_tables[s].iloc[:, 0], errors="coerce") for s in samples}
    tx_lists = ioe_df["total_transcripts"].astype(str).str.split(",").to_numpy()

    out: Dict[str, Dict[str, List[float]]] = {}
    event_ids = ioe_df["event_id"].astype(str).to_numpy()
    for i, event_id in enumerate(event_ids):
        per_cond: Dict[str, List[float]] = {}
        for label, group in (("ref", ref_samples), ("test", test_samples)):
            values: List[float] = []
            for s in group:
                ser = tpm_of.get(s)
                if ser is None:
                    continue
                sub = ser.reindex(tx_lists[i]).dropna()
                if sub.empty:
                    continue
                total = float(sub.sum())
                if not np.isfinite(total) or total <= 0:
                    continue
                values.append(float(math.log10(total)))
            if values:
                per_cond[label] = values
        if len(per_cond) == 2:
            out[str(event_id)] = per_cond
    return out


def event_logtpm(rep_logtpm: Mapping[str, Mapping[str, Sequence[float]]]) -> pd.Series:
    """事件级的表达量坐标：条件内先对重复求平均，再对两个条件求平均。"""
    rows = {}
    for event_id, per_cond in rep_logtpm.items():
        ref_v, test_v = per_cond.get("ref"), per_cond.get("test")
        if not ref_v or not test_v:
            continue
        mean_ref = float(np.mean([v for v in ref_v if np.isfinite(v)]))
        mean_test = float(np.mean([v for v in test_v if np.isfinite(v)]))
        rows[event_id] = 0.5 * (mean_ref + mean_test)
    return pd.Series(rows, dtype=float)


def _slice_local(values: Sequence[float], index: int, area: int) -> List[float]:
    """SUPPA2 ``slice_list``：以 index 为中心取 area 个，越界就整体平移。"""
    half = int(area * 0.5)
    diff = index - half
    if diff < 0:
        left, right = 0, index + half + (-diff) + 1
    elif index + half >= len(values):
        upper = index + half - len(values) + 1
        left, right = diff - upper, index + half + 1
    else:
        left, right = diff, index + half + 1
    return list(values[max(left, 0):max(right, 0)])


def _closest_index(sorted_values: Sequence[float], target: float) -> int:
    """SUPPA2 ``get_closest_number``：返回离 target 最近的位置（并列取较小值）。"""
    pos = bisect_left(list(sorted_values), target)
    if pos == 0:
        return 0
    if pos == len(sorted_values):
        return len(sorted_values) - 1
    before, after = sorted_values[pos - 1], sorted_values[pos]
    return pos if (after - target) < (target - before) else pos - 1


def _ecdf(values: Sequence[float], x: float) -> float:
    """经验分布函数（和 statsmodels 的 ECDF 一样取「≤ x 的比例」）。"""
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return 0.0
    return float((arr <= x).sum()) / float(arr.size)


def empirical_dpsi_table(psi_df: pd.DataFrame, ref_samples: Sequence[str],
                         test_samples: Sequence[str],
                         rep_logtpm: Optional[Mapping[str, Mapping[str, Sequence[float]]]] = None,
                         genes: Optional[Mapping[str, str]] = None,
                         area: int = EMPIRICAL_AREA, cutoff: float = EMPIRICAL_CUTOFF,
                         nan_th: float = EMPIRICAL_NAN_TH,
                         gene_correction: bool = True, alpha: float = EMPIRICAL_ALPHA,
                         min_reps: int = MIN_REPS,
                         reference: str = "reference", test: str = "test") -> pd.DataFrame:
    """ASTK / SUPPA2 经验分布法的 dPSI + 经验 p 值（+ BH 校正）。

    返回列与 :func:`dpsi_table` 一致，另加 ``avg_logtpm`` / ``background`` /
    ``n_background`` 三个诊断列。``n_background`` 是本地背景里参与 ECDF 的噪声值个数。
    """
    ref = [c for c in ref_samples if c in psi_df.columns]
    tst = [c for c in test_samples if c in psi_df.columns]
    if not ref or not tst:
        raise ValueError("参考组和比较组都必须至少有一个样本，且名字要出现在 PSI 表里")

    a, b = psi_df[ref], psi_df[tst]
    out = pd.DataFrame(index=psi_df.index)
    out["comparison"] = comparison_label(reference, test)
    out["reference"] = reference
    out["test"] = test
    out["mean_ref"] = a.mean(axis=1)
    out["mean_test"] = b.mean(axis=1)
    out["dpsi"] = out["mean_test"] - out["mean_ref"]
    out["n_ref"] = len(ref)
    out["n_test"] = len(tst)

    # SUPPA2 的 nan 处理：某一侧缺失比例超过 nan_th 的事件整条丢弃，只留 (nan, 1.0)
    too_many_nan = (a.isna().mean(axis=1) > nan_th) | (b.isna().mean(axis=1) > nan_th)

    rep_logtpm = dict(rep_logtpm or {})
    logtpm = event_logtpm(rep_logtpm)
    out["avg_logtpm"] = logtpm.reindex(out.index)

    enough = min(len(ref), len(tst)) >= min_reps

    # 背景分布：各条件内部两两重复的 ΔPSI。有表达量坐标时按坐标排序，
    # 事件取自己附近的一个窗口；完全没有表达量（只上传 PSI 文件）时退回全局背景。
    pairs_coord: List[Tuple[float, float]] = []   # (ΔPSI, 两个重复的平均 log10 TPM)
    deltas_all: List[float] = []                  # 所有条件内重复对的 ΔPSI
    for event_id in out.index:
        per_cond = rep_logtpm.get(str(event_id)) or {}
        for label, group in (("ref", ref), ("test", tst)):
            values = list(per_cond.get(label) or [])
            obs = psi_df.loc[event_id, group].to_numpy(dtype=float)
            pairs_in_cond = []
            for j, v in enumerate(obs):
                if not np.isfinite(v):
                    continue
                coord = float(values[j]) if j < len(values) else np.nan
                pairs_in_cond.append((float(v), coord))
            for (v1, l1), (v2, l2) in combinations(pairs_in_cond, 2):
                delta = v2 - v1
                deltas_all.append(delta)
                if np.isfinite(l1) and np.isfinite(l2):
                    pairs_coord.append((delta, 0.5 * (l1 + l2)))

    has_coord = bool(len(logtpm)) and bool(pairs_coord)
    if pairs_coord:
        pairs_coord.sort(key=lambda x: x[1])
        grid = [p[1] for p in pairs_coord]
        deltas = [p[0] for p in pairs_coord]
    else:
        grid, deltas = [], []
    if not deltas:
        deltas = deltas_all

    pvals = pd.Series(np.nan, index=out.index, dtype=float)
    backgrounds: List[str] = []
    n_background: List[float] = []
    for event_id in out.index:
        if bool(too_many_nan.get(event_id, False)):
            pvals[event_id] = 1.0          # 与 SUPPA2 一致：丢弃事件给 p = 1.0
            out.loc[event_id, "dpsi"] = np.nan
            backgrounds.append("discarded")
            n_background.append(np.nan)
            continue
        if not deltas:
            backgrounds.append("none")
            n_background.append(np.nan)
            continue
        coord = logtpm.get(str(event_id))
        if has_coord and np.isfinite(coord):
            local = _slice_local(deltas, _closest_index(grid, float(coord)), area)
            backgrounds.append("expression-window")
        else:
            local = deltas or deltas_all      # 没有表达量就退回全局背景
            backgrounds.append("global")
        abs_dpsi = abs(float(out.loc[event_id, "dpsi"]))
        if -cutoff < abs_dpsi < cutoff:
            pvals[event_id] = 1.0             # astk diffSplice -adpsi 的行为
        else:
            pvals[event_id] = (1.0 - _ecdf(local, abs_dpsi)) * 0.5
        n_background.append(len(local))

    out["background"] = backgrounds
    out["n_background"] = n_background

    if not enough:
        out["pval"] = np.nan                  # 组内重复不足：只给 dPSI
        out["qval"] = np.nan
        return out

    out["pval"] = pvals

    # BH 校正：astk dsflow 走 gene-correction=True（按基因内校正），
    # 拿不到基因注释时退回全体事件一起校正（例如只上传 PSI 文件）。
    tested = out.index[pvals.notna()]
    qvals = pd.Series(np.nan, index=out.index, dtype=float)
    if len(tested):
        raw = pvals.loc[tested].to_numpy(dtype=float)
        if gene_correction and genes:
            gene_of = pd.Series([genes.get(str(e), str(e)) for e in tested], index=tested)
            for _, idx in gene_of.groupby(gene_of).groups.items():
                idx = list(idx)
                qvals.loc[idx] = _bh_fdr(pvals.loc[idx].to_numpy(dtype=float))
        else:
            qvals.loc[tested] = _bh_fdr(raw)
    out["qval"] = qvals
    return out


def significant_summary(dpsi: Mapping[str, pd.DataFrame], abs_dpsi: float = 0.1,
                        qval: float = 0.05) -> pd.DataFrame:
    """每个「事件类型 x 比较对」的显著事件数。

    BH 只在单个比较对内部做，所以多个比较对的 n_sig 不能相加当成一个检验看。
    """
    cols = ["event_type", "label", "comparison", "n_events", "n_tested", "n_sig",
            "pct_sig", "n_up", "n_down"]
    rows = []
    for et, df in dpsi.items():
        if df.empty:
            continue
        labels = (df["comparison"].astype(str) if "comparison" in df.columns
                  else pd.Series(["-"] * df.shape[0], index=df.index))
        for comp in labels.drop_duplicates():
            sub = df[labels == comp]
            if sub.empty:
                continue
            tested = sub["qval"].notna()
            sig = sub[(sub["qval"] <= qval) & (sub["dpsi"].abs() >= abs_dpsi)]
            rows.append({
                "event_type": et, "label": AS_LABELS.get(et, et), "comparison": comp,
                "n_events": int(sub.shape[0]), "n_tested": int(tested.sum()),
                "n_sig": int(sig.shape[0]),
                "pct_sig": round(100 * sig.shape[0] / max(int(tested.sum()), 1), 3),
                "n_up": int((sig["dpsi"] > 0).sum()), "n_down": int((sig["dpsi"] < 0).sum()),
            })
    if not rows:
        return pd.DataFrame(columns=cols)
    order = {et: i for i, et in enumerate(AS_TYPES)}
    out = pd.DataFrame(rows)[cols]
    return out.sort_values(["event_type", "comparison"],
                           key=lambda s: s.map(order) if s.name == "event_type" else s,
                           ).reset_index(drop=True)

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
    # 三组演示数据：Control（参考组）/ cKO / Rescue（效应减半），
    # 一打开就能看到「参考组 vs 其余各组」是怎么走的。
    groups = dict(groups or {**{f"Control_{i}": "Control" for i in range(1, 4)},
                             **{f"cKO_{i}": "cKO" for i in range(1, 4)},
                             **{f"Rescue_{i}": "Rescue" for i in range(1, 4)}})
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
    # 各组的效应强度：Control 不动，cKO 全额，Rescue 减半
    effect_by_condition = {"Control": 0.0, "cKO": 1.0, "Rescue": 0.5}
    for sample, cond in groups.items():
        tpm = base_level * rng.lognormal(0.0, 0.25, size=len(txs))
        strength = effect_by_condition.get(
            str(cond), 1.0 if str(cond).strip().lower() in ("case", "cko", "ko") else 0.0)
        if strength:
            tpm = np.where(in_shifted, tpm * factor ** strength, tpm)
        tpm_tables[sample] = pd.Series(tpm, index=txs).to_frame(sample)
    return ioe_tables, tpm_tables, groups
