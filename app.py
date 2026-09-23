"""ASTK web app - Streamlit entry point.

Run locally:      streamlit run app.py
Deploy (free):    Streamlit Community Cloud, main file = astk_web/app.py
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import streamlit as st

import core
import plots

st.set_page_config(page_title="ASTK Web - 可变剪接分析", layout="wide",
                   initial_sidebar_state="expanded")

AS_ORDER = core.AS_TYPES


# ------------------------------------------------------------------ helpers --

def _sample_name_from_upload(name: str, idx: int) -> str:
    stem = Path(name).stem
    if stem.lower() in ("quant", "abundance"):
        return f"sample_{idx + 1}"
    for suffix in ("_quant", ".quant"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem


def _guess_condition(sample: str) -> str:
    """从样本名猜一个初始分组。猜错无所谓 —— 分组列可以随便改，组数不限。"""
    low = sample.lower()
    if any(k in low for k in ("case", "cko", "ko", "treat", "mut", "kd", "oe")):
        return "cKO"
    return "Control"

def _figure_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    return buf.getvalue()


def _show(fig, **kwargs) -> None:
    """Render a figure, then release it - Streamlit keeps the PNG, not the Figure."""
    if fig is None:
        return
    st.pyplot(fig, **kwargs)
    plots.close(fig)


def _zip_results(payload: Dict[str, pd.DataFrame], figures: Dict[str, object]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("CITATION.txt", core.CITATION_NOTE)
        for name, df in payload.items():
            zf.writestr(name, df.to_csv(sep="\t"))
        for name, fig in figures.items():
            if fig is not None:
                zf.writestr(name, _figure_bytes(fig))
                plots.close(fig)
    return buf.getvalue()


@st.cache_data(show_spinner="正在生成演示数据 ...")
def _demo_bundle():
    ioe, tpm, groups = core.demo_dataset()
    return ioe, tpm, groups


# ------------------------------------------------------------------ sidebar --

st.sidebar.title("ASTK Web")
st.sidebar.caption("转录组 → 七类可变剪接事件 → PSI / dPSI 与图表")

source_kind = st.sidebar.radio(
    "数据来源",
    ["演示数据（免上传）", "salmon quant.sf", "TPM 矩阵", "已有 PSI 结果"],
    help="云端部署时只能上传表达定量（quant.sf / TPM）。FASTQ 比对需要基因组索引，"
         "请在本地或服务器上完成。",
)

ref_kind = st.sidebar.radio(
    "事件参考集",
    ["随数据提供 / 演示", "使用仓库内置 ref/", "上传 .ioe", "上传 GTF（现场生成，慢）"],
    help="ioe 是 ASTK 的事件定义文件（annotation_<ET>_strict.ioe）。"
         "用同一份 ioe 才能让不同样本、不同批次的 PSI 可比。"
         "把一次生成好的 ref/ 提交进仓库，别人上传 quant.sf 就能直接出结果，"
         "云端也不需要装 astk。",
)

with st.sidebar.expander("参数", expanded=True):
    tpm_threshold = st.number_input("TPM 阈值（低于则该事件记 NaN）", 0.0, 100.0, 1.0, 0.5)
    method_label = st.radio(
        "显著性方法", ["ASTK empirical（默认，与 astk dsflow 一致）", "Welch t 检验（旧口径）"],
        help=core.EMPIRICAL_NOTE)
    method = "empirical" if method_label.startswith("ASTK") else "welch"
    abs_dpsi = st.slider("|dPSI| 显著阈值", 0.0, 1.0, 0.10, 0.05)
    qval_cut = st.slider("q 值（BH FDR）阈值", 0.01, 0.5, 0.05, 0.01)
    top_n_events = st.slider("热图展示事件数", 10, 200, 40, 10)
    top_label = st.number_input("火山图标注事件数", 0, 50, 0, 1)

astk_ok, astk_msg = core.astk_status()
st.sidebar.caption(("✅ " if astk_ok else "⚠️ ") + astk_msg)


# ------------------------------------------------------------------- inputs --

tpm_tables: Dict[str, pd.DataFrame] = {}
ioe_tables: Dict[str, pd.DataFrame] = {}
groups: Dict[str, str] = {}
notes = []

if source_kind.startswith("演示"):
    ioe_tables, tpm_tables, groups = _demo_bundle()
    notes.append("演示数据为程序生成的模拟数据，仅用于体验流程。")

elif source_kind == "salmon quant.sf":
    up = st.sidebar.file_uploader("quant.sf 文件（可多选）", type=["sf", "tsv", "txt"],
                                  accept_multiple_files=True)
    zp = st.sidebar.file_uploader("或上传 zip（<样本名>/quant.sf 结构）", type=["zip"])
    if zp is not None:
        try:
            tpm_tables = core.read_quant_zip(zp.getvalue())
        except Exception as exc:
            st.error(f"读取 zip 失败：{exc}")
    elif up:
        for i, f in enumerate(up):
            sample = _sample_name_from_upload(f.name, i)
            try:
                tpm_tables[sample] = core.read_quant_sf(io.BytesIO(f.getvalue()), sample=sample)
            except Exception as exc:
                st.error(f"{f.name} 读取失败：{exc}")
    if tpm_tables:
        notes.append(f"读入 {len(tpm_tables)} 个样本的定量文件（TPM 取第 4 列，与 astk 一致）。")

elif source_kind == "TPM 矩阵":
    up = st.sidebar.file_uploader("TPM 矩阵（首列=转录本 ID，其余列为样本）", type=["tsv", "txt", "csv"])
    if up:
        try:
            mat = core.read_tpm_matrix(io.BytesIO(up.getvalue()))
            tpm_tables = {c: mat[[c]] for c in mat.columns}
            notes.append(f"读入 TPM 矩阵：{mat.shape[0]} 个转录本 x {mat.shape[1]} 个样本。")
        except Exception as exc:
            st.error(f"读取失败：{exc}")

else:  # 已有 PSI 结果
    up = st.sidebar.file_uploader("PSI 文件（每行一个事件，列为样本；可多选，按事件名取交集）",
                                  type=["psi", "tsv", "txt", "csv"], accept_multiple_files=True)
    merged = []
    for f in up or []:
        try:
            merged.append(core.read_psi_file(io.BytesIO(f.getvalue())))
        except Exception as exc:
            st.error(f"{f.name} 读取失败：{exc}")
    if merged:
        n_cols = len({c for m in merged for c in m.columns})
        notes.append(f"读入 {len(merged)} 个 PSI 文件，覆盖 {n_cols} 个样本；同名事件取交集。")
    # PSI 文件本身已含事件，无需参考集


# ------------------------------------------------------------------ 参考事件 --

if source_kind != "已有 PSI 结果" and tpm_tables:
    if ref_kind.startswith("使用仓库内置"):
        ref_dir = Path(__file__).parent / "ref"
        sets = core.discover_species(ref_dir)
        if not sets:
            st.error(f"没在 {ref_dir} 找到 *_strict.ioe。请先在有 astk 的机器上，为每个物种各跑一次 "
                     "`python make_ioe.py <gencode.vM25.annotation.gtf> ref/mouse`，"
                     "把整个 ref/ 提交到仓库。公开站点建议至少带 mouse + human 两套，别人传个 "
                     "quant.sf 就能直接出结果。")
        else:
            pick = st.sidebar.selectbox(
                "参考集", sorted(sets), format_func=core.species_label,
                help="ref/<物种>/ 下的事件定义。别人不必自己准备 ioe，传 quant.sf 就能算。")
            ioe_tables = sets[pick]
            notes.append(f"参考集 {core.species_label(pick)}：" + ", ".join(sorted(ioe_tables)) + "。")
    elif ref_kind == "上传 .ioe":
        ioe_up = st.sidebar.file_uploader("ioe 文件（每种事件一个，可多选）", type=["ioe"],
                                          accept_multiple_files=True)
        if ioe_up:
            for f in ioe_up:
                et = f.name.split("_")[-2] if "_" in f.name else Path(f.name).stem
                try:
                    ioe_tables[et] = core.read_ioe(io.BytesIO(f.getvalue()))
                except Exception as exc:
                    st.error(f"{f.name} 读取失败：{exc}")
    elif ref_kind.startswith("上传 GTF"):
        gtf = st.sidebar.file_uploader("GTF 注释（建议用 gencode）", type=["gtf", "gtf.gz", "gz"])
        events = st.sidebar.multiselect("事件类型", AS_ORDER, default=AS_ORDER)
        if gtf is not None:
            if not astk_ok:
                st.error("生成事件需要 ASTK，请先在环境中安装 astk。")
            elif st.sidebar.button("生成事件参考集"):
                import tempfile
                with st.spinner("正在解析 GTF 并构建事件（大基因组可能要好几分钟）..."):
                    workdir = Path(tempfile.mkdtemp(prefix="astk_"))
                    gtf_path = workdir / gtf.name
                    gtf_path.write_bytes(gtf.getvalue())
                    paths = core.generate_ioe(gtf_path, workdir / "ref", events=events)
                    st.session_state["gtf_ioe"] = {et: core.read_ioe(p) for et, p in paths.items()}
                st.success("事件参考集已生成。")
            if st.session_state.get("gtf_ioe"):
                ioe_tables = st.session_state["gtf_ioe"]
                notes.append(f"现场生成事件参考：{', '.join(ioe_tables)}。")


# ------------------------------------------------------------------ 分组设置 --

# 分组名不写死、组数不限：演示数据自带 Control / cKO / Rescue，真实数据先按样本名
# 猜一版填进表格，用户在「① 数据与分组」里随便改，参考组与比较对也跟着变。
if source_kind == "已有 PSI 结果":
    groups = {}

st.title("可变剪接事件分析")
st.caption("输入转录表达定量 → 七类 AS 事件的 PSI / 差异 PSI → 事件谱、PCA、热图、散点图、火山图")

(tab_input, tab_overview, tab_tables, tab_pca, tab_heat, tab_diff, tab_dl,
 tab_about) = st.tabs(
    ["① 数据与分组", "② 概览", "③ 七类事件表", "④ PCA", "⑤ 热图", "⑥ 差异分析",
     "⑦ 下载", "⑧ 关于与引用"]
)

with tab_input:
    st.subheader("样本与分组")
    st.info("本网页从**表达定量**（salmon `quant.sf` / TPM 矩阵）开始计算 PSI。"
            "FASTQ 比对需要基因组索引，请在本地或服务器上完成，再把 `quant.sf` 传上来。")
    reference, all_pairs = None, False
    if source_kind == "已有 PSI 结果":
        st.info("当前使用已有的 PSI 结果文件，请在下方表格里为每个样本填写分组。")
        sample_cols = sorted({c for m in merged for c in m.columns})
        base = pd.DataFrame({"sample": sample_cols,
                             "condition": [_guess_condition(s) for s in sample_cols]})
    else:
        base = pd.DataFrame({"sample": list(tpm_tables),
                             "condition": [groups.get(s) or _guess_condition(s) for s in tpm_tables]})
    if base.empty:
        st.warning("还没有数据。请在左侧选择数据来源并上传文件，或直接使用演示数据。")
    else:
        st.caption("分组名可以随便填：**相同字符串就是同一组，填几组都行**。"
                   "时间序列填 11.5 / 12.5 / 13.5，实验组填 Control / cKO / Rescue 都可以。")
        edited = st.data_editor(base, hide_index=True, key="group_editor",
                                column_config={
                                    "sample": st.column_config.TextColumn("样本名", disabled=True),
                                    "condition": st.column_config.TextColumn("分组（可自由填写）",
                                                                            required=True),
                                })
        groups = {str(s): str(c).strip() for s, c in zip(edited["sample"], edited["condition"])
                  if str(c).strip()}
        conds = core.condition_names(groups)
        st.dataframe(pd.DataFrame(
            [{"分组": c, "样本数": len(core.samples_of(groups, c))} for c in conds]),
            hide_index=True)
        if len(conds) < 2:
            st.warning("要至少 2 个分组才能做差异分析；现在只有 "
                       + (str(len(conds)) + " 组。" if conds else "0 组。"))
        else:
            guess = core.pick_reference(groups, conds)
            col_r, col_p = st.columns([2, 3])
            with col_r:
                reference = st.selectbox(
                    "参考组（dPSI = 比较组 − 参考组）", conds, index=conds.index(guess),
                    help="默认取名字像对照的组（control / ctrl / wt / mock …）；"
                         "都不像就取样本数最多的那组。可以手动改。")
            with col_p:
                all_pairs = st.checkbox("显示所有两两比较", value=False,
                                        help=core.COMPARISON_NOTE)
            plan = core.comparison_plan(groups, reference, all_pairs)
            st.markdown(f"**将做 {plan.shape[0]} 个比较对**")
            st.dataframe(plan, hide_index=True)
            st.caption(core.COMPARISON_NOTE)
            bad = plan.loc[plan["可算 p 值"] != "是", "比较对"].tolist()
            if bad:
                st.warning("这些比较对里有分组的样本数不足 2 个：只会给出 dPSI，"
                           "p 值 / q 值留空，不计入显著事件 —— " + "、".join(bad))
        if notes:
            st.info("　".join(notes))

    run = st.button("开始分析", type="primary", disabled=not (tpm_tables or (source_kind == "已有 PSI 结果" and up)))

if run:
    st.session_state.pop("results", None)
    psi: Dict[str, pd.DataFrame] = {}
    if source_kind == "已有 PSI 结果":
        psi = core.merge_psi_files(merged)
    else:
        if not ioe_tables:
            st.error("缺少事件参考集：请上传 ioe，或用 GTF 现场生成。")
        elif not tpm_tables:
            st.error("缺少表达定量文件。")
        else:
            bar = st.progress(0.0, text="正在计算 PSI ...")
            for i, (et, ioe) in enumerate(ioe_tables.items()):
                psi[et] = core.psi_table(ioe, tpm_tables, tpm_threshold)
                bar.progress((i + 1) / len(ioe_tables), text=f"PSI {et} ({i + 1}/{len(ioe_tables)})")
            bar.empty()
    if psi:
        psi = {et: df for et, df in psi.items() if not df.empty and df.shape[0] > 0}
        comps = core.build_comparisons(groups, reference, all_pairs)
        dpsi = core.dpsi_tables(psi, groups, comparisons=comps, method=method,
                                ioe=ioe_tables, tpm=tpm_tables) if comps else {}
        st.session_state["results"] = {"psi": psi, "dpsi": dpsi, "groups": dict(groups),
                                       "ioe": ioe_tables, "tpm": list(tpm_tables),
                                       "comparisons": comps, "reference": reference,
                                       "method": method}
        extra = f"，{len(comps)} 个比较对" if comps else ""
        st.success(f"完成：{len(psi)} 类事件，"
                   f"{sum(df.shape[0] for df in psi.values()):,} 个事件条目{extra}。")
    else:
        st.error("没有算出任何 PSI。请检查事件参考集（ioe）是否覆盖了上传的转录本 ID。")

with tab_about:
    st.subheader("这个网站是什么")
    st.markdown(
        "上传转录组表达定量（salmon `quant.sf` 或 TPM 矩阵），在线得到**七类可变剪接事件**"
        "（SE / A5 / A3 / MX / RI / AF / AL）的 PSI 与差异分析，以及事件谱、PCA、"
        "样本相关性热图、PSI 热图、火山图和散点图。\n\n"
        "事件定义（`.ioe`）与 PSI 计算口径和 **ASTK / SUPPA2** 一致；做这个网页版，"
        "是为了让人不必先装命令行环境就能把结果跑出来。")
    st.divider()
    st.subheader("请引用 ASTK")
    st.markdown("本网站是 ASTK 的网页前端。**如果你用本网站的结果发表了工作，请引用 ASTK 原文**"
                "——这也是这个网站存在的意义。")
    st.info(core.ASTK_CITATION + "  " + core.ASTK_DOI)
    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.markdown("**BibTeX**（代码块右上角可复制）")
        st.code(core.ASTK_BIBTEX, language="bibtex")
    with col_b:
        st.markdown("**链接**")
        st.markdown(f"- 论文原文：{core.ASTK_DOI}\n"
                    f"- 源码：{core.ASTK_REPO}\n"
                    f"- 文档：{core.ASTK_DOCS}\n"
                    f"- 联系作者：{core.ASTK_CONTACT}")
        st.markdown("**要做完整分析**（表观信号 / 序列特征 / 随机森林）")
        st.code("pip install astk\nastk --help", language="bash")
    st.divider()
    st.caption("口径提醒：差异分析默认走 ASTK 的 empirical 经验分布法（area=1000、按基因内 "
               "BH 校正），和命令行 `astk dsflow` 对齐；想对比旧口径可以在左侧参数里切回 "
               "Welch t 检验。FASTQ 比对和 ChIP-seq bigWig 不在本网站范围内。")

st.sidebar.divider()
st.sidebar.markdown("**请引用 ASTK**")
st.sidebar.caption(core.ASTK_CITATION)
st.sidebar.markdown(f"[论文]({core.ASTK_DOI}) ｜ [源码]({core.ASTK_REPO}) ｜ [文档]({core.ASTK_DOCS})")

res = st.session_state.get("results")
if not res:
    st.stop()

psi, dpsi, groups, ioe_tables = res["psi"], res["dpsi"], res["groups"], res["ioe"]
# the seven standard types first, then anything extra the user uploaded
present = [et for et in AS_ORDER if et in psi] + [et for et in psi if et not in AS_ORDER]

with tab_overview:
    counts = core.event_counts(ioe_tables) if ioe_tables else pd.DataFrame(
        [{"event_type": et, "label": core.AS_LABELS[et], "n_events": psi[et].shape[0],
          "n_genes": np.nan} for et in present])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("事件类型", len(present))
    c2.metric("事件条目合计", f"{sum(psi[et].shape[0] for et in present):,}")
    c3.metric("样本数", psi[present[0]].shape[1])
    c4.metric("显著事件（各比较对合计）",
              f"{int(core.significant_summary(dpsi, abs_dpsi, qval_cut)['n_sig'].sum()):,}"
              if dpsi else "-")
    left, right = st.columns([1, 1])
    with left:
        _show(plots.fig_event_counts(counts))
    with right:
        _show(plots.fig_psi_distribution(psi, groups))
    st.subheader("PSI 概览")
    st.dataframe(core.psi_overview(psi, groups), hide_index=True)

with tab_tables:
    et = st.selectbox("事件类型", present, format_func=lambda x: core.AS_LABELS.get(x, x))
    df = psi[et].copy()
    st.caption(f"{df.shape[0]:,} 个事件 x {df.shape[1]} 个样本；NaN 表示该样本 TPM 低于阈值。")
    if et in ioe_tables:
        meta_cols = [c for c in ("seqname", "gene_id", "alternative_transcripts", "total_transcripts")
                     if c in ioe_tables[et].columns]
        show = ioe_tables[et].set_index("event_id")[meta_cols].join(df, how="inner")
    else:
        show = df
    st.dataframe(show.round(4), height=460)
    st.download_button(f"下载 {et} PSI 表", df.to_csv(sep="\t").encode("utf-8-sig"),
                       file_name=f"{et}_psi.tsv", mime="text/tab-separated-values")

with tab_pca:
    sel = st.radio("用哪些事件做 PCA", ["全部事件（合并）", "单一事件类型"], horizontal=True)
    if sel == "全部事件（合并）":
        merged_psi = pd.concat([psi[et] for et in present], axis=0)
        et_sel = None
    else:
        et_sel = st.selectbox("事件类型 ", present, format_func=lambda x: core.AS_LABELS.get(x, x))
        merged_psi = psi[et_sel]
    fig = plots.fig_pca(merged_psi, groups, title=f"PCA - {et_sel or 'ALL'}")
    if fig is None:
        st.warning("样本或事件太少（或缺失值过多），无法做 PCA。")
    else:
        _show(fig)
        st.caption("输入为样本 x 事件 的 PSI 矩阵，按事件标准化后取前两个主成分；"
                   "同一组的样本应聚在一起，若 ctrl/case 完全混在一起说明整体剪接谱差异不大。")

with tab_heat:
    col1, col2 = st.columns(2)
    with col1:
        f = plots.fig_corr_heatmap(pd.concat([psi[et] for et in present], axis=0), groups)
        if f is not None:
            _show(f)
    with col2:
        et_h = st.selectbox("热图事件类型", present, key="heat_et",
                            format_func=lambda x: core.AS_LABELS.get(x, x))
        f = plots.fig_psi_heatmap(psi[et_h], groups, top_n=top_n_events)
        if f is not None:
            _show(f)

with tab_diff:
    if not dpsi:
        if not res.get("comparisons"):
            st.info("要至少 2 个分组才能做差异分析。")
        else:
            st.info("没有算出 dPSI，请检查分组名是否和 PSI 表的列名对得上。")
    else:
        used = res.get("method", "empirical")
        st.subheader("显著事件汇总（ASTK empirical 经验分布法 + 基因内 BH）" if used == "empirical"
                     else "显著事件汇总（Welch t 检验 + BH FDR）")
        sig_sum = core.significant_summary(dpsi, abs_dpsi=abs_dpsi, qval=qval_cut)
        st.dataframe(sig_sum, hide_index=True)
        st.caption((core.EMPIRICAL_NOTE if used == "empirical" else
                    "网页端用 Welch t 检验 + BH 校正；这个口径和 `astk dsflow` 不同，"
                    "显著事件数对不上是正常的。") + " " + core.COMPARISON_NOTE)
        if used == "empirical":
            bg = pd.concat([d["background"] for d in dpsi.values()]) if dpsi else pd.Series(dtype=str)
            if len(bg) and bool((bg == "global").any()):
                st.info("这批数据没有表达量信息（只上传了 PSI 文件），本地背景退回「全部事件的噪声」，"
                        "和命令行 `astk dsflow`（用表达量窗口）会有差异；"
                        "上传 quant.sf 或 TPM 矩阵就能完全对齐。")
        avail = [et for et in AS_ORDER if et in dpsi] + [et for et in dpsi if et not in AS_ORDER]
        col_et, col_cmp = st.columns(2)
        with col_et:
            et_d = st.selectbox("事件类型", avail, key="diff_et",
                                format_func=lambda x: core.AS_LABELS.get(x, x))
        comps_here = list(dict.fromkeys(dpsi[et_d]["comparison"].astype(str)))
        with col_cmp:
            if len(comps_here) > 1:
                comp_d = st.selectbox("比较对", comps_here, key="diff_comp")
            else:
                comp_d = comps_here[0]
                st.text_input("比较对", value=comp_d, disabled=True, key="diff_comp_1")
        d_d = dpsi[et_d]
        d_d = d_d[d_d["comparison"].astype(str) == comp_d]
        c1, c2 = st.columns(2)
        with c1:
            fig = plots.fig_volcano(d_d, abs_dpsi=abs_dpsi, qval=qval_cut, event_type=et_d,
                                    top_label=int(top_label))
            if fig is None:
                st.info("这一比较对没有 p 值（有分组的样本数不足 2），火山图无法绘制；"
                        "看右边的散点图和下面的表格。")
            else:
                _show(fig)
        with c2:
            fig = plots.fig_scatter(d_d, event_type=et_d, abs_dpsi=abs_dpsi, qval=qval_cut)
            if fig is not None:
                _show(fig)
        if et_d in ioe_tables:
            meta = ioe_tables[et_d].set_index("event_id")[["seqname", "gene_id"]]
            meta = meta[~meta.index.duplicated()]
            stdata = d_d.join(meta, how="left")
        else:
            stdata = d_d
        st.dataframe(stdata.sort_values("dpsi", key=lambda s: s.abs(), ascending=False).round(4),
                     height=380)

with tab_dl:
    payload = {f"{et}_psi.tsv": psi[et] for et in present}
    for et in dpsi:
        payload[f"{et}_dpsi.tsv"] = dpsi[et]
    if dpsi:
        payload["significant_summary.tsv"] = core.significant_summary(dpsi, abs_dpsi, qval_cut)
    figures = {
        "fig_event_counts.png": plots.fig_event_counts(
            core.event_counts(ioe_tables) if ioe_tables else pd.DataFrame(
                [{"event_type": et, "n_events": psi[et].shape[0]} for et in present])),
        "fig_psi_distribution.png": plots.fig_psi_distribution(psi, groups),
        "fig_pca.png": plots.fig_pca(pd.concat([psi[et] for et in present], axis=0), groups),
        "fig_corr_heatmap.png": plots.fig_corr_heatmap(pd.concat([psi[et] for et in present], axis=0), groups),
    }
    diff_ets = [et for et in AS_ORDER if et in dpsi] + [et for et in dpsi if et not in AS_ORDER]
    if diff_ets:
        default_et = ["AF"] if "AF" in diff_ets else diff_ets[:1]
        st.markdown("**差异图要打包哪几类事件？** 多分组时「每类事件 x 每个比较对」都要出两张图，"
                    "全打包会有几十张；只留 AF 最省事。")
        picked_et = st.multiselect("要打包的事件类型", diff_ets, default=default_et,
                                   format_func=lambda x: core.AS_LABELS.get(x, x), key="dl_et")
        for et in picked_et:
            sub = dpsi[et]
            for comp in dict.fromkeys(sub["comparison"].astype(str)):
                part = sub[sub["comparison"].astype(str) == comp]
                figures[f"volcano_{et}_{comp}.png"] = plots.fig_volcano(part, abs_dpsi, qval_cut, et)
                figures[f"scatter_{et}_{comp}.png"] = plots.fig_scatter(part, et, abs_dpsi, qval_cut)
    st.download_button("下载全部结果（zip）", _zip_results(payload, figures),
                       file_name="astk_web_results.zip", mime="application/zip", type="primary")
    st.caption(f"包含 {len(payload)} 个表格、"
               f"{sum(1 for f in figures.values() if f is not None)} 张图，以及 CITATION.txt（引用信息）。")
    st.warning("用这些结果发表工作时，请引用 ASTK：" + core.ASTK_CITATION)
