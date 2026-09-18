# ASTK Web —— 转录组可变剪接（AS）在线分析

上传 **转录本表达定量**（salmon `quant.sf` 或 TPM 矩阵），网页输出 **七类可变剪接事件
（SE / A5 / A3 / MX / RI / AF / AL）** 的 PSI 矩阵、事件谱、PCA、样本相关性热图、
PSI 热图、火山图、PSI 散点图，并打包下载全部表格与图片。

> **请引用 ASTK。** 本网站是 [ASTK](https://github.com/huang-sh/astk) 的网页前端，
> 事件定义与 PSI 口径都和 ASTK / SUPPA2 一致。用本网站的结果发表工作时请引用：
>
> Huang, S., He, J., Yu, L., Guo, J., Jiang, S., Sun, Z., Cheng, L., Chen, X., Ji, X. and
> Zhang, Y. (2024), ASTK: A Machine Learning-Based Integrative Software for Alternative
> Splicing Analysis. *Adv. Intell. Syst.* 2300594. <https://doi.org/10.1002/aisy.202300594>
>
> 网页「⑧ 关于与引用」里有可复制的 BibTeX，下载的 zip 里带 `CITATION.txt`，
> 导出的每张图右下角带引用水印。

---

## 1. 部署成公开网站（Streamlit Community Cloud，免费）

这是给别人用的方式：部署完拿到一个公开网址（形如 `https://<名字>.streamlit.app`），
任何人打开就能传 `quant.sf` 出结果。

1. 推到 GitHub（**建议用 public 仓库**：别人能看源码，也更容易被搜到）：

   ```bash
   cd <仓库根目录>
   git add astk_web
   git commit -m "astk web"
   git push
   ```

2. 打开 <https://share.streamlit.io> → **New app** → 选仓库和分支。
3. **Main file path** 填 `astk_web/app.py`（若仓库根目录就是 `astk_web/`，填 `app.py`）。
4. **Deploy**。首次构建 3–5 分钟。

`requirements.txt`、`packages.txt`（中文字体）、`.streamlit/config.toml` 会被自动识别。

**免费版的真实限制，心里有数就行：**

- 一段时间没人访问会**休眠**，下一个访问者要等 20–40 秒才醒（醒来后正常）。
- 内存约 1 GB，别指望在上面做比对或者处理几万个转录本的大矩阵。
- 单次上传上限 500 MB（可在 `.streamlit/config.toml` 改）。
- 让别人真正能用的关键，是仓库里带上内置参考集 `ref/` —— 见下一节。

### 用 Docker 跑（自己的服务器 / 内网）

```bash
docker build -t astk-web ./astk_web
docker run -p 8501:8501 astk-web
```

---

## 2. 关键一步：内置参考集 `ref/`

这一步决定「别人能不能直接用」。没有参考集，访客得自己先准备 7 个 `.ioe`，
绝大多数人到这一步就走了 —— 所以公开站点**一定要带**。

在有 astk 的机器上，每个物种各跑一次：

```bash
python make_ioe.py /home/public/ref/genome/mm/release_M25/gencode.vM25.annotation.gtf ref/mouse
python make_ioe.py /path/to/gencode.v38.annotation.gtf                        ref/human
```

生成 `ref/mouse/annotation_SE_strict.ioe` 等 7 个文件（一个物种只有几 MB）。
把整个 `ref/` 提交到仓库，网站上「事件参考集 → 使用仓库内置 ref/」就能选到物种。

> 细节见 `ref/README.md`。小鼠用 `gencode.vM25.annotation.gtf`（GRCm38 / M25），
> 人用 `gencode.v4x.annotation.gtf`。参考集的转录本 ID 体系必须和样本一致，
> 对不上时 PSI 会全是 `NaN`。

---

## 3. 本地运行（自己先试）

### Windows：双击 `启动.bat`

第一次会自动建 `.venv` 并装依赖（几分钟），之后每次双击就能打开浏览器。
前提是装了 Python 3.10+（安装时勾上 **Add python.exe to PATH**）。

### 命令行方式（Windows / macOS / Linux）

```bash
conda create -n astk_web python=3.11 -y
conda activate astk_web
pip install -r requirements.txt
streamlit run app.py

# 自检（不需要 astk）：合成数据跑通全流程 + 上传路径，26 项检查
python selftest.py
```

打开 `http://localhost:8501`。先点「演示数据（免上传）」就能看到全部图表，不用准备任何文件。

> `localhost:8501` 只有在**跑浏览器的这台机器**上启动了网站才打得开。
> 如果网站跑在服务器上，要先建隧道：
> `ssh -L 8501:localhost:8501 huhaoran@<服务器地址>`，
> 再在服务器上 `streamlit run astk_web/app.py --server.port 8501`。
> 在服务器上跑还有个好处：那台机器有 astk，「事件参考集」可以直接选
> **上传 GTF（现场生成）**，不用事先做 `ref/`。

---

## 4. 输入文件格式

| 输入 | 说明 |
| --- | --- |
| `quant.sf`（salmon） | 每样本一个文件。第 1 列 = 转录本 ID，**第 4 列 = TPM**。直接多选上传，或用 zip 打包成 `<样本名>/quant.sf` |
| TPM 矩阵 | 首列 = 转录本 ID，其余每列一个样本 |
| `.ioe` 事件参考 | ASTK 生成，列固定为 `seqname / gene_id / event_id / alternative_transcripts / total_transcripts`，每种事件一个文件 |
| GTF | 现场生成事件参考（需要服务器装了 `astk`，大基因组较慢） |
| 已有 PSI 结果 | 每行一个事件、每列一个样本的 PSI 表，直接跳到图表 |

分组：在「① 数据与分组」表格里把每个样本标成 `ctrl` / `case`
（默认按样本名里的 `cko / ko / case / treat / mut` 猜 case）。

---

## 5. 云端做不到的事（重要边界）

Streamlit Community Cloud **不适合做 FASTQ 比对**：

- 没有参考基因组索引（bowtie2 / STAR / HISAT2 索引动辄 3–40 GB，超过云端磁盘与内存上限）
- 单次运行时间与内存都有硬限制，长任务会被 kill

所以推荐的**本地 / 云端分工**是：

```
FASTQ
  └─ (本地或服务器) fastp / trim_galore   质控
  └─ (本地或服务器) bowtie2 / STAR 比对 + samtools 排序
  └─ (本地或服务器) salmon quant  ->  quant.sf     ★ 这一步之后的都交给网站
        └─ (网站) quant.sf -> PSI -> dPSI -> 图表 -> 下载
```

同理，ChIP-seq 的 bigWig / 表观信号叠加（`astk se`、`astk signalHeatmap`）
需要 bam/bw 大文件，也建议在本地或服务器上跑。

---

## 6. 网页端统计的口径

- **PSI**：`sum(alt TPM) / sum(total TPM)`；当 total 转录本平均 TPM < 阈值 → `NaN`；alt TPM ≤ 0.0001 → 0
- **dPSI** = `mean(PSI, case) - mean(PSI, ctrl)`
- **检验**：Welch t 检验 + Benjamini–Hochberg FDR（`qval`），显著 = `qval ≤ 阈值` 且 `|dPSI| ≥ 阈值`
  - 这是给网页用的**快速筛选**；正式发表口径请回到命令行
    `astk dsflow` / `astk diffSplice`（SUPPA2 经验法）

---

## 7. 文件说明

| 文件 | 作用 |
| --- | --- |
| `app.py` | Streamlit 主程序（8 个 tab：数据与分组 / 概览 / 七类事件表 / PCA / 热图 / 差异分析 / 下载 / 关于与引用） |
| `core.py` | 数据层：读 ioe / quant.sf / TPM 矩阵 / PSI，计算 PSI 与 dPSI，引用信息常量。**唯一可能 import astk 的模块** |
| `plots.py` | 绘图层：纯 matplotlib 函数，每个函数返回一个 Figure（含引用水印） |
| `make_ioe.py` | 在有 astk 的机器上生成 `ref/<物种>/*.ioe`（每个物种跑一次） |
| `selftest.py` | 自检脚本：合成数据跑通全流程 + 上传路径，不需要 astk |
| `ref/` | 内置事件参考集，一个物种一个子目录（见 `ref/README.md`） |
| `启动.bat` | Windows 双击启动 |
| `requirements.txt` | Python 依赖 |
| `packages.txt` | 系统包（中文字体） |
| `Dockerfile` / `.dockerignore` | 容器化部署 |
| `.streamlit/config.toml` | 上传大小上限、主题色 |

---

## 8. 引用信息改在哪里

给这个站点的维护者：所有引用文案都集中在 `core.py` 顶部，改一处就全局生效。

- `ASTK_CITATION` / `ASTK_DOI` —— 正文引用与链接
- `ASTK_BIBTEX` —— 「关于与引用」里可复制的 BibTeX
- `plots.ATTRIBUTION` —— 导出的 PNG 右下角水印
- `core.CITATION_NOTE` —— 下载 zip 里的 `CITATION.txt`
