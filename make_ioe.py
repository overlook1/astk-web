"""Generate the seven .ioe event references, once, on a machine that has astk.

This script is self-contained - it imports nothing from the astk_web package -
so you can copy this single file anywhere. It only needs the astk Python package.

    python make_ioe.py <annotation.gtf> ref/<species>

For a public site that strangers can use, run it once per species:

    python make_ioe.py /home/public/ref/genome/mm/release_M25/gencode.vM25.annotation.gtf ref/mouse
    python make_ioe.py /path/to/gencode.v38.annotation.gtf                        ref/human

Then commit the resulting ref/ folder (a few MB per species). After that any
visitor can upload a salmon quant.sf, pick their species in the sidebar and get
PSI / dPSI / PCA / heatmaps - and the deployed site never needs ASTK installed
(astk is heavy and often fails to build on a cloud container).

Requires:  pip install astk     (or run it inside your astk_env environment)
"""
from __future__ import annotations

import sys
from pathlib import Path

AS_TYPES = ["SE", "A5", "A3", "MX", "RI", "AF", "AL"]


def astk_status() -> tuple:
    """Return (available, message) for the astk installation."""
    try:
        import astk  # noqa: F401
    except Exception as exc:
        return False, "%s: %s" % (exc.__class__.__name__, exc)
    return True, "ASTK OK"


def generate_ioe(gtf_path, outdir, events=None, idtype="SUPPA2", event_pos=None):
    """Build one ioe file per event type from a GTF.

    Mirrors `astk generateEvents`: writes <outdir>/annotation[_<pos>]_<ET>_strict.ioe
    """
    from astk.suppa.AS_event import make_events
    from astk.suppa.gtf_parse import construct_genome

    events = list(events or AS_TYPES)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    genome = construct_genome(str(gtf_path))
    make_events(str(outdir / "annotation"), genome, events, idtype, event_pos)
    suffix = "" if event_pos is None else "_%s" % event_pos
    return {et: outdir / ("annotation%s_%s_strict.ioe" % (suffix, et)) for et in events}


def main(argv) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2

    gtf = Path(argv[1])
    outdir = Path(argv[2]) if len(argv) > 2 else Path("ref")
    if not gtf.is_file():
        print("GTF 不存在: %s" % gtf)
        return 1

    ok, msg = astk_status()
    print(("OK   " if ok else "WARN ") + msg)
    if not ok:
        print("生成事件必须有 astk：pip install astk，或在服务器的 astk_env 环境里跑本脚本")
        return 1

    print("正在解析 %s（大基因组要几分钟）..." % gtf)
    paths = generate_ioe(gtf, outdir, events=AS_TYPES)

    missing = [et for et, p in paths.items() if not p.is_file()]
    for et, path in paths.items():
        print("  %s: %s  %s" % (et, "ok" if path.is_file() else "MISSING", path))
    if missing:
        print("未生成的事件类型: %s" % ", ".join(missing))
        return 1

    print("")
    print("完成。把 %s/ 提交到仓库，网站左侧「参考集」里就能选到它。" % outdir)
    print("公开部署建议至少带 mouse + human 两套，这样陌生人才不用自己准备 ioe。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))