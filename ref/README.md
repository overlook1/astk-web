# ref/ —— 内置事件参考集

把生成好的 `.ioe` 放在这里。部署之后，别人只要上传 `quant.sf` 就能算出 PSI，
既不用自己准备事件定义，云端也不需要装 astk。**这是让陌生人真的用起来的关键一步。**

目录结构：一个物种一个子目录。

```
ref/
├── mouse/                      # 小鼠 GRCm38 / M25
│   ├── annotation_SE_strict.ioe
│   ├── annotation_A5_strict.ioe
│   └── ... 共 7 个（SE A5 A3 MX RI AF AL）
├── human/                      # 人 GRCh38
│   └── ...
└── rat/
    └── ...
```

## 怎么生成

在有 astk 的机器上（例如服务器的 `astk_env`），每个物种跑一次：

```bash
python make_ioe.py /home/public/ref/genome/mm/release_M25/gencode.vM25.annotation.gtf ref/mouse
python make_ioe.py /path/to/gencode.v38.annotation.gtf                        ref/human
```

一个物种只有几 MB，建议直接提交进仓库。

## 物种目录名

`human` / `mouse` / `rat` / `zebrafish` / `fly` 会显示成中文名（见 `core.SPECIES_LABELS`），
用别的名字也可以，界面会直接显示目录名。也允许把 `.ioe` 直接放在 `ref/` 下（不分子目录），
这时显示为「默认参考集」。

## 注意

- 参考集必须和样本的**转录本 ID 体系**一致（都用 GENCODE，或都用 Ensembl）。
  对不上时 PSI 会全是 `NaN`。
- 不同物种不要混用；同一批样本始终用同一份参考集，不同批次的 PSI 才可比。
