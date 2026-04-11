# Reproduction Checklist

这份文档的目标是：让你从零开始，在当前仓库里复现 SVDomain 论文主线所需的关键材料。

---

## 1. 环境前置

### 1.1 Python

- 推荐：`Python 3.10`
- 最低：`Python >= 3.9`

### 1.2 数据

需要可访问：

- `MUI_HUB/cache`
- `MUI_HUB/cache_train`

仓库根目录通常通过：

```bash
ln -s /home/jovyan/public-ro/MUI_HUB MUI_HUB
```

建立软链接。

### 1.3 依赖

先看：

- `SVDomain/env/requirements-paper.txt`
- `SVDomain/env/environment.yml`
- `SVDomain/env/verify_imports.py`

---

## 2. 推荐安装方式

### 方案 A：仓库 `.venv`

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python SVDomain/env/verify_imports.py
```

### 方案 B：Conda

```bash
conda env create -f SVDomain/env/environment.yml
conda activate svdomain-paper
python SVDomain/env/verify_imports.py
```

---

## 3. 官方安装 / 校验脚本

仓库已有：

```bash
bash cookbook/00_setup/install.sh
bash cookbook/00_setup/verify.sh
```

注意：

- 当前机器的 system `python3` 可能缺少 `flask / plotly / hmmlearn / tokenizers / transformers`
- 已有工作通常是在仓库 `.venv` 内完成

因此如果 `verify.sh` 不全绿，先确认你是不是在正确环境里执行。

---

## 4. 复现 canonical `r1` 主线

### 4.1 训练 `math + science + ms`

```bash
python3 SVDomain/train_es_svd_ms_rr_r1.py
```

主要产物：

- `models/ml_selectors/es_svd_math_rr_r1.pkl`
- `models/ml_selectors/es_svd_science_rr_r1.pkl`
- `models/ml_selectors/es_svd_ms_rr_r1.pkl`
- `results/scans/earlystop/es_svd_ms_rr_r1_summary.json`
- `results/scans/earlystop/es_svd_ms_rr_r1_eval.json`

### 4.2 训练 coding 分支

```bash
python3 SVDomain/train_es_svd_coding_rr_r1.py
```

主要产物：

- `models/ml_selectors/es_svd_coding_rr_r1.pkl`
- `results/scans/earlystop/es_svd_coding_rr_r1_summary.json`
- `results/scans/earlystop/es_svd_coding_rr_r1_eval.json`
- `submission/EarlyStop/es_svd_ms_rr_r1__coding_rr_r1.json`

---

## 5. 复现解释性产物

### 5.1 smoke export

```bash
python3 scripts/export_svd_explanations.py --max-problems 1
```

### 5.2 full export

```bash
python3 scripts/export_svd_explanations.py
```

输出目录：

- `results/interpretability/es_svd_math_rr_r1/`
- `results/interpretability/es_svd_science_rr_r1/`
- `results/interpretability/es_svd_ms_rr_r1/`

---

## 6. 启动 viewer

推荐直接启动：

```bash
python3 cot_viewer/app.py
```

然后在浏览器里看：

- canonical SVD methods
- decision summary
- feature panel
- top1 / top2 / top3
- trajectory hover correctness

---

## 7. 关键 smoke checks

### 7.1 Python 语法

```bash
python3 -m py_compile \
  nad/explain/svd_explain.py \
  scripts/export_svd_explanations.py \
  cot_viewer/app.py
```

### 7.2 前端语法

```bash
node --check cot_viewer/static/app.js
```

### 7.3 API smoke

至少检查：

- `/api/svd/explain/model_summary`
- `/api/svd/explain/problem_top1_vs_top2`
- `/api/svd/explain/run_contributions`

---

## 8. 论文复现最小清单

如果你的目标只是“够写 paper”，最小需要复现这些 artifact：

1. `es_svd_ms_rr_r1_summary.json`
2. `es_svd_ms_rr_r1_eval.json`
3. `docs/ES_SVD_MS_RR_R1.md`
4. `results/interpretability/es_svd_ms_rr_r1/*`
5. blind leaderboard receipt / result summary

---

## 9. 常见坑

### 9.1 用错 Python 环境

这是最常见问题。  
症状通常是：

- `verify.sh` 报缺包
- `cot_viewer` 无法启动
- `transformers` 或 `plotly` import error

### 9.2 coding 域预期过高

`es_svd_coding_rr_r1` 目前不是更优主线，只是一个已验证分支。  
复现时请预期它更像 negative result。

### 9.3 interpretability artifact 体积

当前仓库 checked-in 的是 compact smoke export。  
如果你要画全量图表，需要自行跑 full export。

---

## 10. 最终检查

在你准备写正文前，建议确保下面几件事都具备：

- [ ] 环境可复现
- [ ] canonical `r1` 结果文件齐全
- [ ] blind leaderboard 关键数字已核对
- [ ] interpretability artifact 可读
- [ ] viewer 可以回答“为什么它是 top1？”
