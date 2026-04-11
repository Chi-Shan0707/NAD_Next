# Environment Guide

本目录提供一套面向 `SVDomain` 论文复现的环境文件。

---

## 文件说明

- `requirements-paper.txt`
  - 面向 `pip` / `.venv`
  - 与当前仓库实际依赖保持一致
- `environment.yml`
  - 面向 `conda`
  - 方便做独立复现实验
- `verify_imports.py`
  - 最小化导入检查脚本

---

## 推荐方式

### 方案 A：`.venv`

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r SVDomain/env/requirements-paper.txt
python SVDomain/env/verify_imports.py
```

### 方案 B：conda

```bash
conda env create -f SVDomain/env/environment.yml
conda activate svdomain-paper
python SVDomain/env/verify_imports.py
```

---

## 依赖设计原则

我们没有在这个目录里单独发明一套新的环境体系，而是遵循：

1. 与仓库 `requirements.txt` 对齐
2. 覆盖训练、viewer、解释性导出三类依赖
3. 保持尽量小而完整

---

## 额外说明

如果你直接运行：

```bash
bash cookbook/00_setup/install.sh
bash cookbook/00_setup/verify.sh
```

也可以完成环境配置。  
但如果你是为了 paper artifact 复现，更推荐本目录提供的独立环境文件。
