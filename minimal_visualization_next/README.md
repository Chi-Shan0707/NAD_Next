# NAD Next Streaming Visualization

`minimal_visualization_next` 提供了一个面向 NAD Next 缓存的最小可视化服务器。
相比旧目录，它只保留最新的 **流式加载** 逻辑，不再兼容 `viz_pack`、`npy_pack` 或 `npz`。

## Quick Start

```bash
pip install flask plotly numpy tqdm transformers  # 可选：transformers for tokenizer
export PYTHONPATH="/path/to/NAD_Next:$PYTHONPATH"
```

### 单缓存模式（直接指定一个 cache 目录）

```bash
python minimal_visualization_next/app.py --data-dir /path/to/cache --port 5002
```

### 多缓存模式（浏览 model/dataset/cache 层级）

```bash
python minimal_visualization_next/app.py --cache-root /path/to/MUI_HUB --port 5002
```

多缓存模式会扫描 `--cache-root` 下的三级目录结构：

```
MUI_HUB/cache/{model}/{dataset}/{cache_dir}/
  └── manifest.json   ← 用于验证是否为合法缓存
```

启动后页面顶部会出现 **Model / Dataset / Cache** 三级联动下拉菜单，
选择后点击 **Load** 即可在不重启服务器的情况下切换缓存。

运行成功后访问 `http://localhost:5002` 即可。

## CLI 参数

| 参数 | 说明 |
|------|------|
| `--data-dir` | 单缓存模式：指向一个 NAD_NEXT cache 目录 |
| `--cache-root` | 多缓存模式：指向包含 model/dataset/cache 层级的根目录 |
| `--port` | Web 服务端口（默认 5001） |
| `--host` | 绑定地址（默认 0.0.0.0） |
| `--max-cache-mb` | LRU 缓存内存上限 MB（默认 256） |

> `--data-dir` 和 `--cache-root` 二选一，不可同时使用。

## 主要特性

- **NadNextLoader** 流式按需加载切片与 token 数据。
- LRU 缓存与预聚合状态监控 (`/api/v1/precompute_status`)。
- 完整的 Plotly 前端（模板位于 `templates/index.html`）。
- Token 解码接口 `/api/decode_tokens`，自动尝试加载 `meta.json` 中的 tokenizer。
- **多缓存浏览**：通过 `--cache-root` 启用，支持运行时通过 UI 下拉菜单切换不同 model/dataset/cache。

## API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/status` | GET | 服务器状态（含 `multi_cache_mode` 字段） |
| `/api/cache_tree` | GET | 返回完整缓存树和当前选择（多缓存模式） |
| `/api/switch_cache` | POST | 切换到指定缓存（需传 `model`, `dataset`, `cache`） |
| `/api/plotly_data/<pid>` | GET | 获取问题的可视化数据 |
| `/api/decode_tokens` | POST | Token ID 解码 |
| `/api/search` | GET | 问题 ID 搜索 |

## 主要文件

- `app.py`：Flask 服务器入口。
- `templates/index.html`：前端页面（含 JS 逻辑）。
- `__init__.py`：便于 `python -m minimal_visualization_next.app` 启动。

## 运行脚本

本目录下的 `run_example.sh` 包含启动示例，支持单缓存和多缓存模式。
