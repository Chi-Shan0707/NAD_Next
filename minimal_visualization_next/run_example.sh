#!/bin/bash
# 示例启动脚本（NAD_NEXT 流式读取版本）

# 切换到脚本所在目录，确保相对路径正确
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 获取NAD_Next根目录（父目录）
NAD_NEXT_ROOT="$(cd .. && pwd)"

# 将NAD_Next加入PYTHONPATH
export PYTHONPATH="${NAD_NEXT_ROOT}:$PYTHONPATH"

# 设置NAD_NEXT cache 根目录（使用相对路径）
# 默认使用 ../cache_aime24，可通过环境变量覆盖
DATA_DIR="${VIS_DATA_DIR:-${NAD_NEXT_ROOT}/cache_aime24}"

# 设置端口
PORT=5002

# 设置 LRU 缓存上限（MB）
MAX_CACHE_MB=256

# 启动可视化服务器（流式读取模式）
python "$SCRIPT_DIR/app.py" \
    --data-dir "$DATA_DIR" \
    --port $PORT \
    --max-cache-mb $MAX_CACHE_MB

# Multi-cache mode (browse models/datasets in UI):
# VIS_DATA_DIR is ignored; use --cache-root instead
# python "$SCRIPT_DIR/app.py" --cache-root "$NAD_NEXT_ROOT/MUI_HUB" --port $PORT --max-cache-mb $MAX_CACHE_MB

# 可选参数：
# --max-cache-mb N  # LRU 缓存内存上限（MB），默认 256
#                   # 增大可减少重复计算，但会占用更多内存
# --cache-root DIR  # 多缓存模式：指定包含 model/dataset/cache 层级的根目录
#                   # 在 Web UI 中通过下拉菜单选择并切换缓存
#
# 注意：不再支持 --num-shards 和 --max-workers 参数（已移除）
#       流式架构按需加载，无需预先指定分片数量
