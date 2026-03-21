#!/bin/bash

################################################################################
# 交互式清理 window_cache 目录
#
# 使用方法:
#   ./tools/clean_window_cache.sh [--dry-run] [--yes-all] [--base-dir DIR]
#
# 选项:
#   --dry-run    仅显示，不实际删除
#   --yes-all    自动确认删除所有（危险！）
#   --base-dir   指定搜索目录（默认: ./MUI_HUB/cache）
#
################################################################################

set -euo pipefail

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# 默认参数
DRY_RUN=false
YES_ALL=false
BASE_DIR="./MUI_HUB/cache"

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --yes-all)
            YES_ALL=true
            shift
            ;;
        --base-dir)
            BASE_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "使用方法: $0 [--dry-run] [--yes-all] [--base-dir DIR]"
            echo ""
            echo "选项:"
            echo "  --dry-run    仅显示，不实际删除"
            echo "  --yes-all    自动确认删除所有"
            echo "  --base-dir   指定搜索目录（默认: ./MUI_HUB/cache）"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

# 检查目录
if [[ ! -d "$BASE_DIR" ]]; then
    echo -e "${RED}错误: 目录不存在: $BASE_DIR${NC}"
    exit 1
fi

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  交互式清理 window_cache 目录${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

if [[ "$DRY_RUN" == "true" ]]; then
    echo -e "${YELLOW}[DRY-RUN 模式] 仅显示，不实际删除${NC}"
    echo ""
fi

# 查找所有 window_cache 目录
echo -e "${CYAN}正在扫描...${NC}"
CACHE_DIRS=$(find "$BASE_DIR" -maxdepth 5 -type d -name "window_cache" 2>/dev/null | sort)
TOTAL=$(echo "$CACHE_DIRS" | grep -c . || echo 0)

if [[ "$TOTAL" -eq 0 ]]; then
    echo -e "${GREEN}未发现 window_cache 目录${NC}"
    exit 0
fi

# 计算总大小
TOTAL_SIZE=$(echo "$CACHE_DIRS" | xargs -I {} du -s {} 2>/dev/null | awk '{sum+=$1} END {print sum}')
TOTAL_SIZE_GB=$(echo "scale=2; $TOTAL_SIZE / 1024 / 1024" | bc)

echo ""
echo -e "发现 ${YELLOW}$TOTAL${NC} 个 window_cache 目录，总计 ${YELLOW}${TOTAL_SIZE_GB} GB${NC}"
echo ""
echo -e "${CYAN}操作说明 (每10个一批):${NC}"
echo "  y/Y     - 删除本批次"
echo "  n/N/回车 - 跳过本批次"
echo "  a/A     - 删除所有剩余目录"
echo "  q/Q     - 退出"
echo ""
echo -e "${BLUE}────────────────────────────────────────────────────────────────────────────────${NC}"

# 统计
DELETED_COUNT=0
SKIPPED_COUNT=0
DELETED_SIZE=0
BATCH_SIZE=10

# 转换为数组
mapfile -t DIR_ARRAY <<< "$CACHE_DIRS"

# 按批次处理
BATCH_START=0
while [[ $BATCH_START -lt $TOTAL ]]; do
    BATCH_END=$((BATCH_START + BATCH_SIZE))
    [[ $BATCH_END -gt $TOTAL ]] && BATCH_END=$TOTAL

    echo ""
    echo -e "${BLUE}────────────────────────────────────────────────────────────────────────────────${NC}"
    echo -e "${CYAN}批次 $((BATCH_START/BATCH_SIZE + 1)): 目录 $((BATCH_START + 1)) - $BATCH_END / $TOTAL${NC}"
    echo -e "${BLUE}────────────────────────────────────────────────────────────────────────────────${NC}"

    # 收集本批次信息
    BATCH_DIRS=()
    BATCH_SIZES=()
    BATCH_TOTAL_SIZE=0

    for ((i=BATCH_START; i<BATCH_END; i++)); do
        dir="${DIR_ARRAY[$i]}"
        [[ -z "$dir" ]] && continue

        SIZE=$(du -sh "$dir" 2>/dev/null | cut -f1)
        SIZE_BYTES=$(du -s "$dir" 2>/dev/null | cut -f1)
        REL_PATH="${dir#$BASE_DIR/}"

        BATCH_DIRS+=("$dir")
        BATCH_SIZES+=("$SIZE_BYTES")
        BATCH_TOTAL_SIZE=$((BATCH_TOTAL_SIZE + SIZE_BYTES))

        echo -e "  [$((i + 1))] ${YELLOW}$SIZE${NC}  $REL_PATH"
    done

    BATCH_TOTAL_SIZE_GB=$(echo "scale=2; $BATCH_TOTAL_SIZE / 1024 / 1024" | bc)
    echo ""
    echo -e "  本批次: ${#BATCH_DIRS[@]} 个目录, ${YELLOW}${BATCH_TOTAL_SIZE_GB} GB${NC}"

    if [[ "$DRY_RUN" == "true" ]]; then
        echo -e "  ${YELLOW}[DRY-RUN] 将被删除${NC}"
        DELETED_COUNT=$((DELETED_COUNT + ${#BATCH_DIRS[@]}))
        DELETED_SIZE=$((DELETED_SIZE + BATCH_TOTAL_SIZE))
        BATCH_START=$BATCH_END
        continue
    fi

    if [[ "$YES_ALL" == "true" ]]; then
        echo -e "  ${RED}删除中...${NC}"
        for dir in "${BATCH_DIRS[@]}"; do
            rm -rf "$dir"
        done
        echo -e "  ${GREEN}✓ 已删除 ${#BATCH_DIRS[@]} 个目录${NC}"
        DELETED_COUNT=$((DELETED_COUNT + ${#BATCH_DIRS[@]}))
        DELETED_SIZE=$((DELETED_SIZE + BATCH_TOTAL_SIZE))
        BATCH_START=$BATCH_END
        continue
    fi

    # 交互式确认
    echo ""
    echo -n -e "  删除这 ${#BATCH_DIRS[@]} 个目录? [y/n/a/q]: "
    read -r REPLY </dev/tty
    REPLY="${REPLY:-n}"

    case "$REPLY" in
        y|Y)
            echo -e "  ${RED}删除中...${NC}"
            for dir in "${BATCH_DIRS[@]}"; do
                rm -rf "$dir"
            done
            echo -e "  ${GREEN}✓ 已删除 ${#BATCH_DIRS[@]} 个目录${NC}"
            DELETED_COUNT=$((DELETED_COUNT + ${#BATCH_DIRS[@]}))
            DELETED_SIZE=$((DELETED_SIZE + BATCH_TOTAL_SIZE))
            ;;
        a|A)
            echo -e "  ${RED}删除中...${NC}"
            for dir in "${BATCH_DIRS[@]}"; do
                rm -rf "$dir"
            done
            echo -e "  ${GREEN}✓ 已删除 ${#BATCH_DIRS[@]} 个目录${NC}"
            DELETED_COUNT=$((DELETED_COUNT + ${#BATCH_DIRS[@]}))
            DELETED_SIZE=$((DELETED_SIZE + BATCH_TOTAL_SIZE))
            YES_ALL=true
            echo -e "  ${YELLOW}已切换为自动删除模式${NC}"
            ;;
        q|Q)
            echo ""
            echo -e "${YELLOW}用户中断${NC}"
            break
            ;;
        *)
            echo -e "  ${CYAN}跳过本批次${NC}"
            SKIPPED_COUNT=$((SKIPPED_COUNT + ${#BATCH_DIRS[@]}))
            ;;
    esac

    BATCH_START=$BATCH_END
done

# 汇总
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  清理完成${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

DELETED_SIZE_GB=$(echo "scale=2; $DELETED_SIZE / 1024 / 1024" | bc)

if [[ "$DRY_RUN" == "true" ]]; then
    echo -e "${YELLOW}[DRY-RUN] 统计:${NC}"
    echo -e "  将删除: $DELETED_COUNT 个目录"
    echo -e "  将释放: ${DELETED_SIZE_GB} GB"
else
    echo -e "已删除: ${GREEN}$DELETED_COUNT${NC} 个目录"
    echo -e "已跳过: ${CYAN}$SKIPPED_COUNT${NC} 个目录"
    echo -e "释放空间: ${GREEN}${DELETED_SIZE_GB} GB${NC}"
fi
echo ""
