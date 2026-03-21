#!/bin/bash

################################################################################
# Step 3.1: DeepConf 选择器专项分析
#
# 用途: 运行 DeepConf-style token confidence 选择器的专项分析和对比评估
#
# 使用方法:
#   ./step3.1_deepconf_analysis.sh [cache_name] [mode] [选项...]
#
# 参数:
#   cache_name  - Cache目录名称（可选，默认：MUI_public/cache/DeepSeek-R1-0528-Qwen3-8B/aime24/cache_neuron_output_1_act_no_rms_20250902_0256）
#   mode        - 分析模式（可选，默认：standard）
#                 - quick: 快速测试（仅默认配置）
#                 - standard: 标准分析（多种配置对比）
#                 - full: 完整分析（全序列 + position窗口）
#                 - custom: 自定义配置
#
# 选项:
#   --metric METRIC         - Token指标类型（默认：tok_conf）
#                             - tok_conf: DeepConf token confidence (-mean(log p_topk)) [推荐]
#                             - tok_neg_entropy: Negative entropy (sum p log p)
#   --reduction METHOD      - 聚合方法（默认：min_group）
#                             - min_group: 移动平均窗口的最小值 [推荐]
#                             - mean: 简单平均
#   --group-size N          - 移动平均窗口大小（默认：20）
#   --compare-all           - 与所有标准选择器对比
#   --compare-baseline      - 仅与baseline选择器对比
#   --threads N             - 距离计算线程数（默认：64）
#   --backend BACKEND       - Jaccard后端（默认：roaring）
#   --enable-profiling      - 启用性能分析
#   --log-level LEVEL       - 日志级别（默认：INFO）
#
# DeepConf 选择器说明:
#   DeepConf 使用缓存的 token-level 信息（confidence、entropy等）来选择
#   最佳样本。它基于以下假设：
#   - 高置信度的token序列代表更可靠的推理过程
#   - 通过聚合token级别的指标，可以评估整个response的质量
#   - 使用移动平均可以捕捉局部模式，避免单个token噪声
#
# Token 指标对比:
#   tok_conf (推荐):
#     - 计算: -mean(log p_topk)，即top-k概率的负对数均值
#     - 解释: 值越低，表示模型越有信心，推理越可靠
#     - 适用: 通用场景，对大多数任务表现良好
#
#   tok_neg_entropy:
#     - 计算: sum p log p，即负熵
#     - 解释: 值越接近0，表示分布越确定
#     - 适用: 需要精确控制不确定性的场景
#
# 聚合方法对比:
#   min_group (推荐):
#     - 先计算移动平均（平滑token序列）
#     - 再取最小值（找到最不确定的局部区域）
#     - 优势: 对局部不确定性敏感，捕捉推理瓶颈
#     - 适用: 需要保守选择的场景
#
#   mean:
#     - 简单平均所有token的指标值
#     - 优势: 计算简单，全局评估
#     - 适用: 对整体质量有要求的场景
#
# 示例:
#   # 使用默认配置（tok_conf + min_group + 窗口20）
#   ./step3.1_deepconf_analysis.sh
#
#   # 快速测试
#   ./step3.1_deepconf_analysis.sh quick
#
#   # 标准分析（多种配置对比）
#   ./step3.1_deepconf_analysis.sh standard
#
#   # 完整分析（包含position窗口）
#   ./step3.1_deepconf_analysis.sh full
#
#   # 自定义配置
#   ./step3.1_deepconf_analysis.sh custom --metric tok_neg_entropy --reduction mean
#
#   # 与所有选择器对比
#   ./step3.1_deepconf_analysis.sh standard --compare-all
#
#   # 指定缓存目录
#   ./step3.1_deepconf_analysis.sh ./cache_aime24 standard
#
#   # 启用性能分析
#   ./step3.1_deepconf_analysis.sh standard --enable-profiling
#
################################################################################

set -euo pipefail

# 获取脚本所在目录，切换到 repo root（上两级）
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}/../.."

################################################################################
# 颜色输出
################################################################################

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_info() {
    echo -e "${CYAN}ℹ${NC} $1"
}

################################################################################
# 默认配置
################################################################################

# 检查帮助参数
for arg in "$@"; do
    if [[ "$arg" == "--help" || "$arg" == "-h" ]]; then
        head -117 "$0" | tail -113
        exit 0
    fi
done

# 解析参数
CACHE_NAME=""
MODE=""
METRIC=""
REDUCTION=""
GROUP_SIZE=""
COMPARE_ALL=false
COMPARE_BASELINE=false
THREADS=""
BACKEND=""
ENABLE_PROFILING=false
LOG_LEVEL="INFO"

# 遍历所有参数
i=0
args=("$@")
while [ $i -lt ${#args[@]} ]; do
    arg="${args[$i]}"

    if [[ "$arg" == "--metric" ]]; then
        i=$((i + 1))
        if [ $i -lt ${#args[@]} ]; then
            METRIC="${args[$i]}"
        fi
    elif [[ "$arg" == "--reduction" ]]; then
        i=$((i + 1))
        if [ $i -lt ${#args[@]} ]; then
            REDUCTION="${args[$i]}"
        fi
    elif [[ "$arg" == "--group-size" ]]; then
        i=$((i + 1))
        if [ $i -lt ${#args[@]} ]; then
            GROUP_SIZE="${args[$i]}"
        fi
    elif [[ "$arg" == "--compare-all" ]]; then
        COMPARE_ALL=true
    elif [[ "$arg" == "--compare-baseline" ]]; then
        COMPARE_BASELINE=true
    elif [[ "$arg" == "--threads" ]]; then
        i=$((i + 1))
        if [ $i -lt ${#args[@]} ]; then
            THREADS="${args[$i]}"
        fi
    elif [[ "$arg" == "--backend" ]]; then
        i=$((i + 1))
        if [ $i -lt ${#args[@]} ]; then
            BACKEND="${args[$i]}"
        fi
    elif [[ "$arg" == "--enable-profiling" ]]; then
        ENABLE_PROFILING=true
    elif [[ "$arg" == "--log-level" ]]; then
        i=$((i + 1))
        if [ $i -lt ${#args[@]} ]; then
            LOG_LEVEL="${args[$i]}"
        fi
    elif [[ -z "$CACHE_NAME" && ! "$arg" =~ ^(quick|standard|full|custom)$ ]]; then
        CACHE_NAME="$arg"
    elif [[ -z "$MODE" ]]; then
        MODE="$arg"
    fi

    i=$((i + 1))
done

# 设置默认值
CACHE_NAME="${CACHE_NAME:-MUI_public/cache/DeepSeek-R1-0528-Qwen3-8B/aime24/cache_neuron_output_1_act_no_rms_20250902_0256}"
MODE="${MODE:-standard}"
METRIC="${METRIC:-tok_conf}"
REDUCTION="${REDUCTION:-min_group}"
GROUP_SIZE="${GROUP_SIZE:-20}"
THREADS="${THREADS:-16}"
BACKEND="${BACKEND:-roaring}"

# 设置 NAD_JA_BACKEND 环境变量
if [ -z "${NAD_JA_BACKEND:-}" ]; then
    export NAD_JA_BACKEND="$BACKEND"
fi

# 规范化 CACHE_NAME 路径
CACHE_NAME="${CACHE_NAME%/}"

if [[ "$CACHE_NAME" = /* ]]; then
    CACHE_ROOT="$CACHE_NAME"
    CACHE_BASENAME=$(basename "$CACHE_NAME")
    OUTPUT_DIR="./results_deepconf_${CACHE_BASENAME}"
else
    CACHE_NAME="${CACHE_NAME#./}"
    CACHE_ROOT="./${CACHE_NAME}"
    OUTPUT_DIR="./results_deepconf_${CACHE_NAME}"
fi

# 验证模式参数
if [[ "$MODE" != "quick" && "$MODE" != "standard" && "$MODE" != "full" && "$MODE" != "custom" ]]; then
    print_error "无效的模式: $MODE"
    echo ""
    echo "支持的模式:"
    echo "  - quick: 快速测试（仅默认配置）"
    echo "  - standard: 标准分析（多种配置对比）"
    echo "  - full: 完整分析（全序列 + position窗口）"
    echo "  - custom: 自定义配置"
    exit 1
fi

# 验证cache目录
if [ ! -d "$CACHE_ROOT" ]; then
    print_error "Cache目录不存在: $CACHE_ROOT"
    echo ""
    echo "提示: 请先运行 step1_build_cache.sh 构建cache"
    exit 1
fi

# 验证meta.json存在
if [ ! -f "$CACHE_ROOT/meta.json" ]; then
    print_error "Cache目录中缺少 meta.json: $CACHE_ROOT/meta.json"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

################################################################################
# 显示配置信息
################################################################################

print_header "Step 3.1: DeepConf 选择器专项分析"
echo ""
echo -e "${BLUE}配置信息:${NC}"
echo "  Cache目录:      $CACHE_ROOT"
echo "  分析模式:       $MODE"
echo "  Token指标:      $METRIC"
echo "  聚合方法:       $REDUCTION"
echo "  窗口大小:       $GROUP_SIZE"
echo "  距离计算线程:   $THREADS"
echo "  Jaccard后端:    $NAD_JA_BACKEND"
echo "  日志级别:       $LOG_LEVEL"
echo "  性能分析:       $([ "$ENABLE_PROFILING" = true ] && echo "启用" || echo "禁用")"
echo "  输出目录:       $OUTPUT_DIR"
echo ""

if [ "$COMPARE_ALL" = true ]; then
    print_info "将与所有标准选择器进行对比"
elif [ "$COMPARE_BASELINE" = true ]; then
    print_info "将与baseline选择器进行对比"
fi

echo ""

################################################################################
# 缓存验证
################################################################################

print_info "验证缓存支持的 token metrics..."

python3 << EOF
import sys
from pathlib import Path

try:
    # Import after changing to script directory
    from nad.core.views.reader import CacheReader
except ImportError as e:
    print(f"ERROR: Failed to import NAD modules: {e}")
    print("       Ensure you are running from NAD_Next directory")
    sys.exit(1)

cache_root = "${CACHE_ROOT}"
requested_metric = "${METRIC}"

try:
    # Read cache
    reader = CacheReader(cache_root)

    if reader.num_runs == 0:
        print("ERROR: Cache contains no samples")
        sys.exit(1)

    # Check token data on first sample
    token_view = reader.get_token_view(0)

    # Detect available metrics
    available = []
    if hasattr(token_view, 'tok_conf') and token_view.tok_conf is not None:
        available.append("tok_conf")
    if hasattr(token_view, 'tok_neg_entropy') and token_view.tok_neg_entropy is not None:
        available.append("tok_neg_entropy")

    if not available:
        print("")
        print("ERROR: Cache does not contain token-level metrics")
        print("       Cache was likely built without token data")
        print("       Please rebuild cache with NAD v4+ and enable token collection")
        print("")
        sys.exit(1)

    # Check if requested metric is available
    if requested_metric not in available:
        print("")
        print(f"ERROR: Requested metric '{requested_metric}' not found in cache")
        print(f"       Available metrics: {', '.join(available)}")
        print("")
        print("       Please use one of the available metrics:")
        for m in available:
            print(f"         --metric {m}")
        print("")
        sys.exit(1)

    # Validation passed
    print(f"✓ Cache validation passed")
    print(f"  Available metrics: {', '.join(available)}")
    print(f"  Using metric: {requested_metric}")

except Exception as e:
    print("")
    print(f"ERROR: Cache validation failed: {e}")
    print("")
    sys.exit(1)
EOF

if [ $? -ne 0 ]; then
    echo ""
    print_error "缓存验证失败，无法继续"
    exit 1
fi

echo ""

################################################################################
# 辅助函数
################################################################################

# 运行单个 DeepConf 配置分析
run_deepconf_analysis() {
    local METRIC="$1"
    local REDUCTION="$2"
    local GROUP_SIZE="$3"
    local OUTPUT_FILE="$4"
    local DESC="$5"

    echo ""
    print_info "$DESC"
    echo "--------------------------------------------------------------------------------"

    # 构建选择器规格
    local SELECTOR_SPEC="deepconf"
    if [ -n "$METRIC" ]; then
        SELECTOR_SPEC="${SELECTOR_SPEC}[metric=${METRIC}"
        if [ -n "$REDUCTION" ]; then
            SELECTOR_SPEC="${SELECTOR_SPEC},reduction=${REDUCTION}"
            if [ -n "$GROUP_SIZE" ]; then
                SELECTOR_SPEC="${SELECTOR_SPEC},group_size=${GROUP_SIZE}"
            fi
        fi
        SELECTOR_SPEC="${SELECTOR_SPEC}]"
    fi

    # 构建命令
    local CMD="python3 -m nad.cli --log-level ${LOG_LEVEL} analyze"
    CMD="${CMD} --cache-root ${CACHE_ROOT}"
    CMD="${CMD} --distance ja"
    CMD="${CMD} --group-topk-policy legacy-min"
    CMD="${CMD} --selectors ${SELECTOR_SPEC}"
    CMD="${CMD} --distance-threads ${THREADS}"
    CMD="${CMD} --out ${OUTPUT_FILE}"

    if [ "$ENABLE_PROFILING" = true ]; then
        CMD="${CMD} --enable-profiling"
    fi

    # 运行分析
    ${CMD}

    if [ $? -eq 0 ]; then
        print_success "分析完成"

        # 计算准确率
        echo ""
        print_info "计算准确率..."

        local ACC_FILE="${OUTPUT_FILE%.json}_accuracy.json"
        python3 -m nad.cli accuracy \
            --selection "${OUTPUT_FILE}" \
            --cache-root "${CACHE_ROOT}" \
            --out "${ACC_FILE}"

        if [ $? -eq 0 ]; then
            print_success "准确率计算完成"

            # 显示准确率
            echo ""
            python3 << EOF
import json
from pathlib import Path

acc_file = Path("${ACC_FILE}")
if acc_file.exists():
    with open(acc_file) as f:
        data = json.load(f)

    selector_accuracy = data.get('selector_accuracy', {})
    selector_counts = data.get('selector_counts', {})

    if selector_accuracy:
        print("DeepConf 准确率:")
        for name, acc_value in selector_accuracy.items():
            counts = selector_counts.get(name, {"correct": 0, "total": 0})
            correct = counts['correct']
            total = counts['total']
            print(f"  {name}: {acc_value:.2f}% ({correct}/{total})")
EOF
        else
            print_warning "准确率计算失败"
        fi
    else
        print_error "分析失败"
        return 1
    fi

    echo ""
}

# 运行对比分析
run_comparison_analysis() {
    local OUTPUT_FILE="$1"
    local SELECTOR_LIST="$2"
    local DESC="$3"

    echo ""
    print_info "$DESC"
    echo "--------------------------------------------------------------------------------"

    # 构建命令
    local CMD="python3 -m nad.cli --log-level ${LOG_LEVEL} analyze"
    CMD="${CMD} --cache-root ${CACHE_ROOT}"
    CMD="${CMD} --distance ja"
    CMD="${CMD} --group-topk-policy legacy-min"
    CMD="${CMD} --selectors ${SELECTOR_LIST}"
    CMD="${CMD} --distance-threads ${THREADS}"
    CMD="${CMD} --out ${OUTPUT_FILE}"

    if [ "$ENABLE_PROFILING" = true ]; then
        CMD="${CMD} --enable-profiling"
    fi

    # 运行分析
    ${CMD}

    if [ $? -eq 0 ]; then
        print_success "对比分析完成"

        # 计算准确率
        echo ""
        print_info "计算准确率..."

        local ACC_FILE="${OUTPUT_FILE%.json}_accuracy.json"
        python3 -m nad.cli accuracy \
            --selection "${OUTPUT_FILE}" \
            --cache-root "${CACHE_ROOT}" \
            --out "${ACC_FILE}"

        if [ $? -eq 0 ]; then
            print_success "准确率计算完成"

            # 显示排名
            echo ""
            python3 << EOF
import json
from pathlib import Path

acc_file = Path("${ACC_FILE}")
if acc_file.exists():
    with open(acc_file) as f:
        data = json.load(f)

    selector_accuracy = data.get('selector_accuracy', {})
    selector_counts = data.get('selector_counts', {})

    if selector_accuracy:
        sorted_selectors = sorted(
            selector_accuracy.items(),
            key=lambda x: x[1],
            reverse=True
        )

        print("选择器准确率排名（从高到低）:")
        print("")
        print(f"{'排名':<6s} {'选择器名称':<40s} {'准确率':>12s} {'正确数/总数':>15s}")
        print("-" * 80)

        for rank, (name, acc_value) in enumerate(sorted_selectors, 1):
            counts = selector_counts.get(name, {"correct": 0, "total": 0})
            correct = counts['correct']
            total = counts['total']

            # 标记 DeepConf 选择器
            marker = "★" if "deepconf" in name.lower() or "deep" in name.lower() else " "
            print(f"{marker} {rank:<4d} {name:<40s} {acc_value:>11.2f}% {correct:>6d}/{total:<6d}")
        print()
EOF
        fi
    else
        print_error "对比分析失败"
        return 1
    fi

    echo ""
}

################################################################################
# 主程序
################################################################################

# 记录开始时间
START_TIME=$(date +%s)

echo "================================================================================"
echo "开始分析..."
echo "================================================================================"

if [[ "$MODE" == "quick" ]]; then
    # 快速测试：仅运行默认配置
    run_deepconf_analysis \
        "$METRIC" \
        "$REDUCTION" \
        "$GROUP_SIZE" \
        "${OUTPUT_DIR}/deepconf_default.json" \
        "运行默认 DeepConf 配置 (${METRIC}, ${REDUCTION}, window=${GROUP_SIZE})"

elif [[ "$MODE" == "standard" ]]; then
    # 标准分析：测试多种配置
    print_info "运行多种 DeepConf 配置对比..."
    echo ""

    # 配置1: 默认配置 (tok_conf + min_group + 20)
    run_deepconf_analysis \
        "tok_conf" \
        "min_group" \
        "20" \
        "${OUTPUT_DIR}/deepconf_conf_mingroup_20.json" \
        "[1/4] tok_conf + min_group + window=20 (默认推荐配置)"

    # 配置2: tok_conf + mean
    run_deepconf_analysis \
        "tok_conf" \
        "mean" \
        "" \
        "${OUTPUT_DIR}/deepconf_conf_mean.json" \
        "[2/4] tok_conf + mean (全局平均)"

    # 配置3: tok_neg_entropy + min_group
    run_deepconf_analysis \
        "tok_neg_entropy" \
        "min_group" \
        "20" \
        "${OUTPUT_DIR}/deepconf_negent_mingroup_20.json" \
        "[3/4] tok_neg_entropy + min_group + window=20"

    # 配置4: 窗口大小对比 (window=10)
    run_deepconf_analysis \
        "tok_conf" \
        "min_group" \
        "10" \
        "${OUTPUT_DIR}/deepconf_conf_mingroup_10.json" \
        "[4/4] tok_conf + min_group + window=10 (小窗口)"

    # 对比分析
    if [ "$COMPARE_ALL" = true ]; then
        run_comparison_analysis \
            "${OUTPUT_DIR}/comparison_all.json" \
            "deepconf,medoid,knn-medoid,dbscan-medoid,con64@,avg64@" \
            "DeepConf 与所有标准选择器对比"
    elif [ "$COMPARE_BASELINE" = true ]; then
        run_comparison_analysis \
            "${OUTPUT_DIR}/comparison_baseline.json" \
            "deepconf,con64@,avg64@" \
            "DeepConf 与 Baseline 选择器对比"
    fi

elif [[ "$MODE" == "full" ]]; then
    # 完整分析：全序列 + position窗口
    print_info "运行完整分析（全序列 + position窗口）..."
    echo ""

    # 全序列分析
    run_deepconf_analysis \
        "$METRIC" \
        "$REDUCTION" \
        "$GROUP_SIZE" \
        "${OUTPUT_DIR}/deepconf_full_sequence.json" \
        "[1/4] 全序列分析"

    # Position 0-1 (tokens 0-31)
    echo ""
    print_info "[2/4] Position 0-1 分析 (tokens 0-31)..."

    CMD="python3 -m nad.cli --log-level ${LOG_LEVEL} analyze"
    CMD="${CMD} --cache-root ${CACHE_ROOT}"
    CMD="${CMD} --distance ja"
    CMD="${CMD} --group-topk-policy legacy-min"
    CMD="${CMD} --selectors deepconf[metric=${METRIC},reduction=${REDUCTION},group_size=${GROUP_SIZE}]"
    CMD="${CMD} --distance-threads ${THREADS}"
    CMD="${CMD} --pos-window 0-1 --pos-size 32"
    CMD="${CMD} --out ${OUTPUT_DIR}/deepconf_window_0-1.json"

    if [ "$ENABLE_PROFILING" = true ]; then
        CMD="${CMD} --enable-profiling"
    fi

    ${CMD}

    if [ $? -eq 0 ]; then
        print_success "Position 0-1 分析完成"
        python3 -m nad.cli accuracy \
            --selection "${OUTPUT_DIR}/deepconf_window_0-1.json" \
            --cache-root "${CACHE_ROOT}" \
            --out "${OUTPUT_DIR}/deepconf_window_0-1_accuracy.json" > /dev/null 2>&1
    fi

    # Position 0-2 (tokens 0-63)
    echo ""
    print_info "[3/4] Position 0-2 分析 (tokens 0-63)..."

    CMD="python3 -m nad.cli --log-level ${LOG_LEVEL} analyze"
    CMD="${CMD} --cache-root ${CACHE_ROOT}"
    CMD="${CMD} --distance ja"
    CMD="${CMD} --group-topk-policy legacy-min"
    CMD="${CMD} --selectors deepconf[metric=${METRIC},reduction=${REDUCTION},group_size=${GROUP_SIZE}]"
    CMD="${CMD} --distance-threads ${THREADS}"
    CMD="${CMD} --pos-window 0-2 --pos-size 32"
    CMD="${CMD} --out ${OUTPUT_DIR}/deepconf_window_0-2.json"

    if [ "$ENABLE_PROFILING" = true ]; then
        CMD="${CMD} --enable-profiling"
    fi

    ${CMD}

    if [ $? -eq 0 ]; then
        print_success "Position 0-2 分析完成"
        python3 -m nad.cli accuracy \
            --selection "${OUTPUT_DIR}/deepconf_window_0-2.json" \
            --cache-root "${CACHE_ROOT}" \
            --out "${OUTPUT_DIR}/deepconf_window_0-2_accuracy.json" > /dev/null 2>&1
    fi

    # Position 0-8 (tokens 0-255)
    echo ""
    print_info "[4/4] Position 0-8 分析 (tokens 0-255)..."

    CMD="python3 -m nad.cli --log-level ${LOG_LEVEL} analyze"
    CMD="${CMD} --cache-root ${CACHE_ROOT}"
    CMD="${CMD} --distance ja"
    CMD="${CMD} --group-topk-policy legacy-min"
    CMD="${CMD} --selectors deepconf[metric=${METRIC},reduction=${REDUCTION},group_size=${GROUP_SIZE}]"
    CMD="${CMD} --distance-threads ${THREADS}"
    CMD="${CMD} --pos-window 0-8 --pos-size 32"
    CMD="${CMD} --out ${OUTPUT_DIR}/deepconf_window_0-8.json"

    if [ "$ENABLE_PROFILING" = true ]; then
        CMD="${CMD} --enable-profiling"
    fi

    ${CMD}

    if [ $? -eq 0 ]; then
        print_success "Position 0-8 分析完成"
        python3 -m nad.cli accuracy \
            --selection "${OUTPUT_DIR}/deepconf_window_0-8.json" \
            --cache-root "${CACHE_ROOT}" \
            --out "${OUTPUT_DIR}/deepconf_window_0-8_accuracy.json" > /dev/null 2>&1
    fi

elif [[ "$MODE" == "custom" ]]; then
    # 自定义配置
    run_deepconf_analysis \
        "$METRIC" \
        "$REDUCTION" \
        "$GROUP_SIZE" \
        "${OUTPUT_DIR}/deepconf_custom.json" \
        "运行自定义 DeepConf 配置 (${METRIC}, ${REDUCTION}, window=${GROUP_SIZE})"
fi

# 记录结束时间
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
MINUTES=$((DURATION / 60))
SECONDS=$((DURATION % 60))

################################################################################
# 分析完成总结
################################################################################

echo ""
print_header "分析完成"
echo ""
print_success "所有 DeepConf 分析完成！"
echo ""

echo -e "${BLUE}配置信息:${NC}"
echo "  Cache目录:      $CACHE_ROOT"
echo "  分析模式:       $MODE"
echo "  分析耗时:       ${MINUTES}分${SECONDS}秒"
echo ""

echo -e "${BLUE}生成的结果文件:${NC}"
ls -lh "$OUTPUT_DIR"/*.json 2>/dev/null | while IFS= read -r line; do
    filename=$(echo "$line" | awk '{print $NF}')
    size=$(echo "$line" | awk '{print $5}')
    echo "  $(basename "$filename") ($size)"
done
echo ""

echo "================================================================================"
echo ""
echo "查看详细结果:"
echo "  ls -lh ${OUTPUT_DIR}/"
echo ""
echo "查看准确率对比:"
echo "  cat ${OUTPUT_DIR}/*_accuracy.json | jq '.selector_accuracy'"
echo ""

# 生成汇总报告
if [[ "$MODE" == "standard" ]]; then
    echo "生成配置对比汇总..."
    python3 << 'EOF'
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
accuracy_files = sorted(output_dir.glob("deepconf_*_accuracy.json"))

if not accuracy_files:
    print("  未找到准确率文件")
    sys.exit(0)

print("\n" + "="*80)
print("DeepConf 配置对比汇总")
print("="*80)
print()

results = []
for acc_file in accuracy_files:
    config_name = acc_file.stem.replace("_accuracy", "").replace("deepconf_", "")

    try:
        with open(acc_file) as f:
            data = json.load(f)

        selector_accuracy = data.get('selector_accuracy', {})
        selector_counts = data.get('selector_counts', {})

        # Get the deepconf selector accuracy
        for name, acc_value in selector_accuracy.items():
            if 'deepconf' in name.lower():
                counts = selector_counts.get(name, {"correct": 0, "total": 0})
                results.append({
                    'config': config_name,
                    'accuracy': acc_value,
                    'correct': counts['correct'],
                    'total': counts['total']
                })
                break
    except:
        continue

if results:
    # Sort by accuracy
    results.sort(key=lambda x: x['accuracy'], reverse=True)

    print(f"{'排名':<6s} {'配置':<30s} {'准确率':>12s} {'正确数/总数':>15s}")
    print("-" * 70)

    for rank, r in enumerate(results, 1):
        marker = "★" if rank == 1 else " "
        print(f"{marker} {rank:<4d} {r['config']:<30s} {r['accuracy']:>11.2f}% {r['correct']:>6d}/{r['total']:<6d}")

    print()
    print("推荐配置: " + results[0]['config'])
    print()

EOF
python3 - "$OUTPUT_DIR"
fi

echo "================================================================================"
