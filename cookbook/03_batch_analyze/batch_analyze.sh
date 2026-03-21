#!/bin/bash

################################################################################
# Step 2.2.1: NAD 批量分析脚本 (并行版本)
#
# 用途: 并行批量分析 MUI_HUB/cache 中所有模型和数据集的缓存
#
# 使用方法:
#   ./step2.2.1_batch_analyze_parallel.sh [选项...]
#
# 选项:
#   --mode MODE             - 分析模式 (默认: full)
#                             - full: 仅全序列分析
#                             - positions: 仅固定窗口分析 (0-1, 0-2, 0-8)
#                             - all: full + positions
#                             - all-positions: 全position细粒度分析 (0-1, 0-2, ..., 0-N)
#   --pos-max N             - all-positions模式下限制最大position (默认: 无限制)
#   --threads N             - 距离计算线程数 (默认: 8)
#   --backend BACKEND       - Jaccard后端 roaring/numpy/auto (默认: roaring)
#   --limit N               - 限制处理数量 (默认: 0表示全部)
#   --parallel N, -j N      - 并行任务数 (默认: 4)
#   --output-dir DIR        - 输出目录 (默认: ./result/all_model_TIMESTAMP)
#   --dry-run               - 仅显示任务列表，不实际运行
#   --auto-yes, -y          - 跳过交互式确认
#   --help, -h              - 显示帮助信息
#
# 与 step2.2 的区别:
#   - 支持 N 个任务并行处理，显著提升处理速度
#   - 每个任务有独立日志文件
#   - 简化的进度显示（适合并行场景）
#   - 支持 all-positions 全position细粒度分析模式
#
# 示例:
#   # 使用4个并行任务（默认）
#   ./step2.2.1_batch_analyze_parallel.sh
#
#   # 使用8个并行任务
#   ./step2.2.1_batch_analyze_parallel.sh -j 8
#
#   # 快速测试：仅处理前3个缓存，4并行
#   ./step2.2.1_batch_analyze_parallel.sh --limit 3 -j 4 -y
#
#   # 干运行模式：仅显示任务
#   ./step2.2.1_batch_analyze_parallel.sh --dry-run
#
#   # 完整配置
#   ./step2.2.1_batch_analyze_parallel.sh --mode all --threads 32 -j 8 --backend roaring
#
#   # 全position细粒度分析（限制前10个position）
#   ./step2.2.1_batch_analyze_parallel.sh --mode all-positions --pos-max 10 -j 4 -y
#
#   # 全position细粒度分析（不限制，分析所有position）
#   ./step2.2.1_batch_analyze_parallel.sh --mode all-positions -j 4 -y
#
################################################################################

set -euo pipefail

# 获取脚本所在目录，切换到 repo root（上两级）
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}/../.."

# 加载配置文件（如果存在）
if [ -f "${SCRIPT_DIR}/config.sh" ]; then
    source "${SCRIPT_DIR}/config.sh"
elif [ -f "${SCRIPT_DIR}/config_template.sh" ]; then
    echo "提示: 建议复制 config_template.sh 为 config.sh 并配置路径" >&2
fi

################################################################################
# 默认参数
################################################################################

CACHE_BASE="${MUI_PUBLIC_CACHE:-${BATCH_CACHE_BASE:-./MUI_HUB/cache}}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_BASE=""

# 分析参数
MODE="full"
THREADS=4
BACKEND="roaring"
LIMIT=0
POS_MAX=""  # 用于 all-positions 模式

# 并行参数
PARALLEL_JOBS=4
DRY_RUN=false
AUTO_YES=false

# 固定配置
LOG_LEVEL="INFO"

################################################################################
# 颜色输出
################################################################################

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

print_header() {
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

print_success() { echo -e "${GREEN}✓${NC} $1"; }
print_error() { echo -e "${RED}✗${NC} $1"; }
print_warning() { echo -e "${YELLOW}⚠${NC} $1"; }
print_info() { echo -e "${CYAN}ℹ${NC} $1"; }

################################################################################
# 参数解析
################################################################################

show_help() {
    head -55 "$0" | tail -51
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --pos-max)
            POS_MAX="$2"
            shift 2
            ;;
        --threads)
            THREADS="$2"
            shift 2
            ;;
        --backend)
            BACKEND="$2"
            shift 2
            ;;
        --limit)
            LIMIT="$2"
            shift 2
            ;;
        --parallel|-j)
            PARALLEL_JOBS="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_BASE="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --auto-yes|-y)
            AUTO_YES=true
            shift
            ;;
        --help|-h)
            show_help
            ;;
        *)
            # 兼容旧参数格式: mode threads backend limit auto_yes
            if [[ "$1" =~ ^(full|positions|all|all-positions)$ ]]; then
                MODE="$1"
                shift
            elif [[ "$1" =~ ^[0-9]+$ ]]; then
                if [[ -z "${_PARSED_THREADS:-}" ]]; then
                    THREADS="$1"
                    _PARSED_THREADS=1
                elif [[ -z "${_PARSED_LIMIT:-}" ]]; then
                    LIMIT="$1"
                    _PARSED_LIMIT=1
                fi
                shift
            elif [[ "$1" =~ ^(roaring|numpy|auto)$ ]]; then
                BACKEND="$1"
                shift
            elif [[ "$1" == "yes" ]]; then
                AUTO_YES=true
                shift
            else
                print_error "未知参数: $1"
                echo "使用 --help 查看帮助"
                exit 1
            fi
            ;;
    esac
done

# 设置输出目录
if [[ -z "$OUTPUT_BASE" ]]; then
    OUTPUT_BASE="./result/all_model_${TIMESTAMP}"
fi

# 设置环境变量
if [ -z "${NAD_JA_BACKEND:-}" ]; then
    export NAD_JA_BACKEND="$BACKEND"
fi

################################################################################
# 验证参数
################################################################################

print_header "Step 2.2.1: NAD 批量分析 (并行版本)"
echo ""

# 验证模式参数
if [[ "$MODE" != "full" && "$MODE" != "positions" && "$MODE" != "all" && "$MODE" != "all-positions" ]]; then
    print_error "无效的模式: $MODE"
    echo "支持的模式: full, positions, all, all-positions"
    exit 1
fi

# 验证cache基础目录
if [ ! -d "$CACHE_BASE" ]; then
    print_error "Cache基础目录不存在: $CACHE_BASE"
    exit 1
fi

################################################################################
# 扫描缓存目录
################################################################################

print_info "扫描缓存目录..."
CACHE_LIST=()
MODEL_COUNT=0

for MODEL_DIR in "$CACHE_BASE"/*; do
    if [ ! -d "$MODEL_DIR" ]; then
        continue
    fi

    MODEL_COUNT=$((MODEL_COUNT + 1))

    for DATASET_DIR in "$MODEL_DIR"/*; do
        if [ ! -d "$DATASET_DIR" ]; then
            continue
        fi

        # 仅处理 cache_neuron_output_1_* 开头的目录
        for CACHE_DIR in "$DATASET_DIR"/cache_neuron_output_1_*; do
            if [ -d "$CACHE_DIR" ]; then
                CACHE_LIST+=("$CACHE_DIR")
            fi
        done
    done
done

TOTAL_CACHES=${#CACHE_LIST[@]}

if [ $TOTAL_CACHES -eq 0 ]; then
    print_error "未发现任何缓存目录"
    exit 1
fi

# 应用限制
if [ $LIMIT -gt 0 ] && [ $LIMIT -lt $TOTAL_CACHES ]; then
    CACHE_LIST=("${CACHE_LIST[@]:0:$LIMIT}")
    TOTAL_CACHES=$LIMIT
    print_warning "测试模式：仅处理前 $LIMIT 个缓存"
fi

print_success "发现 $MODEL_COUNT 个模型"
print_success "发现 $TOTAL_CACHES 个待处理缓存（仅 type-1）"
echo ""

################################################################################
# 显示配置
################################################################################

echo -e "${BLUE}配置:${NC}"
echo "  分析模式:     $MODE"
if [[ "$MODE" == "all-positions" ]]; then
    if [[ -n "$POS_MAX" ]]; then
        echo "  最大Position: $POS_MAX"
    else
        echo "  最大Position: 自动检测"
    fi
fi
echo "  距离计算线程: $THREADS"
echo "  Jaccard后端:  $NAD_JA_BACKEND"
echo "  并行任务数:   $PARALLEL_JOBS"
echo "  待处理缓存:   $TOTAL_CACHES"
echo "  输出目录:     $OUTPUT_BASE/"
echo "  模式:         $([ "$DRY_RUN" = true ] && echo "干运行" || echo "实际执行")"
echo ""

if [[ "$DRY_RUN" == true ]]; then
    print_warning "干运行模式 - 仅显示任务，不实际执行"
    echo ""
fi

# 确认
if [[ "$AUTO_YES" != true && "$DRY_RUN" != true ]]; then
    read -p "是否开始批量处理？[Y/n] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]] && [[ ! -z $REPLY ]]; then
        echo "已取消"
        exit 0
    fi
fi

################################################################################
# 创建输出目录
################################################################################

if [[ "$DRY_RUN" != true ]]; then
    mkdir -p "$OUTPUT_BASE/.logs"
fi

################################################################################
# 定义单个任务执行函数
################################################################################

run_single_cache() {
    local model="$1"
    local dataset="$2"
    local cache_name="$3"
    local cache_path="$4"
    local output_dir="$5"
    local mode="$6"
    local threads="$7"
    local log_level="$8"
    local pos_max="${9:-}"  # 可选参数

    # 日志文件
    local log_file="${output_dir}/.logs/${model}_${dataset}_${cache_name}.log"
    local result_dir="${output_dir}/${model}/${dataset}/${cache_name}"

    # 创建结果目录
    mkdir -p "$result_dir"

    # 开始时间
    local start_time=$(date +%s)

    {
        echo "=========================================="
        echo "Task: ${model}/${dataset}/${cache_name}"
        echo "Mode: ${mode}"
        echo "Threads: ${threads}"
        if [[ "$mode" == "all-positions" && -n "$pos_max" ]]; then
            echo "PosMax: ${pos_max}"
        fi
        echo "Start: $(date)"
        echo "=========================================="
        echo ""

        # 构建基础命令
        local BASE_CMD="python3 -m nad.cli --log-level ${log_level} analyze"
        BASE_CMD="${BASE_CMD} --cache-root ${cache_path}"
        BASE_CMD="${BASE_CMD} --distance ja"
        BASE_CMD="${BASE_CMD} --group-topk-policy legacy-min"
        BASE_CMD="${BASE_CMD} --selectors all"
        BASE_CMD="${BASE_CMD} --distance-threads ${threads}"

        local success=true

        # 运行全序列分析
        if [[ "$mode" == "full" || "$mode" == "all" ]]; then
            echo "[Full Sequence Analysis]"
            if ${BASE_CMD} --out "${result_dir}/full_sequence_result.json"; then
                echo "✓ Full sequence analysis completed"

                # 计算准确率
                if python3 -m nad.cli accuracy \
                    --selection "${result_dir}/full_sequence_result.json" \
                    --cache-root "${cache_path}" \
                    --out "${result_dir}/accuracy_full_sequence.json"; then
                    echo "✓ Accuracy calculation completed"
                else
                    echo "✗ Accuracy calculation failed"
                    success=false
                fi
            else
                echo "✗ Full sequence analysis failed"
                success=false
            fi
            echo ""
        fi

        # 运行 position 分析 (固定窗口)
        if [[ "$mode" == "positions" || "$mode" == "all" ]]; then
            for window in "0-1" "0-2" "0-8"; do
                echo "[Window ${window} Analysis]"
                if ${BASE_CMD} --pos-window "$window" --pos-size 32 \
                    --out "${result_dir}/window_${window}_result.json"; then
                    echo "✓ Window ${window} analysis completed"

                    if python3 -m nad.cli accuracy \
                        --selection "${result_dir}/window_${window}_result.json" \
                        --cache-root "${cache_path}" \
                        --out "${result_dir}/accuracy_window_${window}.json"; then
                        echo "✓ Window ${window} accuracy completed"
                    else
                        echo "✗ Window ${window} accuracy failed"
                    fi
                else
                    echo "✗ Window ${window} analysis failed"
                    success=false
                fi
                echo ""
            done
        fi

        # 运行全 position 细粒度分析
        if [[ "$mode" == "all-positions" ]]; then
            echo "[All Positions Analysis]"
            local pos_max_arg=""
            if [[ -n "$pos_max" ]]; then
                pos_max_arg="--pos-max ${pos_max}"
            fi

            if ${BASE_CMD} --pos-window all ${pos_max_arg} \
                --out "${result_dir}/all_positions_result.json"; then
                echo "✓ All positions analysis completed"

                # 为每个窗口计算准确率
                python3 << PYEOF
import json
from pathlib import Path
import tempfile

result_file = Path("${result_dir}/all_positions_result.json")
cache_path = "${cache_path}"

if result_file.exists():
    with open(result_file) as f:
        data = json.load(f)

    from nad.ops.accuracy import compute_accuracy_report

    accuracy_summary = {
        "mode": "all_positions",
        "max_position": data.get("max_position"),
        "pos_size": data.get("pos_size"),
        "windows": {}
    }

    windows = data.get("windows", {})
    total_windows = len(windows)
    for i, (window_name, window_data) in enumerate(windows.items()):
        try:
            # 临时保存单个窗口结果
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
                json.dump(window_data, tmp)
                tmp_path = tmp.name

            acc = compute_accuracy_report(tmp_path, cache_path)
            accuracy_summary["windows"][window_name] = acc.selector_accuracy

            # 清理临时文件
            Path(tmp_path).unlink()

            if (i + 1) % 50 == 0 or (i + 1) == total_windows:
                print(f"  Accuracy: {i+1}/{total_windows} windows processed")
        except Exception as e:
            print(f"  Warning: accuracy for {window_name} failed: {e}")

    # 保存汇总准确率
    acc_file = Path("${result_dir}/accuracy_all_positions.json")
    with open(acc_file, "w") as f:
        json.dump(accuracy_summary, f, indent=2)
    print("✓ All positions accuracy completed")
PYEOF
            else
                echo "✗ All positions analysis failed"
                success=false
            fi
            echo ""
        fi

        local end_time=$(date +%s)
        local duration=$((end_time - start_time))

        echo "=========================================="
        echo "End: $(date)"
        echo "Duration: ${duration} seconds"
        echo "Status: $([ "$success" = true ] && echo "SUCCESS" || echo "FAILED")"
        echo "=========================================="

    } > "$log_file" 2>&1

    # 返回状态
    if [[ -f "${result_dir}/accuracy_full_sequence.json" ]] || \
       [[ -f "${result_dir}/accuracy_window_0-1.json" ]] || \
       [[ -f "${result_dir}/all_positions_result.json" ]]; then
        echo "OK:${model}/${dataset}/${cache_name}"
    else
        echo "FAIL:${model}/${dataset}/${cache_name}"
    fi
}
export -f run_single_cache

################################################################################
# 生成任务列表
################################################################################

START_TIME=$(date +%s)

print_header "开始批量分析 (并行: ${PARALLEL_JOBS} 任务)"
echo ""

TASK_LIST_FILE="${OUTPUT_BASE}/.task_list.txt"

if [[ "$DRY_RUN" != true ]]; then
    > "$TASK_LIST_FILE"
fi

# 生成任务列表
for CACHE_DIR in "${CACHE_LIST[@]}"; do
    MODEL_NAME=$(basename "$(dirname "$(dirname "$CACHE_DIR")")")
    DATASET_NAME=$(basename "$(dirname "$CACHE_DIR")")
    CACHE_NAME=$(basename "$CACHE_DIR")

    if [[ "$DRY_RUN" != true ]]; then
        echo "${MODEL_NAME}|${DATASET_NAME}|${CACHE_NAME}|${CACHE_DIR}|${OUTPUT_BASE}|${MODE}|${THREADS}|${LOG_LEVEL}|${POS_MAX}" >> "$TASK_LIST_FILE"
    fi
done

echo "总任务数: $TOTAL_CACHES, 并行数: $PARALLEL_JOBS"
echo ""

################################################################################
# 执行任务
################################################################################

if [[ "$DRY_RUN" == true ]]; then
    print_warning "干运行模式 - 显示任务列表:"
    echo ""
    for CACHE_DIR in "${CACHE_LIST[@]}"; do
        MODEL_NAME=$(basename "$(dirname "$(dirname "$CACHE_DIR")")")
        DATASET_NAME=$(basename "$(dirname "$CACHE_DIR")")
        CACHE_NAME=$(basename "$CACHE_DIR")
        echo "  ${MODEL_NAME}/${DATASET_NAME}/${CACHE_NAME}"
    done
    echo ""
    echo "使用以下命令实际执行:"
    echo "  $0 --mode $MODE --threads $THREADS -j $PARALLEL_JOBS -y"
else
    print_info "开始并行处理..."
    echo ""

    # 创建完成计数器
    COMPLETED=0
    SUCCESS_COUNT=0
    FAILED_COUNT=0
    FAILED_LIST=()

    # 使用 GNU parallel 或 xargs 并行执行
    if command -v parallel &> /dev/null; then
        print_info "使用 GNU parallel 执行"
        echo ""

        # 使用 parallel 执行，收集输出
        while IFS= read -r result; do
            COMPLETED=$((COMPLETED + 1))
            if [[ "$result" == OK:* ]]; then
                SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
                print_success "[${COMPLETED}/${TOTAL_CACHES}] ${result#OK:}"
            else
                FAILED_COUNT=$((FAILED_COUNT + 1))
                FAILED_LIST+=("${result#FAIL:}")
                print_error "[${COMPLETED}/${TOTAL_CACHES}] ${result#FAIL:}"
            fi
        done < <(cat "$TASK_LIST_FILE" | parallel -j "$PARALLEL_JOBS" --colsep '\|' \
            'run_single_cache {1} {2} {3} {4} {5} {6} {7} {8} {9}')

    else
        print_info "使用 xargs 执行 (未安装 GNU parallel)"
        echo ""

        # 使用 xargs 并行执行
        while IFS= read -r result; do
            COMPLETED=$((COMPLETED + 1))
            if [[ "$result" == OK:* ]]; then
                SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
                print_success "[${COMPLETED}/${TOTAL_CACHES}] ${result#OK:}"
            else
                FAILED_COUNT=$((FAILED_COUNT + 1))
                FAILED_LIST+=("${result#FAIL:}")
                print_error "[${COMPLETED}/${TOTAL_CACHES}] ${result#FAIL:}"
            fi
        done < <(cat "$TASK_LIST_FILE" | xargs -P "$PARALLEL_JOBS" -I {} bash -c '
            IFS="|" read -r model dataset cache_name cache_path output_dir mode threads log_level pos_max <<< "{}"
            run_single_cache "$model" "$dataset" "$cache_name" "$cache_path" "$output_dir" "$mode" "$threads" "$log_level" "$pos_max"
        ')
    fi

    echo ""
fi

################################################################################
# 生成汇总报告
################################################################################

if [[ "$DRY_RUN" != true ]]; then
    print_info "生成汇总报告..."

    python3 << EOF
import json
from pathlib import Path
from datetime import datetime

mode = "$MODE"
pos_max = "$POS_MAX" if "$POS_MAX" else None

summary = {
    "metadata": {
        "generated_at": datetime.now().isoformat(),
        "mode": mode,
        "parallel_jobs": $PARALLEL_JOBS,
        "analysis_modes": {
            "full_sequence": mode in ["full", "all"],
            "position_windows": mode in ["positions", "all"],
            "all_positions": mode == "all-positions",
            "windows": ["0-1", "0-2", "0-8"] if mode in ["positions", "all"] else [],
            "pos_max": int(pos_max) if pos_max else None
        },
        "total_caches": $TOTAL_CACHES,
        "successful": $SUCCESS_COUNT,
        "failed": $FAILED_COUNT
    },
    "models": {},
    "failures": []
}

# Counters for statistics
full_sequence_count = 0
position_window_counts = {"window_0-1": 0, "window_0-2": 0, "window_0-8": 0}
all_positions_count = 0

# 遍历所有结果文件
results_dir = Path("$OUTPUT_BASE")
if results_dir.exists():
    for model_dir in results_dir.iterdir():
        if not model_dir.is_dir() or model_dir.name.startswith('.'):
            continue

        model_name = model_dir.name
        summary["models"][model_name] = {}

        for dataset_dir in model_dir.iterdir():
            if not dataset_dir.is_dir():
                continue

            dataset_name = dataset_dir.name

            for cache_dir in dataset_dir.iterdir():
                if not cache_dir.is_dir():
                    continue

                cache_name = cache_dir.name

                # Initialize dataset entry
                dataset_entry = {
                    "cache": cache_name,
                    "status": "success"
                }

                # Load full sequence accuracy if exists
                acc_file = cache_dir / "accuracy_full_sequence.json"
                if acc_file.exists():
                    with open(acc_file) as f:
                        acc_data = json.load(f)
                    dataset_entry["full_sequence"] = acc_data.get("selector_accuracy", {})
                    full_sequence_count += 1

                # Load position window accuracies if they exist
                for window_name in ["window_0-1", "window_0-2", "window_0-8"]:
                    acc_file = cache_dir / f"accuracy_{window_name}.json"
                    if acc_file.exists():
                        with open(acc_file) as f:
                            acc_data = json.load(f)
                        dataset_entry[window_name] = acc_data.get("selector_accuracy", {})
                        position_window_counts[window_name] += 1

                # Load all-positions accuracy if exists
                acc_file = cache_dir / "accuracy_all_positions.json"
                if acc_file.exists():
                    with open(acc_file) as f:
                        acc_data = json.load(f)
                    dataset_entry["all_positions"] = {
                        "max_position": acc_data.get("max_position"),
                        "num_windows": len(acc_data.get("windows", {})),
                        "windows": acc_data.get("windows", {})
                    }
                    all_positions_count += 1

                # Only add if we have at least some data
                if any(key in dataset_entry for key in ["full_sequence", "window_0-1", "window_0-2", "window_0-8", "all_positions"]):
                    summary["models"][model_name][dataset_name] = dataset_entry

# Add statistics to metadata
summary["metadata"]["data_counts"] = {
    "full_sequence": full_sequence_count,
    "position_windows": position_window_counts,
    "all_positions": all_positions_count
}

# Calculate total duration
import os
start_time = $START_TIME
end_time = int(datetime.now().timestamp())
summary["metadata"]["total_duration_seconds"] = end_time - start_time

# 保存汇总
with open("$OUTPUT_BASE/summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"✓ 汇总报告已保存: $OUTPUT_BASE/summary.json")

# Print statistics
if full_sequence_count > 0:
    print(f"  - 全序列数据: {full_sequence_count} 个任务")
if any(count > 0 for count in position_window_counts.values()):
    print(f"  - Position窗口数据:")
    for window, count in position_window_counts.items():
        if count > 0:
            print(f"    * {window}: {count} 个任务")
if all_positions_count > 0:
    print(f"  - 全Position数据: {all_positions_count} 个任务")
EOF

    echo ""
fi

################################################################################
# 完成总结
################################################################################

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
MINUTES=$((DURATION / 60))
SECONDS_REMAINING=$((DURATION % 60))

print_header "批量分析完成"
echo ""

if [[ "$DRY_RUN" != true ]]; then
    echo -e "${BLUE}处理统计:${NC}"
    echo "  成功:       $SUCCESS_COUNT/$TOTAL_CACHES"
    if [ ${FAILED_COUNT:-0} -gt 0 ]; then
        echo "  失败:       $FAILED_COUNT/$TOTAL_CACHES"
    fi
    echo "  并行任务数: $PARALLEL_JOBS"
    echo "  总耗时:     ${MINUTES}分${SECONDS_REMAINING}秒"
    echo ""

    if [ ${FAILED_COUNT:-0} -gt 0 ]; then
        echo "失败列表:"
        for failed in "${FAILED_LIST[@]:0:10}"; do
            echo "  ✗ $failed"
        done
        if [ ${#FAILED_LIST[@]} -gt 10 ]; then
            echo "  ... 共 ${#FAILED_LIST[@]} 个失败任务"
        fi
        echo ""
        echo "查看失败日志:"
        echo "  ls $OUTPUT_BASE/.logs/ | head"
        echo ""
    fi

    echo "输出文件:"
    echo "  结果目录: $OUTPUT_BASE/"
    echo "  汇总报告: $OUTPUT_BASE/summary.json"
    echo "  任务日志: $OUTPUT_BASE/.logs/"
    echo ""

    echo "下一步:"
    echo "  # 运行选择器排名分析:"
    echo "  ./step2.3_selector_ranking.sh --results-dir $OUTPUT_BASE"
    echo ""

    echo "查看汇总:"
    echo "  cat $OUTPUT_BASE/summary.json | jq '.metadata'"
    echo ""
else
    echo "干运行完成，未执行实际任务"
    echo ""
fi

print_success "NAD 批量分析 (并行版本) 完成!"
