#!/bin/bash

################################################################################
# Step 2.7: Position Window 消融实验脚本
#
# 用途: 对所有 cache 运行不同 position window 的消融实验
#
# 使用方法:
#   ./step2.7_position_ablation.sh [选项...]
#
# 选项:
#   --windows LIST      - position window 列表，逗号分隔 (默认: 0-1,0-2,0-8,0-32,0-128,0-512)
#   --threads N         - 距离计算线程数 (默认: 64)
#   --backend BACKEND   - Jaccard后端 roaring/numpy/auto (默认: roaring)
#   --output-dir DIR    - 输出目录 (默认: ./result/position_ablation_TIMESTAMP)
#   --dry-run           - 仅显示将要执行的命令，不实际运行
#   --quick             - 快速模式，仅测试 0-1,0-8,0-128
#   --parallel N, -j N  - 并行任务数 (默认: 4)
#   --model MODEL       - 指定模型 (默认: DeepSeek-R1-0528-Qwen3-8B)
#   --datasets LIST     - 数据集列表，逗号分隔 (默认: 全部)
#   --latest-only       - 仅使用每个数据集的最新 cache
#   --help              - 显示帮助信息
#
# 示例:
#   # 完整消融 (所有 cache, 所有 windows)
#   ./step2.7_position_ablation.sh
#
#   # 快速模式 (所有 cache, 3 个 windows)
#   ./step2.7_position_ablation.sh --quick
#
#   # 仅最新 cache
#   ./step2.7_position_ablation.sh --latest-only
#
#   # 指定数据集
#   ./step2.7_position_ablation.sh --datasets aime24,aime25
#
#   # 高并发
#   ./step2.7_position_ablation.sh -j 8
#
################################################################################

set -euo pipefail

# 获取脚本所在目录，切换到 repo root（上两级）
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}/../.."

################################################################################
# 默认参数
################################################################################

CACHE_BASE="MUI_HUB/cache"
MODEL="DeepSeek-R1-0528-Qwen3-8B"
DATASETS_STR=""  # 空表示全部
WINDOWS_STR="0-1,0-2,0-8,0-32,0-128,0-512"
THREADS=4
BACKEND="roaring"
OUTPUT_DIR=""
DRY_RUN=false
QUICK_MODE=false
LATEST_ONLY=false
PARALLEL_JOBS=4
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

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
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

print_success() { echo -e "${GREEN}✓${NC} $1"; }
print_error() { echo -e "${RED}✗${NC} $1"; }
print_warning() { echo -e "${YELLOW}⚠${NC} $1"; }
print_info() { echo -e "${CYAN}ℹ${NC} $1"; }

################################################################################
# 参数解析
################################################################################

show_help() {
    head -40 "$0" | tail -36
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --windows)
            WINDOWS_STR="$2"
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
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --quick)
            QUICK_MODE=true
            shift
            ;;
        --latest-only)
            LATEST_ONLY=true
            shift
            ;;
        --parallel|-j)
            PARALLEL_JOBS="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --datasets)
            DATASETS_STR="$2"
            shift 2
            ;;
        --help|-h)
            show_help
            ;;
        *)
            print_error "未知参数: $1"
            echo "使用 --help 查看帮助"
            exit 1
            ;;
    esac
done

# 快速模式覆盖参数
if [[ "$QUICK_MODE" == true ]]; then
    WINDOWS_STR="0-1,0-8,0-128"
fi

# 设置输出目录
if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="./result/position_ablation_${TIMESTAMP}"
fi

# 设置环境变量
export NAD_JA_BACKEND="$BACKEND"

################################################################################
# 解析窗口列表
################################################################################

IFS=',' read -ra WINDOWS <<< "$WINDOWS_STR"

################################################################################
# 扫描数据集和 cache
################################################################################

print_header "Step 2.7: Position Window 消融实验"
echo ""

MODEL_DIR="${CACHE_BASE}/${MODEL}"

if [[ ! -d "$MODEL_DIR" ]]; then
    print_error "模型目录不存在: $MODEL_DIR"
    exit 1
fi

# 获取数据集列表
if [[ -z "$DATASETS_STR" ]]; then
    # 自动扫描所有数据集
    DATASETS=($(ls -1 "$MODEL_DIR" 2>/dev/null | grep -v "^\." || true))
else
    IFS=',' read -ra DATASETS <<< "$DATASETS_STR"
fi

print_info "扫描数据集和cache..."
echo ""

# 收集所有有效的 cache 配置
declare -a VALID_CONFIGS=()

for dataset in "${DATASETS[@]}"; do
    dataset_dir="${MODEL_DIR}/${dataset}"

    if [[ ! -d "$dataset_dir" ]]; then
        print_warning "${dataset}: 目录不存在 - 跳过"
        continue
    fi

    # 获取该数据集下的所有 cache
    if [[ "$LATEST_ONLY" == true ]]; then
        # 仅最新的 cache
        caches=($(ls -1 "$dataset_dir" 2>/dev/null | grep "^cache_neuron" | sort -r | head -1 || true))
    else
        # 所有 cache
        caches=($(ls -1 "$dataset_dir" 2>/dev/null | grep "^cache_neuron" | sort || true))
    fi

    if [[ ${#caches[@]} -eq 0 ]]; then
        print_warning "${dataset}: 没有找到 cache - 跳过"
        continue
    fi

    for cache in "${caches[@]}"; do
        cache_path="${dataset_dir}/${cache}"
        if [[ -f "${cache_path}/meta.json" ]]; then
            print_success "${dataset}/${cache}"
            VALID_CONFIGS+=("${dataset}|${cache}|${cache_path}")
        else
            print_warning "${dataset}/${cache}: 缺少 meta.json - 跳过"
        fi
    done
done

echo ""

if [[ ${#VALID_CONFIGS[@]} -eq 0 ]]; then
    print_error "没有找到有效的cache，请检查路径配置"
    exit 1
fi

################################################################################
# 显示配置信息
################################################################################

TOTAL_TASKS=$((${#VALID_CONFIGS[@]} * ${#WINDOWS[@]}))

echo -e "${BLUE}实验配置:${NC}"
echo "  模型:           $MODEL"
echo "  数据集数:       ${#DATASETS[@]}"
echo "  Cache 数:       ${#VALID_CONFIGS[@]}"
echo "  Position窗口:   ${WINDOWS[*]} (共 ${#WINDOWS[@]} 个)"
echo "  并行任务数:     $PARALLEL_JOBS"
echo "  距离计算线程:   $THREADS"
echo "  Jaccard后端:    $BACKEND"
echo "  输出目录:       $OUTPUT_DIR"
echo "  仅最新cache:    $([ "$LATEST_ONLY" = true ] && echo "是" || echo "否")"
echo "  模式:           $([ "$DRY_RUN" = true ] && echo "干运行" || echo "实际执行")"
echo ""

echo -e "${BLUE}总任务数: ${TOTAL_TASKS} (${#VALID_CONFIGS[@]} cache x ${#WINDOWS[@]} 窗口)${NC}"
echo ""

# 预估时间 (每个任务约15秒)
EST_SECONDS=$(( (TOTAL_TASKS * 15) / PARALLEL_JOBS ))
EST_MINUTES=$((EST_SECONDS / 60))
echo -e "${CYAN}预估耗时: 约 ${EST_MINUTES} 分钟${NC}"
echo ""

if [[ "$DRY_RUN" == true ]]; then
    print_warning "干运行模式 - 仅显示命令，不实际执行"
    echo ""
fi

################################################################################
# 创建输出目录结构
################################################################################

if [[ "$DRY_RUN" != true ]]; then
    mkdir -p "$OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR/.logs"

    # 为每个数据集/cache创建子目录
    for config in "${VALID_CONFIGS[@]}"; do
        IFS='|' read -r dataset cache cache_path <<< "$config"
        mkdir -p "$OUTPUT_DIR/$dataset/$cache"
    done
fi

################################################################################
# 定义单个任务执行函数
################################################################################

run_single_task() {
    local dataset="$1"
    local cache="$2"
    local window="$3"
    local cache_path="$4"
    local output_dir="$5"
    local threads="$6"

    local result_file="${output_dir}/${dataset}/${cache}/window_${window}_result.json"
    local accuracy_file="${output_dir}/${dataset}/${cache}/accuracy_${window}.json"
    local log_file="${output_dir}/.logs/${dataset}_${cache}_${window}.log"

    # 运行分析
    python3 -m nad.cli --log-level WARNING analyze \
        --cache-root "${cache_path}" \
        --distance ja \
        --group-topk-policy legacy-min \
        --selectors all \
        --distance-threads "${threads}" \
        --pos-window "${window}" \
        --pos-size 32 \
        --out "${result_file}" > "$log_file" 2>&1

    if [[ $? -eq 0 ]]; then
        # 计算准确率
        python3 -m nad.cli accuracy \
            --selection "${result_file}" \
            --cache-root "${cache_path}" \
            --out "${accuracy_file}" 2>> "$log_file"

        if [[ $? -eq 0 ]]; then
            echo "OK:${dataset}/${cache}/${window}"
        else
            echo "ACC_FAIL:${dataset}/${cache}/${window}"
        fi
    else
        echo "FAIL:${dataset}/${cache}/${window}"
    fi
}
export -f run_single_task

################################################################################
# 运行消融实验 (并行模式)
################################################################################

START_TIME=$(date +%s)

print_header "开始 Position Window 消融实验 (并行: ${PARALLEL_JOBS} 任务)"
echo ""

# 生成所有任务列表
TASK_LIST_FILE="${OUTPUT_DIR}/.task_list.txt"

if [[ "$DRY_RUN" != true ]]; then
    > "$TASK_LIST_FILE"

    for config in "${VALID_CONFIGS[@]}"; do
        IFS='|' read -r dataset cache cache_path <<< "$config"
        for window in "${WINDOWS[@]}"; do
            echo "${dataset}|${cache}|${window}|${cache_path}|${OUTPUT_DIR}|${THREADS}" >> "$TASK_LIST_FILE"
        done
    done
fi

echo "总任务数: $TOTAL_TASKS, 并行数: $PARALLEL_JOBS"
echo ""

if [[ "$DRY_RUN" == true ]]; then
    print_warning "干运行模式 - 显示任务示例:"
    for config in "${VALID_CONFIGS[@]:0:5}"; do
        IFS='|' read -r dataset cache cache_path <<< "$config"
        echo "  ${dataset}/${cache}: windows=[${WINDOWS[*]}]"
    done
    if [[ ${#VALID_CONFIGS[@]} -gt 5 ]]; then
        echo "  ... (共 ${#VALID_CONFIGS[@]} 个 cache)"
    fi
else
    # 进度追踪
    COMPLETED=0

    # 使用 GNU parallel 或 xargs 并行执行
    if command -v parallel &> /dev/null; then
        cat "$TASK_LIST_FILE" | parallel -j "$PARALLEL_JOBS" --colsep '\|' \
            'run_single_task {1} {2} {3} {4} {5} {6}' | while read line; do
            COMPLETED=$((COMPLETED + 1))
            echo "[${COMPLETED}/${TOTAL_TASKS}] $line"
        done
    else
        cat "$TASK_LIST_FILE" | xargs -P "$PARALLEL_JOBS" -I {} bash -c '
            IFS="|" read -r dataset cache window cache_path output_dir threads <<< "{}"
            run_single_task "$dataset" "$cache" "$window" "$cache_path" "$output_dir" "$threads"
        ' | while read line; do
            COMPLETED=$((COMPLETED + 1))
            echo "[${COMPLETED}/${TOTAL_TASKS}] $line"
        done
    fi
fi

################################################################################
# 统计失败任务
################################################################################

FAILED_TASKS=()
if [[ "$DRY_RUN" != true ]]; then
    for config in "${VALID_CONFIGS[@]}"; do
        IFS='|' read -r dataset cache cache_path <<< "$config"
        for window in "${WINDOWS[@]}"; do
            if [[ ! -f "${OUTPUT_DIR}/${dataset}/${cache}/accuracy_${window}.json" ]]; then
                FAILED_TASKS+=("${dataset}/${cache}/${window}")
            fi
        done
    done
fi

################################################################################
# 生成汇总报告
################################################################################

if [[ "$DRY_RUN" != true ]]; then
    echo ""
    print_header "生成汇总报告"
    echo ""

    # 生成 Python 汇总脚本
    python3 << EOF
import json
from pathlib import Path
from collections import defaultdict

output_dir = Path("${OUTPUT_DIR}")
windows = "${WINDOWS_STR}".split(",")
datasets = "${DATASETS[*]}".split()

# 收集所有结果
# 结构: {dataset: {cache: {window: {selector: accuracy}}}}
all_results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
all_selectors = set()

for dataset in datasets:
    dataset_dir = output_dir / dataset
    if not dataset_dir.exists():
        continue

    for cache_dir in dataset_dir.iterdir():
        if not cache_dir.is_dir():
            continue
        cache = cache_dir.name

        for window in windows:
            acc_file = cache_dir / f"accuracy_{window}.json"
            if acc_file.exists():
                try:
                    with open(acc_file) as f:
                        data = json.load(f)
                    selector_acc = data.get("selector_accuracy", {})
                    all_results[dataset][cache][window] = selector_acc
                    all_selectors.update(selector_acc.keys())
                except Exception as e:
                    print(f"警告: 无法读取 {acc_file}: {e}")

# 按窗口汇总 (所有 cache 平均)
window_summary = defaultdict(lambda: defaultdict(list))
for dataset in all_results:
    for cache in all_results[dataset]:
        for window in all_results[dataset][cache]:
            for selector, acc in all_results[dataset][cache][window].items():
                window_summary[window][selector].append(acc)

# 计算平均
window_avg = {}
for window in windows:
    window_avg[window] = {}
    for selector in all_selectors:
        if window in window_summary and selector in window_summary[window]:
            acc_list = window_summary[window][selector]
            window_avg[window][selector] = sum(acc_list) / len(acc_list)

# 保存汇总
summary = {
    "model": "${MODEL}",
    "windows": windows,
    "selectors": sorted(list(all_selectors)),
    "num_caches": sum(len(caches) for caches in all_results.values()),
    "window_avg": window_avg,
    "detailed_results": {d: {c: dict(w) for c, w in caches.items()} for d, caches in all_results.items()}
}

summary_file = output_dir / "position_ablation_summary.json"
with open(summary_file, "w") as f:
    json.dump(summary, f, indent=2)

print(f"汇总报告已保存: {summary_file}")

# 打印汇总表格
print("\n" + "=" * 100)
print("Position Window 消融实验结果 (所有 cache 平均准确率 %)")
print("=" * 100)

# 表头
header = f"{'选择器':<25s}"
for w in windows:
    header += f"{w:>12s}"
header += f"{'最佳窗口':>14s}"
print(header)
print("-" * 100)

# 按选择器打印
for selector in sorted(all_selectors):
    row = f"{selector:<25s}"
    best_window = None
    best_acc = -1

    for window in windows:
        if window in window_avg and selector in window_avg[window]:
            acc = window_avg[window][selector]
            row += f"{acc:>11.2f}%"
            if acc > best_acc:
                best_acc = acc
                best_window = window
        else:
            row += f"{'N/A':>12s}"

    if best_window:
        row += f"{best_window:>14s}"
    print(row)

print("-" * 100)

# 每个窗口的最佳选择器
print("\n各窗口最佳选择器:")
for window in windows:
    if window in window_avg and window_avg[window]:
        best_selector = max(window_avg[window].keys(), key=lambda s: window_avg[window][s])
        best_acc = window_avg[window][best_selector]
        print(f"  {window:<10s}: {best_selector:<25s} ({best_acc:.2f}%)")

# 按数据集汇总
print("\n" + "=" * 100)
print("按数据集汇总 (medoid 选择器)")
print("=" * 100)

header = f"{'数据集':<20s}{'Cache数':>8s}"
for w in windows:
    header += f"{w:>12s}"
print(header)
print("-" * 100)

for dataset in sorted(all_results.keys()):
    num_caches = len(all_results[dataset])
    row = f"{dataset:<20s}{num_caches:>8d}"

    for window in windows:
        acc_list = []
        for cache in all_results[dataset]:
            if window in all_results[dataset][cache]:
                if "medoid" in all_results[dataset][cache][window]:
                    acc_list.append(all_results[dataset][cache][window]["medoid"])

        if acc_list:
            avg = sum(acc_list) / len(acc_list)
            row += f"{avg:>11.2f}%"
        else:
            row += f"{'N/A':>12s}"

    print(row)

print("-" * 100)
EOF

fi

################################################################################
# 完成总结
################################################################################

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
MINUTES=$((DURATION / 60))
SECONDS_REMAIN=$((DURATION % 60))

echo ""
print_header "实验完成"
echo ""

echo -e "${BLUE}实验配置:${NC}"
echo "  模型:           $MODEL"
echo "  数据集数:       ${#DATASETS[@]}"
echo "  Cache 数:       ${#VALID_CONFIGS[@]}"
echo "  Position窗口:   ${WINDOWS[*]}"
echo "  总任务数:       $TOTAL_TASKS"
echo "  失败任务数:     ${#FAILED_TASKS[@]}"
echo "  总耗时:         ${MINUTES}分${SECONDS_REMAIN}秒"
echo ""

if [[ "$DRY_RUN" != true ]]; then
    echo -e "${BLUE}输出文件:${NC}"
    echo "  结果目录:       $OUTPUT_DIR"
    echo "  汇总报告:       $OUTPUT_DIR/position_ablation_summary.json"
    echo ""

    echo "查看结果:"
    echo "  cat $OUTPUT_DIR/position_ablation_summary.json | jq '.window_avg'"
    echo ""
fi

if [[ ${#FAILED_TASKS[@]} -gt 0 ]]; then
    print_warning "以下任务失败:"
    for task in "${FAILED_TASKS[@]:0:10}"; do
        echo "  - $task"
    done
    if [[ ${#FAILED_TASKS[@]} -gt 10 ]]; then
        echo "  ... 共 ${#FAILED_TASKS[@]} 个失败任务"
    fi
fi

echo ""
print_success "Position Window 消融实验完成!"
