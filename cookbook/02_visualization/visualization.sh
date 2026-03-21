#!/bin/bash

################################################################################
# Step 1.1: NAD Cache Visualization Server
#
# 用途: 启动NAD缓存可视化Web服务器，用于交互式探索和分析缓存数据
#
# 使用方法:
#   ./step1.1_visualization.sh [cache_dir] [port] [options...]
#
# 参数:
#   cache_dir   - 缓存目录路径（默认：MUI_public/cache/.../cache_neuron_output_1_*）
#   port        - 服务器端口（默认：5002）
#
# 选项:
#   --list-caches            - 列出所有可用的缓存目录
#   --max-cache-mb SIZE      - 最大缓存内存大小MB（默认：256）
#   --host HOST              - 绑定主机地址（默认：0.0.0.0）
#   --debug                  - 启用Flask调试模式
#   --no-browser             - 不自动打开浏览器
#   --background             - 后台运行（使用nohup）
#   --kill                   - 停止所有运行的可视化服务器
#   --status                 - 检查服务器运行状态
#
# 示例:
#   ./step1.1_visualization.sh MUI_public/cache/Qwen3-4B-Instruct-2507/gpqa/cache_neuron_output_1_act_no_rms_20250913_023836 5006
#   ./step1.1_visualization.sh                           # 使用默认设置启动
#   ./step1.1_visualization.sh --list-caches             # 列出所有可用缓存
#   ./step1.1_visualization.sh cache_aime24              # 指定缓存目录
#   ./step1.1_visualization.sh cache_aime24 8080         # 指定缓存和端口
#   ./step1.1_visualization.sh --max-cache-mb 512        # 增加内存缓存
#   ./step1.1_visualization.sh --background              # 后台运行
#   ./step1.1_visualization.sh --kill                    # 停止服务器
#   ./step1.1_visualization.sh --status                  # 检查状态
#
# 功能说明:
#   - 提供Web界面浏览缓存结构和内容
#   - 支持选择器性能可视化
#   - 实时查看问题级别的准确率
#   - 交互式探索激活分布
#   - 支持多种缓存格式（v4.0/4.1）
#
# 注意事项:
#   - 需要安装Flask和相关依赖
#   - 大型缓存可能需要增加 --max-cache-mb
#   - 后台模式日志保存在 visualization.log
#
################################################################################

set -e  # 遇到错误时停止

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 默认参数
# 默认使用MUI_public中的第一个模型的aime24缓存
DEFAULT_CACHE="./MUI_public/cache/DeepSeek-R1-0528-Qwen3-8B/aime24/cache_neuron_output_1_act_no_rms_20250902_025610"
# 如果指定的默认缓存不存在，尝试查找其他可用缓存
if [ ! -d "$DEFAULT_CACHE" ]; then
    # 尝试查找任意可用的cache_neuron_output_1缓存
    DEFAULT_CACHE=$(find ./MUI_public/cache -type d -name "cache_neuron_output_1_act_no_rms_*" 2>/dev/null | head -1)
    if [ -z "$DEFAULT_CACHE" ]; then
        # 如果还是没找到，使用本地缓存
        DEFAULT_CACHE=$(ls -d cache_* 2>/dev/null | head -1 || echo "cache_aime24")
    fi
fi
CACHE_DIR="$DEFAULT_CACHE"
PORT=5002
MAX_CACHE_MB=256
HOST="0.0.0.0"
DEBUG_MODE=""
AUTO_BROWSER=true
BACKGROUND=false
KILL_SERVER=false
CHECK_STATUS=false

# PID文件位置
PID_FILE="/tmp/nad_visualization_server.pid"
LOG_FILE="visualization.log"

# 函数：显示帮助信息
show_help() {
    sed -n '3,45p' "$0" | sed 's/^#//'
    exit 0
}

# 函数：列出所有可用的缓存
list_caches() {
    echo -e "${BLUE}[INFO]${NC} 查找可用的缓存目录..."
    echo

    # MUI_public缓存
    echo -e "${GREEN}MUI_public 缓存:${NC}"
    if [ -d "./MUI_public/cache" ]; then
        for model_dir in ./MUI_public/cache/*/; do
            if [ -d "$model_dir" ]; then
                model_name=$(basename "$model_dir")
                echo -e "  ${YELLOW}$model_name:${NC}"
                for dataset_dir in "$model_dir"*/; do
                    if [ -d "$dataset_dir" ]; then
                        dataset_name=$(basename "$dataset_dir")
                        for cache_dir in "$dataset_dir"cache_neuron_output_1_*; do
                            if [ -d "$cache_dir" ]; then
                                cache_name=$(basename "$cache_dir")
                                # 获取相对路径
                                rel_path="${cache_dir#./}"
                                echo "    $dataset_name/$cache_name"
                                echo "      路径: $rel_path"
                            fi
                        done
                    fi
                done
            fi
        done
    else
        echo "  未找到 MUI_public/cache 目录"
    fi

    echo
    echo -e "${GREEN}本地缓存:${NC}"
    local_caches=$(ls -d cache_* 2>/dev/null)
    if [ -n "$local_caches" ]; then
        for cache in $local_caches; do
            echo "  $cache"
        done
    else
        echo "  未找到本地缓存目录"
    fi

    echo
    echo -e "${BLUE}使用示例:${NC}"
    echo "  ./step1.1_visualization.sh cache_aime24"
    echo "  ./step1.1_visualization.sh ./MUI_public/cache/DeepSeek-R1-0528-Qwen3-8B/aime24/cache_neuron_output_1_act_no_rms_20250902_025610"
    exit 0
}

# 函数：检查Python和依赖
check_dependencies() {
    # 检查Python
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}[ERROR]${NC} Python3 未安装"
        exit 1
    fi

    # 检查Flask
    if ! python3 -c "import flask" 2>/dev/null; then
        echo -e "${YELLOW}[WARNING]${NC} Flask 未安装，尝试安装..."
        pip3 install flask
    fi

    # 检查可视化目录
    if [ ! -d "minimal_visualization_next" ]; then
        echo -e "${RED}[ERROR]${NC} 未找到 minimal_visualization_next 目录"
        echo "请确保在 NAD_Next 根目录运行此脚本"
        exit 1
    fi

    # 检查app.py
    if [ ! -f "minimal_visualization_next/app.py" ]; then
        echo -e "${RED}[ERROR]${NC} 未找到 minimal_visualization_next/app.py"
        exit 1
    fi
}

# 函数：停止服务器
kill_server() {
    echo -e "${BLUE}[INFO]${NC} 查找运行中的可视化服务器..."

    # 方法1：使用PID文件
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p $PID > /dev/null 2>&1; then
            echo -e "${BLUE}[INFO]${NC} 停止进程 PID=$PID"
            kill $PID 2>/dev/null || true
            rm -f "$PID_FILE"
            echo -e "${GREEN}[SUCCESS]${NC} 服务器已停止"
        else
            echo -e "${YELLOW}[INFO]${NC} PID文件存在但进程未运行"
            rm -f "$PID_FILE"
        fi
    fi

    # 方法2：查找所有相关进程
    PIDS=$(ps aux | grep "[p]ython3.*minimal_visualization_next/app.py" | awk '{print $2}')
    if [ -n "$PIDS" ]; then
        echo -e "${BLUE}[INFO]${NC} 找到运行中的服务器进程: $PIDS"
        for PID in $PIDS; do
            echo -e "${BLUE}[INFO]${NC} 停止进程 PID=$PID"
            kill $PID 2>/dev/null || true
        done
        echo -e "${GREEN}[SUCCESS]${NC} 所有服务器进程已停止"
    else
        echo -e "${YELLOW}[INFO]${NC} 未找到运行中的可视化服务器"
    fi
}

# 函数：检查服务器状态
check_status() {
    echo -e "${BLUE}[INFO]${NC} 检查可视化服务器状态..."

    # 检查PID文件
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p $PID > /dev/null 2>&1; then
            echo -e "${GREEN}[RUNNING]${NC} 服务器运行中 (PID=$PID)"

            # 尝试获取端口信息
            PORT_INFO=$(netstat -tlnp 2>/dev/null | grep "$PID" | awk '{print $4}' | tail -1)
            if [ -n "$PORT_INFO" ]; then
                echo -e "${GREEN}[INFO]${NC} 监听地址: $PORT_INFO"
            fi

            # 显示访问URL
            if [[ "$PORT_INFO" =~ :([0-9]+)$ ]]; then
                PORT_NUM="${BASH_REMATCH[1]}"
                echo -e "${GREEN}[INFO]${NC} 访问地址: http://localhost:$PORT_NUM"
            fi

            return 0
        fi
    fi

    # 检查进程
    PIDS=$(ps aux | grep "[p]ython3.*minimal_visualization_next/app.py" | awk '{print $2}')
    if [ -n "$PIDS" ]; then
        echo -e "${GREEN}[RUNNING]${NC} 找到服务器进程: $PIDS"

        for PID in $PIDS; do
            # 尝试获取端口信息
            PORT_INFO=$(netstat -tlnp 2>/dev/null | grep "$PID" | awk '{print $4}' | head -1)
            if [ -n "$PORT_INFO" ]; then
                echo -e "${GREEN}[INFO]${NC} PID $PID 监听: $PORT_INFO"
            fi
        done

        return 0
    fi

    echo -e "${YELLOW}[STOPPED]${NC} 服务器未运行"
    return 1
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            ;;
        --list-caches)
            list_caches
            ;;
        --max-cache-mb)
            MAX_CACHE_MB="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --debug)
            DEBUG_MODE="--debug"
            shift
            ;;
        --no-browser)
            AUTO_BROWSER=false
            shift
            ;;
        --background)
            BACKGROUND=true
            shift
            ;;
        --kill)
            KILL_SERVER=true
            shift
            ;;
        --status)
            CHECK_STATUS=true
            shift
            ;;
        --*)
            echo -e "${RED}[ERROR]${NC} 未知选项: $1"
            show_help
            ;;
        *)
            # 位置参数
            # 检查是否已经设置了用户提供的缓存目录
            if [ "$CACHE_DIR" = "$DEFAULT_CACHE" ]; then
                # 第一个位置参数是缓存目录
                CACHE_DIR="$1"
            elif [ "$PORT" = "5002" ]; then
                # 第二个位置参数是端口
                PORT="$1"
            else
                echo -e "${RED}[ERROR]${NC} 多余的参数: $1"
                show_help
            fi
            shift
            ;;
    esac
done

# 执行特殊命令
if [ "$KILL_SERVER" = true ]; then
    kill_server
    exit 0
fi

if [ "$CHECK_STATUS" = true ]; then
    check_status
    exit $?
fi

# 检查依赖
check_dependencies

# 检查缓存目录
if [ ! -d "$CACHE_DIR" ]; then
    echo -e "${YELLOW}[WARNING]${NC} 缓存目录不存在: $CACHE_DIR"
    echo -e "${BLUE}[INFO]${NC} 可用的缓存目录:"
    ls -d cache_* 2>/dev/null | head -10 || echo "  无"
    echo
    read -p "是否继续？[y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 获取绝对路径
CACHE_DIR_ABS=$(realpath "$CACHE_DIR" 2>/dev/null || echo "$CACHE_DIR")

# 显示配置
echo "=================================="
echo "NAD 可视化服务器"
echo "=================================="
echo -e "${GREEN}配置:${NC}"
echo "  缓存目录: $CACHE_DIR_ABS"
echo "  服务端口: $PORT"
echo "  绑定地址: $HOST:$PORT"
echo "  内存缓存: ${MAX_CACHE_MB}MB"
[ -n "$DEBUG_MODE" ] && echo "  调试模式: 启用"
[ "$BACKGROUND" = true ] && echo "  运行模式: 后台"
echo "=================================="
echo

# 构建命令
CMD="python3 app.py"
CMD="$CMD --data-dir ../$CACHE_DIR"
CMD="$CMD --port $PORT"
CMD="$CMD --host $HOST"
CMD="$CMD --max-cache-mb $MAX_CACHE_MB"
[ -n "$DEBUG_MODE" ] && CMD="$CMD $DEBUG_MODE"

# 设置环境变量
export PYTHONPATH="../:$PYTHONPATH"

# 切换到可视化目录
cd minimal_visualization_next

# 启动服务器
if [ "$BACKGROUND" = true ]; then
    echo -e "${BLUE}[INFO]${NC} 在后台启动服务器..."
    echo -e "${BLUE}[CMD]${NC} $CMD"

    # 使用nohup后台运行
    nohup $CMD > "../$LOG_FILE" 2>&1 &
    PID=$!

    # 保存PID
    echo $PID > "$PID_FILE"

    # 等待服务器启动
    sleep 2

    # 检查是否成功启动
    if ps -p $PID > /dev/null 2>&1; then
        echo -e "${GREEN}[SUCCESS]${NC} 服务器已在后台启动 (PID=$PID)"
        echo -e "${GREEN}[INFO]${NC} 日志文件: $LOG_FILE"
        echo -e "${GREEN}[INFO]${NC} 访问地址: http://localhost:$PORT"
        echo
        echo "管理命令:"
        echo "  查看日志: tail -f $LOG_FILE"
        echo "  检查状态: $0 --status"
        echo "  停止服务: $0 --kill"
    else
        echo -e "${RED}[ERROR]${NC} 服务器启动失败"
        echo "查看日志: tail -20 $LOG_FILE"
        exit 1
    fi
else
    echo -e "${BLUE}[INFO]${NC} 启动可视化服务器..."
    echo -e "${BLUE}[CMD]${NC} $CMD"
    echo
    echo -e "${GREEN}[INFO]${NC} 服务器运行中，访问: http://localhost:$PORT"
    echo -e "${YELLOW}[INFO]${NC} 按 Ctrl+C 停止服务器"
    echo

    # 如果需要自动打开浏览器
    if [ "$AUTO_BROWSER" = true ]; then
        # 等待服务器启动
        (sleep 2 && python3 -m webbrowser "http://localhost:$PORT") 2>/dev/null &
    fi

    # 前台运行
    $CMD
fi