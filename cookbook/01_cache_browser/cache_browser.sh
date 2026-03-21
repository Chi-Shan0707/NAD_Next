#!/bin/bash

################################################################################
# Step 1.2: NAD Cache Browser Web Server
#
# A dedicated web interface for browsing and managing NAD caches.
#
# Usage:
#   ./step1.2_cache_browser.sh [options]
#
# Options:
#   --port PORT          Server port (default: 5003)
#   --host HOST          Bind host (default: 0.0.0.0)
#   --vis-port PORT      Visualization server port for links (default: 5002)
#   --debug              Enable Flask debug mode
#   --background         Run in background
#   --kill               Stop running cache browser server
#   --status             Check server status
#   -h, --help           Show this help
#
# Features:
#   - Web-based cache directory browser
#   - Shows MUI_public and local caches
#   - Displays cache metadata (samples, problems, version)
#   - Search and filter functionality
#   - Quick links to visualization server
#   - API endpoints for programmatic access
#
# Examples:
#   ./step1.2_cache_browser.sh                    # Start on default port 5003
#   ./step1.2_cache_browser.sh --port 8080        # Use custom port
#   ./step1.2_cache_browser.sh --background       # Run in background
#   ./step1.2_cache_browser.sh --kill             # Stop server
#
################################################################################

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Defaults
PORT=5003
HOST="0.0.0.0"
VIS_PORT=5002
DEBUG_MODE=""
BACKGROUND=false
KILL_SERVER=false
CHECK_STATUS=false

# PID file
PID_FILE="/tmp/nad_cache_browser.pid"
LOG_FILE="cache_browser.log"

# Functions
show_help() {
    sed -n '3,35p' "$0" | sed 's/^#//'
    exit 0
}

kill_server() {
    echo -e "${BLUE}[INFO]${NC} Looking for cache browser server..."

    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p $PID > /dev/null 2>&1; then
            echo -e "${BLUE}[INFO]${NC} Stopping process PID=$PID"
            kill $PID 2>/dev/null || true
            rm -f "$PID_FILE"
            echo -e "${GREEN}[SUCCESS]${NC} Server stopped"
            exit 0
        fi
        rm -f "$PID_FILE"
    fi

    # Find by process name
    PIDS=$(ps aux | grep "[p]ython3.*cache_browser.py" | awk '{print $2}')
    if [ -n "$PIDS" ]; then
        echo -e "${BLUE}[INFO]${NC} Found server processes: $PIDS"
        for PID in $PIDS; do
            kill $PID 2>/dev/null || true
        done
        echo -e "${GREEN}[SUCCESS]${NC} All servers stopped"
    else
        echo -e "${YELLOW}[INFO]${NC} No cache browser server running"
    fi
    exit 0
}

check_status() {
    echo -e "${BLUE}[INFO]${NC} Checking cache browser status..."

    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p $PID > /dev/null 2>&1; then
            echo -e "${GREEN}[RUNNING]${NC} Cache browser running (PID=$PID)"
            PORT_INFO=$(netstat -tlnp 2>/dev/null | grep "$PID" | awk '{print $4}' | tail -1)
            [ -n "$PORT_INFO" ] && echo -e "${GREEN}[INFO]${NC} Listening: $PORT_INFO"
            exit 0
        fi
    fi

    PIDS=$(ps aux | grep "[p]ython3.*cache_browser.py" | awk '{print $2}')
    if [ -n "$PIDS" ]; then
        echo -e "${GREEN}[RUNNING]${NC} Found processes: $PIDS"
        exit 0
    fi

    echo -e "${YELLOW}[STOPPED]${NC} Cache browser not running"
    exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --vis-port)
            VIS_PORT="$2"
            shift 2
            ;;
        --debug)
            DEBUG_MODE="--debug"
            shift
            ;;
        --background)
            BACKGROUND=true
            shift
            ;;
        --kill)
            kill_server
            ;;
        --status)
            check_status
            ;;
        *)
            echo -e "${RED}[ERROR]${NC} Unknown option: $1"
            show_help
            ;;
    esac
done

# Check dependencies
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}[ERROR]${NC} Python3 not installed"
    exit 1
fi

if ! python3 -c "import flask" 2>/dev/null; then
    echo -e "${YELLOW}[WARNING]${NC} Flask not installed, installing..."
    pip3 install flask
fi

# Check script exists
SCRIPT_PATH="$(dirname "$0")/../../tools/cache_browser.py"
if [ ! -f "$SCRIPT_PATH" ]; then
    echo -e "${RED}[ERROR]${NC} cache_browser.py not found at $SCRIPT_PATH"
    exit 1
fi

# Display configuration
echo "=================================="
echo "NAD Cache Browser"
echo "=================================="
echo -e "${GREEN}Configuration:${NC}"
echo "  Server: http://$HOST:$PORT"
echo "  Visualization port: $VIS_PORT"
[ -n "$DEBUG_MODE" ] && echo "  Debug: enabled"
[ "$BACKGROUND" = true ] && echo "  Mode: background"
echo "=================================="
echo

# Build command
CMD="python3 $SCRIPT_PATH --port $PORT --host $HOST --vis-port $VIS_PORT"
[ -n "$DEBUG_MODE" ] && CMD="$CMD $DEBUG_MODE"

# Start server
if [ "$BACKGROUND" = true ]; then
    echo -e "${BLUE}[INFO]${NC} Starting in background..."
    nohup $CMD > "$LOG_FILE" 2>&1 &
    PID=$!
    echo $PID > "$PID_FILE"
    sleep 2

    if ps -p $PID > /dev/null 2>&1; then
        echo -e "${GREEN}[SUCCESS]${NC} Server started (PID=$PID)"
        echo -e "${GREEN}[INFO]${NC} Access: http://localhost:$PORT"
        echo -e "${GREEN}[INFO]${NC} Log: $LOG_FILE"
        echo
        echo "Management:"
        echo "  View logs: tail -f $LOG_FILE"
        echo "  Status: $0 --status"
        echo "  Stop: $0 --kill"
    else
        echo -e "${RED}[ERROR]${NC} Failed to start server"
        tail -20 "$LOG_FILE"
        exit 1
    fi
else
    echo -e "${BLUE}[INFO]${NC} Starting cache browser..."
    echo -e "${GREEN}[INFO]${NC} Access: http://localhost:$PORT"
    echo -e "${YELLOW}[INFO]${NC} Press Ctrl+C to stop"
    echo
    $CMD
fi
