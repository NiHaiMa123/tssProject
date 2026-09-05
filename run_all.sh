#!/bin/bash
# ============================================================
# ChatTTS 零样本音色克隆与情感合成流水线
# 一键启动脚本
# 目标平台: macOS Apple Silicon (M1/M2/M3)
# ============================================================
set -euo pipefail

# ── 颜色输出 ───────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }
step()  { echo -e "${CYAN}[STEP]${NC} $*"; }

# ── 项目根目录 ─────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── 环境自检 ───────────────────────────────────────────────
step "环境自检..."

# Python 版本检查
PYTHON_CMD=""
for cmd in python3.10 python3 python; do
    if command -v "$cmd" &>/dev/null; then
        ver=$("$cmd" --version 2>&1 | awk '{print $2}')
        major=$(echo "$ver" | cut -d. -f1)
        minor=$(echo "$ver" | cut -d. -f2)
        if [ "$major" -ge 3 ] && [ "$minor" -ge 10 ]; then
            PYTHON_CMD="$cmd"
            info "Python: $ver ($cmd)"
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    error "需要 Python 3.10+，请先安装。"
    error "推荐: brew install python@3.10"
    exit 1
fi

# macOS / Apple Silicon 检测
if [[ "$(uname)" == "Darwin" ]]; then
    arch=$(uname -m)
    if [[ "$arch" == "arm64" ]]; then
        info "检测到 Apple Silicon ($arch)"
    else
        warn "非 Apple Silicon 架构 ($arch)，MPS 加速可能不可用"
    fi
else
    warn "非 macOS 系统 ($(uname))，MPS 加速不可用，将使用 CPU"
fi

# FFmpeg 检查
if ! command -v ffmpeg &>/dev/null; then
    warn "FFmpeg 未安装，音频格式转换可能受限"
    warn "安装: brew install ffmpeg"
fi

# ── HuggingFace 镜像（国内加速） ───────────────────────────
# 自动检测并启用国内镜像，可通过环境变量 HF_ENDPOINT 覆盖
if [ -z "${HF_ENDPOINT:-}" ]; then
    # 检查是否能访问官方站（简单连通性探测）
    if ! curl -s --max-time 5 https://huggingface.co >/dev/null 2>&1; then
        if [ -z "${CHATTTS_HF_MIRROR:-}" ]; then
            CHATTTS_HF_MIRROR="https://hf-mirror.com"
        fi
        export HF_ENDPOINT="$CHATTTS_HF_MIRROR"
        info "启用 HuggingFace 镜像: $HF_ENDPOINT"
    fi
fi

# ── 虚拟环境 ───────────────────────────────────────────────
VENV_DIR="$SCRIPT_DIR/.venv"

if [ ! -d "$VENV_DIR" ]; then
    step "创建虚拟环境..."
    $PYTHON_CMD -m venv "$VENV_DIR"
    info "虚拟环境已创建: $VENV_DIR"
fi

# 激活虚拟环境
source "$VENV_DIR/bin/activate"
info "已激活虚拟环境"

# ── 依赖安装 ───────────────────────────────────────────────
step "检查依赖..."

if [ -f "requirements.txt" ]; then
    # 仅检查关键包是否已安装
    if ! $PYTHON_CMD -c "import torch" 2>/dev/null; then
        warn "依赖未安装，开始安装..."
        info "这可能需要几分钟，请耐心等待..."
        pip install --upgrade pip -q
        pip install -r requirements.txt
        info "依赖安装完成"
    else
        info "依赖已就绪"
    fi
else
    warn "未找到 requirements.txt，跳过依赖检查"
fi

# ── 创建输入输出目录 ───────────────────────────────────────
step "确保目录结构..."
mkdir -p input/raw_audio input/texts input/emotion_ref
mkdir -p output/audio output/speaker_emb
mkdir -p temp

# ── 运行主流水线 ───────────────────────────────────────────
step "启动流水线..."
echo ""

$PYTHON_CMD main.py "$@"
EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    info "流水线执行成功"
else
    error "流水线执行失败 (退出码: $EXIT_CODE)"
fi

# ── 清理临时文件 ───────────────────────────────────────────
if [ -d "temp" ]; then
    rm -rf temp/*
fi

exit $EXIT_CODE