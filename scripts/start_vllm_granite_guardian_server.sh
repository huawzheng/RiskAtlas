#!/bin/bash
# Start Granite-Guardian-3.1-8B vLLM server (for toxicity evaluation)
# Adapted for A100 GPU - Enhanced version

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "🚀 Starting Granite-Guardian-3.1-8B vLLM server (A100 enhanced version)..."
echo "Purpose: Toxicity evaluation (Stage 2)"

# ---------- Configuration Parameters ----------
GRANITE_MODEL_PATH=""  # cache path of the model 'granite-guardian-3.1-8b'
GRANITE_PORT="8001"
GRANITE_HOST="0.0.0.0"
GRANITE_MAX_LEN="4096"       # Length suitable for toxicity evaluation
GRANITE_TP_SIZE="1"          # Use 1 A100 GPU
GRANITE_GPU_UTIL="0.8"       # A100 memory utilization
GRANITE_GPUS="0"             # Use GPU 0
# GRANITE_GPUS="1"             # Use GPU 0

LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

# ---------- Conda Environment Activation ----------
activate_conda() {
    echo "📦 Activating conda environment: RiskAtlas"
    if command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook)"
    else
        # Common installation paths fallback
        for p in "$HOME/miniconda3/etc/profile.d/conda.sh" \
                 "$HOME/anaconda3/etc/profile.d/conda.sh" \
                 "/opt/conda/etc/profile.d/conda.sh"; do
            if [ -f "$p" ]; then
                source "$p"
                break
            fi
        done
    fi
    conda activate RiskAtlas
}

check_port_free() {
    local PORT="$1"
    if command -v lsof >/dev/null 2>&1; then
        if lsof -iTCP:"$PORT" -sTCP:LISTEN -Pn >/dev/null 2>&1; then
            echo "⚠️  Port $PORT is occupied, trying to kill process..."
            kill -9 $(lsof -Pi :$PORT -sTCP:LISTEN -t) 2>/dev/null || true
            sleep 2
        fi
    fi
    return 0
}

check_gpu_available() {
    local GPUS="$1"
    echo "🔍 Checking if required GPUs for Granite Guardian ($GPUS) are available..."
    
    # Convert comma-separated GPU list to array
    IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
    
    for gpu in "${GPU_ARRAY[@]}"; do
        if ! nvidia-smi -i "$gpu" >/dev/null 2>&1; then
            echo "❌ GPU $gpu is not available, cannot start Granite Guardian"
            return 1
        fi
    done
    
    echo "✅ GPU ($GPUS) check passed"
    return 0
}

launch_granite_guardian() {
    echo "🚀 Starting Granite Guardian vLLM service on GPU${GRANITE_GPUS} (port=$GRANITE_PORT) ..."
    
    if ! check_port_free "$GRANITE_PORT"; then
        echo "⚠️  Port handling completed"
    fi
    
    if ! check_gpu_available "$GRANITE_GPUS"; then
        echo "❌ GPU check failed, exiting"
        exit 1
    fi

    echo "🔧 Startup configuration:"
    echo "   - Model: $GRANITE_MODEL_PATH"
    echo "   - Port: $GRANITE_PORT"
    echo "   - GPU: $GRANITE_GPUS (A100)"
    echo "   - Tensor parallel size: $GRANITE_TP_SIZE"
    echo "   - Max length: $GRANITE_MAX_LEN"
    echo "   - GPU memory utilization: $GRANITE_GPU_UTIL"
    echo ""

    # Check if tmux is installed
    if ! command -v tmux &> /dev/null; then
        echo "❌ tmux is not installed, please install tmux first"
        exit 1
    fi

    # Check if tmux session with same name already exists
    SESSION_NAME="granite_guardian_evaluator"
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        echo "⚠️  tmux session '$SESSION_NAME' already exists, terminating..."
        tmux kill-session -t "$SESSION_NAME"
        sleep 2
    fi

    # Start service in background using tmux
    tmux new-session -d -s "$SESSION_NAME" -c "$PROJECT_ROOT" bash -c "
        eval \"\$(conda shell.bash hook)\";
        conda activate RiskAtlas;
        CUDA_VISIBLE_DEVICES=$GRANITE_GPUS python -m vllm.entrypoints.openai.api_server \
            --model '$GRANITE_MODEL_PATH' \
            --host '$GRANITE_HOST' \
            --port '$GRANITE_PORT' \
            --max-model-len '$GRANITE_MAX_LEN' \
            --tensor-parallel-size '$GRANITE_TP_SIZE' \
            --gpu-memory-utilization '$GRANITE_GPU_UTIL' \
            --trust-remote-code \
            --served-model-name 'granite-guardian-3.1-8b-vllm-server' \
            --disable-log-requests \
            2>&1 | tee '$LOG_DIR/granite_guardian_vllm_${GRANITE_PORT}.log'
    "

    echo "✅ Granite Guardian service has been started in tmux session '$SESSION_NAME'"
    echo "📋 Management commands:"
    echo "   - View service status: tmux attach -t $SESSION_NAME"
    echo "   - Exit tmux but keep service running: Ctrl+B then press D"
    echo "   - Stop service: Press Ctrl+C in tmux session"
    echo "   - Service log: $LOG_DIR/granite_guardian_vllm_${GRANITE_PORT}.log"
    echo ""
}

wait_for_service() {
    echo "📋 Service has been started, please check status manually"
    echo "🌐 Service endpoint: http://localhost:$GRANITE_PORT"
    echo "📝 Test command: curl http://localhost:$GRANITE_PORT/v1/models"
    echo "   Check method: tmux attach -t granite_guardian_evaluator"
    return 0
}

main() {
    echo "============= Granite Guardian A100 Startup Script ============="
    
    activate_conda
    launch_granite_guardian
    wait_for_service
    
    echo ""
    echo "🎉 Granite Guardian service configuration completed!"
    echo ""
    echo "📋 Next steps:"
    echo "   1. Test service: curl http://localhost:$GRANITE_PORT/v1/models"
    echo "   2. Run step2: python step2_toxicity_evaluation.py --input-file <step1 output file>"
    echo ""
    echo "🔧 Troubleshooting:"
    echo "   - View service log: cat $LOG_DIR/granite_guardian_vllm_${GRANITE_PORT}.log"
    echo "   - Check GPU usage: nvidia-smi"
    echo "   - Enter service session: tmux attach -t granite_guardian_evaluator"
}

main "$@"
