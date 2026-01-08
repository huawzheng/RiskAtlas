#!/bin/bash
# Start fine-tuned Llama-3.1-8B model vLLM server
# Adapted for A100 GPU - Enhanced version
export VLLM_USE_TRITON_LORA=0
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

BASE_MODEL_PATH="" # path of the alpaca dataset finetuned model 'Meta-Llama-3.1-8B-bnb-4bit', it should be 'models/Llama-3.1-8b-bnb-4bit-finetune'

LORA_MODEL_PATH="" # path of the LoRA model after safety finetuning, it should be 'experiment/finetune_model/{safety_finetune_model_name}'
# Create log directory
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

echo "🚀 Starting fine-tuned Llama-3.1-8B vLLM server..."
echo "Base model path: $BASE_MODEL_PATH"
echo "LoRA model path: $LORA_MODEL_PATH"

# Check if base model exists
if [ ! -d "$BASE_MODEL_PATH" ]; then
    echo "❌ Base model directory does not exist: $BASE_MODEL_PATH"
    exit 1
fi

if [ ! -f "$BASE_MODEL_PATH/config.json" ]; then
    echo "❌ Base model config file does not exist: $BASE_MODEL_PATH/config.json"
    exit 1
fi

# Check if LoRA model exists
if [ ! -d "$LORA_MODEL_PATH" ]; then
    echo "❌ LoRA model directory does not exist: $LORA_MODEL_PATH"
    exit 1
fi

if [ ! -f "$LORA_MODEL_PATH/adapter_config.json" ]; then
    echo "❌ LoRA config file does not exist: $LORA_MODEL_PATH/adapter_config.json"
    exit 1
fi

echo "✅ Base model directory exists: $BASE_MODEL_PATH"
echo "✅ LoRA model directory exists: $LORA_MODEL_PATH"

# ---------- Conda Environment Activation ----------
activate_conda() {
    echo "📦 Activating conda environment: RiskAtlas"
    if command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook)"
    else
        # Common installation path fallback
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
    # Check if port is occupied (compatible with environments without lsof)
    if command -v lsof >/dev/null 2>&1; then
        if lsof -iTCP:"$PORT" -sTCP:LISTEN -Pn >/dev/null 2>&1; then
            echo "⚠️  Port $PORT is occupied, trying to kill process..."
            kill -9 $(lsof -Pi :$PORT -sTCP:LISTEN -t) 2>/dev/null || true
            sleep 2
        fi
    else
        # Use netstat as fallback
        if command -v netstat >/dev/null 2>&1; then
            if netstat -tln | grep ":$PORT " >/dev/null 2>&1; then
                echo "⚠️  Port $PORT is occupied, please check manually"
            fi
        else
            echo "⚠️  Unable to check port occupation (missing lsof and netstat commands)"
        fi
    fi
    return 0
}

check_gpu_available() {
    echo "🔍 Checking if GPU is available..."
    if command -v nvidia-smi >/dev/null 2>&1; then
        if nvidia-smi >/dev/null 2>&1; then
            echo "✅ GPU check passed"
            return 0
        else
            echo "❌ GPU not available, cannot start service"
            return 1
        fi
    else
        echo "⚠️  nvidia-smi command not available, skipping GPU check"
        return 0
    fi
}

launch_finetune_server() {
    echo "🚀 Starting fine-tuned model vLLM service..."
    
    if ! check_port_free "8002"; then
        echo "⚠️  Port handling completed"
    fi
    
    if ! check_gpu_available; then
        echo "❌ GPU check failed, exiting"
        exit 1
    fi

    echo "🔧 Startup configuration:"
    echo "   - Base model: $BASE_MODEL_PATH"
    echo "   - LoRA adapter: $LORA_MODEL_PATH"
    echo "   - Port: 8002"
    echo "   - GPU: A100-80G"
    echo "   - Tensor parallel size: 1"
    echo "   - Max length: 8192"
    echo "   - GPU memory utilization: 0.9"
    echo ""

    # Check if tmux is installed
    if ! command -v tmux &> /dev/null; then
        echo "❌ tmux not installed, please install tmux first"
        exit 1
    fi

    # Check if tmux session with the same name already exists
    SESSION_NAME="Llama-3.1-8B-finetune"
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        echo "⚠️  tmux session '$SESSION_NAME' already exists, terminating..."
        tmux kill-session -t "$SESSION_NAME"
        sleep 2
    fi

    # Use tmux to start service in background
    tmux new-session -d -s "$SESSION_NAME" -c "$PROJECT_ROOT" bash -c "
        eval \"\$(conda shell.bash hook)\";
        conda activate RiskAtlas;
        python -m vllm.entrypoints.openai.api_server \
            --model '$BASE_MODEL_PATH' \
            --enable-lora \
            --lora-modules Llama-3.1-8B-finetune='$LORA_MODEL_PATH' \
            --max-lora-rank 64 \
            --host 0.0.0.0 \
            --port 8002 \
            --max-model-len 8192 \
            --tensor-parallel-size 1 \
            --gpu-memory-utilization 0.9 \
            --dtype auto \
            --trust-remote-code \
            --served-model-name 'Llama-3.1-8B-finetune' \
            --disable-log-requests \
            2>&1 | tee '$LOG_DIR/Llama-3.1-8B-finetune_vllm_8002.log'
    "

    echo "✅ Fine-tuned model service started in tmux session '$SESSION_NAME'"
    echo "📋 Management commands:"
    echo "   - View service status: tmux attach-session -t $SESSION_NAME"
    echo "   - Exit tmux but keep service running: Ctrl+B then press D"
    echo "   - Stop service: Press Ctrl+C in tmux session"
    echo "   - Service log: $LOG_DIR/Llama-3.1-8B-finetune_vllm_8002.log"
    echo ""
}

wait_for_service() {
    echo "📋 Service started, please check status manually"
    echo "🌐 Service endpoint: http://localhost:8002"
    echo "📝 Test command: curl http://localhost:8002/v1/models"
    echo "   Check method: tmux attach-session -t Llama-3.1-8B-finetune"
    return 0
}

main() {
    echo "============= Llama-3.1-8B Fine-tuned Model Startup Script ============="
    
    activate_conda
    launch_finetune_server
    wait_for_service
    
    echo ""
    echo "🎉 Fine-tuned model service configuration completed!"
    echo ""
    echo "📋 Next steps:"
    echo "   1. Test service: curl http://localhost:8002/v1/models"
    echo "   2. Run step1: python step1_prompt_generation.py"
    echo ""
    echo "🔧 Troubleshooting:"
    echo "   - View service log: cat $LOG_DIR/Llama-3.1-8B-finetune_vllm_8002.log"
    echo "   - Check GPU usage: nvidia-smi"
    echo "   - Enter service session: tmux attach-session -t Llama-3.1-8B-finetune"
}

main "$@"
