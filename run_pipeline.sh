DOMAIN=$1


if [ -z "$DOMAIN" ]; then
  echo "Usage: bash run.sh <domain>"
  exit 1
fi

i=1

for step_file in $(eval echo real_pipeline/step{$i..6}_*.py); do
  if [[ "$step_file" == "real_pipeline/step5_implicit_jailbreak.py" ]]; then
      continue
  fi

  echo "Processing $step_file with step $i..."

  if [ "$i" -eq 2 ]; then
    Step 2
    bash scripts/start_vllm_finetune_server.sh
    echo "Waiting 10 minutes for finetune server..."
    # sleep 600
    python $step_file --domain $DOMAIN

  elif [ "$i" -eq 3 ]; then
    # Step 3
    tmux kill-window -t llama3_1_70b_finetune
    bash scripts/start_vllm_granite_guardian_server.sh
    echo "Waiting 3 minutes for granite guardian server..."
    sleep 180
    python $step_file --domain $DOMAIN

  elif [ "$i" -eq 5 ]; then
    # Step 5
    tmux kill-window -t granite_guardian_evaluator
    bash scripts/start_vllm_finetune_server.sh
    echo "Waiting 10 minutes for finetune server..."
    sleep 600
    python $step_file --domain $DOMAIN

  else
    python $step_file --domain $DOMAIN
  fi

  i=$((i + 1))
done