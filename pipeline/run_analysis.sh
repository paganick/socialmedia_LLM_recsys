#!/usr/bin/env bash
# Convenience script: run all downstream steps (Steps 3a–5) in sequence.
#
# Steps:
#   3a  compute_text_features   — for all 9 experiment directories
#   3b  infer_demographics      — Twitter only; starts/stops Ollama automatically
#   4   compute_bias_metrics
#   5   generate_figures
#
# Requires the apptainer container (llm-recsys.sif) and the module system.
# Alternatively, activate the conda environment and call the Python scripts directly.
#
# Usage:
#   bash run_downstream.sh
#   nohup bash run_downstream.sh > logs/downstream.log 2>&1 &

set -e
cd "$(dirname "$0")"

module load apptainer 2>/dev/null || true
APPTAINER="apptainer exec --bind $(pwd):$(pwd) llm-recsys.sif"

echo "=== Step 3a: compute_text_features ==="
for dataset in twitter bluesky reddit; do
  for provider in anthropic openai gemini; do
    exp_dir=$(ls -d outputs/experiments/${dataset}_${provider}_* 2>/dev/null | head -1)
    if [ -z "$exp_dir" ]; then
      echo "  SKIP $dataset/$provider — no experiment dir found"
      continue
    fi
    echo "  Processing $exp_dir ..."
    $APPTAINER python compute_text_features.py --experiment-dir "$exp_dir"
  done
done
echo "Text features done."

echo ""
echo "=== Step 3b: infer_demographics (Twitter only) ==="
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
  echo "  Starting Ollama..."
  ollama serve > logs/ollama.log 2>&1 &
  OLLAMA_PID=$!
  sleep 10
else
  OLLAMA_PID=""
fi

for provider in anthropic openai gemini; do
  exp_dir=$(ls -d outputs/experiments/twitter_${provider}_* 2>/dev/null | head -1)
  if [ -z "$exp_dir" ]; then
    echo "  SKIP twitter/$provider — no experiment dir"
    continue
  fi
  echo "  Processing $exp_dir ..."
  $APPTAINER python infer_demographics.py --experiment-dir "$exp_dir"
done
[ -n "$OLLAMA_PID" ] && kill $OLLAMA_PID 2>/dev/null || true
echo "Demographics done."

echo ""
echo "=== Step 4: compute_bias_metrics ==="
$APPTAINER python compute_bias_metrics.py
echo "Bias metrics done."

echo ""
echo "=== Step 5: generate_figures ==="
$APPTAINER python generate_figures.py
echo "Figures done."

echo ""
echo "=== All downstream steps complete ==="
