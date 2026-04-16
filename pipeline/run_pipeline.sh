#!/bin/bash
# Full pipeline: build container → gemini experiments → text features → demographics
# Individual step failures are logged but do not abort the pipeline.

cd /home/nicpag/data/llm-recsys-clean

log()  { echo ""; echo "=== [$(date '+%H:%M:%S')] $* ==="; }
ok()   { echo "  -> OK"; }
fail() { echo "  -> FAILED (continuing)"; }

# ── 0. Ensure apptainer is available ─────────────────────────────────────────
if ! command -v apptainer &>/dev/null; then
    module load apptainer 2>/dev/null \
        || { echo "ERROR: apptainer not found. Run: module load apptainer"; exit 1; }
fi

SIF=/home/nicpag/data/llm-recsys-clean/llm-recsys.sif
py() { apptainer exec "$SIF" python "$@"; }

# ── 1. Ensure Ollama is installed ────────────────────────────────────────────
log "Checking Ollama..."
OLLAMA_BIN="$HOME/.local/bin/ollama"
if command -v ollama &>/dev/null; then
    OLLAMA_BIN=$(command -v ollama)
    echo "  Ollama found at $OLLAMA_BIN"
elif [ -x "$OLLAMA_BIN" ]; then
    echo "  Ollama found at $OLLAMA_BIN"
else
    echo "  Ollama not found — downloading tarball from GitHub releases..."
    mkdir -p "$HOME/.local/bin" "$HOME/.local/lib"
    TARBALL_URL=$(curl -fsSL https://api.github.com/repos/ollama/ollama/releases/latest \
        | python3 -c "import sys,json; assets=json.load(sys.stdin)['assets']; \
          url=[a['browser_download_url'] for a in assets if a['name']=='ollama-linux-amd64.tar.zst']; \
          print(url[0] if url else '')")
    if [ -z "$TARBALL_URL" ]; then
        echo "  -> Could not resolve download URL (demographics step will be skipped)."
        OLLAMA_BIN=""
    else
        curl -fsSL "$TARBALL_URL" -o /tmp/ollama-linux-amd64.tar.zst \
            && tar --use-compress-program=unzstd -xf /tmp/ollama-linux-amd64.tar.zst -C "$HOME/.local/" \
            && echo "  -> Ollama installed at $OLLAMA_BIN." \
            || { echo "  -> Ollama install FAILED (demographics step will be skipped)."; OLLAMA_BIN=""; }
    fi
fi

# ── 2. Build container (skip if already built) ───────────────────────────────
if apptainer inspect "$SIF" &>/dev/null; then
    log "Container already built — skipping build."
else
    [ -f "$SIF" ] && { echo "  Removing incomplete/corrupt .sif..."; rm -f "$SIF"; }
    log "Building apptainer container (may take 20-30 min)..."
    apptainer build --fakeroot "$SIF" llm-recsys.def \
        && ok || { echo "  -> Build FAILED — aborting."; exit 1; }
fi

# ── 3. Gemini experiments (only the 3 that failed) ───────────────────────────
for dataset in twitter bluesky reddit; do
    out="outputs/experiments/${dataset}_gemini_gemini-2.0-flash/post_level_data.csv"
    if [ -f "$out" ]; then
        log "Gemini $dataset — already done, skipping."
    else
        log "Gemini experiment: $dataset"
        py run_llm_recommendation.py \
            --dataset "$dataset" --provider gemini --n-trials 5 \
            && ok || fail
    fi
done

# ── 4. Text features for all experiments ─────────────────────────────────────
# Cache means each platform's 5,000 posts are computed only once.
for exp_dir in outputs/experiments/*/; do
    log "Text features: ${exp_dir%/}"
    py compute_text_features.py \
        --experiment-dir "${exp_dir%/}" \
        && ok || fail
done

# ── 5. LLM demographic inference — Twitter only ──────────────────────────────
if [ -z "$OLLAMA_BIN" ]; then
    log "Skipping demographics — Ollama not available."
else
    # Pull required models (no-op if already present)
    log "Pulling Ollama models (llama3.1:8b + mistral:v0.2)..."
    "$OLLAMA_BIN" pull llama3.1:8b && ok || fail
    "$OLLAMA_BIN" pull mistral:v0.2 && ok || fail

    # Start Ollama server if not already running
    if ! pgrep -x ollama &>/dev/null; then
        log "Starting Ollama server..."
        "$OLLAMA_BIN" serve &>/tmp/ollama.log &
        OLLAMA_PID=$!
        sleep 5  # give it time to start
        echo "  -> Ollama PID $OLLAMA_PID"
    else
        OLLAMA_PID=""
        echo "  Ollama already running."
    fi

    # Run inference (cache means each author processed only once)
    for exp_dir in outputs/experiments/twitter_*/; do
        log "Demographics: ${exp_dir%/}"
        py infer_demographics.py \
            --experiment-dir "${exp_dir%/}" \
            && ok || fail
    done

    # Stop Ollama if we started it
    [ -n "$OLLAMA_PID" ] && kill "$OLLAMA_PID" 2>/dev/null && echo "Ollama stopped."
fi

log "Pipeline complete."
