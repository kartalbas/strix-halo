#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# LLM Server Management Script
# Search HuggingFace for GGUF models, download, and serve via llama-server
# Uses native llama.cpp builds (Vulkan/HIP) as a systemd user service.
# =============================================================================

# ---- Hardcoded Defaults (lowest priority) -----------------------------------
CONTAINER="native"
CONTEXT_SIZE="262144"
PARALLEL_SLOTS="1"
GPU_LAYERS="999"
FLASH_ATTENTION="1"
ROPE_SCALE="1.0"
KV_CACHE_TYPE=""
BATCH_SIZE="4096"
CHECKPOINT_EVERY_NT=""
CTX_CHECKPOINTS="32"
HOST="0.0.0.0"
PORT="8080"
# -----------------------------------------------------------------------------

# Resolve the real home directory (works even under sudo)
REAL_HOME=$(getent passwd "${SUDO_USER:-$USER}" | cut -d: -f6)
REAL_HOME="${REAL_HOME:-$HOME}"

# Add ~/.local/bin to PATH for any user-installed tools
export PATH="$REAL_HOME/.local/bin:$PATH"

SERVICE_NAME="llm-server"
UNIT_DIR="$REAL_HOME/.config/systemd/user"
UNIT_FILE="$UNIT_DIR/$SERVICE_NAME.service"
MODELS_DIR="$REAL_HOME/models/gguf"
CONFIG_DIR="$REAL_HOME/.config/llm-server"
SERVER_CONF="$CONFIG_DIR/server.conf"
REGISTRY_DIR="$CONFIG_DIR/models"
ACTIVE_MODEL_FILE="$CONFIG_DIR/active-model"

# Load saved settings (overrides hardcoded defaults)
# shellcheck source=/dev/null
[[ -f "$SERVER_CONF" ]] && source "$SERVER_CONF"

# Environment variable overrides (highest priority)
[[ -n "${LLM_CONTEXT_SIZE:-}"    ]] && CONTEXT_SIZE="$LLM_CONTEXT_SIZE"
[[ -n "${LLM_PARALLEL_SLOTS:-}"  ]] && PARALLEL_SLOTS="$LLM_PARALLEL_SLOTS"
[[ -n "${LLM_GPU_LAYERS:-}"      ]] && GPU_LAYERS="$LLM_GPU_LAYERS"
[[ -n "${LLM_FLASH_ATTENTION:-}" ]] && FLASH_ATTENTION="$LLM_FLASH_ATTENTION"
[[ -n "${LLM_ROPE_SCALE:-}"      ]] && ROPE_SCALE="$LLM_ROPE_SCALE"
[[ -n "${LLM_KV_CACHE_TYPE:-}"  ]] && KV_CACHE_TYPE="$LLM_KV_CACHE_TYPE"
[[ -n "${LLM_BATCH_SIZE:-}"     ]] && BATCH_SIZE="$LLM_BATCH_SIZE"
[[ -n "${LLM_CHECKPOINT_EVERY_NT:-}" ]] && CHECKPOINT_EVERY_NT="$LLM_CHECKPOINT_EVERY_NT"
[[ -n "${LLM_CTX_CHECKPOINTS:-}" ]] && CTX_CHECKPOINTS="$LLM_CTX_CHECKPOINTS"
[[ -n "${LLM_HOST:-}"            ]] && HOST="$LLM_HOST"
[[ -n "${LLM_PORT:-}"            ]] && PORT="$LLM_PORT"

# Colors
red()    { printf '\033[1;31m%s\033[0m\n' "$*"; }
green()  { printf '\033[1;32m%s\033[0m\n' "$*"; }
yellow() { printf '\033[1;33m%s\033[0m\n' "$*"; }
bold()   { printf '\033[1m%s\033[0m\n' "$*"; }
dim()    { printf '\033[2m%s\033[0m\n' "$*"; }

# ---- Helpers ----------------------------------------------------------------

# Guard: commands that use systemctl --user must not run under sudo
require_no_sudo() {
    if [[ -n "${SUDO_USER:-}" ]]; then
        red "This command uses 'systemctl --user' and cannot run under sudo."
        echo "Run without sudo:  $0 $*"
        exit 1
    fi
}

# Detect llama-server binary path (native builds only)
detect_llama_server() {
    local build_dir
    build_dir=$(_dist_build_dir "${DIST_BACKEND:-vulkan}")
    local bin="$build_dir/bin/llama-server"
    if [[ ! -x "$bin" ]]; then
        red "llama-server not found at $bin (backend=${DIST_BACKEND:-vulkan})"
        echo "Build it:"
        if [[ "${DIST_BACKEND:-vulkan}" == "hip" ]]; then
            echo "  cd ~/llama.cpp && HIPCXX=\"\$(hipconfig -l)/clang\" HIP_PATH=\"\$(hipconfig -R)\" cmake -B build-hip -DGGML_HIP=ON -DGGML_RPC=ON -DAMDGPU_TARGETS=gfx1151 -DCMAKE_BUILD_TYPE=Release && cmake --build build-hip -j\$(nproc)"
        else
            echo "  cd ~/llama.cpp && cmake -B build -DGGML_VULKAN=ON -DGGML_RPC=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build -j\$(nproc)"
        fi
        exit 1
    fi
    echo "$bin"
}

save_conf() {
    local key="$1" value="$2"
    mkdir -p "$CONFIG_DIR"
    touch "$SERVER_CONF"
    if grep -q "^${key}=" "$SERVER_CONF" 2>/dev/null; then
        sed -i "s|^${key}=.*|${key}=${value}|" "$SERVER_CONF"
    else
        echo "${key}=${value}" >> "$SERVER_CONF"
    fi
    # Update in-memory variable so subsequent commands in this invocation use the new value
    printf -v "$key" '%s' "$value"
}

ensure_dirs() {
    mkdir -p "$MODELS_DIR" "$REGISTRY_DIR"
}

# Format a download count for display (e.g. 335000 -> "335k")
fmt_count() {
    local n="$1"
    if (( n >= 1000000 )); then
        printf "%s.%sM" $(( n / 1000000 )) $(( (n % 1000000) / 100000 ))
    elif (( n >= 1000 )); then
        printf "%sk" $(( n / 1000 ))
    else
        printf "%s" "$n"
    fi
}

# Format bytes for display
fmt_bytes() {
    local b="$1"
    if (( b >= 1073741824 )); then
        printf "%.1fGB" "$(echo "scale=1; $b / 1073741824" | bc)"
    elif (( b >= 1048576 )); then
        printf "%.0fMB" "$(echo "scale=0; $b / 1048576" | bc)"
    else
        printf "%sB" "$b"
    fi
}

# Get active model key
get_active_model() {
    if [[ -f "$ACTIVE_MODEL_FILE" ]]; then
        cat "$ACTIVE_MODEL_FILE"
    fi
}

# Get a field from a model config file
get_model_field() {
    local key="$1" field="$2"
    local conf="$REGISTRY_DIR/$key.conf"
    if [[ -f "$conf" ]]; then
        grep "^${field}=" "$conf" 2>/dev/null | cut -d= -f2-
    fi
}

# Get the GGUF path for the active model
get_active_gguf() {
    local active
    active=$(get_active_model)
    if [[ -z "$active" ]]; then
        return 1
    fi
    get_model_field "$active" "gguf"
}

# Auto-detect the first GGUF file in a directory (handles split models)
detect_gguf() {
    local dir="$1"
    # Look for -00001-of- pattern first (split models), then any .gguf
    local first
    first=$(find "$dir" -maxdepth 1 -name '*-00001-of-*.gguf' -type f 2>/dev/null | sort | head -1)
    if [[ -z "$first" ]]; then
        first=$(find "$dir" -maxdepth 1 -name '*.gguf' -type f 2>/dev/null | sort | head -1)
    fi
    echo "$first"
}

# Migrate existing model if no registry exists yet
migrate_existing_models() {
    ensure_dirs
    # If registry already has entries, skip
    if compgen -G "$REGISTRY_DIR/*.conf" >/dev/null 2>&1; then
        return
    fi
    # Check for the known pre-existing Qwen3 model
    local qwen_dir="$MODELS_DIR/qwen3-next-80B/Q6_K"
    if [[ -d "$qwen_dir" ]]; then
        local gguf
        gguf=$(detect_gguf "$qwen_dir")
        if [[ -n "$gguf" ]]; then
            local key="qwen3-next-80b-q6k"
            cat > "$REGISTRY_DIR/$key.conf" <<EOF
repo=unsloth/Qwen3-Next-80B-A3B-Instruct-GGUF
quant=Q6_K
local_dir=$MODELS_DIR/qwen3-next-80B
gguf=$gguf
EOF
            # Set as active if no active model
            if [[ ! -f "$ACTIVE_MODEL_FILE" ]]; then
                echo "$key" > "$ACTIVE_MODEL_FILE"
            fi
            dim "Migrated existing model: $key"
        fi
    fi
}

create_unit_file() {
    local model_path="$1"

    # Detect the correct binary path
    local llama_bin
    llama_bin=$(detect_llama_server)

    local build_dir
    build_dir=$(_dist_build_dir "${DIST_BACKEND:-vulkan}")
    local exec_cmd="$llama_bin"
    local env_lines="Environment=LD_LIBRARY_PATH=$build_dir/bin"
    # Add backend-specific env vars
    local backend_env
    backend_env=$(_dist_backend_env_rpc "${DIST_BACKEND:-vulkan}")
    if [[ -n "$backend_env" && "$backend_env" != \#* ]]; then
        env_lines+=$'\n'"$backend_env"
    fi
    exec_cmd+=" -m $model_path"
    exec_cmd+=" -c $CONTEXT_SIZE"
    exec_cmd+=" -np $PARALLEL_SLOTS"
    exec_cmd+=" -ngl $GPU_LAYERS"
    exec_cmd+=" -b $BATCH_SIZE"
    [[ "$FLASH_ATTENTION" == "1" ]] && exec_cmd+=" -fa 1"
    exec_cmd+=" --no-mmap"
    if [[ -n "$KV_CACHE_TYPE" ]]; then
        exec_cmd+=" -ctk $KV_CACHE_TYPE -ctv $KV_CACHE_TYPE"
    fi
    if [[ "$ROPE_SCALE" != "1" && "$ROPE_SCALE" != "1.0" ]]; then
        exec_cmd+=" --rope-scaling yarn"
        exec_cmd+=" --rope-scale $ROPE_SCALE"
    fi
    if [[ -n "$CHECKPOINT_EVERY_NT" && "$CHECKPOINT_EVERY_NT" != "0" ]]; then
        exec_cmd+=" --checkpoint-every-n-tokens $CHECKPOINT_EVERY_NT"
        exec_cmd+=" --ctx-checkpoints ${CTX_CHECKPOINTS:-32}"
    fi
    exec_cmd+=" --host $HOST"
    exec_cmd+=" --port $PORT"

    local desc="LLM Server"
    desc+=" (native, ${DIST_BACKEND:-vulkan})"

    mkdir -p "$UNIT_DIR"
    cat > "$UNIT_FILE" <<EOF
[Unit]
Description=$desc
After=network-online.target
Wants=network-online.target

[Service]
Type=exec
$env_lines
ExecStart=$exec_cmd
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
EOF
}

# ---- Subcommands ------------------------------------------------------------

cmd_search() {
    if [[ $# -eq 0 ]]; then
        red "Usage: $0 search <query>"
        echo "Example: $0 search qwen3"
        exit 1
    fi

    local query="$*"
    bold "Searching HuggingFace for GGUF models: $query"
    echo ""

    local url="https://huggingface.co/api/models?search=$(printf '%s' "$query gguf" | jq -sRr @uri)&sort=downloads&direction=-1&limit=20"
    local response
    response=$(curl -sf "$url") || { red "Failed to query HuggingFace API"; exit 1; }

    # Filter to repos that have gguf in their tags and display
    local count=0
    while IFS= read -r line; do
        local id downloads likes
        id=$(echo "$line" | jq -r '.id')
        downloads=$(echo "$line" | jq -r '.downloads // 0')
        likes=$(echo "$line" | jq -r '.likes // 0')
        tags=$(echo "$line" | jq -r '.tags // [] | join(",")')

        # Filter: must have gguf in tags or id
        if echo "$tags" | grep -qi "gguf" || echo "$id" | grep -qi "gguf"; then
            count=$((count + 1))
            printf "  %2d) %-55s (%s downloads, %s likes)\n" \
                "$count" "$id" "$(fmt_count "$downloads")" "$(fmt_count "$likes")"
        fi
    done < <(echo "$response" | jq -c '.[]')

    if [[ $count -eq 0 ]]; then
        yellow "No GGUF models found for: $query"
        echo "Try a broader search term."
    else
        echo ""
        dim "Use: $0 download <repo-name> [quant] to download a model"
    fi
}

cmd_download() {
    ensure_dirs

    # Environment workaround for hf download issues
    export HF_HUB_ENABLE_HF_TRANSFER=0
    unset HF_XET_HIGH_PERFORMANCE 2>/dev/null || true

    if [[ $# -ge 1 ]]; then
        # Direct mode: download <repo> [quant]
        _download_direct "$@"
    else
        # Interactive mode
        _download_interactive
    fi
}

_download_interactive() {
    echo ""
    read -rp "Search HuggingFace: " query
    if [[ -z "$query" ]]; then
        red "No query entered."
        exit 1
    fi

    bold "Searching..."
    local url="https://huggingface.co/api/models?search=$(printf '%s' "$query gguf" | jq -sRr @uri)&sort=downloads&direction=-1&limit=20"
    local response
    response=$(curl -sf "$url") || { red "Failed to query HuggingFace API"; exit 1; }

    # Collect matching repos
    local -a repo_ids=()
    while IFS= read -r line; do
        local id downloads likes tags
        id=$(echo "$line" | jq -r '.id')
        downloads=$(echo "$line" | jq -r '.downloads // 0')
        likes=$(echo "$line" | jq -r '.likes // 0')
        tags=$(echo "$line" | jq -r '.tags // [] | join(",")')

        if echo "$tags" | grep -qi "gguf" || echo "$id" | grep -qi "gguf"; then
            repo_ids+=("$id")
            printf "  %2d) %-55s (%s downloads, %s likes)\n" \
                "${#repo_ids[@]}" "$id" "$(fmt_count "$downloads")" "$(fmt_count "$likes")"
        fi
    done < <(echo "$response" | jq -c '.[]')

    if [[ ${#repo_ids[@]} -eq 0 ]]; then
        yellow "No GGUF models found."
        exit 1
    fi

    echo ""
    read -rp "Pick a repo number [1-${#repo_ids[@]}]: " pick
    if [[ -z "$pick" ]] || (( pick < 1 || pick > ${#repo_ids[@]} )); then
        red "Invalid selection."
        exit 1
    fi
    local repo="${repo_ids[$((pick - 1))]}"
    bold "Selected: $repo"
    echo ""

    # Fetch file list and show quant options
    _pick_quant_and_download "$repo"
}

_pick_quant_and_download() {
    local repo="$1"

    bold "Fetching file list for $repo..."
    local api_url="https://huggingface.co/api/models/$repo?blobs=true"
    local repo_info
    repo_info=$(curl -sf "$api_url") || { red "Failed to fetch repo info"; exit 1; }

    # Parse GGUF files: extract paths and sizes
    # Files can be in subdirectories (quant dirs) like Q6_K/file.gguf or at root
    local -a quant_names=()
    local -A quant_sizes=()
    local -A quant_counts=()

    while IFS=$'\t' read -r path size; do
        # Skip non-gguf files
        [[ "$path" == *.gguf ]] || continue

        local quant_dir
        if [[ "$path" == */* ]]; then
            quant_dir="${path%%/*}"
        else
            # Root-level file — extract quant from filename (e.g. model-Q4_K_M.gguf)
            quant_dir=$(echo "$path" | sed -n 's/.*[-_]\(Q[0-9][^.]*\)\.gguf$/\1/p')
            if [[ -z "$quant_dir" ]]; then
                quant_dir=$(echo "$path" | sed -n 's/.*[-_]\([Ff][0-9][0-9]*\)\.gguf$/\1/p')
            fi
            if [[ -z "$quant_dir" ]]; then
                quant_dir="default"
            fi
        fi

        # Track unique quants
        if [[ -z "${quant_sizes[$quant_dir]+x}" ]]; then
            quant_names+=("$quant_dir")
            quant_sizes[$quant_dir]=0
            quant_counts[$quant_dir]=0
        fi
        quant_sizes[$quant_dir]=$(( ${quant_sizes[$quant_dir]} + size ))
        quant_counts[$quant_dir]=$(( ${quant_counts[$quant_dir]} + 1 ))
    done < <(echo "$repo_info" | jq -r '
        .siblings // [] | .[] |
        select(.rfilename | test("\\.gguf$")) |
        [.rfilename, (.size // 0 | tostring)] | @tsv
    ')

    if [[ ${#quant_names[@]} -eq 0 ]]; then
        red "No GGUF files found in this repo."
        exit 1
    fi

    echo ""
    bold "Available quants:"
    for i in "${!quant_names[@]}"; do
        local q="${quant_names[$i]}"
        printf "  %2d) %-20s %d file(s), %s\n" \
            $((i + 1)) "$q" "${quant_counts[$q]}" "$(fmt_bytes "${quant_sizes[$q]}")"
    done

    echo ""
    read -rp "Pick a quant [1-${#quant_names[@]}]: " qpick
    if [[ -z "$qpick" ]] || (( qpick < 1 || qpick > ${#quant_names[@]} )); then
        red "Invalid selection."
        exit 1
    fi
    local quant="${quant_names[$((qpick - 1))]}"

    _do_download "$repo" "$quant"
}

_download_direct() {
    local repo="$1"
    local quant="${2:-}"

    if [[ -z "$quant" ]]; then
        # No quant specified — show available quants and let user pick
        _pick_quant_and_download "$repo"
        return
    fi

    _do_download "$repo" "$quant"
}

_do_download() {
    local repo="$1"
    local quant="$2"

    # Derive a local directory name from the repo
    local model_name
    model_name=$(echo "$repo" | sed 's|.*/||; s|-GGUF$||; s|-gguf$||' | tr '[:upper:]' '[:lower:]')
    local local_dir="$MODELS_DIR/$model_name"

    # Ensure local_dir exists before searching
    mkdir -p "$local_dir"

    # Check if already downloaded
    local existing_gguf=""
    if [[ -d "$local_dir/$quant" ]]; then
        existing_gguf=$(detect_gguf "$local_dir/$quant")
    fi
    if [[ -z "$existing_gguf" ]]; then
        # Only match root-level files whose name contains this quant
        existing_gguf=$(find "$local_dir" -maxdepth 1 -name "*${quant}*.gguf" -type f 2>/dev/null | sort | head -1)
    fi
    if [[ -n "$existing_gguf" ]]; then
        yellow "Already downloaded: $existing_gguf"
        local answer
        read -rp "Re-download? [y/N] " answer
        answer="${answer:-N}"
        if [[ "${answer,,}" != "y" ]]; then
            # Skip download, go straight to registration
            _register_model "$repo" "$quant" "$local_dir" "$existing_gguf"
            return
        fi
    fi

    echo ""
    bold "Downloading: $repo / $quant"
    echo "  Destination: $local_dir/$quant/"
    echo ""

    # Download via Python API (huggingface_hub.snapshot_download).
    # This avoids relying on the huggingface-cli / hf binary, which pip does
    # not always create (known issue with huggingface_hub >=1.4 on Fedora).
    # On failure, partial files are cleaned up automatically.
    local hf_cache="$local_dir/.cache"
    if ! HF_REPO="$repo" HF_QUANT="$quant" HF_DIR="$local_dir" \
    python3 - <<'PYEOF'
import os, sys, shutil, traceback
from huggingface_hub import snapshot_download
repo    = os.environ["HF_REPO"]
quant   = os.environ["HF_QUANT"]
destdir = os.environ["HF_DIR"]
print(f"  Fetching {repo} [{quant}] -> {destdir}", flush=True)
try:
    snapshot_download(
        repo_id=repo,
        allow_patterns=[f"{quant}/*.gguf", f"*{quant}*.gguf"],
        local_dir=destdir,
    )
except Exception as e:
    msg = str(e)
    if "No space left on device" in msg or "os error 28" in msg:
        print(f"\n[ERROR] Disk full — not enough space to download this model.", file=sys.stderr, flush=True)
        print(f"        Free up space or choose a smaller quantisation.", file=sys.stderr)
    else:
        print(f"\n[ERROR] Download failed: {e}", file=sys.stderr, flush=True)
    # Clean up incomplete files so a retry starts fresh
    cache_dir = os.path.join(destdir, ".cache")
    quant_dir = os.path.join(destdir, quant)
    for path in [cache_dir, quant_dir]:
        if os.path.exists(path):
            print(f"  Cleaning up: {path}", file=sys.stderr)
            shutil.rmtree(path, ignore_errors=True)
    sys.exit(1)
PYEOF
    then
        red "Download failed. Partial files have been removed — safe to retry."
        return 1
    fi

    # Detect GGUF file
    local gguf_file=""
    if [[ -d "$local_dir/$quant" ]]; then
        gguf_file=$(detect_gguf "$local_dir/$quant")
    fi
    if [[ -z "$gguf_file" ]]; then
        gguf_file=$(detect_gguf "$local_dir")
    fi

    if [[ -z "$gguf_file" ]]; then
        red "Warning: Could not auto-detect GGUF file after download."
        echo "You may need to manually configure the model."
        return 1
    fi

    _register_model "$repo" "$quant" "$local_dir" "$gguf_file"
}

_register_model() {
    local repo="$1" quant="$2" local_dir="$3" gguf_file="$4"
    local model_name
    model_name=$(echo "$repo" | sed 's|.*/||; s|-GGUF$||; s|-gguf$||' | tr '[:upper:]' '[:lower:]')
    local key="${model_name}-$(echo "$quant" | tr '[:upper:]' '[:lower:]')"

    ensure_dirs
    # Use quant subdirectory as local_dir if it exists
    local reg_dir="$local_dir"
    if [[ -d "$local_dir/$quant" ]]; then
        reg_dir="$local_dir/$quant"
    fi
    cat > "$REGISTRY_DIR/$key.conf" <<EOF
repo=$repo
quant=$quant
local_dir=$reg_dir
gguf=$gguf_file
EOF

    echo ""
    green "Model ready!"
    echo "  Model key: $key"
    echo "  GGUF: $gguf_file"
    echo ""

    # Ask to set as active model
    local current_active answer
    current_active=$(get_active_model)
    if [[ "$current_active" == "$key" ]]; then
        green "Already the active model."
    else
        read -rp "Set as active model? [Y/n] " answer
        answer="${answer:-Y}"
        if [[ "${answer,,}" == "y" ]]; then
            echo "$key" > "$ACTIVE_MODEL_FILE"
            green "Active model → $key"

            # Ask to restart service if running
            if systemctl --user is-active "$SERVICE_NAME.service" &>/dev/null 2>&1; then
                read -rp "Restart service now to load new model? [Y/n] " answer
                answer="${answer:-Y}"
                if [[ "${answer,,}" == "y" ]]; then
                    systemctl --user restart "$SERVICE_NAME.service"
                    green "Service restarted"
                fi
            fi
        else
            dim "Use '$0 select $key' to activate later"
        fi
    fi
}

cmd_models() {
    ensure_dirs
    migrate_existing_models

    local active
    active=$(get_active_model)

    if ! compgen -G "$REGISTRY_DIR/*.conf" >/dev/null 2>&1; then
        yellow "No models registered."
        echo "Use '$0 search <query>' to find models, then '$0 download' to get one."
        exit 0
    fi

    echo ""
    bold "Downloaded models:"
    echo ""
    for conf in "$REGISTRY_DIR"/*.conf; do
        local key repo quant marker
        key=$(basename "$conf" .conf)
        repo=$(grep "^repo=" "$conf" | cut -d= -f2-)
        quant=$(grep "^quant=" "$conf" | cut -d= -f2-)

        if [[ "$key" == "$active" ]]; then
            marker="* "
            printf "  \033[1;32m* %-25s %-50s %s  (active)\033[0m\n" "$key" "$repo" "$quant"
        else
            printf "    %-25s %-50s %s\n" "$key" "$repo" "$quant"
        fi
    done
    echo ""
}

cmd_select() {
    migrate_existing_models
    if [[ $# -eq 0 ]]; then
        red "Usage: $0 select <model-key>"
        echo "Run '$0 models' to see available models."
        exit 1
    fi

    local key="$1"
    local conf="$REGISTRY_DIR/$key.conf"

    if [[ ! -f "$conf" ]]; then
        red "Model not found: $key"
        echo "Run '$0 models' to see available models."
        exit 1
    fi

    echo "$key" > "$ACTIVE_MODEL_FILE"
    green "Active model set to: $key"

    local gguf
    gguf=$(get_model_field "$key" "gguf")
    echo "  GGUF: $gguf"

    # Regenerate systemd unit if it exists
    if [[ -f "$UNIT_FILE" ]]; then
        create_unit_file "$gguf"
        systemctl --user daemon-reload
        green "Updated systemd unit file"

        # Check if service is running
        if systemctl --user is-active "$SERVICE_NAME.service" &>/dev/null; then
            echo ""
            read -rp "Service is running. Restart with new model? [y/N] " answer
            if [[ "${answer,,}" == "y" ]]; then
                systemctl --user restart "$SERVICE_NAME.service"
                green "Service restarted"
            else
                yellow "Remember to restart the service to use the new model."
            fi
        fi
    fi
}

# ---- Setup Commands (new machine provisioning) ------------------------------

_run_on_both() {
    local desc="$1"; shift
    bold "[$desc] Local node:"
    eval "$@"
    local rc=$?
    echo ""
    bold "[$desc] Remote node ($DIST_REMOTE_NODE):"
    ssh "$DIST_REMOTE_NODE" "$@"
    local rc2=$?
    [[ $rc -eq 0 && $rc2 -eq 0 ]]
}

_setup_help() {
    bold "Usage: $0 setup <subcommand>"
    echo ""
    echo "Before running setup, configure node IPs:"
    echo "  $0 set head <lan-ip>        # head node LAN IP (e.g. 192.168.0.231)"
    echo "  $0 set worker <lan-ip>      # worker node LAN IP (e.g. 192.168.0.232)"
    echo ""
    echo "Subcommands (run in order):"
    echo "  deps          Install all required packages on both nodes (sudo)"
    echo "  thunderbolt   Configure USB4 v2 networking on both nodes (sudo)"
    echo "  rdma          Configure Soft-RoCE RDMA on both nodes (sudo)"
    echo "  build         Clone + build llama.cpp on both nodes"
    echo "  verify        Verify entire setup (connectivity, binaries, modules)"
    echo ""
    echo "Thunderbolt IPs default to 10.0.0.1 (head) and 10.0.0.2 (worker)."
    echo "Override with: $0 set head-tb <ip> / $0 set worker-tb <ip>"
}

cmd_setup() {
    local sub="${1:-help}"
    shift || true
    case "$sub" in
        deps)        cmd_setup_deps "$@" ;;
        thunderbolt) cmd_setup_thunderbolt "$@" ;;
        rdma)        cmd_setup_rdma "$@" ;;
        build)       cmd_setup_build "$@" ;;
        verify)      cmd_setup_verify "$@" ;;
        help|*)      _setup_help ;;
    esac
}

cmd_setup_deps() {
    _require_worker_ip
    bold "Installing required packages on both nodes..."
    echo ""

    local pkgs="cmake gcc-c++ git vulkan-headers vulkan-loader-devel glslang glslc iperf3 jq python3-pip rdma-core libibverbs-utils"

    bold "Local node:"
    sudo dnf install -y $pkgs
    pip install --user huggingface_hub 2>/dev/null || pip install --user huggingface_hub --break-system-packages
    echo ""

    bold "Remote node ($DIST_REMOTE_NODE):"
    ssh "$DIST_REMOTE_NODE" "sudo dnf install -y $pkgs"
    ssh "$DIST_REMOTE_NODE" "pip install --user huggingface_hub 2>/dev/null || pip install --user huggingface_hub --break-system-packages"

    echo ""
    green "All packages installed on both nodes"
}

cmd_setup_thunderbolt() {
    _require_worker_ip
    bold "Configuring USB4 v2 Thunderbolt networking..."
    echo ""

    # Use configured thunderbolt IPs
    local local_ip="$HEAD_TB"
    local remote_ip="$WORKER_TB"
    bold "Head node thunderbolt IP: $local_ip"
    bold "Worker node thunderbolt IP: $remote_ip"
    echo ""

    # 1. Load kernel modules
    bold "Loading kernel modules..."
    sudo modprobe thunderbolt_net || true
    sudo modprobe rdma_rxe || true

    # 2. Persist modules
    bold "Creating /etc/modules-load.d/thunderbolt-rdma.conf..."
    sudo tee /etc/modules-load.d/thunderbolt-rdma.conf > /dev/null <<'MODEOF'
thunderbolt_net
rdma_rxe
MODEOF
    green "  Module autoload configured"

    # 3. Wait for thunderbolt0 to appear
    echo -n "Waiting for thunderbolt0..."
    for i in $(seq 1 10); do
        if ip link show thunderbolt0 &>/dev/null; then
            echo " found"
            break
        fi
        echo -n "."
        sleep 1
    done
    if ! ip link show thunderbolt0 &>/dev/null; then
        echo ""
        red "thunderbolt0 not found. Is the USB4 v2 cable connected to the REAR port?"
        return 1
    fi

    # 4. Create NetworkManager profile for thunderbolt0
    bold "Configuring thunderbolt0 with IP $local_ip/24..."
    sudo nmcli connection delete thunderbolt-link 2>/dev/null || true
    sudo nmcli connection add type ethernet ifname thunderbolt0 \
        con-name thunderbolt-link \
        ipv4.method manual \
        ipv4.addresses "$local_ip/24" \
        connection.zone trusted
    sudo nmcli connection up thunderbolt-link || true
    green "  thunderbolt0 = $local_ip/24"

    # 5. Check for thunderbolt1 (second cable)
    if ip link show thunderbolt1 &>/dev/null; then
        local local_ip2 remote_ip2
        if [[ "$local_ip" == "10.0.0.1" ]]; then
            local_ip2="10.0.1.1"
        else
            local_ip2="10.0.1.2"
        fi
        bold "Found thunderbolt1, configuring with IP $local_ip2/24..."
        sudo nmcli connection delete thunderbolt-link2 2>/dev/null || true
        sudo nmcli connection add type ethernet ifname thunderbolt1 \
            con-name thunderbolt-link2 \
            ipv4.method manual \
            ipv4.addresses "$local_ip2/24" \
            connection.zone trusted
        sudo nmcli connection up thunderbolt-link2 || true
        green "  thunderbolt1 = $local_ip2/24"
    fi

    # 6. Firewall
    if command -v firewall-cmd &>/dev/null; then
        bold "Configuring firewall..."
        sudo firewall-cmd --zone=trusted --add-interface=thunderbolt0 --permanent 2>/dev/null || true
        if ip link show thunderbolt1 &>/dev/null; then
            sudo firewall-cmd --zone=trusted --add-interface=thunderbolt1 --permanent 2>/dev/null || true
        fi
        sudo firewall-cmd --reload 2>/dev/null || true
        green "  Firewall: thunderbolt interfaces in trusted zone"
    fi

    # 7. TCP tuning
    bold "Applying TCP tuning..."
    sudo tee /etc/sysctl.d/99-thunderbolt.conf > /dev/null <<'SYSEOF'
net.core.rmem_max = 16777216
net.core.wmem_max = 16777216
net.core.rmem_default = 1048576
net.core.wmem_default = 1048576
net.ipv4.tcp_rmem = 4096 1048576 16777216
net.ipv4.tcp_wmem = 4096 1048576 16777216
net.ipv4.tcp_congestion_control = bbr
net.ipv4.tcp_window_scaling = 1
net.core.netdev_max_backlog = 5000
SYSEOF
    sudo sysctl -p /etc/sysctl.d/99-thunderbolt.conf &>/dev/null
    green "  TCP tuning applied"

    echo ""
    green "Local thunderbolt networking configured!"
    echo ""

    # 8. Configure remote node
    bold "Configuring remote node ($DIST_REMOTE_NODE)..."
    ssh "$DIST_REMOTE_NODE" bash <<REMOTE_EOF
set -e
sudo modprobe thunderbolt_net || true
sudo modprobe rdma_rxe || true
sudo tee /etc/modules-load.d/thunderbolt-rdma.conf > /dev/null <<'MODEOF'
thunderbolt_net
rdma_rxe
MODEOF
# Wait for thunderbolt0
for i in \$(seq 1 10); do
    ip link show thunderbolt0 &>/dev/null && break
    sleep 1
done
sudo nmcli connection delete thunderbolt-link 2>/dev/null || true
sudo nmcli connection add type ethernet ifname thunderbolt0 \
    con-name thunderbolt-link \
    ipv4.method manual \
    ipv4.addresses "$remote_ip/24" \
    connection.zone trusted
sudo nmcli connection up thunderbolt-link || true
if command -v firewall-cmd &>/dev/null; then
    sudo firewall-cmd --zone=trusted --add-interface=thunderbolt0 --permanent 2>/dev/null || true
    sudo firewall-cmd --reload 2>/dev/null || true
fi
sudo tee /etc/sysctl.d/99-thunderbolt.conf > /dev/null <<'SYSEOF'
net.core.rmem_max = 16777216
net.core.wmem_max = 16777216
net.core.rmem_default = 1048576
net.core.wmem_default = 1048576
net.ipv4.tcp_rmem = 4096 1048576 16777216
net.ipv4.tcp_wmem = 4096 1048576 16777216
net.ipv4.tcp_congestion_control = bbr
net.ipv4.tcp_window_scaling = 1
net.core.netdev_max_backlog = 5000
SYSEOF
sudo sysctl -p /etc/sysctl.d/99-thunderbolt.conf &>/dev/null
REMOTE_EOF
    green "Remote node thunderbolt configured ($remote_ip)"

    echo ""
    yellow "Recommendation: Reboot both nodes for modules to load cleanly at boot."
    echo -n "Reboot both nodes now? [y/N] "
    read -r reboot_answer
    if [[ "$reboot_answer" =~ ^[Yy] ]]; then
        bold "Rebooting remote node..."
        ssh "$DIST_REMOTE_NODE" "sudo reboot" &>/dev/null || true
        bold "Rebooting local node in 3 seconds..."
        sleep 3
        sudo reboot
    fi
}

cmd_setup_rdma() {
    _require_worker_ip
    bold "Configuring Soft-RoCE RDMA on both nodes..."
    echo ""

    # Local
    bold "Local node:"
    sudo modprobe rdma_rxe
    if ! rdma link show rxe0 &>/dev/null; then
        sudo rdma link add rxe0 type rxe netdev thunderbolt0
        green "  Created RDMA device rxe0 on thunderbolt0"
    else
        green "  RDMA device rxe0 already exists"
    fi

    # Create systemd service to persist
    bold "Creating /etc/systemd/system/rdma-rxe.service..."
    sudo tee /etc/systemd/system/rdma-rxe.service > /dev/null <<'RDMAEOF'
[Unit]
Description=Soft-RoCE (RXE) RDMA over Thunderbolt
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/usr/sbin/rdma link add rxe0 type rxe netdev thunderbolt0

[Install]
WantedBy=multi-user.target
RDMAEOF
    sudo systemctl daemon-reload
    sudo systemctl enable rdma-rxe.service
    green "  rdma-rxe.service enabled"
    echo ""

    # Remote
    bold "Remote node ($DIST_REMOTE_NODE):"
    ssh "$DIST_REMOTE_NODE" bash <<'REMOTE_EOF'
set -e
sudo modprobe rdma_rxe
if ! rdma link show rxe0 &>/dev/null; then
    sudo rdma link add rxe0 type rxe netdev thunderbolt0
    echo "  Created RDMA device rxe0 on thunderbolt0"
else
    echo "  RDMA device rxe0 already exists"
fi
sudo tee /etc/systemd/system/rdma-rxe.service > /dev/null <<'SVCEOF'
[Unit]
Description=Soft-RoCE (RXE) RDMA over Thunderbolt
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/usr/sbin/rdma link add rxe0 type rxe netdev thunderbolt0

[Install]
WantedBy=multi-user.target
SVCEOF
sudo systemctl daemon-reload
sudo systemctl enable rdma-rxe.service
REMOTE_EOF
    green "  Remote rdma-rxe.service enabled"

    echo ""
    green "Soft-RoCE RDMA configured on both nodes"
    dim "Note: RDMA is future-proofing — llama.cpp RPC currently uses TCP only"
}

cmd_setup_build() {
    _require_worker_ip
    bold "Building llama.cpp on both nodes..."
    echo ""

    local llama_dir="$REAL_HOME/llama.cpp"

    # Local build
    bold "Local node:"
    if [[ ! -d "$llama_dir" ]]; then
        echo "  Cloning llama.cpp..."
        git clone https://github.com/ggml-org/llama.cpp.git "$llama_dir"
    else
        echo "  Updating llama.cpp..."
        git -C "$llama_dir" pull
    fi
    echo "  Building with Vulkan + RPC..."
    cmake -S "$llama_dir" -B "$llama_dir/build" \
        -DGGML_VULKAN=ON -DGGML_RPC=ON -DCMAKE_BUILD_TYPE=Release
    cmake --build "$llama_dir/build" -j"$(nproc)"
    if [[ -x "$llama_dir/build/bin/llama-server" ]] && [[ -x "$llama_dir/build/bin/rpc-server" ]]; then
        green "  Local build successful"
    else
        red "  Local build failed — binaries not found"
        return 1
    fi
    echo ""

    # Remote build
    bold "Remote node ($DIST_REMOTE_NODE):"
    ssh "$DIST_REMOTE_NODE" bash <<'REMOTE_EOF'
set -e
LLAMA_DIR="$HOME/llama.cpp"
if [[ ! -d "$LLAMA_DIR" ]]; then
    echo "  Cloning llama.cpp..."
    git clone https://github.com/ggml-org/llama.cpp.git "$LLAMA_DIR"
else
    echo "  Updating llama.cpp..."
    git -C "$LLAMA_DIR" pull
fi
echo "  Building with Vulkan + RPC..."
cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" \
    -DGGML_VULKAN=ON -DGGML_RPC=ON -DCMAKE_BUILD_TYPE=Release
cmake --build "$LLAMA_DIR/build" -j"$(nproc)"
if [[ -x "$LLAMA_DIR/build/bin/llama-server" ]] && [[ -x "$LLAMA_DIR/build/bin/rpc-server" ]]; then
    echo "  Remote build successful"
else
    echo "  Remote build FAILED"
    exit 1
fi
REMOTE_EOF
    green "  Remote build successful"

    echo ""
    green "llama.cpp built on both nodes"
}

cmd_setup_verify() {
    _require_worker_ip
    bold "Verifying distributed inference setup..."
    echo ""
    local pass=0 fail=0

    # 1. Kernel modules
    bold "Kernel modules:"
    if lsmod | grep -q thunderbolt_net; then
        green "  PASS: thunderbolt_net loaded"
        ((pass++))
    else
        red "  FAIL: thunderbolt_net not loaded"
        ((fail++))
    fi
    if lsmod | grep -q rdma_rxe; then
        green "  PASS: rdma_rxe loaded"
        ((pass++))
    else
        yellow "  WARN: rdma_rxe not loaded (optional — future-proofing)"
    fi
    echo ""

    # 2. Thunderbolt interface
    bold "Thunderbolt interface:"
    local local_tb_ip
    local_tb_ip=$(ip -4 addr show thunderbolt0 2>/dev/null | grep -oP 'inet \K[\d.]+') || true
    if [[ -n "$local_tb_ip" ]]; then
        green "  PASS: thunderbolt0 has IP $local_tb_ip"
        ((pass++))
    else
        red "  FAIL: thunderbolt0 has no IP or does not exist"
        ((fail++))
    fi
    echo ""

    # 3. Ping remote
    bold "Remote connectivity:"
    if ping -c 2 -W 2 "$DIST_REMOTE_TB" &>/dev/null; then
        green "  PASS: can ping $DIST_REMOTE_TB (max2)"
        ((pass++))
    else
        red "  FAIL: cannot ping $DIST_REMOTE_TB"
        ((fail++))
    fi

    # SSH to remote
    if ssh -o ConnectTimeout=5 "$DIST_REMOTE_NODE" "echo ok" &>/dev/null; then
        green "  PASS: SSH to $DIST_REMOTE_NODE works"
        ((pass++))
    else
        red "  FAIL: cannot SSH to $DIST_REMOTE_NODE"
        ((fail++))
    fi
    echo ""

    # 4. Bandwidth test
    bold "Bandwidth (iperf3):"
    if command -v iperf3 &>/dev/null; then
        ssh "$DIST_REMOTE_NODE" "iperf3 -s -1 -D -p 5198" 2>/dev/null
        sleep 1
        local result
        result=$(iperf3 -c "$DIST_REMOTE_TB" -p 5198 -t 3 2>&1) || true
        local bw
        bw=$(echo "$result" | grep "sender" | awk '{for(i=1;i<=NF;i++) if($i ~ /bits\/sec/) print $(i-1), $i}')
        if [[ -n "$bw" ]]; then
            green "  Bandwidth: $bw"
            ((pass++))
        else
            yellow "  WARN: bandwidth test failed (non-critical)"
        fi
    else
        yellow "  SKIP: iperf3 not installed"
    fi
    echo ""

    # 5. llama.cpp binaries
    bold "llama.cpp binaries:"
    local build_dir
    build_dir=$(_dist_build_dir "${DIST_BACKEND:-vulkan}")
    if [[ -x "$build_dir/bin/llama-server" ]] && [[ -x "$build_dir/bin/rpc-server" ]]; then
        green "  PASS: local llama-server and rpc-server found ($build_dir/bin/)"
        ((pass++))
    else
        red "  FAIL: local binaries not found in $build_dir/bin/"
        ((fail++))
    fi

    local remote_check
    remote_check=$(ssh "$DIST_REMOTE_NODE" "test -x ~/llama.cpp/build/bin/llama-server && test -x ~/llama.cpp/build/bin/rpc-server && echo ok" 2>/dev/null) || true
    if [[ "$remote_check" == "ok" ]]; then
        green "  PASS: remote llama-server and rpc-server found"
        ((pass++))
    else
        red "  FAIL: remote binaries not found"
        ((fail++))
    fi
    echo ""

    # 6. RDMA devices
    bold "RDMA devices:"
    if rdma link show rxe0 &>/dev/null; then
        green "  PASS: local rxe0 device exists"
        ((pass++))
    else
        yellow "  WARN: local rxe0 not found (optional)"
    fi
    local remote_rdma
    remote_rdma=$(ssh "$DIST_REMOTE_NODE" "rdma link show rxe0 2>/dev/null && echo ok" 2>/dev/null) || true
    if [[ "$remote_rdma" == *"ok"* ]]; then
        green "  PASS: remote rxe0 device exists"
        ((pass++))
    else
        yellow "  WARN: remote rxe0 not found (optional)"
    fi
    echo ""

    # Summary
    bold "========================================="
    green "Passed: $pass"
    if [[ $fail -gt 0 ]]; then
        red "Failed: $fail"
        echo ""
        yellow "Fix the failures above before proceeding."
        return 1
    else
        green "All critical checks passed!"
        echo ""
        echo "Next steps:"
        echo "  $0 download              # download a model"
        echo "  $0 select <model-key>    # set active model"
        echo "  $0 dist-install          # create services on both nodes"
        echo "  $0 dist-start            # start distributed inference"
    fi
}

cmd_install() {
    require_no_sudo install
    migrate_existing_models

    local gguf
    gguf=$(get_active_gguf) || true

    if [[ -z "$gguf" ]]; then
        red "No active model selected."
        echo "Run '$0 models' to see models, or '$0 download' to get one."
        exit 1
    fi

    local active
    active=$(get_active_model)
    bold "Installing $SERVICE_NAME systemd user service..."
    echo "  Model: $active"
    echo "  GGUF: $gguf"
    echo ""

    # Create the unit file
    create_unit_file "$gguf"
    green "Created $UNIT_FILE"

    # Enable linger so the user service survives logout
    if ! loginctl show-user "$USER" --property=Linger 2>/dev/null | grep -q "Linger=yes"; then
        loginctl enable-linger "$USER"
        green "Enabled linger for $USER"
    else
        echo "Linger already enabled for $USER"
    fi

    # Reload and enable
    systemctl --user daemon-reload
    systemctl --user enable "$SERVICE_NAME.service"
    green "Service enabled"

    echo ""
    bold "Done! You can now run:"
    echo "  $0 start    - start the server"
    echo "  $0 status   - check status"
    echo "  $0 logs     - view logs"
}

cmd_uninstall() {
    require_no_sudo uninstall
    bold "Uninstalling $SERVICE_NAME systemd user service..."

    # Stop if running
    if systemctl --user is-active "$SERVICE_NAME.service" &>/dev/null; then
        systemctl --user stop "$SERVICE_NAME.service"
        green "Stopped service"
    fi

    # Disable
    if systemctl --user is-enabled "$SERVICE_NAME.service" &>/dev/null; then
        systemctl --user disable "$SERVICE_NAME.service"
        green "Disabled service"
    fi

    # Remove unit file
    if [[ -f "$UNIT_FILE" ]]; then
        rm "$UNIT_FILE"
        systemctl --user daemon-reload
        green "Removed $UNIT_FILE"
    fi

    # Optionally disable linger
    echo ""
    read -rp "Disable linger for $USER? (this affects ALL user services) [y/N] " answer
    if [[ "${answer,,}" == "y" ]]; then
        loginctl disable-linger "$USER"
        green "Disabled linger for $USER"
    fi

    green "Uninstall complete"
}

cmd_start() {
    require_no_sudo start
    bold "Starting $SERVICE_NAME..."
    systemctl --user start "$SERVICE_NAME.service"
    green "Started"
    echo "Check health: curl http://localhost:$PORT/health"
}

cmd_stop() {
    require_no_sudo stop
    bold "Stopping $SERVICE_NAME..."
    systemctl --user stop "$SERVICE_NAME.service"
    green "Stopped"
}

cmd_restart() {
    require_no_sudo restart
    bold "Restarting $SERVICE_NAME..."
    systemctl --user restart "$SERVICE_NAME.service"
    green "Restarted"
}

cmd_gpu() {
    local watch_mode=false
    [[ "${1:-}" == "-w" || "${1:-}" == "--watch" ]] && watch_mode=true

    # Auto-detect GPU: find the DRM card that has VRAM
    local gpu_dev=""
    for _card in /sys/class/drm/card[0-9]*/device; do
        if [[ -f "$_card/mem_info_vram_total" ]]; then
            gpu_dev="$_card"
            break
        fi
    done
    if [[ -z "$gpu_dev" ]]; then
        red "No GPU with VRAM found"; return 1
    fi
    # Auto-detect hwmon for amdgpu
    local hwmon=""
    for _hw in /sys/class/hwmon/hwmon[0-9]*; do
        if [[ "$(cat "$_hw/name" 2>/dev/null)" == "amdgpu" ]]; then
            hwmon="$_hw"
            break
        fi
    done
    if [[ -z "$hwmon" ]]; then
        red "No amdgpu hwmon found"; return 1
    fi

    _gpu_snapshot() {
        local vram_used vram_total gtt_used gtt_total gpu_load power_uw freq_hz temp_mc
        vram_used=$(cat "$gpu_dev/mem_info_vram_used" 2>/dev/null || echo 0)
        vram_total=$(cat "$gpu_dev/mem_info_vram_total" 2>/dev/null || echo 1)
        gtt_used=$(cat "$gpu_dev/mem_info_gtt_used" 2>/dev/null || echo 0)
        gtt_total=$(cat "$gpu_dev/mem_info_gtt_total" 2>/dev/null || echo 0)
        gpu_load=$(cat "$gpu_dev/gpu_busy_percent" 2>/dev/null || echo 0)
        power_uw=$(cat "$hwmon/power1_average" 2>/dev/null || echo 0)
        freq_hz=$(cat "$hwmon/freq1_input" 2>/dev/null || echo 0)
        temp_mc=$(cat "$hwmon/temp1_input" 2>/dev/null || echo 0)

        # Read CPU RAM from /proc/meminfo
        local mem_total mem_avail mem_used
        mem_total=$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)
        mem_avail=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
        mem_used=$((mem_total - mem_avail))

        awk -v vu="$vram_used" -v vt="$vram_total" \
            -v gu="$gtt_used" -v gt="$gtt_total" \
            -v gpu_load="$gpu_load" -v pw="$power_uw" \
            -v freq="$freq_hz" -v temp="$temp_mc" \
            -v mu="$mem_used" -v mt="$mem_total" '
        BEGIN {
            # GPU memory: use VRAM + GTT combined if GTT is present
            if (gt > 0) {
                gpu_used_b = vu + gu
                gpu_total_b = vt + gt
            } else {
                gpu_used_b = vu
                gpu_total_b = vt
            }
            gpu_used_gb  = gpu_used_b  / 1073741824
            gpu_total_gb = gpu_total_b / 1073741824
            gpu_p = (gpu_total_b > 0) ? int(gpu_used_b / gpu_total_b * 100) : 0

            pw_w   = pw   / 1000000
            freq_g = freq / 1000000000
            temp_c = temp / 1000

            gpu_all_kb = (vt + gt) / 1024
            os_total = mt - gpu_all_kb
            if (os_total < 1048576) os_total = mt
            os_avail = mt - mu
            os_used = os_total - os_avail
            if (os_used < 0) os_used = 0
            if (os_used > os_total) os_used = os_total
            ram_used_gb  = os_used / 1048576
            ram_total_gb = os_total / 1048576
            ram_p = (os_total > 0) ? int(os_used / os_total * 100) : 0

            bar_len = 30

            gpu_bar_n = int(gpu_p * bar_len / 100)
            gbar = ""; for(i=0;i<gpu_bar_n;i++) gbar = gbar "█"
            for(i=gpu_bar_n;i<bar_len;i++) gbar = gbar "░"

            load_bar_n = int(gpu_load * bar_len / 100)
            lbar = ""; for(i=0;i<load_bar_n;i++) lbar = lbar "█"
            for(i=load_bar_n;i<bar_len;i++) lbar = lbar "░"

            ram_bar_n = int(ram_p * bar_len / 100)
            rbar = ""; for(i=0;i<ram_bar_n;i++) rbar = rbar "█"
            for(i=ram_bar_n;i<bar_len;i++) rbar = rbar "░"

            printf "  VRAM  [%s] %5.1f / %.0f GB (%d%%)\n", gbar, gpu_used_gb, gpu_total_gb, gpu_p
            printf "  RAM   [%s] %5.1f / %.0f GB (%d%%)\n", rbar, ram_used_gb, ram_total_gb, ram_p
            printf "  Load  [%s] %d%%\n",                   lbar, gpu_load
            printf "  Power  %6.1f W\n",  pw_w
            printf "  Clock  %6.2f GHz\n", freq_g
            printf "  Temp   %6.1f °C\n",  temp_c
        }'
    }

    if $watch_mode; then
        echo "GPU monitor — Ctrl-C to exit"
        echo ""
        while true; do
            tput cup 2 0 2>/dev/null || true
            echo "  $(date '+%H:%M:%S')"
            _gpu_snapshot
            sleep 1
        done
    else
        echo ""
        _gpu_snapshot
        echo ""
    fi
}

cmd_bench() {
    require_no_sudo bench

    local quick=false
    [[ "${1:-}" == "-q" || "${1:-}" == "--quick" ]] && quick=true

    local gguf
    gguf=$(get_active_gguf) || true
    if [[ -z "$gguf" ]]; then
        red "No active model. Run '$0 download' first."
        exit 1
    fi
    local active
    active=$(get_active_model)

    if $quick; then
        bold "LLM Quick Benchmark (~2 min)"
    else
        bold "LLM Intensive Benchmark (~20 min)"
    fi
    echo "  Model:     $active"
    echo "  GGUF:      $gguf"
    echo "  Container: $CONTAINER"
    echo ""

    # Stop server to free VRAM
    local was_running=false
    if systemctl --user is-active "$SERVICE_NAME.service" &>/dev/null; then
        was_running=true
        bold "Stopping server to free VRAM..."
        systemctl --user stop "$SERVICE_NAME.service"
        sleep 3
    fi

    local results_dir="$HOME/llm-benchmarks"
    mkdir -p "$results_dir"
    local ts results_file mode_tag
    ts=$(date +%Y%m%d-%H%M%S)
    $quick && mode_tag="quick" || mode_tag="full"
    results_file="$results_dir/${active}-${mode_tag}-${ts}.md"

    if $quick; then
        bold "Running benchmark — tests:"
        echo "  Prompt processing : 512 / 2048 tokens"
        echo "  Token generation  : 512 tokens"
        echo "  End-to-end        : 512→512 tokens"
        echo "  Repetitions       : 3 each"
        echo ""
        {
            echo "# LLM Quick Benchmark: $active"
            echo "Date: $(date)"
            echo "GGUF: $gguf"
            echo ""
            $(_dist_build_dir "${DIST_BACKEND:-vulkan}")/bin/llama-bench \
                -m "$gguf" \
                -ngl "$GPU_LAYERS" \
                -fa "$FLASH_ATTENTION" \
                -mmp 0 \
                -p "512,2048" \
                -n 512 \
                -pg "512,512" \
                -r 3 \
                --progress \
                -o md 2>&1
        } | tee "$results_file"
    else
        bold "Running benchmark — tests:"
        echo "  Prompt processing : 128 / 512 / 2048 / 8192 / 32768 tokens"
        echo "  Token generation  : 128 / 512 tokens"
        echo "  End-to-end        : 512→512 / 4096→512 / 16384→512 tokens"
        echo "  Repetitions       : 5 each"
        echo ""
        {
            echo "# LLM Intensive Benchmark: $active"
            echo "Date: $(date)"
            echo "GGUF: $gguf"
            echo ""
            $(_dist_build_dir "${DIST_BACKEND:-vulkan}")/bin/llama-bench \
                -m "$gguf" \
                -ngl "$GPU_LAYERS" \
                -fa "$FLASH_ATTENTION" \
                -mmp 0 \
                -p "128,512,2048,8192,32768" \
                -n "128,512" \
                -pg "512,512" \
                -pg "4096,512" \
                -pg "16384,512" \
                -r 5 \
                --progress \
                -o md 2>&1
        } | tee "$results_file"
    fi

    echo ""
    green "Results saved: $results_file"
    echo ""

    if $was_running; then
        bold "Restarting server..."
        systemctl --user start "$SERVICE_NAME.service"
        green "Server restarted"
    fi
}

cmd_set() {
    require_no_sudo set
    if [[ $# -lt 2 ]]; then
        bold "Usage: $0 set <setting> <value>"
        echo ""
        echo "Settings:"
        printf "  %-18s  %s\n" "head <ip>"        "Head node LAN IP for SSH (current: ${HEAD_IP:-not set})"
        printf "  %-18s  %s\n" "worker <ip>"      "Worker node LAN IP for SSH (current: ${WORKER_IP:-not set})"
        printf "  %-18s  %s\n" "head-tb <ip>"     "Head node Thunderbolt IP (current: ${HEAD_TB})"
        printf "  %-18s  %s\n" "worker-tb <ip>"   "Worker node Thunderbolt IP (current: ${WORKER_TB})"
        printf "  %-18s  %s\n" "parallel <n>"     "Parallel request slots — each gets its own context window (current: $PARALLEL_SLOTS)"
        printf "  %-18s  %s\n" "context <tokens>"  "Context size per slot in tokens (current: $CONTEXT_SIZE)"
        printf "  %-18s  %s\n" "port <n>"          "Server port (current: $PORT)"
        printf "  %-18s  %s\n" "kv-cache <type>"   "KV cache quantization: q4_0, q8_0, f16/off (current: ${KV_CACHE_TYPE:-f16})"
        printf "  %-18s  %s\n" "container <name>"  "Toolbox container (current: $CONTAINER)"
        echo ""
        echo "Examples:"
        echo "  $0 set parallel 4                  # 4 users, each with ${CONTEXT_SIZE} tokens"
        echo "  $0 set context 524288              # 512k context per slot"
        echo "  $0 set parallel 4 context 262144   # not valid — run set twice or see below"
        exit 1
    fi

    local setting="$1" value="$2"
    case "$setting" in
        head)
            [[ "$value" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]] || { red "head must be a valid IP address"; exit 1; }
            save_conf "HEAD_IP" "$value"
            HEAD_IP="$value"
            green "Head node LAN IP → $value"
            ;;
        worker)
            [[ "$value" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]] || { red "worker must be a valid IP address"; exit 1; }
            save_conf "WORKER_IP" "$value"
            WORKER_IP="$value"
            DIST_REMOTE_NODE="$value"
            green "Worker node LAN IP → $value"
            ;;
        head-tb)
            [[ "$value" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]] || { red "head-tb must be a valid IP address"; exit 1; }
            save_conf "HEAD_TB" "$value"
            HEAD_TB="$value"
            green "Head node Thunderbolt IP → $value"
            ;;
        worker-tb)
            [[ "$value" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]] || { red "worker-tb must be a valid IP address"; exit 1; }
            save_conf "WORKER_TB" "$value"
            WORKER_TB="$value"
            DIST_REMOTE_TB="$value"
            green "Worker node Thunderbolt IP → $value"
            ;;
        parallel|np)
            [[ "$value" =~ ^[0-9]+$ ]] && [[ "$value" -ge 1 ]] || { red "parallel must be a positive integer"; exit 1; }
            save_conf "PARALLEL_SLOTS" "$value"
            local ctx_per_slot=$(( CONTEXT_SIZE / value ))
            green "Parallel slots → $value"
            echo "  Each slot gets: ~${ctx_per_slot} tokens (${CONTEXT_SIZE} model max ÷ ${value} slots)"
            ;;
        context|ctx)
            [[ "$value" =~ ^[0-9]+$ ]] && [[ "$value" -ge 512 ]] || { red "context must be a positive integer ≥ 512"; exit 1; }
            save_conf "CONTEXT_SIZE" "$value"
            green "Context size → $value tokens"
            ;;
        rope-scale|rope|yarn)
            [[ "$value" =~ ^[0-9]+(\.[0-9]+)?$ ]] || { red "rope-scale must be a number (e.g. 1.0, 2.0, 4.0)"; exit 1; }
            save_conf "ROPE_SCALE" "$value"
            if [[ "$value" == "1" || "$value" == "1.0" ]]; then
                save_conf "CONTEXT_SIZE" "262144"
                green "RoPE scale → disabled (native 256k context restored)"
            else
                local new_ctx
                new_ctx=$(python3 -c "print(int($value * 262144))")
                save_conf "CONTEXT_SIZE" "$new_ctx"
                green "RoPE scale → ${value}x YaRN  (context: ${new_ctx} tokens)"
            fi
            ;;
        port)
            [[ "$value" =~ ^[0-9]+$ ]] || { red "port must be a positive integer"; exit 1; }
            save_conf "PORT" "$value"
            green "Port → $value"
            ;;
        kv-cache|kv|ctk)
            local valid_types="q4_0 q8_0 f16"
            if [[ "$value" == "off" || "$value" == "f16" || "$value" == "fp16" ]]; then
                save_conf "KV_CACHE_TYPE" ""
                green "KV cache type → f16 (default, no quantization)"
            elif [[ " $valid_types " == *" $value "* ]]; then
                save_conf "KV_CACHE_TYPE" "$value"
                green "KV cache type → $value"
            else
                red "Invalid KV cache type: $value"
                echo "Valid types: q4_0, q8_0, f16 (or off)"
                exit 1
            fi
            ;;
        container)
            save_conf "CONTAINER" "$value"
            green "Container → $value"
            ;;
        *)
            red "Unknown setting: $setting"
            echo "Valid settings: head, worker, head-tb, worker-tb, parallel, context, port, kv-cache, container"
            exit 1
            ;;
    esac
    echo ""
    if [[ -f "$UNIT_FILE" ]]; then
        bold "Reinstalling service with updated settings..."
        cmd_install
        if systemctl --user is-active "$SERVICE_NAME.service" &>/dev/null; then
            systemctl --user restart "$SERVICE_NAME.service"
            green "Service restarted"
        fi
    else
        echo "Run '$0 install && $0 start' to apply."
    fi
}

cmd_status() {
    require_no_sudo status
    migrate_existing_models
    local active
    active=$(get_active_model)
    if [[ -n "$active" ]]; then
        local repo quant
        repo=$(get_model_field "$active" "repo")
        quant=$(get_model_field "$active" "quant")
        bold "Active model: $active ($repo $quant)"
    else
        yellow "No active model selected"
    fi
    echo ""
    local rope_info=""
    [[ "$ROPE_SCALE" != "1" && "$ROPE_SCALE" != "1.0" ]] && rope_info="  rope=${ROPE_SCALE}x"
    local kv_info=""
    [[ -n "$KV_CACHE_TYPE" ]] && kv_info="  kv=$KV_CACHE_TYPE"
    echo "Server settings:  parallel=$PARALLEL_SLOTS  context=$CONTEXT_SIZE  port=$PORT  container=$CONTAINER${rope_info}${kv_info}"
    echo ""
    systemctl --user status "$SERVICE_NAME.service" --no-pager || true
}

cmd_logs() {
    require_no_sudo logs
    if [[ "${1:-}" == "-f" || "${1:-}" == "--follow" ]]; then
        journalctl --user -u "$SERVICE_NAME.service" -f
    else
        journalctl --user -u "$SERVICE_NAME.service" --no-pager -n 50
    fi
}

# ---- Distributed Mode (2-node RPC) -----------------------------------------

DIST_RPC_SERVICE="rpc-server"
DIST_LLM_SERVICE="llm-distributed"
DIST_BACKEND="${DIST_BACKEND:-vulkan}"   # "vulkan" or "hip"

# Distributed node IPs — loaded from server.conf (set via: llm-server.sh set head/worker)
# HEAD_IP:   LAN IP of head node (this machine), used by worker to reach us
# WORKER_IP: LAN IP of worker node, used for SSH
# HEAD_TB:   Thunderbolt IP of head node (default: 10.0.0.1)
# WORKER_TB: Thunderbolt IP of worker node (default: 10.0.0.2)
HEAD_IP="${HEAD_IP:-}"
WORKER_IP="${WORKER_IP:-}"
HEAD_TB="${HEAD_TB:-10.0.0.1}"
WORKER_TB="${WORKER_TB:-10.0.0.2}"

# Legacy compatibility
DIST_REMOTE_NODE="$WORKER_IP"
DIST_REMOTE_TB="$WORKER_TB"

# Load saved backend from config
if [[ -f "$CONFIG_DIR/dist-backend" ]]; then
    DIST_BACKEND=$(< "$CONFIG_DIR/dist-backend")
fi

# Guard: require worker IP to be configured for distributed/setup commands
_require_worker_ip() {
    if [[ -z "$WORKER_IP" ]]; then
        red "Worker node IP not configured."
        echo ""
        echo "Set the LAN IPs of both machines first:"
        echo "  $0 set head <head-lan-ip>       # e.g. $0 set head 192.168.0.231"
        echo "  $0 set worker <worker-lan-ip>   # e.g. $0 set worker 192.168.0.232"
        echo ""
        echo "Optionally set thunderbolt IPs (defaults: 10.0.0.1 / 10.0.0.2):"
        echo "  $0 set head-tb <ip>             # default: 10.0.0.1"
        echo "  $0 set worker-tb <ip>            # default: 10.0.0.2"
        exit 1
    fi
    # Keep legacy vars in sync
    DIST_REMOTE_NODE="$WORKER_IP"
    DIST_REMOTE_TB="$WORKER_TB"
}

_dist_build_dir() {
    case "$1" in
        hip)    echo "$REAL_HOME/llama.cpp/build-hip" ;;
        vulkan) echo "$REAL_HOME/llama.cpp/build" ;;
        *)      red "Unknown backend: $1"; return 1 ;;
    esac
}

_dist_backend_env_rpc() {
    # Environment for RPC server (needs GPU access)
    case "$1" in
        hip)
            echo "Environment=ROCBLAS_USE_HIPBLASLT=1"
            echo "Environment=HSA_OVERRIDE_GFX_VERSION=11.5.1"
            ;;
        vulkan)
            echo "# Vulkan RPC: no extra env needed"
            ;;
    esac
}

_dist_backend_env_server() {
    # Environment for llama-server (must NOT use local GPU directly)
    case "$1" in
        hip)
            echo "Environment=ROCBLAS_USE_HIPBLASLT=1"
            echo "Environment=HSA_OVERRIDE_GFX_VERSION=11.5.1"
            ;;
        vulkan)
            echo "Environment=GGML_VK_VISIBLE_DEVICES="
            ;;
    esac
}

cmd_dist_backend() {
    require_no_sudo dist-backend
    local new_backend="${1:-}"
    if [[ -z "$new_backend" ]]; then
        bold "Current distributed backend: $DIST_BACKEND"
        echo "  Build dir: $(_dist_build_dir "$DIST_BACKEND")/bin"
        echo ""
        echo "Switch with: $0 dist-backend <vulkan|hip>"
        return
    fi
    if [[ "$new_backend" != "vulkan" && "$new_backend" != "hip" ]]; then
        red "Invalid backend: $new_backend (must be 'vulkan' or 'hip')"
        return 1
    fi
    local build_dir
    build_dir=$(_dist_build_dir "$new_backend")
    if [[ ! -x "$build_dir/bin/rpc-server" ]] || [[ ! -x "$build_dir/bin/llama-server" ]]; then
        red "Backend '$new_backend' not built! Missing binaries in $build_dir/bin/"
        return 1
    fi
    mkdir -p "$CONFIG_DIR"
    echo "$new_backend" > "$CONFIG_DIR/dist-backend"
    DIST_BACKEND="$new_backend"
    green "Backend switched to: $new_backend"
    echo "  Build dir: $build_dir/bin"
    echo ""
    yellow "Run '$0 dist-install' to regenerate services, then '$0 dist-restart'"
}

cmd_dist_start() {
    require_no_sudo dist-start
    _require_worker_ip
    bold "Starting distributed inference (2-node RPC)..."
    echo "  Starting RPC on max2 ($DIST_REMOTE_NODE)..."
    ssh "$DIST_REMOTE_NODE" "systemctl --user start $DIST_RPC_SERVICE.service" || {
        red "Failed to start RPC on max2. Is SSH configured?"
        return 1
    }
    green "  max2 RPC started"

    echo "  Starting RPC on max1 (local)..."
    systemctl --user start "$DIST_RPC_SERVICE.service"
    green "  max1 RPC started"

    echo "  Starting llama-server (distributed)..."
    systemctl --user start "$DIST_LLM_SERVICE.service"
    green "  llama-server started"

    echo ""
    green "Distributed inference running!"
    echo "Check health: curl http://localhost:$PORT/health"
}

cmd_dist_stop() {
    require_no_sudo dist-stop
    _require_worker_ip
    bold "Stopping distributed inference..."
    systemctl --user stop "$DIST_LLM_SERVICE.service" 2>/dev/null && green "  llama-server stopped" || yellow "  llama-server was not running"
    systemctl --user stop "$DIST_RPC_SERVICE.service" 2>/dev/null && green "  max1 RPC stopped" || yellow "  max1 RPC was not running"
    ssh "$DIST_REMOTE_NODE" "systemctl --user stop $DIST_RPC_SERVICE.service" 2>/dev/null && green "  max2 RPC stopped" || yellow "  max2 RPC was not running"
    green "All stopped"
}

cmd_dist_restart() {
    require_no_sudo dist-restart
    cmd_dist_stop
    echo ""
    cmd_dist_start
}

cmd_dist_status() {
    require_no_sudo dist-status
    _require_worker_ip
    bold "Distributed inference status (backend: $DIST_BACKEND)"
    echo ""
    echo "=== max1 (local) ==="
    systemctl --user status "$DIST_RPC_SERVICE.service" --no-pager 2>/dev/null | head -5 || yellow "  rpc-server: not installed"
    echo ""
    systemctl --user status "$DIST_LLM_SERVICE.service" --no-pager 2>/dev/null | head -8 || yellow "  llm-distributed: not installed"
    echo ""
    echo "=== worker ($DIST_REMOTE_NODE) ==="
    ssh "$DIST_REMOTE_NODE" "systemctl --user status $DIST_RPC_SERVICE.service --no-pager 2>/dev/null | head -5" || yellow "  rpc-server: unreachable"
    echo ""

    # Health check
    local health
    health=$(curl -s --max-time 3 "http://localhost:$PORT/health" 2>/dev/null) || true
    if [[ "$health" == *'"ok"'* ]]; then
        green "Health: OK (http://localhost:$PORT/health)"
    else
        yellow "Health: not responding"
    fi
}

cmd_dist_logs() {
    require_no_sudo dist-logs
    if [[ "${1:-}" == "-f" || "${1:-}" == "--follow" ]]; then
        journalctl --user -u "$DIST_LLM_SERVICE.service" -f
    else
        journalctl --user -u "$DIST_LLM_SERVICE.service" --no-pager -n 50
    fi
}

cmd_dist_test() {
    require_no_sudo dist-test
    _require_worker_ip
    bold "Testing USB4 v2 Thunderbolt connectivity to worker ($WORKER_IP)..."
    echo ""

    # 1. Check thunderbolt interface exists locally
    local local_tb_ip
    local_tb_ip=$(ip -4 addr show thunderbolt0 2>/dev/null | grep -oP 'inet \K[\d.]+') || true
    if [[ -z "$local_tb_ip" ]]; then
        red "FAIL: thunderbolt0 interface not found on max1"
        echo "  Try: sudo modprobe -r thunderbolt_net rdma_rxe && sudo modprobe thunderbolt_net && sudo modprobe rdma_rxe"
        return 1
    fi
    green "  max1 thunderbolt0: $local_tb_ip"

    # 2. Ping remote TB IP
    if ping -c 2 -W 2 "$DIST_REMOTE_TB" &>/dev/null; then
        green "  max2 thunderbolt0: $DIST_REMOTE_TB (reachable)"
    else
        red "FAIL: cannot ping max2 at $DIST_REMOTE_TB"
        echo "  Check cable, or try: ssh $DIST_REMOTE_NODE 'sudo modprobe -r thunderbolt_net rdma_rxe && sudo modprobe thunderbolt_net && sudo modprobe rdma_rxe'"
        return 1
    fi

    # 3. SSH over TB
    if ssh -o ConnectTimeout=5 "$DIST_REMOTE_TB" "echo ok" &>/dev/null; then
        green "  SSH over thunderbolt: OK"
    else
        yellow "  SSH over thunderbolt: failed (using LAN for SSH is fine)"
    fi

    # 4. Bandwidth test with iperf3
    if command -v iperf3 &>/dev/null; then
        echo ""
        bold "  Running bandwidth test (5s)..."
        ssh "$DIST_REMOTE_NODE" "iperf3 -s -1 -D -p 5199" 2>/dev/null
        sleep 1
        local result
        result=$(iperf3 -c "$DIST_REMOTE_TB" -p 5199 -t 5 2>&1)
        local bw
        bw=$(echo "$result" | grep "sender" | awk '{for(i=1;i<=NF;i++) if($i ~ /bits\/sec/) print $(i-1), $i}')
        if [[ -n "$bw" ]]; then
            green "  Bandwidth: $bw"
            # Warn if below 20 Gbps
            local gbps
            gbps=$(echo "$result" | grep "sender" | awk '{for(i=1;i<=NF;i++) if($i ~ /bits\/sec/) print $(i-1)}')
            if (( $(echo "$gbps < 20" | bc -l 2>/dev/null || echo 0) )); then
                yellow "  WARNING: bandwidth below 20 Gbps — check cable or port"
            fi
        else
            yellow "  Bandwidth test failed"
            echo "$result" | tail -3
        fi
    else
        yellow "  iperf3 not installed — skipping bandwidth test"
    fi

    # 5. RPC port check
    echo ""
    bold "  Checking RPC port connectivity..."
    if nc -z -w 3 "$DIST_REMOTE_TB" 50052 2>/dev/null; then
        green "  max2 RPC port 50052: listening"
    else
        yellow "  max2 RPC port 50052: not listening (start with: $0 dist-start)"
    fi
    if nc -z -w 3 "$local_tb_ip" 50052 2>/dev/null; then
        green "  max1 RPC port 50052: listening"
    else
        yellow "  max1 RPC port 50052: not listening (start with: $0 dist-start)"
    fi

    echo ""
    green "Thunderbolt connectivity test complete"
}

cmd_dist_install() {
    require_no_sudo dist-install
    _require_worker_ip
    bold "Installing distributed inference services (backend: $DIST_BACKEND)..."
    echo ""

    local build_dir
    build_dir=$(_dist_build_dir "$DIST_BACKEND")
    local llama_bin="$build_dir/bin"
    if [[ ! -x "$llama_bin/rpc-server" ]] || [[ ! -x "$llama_bin/llama-server" ]]; then
        red "llama.cpp not built for '$DIST_BACKEND' backend!"
        if [[ "$DIST_BACKEND" == "hip" ]]; then
            echo "Build it:"
            echo "  cd ~/llama.cpp && HIPCXX=\"\$(hipconfig -l)/clang\" HIP_PATH=\"\$(hipconfig -R)\" cmake -B build-hip -DGGML_HIP=ON -DGGML_RPC=ON -DAMDGPU_TARGETS=gfx1151 -DCMAKE_BUILD_TYPE=Release && cmake --build build-hip -j\$(nproc)"
        else
            echo "Build it:"
            echo "  cd ~/llama.cpp && cmake -B build -DGGML_VULKAN=ON -DGGML_RPC=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build -j\$(nproc)"
        fi
        return 1
    fi

    # Detect active model GGUF path
    local active gguf_path
    active=$(get_active_model)
    if [[ -z "$active" ]]; then
        red "No active model selected. Run: $0 select <model>"
        return 1
    fi
    gguf_path=$(get_model_field "$active" "gguf")
    if [[ -z "$gguf_path" ]] || [[ ! -f "$gguf_path" ]]; then
        red "GGUF file not found for model $active"
        return 1
    fi
    bold "Model: $active"
    echo "  GGUF: $gguf_path"
    echo ""

    # Detect local TB IP
    local local_tb_ip
    local_tb_ip=$(ip -4 addr show thunderbolt0 2>/dev/null | grep -oP 'inet \K[\d.]+') || true
    if [[ -z "$local_tb_ip" ]]; then
        red "thunderbolt0 not found or has no IP. Is the USB4 link configured?"
        return 1
    fi

    # Backend-specific environment lines (different for RPC vs server)
    local rpc_env server_env
    rpc_env=$(_dist_backend_env_rpc "$DIST_BACKEND")
    server_env=$(_dist_backend_env_server "$DIST_BACKEND")

    # Create max1 rpc-server.service
    mkdir -p "$UNIT_DIR"
    cat > "$UNIT_DIR/$DIST_RPC_SERVICE.service" <<EOF
[Unit]
Description=llama.cpp RPC Server [$DIST_BACKEND] (exposes local GPU for distributed inference)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
Environment=LD_LIBRARY_PATH=$llama_bin
$rpc_env
ExecStart=$llama_bin/rpc-server -H $local_tb_ip -p 50052
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
EOF
    green "Created $UNIT_DIR/$DIST_RPC_SERVICE.service (local, $local_tb_ip, $DIST_BACKEND)"

    # Build ExecStart for llama-server
    local exec_cmd="$llama_bin/llama-server"
    exec_cmd+=" -m $gguf_path"
    exec_cmd+=" --rpc $local_tb_ip:50052,$DIST_REMOTE_TB:50052"
    exec_cmd+=" -ngl $GPU_LAYERS"
    exec_cmd+=" -ts 1,1"
    exec_cmd+=" -c $CONTEXT_SIZE"
    exec_cmd+=" -np $PARALLEL_SLOTS"
    exec_cmd+=" -b $BATCH_SIZE"
    [[ "$FLASH_ATTENTION" == "1" ]] && exec_cmd+=" -fa 1"
    exec_cmd+=" --no-mmap"
    if [[ -n "$KV_CACHE_TYPE" ]]; then
        exec_cmd+=" -ctk $KV_CACHE_TYPE -ctv $KV_CACHE_TYPE"
    fi
    exec_cmd+=" --host $HOST --port $PORT"

    # Create max1 llm-distributed.service
    cat > "$UNIT_DIR/$DIST_LLM_SERVICE.service" <<EOF
[Unit]
Description=Distributed LLM Server [$DIST_BACKEND] (2-node RPC via USB4 v2)
After=$DIST_RPC_SERVICE.service
Requires=$DIST_RPC_SERVICE.service

[Service]
Type=simple
Environment=LD_LIBRARY_PATH=$llama_bin
$server_env
ExecStartPre=/bin/bash -c 'for i in \$(seq 1 30); do nc -z $DIST_REMOTE_TB 50052 && exit 0; sleep 2; done; echo "Remote RPC not reachable at $DIST_REMOTE_TB:50052"; exit 1'
ExecStart=$exec_cmd
Restart=on-failure
RestartSec=10

[Install]
WantedBy=default.target
EOF
    green "Created $UNIT_DIR/$DIST_LLM_SERVICE.service ($DIST_BACKEND)"

    # Create max2 rpc-server.service via SSH
    echo ""
    bold "Creating RPC service on max2 ($DIST_REMOTE_NODE)..."
    ssh "$DIST_REMOTE_NODE" "mkdir -p ~/.config/systemd/user && cat > ~/.config/systemd/user/$DIST_RPC_SERVICE.service" <<EOF
[Unit]
Description=llama.cpp RPC Server [$DIST_BACKEND] (exposes local GPU for distributed inference)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
Environment=LD_LIBRARY_PATH=$llama_bin
$rpc_env
ExecStart=$llama_bin/rpc-server -H $DIST_REMOTE_TB -p 50052
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
EOF
    green "Created rpc-server.service on max2 ($DIST_BACKEND)"

    # Enable all
    systemctl --user daemon-reload
    systemctl --user enable "$DIST_RPC_SERVICE.service" "$DIST_LLM_SERVICE.service"
    ssh "$DIST_REMOTE_NODE" "systemctl --user daemon-reload && systemctl --user enable $DIST_RPC_SERVICE.service"
    green "All services enabled (auto-start on boot)"

    echo ""
    bold "Done! Start with: $0 dist-start"
}

cmd_dist_uninstall() {
    require_no_sudo dist-uninstall
    _require_worker_ip
    bold "Uninstalling distributed inference services..."
    cmd_dist_stop 2>/dev/null || true
    systemctl --user disable "$DIST_LLM_SERVICE.service" 2>/dev/null || true
    systemctl --user disable "$DIST_RPC_SERVICE.service" 2>/dev/null || true
    rm -f "$UNIT_DIR/$DIST_LLM_SERVICE.service" "$UNIT_DIR/$DIST_RPC_SERVICE.service"
    systemctl --user daemon-reload
    green "Removed local services"

    ssh "$DIST_REMOTE_NODE" "systemctl --user disable $DIST_RPC_SERVICE.service 2>/dev/null; rm -f ~/.config/systemd/user/$DIST_RPC_SERVICE.service; systemctl --user daemon-reload" 2>/dev/null || true
    green "Removed max2 services"
    green "Done"
}

cmd_help() {
    bold "Usage: $0 <command> [args]"
    echo ""
    echo "New machine quickstart:"
    echo "  $0 set head <ip>     Set head node LAN IP (required for distributed)"
    echo "  $0 set worker <ip>   Set worker node LAN IP (required for distributed)"
    echo "  $0 setup <sub>       Setup: deps | thunderbolt | rdma | build | verify"
    echo "  $0 download          Find + download a model interactively"
    echo "  $0 install           Create + enable the systemd service"
    echo "  $0 start             Start serving"
    echo ""
    echo "Model Discovery & Download:"
    echo "  search <query>       Search HuggingFace for GGUF models"
    echo "  download [repo] [q]  Download a model (interactive if no args)"
    echo ""
    echo "Local Model Management:"
    echo "  models               List downloaded models, show active"
    echo "  select <key>         Set active model for the server"
    echo ""
    echo "Server Configuration:"
    echo "  set head <ip>        Set head node LAN IP (for SSH)"
    echo "  set worker <ip>      Set worker node LAN IP (for SSH)"
    echo "  set head-tb <ip>     Set head node Thunderbolt IP (default: 10.0.0.1)"
    echo "  set worker-tb <ip>   Set worker node Thunderbolt IP (default: 10.0.0.2)"
    echo "  set parallel <n>     Set parallel request slots (divides context equally)"
    echo "  set context <tokens> Set total context pool size"
    echo "  set rope-scale <n>   Enable YaRN RoPE scaling (2.0=512k, 4.0=1M, 1.0=off)"
    echo "  set kv-cache <type>  KV cache quantization (q4_0, q8_0, f16/off)"
    echo "  set port <n>         Set server port"
    echo ""
    echo "Toolbox Management:"
    echo ""
    echo "Service Lifecycle:"
    echo "  install              Create systemd unit + enable service"
    echo "  uninstall            Stop + disable + remove service"
    echo "  start                Start the LLM server"
    echo "  stop                 Stop the LLM server"
    echo "  restart              Restart the LLM server"
    echo "  status               Show active model + service status"
    echo "  logs [-f]            Show logs (use -f to follow)"
    echo "  bench [-q]           Run benchmark — -q for quick (~2 min), full is ~20 min"
    echo "  gpu [-w]             Show GPU stats (VRAM, load, power, clock, temp)"
    echo "                       Use -w for live watch mode"
    echo ""
    echo "Distributed Mode (2-node RPC over USB4 v2):"
    echo "  dist-backend [b]     Show or switch backend (vulkan|hip)"
    echo "  dist-test            Test USB4 v2 Thunderbolt link (ping, bandwidth, RPC)"
    echo "  dist-install         Create systemd services on both nodes"
    echo "  dist-uninstall       Remove distributed services from both nodes"
    echo "  dist-start           Start RPC servers + distributed llama-server"
    echo "  dist-stop            Stop all distributed services"
    echo "  dist-restart         Restart distributed inference"
    echo "  dist-status          Show status of all distributed services"
    echo "  dist-logs [-f]       Show distributed server logs"
}

# ---- Main -------------------------------------------------------------------
case "${1:-}" in
    setup)     shift; cmd_setup "$@" ;;
    search)    shift; cmd_search "$@" ;;
    download)  shift; cmd_download "$@" ;;
    models)    cmd_models ;;
    select)    shift; cmd_select "$@" ;;
    set)       shift; cmd_set "$@" ;;
    containers) echo "Native mode only — no toolbox containers used." ;;
    install)   cmd_install ;;
    uninstall) cmd_uninstall ;;
    start)     cmd_start ;;
    stop)      cmd_stop ;;
    restart)   cmd_restart ;;
    status)    cmd_status ;;
    logs)      shift; cmd_logs "$@" ;;
    bench)     shift; cmd_bench "$@" ;;
    gpu)       shift; cmd_gpu "$@" ;;
    dist-test)      cmd_dist_test ;;
    dist-install)   cmd_dist_install ;;
    dist-uninstall) cmd_dist_uninstall ;;
    dist-start)     cmd_dist_start ;;
    dist-stop)      cmd_dist_stop ;;
    dist-restart)   cmd_dist_restart ;;
    dist-status)    cmd_dist_status ;;
    dist-logs)      shift; cmd_dist_logs "$@" ;;
    dist-backend)   shift; cmd_dist_backend "$@" ;;
    help|--help|-h) cmd_help ;;
    *)
        if [[ -n "${1:-}" ]]; then
            red "Unknown command: $1"
            echo ""
        fi
        cmd_help
        exit 1
        ;;
esac
