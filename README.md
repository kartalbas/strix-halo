# Dual-Node MiniMax M2.5 229B Inference — 2x Minisforum MS-S1 MAX (AMD Strix Halo) over USB4 Thunderbolt

Run [**MiniMax M2.5 229B**](https://huggingface.co/MiniMaxAI) — the 229-billion-parameter Mixture-of-Experts LLM by **MiniMax (稀宇科技)** — locally on two **Minisforum MS-S1 MAX** mini PCs using distributed [**llama.cpp**](https://github.com/ggml-org/llama.cpp) with the **Vulkan** (RADV) backend and **RPC** layer splitting over a **USB4 v2 Thunderbolt** cable. No cloud GPUs, no NVIDIA — just two AMD APUs with 256 GB of unified memory.

**~13 tok/s generation | ~195 tok/s prompt processing | 131K context | under $3,000 total hardware cost**

| Metric | Value |
|---|---|
| Model | [MiniMax-M2.5](https://huggingface.co/MiniMaxAI/MiniMax-M2.5) 229B (MoE: 32 experts, 2 active per token, 61 layers) |
| Developer | [MiniMax (稀宇科技)](https://www.minimax.io/) |
| Quantization | UD-Q6_K_XL via [unsloth](https://huggingface.co/unsloth/MiniMax-M2.5-GGUF) (~181 GB GGUF) |
| Nodes | 2x [Minisforum MS-S1 MAX](https://www.minisforum.com/) |
| CPU | AMD Ryzen AI MAX+ 395 (Strix Halo) — 16C/32T Zen 5 |
| iGPU | Radeon 8060S — 40 CUs, RDNA 3.5, up to 121 GB VRAM per node ([dynamic GTT + .drirc](docs/hardware.md#approach-2-dynamic-gtt-advanced--more-vram)) |
| Memory | 128 GB LPDDR5x-8000 unified memory per node (~256 GB/s) |
| Backend | Vulkan (RADV / Mesa) — no ROCm or CUDA required |
| Interconnect | USB4 v2 Thunderbolt (rear ports), ~38 Gbps measured |
| Distributed inference | llama.cpp RPC — 2-node tensor-parallel split (`-ts 1,1`) |
| Generation speed | ~13 tok/s |
| Prompt processing | ~195 tok/s |
| Context window | 131K tokens (q8_0 KV cache) |
| SWE-bench Verified | 80.2% (per MiniMax) |
| Total hardware cost | ~$2,800 (2x MS-S1 MAX + USB4 v2 cable) |
| Power consumption | ~100W combined (both nodes) |

For comparison, cloud GPU inference for a 229B parameter model typically requires multiple NVIDIA A100/H100 GPUs at $2-8/hour. This setup runs 24/7 on hardware you own, with no recurring costs beyond electricity.

> **Who is this for?** Anyone who wants to run frontier-class LLMs locally — AI developers, researchers, self-hosters, home lab enthusiasts — without spending $10K+ on NVIDIA GPUs or paying per-token cloud API fees. The entire setup is automated by a single management script.

---

## Hardware — 2x Minisforum MS-S1 MAX (AMD Ryzen AI MAX+ 395)

### Bill of Materials

| Component | Specification | Qty | Notes |
|---|---|---|---|
| Minisforum MS-S1 MAX | AMD Ryzen AI MAX+ 395 (Strix Halo), 128 GB LPDDR5x-8000 | 2 | 96 GB iGPU VRAM (BIOS) or up to 112 GB ([dynamic GTT](docs/hardware.md#approach-2-dynamic-gtt-advanced--more-vram)) |
| USB4 v2 Thunderbolt cable | 80 Gbps rated, 0.5-1m recommended | 1 | Must use **rear ports** only |
| Ethernet | Standard CAT6 or existing LAN | 1 | Required for SSH between nodes and API access |

### Strix Halo APU — Radeon 8060S iGPU (RDNA 3.5, Vulkan)

The AMD Ryzen AI MAX+ 395 (Strix Halo) integrates a Radeon 8060S iGPU with 40 RDNA 3.5 compute units. The unified memory architecture shares the 128 GB LPDDR5x-8000 pool between CPU and GPU — there is no separate VRAM chip, so both allocations run at the same ~256 GB/s bandwidth. By default, we configure 96 GB VRAM per node (192 GB total) via a static BIOS setting. For more headroom, dynamic GTT allocation via kernel parameters can provide up to 112 GB per node (224 GB total). See [docs/hardware.md](docs/hardware.md#uma--vram-allocation) for both approaches.

### BIOS Settings (AMI, UMA Frame Buffer, VRAM Allocation)

- **Board**: SHWSA
- **BIOS vendor**: AMI, version 1.06 or later
- **Key setting**: UMA Frame Buffer Size must be set to 96 GB (see Prerequisites below)

### USB4 v2 vs v1 Ports — Thunderbolt Bandwidth

The MS-S1 MAX has two classes of USB4 ports:

| Location | Version | Measured bandwidth | Use for inference? |
|---|---|---|---|
| **Rear** | USB4 v2 | ~38 Gbps | Yes |
| Front | USB4 v1 | ~9.3 Gbps | No — too slow |

Always connect the Thunderbolt cable to the **rear USB4 v2 ports**. The front ports are USB4 v1 and will bottleneck distributed inference significantly.

---

## Prerequisites (BIOS, Fedora, SSH, USB4 Cable)

Only four things require manual configuration. Everything else is handled by `llm-server.sh setup` subcommands.

### 3a. BIOS Configuration (iGPU VRAM, Thunderbolt)

Power on each machine and enter BIOS setup (press `DEL` during POST). For BIOS updates, see [capetron/minisforum-ms-s1-max-bios](https://github.com/capetron/minisforum-ms-s1-max-bios/tree/main).

1. **Set iGPU VRAM to 96 GB** (default approach):
   ```
   Advanced → AMD CBS → NBIO → GFX Configuration → UMA Frame Buffer Size → 96GB
   ```
   This allocates 96 GB to the iGPU and leaves 32 GB for the OS. For more VRAM (up to 121 GB) using dynamic GTT allocation via kernel parameters + `.drirc`, see [docs/hardware.md](docs/hardware.md#approach-2-dynamic-gtt-advanced--more-vram).

2. **Enable Thunderbolt** (should be enabled by default):
   ```
   Advanced → AMD PBS → Thunderbolt Configuration → Thunderbolt Support → Enabled
   ```

3. **Save and exit** BIOS.

### 3b. Install Fedora 43 Server + Kernel Upgrade

Install Fedora 43 Server on both machines. A minimal server installation is sufficient — no desktop environment is needed. Create the same user account on both machines (e.g., `mkadm`).

After installation, **upgrade the kernel and all packages** on both machines. A recent kernel is critical for USB4 v2 Thunderbolt support, RADV Vulkan driver, and AMDGPU stability:

```bash
sudo dnf upgrade -y
sudo reboot
```

Ensure both machines have the same username. Enable lingering so user services persist after logout:
```bash
loginctl enable-linger $USER
```

**Add IOMMU kernel parameter** on both machines (required for optimal GPU/DMA performance with the RADV Vulkan driver):
```bash
sudo grubby --update-kernel=ALL --args="iommu=pt"
sudo reboot
```

Also ensure both machines have static LAN IPs (or DHCP reservations) so they can always reach each other.

### 3c. Passwordless SSH Between Both Machines

From each machine, copy your SSH key to the other:

```bash
# On the head node:
ssh-keygen -t ed25519    # if you don't already have a key
ssh-copy-id <worker-ip>

# On the worker node:
ssh-keygen -t ed25519
ssh-copy-id <head-ip>
```

Verify passwordless access works in both directions:
```bash
ssh <worker-ip> hostname    # should print worker's hostname
ssh <head-ip> hostname      # should print head's hostname
```

### 3d. USB4 v2 Thunderbolt Cable (Rear Ports Only)

Connect a USB4 v2 (80 Gbps) Thunderbolt cable between the **rear** USB4 v2 ports of both machines. The cable should be 0.5m to 1m for best signal integrity.

After connecting, verify the link is detected:
```bash
# Should show thunderbolt device entries (domain0, 0-0, 0-1, etc.)
ls /sys/bus/thunderbolt/devices/
```

Do not use an adapter or hub — the cable must connect directly between the two rear USB4 v2 ports. Active cables (with electronics for signal boosting) are fine for longer runs but passive cables are preferred for sub-1m distances.

---

## Automated Setup — llm-server.sh (Vulkan, RPC, Thunderbolt Networking)

The `llm-server.sh` script automates all remaining setup. Copy it to your home directory on `max1` (the head node):

```bash
cp scripts/llm-server.sh ~/llm-server.sh
chmod +x ~/llm-server.sh
```

All setup subcommands are idempotent — safe to re-run if anything fails partway through.

### 4a. Configure Node IPs

Tell the script the LAN IPs of both machines (used for SSH between nodes):

```bash
~/llm-server.sh set head <head-ip>      # head node LAN IP (this machine)
~/llm-server.sh set worker <worker-ip>    # worker node LAN IP
```

Replace the IPs above with your actual LAN addresses. These are saved to `~/.config/llm-server/server.conf` and persist across reboots.

The Thunderbolt IPs default to `10.0.0.1` (head) and `10.0.0.2` (worker) — these are assigned automatically by `setup thunderbolt`. To override:

```bash
~/llm-server.sh set head-tb 10.0.0.1       # default, usually no need to change
~/llm-server.sh set worker-tb 10.0.0.2     # default, usually no need to change
```

### 4b. Install Dependencies (Vulkan SDK, Mesa RADV, Build Tools)

```bash
~/llm-server.sh setup deps
```

This installs required packages on **both nodes** via SSH:

**Build tools**:
- `cmake`, `gcc-c++`, `git`, `ninja-build`

**Vulkan SDK and drivers**:
- `vulkan-headers`, `vulkan-loader-devel`, `vulkan-tools`
- `mesa-vulkan-drivers` (provides the RADV driver for RDNA 3.5)

**Networking and RDMA**:
- `iperf3` (bandwidth testing)
- `nmap-ncat` (port connectivity checks)
- `rdma-core`, `libibverbs-utils` (Soft-RoCE RDMA support)

**Python**:
- `python3-pip`, `python3-huggingface-hub` (model downloads from HuggingFace)

**Utilities**:
- `jq` (JSON parsing for HuggingFace API)
- `bc` (arithmetic in scripts)
- `curl` (HTTP requests)

Verify GPU detection after installing:
```bash
vulkaninfo --summary
```

Expected output should include:
```
GPU0:
    apiVersion    = 1.3.xxx
    driverVersion = 25.x.x
    vendorID      = 0x1002
    deviceID      = 0x150e
    deviceType    = INTEGRATED_GPU
    deviceName    = AMD Radeon 8060S (RADV GFX1151)
    driverName    = radv
```

### 4c. Configure USB4 v2 Thunderbolt Networking (Node-to-Node)

```bash
~/llm-server.sh setup thunderbolt
```

This configures on **both nodes**:

1. **Kernel modules**: Loads `thunderbolt_net` and sets up auto-load at boot via `/etc/modules-load.d/thunderbolt-rdma.conf`:
   ```
   thunderbolt_net
   rdma_rxe
   ```

2. **Static IP addressing**: Assigns `10.0.0.1` to `max1` and `10.0.0.2` to `max2` on the `thunderbolt0` interface using NetworkManager:
   ```bash
   # Example of what runs on max1:
   nmcli connection add type ethernet con-name thunderbolt0 \
       ifname thunderbolt0 ipv4.method manual \
       ipv4.addresses 10.0.0.1/24
   ```

3. **Firewall rules**: Opens port 50052/tcp (RPC) on the Thunderbolt zone:
   ```bash
   firewall-cmd --permanent --zone=trusted --add-interface=thunderbolt0
   firewall-cmd --permanent --zone=trusted --add-port=50052/tcp
   firewall-cmd --reload
   ```

4. **TCP tuning**: Installs `/etc/sysctl.d/99-thunderbolt-tcp.conf` with settings optimized for high-bandwidth, low-latency tensor transfers:
   - 16 MB socket buffers (`rmem_max`, `wmem_max`)
   - BBR congestion control (better throughput than CUBIC on high-BDP links)
   - TCP window scaling enabled
   - Increased netdev backlog (5000)

The static IPs used throughout this project:

| Node | LAN IP | Thunderbolt IP | Role |
|---|---|---|---|
| Head | `<head-ip>` | 10.0.0.1 (default) | Runs llama-server + local RPC |
| Worker | `<worker-ip>` | 10.0.0.2 (default) | Runs remote RPC |

### 4d. Configure Soft-RoCE RDMA (Future-Proofing)

```bash
~/llm-server.sh setup rdma
```

This sets up Soft-RoCE (software RDMA over Converged Ethernet) on both nodes:
- Loads the `rdma_rxe` kernel module
- Creates a systemd service (`rdma-rxe.service`) that brings up the `rxe0` RDMA device over `thunderbolt0` at boot
- Verifies RDMA device creation with `ibv_devices`

**Note**: llama.cpp RPC currently uses TCP, not RDMA. This step is future-proofing for when llama.cpp gains native RDMA support, which would reduce latency and CPU overhead for tensor transfers. It is optional but harmless to configure now.

### 4e. Build llama.cpp (Vulkan + RPC Backends)

```bash
~/llm-server.sh setup build
```

This clones and builds llama.cpp with Vulkan and RPC support on **both nodes**:

```bash
# What it runs on each node:
cd ~/llama.cpp
git pull    # or git clone https://github.com/ggml-org/llama.cpp.git
cmake -B build \
    -DGGML_VULKAN=ON \
    -DGGML_RPC=ON \
    -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

Key build flags:
- `GGML_VULKAN=ON` — uses the RADV Vulkan driver (Mesa) for GPU compute
- `GGML_RPC=ON` — enables the RPC server/client for distributed inference
- Build output goes to `~/llama.cpp/build/bin/` containing `llama-server`, `rpc-server`, `llama-bench`, etc.

The build takes approximately 5-10 minutes per node on the Ryzen AI MAX+ 395.

### 4f. Verify Setup

```bash
~/llm-server.sh setup verify
```

Runs a comprehensive verification checklist on both nodes:

```
[PASS] Vulkan GPU detected: AMD Radeon 8060S (RADV GFX1151)
[PASS] VRAM allocated: 96 GB
[PASS] llama-server binary: ~/llama.cpp/build/bin/llama-server
[PASS] rpc-server binary: ~/llama.cpp/build/bin/rpc-server
[PASS] llama-bench binary: ~/llama.cpp/build/bin/llama-bench
[PASS] thunderbolt0 interface: 10.0.0.1/24
[PASS] Ping max2 (10.0.0.2): OK
[PASS] iperf3 bandwidth: 38.1 Gbps
[PASS] SSH to max2 (<worker-ip>): OK
[PASS] Firewall port 50052: open
[PASS] Kernel module thunderbolt_net: loaded
[PASS] Kernel module rdma_rxe: loaded
[PASS] max2 Vulkan GPU: AMD Radeon 8060S (RADV GFX1151)
[PASS] max2 VRAM: 96 GB
[PASS] max2 llama.cpp binaries: present
```

Any `[FAIL]` items include a suggested fix command. All checks must pass before proceeding to model download and distributed inference.

### Full Setup Flow

On a fresh pair of machines with BIOS configured and Fedora 43 installed:

```bash
# Copy the script to your home directory
cp scripts/llm-server.sh ~/llm-server.sh
chmod +x ~/llm-server.sh

# Configure node IPs (replace with your actual LAN IPs)
~/llm-server.sh set head <head-ip>
~/llm-server.sh set worker <worker-ip>

# Run all setup steps in order
~/llm-server.sh setup deps          # Install packages on both nodes
~/llm-server.sh setup thunderbolt   # Configure USB4 networking
~/llm-server.sh setup rdma          # Configure Soft-RoCE (optional)
~/llm-server.sh setup build         # Clone and build llama.cpp on both nodes
~/llm-server.sh setup verify        # Verify everything is ready
```

After setup completes, proceed to download a model and start serving.

---

## Model Download and Distributed Inference

### 5a. Search and Download GGUF Models from HuggingFace

Search HuggingFace for GGUF models:
```bash
~/llm-server.sh search minimax
```

Output shows matching repos with download counts:
```
   1) unsloth/MiniMax-M2.5-GGUF                               (335k downloads, 120 likes)
   2) bartowski/MiniMax-M2.5-GGUF                              (42k downloads, 35 likes)
```

Download a specific quantization:
```bash
~/llm-server.sh download unsloth/MiniMax-M2.5-GGUF UD-Q6_K_XL
```

This downloads the model files to `~/models/gguf/UD-Q6_K_XL/` (~181 GB, allow 30-60 minutes depending on your internet connection). The download uses `huggingface_hub` and automatically resumes if interrupted.

After downloading, the script registers the model and prompts you to set it as the active model.

**Note**: The model files only need to exist on the **head node**. With distributed RPC and `--no-mmap`, the head node loads the model and distributes tensor data to the worker's GPU over the USB4 v2 link. No model files are needed on the worker.

### 5b. Select Model and Configure Settings

List downloaded models:
```bash
~/llm-server.sh models
```

Select the active model:
```bash
~/llm-server.sh select minimax-m2.5-ud-q6_k_xl
```

Configure server settings:
```bash
# Set context window (131K recommended for this model with q8_0 KV cache)
~/llm-server.sh set context 131072

# Set KV cache quantization (q8_0 saves ~50% KV memory vs f16)
~/llm-server.sh set kv-cache q8_0
```

Available settings:

| Setting | Command | Default | Notes |
|---|---|---|---|
| Context size | `set context <tokens>` | 262144 | Total context pool across all slots |
| KV cache type | `set kv-cache <type>` | f16 | `q8_0` recommended; `q4_0` saves more but lower quality |
| Parallel slots | `set parallel <n>` | 1 | Each slot gets `context / n` tokens |
| Port | `set port <n>` | 8080 | Server listen port |
| RoPE scale | `set rope-scale <n>` | 1.0 | YaRN scaling factor (1.0 = disabled) |

### 5c. Test USB4 Link

Before starting distributed inference, verify the Thunderbolt link:

```bash
~/llm-server.sh dist-test
```

Expected output:
```
Testing USB4 v2 Thunderbolt connectivity to max2...

  max1 thunderbolt0: 10.0.0.1
  max2 thunderbolt0: 10.0.0.2 (reachable)
  SSH over thunderbolt: OK

  Running bandwidth test (5s)...
  Bandwidth: 38.1 Gbits/sec

  Checking RPC port connectivity...
  max2 RPC port 50052: not listening (start with: ./llm-server.sh dist-start)
  max1 RPC port 50052: not listening (start with: ./llm-server.sh dist-start)

Thunderbolt connectivity test complete
```

If bandwidth is below 20 Gbps, check that you are using the rear USB4 v2 ports, not the front USB4 v1 ports.

### 5d. Install Distributed Services (RPC Server + llama-server)

```bash
~/llm-server.sh dist-install
```

This creates three systemd user services across both nodes:

| Service | Node | Description |
|---|---|---|
| `rpc-server.service` | max1 | Exposes max1's GPU via RPC on `10.0.0.1:50052` |
| `rpc-server.service` | max2 | Exposes max2's GPU via RPC on `10.0.0.2:50052` |
| `llm-distributed.service` | max1 | Runs `llama-server` connecting to both RPC servers |

The `dist-install` command automatically:
- Creates the service files on both nodes (max2's service is deployed via SSH)
- Enables all services for auto-start on boot
- Sets `GGML_VK_VISIBLE_DEVICES=""` on the head node to prevent the llama-server process from double-counting the local GPU (it accesses GPUs exclusively through RPC)
- Configures `ExecStartPre` health check that waits up to 60 seconds for the remote RPC server to become available
- Sets `Requires=rpc-server.service` so the local RPC server must be running before llama-server starts

The generated `llm-distributed.service` ExecStart looks like:

```bash
llama-server \
    -m ~/models/gguf/UD-Q6_K_XL/MiniMax-M2.5-UD-Q6_K_XL-00001-of-00005.gguf \
    --rpc 10.0.0.1:50052,10.0.0.2:50052 \
    -ngl 999 -ts 1,1 \
    -c 131072 -np 1 -b 4096 \
    -fa 1 --no-mmap \
    -ctk q8_0 -ctv q8_0 \
    --host 0.0.0.0 --port 8080
```

All flags are derived from your current `set` configuration, so changing settings and re-running `dist-install` regenerates the services automatically.

### 5e. Start, Stop, and Monitor

**Start distributed inference** (starts all three services in the correct order):
```bash
~/llm-server.sh dist-start
```

**Check status** of all services across both nodes:
```bash
~/llm-server.sh dist-status
```

**View server logs** (the llama-server process on max1):
```bash
~/llm-server.sh dist-logs        # last 50 lines
~/llm-server.sh dist-logs -f     # follow live
```

**Stop all services**:
```bash
~/llm-server.sh dist-stop
```

**Restart** (stops then starts all services):
```bash
~/llm-server.sh dist-restart
```

### 5f. Run Benchmarks

Quick benchmark (~2 minutes, stops server temporarily to free VRAM):
```bash
~/llm-server.sh bench -q
```

Full benchmark (~20 minutes, tests multiple prompt/generation sizes):
```bash
~/llm-server.sh bench
```

Results are saved as Markdown files in `~/llm-benchmarks/`.

### 5g. GPU Monitoring

View GPU stats (VRAM usage, load, power, clock, temperature):
```bash
~/llm-server.sh gpu
```

Live monitoring mode (updates every second):
```bash
~/llm-server.sh gpu -w
```

Example output:
```
  VRAM  [████████████████████████████░░] 89.2 / 96 GB (93%)
  Load  [██████████████████████████████] 98%
  Power    42.3 W
  Clock     2.90 GHz
  Temp     72.1 °C
```

### 5h. Uninstall

Remove all distributed services from both nodes:
```bash
~/llm-server.sh dist-uninstall
```

This stops all running services, disables them, and removes the unit files from both max1 and max2.

---

## Architecture — Distributed llama.cpp RPC over USB4 Thunderbolt

### Three-Process Architecture (2x rpc-server + llama-server)

Distributed inference uses three cooperating processes across two machines:

```
┌─────────────── max1 (head node) ───────────────┐     ┌──── max2 (worker) ────┐
│                                                 │     │                       │
│  llama-server         rpc-server                │     │  rpc-server           │
│  (orchestrator)       (local GPU)               │     │  (remote GPU)         │
│       │                    │                    │     │       │               │
│       ├──── RPC TCP ──────►│ 10.0.0.1:50052     │     │       │ 10.0.0.2     │
│       │                    │                    │     │       │ :50052        │
│       └──── RPC TCP ──────►├────────────────────┼─────┼──────►│              │
│                            │   USB4 v2 (~38Gbps)│     │       │              │
│  :8080 (OpenAI API)        │                    │     │       │              │
└─────────────────────────────────────────────────┘     └───────────────────────┘
```

1. **rpc-server** (on each node) — exposes the local GPU's compute and memory over TCP. Each RPC server binds to its Thunderbolt IP on port 50052. It handles tensor operations (matrix multiply, etc.) on behalf of the orchestrator. The `rpc-server` is needed because llama.cpp's distributed inference requires a uniform interface to all GPUs — the orchestrator (`llama-server`) treats every GPU as a remote RPC backend, whether it's on the same machine or across the network. This is why `rpc-server` runs on the head node too, not just the worker: `llama-server` itself does not access any GPU directly. Instead, `GGML_VK_VISIBLE_DEVICES=""` hides the local GPU from the orchestrator, and all GPU work goes through the two RPC servers symmetrically.

2. **llama-server** (on max1 only) — the orchestrator. It loads the model, splits layers evenly across the two RPC backends (`-ts 1,1`), and serves an OpenAI-compatible API on port 8080.

### Why USB4 v2 Thunderbolt? (Interconnect Bandwidth Comparison)

The bottleneck in distributed inference is moving tensor data between nodes. During each forward pass, intermediate activations must cross the interconnect. USB4 v2 provides ~38 Gbps measured throughput, which is sufficient for MoE models where only 2 of 32 experts are active per token — the amount of data transferred per token is relatively small compared to a dense 229B model.

For comparison:
- USB4 v1 (front ports): ~9.3 Gbps — workable but noticeably slower generation
- USB4 v2 (rear ports): ~38 Gbps — used here
- 100GbE: ~90 Gbps — would be faster but requires NICs and a switch
- InfiniBand: ~200+ Gbps — overkill for this use case and prohibitively expensive

### Key llama.cpp Flags for Distributed Vulkan Inference

| Flag | Value | Purpose |
|---|---|---|
| `--rpc` | `10.0.0.1:50052,10.0.0.2:50052` | Connect to both RPC servers |
| `-ts` | `1,1` | Split layers evenly across both GPUs |
| `-ngl` | `999` | Offload all layers to GPU |
| `--no-mmap` | (set) | Required for distributed inference — model must be fully loaded into memory, not memory-mapped |
| `-fa 1` | (set) | Enable flash attention for memory efficiency |
| `-ctk q8_0 -ctv q8_0` | (set) | Quantize KV cache to q8_0, halving KV memory vs f16 |
| `GGML_VK_VISIBLE_DEVICES=""` | (env) | Prevents the llama-server process from seeing the local GPU directly, avoiding double-counting |
| `-c 131072` | (set) | 131K context window (limited by available memory after model loading) |
| `-np 1` | (set) | Single parallel slot (all context allocated to one user) |
| `-b 4096` | (set) | Batch size for prompt processing |

### How Tensor-Parallel Layer Splitting Works (MoE)

With `-ts 1,1`, llama.cpp distributes the 61 model layers approximately evenly across the two RPC backends (~30-31 layers each). During inference:

1. The orchestrator sends the input tokens to the first RPC backend (max1's GPU)
2. max1's GPU processes layers 0-30, producing intermediate activations
3. The activations are sent over the USB4 v2 link to max2's RPC backend
4. max2's GPU processes layers 31-60, producing the final logits
5. The logits are sent back to the orchestrator for token sampling

For MoE models like MiniMax M2.5, each layer activates only 2 of 32 experts, so the actual computation per layer is modest (equivalent to a ~10B dense model). The interconnect bandwidth is the primary bottleneck.

### Configuration Files

The config files in `configs/` are provided as **reference examples only**. You do not need to edit them manually — `llm-server.sh` generates all configuration and systemd unit files automatically. They are included in the repository to document the exact settings used:

- `configs/systemd/llm-distributed.service` — example distributed llama-server unit
- `configs/systemd/rpc-server.service` — example RPC server unit
- `configs/server/server.conf` — example server settings
- `configs/system/99-thunderbolt-tcp.conf` — TCP tuning sysctl values
- `configs/system/rdma-rxe.service` — Soft-RoCE RDMA service
- `configs/system/thunderbolt-rdma.conf` — kernel module auto-load config
- `configs/system/drirc` — Mesa/RADV unified heap config (install to `~/.drirc`, required for dynamic GTT)

---

## Verify Everything Works

After `dist-install` and `dist-start`, verify the full stack:

```bash
# 1. Check all services are running
~/llm-server.sh dist-status

# 2. Health check via API
curl http://localhost:8080/health
# Expected: {"status":"ok"}

# 3. Test inference
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MiniMax-M2.5-UD-Q6_K_XL-00001-of-00005.gguf",
    "messages": [{"role": "user", "content": "Write a Python function to compute fibonacci numbers."}],
    "max_tokens": 256,
    "temperature": 0.7
  }'
```

If the health check returns `{"status":"ok"}` and the inference test returns a coherent response, everything is working correctly.

Check GPU utilization on both nodes during inference:
```bash
# On max1
~/llm-server.sh gpu

# On max2 (via SSH)
ssh <worker-ip> "~/llm-server.sh gpu"
```

Both GPUs should show high VRAM usage (~89-93 GB each) when the model is loaded.

---

## llm-server.sh Complete Reference

### Setup Commands

| Command | Description |
|---|---|
| `setup deps` | Install required packages on both nodes |
| `setup thunderbolt` | Configure USB4 v2 networking, firewall, TCP tuning |
| `setup rdma` | Configure Soft-RoCE RDMA (future-proofing, optional) |
| `setup build` | Clone and build llama.cpp on both nodes |
| `setup verify` | Run comprehensive verification on both nodes |

### Model Management

| Command | Description |
|---|---|
| `search <query>` | Search HuggingFace for GGUF models |
| `download [repo] [quant]` | Download a model (interactive if no args) |
| `models` | List downloaded models, show active |
| `select <key>` | Set active model for the server |

### Node Configuration

| Command | Description |
|---|---|
| `set head <ip>` | Head node LAN IP (required for distributed/setup) |
| `set worker <ip>` | Worker node LAN IP (required for distributed/setup) |
| `set head-tb <ip>` | Head node Thunderbolt IP (default: `10.0.0.1`) |
| `set worker-tb <ip>` | Worker node Thunderbolt IP (default: `10.0.0.2`) |

### Server Configuration

| Command | Description |
|---|---|
| `set context <tokens>` | Set total context pool size |
| `set kv-cache <type>` | KV cache quantization: `q4_0`, `q8_0`, `f16`/`off` |
| `set parallel <n>` | Set parallel request slots |
| `set port <n>` | Set server listen port |
| `set rope-scale <n>` | YaRN RoPE scaling factor (1.0 = disabled) |

### Single-Node Mode

| Command | Description |
|---|---|
| `install` | Create systemd unit and enable service |
| `uninstall` | Stop, disable, and remove service |
| `start` | Start the LLM server |
| `stop` | Stop the LLM server |
| `restart` | Restart the LLM server |
| `status` | Show active model and service status |
| `logs [-f]` | Show logs (`-f` to follow live) |

### Distributed Mode (Dual-Node RPC over Thunderbolt)

| Command | Description |
|---|---|
| `dist-backend [vulkan\|hip]` | Show or switch GPU backend |
| `dist-test` | Test USB4 v2 link (ping, bandwidth, RPC ports) |
| `dist-install` | Create systemd services on both nodes |
| `dist-uninstall` | Remove distributed services from both nodes |
| `dist-start` | Start all RPC servers + distributed llama-server |
| `dist-stop` | Stop all distributed services |
| `dist-restart` | Restart distributed inference |
| `dist-status` | Show status of all services across both nodes |
| `dist-logs [-f]` | Show distributed server logs |

### Monitoring

| Command | Description |
|---|---|
| `bench [-q]` | Run benchmark (`-q` for quick ~2 min, full ~20 min) |
| `gpu [-w]` | Show GPU stats (`-w` for live watch mode) |

### Typical Workflow

1. Complete the four prerequisites (BIOS, Fedora, SSH, USB4 cable)
2. `~/llm-server.sh set head <head-ip>` — set head node LAN IP
3. `~/llm-server.sh set worker <worker-ip>` — set worker node LAN IP
4. `~/llm-server.sh setup deps` — install packages
5. `~/llm-server.sh setup thunderbolt` — configure networking
6. `~/llm-server.sh setup build` — build llama.cpp
7. `~/llm-server.sh setup verify` — verify setup
8. `~/llm-server.sh search minimax` — find the model
9. `~/llm-server.sh download unsloth/MiniMax-M2.5-GGUF UD-Q6_K_XL` — download it
10. `~/llm-server.sh select minimax-m2.5-ud-q6_k_xl` — set active model
11. `~/llm-server.sh set context 131072` — configure context
12. `~/llm-server.sh set kv-cache q8_0` — enable KV cache quantization
13. `~/llm-server.sh dist-test` — verify Thunderbolt link
14. `~/llm-server.sh dist-install` — install services
15. `~/llm-server.sh dist-start` — start inference
16. `~/llm-server.sh dist-status` — verify running
17. `~/llm-server.sh bench -q` — benchmark performance

### Configuration File Locations

| File | Path | Description |
|---|---|---|
| Server config | `~/.config/llm-server/server.conf` | Saved settings (node IPs, context, KV cache, port, etc.) |
| Active model | `~/.config/llm-server/active-model` | Currently selected model key |
| Model registry | `~/.config/llm-server/models/*.conf` | Downloaded model metadata |
| Service units | `~/.config/systemd/user/*.service` | Generated systemd user services |
| Model files | `~/models/gguf/` | Downloaded GGUF model files |
| Benchmark results | `~/llm-benchmarks/` | Saved benchmark output |
| TCP tuning | `/etc/sysctl.d/99-thunderbolt-tcp.conf` | Thunderbolt TCP optimization |
| RDMA service | `/etc/systemd/system/rdma-rxe.service` | Soft-RoCE RDMA device creation |
| Module autoload | `/etc/modules-load.d/thunderbolt-rdma.conf` | Kernel module auto-load |
| Mesa/RADV config | `~/.drirc` | Unified Vulkan heap for dynamic GTT (see [docs/hardware.md](docs/hardware.md#step-3--critical-enable-unified-vulkan-heap-drirc)) |

---

## Client Configuration (OpenAI-Compatible API)

### OpenAI-Compatible API (Chat, Completions, Models)

The llama-server exposes an OpenAI-compatible API at `http://<head-node-ip>:8080/v1`. Any client that supports the OpenAI API format can connect.

**Endpoints**:
- `POST /v1/chat/completions` — chat completions
- `POST /v1/completions` — text completions
- `GET /v1/models` — list available models
- `GET /health` — health check

No API key is required (any non-empty string works as a placeholder).

**Example with curl**:
```bash
curl http://<head-node-ip>:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MiniMax-M2.5-UD-Q6_K_XL-00001-of-00005.gguf",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 100
  }'
```

**Example with Python (openai library)**:
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://<head-node-ip>:8080/v1",
    api_key="dummy",
)

response = client.chat.completions.create(
    model="MiniMax-M2.5-UD-Q6_K_XL-00001-of-00005.gguf",
    messages=[{"role": "user", "content": "Hello"}],
    max_tokens=100,
)
print(response.choices[0].message.content)
```

### OpenCode AI Coding Assistant Configuration

For [opencode](https://opencode.ai), create or edit `opencode.json` in your project directory:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "strix-halo": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Strix Halo Cluster",
      "options": {
        "baseURL": "http://<head-node-ip>:8080/v1",
        "apiKey": "dummy"
      },
      "models": {
        "MiniMax-M2.5-UD-Q6_K_XL-00001-of-00005.gguf": {
          "name": "MiniMax-M2.5 UD-Q6_K_XL (2-node distributed)",
          "limit": {
            "context": 131072,
            "output": 32768
          }
        }
      }
    }
  },
  "model": "strix-halo/MiniMax-M2.5-UD-Q6_K_XL-00001-of-00005.gguf"
}
```

See `configs/clients/opencode.json` for the full example.

---

## Benchmarks — MiniMax M2.5 229B on Dual Strix Halo

### Performance Summary

| Test | Result |
|---|---|
| Token generation (tg512) | ~13 tok/s |
| Prompt processing (pp512) | ~195 tok/s |
| Prompt processing (pp2048) | ~170 tok/s |
| End-to-end (pp512 + tg512) | ~12.5 tok/s effective |
| USB4 v2 bandwidth | ~38 Gbps (iperf3) |
| Model load time | ~90 seconds |
| VRAM usage per node | ~89-93 GB |
| Total model size (UD-Q6_K_XL) | ~181 GB |

Generation speed is primarily limited by interconnect bandwidth (tensor transfers between nodes) rather than GPU compute. Prompt processing is faster because it can be parallelized more effectively across both GPUs, with larger batches amortizing the transfer overhead.

For detailed benchmark results across different context sizes, quantizations, and configurations, see [docs/benchmarks.md](docs/benchmarks.md).

---

## Troubleshooting (Thunderbolt, VRAM, RPC, Vulkan)

| Symptom | Likely cause | Fix |
|---|---|---|
| `thunderbolt0` interface missing | Kernel module not loaded or cable not detected | `sudo modprobe thunderbolt_net` and check cable seating |
| Bandwidth < 20 Gbps | Using front USB4 v1 port | Move cable to rear USB4 v2 port |
| RPC timeout on startup | max2 RPC not started or firewall blocking | Check `dist-status`, verify port 50052 is open |
| `VRAM allocation failed` | UMA frame buffer not set correctly | Static: set to 96 GB in BIOS. Dynamic GTT: check `ttm.pages_limit` kernel param. See [docs/hardware.md](docs/hardware.md) |
| Vulkan shows less VRAM than expected with GTT | Missing `~/.drirc` — RADV splits GTT into 2/3 device-local + 1/3 host | Install `configs/system/drirc` to `~/.drirc`. See [docs/hardware.md](docs/hardware.md#step-3--critical-enable-unified-vulkan-heap-drirc) |
| Segfault or crash on load | Missing `--no-mmap` flag | Ensure `--no-mmap` is set (automatic with `dist-install`) |
| Double VRAM usage on max1 | `GGML_VK_VISIBLE_DEVICES` not empty | Ensure the env var is set to empty string (automatic with `dist-install`) |
| Slow generation (<8 tok/s) | KV cache using f16 instead of q8_0 | `~/llm-server.sh set kv-cache q8_0` then `dist-install` and `dist-restart` |

**Common diagnostic commands**:

```bash
# Check if thunderbolt interface exists and has an IP
ip addr show thunderbolt0

# Check if RPC port is reachable from max1 to max2
nc -z -w 3 10.0.0.2 50052 && echo "OK" || echo "FAIL"

# Check loaded kernel modules
lsmod | grep -E "thunderbolt_net|rdma_rxe"

# Check VRAM allocation (should show ~96 GB)
cat /sys/class/drm/card*/device/mem_info_vram_total

# Check firewall rules for thunderbolt zone
firewall-cmd --zone=trusted --list-all

# View systemd service logs for errors
journalctl --user -u llm-distributed.service --no-pager -n 100

# Verify llama.cpp binaries exist
ls -la ~/llama.cpp/build/bin/{llama-server,rpc-server,llama-bench}
```

For a comprehensive troubleshooting guide, see [docs/troubleshooting.md](docs/troubleshooting.md).

---

## References and Related Projects

- [llama.cpp](https://github.com/ggml-org/llama.cpp) — the inference engine powering this setup
- [MiniMax (稀宇科技)](https://www.minimax.io/) — developer of the MiniMax-M2.5 229B model
- [MiniMax-M2.5 on HuggingFace](https://huggingface.co/MiniMaxAI/MiniMax-M2.5) — official model page
- [unsloth GGUF quants](https://huggingface.co/unsloth/MiniMax-M2.5-GGUF) — quantized GGUF files used here
- [Minisforum MS-S1 MAX](https://www.minisforum.com/) — the hardware platform
- [AMD ROCm: Strix Halo optimization](https://rocm.docs.amd.com/en/latest/how-to/system-optimization/strixhalo.html) — official memory management guide
- [Jeff Geerling: VRAM allocation on AMD AI APUs](https://www.jeffgeerling.com/blog/2025/increasing-vram-allocation-on-amd-ai-apus-under-linux/) — GTT testing and limits
- [Framework Community: Strix Halo LLM tests](https://community.frame.work/t/amd-strix-halo-ryzen-ai-max-395-gpu-llm-performance-tests/72521) — community benchmarks
- [AMD: Trillion-parameter LLM on Ryzen AI Max+](https://www.amd.com/en/developer/resources/technical-articles/2026/how-to-run-a-one-trillion-parameter-llm-locally-an-amd.html) — AMD's multi-node guide

See [docs/hardware.md](docs/hardware.md#references) for the full reference list.

---

## License

See [LICENSE](LICENSE) for details.

---

<sub>**Keywords**: AMD Strix Halo, Ryzen AI MAX+ 395, Minisforum MS-S1 MAX, Radeon 8060S, RDNA 3.5, Vulkan RADV, llama.cpp, distributed inference, RPC server, dual node, two node, multi-node, USB4 v2, Thunderbolt, MiniMax M2.5, MiniMax 稀宇科技, 229B, Mixture of Experts, MoE, GGUF, LPDDR5x, unified memory, 128GB, 256GB, iGPU, local LLM, self-hosted AI, OpenAI compatible API, tensor parallel, layer splitting, no cloud GPU, consumer hardware, mini PC, home lab, llama-server, rpc-server</sub>
