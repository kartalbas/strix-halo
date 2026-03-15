# Troubleshooting

## 1. Thunderbolt Hot-Plug Failure

**Symptom:** After unplugging and re-plugging the Thunderbolt cable, the `thunderbolt0` network interface does not come back.

**Fix:** Reload the kernel modules on both nodes:

```bash
sudo modprobe -r thunderbolt_net rdma_rxe
sudo modprobe thunderbolt_net
sudo modprobe rdma_rxe
```

Run this on both max1 and max2.

## 2. linux-firmware Regression

**Symptom:** ROCm breaks on gfx1151 after updating linux-firmware to version 20251125 or later.

**Fix:** Downgrade to the last known-good version and pin it:

```bash
sudo dnf downgrade linux-firmware-20251111
sudo dnf versionlock add linux-firmware-20251111
```

## 3. VRAM Exhaustion

**Symptom:** `llama-server` starts but immediately exits with no obvious error. The service appears to launch successfully but produces no output.

**Cause:** The combined model shard + KV cache exceeds the 96 GB VRAM allocation on one or both nodes.

**Fix:** Reduce memory usage by either lowering context size or using more aggressive KV cache quantization:

```bash
llm-server.sh set context <smaller_value>
llm-server.sh set kv-cache q4_0
```

## 4. RPC Connection Timeout

**Symptom:** `llm-distributed.service` times out after 60 seconds waiting for the remote RPC server.

**Diagnosis checklist:**

1. Check that the remote RPC server is running:
   ```bash
   ssh max2 systemctl --user status rpc-server
   ```

2. Check the Thunderbolt link:
   ```bash
   ping 10.0.0.2
   ```

3. Check the firewall allows traffic on the Thunderbolt interface:
   ```bash
   firewall-cmd --zone=trusted --list-interfaces
   ```
   The Thunderbolt interface should be listed in the trusted zone.

## 5. Model Loading Errors with Distributed Inference

**Symptom:** Distributed inference crashes during model loading.

**Cause:** Memory-mapped I/O does not work across RPC nodes. The `--no-mmap` flag is required.

**Fix:** The `dist-install` command in `llm-server.sh` adds `--no-mmap` automatically. If you are running manually, ensure `--no-mmap` is passed to `llama-server`.

## 6. Front USB4 Ports Are Slow

**Symptom:** Distributed inference is much slower than expected, or `llm-server.sh dist-test` reports bandwidth below 20 Gbps.

**Cause:** The front ports on the MS-S1 MAX are USB4 v1 (~9.3 Gbps). The rear ports are USB4 v2 (~38 Gbps).

**Fix:** Always use the rear USB4 ports for the node-to-node Thunderbolt link. `llm-server.sh dist-test` will warn if bandwidth is below 20 Gbps.

## 7. Services Do Not Survive Logout

**Symptom:** User-level systemd services (rpc-server, llm-distributed) stop when the user logs out.

**Cause:** `loginctl linger` is not enabled for the user.

**Fix:**

```bash
loginctl enable-linger $USER
```

The `dist-install` command in `llm-server.sh` enables this automatically.

## 8. Dynamic GTT Segfaults or OOM

**Symptom:** System crashes, segfaults, or OOM kills when using dynamic GTT allocation (`ttm.pages_limit`) at high values (>108 GB).

**Cause:** Leaving too little RAM for the OS and llama-server orchestrator. The head node typically needs ~15 GB for the OS + orchestrator + SSH + system services.

**Fix:** Reduce the GTT allocation. Start conservative and increase gradually:

```bash
# Safe: 104 GB VRAM, 24 GB for OS
sudo grubby --update-kernel=ALL --args="ttm.pages_limit=27262976"

# Moderate: 108 GB VRAM, 20 GB for OS
sudo grubby --update-kernel=ALL --args="ttm.pages_limit=28311552"

# Aggressive: 112 GB VRAM, 16 GB for OS
sudo grubby --update-kernel=ALL --args="ttm.pages_limit=29360128"
```

Monitor OS memory after booting: `free -h`. If "available" is below 2 GB during inference, reduce the allocation. See [docs/hardware.md](hardware.md#approach-2-dynamic-gtt-advanced--more-vram) for details and tested limits.

**References:** [Jeff Geerling's testing](https://www.jeffgeerling.com/blog/2025/increasing-vram-allocation-on-amd-ai-apus-under-linux/), [Framework community](https://community.frame.work/t/igpu-vram-how-much-can-be-assigned/73081)

## 9. Slow Model Loading on Vulkan (>64 GB)

**Symptom:** Model loading takes significantly longer than expected (several minutes instead of ~90 seconds) when the model exceeds ~64 GB.

**Cause:** Known Vulkan/RADV issue with large memory allocations. See [llama.cpp #14854](https://github.com/ggml-org/llama.cpp/issues/14854).

**Workaround:** This primarily affects initial load time, not inference speed. Wait for the load to complete — once loaded, inference runs at normal speed. Newer Mesa/RADV versions may improve this.

## 10. ROCm/HIP Build Issues

**Symptom:** Build failures or runtime errors when compiling llama.cpp with HIP support.

**What works:** ROCm 6.4.4 from the Fedora repos. Do **not** use AMD's upstream repositories as they conflict with mesa.

**Build command for HIP:**

```bash
HIPCXX="$(hipconfig -l)/clang" \
HIP_PATH="$(hipconfig -R)" \
cmake -B build-hip \
  -DGGML_HIP=ON \
  -DGGML_RPC=ON \
  -DAMDGPU_TARGETS=gfx1151
```
