# Hardware Documentation

## System Specifications

| Component | Details |
|-----------|---------|
| Nodes | 2x Minisforum MS-S1 MAX |
| CPU | AMD Ryzen AI MAX+ 395 (Strix Halo), 16C/32T Zen 5 |
| iGPU | Radeon 8060S (40 CUs, RDNA 3.5) |
| Memory | 128 GB LPDDR5x-8000 unified memory per node (~256 GB/s bandwidth) |
| Storage | NVMe SSD |
| Total Memory | 256 GB across both nodes |

## BIOS Configuration

### BIOS Information

- **Vendor:** American Megatrends International (AMI)
- **Version:** 1.06
- **Date:** 2026-01-04
- **Board:** SHWSA
- **BIOS updates:** [capetron/minisforum-ms-s1-max-bios](https://github.com/capetron/minisforum-ms-s1-max-bios/tree/main)

### UMA / VRAM Allocation

The iGPU shares system memory (unified memory architecture). There is no separate VRAM chip — both "VRAM" and "system RAM" are the same physical LPDDR5x-8000 pool. The allocation simply controls how much of that pool the GPU driver can use.

There are two approaches:

| Approach | VRAM | OS RAM | Complexity | Stability |
|----------|------|--------|------------|-----------|
| **Static 96 GB in BIOS** | 96 GB | 32 GB | Simple — set once | Rock-solid, well tested |
| **Dynamic GTT via kernel params** | 104-112 GB | 16-24 GB | Moderate — BIOS + GRUB | Stable at ≤112 GB ([ref](#references)) |

#### Approach 1: Static 96 GB (Default)

This is the simpler approach and what we use in our standard setup.

1. Enter BIOS setup (press DEL during POST).
2. Navigate to **Advanced** > **AMD CBS** > **NBIO** > **GFX Configuration**.
3. Set **UMA Frame Buffer Size** to **96 GB**.
4. Save and exit.

This gives each node 96 GB VRAM + 32 GB system RAM. The MiniMax M2.5 UD-Q6_K_XL model uses ~90 GB per node, leaving ~6 GB headroom for KV cache.

#### Approach 2: Dynamic GTT (Advanced — More VRAM)

Instead of reserving a fixed 96 GB in BIOS, set a small BIOS reservation and let the kernel dynamically manage a larger GPU-accessible pool via GTT (Graphics Translation Table). This is [recommended by AMD's own ROCm documentation](https://rocm.docs.amd.com/en/latest/how-to/system-optimization/strixhalo.html) for Strix Halo.

Since all memory is physically the same LPDDR5x-8000, there is **no performance difference** between static UMA and dynamic GTT allocation — both provide the same ~256 GB/s bandwidth.

**Step 1 — BIOS: Set UMA to 1 GB (or 512 MB)**

1. Enter BIOS setup (press DEL during POST).
2. Navigate to **Advanced** > **AMD CBS** > **NBIO** > **GFX Configuration**.
3. Set **iGPU Configuration** to **UMA_Specified**.
4. Set **UMA Frame Buffer Size** to **1G** (some boards don't allow lower than 1 GB).
5. Ensure **Above 4G Decoding** and **Resizable BAR** are enabled.
6. Save and exit.

**Step 2 — GRUB: Set TTM pages limit**

The `ttm.pages_limit` kernel parameter controls how many 4 KB pages the GPU driver can map from system RAM. The older `amdgpu.gttsize` parameter still works but is [deprecated](https://www.mail-archive.com/amd-gfx@lists.freedesktop.org/msg117333.html).

```bash
# For 120 GB GPU-accessible memory (leaves 8 GB for OS — tested stable on single-node):
sudo grubby --update-kernel=ALL --args="ttm.pages_limit=31457280"

# For 112 GB (safer — leaves 16 GB for OS):
sudo grubby --update-kernel=ALL --args="ttm.pages_limit=29360128"

# For 104 GB (conservative — leaves 24 GB for OS):
sudo grubby --update-kernel=ALL --args="ttm.pages_limit=27262976"

sudo reboot
```

The formula: `desired_GB × 1024 × 1024 / 4 = pages`. For 120 GB: `120 × 1024 × 1024 / 4 = 31,457,280`.

**Step 3 — CRITICAL: Enable unified Vulkan heap (`.drirc`)**

By default, the RADV Vulkan driver splits GTT memory into two heaps: 2/3 device-local + 1/3 host-visible. This means a 120 GB GTT allocation only gives ~80 GB of usable device-local VRAM — **worse than the static 96 GB BIOS approach**.

The fix is a Mesa drirc option that tells RADV to expose all memory as a single device-local heap. This is safe on APUs because all memory is physically the same LPDDR5x — the split only exists as a workaround for poorly written games.

Create `~/.drirc` on each node:

```xml
<?xml version="1.0" standalone="yes"?>
<driconf>
    <device driver="radv">
        <application name="Default">
            <option name="radv_enable_unified_heap_on_apu" value="true" />
        </application>
    </device>
</driconf>
```

This option was added in [Mesa 22.3](https://gitlab.freedesktop.org/mesa/mesa/-/merge_requests/18884) and is present in all recent Mesa versions. It is already used by default for some games (e.g., Red Dead Redemption 2) in `/usr/share/drirc.d/00-radv-defaults.conf`. No reboot required — takes effect on the next Vulkan application launch.

> **Without this step**, the GTT trick is counterproductive: you get ~80 GB device-local instead of 96 GB. **With this step**, all ~121 GB is exposed as a single device-local heap.

**Step 4 — Verify**

After reboot (for the kernel parameter) and creating `.drirc` (no reboot needed):

```bash
# Check Vulkan sees a single large heap:
vulkaninfo 2>/dev/null | grep -A5 "memoryHeaps:"
# Expected: 1 heap, ~121 GB, DEVICE_LOCAL

# Check OS memory:
free -h
# Expected: ~7-8 GB total (for 120 GB GTT) or ~16 GB (for 112 GB GTT)
```

**Tested results on MS-S1 MAX (128 GB, BIOS UMA = 1 GB):**

| ttm.pages_limit | GTT Size | Vulkan Heap (without .drirc) | Vulkan Heap (with .drirc) | OS RAM |
|-----------------|----------|------------------------------|---------------------------|--------|
| 31,457,280 | 120 GB | 80 GB device-local + 40 GB host | **121 GB single device-local** | 8 GB |
| 29,360,128 | 112 GB | ~75 GB device-local + 37 GB host | **~113 GB single device-local** | 16 GB |
| 27,262,976 | 104 GB | ~69 GB device-local + 35 GB host | **~105 GB single device-local** | 24 GB |

**Community-tested stability limits:**

| Target VRAM | ttm.pages_limit | OS RAM Left | Status |
|-------------|-----------------|-------------|--------|
| 104 GB | 27,262,976 | 24 GB | Safe |
| 108 GB | 28,311,552 | 20 GB | Stable ([Geerling](https://www.jeffgeerling.com/blog/2025/increasing-vram-allocation-on-amd-ai-apus-under-linux)) |
| 112 GB | 29,360,128 | 16 GB | Stable ([CachyOS guide](https://brian.th3rogers.com/posts/strixhalo-cachyos/)) |
| 115 GB | 30,146,560 | 13 GB | Used by [Framework community](https://community.frame.work/t/igpu-vram-how-much-can-be-assigned/73081) |
| 120 GB | 31,457,280 | 8 GB | Tested stable on single-node (this project) |

> **Warning**: Jeff Geerling reported [segfaults at 110 GB](https://www.jeffgeerling.com/blog/2025/increasing-vram-allocation-on-amd-ai-apus-under-linux) during large model loads. With the unified heap fix, the full GTT pool is usable. Start conservative and increase if stable.

**What this means for inference:**

| Allocation | VRAM per Node | Single-Node | Dual-Node Total | MiniMax M2.5 (~181 GB) |
|-----------|--------------|-------------|-----------------|------------------------|
| Static 96 GB | 96 GB | 96 GB | 192 GB | Fits (dual only) |
| Dynamic 112 GB | 113 GB | 113 GB | 226 GB | Fits (dual only) |
| Dynamic 120 GB | 121 GB | 121 GB | 242 GB | Fits (dual), more KV headroom |

The extra headroom enables larger context windows, more parallel slots, or less aggressive KV quantization (f16 instead of q8_0).

#### Which approach to use?

- **New to this setup?** Start with static 96 GB in BIOS. It's simple, one setting, no `.drirc` needed.
- **Need more VRAM?** (larger context, more parallel slots, larger quant) — use dynamic GTT + `.drirc`. We recommend 120 GB for single-node or 112 GB for dual-node (where the head node runs more services).
- **Single-node setup?** Dynamic GTT at 120 GB is ideal — you get 121 GB usable VRAM with 8 GB OS RAM, which is sufficient when only running llama-server.

### Kernel Boot Parameters

`iommu=pt` is required for optimal GPU passthrough and DMA performance.

```bash
sudo grubby --update-kernel=ALL --args="iommu=pt"
sudo reboot
```

If using dynamic GTT, combine both parameters:

```bash
sudo grubby --update-kernel=ALL --args="iommu=pt ttm.pages_limit=29360128"
sudo reboot
```

### Thunderbolt BIOS Settings

Thunderbolt is usually enabled by default on this board. Verify under **Advanced** > **AMD PBS** that Thunderbolt support is enabled.

### Power Management Recommendations

- Set power profile to **Performance** in BIOS if available.
- On Fedora, use `tuned-adm profile throughput-performance` for consistent clock speeds.
- Disable CPU idle states if latency-sensitive workloads require it.

## USB4 Port Map

The MS-S1 MAX has both USB4 v1 and v2 ports. **Always use the rear ports** for node-to-node links.

| Location | Standard | Link Speed | Measured TCP Throughput |
|----------|----------|------------|------------------------|
| Rear | USB4 v2 | 80 Gbps | ~38 Gbps |
| Front | USB4 v1 | 40 Gbps | ~9.3 Gbps |

### Cable Specifications

- 40 Gbps USB4 cables work fine on the rear ports (the link negotiates at 80 Gbps regardless).
- Keep cables short (0.8 m or less recommended) for reliable signal integrity.
- Tested cables: standard 40 Gbps USB4 certified cables.

## Network Topology

| Network | Subnet | Head Node | Worker Node | Purpose |
|---------|--------|-----------|-------------|---------|
| LAN | Your LAN subnet | `set head <ip>` | `set worker <ip>` | Management, SSH, general traffic |
| Thunderbolt | 10.0.0.0/24 (default) | 10.0.0.1 | 10.0.0.2 | High-speed RPC for distributed inference |

The Thunderbolt link is the primary data path for distributed llama.cpp inference. LAN is used for SSH access and management.

## References

### VRAM / Memory Allocation

- [AMD ROCm: Strix Halo System Optimization](https://rocm.docs.amd.com/en/latest/how-to/system-optimization/strixhalo.html) — AMD's official guidance on memory configuration for Strix Halo APUs. Recommends keeping dedicated VRAM small and using shared memory via TTM.
- [Jeff Geerling: Increasing VRAM allocation on AMD AI APUs under Linux](https://www.jeffgeerling.com/blog/2025/increasing-vram-allocation-on-amd-ai-apus-under-linux/) — Tested up to 108 GB GTT on AI Max+ 395, documents stability limits.
- [Framework Community: iGPU VRAM — How much can be assigned?](https://community.frame.work/t/igpu-vram-how-much-can-be-assigned/73081) — Community testing of GTT limits on Framework Desktop with Strix Halo.
- [Framework Community: Strix Halo llama.cpp Installation Guide (Fedora 42)](https://community.frame.work/t/amd-strix-halo-llama-cpp-installation-guide-for-fedora-42/75856) — Step-by-step llama.cpp setup with Vulkan on Strix Halo.
- [CachyOS Strix Halo LLM Configuration](https://brian.th3rogers.com/posts/strixhalo-cachyos/) — 112 GB TTM configuration guide.
- [Setting up unified memory for Strix Halo on Ubuntu 25.04/25.10](https://dev.webonomic.nl/setting-up-unified-memory-for-strix-halo-correctly-on-ubuntu-25-04-or-25-10) — TTM parameter documentation with Ubuntu.
- [amdgpu.gttsize deprecation (kernel mailing list)](https://www.mail-archive.com/amd-gfx@lists.freedesktop.org/msg117333.html) — Announcement that `amdgpu.gttsize` is deprecated in favor of `ttm.pages_limit`.
- [Mesa MR !18884: radv_enable_unified_heap_on_apu](https://gitlab.freedesktop.org/mesa/mesa/-/merge_requests/18884) — The Mesa merge request that added the unified heap option for APUs. Without this, RADV splits GTT into 2/3 device-local + 1/3 host-visible, reducing usable VRAM.

### Strix Halo LLM Performance

- [Framework Community: Strix Halo GPU LLM Performance Tests](https://community.frame.work/t/amd-strix-halo-ryzen-ai-max-395-gpu-llm-performance-tests/72521) — Community benchmarks across multiple models and backends.
- [llm-tracker.info: Strix Halo GPU Performance](https://llm-tracker.info/AMD-Strix-Halo-(Ryzen-AI-Max+-395)-GPU-Performance) — Aggregated benchmark data for Strix Halo.
- [Level1Techs: Strix Halo LLM Benchmark Results](https://forum.level1techs.com/t/strix-halo-ryzen-ai-max-395-llm-benchmark-results/233796) — Community benchmark thread.
- [AMD: Trillion-Parameter LLM on Ryzen AI Max+ Cluster](https://www.amd.com/en/developer/resources/technical-articles/2026/how-to-run-a-one-trillion-parameter-llm-locally-an-amd.html) — AMD's own guide for multi-node inference.

### Setup Guides

- [Gygeek/Framework-strix-halo-llm-setup](https://github.com/Gygeek/Framework-strix-halo-llm-setup) — Complete setup guide using 512 MB BIOS VRAM + 115 GB GTT.
- [kyuz0/amd-strix-halo-toolboxes](https://github.com/kyuz0/amd-strix-halo-toolboxes) — Containerized LLM setup for Strix Halo.
- [capetron/minisforum-ms-s1-max-bios](https://github.com/capetron/minisforum-ms-s1-max-bios) — BIOS updates for the MS-S1 MAX.

### Known Issues

- [llama.cpp #19818: ROCm SIGKILL on APU — GTT not utilized](https://github.com/ggml-org/llama.cpp/issues/19818) — ROCm ignores GTT on Strix Halo; Vulkan/RADV does not have this issue.
- [llama.cpp #14854: Slow model loading >64 GB on Vulkan](https://github.com/ggml-org/llama.cpp/issues/14854) — Vulkan model loading slows past 64 GB.
- [ollama #12062: GTT ignored on gfx1151](https://github.com/ollama/ollama/issues/12062) — Related ROCm GTT issue.
