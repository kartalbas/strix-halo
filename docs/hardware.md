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

The iGPU shares system memory. To run large models, allocate 96 GB as VRAM per node.

**Step-by-step (static allocation -- our approach):**

1. Enter BIOS setup (press DEL during POST).
2. Navigate to **Advanced** > **AMD CBS** > **NBIO** > **GFX Configuration**.
3. Set **UMA Frame Buffer Size** to **96 GB**.
4. Save and exit.

This is the simpler of two approaches:

| Approach | Description |
|----------|-------------|
| Static 96 GB in BIOS | Set once, always available. This is what we use. |
| Dynamic GTT with kernel params | More flexible but more complex. Uses kernel boot parameters to adjust GTT size at runtime. |

### Kernel Boot Parameters

`iommu=pt` is required for optimal GPU passthrough and DMA performance.

```bash
# Edit GRUB defaults
sudo vi /etc/default/grub

# Add to GRUB_CMDLINE_LINUX:
GRUB_CMDLINE_LINUX="... iommu=pt"

# Regenerate GRUB config
sudo grub2-mkconfig -o /boot/grub2/grub.cfg
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
