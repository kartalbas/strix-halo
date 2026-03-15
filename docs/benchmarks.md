# Performance Benchmarks

All benchmarks are collected using `llm-server.sh bench`.

## Quantization Comparison (MiniMax M2.5)

Two-node distributed inference over Thunderbolt.

| Quant | Size | Gen (tok/s) | Prompt (tok/s) | Quality |
|-------|------|-------------|----------------|---------|
| UD-Q6_K_XL | ~181 GB | ~13 | ~195 | Highest |
| Q6_K | ~175 GB | ~13 | ~195 | Very High |
| Q5_K_M | ~155 GB | ~14 | ~210 | High |
| Q4_K_XL | ~130 GB | ~16 | ~230 | Good |

Lower quantization trades some quality for faster speed and smaller memory footprint. Q4_K_XL is the sweet spot if the model needs to fit alongside a large KV cache.

## Backend Comparison

Same model and quantization, comparing Vulkan (RADV) vs ROCm/HIP backends.

| Backend | Gen (tok/s) | Prompt (tok/s) | Notes |
|---------|-------------|----------------|-------|
| Vulkan (RADV) | ~13 | ~195 | Better generation speed |
| ROCm/HIP | ~11 | ~225 | Better prompt processing |

**Vulkan is the default** because generation speed (tok/s during output) matters more for interactive use. Prompt processing is a one-time cost per request, while generation speed determines the perceived responsiveness.

## Context Size vs Memory

### With Static 96 GB VRAM (default)

| Context Size | KV Cache Quant | Approx Memory per Node | Fits in 192 GB (96 GB x2)? |
|--------------|----------------|-------------------------|-----------------------------|
| 131K | q8_0 | ~90 GB model + KV cache | Yes (~6 GB headroom) |
| 131K | f16 | ~93 GB model + KV cache | Tight |
| 262K | q8_0 | Exceeds available | No |
| 262K | q4_0 | May fit with smaller model quant | Possibly |

### With Dynamic GTT 112 GB VRAM

Using [dynamic GTT allocation](../docs/hardware.md#approach-2-dynamic-gtt-advanced--more-vram), each node can access up to 112 GB:

| Context Size | KV Cache Quant | Approx Memory per Node | Fits in 224 GB (112 GB x2)? |
|--------------|----------------|-------------------------|-----------------------------|
| 131K | q8_0 | ~90 GB model + KV cache | Yes (~22 GB headroom) |
| 131K | f16 | ~93 GB model + KV cache | Yes (~19 GB headroom) |
| 262K | q8_0 | ~105 GB model + KV cache | Yes (~7 GB headroom) |
| 262K | q4_0 | ~98 GB model + KV cache | Yes (~14 GB headroom) |

The extra 16 GB per node (32 GB total) unlocks:
- **f16 KV cache** at 131K context — higher quality without memory pressure
- **262K context** with quantized KV cache — double the context window
- **More parallel slots** — e.g., 2 slots at 65K context each

At 131K context with q8_0 KV cache, each node uses approximately 90 GB (model shard + KV cache), fitting well within either VRAM configuration.

## Interconnect Bandwidth

| Configuration | Bandwidth | Notes |
|---------------|-----------|-------|
| Single USB4 v2 cable (rear) | ~38 Gbps | Standard setup |
| Dual cable | Higher aggregate | Second cable on 10.0.1.0/24 subnet |
| Single USB4 v1 cable (front) | ~9.3 Gbps | Not recommended |

Single rear cable at ~38 Gbps is sufficient for the current generation speeds. The interconnect is not the bottleneck for models at ~13 tok/s generation.
